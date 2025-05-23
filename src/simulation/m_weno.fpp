!>
!! @file m_weno.f90
!! @brief Contains module m_weno
#:include 'macros.fpp'

!> @brief  Weighted essentially non-oscillatory (WENO) reconstruction scheme
!!              that is supplemented with monotonicity preserving bounds (MPWENO)
!!              and a mapping function that boosts the accuracy of the non-linear
!!              weights (WENOM). MPWENO, see Balsara and Shu (2000), prevents the
!!              reconstructed values to lay outside the range set by the stencil,
!!              while WENOM, see Henrick et al. (2005), recovers the formal order
!!              of accuracy of the reconstruction at critical points. Please note
!!              that the basic WENO approach is implemented according to the work
!!              of Jiang and Shu (1996). WENO-Z, which is less dissipative than
!!              WENO-JS and WENO-M, is implemented according to the work of
!!              Borges, et al. (2008). TENO, which is even less dissipative than
!!              WENO-Z but is less robust, is implemented according to the work
!!              of Fu et al. (2016).
module m_weno
    ! Dependencies =============================================================
    use m_derived_types        !< Definitions of the derived types

    use m_global_parameters    !< Definitions of the global parameters

    use m_variables_conversion !< State variables type conversion procedures

#ifdef MFC_OPENACC
    use openacc
#endif

    use m_mpi_proxy
    ! ==========================================================================

    !implicit none

    private; public :: s_initialize_weno_module, s_initialize_weno, s_finalize_weno_module, s_weno

    !> @name The cell-average variables that will be WENO-reconstructed. Formerly, they
    !! are stored in v_vf. However, they are transferred to v_rs_wsL and v_rs_wsR
    !! as to be reshaped (RS) and/or characteristically decomposed. The reshaping
    !! allows the WENO procedure to be independent of the coordinate direction of
    !! the reconstruction. Lastly, notice that the left (L) and right (R) results
    !! of the characteristic decomposition are stored in custom-constructed WENO-
    !! stencils (WS) that are annexed to each position of a given scalar field.
    !> @{

    real(kind(0d0)), allocatable, dimension(:, :, :, :) :: v_rs_ws_x, v_rs_ws_y, v_rs_ws_z
    !> @}

    ! WENO Coefficients ========================================================

    !> @name Polynomial coefficients at the left and right cell-boundaries (CB) and at
    !! the left and right quadrature points (QP), in the x-, y- and z-directions.
    !! Note that the first dimension of the array identifies the polynomial, the
    !! second dimension identifies the position of its coefficients and the last
    !! dimension denotes the cell-location in the relevant coordinate direction.
    !> @{
    real(kind(0d0)), target, allocatable, dimension(:, :, :) :: poly_coef_cbL_x
    real(kind(0d0)), target, allocatable, dimension(:, :, :) :: poly_coef_cbL_y
    real(kind(0d0)), target, allocatable, dimension(:, :, :) :: poly_coef_cbL_z

    real(kind(0d0)), target, allocatable, dimension(:, :, :) :: poly_coef_cbR_x
    real(kind(0d0)), target, allocatable, dimension(:, :, :) :: poly_coef_cbR_y
    real(kind(0d0)), target, allocatable, dimension(:, :, :) :: poly_coef_cbR_z

    !    real(kind(0d0)), pointer, dimension(:, :, :) :: poly_coef_L => null()
    !    real(kind(0d0)), pointer, dimension(:, :, :) :: poly_coef_R => null()
    !> @}

    !> @name The ideal weights at the left and the right cell-boundaries and at the
    !! left and the right quadrature points, in x-, y- and z-directions. Note
    !! that the first dimension of the array identifies the weight, while the
    !! last denotes the cell-location in the relevant coordinate direction.
    !> @{
    real(kind(0d0)), target, allocatable, dimension(:, :) :: d_cbL_x
    real(kind(0d0)), target, allocatable, dimension(:, :) :: d_cbL_y
    real(kind(0d0)), target, allocatable, dimension(:, :) :: d_cbL_z

    real(kind(0d0)), target, allocatable, dimension(:, :) :: d_cbR_x
    real(kind(0d0)), target, allocatable, dimension(:, :) :: d_cbR_y
    real(kind(0d0)), target, allocatable, dimension(:, :) :: d_cbR_z
!    real(kind(0d0)), pointer, dimension(:, :) :: d_L => null()
!    real(kind(0d0)), pointer, dimension(:, :) :: d_R => null()
    !> @}

    !> @name Smoothness indicator coefficients in the x-, y-, and z-directions. Note
    !! that the first array dimension identifies the smoothness indicator, the
    !! second identifies the position of its coefficients and the last denotes
    !! the cell-location in the relevant coordinate direction.
    !> @{
    real(kind(0d0)), target, allocatable, dimension(:, :, :) :: beta_coef_x
    real(kind(0d0)), target, allocatable, dimension(:, :, :) :: beta_coef_y
    real(kind(0d0)), target, allocatable, dimension(:, :, :) :: beta_coef_z
!    real(kind(0d0)), pointer, dimension(:, :, :) :: beta_coef => null()
    !> @}

    ! END: WENO Coefficients ===================================================

    integer :: v_size !< Number of WENO-reconstructed cell-average variables

    !$acc declare create(v_size)

    !> @name Indical bounds in the s1-, s2- and s3-directions
    !> @{
    type(int_bounds_info) :: is1_weno, is2_weno, is3_weno
    !$acc declare create(is1_weno, is2_weno, is3_weno)
    !
    !> @}

    real(kind(0d0)) :: test
    !$acc declare create(test)

    !$acc declare create( &
    !$acc                v_rs_ws_x, v_rs_ws_y, v_rs_ws_z, &
    !$acc                poly_coef_cbL_x,poly_coef_cbL_y,poly_coef_cbL_z, &
    !$acc                poly_coef_cbR_x,poly_coef_cbR_y,poly_coef_cbR_z,d_cbL_x,       &
    !$acc                d_cbL_y,d_cbL_z,d_cbR_x,d_cbR_y,d_cbR_z,beta_coef_x,beta_coef_y,beta_coef_z)

contains

    !>  The computation of parameters, the allocation of memory,
        !!      the association of pointers and/or the execution of any
        !!      other procedures that are necessary to setup the module.
    subroutine s_initialize_weno_module

        if (weno_order == 1) return

        ! Allocating/Computing WENO Coefficients in x-direction ============
        is1_weno%beg = -buff_size; is1_weno%end = m - is1_weno%beg
        if (n == 0) then
            is2_weno%beg = 0
        else
            is2_weno%beg = -buff_size; 
        end if

        is2_weno%end = n - is2_weno%beg

        if (p == 0) then
            is3_weno%beg = 0
        else
            is3_weno%beg = -buff_size
        end if

        is3_weno%end = p - is3_weno%beg

        @:ALLOCATE_GLOBAL(poly_coef_cbL_x(is1_weno%beg + weno_polyn:is1_weno%end - weno_polyn, 0:weno_polyn, &
            0:weno_polyn - 1))
        @:ALLOCATE_GLOBAL(poly_coef_cbR_x(is1_weno%beg + weno_polyn:is1_weno%end - weno_polyn, 0:weno_polyn, &
            0:weno_polyn - 1))

        @:ALLOCATE_GLOBAL(d_cbL_x(0:weno_num_stencils, is1_weno%beg + weno_polyn:is1_weno%end - weno_polyn))
        @:ALLOCATE_GLOBAL(d_cbR_x(0:weno_num_stencils, is1_weno%beg + weno_polyn:is1_weno%end - weno_polyn))

        @:ALLOCATE_GLOBAL(beta_coef_x(is1_weno%beg + weno_polyn:is1_weno%end - weno_polyn, 0:weno_polyn, &
            0:2*(weno_polyn - 1)))

        call s_compute_weno_coefficients(1, is1_weno)

        @:ALLOCATE_GLOBAL(v_rs_ws_x(is1_weno%beg:is1_weno%end, &
            is2_weno%beg:is2_weno%end, is3_weno%beg:is3_weno%end, 1:sys_size))

        ! ==================================================================

        ! Allocating/Computing WENO Coefficients in y-direction ============
        if (n == 0) return

        is2_weno%beg = -buff_size; is2_weno%end = n - is2_weno%beg
        is1_weno%beg = -buff_size; is1_weno%end = m - is1_weno%beg

        if (p == 0) then
            is3_weno%beg = 0
        else
            is3_weno%beg = -buff_size
        end if

        is3_weno%end = p - is3_weno%beg

        @:ALLOCATE_GLOBAL(poly_coef_cbL_y(is2_weno%beg + weno_polyn:is2_weno%end - weno_polyn, 0:weno_polyn, &
            0:weno_polyn - 1))
        @:ALLOCATE_GLOBAL(poly_coef_cbR_y(is2_weno%beg + weno_polyn:is2_weno%end - weno_polyn, 0:weno_polyn, &
            0:weno_polyn - 1))

        @:ALLOCATE_GLOBAL(d_cbL_y(0:weno_num_stencils, is2_weno%beg + weno_polyn:is2_weno%end - weno_polyn))
        @:ALLOCATE_GLOBAL(d_cbR_y(0:weno_num_stencils, is2_weno%beg + weno_polyn:is2_weno%end - weno_polyn))

        @:ALLOCATE_GLOBAL(beta_coef_y(is2_weno%beg + weno_polyn:is2_weno%end - weno_polyn, 0:weno_polyn, &
            0:2*(weno_polyn - 1)))

        call s_compute_weno_coefficients(2, is2_weno)

        @:ALLOCATE_GLOBAL(v_rs_ws_y(is2_weno%beg:is2_weno%end, &
            is1_weno%beg:is1_weno%end, is3_weno%beg:is3_weno%end, 1:sys_size))

        ! ==================================================================

        ! Allocating/Computing WENO Coefficients in z-direction ============
        if (p == 0) return

        is2_weno%beg = -buff_size; is2_weno%end = n - is2_weno%beg
        is1_weno%beg = -buff_size; is1_weno%end = m - is1_weno%beg
        is3_weno%beg = -buff_size; is3_weno%end = p - is3_weno%beg

        @:ALLOCATE_GLOBAL(poly_coef_cbL_z(is3_weno%beg + weno_polyn:is3_weno%end - weno_polyn, 0:weno_polyn, &
            0:weno_polyn - 1))
        @:ALLOCATE_GLOBAL(poly_coef_cbR_z(is3_weno%beg + weno_polyn:is3_weno%end - weno_polyn, 0:weno_polyn, &
            0:weno_polyn - 1))

        @:ALLOCATE_GLOBAL(d_cbL_z(0:weno_num_stencils, is3_weno%beg + weno_polyn:is3_weno%end - weno_polyn))
        @:ALLOCATE_GLOBAL(d_cbR_z(0:weno_num_stencils, is3_weno%beg + weno_polyn:is3_weno%end - weno_polyn))

        @:ALLOCATE_GLOBAL(beta_coef_z(is3_weno%beg + weno_polyn:is3_weno%end - weno_polyn, 0:weno_polyn, &
            0:2*(weno_polyn - 1)))

        call s_compute_weno_coefficients(3, is3_weno)

        @:ALLOCATE_GLOBAL(v_rs_ws_z(is3_weno%beg:is3_weno%end, &
            is2_weno%beg:is2_weno%end, is1_weno%beg:is1_weno%end, 1:sys_size))

        ! ==================================================================

    end subroutine s_initialize_weno_module

    !>  The purpose of this subroutine is to compute the grid
        !!      dependent coefficients of the WENO polynomials, ideal
        !!      weights and smoothness indicators, provided the order,
        !!      the coordinate direction and the location of the WENO
        !!      reconstruction.
        !! @param weno_dir Coordinate direction of the WENO reconstruction
        !! @param is Index bounds in the s-direction
    subroutine s_compute_weno_coefficients(weno_dir, is)

        integer, intent(in) :: weno_dir
        type(int_bounds_info), intent(in) :: is
        integer :: s

        real(kind(0d0)), pointer, dimension(:) :: s_cb => null() !<
            !! Cell-boundary locations in the s-direction

        type(int_bounds_info) :: bc_s !< Boundary conditions (BC) in the s-direction

        integer :: i !< Generic loop iterator

        ! Determining the number of cells, the cell-boundary locations and
        ! the boundary conditions in the coordinate direction selected for
        ! the WENO reconstruction
        if (weno_dir == 1) then
            s = m; s_cb => x_cb; bc_s = bc_x
        elseif (weno_dir == 2) then
            s = n; s_cb => y_cb; bc_s = bc_y
        else
            s = p; s_cb => z_cb; bc_s = bc_z
        end if

        #:for WENO_DIR, XYZ in [(1, 'x'), (2, 'y'), (3, 'z')]
            ! Computing WENO3 Coefficients =====================================
            if (weno_dir == ${WENO_DIR}$) then
                if (weno_order == 3) then
                    do i = is%beg - 1 + weno_polyn, is%end - 1 - weno_polyn

                        poly_coef_cbR_${XYZ}$ (i + 1, 0, 0) = (s_cb(i) - s_cb(i + 1))/ &
                                                              (s_cb(i) - s_cb(i + 2))
                        poly_coef_cbR_${XYZ}$ (i + 1, 1, 0) = (s_cb(i) - s_cb(i + 1))/ &
                                                              (s_cb(i - 1) - s_cb(i + 1))

                        poly_coef_cbL_${XYZ}$ (i + 1, 0, 0) = -poly_coef_cbR_${XYZ}$ (i + 1, 0, 0)
                        poly_coef_cbL_${XYZ}$ (i + 1, 1, 0) = -poly_coef_cbR_${XYZ}$ (i + 1, 1, 0)

                        d_cbR_${XYZ}$ (0, i + 1) = (s_cb(i - 1) - s_cb(i + 1))/ &
                                                   (s_cb(i - 1) - s_cb(i + 2))
                        d_cbL_${XYZ}$ (0, i + 1) = (s_cb(i - 1) - s_cb(i))/ &
                                                   (s_cb(i - 1) - s_cb(i + 2))

                        d_cbR_${XYZ}$ (1, i + 1) = 1d0 - d_cbR_${XYZ}$ (0, i + 1)
                        d_cbL_${XYZ}$ (1, i + 1) = 1d0 - d_cbL_${XYZ}$ (0, i + 1)

                        beta_coef_${XYZ}$ (i + 1, 0, 0) = 4d0*(s_cb(i) - s_cb(i + 1))**2d0/ &
                                                          (s_cb(i) - s_cb(i + 2))**2d0
                        beta_coef_${XYZ}$ (i + 1, 1, 0) = 4d0*(s_cb(i) - s_cb(i + 1))**2d0/ &
                                                          (s_cb(i - 1) - s_cb(i + 1))**2d0

                    end do

                    ! Modifying the ideal weights coefficients in the neighborhood
                    ! of beginning and end Riemann state extrapolation BC to avoid
                    ! any contributions from outside of the physical domain during
                    ! the WENO reconstruction
                    if (null_weights) then
                        if (bc_s%beg == -4) then
                            d_cbR_${XYZ}$ (1, 0) = 0d0; d_cbR_${XYZ}$ (0, 0) = 1d0
                            d_cbL_${XYZ}$ (1, 0) = 0d0; d_cbL_${XYZ}$ (0, 0) = 1d0
                        end if

                        if (bc_s%end == -4) then
                            d_cbR_${XYZ}$ (0, s) = 0d0; d_cbR_${XYZ}$ (1, s) = 1d0
                            d_cbL_${XYZ}$ (0, s) = 0d0; d_cbL_${XYZ}$ (1, s) = 1d0
                        end if
                    end if
                    ! END: Computing WENO3 Coefficients ================================

                    ! Computing WENO5 Coefficients =====================================
                elseif (weno_order == 5) then

                    do i = is%beg - 1 + weno_polyn, is%end - 1 - weno_polyn

                        poly_coef_cbR_${XYZ}$ (i + 1, 0, 0) = &
                            ((s_cb(i) - s_cb(i + 1))*(s_cb(i + 1) - s_cb(i + 2)))/ &
                            ((s_cb(i) - s_cb(i + 3))*(s_cb(i + 3) - s_cb(i + 1)))
                        poly_coef_cbR_${XYZ}$ (i + 1, 1, 0) = &
                            ((s_cb(i - 1) - s_cb(i + 1))*(s_cb(i + 1) - s_cb(i)))/ &
                            ((s_cb(i - 1) - s_cb(i + 2))*(s_cb(i + 2) - s_cb(i)))
                        poly_coef_cbR_${XYZ}$ (i + 1, 1, 1) = &
                            ((s_cb(i) - s_cb(i + 1))*(s_cb(i + 1) - s_cb(i + 2)))/ &
                            ((s_cb(i - 1) - s_cb(i + 1))*(s_cb(i - 1) - s_cb(i + 2)))
                        poly_coef_cbR_${XYZ}$ (i + 1, 2, 1) = &
                            ((s_cb(i) - s_cb(i + 1))*(s_cb(i + 1) - s_cb(i - 1)))/ &
                            ((s_cb(i - 2) - s_cb(i))*(s_cb(i - 2) - s_cb(i + 1)))
                        poly_coef_cbL_${XYZ}$ (i + 1, 0, 0) = &
                            ((s_cb(i + 1) - s_cb(i))*(s_cb(i) - s_cb(i + 2)))/ &
                            ((s_cb(i) - s_cb(i + 3))*(s_cb(i + 3) - s_cb(i + 1)))
                        poly_coef_cbL_${XYZ}$ (i + 1, 1, 0) = &
                            ((s_cb(i) - s_cb(i - 1))*(s_cb(i) - s_cb(i + 1)))/ &
                            ((s_cb(i - 1) - s_cb(i + 2))*(s_cb(i) - s_cb(i + 2)))
                        poly_coef_cbL_${XYZ}$ (i + 1, 1, 1) = &
                            ((s_cb(i + 1) - s_cb(i))*(s_cb(i) - s_cb(i + 2)))/ &
                            ((s_cb(i - 1) - s_cb(i + 1))*(s_cb(i - 1) - s_cb(i + 2)))
                        poly_coef_cbL_${XYZ}$ (i + 1, 2, 1) = &
                            ((s_cb(i - 1) - s_cb(i))*(s_cb(i) - s_cb(i + 1)))/ &
                            ((s_cb(i - 2) - s_cb(i))*(s_cb(i - 2) - s_cb(i + 1)))

                        poly_coef_cbR_${XYZ}$ (i + 1, 0, 1) = &
                            ((s_cb(i) - s_cb(i + 2)) + (s_cb(i + 1) - s_cb(i + 3)))/ &
                            ((s_cb(i) - s_cb(i + 2))*(s_cb(i) - s_cb(i + 3)))* &
                            ((s_cb(i) - s_cb(i + 1)))
                        poly_coef_cbR_${XYZ}$ (i + 1, 2, 0) = &
                            ((s_cb(i - 2) - s_cb(i + 1)) + (s_cb(i - 1) - s_cb(i + 1)))/ &
                            ((s_cb(i - 1) - s_cb(i + 1))*(s_cb(i + 1) - s_cb(i - 2)))* &
                            ((s_cb(i + 1) - s_cb(i)))
                        poly_coef_cbL_${XYZ}$ (i + 1, 0, 1) = &
                            ((s_cb(i) - s_cb(i + 2)) + (s_cb(i) - s_cb(i + 3)))/ &
                            ((s_cb(i) - s_cb(i + 2))*(s_cb(i) - s_cb(i + 3)))* &
                            ((s_cb(i + 1) - s_cb(i)))
                        poly_coef_cbL_${XYZ}$ (i + 1, 2, 0) = &
                            ((s_cb(i - 2) - s_cb(i)) + (s_cb(i - 1) - s_cb(i + 1)))/ &
                            ((s_cb(i - 2) - s_cb(i + 1))*(s_cb(i + 1) - s_cb(i - 1)))* &
                            ((s_cb(i) - s_cb(i + 1)))

                        d_cbR_${XYZ}$ (0, i + 1) = &
                            ((s_cb(i - 2) - s_cb(i + 1))*(s_cb(i + 1) - s_cb(i - 1)))/ &
                            ((s_cb(i - 2) - s_cb(i + 3))*(s_cb(i + 3) - s_cb(i - 1)))
                        d_cbR_${XYZ}$ (2, i + 1) = &
                            ((s_cb(i + 1) - s_cb(i + 2))*(s_cb(i + 1) - s_cb(i + 3)))/ &
                            ((s_cb(i - 2) - s_cb(i + 2))*(s_cb(i - 2) - s_cb(i + 3)))
                        d_cbL_${XYZ}$ (0, i + 1) = &
                            ((s_cb(i - 2) - s_cb(i))*(s_cb(i) - s_cb(i - 1)))/ &
                            ((s_cb(i - 2) - s_cb(i + 3))*(s_cb(i + 3) - s_cb(i - 1)))
                        d_cbL_${XYZ}$ (2, i + 1) = &
                            ((s_cb(i) - s_cb(i + 2))*(s_cb(i) - s_cb(i + 3)))/ &
                            ((s_cb(i - 2) - s_cb(i + 2))*(s_cb(i - 2) - s_cb(i + 3)))

                        d_cbR_${XYZ}$ (1, i + 1) = 1d0 - d_cbR_${XYZ}$ (0, i + 1) - d_cbR_${XYZ}$ (2, i + 1)
                        d_cbL_${XYZ}$ (1, i + 1) = 1d0 - d_cbL_${XYZ}$ (0, i + 1) - d_cbL_${XYZ}$ (2, i + 1)

                        beta_coef_${XYZ}$ (i + 1, 0, 0) = &
                            4d0*(s_cb(i) - s_cb(i + 1))**2d0*(10d0*(s_cb(i + 1) - &
                                                                    s_cb(i))**2d0 + (s_cb(i + 1) - s_cb(i))*(s_cb(i + 2) - &
                                                                                                             s_cb(i + 1)) + (s_cb(i + 2) - s_cb(i + 1))**2d0)/((s_cb(i) - &
                                                                                                                                                                s_cb(i + 3))**2d0*(s_cb(i + 1) - s_cb(i + 3))**2d0)

                        beta_coef_${XYZ}$ (i + 1, 0, 1) = &
                            4d0*(s_cb(i) - s_cb(i + 1))**2d0*(19d0*(s_cb(i + 1) - &
                                                                    s_cb(i))**2d0 - (s_cb(i + 1) - s_cb(i))*(s_cb(i + 3) - &
                                                                                                             s_cb(i + 1)) + 2d0*(s_cb(i + 2) - s_cb(i))*((s_cb(i + 2) - &
                                                                                                                                                          s_cb(i)) + (s_cb(i + 3) - s_cb(i + 1))))/((s_cb(i) - &
                                                                                                                                                                                                     s_cb(i + 2))*(s_cb(i) - s_cb(i + 3))**2d0*(s_cb(i + 3) - &
                                                                                                                                                                                                                                                s_cb(i + 1)))

                        beta_coef_${XYZ}$ (i + 1, 0, 2) = &
                            4d0*(s_cb(i) - s_cb(i + 1))**2d0*(10d0*(s_cb(i + 1) - &
                                                                    s_cb(i))**2d0 + (s_cb(i + 1) - s_cb(i))*((s_cb(i + 2) - &
                                                                                                              s_cb(i)) + (s_cb(i + 3) - s_cb(i + 1))) + ((s_cb(i + 2) - &
                                                                                                                                                          s_cb(i)) + (s_cb(i + 3) - s_cb(i + 1)))**2d0)/((s_cb(i) - &
                                                                                                                                                                                                          s_cb(i + 2))**2d0*(s_cb(i) - s_cb(i + 3))**2d0)

                        beta_coef_${XYZ}$ (i + 1, 1, 0) = &
                            4d0*(s_cb(i) - s_cb(i + 1))**2d0*(10d0*(s_cb(i + 1) - &
                                                                    s_cb(i))**2d0 + (s_cb(i) - s_cb(i - 1))**2d0 + (s_cb(i) - &
                                                                                                                    s_cb(i - 1))*(s_cb(i + 1) - s_cb(i)))/((s_cb(i - 1) - &
                                                                                                                                                            s_cb(i + 2))**2d0*(s_cb(i) - s_cb(i + 2))**2d0)

                        beta_coef_${XYZ}$ (i + 1, 1, 1) = &
                            4d0*(s_cb(i) - s_cb(i + 1))**2d0*((s_cb(i) - &
                                                               s_cb(i + 1))*((s_cb(i) - s_cb(i - 1)) + 20d0*(s_cb(i + 1) - &
                                                                                                             s_cb(i))) + (2d0*(s_cb(i) - s_cb(i - 1)) + (s_cb(i + 1) - &
                                                                                                                                                         s_cb(i)))*(s_cb(i + 2) - s_cb(i)))/((s_cb(i + 1) - &
                                                                                                                                                                                              s_cb(i - 1))*(s_cb(i - 1) - s_cb(i + 2))**2d0*(s_cb(i + 2) - &
                                                                                                                                                                                                                                             s_cb(i)))

                        beta_coef_${XYZ}$ (i + 1, 1, 2) = &
                            4d0*(s_cb(i) - s_cb(i + 1))**2d0*(10d0*(s_cb(i + 1) - &
                                                                    s_cb(i))**2d0 + (s_cb(i + 1) - s_cb(i))*(s_cb(i + 2) - &
                                                                                                             s_cb(i + 1)) + (s_cb(i + 2) - s_cb(i + 1))**2d0)/ &
                            ((s_cb(i - 1) - s_cb(i + 1))**2d0*(s_cb(i - 1) - &
                                                               s_cb(i + 2))**2d0)

                        beta_coef_${XYZ}$ (i + 1, 2, 0) = &
                            4d0*(s_cb(i) - s_cb(i + 1))**2d0*(12d0*(s_cb(i + 1) - &
                                                                    s_cb(i))**2d0 + ((s_cb(i) - s_cb(i - 2)) + (s_cb(i) - &
                                                                                                                s_cb(i - 1)))**2d0 + 3d0*((s_cb(i) - s_cb(i - 2)) + &
                                                                                                                                          (s_cb(i) - s_cb(i - 1)))*(s_cb(i + 1) - s_cb(i)))/ &
                            ((s_cb(i - 2) - s_cb(i + 1))**2d0*(s_cb(i - 1) - &
                                                               s_cb(i + 1))**2d0)

                        beta_coef_${XYZ}$ (i + 1, 2, 1) = &
                            4d0*(s_cb(i) - s_cb(i + 1))**2d0*(19d0*(s_cb(i + 1) - &
                                                                    s_cb(i))**2d0 + ((s_cb(i) - s_cb(i - 2))*(s_cb(i) - &
                                                                                                              s_cb(i + 1))) + 2d0*(s_cb(i + 1) - s_cb(i - 1))*((s_cb(i) - &
                                                                                                                                                                s_cb(i - 2)) + (s_cb(i + 1) - s_cb(i - 1))))/((s_cb(i - 2) - &
                                                                                                                                                                                                               s_cb(i))*(s_cb(i - 2) - s_cb(i + 1))**2d0*(s_cb(i + 1) - &
                                                                                                                                                                                                                                                          s_cb(i - 1)))

                        beta_coef_${XYZ}$ (i + 1, 2, 2) = &
                            4d0*(s_cb(i) - s_cb(i + 1))**2d0*(10d0*(s_cb(i + 1) - &
                                                                    s_cb(i))**2d0 + (s_cb(i) - s_cb(i - 1))**2d0 + (s_cb(i) - &
                                                                                                                    s_cb(i - 1))*(s_cb(i + 1) - s_cb(i)))/((s_cb(i - 2) - &
                                                                                                                                                            s_cb(i))**2d0*(s_cb(i - 2) - s_cb(i + 1))**2d0)

                    end do

                    ! Modifying the ideal weights coefficients in the neighborhood
                    ! of beginning and end Riemann state extrapolation BC to avoid
                    ! any contributions from outside of the physical domain during
                    ! the WENO reconstruction
                    if (null_weights) then
                        if (bc_s%beg == -4) then
                            d_cbR_${XYZ}$ (1:2, 0) = 0d0; d_cbR_${XYZ}$ (0, 0) = 1d0
                            d_cbL_${XYZ}$ (1:2, 0) = 0d0; d_cbL_${XYZ}$ (0, 0) = 1d0
                            d_cbR_${XYZ}$ (2, 1) = 0d0; d_cbR_${XYZ}$ (:, 1) = d_cbR_${XYZ}$ (:, 1)/sum(d_cbR_${XYZ}$ (:, 1))
                            d_cbL_${XYZ}$ (2, 1) = 0d0; d_cbL_${XYZ}$ (:, 1) = d_cbL_${XYZ}$ (:, 1)/sum(d_cbL_${XYZ}$ (:, 1))
                        end if

                        if (bc_s%end == -4) then
                            d_cbR_${XYZ}$ (0, s - 1) = 0d0; d_cbR_${XYZ}$ (:, s - 1) = d_cbR_${XYZ}$ (:, s - 1)/sum(d_cbR_${XYZ}$ (:, s - 1))
                            d_cbL_${XYZ}$ (0, s - 1) = 0d0; d_cbL_${XYZ}$ (:, s - 1) = d_cbL_${XYZ}$ (:, s - 1)/sum(d_cbL_${XYZ}$ (:, s - 1))
                            d_cbR_${XYZ}$ (0:1, s) = 0d0; d_cbR_${XYZ}$ (2, s) = 1d0
                            d_cbL_${XYZ}$ (0:1, s) = 0d0; d_cbL_${XYZ}$ (2, s) = 1d0
                        end if
                    end if

                else ! WENO7
                    ! Note: WENO7 only supports uniform grid
                    if (.not. teno) then
                        ! (Balsara & Shu, 2000) Page 11 Section III.a
                        d_cbL_${XYZ}$ (0, :) = 4d0/35d0
                        d_cbL_${XYZ}$ (1, :) = 18d0/35d0
                        d_cbL_${XYZ}$ (2, :) = 12d0/35d0
                        d_cbL_${XYZ}$ (3, :) = 1d0/35d0

                        d_cbR_${XYZ}$ (0, :) = 1d0/35d0
                        d_cbR_${XYZ}$ (1, :) = 12d0/35d0
                        d_cbR_${XYZ}$ (2, :) = 18d0/35d0
                        d_cbR_${XYZ}$ (3, :) = 4d0/35d0

                    else ! TENO
                        ! (Fu, et al., 2016) Table 2 (for right flux)
                        d_cbL_${XYZ}$ (0, :) = 18d0/35d0
                        d_cbL_${XYZ}$ (1, :) = 3d0/35d0
                        d_cbL_${XYZ}$ (2, :) = 9d0/35d0
                        d_cbL_${XYZ}$ (3, :) = 1d0/35d0
                        d_cbL_${XYZ}$ (4, :) = 4d0/35d0

                        d_cbR_${XYZ}$ (0, :) = 18d0/35d0
                        d_cbR_${XYZ}$ (1, :) = 9d0/35d0
                        d_cbR_${XYZ}$ (2, :) = 3d0/35d0
                        d_cbR_${XYZ}$ (3, :) = 4d0/35d0
                        d_cbR_${XYZ}$ (4, :) = 1d0/35d0

                    end if
                end if

            end if
        #:endfor

! END: Computing WENO Coefficients ================================
        if (weno_dir == 1) then
            !$acc update device(poly_coef_cbL_x, poly_coef_cbR_x, d_cbL_x, d_cbR_x, beta_coef_x)
        elseif (weno_dir == 2) then
            !$acc update device(poly_coef_cbL_y, poly_coef_cbR_y, d_cbL_y, d_cbR_y, beta_coef_y)
        else
            !$acc update device(poly_coef_cbL_z, poly_coef_cbR_z, d_cbL_z, d_cbR_z, beta_coef_z)
        end if

        ! Nullifying WENO coefficients and cell-boundary locations pointers

        nullify (s_cb)

    end subroutine s_compute_weno_coefficients

    subroutine s_weno(v_vf, vL_rs_vf_x, vL_rs_vf_y, vL_rs_vf_z, vR_rs_vf_x, vR_rs_vf_y, vR_rs_vf_z, &
                      norm_dir, weno_dir, &
                      is1_weno_d, is2_weno_d, is3_weno_d)

        type(scalar_field), dimension(1:), intent(in) :: v_vf
        real(kind(0d0)), dimension(startx:, starty:, startz:, 1:), intent(inout) :: vL_rs_vf_x, vL_rs_vf_y, vL_rs_vf_z
        real(kind(0d0)), dimension(startx:, starty:, startz:, 1:), intent(inout) :: vR_rs_vf_x, vR_rs_vf_y, vR_rs_vf_z
        integer, intent(in) :: norm_dir
        integer, intent(in) :: weno_dir
        type(int_bounds_info), intent(in) :: is1_weno_d, is2_weno_d, is3_weno_d

        real(kind(0d0)), dimension(-weno_polyn:weno_polyn - 1) :: dvd
        real(kind(0d0)), dimension(0:weno_num_stencils) :: poly
        real(kind(0d0)), dimension(0:weno_num_stencils) :: alpha
        real(kind(0d0)), dimension(0:weno_num_stencils) :: omega
        real(kind(0d0)), dimension(0:weno_num_stencils) :: beta
        real(kind(0d0)), dimension(0:weno_num_stencils) :: delta
        real(kind(0d0)), dimension(-3:3) :: v ! temporary field value array for clarity (WENO7 only)
        real(kind(0d0)) :: tau

        integer :: i, j, k, l

        is1_weno = is1_weno_d
        is2_weno = is2_weno_d
        is3_weno = is3_weno_d

        !$acc update device(is1_weno, is2_weno, is3_weno)

        if (weno_order /= 1) then
            call s_initialize_weno(v_vf, &
                                   norm_dir, weno_dir)
        end if

        if (weno_order == 1) then
            if (weno_dir == 1) then
                !$acc parallel loop collapse(4) default(present)
                do i = 1, ubound(v_vf, 1)
                    do l = is3_weno%beg, is3_weno%end
                        do k = is2_weno%beg, is2_weno%end
                            do j = is1_weno%beg, is1_weno%end
                                vL_rs_vf_x(j, k, l, i) = v_vf(i)%sf(j, k, l)
                                vR_rs_vf_x(j, k, l, i) = v_vf(i)%sf(j, k, l)
                            end do
                        end do
                    end do
                end do
                !$acc end parallel loop
            else if (weno_dir == 2) then
                !$acc parallel loop collapse(4) default(present)
                do i = 1, ubound(v_vf, 1)
                    do l = is3_weno%beg, is3_weno%end
                        do k = is2_weno%beg, is2_weno%end
                            do j = is1_weno%beg, is1_weno%end
                                vL_rs_vf_y(j, k, l, i) = v_vf(i)%sf(k, j, l)
                                vR_rs_vf_y(j, k, l, i) = v_vf(i)%sf(k, j, l)
                            end do
                        end do
                    end do
                end do
                !$acc end parallel loop
            else if (weno_dir == 3) then
                !$acc parallel loop collapse(4) default(present)
                do i = 1, ubound(v_vf, 1)
                    do l = is3_weno%beg, is3_weno%end
                        do k = is2_weno%beg, is2_weno%end
                            do j = is1_weno%beg, is1_weno%end
                                vL_rs_vf_z(j, k, l, i) = v_vf(i)%sf(l, k, j)
                                vR_rs_vf_z(j, k, l, i) = v_vf(i)%sf(l, k, j)
                            end do
                        end do
                    end do
                end do
                !$acc end parallel loop
            end if
        elseif (weno_order == 3) then
            #:for WENO_DIR, XYZ in [(1, 'x'), (2, 'y'), (3, 'z')]
                if (weno_dir == ${WENO_DIR}$) then
                    !$acc parallel loop collapse(4) gang vector default(present) private(beta,dvd,poly,omega,alpha,tau)
                    do l = is3_weno%beg, is3_weno%end
                        do k = is2_weno%beg, is2_weno%end
                            do j = is1_weno%beg, is1_weno%end
                                do i = 1, v_size
                                    ! reconstruct from left side

                                    dvd(0) = v_rs_ws_${XYZ}$ (j + 1, k, l, i) &
                                             - v_rs_ws_${XYZ}$ (j, k, l, i)
                                    dvd(-1) = v_rs_ws_${XYZ}$ (j, k, l, i) &
                                              - v_rs_ws_${XYZ}$ (j - 1, k, l, i)

                                    poly(0) = v_rs_ws_${XYZ}$ (j, k, l, i) &
                                              + poly_coef_cbL_${XYZ}$ (j, 0, 0)*dvd(0)
                                    poly(1) = v_rs_ws_${XYZ}$ (j, k, l, i) &
                                              + poly_coef_cbL_${XYZ}$ (j, 1, 0)*dvd(-1)

                                    beta(0) = beta_coef_${XYZ}$ (j, 0, 0)*dvd(0)*dvd(0) &
                                              + weno_eps
                                    beta(1) = beta_coef_${XYZ}$ (j, 1, 0)*dvd(-1)*dvd(-1) &
                                              + weno_eps

                                    if (wenojs) then
                                        alpha = d_cbL_${XYZ}$ (:, j)/(beta*beta)

                                    elseif (mapped_weno) then
                                        alpha = d_cbL_${XYZ}$ (:, j)/(beta*beta)
                                        omega = alpha/sum(alpha)
                                        alpha = (d_cbL_${XYZ}$ (:, j)*(1d0 + d_cbL_${XYZ}$ (:, j) - 3d0*omega) + omega**2d0) &
                                                *(omega/(d_cbL_${XYZ}$ (:, j)**2d0 + omega*(1d0 - 2d0*d_cbL_${XYZ}$ (:, j))))

                                    elseif (wenoz) then
                                        ! Borges, et al. (2008)
                                        tau = abs(beta(1) - beta(0))
                                        alpha = d_cbL_${XYZ}$ (:, j)*(1d0 + tau/beta)

                                    end if

                                    omega = alpha/sum(alpha)

                                    vL_rs_vf_${XYZ}$ (j, k, l, i) = omega(0)*poly(0) + omega(1)*poly(1)

                                    ! reconstruct from right side

                                    poly(0) = v_rs_ws_${XYZ}$ (j, k, l, i) &
                                              + poly_coef_cbR_${XYZ}$ (j, 0, 0)*dvd(0)
                                    poly(1) = v_rs_ws_${XYZ}$ (j, k, l, i) &
                                              + poly_coef_cbR_${XYZ}$ (j, 1, 0)*dvd(-1)

                                    if (wenojs) then
                                        alpha = d_cbR_${XYZ}$ (:, j)/(beta*beta)

                                    elseif (mapped_weno) then
                                        alpha = d_cbR_${XYZ}$ (:, j)/(beta*beta)
                                        omega = alpha/sum(alpha)
                                        alpha = (d_cbR_${XYZ}$ (:, j)*(1d0 + d_cbR_${XYZ}$ (:, j) - 3d0*omega) + omega**2d0) &
                                                *(omega/(d_cbR_${XYZ}$ (:, j)**2d0 + omega*(1d0 - 2d0*d_cbR_${XYZ}$ (:, j))))

                                    elseif (wenoz) then
                                        alpha = d_cbR_${XYZ}$ (:, j)*(1d0 + tau/beta)

                                    end if

                                    omega = alpha/sum(alpha)

                                    vR_rs_vf_${XYZ}$ (j, k, l, i) = omega(0)*poly(0) + omega(1)*poly(1)

                                end do
                            end do
                        end do
                    end do
                    !$acc end parallel loop
                end if
            #:endfor
        elseif (weno_order == 5) then
            #:for WENO_DIR, XYZ in [(1, 'x'), (2, 'y'), (3, 'z')]
                if (weno_dir == ${WENO_DIR}$) then
                    !$acc parallel loop vector gang collapse(3) default(present) private(dvd, poly, beta, alpha, omega, tau, delta)
                    do l = is3_weno%beg, is3_weno%end
                        do k = is2_weno%beg, is2_weno%end
                            do j = is1_weno%beg, is1_weno%end
                                !$acc loop seq
                                do i = 1, v_size
                                    ! reconstruct from left side

                                    dvd(1) = v_rs_ws_${XYZ}$ (j + 2, k, l, i) &
                                             - v_rs_ws_${XYZ}$ (j + 1, k, l, i)
                                    dvd(0) = v_rs_ws_${XYZ}$ (j + 1, k, l, i) &
                                             - v_rs_ws_${XYZ}$ (j, k, l, i)
                                    dvd(-1) = v_rs_ws_${XYZ}$ (j, k, l, i) &
                                              - v_rs_ws_${XYZ}$ (j - 1, k, l, i)
                                    dvd(-2) = v_rs_ws_${XYZ}$ (j - 1, k, l, i) &
                                              - v_rs_ws_${XYZ}$ (j - 2, k, l, i)

                                    poly(0) = v_rs_ws_${XYZ}$ (j, k, l, i) &
                                              + poly_coef_cbL_${XYZ}$ (j, 0, 0)*dvd(1) &
                                              + poly_coef_cbL_${XYZ}$ (j, 0, 1)*dvd(0)
                                    poly(1) = v_rs_ws_${XYZ}$ (j, k, l, i) &
                                              + poly_coef_cbL_${XYZ}$ (j, 1, 0)*dvd(0) &
                                              + poly_coef_cbL_${XYZ}$ (j, 1, 1)*dvd(-1)
                                    poly(2) = v_rs_ws_${XYZ}$ (j, k, l, i) &
                                              + poly_coef_cbL_${XYZ}$ (j, 2, 0)*dvd(-1) &
                                              + poly_coef_cbL_${XYZ}$ (j, 2, 1)*dvd(-2)

                                    beta(0) = beta_coef_${XYZ}$ (j, 0, 0)*dvd(1)*dvd(1) &
                                              + beta_coef_${XYZ}$ (j, 0, 1)*dvd(1)*dvd(0) &
                                              + beta_coef_${XYZ}$ (j, 0, 2)*dvd(0)*dvd(0) &
                                              + weno_eps
                                    beta(1) = beta_coef_${XYZ}$ (j, 1, 0)*dvd(0)*dvd(0) &
                                              + beta_coef_${XYZ}$ (j, 1, 1)*dvd(0)*dvd(-1) &
                                              + beta_coef_${XYZ}$ (j, 1, 2)*dvd(-1)*dvd(-1) &
                                              + weno_eps
                                    beta(2) = beta_coef_${XYZ}$ (j, 2, 0)*dvd(-1)*dvd(-1) &
                                              + beta_coef_${XYZ}$ (j, 2, 1)*dvd(-1)*dvd(-2) &
                                              + beta_coef_${XYZ}$ (j, 2, 2)*dvd(-2)*dvd(-2) &
                                              + weno_eps

                                    if (wenojs) then
                                        alpha = d_cbL_${XYZ}$ (:, j)/(beta*beta)

                                    elseif (mapped_weno) then
                                        alpha = d_cbL_${XYZ}$ (:, j)/(beta*beta)
                                        omega = alpha/sum(alpha)
                                        alpha = (d_cbL_${XYZ}$ (:, j)*(1d0 + d_cbL_${XYZ}$ (:, j) - 3d0*omega) + omega**2d0) &
                                                *(omega/(d_cbL_${XYZ}$ (:, j)**2d0 + omega*(1d0 - 2d0*d_cbL_${XYZ}$ (:, j))))

                                    elseif (wenoz) then
                                        ! Borges, et al. (2008)
                                        tau = abs(beta(2) - beta(0))                   ! Equation 25
                                        alpha = d_cbL_${XYZ}$ (:, j)*(1d0 + tau/beta)  ! Equation 28 (note: weno_eps was already added to beta)

                                    elseif (teno) then
                                        ! Fu, et al. (2016)
                                        ! Fu''s code: https://dx.doi.org/10.13140/RG.2.2.36250.34247
                                        tau = abs(beta(2) - beta(0))
                                        alpha = (1d0 + tau/beta)**6d0              ! Equation 22 (reuse alpha as gamma; pick C=1 & q=6)
                                        omega = alpha/sum(alpha)                    ! Equation 25 (reuse omega as xi)
                                        delta = merge(0d0, 1d0, omega < teno_CT)    ! Equation 26
                                        alpha = delta*d_cbL_${XYZ}$ (:, j)          ! Equation 27

                                    end if

                                    omega = alpha/sum(alpha)

                                    vL_rs_vf_${XYZ}$ (j, k, l, i) = sum(omega*poly)

                                    ! reconstruct from right side

                                    poly(0) = v_rs_ws_${XYZ}$ (j, k, l, i) &
                                              + poly_coef_cbR_${XYZ}$ (j, 0, 0)*dvd(1) &
                                              + poly_coef_cbR_${XYZ}$ (j, 0, 1)*dvd(0)
                                    poly(1) = v_rs_ws_${XYZ}$ (j, k, l, i) &
                                              + poly_coef_cbR_${XYZ}$ (j, 1, 0)*dvd(0) &
                                              + poly_coef_cbR_${XYZ}$ (j, 1, 1)*dvd(-1)
                                    poly(2) = v_rs_ws_${XYZ}$ (j, k, l, i) &
                                              + poly_coef_cbR_${XYZ}$ (j, 2, 0)*dvd(-1) &
                                              + poly_coef_cbR_${XYZ}$ (j, 2, 1)*dvd(-2)

                                    if (wenojs) then
                                        alpha = d_cbR_${XYZ}$ (:, j)/(beta*beta)

                                    elseif (mapped_weno) then
                                        alpha = d_cbR_${XYZ}$ (:, j)/(beta*beta)
                                        omega = alpha/sum(alpha)
                                        alpha = (d_cbR_${XYZ}$ (:, j)*(1d0 + d_cbR_${XYZ}$ (:, j) - 3d0*omega) + omega**2d0) &
                                                *(omega/(d_cbR_${XYZ}$ (:, j)**2d0 + omega*(1d0 - 2d0*d_cbR_${XYZ}$ (:, j))))

                                    elseif (wenoz) then
                                        alpha = d_cbR_${XYZ}$ (:, j)*(1d0 + tau/beta)

                                    elseif (teno) then
                                        alpha = delta*d_cbR_${XYZ}$ (:, j)

                                    end if

                                    omega = alpha/sum(alpha)

                                    vR_rs_vf_${XYZ}$ (j, k, l, i) = sum(omega*poly)

                                end do
                            end do
                        end do
                    end do
                    !$acc end parallel loop

                    if (mp_weno) then
                        call s_preserve_monotonicity(v_rs_ws_${XYZ}$, vL_rs_vf_${XYZ}$, &
                                                     vR_rs_vf_${XYZ}$)
                    end if
                end if
            #:endfor
        elseif (weno_order == 7) then
            #:for WENO_DIR, XYZ in [(1, 'x'), (2, 'y'), (3, 'z')]
                if (weno_dir == ${WENO_DIR}$) then
                    !$acc parallel loop vector gang collapse(3) default(present) private(poly, beta, alpha, omega, tau, delta, v)
                    ! Note: dvd is not used as the equations are not cast in terms of the differences
                    do l = is3_weno%beg, is3_weno%end
                        do k = is2_weno%beg, is2_weno%end
                            do j = is1_weno%beg, is1_weno%end
                                !$acc loop seq
                                do i = 1, v_size

                                    v = v_rs_ws_${XYZ}$ (j - 3:j + 3, k, l, i) ! temporary field value array for clarity

                                    if (.not. teno) then
                                        ! (Balsara & Shu, 2000) Page 11 Table I
                                        poly(0) = ( 1d0*v(-3) -  5d0*v(-2) + 13d0*v(-1) + 3d0*v( 0)) / 12d0 !&
                                        poly(1) = (-1d0*v(-2) +  7d0*v(-1) +  7d0*v( 0) - 1d0*v( 1)) / 12d0 !&
                                        poly(2) = ( 3d0*v(-1) + 13d0*v( 0) -  5d0*v( 1) + 1d0*v( 2)) / 12d0 !&
                                        poly(3) = (25d0*v( 0) - 23d0*v( 1) + 13d0*v( 2) - 3d0*v( 3)) / 12d0 !&
                                    else
                                        ! (Fu, et al., 2016) Table 1
                                        ! Note: Unlike TENO5, TENO7 stencils differ from WENO7 stencils
                                        ! See Figure 2 (right) for right-sided flux (at i+1/2)
                                        ! Here we need the left-sided flux, so we flip the weights with respect to the x=i point
                                        ! But we need to keep the stencil order to reuse the beta coefficients
                                        poly(0) = ( 2d0*v(-1) +  5d0*v( 0) -  1d0*v( 1)) / 6d0 !&
                                        poly(1) = (11d0*v( 0) -  7d0*v( 1) +  2d0*v( 2)) / 6d0 !&
                                        poly(2) = (-1d0*v(-2) +  5d0*v(-1) +  2d0*v( 0)) / 6d0 !&
                                        poly(3) = (25d0*v( 0) - 23d0*v( 1) + 13d0*v( 2) - 3d0*v( 3)) / 12d0 !&
                                        poly(4) = ( 1d0*v(-3) -  5d0*v(-2) + 13d0*v(-1) + 3d0*v( 0)) / 12d0 !&
                                    end if

                                    if (.not. teno) then
                                        ! (Balsara & Shu, 2000) Page 11 Section III.a
                                        ! Note: parentheses are needed to group the terms before '+ weno_eps' to avoid unintended floating point errors
                                        beta(0) = ( v(-3)*(547d0*v(-3) - 3882d0*v(-2) +  4642d0*v(-1) - 1854d0*v( 0)) & !&
                                                  + v(-2)*(              7043d0*v(-2) - 17246d0*v(-1) + 7042d0*v( 0)) & !&
                                                  + v(-1)*(                             11003d0*v(-1) - 9402d0*v( 0)) & !&
                                                  + v( 0)*(                                             2107d0*v( 0)) ) & !&
                                                  + weno_eps !&

                                        beta(1) = ( v(-2)*(267d0*v(-2) - 1642d0*v(-1) + 1602d0*v( 0) -  494d0*v( 1)) & !&
                                                  + v(-1)*(              2843d0*v(-1) - 5966d0*v( 0) + 1922d0*v( 1)) & !&
                                                  + v( 0)*(                             3443d0*v( 0) - 2522d0*v( 1)) & !&
                                                  + v( 1)*(                                             547d0*v( 1)) ) & !&
                                                  + weno_eps !&

                                        beta(2) = ( v(-1)*(547d0*v(-1) - 2522d0*v( 0) + 1922d0*v( 1) -  494d0*v( 2)) & !&
                                                  + v( 0)*(              3443d0*v( 0) - 5966d0*v( 1) + 1602d0*v( 2)) & !&
                                                  + v( 1)*(                             2843d0*v( 1) - 1642d0*v( 2)) & !&
                                                  + v( 2)*(                                             267d0*v( 2)) ) & !&
                                                  + weno_eps !&

                                        beta(3) = ( v( 0)*(2107d0*v( 0) - 9402d0*v( 1) +  7042d0*v( 2) - 1854d0*v( 3)) & !&
                                                  + v( 1)*(              11003d0*v( 1) - 17246d0*v( 2) + 4642d0*v( 3)) & !&
                                                  + v( 2)*(                               7043d0*v( 2) - 3882d0*v( 3)) & !&
                                                  + v( 3)*(                                               547d0*v( 3)) ) & !&
                                                  + weno_eps !&

                                    else ! TENO
                                        ! High-Order Low-Dissipation Targeted ENO Schemes for Ideal Magnetohydrodynamics (Fu & Tang, 2019) Section 3.2
                                        beta(0) = 13d0/12d0*(v(-1) - 2d0*v( 0) + v( 1))**2d0 + ((    v(-1)             -     v( 1))**2d0)/4d0 + weno_eps !&
                                        beta(1) = 13d0/12d0*(v( 0) - 2d0*v( 1) + v( 2))**2d0 + ((3d0*v( 0) - 4d0*v( 1) +     v( 2))**2d0)/4d0 + weno_eps !&
                                        beta(2) = 13d0/12d0*(v(-2) - 2d0*v(-1) + v( 0))**2d0 + ((    v(-2) - 4d0*v(-1) + 3d0*v( 0))**2d0)/4d0 + weno_eps !&

                                        beta(3) = ( v( 0)*(2107d0*v( 0) - 9402d0*v( 1) +  7042d0*v( 2) - 1854d0*v( 3)) & !&
                                                  + v( 1)*(              11003d0*v( 1) - 17246d0*v( 2) + 4642d0*v( 3)) & !&
                                                  + v( 2)*(                               7043d0*v( 2) - 3882d0*v( 3)) & !&
                                                  + v( 3)*(                                               547d0*v( 3)) ) / 240d0 & !&
                                                  + weno_eps !&

                                        beta(4) = ( v(-3)*(547d0*v(-3) - 3882d0*v(-2) +  4642d0*v(-1) - 1854d0*v( 0)) & !&
                                                  + v(-2)*(              7043d0*v(-2) - 17246d0*v(-1) + 7042d0*v( 0)) & !&
                                                  + v(-1)*(                             11003d0*v(-1) - 9402d0*v( 0)) & !&
                                                  + v( 0)*(                                             2107d0*v( 0)) ) / 240d0 & !&
                                                  + weno_eps !&
                                    end if

                                    if (wenojs) then
                                        alpha = d_cbL_${XYZ}$ (:, j)/(beta*beta)

                                    elseif (mapped_weno) then
                                        alpha = d_cbL_${XYZ}$ (:, j)/(beta*beta)
                                        omega = alpha/sum(alpha)
                                        alpha = (d_cbL_${XYZ}$ (:, j)*(1d0 + d_cbL_${XYZ}$ (:, j) - 3d0*omega) + omega**2d0) &
                                                *(omega/(d_cbL_${XYZ}$ (:, j)**2d0 + omega*(1d0 - 2d0*d_cbL_${XYZ}$ (:, j))))

                                    elseif (wenoz) then
                                        ! Castro, et al. (2010)
                                        ! Don & Borges (2013) also helps
                                        tau = abs(beta(3) - beta(0)) ! Equation 50
                                        alpha = d_cbL_${XYZ}$ (:, j)*(1d0 + (tau/beta)**wenoz_q) ! q = 2,3,4 for stability

                                    elseif (teno) then
                                        tau = abs(beta(4) - beta(3)) ! Note the reordering of stencils
                                        alpha = (1d0 + tau/beta)**6d0
                                        omega = alpha/sum(alpha)
                                        delta = merge(0d0, 1d0, omega < teno_CT)
                                        alpha = delta*d_cbL_${XYZ}$ (:, j)

                                    end if

                                    omega = alpha/sum(alpha)

                                    vL_rs_vf_${XYZ}$ (j, k, l, i) = sum(omega*poly)

                                    if (.not. teno) then
                                        poly(0) = (-3d0*v(-3) + 13d0*v(-2) - 23d0*v(-1) + 25d0*v( 0)) / 12d0 !&
                                        poly(1) = ( 1d0*v(-2) -  5d0*v(-1) + 13d0*v( 0) +  3d0*v( 1)) / 12d0 !&
                                        poly(2) = (-1d0*v(-1) +  7d0*v( 0) +  7d0*v( 1) -  1d0*v( 2)) / 12d0 !&
                                        poly(3) = ( 3d0*v( 0) + 13d0*v( 1) -  5d0*v( 2) +  1d0*v( 3)) / 12d0 !&
                                    else
                                        poly(0) = (-1d0*v(-1) +  5d0*v( 0) +  2d0*v( 1)) / 6d0 !&
                                        poly(1) = ( 2d0*v( 0) +  5d0*v( 1) -  1d0*v( 2)) / 6d0 !&
                                        poly(2) = ( 2d0*v(-2) -  7d0*v(-1) + 11d0*v( 0)) / 6d0 !&
                                        poly(3) = ( 3d0*v( 0) + 13d0*v( 1) -  5d0*v( 2) +  1d0*v( 3)) / 12d0 !&
                                        poly(4) = (-3d0*v(-3) + 13d0*v(-2) - 23d0*v(-1) + 25d0*v( 0)) / 12d0 !&
                                    end if

                                    if (wenojs) then
                                        alpha = d_cbR_${XYZ}$ (:, j)/(beta*beta)

                                    elseif (mapped_weno) then
                                        alpha = d_cbR_${XYZ}$ (:, j)/(beta*beta)
                                        omega = alpha/sum(alpha)
                                        alpha = (d_cbR_${XYZ}$ (:, j)*(1d0 + d_cbR_${XYZ}$ (:, j) - 3d0*omega) + omega**2d0) &
                                                *(omega/(d_cbR_${XYZ}$ (:, j)**2d0 + omega*(1d0 - 2d0*d_cbR_${XYZ}$ (:, j))))

                                    elseif (wenoz) then
                                        alpha = d_cbR_${XYZ}$ (:, j)*(1d0 + (tau/beta)**wenoz_q)

                                    elseif (teno) then
                                        alpha = delta*d_cbR_${XYZ}$ (:, j)

                                    end if

                                    omega = alpha/sum(alpha)

                                    vR_rs_vf_${XYZ}$ (j, k, l, i) = sum(omega*poly)

                                end do
                            end do
                        end do
                    end do
                    !$acc end parallel loop

                end if
            #:endfor
        end if

    end subroutine s_weno

    !> The computation of parameters, the allocation of memory,
        !!      the association of pointers and/or the execution of any
        !!      other procedures that are required for the setup of the
        !!      WENO reconstruction.
        !! @param v_vf Cell-averaged variables
        !! @param vL_vf Left WENO reconstructed cell-boundary values
        !! @param vR_vf Right WENO reconstructed cell-boundary values
        !! @param norm_dir Characteristic decommposition coordinate direction
        !! @param weno_dir Coordinate direction of the WENO reconstruction
        !! @param is1_weno Index bounds in first coordinate direction
        !! @param is2_weno Index bounds in second coordinate direction
        !! @param is3_weno Index bounds in third coordinate direction
    subroutine s_initialize_weno(v_vf, &
                                 norm_dir, weno_dir)

        type(scalar_field), dimension(:), intent(IN) :: v_vf

        integer, intent(IN) :: norm_dir
        integer, intent(IN) :: weno_dir

        integer :: j, k, l, q

        ! Determining the number of cell-average variables which will be
        ! WENO-reconstructed and mapping their indical bounds in the x-,
        ! y- and z-directions to those in the s1-, s2- and s3-directions
        ! as to reshape the inputted data in the coordinate direction of
        ! the WENO reconstruction
        v_size = ubound(v_vf, 1)
        !$acc update device(v_size)

        if (weno_dir == 1) then
            !$acc parallel loop collapse(4) gang vector default(present)
            do j = 1, v_size
                do q = is3_weno%beg, is3_weno%end
                    do l = is2_weno%beg, is2_weno%end
                        do k = is1_weno%beg - weno_polyn, is1_weno%end + weno_polyn
                            v_rs_ws_x(k, l, q, j) = v_vf(j)%sf(k, l, q)
                        end do
                    end do
                end do
            end do
            !$acc end parallel loop
        end if
        ! ==================================================================

        ! Reshaping/Projecting onto Characteristic Fields in y-direction ===
        if (n == 0) return

        if (weno_dir == 2) then
#if MFC_cuTENSOR
            if (cu_tensor) then
                if (p == 0) then
                    block
                        use CuTensorEx

                        !$acc host_data use_device(v_rs_ws_x, v_rs_ws_y)
                        v_rs_ws_y = reshape(v_rs_ws_x, shape=[n + 1 + 2*buff_size, m + 2*buff_size + 1, p + 1, sys_size], order=[2, 1, 3, 4])
                        !$acc end host_data
                    end block
                else
                    block
                        use CuTensorEx

                        !$acc host_data use_device(v_rs_ws_x, v_rs_ws_y)
                        v_rs_ws_y = reshape(v_rs_ws_x, shape=[n + 1 + 2*buff_size, m + 2*buff_size + 1, p + 1 + 2*buff_size, sys_size], order=[2, 1, 3, 4])
                        !$acc end host_data
                    end block
                end if
            else
#endif
                !$acc parallel loop collapse(4) gang vector default(present)
                do j = 1, v_size
                    do q = is3_weno%beg, is3_weno%end
                        do l = is2_weno%beg, is2_weno%end
                            do k = is1_weno%beg - weno_polyn, is1_weno%end + weno_polyn
                                v_rs_ws_y(k, l, q, j) = v_vf(j)%sf(l, k, q)
                            end do
                        end do
                    end do
                end do
!$acc end parallel loop
#if MFC_cuTENSOR
            end if
#endif
        end if

        ! ==================================================================

        ! Reshaping/Projecting onto Characteristic Fields in z-direction ===
        if (p == 0) return
        if (weno_dir == 3) then
#if MFC_cuTENSOR
            if (cu_tensor) then
                block
                    use CuTensorEx

                    !$acc host_data use_device(v_rs_ws_x, v_rs_ws_z)
                    v_rs_ws_z = reshape(v_rs_ws_x, shape=[p + 1 + 2*buff_size, n + 2*buff_size + 1, m + 2*buff_size + 1, sys_size], order=[3, 2, 1, 4])
                    !$acc end host_data
                end block
            else
#endif
                !$acc parallel loop collapse(4) gang vector default(present)
                do j = 1, v_size
                    do q = is3_weno%beg, is3_weno%end
                        do l = is2_weno%beg, is2_weno%end
                            do k = is1_weno%beg - weno_polyn, is1_weno%end + weno_polyn
                                v_rs_ws_z(k, l, q, j) = v_vf(j)%sf(q, l, k)
                            end do
                        end do
                    end do
                end do
!$acc end parallel loop
#if MFC_cuTENSOR
            end if
#endif
        end if

        ! ==================================================================

    end subroutine s_initialize_weno

    !>  The goal of this subroutine is to ensure that the WENO
        !!      reconstruction is monotonic. The latter is achieved by
        !!      enforcing monotonicity preserving bounds of Suresh and
        !!      Huynh (1997). The resulting MPWENO reconstruction, see
        !!      Balsara and Shu (2000), ensures that the reconstructed
        !!      values do not reside outside the range spanned by WENO
        !!      stencil.
        !!  @param i Equation number
        !!  @param j First-coordinate cell index
        !!  @param k Second-coordinate cell index
        !!  @param l Third-coordinate cell index
    subroutine s_preserve_monotonicity(v_rs_ws, vL_rs_vf, vR_rs_vf)

        real(kind(0d0)), dimension(startx:, starty:, startz:, 1:), intent(IN) :: v_rs_ws
        real(kind(0d0)), dimension(startx:, starty:, startz:, 1:), intent(INOUT) :: vL_rs_vf, vR_rs_vf

        integer :: i, j, k, l

        real(kind(0d0)), dimension(-1:1) :: d !< Curvature measures at the zone centers

        real(kind(0d0)) :: d_MD, d_LC !<
            !! Median (md) curvature and large curvature (LC) measures

        ! The left and right upper bounds (UL), medians, large curvatures,
        ! minima, and maxima of the WENO-reconstructed values of the cell-
        ! average variables.
        real(kind(0d0)) :: vL_UL, vR_UL
        real(kind(0d0)) :: vL_MD, vR_MD
        real(kind(0d0)) :: vL_LC, vR_LC
        real(kind(0d0)) :: vL_min, vR_min
        real(kind(0d0)) :: vL_max, vR_max

        real(kind(0d0)), parameter :: alpha = 2d0 !>
            !! Determines the maximum Courant–Friedrichs–Lewy (CFL) number that
            !! may be utilized with the scheme. In theory, for stability, a CFL
            !! number less than 1/(1+alpha) is necessary. The default value for
            !! alpha is 2.

        real(kind(0d0)), parameter :: beta = 4d0/3d0 !<
            !! Determines the amount of freedom available from utilizing a large
            !! value for the local curvature. The default value for beta is 4/3.

        real(kind(0d0)), parameter :: alpha_mp = 2d0
        real(kind(0d0)), parameter :: beta_mp = 4d0/3d0

        !$acc parallel loop gang vector collapse (4)  default(present) private(d)
        do l = is3_weno%beg, is3_weno%end
            do k = is2_weno%beg, is2_weno%end
                do j = is1_weno%beg, is1_weno%end
                    do i = 1, v_size
                        d(-1) = v_rs_ws(j, k, l, i) &
                                + v_rs_ws(j - 2, k, l, i) &
                                - v_rs_ws(j - 1, k, l, i) &
                                *2d0
                        d(0) = v_rs_ws(j + 1, k, l, i) &
                               + v_rs_ws(j - 1, k, l, i) &
                               - v_rs_ws(j, k, l, i) &
                               *2d0
                        d(1) = v_rs_ws(j + 2, k, l, i) &
                               + v_rs_ws(j, k, l, i) &
                               - v_rs_ws(j + 1, k, l, i) &
                               *2d0

                        d_MD = (sign(1d0, 4d0*d(-1) - d(0)) + sign(1d0, 4d0*d(0) - d(-1))) &
                               *abs((sign(1d0, 4d0*d(-1) - d(0)) + sign(1d0, d(-1))) &
                                    *(sign(1d0, 4d0*d(-1) - d(0)) + sign(1d0, d(0)))) &
                               *min(abs(4d0*d(-1) - d(0)), abs(d(-1)), &
                                    abs(4d0*d(0) - d(-1)), abs(d(0)))/8d0

                        d_LC = (sign(1d0, 4d0*d(0) - d(1)) + sign(1d0, 4d0*d(1) - d(0))) &
                               *abs((sign(1d0, 4d0*d(0) - d(1)) + sign(1d0, d(0))) &
                                    *(sign(1d0, 4d0*d(0) - d(1)) + sign(1d0, d(1)))) &
                               *min(abs(4d0*d(0) - d(1)), abs(d(0)), &
                                    abs(4d0*d(1) - d(0)), abs(d(1)))/8d0

                        vL_UL = v_rs_ws(j, k, l, i) &
                                - (v_rs_ws(j + 1, k, l, i) &
                                   - v_rs_ws(j, k, l, i))*alpha_mp

                        vL_MD = (v_rs_ws(j, k, l, i) &
                                 + v_rs_ws(j - 1, k, l, i) &
                                 - d_MD)*5d-1

                        vL_LC = v_rs_ws(j, k, l, i) &
                                - (v_rs_ws(j + 1, k, l, i) &
                                   - v_rs_ws(j, k, l, i))*5d-1 + beta_mp*d_LC

                        vL_min = max(min(v_rs_ws(j, k, l, i), &
                                         v_rs_ws(j - 1, k, l, i), &
                                         vL_MD), &
                                     min(v_rs_ws(j, k, l, i), &
                                         vL_UL, &
                                         vL_LC))

                        vL_max = min(max(v_rs_ws(j, k, l, i), &
                                         v_rs_ws(j - 1, k, l, i), &
                                         vL_MD), &
                                     max(v_rs_ws(j, k, l, i), &
                                         vL_UL, &
                                         vL_LC))

                        vL_rs_vf(j, k, l, i) = vL_rs_vf(j, k, l, i) &
                                               + (sign(5d-1, vL_min - vL_rs_vf(j, k, l, i)) &
                                                  + sign(5d-1, vL_max - vL_rs_vf(j, k, l, i))) &
                                               *min(abs(vL_min - vL_rs_vf(j, k, l, i)), &
                                                    abs(vL_max - vL_rs_vf(j, k, l, i)))
                        ! END: Left Monotonicity Preserving Bound ==========================

                        ! Right Monotonicity Preserving Bound ==============================
                        d(-1) = v_rs_ws(j, k, l, i) &
                                + v_rs_ws(j - 2, k, l, i) &
                                - v_rs_ws(j - 1, k, l, i) &
                                *2d0
                        d(0) = v_rs_ws(j + 1, k, l, i) &
                               + v_rs_ws(j - 1, k, l, i) &
                               - v_rs_ws(j, k, l, i) &
                               *2d0
                        d(1) = v_rs_ws(j + 2, k, l, i) &
                               + v_rs_ws(j, k, l, i) &
                               - v_rs_ws(j + 1, k, l, i) &
                               *2d0

                        d_MD = (sign(1d0, 4d0*d(0) - d(1)) + sign(1d0, 4d0*d(1) - d(0))) &
                               *abs((sign(1d0, 4d0*d(0) - d(1)) + sign(1d0, d(0))) &
                                    *(sign(1d0, 4d0*d(0) - d(1)) + sign(1d0, d(1)))) &
                               *min(abs(4d0*d(0) - d(1)), abs(d(0)), &
                                    abs(4d0*d(1) - d(0)), abs(d(1)))/8d0

                        d_LC = (sign(1d0, 4d0*d(-1) - d(0)) + sign(1d0, 4d0*d(0) - d(-1))) &
                               *abs((sign(1d0, 4d0*d(-1) - d(0)) + sign(1d0, d(-1))) &
                                    *(sign(1d0, 4d0*d(-1) - d(0)) + sign(1d0, d(0)))) &
                               *min(abs(4d0*d(-1) - d(0)), abs(d(-1)), &
                                    abs(4d0*d(0) - d(-1)), abs(d(0)))/8d0

                        vR_UL = v_rs_ws(j, k, l, i) &
                                + (v_rs_ws(j, k, l, i) &
                                   - v_rs_ws(j - 1, k, l, i))*alpha_mp

                        vR_MD = (v_rs_ws(j, k, l, i) &
                                 + v_rs_ws(j + 1, k, l, i) &
                                 - d_MD)*5d-1

                        vR_LC = v_rs_ws(j, k, l, i) &
                                + (v_rs_ws(j, k, l, i) &
                                   - v_rs_ws(j - 1, k, l, i))*5d-1 + beta_mp*d_LC

                        vR_min = max(min(v_rs_ws(j, k, l, i), &
                                         v_rs_ws(j + 1, k, l, i), &
                                         vR_MD), &
                                     min(v_rs_ws(j, k, l, i), &
                                         vR_UL, &
                                         vR_LC))

                        vR_max = min(max(v_rs_ws(j, k, l, i), &
                                         v_rs_ws(j + 1, k, l, i), &
                                         vR_MD), &
                                     max(v_rs_ws(j, k, l, i), &
                                         vR_UL, &
                                         vR_LC))

                        vR_rs_vf(j, k, l, i) = vR_rs_vf(j, k, l, i) &
                                               + (sign(5d-1, vR_min - vR_rs_vf(j, k, l, i)) &
                                                  + sign(5d-1, vR_max - vR_rs_vf(j, k, l, i))) &
                                               *min(abs(vR_min - vR_rs_vf(j, k, l, i)), &
                                                    abs(vR_max - vR_rs_vf(j, k, l, i)))
                        ! END: Right Monotonicity Preserving Bound =========================
                    end do
                end do
            end do
        end do
        !$acc end parallel loop

    end subroutine s_preserve_monotonicity

    !>  Module deallocation and/or disassociation procedures
    subroutine s_finalize_weno_module()

        if (weno_order == 1) return

        ! Deallocating the WENO-stencil of the WENO-reconstructed variables

        !deallocate(vL_rs_vf_x, vR_rs_vf_x)
        @:DEALLOCATE_GLOBAL(v_rs_ws_x)

        ! Deallocating WENO coefficients in x-direction ====================
        @:DEALLOCATE_GLOBAL(poly_coef_cbL_x, poly_coef_cbR_x)
        @:DEALLOCATE_GLOBAL(d_cbL_x, d_cbR_x)
        @:DEALLOCATE_GLOBAL(beta_coef_x)
        ! ==================================================================

        ! Deallocating WENO coefficients in y-direction ====================
        if (n == 0) return

        !deallocate(vL_rs_vf_y, vR_rs_vf_y)
        @:DEALLOCATE_GLOBAL(v_rs_ws_y)

        @:DEALLOCATE_GLOBAL(poly_coef_cbL_y, poly_coef_cbR_y)
        @:DEALLOCATE_GLOBAL(d_cbL_y, d_cbR_y)
        @:DEALLOCATE_GLOBAL(beta_coef_y)
        ! ==================================================================

        ! Deallocating WENO coefficients in z-direction ====================
        if (p == 0) return

        !deallocate(vL_rs_vf_z, vR_rs_vf_z)
        @:DEALLOCATE_GLOBAL(v_rs_ws_z)

        @:DEALLOCATE_GLOBAL(poly_coef_cbL_z, poly_coef_cbR_z)
        @:DEALLOCATE_GLOBAL(d_cbL_z, d_cbR_z)
        @:DEALLOCATE_GLOBAL(beta_coef_z)
        ! ==================================================================

    end subroutine s_finalize_weno_module

end module m_weno
