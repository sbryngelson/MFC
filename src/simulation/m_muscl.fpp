!>
!! @file
!! @brief Contains module m_muscl

#:include 'macros.fpp'

!> @brief MUSCL reconstruction with interface sharpening for contact-preserving advection
module m_muscl

    use m_derived_types
    use m_global_parameters
    use m_variables_conversion
    use m_constants, only: muscl_order_first_order, muscl_order_second_order, muscl_lim_unlimited, muscl_lim_minmod, &
        & muscl_lim_mc, muscl_lim_van_albada, muscl_lim_van_leer, muscl_lim_superbee
#ifdef MFC_OpenACC
    use openacc
#endif

    use m_mpi_proxy
    use m_helper
    use m_thinc
    use m_helper_basic, only: f_optional_true
    use m_nvtx

    private; public :: s_initialize_muscl_module, s_muscl, s_finalize_muscl_module

    integer :: v_size
    $:GPU_DECLARE(create='[v_size]')

    type(int_bounds_info) :: is1_muscl, is2_muscl, is3_muscl
    $:GPU_DECLARE(create='[is1_muscl, is2_muscl, is3_muscl]')

    !> @name The cell-average variables that will be MUSCL-reconstructed, unpacked into an array for performance
    !> @{
    real(wp), allocatable, dimension(:,:,:,:) :: v_rs_ws_muscl
    !> @}
    $:GPU_DECLARE(create='[v_rs_ws_muscl]')

contains

    !> Allocate and initialize MUSCL reconstruction working arrays
    subroutine s_initialize_muscl_module()

        ! Initializing in x-direction
        is1_muscl%beg = -buff_size; is1_muscl%end = m - is1_muscl%beg
        if (n == 0) then
            is2_muscl%beg = 0
        else
            is2_muscl%beg = -buff_size
        end if

        is2_muscl%end = n - is2_muscl%beg

        if (p == 0) then
            is3_muscl%beg = 0
        else
            is3_muscl%beg = -buff_size
        end if

        is3_muscl%end = p - is3_muscl%beg

        @:ALLOCATE(v_rs_ws_muscl(is1_muscl%beg:is1_muscl%end, is2_muscl%beg:is2_muscl%end, is3_muscl%beg:is3_muscl%end, 1:sys_size))

        if (n == 0) return

        ! initializing in y-direction
        is2_muscl%beg = -buff_size; is2_muscl%end = n - is2_muscl%beg
        is1_muscl%beg = -buff_size; is1_muscl%end = m - is1_muscl%beg

        if (p == 0) then
            is3_muscl%beg = 0
        else
            is3_muscl%beg = -buff_size
        end if

        is3_muscl%end = p - is3_muscl%beg

        if (p == 0) return

        ! initializing in z-direction
        is2_muscl%beg = -buff_size; is2_muscl%end = n - is2_muscl%beg
        is1_muscl%beg = -buff_size; is1_muscl%end = m - is1_muscl%beg
        is3_muscl%beg = -buff_size; is3_muscl%end = p - is3_muscl%beg

    end subroutine s_initialize_muscl_module

    !> Perform MUSCL reconstruction of left and right cell-boundary values from cell-averaged variables
    subroutine s_muscl(v_vf, vL_rs_vf_x, vR_rs_vf_x, muscl_dir, is1_muscl_d, is2_muscl_d, is3_muscl_d, packed)

        !> collapsed-loop bounds as scalars (see gpuParallelization.md, NVHPC 26.5)
        integer :: bnd0_hi, bnd0_lo, bnd1_hi, bnd1_lo, bnd2_hi, bnd2_lo, bnd3_hi, bnd3_lo
        type(scalar_field), dimension(1:), intent(in) :: v_vf
        real(wp), dimension(idwbuff(1)%beg:,idwbuff(2)%beg:,idwbuff(3)%beg:,1:), intent(inout) :: vL_rs_vf_x, vR_rs_vf_x
        integer, intent(in) :: muscl_dir
        type(int_bounds_info), intent(in) :: is1_muscl_d, is2_muscl_d, is3_muscl_d
        logical, optional, intent(in) :: packed  !< v_rs_ws_muscl holds v_vf
        integer :: j, k, l, i
        real(wp) :: slopeL, slopeR, slope

        is1_muscl = is1_muscl_d
        is2_muscl = is2_muscl_d
        is3_muscl = is3_muscl_d

        $:GPU_UPDATE(device='[is1_muscl, is2_muscl, is3_muscl]')

        if (muscl_order == muscl_order_first_order) then
            if (muscl_dir == 1) then
                bnd1_lo = is3_muscl%beg; bnd1_hi = is3_muscl%end
                bnd2_lo = is2_muscl%beg; bnd2_hi = is2_muscl%end
                bnd3_lo = is1_muscl%beg; bnd3_hi = is1_muscl%end
                $:GPU_PARALLEL_LOOP(collapse=4)
                do i = 1, ubound(v_vf, 1)
                    do l = bnd1_lo, bnd1_hi
                        do k = bnd2_lo, bnd2_hi
                            do j = bnd3_lo, bnd3_hi
                                vL_rs_vf_x(j, k, l, i) = v_vf(i)%sf(j, k, l)
                                vR_rs_vf_x(j, k, l, i) = v_vf(i)%sf(j, k, l)
                            end do
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            else if (muscl_dir == 2) then
                bnd1_lo = is3_muscl%beg; bnd1_hi = is3_muscl%end
                bnd2_lo = is1_muscl%beg; bnd2_hi = is1_muscl%end
                bnd3_lo = is2_muscl%beg; bnd3_hi = is2_muscl%end
                $:GPU_PARALLEL_LOOP(collapse=4)
                do i = 1, ubound(v_vf, 1)
                    do l = bnd1_lo, bnd1_hi
                        do j = bnd2_lo, bnd2_hi
                            do k = bnd3_lo, bnd3_hi
                                vL_rs_vf_x(k, j, l, i) = v_vf(i)%sf(k, j, l)
                                vR_rs_vf_x(k, j, l, i) = v_vf(i)%sf(k, j, l)
                            end do
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            else if (muscl_dir == 3) then
                bnd1_lo = is1_muscl%beg; bnd1_hi = is1_muscl%end
                bnd2_lo = is2_muscl%beg; bnd2_hi = is2_muscl%end
                bnd3_lo = is3_muscl%beg; bnd3_hi = is3_muscl%end
                $:GPU_PARALLEL_LOOP(collapse=4)
                do i = 1, ubound(v_vf, 1)
                    do j = bnd1_lo, bnd1_hi
                        do k = bnd2_lo, bnd2_hi
                            do l = bnd3_lo, bnd3_hi
                                vL_rs_vf_x(l, k, j, i) = v_vf(i)%sf(l, k, j)
                                vR_rs_vf_x(l, k, j, i) = v_vf(i)%sf(l, k, j)
                            end do
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            end if
        end if

        v_size = ubound(v_vf, 1)
        $:GPU_UPDATE(device='[v_size]')

        if (muscl_order /= muscl_order_first_order .and. .not. f_optional_true(packed)) then
            bnd1_lo = idwbuff(3)%beg; bnd1_hi = idwbuff(3)%end
            bnd2_lo = idwbuff(2)%beg; bnd2_hi = idwbuff(2)%end
            bnd3_lo = idwbuff(1)%beg; bnd3_hi = idwbuff(1)%end
            $:GPU_PARALLEL_LOOP(private='[j, k, l, i]', collapse=4)
            do i = 1, v_size
                do l = bnd1_lo, bnd1_hi
                    do k = bnd2_lo, bnd2_hi
                        do j = bnd3_lo, bnd3_hi
                            v_rs_ws_muscl(j, k, l, i) = v_vf(i)%sf(j, k, l)
                        end do
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()
        end if

        if (muscl_order == muscl_order_second_order) then
            ! MUSCL Reconstruction
            #:for MUSCL_DIR, XYZ, STENCIL_VAR, COORDS, X_BND, Y_BND, Z_BND in &
                    [(1, 'x', 'j', '{STENCIL_IDX}, k, l', 'is1_muscl', 'is2_muscl', 'is3_muscl'), &
                     (2, 'y', 'k', 'j, {STENCIL_IDX}, l', 'is2_muscl', 'is1_muscl', 'is3_muscl'), &
                     (3, 'z', 'l', 'j, k, {STENCIL_IDX}', 'is3_muscl', 'is2_muscl', 'is1_muscl')]
                #:set SV = STENCIL_VAR
                #:set SF = lambda offs: COORDS.format(STENCIL_IDX = SV + offs)
                if (muscl_dir == ${MUSCL_DIR}$) then
                    bnd0_lo = ${Z_BND}$%beg; bnd0_hi = ${Z_BND}$%end
                    bnd1_lo = ${Y_BND}$%beg; bnd1_hi = ${Y_BND}$%end
                    bnd2_lo = ${X_BND}$%beg; bnd2_hi = ${X_BND}$%end
                    $:GPU_PARALLEL_LOOP(collapse=4,private='[i, j, k, l, slopeL, slopeR, slope]')
                    do l = bnd0_lo, bnd0_hi
                        do k = bnd1_lo, bnd1_hi
                            do j = bnd2_lo, bnd2_hi
                                do i = 1, v_size
                                    slopeL = v_rs_ws_muscl(${SF(' + 1')}$, i) - v_rs_ws_muscl(${SF('')}$, i)
                                    slopeR = v_rs_ws_muscl(${SF('')}$, i) - v_rs_ws_muscl(${SF(' - 1')}$, i)
                                    slope = 0._wp

                                    if (muscl_lim == muscl_lim_unlimited) then  ! unlimited (central difference)
                                        slope = 5e-1_wp*(slopeL + slopeR)
                                    else if (muscl_lim == muscl_lim_minmod) then  ! minmod
                                        if (slopeL*slopeR > muscl_eps) then
                                            slope = min(abs(slopeL), abs(slopeR))
                                        end if
                                        if (slopeL < 0._wp) slope = -slope
                                    else if (muscl_lim == muscl_lim_mc) then  ! MC
                                        if (slopeL*slopeR > muscl_eps) then
                                            slope = min(2._wp*abs(slopeL), 2._wp*abs(slopeR))
                                            slope = min(slope, 5e-1_wp*(abs(slopeL) + abs(slopeR)))
                                        end if
                                        if (slopeL < 0._wp) slope = -slope
                                    else if (muscl_lim == muscl_lim_van_albada) then  ! Van Albada
                                        if (slopeL*slopeR > muscl_eps) then
                                            slope = ((slopeL + slopeR)*slopeL*slopeR)/(slopeL**2._wp + slopeR**2._wp)
                                        end if
                                    else if (muscl_lim == muscl_lim_van_leer) then  ! Van Leer
                                        if (slopeL*slopeR > muscl_eps) then
                                            slope = 2._wp*slopeL*slopeR/(slopeL + slopeR)
                                        end if
                                    else if (muscl_lim == muscl_lim_superbee) then  ! SUPERBEE
                                        if (slopeL*slopeR > muscl_eps) then
                                            slope = -1._wp*min(-min(2._wp*abs(slopeL), abs(slopeR)), -min(abs(slopeL), &
                                                               & 2._wp*abs(slopeR)))
                                        end if
                                        if (slopeL < 0._wp) slope = -slope
                                    end if

                                    ! reconstruct from left side
                                    vL_rs_vf_x(j, k, l, i) = v_rs_ws_muscl(${SF('')}$, i) - (5.e-1_wp*slope)

                                    ! reconstruct from the right side
                                    vR_rs_vf_x(j, k, l, i) = v_rs_ws_muscl(${SF('')}$, i) + (5.e-1_wp*slope)
                                end do
                            end do
                        end do
                    end do
                    $:END_GPU_PARALLEL_LOOP()
                end if
            #:endfor
        end if

        if (int_comp > 0 .and. v_size >= eqn_idx%adv%end) then
            call nvtxStartRange("WENO-INTCOMP")
            #:for MUSCL_DIR, XYZ in [(1, 'x'), (2, 'y'), (3, 'z')]
                if (muscl_dir == ${MUSCL_DIR}$) then
                    call s_thinc_compression(v_rs_ws_muscl, vL_rs_vf_x, vR_rs_vf_x, muscl_dir, is1_muscl, is2_muscl, is3_muscl)
                end if
            #:endfor
            call nvtxEndRange()
        end if

    end subroutine s_muscl

    !> Finalize the MUSCL module
    subroutine s_finalize_muscl_module()

        @:DEALLOCATE(v_rs_ws_muscl)

    end subroutine s_finalize_muscl_module

end module m_muscl
