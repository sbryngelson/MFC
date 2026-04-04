!>
!! @file m_fused_kernels.fpp
!! @brief Per-cell fused WENO5+HLLC kernel for maximum memory efficiency.
!!
!! This module provides a single per-cell GPU kernel that fuses WENO reconstruction,
!! Riemann solving, and flux differencing with ZERO intermediate global memory.
!! Conservative variables are read directly from q_cons_vf; primitives (velocity,
!! pressure) are computed on the fly at each WENO stencil point.
!!
!! The kernel uses Fypp macros (from include/inline_weno.fpp and inline_riemann.fpp)
!! for the WENO and Riemann computations. These expand to flat GPU code that nvfortran
!! compiles efficiently. Device subroutines ($:GPU_ROUTINE(seq)) are NOT used because
!! nvfortran's OpenACC frontend hangs on large device subroutines.
!!
!! To extend:
!!   - New WENO order: add INLINE_WENO3_CONS macro to include/inline_weno.fpp
!!   - New Riemann solver: add INLINE_HLL_FLUX macro to include/inline_riemann.fpp
!!   - Then call the new macro in s_fused_weno_riemann_rhs below

#:include 'case.fpp'
#:include 'macros.fpp'
#:include 'inline_weno.fpp'
#:include 'inline_riemann.fpp'

module m_fused_kernels

    use m_derived_types
    use m_global_parameters
    use m_variables_conversion
    use m_weno, only: poly_coef_cbL_x, poly_coef_cbL_y, poly_coef_cbL_z, &
        & poly_coef_cbR_x, poly_coef_cbR_y, poly_coef_cbR_z, &
        & d_cbL_x, d_cbL_y, d_cbL_z, d_cbR_x, d_cbR_y, d_cbR_z, &
        & beta_coef_x, beta_coef_y, beta_coef_z
    use m_thermochem, only: gas_constant, get_mixture_molecular_weight, &
        & get_mixture_specific_heat_cv_mass, get_mixture_energy_mass, &
        & get_species_specific_heats_r, get_species_enthalpies_rt, &
        & get_mixture_specific_heat_cp_mass
    use m_chemistry

    implicit none

    private; public :: s_initialize_fused_kernels_module, s_fused_weno_riemann_rhs, s_finalize_fused_kernels_module

    ! Local copy of shear moduli (mirrors Gs_rs in m_riemann_solvers, avoids circular dependency)
    real(wp), allocatable, dimension(:) :: Gs_rs
    $:GPU_DECLARE(create='[Gs_rs]')

contains

    subroutine s_initialize_fused_kernels_module()

        integer :: i

        @:ALLOCATE(Gs_rs(1:num_fluids))
        do i = 1, num_fluids
            Gs_rs(i) = fluid_pp(i)%G
        end do
        $:GPU_UPDATE(device='[Gs_rs]')

    end subroutine s_initialize_fused_kernels_module

    subroutine s_finalize_fused_kernels_module()

        @:DEALLOCATE(Gs_rs)

    end subroutine s_finalize_fused_kernels_module

    !> Per-cell fused WENO5+HLLC+flux_diff kernel.
    !! Each GPU thread handles one cell, computing BOTH adjacent face fluxes
    !! and accumulating the flux difference directly into rhs_vf.
    !! No intermediate global arrays are used.
    subroutine s_fused_weno_riemann_rhs(q_cons_vf_arg, rhs_vf, norm_dir)

        type(vector_field), intent(in)                         :: q_cons_vf_arg
        type(scalar_field), dimension(sys_size), intent(inout) :: rhs_vf
        integer, intent(in)                                    :: norm_dir

        ! WENO scratch (INLINE_WENO5_CONS uses these names)
        real(wp) :: w5_dvd(-2:1), w5_poly(0:2), w5_beta(0:2), w5_alpha(0:2)
        real(wp) :: w5_omega(0:2), w5_tau, w5_delta(0:2), w5_dummy
        real(wp) :: w5c_stencil(-2:2), w5c_rho, w5c_vel_sqr
        real(wp) :: w5c_gamma, w5c_pi_inf, w5c_qv
        integer  :: w5_idx

        ! Reconstructed states and Riemann output
        real(wp) :: qL_local(sys_size), qR_local(sys_size)
        real(wp) :: flux_left(sys_size), flux_right(sys_size)
        real(wp) :: vel_src_left, vel_src_right

        ! HLLC solver scratch — full declaration block matching m_riemann_solvers.fpp
        ! (required by INLINE_HLLC_FLUX macro which reads these by name)
        #:if not MFC_CASE_OPTIMIZATION and USING_AMD
            real(wp), dimension(3) :: alpha_rho_L, alpha_rho_R
            real(wp), dimension(3) :: alpha_L, alpha_R
            real(wp), dimension(3) :: vel_L, vel_R
        #:else
            real(wp), dimension(num_fluids) :: alpha_rho_L, alpha_rho_R
            real(wp), dimension(num_fluids) :: alpha_L, alpha_R
            real(wp), dimension(num_dims)   :: vel_L, vel_R
        #:endif
        real(wp) :: rho_L, rho_R, pres_L, pres_R, E_L, E_R, H_L, H_R
        #:if not MFC_CASE_OPTIMIZATION and USING_AMD
            real(wp), dimension(10) :: Ys_L, Ys_R, Xs_L, Xs_R, Gamma_iL, Gamma_iR, Cp_iL, Cp_iR
            real(wp), dimension(10) :: Yi_avg, Phi_avg, h_iL, h_iR, h_avg_2
        #:else
            real(wp), dimension(num_species) :: Ys_L, Ys_R, Xs_L, Xs_R, Gamma_iL, Gamma_iR, Cp_iL, Cp_iR
            real(wp), dimension(num_species) :: Yi_avg, Phi_avg, h_iL, h_iR, h_avg_2
        #:endif
        real(wp) :: Cp_avg, Cv_avg, T_avg, c_sum_Yi_Phi, eps
        real(wp) :: T_L, T_R, MW_L, MW_R, R_gas_L, R_gas_R
        real(wp) :: Cp_L, Cp_R, Cv_L, Cv_R, Gamm_L, Gamm_R, Y_L, Y_R
        real(wp) :: gamma_L, gamma_R, pi_inf_L, pi_inf_R, qv_L, qv_R
        real(wp) :: c_L, c_R, rho_avg, H_avg, gamma_avg, qv_avg, c_avg
        real(wp) :: s_L, s_R, s_M, s_P, s_S, xi_L, xi_R, xi_M, xi_P, xi_MP, xi_PP
        real(wp), dimension(2) :: Re_L, Re_R
        real(wp) :: vel_L_rms, vel_R_rms, vel_avg_rms, alpha_L_sum, alpha_R_sum
        real(wp) :: vel_L_tmp, vel_R_tmp, pcorr, zcoef, Ms_L, Ms_R, pres_SL, pres_SR
        real(wp), dimension(6) :: tau_e_L, tau_e_R
        real(wp) :: flux_ene_e, G_L, G_R
        #:if not MFC_CASE_OPTIMIZATION and USING_AMD
            real(wp), dimension(3) :: xi_field_L, xi_field_R
        #:else
            real(wp), dimension(num_dims) :: xi_field_L, xi_field_R
        #:endif
        real(wp) :: rho_Star, E_Star, p_Star, p_K_Star, vel_K_star
        real(wp) :: Re_max, nbub_L, nbub_R, ptilde_L, ptilde_R
        integer  :: i, q

        real(wp) :: inv_ds
        integer  :: j, k, l, iv, j_adv

        ! --- Direction-dependent loop (only Fypp expansion in the orchestrator) ---

        #:for NORM_DIR, XYZ in [(1, 'x'), (2, 'y'), (3, 'z')]
            if (norm_dir == ${NORM_DIR}$) then
                $:GPU_PARALLEL_LOOP(collapse=3)
                do l = 0, p
                    do k = 0, n
                        do j = 0, m
                            #:if NORM_DIR == 1
                                inv_ds = 1._wp/dx(j)
                            #:elif NORM_DIR == 2
                                inv_ds = 1._wp/dy(k)
                            #:else
                                inv_ds = 1._wp/dz(l)
                            #:endif

                            ! ---- LEFT FACE (between cell j-1 and j) ----

                            ! Left-biased reconstruction at left face
                            do iv = 1, sys_size
                                #:if NORM_DIR == 1
                                    @:INLINE_WENO5_CONS(${NORM_DIR}$, j - 1, iv, poly_coef_cbL_x, poly_coef_cbR_x, d_cbL_x, d_cbR_x, beta_coef_x, qL_local(iv), w5_dummy)
                                #:elif NORM_DIR == 2
                                    @:INLINE_WENO5_CONS(${NORM_DIR}$, k - 1, iv, poly_coef_cbL_y, poly_coef_cbR_y, d_cbL_y, d_cbR_y, beta_coef_y, qL_local(iv), w5_dummy)
                                #:else
                                    @:INLINE_WENO5_CONS(${NORM_DIR}$, l - 1, iv, poly_coef_cbL_z, poly_coef_cbR_z, d_cbL_z, d_cbR_z, beta_coef_z, qL_local(iv), w5_dummy)
                                #:endif
                            end do
                            ! Right-biased reconstruction at left face
                            do iv = 1, sys_size
                                #:if NORM_DIR == 1
                                    @:INLINE_WENO5_CONS(${NORM_DIR}$, j, iv, poly_coef_cbL_x, poly_coef_cbR_x, d_cbL_x, d_cbR_x, beta_coef_x, w5_dummy, qR_local(iv))
                                #:elif NORM_DIR == 2
                                    @:INLINE_WENO5_CONS(${NORM_DIR}$, k, iv, poly_coef_cbL_y, poly_coef_cbR_y, d_cbL_y, d_cbR_y, beta_coef_y, w5_dummy, qR_local(iv))
                                #:else
                                    @:INLINE_WENO5_CONS(${NORM_DIR}$, l, iv, poly_coef_cbL_z, poly_coef_cbR_z, d_cbL_z, d_cbR_z, beta_coef_z, w5_dummy, qR_local(iv))
                                #:endif
                            end do
                            @:INLINE_HLLC_FLUX(flux_left, vel_src_left)

                            ! ---- RIGHT FACE (between cell j and j+1) ----

                            do iv = 1, sys_size
                                #:if NORM_DIR == 1
                                    @:INLINE_WENO5_CONS(${NORM_DIR}$, j, iv, poly_coef_cbL_x, poly_coef_cbR_x, d_cbL_x, d_cbR_x, beta_coef_x, qL_local(iv), w5_dummy)
                                #:elif NORM_DIR == 2
                                    @:INLINE_WENO5_CONS(${NORM_DIR}$, k, iv, poly_coef_cbL_y, poly_coef_cbR_y, d_cbL_y, d_cbR_y, beta_coef_y, qL_local(iv), w5_dummy)
                                #:else
                                    @:INLINE_WENO5_CONS(${NORM_DIR}$, l, iv, poly_coef_cbL_z, poly_coef_cbR_z, d_cbL_z, d_cbR_z, beta_coef_z, qL_local(iv), w5_dummy)
                                #:endif
                            end do
                            do iv = 1, sys_size
                                #:if NORM_DIR == 1
                                    @:INLINE_WENO5_CONS(${NORM_DIR}$, j + 1, iv, poly_coef_cbL_x, poly_coef_cbR_x, d_cbL_x, d_cbR_x, beta_coef_x, w5_dummy, qR_local(iv))
                                #:elif NORM_DIR == 2
                                    @:INLINE_WENO5_CONS(${NORM_DIR}$, k + 1, iv, poly_coef_cbL_y, poly_coef_cbR_y, d_cbL_y, d_cbR_y, beta_coef_y, w5_dummy, qR_local(iv))
                                #:else
                                    @:INLINE_WENO5_CONS(${NORM_DIR}$, l + 1, iv, poly_coef_cbL_z, poly_coef_cbR_z, d_cbL_z, d_cbR_z, beta_coef_z, w5_dummy, qR_local(iv))
                                #:endif
                            end do
                            @:INLINE_HLLC_FLUX(flux_right, vel_src_right)

                            ! ---- FLUX DIFFERENCING → rhs_vf ----

                            do iv = 1, sys_size
                                rhs_vf(iv)%sf(j, k, l) = rhs_vf(iv)%sf(j, k, l) + inv_ds*(flux_left(iv) - flux_right(iv))
                            end do
                            do j_adv = advxb, advxe
                                rhs_vf(j_adv)%sf(j, k, l) = rhs_vf(j_adv)%sf(j, k, l) + &
                                    & inv_ds*q_cons_vf_arg%vf(j_adv)%sf(j, k, l)*(vel_src_right - vel_src_left)
                            end do

                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            end if
        #:endfor

    end subroutine s_fused_weno_riemann_rhs

end module m_fused_kernels
