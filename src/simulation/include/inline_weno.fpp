!> @brief Inline WENO5 reconstruction macros for fused WENO+Riemann kernels. These macros reconstruct left and right states at a
!! cell face from the transposed WENO workspace, storing results in local register arrays.

!> Reconstruct left and right states for variable IV at face J in the transposed workspace V_WS. Results written to QL_VAR and
!! QR_VAR. Requires w5_dvd, w5_poly, w5_beta, w5_alpha, w5_omega, w5_tau, w5_delta as local variables. Uses w5_idx as a loop index.
!! POLY_CBL/R, D_CBL/R, BETA_C are the direction-specific WENO coefficient arrays.
#:def INLINE_WENO5_RECONSTRUCT(V_WS, J, K, L, IV, POLY_CBL, POLY_CBR, D_CBL, D_CBR, BETA_C, QL_VAR, QR_VAR)
    w5_dvd(1) = ${V_WS}$(${J}$ + 2, ${K}$, ${L}$, ${IV}$) - ${V_WS}$(${J}$ + 1, ${K}$, ${L}$, ${IV}$)
    w5_dvd(0) = ${V_WS}$(${J}$ + 1, ${K}$, ${L}$, ${IV}$) - ${V_WS}$(${J}$, ${K}$, ${L}$, ${IV}$)
    w5_dvd(-1) = ${V_WS}$(${J}$, ${K}$, ${L}$, ${IV}$) - ${V_WS}$(${J}$ - 1, ${K}$, ${L}$, ${IV}$)
    w5_dvd(-2) = ${V_WS}$(${J}$ - 1, ${K}$, ${L}$, ${IV}$) - ${V_WS}$(${J}$ - 2, ${K}$, ${L}$, ${IV}$)

    ! Left reconstruction polynomials
    w5_poly(0) = ${V_WS}$(${J}$, ${K}$, ${L}$, ${IV}$) + ${POLY_CBL}$(${J}$, 0, 0)*w5_dvd(1) + ${POLY_CBL}$(${J}$, 0, 1)*w5_dvd(0)
    w5_poly(1) = ${V_WS}$(${J}$, ${K}$, ${L}$, ${IV}$) + ${POLY_CBL}$(${J}$, 1, 0)*w5_dvd(0) + ${POLY_CBL}$(${J}$, 1, 1)*w5_dvd(-1)
    w5_poly(2) = ${V_WS}$(${J}$, ${K}$, ${L}$, ${IV}$) + ${POLY_CBL}$(${J}$, 2, 0)*w5_dvd(-1) + ${POLY_CBL}$(${J}$, 2, 1)*w5_dvd(-2)

    ! Smoothness indicators
    w5_beta(0) = ${BETA_C}$(${J}$, 0, 0)*w5_dvd(1)*w5_dvd(1) + ${BETA_C}$(${J}$, 0, 1)*w5_dvd(1)*w5_dvd(0) + ${BETA_C}$(${J}$, 0, &
            & 2)*w5_dvd(0)*w5_dvd(0) + weno_eps
    w5_beta(1) = ${BETA_C}$(${J}$, 1, 0)*w5_dvd(0)*w5_dvd(0) + ${BETA_C}$(${J}$, 1, 1)*w5_dvd(0)*w5_dvd(-1) + ${BETA_C}$(${J}$, &
            & 1, 2)*w5_dvd(-1)*w5_dvd(-1) + weno_eps
    w5_beta(2) = ${BETA_C}$(${J}$, 2, 0)*w5_dvd(-1)*w5_dvd(-1) + ${BETA_C}$(${J}$, 2, &
            & 1)*w5_dvd(-1)*w5_dvd(-2) + ${BETA_C}$(${J}$, 2, 2)*w5_dvd(-2)*w5_dvd(-2) + weno_eps

    ! Left weights (WENO-Z by default, same as mapped_weno=F, wenoz=T)
    if (wenojs) then
        w5_alpha(0:weno_num_stencils) = ${D_CBL}$(0:weno_num_stencils,${J}$)/(w5_beta(0:weno_num_stencils)**2._wp)
    else if (mapped_weno) then
        w5_alpha(0:weno_num_stencils) = ${D_CBL}$(0:weno_num_stencils,${J}$)/(w5_beta(0:weno_num_stencils)**2._wp)
        w5_omega = w5_alpha/sum(w5_alpha)
        w5_alpha(0:weno_num_stencils) = (${D_CBL}$(0:weno_num_stencils,${J}$)*(1._wp + ${D_CBL}$(0:weno_num_stencils, &
                 & ${J}$) - 3._wp*w5_omega(0:weno_num_stencils)) + w5_omega(0:weno_num_stencils)**2._wp) &
                 & *(w5_omega(0:weno_num_stencils)/(${D_CBL}$(0:weno_num_stencils, &
                 & ${J}$)**2._wp + w5_omega(0:weno_num_stencils)*(1._wp - 2._wp*${D_CBL}$(0:weno_num_stencils,${J}$))))
    else if (wenoz) then
        w5_tau = abs(w5_beta(2) - w5_beta(0))
        $:GPU_LOOP(parallelism='[seq]')
        do w5_idx = 0, weno_num_stencils
            w5_alpha(w5_idx) = ${D_CBL}$(w5_idx, ${J}$)*(1._wp + (w5_tau/w5_beta(w5_idx)))
        end do
    else if (teno) then
        w5_tau = abs(w5_beta(2) - w5_beta(0))
        $:GPU_LOOP(parallelism='[seq]')
        do w5_idx = 0, weno_num_stencils
            w5_alpha(w5_idx) = 1._wp + w5_tau/w5_beta(w5_idx)
            w5_alpha(w5_idx) = (w5_alpha(w5_idx)**3._wp)**2._wp
        end do
        w5_omega = w5_alpha/sum(w5_alpha)
        $:GPU_LOOP(parallelism='[seq]')
        do w5_idx = 0, weno_num_stencils
            if (w5_omega(w5_idx) < teno_CT) then
                w5_delta(w5_idx) = 0._wp
            else
                w5_delta(w5_idx) = 1._wp
            end if
            w5_alpha(w5_idx) = w5_delta(w5_idx)*${D_CBL}$(w5_idx, ${J}$)
        end do
    end if
    w5_omega = w5_alpha/sum(w5_alpha)
    ${QL_VAR}$ = w5_omega(0)*w5_poly(0) + w5_omega(1)*w5_poly(1) + w5_omega(2)*w5_poly(2)

    ! Right reconstruction polynomials (reuse w5_dvd, w5_beta, w5_tau from left)
    w5_poly(0) = ${V_WS}$(${J}$, ${K}$, ${L}$, ${IV}$) + ${POLY_CBR}$(${J}$, 0, 0)*w5_dvd(1) + ${POLY_CBR}$(${J}$, 0, 1)*w5_dvd(0)
    w5_poly(1) = ${V_WS}$(${J}$, ${K}$, ${L}$, ${IV}$) + ${POLY_CBR}$(${J}$, 1, 0)*w5_dvd(0) + ${POLY_CBR}$(${J}$, 1, 1)*w5_dvd(-1)
    w5_poly(2) = ${V_WS}$(${J}$, ${K}$, ${L}$, ${IV}$) + ${POLY_CBR}$(${J}$, 2, 0)*w5_dvd(-1) + ${POLY_CBR}$(${J}$, 2, 1)*w5_dvd(-2)

    ! Right weights
    if (wenojs) then
        w5_alpha(0:weno_num_stencils) = ${D_CBR}$(0:weno_num_stencils,${J}$)/(w5_beta(0:weno_num_stencils)**2._wp)
    else if (mapped_weno) then
        w5_alpha(0:weno_num_stencils) = ${D_CBR}$(0:weno_num_stencils,${J}$)/(w5_beta(0:weno_num_stencils)**2._wp)
        w5_omega = w5_alpha/sum(w5_alpha)
        w5_alpha(0:weno_num_stencils) = (${D_CBR}$(0:weno_num_stencils,${J}$)*(1._wp + ${D_CBR}$(0:weno_num_stencils, &
                 & ${J}$) - 3._wp*w5_omega(0:weno_num_stencils)) + w5_omega(0:weno_num_stencils)**2._wp) &
                 & *(w5_omega(0:weno_num_stencils)/(${D_CBR}$(0:weno_num_stencils, &
                 & ${J}$)**2._wp + w5_omega(0:weno_num_stencils)*(1._wp - 2._wp*${D_CBR}$(0:weno_num_stencils,${J}$))))
    else if (wenoz) then
        $:GPU_LOOP(parallelism='[seq]')
        do w5_idx = 0, weno_num_stencils
            w5_alpha(w5_idx) = ${D_CBR}$(w5_idx, ${J}$)*(1._wp + (w5_tau/w5_beta(w5_idx)))
        end do
    else if (teno) then
        $:GPU_LOOP(parallelism='[seq]')
        do w5_idx = 0, weno_num_stencils
            w5_alpha(w5_idx) = w5_delta(w5_idx)*${D_CBR}$(w5_idx, ${J}$)
        end do
    end if
    w5_omega = w5_alpha/sum(w5_alpha)
    ${QR_VAR}$ = w5_omega(0)*w5_poly(0) + w5_omega(1)*w5_poly(1) + w5_omega(2)*w5_poly(2)
#:enddef INLINE_WENO5_RECONSTRUCT
