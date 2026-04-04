#:def arithmetic_avg()
    rho_avg = 5.e-1_wp*(rho_L + rho_R)
    vel_avg_rms = 0._wp
    $:GPU_LOOP(parallelism='[seq]')
    do i = 1, num_vels
        vel_avg_rms = vel_avg_rms + (5.e-1_wp*(vel_L(i) + vel_R(i)))**2._wp
    end do

    H_avg = 5.e-1_wp*(H_L + H_R)
    gamma_avg = 5.e-1_wp*(gamma_L + gamma_R)
    qv_avg = 5.e-1_wp*(qv_L + qv_R)
#:enddef arithmetic_avg

#:def roe_avg()
    rho_avg = sqrt(rho_L*rho_R)

    vel_avg_rms = 0._wp

    $:GPU_LOOP(parallelism='[seq]')
    do i = 1, num_vels
        vel_avg_rms = vel_avg_rms + (sqrt(rho_L)*vel_L(i) + sqrt(rho_R)*vel_R(i))**2._wp/(sqrt(rho_L) + sqrt(rho_R))**2._wp
    end do

    H_avg = (sqrt(rho_L)*H_L + sqrt(rho_R)*H_R)/(sqrt(rho_L) + sqrt(rho_R))

    gamma_avg = (sqrt(rho_L)*gamma_L + sqrt(rho_R)*gamma_R)/(sqrt(rho_L) + sqrt(rho_R))

    vel_avg_rms = (sqrt(rho_L)*vel_L(1) + sqrt(rho_R)*vel_R(1))**2._wp/(sqrt(rho_L) + sqrt(rho_R))**2._wp

    qv_avg = (sqrt(rho_L)*qv_L + sqrt(rho_R)*qv_R)/(sqrt(rho_L) + sqrt(rho_R))

    if (chemistry) then
        eps = 0.001_wp
        call get_species_enthalpies_rt(T_L, h_iL)
        call get_species_enthalpies_rt(T_R, h_iR)
        #:if USING_AMD
            h_iL = h_iL*gas_constant/molecular_weights_nonparameter*T_L
            h_iR = h_iR*gas_constant/molecular_weights_nonparameter*T_R
        #:else
            h_iL = h_iL*gas_constant/molecular_weights*T_L
            h_iR = h_iR*gas_constant/molecular_weights*T_R
        #:endif
        call get_species_specific_heats_r(T_L, Cp_iL)
        call get_species_specific_heats_r(T_R, Cp_iR)

        h_avg_2 = (sqrt(rho_L)*h_iL + sqrt(rho_R)*h_iR)/(sqrt(rho_L) + sqrt(rho_R))
        Yi_avg = (sqrt(rho_L)*Ys_L + sqrt(rho_R)*Ys_R)/(sqrt(rho_L) + sqrt(rho_R))
        T_avg = (sqrt(rho_L)*T_L + sqrt(rho_R)*T_R)/(sqrt(rho_L) + sqrt(rho_R))
        #:if USING_AMD
            if (abs(T_L - T_R) < eps) then
                ! Case when T_L and T_R are very close
                Cp_avg = sum(Yi_avg(:)*(0.5_wp*Cp_iL(:) + 0.5_wp*Cp_iR(:))*gas_constant/molecular_weights_nonparameter(:))
                Cv_avg = sum(Yi_avg(:)*((0.5_wp*Cp_iL(:) + 0.5_wp*Cp_iR(:))*gas_constant/molecular_weights_nonparameter(:) &
                             & - gas_constant/molecular_weights_nonparameter(:)))
            else
                ! Normal calculation when T_L and T_R are sufficiently different
                Cp_avg = sum(Yi_avg(:)*(h_iR(:) - h_iL(:))/(T_R - T_L))
                Cv_avg = sum(Yi_avg(:)*((h_iR(:) - h_iL(:))/(T_R - T_L) - gas_constant/molecular_weights_nonparameter(:)))
            end if
            gamma_avg = Cp_avg/Cv_avg

            Phi_avg(:) = (gamma_avg - 1._wp)*(vel_avg_rms/2.0_wp - h_avg_2(:)) &
                    & + gamma_avg*gas_constant/molecular_weights_nonparameter(:)*T_avg
            c_sum_Yi_Phi = sum(Yi_avg(:)*Phi_avg(:))
        #:else
            if (abs(T_L - T_R) < eps) then
                ! Case when T_L and T_R are very close
                Cp_avg = sum(Yi_avg(:)*(0.5_wp*Cp_iL(:) + 0.5_wp*Cp_iR(:))*gas_constant/molecular_weights(:))
                Cv_avg = sum(Yi_avg(:)*((0.5_wp*Cp_iL(:) + 0.5_wp*Cp_iR(:))*gas_constant/molecular_weights(:) &
                             & - gas_constant/molecular_weights(:)))
            else
                ! Normal calculation when T_L and T_R are sufficiently different
                Cp_avg = sum(Yi_avg(:)*(h_iR(:) - h_iL(:))/(T_R - T_L))
                Cv_avg = sum(Yi_avg(:)*((h_iR(:) - h_iL(:))/(T_R - T_L) - gas_constant/molecular_weights(:)))
            end if
            gamma_avg = Cp_avg/Cv_avg

            Phi_avg(:) = (gamma_avg - 1._wp)*(vel_avg_rms/2.0_wp - h_avg_2(:)) + gamma_avg*gas_constant/molecular_weights(:)*T_avg
            c_sum_Yi_Phi = sum(Yi_avg(:)*Phi_avg(:))
        #:endif
    end if
#:enddef roe_avg

#:def compute_average_state()
    if (avg_state == 1) then
        @:roe_avg()
    end if

    if (avg_state == 2) then
        @:arithmetic_avg()
    end if
#:enddef compute_average_state

#:def compute_low_Mach_correction()
    if (riemann_solver == 1 .or. riemann_solver == 5) then
        zcoef = min(1._wp, max(vel_L_rms**5.e-1_wp/c_L, vel_R_rms**5.e-1_wp/c_R))
        pcorr = 0._wp

        if (low_Mach == 1) then
            pcorr = -(s_P - s_M)*(rho_L + rho_R)/8._wp*(zcoef - 1._wp)
        end if
    else if (riemann_solver == 2) then
        zcoef = min(1._wp, max(vel_L_rms**5.e-1_wp/c_L, vel_R_rms**5.e-1_wp/c_R))
        pcorr = 0._wp

        if (low_Mach == 1) then
            pcorr = rho_L*rho_R*(s_L - vel_L(dir_idx(1)))*(s_R - vel_R(dir_idx(1)))*(vel_R(dir_idx(1)) - vel_L(dir_idx(1))) &
                                 & /(rho_R*(s_R - vel_R(dir_idx(1))) - rho_L*(s_L - vel_L(dir_idx(1))))*(zcoef - 1._wp)
        else if (low_Mach == 2) then
            vel_L_tmp = 5.e-1_wp*((vel_L(dir_idx(1)) + vel_R(dir_idx(1))) + zcoef*(vel_L(dir_idx(1)) - vel_R(dir_idx(1))))
            vel_R_tmp = 5.e-1_wp*((vel_L(dir_idx(1)) + vel_R(dir_idx(1))) + zcoef*(vel_R(dir_idx(1)) - vel_L(dir_idx(1))))
            vel_L(dir_idx(1)) = vel_L_tmp
            vel_R(dir_idx(1)) = vel_R_tmp
        end if
    end if
#:enddef compute_low_Mach_correction

!> Inline HLLC flux computation for per-cell fused kernel. Reads from qL_local/qR_local (already populated by WENO). Writes the
!! Riemann flux to FLUX_ARR(1:sys_size) and the normal advection velocity to VEL_SRC_VAR. All intermediate variables (rho_L, vel_L,
!! etc.) must already be declared in the enclosing scope.
#:def INLINE_HLLC_FLUX(FLUX_ARR, VEL_SRC_VAR)
    vel_L_rms = 0._wp; vel_R_rms = 0._wp
    rho_L = 0._wp; rho_R = 0._wp
    gamma_L = 0._wp; gamma_R = 0._wp
    pi_inf_L = 0._wp; pi_inf_R = 0._wp
    qv_L = 0._wp; qv_R = 0._wp
    alpha_L_sum = 0._wp; alpha_R_sum = 0._wp

    $:GPU_LOOP(parallelism='[seq]')
    do i = 1, num_fluids
        alpha_L(i) = qL_local(E_idx + i)
        alpha_R(i) = qR_local(E_idx + i)
    end do

    $:GPU_LOOP(parallelism='[seq]')
    do i = 1, num_dims
        vel_L(i) = qL_local(contxe + i)
        vel_R(i) = qR_local(contxe + i)
        vel_L_rms = vel_L_rms + vel_L(i)**2._wp
        vel_R_rms = vel_R_rms + vel_R(i)**2._wp
    end do

    pres_L = qL_local(E_idx)
    pres_R = qR_local(E_idx)

    if (mpp_lim) then
        $:GPU_LOOP(parallelism='[seq]')
        do i = 1, num_fluids
            qL_local(i) = max(0._wp, qL_local(i))
            qL_local(E_idx + i) = min(max(0._wp, qL_local(E_idx + i)), 1._wp)
            qR_local(i) = max(0._wp, qR_local(i))
            qR_local(E_idx + i) = min(max(0._wp, qR_local(E_idx + i)), 1._wp)
            alpha_L_sum = alpha_L_sum + qL_local(E_idx + i)
            alpha_R_sum = alpha_R_sum + qR_local(E_idx + i)
        end do
        $:GPU_LOOP(parallelism='[seq]')
        do i = 1, num_fluids
            qL_local(E_idx + i) = qL_local(E_idx + i)/max(alpha_L_sum, sgm_eps)
            qR_local(E_idx + i) = qR_local(E_idx + i)/max(alpha_R_sum, sgm_eps)
        end do
        $:GPU_LOOP(parallelism='[seq]')
        do i = 1, num_fluids
            alpha_L(i) = qL_local(E_idx + i)
            alpha_R(i) = qR_local(E_idx + i)
        end do
    end if

    $:GPU_LOOP(parallelism='[seq]')
    do i = 1, num_fluids
        rho_L = rho_L + qL_local(i)
        gamma_L = gamma_L + qL_local(E_idx + i)*gammas(i)
        pi_inf_L = pi_inf_L + qL_local(E_idx + i)*pi_infs(i)
        qv_L = qv_L + qL_local(i)*qvs(i)
        rho_R = rho_R + qR_local(i)
        gamma_R = gamma_R + qR_local(E_idx + i)*gammas(i)
        pi_inf_R = pi_inf_R + qR_local(E_idx + i)*pi_infs(i)
        qv_R = qv_R + qR_local(i)*qvs(i)
    end do

    E_L = gamma_L*pres_L + pi_inf_L + 5.e-1*rho_L*vel_L_rms + qv_L
    E_R = gamma_R*pres_R + pi_inf_R + 5.e-1*rho_R*vel_R_rms + qv_R
    H_L = (E_L + pres_L)/rho_L
    H_R = (E_R + pres_R)/rho_R

    if (hypoelasticity) then
        $:GPU_LOOP(parallelism='[seq]')
        do i = 1, strxe - strxb + 1
            tau_e_L(i) = qL_local(strxb - 1 + i)
            tau_e_R(i) = qR_local(strxb - 1 + i)
        end do
        G_L = 0._wp; G_R = 0._wp
        $:GPU_LOOP(parallelism='[seq]')
        do i = 1, num_fluids
            G_L = G_L + alpha_L(i)*Gs_rs(i)
            G_R = G_R + alpha_R(i)*Gs_rs(i)
        end do
        $:GPU_LOOP(parallelism='[seq]')
        do i = 1, strxe - strxb + 1
            if ((G_L > verysmall) .and. (G_R > verysmall)) then
                E_L = E_L + (tau_e_L(i)*tau_e_L(i))/(4._wp*G_L)
                E_R = E_R + (tau_e_R(i)*tau_e_R(i))/(4._wp*G_R)
                if ((i == 2) .or. (i == 4) .or. (i == 5)) then
                    E_L = E_L + (tau_e_L(i)*tau_e_L(i))/(4._wp*G_L)
                    E_R = E_R + (tau_e_R(i)*tau_e_R(i))/(4._wp*G_R)
                end if
            end if
        end do
    end if

    if (hyperelasticity) then
        $:GPU_LOOP(parallelism='[seq]')
        do i = 1, num_dims
            xi_field_L(i) = qL_local(xibeg - 1 + i)
            xi_field_R(i) = qR_local(xibeg - 1 + i)
        end do
        G_L = 0._wp; G_R = 0._wp
        $:GPU_LOOP(parallelism='[seq]')
        do i = 1, num_fluids
            G_L = G_L + alpha_L(i)*Gs_rs(i)
            G_R = G_R + alpha_R(i)*Gs_rs(i)
        end do
        if (G_L > verysmall .and. G_R > verysmall) then
            E_L = E_L + G_L*qL_local(xiend + 1)
            E_R = E_R + G_R*qR_local(xiend + 1)
        end if
        $:GPU_LOOP(parallelism='[seq]')
        do i = 1, b_size - 1
            tau_e_L(i) = qL_local(strxb - 1 + i)
            tau_e_R(i) = qR_local(strxb - 1 + i)
        end do
    end if

    H_L = (E_L + pres_L)/rho_L
    H_R = (E_R + pres_R)/rho_R

    @:compute_average_state()

    call s_compute_speed_of_sound(pres_L, rho_L, gamma_L, pi_inf_L, H_L, alpha_L, vel_L_rms, 0._wp, c_L, qv_L)
    call s_compute_speed_of_sound(pres_R, rho_R, gamma_R, pi_inf_R, H_R, alpha_R, vel_R_rms, 0._wp, c_R, qv_R)
    call s_compute_speed_of_sound(pres_R, rho_avg, gamma_avg, pi_inf_R, H_avg, alpha_R, vel_avg_rms, c_sum_Yi_Phi, c_avg, qv_avg)

    if (low_Mach == 2) then
        @:compute_low_Mach_correction()
    end if

    if (wave_speeds == 1) then
        if (elasticity) then
            s_L = min(vel_L(dir_idx(1)) - sqrt(c_L*c_L + (((4._wp*G_L)/3._wp) + tau_e_L(dir_idx_tau(1)))/rho_L), &
                      & vel_R(dir_idx(1)) - sqrt(c_R*c_R + (((4._wp*G_R)/3._wp) + tau_e_R(dir_idx_tau(1)))/rho_R))
            s_R = max(vel_R(dir_idx(1)) + sqrt(c_R*c_R + (((4._wp*G_R)/3._wp) + tau_e_R(dir_idx_tau(1)))/rho_R), &
                      & vel_L(dir_idx(1)) + sqrt(c_L*c_L + (((4._wp*G_L)/3._wp) + tau_e_L(dir_idx_tau(1)))/rho_L))
            s_S = (pres_R - tau_e_R(dir_idx_tau(1)) - pres_L + tau_e_L(dir_idx_tau(1)) + rho_L*vel_L(dir_idx(1))*(s_L &
                   & - vel_L(dir_idx(1))) - rho_R*vel_R(dir_idx(1))*(s_R - vel_R(dir_idx(1))))/(rho_L*(s_L - vel_L(dir_idx(1))) &
                   & - rho_R*(s_R - vel_R(dir_idx(1))))
        else
            s_L = min(vel_L(dir_idx(1)) - c_L, vel_R(dir_idx(1)) - c_R)
            s_R = max(vel_R(dir_idx(1)) + c_R, vel_L(dir_idx(1)) + c_L)
            s_S = (pres_R - pres_L + rho_L*vel_L(dir_idx(1))*(s_L - vel_L(dir_idx(1))) - rho_R*vel_R(dir_idx(1))*(s_R &
                   & - vel_R(dir_idx(1))))/(rho_L*(s_L - vel_L(dir_idx(1))) - rho_R*(s_R - vel_R(dir_idx(1))))
        end if
    else if (wave_speeds == 2) then
        pres_SL = 5.e-1_wp*(pres_L + pres_R + rho_avg*c_avg*(vel_L(dir_idx(1)) - vel_R(dir_idx(1))))
        pres_SR = pres_SL
        Ms_L = max(1._wp, &
                   & sqrt(1._wp + ((5.e-1_wp + gamma_L)/(1._wp + gamma_L))*(pres_SL/pres_L - 1._wp)*pres_L/((pres_L &
                   & + pi_inf_L/(1._wp + gamma_L)))))
        Ms_R = max(1._wp, &
                   & sqrt(1._wp + ((5.e-1_wp + gamma_R)/(1._wp + gamma_R))*(pres_SR/pres_R - 1._wp)*pres_R/((pres_R &
                   & + pi_inf_R/(1._wp + gamma_R)))))
        s_L = vel_L(dir_idx(1)) - c_L*Ms_L
        s_R = vel_R(dir_idx(1)) + c_R*Ms_R
        s_S = 5.e-1_wp*((vel_L(dir_idx(1)) + vel_R(dir_idx(1))) + (pres_L - pres_R)/(rho_avg*c_avg))
    end if

    s_M = min(0._wp, s_L); s_P = max(0._wp, s_R)
    xi_L = (s_L - vel_L(dir_idx(1)))/(s_L - s_S)
    xi_R = (s_R - vel_R(dir_idx(1)))/(s_R - s_S)
    xi_M = (5.e-1_wp + sign(5.e-1_wp, s_S))
    xi_P = (5.e-1_wp - sign(5.e-1_wp, s_S))

    if (low_Mach == 1) then
        @:compute_low_Mach_correction()
    else
        pcorr = 0._wp
    end if

    ! Mass flux
    $:GPU_LOOP(parallelism='[seq]')
    do i = 1, contxe
        ${FLUX_ARR}$(i) = xi_M*qL_local(i)*(vel_L(dir_idx(1)) + s_M*(xi_L - 1._wp)) + xi_P*qR_local(i)*(vel_R(dir_idx(1)) &
                     & + s_P*(xi_R - 1._wp))
    end do

    ! Momentum flux
    $:GPU_LOOP(parallelism='[seq]')
    do i = 1, num_dims
        ${FLUX_ARR}$(contxe + dir_idx(i)) = xi_M*(rho_L*(vel_L(dir_idx(1))*vel_L(dir_idx(i)) + s_M*(xi_L*(dir_flg(dir_idx(i))*s_S &
                     & + (1._wp - dir_flg(dir_idx(i)))*vel_L(dir_idx(i))) - vel_L(dir_idx(i)))) + dir_flg(dir_idx(i))*(pres_L)) &
                     & + xi_P*(rho_R*(vel_R(dir_idx(1))*vel_R(dir_idx(i)) + s_P*(xi_R*(dir_flg(dir_idx(i))*s_S + (1._wp &
                     & - dir_flg(dir_idx(i)))*vel_R(dir_idx(i))) - vel_R(dir_idx(i)))) + dir_flg(dir_idx(i))*(pres_R)) + (s_M/s_L) &
                     & *(s_P/s_R)*dir_flg(dir_idx(i))*pcorr
    end do

    ! Energy flux
    ${FLUX_ARR}$(E_idx) = xi_M*(vel_L(dir_idx(1))*(E_L + pres_L) + s_M*(xi_L*(E_L + (s_S - vel_L(dir_idx(1)))*(rho_L*s_S &
                 & + pres_L/(s_L - vel_L(dir_idx(1))))) - E_L)) + xi_P*(vel_R(dir_idx(1))*(E_R + pres_R) + s_P*(xi_R*(E_R + (s_S &
                 & - vel_R(dir_idx(1)))*(rho_R*s_S + pres_R/(s_R - vel_R(dir_idx(1))))) - E_R)) + (s_M/s_L)*(s_P/s_R)*pcorr*s_S

    ! Elastic shear stress additions
    if (elasticity) then
        flux_ene_e = 0._wp
        $:GPU_LOOP(parallelism='[seq]')
        do i = 1, num_dims
            ${FLUX_ARR}$(contxe + dir_idx(i)) = ${FLUX_ARR}$(contxe + dir_idx(i)) - xi_M*tau_e_L(dir_idx_tau(i)) &
                         & - xi_P*tau_e_R(dir_idx_tau(i))
            flux_ene_e = flux_ene_e - xi_M*(vel_L(dir_idx(i))*tau_e_L(dir_idx_tau(i)) + s_M*(xi_L*((s_S - vel_L(i)) &
                                            & *(tau_e_L(dir_idx_tau(i))/(s_L - vel_L(i)))))) - xi_P*(vel_R(dir_idx(i)) &
                                            & *tau_e_R(dir_idx_tau(i)) + s_P*(xi_R*((s_S - vel_R(i))*(tau_e_R(dir_idx_tau(i)) &
                                            & /(s_R - vel_R(i))))))
        end do
        ${FLUX_ARR}$(E_idx) = ${FLUX_ARR}$(E_idx) + flux_ene_e
    end if

    ! Hypoelastic stress evolution flux
    if (hypoelasticity) then
        $:GPU_LOOP(parallelism='[seq]')
        do i = 1, strxe - strxb + 1
            ${FLUX_ARR}$(strxb - 1 + i) = xi_M*(s_S/(s_L - s_S))*(s_L*rho_L*tau_e_L(i) - rho_L*vel_L(dir_idx(1))*tau_e_L(i)) &
                         & + xi_P*(s_S/(s_R - s_S))*(s_R*rho_R*tau_e_R(i) - rho_R*vel_R(dir_idx(1))*tau_e_R(i))
        end do
    end if

    ! Volume fraction flux
    $:GPU_LOOP(parallelism='[seq]')
    do i = advxb, advxe
        ${FLUX_ARR}$(i) = xi_M*qL_local(i)*(vel_L(dir_idx(1)) + s_M*(xi_L - 1._wp)) + xi_P*qR_local(i)*(vel_R(dir_idx(1)) &
                     & + s_P*(xi_R - 1._wp))
    end do

    ! Hyperelastic reference map flux
    if (hyperelasticity) then
        $:GPU_LOOP(parallelism='[seq]')
        do i = 1, num_dims
            ${FLUX_ARR}$(xibeg - 1 + i) = xi_M*(s_S/(s_L - s_S))*(s_L*rho_L*xi_field_L(i) - rho_L*vel_L(dir_idx(1))*xi_field_L(i)) &
                         & + xi_P*(s_S/(s_R - s_S))*(s_R*rho_R*xi_field_R(i) - rho_R*vel_R(dir_idx(1))*xi_field_R(i))
        end do
    end if

    ! Normal advection velocity at this face (for volume fraction source terms)
    ${VEL_SRC_VAR}$ = xi_M*(vel_L(dir_idx(1)) + dir_flg(dir_idx(1))*s_M*(xi_L - 1._wp)) + xi_P*(vel_R(dir_idx(1)) &
                            & + dir_flg(dir_idx(1))*s_P*(xi_R - 1._wp))
#:enddef INLINE_HLLC_FLUX
