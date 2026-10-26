!>
!! @file
!! @brief Contains module m_ibm

#:include 'macros.fpp'

!> @brief Ghost-node immersed boundary method: locates ghost/image points, computes interpolation coefficients, and corrects the
!! flow state
module m_ibm

    use m_derived_types
    use m_global_parameters
    use m_mpi_proxy
    use m_variables_conversion
    use m_eos
    use m_helper
    use m_helper_basic
    use m_constants
    use m_compute_levelset
    use m_ib_patches
    use m_model
    use m_patch_geometries
    use m_collisions
    use m_thermochem, only: num_species, gas_constant, molecular_weights, get_mixture_molecular_weight, get_mixture_energy_mass, &
        & get_mixture_thermal_conductivity_mixavg, get_species_mass_diffusivities_mixavg

    use m_surface_thermochem, only: get_surface_net_production_rates, get_surface_reaction_heat_flux

    implicit none

    private :: s_compute_image_points, s_compute_interpolation_coeffs, s_interpolate_image_point, s_find_ghost_points, &
        & s_find_num_ghost_points, s_compute_ghost_point_pressure, s_compute_ghost_point_velocity
    ; public :: s_initialize_ibm_module, s_ibm_setup, s_ibm_correct_state, s_finalize_ibm_module, s_report_ibm_surface, &
        & s_write_ib_surface, s_update_ib_temperatures, s_conserve_ib_mass, s_accumulate_ib_face_fluxes, &
        & s_ibm_set_conduction_temperature, s_ibm_restore_conduction_temperature, s_add_swept_cells, s_cutvol_advance

    !> Ghost points at which the reacting-surface Newton solve did not reach its tolerance, so the point fell back to a chemically
    !! inert wall (keeping a prescribed Twall). Counted because that fallback is otherwise indistinguishable from a surface
    !! mechanism that simply does nothing, and a run can look converged while the surface chemistry never engaged.
    !> Tier tallies of the two-probe wall treatment (ib_wall_order = 2), summed over the ghost points updated: tier 1 the two-probe
    !! quadratic, tier 2 the one-probe linear fallback where a second probe has too little fluid around it, tier 3 the caller's own
    !! treatment. Reported because a tight packing can sit mostly in tier 2, and a second-order claim is then not true of that
    !! surface. Gradient tiers are for the surface balance, fill tiers for the ghost state the flow solve sees.
    integer, dimension(3) :: n_grad_tier = 0, n_fill_tier = 0
    integer               :: n_surface_not_converged = 0

    !> Ghost points whose levelset distance was not positive, leaving the surface gradient (X_IP - X_s)/d undefined. Separate from
    !! the counter above because this is a grid/geometry degeneracy, not a kinetics failure.
    integer                     :: n_surface_ill_posed = 0
    type(integer_field), public :: ib_markers
    $:GPU_DECLARE(create='[ib_markers]')

    !> 1 at ghost points whose image point reaches the fluid, 0 elsewhere. Ghost points with a buried image point average over these
    !! neighbors, and never over each other.
    type(integer_field) :: corrected_gps
    $:GPU_DECLARE(create='[corrected_gps]')
    type(integer_field) :: ib_markers_prev  !< Markers at the last placement (moving bodies with ib_exchange)
    $:GPU_DECLARE(create='[ib_markers_prev]')
    type(integer_field) :: ib_shell  !< ib_mass_relax: global IB id of each fluid cell within ib_shell_width cells of it, else 0
    $:GPU_DECLARE(create='[ib_shell]')
    !> ib_cutvol (conservation-only cut cells, Seo & Mittal JCP 2011): fluid fraction of each fluid cell (1 away from a body), its
    !! value and the markers before the step's move, and the translation velocity of each body cell (0 in a fixed body)
    type(scalar_field)                        :: ib_alpha, ib_alpha_prev
    type(scalar_field)                        :: ib_cv_f  !< share of each donor an uncovered cell takes
    type(integer_field)                       :: ib_markers_cv
    real(wp), allocatable, dimension(:,:,:,:) :: ib_wall_vel
    $:GPU_DECLARE(create='[ib_alpha, ib_alpha_prev, ib_cv_f, ib_markers_cv, ib_wall_vel]')

    !> Quadratic wall-normal fill of one ghost point: two probes along the surface normal, with TSC weights, and the Lagrange
    !! weights that extrapolate the profile through them back to the ghost (see s_wall_fill).
    type :: wall_fill
        integer :: tier  !< 1 quadratic, 2 linear (the mirror), 3 off or too little fluid: the caller's own treatment
        real(wp), dimension(3) :: lw  !< Lagrange weights on (phi_w, phi_1, phi_2)
        real(wp), dimension(2) :: llin  !< The linear mirror on (phi_w, phi_1), which bounds the quadratic
        integer, dimension(3) :: ic1, ic2  !< TSC stencil centres of the two probes
        real(wp), dimension(3, 3) :: wd1, wd2  !< TSC weights of the two probes
    end type wall_fill

    !> Wall-normal temperature gradient for the surface balance, dT/dn|_w = a - b*T_s (linear in the Newton unknown T_s). Where quad
    !! is false the one-sided difference (T_s - T_IP)/d is used, as before.
    type :: wall_grad
        logical  :: quad
        real(wp) :: a, b
    end type wall_grad

    type(ghost_point), dimension(:), allocatable :: ghost_points

    !> New primitive state per ghost point, staged so it is published only after every image-point stencil has been read
    !! (MFlowCode/MFC#1901). Writing q_prim inside the same loop that reads it at neighboring image-point stencils makes a ghost's
    !! result depend on whether its neighbors were updated first.
    real(wp), dimension(:,:), allocatable :: gp_prim_new
    $:GPU_DECLARE(create='[ghost_points]')

    !> Surface record per ghost point from the latest s_ibm_correct_state, kept when ib_surface_wrt: (1) area weight, (2) wall
    !! temperature, (3) gasified mass flux. Only ghost points within two cells of the surface carry a weight, so summing weight*flux
    !! over them integrates over the surface; see s_record_gp_surface.
    integer, parameter                    :: ib_surf_nvars = 3
    real(wp), allocatable, dimension(:,:) :: gp_surf
    $:GPU_DECLARE(create='[gp_surf]')

    !> gp_surf is filled when it is written (ib_surface_wrt) or integrated (lumped_ib, ib_heat_wrt).
    logical         :: record_surface
    logical, public :: lumped_ib    !< Some IB has thermal_bc = 3
    logical, public :: ib_exchange  !< ib_heat is accumulated: lumped_ib, ib_heat_wrt or ib_mass_relax
    $:GPU_DECLARE(create='[record_surface]')

    !> Per global IB, for the thermal_bc = 3 update and ib_heat_wrt: (1) surface area and (4) gasification rate, from gp_surf; (2)
    !! energy into the solid and (3) mass out of it, both summed over the faces between the body's cells and fluid cells with the RK
    !! stage weights, so over a step they are exactly the energy and mass the fluid exchanged with the body.
    real(wp), allocatable, dimension(:,:) :: ib_heat
    $:GPU_DECLARE(create='[ib_heat]')

    !> ib_mass_relax, per global IB: (1) mass the gas gained from the body beyond its gasification and not yet returned, (2) the
    !! share of it removed this step; ib_shell_mass(1, :): gas mass of the body's shell.
    real(wp), allocatable, dimension(:,:) :: ib_mass_res, ib_shell_mass
    $:GPU_DECLARE(create='[ib_mass_res, ib_shell_mass]')
    integer, parameter :: ib_shell_width = 2
    integer            :: num_gps  !< Number of ghost points
#if defined(MFC_OpenACC)
    $:GPU_DECLARE(create='[num_gps]')
#elif defined(MFC_OpenMP)
    $:GPU_DECLARE(create='[num_gps]')
#endif
    logical :: moving_immersed_boundary_flag
    logical :: centroid_offsets_active = .false.  !< some patch is a moving airfoil or STL, which carries a centroid_offset

    ! IB force reduction (s_communicate_ib_forces): distinct neighborhood ranks in ascending order, and buffers allocated once
    integer, parameter    :: ib_rec_len = 7  !< gbl_patch_id, force(1:3), torque(1:3)
    integer               :: n_ib_nbrs = 0
    integer, allocatable  :: ib_nbrs(:), ib_send_counts(:), ib_reqs(:), ib_stats(:,:)
    real(wp), allocatable :: ib_send_bufs(:,:), ib_recv_bufs(:,:)
    private :: ib_rec_len, n_ib_nbrs, ib_nbrs, ib_send_counts, ib_reqs, ib_stats, ib_send_bufs, ib_recv_bufs

    !> Force (1:3) and torque (4:6) on each local IB from the fluid-body face fluxes of the last RHS evaluation
    real(wp), allocatable, dimension(:,:) :: ib_face_ft
    $:GPU_DECLARE(create='[ib_face_ft]')

contains

    !> Shell layer of the cell (I, J, K) into LAY: 0 if it needs no ghost value, 1 .. gp_layers_full for a cell within that box
    !! (Chebyshev) distance of a fluid cell, which is filled from an image point, and gp_layers_full + 1 for a deeper cell.
    !!
    !! Deep cells exist only when gp_layers > gp_layers_full, i.e. for a viscous reconstruction that reaches further into the
    !! body (see f_ghost_point_layers). Anything such a stencil reads has to hold a current value, not the state the cell was
    !! initialised with. Those stencils run one coordinate direction at a time, so the cells that matter form a cross, and a
    !! cell is deep when a fluid cell lies within a Manhattan distance of gp_layers. A box of that radius would swallow all of a
    !! small body: the inscribed radius of a cube of half-width 6 is 6/sqrt(3) of a sphere's radius in cells.
    !!
    !! Inlined in the kernels that need it: as a device function it found no ghost points under NVHPC OpenACC. Needs private
    !! integers gl_di, gl_dj, gl_dk, gl_dk_max, gl_rem_j, gl_rem_k, gl_cheb.
    #:def GHOST_LAYER(LAY, I, J, K)
        ${LAY}$ = 0
        if (ib_markers%sf(${I}$, ${J}$, ${K}$) /= 0) then
            gl_dk_max = merge(gp_layers_full, 0, p > 0)
            gl_cheb = gp_layers_full + 1
            do gl_di = -gp_layers_full, gp_layers_full
                do gl_dj = -gp_layers_full, gp_layers_full
                    do gl_dk = -gl_dk_max, gl_dk_max
                        if (ib_markers%sf(${I}$ + gl_di, ${J}$ + gl_dj, ${K}$ + gl_dk) == 0) gl_cheb = min(gl_cheb, &
                            & max(abs(gl_di), abs(gl_dj), abs(gl_dk)))
                    end do
                end do
            end do
            if (gl_cheb <= gp_layers_full) then
                ${LAY}$ = gl_cheb
            else if (gp_layers > gp_layers_full) then
                gl_dk_max = merge(gp_layers, 0, p > 0)
                do gl_di = -gp_layers, gp_layers
                    gl_rem_j = gp_layers - abs(gl_di)
                    do gl_dj = -gl_rem_j, gl_rem_j
                        gl_rem_k = min(gl_rem_j - abs(gl_dj), gl_dk_max)
                        do gl_dk = -gl_rem_k, gl_rem_k
                            if (ib_markers%sf(${I}$ + gl_di, ${J}$ + gl_dj, ${K}$ + gl_dk) == 0) ${LAY}$ = gp_layers_full + 1
                        end do
                    end do
                end do
            end if
        end if
    #:enddef

    !> Allocates memory for the variables in the IBM module
    impure subroutine s_initialize_ibm_module()

        if (p > 0) then
            @:ALLOCATE(ib_markers%sf(-buff_size:m+buff_size, -buff_size:n+buff_size, -buff_size:p+buff_size))
            @:ALLOCATE(corrected_gps%sf(-buff_size:m+buff_size, -buff_size:n+buff_size, -buff_size:p+buff_size))
        else
            @:ALLOCATE(ib_markers%sf(-buff_size:m+buff_size, -buff_size:n+buff_size, 0:0))
            @:ALLOCATE(corrected_gps%sf(-buff_size:m+buff_size, -buff_size:n+buff_size, 0:0))
        end if

        @:ACC_SETUP_SFs(ib_markers)
        @:ACC_SETUP_SFs(corrected_gps)

        $:GPU_ENTER_DATA(copyin='[num_gps]')
        $:GPU_UPDATE(device='[ib_second_order_vel, ib_ip_min_dist, ib_cutvol]')

        if (collision_model > 0) call s_initialize_collisions_module()

    end subroutine s_initialize_ibm_module

    !> Initializes the values of various IBM variables, such as ghost points and image points.
    impure subroutine s_ibm_setup()

        integer         :: i, j, k, gid
        integer(kind=8) :: n_need
        real(wp)        :: t_init  !< initial time for prescribed kinematics
        real(wp)        :: max_num_gps_rank
        integer(kind=8) :: max_num_gps
        integer(kind=8) :: n_lumped
        integer         :: zb

        call nvtxStartRange("SETUP-IBM-MODULE")

        ! GPU routines require updated cell centers
        $:GPU_UPDATE(device='[num_ibs, num_gbl_ibs, x_cc, y_cc, dx, dy, ib_bc_x%beg, ib_bc_x%end, ib_bc_y%beg, ib_bc_y%end]')
        if (p /= 0) then
            $:GPU_UPDATE(device='[z_cc, dz, ib_bc_z%beg, ib_bc_z%end]')
        end if
        $:GPU_UPDATE(device='[patch_ib(1:num_ibs), glb_bounds]')

        ! do all set up for moving immersed boundaries; prescribed kinematics are evaluated at the initial time so the
        ! first stage already sees the correct body state, velocity and angular velocity
        t_init = t_step_start*dt
        if (cfl_dt) t_init = t_save*n_start
        $:GPU_PARALLEL_LOOP(private='[i]', copyin='[t_init]')
        do i = 1, num_ibs
            if (patch_ib(i)%moving_ibm /= 0) then
                if (patch_ib(i)%kin_model > 0) call s_prescribed_kinematics(i, t_init)
                call s_compute_moment_of_inertia(patch_ib(i), patch_ib(i)%angular_vel, patch_ib(i)%moment)
            end if
            call s_update_ib_rotation_matrix(i)
        end do
        $:END_GPU_PARALLEL_LOOP()
        $:GPU_UPDATE(host='[patch_ib(1:num_ibs)]')

        ! allocate some arrays for MPI communication, if required by this simulation
#ifdef MFC_MPI
        if (num_procs > 1) call s_setup_ib_force_comm()
#endif

        @:ALLOCATE(ib_face_ft(6, num_local_ibs_max))
        ib_face_ft = 0._wp
        $:GPU_UPDATE(device='[ib_face_ft]')

        call s_update_ib_lookup()

        ! recompute the new ib_patch locations
        ib_markers%sf = 0._wp
        corrected_gps%sf = 0
        $:GPU_UPDATE(device='[ib_markers%sf, corrected_gps%sf]')
        call s_apply_ib_patches(ib_markers)
        $:GPU_UPDATE(host='[ib_markers%sf]')
        call s_check_every_patch_marked()
        ! One reduction decides whether any patch needs an offset, so cases without one (particle clouds have thousands of
        ! patches) skip the per-patch collectives. Then loop over global ids: ranks hold different patches.
        call s_mpi_allreduce_integer_sum(int(count([(f_needs_centroid_offset(patch_ib(i)), i=1, num_ibs)]), 8), n_need)
        centroid_offsets_active = n_need > 0_8
        if (centroid_offsets_active) then
            do gid = 1, num_gbl_ibs
                call s_get_neighborhood_idx(gid, i)
                call s_compute_centroid_offset(gid, i)
            end do
            call s_restore_centroid_offsets(t_init)
        end if
        do i = 1, num_ibs
            $:GPU_UPDATE(device='[patch_ib(i)]')
        end do

        ! find the number of ghost points and set them to be the maximum total across ranks
        call s_find_num_ghost_points(num_gps)
        if (moving_immersed_boundary_flag) then
            ! Every rank allocates the same size, which must hold any one rank's peak as the body moves: twice the global
            ! total, but at least eight times the largest per-rank count (a thin body crossing few of many ranks)
            call s_mpi_allreduce_integer_sum(int(num_gps, 8), max_num_gps)
            call s_mpi_allreduce_max(real(num_gps, wp), max_num_gps_rank)
            max_num_gps = max(max_num_gps*2_8, 8_8*int(max_num_gps_rank, 8))
            max_num_gps = min(max_num_gps, int(m + 1, 8)*int(n + 1, 8)*int(p + 1, 8))
        else
            max_num_gps = int(num_gps, 8)
        end if

        ! set the size of the ghost point arrays to be the amount of points total, plus a factor of 2 buffer
        $:GPU_UPDATE(device='[num_gps]')
        @:ALLOCATE(ghost_points(1:max_num_gps))
        @:ALLOCATE(gp_prim_new(1:sys_size, 1:max_num_gps))

        $:GPU_ENTER_DATA(copyin='[ghost_points, gp_prim_new]')
        ! Global, not per rank: s_update_ib_temperatures is collective, and a rank whose neighborhood holds no thermal_bc = 3
        ! body must still take part in its allreduce.
        call s_mpi_allreduce_integer_sum(int(count(patch_ib(1:num_ibs)%thermal_bc == 3), 8), n_lumped)
        lumped_ib = n_lumped > 0
        record_surface = ib_surface_wrt .or. lumped_ib .or. ib_heat_wrt .or. ib_mass_relax > 0
        $:GPU_UPDATE(device='[record_surface]')
        if (record_surface) then
            @:ALLOCATE(gp_surf(ib_surf_nvars, 1:max_num_gps))
            gp_surf = 0._wp
            $:GPU_UPDATE(device='[gp_surf]')
        end if
        ib_exchange = lumped_ib .or. ib_heat_wrt .or. ib_mass_relax > 0
        if (ib_exchange) then
            @:ALLOCATE(ib_heat(4, num_gbl_ibs))
            ib_heat = 0._wp
            $:GPU_UPDATE(device='[ib_heat]')
        end if
        if (ib_mass_relax > 0) then
            @:ALLOCATE(ib_mass_res(2, num_gbl_ibs), ib_shell_mass(1, num_gbl_ibs))
            ib_mass_res = 0._wp
            $:GPU_UPDATE(device='[ib_mass_res]')
            @:ALLOCATE(ib_shell%sf(0:m, 0:n, 0:p))
            @:ACC_SETUP_SFs(ib_shell)
        end if
        if (ib_exchange .and. moving_immersed_boundary_flag) then
            @:ALLOCATE(ib_markers_prev%sf(0:m, 0:n, 0:p))
            @:ACC_SETUP_SFs(ib_markers_prev)
            ib_markers_prev%sf(0:m,0:n,0:p) = ib_markers%sf(0:m,0:n,0:p)
            $:GPU_UPDATE(device='[ib_markers_prev%sf]')
        end if
        if (ib_cutvol) then
            zb = 0; if (p > 0) zb = buff_size
            @:ALLOCATE(ib_alpha%sf(-buff_size:m + buff_size, -buff_size:n + buff_size, -zb:p + zb))
            @:ACC_SETUP_SFs(ib_alpha)
            @:ALLOCATE(ib_alpha_prev%sf(-buff_size:m + buff_size, -buff_size:n + buff_size, -zb:p + zb))
            @:ACC_SETUP_SFs(ib_alpha_prev)
            @:ALLOCATE(ib_cv_f%sf(-buff_size:m + buff_size, -buff_size:n + buff_size, -zb:p + zb))
            @:ACC_SETUP_SFs(ib_cv_f)
            if (buff_size < 4) call s_mpi_abort('ib_cutvol needs buff_size >= 4 (an uncovered cell draws from cells two away)')
            @:ALLOCATE(ib_markers_cv%sf(-buff_size:m + buff_size, -buff_size:n + buff_size, -zb:p + zb))
            @:ACC_SETUP_SFs(ib_markers_cv)
            @:ALLOCATE(ib_wall_vel(1:3, -buff_size:m + buff_size, -buff_size:n + buff_size, -zb:p + zb))
            call s_cutvol_fields()
        end if
        ! Ghost-cell IBM, Tseng & Ferziger JCP (2003), Mittal & Iaccarino ARFM (2005)
        call s_find_ghost_points()
        call s_apply_levelset(ghost_points, num_gps)

        call s_compute_image_points()
        call s_compute_interpolation_coeffs()

        call nvtxEndRange

    end subroutine s_ibm_setup

    !> Velocity of the wall point nearest a ghost point: the body's translation plus its rotation about the centroid. Zero for a
    !! fixed body.
    function f_wall_velocity(gp, gp_patch_id, radial_vector) result(vel_w)

        $:GPU_ROUTINE(parallelism='[seq]')

        type(ghost_point), intent(in)      :: gp
        integer, intent(in)                :: gp_patch_id
        real(wp), dimension(3), intent(in) :: radial_vector
        real(wp), dimension(3)             :: vel_w, norm

        vel_w = 0._wp
        if (patch_ib(gp_patch_id)%moving_ibm /= 0) then
            norm = gp%levelset_norm/max(sqrt(sum(gp%levelset_norm**2)), sgm_eps)
            call s_cross_product(patch_ib(gp_patch_id)%angular_vel, radial_vector + abs(real(gp%levelset, kind=wp))*norm, vel_w)
            vel_w = vel_w + patch_ib(gp_patch_id)%vel
        end if

    end function f_wall_velocity

    !> sum_i kappa_i u_i^2 over the principal directions of the wall, for a tangential velocity u_t: the centripetal acceleration of
    !! fluid sliding along a curved wall. Known for a circle, a sphere (1/R in every direction) and the curved side of a cylinder
    !! (none along its axis); zero for any other shape, so those keep the zero-curvature condition.
    pure function f_curvature_u2(gp_patch_id, n, u_t) result(k_u2)

        $:GPU_ROUTINE(parallelism='[seq]')

        integer, intent(in)                :: gp_patch_id
        real(wp), dimension(3), intent(in) :: n, u_t
        real(wp)                           :: k_u2
        real(wp), dimension(3)             :: axis
        integer                            :: ax

        k_u2 = 0._wp
        select case (patch_ib(gp_patch_id)%geometry)
        case (2, 8)
            k_u2 = sum(u_t**2)/patch_ib(gp_patch_id)%radius
        case (10)
            ax = f_cylinder_axis([patch_ib(gp_patch_id)%length_x, patch_ib(gp_patch_id)%length_y, patch_ib(gp_patch_id)%length_z])
            axis = patch_ib(gp_patch_id)%rotation_matrix(:,ax)
            if (abs(dot_product(n, axis)) < 0.5_wp) k_u2 = (sum(u_t**2) - dot_product(u_t, axis)**2)/patch_ib(gp_patch_id)%radius
        end select

    end function f_curvature_u2

    !> Ghost pressure from the wall-normal momentum balance, dp/dn = -rho (a_w . n) + rho sum_i kappa_i u_i^2 (n out of the body):
    !! the first term is the acceleration of the wall point (the body's translation, force/mass for a two-way coupled body, alpha x
    !! r and omega x (omega x r)), the second the centripetal acceleration of fluid sliding along a curved slip wall, taken from the
    !! image point's velocity relative to the wall. Clamped both ways: the linearization holds only while the correction is order
    !! one, and an unbounded one drives the ghost state to vacuum.
    subroutine s_compute_ghost_point_pressure(gp, gp_patch_id, radial_vector, vel_IP, rho, pres_IP, pres_GP)

        $:GPU_ROUTINE(parallelism='[seq]')

        type(ghost_point), intent(in)      :: gp
        integer, intent(in)                :: gp_patch_id
        real(wp), dimension(3), intent(in) :: radial_vector, vel_IP
        real(wp), intent(in)               :: rho, pres_IP
        real(wp), intent(out)              :: pres_GP
        real(wp), dimension(3)             :: accel, r_w, w_x_r, w_x_w_x_r, norm, u_rel
        real(wp)                           :: a_n

        norm = gp%levelset_norm/max(sqrt(sum(gp%levelset_norm**2)), sgm_eps)
        accel = patch_ib(gp_patch_id)%accel
        if (patch_ib(gp_patch_id)%moving_ibm > 1) accel = patch_ib(gp_patch_id)%force/patch_ib(gp_patch_id)%mass
        r_w = radial_vector + abs(real(gp%levelset, kind=wp))*norm
        call s_cross_product(patch_ib(gp_patch_id)%angular_vel, r_w, w_x_r)
        call s_cross_product(patch_ib(gp_patch_id)%angular_vel, w_x_r, w_x_w_x_r)
        call s_cross_product(patch_ib(gp_patch_id)%angular_accel, r_w, w_x_r)
        a_n = dot_product(accel + w_x_w_x_r + w_x_r, norm)

        ! A no-slip wall holds its fluid at the wall velocity, so only a slip wall lets fluid sweep round the curvature
        if (gp%slip) then
            u_rel = vel_IP - f_wall_velocity(gp, gp_patch_id, radial_vector)
            u_rel = u_rel - dot_product(u_rel, norm)*norm
            a_n = a_n - f_curvature_u2(gp_patch_id, norm, u_rel)
        end if

        pres_GP = pres_IP/min(max(1._wp - (abs(gp%levelset) + gp%ip_dist)*rho/pres_IP*a_n, 5.e-1_wp), 2._wp)

    end subroutine s_compute_ghost_point_pressure

    !> Ghost velocity mirrored through the wall about u_w (body velocity at the wall point plus the normal transpiration v_n:
    !! blowing, Stefan flow): u_g = u_w + theta (u_w - u_IP), theta = |levelset|/(sample wall distance), puts u_w on the wall. A
    !! slip wall mirrors only the normal part and keeps the image point's tangential velocity.
    subroutine s_compute_ghost_point_velocity(gp, gp_patch_id, radial_vector, vel_IP, pres_IP, theta, v_n, vel_GP, q_prim_vf, wf, &
        & dh)

        $:GPU_ROUTINE(parallelism='[seq]')

        type(ghost_point), intent(in)       :: gp
        integer, intent(in)                 :: gp_patch_id
        real(wp), dimension(3), intent(in)  :: radial_vector, vel_IP
        real(wp), intent(in)                :: pres_IP, theta, v_n
        real(wp), dimension(3), intent(out) :: vel_GP
        !> Primitive field the quadratic fill probes, and the fill itself (tier 1 uses it). The ghost value is what enforces no-slip
        !! in the solve, so its order sets the order of the wall shear and hence the drag; a linear profile caps both at first
        !! order.
        type(scalar_field), dimension(sys_size), intent(in) :: q_prim_vf
        type(wall_fill), intent(in)                         :: wf
        !> h_IP - h_GP of the slip ghost fill (specific enthalpy at the image point minus at the ghost): the total enthalpy is held
        !! across the wall, which sets the tangential speed (curvature-corrected symmetry technique). Zero where that does not
        !! apply.
        real(wp), intent(in)   :: dh
        real(wp), dimension(3) :: vel_q
        integer                :: cmp
        real(wp), dimension(3) :: norm, vel_w, u_rel
        real(wp)               :: un_g, ut2, ut2_g
        real(wp)               :: v_blow_eff

        ! A zero normal (ghost at a circle's centre or on a flat face) leaves norm = 0 rather than 0/0
        norm = gp%levelset_norm/max(sqrt(sum(gp%levelset_norm**2)), sgm_eps)

        vel_w = f_wall_velocity(gp, gp_patch_id, radial_vector)

        v_blow_eff = patch_ib(gp_patch_id)%v_blow
        ! Pressure-coupled burn rate (Vieille's law r_dot ~ p^n); max(pres_IP, 0) guards the fractional power
        if (v_blow_eff > 0._wp .and. patch_ib(gp_patch_id)%burn_rate_pref > 0._wp) then
            v_blow_eff = v_blow_eff*(max(pres_IP, 0._wp)/patch_ib(gp_patch_id)%burn_rate_pref)**patch_ib(gp_patch_id)%burn_rate_exp
        end if
        vel_w = vel_w + (max(v_blow_eff, 0._wp) + v_n)*norm

        if (wf%tier == 1) then
            vel_q = 0._wp
            do cmp = 1, num_dims
                vel_q(cmp) = f_quad_fill(vel_w(cmp), f_tsc_apply(q_prim_vf(eqn_idx%mom%beg + cmp - 1), wf%ic1, wf%wd1), &
                      & f_tsc_apply(q_prim_vf(eqn_idx%mom%beg + cmp - 1), wf%ic2, wf%wd2), wf%lw, wf%llin)
            end do
            vel_GP = vel_q - vel_IP
        else
            vel_GP = (1._wp + theta)*(vel_w - vel_IP)
        end if
        if (gp%slip) then
            un_g = dot_product(vel_IP + vel_GP, norm) - dot_product(vel_w, norm)  ! ghost velocity relative to the wall, normal part
            u_rel = vel_IP - vel_w
            u_rel = u_rel - dot_product(u_rel, norm)*norm  ! image-point velocity relative to the wall, tangential part
            ut2 = sum(u_rel**2)
            ! Tangential speed from conservation of total enthalpy between the image point and the ghost; where there is no real
            ! solution, or no tangential flow to scale, the tangential velocity is copied
            ut2_g = ut2 + dot_product(vel_IP - vel_w, norm)**2 + 2._wp*dh - un_g**2
            ! The correction adjusts an existing tangential speed; it is bounded so a nearly stagnant image point is not given a
            ! large velocity
            if (dh /= 0._wp .and. ut2 > sgm_eps .and. ut2_g > 0._wp) u_rel = u_rel*min(max(sqrt(ut2_g/ut2), 0.5_wp), 2._wp)
            vel_GP = vel_w + un_g*norm + u_rel - vel_IP
        end if
        vel_GP = vel_IP + vel_GP

    end subroutine s_compute_ghost_point_velocity

    !> Update the conservative variables at the ghost points
    subroutine s_ibm_correct_state(q_cons_vf, q_prim_vf, pb_in, mv_in, q_T_sf)

        type(scalar_field), dimension(sys_size), intent(inout) :: q_cons_vf  !< Primitive Variables
        type(scalar_field), dimension(sys_size), intent(inout) :: q_prim_vf  !< Primitive Variables
        real(stp), dimension(idwbuff(1)%beg:,idwbuff(2)%beg:,idwbuff(3)%beg:,1:,1:), intent(inout) :: pb_in, mv_in
        !> Cell-average temperature of the same stage, probed by the quadratic wall treatment (ib_wall_order = 2, chemistry)
        type(scalar_field), intent(in) :: q_T_sf
        integer                        :: i, j, k, l, q, r              !< Iterator variables
        integer                        :: jj, kk, ll                    !< Neighbor iterators
        integer                        :: rad, rad_z, num_nbrs          !< Neighbor stencil radius and population
        integer                        :: patch_id, patch_id_temp       !< Patch ID of ghost point
        real(wp)                       :: rho, gamma, pi_inf, dyn_pres  !< Mixture variables
        real(wp)                       :: vel_sum_g, E_ghost            !< Ghost-point velocity magnitude and energy
        real(wp), dimension(2)         :: Re_K
        real(wp)                       :: G_K
        real(wp)                       :: qv_K
        real(wp)                       :: pres_IP, pres_GP
        real(wp), dimension(3)         :: vel_IP
        real(wp)                       :: c_IP

        #:if not MFC_CASE_OPTIMIZATION and USING_AMD
            real(wp), dimension(3)  :: Gs
            real(wp), dimension(3)  :: alpha_rho_IP, alpha_IP, alpha_rho_GP
            real(wp), dimension(3)  :: r_IP, v_IP, pb_IP, mv_IP
            real(wp), dimension(18) :: nmom_IP
            real(wp), dimension(12) :: presb_IP, massv_IP
        #:else
            real(wp), dimension(num_fluids) :: Gs
            real(wp), dimension(num_fluids) :: alpha_rho_IP, alpha_IP, alpha_rho_GP
            real(wp), dimension(nb)         :: r_IP, v_IP, pb_IP, mv_IP
            real(wp), dimension(nb*nmom)    :: nmom_IP
            real(wp), dimension(nb*nnode)   :: presb_IP, massv_IP
        #:endif
        real(wp), dimension(${NUM_SPECIES}$) :: Ys_IP, Ys_g, Ys_s, W_species
        real(wp) :: alpha_q, alpha_rho_q, e_q
        real(wp) :: T_IP, mw_IP, e_IP  !< Image-point temperature, mixture MW, and mass-specific internal energy (chemistry)
        real(wp) :: T_s, T_g, mw_s, mw_g, rho_s, mdot_s, v_stefan, d, theta
        real(wp) :: T_adv  !< Temperature of the advected ghost: T_g in the thermodynamic window (moving wall), else T_s
        real(wp) :: h_IP, dh_ccst  !< Enthalpy at the image point, and its excess over the slip ghost (CCST)
        logical :: use_ccst
        logical :: surface_converged
        integer :: n_not_converged, n_ill_posed  !< Per-call reacting-surface failure tallies (see the module-level counters)
        integer :: ng1, ng2, ng3, nf1, nf2, nf3  !< Per-call wall-gradient and wall-fill tier tallies
        type(wall_fill) :: wf  !< Quadratic wall-normal fill of the current ghost point
        type(wall_grad) :: wg  !< Wall-normal temperature gradient of the current ghost point
        integer :: tier_wg
        ! Primitive variables at the image point associated with a ghost point, interpolated from surrounding fluid cells.

        real(wp), dimension(3) :: physical_loc   !< Physical loc of GP
        real(wp), dimension(3) :: vel_g          !< Velocity of GP
        real(wp), dimension(3) :: radial_vector  !< vector from centroid to ghost point
        real(wp)               :: nbub
        real(wp)               :: buf, buf_prim
        type(ghost_point)      :: gp

        ! set the Moving IBM interior conservative variables. A solid cell that is a ghost keeps the value of the last correction
        ! when the interpolation or the wall probes read solid cells (ib_keep_ghost): resetting it to the body velocity, and without
        ! chemistry to a pressure of 1 Pa, would feed those values into the image-point state. Inverse-distance weights drop solid
        ! cells, so they never see it.
        $:GPU_PARALLEL_LOOP(private='[i, j, k, patch_id, rho, patch_id_temp]', collapse=3)
        do l = 0, p
            do k = 0, n
                do j = 0, m
                    patch_id = ib_markers%sf(j, k, l)
                    if (patch_id /= 0) then
                        call s_decode_patch_periodicity(patch_id, patch_id_temp)
                        call s_get_neighborhood_idx(patch_id_temp, patch_id)
                        if (patch_id > 0 .and. .not. (ib_keep_ghost .and. corrected_gps%sf(j, k, l) /= 0)) then
                            ! skip pressure correction with chemistry to prevent unphysical pressure
                            if (.not. chemistry) q_prim_vf(eqn_idx%E)%sf(j, k, l) = 1._wp
                            rho = 0._wp
                            do i = 1, num_fluids
                                rho = rho + q_prim_vf(eqn_idx%cont%beg + i - 1)%sf(j, k, l)
                            end do

                            ! Sets the momentum
                            do i = 1, num_dims
                                q_cons_vf(eqn_idx%mom%beg + i - 1)%sf(j, k, l) = patch_ib(patch_id)%vel(i)*rho
                                q_prim_vf(eqn_idx%mom%beg + i - 1)%sf(j, k, l) = patch_ib(patch_id)%vel(i)
                            end do
                        end if  ! patch_id > 0
                    end if
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        n_not_converged = 0
        n_ill_posed = 0
        ng1 = 0; ng2 = 0; ng3 = 0; nf1 = 0; nf2 = 0; nf3 = 0

        if (num_gps > 0) then
            ! Seed the staging buffer, so a component no ghost point writes survives the flush unchanged.
            $:GPU_PARALLEL_LOOP(private='[i, q, j, k, l, gp]', present='[ghost_points, gp_prim_new]')
            do i = 1, num_gps
                gp = ghost_points(i)
                j = gp%loc(1); k = gp%loc(2); l = gp%loc(3)
                do q = 1, sys_size
                    gp_prim_new(q, i) = q_prim_vf(q)%sf(j, k, l)
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()
            $:GPU_PARALLEL_LOOP(private='[i, physical_loc, dyn_pres, alpha_rho_IP, alpha_IP, alpha_rho_GP, pres_IP, pres_GP, &
                                & vel_IP, vel_g, r_IP, v_IP, pb_IP, mv_IP, nmom_IP, presb_IP, massv_IP, rho, gamma, pi_inf, Re_K, &
                                & G_K, Gs, gp, radial_vector, j, k, l, q, qv_K, c_IP, nbub, patch_id, Ys_IP, W_species, T_IP, &
                                & mw_IP, e_IP, Ys_g, Ys_s, T_s, T_g, T_adv, mw_s, mw_g, rho_s, mdot_s, v_stefan, d, theta, h_IP, &
                                & dh_ccst, use_ccst, surface_converged, vel_sum_g, E_ghost, alpha_q, alpha_rho_q, e_q, wf, wg, &
                                & tier_wg]', reduction='[[n_not_converged, n_ill_posed, ng1, ng2, ng3, nf1, nf2, nf3]]', &
                                & reductionOp='[+]', present='[ghost_points]')
            do i = 1, num_gps
                if (record_surface) gp_surf(:,i) = 0._wp
                ghost_points(i)%T_cond = T_cond_unset
                gp = ghost_points(i)
                if (.not. gp%interp_valid) cycle
                j = gp%loc(1)
                k = gp%loc(2)
                l = gp%loc(3)
                patch_id = ghost_points(i)%ib_patch_id

                ! Quadratic wall-normal fill: the probes and weights are built once per ghost point, before their first use below
                wf%tier = 3
                if (ib_wall_quad) then
                    call s_wall_fill(ib_markers, gp, wf)
                    if (wf%tier == 1) nf1 = nf1 + 1
                    if (wf%tier == 2) nf2 = nf2 + 1
                    if (wf%tier == 3) nf3 = nf3 + 1
                end if

                ! Calculate physical location of GP
                physical_loc = [x_cc(j), y_cc(k), 0._wp]
                if (num_dims == 3) physical_loc(3) = z_cc(l)

                ! Interpolate primitive variables at image point associated w/ GP
                if (bubbles_euler .and. .not. qbmm) then
                    call s_interpolate_image_point(q_prim_vf, gp, alpha_rho_IP, alpha_IP, pres_IP, vel_IP, c_IP, r_IP, v_IP, &
                                                   & pb_IP, mv_IP)
                else if (qbmm .and. polytropic) then
                    call s_interpolate_image_point(q_prim_vf, gp, alpha_rho_IP, alpha_IP, pres_IP, vel_IP, c_IP, r_IP, v_IP, &
                                                   & pb_IP, mv_IP, nmom_IP)
                else if (qbmm .and. .not. polytropic) then
                    call s_interpolate_image_point(q_prim_vf, gp, alpha_rho_IP, alpha_IP, pres_IP, vel_IP, c_IP, r_IP, v_IP, &
                                                   & pb_IP, mv_IP, nmom_IP, pb_in, mv_in, presb_IP, massv_IP)
                else if (chemistry) then
                    call s_interpolate_image_point(q_prim_vf, gp, alpha_rho_IP, alpha_IP, pres_IP, vel_IP, c_IP, Ys_IP=Ys_IP)
                else
                    call s_interpolate_image_point(q_prim_vf, gp, alpha_rho_IP, alpha_IP, pres_IP, vel_IP, c_IP)
                end if

                ! Injecting (burning) surface: replace the mirrored ghost composition with pure
                ! injected fuel at the local pressure and the ambient (image-point) temperature.
                ! Setting a consistent injected density here (rather than reusing the heavy ambient
                ! rho) keeps the light fuel at a physical temperature and feeds the surface flame.
                if (chemistry .and. patch_ib(patch_id)%inj_species > 0) then
                    call get_mixture_molecular_weight(Ys_IP, mw_IP)
                    T_IP = pres_IP*mw_IP/(alpha_rho_IP(1)*gas_constant)
                    Ys_IP = 0._wp
                    Ys_IP(patch_ib(patch_id)%inj_species) = 1._wp
                    call get_mixture_molecular_weight(Ys_IP, mw_IP)
                    alpha_rho_IP(1) = pres_IP*mw_IP/(T_IP*gas_constant)
                end if

                ! Mirror weight: the ghost sits |levelset| behind the wall, the image-point value is sampled d in front
                d = max(gp%ip_dist, abs(real(gp%levelset, kind=wp)))
                theta = 0._wp
                if (d > 0._wp) theta = abs(real(gp%levelset, kind=wp))/d

                ! Thermal and heterogeneous reacting-surface boundary conditions.
                v_stefan = 0._wp
                surface_converged = .false.

                if (chemistry .and. patch_ib(patch_id)%inj_species == 0) then
                    ! Intrinsic gas state at the image point:     rho_IP = (alpha*rho)_IP / alpha_IP
                    call get_mixture_molecular_weight(Ys_IP, mw_IP)
                    T_IP = pres_IP*mw_IP*alpha_IP(1)/(alpha_rho_IP(1)*gas_constant)

                    if (patch_ib(patch_id)%surface_reaction == 1) then
                        ! Heterogeneous reacting surface.

                        W_species(1:num_species) = molecular_weights(:)

                        ! d is the gas-side gradient length for every surface flux, so a ghost point sitting exactly on the
                        ! immersed surface (grid-aligned rectangle patch, coincident STL facet) leaves the whole balance
                        ! undefined. Left alone the 1/d produces NaN residuals, and a NaN then defeats the pivot test in
                        ! s_solve_surface_linear_system -- NaN <= epsilon is false -- so the solve reports success and writes
                        ! NaN into the ghost state. Such a point is not solvable, so it is recorded and skipped rather than
                        ! having its distance rescaled to something it is not.
                        if (gp%layer > gp_layers_full) then
                            ! A deep cell carries no surface chemistry: it keeps the image-point state (the branch below for an
                            ! unconverged solve), so the solve's cost does not grow with the shell depth.
                            surface_converged = .false.
                        else if (d > 0._wp) then
                            wg%quad = .false.; wg%a = 0._wp; wg%b = 0._wp
                            if (ib_wall_quad) then
                                call s_wall_normal_gradient(ib_markers, q_T_sf, gp, wg%a, wg%b, tier_wg)
                                wg%quad = tier_wg < 3
                                if (tier_wg == 1) ng1 = ng1 + 1
                                if (tier_wg == 2) ng2 = ng2 + 1
                                if (tier_wg == 3) ng3 = ng3 + 1
                            end if
                            call s_solve_surface(pres_IP, T_IP, patch_ib(patch_id)%Twall, d, wg, Ys_IP, W_species, merge(1, &
                                                 & patch_ib(patch_id)%thermal_bc, f_holds_twall(patch_ib(patch_id)%thermal_bc)), &
                                                 & Ys_s, T_s, mdot_s, surface_converged)
                            if (.not. surface_converged) n_not_converged = n_not_converged + 1
                        else
                            n_ill_posed = n_ill_posed + 1
                        end if

                        if (surface_converged) then
                            call get_mixture_molecular_weight(Ys_s, mw_s)

                            ! Intrinsic gas density at the reacting surface.
                            rho_s = pres_IP*mw_s/(gas_constant*T_s)
                            if (rho_s > 0._wp) v_stefan = mdot_s/rho_s
                        end if
                    end if

                    ! Inert surface, or a reacting surface whose solve failed: zero species flux, so the surface composition
                    ! is the image-point composition. Only the chemistry is dropped on failure; a prescribed Twall is kept,
                    ! and thermal_bc = 2, which has no wall temperature without the energy balance, takes a zero gradient.
                    if (.not. surface_converged) then
                        Ys_s(1:num_species) = Ys_IP(1:num_species)
                        T_s = T_IP
                        if (f_holds_twall(patch_ib(patch_id)%thermal_bc)) T_s = patch_ib(patch_id)%Twall
                    end if

                    if (record_surface) call s_record_gp_surface(i, gp, T_s, mdot_s, surface_converged)

                    call s_blend_ghost_state(theta, f_face_theta(q_prim_vf, ib_markers, gp, theta, T_IP, T_s), T_IP, T_s, Ys_IP, &
                                             & Ys_s, T_g, Ys_g, q_T_sf, wf)

                    ! A moving wall's lab-frame faces advect the ghost into the fluid, where the flux stencils sample it at the
                    ! ghost centre: there it carries the mirror T_g, as conduction does (T_cond), since wall gas at T_s cut the
                    ! heat flux of a translating body against the same body held fixed in the same relative flow. A fixed wall has
                    ! no normal velocity, the ghost enters only through the numerical dissipation, and wall gas at T_s is kept:
                    ! the mirror of a cold wall in hot gas leaves the window and slowed an unresolved heat-up transient.
                    T_adv = T_s
                    ! ib_cutvol makes the moving wall impermeable, and wall gas is then the consistent ghost (2D replica)
                    if (patch_ib(patch_id)%moving_ibm /= 0 .and. .not. ib_cutvol) T_adv = min(max(T_g, T_surface_min), &
                        & T_surface_max)
                    call get_mixture_molecular_weight(Ys_g, mw_g)
                    alpha_rho_IP(1) = alpha_IP(1)*pres_IP*mw_g/(gas_constant*T_adv)
                end if

                ! If in simulation, use acc mixture subroutines
                if (hypoelasticity) then
                    call s_convert_species_to_mixture_variables_kernel(rho, gamma, pi_inf, qv_K, alpha_IP, alpha_rho_IP, Re_K, &
                        & G_K, Gs)
                else
                    call s_convert_species_to_mixture_variables_kernel(rho, gamma, pi_inf, qv_K, alpha_IP, alpha_rho_IP, Re_K)
                end if

                if (surface_tension) gp_prim_new(eqn_idx%c, i) = c_IP

                ! get the vector that points from the centroid to the ghost
                radial_vector(1) = physical_loc(1) - (patch_ib(patch_id)%x_centroid + real(ghost_points(i)%x_periodicity, &
                              & wp)*(glb_bounds(1)%end - glb_bounds(1)%beg))
                radial_vector(2) = physical_loc(2) - (patch_ib(patch_id)%y_centroid + real(ghost_points(i)%y_periodicity, &
                              & wp)*(glb_bounds(2)%end - glb_bounds(2)%beg))
                radial_vector(3) = 0._wp
                if (num_dims == 3) radial_vector(3) = physical_loc(3) - (patch_ib(patch_id)%z_centroid &
                    & + real(ghost_points(i)%z_periodicity, wp)*(glb_bounds(3)%end - glb_bounds(3)%beg))

                ! set the pressure and density
                dh_ccst = 0._wp
                use_ccst = .false.
                if (patch_ib(patch_id)%moving_ibm == 0 .and. .not. gp%slip) then
                    pres_GP = pres_IP
                    alpha_rho_GP = alpha_rho_IP
                else
                    ! Curvature-corrected symmetry technique on a slip wall: the ghost density follows the isentrope and the total
                    ! enthalpy is conserved across the wall, which sets the tangential speed in s_compute_ghost_point_velocity
                    use_ccst = gp%slip .and. .not. chemistry .and. model_eqns == 2 .and. .not. bubbles_euler
                    h_IP = ((gamma + 1._wp)*pres_IP + pi_inf + qv_K)/rho
                    call s_compute_ghost_point_pressure(gp, patch_id, radial_vector, vel_IP, rho, pres_IP, pres_GP)

                    ! The adiabatic wall condition T_GP = T_IP the correction is derived from also
                    ! fixes the ghost density: p + B = (n - 1)*cv*rho*T at both points under the one
                    ! temperature leaves each partial density carrying the pressure ratio. The volume
                    ! fractions are untouched, so only rho and qv move with it.
                    $:GPU_LOOP(parallelism='[seq]')
                    do q = 1, num_fluids
                        if (use_ccst) then
                            alpha_rho_GP(q) = alpha_rho_IP(q)*((pres_GP + isentrope_B(q))/(pres_IP + isentrope_B(q))) &
                                         & **(1._wp/isentrope_n(q))
                        else
                            alpha_rho_GP(q) = alpha_rho_IP(q)*(pres_GP + isentrope_B(q))/(pres_IP + isentrope_B(q))
                        end if
                    end do
                    call s_compute_mixture_coefficients(alpha_rho_GP, alpha_IP, rho, gamma, pi_inf, qv_K)
                    if (use_ccst) dh_ccst = h_IP - ((gamma + 1._wp)*pres_GP + pi_inf + qv_K)/rho
                end if
                gp_prim_new(eqn_idx%E, i) = pres_GP

                ! Set q_prim_vf params at GP
                $:GPU_LOOP(parallelism='[seq]')
                do q = 1, num_fluids
                    gp_prim_new(q, i) = alpha_rho_GP(q)
                    gp_prim_new(eqn_idx%adv%beg + q - 1, i) = alpha_IP(q)
                end do

                ! Calculate velocity of ghost cell
                call s_compute_ghost_point_velocity(gp, patch_id, radial_vector, vel_IP, pres_IP, theta, v_stefan, vel_g, &
                                                    & q_prim_vf, wf, dh_ccst)

                ! Set momentum
                vel_sum_g = 0._wp
                $:GPU_LOOP(parallelism='[seq]')
                do q = eqn_idx%mom%beg, eqn_idx%mom%end
                    q_cons_vf(q)%sf(j, k, l) = rho*vel_g(q - eqn_idx%mom%beg + 1)
                    gp_prim_new(q, i) = vel_g(q - eqn_idx%mom%beg + 1)
                    vel_sum_g = vel_sum_g + vel_g(q - eqn_idx%mom%beg + 1)**2._wp
                end do
                dyn_pres = 5.e-1_wp*rho*vel_sum_g

                ! Set continuity and adv vars
                $:GPU_LOOP(parallelism='[seq]')
                do q = 1, num_fluids
                    q_cons_vf(q)%sf(j, k, l) = alpha_rho_GP(q)
                    q_cons_vf(eqn_idx%adv%beg + q - 1)%sf(j, k, l) = alpha_IP(q)
                end do

                ! Set color function
                if (surface_tension) q_cons_vf(eqn_idx%c)%sf(j, k, l) = c_IP

                ! Set Energy
                if (chemistry) then
                    ! Use the reconstructed thermal/species ghost state for an inert
                    ! thermal surface or a converged heterogeneous reacting surface.
                    if (patch_ib(patch_id)%inj_species == 0 .and. (patch_ib(patch_id)%surface_reaction == 0 &
                        & .or. surface_converged)) then

                        call get_mixture_energy_mass(T_adv, Ys_g, e_IP)
                        ghost_points(i)%T_cond = T_g
                        $:GPU_LOOP(parallelism='[seq]')
                        do q = 1, num_species
                            q_cons_vf(eqn_idx%species%beg + q - 1)%sf(j, k, l) = rho*Ys_g(q)
                        end do
                    else
                        ! Ordinary chemistry/injection, or fallback after a failed
                        ! heterogeneous surface solve: retain the image-point state.
                        mw_IP = 0._wp
                        call get_mixture_molecular_weight(Ys_IP, mw_IP)
                        T_IP = pres_IP*mw_IP/(rho*gas_constant)
                        call get_mixture_energy_mass(T_IP, Ys_IP, e_IP)
                        $:GPU_LOOP(parallelism='[seq]')
                        do q = 1, num_species
                            q_cons_vf(eqn_idx%species%beg + q - 1)%sf(j, k, l) = rho*Ys_IP(q)
                        end do
                    end if

                    q_cons_vf(eqn_idx%E)%sf(j, k, l) = rho*e_IP + dyn_pres
                else
                    call s_compute_energy(pres_GP, alpha_rho_GP, alpha_IP, vel_sum_g, E_ghost)
                    q_cons_vf(eqn_idx%E)%sf(j, k, l) = E_ghost
                end if
                ! Set bubble vars
                if (bubbles_euler .and. .not. qbmm) then
                    call s_comp_n_from_prim(alpha_IP(1), r_IP, nbub, weight)
                    $:GPU_LOOP(parallelism='[seq]')
                    do q = 1, nb
                        q_cons_vf(eqn_idx%bub%beg + (q - 1)*2)%sf(j, k, l) = nbub*r_IP(q)
                        q_cons_vf(eqn_idx%bub%beg + (q - 1)*2 + 1)%sf(j, k, l) = nbub*v_IP(q)
                        if (.not. polytropic) then
                            q_cons_vf(eqn_idx%bub%beg + (q - 1)*4)%sf(j, k, l) = nbub*r_IP(q)
                            q_cons_vf(eqn_idx%bub%beg + (q - 1)*4 + 1)%sf(j, k, l) = nbub*v_IP(q)
                            q_cons_vf(eqn_idx%bub%beg + (q - 1)*4 + 2)%sf(j, k, l) = nbub*pb_IP(q)
                            q_cons_vf(eqn_idx%bub%beg + (q - 1)*4 + 3)%sf(j, k, l) = nbub*mv_IP(q)
                        end if
                    end do
                end if

                if (qbmm) then
                    nbub = nmom_IP(1)
                    $:GPU_LOOP(parallelism='[seq]')
                    do q = 1, nb*nmom
                        q_cons_vf(eqn_idx%bub%beg + q - 1)%sf(j, k, l) = nbub*nmom_IP(q)
                    end do

                    $:GPU_LOOP(parallelism='[seq]')
                    do q = 1, nb
                        q_cons_vf(eqn_idx%bub%beg + (q - 1)*nmom)%sf(j, k, l) = nbub
                    end do

                    if (.not. polytropic) then
                        $:GPU_LOOP(parallelism='[seq]')
                        do q = 1, nb
                            $:GPU_LOOP(parallelism='[seq]')
                            do r = 1, nnode
                                pb_in(j, k, l, r, q) = presb_IP((q - 1)*nnode + r)
                                mv_in(j, k, l, r, q) = massv_IP((q - 1)*nnode + r)
                            end do
                        end do
                    end if
                end if

                if (model_eqns == model_eqns_6eq) then
                    $:GPU_LOOP(parallelism='[seq]')
                    do q = eqn_idx%int_en%beg, eqn_idx%int_en%end
                        alpha_q = alpha_IP(q - eqn_idx%int_en%beg + 1)
                        alpha_rho_q = alpha_rho_GP(q - eqn_idx%int_en%beg + 1)
                        call s_phase_internal_energy(pres_GP, alpha_q, alpha_rho_q, q - eqn_idx%int_en%beg + 1, e_q)
                        q_cons_vf(q)%sf(j, k, l) = e_q
                    end do
                end if
            end do
            $:END_GPU_PARALLEL_LOOP()

            ! A ghost point whose image point is buried in a neighboring IB has no fluid to mirror, so
            ! it takes the average of the ghost points corrected above, growing the stencil until it
            ! reaches one. Those neighbors already carry the wall condition, so the average does too.
            $:GPU_PARALLEL_LOOP(private='[i, j, k, l, q, r, jj, kk, ll, gp, rad, rad_z, num_nbrs, buf, buf_prim]', &
                                & present='[ghost_points]')
            do i = 1, num_gps
                gp = ghost_points(i)
                if (gp%interp_valid) cycle
                j = gp%loc(1)
                k = gp%loc(2)
                l = gp%loc(3)

                num_nbrs = 0
                rad = 0
                rad_z = 0
                do while (num_nbrs == 0 .and. rad < gp_layers)
                    rad = rad + 1
                    if (p /= 0) rad_z = rad
                    do jj = j - rad, j + rad
                        do kk = k - rad, k + rad
                            do ll = l - rad_z, l + rad_z
                                num_nbrs = num_nbrs + corrected_gps%sf(jj, kk, ll)
                            end do
                        end do
                    end do
                end do

                ! Fully enclosed: nothing to average from, so leave the interior placeholder in place
                if (num_nbrs == 0) cycle

                $:GPU_LOOP(parallelism='[seq]')
                do q = 1, sys_size
                    buf = 0._wp
                    buf_prim = 0._wp
                    do jj = j - rad, j + rad
                        do kk = k - rad, k + rad
                            do ll = l - rad_z, l + rad_z
                                if (corrected_gps%sf(jj, kk, ll) == 1) then
                                    buf = buf + q_cons_vf(q)%sf(jj, kk, ll)
                                    buf_prim = buf_prim + q_prim_vf(q)%sf(jj, kk, ll)
                                end if
                            end do
                        end do
                    end do
                    q_cons_vf(q)%sf(j, k, l) = buf/real(num_nbrs, wp)
                    gp_prim_new(q, i) = buf_prim/real(num_nbrs, wp)
                end do

                if (qbmm .and. .not. polytropic) then
                    $:GPU_LOOP(parallelism='[seq]')
                    do q = 1, nb
                        $:GPU_LOOP(parallelism='[seq]')
                        do r = 1, nnode
                            buf = 0._wp
                            buf_prim = 0._wp
                            do jj = j - rad, j + rad
                                do kk = k - rad, k + rad
                                    do ll = l - rad_z, l + rad_z
                                        if (corrected_gps%sf(jj, kk, ll) == 1) then
                                            buf = buf + pb_in(jj, kk, ll, r, q)
                                            buf_prim = buf_prim + mv_in(jj, kk, ll, r, q)
                                        end if
                                    end do
                                end do
                            end do
                            pb_in(j, k, l, r, q) = buf/real(num_nbrs, wp)
                            mv_in(j, k, l, r, q) = buf_prim/real(num_nbrs, wp)
                        end do
                    end do
                end if
            end do
            $:END_GPU_PARALLEL_LOOP()

            ! Second pass: publish the staged state now that every stencil has been read.
            $:GPU_PARALLEL_LOOP(private='[i, q, j, k, l, gp]', present='[ghost_points, gp_prim_new]')
            do i = 1, num_gps
                gp = ghost_points(i)
                j = gp%loc(1); k = gp%loc(2); l = gp%loc(3)
                do q = 1, sys_size
                    q_prim_vf(q)%sf(j, k, l) = gp_prim_new(q, i)
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()
        end if

        n_surface_not_converged = n_surface_not_converged + n_not_converged
        n_surface_ill_posed = n_surface_ill_posed + n_ill_posed
        n_grad_tier = n_grad_tier + [ng1, ng2, ng3]
        n_fill_tier = n_fill_tier + [nf1, nf2, nf3]

    end subroutine s_ibm_correct_state

    !> Give the wall conduction gradient the mirrored ghost temperature from s_ibm_correct_state in place of the temperature of the
    !! ghost state, which holds the same mirror inside the thermodynamic window. Only the face difference may read it: the mirror of
    !! a cold wall in hot gas lies far below 0 K, so properties at the ghost are evaluated at its own state
    !! (s_compute_chemistry_diffusion_flux, ib_marks) and s_ibm_restore_conduction_temperature puts the ghost's own temperature back
    !! once the fluxes are formed. Call after the conservative-to-primitive conversion, before the halo exchange.
    subroutine s_ibm_set_conduction_temperature(q_T_sf)

        type(scalar_field), intent(inout) :: q_T_sf
        integer                           :: i

        if (num_gps == 0) return
        $:GPU_PARALLEL_LOOP(private='[i]', present='[ghost_points]')
        do i = 1, num_gps
            if (ghost_points(i)%T_cond > T_cond_unset) then
                ghost_points(i)%T_save = q_T_sf%sf(ghost_points(i)%loc(1), ghost_points(i)%loc(2), ghost_points(i)%loc(3))
                q_T_sf%sf(ghost_points(i)%loc(1), ghost_points(i)%loc(2), ghost_points(i)%loc(3)) = ghost_points(i)%T_cond
            end if
        end do
        $:END_GPU_PARALLEL_LOOP()

    end subroutine s_ibm_set_conduction_temperature

    !> Undo s_ibm_set_conduction_temperature on this rank's ghosts, so the reaction source and the split chemistry substep see each
    !! ghost's own temperature
    subroutine s_ibm_restore_conduction_temperature(q_T_sf)

        type(scalar_field), intent(inout) :: q_T_sf
        integer                           :: i

        if (num_gps == 0) return
        $:GPU_PARALLEL_LOOP(private='[i]', present='[ghost_points]')
        do i = 1, num_gps
            if (ghost_points(i)%T_cond > T_cond_unset) then
                q_T_sf%sf(ghost_points(i)%loc(1), ghost_points(i)%loc(2), ghost_points(i)%loc(3)) = ghost_points(i)%T_save
            end if
        end do
        $:END_GPU_PARALLEL_LOOP()

    end subroutine s_ibm_restore_conduction_temperature

    !> One line at the end of a run if the reacting surface ever failed to solve. Silence means every ghost point on every step
    !! reached the surface-balance tolerance, i.e. the surface chemistry was actually applied everywhere it was asked for.
    impure subroutine s_report_ibm_surface

        integer               :: n_not_converged_glb, n_ill_posed_glb
        integer, dimension(3) :: gt, ft  !< Global tier tallies
        integer               :: q

        call s_mpi_reduce_int_sum(n_surface_not_converged, n_not_converged_glb)
        call s_mpi_reduce_int_sum(n_surface_ill_posed, n_ill_posed_glb)
        ! Every rank reduces before the root-only printing: a collective after the early return below would deadlock rank 0
        do q = 1, 3
            call s_mpi_reduce_int_sum(n_grad_tier(q), gt(q))
            call s_mpi_reduce_int_sum(n_fill_tier(q), ft(q))
        end do

        if (proc_rank /= 0) return

        if (n_not_converged_glb > 0) then
            print '(A,I0,A)', ' Immersed-boundary surface chemistry: the Newton solve did not converge at ', n_not_converged_glb, &
                & ' ghost-point updates, which fell back to a chemically inert wall (Twall kept for thermal_bc = 1, ' &
                & // 'zero temperature gradient for thermal_bc = 2).'
        end if

        if (n_ill_posed_glb > 0) then
            print '(A,I0,A)', ' Immersed-boundary surface chemistry: ', n_ill_posed_glb, &
                & ' ghost-point updates had a zero levelset distance, leaving the surface balance undefined; those points ' &
                & // 'fell back to a chemically inert wall (Twall kept for thermal_bc = 1, zero temperature gradient for ' &
                & // 'thermal_bc = 2). Check the immersed geometry against the grid.'
        end if

        if (ib_wall_quad) then
            if (sum(ft) > 0) print '(A,3(F6.1,A))', ' [WALLFILL] ghost fill tiers: quadratic ', 1.e2_wp*ft(1)/sum(ft), &
                & '%, linear ', 1.e2_wp*ft(2)/sum(ft), '%, caller treatment ', 1.e2_wp*ft(3)/sum(ft), '%'
            if (sum(gt) > 0) print '(A,3(F6.1,A))', ' [WALLGRAD] wall gradient tiers: quadratic ', 1.e2_wp*gt(1)/sum(gt), &
                & '%, linear ', 1.e2_wp*gt(2)/sum(gt), '%, one-sided ', 1.e2_wp*gt(3)/sum(gt), '%'
        end if

    end subroutine s_report_ibm_surface

    !> Store ghost point i's surface record in gp_surf. The weight is the surface area the point stands for. Ghost points within two
    !! cell sizes h = dV^(1/d) of the surface fill a band of volume ~2hA, so dV/(2h) each sums to the area A (a length in 2D)
    !! whatever the surface's orientation to the grid. The band lies inside the solid, where a layer at depth s has area A (1 -
    !! s/R)^(d-1) on a curved surface; for circles, spheres and cylinders the weight is scaled back by (R/(R - s))^(d-1), which
    !! removes that bias (sampled on random lattice offsets: < 0.1% mean, 0.3% spread for a sphere at R = 12h).
    subroutine s_record_gp_surface(i, gp, T_s, mdot_s, reacting)

        $:GPU_ROUTINE(parallelism='[seq]')

        integer, intent(in)           :: i
        type(ghost_point), intent(in) :: gp
        real(wp), intent(in)          :: T_s, mdot_s
        logical, intent(in)           :: reacting
        real(wp)                      :: d, dV, h, R
        real(wp), dimension(3)        :: length

        d = abs(real(gp%levelset, kind=wp))
        dV = dx(gp%loc(1))*dy(gp%loc(2))
        if (num_dims == 3) dV = dV*dz(gp%loc(3))
        h = dV**(1._wp/real(num_dims, wp))
        if (.not. (d > 0._wp .and. d <= 2._wp*h)) return

        gp_surf(1, i) = dV/(2._wp*h)
        R = patch_ib(gp%ib_patch_id)%radius
        select case (patch_ib(gp%ib_patch_id)%geometry)
        case (2, 8)
            gp_surf(1, i) = gp_surf(1, i)*(R/(R - d))**(num_dims - 1)
        case (10)  ! curved side only; the flat caps need no correction
            length = [patch_ib(gp%ib_patch_id)%length_x, patch_ib(gp%ib_patch_id)%length_y, patch_ib(gp%ib_patch_id)%length_z]
            if (abs(gp%levelset_norm(f_cylinder_axis(length))) < 0.5_wp) gp_surf(1, i) = gp_surf(1, i)*R/(R - d)
        end select

        gp_surf(2, i) = T_s
        gp_surf(3, i) = 0._wp
        if (reacting) gp_surf(3, i) = mdot_s

    end subroutine s_record_gp_surface

    !> Write this rank's surface records to D/ib_surface_<rank>_<save>.dat, one line per weighted ghost point.
    impure subroutine s_write_ib_surface(save_count)

        integer, intent(in)                  :: save_count
        character(LEN=path_len + 2*name_len) :: file_loc
        real(wp)                             :: x(3)
        integer                              :: i, g, unit
        integer, allocatable                 :: order(:)

        if (.not. allocated(gp_surf)) return

        if (num_gps > 0) then
            $:GPU_UPDATE(host='[gp_surf(:, 1:num_gps), ghost_points(1:num_gps)]')
        end if

        call s_sort_gps_by_location(order)

        write (file_loc, '(A,I0,A,I0,A)') trim(case_dir) // '/D/ib_surface_', proc_rank, '_', save_count, '.dat'
        open (newunit=unit, file=trim(file_loc), status='replace', action='write')
        do i = 1, num_gps
            g = order(i)
            if (.not. gp_surf(1, g) > 0._wp) cycle
            x = 0._wp
            x(1) = x_cc(ghost_points(g)%loc(1))
            x(2) = y_cc(ghost_points(g)%loc(2))
            if (num_dims == 3) x(3) = z_cc(ghost_points(g)%loc(3))
            write (unit, '(3ES16.8,1X,I0,3ES15.6,3ES16.8)') x, patch_ib(ghost_points(g)%ib_patch_id)%gbl_patch_id, &
                   & ghost_points(g)%levelset_norm, gp_surf(:,g)
        end do
        close (unit)

    end subroutine s_write_ib_surface

    !> Ghost-point indices ordered by grid location, so per-point output does not inherit the discovery order. s_find_ghost_points
    !! assigns slots from a GPU atomic, so that order is the thread schedule's and differs between CPU and GPU builds
    !! (MFlowCode/MFC#2039). Heapsort, since a GPU run's input is unordered and a particle array carries O(10^5) ghost points.
    pure subroutine s_sort_gps_by_location(order)

        integer, allocatable, intent(out) :: order(:)
        integer(8), allocatable           :: key(:)
        integer                           :: i, root, g
        integer(8)                        :: k

        allocate (order(max(num_gps, 1)), key(max(num_gps, 1)))
        do i = 1, num_gps
            order(i) = i
            ! Lexicographic (i, j, k) as one key; loc is a buffer-inclusive index, so offset it to stay non-negative.
            k = int(ghost_points(i)%loc(1) + buff_size, 8)
            k = k*int(n + 1 + 2*buff_size, 8) + int(ghost_points(i)%loc(2) + buff_size, 8)
            key(i) = k*int(p + 1 + 2*buff_size, 8) + int(ghost_points(i)%loc(3) + buff_size, 8)
        end do

        do root = num_gps/2, 1, -1
            call s_sift_gp(order, key, root, num_gps)
        end do
        do i = num_gps, 2, -1
            g = order(1); order(1) = order(i); order(i) = g
            call s_sift_gp(order, key, 1, i - 1)
        end do

    end subroutine s_sort_gps_by_location

    !> Restore the max-heap rooted at root over order(1:last), comparing by key.
    pure subroutine s_sift_gp(order, key, root, last)

        integer, intent(inout) :: order(:)
        integer(8), intent(in) :: key(:)
        integer, intent(in)    :: root, last
        integer                :: r, child, g

        r = root
        do
            child = 2*r
            if (child > last) exit
            if (child < last) then
                if (key(order(child + 1)) > key(order(child))) child = child + 1
            end if
            if (key(order(child)) <= key(order(r))) exit
            g = order(r); order(r) = order(child); order(child) = g
            r = child
        end do

    end subroutine s_sift_gp

    !> The surface temperature is held at Twall: prescribed (thermal_bc = 1) or evolved by the heat balance (thermal_bc = 3).
    pure logical function f_holds_twall(thermal_bc)

        $:GPU_ROUTINE(parallelism='[seq]')

        integer, intent(in) :: thermal_bc

        f_holds_twall = thermal_bc == 1 .or. thermal_bc == 3

    end function f_holds_twall

    !> Solid volume of a circle (per unit depth), sphere or cylinder IB.
    pure real(wp) function f_ib_volume(ib_patch)
        $:GPU_ROUTINE(parallelism='[seq]')

        type(ib_patch_parameters), intent(in) :: ib_patch

        select case (ib_patch%geometry)
        case (2)
            f_ib_volume = pi*ib_patch%radius**2
        case (8)
            f_ib_volume = 4._wp/3._wp*pi*ib_patch%radius**3
        case default
            f_ib_volume = pi*ib_patch%radius**2*max(ib_patch%length_x, ib_patch%length_y, ib_patch%length_z)
        end select

    end function f_ib_volume

    !> Advance the temperature of each thermal_bc = 3 IB over one step, treating it as one lumped body (small Biot number). With
    !! E_solid = m c_s (T - T_ref), the step's face energy in from s_accumulate_ib_face_fluxes and the gasification rate from the
    !! surface solve, m c_s dT/dt = Q_in + c_s (T - T_ref) Mdot_out + heat_power - emissivity sigma A (T^4 - T_rad^4). Mdot_out is
    !! the solid's mass loss: the face mass sum also carries the ghost cells' non-conservation (a net inflow on a reacting
    !! cylinder). T_ref = 298.15 K is the reference of the formation enthalpies, so the gas energy the solid's lost carbon carries
    !! is consistent. The step is explicit: the body's thermal time m c_s/(h A) is many orders of magnitude longer than a flow step.
    !! For D/ib_heat.dat every body also keeps its gasification rate, face mass out and fluid heat in, Q_in (+ the c_s term for
    !! thermal_bc = 3), in patch_ib%mdot, %mdot_face and %qdot.
    impure subroutine s_update_ib_temperatures(dt_step, q_cons_vf)

        real(wp), intent(in) :: dt_step
        type(scalar_field), dimension(sys_size), intent(in) :: q_cons_vf
        real(wp), parameter :: sigma_sb = 5.670374419e-8_wp  !< Stefan-Boltzmann constant [W/m^2/K^4]
        real(wp), parameter :: T_ref = 298.15_wp
        real(wp) :: q_in
        integer :: i, g

        ! Area and gasification rate from the surface records of the last stage

        $:GPU_PARALLEL_LOOP(private='[g]')
        do g = 1, num_gbl_ibs
            ib_heat(1, g) = 0._wp
            ib_heat(4, g) = 0._wp
        end do
        $:END_GPU_PARALLEL_LOOP()
        if (num_gps > 0) then
            $:GPU_PARALLEL_LOOP(private='[i, g]')
            do i = 1, num_gps
                if (gp_surf(1, i) > 0._wp) then
                    g = patch_ib(ghost_points(i)%ib_patch_id)%gbl_patch_id
                    $:GPU_ATOMIC(atomic='update')
                    ib_heat(1, g) = ib_heat(1, g) + gp_surf(1, i)
                    $:GPU_ATOMIC(atomic='update')
                    ib_heat(4, g) = ib_heat(4, g) + gp_surf(1, i)*gp_surf(3, i)
                end if
            end do
            $:END_GPU_PARALLEL_LOOP()
        end if
        $:GPU_UPDATE(host='[ib_heat]')
        call s_mpi_allreduce_vectors_sum(ib_heat, ib_heat, 4, num_gbl_ibs)
        $:GPU_UPDATE(device='[ib_heat]')

        ! On the device, where patch_ib is current (moving IBs update it there)
        $:GPU_PARALLEL_LOOP(private='[i, g, q_in]', copyin='[dt_step]')
        do i = 1, num_ibs
            g = patch_ib(i)%gbl_patch_id
            q_in = ib_heat(2, g)
            if (patch_ib(i)%thermal_bc == 3) q_in = q_in + patch_ib(i)%cp_solid*(patch_ib(i)%Twall - T_ref)*ib_heat(4, g)
            patch_ib(i)%mdot = ib_heat(4, g)
            patch_ib(i)%mdot_face = ib_heat(3, g)
            patch_ib(i)%qdot = q_in
            if (patch_ib(i)%thermal_bc == 3) then
                q_in = q_in + patch_ib(i)%heat_power - patch_ib(i)%emissivity*sigma_sb*ib_heat(1, &
                                       & g)*(patch_ib(i)%Twall**4 - patch_ib(i)%T_rad**4)
                patch_ib(i)%Twall = min(max(patch_ib(i)%Twall &
                         & + dt_step*q_in/(patch_ib(i)%rho_solid*patch_ib(i)%cp_solid*f_ib_volume(patch_ib(i))), T_surface_min), &
                         & T_surface_max)
            end if
        end do
        $:END_GPU_PARALLEL_LOOP()

        ! ib_mass_relax: the gas should gain exactly the gasified mass; bank the rest and release 1/ib_mass_relax of the bank
        if (ib_mass_relax > 0) then
            $:GPU_PARALLEL_LOOP(private='[g]', copyin='[dt_step]')
            do g = 1, num_gbl_ibs
                ib_mass_res(1, g) = ib_mass_res(1, g) + (ib_heat(3, g) - ib_heat(4, g))*dt_step
                ib_mass_res(2, g) = ib_mass_res(1, g)/real(ib_mass_relax, wp)
                ib_mass_res(1, g) = ib_mass_res(1, g) - ib_mass_res(2, g)
            end do
            $:END_GPU_PARALLEL_LOOP()
        end if

        ! The next step accumulates afresh
        $:GPU_PARALLEL_LOOP(private='[g]')
        do g = 1, num_gbl_ibs
            ib_heat(2, g) = 0._wp
            ib_heat(3, g) = 0._wp
        end do
        $:END_GPU_PARALLEL_LOOP()

    end subroutine s_update_ib_temperatures

    !> ib_mass_relax: take this step's release of each body's bank (s_update_ib_temperatures) out of the gas within ib_shell_width
    !! cells of the body, scaling the partial densities, momentum, energy and species of those cells by one factor (velocity,
    !! temperature and composition kept). The ghost-cell treatment does not conserve mass at a moving or gasifying wall (face fluxes
    !! into overwritten ghost cells against cells changing side); this returns the gas to exactly the gasified mass, lagged by about
    !! ib_mass_relax steps.
    subroutine s_conserve_ib_mass(q_cons_vf)

        type(scalar_field), dimension(sys_size), intent(inout) :: q_cons_vf
        real(wp)                                               :: rho, vol, f
        integer                                                :: i, j, k, l, jj, kk, ll, wz, mk, ib_tmp, ib_id, g, xp, yp, zp

        wz = 0
        if (num_dims == 3) wz = ib_shell_width
        $:GPU_PARALLEL_LOOP(private='[g]')
        do g = 1, num_gbl_ibs
            ib_shell_mass(1, g) = 0._wp
        end do
        $:END_GPU_PARALLEL_LOOP()

        ! Shell membership from the markers, which carry the halo, so a cell next to another rank's body is found too
        $:GPU_PARALLEL_LOOP(private='[i, j, k, l, jj, kk, ll, mk, ib_tmp, ib_id, g, xp, yp, zp, rho, vol]', collapse=3, &
                            & copyin='[wz]')
        do l = 0, p
            do k = 0, n
                do j = 0, m
                    g = 0
                    if (ib_markers%sf(j, k, l) == 0) then
                        mk = 0
                        do ll = -wz, wz
                            do kk = -ib_shell_width, ib_shell_width
                                do jj = -ib_shell_width, ib_shell_width
                                    if (mk == 0) mk = ib_markers%sf(j + jj, k + kk, l + ll)
                                end do
                            end do
                        end do
                        if (mk /= 0) then
                            call s_decode_patch_periodicity(mk, ib_tmp, xp, yp, zp)
                            call s_get_neighborhood_idx(ib_tmp, ib_id)
                            if (ib_id > 0) g = patch_ib(ib_id)%gbl_patch_id
                        end if
                    end if
                    ib_shell%sf(j, k, l) = g
                    if (g > 0) then
                        rho = 0._wp
                        do i = eqn_idx%cont%beg, eqn_idx%cont%end
                            rho = rho + q_cons_vf(i)%sf(j, k, l)
                        end do
                        vol = dx(j)*dy(k)
                        if (num_dims == 3) vol = vol*dz(l)
                        $:GPU_ATOMIC(atomic='update')
                        ib_shell_mass(1, g) = ib_shell_mass(1, g) + rho*vol
                    end if
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()
        $:GPU_UPDATE(host='[ib_shell_mass]')
        call s_mpi_allreduce_vectors_sum(ib_shell_mass, ib_shell_mass, 1, num_gbl_ibs)
        $:GPU_UPDATE(device='[ib_shell_mass]')

        $:GPU_PARALLEL_LOOP(private='[i, j, k, l, g, f]', collapse=3)
        do l = 0, p
            do k = 0, n
                do j = 0, m
                    g = ib_shell%sf(j, k, l)
                    if (g > 0) then
                        if (ib_shell_mass(1, g) > 0._wp) then
                            f = 1._wp - ib_mass_res(2, g)/ib_shell_mass(1, g)
                            do i = eqn_idx%cont%beg, eqn_idx%E
                                q_cons_vf(i)%sf(j, k, l) = f*q_cons_vf(i)%sf(j, k, l)
                            end do
                            if (chemistry) then
                                do i = eqn_idx%species%beg, eqn_idx%species%end
                                    q_cons_vf(i)%sf(j, k, l) = f*q_cons_vf(i)%sf(j, k, l)
                                end do
                            end if
                        end if
                    end if
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

    end subroutine s_conserve_ib_mass

    !> ib_cutvol: each fluid cell's fluid fraction, alpha = 1/2 + phi/(|n_x| dx + |n_y| dy + |n_z| dz) (the planar-wall area
    !! fraction to first order) for the nearest of the bodies in its 3^d neighbourhood, at least 1/2 (its centre is outside), and
    !! the translation velocity of each body cell
    subroutine s_cutvol_fields()

        type(ghost_point) :: gp
        integer           :: i, j, k, ii, jj, kk, d, kz, zb, mk, pid, idx, xp, yp, zp
        real(wp)          :: h, a

        zb = 0; if (p > 0) zb = buff_size
        kz = 0; if (p > 0) kz = 1
        $:GPU_PARALLEL_LOOP(private='[i, j, k, ii, jj, kk, d, mk, pid, idx, xp, yp, zp, h, a, gp]', collapse=3, copyin='[zb, kz]')
        do k = -zb, p + zb
            do j = -buff_size, n + buff_size
                do i = -buff_size, m + buff_size
                    ib_alpha%sf(i, j, k) = 1._wp
                    do d = 1, 3
                        ib_wall_vel(d, i, j, k) = 0._wp
                    end do
                    if (ib_markers%sf(i, j, k) /= 0) then
                        call s_decode_patch_periodicity(ib_markers%sf(i, j, k), pid)
                        call s_get_neighborhood_idx(pid, idx)
                        if (idx > 0) then
                            if (patch_ib(idx)%moving_ibm /= 0) then
                                do d = 1, 3
                                    ib_wall_vel(d, i, j, k) = patch_ib(idx)%vel(d)
                                end do
                            end if
                        end if
                    else
                        do kk = -kz, kz
                            do jj = -1, 1
                                do ii = -1, 1
                                    if (i + ii < -buff_size .or. i + ii > m + buff_size .or. j + jj < -buff_size .or. j + jj > n &
                                        & + buff_size .or. k + kk < -zb .or. k + kk > p + zb) cycle
                                    mk = ib_markers%sf(i + ii, j + jj, k + kk)
                                    if (mk == 0) cycle
                                    call s_decode_patch_periodicity(mk, pid, xp, yp, zp)
                                    call s_get_neighborhood_idx(pid, idx)
                                    if (idx <= 0) cycle
                                    gp%loc(1) = i; gp%loc(2) = j; gp%loc(3) = k
                                    gp%ib_patch_id = idx
                                    gp%x_periodicity = xp; gp%y_periodicity = yp; gp%z_periodicity = zp
                                    call s_levelset_point(gp)
                                    h = abs(gp%levelset_norm(1))*dx(i) + abs(gp%levelset_norm(2))*dy(j)
                                    if (p > 0) h = h + abs(gp%levelset_norm(3))*dz(k)
                                    a = min(max(0.5_wp + gp%levelset/max(h, sgm_eps), 0.5_wp), 1._wp)
                                    ib_alpha%sf(i, j, k) = min(ib_alpha%sf(i, j, k), a)
                                end do
                            end do
                        end do
                    end if
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

    end subroutine s_cutvol_fields

    !> ib_cutvol: move every prescribed-translation body over the whole step, rebuild the markers and ghost points, and remap the
    !! gas conservatively (s_cutvol_remap). The geometry is then frozen for the step's stages. q_cons_vf's halo must be current.
    impure subroutine s_cutvol_advance(q_cons_vf, dt_step)

        type(scalar_field), dimension(sys_size), intent(inout) :: q_cons_vf
        real(wp), intent(in)                                   :: dt_step
        integer                                                :: i, j, k, zb

        zb = 0; if (p > 0) zb = buff_size
        $:GPU_PARALLEL_LOOP(private='[i, j, k]', collapse=3, copyin='[zb]')
        do k = -zb, p + zb
            do j = -buff_size, n + buff_size
                do i = -buff_size, m + buff_size
                    ib_markers_cv%sf(i, j, k) = ib_markers%sf(i, j, k)
                    ib_alpha_prev%sf(i, j, k) = ib_alpha%sf(i, j, k)
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        $:GPU_PARALLEL_LOOP(private='[i]', copyin='[dt_step]')
        do i = 1, num_ibs
            if (patch_ib(i)%moving_ibm == 1) then
                patch_ib(i)%x_centroid = patch_ib(i)%x_centroid + dt_step*patch_ib(i)%vel(1)
                patch_ib(i)%y_centroid = patch_ib(i)%y_centroid + dt_step*patch_ib(i)%vel(2)
                patch_ib(i)%z_centroid = patch_ib(i)%z_centroid + dt_step*patch_ib(i)%vel(3)
            end if
        end do
        $:END_GPU_PARALLEL_LOOP()

        call s_update_mib(num_ibs)
        call s_cutvol_fields()
        call s_cutvol_remap(q_cons_vf)

    end subroutine s_cutvol_advance

    !> ib_cutvol: conservative remap after a move. A cell fluid before and after (kept) keeps its gas in its new fluid volume, q =
    !! alpha_prev q / alpha (the advancing wall compresses, the receding wall expands). An uncovered cell takes the fraction f =
    !! alpha / sum(alpha_prev) (at most 1/4) of every kept neighbour, or of every kept cell within two cells when its neighbours
    !! cannot spare that (ib_cv_f < 0 marks it); the donors keep their state at (1 - sum f) of their content, so no cell is drained
    !! or its state shifted. A covered cell gives its content alpha_prev q to its kept neighbours in equal shares. Cells in the halo
    !! are processed too, from halo states every rank shares.
    subroutine s_cutvol_remap(q_cons_vf)

        type(scalar_field), dimension(sys_size), intent(inout) :: q_cons_vf
        integer                                                :: i, j, k, ii, jj, kk, q, kz, cnt, r
        real(wp)                                               :: w, w1, f, ws

        kz = 0; if (p > 0) kz = 1

        ! Volume fractions are intensive: kept and covered cells leave them, an uncovered cell takes its donors' mean

        ! uncovered cells: f of each kept donor within reach r (read before anything changes)
        $:GPU_PARALLEL_LOOP(private='[i, j, k, ii, jj, kk, q, w, w1, f, r, ws]', collapse=3, copyin='[kz]')
        do k = -2*kz, p + 2*kz
            do j = -2, n + 2
                do i = -2, m + 2
                    ib_cv_f%sf(i, j, k) = 0._wp
                    if (ib_markers_cv%sf(i, j, k) /= 0 .and. ib_markers%sf(i, j, k) == 0) then
                        w = 0._wp; w1 = 0._wp
                        do kk = -2*kz, 2*kz
                            do jj = -2, 2
                                do ii = -2, 2
                                    if (f_cutvol_kept(ib_markers, ib_markers_cv, i + ii, j + jj, k + kk)) then
                                        w = w + ib_alpha_prev%sf(i + ii, j + jj, k + kk)
                                        if (max(abs(ii), abs(jj), abs(kk)) <= 1) w1 = w1 + ib_alpha_prev%sf(i + ii, j + jj, k + kk)
                                    end if
                                end do
                            end do
                        end do
                        r = 2
                        if (ib_alpha%sf(i, j, k) <= 0.25_wp*w1) then
                            r = 1; w = w1
                        end if
                        if (w > 0._wp) then
                            ws = w
                            f = min(ib_alpha%sf(i, j, k)/w, 0.25_wp)
                            ib_cv_f%sf(i, j, k) = merge(f, -f, r == 1)
                            do q = 1, sys_size
                                w = 0._wp
                                do kk = -r*kz, r*kz
                                    do jj = -r, r
                                        do ii = -r, r
                                            if (f_cutvol_kept(ib_markers, ib_markers_cv, i + ii, j + jj, &
                                                & k + kk)) w = w + ib_alpha_prev%sf(i + ii, j + jj, &
                                                & k + kk)*q_cons_vf(q)%sf(i + ii, j + jj, k + kk)
                                        end do
                                    end do
                                end do
                                if (q >= eqn_idx%adv%beg .and. q <= eqn_idx%adv%end) then
                                    q_cons_vf(q)%sf(i, j, k) = w/ws
                                else
                                    q_cons_vf(q)%sf(i, j, k) = f*w/ib_alpha%sf(i, j, k)
                                end if
                            end do
                        end if
                    end if
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        ! kept cells: what the uncovered cells took, then the new fluid volume
        $:GPU_PARALLEL_LOOP(private='[i, j, k, ii, jj, kk, q, w, f]', collapse=3, copyin='[kz]')
        do k = 0, p
            do j = 0, n
                do i = 0, m
                    if (f_cutvol_kept(ib_markers, ib_markers_cv, i, j, k)) then
                        w = 0._wp
                        do kk = -2*kz, 2*kz
                            do jj = -2, 2
                                do ii = -2, 2
                                    f = ib_cv_f%sf(i + ii, j + jj, k + kk)  ! > 0: reach 1, < 0: reach 2
                                    if (ib_markers_cv%sf(i + ii, j + jj, k + kk) /= 0 .and. ib_markers%sf(i + ii, j + jj, &
                                        & k + kk) == 0 .and. (f < 0._wp .or. max(abs(ii), abs(jj), abs(kk)) <= 1)) w = w + abs(f)
                                end do
                            end do
                        end do
                        w = (1._wp - w)*ib_alpha_prev%sf(i, j, k)/ib_alpha%sf(i, j, k)
                        if (w /= 1._wp) then
                            do q = 1, sys_size
                                if (q >= eqn_idx%adv%beg .and. q <= eqn_idx%adv%end) cycle
                                q_cons_vf(q)%sf(i, j, k) = w*q_cons_vf(q)%sf(i, j, k)
                            end do
                        end if
                    end if
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        ! covered cells: their content alpha_prev q into the kept neighbours, per unit fluid volume
        $:GPU_PARALLEL_LOOP(private='[i, j, k, ii, jj, kk, q, cnt, w]', collapse=3, copyin='[kz]')
        do k = -kz, p + kz
            do j = -1, n + 1
                do i = -1, m + 1
                    if (ib_markers_cv%sf(i, j, k) == 0 .and. ib_markers%sf(i, j, k) /= 0) then
                        cnt = f_cutvol_count(ib_markers, ib_markers_cv, i, j, k, kz)
                        if (cnt > 0) then
                            w = ib_alpha_prev%sf(i, j, k)/real(cnt, wp)
                            do kk = -kz, kz
                                do jj = -1, 1
                                    do ii = -1, 1
                                        if (i + ii < 0 .or. i + ii > m .or. j + jj < 0 .or. j + jj > n .or. k + kk < 0 .or. k &
                                            & + kk > p) cycle
                                        if (.not. f_cutvol_kept(ib_markers, ib_markers_cv, i + ii, j + jj, k + kk)) cycle
                                        do q = 1, sys_size
                                            if (q >= eqn_idx%adv%beg .and. q <= eqn_idx%adv%end) cycle
                                            $:GPU_ATOMIC(atomic='update')
                                            q_cons_vf(q)%sf(i + ii, j + jj, k + kk) = q_cons_vf(q)%sf(i + ii, j + jj, &
                                                      & k + kk) + w*q_cons_vf(q)%sf(i, j, k)/ib_alpha%sf(i + ii, j + jj, k + kk)
                                        end do
                                    end do
                                end do
                            end do
                        end if
                    end if
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

    end subroutine s_cutvol_remap

    !> ib_cutvol: a cell that is fluid now and was before the step's move
    function f_cutvol_kept(mk, mk_cv, i, j, k) result(kept)

        $:GPU_ROUTINE(function_name='f_cutvol_kept', parallelism='[seq]', cray_inline=True)
        type(integer_field), intent(in) :: mk, mk_cv  !< ib_markers and ib_markers_cv, passed (see f_ghost_layer)
        integer, intent(in)             :: i, j, k
        logical                         :: kept

        kept = mk%sf(i, j, k) == 0 .and. mk_cv%sf(i, j, k) == 0

    end function f_cutvol_kept

    !> ib_cutvol: number of kept cells in the 3^d neighbourhood of (i, j, k)
    function f_cutvol_count(mk, mk_cv, i, j, k, kz) result(cnt)

        $:GPU_ROUTINE(function_name='f_cutvol_count', parallelism='[seq]', cray_inline=True)
        type(integer_field), intent(in) :: mk, mk_cv
        integer, intent(in)             :: i, j, k, kz
        integer                         :: cnt, ii, jj, kk

        cnt = 0
        do kk = -kz, kz
            do jj = -1, 1
                do ii = -1, 1
                    if ((ii /= 0 .or. jj /= 0 .or. kk /= 0) .and. f_cutvol_kept(mk, mk_cv, i + ii, j + jj, k + kk)) cnt = cnt + 1
                end do
            end do
        end do

    end function f_cutvol_count

    !> Moving bodies: the body-frame energy E' = E - rho u.V_b + rho |V_b|^2/2 and the mass of each cell a body covered (+) or
    !! uncovered (-) at this stage's placement, as rates into ib_heat(2) and out of ib_heat(3). The grid faces do not move with the
    !! wall, so the face fluxes alone miss the gas swept between face and wall. Called at every placement, after the stage update
    !! and before the ghost correction overwrites the covered cells: a cell's content enters the step's result with weight a_w (the
    !! product of the later stages' rk_coef(:,1)/rk_coef(:,4)) and the step-start state with 1 - a_w, so it is booked as a_w q + (1
    !! - a_w) q_start, which makes the booking exact under the RK blend (checked to 1e-14 per step in a 2D replica). A cell passing
    !! directly from one body to another is not counted.
    subroutine s_add_swept_cells(q_cons_vf, q_start_vf, a_w, dt_step)

        type(scalar_field), dimension(sys_size), intent(in) :: q_cons_vf, q_start_vf
        real(wp), intent(in)                                :: a_w, dt_step
        real(wp), dimension(3)                              :: r, vb, box_len
        real(wp)                                            :: rho, e_b, vol, b_w
        integer                                             :: i, j, k, l, mk, sgn, ib_tmp, ib_id, g, xp, yp, zp

        box_len = [glb_bounds(1)%end - glb_bounds(1)%beg, glb_bounds(2)%end - glb_bounds(2)%beg, 0._wp]
        if (num_dims == 3) box_len(3) = glb_bounds(3)%end - glb_bounds(3)%beg
        b_w = 1._wp - a_w
        $:GPU_PARALLEL_LOOP(private='[i, j, k, l, mk, sgn, ib_tmp, ib_id, g, xp, yp, zp, r, vb, rho, e_b, vol]', collapse=3, &
                            & copyin='[dt_step, box_len, a_w, b_w]')
        do l = 0, p
            do k = 0, n
                do j = 0, m
                    if ((ib_markers_prev%sf(j, k, l) == 0) .neqv. (ib_markers%sf(j, k, l) == 0)) then
                        sgn = merge(1, -1, ib_markers%sf(j, k, l) /= 0)
                        mk = merge(ib_markers%sf(j, k, l), ib_markers_prev%sf(j, k, l), sgn == 1)
                        call s_decode_patch_periodicity(mk, ib_tmp, xp, yp, zp)
                        call s_get_neighborhood_idx(ib_tmp, ib_id)
                        if (ib_id > 0) then
                            ! Lever arm to the centroid, minimum image (a body may have just been wrapped)
                            r(1) = x_cc(j) - patch_ib(ib_id)%x_centroid
                            r(2) = y_cc(k) - patch_ib(ib_id)%y_centroid
                            r(3) = 0._wp
                            if (num_dims == 3) r(3) = z_cc(l) - patch_ib(ib_id)%z_centroid
                            do i = 1, 3
                                if (box_len(i) > 0._wp) r(i) = r(i) - box_len(i)*anint(r(i)/box_len(i))
                            end do
                            vb(1) = patch_ib(ib_id)%vel(1) + patch_ib(ib_id)%angular_vel(2)*r(3) - patch_ib(ib_id)%angular_vel(3) &
                               & *r(2)
                            vb(2) = patch_ib(ib_id)%vel(2) + patch_ib(ib_id)%angular_vel(3)*r(1) - patch_ib(ib_id)%angular_vel(1) &
                               & *r(3)
                            vb(3) = patch_ib(ib_id)%vel(3) + patch_ib(ib_id)%angular_vel(1)*r(2) - patch_ib(ib_id)%angular_vel(2) &
                               & *r(1)
                            rho = 0._wp
                            do i = eqn_idx%cont%beg, eqn_idx%cont%end
                                rho = rho + a_w*q_cons_vf(i)%sf(j, k, l) + b_w*q_start_vf(i)%sf(j, k, l)
                            end do
                            e_b = a_w*q_cons_vf(eqn_idx%E)%sf(j, k, l) + b_w*q_start_vf(eqn_idx%E)%sf(j, k, &
                                                & l) + 0.5_wp*rho*(vb(1)**2 + vb(2)**2 + vb(3)**2)
                            do i = 1, num_dims
                                e_b = e_b - (a_w*q_cons_vf(eqn_idx%mom%beg + i - 1)%sf(j, k, &
                                             & l) + b_w*q_start_vf(eqn_idx%mom%beg + i - 1)%sf(j, k, l))*vb(i)
                            end do
                            vol = dx(j)*dy(k)
                            if (num_dims == 3) vol = vol*dz(l)
                            g = patch_ib(ib_id)%gbl_patch_id
                            $:GPU_ATOMIC(atomic='update')
                            ib_heat(2, g) = ib_heat(2, g) + real(sgn, wp)*e_b*vol/dt_step
                            $:GPU_ATOMIC(atomic='update')
                            ib_heat(3, g) = ib_heat(3, g) - real(sgn, wp)*rho*vol/dt_step
                        end if
                    end if
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        $:GPU_PARALLEL_LOOP(private='[j, k, l]', collapse=3)
        do l = 0, p
            do k = 0, n
                do j = 0, m
                    ib_markers_prev%sf(j, k, l) = ib_markers%sf(j, k, l)
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

    end subroutine s_add_swept_cells

    !> Compute the image points for each ghost point
    impure subroutine s_compute_image_points()

        real(wp)                        :: dist
        real(wp)                        :: min_cell_width
        real(wp)                        :: temp_loc
        real(wp), pointer, dimension(:) :: s_cc => null()
        integer                         :: bound
        type(ghost_point)               :: gp
        integer                         :: q, dim      !< Iterator variables
        integer                         :: i, j, k, l  !< Location indexes
        integer                         :: patch_id    !< IB Patch ID
        integer                         :: dir
        integer                         :: index
        logical                         :: bounds_error

        bounds_error = .false.

        $:GPU_PARALLEL_LOOP(private='[q, gp, i, j, k, patch_id, dist, min_cell_width, dim, bound, dir, index, temp_loc, s_cc]', &
                            & copy='[bounds_error]', present='[ghost_points]')
        do q = 1, num_gps
            gp = ghost_points(q)
            i = gp%loc(1)
            j = gp%loc(2)
            k = gp%loc(3)

            ! Calculate and store the precise location of the image point
            patch_id = gp%ib_patch_id
            dist = abs(real(gp%levelset, kind=wp))
            ghost_points(q)%ip_dist = dist
            ! keep the image point stencil off the wall
            if (ib_second_order_vel) then
                min_cell_width = min(dx(i), dy(j))
                if (p > 0) min_cell_width = min(min_cell_width, dz(k))
                ghost_points(q)%ip_dist = max(dist, ib_ip_min_dist*min_cell_width)
            end if
            ! Per-component scalar stores: the array-valued form of this assignment was intermittently overwritten by another
            ! thread on AMD GPUs (a few ip_loc(2) of ~4000 per call), which sent the image-point search past the halo
            ghost_points(q)%ip_loc(1) = x_cc(i) + (dist + ghost_points(q)%ip_dist)*gp%levelset_norm(1)
            ghost_points(q)%ip_loc(2) = y_cc(j) + (dist + ghost_points(q)%ip_dist)*gp%levelset_norm(2)
            ghost_points(q)%ip_loc(3) = 0._wp
            if (p > 0) ghost_points(q)%ip_loc(3) = z_cc(k) + (dist + ghost_points(q)%ip_dist)*gp%levelset_norm(3)

            ! Find the closest grid point to the image point
            do dim = 1, num_dims
                ! s_cc points to the dim array we need
                if (dim == 1) then
                    s_cc => x_cc
                    bound = m + buff_size - 1
                else if (dim == 2) then
                    s_cc => y_cc
                    bound = n + buff_size - 1
                else
                    s_cc => z_cc
                    bound = p + buff_size - 1
                end if

                if (f_approx_equal(gp%levelset_norm(dim), 0._wp)) then
                    ! if the ghost point is almost equal to a cell location, we set it equal and continue
                    ghost_points(q)%ip_grid(dim) = ghost_points(q)%loc(dim)
                else
                    if (gp%levelset_norm(dim) > 0) then
                        dir = 1
                    else
                        dir = -1
                    end if

                    index = ghost_points(q)%loc(dim)
                    temp_loc = ghost_points(q)%ip_loc(dim)
                    do while ((temp_loc < s_cc(index) .or. temp_loc > s_cc(index + 1)) .and. (.not. bounds_error) &
                              & .and. ghost_points(q)%ip_ok)
                        index = index + dir
                        ! A deep cell exists only so the viscous stencils never read a stale cell, and its image point lies
                        ! twice its depth away, which can pass the halo near a rank boundary. That is not an error: the cell takes
                        ! the neighbour average in s_ibm_correct_state. A margin of two keeps the interpolation stencil inside.
                        if (ghost_points(q)%layer > gp_layers_full .and. (index < -buff_size + 2 .or. index > bound - 2)) then
                            ghost_points(q)%ip_ok = .false.
                            exit  ! before the loop condition reads s_cc past its end
                        end if
                        if (index < -buff_size .or. index > bound) then
#if !defined(MFC_OpenACC) && !defined(MFC_OpenMP)
                            print *, "A required image point is not located in this computational domain."
                            print *, "Ghost Point is located at :"
                            if (p == 0) then
                                print *, [x_cc(i), y_cc(j)]
                            else
                                print *, [x_cc(i), y_cc(j), z_cc(k)]
                            end if
                            print *, "We are searching in dimension ", dim, " for image point at ", ghost_points(q)%ip_loc(:)
                            print *, "Domain size: "
                            print *, "x: ", x_cc(-buff_size), " to: ", x_cc(m + buff_size - 1)
                            print *, "y: ", y_cc(-buff_size), " to: ", y_cc(n + buff_size - 1)
                            if (p /= 0) print *, "z: ", z_cc(-buff_size), " to: ", z_cc(p + buff_size - 1)
                            print *, "Image point is located approximately ", &
                                & (ghost_points(q)%loc(dim) - ghost_points(q) %ip_loc(dim))/(s_cc(1) - s_cc(0)), " grid cells away"
                            print *, "Levelset ", dist, " and Norm: ", gp%levelset_norm
                            print *, &
                                & "A short term fix may include increasing buff_size further in m_helper_basic (currently set to a minimum of 10)"
#endif
                            bounds_error = .true.
                        end if
                    end do

                    ghost_points(q)%ip_grid(dim) = index
                    if (ghost_points(q)%DB(dim) == -1) then
                        ghost_points(q)%ip_grid(dim) = ghost_points(q)%loc(dim) + 1
                    else if (ghost_points(q)%DB(dim) == 1) then
                        ghost_points(q)%ip_grid(dim) = ghost_points(q)%loc(dim) - 1
                    end if
                end if
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        @:PROHIBIT(bounds_error, "Ghost Point and Image Point on Different Processors. Exiting")

    end subroutine s_compute_image_points

    !> Count the number of ghost points for memory allocation
    subroutine s_find_num_ghost_points(num_gps_out)

        integer, intent(out) :: num_gps_out
        integer              :: i, j, k        !< Iterator variables
        integer              :: num_gps_local  !< local copies of the gp count to support GPU compute
        integer              :: lay, gl_di, gl_dj, gl_dk, gl_dk_max, gl_rem_j, gl_rem_k, gl_cheb

        num_gps_local = 0

        $:GPU_PARALLEL_LOOP(private='[i, j, k, lay, gl_di, gl_dj, gl_dk, gl_dk_max, gl_rem_j, gl_rem_k, gl_cheb]', &
                            & copy='[num_gps_local]', collapse=3)
        do i = 0, m
            do j = 0, n
                do k = 0, p
                    @:GHOST_LAYER(lay, i, j, k)
                    if (lay > 0) then
                        $:GPU_ATOMIC(atomic='update')
                        num_gps_local = num_gps_local + 1
                    end if
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        num_gps_out = num_gps_local

    end subroutine s_find_num_ghost_points

    !> Locate all ghost points in the domain
    subroutine s_find_ghost_points()

        integer :: i, j, k, lay  !< Iterator variables
        integer :: xp, yp, zp    !< periodicities
        integer :: count, count_i, local_idx
        integer :: patch_id, encoded_patch_id, neighborhood_patch_id
        integer :: gl_di, gl_dj, gl_dk, gl_dk_max, gl_rem_j, gl_rem_k, gl_cheb

        count = 0
        count_i = 0

        $:GPU_PARALLEL_LOOP(private='[i, j, k, lay, local_idx, patch_id, encoded_patch_id, neighborhood_patch_id, xp, yp, zp, &
                            & gl_di, gl_dj, gl_dk, gl_dk_max, gl_rem_j, gl_rem_k, gl_cheb]', copyin='[count, count_i, &
                            & glb_bounds]', present='[ghost_points]', collapse=3)
        do i = 0, m
            do j = 0, n
                do k = 0, p
                    if (ib_markers%sf(i, j, k) /= 0) then
                        @:GHOST_LAYER(lay, i, j, k)

                        if (lay > 0) then
                            $:GPU_ATOMIC(atomic='capture')
                            count = count + 1
                            local_idx = count
                            $:END_GPU_ATOMIC_CAPTURE()
                            ghost_points(local_idx)%loc(1) = i
                            ghost_points(local_idx)%loc(2) = j
                            ghost_points(local_idx)%loc(3) = k
                            ghost_points(local_idx)%T_cond = T_cond_unset
                            ghost_points(local_idx)%layer = lay
                            ghost_points(local_idx)%ip_ok = .true.
                            encoded_patch_id = ib_markers%sf(i, j, k)
                            call s_decode_patch_periodicity(encoded_patch_id, patch_id, xp, yp, zp)
                            call s_get_neighborhood_idx(patch_id, neighborhood_patch_id)
                            ghost_points(local_idx)%ib_patch_id = neighborhood_patch_id
                            ghost_points(local_idx)%x_periodicity = xp
                            ghost_points(local_idx)%y_periodicity = yp
                            ghost_points(local_idx)%z_periodicity = zp
                            ghost_points(local_idx)%slip = patch_ib(neighborhood_patch_id)%slip

                            #:for X, ID, IDX in [('x', 1, 'i'), ('y', 2, 'j'), ('z', 3, 'k')]
                                ghost_points(local_idx)%DB(${ID}$) = 0
                                if (${ID}$ <= num_dims) then  ! Onlyrun the numeher of dimensions present
                                    if (ib_bc_${X}$%beg /= BC_PERIODIC) then
                                        if ((${X}$_cc(${IDX}$) - d${X}$(${IDX}$)) < glb_bounds(${ID}$)%beg) then
                                            ! if the grid cell is in a wall on the "left"
                                            ghost_points(local_idx)%DB(${ID}$) = -1
                                        else if ((${X}$_cc(${IDX}$) + d${X}$(${IDX}$)) > glb_bounds(${ID}$)%end) then
                                            ! if the grid cell is in a wall on the "right"
                                            ghost_points(local_idx)%DB(${ID}$) = 1
                                        end if
                                    end if
                                end if
                            #:endfor
                        end if
                    end if
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

    end subroutine s_find_ghost_points

    !> Compute the interpolation coefficients for image points
    subroutine s_compute_interpolation_coeffs()

        real(wp), dimension(2, 2, 2) :: dist
        real(wp), dimension(2, 2, 2) :: alpha
        real(wp), dimension(2, 2, 2) :: interp_coeffs
        real(wp)                     :: buf
        real(wp), dimension(2, 2, 2) :: eta
        real(wp), dimension(3, 3, 3) :: stencil_coeffs          !< Coefficients from the stencil's lower corner
        real(wp)                     :: fluid_w                 !< Weight carried by fluid cells (TSC)
        integer, dimension(3)        :: lo                      !< Lower corner of the stencil
        integer, dimension(3)        :: ic                      !< Cell centring a TSC stencil
        real(wp), dimension(3, 3)    :: wd                      !< 1D TSC weights per direction
        real(wp), dimension(3)       :: sample_loc              !< Weighted stencil centroid relative to the ghost point
        type(ghost_point)            :: gp
        integer                      :: q, i, j, k, ii, jj, kk  !< Grid indexes and iterators
        logical                      :: is_cell_center

        $:GPU_PARALLEL_LOOP(private='[q, i, j, k, ii, jj, kk, dist, buf, gp, interp_coeffs, eta, alpha, is_cell_center, &
                            & stencil_coeffs, fluid_w, lo, ic, wd, sample_loc]', present='[ghost_points]')
        do q = 1, num_gps
            gp = ghost_points(q)
            if (.not. gp%ip_ok) then
                ghost_points(q)%interp_valid = .false.  ! takes the neighbour average in s_ibm_correct_state
                cycle
            end if
            stencil_coeffs = 0._wp
            if (ib_ip_interp == 2) then
                call s_tsc_weights(ib_markers, gp%ip_loc, ic, wd, fluid_w)
                lo = ic - 1
                if (p == 0) lo(3) = 0
                stencil_coeffs = f_tsc_stencil(wd)
                ! An injecting body's ghost cells hold the imposed fuel state, not a mirror of the gas, so they must not feed its
                ! own image point: weight the fluid cells only, as the inverse-distance stencil does
                if (patch_ib(gp%ib_patch_id)%inj_species > 0) then
                    buf = 0._wp
                    do kk = 1, merge(3, 1, p /= 0)
                        do jj = 1, 3
                            do ii = 1, 3
                                if (ib_markers%sf(lo(1) + ii - 1, lo(2) + jj - 1, lo(3) + kk - 1) /= 0) stencil_coeffs(ii, jj, &
                                    & kk) = 0._wp
                                buf = buf + stencil_coeffs(ii, jj, kk)
                            end do
                        end do
                    end do
                    if (buf > 0._wp) stencil_coeffs = stencil_coeffs/buf
                    fluid_w = buf
                end if
                ghost_points(q)%interp_valid = fluid_w > 0._wp
            else
                ! Get the interpolation points
                i = gp%ip_grid(1)
                j = gp%ip_grid(2)
                if (p /= 0) then
                    k = gp%ip_grid(3)
                else
                    k = 0
                end if

                ! get the distance to a cell in each direction
                dist = 0._wp
                buf = 1._wp
                do ii = 0, 1
                    do jj = 0, 1
                        if (p == 0) then
                            dist(1 + ii, 1 + jj, 1) = sqrt((x_cc(i + ii) - gp%ip_loc(1))**2 + (y_cc(j + jj) - gp%ip_loc(2))**2)
                        else
                            do kk = 0, 1
                                dist(1 + ii, 1 + jj, &
                                     & 1 + kk) = sqrt((x_cc(i + ii) - gp%ip_loc(1))**2 + (y_cc(j + jj) - gp%ip_loc(2))**2 &
                                     & + (z_cc(k + kk) - gp%ip_loc(3))**2)
                            end do
                        end if
                    end do
                end do

                ! check if we are arbitrarily close to a cell center
                interp_coeffs = 0._wp
                is_cell_center = .false.
                check_is_cell_center: do ii = 0, 1
                    do jj = 0, 1
                        ! A coincident solid cell leaves the point invalid: falling through to the eta
                        ! branch with dist = 0 would give alpha*eta = 0*Inf = NaN.
                        if (dist(ii + 1, jj + 1, 1) <= 1.e-16_wp) then
                            if (ib_markers%sf(i + ii, j + jj, k) == 0) interp_coeffs(ii + 1, jj + 1, 1) = 1._wp
                            is_cell_center = .true.
                            exit check_is_cell_center
                        else
                            if (p /= 0) then
                                if (dist(ii + 1, jj + 1, 2) <= 1.e-16_wp) then
                                    if (ib_markers%sf(i + ii, j + jj, k + 1) == 0) interp_coeffs(ii + 1, jj + 1, 2) = 1._wp
                                    is_cell_center = .true.
                                    exit check_is_cell_center
                                end if
                            end if
                        end if
                    end do
                end do check_is_cell_center

                if (.not. is_cell_center) then
                    ! if we are not arbitrarily close, interpolate
                    alpha = 1._wp
                    if (ib_markers%sf(i, j, k) /= 0) alpha(1, 1, 1) = 0._wp
                    if (ib_markers%sf(i + 1, j, k) /= 0) alpha(2, 1, 1) = 0._wp
                    if (ib_markers%sf(i, j + 1, k) /= 0) alpha(1, 2, 1) = 0._wp
                    if (ib_markers%sf(i + 1, j + 1, k) /= 0) alpha(2, 2, 1) = 0._wp

                    if (p == 0) then
                        eta(:,:,1) = 1._wp/dist(:,:,1)**2
                        buf = sum(alpha(:,:,1)*eta(:,:,1))
                        if (buf > 0._wp) interp_coeffs(:,:,1) = alpha(:,:,1)*eta(:,:,1)/buf
                    else
                        if (ib_markers%sf(i, j, k + 1) /= 0) alpha(1, 1, 2) = 0._wp
                        if (ib_markers%sf(i + 1, j, k + 1) /= 0) alpha(2, 1, 2) = 0._wp
                        if (ib_markers%sf(i, j + 1, k + 1) /= 0) alpha(1, 2, 2) = 0._wp
                        if (ib_markers%sf(i + 1, j + 1, k + 1) /= 0) alpha(2, 2, 2) = 0._wp
                        eta = 1._wp/dist**2
                        buf = sum(alpha*eta)
                        if (buf > 0._wp) interp_coeffs = alpha*eta/buf
                    end if
                end if
                lo = gp%ip_grid
                if (p == 0) lo(3) = 0
                stencil_coeffs(1:2,1:2,1:2) = interp_coeffs
                ghost_points(q)%interp_valid = any(interp_coeffs > 0._wp)
            end if

            ! An image point buried in a neighboring IB gets no weights at all: its ghost point is
            ! averaged from its own neighbors instead, in s_ibm_correct_state.
            do kk = 1, 3
                do jj = 1, 3
                    do ii = 1, 3
                        ghost_points(q)%interp_coeffs(ii, jj, kk) = stencil_coeffs(ii, jj, kk)
                    end do
                end do
                ghost_points(q)%ip_stencil(kk) = lo(kk)
            end do

            ! Wall distance of the weighted stencil centroid, which is where the image-point value is
            ! actually sampled: inverse-distance weights over a partly solid stencil sit farther out
            ! than the mirror point at |levelset|, most of all for a ghost point on the surface.
            sample_loc(1) = x_cc(gp%loc(1))
            sample_loc(2) = y_cc(gp%loc(2))
            sample_loc(3) = 0._wp
            if (p /= 0) sample_loc(3) = z_cc(gp%loc(3))
            sample_loc = -sample_loc
            do ii = 1, ib_ip_width
                do jj = 1, ib_ip_width
                    do kk = 1, merge(ib_ip_width, 1, p /= 0)
                        buf = stencil_coeffs(ii, jj, kk)
                        sample_loc(1) = sample_loc(1) + buf*x_cc(lo(1) + ii - 1)
                        sample_loc(2) = sample_loc(2) + buf*y_cc(lo(2) + jj - 1)
                        if (p /= 0) sample_loc(3) = sample_loc(3) + buf*z_cc(lo(3) + kk - 1)
                    end do
                end do
            end do
            ghost_points(q)%ip_dist = dot_product(sample_loc, gp%levelset_norm) - abs(gp%levelset)
            if (ghost_points(q)%interp_valid) corrected_gps%sf(gp%loc(1), gp%loc(2), gp%loc(3)) = 1
        end do
        $:END_GPU_PARALLEL_LOOP()

    end subroutine s_compute_interpolation_coeffs

    !> Interpolate primitive variables to a ghost point's image point from the weights in gp%interp_coeffs (see
    !! s_compute_interpolation_coeffs)
    subroutine s_interpolate_image_point(q_prim_vf, gp, alpha_rho_IP, alpha_IP, pres_IP, vel_IP, c_IP, r_IP, v_IP, pb_IP, mv_IP, &
                                         & nmom_IP, pb_in, mv_in, presb_IP, massv_IP, Ys_IP)
        $:GPU_ROUTINE(parallelism='[seq]')

        type(scalar_field), dimension(sys_size), intent(in) :: q_prim_vf  !< Primitive Variables
        real(stp), optional, dimension(idwbuff(1)%beg:,idwbuff(2)%beg:,idwbuff(3)%beg:,1:,1:), intent(in) :: pb_in, mv_in
        type(ghost_point), intent(in) :: gp
        real(wp), intent(inout) :: pres_IP
        real(wp), dimension(3), intent(inout) :: vel_IP
        real(wp), intent(inout) :: c_IP
        #:if not MFC_CASE_OPTIMIZATION and USING_AMD
            real(wp), dimension(3), intent(inout) :: alpha_IP, alpha_rho_IP
        #:else
            real(wp), dimension(num_fluids), intent(inout) :: alpha_IP, alpha_rho_IP
        #:endif
        real(wp), optional, dimension(:), intent(inout) :: r_IP, v_IP, pb_IP, mv_IP
        real(wp), optional, dimension(:), intent(inout) :: nmom_IP
        real(wp), optional, dimension(:), intent(inout) :: presb_IP, massv_IP
        real(wp), optional, dimension(:), intent(inout) :: Ys_IP  !< Interpolated species mass fractions (chemistry)
        integer                                         :: i, j, k, l, q  !< Iterator variables
        integer                                         :: i1, i2, j1, j2, k1, k2  !< Iterator variables
        real(wp)                                        :: coeff

        i1 = gp%ip_stencil(1); i2 = i1 + ib_ip_width - 1
        j1 = gp%ip_stencil(2); j2 = j1 + ib_ip_width - 1
        k1 = gp%ip_stencil(3); k2 = k1 + ib_ip_width - 1

        if (p == 0) then
            k1 = 0
            k2 = 0
        end if

        alpha_rho_IP = 0._wp
        alpha_IP = 0._wp
        pres_IP = 0._wp
        vel_IP = 0._wp

        if (chemistry) Ys_IP = 0._wp

        if (surface_tension) c_IP = 0._wp

        if (bubbles_euler) then
            r_IP = 0._wp
            v_IP = 0._wp
            if (.not. polytropic) then
                mv_IP = 0._wp
                pb_IP = 0._wp
            end if
        end if

        if (qbmm) then
            nmom_IP = 0._wp
            if (.not. polytropic) then
                presb_IP = 0._wp
                massv_IP = 0._wp
            end if
        end if

        $:GPU_LOOP(parallelism='[seq]')
        do i = i1, i2
            $:GPU_LOOP(parallelism='[seq]')
            do j = j1, j2
                $:GPU_LOOP(parallelism='[seq]')
                do k = k1, k2
                    coeff = gp%interp_coeffs(i - i1 + 1, j - j1 + 1, k - k1 + 1)

                    pres_IP = pres_IP + coeff*q_prim_vf(eqn_idx%E)%sf(i, j, k)

                    $:GPU_LOOP(parallelism='[seq]')
                    do q = eqn_idx%mom%beg, eqn_idx%mom%end
                        vel_IP(q + 1 - eqn_idx%mom%beg) = vel_IP(q + 1 - eqn_idx%mom%beg) + coeff*q_prim_vf(q)%sf(i, j, k)
                    end do

                    $:GPU_LOOP(parallelism='[seq]')
                    do l = eqn_idx%cont%beg, eqn_idx%cont%end
                        alpha_rho_IP(l) = alpha_rho_IP(l) + coeff*q_prim_vf(l)%sf(i, j, k)
                        alpha_IP(l) = alpha_IP(l) + coeff*q_prim_vf(eqn_idx%adv%beg + l - 1)%sf(i, j, k)
                    end do

                    if (surface_tension) then
                        c_IP = c_IP + coeff*q_prim_vf(eqn_idx%c)%sf(i, j, k)
                    end if

                    if (chemistry) then
                        $:GPU_LOOP(parallelism='[seq]')
                        do q = 1, num_species
                            Ys_IP(q) = Ys_IP(q) + coeff*q_prim_vf(eqn_idx%species%beg + q - 1)%sf(i, j, k)
                        end do
                    end if

                    if (bubbles_euler .and. .not. qbmm) then
                        $:GPU_LOOP(parallelism='[seq]')
                        do l = 1, nb
                            if (polytropic) then
                                r_IP(l) = r_IP(l) + coeff*q_prim_vf(eqn_idx%bub%beg + (l - 1)*2)%sf(i, j, k)
                                v_IP(l) = v_IP(l) + coeff*q_prim_vf(eqn_idx%bub%beg + 1 + (l - 1)*2)%sf(i, j, k)
                            else
                                r_IP(l) = r_IP(l) + coeff*q_prim_vf(eqn_idx%bub%beg + (l - 1)*4)%sf(i, j, k)
                                v_IP(l) = v_IP(l) + coeff*q_prim_vf(eqn_idx%bub%beg + 1 + (l - 1)*4)%sf(i, j, k)
                                pb_IP(l) = pb_IP(l) + coeff*q_prim_vf(eqn_idx%bub%beg + 2 + (l - 1)*4)%sf(i, j, k)
                                mv_IP(l) = mv_IP(l) + coeff*q_prim_vf(eqn_idx%bub%beg + 3 + (l - 1)*4)%sf(i, j, k)
                            end if
                        end do
                    end if

                    if (qbmm) then
                        do l = 1, nb*nmom
                            nmom_IP(l) = nmom_IP(l) + coeff*q_prim_vf(eqn_idx%bub%beg - 1 + l)%sf(i, j, k)
                        end do
                        if (.not. polytropic) then
                            do q = 1, nb
                                do l = 1, nnode
                                    presb_IP((q - 1)*nnode + l) = presb_IP((q - 1)*nnode + l) + coeff*real(pb_in(i, j, k, l, q), &
                                             & kind=wp)
                                    massv_IP((q - 1)*nnode + l) = massv_IP((q - 1)*nnode + l) + coeff*real(mv_in(i, j, k, l, q), &
                                             & kind=wp)
                                end do
                            end do
                        end if
                    end if
                end do
            end do
        end do

    end subroutine s_interpolate_image_point

    !> Triangular-shaped-cloud (quadratic B-spline) weights of a point on the 3x3(x3) stencil centred on the cell nearest to it
    !! (Ezra, Peles & Kozak 2025, Int. J. Heat Mass Transf. 252:127314), and the fraction of that weight carried by fluid cells. The
    !! kernel is a partition of unity and smooth across cell boundaries, so an interpolated wall quantity converges one order faster
    !! than from inverse-distance or bilinear weights, which are only continuous. The weights use the fractional position in the
    !! bracketing cell at integer offsets, which keeps the partition of unity exactly on a stretched grid. The stencil is kept
    !! inside the halo. Solid cells in it carry the ghost values of the last correction (staged, #1901); a point buried in a body
    !! has no fluid weight.
    subroutine s_tsc_weights(markers, loc, ic, wd, fluid_frac)

        $:GPU_ROUTINE(parallelism='[seq]')
        type(integer_field), intent(in) :: markers    !< ib_markers, passed so the device routine reads the caller's present copy
        real(wp), dimension(3), intent(in) :: loc
        integer, dimension(3), intent(out) :: ic      !< Cell centring the stencil
        real(wp), dimension(3, 3), intent(out) :: wd  !< 1D weights on ic - 1, ic, ic + 1, per direction
        real(wp), intent(out) :: fluid_frac
        real(wp) :: ds, frac, wsum, wfluid, buf
        integer :: dim, ii, jj, kk

        ic = 0
        wd = 0._wp
        do dim = 1, merge(3, 2, p /= 0)
            if (dim == 1) then
                ic(1) = max(min(floor((loc(1) - x_cc(0))/(x_cc(1) - x_cc(0))), m + buff_size - 2), -buff_size + 2)
                ! the estimate assumes a uniform grid: one step brackets the point on a mildly stretched one
                if (loc(1) < x_cc(ic(1))) ic(1) = max(ic(1) - 1, -buff_size + 2)
                if (loc(1) >= x_cc(ic(1) + 1)) ic(1) = min(ic(1) + 1, m + buff_size - 2)
                ds = x_cc(ic(1) + 1) - x_cc(ic(1)); frac = (loc(1) - x_cc(ic(1)))/ds
            else if (dim == 2) then
                ic(2) = max(min(floor((loc(2) - y_cc(0))/(y_cc(1) - y_cc(0))), n + buff_size - 2), -buff_size + 2)
                if (loc(2) < y_cc(ic(2))) ic(2) = max(ic(2) - 1, -buff_size + 2)
                if (loc(2) >= y_cc(ic(2) + 1)) ic(2) = min(ic(2) + 1, n + buff_size - 2)
                ds = y_cc(ic(2) + 1) - y_cc(ic(2)); frac = (loc(2) - y_cc(ic(2)))/ds
            else
                ic(3) = max(min(floor((loc(3) - z_cc(0))/(z_cc(1) - z_cc(0))), p + buff_size - 2), -buff_size + 2)
                if (loc(3) < z_cc(ic(3))) ic(3) = max(ic(3) - 1, -buff_size + 2)
                if (loc(3) >= z_cc(ic(3) + 1)) ic(3) = min(ic(3) + 1, p + buff_size - 2)
                ds = z_cc(ic(3) + 1) - z_cc(ic(3)); frac = (loc(3) - z_cc(ic(3)))/ds
            end if
            if (frac > 0.5_wp) then
                ic(dim) = ic(dim) + 1
                frac = frac - 1._wp
            end if
            wd(1, dim) = 0.5_wp*(1.5_wp - abs(frac + 1._wp))**2
            wd(2, dim) = 0.75_wp - frac*frac
            wd(3, dim) = 0.5_wp*(1.5_wp - abs(frac - 1._wp))**2
        end do
        if (p == 0) wd(1, 3) = 1._wp  ! 2D: the consumers index this slot as k - k1 + 1 = 1

        wsum = 0._wp; wfluid = 0._wp
        do ii = 1, 3
            do jj = 1, 3
                do kk = 1, merge(3, 1, p /= 0)
                    buf = wd(ii, 1)*wd(jj, 2)*wd(kk, 3)
                    wsum = wsum + buf
                    if (markers%sf(ic(1) + ii - 2, ic(2) + jj - 2, ic(3) + (kk - 2)*min(p, 1)) == 0) wfluid = wfluid + buf
                end do
            end do
        end do
        fluid_frac = wfluid/wsum

    end subroutine s_tsc_weights

    !> Tensor product of the 1D TSC weights: the coefficient on each cell of the 3x3x3 stencil.
    function f_tsc_stencil(wd) result(c)

        $:GPU_ROUTINE(function_name='f_tsc_stencil', parallelism='[seq]', cray_inline=True)
        real(wp), dimension(3, 3), intent(in) :: wd
        real(wp), dimension(3, 3, 3)          :: c
        integer                               :: ii, jj, kk

        do ii = 1, 3
            do jj = 1, 3
                do kk = 1, 3
                    c(ii, jj, kk) = wd(ii, 1)*wd(jj, 2)*wd(kk, 3)
                end do
            end do
        end do

    end function f_tsc_stencil

    !> Apply TSC weights from s_tsc_weights to one field.
    function f_tsc_apply(sf, ic, wd) result(val)

        $:GPU_ROUTINE(function_name='f_tsc_apply', parallelism='[seq]', cray_inline=True)
        type(scalar_field), intent(in)        :: sf
        integer, dimension(3), intent(in)     :: ic
        real(wp), dimension(3, 3), intent(in) :: wd
        real(wp)                              :: val
        integer                               :: ii, jj, kk

        val = 0._wp
        do ii = 1, 3
            do jj = 1, 3
                do kk = 1, merge(3, 1, p /= 0)
                    val = val + wd(ii, 1)*wd(jj, 2)*wd(kk, 3)*sf%sf(ic(1) + ii - 2, ic(2) + jj - 2, ic(3) + (kk - 2)*min(p, 1))
                end do
            end do
        end do

    end function f_tsc_apply

    !> Ghost value from the quadratic wall-normal profile, bounded by the linear mirror (minmod). The two extrapolations agree
    !! exactly for a linear profile, so this limits only the curvature correction, and to the magnitude the linear mirror already
    !! produces. Unlimited, the quadratic amplifies that correction with ghost depth and the run fails with unphysical near-wall
    !! states, the positivity risk the literature attaches to higher-order ghost fills (Seshadri & De 2019/2021; Tan & Shu 2010).
    function f_quad_fill(phi_w, phi_1, phi_2, lw, llin) result(phi_g)

        $:GPU_ROUTINE(function_name='f_quad_fill', parallelism='[seq]', cray_inline=True)
        real(wp), intent(in)               :: phi_w, phi_1, phi_2
        real(wp), dimension(3), intent(in) :: lw
        real(wp), dimension(2), intent(in) :: llin
        real(wp)                           :: phi_g, d_quad, d_lin

        d_quad = lw(1)*phi_w + lw(2)*phi_1 + lw(3)*phi_2 - phi_w
        d_lin = llin(1)*phi_w + llin(2)*phi_1 - phi_w
        if (d_quad*d_lin <= 0._wp) then
            phi_g = phi_w + d_lin  ! opposite signs: the curvature term has reversed the extrapolation
        else if (abs(d_quad) <= abs(d_lin)) then
            phi_g = phi_w + d_quad
        else
            phi_g = phi_w + d_lin
        end if

    end function f_quad_fill

    !> Cell scale of a ghost point: the geometric mean of the cell widths.
    function f_cell_scale(gp) result(h)

        $:GPU_ROUTINE(function_name='f_cell_scale', parallelism='[seq]', cray_inline=True)
        type(ghost_point), intent(in) :: gp
        real(wp)                      :: h, dV

        dV = dx(gp%loc(1))*dy(gp%loc(2))
        if (p > 0) dV = dV*dz(gp%loc(3))
        h = dV**(1._wp/real(num_dims, wp))

    end function f_cell_scale

    !> TSC stencils of the two probes at wall distances s1 and 2 s1 along the surface normal from a ghost point's wall foot, and the
    !! fluid fraction of each.
    subroutine s_wall_probes(markers, gp, s1, ic1, wd1, f1, ic2, wd2, f2)

        $:GPU_ROUTINE(parallelism='[seq]')
        type(integer_field), intent(in) :: markers  !< ib_markers, passed so the device routine reads the caller's present copy
        type(ghost_point), intent(in) :: gp
        real(wp), intent(in) :: s1
        integer, dimension(3), intent(out) :: ic1, ic2
        real(wp), dimension(3, 3), intent(out) :: wd1, wd2
        real(wp), intent(out) :: f1, f2
        real(wp), dimension(3) :: norm, wall_loc

        norm = gp%levelset_norm/max(sqrt(sum(gp%levelset_norm**2)), sgm_eps)
        wall_loc = 0._wp
        wall_loc(1) = x_cc(gp%loc(1)); wall_loc(2) = y_cc(gp%loc(2))
        if (p > 0) wall_loc(3) = z_cc(gp%loc(3))
        wall_loc = wall_loc + abs(real(gp%levelset, kind=wp))*norm
        call s_tsc_weights(markers, wall_loc + s1*norm, ic1, wd1, f1)
        call s_tsc_weights(markers, wall_loc + 2._wp*s1*norm, ic2, wd2, f2)

    end subroutine s_wall_probes

    !> Lagrange weights that extrapolate a quadratic wall-normal profile back to a ghost point, with the wall at s = 0, probes at s1
    !! and s2 = 2 s1 and the ghost at s = -d (d = |levelset|):
    !!
    !!     phi_g = L0 phi_w + L1 phi_1 + L2 phi_2
    !!     L0 = (d + s1)(d + s2)/(s1 s2),  L1 = -d (d + s2)/(s1 (s2 - s1)),  L2 = d (d + s1)/(s2 (s2 - s1))
    !!
    !! L0 + L1 + L2 = 1, and at d = 0 they collapse to (1, 0, 0), so the surface value is still enforced exactly on the surface; the
    !! fill is linear in phi_w. This is what the flow solve consumes: its order is the order of the boundary condition and hence of
    !! the wall shear, the drag and the wall heat flux. Fixed probes at a fixed multiple of the spacing keep the fit conditioned for a
    !! ghost almost on the surface, but a ghost can sit 3h inside the body and then d/s1 = 3 and the weights reach (10, -15, 6);
    !! s1 = max(h, d) caps them at (3, -3, 1) at any depth and leaves shallow ghosts on fixed probes.
    !!
    !! Degradation is explicit: tier 1 uses both probes; tier 2 is the linear mirror where the outer probe has too little fluid
    !! around it (a narrow inter-particle gap, a rim, a neighbouring body); tier 3 leaves the caller's own treatment. The tier mix
    !! is reported, so a second-order claim is never made for a surface that did not earn it.
    subroutine s_wall_fill(markers, gp, wf)

        $:GPU_ROUTINE(parallelism='[seq]')
        type(integer_field), intent(in) :: markers  !< ib_markers, passed so the device routine reads the caller's present copy
        type(ghost_point), intent(in)   :: gp
        type(wall_fill), intent(out)    :: wf
        real(wp), parameter             :: FLUID_FRAC_MIN = 0.5_wp  !< A probe needs this much fluid weight to be used
        real(wp)                        :: d, s1, s2, f1, f2

        wf%tier = 3
        wf%lw = [1._wp, 0._wp, 0._wp]
        wf%ic1 = 0; wf%ic2 = 0; wf%wd1 = 0._wp; wf%wd2 = 0._wp

        d = abs(real(gp%levelset, kind=wp))
        s1 = max(f_cell_scale(gp), d)
        s2 = 2._wp*s1
        call s_wall_probes(markers, gp, s1, wf%ic1, wf%wd1, f1, wf%ic2, wf%wd2, f2)

        wf%llin = [1._wp + d/s1, -d/s1]  ! the linear mirror, which also bounds the quadratic
        if (f1 < FLUID_FRAC_MIN) return
        if (f2 < FLUID_FRAC_MIN) then
            wf%lw = [wf%llin(1), wf%llin(2), 0._wp]
            wf%tier = 2
            return
        end if
        wf%lw = [(d + s1)*(d + s2)/(s1*s2), -d*(d + s2)/(s1*(s2 - s1)), d*(d + s1)/(s2*(s2 - s1))]
        wf%tier = 1

    end subroutine s_wall_fill

    !> Wall-normal gradient of a scalar at a ghost point's wall foot, as the coefficients of dphi/dn|_w = a - b*phi_w: the wall
    !! value is the Newton unknown of the surface balance and the fit is linear in it. Two probes at h and 2h give the quadratic
    !! gradient dphi/dn|_w = [s2^2 (phi_1 - phi_w) - s1^2 (phi_2 - phi_w)] / [s1 s2 (s2 - s1)], second order where the one-probe
    !! difference (phi_IP - phi_w)/d it replaces is first. Raising the profile degree alone does not help: the truncation is O(s1)
    !! and the interpolation error O(h^2)/s1 is also O(h), so the interpolation must reach O(h^3) too, which TSC does (Ezra, Peles &
    !! Kozak 2025; Jost & Glockner 2021; Yousefzadeh & Battiato 2019). Tier 1 uses both probes, tier 2 the one-probe linear
    !! gradient, tier 3 reports failure so the caller keeps its one-sided difference.
    subroutine s_wall_normal_gradient(markers, sf, gp, a_coef, b_coef, tier)

        $:GPU_ROUTINE(parallelism='[seq]')
        type(integer_field), intent(in) :: markers  !< ib_markers, passed so the device routine reads the caller's present copy
        type(scalar_field), intent(in)  :: sf
        type(ghost_point), intent(in)   :: gp
        real(wp), intent(out)           :: a_coef, b_coef
        integer, intent(out)            :: tier
        real(wp), parameter             :: FLUID_FRAC_MIN = 0.5_wp
        real(wp)                        :: s1, s2, phi1, phi2, f1, f2
        integer, dimension(3)           :: ic1, ic2
        real(wp), dimension(3, 3)       :: wd1, wd2

        a_coef = 0._wp
        b_coef = 0._wp
        tier = 3
        s1 = f_cell_scale(gp)
        s2 = 2._wp*s1
        call s_wall_probes(markers, gp, s1, ic1, wd1, f1, ic2, wd2, f2)
        if (f1 < FLUID_FRAC_MIN) return
        phi1 = f_tsc_apply(sf, ic1, wd1)
        if (f2 < FLUID_FRAC_MIN) then
            a_coef = phi1/s1
            b_coef = 1._wp/s1
            tier = 2
            return
        end if
        phi2 = f_tsc_apply(sf, ic2, wd2)
        a_coef = (s2*s2*phi1 - s1*s1*phi2)/(s1*s2*(s2 - s1))
        b_coef = (s1 + s2)/(s1*s2)
        tier = 1

    end subroutine s_wall_normal_gradient

    !> Resets the current indexes of immersed boundaries and replaces them after updating
    !> the position of each moving immersed boundary
    impure subroutine s_update_mib(num_ibs)

        integer, intent(in) :: num_ibs
        integer             :: i, j, k, z_buff_size

        call nvtxStartRange("UPDATE-MIBM")

        ! Clears the existing immersed boundary indices over the same halo extent s_get_bounding_indices draws them
        z_buff_size = 0; if (p /= 0) z_buff_size = buff_size
        $:GPU_PARALLEL_LOOP(private='[i, j, k]')
        do i = -buff_size, m + buff_size; do j = -buff_size, n + buff_size; do k = -z_buff_size, p + z_buff_size
            ib_markers%sf(i, j, k) = 0._wp
            corrected_gps%sf(i, j, k) = 0
        end do; end do; end do
        $:END_GPU_PARALLEL_LOOP()

        ! recalulcate the rotation matrix based upon the new angles
        $:GPU_PARALLEL_LOOP(private='[i]')
        do i = 1, num_ibs
            if (patch_ib(i)%moving_ibm /= 0) then
                call s_update_ib_rotation_matrix(i)
            end if
        end do
        $:END_GPU_PARALLEL_LOOP()

        ! recompute the new ib_patch locations and broadcast them.
        call nvtxStartRange("APPLY-IB-PATCHES")
        call s_apply_ib_patches(ib_markers)
        call nvtxEndRange

        call nvtxStartRange("COMPUTE-GHOST-POINTS")
        ! recalculate the ghost point locations and coefficients
        call s_find_num_ghost_points(num_gps)
        ! overrunning ghost_points is a silent device fault; name the numbers instead
        if (int(num_gps, 8) > int(size(ghost_points), 8)) then
            print '(A,I0,A,I0,A,I0)', 'ghost points on rank ', proc_rank, ' grew to ', num_gps, ' but the array holds ', &
                & size(ghost_points)
            call s_mpi_abort('moving immersed boundary: ghost-point count exceeded its allocation')
        end if
        ! num_gps is a declare-target module variable and bounds every device loop over the ghost points; it is
        ! copied to the device once in s_ibm_setup, so without this refresh the kernels keep the setup-time count
        ! as a moving body's count changes: stale list entries beyond the current count are written as wall
        ! states into cells that are now fluid, new ghost cells are left uncorrected, and which entries those are
        ! depends on the atomic fill order -- nondeterministic results run to run (#1886).
        $:GPU_UPDATE(device='[num_gps]')
        call s_find_ghost_points()
        call nvtxEndRange

        call nvtxStartRange("COMPUTE-IMAGE-POINTS")
        call s_apply_levelset(ghost_points, num_gps)
        call s_compute_image_points()
        call s_compute_interpolation_coeffs()
        call nvtxEndRange

        call nvtxEndRange

    end subroutine s_update_mib

    !> log(cosh(x)) without overflow for large |x|
    pure function f_log_cosh(x) result(y)

        $:GPU_ROUTINE(parallelism='[seq]')
        real(wp), intent(in) :: x
        real(wp)             :: y
        y = abs(x) + log(0.5_wp*(1._wp + exp(-2._wp*abs(x))))

    end function f_log_cosh

    !> Prescribed kinematics for patch i at time t: hinged flapping (kin_model = 1) or the Eldredge pitch ramp (kin_model = 2).
    !! Flapping rolls phi about the lab x axis through the hinge and pitches theta about the body spanwise (y) axis through it; the
    !! ramp holds phi at zero and drives theta alone. Both compose as R = Rx(phi) Ry(theta) (MFC's angle convention) and set the
    !! angles, centroid, velocity and lab-frame angular velocity; nothing is integrated, so restarts are exact.
    subroutine s_prescribed_kinematics(i, t)

        $:GPU_ROUTINE(parallelism='[seq]')

        integer, intent(in)    :: i
        real(wp), intent(in)   :: t
        real(wp)               :: tau, amp, damp, omega, phi, theta, phid, thetad, arg_phi, arg_th, t_p, a_s
        real(wp), dimension(3) :: r, c, w, v
        integer                :: d

        tau = t - patch_ib(i)%kin_t0
        omega = 2._wp*pi*patch_ib(i)%kin_freq
        amp = 0._wp
        damp = 0._wp
        if (tau >= 0._wp) then
            amp = 1._wp
            if (patch_ib(i)%kin_ramp > 0._wp .and. tau < patch_ib(i)%kin_ramp) then
                amp = 0.5_wp*(1._wp - cos(pi*tau/patch_ib(i)%kin_ramp))
                damp = 0.5_wp*pi/patch_ib(i)%kin_ramp*sin(pi*tau/patch_ib(i)%kin_ramp)
            end if
        end if
        arg_phi = omega*tau
        arg_th = omega*tau + patch_ib(i)%kin_phase

        if (patch_ib(i)%kin_model == 2) then
            ! Eldredge smoothed linear pitch ramp and hold (AIAA canonical pitch-up): with a = kin_smooth, nominal rate
            ! Omega = kin_pitch_rate and pitch time t_p = theta0/Omega, G = log[cosh(a tau)/cosh(a (tau - t_p))] rises from
            ! -a t_p to +a t_p, so theta = mean + theta0/2 [1 + G/(a t_p)] and theta' = Omega/2 [tanh(a tau) - tanh(a (tau - t_p))]
            phi = 0._wp
            phid = 0._wp
            t_p = patch_ib(i)%kin_theta0/patch_ib(i)%kin_pitch_rate
            a_s = patch_ib(i)%kin_smooth
            theta = patch_ib(i)%kin_theta_mean + 0.5_wp*patch_ib(i)%kin_theta0*(1._wp + (f_log_cosh(a_s*tau) &
                             & - f_log_cosh(a_s*(tau - t_p)))/(a_s*t_p))
            thetad = 0.5_wp*patch_ib(i)%kin_pitch_rate*(tanh(a_s*tau) - tanh(a_s*(tau - t_p)))
        else
            phi = amp*patch_ib(i)%kin_phi0*sin(arg_phi)
            phid = damp*patch_ib(i)%kin_phi0*sin(arg_phi) + amp*patch_ib(i)%kin_phi0*omega*cos(arg_phi)
            theta = patch_ib(i)%kin_theta_mean + amp*patch_ib(i)%kin_theta0*sin(arg_th)
            thetad = damp*patch_ib(i)%kin_theta0*sin(arg_th) + amp*patch_ib(i)%kin_theta0*omega*cos(arg_th)
        end if

        ! centroid relative to the hinge: Rx(phi) Ry(theta) offset. kin_offset points to the case-file centroid; a
        ! centroid moved to the centre of mass (geometries 4, 5, 11, 12) sits centroid_offset short of it
        r = patch_ib(i)%kin_offset - patch_ib(i)%centroid_offset
        c(1) = cos(theta)*r(1) + sin(theta)*r(3)
        c(2) = r(2)
        c(3) = -sin(theta)*r(1) + cos(theta)*r(3)
        v(1) = c(1)
        v(2) = cos(phi)*c(2) - sin(phi)*c(3)
        v(3) = sin(phi)*c(2) + cos(phi)*c(3)
        c = v

        ! lab-frame angular velocity: phi' e_x + theta' Rx(phi) e_y
        w(1) = phid
        w(2) = thetad*cos(phi)
        w(3) = thetad*sin(phi)
        call s_cross_product(w, c, v)

        patch_ib(i)%angles(1) = phi
        patch_ib(i)%angles(2) = theta
        patch_ib(i)%angles(3) = 0._wp
        patch_ib(i)%x_centroid = patch_ib(i)%kin_hinge(1) + c(1)
        patch_ib(i)%y_centroid = patch_ib(i)%kin_hinge(2) + c(2)
        patch_ib(i)%z_centroid = patch_ib(i)%kin_hinge(3) + c(3)
        do d = 1, 3  ! component stores into patch_ib on the device (see s_update_ib_rotation_matrix)
            patch_ib(i)%vel(d) = v(d)
            patch_ib(i)%angular_vel(d) = w(d)
        end do

    end subroutine s_prescribed_kinematics

    !> Add direction idir's fluid-body face fluxes to each IB. These are the fluxes the RHS differences, Riemann (flux_vf) plus
    !! viscous, capillary, conductive and diffusive (flux_src_vf), so each body receives exactly what the fluid loses. A face counts
    !! on the rank owning its fluid cell. With IB forces: force (1:3) and torque about the centroid (4:6) of each local IB into
    !! ib_face_ft, from the last RHS evaluation (idir = 1 starts afresh). With ib_exchange: energy in and mass out of each lumped
    !! body (every body with ib_heat_wrt) into ib_heat(2:3), weighted so a step's stages sum to its exchange (SSP-RK: 1; 1/2, 1/2;
    !! 1/6, 1/6, 2/3).
    subroutine s_accumulate_ib_face_fluxes(idir, flux_vf, flux_src_vf, stage)

        integer, intent(in)                                 :: idir, stage
        type(scalar_field), dimension(sys_size), intent(in) :: flux_vf, flux_src_vf
        real(wp), dimension(3), parameter                   :: w_rk3 = [1._wp/6._wp, 1._wp/6._wp, 2._wp/3._wp]
        real(wp), dimension(3)                              :: f, r, tq, vb
        real(wp)                                            :: area, w, f_E, f_m, f_mom
        integer                                             :: i, j, k, l, jn, kn, ln, cell(3), side, ib_id, ib_tmp, xp, yp, zp, g
        integer                                             :: j0, k0, l0
        logical                                             :: has_src, has_src_E, do_ft, all_ibs, moving

        if (num_ibs == 0) return
        do_ft = moving_immersed_boundary_flag .or. ib_state_wrt
        all_ibs = ib_heat_wrt .or. ib_mass_relax > 0

        if (do_ft .and. idir == 1) then
            $:GPU_PARALLEL_LOOP(private='[i, l]', collapse=2)
            do i = 1, num_ibs
                do l = 1, 6
                    ib_face_ft(l, i) = 0._wp
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()
        end if

        ! The conditions under which s_compute_additional_physics_rhs differences flux_src for the momentum and the energy
        has_src = viscous .or. surface_tension .or. heat_conduction
        has_src_E = has_src .or. chem_params%diffusion
        w = 1._wp
        if (time_stepper == time_stepper_rk2) w = 0.5_wp
        if (time_stepper == time_stepper_rk3) w = w_rk3(stage)
        j0 = 0; k0 = 0; l0 = 0
        if (idir == 1) j0 = -1
        if (idir == 2) k0 = -1
        if (idir == 3) l0 = -1

        ! Face (j, k, l) lies between that cell and the next one along idir
        $:GPU_PARALLEL_LOOP(collapse=3, private='[i, j, k, l, jn, kn, ln, cell, side, ib_id, ib_tmp, xp, yp, zp, g, area, f, r, &
                            & tq, f_E, f_m, vb, f_mom, moving]', copyin='[has_src, has_src_E, do_ft, all_ibs, w, idir, j0, k0, l0]')
        do l = l0, p
            do k = k0, n
                do j = j0, m
                    jn = j; kn = k; ln = l
                    if (idir == 1) jn = j + 1
                    if (idir == 2) kn = k + 1
                    if (idir == 3) ln = l + 1
                    ! side = +1: body on the far side, the face flux enters it; -1: body on the near side, the flux leaves it
                    side = 0
                    if (ib_markers%sf(j, k, l) == 0 .and. ib_markers%sf(jn, kn, ln) /= 0) then
                        if (j >= 0 .and. k >= 0 .and. l >= 0) side = 1
                    else if (ib_markers%sf(j, k, l) /= 0 .and. ib_markers%sf(jn, kn, ln) == 0) then
                        if (jn <= m .and. kn <= n .and. ln <= p) side = -1
                    end if
                    if (side /= 0) then
                        cell = [jn, kn, ln]
                        if (side == -1) cell = [j, k, l]
                        call s_decode_patch_periodicity(ib_markers%sf(cell(1), cell(2), cell(3)), ib_tmp, xp, yp, zp)
                        call s_get_neighborhood_idx(ib_tmp, ib_id)
                        if (ib_id > 0) then
                            if (idir == 1) area = dy(k)
                            if (idir == 2) area = dx(j)
                            if (idir == 3) area = dx(j)*dy(k)
                            if (num_dims == 3 .and. idir /= 3) area = area*dz(l)
                            moving = patch_ib(ib_id)%moving_ibm > 0
                            if (do_ft .or. moving) then
                                ! Lever arm: face centre minus the centroid of the body's periodic image
                                r = 0._wp
                                r(1) = merge(x_cb(j), x_cc(j), idir == 1) - (patch_ib(ib_id)%x_centroid + real(xp, &
                                  & wp)*(glb_bounds(1)%end - glb_bounds(1)%beg))
                                r(2) = merge(y_cb(k), y_cc(k), idir == 2) - (patch_ib(ib_id)%y_centroid + real(yp, &
                                  & wp)*(glb_bounds(2)%end - glb_bounds(2)%beg))
                                if (num_dims == 3) r(3) = merge(z_cb(l), z_cc(l), &
                                    & idir == 3) - (patch_ib(ib_id)%z_centroid + real(zp, &
                                    & wp)*(glb_bounds(3)%end - glb_bounds(3)%beg))
                            end if
                            if (do_ft) then
                                f = 0._wp
                                do i = 1, num_dims
                                    f(i) = flux_vf(eqn_idx%mom%beg + i - 1)%sf(j, k, l)
                                    if (has_src) f(i) = f(i) + flux_src_vf(eqn_idx%mom%beg + i - 1)%sf(j, k, l)
                                    f(i) = real(side, wp)*area*f(i)
                                end do
                                ! mean pressure gradient: the uniform gradient's surface pressure -f.r gives the displacement force
                                ! f*V
                                f(idir) = f(idir) - real(side, wp)*area*dot_product(mpg_force, r)
                                call s_cross_product(r, f, tq)
                                do i = 1, 3
                                    $:GPU_ATOMIC(atomic='update')
                                    ib_face_ft(i, ib_id) = ib_face_ft(i, ib_id) + f(i)
                                    $:GPU_ATOMIC(atomic='update')
                                    ib_face_ft(i + 3, ib_id) = ib_face_ft(i + 3, ib_id) + tq(i)
                                end do
                            end if
                            if (all_ibs .or. patch_ib(ib_id)%thermal_bc == 3) then
                                f_E = flux_vf(eqn_idx%E)%sf(j, k, l)
                                if (has_src_E) f_E = f_E + flux_src_vf(eqn_idx%E)%sf(j, k, l)
                                f_m = 0._wp
                                do i = eqn_idx%cont%beg, eqn_idx%cont%end
                                    f_m = f_m + flux_vf(i)%sf(j, k, l)
                                end do
                                if (moving) then
                                    ! Energy flux in the body frame, f_E - V_b.f_mom + |V_b|^2 f_m/2 (V_b = V + omega x r at the
                                    ! face): the lab flux also carries the wall's pressure/viscous work and the swept gas energy,
                                    ! which are not heat into the body
                                    vb(1) = patch_ib(ib_id)%vel(1) + patch_ib(ib_id)%angular_vel(2)*r(3) &
                                       & - patch_ib(ib_id)%angular_vel(3)*r(2)
                                    vb(2) = patch_ib(ib_id)%vel(2) + patch_ib(ib_id)%angular_vel(3)*r(1) &
                                       & - patch_ib(ib_id)%angular_vel(1)*r(3)
                                    vb(3) = patch_ib(ib_id)%vel(3) + patch_ib(ib_id)%angular_vel(1)*r(2) &
                                       & - patch_ib(ib_id)%angular_vel(2)*r(1)
                                    do i = 1, num_dims
                                        f_mom = flux_vf(eqn_idx%mom%beg + i - 1)%sf(j, k, l)
                                        if (has_src) f_mom = f_mom + flux_src_vf(eqn_idx%mom%beg + i - 1)%sf(j, k, l)
                                        f_E = f_E - vb(i)*f_mom
                                    end do
                                    f_E = f_E + 0.5_wp*(vb(1)**2 + vb(2)**2 + vb(3)**2)*f_m
                                end if
                                g = patch_ib(ib_id)%gbl_patch_id
                                $:GPU_ATOMIC(atomic='update')
                                ib_heat(2, g) = ib_heat(2, g) + real(side, wp)*w*area*f_E
                                $:GPU_ATOMIC(atomic='update')
                                ib_heat(3, g) = ib_heat(3, g) - real(side, wp)*w*area*f_m
                            end if
                        end if
                    end if
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

    end subroutine s_accumulate_ib_face_fluxes

    !> Set each IB's force and torque: the fluid's share from the face fluxes of the last RHS evaluation
    !! (s_accumulate_ib_face_fluxes), plus collisions and body forces
    subroutine s_compute_ib_forces()

        integer                         :: i, l
        real(wp), dimension(num_ibs, 3) :: forces, torques

        call nvtxStartRange("COMPUTE-IB-FORCES")

        ! no kernel over zero-size mapped arrays: a rank can hold no patch after a handoff
        if (num_ibs > 0) then
            $:GPU_PARALLEL_LOOP(private='[i, l]', copyout='[forces, torques]')
            do i = 1, num_ibs
                do l = 1, 3
                    forces(i, l) = ib_face_ft(l, i)
                    torques(i, l) = ib_face_ft(l + 3, i)
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()
        end if

        call nvtxStartRange("APPLY-COLLISION-FORCES")
        call s_apply_collision_forces(ghost_points, num_gps, ib_markers, forces, torques)
        call nvtxEndRange

        ! reduce the forces across local neighborhood ranks
        call nvtxStartRange("COMMUNICATE-IB-FORCES")
        call s_communicate_ib_forces(forces, torques)
        call nvtxEndRange

        ! consider body forces after reducing to avoid double counting; fixed IBs have no mass input
        do i = 1, num_ibs
            if (patch_ib(i)%moving_ibm == 0) cycle
            if (bf_x) then
                forces(i, 1) = forces(i, 1) + accel_bf(1)*patch_ib(i)%mass
            end if
            if (bf_y) then
                forces(i, 2) = forces(i, 2) + accel_bf(2)*patch_ib(i)%mass
            end if
            if (bf_z) then
                forces(i, 3) = forces(i, 3) + accel_bf(3)*patch_ib(i)%mass
            end if
        end do

        ! apply the summed forces
        if (num_ibs > 0) then
            $:GPU_PARALLEL_LOOP(private='[i, l]', copyin='[forces, torques]')
            do i = 1, num_ibs
                do l = 1, 3
                    patch_ib(i)%force(l) = forces(i, l)
                    patch_ib(i)%torque(l) = torques(i, l)
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()
        end if

        call nvtxEndRange

    end subroutine s_compute_ib_forces

    !> Computes the center of mass for IB patch types where we are unable to determine their center of mass analytically.
    !> These patches include things like NACA airfoils and STL models
    !> Move a moving airfoil/STL patch's centroid to its centre of mass, keeping the difference in centroid_offset. Collective:
    !! every rank calls it for every global id, contributing zeros for patches it does not hold.
    subroutine s_compute_centroid_offset(gid, ib_marker)

        integer, intent(in)      :: gid        !< global patch id
        integer, intent(in)      :: ib_marker  !< local index on this rank; <= 0 if not held
        integer                  :: i, j, k, num_cells_local, decoded_gbl_id
        integer(kind=8)          :: num_cells, needs_loc, needs_glb
        real(wp), dimension(1:3) :: center_of_mass, center_of_mass_local

        needs_loc = 0_8
        if (ib_marker > 0) then
            if (f_needs_centroid_offset(patch_ib(ib_marker))) needs_loc = 1_8
        end if
        call s_mpi_allreduce_integer_sum(needs_loc, needs_glb)
        if (needs_glb == 0_8) then
            if (ib_marker > 0) patch_ib(ib_marker)%centroid_offset(:) = 0._wp
            return
        end if

        center_of_mass_local = [0._wp, 0._wp, 0._wp]
        num_cells_local = 0
        if (ib_marker > 0) then
            ! get the summed mass distribution and number of cells to divide by
            do i = 0, m
                do j = 0, n
                    do k = 0, p
                        if (ib_markers%sf(i, j, k) /= 0) then
                            call s_decode_patch_periodicity(ib_markers%sf(i, j, k), decoded_gbl_id)
                            if (decoded_gbl_id == gid) then
                                num_cells_local = num_cells_local + 1
                                center_of_mass_local = center_of_mass_local + [x_cc(i), y_cc(j), 0._wp]
                                if (num_dims == 3) center_of_mass_local(3) = center_of_mass_local(3) + z_cc(k)
                            end if
                        end if
                    end do
                end do
            end do
        end if

        ! reduce the mass contribution over all MPI ranks and compute COM
        call s_mpi_allreduce_integer_sum(int(num_cells_local, 8), num_cells)
        call s_mpi_allreduce_sum(center_of_mass_local(1), center_of_mass(1))
        call s_mpi_allreduce_sum(center_of_mass_local(2), center_of_mass(2))
        call s_mpi_allreduce_sum(center_of_mass_local(3), center_of_mass(3))
        if (ib_marker <= 0) return
        if (num_cells == 0) then
            patch_ib(ib_marker)%centroid_offset = [0._wp, 0._wp, 0._wp]
            return
        end if
        center_of_mass = center_of_mass/real(num_cells, wp)

        ! assign the centroid offset as a vector pointing from the true COM to the "centroid" in the input file and replace the
        ! current centroid
        patch_ib(ib_marker)%centroid_offset = [patch_ib(ib_marker)%x_centroid, patch_ib(ib_marker)%y_centroid, &
                 & patch_ib(ib_marker)%z_centroid] - center_of_mass
        patch_ib(ib_marker)%x_centroid = center_of_mass(1)
        patch_ib(ib_marker)%y_centroid = center_of_mass(2)
        patch_ib(ib_marker)%z_centroid = center_of_mass(3)

        ! rotate the centroid offset back into the local coords of the IB
        patch_ib(ib_marker)%centroid_offset = matmul(patch_ib(ib_marker)%rotation_matrix_inverse, &
                 & patch_ib(ib_marker)%centroid_offset)

    end subroutine s_compute_centroid_offset

    !> A moving airfoil or STL (geometries 4, 5, 11, 12) moves its centroid to the centre of mass and keeps a centroid_offset
    pure logical function f_needs_centroid_offset(patch)

        type(ib_patch_parameters), intent(in) :: patch

        f_needs_centroid_offset = patch%moving_ibm /= 0 .and. any(patch%geometry == [4, 5, 11, 12])

    end function f_needs_centroid_offset

    !> On restart, replace the re-measured centroid offsets with the ones the run was using (restart_data/ib_offset_<step>.dat,
    !! written with each checkpoint), and re-place kinematics-driven bodies about them. No file: the measured offsets stand.
    impure subroutine s_restore_centroid_offsets(t_init)

        real(wp), intent(in)                 :: t_init
        character(len=path_len + 2*name_len) :: file_loc
        logical                              :: file_exist
        integer                              :: gid, i, ios, file_unit, step
        real(wp), dimension(3)               :: off

        step = t_step_start
        if (cfl_dt) step = n_start
        if (step == 0) return
        write (file_loc, '(A,I0,A)') trim(case_dir) // '/restart_data/ib_offset_', step, '.dat'
        inquire (file=trim(file_loc), exist=file_exist)
        if (.not. file_exist) return
        open (newunit=file_unit, file=trim(file_loc), status='old', action='read', iostat=ios)
        if (ios /= 0) return
        do
            read (file_unit, *, iostat=ios) gid, off
            if (ios /= 0) exit
            call s_get_neighborhood_idx(gid, i)
            if (i > 0) then
                patch_ib(i)%centroid_offset = off
                if (patch_ib(i)%moving_ibm /= 0 .and. patch_ib(i)%kin_model > 0) call s_prescribed_kinematics(i, t_init)
            end if
        end do
        close (file_unit)

    end subroutine s_restore_centroid_offsets

    !> Computes the moment of inertia for an immersed boundary
    subroutine s_compute_moment_of_inertia(patch, axis, moment)

        $:GPU_ROUTINE(parallelism='[seq]')

        type(ib_patch_parameters), intent(in) :: patch
        real(wp), dimension(3), intent(in)    :: axis
        real(wp), intent(out)                 :: moment
        real(wp)                              :: distance_to_axis, cell_volume
        real(wp), dimension(3)                :: position, closest_point_along_axis, vector_to_axis, normal_axis, centroid
        integer                               :: i, j, k, count, ib_marker, gbl_id, xp, yp, zp

        ! if the IB is in 2D or a 3D sphere, we can compute this exactly
        if (patch%geometry == 2) then  ! circle
            moment = 0.5_wp*patch%mass*(patch%radius)**2
        else if (patch%geometry == 3) then  ! rectangle
            moment = patch%mass*(patch%length_x**2 + patch %length_y**2)/6._wp
        else if (patch%geometry == 6) then  ! ellipse
            moment = 0.0625_wp*patch%mass*(patch%length_x**2 + patch %length_y**2)
        else if (patch%geometry == 8) then  ! sphere
            moment = 0.4*patch%mass*(patch%radius)**2
        else  ! we do not have an analytic moment of inertia calculation and need to approximate it directly via a sum
            count = 0
            cell_volume = (x_cc(1) - x_cc(0))*(y_cc(1) - y_cc(0))
            ! computed without grid stretching. Update in the loop to perform with stretching
            if (p /= 0) then
                cell_volume = cell_volume*(z_cc(1) - z_cc(0))
            end if

            if (p == 0) then
                normal_axis = [0, 0, 1]
            else if (sqrt(sum(axis**2)) < sgm_eps) then
                ! if the object is not actually rotating at this time, return a dummy value and exit
                moment = 1._wp
                return
            else
                normal_axis = axis/sqrt(sum(axis**2))
            end if

            moment = 0._wp

            do i = 0, m
                do j = 0, n
                    do k = 0, p
                        ib_marker = ib_markers%sf(i, j, k)
                        call s_decode_patch_periodicity(ib_marker, gbl_id, xp, yp, zp)
                        if (ib_marker /= 0 .and. gbl_id == patch%gbl_patch_id) then
                            count = count + 1  ! increment the count of total cells in the boundary

                            ! centroid of the periodic image this cell belongs to
                            centroid = [patch%x_centroid, patch%y_centroid, patch%z_centroid] + real([xp, yp, zp], &
                                & wp)*(glb_bounds(:)%end - glb_bounds(:)%beg)

                            ! get the position in local coordinates so that the axis passes through 0, 0, 0
                            if (num_dims < 3) then
                                position = [x_cc(i) - centroid(1), y_cc(j) - centroid(2), 0._wp]
                            else
                                position = [x_cc(i), y_cc(j), z_cc(k)] - centroid
                            end if

                            ! project the position along the axis to find the closest distance to the rotation axis
                            closest_point_along_axis = normal_axis*dot_product(normal_axis, position)
                            vector_to_axis = position - closest_point_along_axis
                            distance_to_axis = dot_product(vector_to_axis, vector_to_axis)  ! saves the distance to the axis squared

                            ! compute the position component of the moment
                            moment = moment + distance_to_axis
                        end if
                    end do
                end do
            end do

            ! write the final moment assuming the points are all uniform density; a rank holding no cell of the body gets the
            ! dummy value instead of 0/0
            if (count > 0) then
                moment = moment*patch%mass/(count*cell_volume)
            else
                moment = 1._wp
            end if
        end if

    end subroutine s_compute_moment_of_inertia

    !> Wrap immersed boundary positions across periodic domain boundaries
    subroutine s_wrap_periodic_ibs()

        integer :: patch_id

        $:GPU_PARALLEL_LOOP(private='[patch_id]')
        do patch_id = 1, num_ibs
            ! check domain wraps in x, y,
            #:for X, DIR in [('x', 1), ('y', 2), ('z', 3)]
                if (num_dims >= ${DIR}$) then
                    ! check for periodicity
                    if (ib_bc_${X}$%beg == BC_PERIODIC) then
                        ! check if the boundary has left the domain, and then correct
                        if (patch_ib(patch_id)%${X}$_centroid < glb_bounds(${DIR}$)%beg) then
                            ! if the boundary exited "left", wrap it back around to the "right"
                            patch_ib(patch_id)%${X}$_centroid = patch_ib(patch_id)%${X}$_centroid + (glb_bounds(${DIR}$)%end &
                                     & - glb_bounds(${DIR}$)%beg)
                        else if (patch_ib(patch_id)%${X}$_centroid > glb_bounds(${DIR}$)%end) then
                            ! if the boundary exited "right", wrap it back around to the "left"
                            patch_ib(patch_id)%${X}$_centroid = patch_ib(patch_id)%${X}$_centroid - (glb_bounds(${DIR}$)%end &
                                     & - glb_bounds(${DIR}$)%beg)
                        end if
                    end if
                end if
            #:endfor
        end do
        $:END_GPU_PARALLEL_LOOP()

    end subroutine s_wrap_periodic_ibs

    !> Builds the ascending list of distinct neighborhood ranks used by s_communicate_ib_forces and allocates its buffers once. The
    !! ascending order is the fixed order in which an owner adds its neighbors' contributions.
    impure subroutine s_setup_ib_force_comm()

#ifdef MFC_MPI
        integer                            :: i, j, r, max_recs
        integer, allocatable, dimension(:) :: flat

        if (num_procs == 1 .or. allocated(ib_nbrs)) return
        @:PROHIBIT(num_gbl_ibs >= min(2_8**digits(0._wp), int(huge(0), 8)), &
                   & "Too many IBs to encode their ids exactly in real(wp) force messages")

        ! a rank can fill several table slots (periodicity, few ranks) or be its own neighbor; keep each distinct rank once
        flat = reshape(ib_neighbor_ranks, [size(ib_neighbor_ranks)])
        allocate (ib_nbrs(size(flat)))
        n_ib_nbrs = 0
        do i = 1, size(flat)
            r = flat(i)
            if (r < 0 .or. r == proc_rank) cycle
            if (any(ib_nbrs(1:n_ib_nbrs) == r)) cycle
            ! insert in ascending order
            j = n_ib_nbrs
            do while (j > 0)
                if (ib_nbrs(j) < r) exit
                ib_nbrs(j + 1) = ib_nbrs(j)
                j = j - 1
            end do
            ib_nbrs(j + 1) = r
            n_ib_nbrs = n_ib_nbrs + 1
        end do

        ! an owner never receives more records than it owns, and a broadcast never carries more than its sender owns
        max_recs = min(size(patch_ib), num_local_ibs_max)
        allocate (ib_send_counts(n_ib_nbrs), ib_reqs(2*n_ib_nbrs), ib_stats(MPI_STATUS_SIZE, 2*n_ib_nbrs))
        allocate (ib_send_bufs(ib_rec_len*max_recs, max(n_ib_nbrs, 1)), ib_recv_bufs(ib_rec_len*max_recs, max(n_ib_nbrs, 1)))
#endif

    end subroutine s_setup_ib_force_comm

    !> Position of rank in ib_nbrs
    function f_ib_nbr_slot(rank) result(slot)

        integer, intent(in) :: rank
        integer             :: slot

        do slot = 1, n_ib_nbrs
            if (ib_nbrs(slot) == rank) return
        end do
        call s_mpi_abort('IB owner is not a neighborhood rank')

    end function f_ib_nbr_slot

    !> Reduces forces and torques so every rank holding an IB ends with bit-identical values. Each IB has one owner, the rank whose
    !! subdomain holds its centroid (patch_ib%owner_rank, stamped at handoff). Phase 1: each rank sends its nonzero partials to the
    !! owner, which adds them to its own partial in ascending source-rank order once all have arrived. Phase 2: each owner sends its
    !! totals to every neighborhood rank, which overwrite their partials. One nonblocking message per neighbor per phase.
    subroutine s_communicate_ib_forces(forces, torques)

        real(wp), dimension(num_ibs, 3), intent(inout) :: forces, torques

#ifdef MFC_MPI
        integer :: i, j, s, r, slot, pos, nvals, ierr

        if (num_procs == 1 .or. n_ib_nbrs == 0) return

        ! Phase 1: send each partial to the owner of its IB
        do s = 1, n_ib_nbrs
            call MPI_IRECV(ib_recv_bufs(:,s), size(ib_recv_bufs, 1), mpi_p, ib_nbrs(s), 610, MPI_COMM_WORLD, ib_reqs(s), ierr)
        end do

        ib_send_counts = 0
        do i = 1, num_ibs
            if (patch_ib(i)%owner_rank == proc_rank) cycle
            if (all(forces(i,:) == 0._wp) .and. all(torques(i,:) == 0._wp)) cycle
            slot = f_ib_nbr_slot(patch_ib(i)%owner_rank)
            pos = ib_rec_len*ib_send_counts(slot)
            ib_send_bufs(pos + 1, slot) = real(patch_ib(i)%gbl_patch_id, wp)
            ib_send_bufs(pos + 2:pos + 4,slot) = forces(i,1:3)
            ib_send_bufs(pos + 5:pos + 7,slot) = torques(i,1:3)
            ib_send_counts(slot) = ib_send_counts(slot) + 1
        end do

        ! every neighbor gets a message, possibly empty, so every posted receive completes
        do s = 1, n_ib_nbrs
            call MPI_ISEND(ib_send_bufs(:,s), ib_rec_len*ib_send_counts(s), mpi_p, ib_nbrs(s), 610, MPI_COMM_WORLD, &
                           & ib_reqs(n_ib_nbrs + s), ierr)
        end do
        call MPI_WAITALL(2*n_ib_nbrs, ib_reqs, ib_stats, ierr)

        ! owner sum: own partial first, then each neighbor in ascending rank order, whatever order the messages arrived in
        do s = 1, n_ib_nbrs
            call MPI_GET_COUNT(ib_stats(:,s), mpi_p, nvals, ierr)
            do r = 0, nvals/ib_rec_len - 1
                pos = ib_rec_len*r
                call s_get_neighborhood_idx(nint(ib_recv_bufs(pos + 1, s)), j)
                @:ASSERT(j > 0, 'IB force contribution for an IB this rank does not hold')
                @:ASSERT(patch_ib(j)%owner_rank == proc_rank, 'IB force contribution for an IB this rank does not own')
                forces(j,1:3) = forces(j,1:3) + ib_recv_bufs(pos + 2:pos + 4,s)
                torques(j,1:3) = torques(j,1:3) + ib_recv_bufs(pos + 5:pos + 7,s)
            end do
        end do

        ! Phase 2: each owner sends its totals to every neighbor
        do s = 1, n_ib_nbrs
            call MPI_IRECV(ib_recv_bufs(:,s), size(ib_recv_bufs, 1), mpi_p, ib_nbrs(s), 611, MPI_COMM_WORLD, ib_reqs(s), ierr)
        end do

        do i = 1, num_local_ibs
            j = local_ib_patch_ids(i)
            pos = ib_rec_len*(i - 1)
            ib_send_bufs(pos + 1, 1) = real(patch_ib(j)%gbl_patch_id, wp)
            ib_send_bufs(pos + 2:pos + 4,1) = forces(j,1:3)
            ib_send_bufs(pos + 5:pos + 7,1) = torques(j,1:3)
        end do

        do s = 1, n_ib_nbrs
            call MPI_ISEND(ib_send_bufs(:,1), ib_rec_len*num_local_ibs, mpi_p, ib_nbrs(s), 611, MPI_COMM_WORLD, &
                           & ib_reqs(n_ib_nbrs + s), ierr)
        end do
        call MPI_WAITALL(2*n_ib_nbrs, ib_reqs, ib_stats, ierr)

        ! holders take the owner's bits
        do s = 1, n_ib_nbrs
            call MPI_GET_COUNT(ib_stats(:,s), mpi_p, nvals, ierr)
            do r = 0, nvals/ib_rec_len - 1
                pos = ib_rec_len*r
                call s_get_neighborhood_idx(nint(ib_recv_bufs(pos + 1, s)), j)
                if (j <= 0) cycle
                @:ASSERT(patch_ib(j)%owner_rank == ib_nbrs(s), 'IB force total from a rank that does not own the IB')
                forces(j,1:3) = ib_recv_bufs(pos + 2:pos + 4,s)
                torques(j,1:3) = ib_recv_bufs(pos + 5:pos + 7,s)
            end do
        end do
#endif

    end subroutine s_communicate_ib_forces

    !> @brief Swaps ownership of IBs and passes ownership of IBs to neighbor processors
    subroutine s_handoff_ib_ownership()

        integer                                         :: i, j, k, output_idx, local_output_idx
        integer                                         :: old_num_local_ibs, num_ibs_old, num_ibs_pre
        integer                                         :: new_count, recv_count
        integer                                         :: pack_pos, unpack_pos, buf_size, patch_bytes
        integer                                         :: send_neighbor, recv_neighbor, ierr
        integer                                         :: dx, dy, dz, tag, nbr_idx, nreqs
        real(wp), dimension(3)                          :: centroid
        logical                                         :: is_new
        type(ib_patch_parameters)                       :: tmp_patch
        integer, dimension(num_local_ibs_max)           :: local_ib_idx_old
        integer, dimension(num_ib_patches_max_namelist) :: old_to_new  ! old patch_ib slot -> slot after compaction
        ! (2R+1)^num_dims - 1 neighbors in the radius-R neighborhood; each gets its own recv buffer
        integer                       :: max_nbrs
        character(len=1), allocatable :: send_buf(:), recv_bufs(:,:)
        integer, allocatable          :: requests(:), recv_neighbor_list(:)

#ifdef MFC_MPI
        call nvtxStartRange("HANDOFF-IB-OWNERSHIP")

        if (num_procs > 1) then
            ! save a copy of the local IB's global indices to cross-reference for later.
            local_ib_idx_old = 0
            old_num_local_ibs = num_local_ibs
            do i = 1, num_local_ibs
                local_ib_idx_old(i) = patch_ib(local_ib_patch_ids(i))%gbl_patch_id
            end do

            ! Sync GPU-updated fields (angles, angular_vel, centroids) to host before
            ! compaction and MPI packing, which read from host memory.
            $:GPU_UPDATE(host='[patch_ib]')

            ! delete any particles that no longer need to be tracked and coalesce the array
            num_ibs_old = num_ibs
            output_idx = 0
            local_output_idx = 0
            do i = 1, num_ibs
                old_to_new(i) = -1
                centroid = [patch_ib(i)%x_centroid, patch_ib(i)%y_centroid, 0._wp]
                if (num_dims == 3) centroid(3) = patch_ib(i)%z_centroid

                ! delete if not in neighborhood
                if (f_neighborhood_ranks_own_location(centroid)) then
                    output_idx = output_idx + 1
                    old_to_new(i) = output_idx
                    if (i /= output_idx) then
                        patch_ib(output_idx) = patch_ib(i)
                    end if

                    ! check if in local domain
                    if (f_local_rank_owns_location(centroid, glb_bounds)) then
                        local_output_idx = local_output_idx + 1
                        @:PROHIBIT(local_output_idx > num_local_ibs_max, &
                                   & "Too many IBs on a single processor rank. Modify case file or increase limit of num_local_ibs_max to resolve.")
                        local_ib_patch_ids(local_output_idx) = output_idx
                        patch_ib(output_idx)%owner_rank = proc_rank
                    end if
                end if
            end do
            num_ibs = output_idx
            num_local_ibs = local_output_idx
            ! num_ibs shrinks here, so refresh it with patch_ib: s_get_neighborhood_idx bounds its search by it on the device
            $:GPU_UPDATE(device='[patch_ib, num_ibs]')
            call s_compact_ib_lookup(old_to_new, num_ibs_old)

            ! Broadcast newly-owned patches to all neighborhood neighbors
            patch_bytes = storage_size(tmp_patch)/8
            ! a rank can hand off at most every global patch, so size by num_gbl_ibs, not num_local_ibs_max
            buf_size = storage_size(0)/8 + patch_bytes*max(1, min(num_local_ibs_max, num_gbl_ibs))
            max_nbrs = (2*ib_neighborhood_radius + 1)**num_dims - 1
            allocate (send_buf(buf_size), recv_bufs(buf_size, max_nbrs), requests(2*max_nbrs), recv_neighbor_list(max_nbrs))

            ! Write placeholder count at position 0
            pack_pos = 0
            call MPI_PACK(0, 1, MPI_INTEGER, send_buf, buf_size, pack_pos, MPI_COMM_WORLD, ierr)

            ! pack new patches and count them
            new_count = 0
            do i = 1, num_local_ibs
                k = local_ib_patch_ids(i)
                is_new = .true.
                do j = 1, old_num_local_ibs
                    if (patch_ib(k)%gbl_patch_id == local_ib_idx_old(j)) then
                        is_new = .false.
                        exit
                    end if
                end do
                if (is_new) then
                    call MPI_PACK(patch_ib(k), patch_bytes, MPI_BYTE, send_buf, buf_size, pack_pos, MPI_COMM_WORLD, ierr)
                    new_count = new_count + 1
                end if
            end do

            ! Overwrite the placeholder with the real count
            pack_pos = 0
            call MPI_PACK(new_count, 1, MPI_INTEGER, send_buf, buf_size, pack_pos, MPI_COMM_WORLD, ierr)
            pack_pos = storage_size(0)/8 + new_count*patch_bytes

            ! Post all receives first, then sends
            nreqs = 0
            nbr_idx = 0
            do dz = merge(-ib_neighborhood_radius, 0, num_dims == 3), merge(ib_neighborhood_radius, 0, num_dims == 3)
                do dy = -ib_neighborhood_radius, ib_neighborhood_radius
                    do dx = -ib_neighborhood_radius, ib_neighborhood_radius
                        if (dx == 0 .and. dy == 0 .and. dz == 0) cycle
                        nbr_idx = nbr_idx + 1
                        ! one tag per offset in the (2R+1)^3 neighbourhood: a radix-3 encoding collides once R > 1
                        tag = 200 + ((dx + ib_neighborhood_radius)*(2*ib_neighborhood_radius + 1) + (dy + ib_neighborhood_radius)) &
                                     & *(2*ib_neighborhood_radius + 1) + (dz + ib_neighborhood_radius)
                        recv_neighbor = ib_neighbor_ranks(-dx, -dy, -dz)
                        recv_neighbor_list(nbr_idx) = MPI_PROC_NULL
                        if (recv_neighbor < 0) cycle
                        recv_neighbor_list(nbr_idx) = recv_neighbor
                        nreqs = nreqs + 1
                        call MPI_IRECV(recv_bufs(:,nbr_idx), buf_size, MPI_PACKED, recv_neighbor, tag, MPI_COMM_WORLD, &
                                       & requests(nreqs), ierr)
                    end do
                end do
            end do

            do dz = merge(-ib_neighborhood_radius, 0, num_dims == 3), merge(ib_neighborhood_radius, 0, num_dims == 3)
                do dy = -ib_neighborhood_radius, ib_neighborhood_radius
                    do dx = -ib_neighborhood_radius, ib_neighborhood_radius
                        if (dx == 0 .and. dy == 0 .and. dz == 0) cycle
                        ! one tag per offset in the (2R+1)^3 neighbourhood: a radix-3 encoding collides once R > 1
                        tag = 200 + ((dx + ib_neighborhood_radius)*(2*ib_neighborhood_radius + 1) + (dy + ib_neighborhood_radius)) &
                                     & *(2*ib_neighborhood_radius + 1) + (dz + ib_neighborhood_radius)
                        send_neighbor = ib_neighbor_ranks(dx, dy, dz)
                        if (send_neighbor < 0) cycle
                        nreqs = nreqs + 1
                        call MPI_ISEND(send_buf, pack_pos, MPI_PACKED, send_neighbor, tag, MPI_COMM_WORLD, requests(nreqs), ierr)
                    end do
                end do
            end do

            call MPI_WAITALL(nreqs, requests, MPI_STATUSES_IGNORE, ierr)

            ! Unpack all received buffers
            num_ibs_pre = num_ibs
            do nbr_idx = 1, max_nbrs
                if (recv_neighbor_list(nbr_idx) == MPI_PROC_NULL) cycle
                unpack_pos = 0
                call MPI_UNPACK(recv_bufs(:,nbr_idx), buf_size, unpack_pos, recv_count, 1, MPI_INTEGER, MPI_COMM_WORLD, ierr)
                do i = 1, recv_count
                    call MPI_UNPACK(recv_bufs(:,nbr_idx), buf_size, unpack_pos, tmp_patch, patch_bytes, MPI_BYTE, MPI_COMM_WORLD, &
                                    & ierr)
                    call s_get_neighborhood_idx(tmp_patch%gbl_patch_id, j, num_ibs_pre)
                    ! A rank filling several neighbor slots (two ranks across a periodic direction) receives the patch
                    ! once per slot. s_merge_ib_lookup only runs after this loop, so the lookup cannot see arrivals from
                    ! earlier iterations; scan them directly to drop the copies after the first.
                    if (j < 0) then
                        do k = num_ibs_pre + 1, num_ibs
                            if (patch_ib(k)%gbl_patch_id == tmp_patch%gbl_patch_id) then
                                j = k
                                exit
                            end if
                        end do
                    end if
                    if (j < 0) then
                        num_ibs = num_ibs + 1
                        @:ASSERT(num_ibs <= size(patch_ib), 'patch_ib overflow in neighborhood handoff')
                        patch_ib(num_ibs) = tmp_patch
                    else
                        ! an IB we already hold changed owner: record the new owner, keep our own copy of its state
                        patch_ib(j)%owner_rank = tmp_patch%owner_rank
                    end if
                end do
            end do

            deallocate (send_buf, recv_bufs, requests, recv_neighbor_list)
            $:GPU_UPDATE(device='[patch_ib, num_ibs]')
            call s_merge_ib_lookup(num_ibs_pre)
        end if

        call nvtxEndRange()
#endif

    end subroutine s_handoff_ib_ownership

    !> Abort if any namelist immersed boundary marks no cell anywhere. The owning rank is chosen by the patch centroid, so an STL
    !! placed with model_translate while its centroid is left elsewhere is handed to ranks that do not hold it, and it vanishes
    !! silently. Cloud particles are skipped: a particle's centroid is its body, so it cannot vanish this way.
    impure subroutine s_check_every_patch_marked()

        integer(kind=8), allocatable :: cnt_loc(:), cnt_glb(:)
        integer                      :: gid, i, j, k, num_checked

        num_checked = num_gbl_ibs - sum(particle_cloud(1:num_particle_clouds)%num_particles)
        if (num_checked <= 0) return
        allocate (cnt_loc(num_checked), cnt_glb(num_checked))
        cnt_loc = 0_8
        do k = 0, p
            do j = 0, n
                do i = 0, m
                    if (ib_markers%sf(i, j, k) /= 0) then
                        call s_decode_patch_periodicity(ib_markers%sf(i, j, k), gid)
                        if (gid <= num_checked) cnt_loc(gid) = cnt_loc(gid) + 1_8
                    end if
                end do
            end do
        end do
        call s_mpi_allreduce_integer_sum_vec(cnt_loc, cnt_glb)
        @:PROHIBIT(any(cnt_glb == 0_8), &
                   & "An immersed boundary marked no cell anywhere: its centroid decides which rank owns it, so a body placed " &
                   & // "elsewhere with model_translate is handed to a rank that does not hold it. Set patch_ib%x/y/z_centroid " &
                   & // "to where the body actually is.")
        deallocate (cnt_loc, cnt_glb)

    end subroutine s_check_every_patch_marked

    !> Finalize the IBM module
    impure subroutine s_finalize_ibm_module()

        integer :: i

        @:DEALLOCATE(ib_markers%sf)
        @:DEALLOCATE(corrected_gps%sf)
        if (ib_exchange .and. moving_immersed_boundary_flag) then
            @:DEALLOCATE(ib_markers_prev%sf)
        end if
        if (ib_cutvol) then
            @:DEALLOCATE(ib_alpha%sf, ib_alpha_prev%sf, ib_cv_f%sf, ib_markers_cv%sf, ib_wall_vel)
        end if
        @:DEALLOCATE(ib_face_ft)
        do i = 1, num_ib_airfoils_max
            if (allocated(ib_airfoil_grids(i)%upper)) then
                @:DEALLOCATE(ib_airfoil_grids(i)%upper)
                @:DEALLOCATE(ib_airfoil_grids(i)%lower)
            end if
        end do
        if (allocated(models)) then
            @:DEALLOCATE(models)
        end if
        if (allocated(ghost_points)) then
            @:DEALLOCATE(ghost_points)
            @:DEALLOCATE(gp_prim_new)
        end if
        if (allocated(gp_surf)) then
            @:DEALLOCATE(gp_surf)
        end if
        if (allocated(ib_heat)) then
            @:DEALLOCATE(ib_heat)
        end if
        if (allocated(ib_mass_res)) then
            @:DEALLOCATE(ib_mass_res, ib_shell_mass)
            @:DEALLOCATE(ib_shell%sf)
        end if
        if (collision_model > 0) call s_finalize_collisions_module()
#ifdef MFC_MPI
        if (allocated(ib_nbrs)) deallocate (ib_nbrs, ib_send_counts, ib_reqs, ib_stats, ib_send_bufs, ib_recv_bufs)
#endif

    end subroutine s_finalize_ibm_module

    !> Species flux residual at a heterogeneous reacting surface.
    subroutine s_surface_species_residual(pres, T_s, d, Ys_IP, Ys_s, W_species, R_species, omega_s, mdot_s)

        $:GPU_ROUTINE(parallelism='[seq]')

        real(wp), intent(in)  :: pres, T_s, d
        real(wp), intent(in)  :: Ys_IP(num_species), Ys_s(num_species), W_species(num_species)
        real(wp), intent(out) :: R_species(num_species), omega_s(num_species), mdot_s
        real(wp)              :: mw_IP, mw_s, rho_s, sum_BG
        real(wp)              :: Xs_IP(num_species), Xs_s(num_species)
        real(wp)              :: D_s(num_species), B_s(num_species), G_s(num_species)
        integer               :: k

        call get_mixture_molecular_weight(Ys_IP, mw_IP)
        call get_mixture_molecular_weight(Ys_s, mw_s)
        rho_s = pres*mw_s/(gas_constant*T_s)

        do k = 1, num_species
            Xs_IP(k) = Ys_IP(k)*mw_IP/W_species(k)
            Xs_s(k) = Ys_s(k)*mw_s/W_species(k)
        end do

        call get_species_mass_diffusivities_mixavg(pres, T_s, Ys_s, D_s)
        call get_surface_net_production_rates(rho_s, T_s, Ys_s, omega_s)

        mdot_s = 0._wp
        do k = 1, num_species
            mdot_s = mdot_s + W_species(k)*omega_s(k)
        end do

        sum_BG = 0._wp
        do k = 1, num_species
            B_s(k) = rho_s*D_s(k)*W_species(k)/mw_s
            G_s(k) = (Xs_IP(k) - Xs_s(k))/d
            sum_BG = sum_BG + B_s(k)*G_s(k)
        end do

        do k = 1, num_species
            R_species(k) = -B_s(k)*G_s(k) + Ys_s(k)*(sum_BG + mdot_s) - W_species(k)*omega_s(k)
        end do

    end subroutine s_surface_species_residual

    !> Surface energy residual: gas-side conduction balances heterogeneous reaction heat. Radiation and solid-side conduction are
    !! omitted.
    subroutine s_surface_energy_residual(pres, T_IP, T_s, d, wg, Ys_s, R_energy)

        $:GPU_ROUTINE(parallelism='[seq]')

        real(wp), intent(in)        :: pres, T_IP, T_s, d
        type(wall_grad), intent(in) :: wg
        real(wp), intent(in)        :: Ys_s(num_species)
        real(wp), intent(out)       :: R_energy
        real(wp)                    :: mw_s, rho_s, k_s, q_rxn

        call get_mixture_molecular_weight(Ys_s, mw_s)
        rho_s = pres*mw_s/(gas_constant*T_s)

        call get_mixture_thermal_conductivity_mixavg(T_s, Ys_s, k_s)
        call get_surface_reaction_heat_flux(rho_s, T_s, Ys_s, q_rxn)

        if (wg%quad) then
            R_energy = k_s*(wg%b*T_s - wg%a) - q_rxn
        else
            R_energy = k_s*(T_s - T_IP)/d - q_rxn
        end if

    end subroutine s_surface_energy_residual

    !> Assemble the Newton residual for Ns species, with temperature appended only when it is solved. k_bath selects which species
    !! balance the sum constraint displaces; see s_solve_surface for why it must be the most abundant one.
    subroutine s_surface_residual(pres, T_IP, T_s, d, wg, Ys_IP, Ys_s, W_species, k_bath, solve_temperature, flux_scale, &
                                  & energy_scale, R, R_species, omega_s, mdot_s)

        $:GPU_ROUTINE(parallelism='[seq]')
        type(wall_grad), intent(in) :: wg  !< dT/dn|_w = wg%a - wg%b*T_s where wg%quad, else the one-sided difference
        real(wp), intent(in)        :: pres, T_IP, T_s, d, flux_scale, energy_scale
        real(wp), intent(in)        :: Ys_IP(num_species), Ys_s(num_species), W_species(num_species)
        integer, intent(in)         :: k_bath
        logical, intent(in)         :: solve_temperature
        real(wp), intent(out)       :: R(num_species + 1), R_species(num_species), omega_s(num_species), mdot_s
        real(wp)                    :: R_energy
        integer                     :: k

        call s_surface_species_residual(pres, T_s, d, Ys_IP, Ys_s, W_species, R_species, omega_s, mdot_s)

        R = 0._wp
        do k = 1, num_species
            R(k) = R_species(k)/flux_scale
        end do
        R(k_bath) = sum(Ys_s) - 1._wp

        if (solve_temperature) then
            call s_surface_energy_residual(pres, T_IP, T_s, d, wg, Ys_s, R_energy)
            R(num_species + 1) = R_energy/energy_scale
        end if

    end subroutine s_surface_residual

    !> Largest temperature mirror weight, at most theta_max, that keeps the ghost consistent with every fluid cell it shares a face
    !! with. Conduction across that face is the two-point difference (T_n - T_g)/h, so the line through the ghost (a behind the
    !! wall) and the cell (b in front) must not reach the wall beyond T_s: T_g - T_s <= (a/b)(T_s - T_n) on the mirror's side, and
    !! T_g = T_s once the cell is past T_s. The image point sits about a cell out, so without this an under-resolved layer leaves
    !! the mirror set by the second gas cell and drives the first past the wall temperature. A resolved, convex profile puts the
    !! mirror inside the bound, so it leaves the mirror untouched there. A cell closer than a quarter of its spacing to the wall
    !! plane is left out: its temperature is within noise of T_s, and the bound, whose slope there is a/b, then switched theta
    !! between 0 and 1 from step to step and kept a steady conduction annulus from converging.
    function f_face_theta(q_prim_vf, markers, gp, theta_max, T_IP, T_s) result(theta)

        $:GPU_ROUTINE(parallelism='[seq]')

        type(scalar_field), dimension(sys_size), intent(in) :: q_prim_vf
        type(integer_field), intent(in) :: markers  !< ib_markers, passed so the device routine reads the caller's present copy
        type(ghost_point), intent(in) :: gp
        real(wp), intent(in) :: theta_max, T_IP, T_s
        real(wp) :: theta, a, b, h, T_n, mw_n, r
        real(wp), dimension(3) :: nrm
        real(wp), dimension(${NUM_SPECIES}$) :: Ys_n
        integer, dimension(3) :: c
        integer :: dir, sgn, q

        theta = theta_max
        if (f_approx_equal(T_s, T_IP)) return
        a = abs(real(gp%levelset, kind=wp))
        nrm = gp%levelset_norm/max(sqrt(sum(gp%levelset_norm**2)), sgm_eps)
        do dir = 1, num_dims
            do sgn = -1, 1, 2
                c = gp%loc
                c(dir) = c(dir) + sgn
                if (markers%sf(c(1), c(2), c(3)) /= 0) cycle
                select case (dir)
                case (1)
                    h = x_cc(c(1)) - x_cc(gp%loc(1))
                case (2)
                    h = y_cc(c(2)) - y_cc(gp%loc(2))
                case default
                    h = z_cc(c(3)) - z_cc(gp%loc(3))
                end select
                do q = 1, num_species
                    Ys_n(q) = q_prim_vf(eqn_idx%species%beg + q - 1)%sf(c(1), c(2), c(3))
                end do
                call get_mixture_molecular_weight(Ys_n, mw_n)
                T_n = q_prim_vf(eqn_idx%E)%sf(c(1), c(2), c(3))*mw_n*q_prim_vf(eqn_idx%adv%beg)%sf(c(1), c(2), &
                                & c(3))/(q_prim_vf(eqn_idx%cont%beg)%sf(c(1), c(2), c(3))*gas_constant)
                r = (T_s - T_n)/(T_s - T_IP)
                ! Wall distance of the neighbor from the ghost's tangent plane (it can come out <= 0 at an edge)
                b = h*nrm(dir) - a
                if (b < 0.25_wp*abs(h)) cycle
                if (r <= 0._wp) then
                    theta = 0._wp
                else if (b > 0._wp) then
                    theta = min(theta, (a/b)*r)
                end if
            end do
        end do

    end function f_face_theta

    !> Ghost state for an immersed surface, from the image-point state and the surface state.
    !!
    !! The natural closure is the linear mirror phi_g = 2*phi_s - phi_IP, which is what makes the midpoint of the ghost/image
    !! pair reproduce the surface value. Its -1 coefficient on phi_IP is unconditional, though, so it drives strictly positive
    !! quantities negative whenever the surface value sits far below the image-point value: a cold wall in hot gas gives a
    !! negative ghost temperature and hence a negative ghost density, and a species the surface consumes faster than half the
    !! free-stream value gives a negative mass fraction. Rather than clamp the result -- which would break sum(Y) = 1 and hide
    !! the excursion -- the ghost state is the largest convex blend of the mirror back toward the surface value that stays
    !! physical:
    !!
    !!     phi_g(theta) = phi_s + theta*(phi_s - phi_IP),   theta in [0, 1]
    !!
    !! theta = 1 is the full second-order mirror when the image-point value is sampled as far in front of the wall as the ghost
    !! point sits behind it; sampled at wall distance d instead, the linear extrapolation weight is |levelset|/d, which the
    !! caller passes as theta_max. theta = 0 is the first-order Dirichlet ghost phi_g = phi_s, which is
    !! Gibou et al. (JCP 176:205, 2002) Eq. 17 and is what they likewise fall back to where the linear form is ill-behaved.
    !! Each species and the temperature get their own theta, and the most abundant (bath) species absorbs the others'
    !! changes, so sum(Y_g) = sum(Y_s) exactly. A shared theta would let a radical the surface depletes (O and OH at a
    !! burning carbon wall) collapse every mirror to the first-order ghost, cutting the wall flux of CO and CO2 the gas
    !! receives by up to half. A shared theta is the fallback if the bath species would go negative. The cost of limiting
    !! is small: the mirror already delivers only first-order wall flux (Ezra et al., Int. J. Heat Mass Transfer, 2025),
    !! and a boundary closure one order below the interior scheme retains the interior convergence rate (Gustafsson,
    !! Math. Comp. 29:396, 1975).
    !!
    !! This bounds the ghost state; it does not make the boundary strictly conservative. Baskaya et al. (Computers & Fluids
    !! 270:106134, 2024) trace the same cold-wall, large-gradient regime in ablation to ghost-cell mass conservation error
    !! that surfaces as spurious blowing, and resolve it only by moving to a flux-based cut-cell boundary -- a different
    !! discretization from this one, not a tuning of it.
    subroutine s_blend_ghost_state(theta_max, theta_T_max, T_IP, T_s, Ys_IP, Ys_s, T_g, Ys_g, q_T_sf, wf)

        $:GPU_ROUTINE(parallelism='[seq]')

        !> Quadratic wall-normal fill of the ghost temperature where the fill reached tier 1. The ghost temperature is what the wall
        !! conduction and the near-wall energy flux see, so its order sets the order of the wall heat flux and hence of Nu. The
        !! thermodynamic window is imposed on the result rather than by limiting a mirror weight, so the quadratic acts wherever it
        !! stays admissible.
        type(scalar_field), intent(in) :: q_T_sf
        type(wall_fill), intent(in)    :: wf
        real(wp), intent(in)           :: theta_max    !< Mirror weight the geometry allows: ghost over sample wall distance
        real(wp), intent(in)           :: theta_T_max  !< Temperature's cap, from f_face_theta
        real(wp), intent(in)           :: T_IP, T_s
        real(wp), intent(in)           :: Ys_IP(num_species), Ys_s(num_species)
        real(wp), intent(out)          :: T_g, Ys_g(num_species)
        ! Stop short of the admissibility boundary rather than landing exactly on it, so that a mass fraction driven to the
        ! limit stays strictly positive instead of becoming a hard zero that roundoff can push negative.
        real(wp), parameter :: blend_safety = 0.9_wp
        real(wp)            :: theta_T, theta_Y
        integer             :: k, k_bath

        ! T_g only carries the conduction gradient across the wall faces (s_ibm_set_conduction_temperature); no property is
        ! evaluated at it, so it is not held inside the thermodynamic window. Capping it there (a 300 K wall in 1500 K gas
        ! allowed theta 0.075) removed a third to a half of the heat a cold particle takes up during ignition.
        theta_T = theta_T_max
        theta_T = max(theta_T, 0._wp)

        if (wf%tier == 1) then
            T_g = f_quad_fill(T_s, f_tsc_apply(q_T_sf, wf%ic1, wf%wd1), f_tsc_apply(q_T_sf, wf%ic2, wf%wd2), wf%lw, wf%llin)
            T_g = min(max(T_g, T_surface_min), T_surface_max)
        else
            T_g = T_s + theta_T*(T_s - T_IP)
        end if

        call s_pick_bath_species(Ys_s, k_bath)
        Ys_g(k_bath) = Ys_s(k_bath)
        do k = 1, num_species
            if (k == k_bath) cycle
            Ys_g(k) = Ys_s(k) + f_species_theta(theta_max, blend_safety, Ys_s(k), Ys_IP(k))*(Ys_s(k) - Ys_IP(k))
            Ys_g(k_bath) = Ys_g(k_bath) + Ys_s(k) - Ys_g(k)
        end do
        if (Ys_g(k_bath) >= 0._wp) return

        theta_Y = theta_max
        do k = 1, num_species
            theta_Y = min(theta_Y, f_species_theta(theta_max, blend_safety, Ys_s(k), Ys_IP(k)))
        end do
        do k = 1, num_species
            Ys_g(k) = Ys_s(k) + theta_Y*(Ys_s(k) - Ys_IP(k))
        end do

    end subroutine s_blend_ghost_state

    !> Largest mirror weight, at most theta_max, that keeps a mass fraction positive (short of zero by the safety factor).
    pure real(wp) function f_species_theta(theta_max, safety, Y_s, Y_IP) result(theta)
        $:GPU_ROUTINE(parallelism='[seq]')

        real(wp), intent(in) :: theta_max, safety, Y_s, Y_IP

        theta = theta_max
        if (Y_s < Y_IP) theta = min(theta, safety*Y_s/(Y_IP - Y_s))

    end function f_species_theta

    !> Index of the most abundant species, i.e. the balance the sum constraint displaces.
    !!
    !! Closing sum(Y) = 1 by dropping one species' flux balance is standard (Surface CHEMKIN), but the dropped species absorbs
    !! the roundoff of every other balance, so it must be the bath gas. Dropping a fixed index instead -- the last species in
    !! the mechanism -- lands that error on whatever the mechanism happens to list last, which for the reduced GRI mechanism
    !! shipped with the reacting-surface example is H2O2, a trace radical whose own surface balance then goes unenforced.
    !! Cantera's solveSP re-scans for the largest species each iteration (evalSurfLarge) for this reason.
    subroutine s_pick_bath_species(Ys_s, k_bath)

        $:GPU_ROUTINE(parallelism='[seq]')

        real(wp), intent(in) :: Ys_s(num_species)
        integer, intent(out) :: k_bath
        integer              :: k

        k_bath = 1
        do k = 2, num_species
            if (Ys_s(k) > Ys_s(k_bath)) k_bath = k
        end do

    end subroutine s_pick_bath_species

    !> Newton solve for a reacting surface. thermal_bc=0: zero-normal-gradient T; 1: prescribed T; 2: energy balance.
    subroutine s_solve_surface(pres, T_IP, T_wall, d, wg, Ys_IP, W_species, thermal_bc, Ys_s, T_s, mdot_s, converged)

        $:GPU_ROUTINE(parallelism='[seq]')

        real(wp), intent(in)        :: pres, T_IP, T_wall, d
        type(wall_grad), intent(in) :: wg  !< see s_surface_residual
        real(wp), intent(in)        :: Ys_IP(num_species), W_species(num_species)
        integer, intent(in)         :: thermal_bc
        real(wp), intent(out)       :: Ys_s(num_species), T_s, mdot_s
        logical, intent(out)        :: converged
        integer, parameter          :: max_iter = 30, max_backtrack = 20
        real(wp), parameter         :: fd_eps_Y = 1.e-7_wp, fd_eps_T = 1.e-6_wp
        ! The Jacobian is a forward difference, so the Newton update carries O(sqrt(epsilon)) noise and the iteration turns
        ! linear near the root. The residual itself is evaluated exactly, so a tighter tolerance is not unreachable, but it is
        ! reached by stalling rather than converging -- which the caller then reports as a failed surface. Cantera's solveSP
        ! runs the same 1e-7 forward-difference step against a 1e-4 relative tolerance; 1e-6 here stays well inside that.
        real(wp), parameter :: tol = 1.e-6_wp
        real(wp), parameter :: Y_tol = 100._wp*epsilon(1._wp)
        real(wp)            :: A(num_species + 1, num_species + 1), rhs(num_species + 1), delta(num_species + 1)
        real(wp)            :: R(num_species + 1), R_pert(num_species + 1), R_trial(num_species + 1)
        real(wp)            :: R_species(num_species), R_species_pert(num_species), R_species_trial(num_species)
        real(wp)            :: omega_s(num_species), omega_pert(num_species), omega_trial(num_species)
        real(wp)            :: Ys_pert(num_species), Ys_trial(num_species)
        real(wp)            :: mdot_pert, mdot_trial, T_pert, T_trial
        real(wp)            :: flux_scale, energy_scale, R_energy, dx, lambda, norm_R, norm_trial
        logical             :: solve_temperature, linear_success, accepted
        integer             :: nsolve, iter, j, iback, k_bath, k_bath_prev

        converged = .false.
        Ys_s = Ys_IP

        select case (thermal_bc)
        case (0)
            T_s = T_IP
            solve_temperature = .false.
        case (1)
            T_s = T_wall
            solve_temperature = .false.
        case (2)
            T_s = min(max(T_IP, T_surface_min), T_surface_max)
            solve_temperature = .true.
        case default
            T_s = T_IP
            omega_s = 0._wp
            mdot_s = 0._wp
            return
        end select

        nsolve = num_species + merge(1, 0, solve_temperature)

        call s_pick_bath_species(Ys_s, k_bath)

        call s_surface_species_residual(pres, T_s, d, Ys_IP, Ys_s, W_species, R_species, omega_s, mdot_s)
        flux_scale = max(maxval(abs(R_species)), 1.e-12_wp)
        energy_scale = 1._wp
        if (solve_temperature) then
            call s_surface_energy_residual(pres, T_IP, T_s, d, wg, Ys_s, R_energy)
            energy_scale = max(abs(R_energy), 1._wp)
        end if

        call s_surface_residual(pres, T_IP, T_s, d, wg, Ys_IP, Ys_s, W_species, k_bath, solve_temperature, flux_scale, &
                                & energy_scale, R, R_species, omega_s, mdot_s)
        norm_R = maxval(abs(R(1:nsolve)))
        if (norm_R < tol) then
            converged = .true.
            return
        end if

        do iter = 1, max_iter
            do j = 1, num_species
                Ys_pert = Ys_s
                T_pert = T_s
                dx = fd_eps_Y*max(abs(Ys_s(j)), 1._wp)
                if (Ys_s(j) + dx > 1._wp) dx = -dx
                Ys_pert(j) = Ys_pert(j) + dx

                call s_surface_residual(pres, T_IP, T_pert, d, wg, Ys_IP, Ys_pert, W_species, k_bath, solve_temperature, &
                                        & flux_scale, energy_scale, R_pert, R_species_pert, omega_pert, mdot_pert)
                A(1:nsolve,j) = (R_pert(1:nsolve) - R(1:nsolve))/dx
            end do

            if (solve_temperature) then
                Ys_pert = Ys_s
                dx = fd_eps_T*max(abs(T_s), 1._wp)
                T_pert = T_s + dx
                if (T_pert > T_surface_max) then
                    dx = -dx
                    T_pert = T_s + dx
                end if

                call s_surface_residual(pres, T_IP, T_pert, d, wg, Ys_IP, Ys_pert, W_species, k_bath, solve_temperature, &
                                        & flux_scale, energy_scale, R_pert, R_species_pert, omega_pert, mdot_pert)
                A(1:nsolve,nsolve) = (R_pert(1:nsolve) - R(1:nsolve))/dx
            end if

            rhs(1:nsolve) = -R(1:nsolve)
            call s_solve_surface_linear_system(A, rhs, delta, nsolve, linear_success)
            if (.not. linear_success) return

            lambda = 1._wp
            accepted = .false.
            do iback = 1, max_backtrack
                Ys_trial = Ys_s + lambda*delta(1:num_species)
                T_trial = T_s
                if (solve_temperature) T_trial = T_s + lambda*delta(nsolve)

                if (minval(Ys_trial) < -Y_tol .or. maxval(Ys_trial) > 1._wp + Y_tol .or. T_trial < T_surface_min &
                    & .or. T_trial > T_surface_max) then
                    lambda = 0.5_wp*lambda
                    cycle
                end if

                where (Ys_trial < 0._wp) Ys_trial = 0._wp
                where (Ys_trial > 1._wp) Ys_trial = 1._wp

                call s_surface_residual(pres, T_IP, T_trial, d, wg, Ys_IP, Ys_trial, W_species, k_bath, solve_temperature, &
                                        & flux_scale, energy_scale, R_trial, R_species_trial, omega_trial, mdot_trial)
                norm_trial = maxval(abs(R_trial(1:nsolve)))
                if (norm_trial < norm_R) then
                    accepted = .true.
                    exit
                end if
                lambda = 0.5_wp*lambda
            end do

            if (.not. accepted) return

            Ys_s = Ys_trial
            T_s = T_trial
            R = R_trial
            R_species = R_species_trial
            omega_s = omega_trial
            mdot_s = mdot_trial
            norm_R = norm_trial

            ! The composition has moved, so the species the sum constraint displaces may no longer be the most abundant one.
            ! Re-closing changes what R means, so the residual is rebuilt on the rare steps where the choice actually changes.
            k_bath_prev = k_bath
            call s_pick_bath_species(Ys_s, k_bath)
            if (k_bath /= k_bath_prev) then
                call s_surface_residual(pres, T_IP, T_s, d, wg, Ys_IP, Ys_s, W_species, k_bath, solve_temperature, flux_scale, &
                                        & energy_scale, R, R_species, omega_s, mdot_s)
                norm_R = maxval(abs(R(1:nsolve)))
            end if

            if (norm_R < tol) then
                converged = .true.
                return
            end if
        end do

    end subroutine s_solve_surface

    !> Small dense linear solve with partial pivoting for the local surface Newton system.
    subroutine s_solve_surface_linear_system(A, b, x, nsolve, success)

        $:GPU_ROUTINE(parallelism='[seq]')

        integer, intent(in)     :: nsolve
        real(wp), intent(inout) :: A(num_species + 1, num_species + 1)
        real(wp), intent(inout) :: b(num_species + 1)
        real(wp), intent(out)   :: x(num_species + 1)
        logical, intent(out)    :: success
        real(wp)                :: factor, pivot_value, tmp, row_tmp(num_species + 1)
        integer                 :: i, j, k, pivot

        success = .true.
        x = 0._wp

        do k = 1, nsolve - 1
            pivot = k
            pivot_value = abs(A(k, k))
            do i = k + 1, nsolve
                if (abs(A(i, k)) > pivot_value) then
                    pivot = i
                    pivot_value = abs(A(i, k))
                end if
            end do
            if (pivot_value <= epsilon(1._wp)) then
                success = .false.
                return
            end if

            if (pivot /= k) then
                row_tmp(1:nsolve) = A(k,1:nsolve)
                A(k,1:nsolve) = A(pivot,1:nsolve)
                A(pivot,1:nsolve) = row_tmp(1:nsolve)
                tmp = b(k)
                b(k) = b(pivot)
                b(pivot) = tmp
            end if

            do i = k + 1, nsolve
                factor = A(i, k)/A(k, k)
                A(i, k) = 0._wp
                do j = k + 1, nsolve
                    A(i, j) = A(i, j) - factor*A(k, j)
                end do
                b(i) = b(i) - factor*b(k)
            end do
        end do

        if (abs(A(nsolve, nsolve)) <= epsilon(1._wp)) then
            success = .false.
            return
        end if

        x(nsolve) = b(nsolve)/A(nsolve, nsolve)
        do i = nsolve - 1, 1, -1
            tmp = b(i)
            do j = i + 1, nsolve
                tmp = tmp - A(i, j)*x(j)
            end do
            if (abs(A(i, i)) <= epsilon(1._wp)) then
                success = .false.
                return
            end if
            x(i) = tmp/A(i, i)
        end do

    end subroutine s_solve_surface_linear_system

end module m_ibm
