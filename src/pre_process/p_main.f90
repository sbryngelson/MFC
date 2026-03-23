!>
!! @file
!! @brief Contains program p_main

!> @brief This program takes care of setting up the initial condition and
!!              grid data for the multicomponent flow code.
program p_main

    use m_global_parameters     !< Global parameters for the code

    use m_start_up
    use m_sphere_packing
    use m_sphere_pack_data

    implicit none

    logical :: file_exists
    real(wp) :: start, finish, time_avg, time_final
    real(wp), allocatable, dimension(:) :: proc_time

    call random_seed()

    call s_initialize_mpi_domain()

    ! Initialization of the MPI environment

    call s_initialize_modules()

    ! Pack spheres before s_read_grid(), which overwrites x_domain/y_domain/
    ! z_domain with local (per-rank) subdomain bounds. Sphere packing needs
    ! the global domain bounds from the namelist.
    call s_pack_spheres()
    if (sphere_pack) call s_write_sphere_pack_file(case_dir)

    call s_read_grid()

    allocate (proc_time(0:num_procs - 1))

    call s_apply_initial_condition(start, finish)

    time_avg = abs(finish - start)

    call s_save_data(proc_time, time_avg, time_final, file_exists)

    deallocate (proc_time)

    call s_finalize_modules()

end program p_main
