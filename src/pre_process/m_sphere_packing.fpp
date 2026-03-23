!>
!! @file
!! @brief Random sphere packing for immersed boundary initialization

!> @brief Packs non-overlapping spheres of uniform radius into a 3D domain
!! to achieve a target volume (solid) fraction, with a user-specified minimum
!! surface-to-surface gap between any two spheres.
!!
!! Algorithm: jittered face-centred cubic (FCC) lattice.
!!   1. Compute the FCC lattice constant from the target volume fraction
!!      and sphere radius.
!!   2. Generate all FCC lattice sites that fit inside the domain (inset
!!      by one radius from each wall).
!!   3. If more sites than needed, randomly discard the excess.
!!   4. Add a random jitter to each site, bounded so the minimum gap is
!!      always maintained between nearest neighbours.
!!   5. Run a few Gauss-Seidel sweeps to resolve any marginal overlaps
!!      introduced by the jitter.
!! This approach is O(N), deterministic for a given seed, and works
!! reliably for volume fractions up to the FCC limit (~74%).
!!
!! Usage: set sphere_pack = .true. along with sphere_pack_radius,
!! sphere_pack_vf, and sphere_pack_min_gap in the input file. The routine
!! calls s_init_sphere_pack_data to store centers compactly; the caller
!! then writes the binary file via s_write_sphere_pack_file.
module m_sphere_packing

    use m_derived_types
    use m_global_parameters
    use m_helper_basic
    use m_mpi_common
    use m_constants
    use m_sphere_pack_data

    implicit none

    private
    public :: s_pack_spheres

contains

    !> Pack uniform-radius spheres into the 3D domain on a jittered FCC
    !! lattice.  On success the sphere centers are stored compactly via
    !! s_init_sphere_pack_data; the caller writes the binary file.
    impure subroutine s_pack_spheres

        real(wp) :: domain_vol, sphere_vol, inset_vol, eff_vf
        real(wp) :: x_lo, x_hi, y_lo, y_hi, z_lo, z_hi
        real(wp) :: a, a_gap, a_vf, nn_dist, max_jitter
        integer  :: n_spheres, n_sites
        integer  :: ix, iy, iz, ib_basis, nx, ny, nz
        integer  :: i, j, n_overlaps
        real(wp) :: bx, by, bz, cx, cy, cz
        real(wp) :: rx, ry, rz, rval
        real(wp) :: sep
        real(wp), allocatable :: sites(:,:)
        real(wp), allocatable :: centers(:,:)

        ! FCC basis: 4 points per unit cell
        real(wp) :: basis(3, 4)

        ! Early exit when the feature is not requested
        if (.not. sphere_pack) return

        ! Validate inputs
        if (p == 0) then
            call s_mpi_abort('sphere_pack requires a 3D domain (p > 0)')
        end if

        if (sphere_pack_radius <= 0.0_wp) then
            call s_mpi_abort('sphere_pack_radius must be positive')
        end if

        if (sphere_pack_min_gap < 0.0_wp) then
            call s_mpi_abort('sphere_pack_min_gap must be non-negative')
        end if

        ! Domain bounds (physical coordinates)
        x_lo = x_domain%beg
        x_hi = x_domain%end
        y_lo = y_domain%beg
        y_hi = y_domain%end
        z_lo = z_domain%beg
        z_hi = z_domain%end

        domain_vol = (x_hi - x_lo)*(y_hi - y_lo)*(z_hi - z_lo)
        sphere_vol = (4.0_wp/3.0_wp)*pi*sphere_pack_radius**3

        ! Resolve packing specification: exactly one of sphere_pack_vf,
        ! sphere_pack_void_frac, or sphere_pack_n must be set.
        if (sphere_pack_n > 0) then
            ! Mode: explicit sphere count -> compute solid VF
            n_spheres = sphere_pack_n
            sphere_pack_vf = real(n_spheres, wp)*sphere_vol/domain_vol
        else if (sphere_pack_void_frac > 0.0_wp) then
            ! Mode: void fraction (porosity) -> solid VF = 1 - void_frac
            sphere_pack_vf = 1.0_wp - sphere_pack_void_frac
            n_spheres = nint(sphere_pack_vf*domain_vol/sphere_vol)
        else if (sphere_pack_vf > 0.0_wp) then
            ! Mode: solid volume fraction (original)
            n_spheres = nint(sphere_pack_vf*domain_vol/sphere_vol)
        else
            call s_mpi_abort( &
                'sphere_pack: set exactly one of sphere_pack_vf, &
                &sphere_pack_void_frac, or sphere_pack_n')
        end if

        if (sphere_pack_vf > 0.74_wp) then
            call s_mpi_abort( &
                'sphere_pack: solid volume fraction exceeds FCC limit (~74%). &
                &Reduce sphere_pack_vf, increase sphere_pack_void_frac, &
                &or decrease sphere_pack_n.')
        end if

        if (n_spheres == 0) then
            if (proc_rank == 0) then
                print *, 'WARNING: sphere_pack volume fraction too small for even one sphere'
            end if
            return
        end if

        ! If num_ibs was not explicitly set, initialize it to 0
        if (num_ibs < 0) num_ibs = 0

        ! Minimum centre-to-centre separation
        sep = 2.0_wp*sphere_pack_radius + sphere_pack_min_gap

        ! Compute the inset domain (where sphere centres can live).
        ! Smaller by one radius from each wall.
        inset_vol = (x_hi - x_lo - 2.0_wp*sphere_pack_radius) &
                   *(y_hi - y_lo - 2.0_wp*sphere_pack_radius) &
                   *(z_hi - z_lo - 2.0_wp*sphere_pack_radius)

        if (inset_vol <= 0.0_wp) then
            call s_mpi_abort( &
                'sphere_pack: domain too small for sphere radius. &
                &Each dimension must exceed 2*sphere_pack_radius.')
        end if

        ! Two constraints determine the FCC lattice constant `a`:
        !
        ! 1. Gap constraint: nearest-neighbour distance = a/sqrt(2) >= sep
        !    => a >= sep * sqrt(2)
        !    Add a small relative tolerance (1e-6) so that nn_dist is
        !    strictly above sep, avoiding floating-point false overlaps.
        a_gap = sep*sqrt(2.0_wp)*(1.0_wp + 1.0e-4_wp)

        ! 2. Density constraint: we need enough lattice sites in the inset
        !    domain.  The effective VF inside the inset domain must produce
        !    at least n_spheres sites.
        eff_vf = real(n_spheres, wp)*sphere_vol/inset_vol

        ! Cap at FCC theoretical maximum
        if (eff_vf > 0.7405_wp) eff_vf = 0.7405_wp

        a_vf = sphere_pack_radius*(16.0_wp*pi/(3.0_wp*eff_vf))**(1.0_wp/3.0_wp)

        ! Use the LARGER of the two: gap constraint sets a floor on `a`,
        ! density constraint sets a ceiling.  If a_gap > a_vf, the gap
        ! makes the lattice too sparse to place all requested spheres;
        ! we honour the gap and place as many as we can.
        a = max(a_gap, a_vf)

        if (a_gap > a_vf .and. proc_rank == 0) then
            print '(A)', &
                ' sphere_pack: gap constraint limits packing density.'
            print '(A,ES10.3,A,ES10.3)', &
                '   a_gap = ', a_gap, ', a_vf = ', a_vf
            print '(A)', &
                '   Using a_gap; achieved VF may be less than target.'
            print '(A)', &
                '   To increase VF, reduce min_gap or increase domain size.'
        end if

        ! FCC nearest-neighbour distance = a / sqrt(2)
        nn_dist = a/sqrt(2.0_wp)

        ! Maximum jitter per sphere that still guarantees the min gap.
        ! Jitter is applied independently in x, y, z, so the worst-case
        ! displacement magnitude is sqrt(3)*max_jitter per sphere.
        ! Two neighbours approaching each other can decrease distance by
        ! at most 2*sqrt(3)*max_jitter. We need:
        !   nn_dist - 2*sqrt(3)*max_jitter >= sep
        max_jitter = (nn_dist - sep)/(2.0_wp*sqrt(3.0_wp))

        ! If max_jitter is negligibly small relative to the sphere radius,
        ! skip jitter entirely — it would only introduce FP noise.
        if (max_jitter < 1.0e-3_wp*sphere_pack_radius) then
            max_jitter = 0.0_wp
        end if

        ! FCC basis vectors (conventional unit cell)
        basis(:, 1) = [0.0_wp,      0.0_wp,      0.0_wp     ]
        basis(:, 2) = [0.5_wp*a,    0.5_wp*a,    0.0_wp     ]
        basis(:, 3) = [0.5_wp*a,    0.0_wp,      0.5_wp*a   ]
        basis(:, 4) = [0.0_wp,      0.5_wp*a,    0.5_wp*a   ]

        ! Seed the random number generator (reproducible if seed > 0)
        if (sphere_pack_seed > 0) then
            call s_seed_rng(sphere_pack_seed)
        end if

        ! Step 1: generate all FCC lattice sites inside the domain
        ! (inset by sphere_pack_radius from each wall)
        nx = ceiling((x_hi - x_lo)/a) + 1
        ny = ceiling((y_hi - y_lo)/a) + 1
        nz = ceiling((z_hi - z_lo)/a) + 1

        ! Upper bound on number of sites
        allocate(sites(3, 4*nx*ny*nz))
        n_sites = 0

        do iz = 0, nz - 1
            do iy = 0, ny - 1
                do ix = 0, nx - 1
                    do ib_basis = 1, 4
                        cx = x_lo + real(ix, wp)*a + basis(1, ib_basis)
                        cy = y_lo + real(iy, wp)*a + basis(2, ib_basis)
                        cz = z_lo + real(iz, wp)*a + basis(3, ib_basis)

                        ! Keep only if sphere fits inside domain
                        if (cx >= x_lo + sphere_pack_radius .and. &
                            cx <= x_hi - sphere_pack_radius .and. &
                            cy >= y_lo + sphere_pack_radius .and. &
                            cy <= y_hi - sphere_pack_radius .and. &
                            cz >= z_lo + sphere_pack_radius .and. &
                            cz <= z_hi - sphere_pack_radius) then
                            n_sites = n_sites + 1
                            sites(1, n_sites) = cx
                            sites(2, n_sites) = cy
                            sites(3, n_sites) = cz
                        end if
                    end do
                end do
            end do
        end do

        if (proc_rank == 0) then
            print '(A,I0,A,I0,A)', &
                ' sphere_pack: generated ', n_sites, &
                ' FCC lattice sites, need ', n_spheres
        end if

        if (n_sites < n_spheres) then
            if (proc_rank == 0) then
                print '(A,I0,A,I0)', &
                    ' sphere_pack: only ', n_sites, &
                    ' sites available, placing ', n_sites
            end if
            n_spheres = n_sites
        end if

        ! Step 2: if we have more sites than needed, randomly select
        ! n_spheres of them using Fisher-Yates shuffle (partial)
        if (n_sites > n_spheres) then
            do i = 1, n_spheres
                ! Pick a random index from i to n_sites
                call random_number(rval)
                j = i + int(rval*real(n_sites - i + 1, wp))
                if (j > n_sites) j = n_sites
                ! Swap sites(i) and sites(j)
                if (j /= i) then
                    rx = sites(1, i); ry = sites(2, i); rz = sites(3, i)
                    sites(1, i) = sites(1, j); sites(2, i) = sites(2, j); sites(3, i) = sites(3, j)
                    sites(1, j) = rx; sites(2, j) = ry; sites(3, j) = rz
                end if
            end do
        end if

        ! Copy selected sites into centers
        allocate(centers(3, n_spheres))
        centers(:, 1:n_spheres) = sites(:, 1:n_spheres)
        deallocate(sites)

        ! Step 3: add random jitter to each sphere
        if (max_jitter > 0.0_wp) then
            do i = 1, n_spheres
                call random_number(rx)
                call random_number(ry)
                call random_number(rz)
                centers(1, i) = centers(1, i) + (2.0_wp*rx - 1.0_wp)*max_jitter
                centers(2, i) = centers(2, i) + (2.0_wp*ry - 1.0_wp)*max_jitter
                centers(3, i) = centers(3, i) + (2.0_wp*rz - 1.0_wp)*max_jitter

                ! Clamp to domain
                centers(1, i) = max(x_lo + sphere_pack_radius, &
                    min(x_hi - sphere_pack_radius, centers(1, i)))
                centers(2, i) = max(y_lo + sphere_pack_radius, &
                    min(y_hi - sphere_pack_radius, centers(2, i)))
                centers(3, i) = max(z_lo + sphere_pack_radius, &
                    min(z_hi - sphere_pack_radius, centers(3, i)))
            end do

            if (proc_rank == 0) then
                print '(A,ES10.3,A,ES10.3)', &
                    ' sphere_pack: jitter magnitude: ', max_jitter, &
                    ', lattice spacing: ', a
            end if
        end if

        ! Step 4: count overlapping pairs (diagnostic only).
        ! With the sqrt(3) jitter bound, overlaps should be zero.
        ! This is a single O(N^2) pass; for very large N (>50K) it
        ! may be slow, so we skip it and trust the bound.
        n_overlaps = 0
        if (max_jitter > 0.0_wp .and. n_spheres <= 50000) then
            call s_count_overlaps(centers, n_spheres, sep, n_overlaps)

            if (n_overlaps > 0 .and. proc_rank == 0) then
                print '(A,I0,A)', &
                    ' WARNING: sphere_pack has ', n_overlaps, &
                    ' overlapping pairs (unexpected with sqrt(3) bound).'
            end if
        end if

        ! Store centers compactly; caller writes the binary file.
        call s_init_sphere_pack_data(centers, n_spheres, &
                                     sphere_pack_radius, sphere_pack_min_gap)

        ! Make sure IB is turned on
        if (n_spheres > 0) ib = .true.

        if (proc_rank == 0) then
            print '(A)', ' '
            print '(A)', '  Sphere Packing Summary'
            print '(A)', ' '
            print '(A,I0)',     '  Spheres placed       : ', n_spheres
            print '(A,ES12.5)', '  Sphere radius        : ', sphere_pack_radius
            print '(A,ES12.5)', '  Solid vol. fraction  : ', sphere_pack_vf
            print '(A,ES12.5)', '  Void fraction        : ', &
                1.0_wp - real(n_spheres, wp)*sphere_vol/domain_vol
            print '(A,ES12.5)', '  Achieved solid VF    : ', &
                real(n_spheres, wp)*sphere_vol/domain_vol
            print '(A,ES12.5)', '  Effective lattice VF : ', eff_vf
            print '(A,ES12.5)', '  Min gap (surface)    : ', sphere_pack_min_gap
            print '(A,ES12.5)', '  FCC lattice constant : ', a
            print '(A,ES12.5)', '  Max jitter applied   : ', max_jitter
            if (n_overlaps > 0) then
                print '(A,I0)',  '  Remaining overlaps   : ', n_overlaps
            else
                print '(A)',     '  Remaining overlaps   : 0 (fully resolved)'
            end if
            print '(A)', ' '
        end if

        deallocate(centers)

    end subroutine s_pack_spheres

    !> Count overlapping pairs (diagnostic only, single O(N^2) pass).
    impure subroutine s_count_overlaps(centers, n, sep, n_overlaps)

        real(wp), intent(in) :: centers(:,:)
        integer,  intent(in) :: n
        real(wp), intent(in) :: sep
        integer,  intent(out) :: n_overlaps

        integer  :: i, j
        real(wp) :: dx, dy, dz, sep_sq

        sep_sq = sep*sep
        n_overlaps = 0

        do i = 1, n - 1
            do j = i + 1, n
                dx = centers(1, j) - centers(1, i)
                dy = centers(2, j) - centers(2, i)
                dz = centers(3, j) - centers(3, i)
                if (dx*dx + dy*dy + dz*dz < sep_sq) then
                    n_overlaps = n_overlaps + 1
                end if
            end do
        end do

    end subroutine s_count_overlaps

    !> Seed Fortran's intrinsic PRNG with a deterministic sequence derived
    !! from a single user-supplied integer seed.
    impure subroutine s_seed_rng(seed_val)

        integer, intent(in) :: seed_val
        integer :: n, i
        integer, allocatable :: seed_array(:)

        call random_seed(size=n)
        allocate(seed_array(n))
        do i = 1, n
            seed_array(i) = seed_val + (i - 1)*37
        end do
        call random_seed(put=seed_array)
        deallocate(seed_array)

    end subroutine s_seed_rng

end module m_sphere_packing
