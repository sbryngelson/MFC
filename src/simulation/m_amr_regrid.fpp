!>
!!@file
!!@brief Contains module m_amr_regrid

#:include 'macros.fpp'

!> @brief Dynamic regrid for the AMR level set: density-gradient tagging, clustering (m_amr_cluster), box shaping, hierarchical
!! child nesting, and slot rebuild with cross-rank fine-state migration.
module m_amr_regrid

#ifdef MFC_MPI
    use mpi
#endif

    use m_derived_types
    use m_global_parameters
    use m_constants, only: mapCells
    use m_mpi_proxy, only: s_mpi_abort
    use m_mpi_common, only: s_mpi_allreduce_min, s_mpi_allreduce_max
    use m_amr_wave
    use m_phase_timing
    use m_amr_cluster, only: s_amr_cluster
    use m_amr_distribution, only: s_amr_sort_by_key
    use m_amr, only: amr_slots, amr_cons_st, amr_stor_st, amr_loc_of, amr_slot_live, amr_my_blk, amr_n_my, &
        & s_amr_refresh_my_blocks, amr_maxc_fit, amr_seam_pairs_dirty, amr_mesh_epoch, amr_cpat_mar, s_amr_alloc_slot, &
        & s_amr_dev_mem, s_amr_free_slot, s_amr_reduce_xchg_flag, s_amr_reconcile_slots, s_amr_assign_block_owners, &
        & s_amr_exchange_coarse_cons_halo, s_amr_phys_to_cells, s_amr_body_bbox, s_amr_expand_box_over_bodies, s_amr_tile_box, &
        & f_amr_seam_dim, f_amr_boxes_overlap, s_set_amr_fine_geometry, s_interpolate_coarse_to_fine, s_amr_setup_ib, &
        & amr_cad_tot, amr_cad_esc, amr_cad_armed, s_amr_l1_fill_exchange, s_amr_l1_fill_consume, s_amr_parent_fill_exchange, &
        & s_amr_parent_fill_consume, s_amr_fill_wave_done, amr_fw_dev
    use m_acoustic_src, only: acoustic_supp_lo, acoustic_supp_hi
    use m_bubbles_EL, only: s_lag_cloud_bbox_local

    implicit none

    private
    public :: s_amr_regrid, s_amr_check_seam_topology

    !> Padded global coarse-index bbox of the Lagrangian bubble cloud; blocks and regrid boxes stay clear of it
    integer :: lag_supp_lo(3), lag_supp_hi(3)
    !> Nested children were dropped for lack of block capacity (warn once per regrid)
    logical :: amr_child_dropped = .false.
    logical :: lag_supp_on = .false.
    !> Migration send/receive pools (freed after the regrid); amr_mig_win(:,old block) = payload start in amr_mig_rq, fine window
    !! lo(3), hi(3) (start 0 = none)
    real(wp), allocatable :: amr_mig_sq(:), amr_mig_rq(:)
    integer, allocatable  :: amr_mig_win(:,:)

contains

    !> Abort on same-level seam topologies no halo reconciles: partial-transverse face adjacency or box intersection (both arise
    !! only from IB body-bbox expansion). Each rank tests its own blocks against all blocks.
    impure subroutine s_amr_check_seam_topology()

        integer :: ix, xb, yb, d, t
        logical :: adj, tover

        call s_amr_refresh_my_blocks()
        do ix = 1, amr_n_my
            xb = amr_my_blk(ix)
            do yb = 1, amr_num_blocks
                if (xb == yb) cycle
                if (amr_block_level(xb) /= amr_block_level(yb)) cycle
                if (f_amr_boxes_overlap(amr_region_lo_all(:,xb), amr_region_hi_all(:,xb), amr_region_lo_all(:,yb), &
                    & amr_region_hi_all(:,yb))) then
                    write (0, '(A,6I6,A,6I6)') ' [amr] intersecting blocks (lo,hi): ', amr_region_lo_all(:,xb), &
                           & amr_region_hi_all(:,xb), ' |', amr_region_lo_all(:,yb), amr_region_hi_all(:,yb)
                    call s_mpi_abort("AMR: two same-level blocks INTERSECT (the child IB body-bbox expansion route can " &
                                     & // "produce this - it has no overlap-merge pass): the overlapping cells would be " &
                                     & // "restricted and refluxed twice, silently breaking conservation. Adjust the body/" &
                                     & // "regrid inputs so body-expanded child boxes merge or separate.")
                end if
                do d = 1, num_dims
                    adj = amr_region_lo_all(d, yb) == amr_region_hi_all(d, xb) + 1
                    if (.not. adj) cycle
                    tover = .true.
                    do t = 1, num_dims
                        if (t /= d) tover = tover .and. amr_region_lo_all(t, xb) <= amr_region_hi_all(t, &
                            & yb) .and. amr_region_lo_all(t, yb) <= amr_region_hi_all(t, xb)
                    end do
                    if (tover .and. f_amr_seam_dim(xb, yb) == 0) then
                        write (0, '(A,6I6,A,6I6)') ' [amr] touching blocks (lo,hi): ', amr_region_lo_all(:,xb), &
                               & amr_region_hi_all(:,xb), ' |', amr_region_lo_all(:,yb), amr_region_hi_all(:,yb)
                        call s_mpi_abort("AMR: two same-level blocks touch with PARTIAL transverse overlap (IB body-bbox " &
                                         & // "expansion can produce this): the fine-fine seam halo only reconciles exact-match " &
                                         & // "faces, so the shared-face flux would silently leak. Adjust amr_block/regrid inputs " // "so body-expanded boxes merge or separate.")
                    end if
                end do
            end do
        end do

    end subroutine s_amr_check_seam_topology

    !> Abort if a box exceeds its level's slot cap, amr_maxc_fit/amr_ref_ratio**(lev-1) coarse cells (it would overrun the fine
    !! coord arrays); floor of 1 as in s_amr_tile_box.
    impure subroutine s_amr_check_box_caps(boxes, nboxes, box_level)

        type(t_box), intent(in) :: boxes(:)
        integer, intent(in)     :: nboxes, box_level(:)
        integer                 :: k, d, lev, cap, span

        do k = 1, nboxes
            lev = box_level(k)
            do d = 1, num_dims
                cap = max(amr_maxc_fit(d)/amr_ref_ratio**(lev - 1), 1)
                span = boxes(k)%hi(d) - boxes(k)%lo(d) + 1
                if (span > cap) then
                    if (proc_rank == 0) print '(A,I0,A,I0,A,I0,A,I0)', ' [amr] box cap violated: level ', lev, ' dim ', d, &
                        & ' span ', span, ' > cap ', cap
                    call s_mpi_abort("AMR regrid: a fine box exceeds the slot cap for its level (span and cap printed above). " &
                                     & // "The fine coord arrays are sized to amr_ref_ratio*amr_maxc_fit, so this would write " &
                                     & // "past x_cb and corrupt the heap. A box emitter did not route through s_amr_tile_box.")
                end if
            end do
        end do

    end subroutine s_amr_check_box_caps

    !> Abort if two same-level boxes overlap (the overlap carry-forward and exchange plans assume disjointness). All levels share
    !! the coarse index space; an x-lo sweep keeps it cheap.
    impure subroutine s_amr_check_box_disjoint(boxes, nboxes, box_level)

        type(t_box), intent(in) :: boxes(:)
        integer, intent(in)     :: nboxes, box_level(:)
        integer(8), allocatable :: key(:)
        integer, allocatable    :: ord(:)
        integer                 :: i, j, k, kk

        allocate (ord(nboxes)); key = [(int(boxes(k)%lo(1), 8), k=1, nboxes)]
        call s_amr_sort_by_key(key, nboxes, ord)
        do i = 1, nboxes
            k = ord(i)
            if (box_level(k) < 1) cycle
            do j = i + 1, nboxes
                kk = ord(j)
                if (boxes(kk)%lo(1) > boxes(k)%hi(1)) exit
                if (box_level(kk) /= box_level(k)) cycle
                if (all(boxes(k)%lo <= boxes(kk)%hi .and. boxes(kk)%lo <= boxes(k)%hi)) then
                    if (proc_rank == 0) print '(A,I0,A,I0,A,I0)', ' [amr] same-level box overlap: level ', box_level(k), &
                        & ' boxes ', k, ' and ', kk
                    call s_mpi_abort("AMR regrid: two same-level boxes overlap (indices printed above). The overlap " &
                                     & // "carry-forward and the exchange plans both assume same-level disjointness.")
                end if
            end do
        end do

    end subroutine s_amr_check_box_disjoint

    !> True iff global level-0 cell (gi, gj, gk) lies inside any acoustic source support bbox.
    pure logical function f_in_acoustic_support(gi, gj, gk) result(insup)

        integer, intent(in) :: gi, gj, gk
        integer             :: s

        insup = .false.
        do s = 1, num_source
            if (gi >= acoustic_supp_lo(1, s) .and. gi <= acoustic_supp_hi(1, s) .and. (n_glb == 0 .or. (gj >= acoustic_supp_lo(2, &
                & s) .and. gj <= acoustic_supp_hi(2, s))) .and. (p_glb == 0 .or. (gk >= acoustic_supp_lo(3, &
                & s) .and. gk <= acoustic_supp_hi(3, s)))) then
                insup = .true.; return
            end if
        end do

    end function f_in_acoustic_support

    !> Clip a box (global indices) clear of every acoustic source support; may empty it (hi < lo).
    impure subroutine s_amr_clip_box_from_sources(lo, hi)

        integer, intent(inout) :: lo(3), hi(3)
        integer                :: s

        do s = 1, num_source
            call s_amr_clip_box_from_supp(lo, hi, acoustic_supp_lo(:,s), acoustic_supp_hi(:,s))
        end do
        do s = 1, num_source
            if (hi(1) < lo(1) .or. hi(2) < lo(2) .or. hi(3) < lo(3)) return
            if (f_amr_boxes_overlap(lo, hi, acoustic_supp_lo(:,s), acoustic_supp_hi(:, &
                & s))) call s_mpi_abort('amr regrid: acoustic source exclusion clip failed (internal error)')
        end do

    end subroutine s_amr_clip_box_from_sources

    !> Collectively recompute the global bubble-cloud exclusion bbox, padded by pad_cells; lag_supp_on = false when no rank holds a
    !! bubble.
    impure subroutine s_amr_compute_lag_supp(pad_cells)

        integer, intent(in)    :: pad_cells
        real(wp), dimension(3) :: pmin_loc, pmax_loc, pmin_glb, pmax_glb
        integer                :: d

        call s_lag_cloud_bbox_local(pmin_loc, pmax_loc)
        do d = 1, 3
            call s_mpi_allreduce_min(pmin_loc(d), pmin_glb(d))
            call s_mpi_allreduce_max(pmax_loc(d), pmax_glb(d))
        end do
        lag_supp_on = pmin_glb(1) <= pmax_glb(1)
        if (.not. lag_supp_on) return
        call s_amr_phys_to_cells(pmin_glb, pmax_glb, pad_cells, lag_supp_lo, lag_supp_hi)

    end subroutine s_amr_compute_lag_supp

    !> True iff global level-0 cell (gi, gj, gk) lies inside the Lagrangian-cloud exclusion bbox.
    pure logical function f_in_lag_support(gi, gj, gk) result(insup)

        integer, intent(in) :: gi, gj, gk

        insup = .false.
        if (.not. lag_supp_on) return
        insup = gi >= lag_supp_lo(1) .and. gi <= lag_supp_hi(1) .and. (n_glb == 0 .or. (gj >= lag_supp_lo(2) &
                                  & .and. gj <= lag_supp_hi(2))) .and. (p_glb == 0 .or. (gk >= lag_supp_lo(3) &
                                  & .and. gk <= lag_supp_hi(3)))

    end function f_in_lag_support

    !> Clip a box (global indices) clear of one support bbox along the axis/side keeping the most extent (ties: lower axis, begin
    !! side); may empty it.
    pure subroutine s_amr_clip_box_from_supp(lo, hi, slo, shi)

        integer, intent(inout) :: lo(3), hi(3)
        integer, intent(in)    :: slo(3), shi(3)
        integer                :: d, best_d, best_side, best_ext, ext_l, ext_r

        if (hi(1) < lo(1) .or. hi(2) < lo(2) .or. hi(3) < lo(3)) return
        if (.not. f_amr_boxes_overlap(lo, hi, slo, shi)) return
        best_d = 1; best_side = 1; best_ext = -1
        do d = 1, num_dims
            ext_l = slo(d) - lo(d)
            ext_r = hi(d) - shi(d)
            if (ext_l > best_ext) then; best_ext = ext_l; best_d = d; best_side = 1; end if
            if (ext_r > best_ext) then; best_ext = ext_r; best_d = d; best_side = 2; end if
        end do
        if (best_side == 1) then
            hi(best_d) = slo(best_d) - 1
        else
            lo(best_d) = shi(best_d) + 1
        end if

    end subroutine s_amr_clip_box_from_supp

    !> Clip a box clear of the acoustic source and bubble-cloud supports.
    impure subroutine s_amr_clip_box_clear(lo, hi)

        integer, intent(inout) :: lo(3), hi(3)

        if (acoustic_source) call s_amr_clip_box_from_sources(lo, hi)
        if (bubbles_lagrange .and. lag_supp_on) call s_amr_clip_box_from_supp(lo, hi, lag_supp_lo, lag_supp_hi)

    end subroutine s_amr_clip_box_clear

    !> This rank's tagged cells as global level-0 coordinates; rank interiors are disjoint, so a SUM reduction counts each cell
    !! once.
    impure subroutine s_amr_local_tags(tag_grid, sidx, tags, ntag)

        logical, intent(in)               :: tag_grid(0:,0:,0:)
        integer, intent(in)               :: sidx(3)
        integer, allocatable, intent(out) :: tags(:,:)
        integer, intent(out)              :: ntag
        integer                           :: ci, cj, ck

        ntag = 0
        do ck = 0, p; do cj = 0, n; do ci = 0, m
            if (tag_grid(ci, cj, ck)) ntag = ntag + 1
        end do; end do; end do
        allocate (tags(3, max(ntag, 1)))
        ntag = 0
        do ck = 0, p; do cj = 0, n; do ci = 0, m
            if (tag_grid(ci, cj, ck)) then
                ntag = ntag + 1
                tags(1, ntag) = ci + sidx(1)
                tags(2, ntag) = 0
                tags(3, ntag) = 0
                if (n_glb > 0) tags(2, ntag) = cj + sidx(2)
                if (p_glb > 0) tags(3, ntag) = ck + sidx(3)
            end if
        end do; end do; end do

    end subroutine s_amr_local_tags

    !> Grow the pack buffers sidx/skb geometrically to hold nloc+extra entries, preserving the first nloc.
    impure subroutine s_amr_grow_pack(sidx, skb, nloc, extra)

        integer(8), allocatable, intent(inout) :: sidx(:)
        integer, allocatable, intent(inout)    :: skb(:)
        integer, intent(in)                    :: nloc, extra
        integer                                :: cap, newcap
        integer(8), allocatable                :: t8(:)
        integer, allocatable                   :: ti(:)

        cap = 0
        if (allocated(sidx)) cap = size(sidx)
        if (nloc + extra <= cap) return
        newcap = max(2*cap, max(nloc + extra, 1024))
        allocate (t8(newcap), ti(newcap))
        if (nloc > 0) then
            t8(1:nloc) = sidx(1:nloc)
            ti(1:nloc) = skb(1:nloc)
        end if
        call move_alloc(t8, sidx)
        call move_alloc(ti, skb)

    end subroutine s_amr_grow_pack

    !> Append this rank's tagged cells of child window [mlo:mhi] as (int8 linear index, parent kb) pairs; the encode matches the
    !! s_amr_nest_window_tags decode. gwin is not modified.
    impure subroutine s_amr_pack_gwin_pairs(gwin, mlo, mhi, mg, ng, kb, sidx, skb, nloc)

        integer, intent(in)                    :: mlo(3), mhi(3), mg, ng, kb
        logical, intent(in)                    :: gwin(mlo(1):,mlo(2):,mlo(3):)
        integer(8), allocatable, intent(inout) :: sidx(:)
        integer, allocatable, intent(inout)    :: skb(:)
        integer, intent(inout)                 :: nloc
        integer                                :: gi, gj, gk

        do gk = mlo(3), mhi(3)
            do gj = mlo(2), mhi(2)
                do gi = mlo(1), mhi(1)
                    if (.not. gwin(gi, gj, gk)) cycle
                    call s_amr_grow_pack(sidx, skb, nloc, 1)
                    nloc = nloc + 1
                    sidx(nloc) = int(gi, 8) + int(mg + 1, 8)*(int(gj, 8) + int(ng + 1, 8)*int(gk, 8))
                    skb(nloc) = kb
                end do
            end do
        end do

    end subroutine s_amr_pack_gwin_pairs

    !> Diagnostic (rank_time_wrt): free device memory at a named regrid point (AMD only).
    impure subroutine s_amr_report_mem(tag)

        character(len=*), intent(in) :: tag
        integer(8)                   :: fb, tb

        if (.not. rank_time_wrt) return
        call s_amr_dev_mem(fb, tb)
        if (fb < 0) return
        write (0, '(A,I0,A,A,A,F8.2,A,F8.2)') ' [amr-mem] rank ', proc_rank, ' ', tag, ' free GiB ', real(fb, wp)/2._wp**30, &
               & ' of ', real(tb, wp)/2._wp**30

    end subroutine s_amr_report_mem

    !> Regrid: tag by density gradient, cluster, shape, nest and rebuild the slots, carrying old data into the overlaps. Called
    !! between steps; no-op if nothing is tagged or the box set is unchanged.
    impure subroutine s_amr_regrid(q_cons_base)

        type(scalar_field), dimension(sys_size), intent(inout) :: q_cons_base
        logical, allocatable                                   :: tag_grid(:,:,:)
        type(t_box), allocatable                               :: boxes(:)
        integer                                                :: sidx(3), nboxes
        integer                                                :: old_np
        ! heap: O(global boxes) arrays overflow the default stack
        integer, allocatable :: box_level(:)
        integer, allocatable :: old_ilo(:,:), old_ext(:,:)
        integer, allocatable :: old_level(:)
        logical, allocatable :: old_owns(:)
        logical              :: same
        integer              :: i

        allocate (box_level(amr_max_blocks), old_ilo(3, amr_max_blocks), old_ext(3, amr_max_blocks), old_level(amr_max_blocks), &
                  & old_owns(amr_max_blocks))

        ! valid coarse ghosts at rank boundaries: the tag sweep and rebuild prolongation read across them

        call s_phase_tic(PH_RG_HALO)
        call s_amr_exchange_coarse_cons_halo(q_cons_base)
        do i = 1, sys_size
            $:GPU_UPDATE(host='[q_cons_base(i)%sf]')
        end do
        call s_phase_toc(PH_RG_HALO)

        ! cloud pad: smearing (mapCells) + stencil (2) + drift (amr_buf)
        if (bubbles_lagrange) call s_amr_compute_lag_supp(mapCells + 2 + amr_buf)

        call s_amr_report_mem('regrid-start')
        call s_phase_tic(PH_RG_TAG)
        call s_amr_regrid_tag_cells(q_cons_base, tag_grid, sidx)
        call s_phase_toc(PH_RG_TAG)
        call s_amr_report_mem('tagged')
        call s_amr_cad_count(tag_grid, sidx)
        call s_phase_tic(PH_RG_CLUS)
        call s_amr_regrid_cluster_tags(tag_grid, sidx, boxes, nboxes)
        call s_phase_toc(PH_RG_CLUS)
        call s_amr_report_mem('clustered')
        if (nboxes == 0) return  ! nothing tagged on any rank; keep the current blocks
        call s_amr_regrid_shape_boxes(boxes, nboxes)
        call s_amr_report_mem('shaped')
        if (nboxes == 0) return  ! every box was confined to the domain margin
        call s_phase_tic(PH_RG_CLUS)
        call s_amr_regrid_nest_children(boxes, nboxes, box_level)
        if (amr_snap > 0) call s_amr_regrid_snap_boxes(boxes, nboxes, box_level)
        call s_amr_check_box_caps(boxes, nboxes, box_level)  ! invariant: no box may exceed its level's slot cap
        call s_amr_check_box_disjoint(boxes, nboxes, box_level)  ! invariant: same-level boxes are pairwise disjoint
        if (rank_time_wrt .and. proc_rank == 0) call s_amr_report_coverage(boxes, nboxes, box_level)
        call s_amr_regrid_boxes_unchanged(boxes, nboxes, box_level, same)
        call s_phase_toc(PH_RG_CLUS)
        call s_amr_report_mem('compared')
        if (same) return  ! identical box set and levels: keep the live slots
        call s_amr_regrid_stash_migrate(boxes, nboxes, box_level, old_np, old_ilo, old_ext, old_level, old_owns)
        call s_amr_report_mem('stashed')
        call s_amr_report_mem('pre-rebuild')
        call s_amr_regrid_rebuild_slots(q_cons_base, boxes, nboxes, old_np, old_ilo, old_ext, old_level, old_owns)
        call s_amr_report_mem('post-rebuild')

    end subroutine s_amr_regrid

    !> Print the share of the base domain covered by each level's boxes.
    impure subroutine s_amr_report_coverage(boxes, nboxes, box_level)

        type(t_box), intent(in) :: boxes(:)
        integer, intent(in)     :: nboxes, box_level(:)
        real(wp)                :: cover(1:max(amr_max_level, 1)), base
        integer                 :: k

        cover = 0._wp
        do k = 1, nboxes
            cover(box_level(k)) = cover(box_level(k)) + real(product(merge(boxes(k)%hi - boxes(k)%lo + 1, 1, amr_dim)), wp)
        end do
        base = real(m_glb + 1, wp)*real(n_glb + 1, wp)*real(max(p_glb + 1, 1), wp)
        print '(A,I0,A,*(F8.4))', ' [amr-cov] boxes ', nboxes, '  level coverage (share of domain) ', cover/base

    end subroutine s_amr_report_coverage

    !> Regrid phase 1: density-gradient tag field, skipping the global boundary cells; tags over acoustic and bubble-cloud supports
    !! are suppressed. sidx = this rank's global start offsets.
    impure subroutine s_amr_regrid_tag_cells(q_cons_base, tag_grid, sidx)

        type(scalar_field), dimension(sys_size), intent(in) :: q_cons_base
        logical, allocatable, intent(inout)                 :: tag_grid(:,:,:)
        integer, intent(out)                                :: sidx(3)
        integer                                             :: tg_lo(3), tg_hi(3), ci, cj, ck
        real(wp)                                            :: r0, g

        sidx = amr_sidx
        tg_lo = merge(merge(1, 0, sidx == 0), 0, amr_dim)
        tg_hi = merge(merge(amr_ext - 1, amr_ext, sidx + amr_ext == [m_glb, n_glb, p_glb]), 0, amr_dim)
        allocate (tag_grid(0:m,0:n,0:p)); tag_grid = .false.
        do ck = tg_lo(3), tg_hi(3)
            do cj = tg_lo(2), tg_hi(2)
                do ci = tg_lo(1), tg_hi(1)
                    ! total density (sum of continuity variables): single-fluid tagger, immune to trace-fluid noise
                    r0 = f_amr_rho_tot_sf(q_cons_base, ci, cj, ck)
                    g = f_amr_tag_ind(f_amr_rho_tot_sf(q_cons_base, ci + 1, cj, ck), r0, f_amr_rho_tot_sf(q_cons_base, ci - 1, &
                                      & cj, ck))
                    if (n_glb > 0) g = max(g, f_amr_tag_ind(f_amr_rho_tot_sf(q_cons_base, ci, cj + 1, ck), r0, &
                        & f_amr_rho_tot_sf(q_cons_base, ci, cj - 1, ck)))
                    if (p_glb > 0) g = max(g, f_amr_tag_ind(f_amr_rho_tot_sf(q_cons_base, ci, cj, ck + 1), r0, &
                        & f_amr_rho_tot_sf(q_cons_base, ci, cj, ck - 1)))
                    if (g > amr_tag_eps) tag_grid(ci, cj, ck) = .true.
                    ! acoustic source support stays coarse
                    if (acoustic_source .and. tag_grid(ci, cj, ck)) then
                        if (f_in_acoustic_support(ci + sidx(1), cj + sidx(2), ck + sidx(3))) tag_grid(ci, cj, ck) = .false.
                    end if
                    ! bubble cloud stays coarse (two-way coupling lives there)
                    if (bubbles_lagrange .and. tag_grid(ci, cj, ck)) then
                        if (f_in_lag_support(ci + sidx(1), cj + sidx(2), ck + sidx(3))) tag_grid(ci, cj, ck) = .false.
                    end if
                end do
            end do
        end do

    end subroutine s_amr_regrid_tag_cells

    !> Cadence audit: count this rank's level-1 tags and those outside the pre-regrid level-1 coverage (a feature that drifted past
    !! amr_buf). Skipped on the first regrid.
    impure subroutine s_amr_cad_count(tag_grid, sidx)

        logical, intent(in)  :: tag_grid(0:,0:,0:)
        integer, intent(in)  :: sidx(3)
        logical, allocatable :: cov(:,:,:)
        integer              :: k, ci, cj, ck, bl(3), bh(3)
        logical              :: any_l1

        if (.not. amr_cad_armed) then
            amr_cad_armed = .true.
            return
        end if
        any_l1 = .false.
        do k = 1, amr_num_blocks
            if (amr_block_level(k) == 1) any_l1 = .true.
        end do
        if (.not. any_l1) return
        allocate (cov(0:m,0:n,0:p)); cov = .false.
        do k = 1, amr_num_blocks
            if (amr_block_level(k) /= 1) cycle
            bl = merge(max(amr_region_lo_all(:,k) - sidx, 0), 0, amr_dim)
            bh = merge(min(amr_region_hi_all(:,k) - sidx, amr_ext), 0, amr_dim)
            if (any(bl > bh)) cycle
            cov(bl(1):bh(1),bl(2):bh(2),bl(3):bh(3)) = .true.
        end do
        do ck = 0, p
            do cj = 0, n
                do ci = 0, m
                    if (tag_grid(ci, cj, ck)) then
                        amr_cad_tot = amr_cad_tot + 1
                        if (.not. cov(ci, cj, ck)) amr_cad_esc = amr_cad_esc + 1
                    end if
                end do
            end do
        end do
        deallocate (cov)

    end subroutine s_amr_cad_count

    !> Regrid phase 2: cluster this rank's tags (rank-invariant, via signature reductions) into separated boxes; nboxes = 0 if
    !! nothing is tagged.
    impure subroutine s_amr_regrid_cluster_tags(tag_grid, sidx, boxes, nboxes)

        logical, allocatable, intent(inout)   :: tag_grid(:,:,:)
        integer, intent(in)                   :: sidx(3)
        type(t_box), allocatable, intent(out) :: boxes(:)
        integer, intent(out)                  :: nboxes
        integer, allocatable                  :: tags(:,:)
        integer                               :: ntag

        call s_amr_local_tags(tag_grid, sidx, tags, ntag)
        deallocate (tag_grid)
        call s_amr_cluster(tags, ntag, boxes, nboxes, .true.)
        deallocate (tags)

    end subroutine s_amr_regrid_cluster_tags

    !> True iff two same-level blocks can coexist: disjoint, and either >= 2 empty coarse cells apart or face-adjacent with an exact
    !! transverse match. Mirrors s_amr_check_seam_topology.
    pure logical function f_amr_boxes_compatible(alo, ahi, blo, bhi) result(ok)

        integer, intent(in) :: alo(3), ahi(3), blo(3), bhi(3)
        integer             :: d, t, gap
        logical             :: tmatch, tover

        ok = .true.
        if (all(alo <= bhi .and. blo <= ahi)) then; ok = .false.; return; end if
        do d = 1, 3
            if (.not. amr_dim(d)) cycle
            gap = max(blo(d) - ahi(d), alo(d) - bhi(d)) - 1  ! empty cells along d (< 0: overlap)
            if (gap >= 2 .or. gap < 0) cycle
            tover = .true.; tmatch = .true.
            do t = 1, 3
                if (t == d .or. .not. amr_dim(t)) cycle
                tover = tover .and. alo(t) <= bhi(t) .and. blo(t) <= ahi(t)
                tmatch = tmatch .and. alo(t) == blo(t) .and. ahi(t) == bhi(t)
            end do
            if (tover .and. .not. (gap == 0 .and. tmatch)) then; ok = .false.; return; end if
        end do

    end function f_amr_boxes_compatible

    !> Hysteresis (amr_snap > 0): a box within amr_snap cells per face of a live same-level block takes that block's box.
    !! All-or-none: the snapped set must stay disjoint, compatible and singly nested (within amr_cpat_mar), else the fresh boxes
    !! stand.
    impure subroutine s_amr_regrid_snap_boxes(boxes, nboxes, box_level)

        type(t_box), intent(inout) :: boxes(:)
        integer, intent(in)        :: nboxes, box_level(:)
        type(t_box), allocatable   :: snapped(:)
        integer                    :: k, kk, ks, npar, nsnap, mlo(3), mhi(3)
        logical                    :: ok

        allocate (snapped(nboxes)); snapped(1:nboxes) = boxes(1:nboxes)
        nsnap = 0
        do k = 1, nboxes
            do ks = 1, amr_num_blocks
                if (amr_block_level(ks) /= box_level(k)) cycle
                if (all(abs(amr_region_lo_all(:,ks) - boxes(k)%lo) <= amr_snap) .and. all(abs(amr_region_hi_all(:, &
                    & ks) - boxes(k)%hi) <= amr_snap)) then
                    ! with IB the old block must contain the fresh box (a lagging body margin would be lost)
                    if (ib .and. .not. (all(amr_region_lo_all(:,ks) <= boxes(k)%lo) .and. all(amr_region_hi_all(:, &
                        & ks) >= boxes(k)%hi))) cycle
                    if (any(amr_region_lo_all(:,ks) /= boxes(k)%lo) .or. any(amr_region_hi_all(:, &
                        & ks) /= boxes(k)%hi)) nsnap = nsnap + 1
                    snapped(k)%lo = amr_region_lo_all(:,ks); snapped(k)%hi = amr_region_hi_all(:,ks)
                    exit
                end if
            end do
        end do
        ok = nsnap > 0
        do k = 1, nboxes
            if (.not. ok) exit
            do kk = k + 1, nboxes
                if (box_level(kk) /= box_level(k)) cycle
                if (all(snapped(k)%lo <= snapped(kk)%hi .and. snapped(kk)%lo <= snapped(k)%hi)) then
                    ok = .false.; exit
                end if
                if (.not. f_amr_boxes_compatible(snapped(k)%lo, snapped(k)%hi, snapped(kk)%lo, snapped(kk)%hi)) then
                    ok = .false.; exit
                end if
            end do
        end do
        ! a level >= 2 box lies inside exactly one parent box, inset by the nesting margin
        do k = 1, nboxes
            if (.not. ok) exit
            if (box_level(k) < 2) cycle
            npar = 0
            do kk = 1, nboxes
                if (box_level(kk) /= box_level(k) - 1) cycle
                if (.not. all(snapped(k)%lo <= snapped(kk)%hi .and. snapped(kk)%lo <= snapped(k)%hi)) cycle
                npar = npar + 1
                call s_amr_nest_window(snapped(kk), mlo, mhi)
                if (any(snapped(k)%lo < mlo) .or. any(snapped(k)%hi > mhi)) ok = .false.
            end do
            if (npar /= 1) ok = .false.
        end do
        if (ok) boxes(1:nboxes) = snapped(1:nboxes)
        if (rank_time_wrt .and. proc_rank == 0) write (0, '(A,I0,A,I0,A,L1)') '[amr-snap] boxes ', nboxes, ' snapped ', nsnap, &
            & ' applied ', ok
        deallocate (snapped)

    end subroutine s_amr_regrid_snap_boxes

    !> Append to out(n+1:) the pieces of `box` that stay `gap` empty cells clear of [glo:ghi] and of each other, cutting dim by dim;
    !! the part inside the block's span is dropped. The caller guarantees `box` overlaps the gap-grown block.
    pure subroutine s_amr_box_carve(box, glo, ghi, gap, out, n)

        type(t_box), intent(in)    :: box
        integer, intent(in)        :: glo(3), ghi(3), gap
        type(t_box), intent(inout) :: out(:)
        integer, intent(inout)     :: n
        type(t_box)                :: rem
        integer                    :: d

        rem = box
        do d = 1, 3
            if (.not. amr_dim(d)) cycle
            if (rem%lo(d) < glo(d) - gap) then
                n = n + 1; out(n) = rem; out(n)%hi(d) = glo(d) - gap - 1
            end if
            if (rem%hi(d) > ghi(d) + gap) then
                n = n + 1; out(n) = rem; out(n)%lo(d) = ghi(d) + gap + 1
            end if
            rem%lo(d) = max(rem%lo(d), glo(d)); rem%hi(d) = min(rem%hi(d), ghi(d))
            if (rem%lo(d) > rem%hi(d)) return
        end do

    end subroutine s_amr_box_carve

    !> Abort if ranks hold different box lists (each rank sees only nearby IB bodies; a mismatch hangs the ownership collectives).
    impure subroutine s_amr_check_boxes_agree(boxes, nboxes)

        type(t_box), intent(in) :: boxes(:)
        integer, intent(in)     :: nboxes
        real(wp)                :: chk, cmin, cmax
        integer                 :: k

        chk = real(nboxes, wp)
        do k = 1, nboxes
            chk = chk + real(modulo(k*7919, 1009), wp)*real(sum(boxes(k)%lo*(/1, 131, 17161/)) + 3*sum(boxes(k)%hi*(/1, 131, &
                             & 17161/)), wp)
        end do
        call s_mpi_allreduce_min(chk, cmin)
        call s_mpi_allreduce_max(chk, cmax)
        if (cmin /= cmax) then
            call s_mpi_abort('amr regrid with ib: the ranks built different block lists, because they hold different sets of ' &
                             & // 'immersed bodies; raise ib_neighborhood_radius until every rank sees every body')
        end if

    end subroutine s_amr_check_boxes_agree

    !> Regrid phase 3a (IB): tile the boxes around the immersed bodies. Each body (bbox + margin; close bodies share one) owns a
    !! block; boxes give up that block plus a gap ring, the remainder tiles to <= amr_maxc_fit, and body blocks are appended last.
    !! The ring keeps body blocks off other blocks, since the seam halo and reflux need exact-match transverse faces.
    impure subroutine s_amr_carve_tile_around_bodies(boxes, nboxes, out, nt, capt)

        type(t_box), intent(in)    :: boxes(:)
        integer, intent(in)        :: nboxes
        type(t_box), intent(inout) :: out(:)
        integer, intent(inout)     :: nt, capt
        ! empty coarse cells between blocks off a shared grid (one cell gives both the same reflux outside cell)
        integer, parameter       :: gap = 2
        type(t_box), allocatable :: pieces(:), rest(:)
        integer, allocatable     :: glo(:,:), ghi(:,:), clo(:,:), chi(:,:), gbx(:,:)
        integer, parameter       :: BOX_UNSET = -1000000000
        integer                  :: nbody, gid, ierr
        integer                  :: e(3), d, dsep, nsep, lowb, upb, need, avail_l, avail_u, tl, tu, cl(3), ch(3)
        logical, allocatable     :: hit(:)
        integer                  :: ng, i, j, g, k, p, npc, nn, mrg, blo(3), bhi(3)
        logical                  :: merged

        ! margin as in s_amr_expand_box_over_bodies: image-point stencil plus nested-level insets

        mrg = max(amr_buf, 4) + max(0, amr_max_level - 1)*amr_cpat_mar
        nbody = max(num_gbl_ibs, num_ibs, 1)
        allocate (glo(3, nbody), ghi(3, nbody), clo(3, nbody), chi(3, nbody), hit(nbody))

        ! Every rank must build identical boxes though it sees only nearby bodies whose positions differ in the last digits: each
        ! rank fills the boxes of its bodies by global id and one MAX reduction shares them.
        allocate (gbx(12, nbody)); gbx = BOX_UNSET
        do i = 1, num_ibs
            gid = patch_ib(i)%gbl_patch_id
            if (gid < 1 .or. gid > nbody) gid = i
            call s_amr_body_bbox(i, mrg, blo, bhi); call s_amr_body_bbox(i, 0, cl, ch)
            gbx(1:3,gid) = blo; gbx(4:6,gid) = bhi; gbx(7:9,gid) = cl; gbx(10:12,gid) = ch
        end do
#ifdef MFC_MPI
        call MPI_ALLREDUCE(MPI_IN_PLACE, gbx, size(gbx), MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD, ierr)
#endif
        ng = 0
        do gid = 1, nbody
            if (gbx(1, gid) <= BOX_UNSET) cycle
            blo = gbx(1:3,gid); bhi = gbx(4:6,gid)
            ! blocks stay buff_size inside the domain
            if (any(amr_dim .and. (blo < buff_size .or. bhi > [m_glb, n_glb, p_glb] - buff_size))) then
                call s_mpi_abort('amr dynamic regrid with ib: the immersed body plus its containment ' &
                                 & // 'margin does not fit inside the refinable domain interior (blocks stay buff_size off the edges)')
            end if
            ng = ng + 1; glo(:,ng) = blo; ghi(:,ng) = bhi; clo(:,ng) = gbx(7:9,gid); chi(:,ng) = gbx(10:12,gid)
        end do
        deallocate (gbx)

        ! Resolve body blocks facing each other across < gap empty cells, or overlapping: trim the margins apart (floored at 3
        ! cells beyond the body: image-point stencil and moving-body guard), else merge the groups.
        merged = .true.
        do while (merged)
            merged = .false.
            outer: do i = 1, ng - 1
                do j = i + 1, ng
                    e = 0
                    do d = 1, 3
                        if (amr_dim(d)) e(d) = max(glo(d, j) - ghi(d, i), glo(d, i) - ghi(d, j)) - 1
                    end do
                    nsep = count(amr_dim .and. e >= 0)  ! dimensions in which the two boxes are apart
                    if (nsep == 0) then
                        glo(:,i) = min(glo(:,i), glo(:,j)); ghi(:,i) = max(ghi(:,i), ghi(:,j))
                        clo(:,i) = min(clo(:,i), clo(:,j)); chi(:,i) = max(chi(:,i), chi(:,j))
                        glo(:,j) = glo(:,ng); ghi(:,j) = ghi(:,ng); clo(:,j) = clo(:,ng); chi(:,j) = chi(:,ng); ng = ng - 1
                        merged = .true.; exit outer
                    else if (nsep == 1) then
                        dsep = 0
                        do d = 1, 3
                            if (amr_dim(d) .and. e(d) >= 0) dsep = d
                        end do
                        if (e(dsep) < gap) then
                            need = gap - e(dsep)
                            lowb = merge(i, j, ghi(dsep, i) < ghi(dsep, j)); upb = merge(j, i, lowb == i)
                            avail_l = max(0, ghi(dsep, lowb) - (chi(dsep, lowb) + 3))
                            avail_u = max(0, (clo(dsep, upb) - 3) - glo(dsep, upb))
                            if (avail_l + avail_u >= need) then
                                tl = min(need, avail_l); tu = need - tl
                                ghi(dsep, lowb) = ghi(dsep, lowb) - tl; glo(dsep, upb) = glo(dsep, upb) + tu
                            else
                                glo(:,i) = min(glo(:,i), glo(:,j)); ghi(:,i) = max(ghi(:,i), ghi(:,j))
                                clo(:,i) = min(clo(:,i), clo(:,j)); chi(:,i) = max(chi(:,i), chi(:,j))
                                glo(:,j) = glo(:,ng); ghi(:,j) = ghi(:,ng); clo(:,j) = clo(:,ng); chi(:,j) = chi(:,ng); ng = ng - 1
                                merged = .true.; exit outer
                            end if
                        end if
                    end if
                end do
            end do outer
        end do
        do g = 1, ng
            if (any(amr_dim .and. ghi(:,g) - glo(:,g) + 1 > amr_maxc_fit)) then
                write (0, '(A,I0,A,I0,A,3I6,A,3I6,A,3I6,A,I0)') ' [amr] body group ', g, ' of ', ng, ' lo ', glo(:,g), ' hi ', &
                       & ghi(:,g), ' cap ', amr_maxc_fit, ' num_ibs ', num_ibs
                call s_mpi_abort('amr dynamic regrid with ib: a body (or bodies closer than the block gap) plus ' &
                                 & // 'margin exceeds the per-rank block size cap; use fewer ranks or a larger amr_maxc_fit')
            end if
        end do

        ! canonical order by low corner: patch_ib order differs per rank and the carve depends on it
        do i = 2, ng
            blo = glo(:,i); bhi = ghi(:,i); j = i - 1
            do while (j >= 1)
                if (.not. (glo(3, j) > blo(3) .or. (glo(3, j) == blo(3) .and. (glo(2, j) > blo(2) .or. (glo(2, &
                    & j) == blo(2) .and. glo(1, j) > blo(1)))))) exit
                glo(:,j + 1) = glo(:,j); ghi(:,j + 1) = ghi(:,j); j = j - 1
            end do
            glo(:,j + 1) = blo; ghi(:,j + 1) = bhi
        end do

        allocate (pieces(amr_max_blocks), rest(amr_max_blocks))
        hit = .false.
        do k = 1, nboxes
            npc = 1; pieces(1) = boxes(k)
            do g = 1, ng
                nn = 0
                do p = 1, npc
                    if (f_amr_boxes_overlap(pieces(p)%lo, pieces(p)%hi, glo(:,g) - gap, ghi(:,g) + gap)) then
                        if (f_amr_boxes_overlap(pieces(p)%lo, pieces(p)%hi, glo(:,g), ghi(:,g))) hit(g) = .true.
                        if (nn + 2*num_dims > amr_max_blocks) call s_mpi_abort('amr regrid: too many pieces around a body')
                        call s_amr_box_carve(pieces(p), glo(:,g), ghi(:,g), gap, rest, nn)
                    else
                        nn = nn + 1; rest(nn) = pieces(p)
                    end if
                end do
                pieces(1:nn) = rest(1:nn); npc = nn
            end do
            do p = 1, npc
                call s_amr_tile_box(pieces(p)%lo, pieces(p)%hi, out, nt, amr_max_blocks, capt)
            end do
        end do

        do g = 1, ng
            if (.not. hit(g)) cycle
            if (nt >= amr_max_blocks) then
                capt = 1
            else
                nt = nt + 1; out(nt)%lo = glo(:,g); out(nt)%hi = ghi(:,g)
            end if
        end do

    end subroutine s_amr_carve_tile_around_bodies

    !> Regrid phase 3: pad and clamp each box, clip it clear of the acoustic/bubble supports, then tile oversized boxes (IB: after
    !! carving out the body blocks).
    impure subroutine s_amr_regrid_shape_boxes(boxes, nboxes)

        type(t_box), allocatable, intent(inout) :: boxes(:)
        integer, intent(inout)                  :: nboxes
        type(t_box), allocatable                :: tiled(:)
        integer                                 :: lo(3), hi(3), k, kk, ntl, capt, pad

        pad = amr_buf
        k = 0
        do kk = 1, nboxes
            lo = merge(max(boxes(kk)%lo - pad, buff_size), 0, amr_dim)
            hi = merge(min(boxes(kk)%hi + pad, [m_glb, n_glb, p_glb] - buff_size), 0, amr_dim)
            ! the supports are coarse-grid only; clipping only shrinks, so boxes stay disjoint
            call s_amr_clip_box_clear(lo, hi)
            if (any(hi < lo)) cycle  ! confined to the domain margin
            k = k + 1; boxes(k)%lo = lo; boxes(k)%hi = hi
        end do
        nboxes = k
        if (nboxes == 0) return

        ! tile boxes larger than amr_maxc_fit (IB: after carving out body blocks, so a body is never split)
        allocate (tiled(amr_max_blocks))
        ntl = 0; capt = 0
        if (ib) then
            call s_amr_carve_tile_around_bodies(boxes, nboxes, tiled, ntl, capt)
        else
            do kk = 1, nboxes
                call s_amr_tile_box(boxes(kk)%lo, boxes(kk)%hi, tiled, ntl, amr_max_blocks, capt)
            end do
        end if
        if (capt == 1 .and. proc_rank == 0) print '(A,I0)', ' [amr] WARNING: tiling capped at amr_max_blocks = ', amr_max_blocks
        deallocate (boxes); call move_alloc(tiled, boxes)
        nboxes = ntl
        if (ib) call s_amr_check_boxes_agree(boxes, nboxes)

        ! a body block cannot both contain the body and exclude a source/cloud: fail closed
        if (ib .and. (acoustic_source .or. (bubbles_lagrange .and. lag_supp_on))) then
            do k = 1, nboxes
                lo = boxes(k)%lo; hi = boxes(k)%hi
                call s_amr_clip_box_clear(lo, hi)
                if (any(lo /= boxes(k)%lo) .or. any(hi /= boxes(k)%hi)) then
                    call s_mpi_abort('amr regrid: a block must contain an immersed body AND stay ' &
                                     & // 'clear of an acoustic source support / Lagrangian bubble cloud - the ' &
                                     & // 'constraints conflict; move the body, source, or cloud apart')
                end if
            end do
        end if

    end subroutine s_amr_regrid_shape_boxes

    !> Regrid phase 3b: nest child boxes inside each level-(l-1) box for l = 2..amr_max_level (parents first) and set box_level. A
    !! child's extent is the density-gradient sensor on the old parent-level fine solution, coarsened and clustered; a new region
    !! with no fine data gets a centred inset.
    !!
    !! Per level: pass 1 tags each parent's window from owned old blocks and packs (index, kb) pairs; an alltoall routes them to the
    !! parent's round-robin owner; pass 2 rebuilds the window and clusters it into children; an allgatherv plus a stable sort by kb
    !! replays the children in serial order, so the list and its truncation are rank-invariant. IB nesting is np=1 only.
    impure subroutine s_amr_regrid_nest_children(boxes, nboxes, box_level)

        type(t_box), allocatable, intent(inout) :: boxes(:)
        integer, intent(inout)                  :: nboxes, box_level(:)
        type(t_box), allocatable                :: grown(:)
        integer                                 :: lev, plo, phi, newlo, kb, ob, jch, nloc_send, ntot_g, nmych, ntot_ch, nct
        integer                                 :: ls  !< host scalar slot: NVHPC maps an update-clause subscript array
        integer                                 :: mlo(3), mhi(3)
        integer, allocatable                    :: mlo_all(:,:), mhi_all(:,:), powner(:), mych(:,:), gch(:,:), chord(:)
        integer, allocatable                    :: skb(:), gkb(:), ctags(:,:)
        integer(8), allocatable                 :: sidx(:), gidx(:)
        logical, allocatable                    :: mine(:), covered(:), gwin(:,:,:)

        box_level(1:nboxes) = 1
        if (amr_max_level < 2) return
        amr_child_dropped = .false.

        ! grow to amr_max_blocks for the appends (the IB path leaves boxes at the cluster count)
        if (size(boxes) < amr_max_blocks) then
            allocate (grown(amr_max_blocks)); grown(1:nboxes) = boxes(1:nboxes); call move_alloc(grown, boxes)
        end if
        ! host-refresh the old blocks: the sensor reads the flat store on the host, stale until the stash's update
        do ob = 1, amr_num_blocks
            if (.not. amr_owns_all(ob)) cycle
            ls = amr_loc_of(ob)
            $:GPU_UPDATE(host='[amr_cons_st(:, :, :, :, ls)]')
        end do
        call s_amr_refresh_my_blocks()

        plo = 1; phi = nboxes  ! parents
        do lev = 2, amr_max_level
            if (phi < plo) exit
            newlo = nboxes + 1
            allocate (covered(plo:phi), mine(plo:phi), mlo_all(3,plo:phi), mhi_all(3,plo:phi), powner(plo:phi), mych(7, &
                      & amr_max_blocks))
            call s_amr_nest_mark_parents(boxes, plo, phi, lev, mine, covered)
            do kb = plo, phi
                powner(kb) = mod(kb - plo, max(num_procs, 1))
                call s_amr_nest_window(boxes(kb), mlo_all(:,kb), mhi_all(:,kb))
            end do

            ! pass 1: build a window only where this rank holds an overlapping old block (IB: the owner also builds one for the
            ! body)
            nloc_send = 0
            do kb = plo, phi
                mlo = mlo_all(:,kb); mhi = mhi_all(:,kb)
                if (f_amr_nest_window_empty(mlo, mhi)) cycle
                if (.not. (mine(kb) .or. (ib .and. powner(kb) == proc_rank))) cycle
                allocate (gwin(mlo(1):mhi(1),mlo(2):mhi(2),mlo(3):mhi(3)))
                call s_amr_nest_tag_parent(boxes(kb), lev, mlo, mhi, gwin, covered(kb))
                call s_amr_pack_gwin_pairs(gwin, mlo, mhi, m_glb, n_glb, kb, sidx, skb, nloc_send)
                deallocate (gwin)
            end do
            call s_amr_nest_route_pairs(powner, plo, sidx, skb, nloc_send, gidx, gkb, ntot_g)

            ! pass 2: own parents only
            nmych = 0
            do kb = plo, phi
                if (powner(kb) /= proc_rank) cycle
                if (nmych + 1 > amr_max_blocks) exit
                mlo = mlo_all(:,kb); mhi = mhi_all(:,kb)
                if (f_amr_nest_window_empty(mlo, mhi)) cycle
                call s_amr_nest_window_tags(kb, mlo, mhi, gidx, gkb, ntot_g, ctags, nct)
                if (.not. covered(kb)) then
                    call s_amr_nest_inset_child(boxes(kb), lev, kb, mych, nmych)
                else if (nct > 0) then
                    call s_amr_nest_cluster_children(ctags, nct, mlo, mhi, lev, kb, mych, nmych)
                end if
                deallocate (ctags)
            end do

            call s_amr_nest_gather_children(mych, nmych, plo, phi, gch, chord, ntot_ch)
            do jch = 1, ntot_ch
                if (nboxes + 1 > amr_max_blocks) exit  ! canonical order gives the same truncation
                nboxes = nboxes + 1
                boxes(nboxes)%lo = gch(1:3,chord(jch)); boxes(nboxes)%hi = gch(4:6,chord(jch)); box_level(nboxes) = lev
            end do
            deallocate (gch, chord, powner, mych, gidx, gkb, covered, mine, mlo_all, mhi_all)
            plo = newlo; phi = nboxes
        end do
        if (nboxes >= amr_max_blocks .and. proc_rank == 0) print '(A)', &
            & ' [amr] NOTE: block pool full during multi-level nesting; some boxes were not refined further'

    end subroutine s_amr_regrid_nest_children

    !> mine(kb): this rank holds a level-(lev-1) block overlapping parent kb; covered(kb): any rank does (one LOR; every block has
    !! one owner).
    impure subroutine s_amr_nest_mark_parents(boxes, plo, phi, lev, mine, covered)

        type(t_box), intent(in) :: boxes(:)
        integer, intent(in)     :: plo, phi, lev
        logical, intent(out)    :: mine(plo:), covered(plo:)
        integer                 :: obi, ob, kb

#ifdef MFC_MPI
        integer :: ierr
#endif

        mine = .false.
        do obi = 1, amr_n_my
            ob = amr_my_blk(obi)
            if (amr_block_level(ob) /= lev - 1) cycle
            do kb = plo, phi
                if (f_amr_boxes_overlap(boxes(kb)%lo, boxes(kb)%hi, amr_region_lo_all(:,ob), amr_region_hi_all(:, &
                    & ob))) mine(kb) = .true.
            end do
        end do
        covered = mine
#ifdef MFC_MPI
        if (num_procs > 1) call MPI_ALLREDUCE(MPI_IN_PLACE, covered, phi - plo + 1, MPI_LOGICAL, MPI_LOR, MPI_COMM_WORLD, ierr)
#endif

    end subroutine s_amr_nest_mark_parents

    !> The nesting window of a parent: its box inset by amr_cpat_mar, so a child's ghost prolongation reads valid parent interior.
    pure subroutine s_amr_nest_window(box, mlo, mhi)

        type(t_box), intent(in) :: box
        integer, intent(out)    :: mlo(3), mhi(3)

        mlo = box%lo + merge(amr_cpat_mar, 0, amr_dim); mhi = box%hi - merge(amr_cpat_mar, 0, amr_dim)

    end subroutine s_amr_nest_window

    !> Too small to nest a child in some active dimension.
    pure logical function f_amr_nest_window_empty(lo, hi) result(e)

        integer, intent(in) :: lo(3), hi(3)

        e = any(amr_dim .and. hi < lo)

    end function f_amr_nest_window_empty

    !> Pass 1 for one parent: tag its window from owned level-(lev-1) blocks (amr_block_level still holds the old levels). IB: also
    !! mark the body bbox with margin max(amr_buf, 4) + amr_cpat_mar, so the clamp to the window keeps a full image-point stencil.
    impure subroutine s_amr_nest_tag_parent(box, lev, mlo, mhi, gwin, covered)

        type(t_box), intent(in) :: box
        integer, intent(in)     :: lev, mlo(3), mhi(3)
        logical, intent(out)    :: gwin(mlo(1):,mlo(2):,mlo(3):)
        logical, intent(inout)  :: covered
        integer                 :: obi, ob, ib_i, bb_lo(3), bb_hi(3)
        logical                 :: any_tag

        gwin = .false.; any_tag = .false.
        do obi = 1, amr_n_my
            ob = amr_my_blk(obi)
            if (amr_block_level(ob) /= lev - 1) cycle
            if (.not. f_amr_boxes_overlap(box%lo, box%hi, amr_region_lo_all(:,ob), amr_region_hi_all(:,ob))) cycle
            call s_amr_tag_child_from_fine(ob, mlo, mhi, gwin, any_tag)
        end do
        if (.not. ib) return
        do ib_i = 1, num_ibs
            call s_amr_body_bbox(ib_i, max(amr_buf, 4) + amr_cpat_mar, bb_lo, bb_hi)
            bb_lo = max(bb_lo, mlo); bb_hi = min(bb_hi, mhi)
            if (f_amr_nest_window_empty(bb_lo, bb_hi)) cycle
            covered = .true.
            gwin(bb_lo(1):bb_hi(1),bb_lo(2):bb_hi(2),bb_lo(3):bb_hi(3)) = .true.
        end do

    end subroutine s_amr_nest_tag_parent

    !> Route each parent's (index, kb) pairs to its clustering owner with a stable counting sort by owner. Serial: the send list is
    !! the gathered list.
    impure subroutine s_amr_nest_route_pairs(powner, plo, sidx, skb, nloc, gidx, gkb, ntot)

        integer, intent(in)                    :: plo, powner(plo:)
        integer(8), allocatable, intent(inout) :: sidx(:)
        integer, allocatable, intent(inout)    :: skb(:)
        integer, intent(in)                    :: nloc
        integer(8), allocatable, intent(out)   :: gidx(:)
        integer, allocatable, intent(out)      :: gkb(:)
        integer, intent(out)                   :: ntot

#ifdef MFC_MPI
        integer                 :: i, ip, ierr
        integer, allocatable    :: rcnt(:), rdsp(:), scnt(:), sdsp(:), phead(:), pord(:), tkb(:)
        integer(8), allocatable :: tidx(:)
#endif

        if (.not. allocated(sidx)) allocate (sidx(0), skb(0))  ! this rank owned no tags at this level
#ifdef MFC_MPI
        if (num_procs > 1) then
            allocate (rcnt(num_procs), rdsp(num_procs), scnt(num_procs), sdsp(num_procs), phead(num_procs), pord(max(nloc, 1)))
            allocate (tidx(max(nloc, 1)), tkb(max(nloc, 1)))
            phead = 0
            do i = 1, nloc
                ip = powner(skb(i)) + 1; phead(ip) = phead(ip) + 1
            end do
            scnt = phead
            sdsp(1) = 0
            do ip = 2, num_procs
                sdsp(ip) = sdsp(ip - 1) + scnt(ip - 1)
            end do
            phead = sdsp + 1
            do i = 1, nloc
                ip = powner(skb(i)) + 1; pord(phead(ip)) = i; phead(ip) = phead(ip) + 1
            end do
            do i = 1, nloc
                tidx(i) = sidx(pord(i)); tkb(i) = skb(pord(i))
            end do
            call MPI_ALLTOALL(scnt, 1, MPI_INTEGER, rcnt, 1, MPI_INTEGER, MPI_COMM_WORLD, ierr)
            rdsp(1) = 0
            do ip = 2, num_procs
                rdsp(ip) = rdsp(ip - 1) + rcnt(ip - 1)
            end do
            ntot = rdsp(num_procs) + rcnt(num_procs)
            allocate (gidx(max(ntot, 1)), gkb(max(ntot, 1)))
            call MPI_ALLTOALLV(tidx, scnt, sdsp, MPI_INTEGER8, gidx, rcnt, rdsp, MPI_INTEGER8, MPI_COMM_WORLD, ierr)
            call MPI_ALLTOALLV(tkb, scnt, sdsp, MPI_INTEGER, gkb, rcnt, rdsp, MPI_INTEGER, MPI_COMM_WORLD, ierr)
            deallocate (sidx, skb)
            return
        end if
#endif
        call move_alloc(sidx, gidx); call move_alloc(skb, gkb)
        ntot = nloc

    end subroutine s_amr_nest_route_pairs

    !> Pass 2 for one parent: rebuild its dense window from the routed pairs with gkb == kb (dedups replicated tags) and extract the
    !! tags in (k,j,i) order. The int8 decode matches s_amr_pack_gwin_pairs.
    impure subroutine s_amr_nest_window_tags(kb, mlo, mhi, gidx, gkb, ntot, ctags, nct)

        integer, intent(in)               :: kb, mlo(3), mhi(3), ntot, gkb(:)
        integer(8), intent(in)            :: gidx(:)
        integer, allocatable, intent(out) :: ctags(:,:)
        integer, intent(out)              :: nct
        logical, allocatable              :: gwin(:,:,:)
        integer                           :: i, gi, gj, gk
        integer(8)                        :: jrem

        allocate (gwin(mlo(1):mhi(1),mlo(2):mhi(2),mlo(3):mhi(3)))
        gwin = .false.
        do i = 1, ntot
            if (gkb(i) /= kb) cycle
            gk = int(gidx(i)/(int(m_glb + 1, 8)*int(n_glb + 1, 8)))
            jrem = gidx(i) - int(gk, 8)*int(m_glb + 1, 8)*int(n_glb + 1, 8)
            gj = int(jrem/int(m_glb + 1, 8))
            gi = int(jrem - int(gj, 8)*int(m_glb + 1, 8))
            gwin(gi, gj, gk) = .true.
        end do
        nct = count(gwin)
        allocate (ctags(3, max(nct, 1)))
        nct = 0
        do gk = mlo(3), mhi(3); do gj = mlo(2), mhi(2); do gi = mlo(1), mhi(1)
            if (gwin(gi, gj, gk)) then
                nct = nct + 1
                ctags(1, nct) = gi; ctags(2, nct) = gj; ctags(3, nct) = gk
            end if
        end do; end do; end do

    end subroutine s_amr_nest_window_tags

    !> Cluster a parent's tagged cells into children, padded by amr_buf and clamped into the nesting window. IB: expand over bodies
    !! and re-clamp (the window already holds the body plus margin).
    impure subroutine s_amr_nest_cluster_children(ctags, nct, mlo, mhi, lev, kb, mych, nmych)

        integer, intent(inout)   :: ctags(:,:), mych(:,:), nmych
        integer, intent(in)      :: nct, mlo(3), mhi(3), lev, kb
        type(t_box), allocatable :: cboxes(:)
        integer                  :: ncb, kc, clo(3), chi(3)

        call s_amr_cluster(ctags, nct, cboxes, ncb, .false.)
        do kc = 1, ncb
            clo = cboxes(kc)%lo; chi = cboxes(kc)%hi
            call s_amr_nest_clamp(clo, chi, mlo, mhi, amr_buf)
            if (ib) then
                call s_amr_expand_box_over_bodies(clo, chi)
                call s_amr_nest_clamp(clo, chi, mlo, mhi, 0)
            end if
            call s_amr_nest_emit(clo, chi, lev, kb, mych, nmych)
        end do
        if (allocated(cboxes)) deallocate (cboxes)

    end subroutine s_amr_nest_cluster_children

    !> Brand-new region (no old fine data yet): a centred inset so the child still appears this regrid.
    impure subroutine s_amr_nest_inset_child(box, lev, kb, mych, nmych)

        type(t_box), intent(in) :: box
        integer, intent(in)     :: lev, kb
        integer, intent(inout)  :: mych(:,:), nmych
        integer                 :: ins(3), clo(3), chi(3)

        ins = merge(max((box%hi - box%lo + 1)/4, amr_cpat_mar), 0, amr_dim)
        clo = box%lo + ins; chi = box%hi - ins
        if (f_amr_nest_window_empty(clo, chi)) return
        call s_amr_nest_emit(clo, chi, lev, kb, mych, nmych)

    end subroutine s_amr_nest_inset_child

    !> Pad a child by `pad` and clamp it into the nesting window; collapsed dims are pinned to 0.
    pure subroutine s_amr_nest_clamp(clo, chi, mlo, mhi, pad)

        integer, intent(inout) :: clo(3), chi(3)
        integer, intent(in)    :: mlo(3), mhi(3), pad

        clo = merge(max(clo - pad, mlo), 0, amr_dim); chi = merge(min(chi + pad, mhi), 0, amr_dim)

    end subroutine s_amr_nest_clamp

    !> Tile a child to the level's slot cap, amr_maxc_fit/amr_ref_ratio**(lev-1) coarse cells, and append the tiles to this rank's
    !! list.
    impure subroutine s_amr_nest_emit(clo, chi, lev, kb, mych, nmych)

        integer, intent(in)    :: clo(3), chi(3), lev, kb
        integer, intent(inout) :: mych(:,:), nmych
        type(t_box)            :: tiles(amr_max_blocks)
        integer                :: nt, capped, it

        nt = 0; capped = 0
        call s_amr_tile_box(clo, chi, tiles, nt, amr_max_blocks, capped, amr_maxc_fit/amr_ref_ratio**(lev - 1))
        do it = 1, nt
            if (nmych + 1 > amr_max_blocks) then
                ! refinement is dropped: warn once
                if (.not. amr_child_dropped) then
                    write (0, '(A,I0,A)') ' [amr] WARNING: nested refinement dropped, box capacity amr_max_blocks = ', &
                           & amr_max_blocks, ' reached; raise amr_max_blocks'
                    amr_child_dropped = .true.
                end if
                exit
            end if
            nmych = nmych + 1
            mych(1:3,nmych) = tiles(it)%lo; mych(4:6,nmych) = tiles(it)%hi; mych(7, nmych) = kb
        end do

    end subroutine s_amr_nest_emit

    !> Assemble every rank's children (7 ints each: lo, hi, parent kb) and order them by parent kb with a stable counting sort:
    !! chord(1:ntot) indexes gch in the canonical (kb ascending, emission) order.
    impure subroutine s_amr_nest_gather_children(mych, nmych, plo, phi, gch, chord, ntot)

        integer, intent(in)               :: mych(:,:), nmych, plo, phi
        integer, allocatable, intent(out) :: gch(:,:), chord(:)
        integer, intent(out)              :: ntot
        integer, allocatable              :: chhead(:)
        integer                           :: ich, jch, kb

#ifdef MFC_MPI
        integer              :: ip, ierr
        integer, allocatable :: rcnt(:), rdsp(:)

        if (num_procs > 1) then
            allocate (rcnt(num_procs), rdsp(num_procs))
            call MPI_ALLGATHER(nmych, 1, MPI_INTEGER, rcnt, 1, MPI_INTEGER, MPI_COMM_WORLD, ierr)
            rdsp(1) = 0
            do ip = 2, num_procs
                rdsp(ip) = rdsp(ip - 1) + rcnt(ip - 1)
            end do
            ntot = rdsp(num_procs) + rcnt(num_procs)
            allocate (gch(7, max(ntot, 1)))
            rcnt = rcnt*7; rdsp = rdsp*7
            call MPI_ALLGATHERV(mych, nmych*7, MPI_INTEGER, gch, rcnt, rdsp, MPI_INTEGER, MPI_COMM_WORLD, ierr)
        else
#endif
            ntot = nmych
            allocate (gch(7, max(ntot, 1))); gch(:,1:ntot) = mych(:,1:ntot)
#ifdef MFC_MPI
        end if
#endif
        allocate (chhead(plo:phi), chord(max(ntot, 1)))
        chhead = 0
        do ich = 1, ntot
            chhead(gch(7, ich)) = chhead(gch(7, ich)) + 1
        end do
        jch = 1
        do kb = plo, phi
            ich = chhead(kb); chhead(kb) = jch; jch = jch + ich
        end do
        do ich = 1, ntot
            kb = gch(7, ich)
            chord(chhead(kb)) = ich; chhead(kb) = chhead(kb) + 1
        end do

    end subroutine s_amr_nest_gather_children

    !> True iff the box set and levels equal the live slots (a rebuild would reproduce them).
    impure subroutine s_amr_regrid_boxes_unchanged(boxes, nboxes, box_level, same)

        type(t_box), intent(in) :: boxes(:)
        integer, intent(in)     :: nboxes, box_level(:)
        logical, intent(out)    :: same
        integer                 :: k, ks

        same = .false.
        if (nboxes == amr_num_blocks) then
            same = .true.
            do k = 1, nboxes
                if (any(boxes(k)%lo /= amr_slots(k)%region%lo) .or. any(boxes(k)%hi /= amr_slots(k)%region%hi) .or. box_level(k) &
                    & /= amr_block_level(k)) same = .false.
            end do
        end if

    end subroutine s_amr_regrid_boxes_unchanged

    !> Pack fine window [l, h] of an owned old block's stash (device store) into its wire slice buf; gi fastest, then gj, gk, ii.
    impure subroutine s_amr_mig_pack_device(loc, l1, l2, l3, h1, h2, h3, buf)

        integer, intent(in)                 :: loc, l1, l2, l3, h1, h2, h3
        real(wp), intent(inout), contiguous :: buf(:)
        integer                             :: ii, gk, gj, gi, n1, n2, n3

        n1 = h1 - l1 + 1; n2 = h2 - l2 + 1; n3 = h3 - l3 + 1
        $:GPU_PARALLEL_LOOP(collapse=4, copyout='[buf]')
        do ii = 1, sys_size
            do gk = l3, h3
                do gj = l2, h2
                    do gi = l1, h1
                        buf(1 + gi - l1 + n1*(gj - l2 + n2*(gk - l3 + n3*(ii - 1)))) = real(amr_stor_st(gi, gj, gk, ii, loc), wp)
                    end do
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

    end subroutine s_amr_mig_pack_device

    !> Fine-index window [flo, fhi] in level-1 old block ilo's stash of the coarse box [wlo, whi] (0:0 in collapsed dims).
    pure subroutine s_amr_mig_window(wlo, whi, ilo, flo, fhi)

        integer, intent(in)  :: wlo(3), whi(3), ilo(3)
        integer, intent(out) :: flo(3), fhi(3)

        flo = merge(amr_ref_ratio*(wlo - ilo), 0, amr_dim)
        fhi = merge(amr_ref_ratio*(whi - ilo + 1) - 1, 0, amr_dim)

    end subroutine s_amr_mig_window

    !> Free a migration pool (device-resident when amr_fw_dev).
    impure subroutine s_amr_mig_free(pool)

        real(wp), allocatable, intent(inout) :: pool(:)

        if (.not. allocated(pool)) return
        if (amr_fw_dev) then
            $:GPU_EXIT_DATA(delete='[pool]')
        end if
        deallocate (pool)

    end subroutine s_amr_mig_free

    !> Device cons->stor stash copy of one owned old block's fine interior (the store is device-authoritative; no host staging).
    impure subroutine s_amr_stash_copy_device(loc, e1, e2, e3)

        integer, intent(in) :: loc, e1, e2, e3
        integer             :: ii, gk, gj, gi

        $:GPU_PARALLEL_LOOP(collapse=4)
        do ii = 1, sys_size
            do gk = 0, e3
                do gj = 0, e2
                    do gi = 0, e1
                        amr_stor_st(gi, gj, gk, ii, loc) = amr_cons_st(gi, gj, gk, ii, loc)
                    end do
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

    end subroutine s_amr_stash_copy_device

    !> Overwrite the new block's prolonged cons with the covering old block's stashed detail (old fine index = new + sh) over the
    !! old window [l, h], from store column vold or the received wire slice buf.
    #:for SRC in ['st', 'wire']
        impure subroutine s_amr_overlap_copy_${SRC}$(vloc, vm, vn, vp, sh, l1, l2, l3, h1, h2, h3, &
            & #{if SRC == 'st'}#vold#{else}#buf#{endif}#)

            integer, intent(in) :: vloc, vm, vn, vp, sh(3), l1, l2, l3, h1, h2, h3

            #:if SRC == 'st'
                integer, intent(in) :: vold
            #:else
                real(wp), intent(in), contiguous :: buf(:)
            #:endif
            integer :: i, fi, fj, fk, ofi, ofj, ofk, sh1, sh2, sh3, n1, n2, n3
            logical :: vd2, vd3

            sh1 = sh(1); sh2 = sh(2); sh3 = sh(3)
            n1 = h1 - l1 + 1; n2 = h2 - l2 + 1; n3 = h3 - l3 + 1
            vd2 = n_glb > 0; vd3 = p_glb > 0
            #:if SRC == 'st'
                $:GPU_PARALLEL_LOOP(collapse=4, private='[ofi, ofj, ofk]')
            #:else
                $:GPU_PARALLEL_LOOP(collapse=4, private='[ofi, ofj, ofk]', copyin='[buf]')
            #:endif
            do i = 1, sys_size
                do fk = 0, vp
                    do fj = 0, vn
                        do fi = 0, vm
                            ofk = fk + sh3; ofj = fj + sh2; ofi = fi + sh1
                            if (vd3 .and. (ofk < l3 .or. ofk > h3)) cycle
                            if (vd2 .and. (ofj < l2 .or. ofj > h2)) cycle
                            if (ofi < l1 .or. ofi > h1) cycle
                            #:if SRC == 'st'
                                amr_cons_st(fi, fj, fk, i, vloc) = amr_stor_st(ofi, ofj, ofk, i, vold)
                            #:else
                                amr_cons_st(fi, fj, fk, i, &
                                            & vloc) = real(buf(1 + ofi - l1 + n1*(ofj - l2 + n2*(ofk - l3 + n3*(i - 1)))), stp)
                            #:endif
                        end do
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()

        end subroutine s_amr_overlap_copy_${SRC}$
    #:endfor

    !> Regrid phase 5: stash live slots' fine interiors, record the old block set (old_*), commit the new regions/levels/owners, and
    !! migrate stashed old blocks point-to-point to the ranks owning overlapping new blocks.
    impure subroutine s_amr_regrid_stash_migrate(boxes, nboxes, box_level, old_np, old_ilo, old_ext, old_level, old_owns)

        type(t_box), intent(in) :: boxes(:)
        integer, intent(in)     :: nboxes, box_level(:)
        integer, intent(out)    :: old_np, old_ilo(:,:), old_ext(:,:), old_level(:)
        logical, intent(out)    :: old_owns(:)
        integer                 :: old_chi(3, amr_max_blocks), old_owner(amr_max_blocks)
        integer                 :: k, ks, kk, k2, rr, cnt, ix, lo, hi, flo(3), fhi(3)
        integer, allocatable    :: wlo(:,:), whi(:,:)  !< per-rank overlap window of the current old block

        ! NVHPC -Minline: plain scalars for device routines
        integer :: loc_ks, old_ext1_k, old_ext2_k, old_ext3_k

        call s_phase_tic(PH_RG_OWN)
        old_np = amr_num_blocks
        do k = 1, old_np
            ks = k
            ! replicated global origin/extent (valid on every rank, unlike the owner-only isect)
            old_ilo(:,k) = amr_region_lo_all(:,ks)
            old_chi(:,k) = amr_region_hi_all(:,ks)
            ! fine extent = amr_ref_ratio**level*footprint - 1 (a level-2 block is 4x its footprint)
            old_ext(:,k) = merge(amr_ref_ratio**amr_block_level(ks)*(old_chi(:,k) - old_ilo(:,k) + 1) - 1, 0, amr_dim)
            old_owner(k) = amr_block_owner(ks)
            ! the overlap copy must match levels
            old_level(k) = amr_block_level(ks)
            old_owns(k) = amr_owns_all(ks)
            if (old_owns(k)) then
                ! device stash: the store is device-authoritative; s_amr_st_reserve preserves it across a mid-rebuild grow.
                ! amdflang: the kernel is its own routine because target regions nested in BLOCK drop from the device image (HSA
                ! invalid symbol)
                loc_ks = amr_loc_of(ks)
                old_ext1_k = old_ext(1, k)
                old_ext2_k = old_ext(2, k)
                old_ext3_k = old_ext(3, k)
                call s_amr_stash_copy_device(loc_ks, old_ext1_k, old_ext2_k, old_ext3_k)
            end if
        end do

        ! regions and owners first: the migration needs the new owners and s_set_amr_fine_geometry reads amr_block_owner
        amr_num_blocks = nboxes
        do k = 1, nboxes
            ks = k
            amr_region_lo_all(:,ks) = boxes(k)%lo; amr_region_hi_all(:,ks) = boxes(k)%hi
            amr_block_level(ks) = box_level(k)
        end do
        ! dirty the seam-pair and overlap-rank caches before the rebuild's gathers consume them
        amr_seam_pairs_dirty = .true.
        amr_mesh_epoch = amr_mesh_epoch + 1
        ! proper nesting: a level >= 2 block must overlap exactly one parent-level block (f_amr_parent_block takes the first)
        block
            integer :: bk, bkk, npar
            do bk = 1, nboxes
                if (box_level(bk) < 2) cycle
                npar = 0
                do bkk = 1, nboxes
                    if (box_level(bkk) == box_level(bk) - 1 .and. f_amr_boxes_overlap(boxes(bk)%lo, boxes(bk)%hi, boxes(bkk)%lo, &
                        & boxes(bkk)%hi)) npar = npar + 1
                end do
                if (npar /= 1) call s_mpi_abort('amr multi-level: a level>=2 block overlaps more than one (or no) ' &
                    & // 'parent-level block - a fine tile straddling a parent-tile seam is unsupported (gather/reflux ' &
                    & // 'couple to a single parent); reduce max_grid_size or the refined feature extent')
            end do
        end block
        amr_num_levels = maxval(box_level(1:nboxes))
        call s_amr_assign_block_owners()
        call s_phase_toc(PH_RG_OWN)

        ! Cross-rank migration as one wave: each old owner ships the bounding box of its overlaps to every other rank owning an
        ! overlapping new level-1 block (level >= 2 re-prolongs from its parent); both sides enumerate old blocks ascending, so the
        ! transfers line up. Received windows stay in amr_mig_rq. No-op at np=1.
        call s_phase_tic(PH_RG_MIG)
        allocate (amr_mig_win(7, old_np)); amr_mig_win = 0
        if (num_procs > 1) then
            allocate (wlo(3,0:num_procs - 1), whi(3,0:num_procs - 1))
            wlo = huge(0); whi = -huge(0)
            call s_amr_wave_open(amr_wave, 4)
            call s_amr_wave_reset(amr_wsend); call s_amr_wave_reset(amr_wrecv)
            do kk = 1, old_np
                if (old_level(kk) /= 1) cycle
                do k2 = 1, nboxes
                    rr = amr_block_owner(k2)
                    if (box_level(k2) /= 1 .or. rr == old_owner(kk)) cycle
                    if (.not. f_amr_boxes_overlap(boxes(k2)%lo, boxes(k2)%hi, old_ilo(:,kk), old_chi(:,kk))) cycle
                    wlo(:,rr) = min(wlo(:,rr), max(boxes(k2)%lo, old_ilo(:,kk)))
                    whi(:,rr) = max(whi(:,rr), min(boxes(k2)%hi, old_chi(:,kk)))
                end do
                do rr = 0, num_procs - 1
                    if (whi(1, rr) < wlo(1, rr)) cycle
                    call s_amr_mig_window(wlo(:,rr), whi(:,rr), old_ilo(:,kk), flo, fhi)
                    cnt = sys_size*product(fhi - flo + 1)
                    if (rr == proc_rank) then
                        call s_amr_wave_add(amr_wrecv, old_owner(kk), kk, wlo(:,rr), whi(:,rr), cnt)
                    else if (old_owns(kk)) then
                        call s_amr_wave_add(amr_wsend, rr, kk, wlo(:,rr), whi(:,rr), cnt)
                    end if
                    wlo(:,rr) = huge(0); whi(:,rr) = -huge(0)
                end do
            end do
            deallocate (wlo, whi)
            call s_amr_wave_close(amr_wsend, amr_mig_sq, amr_fw_dev)
            call s_amr_wave_close(amr_wrecv, amr_mig_rq, amr_fw_dev)
            call s_amr_wave_post(amr_wave, amr_wrecv, amr_mig_rq, amr_fw_dev)
            do ix = 1, amr_wsend%nx
                kk = amr_wsend%blk(ix)
                call s_amr_wave_slice(amr_wsend, ix, lo, hi)
                call s_amr_mig_window(amr_wsend%bl(:,ix), amr_wsend%bh(:,ix), old_ilo(:,kk), flo, fhi)
                call s_amr_mig_pack_device(amr_loc_of(kk), flo(1), flo(2), flo(3), fhi(1), fhi(2), fhi(3), amr_mig_sq(lo:hi))
            end do
            call s_amr_wave_send(amr_wave, amr_wsend, amr_mig_sq, amr_fw_dev)
            call s_amr_wave_wait(amr_wave)
            call s_amr_mig_free(amr_mig_sq)
            do ix = 1, amr_wrecv%nx
                kk = amr_wrecv%blk(ix)
                call s_amr_wave_slice(amr_wrecv, ix, lo, hi)
                call s_amr_mig_window(amr_wrecv%bl(:,ix), amr_wrecv%bh(:,ix), old_ilo(:,kk), flo, fhi)
                amr_mig_win(:,kk) = [lo, flo, fhi]
            end do
        end if
        call s_phase_toc(PH_RG_MIG)

    end subroutine s_amr_regrid_stash_migrate

    !> Regrid phase 6: build each new slot level by level (prolong from the coarse/parent patch, overwrite the overlap from covering
    !! stashed old blocks), then reconcile the slot pool, rebuild the fine IB state and re-check the seam topology. Old-only slots
    !! are freed after their last reader so device memory does not peak.
    impure subroutine s_amr_regrid_rebuild_slots(q_cons_base, boxes, nboxes, old_np, old_ilo, old_ext, old_level, old_owns)

        type(scalar_field), dimension(sys_size), intent(inout) :: q_cons_base
        type(t_box), intent(in)                                :: boxes(:)
        integer, intent(in)                                    :: nboxes, old_np, old_ilo(:,:), old_ext(:,:), old_level(:)
        logical, intent(in)                                    :: old_owns(:)
        integer                                                :: sh(3), k, kk, i, hh, ks, kks, lev, nh, ohi(3), pos, nlev
        integer, allocatable                                   :: last_use(:), held(:), held_hi(:,:), vpos(:)

        ! NVHPC -Minline: plain scalars for device routines
        integer :: loc_kks, loc_ks, m_ks, n_ks, old_ext1_kk, old_ext2_kk, old_ext3_kk, p_ks, w1, w2, w3, w4, w5, w6, lo, hi

        call s_phase_tic(PH_RG_BUILD)
        do k = 1, nboxes
            ks = k
            if (amr_block_owner(ks) == proc_rank) cycle
            amr_cur = ks
            call s_set_amr_fine_geometry(boxes(k)%lo, boxes(k)%hi)
        end do

        ! visit order (owned boxes, level-major) and each old block's last reader; nlev is replicated so every rank enters every
        ! wave
        call s_amr_refresh_my_blocks()
        nlev = maxval(amr_block_level(1:amr_num_blocks))
        allocate (vpos(nboxes)); vpos = 0; pos = 0
        do lev = 1, nlev
            do i = 1, amr_n_my
                k = amr_my_blk(i)
                if (amr_block_level(amr_my_blk(i)) /= lev) cycle
                pos = pos + 1; vpos(k) = pos
            end do
        end do
        allocate (last_use(old_np), held(old_np), held_hi(3, old_np)); last_use = 0
        nh = 0
        do kk = 1, old_np
            if (.not. amr_slot_live(kk)) cycle
            nh = nh + 1; held(nh) = kk
            ohi = old_ilo(:,kk) + merge((old_ext(:,kk) + 1)/amr_ref_ratio**old_level(kk) - 1, 0, amr_dim)
            held_hi(:,nh) = ohi
            do i = 1, amr_n_my
                k = amr_my_blk(i)
                if (f_amr_boxes_overlap(boxes(k)%lo, boxes(k)%hi, old_ilo(:,kk), ohi)) last_use(kk) = max(last_use(kk), vpos(k))
            end do
        end do

        do lev = 1, nlev
            if (lev == 1) then
                call s_amr_l1_fill_exchange(q_cons_base, .true.)
            else
                call s_amr_parent_fill_exchange(lev, .true.)
            end if
            do i = 1, amr_n_my
                ks = amr_my_blk(i)
                k = ks
                if (amr_block_level(ks) /= lev) cycle
                amr_cur = ks
                ! free old-only slots no later box reads (no-op if dead; slots serving new owned boxes live on)
                do hh = 1, nh
                    kk = held(hh)
                    if (last_use(kk) >= vpos(k)) cycle
                    kks = kk
                    if (kks <= amr_num_blocks) then
                        if (amr_block_owner(kks) == proc_rank) cycle
                    end if
                    call s_amr_free_slot(kks)
                end do
                call s_amr_alloc_slot(ks)
                call s_set_amr_fine_geometry(boxes(k)%lo, boxes(k)%hi)
                call s_amr_select_slot(ks)
                if (lev == 1) then
                    call s_amr_l1_fill_consume(q_cons_base, ks, .true.)
                else
                    call s_amr_parent_fill_consume(ks, .true.)
                end if
                ! level >= 2 re-prolongs from its freshly built parent (its stash is in the parent-fine frame): no carry-forward
                call s_interpolate_coarse_to_fine()
                if (lev >= 2) cycle
                loc_ks = amr_loc_of(ks)
                m_ks = amr_slots(ks)%m
                n_ks = amr_slots(ks)%n
                p_ks = amr_slots(ks)%p
                do kk = 1, old_np
                    if (old_level(kk) /= 1) cycle  ! a child's stash is in the 4x frame
                    ohi = old_ilo(:,kk) + merge((old_ext(:,kk) + 1)/amr_ref_ratio - 1, 0, amr_dim)
                    if (.not. f_amr_boxes_overlap(boxes(k)%lo, boxes(k)%hi, old_ilo(:,kk), ohi)) cycle
                    sh = amr_ref_ratio*(amr_isect_lo - old_ilo(:,kk))  ! old local fine index = new local fine index + sh
                    if (amr_mig_win(1, kk) > 0) then
                        ! owned elsewhere: window in the migration receive pool
                        lo = amr_mig_win(1, kk); hi = lo + sys_size*product(amr_mig_win(5:7,kk) - amr_mig_win(2:4,kk) + 1) - 1
                        w1 = amr_mig_win(2, kk); w2 = amr_mig_win(3, kk); w3 = amr_mig_win(4, kk)
                        w4 = amr_mig_win(5, kk); w5 = amr_mig_win(6, kk); w6 = amr_mig_win(7, kk)
                        call s_amr_overlap_copy_wire(loc_ks, m_ks, n_ks, p_ks, sh, w1, w2, w3, w4, w5, w6, amr_mig_rq(lo:hi))
                    else if (amr_slot_live(kk)) then
                        loc_kks = amr_loc_of(kk)
                        old_ext1_kk = old_ext(1, kk)
                        old_ext2_kk = old_ext(2, kk)
                        old_ext3_kk = old_ext(3, kk)
                        call s_amr_overlap_copy_st(loc_ks, m_ks, n_ks, p_ks, sh, 0, 0, 0, old_ext1_kk, old_ext2_kk, old_ext3_kk, &
                                                   & loc_kks)
                    end if
                end do
            end do
            call s_amr_fill_wave_done()
        end do
        deallocate (last_use, held, held_hi, vpos, amr_mig_win)
        call s_amr_mig_free(amr_mig_rq)
        call s_phase_toc(PH_RG_BUILD)
        call s_amr_report_mem('built')
        call s_phase_tic(PH_RG_REC)

        ! sets amr_xchg_coarse_ghosts if any block needs it
        call s_amr_reduce_xchg_flag()
        ! free the transient slots of old blocks this rank does not now own
        call s_amr_reconcile_slots()
        ! rebuild fine IB state for the new geometry
        call s_amr_report_mem('pre-ib-setup')
        if (ib) call s_amr_setup_ib()
        call s_amr_report_mem('post-ib-setup')
        call s_amr_select_slot(1)
        call s_amr_check_seam_topology()
        call s_phase_toc(PH_RG_REC)

    end subroutine s_amr_regrid_rebuild_slots

    !> Sensor-on-fine child tagging: OR the density-gradient tags of an old fine block into an L0 tag grid over a parent window.
    !! Reads the flat store on the host (caller refreshes it); one-sided differences at the fine edges avoid stale ghosts.
    impure subroutine s_amr_tag_child_from_fine(ob, win_lo, win_hi, ctag, any_tag)

        integer, intent(in)    :: ob, win_lo(3), win_hi(3)
        logical, intent(inout) :: ctag(win_lo(1):,win_lo(2):,win_lo(3):)
        logical, intent(inout) :: any_tag
        integer                :: rr, ci, cj, ck, fi, fj, fk, d1, d2, d3, fm1, fm2, fm3, olo(3), lo(3), hi(3)
        real(wp)               :: r0, g
        logical                :: tagged

        rr = amr_slots(ob)%amr_ref_ratio
        olo = amr_region_lo_all(:,ob)
        fm1 = amr_slots(ob)%m; fm2 = amr_slots(ob)%n; fm3 = amr_slots(ob)%p
        lo = merge(max(win_lo, amr_region_lo_all(:,ob)), 0, amr_dim); hi = merge(min(win_hi, amr_region_hi_all(:,ob)), 0, amr_dim)
        do ck = lo(3), hi(3)
            do cj = lo(2), hi(2)
                do ci = lo(1), hi(1)
                    tagged = .false.
                    do d3 = 0, merge(rr - 1, 0, p_glb > 0)
                        fk = (ck - olo(3))*rr + d3
                        do d2 = 0, merge(rr - 1, 0, n_glb > 0)
                            fj = (cj - olo(2))*rr + d2
                            do d1 = 0, rr - 1
                                fi = (ci - olo(1))*rr + d1
                                r0 = f_amr_rho_tot_st(amr_loc_of(ob), fi, fj, fk)
                                g = f_amr_tag_ind(f_amr_rho_tot_st(amr_loc_of(ob), min(fi + 1, fm1), fj, fk), r0, &
                                                  & f_amr_rho_tot_st(amr_loc_of(ob), max(fi - 1, 0), fj, fk))
                                if (n_glb > 0) g = max(g, f_amr_tag_ind(f_amr_rho_tot_st(amr_loc_of(ob), fi, min(fj + 1, fm2), &
                                    & fk), r0, f_amr_rho_tot_st(amr_loc_of(ob), fi, max(fj - 1, 0), fk)))
                                if (p_glb > 0) g = max(g, f_amr_tag_ind(f_amr_rho_tot_st(amr_loc_of(ob), fi, fj, min(fk + 1, &
                                    & fm3)), r0, f_amr_rho_tot_st(amr_loc_of(ob), fi, fj, max(fk - 1, 0))))
                                if (g > amr_tag_eps) tagged = .true.
                            end do
                        end do
                    end do
                    if (tagged) then
                        ctag(ci, cj, ck) = .true.
                        any_tag = .true.
                    end if
                end do
            end do
        end do

    end subroutine s_amr_tag_child_from_fine

    !> Total density (sum of the continuity variables) at one cell; `_st` reads the flat store, `_sf` the level-0 field.
    #:for RSFX, RSRC in [('st', 'amr_cons_st'), ('sf', '')]
        pure function f_amr_rho_tot_${RSFX}$(${'loc' if RSRC else 'q'}$, ci, cj, ck) result(r)

            #:if RSRC
                integer, intent(in) :: loc  !< flat-store slot
            #:else
                type(scalar_field), dimension(:), intent(in) :: q
            #:endif
            integer, intent(in) :: ci, cj, ck
            real(wp)            :: r
            integer             :: f

            r = 0._wp
            do f = eqn_idx%cont%beg, eqn_idx%cont%end
                #:if RSRC
                    r = r + real(amr_cons_st(ci, cj, ck, f, loc), wp)
                #:else
                    r = r + real(q(f)%sf(ci, cj, ck), wp)
                #:endif
            end do

        end function f_amr_rho_tot_${RSFX}$
    #:endfor

    !> Tag indicator from the densities at i+1, i, i-1: central difference over 2*rho_i, or with amr_tag_abs the larger absolute
    !! face jump.
    pure real(wp) function f_amr_tag_ind(rp, r0, rm) result(v)
        real(wp), intent(in) :: rp, r0, rm

        if (amr_tag_abs) then
            v = max(abs(rp - r0), abs(r0 - rm))
        else
            v = abs(rp - rm)/(2._wp*max(abs(r0), 1.e-30_wp))
        end if

    end function f_amr_tag_ind

end module m_amr_regrid
