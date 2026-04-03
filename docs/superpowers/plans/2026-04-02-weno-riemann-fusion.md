# WENO+Riemann Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate qL_rs_vf and qR_rs_vf (47 MB) by fusing WENO reconstruction into the Riemann solver GPU kernel, so reconstructed states are computed per-face in registers rather than stored in global GPU memory.

**Architecture:** Instead of the current two-phase pipeline (WENO kernel → store to qL/R_rs_vf → Riemann kernel reads qL/R_rs_vf), we create a single fused kernel where each GPU thread reconstructs left/right states for its face using inline WENO, then immediately computes the Riemann flux. The reconstructed states live in GPU registers (16 doubles per thread) instead of global memory. The WENO workspace v_rs_ws must be un-aliased from flux_rs_vf since the fused kernel reads v_rs_ws and writes flux_rs_vf simultaneously.

**Tech Stack:** Fortran 2008 + Fypp preprocessor + OpenACC (GPU_* macros)

**Memory impact:** Eliminates qL_rs_vf (24 MB) + qR_rs_vf (24 MB) = 48 MB. Adds back v_rs_ws as separate allocation (~24 MB, was previously shared with flux_rs_vf). Net savings: ~24 MB. Additionally eliminates qL_prim(1) + qR_prim(1) (18 MB, no longer needed when reconstruction is inline). **Total net: ~42 MB.**

---

### Task 1: Un-alias v_rs_ws from flux_rs_vf

**Files:**
- Modify: `src/simulation/m_weno.fpp` — restore v_rs_ws as independent allocation
- Modify: `src/simulation/m_muscl.fpp` — restore v_rs_ws_muscl as independent allocation
- Modify: `src/simulation/m_riemann_solvers.fpp` — restore flux_rs_vf as independent allocation

**Rationale:** In the fused kernel, the WENO stencil reads from v_rs_ws while the Riemann solver writes to flux_rs_vf. These cannot be the same array. Revert the sharing from commit cce0b302 (which was already reverted once due to CMake issues, then re-applied).

- [ ] **Step 1:** In m_weno.fpp, remove `use m_riemann_solvers, only: v_rs_ws => flux_rs_vf`. Restore `v_rs_ws` as a module-level allocatable with GPU_DECLARE. Restore the allocation in s_initialize_weno (max-bounds shape: `-buff_size - weno_polyn : max(m,n,p) + buff_size + weno_polyn` in dim1, `-buff_size : max(m,n,p) + buff_size` in dims 2,3, `1:sys_size` in dim4). Restore deallocation in s_finalize_weno_module.

- [ ] **Step 2:** Same for m_muscl.fpp — restore v_rs_ws_muscl as independent allocation.

- [ ] **Step 3:** In m_riemann_solvers.fpp, restore flux_rs_vf as a module-level allocatable (remove the use-rename import if present). Restore its allocation at `(-1:max(m,n,p), 0:max(m,n,p), 0:max(m,n,p), 1:sys_size)` in s_initialize_riemann_solvers_module. Restore deallocation in s_finalize_riemann_solvers_module.

- [ ] **Step 4:** Build and test: `source ./mfc.sh load -c p -m g && ./mfc.sh build -j 10 --gpu acc -t simulation`. Run the mem_profile test case to verify correctness.

- [ ] **Step 5:** Commit: `git add src/simulation/m_weno.fpp src/simulation/m_muscl.fpp src/simulation/m_riemann_solvers.fpp && git commit -m "un-alias v_rs_ws from flux_rs_vf for fusion preparation"`

---

### Task 2: Create inline WENO5 device subroutine

**Files:**
- Create: `src/simulation/include/inline_weno.fpp` — Fypp include file with inline WENO5 reconstruction

**Rationale:** Extract the WENO5 stencil computation into a device-callable form that can be inlined into the Riemann solver kernel. Uses `GPU_ROUTINE(parallelism='[seq]')` to mark it as callable from GPU threads.

- [ ] **Step 1:** Create `src/simulation/include/inline_weno.fpp` containing a subroutine:

```fortran
!> Inline WENO5 reconstruction of left and right states at face j
!! for all variables. Reads from workspace v_ws, writes to local arrays qL, qR.
$:GPU_ROUTINE(parallelism='[seq]')
subroutine s_inline_weno5_reconstruct(v_ws, j, k, l, nv, &
        & poly_cL, poly_cR, d_cL, d_cR, beta_c, &
        & is1_beg, qL, qR)

    real(wp), dimension(:,:,:,:), intent(in) :: v_ws
    integer, intent(in) :: j, k, l, nv, is1_beg
    real(wp), dimension(:,:,:), intent(in) :: poly_cL, poly_cR
    real(wp), dimension(:,:), intent(in) :: d_cL, d_cR
    real(wp), dimension(:,:,:), intent(in) :: beta_c
    real(wp), dimension(nv), intent(out) :: qL, qR

    real(wp) :: dvd(-2:1), poly(0:2), beta(0:2), alpha(0:2), omega(0:2), tau
    integer :: i

    do i = 1, nv
        ! Value differences for WENO5 stencil
        dvd(1)  = v_ws(j + 2, k, l, i) - v_ws(j + 1, k, l, i)
        dvd(0)  = v_ws(j + 1, k, l, i) - v_ws(j, k, l, i)
        dvd(-1) = v_ws(j, k, l, i)     - v_ws(j - 1, k, l, i)
        dvd(-2) = v_ws(j - 1, k, l, i) - v_ws(j - 2, k, l, i)

        ! Left reconstruction
        poly(0) = v_ws(j, k, l, i) + poly_cL(j, 0, 0)*dvd(0) + poly_cL(j, 0, 1)*dvd(1)
        poly(1) = v_ws(j, k, l, i) + poly_cL(j, 1, 0)*dvd(-1) + poly_cL(j, 1, 1)*dvd(0)
        poly(2) = v_ws(j, k, l, i) + poly_cL(j, 2, 0)*dvd(-2) + poly_cL(j, 2, 1)*dvd(-1)

        beta(0) = beta_c(j, 0, 0)*dvd(0)*dvd(0)   + beta_c(j, 0, 1)*dvd(0)*dvd(1)   + beta_c(j, 0, 2)*dvd(1)*dvd(1) + weno_eps
        beta(1) = beta_c(j, 1, 0)*dvd(-1)*dvd(-1)  + beta_c(j, 1, 1)*dvd(-1)*dvd(0)  + beta_c(j, 1, 2)*dvd(0)*dvd(0) + weno_eps
        beta(2) = beta_c(j, 2, 0)*dvd(-2)*dvd(-2)  + beta_c(j, 2, 1)*dvd(-2)*dvd(-1) + beta_c(j, 2, 2)*dvd(-1)*dvd(-1) + weno_eps

        tau = abs(beta(2) - beta(0))
        alpha(0) = d_cL(0, j)*(1._wp + tau/beta(0))
        alpha(1) = d_cL(1, j)*(1._wp + tau/beta(1))
        alpha(2) = d_cL(2, j)*(1._wp + tau/beta(2))
        omega = alpha/sum(alpha)
        qL(i) = omega(0)*poly(0) + omega(1)*poly(1) + omega(2)*poly(2)

        ! Right reconstruction
        poly(0) = v_ws(j, k, l, i) + poly_cR(j, 0, 0)*dvd(0) + poly_cR(j, 0, 1)*dvd(1)
        poly(1) = v_ws(j, k, l, i) + poly_cR(j, 1, 0)*dvd(-1) + poly_cR(j, 1, 1)*dvd(0)
        poly(2) = v_ws(j, k, l, i) + poly_cR(j, 2, 0)*dvd(-2) + poly_cR(j, 2, 1)*dvd(-1)

        alpha(0) = d_cR(0, j)*(1._wp + tau/beta(0))
        alpha(1) = d_cR(1, j)*(1._wp + tau/beta(1))
        alpha(2) = d_cR(2, j)*(1._wp + tau/beta(2))
        omega = alpha/sum(alpha)
        qR(i) = omega(0)*poly(0) + omega(1)*poly(1) + omega(2)*poly(2)
    end do

end subroutine s_inline_weno5_reconstruct
```

NOTE: The polynomial coefficient indexing above is approximate — verify against the actual WENO5 code in m_weno.fpp lines 1028-1138. The poly_coef indices and dvd offsets for left vs right reconstruction differ. Also handle WENO-Z vs WENO-JS vs mapped WENO based on the `wenoz`/`wenojs`/`mapped_weno` flags.

- [ ] **Step 2:** Build to verify the include file is syntactically correct.

- [ ] **Step 3:** Commit.

---

### Task 3: Add fused WENO+Riemann path to HLLC solver

**Files:**
- Modify: `src/simulation/m_riemann_solvers.fpp` — add fused kernel inside `s_hllc_riemann_solver`

**Rationale:** The HLLC solver's main GPU_PARALLEL_LOOP currently reads from `qL_prim_rs_vf(j,k,l,var)` and `qR_prim_rs_vf(j+1,k,l,var)`. Replace these reads with inline WENO calls that reconstruct from `v_rs_ws`. The WENO workspace v_rs_ws must be imported into m_riemann_solvers (or passed as argument).

- [ ] **Step 1:** Add `use m_weno, only: v_rs_ws` to m_riemann_solvers.fpp (or pass v_rs_ws as a new argument to s_riemann_solver and forward to s_hllc_riemann_solver).

- [ ] **Step 2:** In the HLLC solver's main Fypp `#:for NORM_DIR, XYZ` block, add a `#:if MFC_CASE_OPTIMIZATION` (or runtime `if`) branch for the fused path. In this branch:
  - Declare private arrays: `real(wp) :: qL_local(sys_size), qR_local(sys_size)`
  - Before the main loop, include WENO coefficient arrays for the current direction
  - Inside the GPU_PARALLEL_LOOP, for each face (j,k,l):
    a. Call `s_inline_weno5_reconstruct(v_rs_ws, j, k, l, sys_size, poly_coef_cbL_${XYZ}$, poly_coef_cbR_${XYZ}$, d_cbL_${XYZ}$, d_cbR_${XYZ}$, beta_coef_${XYZ}$, is1%beg, qL_local, qR_local)` for left state at face j
    b. Call same for right state at face j (note: right state at face j = left state at face j+1, so call with j+1)
    c. Replace all `qL_prim_rs_vf(j,k,l,var)` reads with `qL_local(var)`
    d. Replace all `qR_prim_rs_vf(j+1,k,l,var)` reads with `qR_local(var)`

- [ ] **Step 3:** Handle boundary conditions: for `j == is1%beg` (left boundary) and `j == is1%end` (right boundary), apply the same boundary extrapolation that `s_populate_riemann_states_variables_buffers` does, but inline. For `BC_RIEMANN_EXTRAP`: `qL_local = qR_local` at left boundary, `qR_local = qL_local` at right boundary.

- [ ] **Step 4:** Build and test.

- [ ] **Step 5:** Commit.

---

### Task 4: Update s_compute_rhs to use fused path

**Files:**
- Modify: `src/simulation/m_rhs.fpp` — skip separate WENO + boundary fill calls when using fused path

**Rationale:** With the fused kernel, `s_reconstruct_cell_boundary_values` and `s_populate_riemann_states_variables_buffers` are no longer needed. The fused Riemann solver does reconstruction internally. However, the WENO workspace (v_rs_ws) still needs to be filled by `s_initialize_weno`.

- [ ] **Step 1:** In the `do id = 1, num_dims` loop in s_compute_rhs, reorganize:
  - KEEP: `s_initialize_weno` call (fills v_rs_ws from q_prim_qp)
  - SKIP: `s_reconstruct_cell_boundary_values` calls (WENO is now inline in Riemann)
  - SKIP: `s_populate_riemann_states_variables_buffers` (boundary handling is inline)
  - KEEP: `s_riemann_solver` call (now contains the fused logic)

- [ ] **Step 2:** Remove `qL_rs_vf` and `qR_rs_vf` allocation, declaration, GPU_DECLARE, and deallocation from m_rhs.fpp. Remove `s_ensure_rs_allocated`. Remove `qL_rs_vf` and `qR_rs_vf` from the s_riemann_solver call signature.

- [ ] **Step 3:** Also remove `qL_prim(1)%vf(l)%sf` and `qR_prim(1)%vf(l)%sf` allocations (these were populated by the viscous reconstruction path which won't be called for non-viscous). For viscous cases, keep the original non-fused path as a fallback.

- [ ] **Step 4:** Build, run the mem_profile test case, verify correctness and performance.

- [ ] **Step 5:** Run nsys to verify memory reduction.

- [ ] **Step 6:** Commit.

---

### Task 5: Handle HLL, LF, HLLD solvers

**Files:**
- Modify: `src/simulation/m_riemann_solvers.fpp` — apply same fusion to other solvers

The HLLC solver is the most commonly used, but HLL, LF, and HLLD have the same qL/R_rs_vf access pattern. Apply the same inline WENO technique to each.

- [ ] **Step 1:** Repeat Task 3's changes for `s_hll_riemann_solver`.
- [ ] **Step 2:** Repeat for `s_lf_riemann_solver`.
- [ ] **Step 3:** Repeat for `s_hlld_riemann_solver`.
- [ ] **Step 4:** Build and test with each solver (change `riemann_solver` parameter in the test case).
- [ ] **Step 5:** Commit.

---

### Task 6: Handle MUSCL reconstruction variant

**Files:**
- Modify: `src/simulation/m_muscl.fpp` — create inline MUSCL reconstruction
- Modify: `src/simulation/m_riemann_solvers.fpp` — add MUSCL inline path

If the user selects MUSCL instead of WENO, the same fusion applies but with a simpler stencil (2 neighbors instead of 5). Create `s_inline_muscl_reconstruct` in `src/simulation/include/inline_muscl.fpp`.

- [ ] **Step 1:** Create inline MUSCL device subroutine.
- [ ] **Step 2:** Add MUSCL branch to fused Riemann solvers.
- [ ] **Step 3:** Build and test.
- [ ] **Step 4:** Commit.

---

### Task 7: Clean up and final verification

**Files:**
- Modify: All previously modified files

- [ ] **Step 1:** Run `./mfc.sh format -j 8` to auto-format.
- [ ] **Step 2:** Run `./mfc.sh precheck -j 8` to verify all lint checks pass.
- [ ] **Step 3:** Run the full 3D test: `./mfc.sh test --only 3D -j 8 -% 25`.
- [ ] **Step 4:** Run nsys to verify final memory: expect ~220 MB (down from 265 MB).
- [ ] **Step 5:** Final commit and push.

---

## Key Risks and Mitigations

1. **Register pressure**: The fused kernel has 16+ doubles of local state per thread. If register usage exceeds GPU limits, occupancy drops. Mitigation: Profile with `--ptxas-options=-v` to check register count.

2. **WENO coefficient access**: The inline WENO reads per-cell coefficients (poly_coef, d_cb, beta_coef). These are 1D arrays indexed by cell position — should be in L1/L2 cache. Mitigation: Verify performance doesn't regress.

3. **Boundary condition handling**: Inline BCs add branches to the hot loop. Mitigation: Only check at `j == is1%beg` and `j == is1%end` — minimal branch divergence.

4. **Viscous path**: The viscous code path uses qL_prim/qR_prim populated by a separate WENO call through `s_get_viscous`. Keep the non-fused path as fallback for viscous cases initially. Fuse later if needed.

5. **v_rs_ws read from m_riemann_solvers**: The Riemann solver module doesn't currently use m_weno. Adding `use m_weno` creates a dependency but NOT a circular one (m_weno doesn't use m_riemann_solvers after the un-aliasing in Task 1).
