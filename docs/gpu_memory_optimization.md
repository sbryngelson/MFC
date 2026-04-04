# GPU Memory Optimization — `mem-profile` Branch

## Summary

This branch reduces GPU memory per grid point by **93%** (1787 → 128 bytes/gp) for a
2-fluid 3D simulation using HLLC + WENO5 without viscosity. On a 110 GB GPU, this
increases the maximum problem size from **394³ to 962³** (14.6× more cells).

The optimization eliminates intermediate arrays by computing values on the fly,
pointer-aliasing redundant copies, and fusing separate GPU kernels into a single
per-cell kernel that needs no global intermediate storage.

### Results

| Metric                  | Baseline (master) | Optimized        | Change        |
|-------------------------|-------------------|------------------|---------------|
| Bytes per grid point    | 1787              | 128              | **−93%**      |
| Max grid on 110 GB GPU  | 394³ (61M cells)  | 962³ (890M cells)| **14.6×**     |
| nvidia-smi at N=64      | 1044 MiB          | 618 MiB          | −41%          |
| Remaining fields        | 79 scalar fields  | 16 scalar fields | −80%          |

The nvidia-smi improvement appears modest at N=64 because ~600 MiB is fixed CUDA/OpenACC
runtime overhead (measured at 312 MiB for an empty program, plus ~300 MiB of JIT-compiled
GPU kernel code). This overhead does not scale with grid size — at production scales
(N=256+), the per-grid-point reduction dominates.

### Applicability

The fused kernel path activates when ALL conditions hold:
- `riemann_solver == 2` (HLLC)
- `weno_order == 5`
- `.not. viscous`
- `model_eqns /= 3 .and. model_eqns /= 4`
- `.not. (model_eqns == 2 .and. bubbles_euler)`

All other configurations use the original (fallback) code path with full intermediate
arrays. The fallback path still benefits from the non-fused optimizations (pointer
aliasing, conditional allocation, Williamson RK3).

---

## Techniques Applied

### 1. Pointer Aliasing (eliminate redundant copies)

**Principle**: When two arrays hold identical data, make one a Fortran pointer to the
other instead of allocating separate storage. Use `$:GPU_ENTER_DATA(attach='[...]')` to
update the device-side pointer.

**Applied to**:

- **q_cons_qp → q_cons_vf** (`m_rhs.fpp`): The "quadrature point" conservative variables
  were a full copy of the time stepper state. Since `s_convert_conservative_to_primitive_variables`
  takes `q_cons` as `intent(in)` (read-only), the copy is unnecessary. Now q_cons_qp%vf(i)%sf
  points directly to q_cons_vf(i)%sf.

- **q_prim_qp momentum+energy → q_prim_vf** (`m_rhs.fpp`): The RHS module's primitive
  velocity/pressure fields alias the time stepper's q_prim_vf fields. The conversion writes
  through q_prim_qp, which updates q_prim_vf automatically.

- **q_prim_vf density+volfrac → q_cons_ts(1)** (`m_time_steppers.fpp`): For `model_eqns=2`,
  primitive density equals conservative density and primitive volume fractions equal
  conservative volume fractions. These q_prim_vf fields alias q_cons_ts(1) directly.

- **flux_gsrc_n dirs 2,3 → dir 1** (`m_rhs.fpp`): Geometric source fluxes are used
  sequentially per sweep direction. Directions 2 and 3 point to direction 1's storage,
  same pattern as the existing flux_n aliasing.

- **qL_prim/qR_prim dirs 2,3 → dir 1** (`m_rhs.fpp`): Riemann solver momentum states
  are also sequential per direction.

### 2. Conditional Allocation (skip unused arrays)

- **flux_gsrc_n for Cartesian geometry** (`m_rhs.fpp`): `flux_gsrc_n` and `flux_gsrc_rs_vf`
  are only used inside `if (cyl_coord)` blocks. For Cartesian 3D (the common case), their
  `%sf` fields are never allocated, saving 8 scalar fields + 1 4D array.

- **flux_n, flux_src_n for fused path** (`m_rhs.fpp`): When the fused WENO5+HLLC kernel
  is active, these intermediates are unnecessary — flux differences go directly to rhs_vf.
  Their `%sf` fields are not allocated, saving 9 scalar fields.

- **q_prim_vf momentum+energy for fused path** (`m_time_steppers.fpp`): When the fused
  kernel computes primitives on the fly from conservatives, the persistent velocity/pressure
  fields are allocated as tiny 0×0×0 dummies.

- **qL_rs_vf, qR_rs_vf for fused path** (`m_rhs.fpp`): WENO reconstruction output is
  stored in GPU registers, not global arrays.

- **flux_rs_vf, flux_src_rs_vf, vel_src_rs_vf for per-cell path** (`m_riemann_solvers.fpp`):
  The per-cell kernel computes both adjacent face fluxes inline, so no transposed Riemann
  output buffer is needed.

### 3. Max-Bounds Array Merging (reduce direction-triplicated arrays)

**Principle**: Arrays allocated per direction (x, y, z) with slightly different shapes
but the same total element count can be merged into a single array with max-bounds in
each dimension. For cubic grids (typical GPU case), zero memory waste.

**Applied to**:

- **qL_rs_vf, qR_rs_vf** (`m_rhs.fpp`): 6 direction-specific reconstruction arrays → 2
  max-bounds arrays. Each dimension uses `min(idwbuff)` to `max(idwbuff)`.

- **Riemann solver internals** (`m_riemann_solvers.fpp`): `flux_rs_vf`, `flux_gsrc_rs_vf`,
  `flux_src_rs_vf`, `vel_src_rs_vf`, `mom_sp_rs_vf`, `Re_avg_rs_vf` — each merged from
  3 direction-specific arrays to 1 max-bounds array.

- **v_rs_ws WENO workspace** (`m_weno.fpp`): 3 direction-specific workspaces → 1 max-bounds.

- **v_rs_ws_muscl** (`m_muscl.fpp`): Same treatment.

**Important lesson**: OpenACC's runtime pools GPU memory — `@:DEALLOCATE` does NOT free
device memory. Dynamic reallocation (allocate per direction, deallocate between) provides
zero memory benefit. Max-bounds allocation (allocate once, reuse) is the correct approach.

### 4. Williamson 2N-Storage RK3 (`m_time_steppers.fpp`)

Replace the standard Shu-Osher SSP-RK3 (3 registers: state + backup + RHS) with
Williamson (1980) low-storage RK3 (2 registers: state + dU). This eliminates
q_cons_ts(2) — the backup state array (8 scalar fields).

**Coefficients**: A = [0, −5/9, −153/128], B = [1/3, 15/16, 8/15]

**Update formula**:
```
Before each RHS call:  dU = A_s × dU        (pre-scale)
During RHS call:       dU += L(U)           (accumulate)
After RHS call:        U  = U + B_s × dt × dU
```

The x-direction flux differencing in `s_compute_advection_source_term` was changed from
`rhs = flux_diff` (assignment) to `rhs += flux_diff` (accumulation) to support the
pre-scaled dU pattern.

**Trade-off**: Williamson RK3 is 3rd-order but NOT strong-stability-preserving (SSP).
WENO reconstruction provides oscillation control. Not suitable for problems that
critically depend on the SSP property.

### 5. Fused WENO5+HLLC Kernel (`m_riemann_solvers.fpp`, `include/inline_weno.fpp`)

The biggest single optimization. Instead of separate GPU kernels for WENO reconstruction,
boundary fill, Riemann solving, de-transposition, and flux differencing, a single per-cell
GPU kernel does everything with no intermediate global memory.

**Current pipeline (non-fused, 5 kernel launches + 4 global arrays)**:
```
s_initialize_weno:       q_prim → v_rs_ws (transpose)         [25 MB]
s_weno:                  v_rs_ws → qL_rs_vf, qR_rs_vf         [48 MB]
s_populate_buffers:      fill qL/R boundaries
s_hllc_riemann_solver:   qL/R → flux_rs_vf                     [17 MB]
s_finalize_riemann:      flux_rs_vf → flux_n (de-transpose)    [24 MB]
s_advection_source:      flux_n → rhs_vf (differencing)
```

**Fused pipeline (1 kernel launch, 0 global intermediates)**:
```
Per cell (j, k, l):
  For each adjacent face (left, right):
    Read conservatives from q_cons_vf with direction-dependent indexing
    Compute velocity = momentum/rho at 5 stencil points (on the fly)
    Compute pressure from stiffened gas EOS at 5 stencil points (on the fly)
    WENO5 reconstruct primitives → local register arrays qL, qR
    HLLC Riemann solve → local flux array
  Accumulate (flux_left − flux_right) / ds into rhs_vf
```

**Key implementation files**:

- `src/simulation/include/inline_weno.fpp`:
  - `INLINE_WENO5_RECONSTRUCT`: Original macro reading from 4D workspace
  - `INLINE_WENO5_QPRIM`: Reads from scalar fields with direction-dependent indexing
  - `INLINE_WENO5_CONS`: Computes primitives on the fly from conservative scalar fields.
    For density/volume fractions, reads directly. For velocity, computes momentum/rho.
    For pressure, evaluates the stiffened gas EOS. All at each of the 5 WENO stencil points.

- `src/simulation/include/inline_riemann.fpp`:
  - `INLINE_HLLC_FLUX`: The HLLC Riemann solver flux computation as a Fypp macro,
    reading from `qL_local`/`qR_local` register arrays.

- `src/simulation/m_riemann_solvers.fpp`:
  - Per-cell fused kernel in `s_hllc_riemann_solver`, gated by the fused-path condition.
  - Iterates over cells `(0:m, 0:n, 0:p)` in physical coordinates.
  - Each cell computes both adjacent face fluxes (2× WENO + 2× HLLC per cell).
  - Direction-dependent coordinate mapping via Fypp `#:for NORM_DIR, XYZ`.
  - `skip_flux_rs_alloc` flag controls tiny-dummy allocation of Riemann intermediates.

**Trade-off**: 2× WENO + 2× Riemann compute per cell (each face computed by two
adjacent cells independently). On modern GPUs, the reduced memory traffic and eliminated
kernel launch overhead partially compensate.

### 6. Host-Only bc_buffers (`src/common/m_boundary_common.fpp`)

Boundary condition I/O staging buffers (`bc_buffers%sf`) use plain `allocate` instead of
`@:ALLOCATE` to avoid GPU device copies. These are only used for checkpoint save/load,
never during time stepping.

---

## Remaining Irreducible Arrays (128 bytes/grid-point)

| Array | Fields | Purpose | Why irreducible |
|-------|--------|---------|-----------------|
| q_cons_ts(1) | 8 | Solution state (partial densities, momentum, energy, volume fractions) | IS the solution |
| rhs_vf | 8 | Williamson dU register (RHS accumulator + time integration) | Required by RK algorithm |

These 16 fields × 8 bytes = 128 bytes/gp represent the theoretical minimum for the
Williamson RK3 time integration of an 8-equation system.

---

## Files Modified

### Core simulation files
| File | Changes |
|------|---------|
| `src/simulation/m_rhs.fpp` | Allocation guards, fused flux differencing, pointer aliasing, fused-path flag |
| `src/simulation/m_riemann_solvers.fpp` | Fused WENO5+HLLC per-cell kernel, max-bounds Riemann internals, per-cell iteration |
| `src/simulation/m_time_steppers.fpp` | Williamson RK3, q_prim_vf aliasing, conditional q_prim_vf allocation |
| `src/simulation/m_weno.fpp` | Public WENO coefficients, v_rs_ws management |
| `src/simulation/m_muscl.fpp` | Workspace management |
| `src/simulation/m_viscous.fpp` | Updated signatures (2 arrays instead of 6) |
| `src/simulation/m_surface_tension.fpp` | Updated signatures |
| `src/common/m_boundary_common.fpp` | Host-only bc_buffers |

### New include files
| File | Purpose |
|------|---------|
| `src/simulation/include/inline_weno.fpp` | WENO5 Fypp macros: `INLINE_WENO5_RECONSTRUCT`, `INLINE_WENO5_QPRIM`, `INLINE_WENO5_CONS` |
| `src/simulation/include/inline_riemann.fpp` | `INLINE_HLLC_FLUX` Fypp macro for per-cell Riemann flux |

---

## Measurement Methodology

- **nvidia-smi**: Total GPU memory (data + fixed overhead). Unreliable for small grids
  where fixed overhead dominates. Best used for relative comparisons at the same grid size.

- **ACC_CUDA_VERBOSE=1**: Ground-truth allocation sizes from the OpenACC runtime.
  Pass via `mpirun -np 1 -x ACC_CUDA_VERBOSE ./simulation`. Shows every `cuMemAlloc`
  with size.

- **nsys**: `nsys profile --trace=cuda,nvtx --cuda-memory-usage=true --export=sqlite
  mpirun -np 1 ./simulation`. Query `CUDA_GPU_MEMORY_USAGE_EVENTS` table for allocation
  tracking. Requires `--export=sqlite` flag. Individual allocation sizes may be unreliable
  with OpenACC; use cumulative peak instead.

- **Empty program baseline**: A minimal OpenACC program (`!$acc enter data create(x)` +
  `sleep`) measures 312 MiB on the RTX 6000, establishing the fixed CUDA/OpenACC runtime
  overhead.

- **Slope method**: Run at two grid sizes (e.g., N=32 and N=128), compute
  `(mem_128 − mem_32) / (points_128 − points_32)` for true bytes/grid-point independent
  of fixed overhead.

---

## Known Limitations

1. **Fused path only for HLLC + WENO5 + non-viscous + model_eqns=2**: Other Riemann
   solvers (HLL, LF, HLLD), MUSCL reconstruction, viscous flows, and 6-equation models
   use the fallback path with full intermediate arrays.

2. **Williamson RK3 is not SSP**: May exhibit oscillations near strong shocks that the
   SSP-RK3 would suppress. WENO provides some oscillation control but not a guarantee.

3. **I/O save crashes**: The Williamson RK3 + fused path has a known issue with the
   checkpoint save at the final timestep (`CUDA_ERROR_ILLEGAL_ADDRESS` in save routine).
   The main time-stepping loop runs correctly for 500+ steps.

4. **CFL-based adaptive dt**: `s_compute_dt` reads q_prim_vf for velocity/pressure. When
   q_prim_vf is tiny dummies (fused path), this routine cannot be used. Fixed-dt cases
   work correctly.

5. **Performance trade-off**: The per-cell kernel computes each Riemann problem twice
   (once per adjacent cell). At N=64, this roughly doubles the per-step time compared to
   the per-face kernel. For memory-limited large problems, the trade-off is favorable.
