# Mixture Accumulation: Single Home — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the stiffened-gas mixture accumulation exactly one implementation, so that piece 4 has a single place to add per-fluid EOS dispatch.

**Architecture:** `s_accumulate_mixture_properties` moves from `src/simulation/m_riemann_state.fpp` to `src/common/m_variables_conversion.fpp`, and the `else` branch of `s_convert_species_to_mixture_variables_kernel` — which today contains a byte-for-byte copy of the same loop — calls it instead. No signature changes, no behaviour changes, no golden changes.

**Tech Stack:** Fortran 2008 + Fypp, built and tested via `./mfc.sh`. GPU offload via the `GPU_*` Fypp macros.

**Spec:** `docs/superpowers/specs/2026-08-25-eos-backends-design.md` (piece 1)

## Global Constraints

- Precision only via `wp`/`stp` kinds and generic intrinsics. Never `dsqrt`/`dble`/`real(8)` or `d` exponent literals.
- GPU directives only via `GPU_*` Fypp macros. Never raw `!$acc`/`!$omp`.
- No `optional` dummy arguments on routines carrying `$:GPU_ROUTINE`. This is why the two routines are not merged.
- No derived-type dummy arguments on routines carrying `$:GPU_ROUTINE(parallelism='[seq]')`. Measured: collapses amdflang register allocation, ~20% on gfx90a. See spec, "Constraints".
- Line length ≤ 132 characters.
- `./mfc.sh precheck` must pass before every commit.
- This change must be **bit-identical**. Any golden difference is a defect in this plan's execution, never something to regenerate away.

---

### Task 1: Move the accumulation routine into common

**Files:**
- Modify: `src/common/m_variables_conversion.fpp` (add routine; extend `public` list)
- Modify: `src/simulation/m_riemann_state.fpp:1036-1058` (delete routine)
- Test: the existing golden suite — `./mfc.sh test`

**Interfaces:**
- Consumes: `gammas`, `pi_infs`, `qvs` — module arrays already in scope in `m_variables_conversion`.
- Produces: `s_accumulate_mixture_properties(nf, alpha_rho_K, alpha_K, rho_K, gamma_K, pi_inf_K, qv_K)`, public from `m_variables_conversion`. `nf` is `integer, intent(in)`; `alpha_rho_K`, `alpha_K` are `real(wp), dimension(nf), intent(in)`; the four outputs are `real(wp), intent(out)`. Task 2 calls this.

- [ ] **Step 1: Record the current text of the routine**

Run:
```bash
sed -n '1030,1060p' src/simulation/m_riemann_state.fpp
```
Expected: the routine below. Copy it verbatim — do not retype it, so the arithmetic cannot drift.

```fortran
    subroutine s_accumulate_mixture_properties(nf, alpha_rho_K, alpha_K, rho_K, gamma_K, pi_inf_K, qv_K)

        $:GPU_ROUTINE(function_name='s_accumulate_mixture_properties', parallelism='[seq]', cray_inline=True)

        integer, intent(in)                 :: nf  !< Number of fluids to accumulate over
        real(wp), dimension(nf), intent(in) :: alpha_rho_K, alpha_K
        real(wp), intent(out)               :: rho_K, gamma_K, pi_inf_K, qv_K
        integer                             :: i   !< Loop iterator over fluids

        rho_K = 0._wp
        gamma_K = 0._wp
        pi_inf_K = 0._wp
        qv_K = 0._wp

        $:GPU_LOOP(parallelism='[seq]')
        do i = 1, nf
            rho_K = rho_K + alpha_rho_K(i)
            gamma_K = gamma_K + alpha_K(i)*gammas(i)
            pi_inf_K = pi_inf_K + alpha_K(i)*pi_infs(i)
            qv_K = qv_K + alpha_rho_K(i)*qvs(i)
        end do

    end subroutine s_accumulate_mixture_properties
```

- [ ] **Step 2: Paste it into `m_variables_conversion.fpp`**

Insert the routine verbatim immediately before `s_compute_speed_of_sound`. That places it next to the other EOS code and next to the `gammas`/`pi_infs` declarations it reads.

- [ ] **Step 3: Export it**

In the `public ::` list at the top of `m_variables_conversion.fpp`, add `s_accumulate_mixture_properties`. The list is continued with `&`; keep every line ≤ 132 characters.

- [ ] **Step 4: Delete the original**

Remove the routine from `src/simulation/m_riemann_state.fpp`. Leave everything else in that file alone.

- [ ] **Step 5: Build**

Run:
```bash
source ./mfc.sh load          # interactive picker; or -c <slug> -m c, slugs listed in toolchain/modules
./mfc.sh build -j 8
```
Expected: success. The five Riemann solvers already `use m_variables_conversion`, so their calls resolve without edits. If a solver fails with "no explicit type" or an unresolved reference, that solver was relying on `m_riemann_state` re-exporting the name — add `use m_variables_conversion` to it rather than putting the routine back.

- [ ] **Step 6: Run the full suite**

Run:
```bash
./mfc.sh test -j 8
```
Expected: all pass, 0 failed. A pure move cannot change results; any failure means the routine was altered in transit — diff it against Step 1's output.

- [ ] **Step 7: Commit**

```bash
./mfc.sh precheck -j 8
git add src/common/m_variables_conversion.fpp src/simulation/m_riemann_state.fpp
git commit -m "Move s_accumulate_mixture_properties into m_variables_conversion"
```

---

### Task 2: Route the conversion kernel through it

**Files:**
- Modify: `src/common/m_variables_conversion.fpp`, the `else` branch of `s_convert_species_to_mixture_variables_kernel`
- Test: the existing golden suite

**Interfaces:**
- Consumes: `s_accumulate_mixture_properties(...)` from Task 1.
- Produces: no new names. `s_convert_species_to_mixture_variables_kernel` keeps its signature exactly.

- [ ] **Step 1: Confirm the two loops are identical before touching anything**

Run:
```bash
sed -n '/subroutine s_convert_species_to_mixture_variables_kernel(/,/end subroutine s_convert_species_to_mixture_variables_kernel/p' \
  src/common/m_variables_conversion.fpp | sed -n '25,50p'
```
Expected: the `else` branch ends with

```fortran
            rho_K = 0._wp; gamma_K = 0._wp; pi_inf_K = 0._wp; qv_K = 0._wp
            do i = 1, num_fluids
                rho_K = rho_K + alpha_rho_K(i)
                gamma_K = gamma_K + alpha_K(i)*gammas(i)
                pi_inf_K = pi_inf_K + alpha_K(i)*pi_infs(i)
                qv_K = qv_K + alpha_rho_K(i)*qvs(i)
            end do
```

This is the same arithmetic in the same order as Task 1's routine with `nf = num_fluids`. If it is **not** identical — different order, an extra term, a different array — **stop and report**. The premise of this plan is that they match; if they do not, the difference is a finding, not something to smooth over.

- [ ] **Step 2: Replace the loop with a call**

Replace exactly those seven lines with:

```fortran
            call s_accumulate_mixture_properties(num_fluids, alpha_rho_K, alpha_K, rho_K, gamma_K, pi_inf_K, qv_K)
```

Leave the `mpp_lim` clipping block above it untouched — it must still run first, because it mutates `alpha_K` in place and the accumulation must see the clipped values.

- [ ] **Step 3: Build**

Run:
```bash
./mfc.sh build -j 8
```
Expected: success.

Note: the kernel is declared `cray_noinline=True` and the routine it now calls is `cray_inline=True`. That combination is legal; do not "fix" either directive.

- [ ] **Step 4: Run the full suite**

Run:
```bash
./mfc.sh test -j 8
```
Expected: all pass, 0 failed, bit-identical.

The riskiest silent failure here is the clipping ordering. If `mpp_lim` cases (search `./mfc.sh test -l` for `mpp_lim`) fail while others pass, the call was placed above the clipping block instead of below it.

- [ ] **Step 5: Commit**

```bash
./mfc.sh precheck -j 8
git add src/common/m_variables_conversion.fpp
git commit -m "Route the species-to-mixture kernel through s_accumulate_mixture_properties"
```

---

### Task 3: Prove the two paths cannot silently diverge again

**Files:**
- Modify: `src/common/m_variables_conversion.fpp` (comment only)
- Test: none — this task adds no runtime behaviour

**Interfaces:**
- Consumes: nothing. Produces: nothing.

Tasks 1 and 2 remove the duplicate, but nothing stops someone reintroducing one. There is no unit-test harness for a `[seq]` device routine in this codebase, and adding one is out of scope, so the guard is documentary and must be placed where it will actually be read.

- [ ] **Step 1: Comment the routine with what it is for**

Immediately above `s_accumulate_mixture_properties`, add:

```fortran
    !> Accumulate stiffened-gas mixture coefficients over the first nf fluids.
    !!
    !! This is the only implementation of the stiffened-gas mixture rule. It is deliberately not
    !! merged with s_convert_species_to_mixture_variables_kernel: that routine additionally clips and
    !! renormalises alpha_K in place under mpp_lim, special-cases num_fluids == 1 with bubbles_euler,
    !! and optionally returns Re_K and G_K. Merging would need a clipping flag and optional dummies
    !! on a [seq] device routine, which is not portable across the offload backends.
    !!
    !! nf is not always num_fluids: the bubbles path in m_riemann_solver_hllc passes num_fluids - 1
    !! to exclude the gas phase, and passes limited volume fractions rather than the raw ones.
```

- [ ] **Step 2: Verify the claim about `nf` before committing the comment**

Run:
```bash
grep -rn "s_accumulate_mixture_properties(num_fluids - 1" src/
grep -rn "s_accumulate_mixture_properties(num_fluids, alpha_rho_L, alpha_lim_L" src/
```
Expected: both return hits in `src/simulation/m_riemann_solver_hllc.fpp`. If either returns nothing, the comment is wrong — correct the comment to match the code, not the other way round.

- [ ] **Step 3: Commit**

```bash
./mfc.sh precheck -j 8
git add src/common/m_variables_conversion.fpp
git commit -m "Document why the two mixture routines stay separate"
```

---

## Verification for the whole piece

Before opening a PR:

- [ ] `./mfc.sh format -j 8` — expect "files left unchanged"
- [ ] `./mfc.sh precheck -j 8` — expect 7/7
- [ ] `./mfc.sh test -j 8` — expect 0 failed, **0 goldens regenerated**
- [ ] `git diff master --stat` — expect a net reduction; the only files touched are `m_variables_conversion.fpp` and `m_riemann_state.fpp`

Because `src/common/` is shared by all three executables, the suite must be run in full rather than filtered.

If the build is a GPU build, also confirm no kernel-resource regression on the Riemann kernels — this piece touches a routine called from inside them. Extract the code object and compare `private_segment_fixed_size`, `vgpr_count` and `agpr_count` against the pre-change binary; they should be unchanged. `s_accumulate_mixture_properties` is `cray_inline=True` and takes only scalars and arrays, so no change is expected — but "expected" is what #1714 assumed.

## Notes for the executor

- This piece is worth doing *only* because piece 4 needs one place to add per-fluid EOS dispatch. If any task turns out to require a behaviour change to land, stop: the value of the piece was that it was free.
- Do not rename `s_accumulate_mixture_properties`. Piece 4 relocates it to `m_eos.fpp`; renaming now would make that move harder to review.
