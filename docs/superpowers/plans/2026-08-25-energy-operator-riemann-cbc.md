# Energy Operator: Riemann Solvers and CBC (piece 2a) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the stiffened-gas total energy one implementation, called from the Riemann solvers, instead of fourteen open-coded copies.

**Architecture:** A new `s_compute_energy` in `src/common/m_variables_conversion.fpp`, beside `s_accumulate_mixture_properties` and the sound-speed operators. It returns the *thermodynamic* energy only; non-thermodynamic contributions (magnetic, elastic) stay at the call site, because they are not equation-of-state terms.

**Tech Stack:** Fortran 2008 + Fypp, built and tested via `./mfc.sh`. GPU offload via `GPU_*` Fypp macros.

**Spec:** `docs/superpowers/specs/2026-08-25-eos-backends-design.md` (piece 2a)

## Global Constraints

- Precision only via `wp`/`stp` kinds and generic intrinsics. Never `dsqrt`/`dble`/`real(8)` or `d` exponent literals.
- GPU directives only via `GPU_*` Fypp macros. Never raw `!$acc`/`!$omp`.
- No `optional` dummies and no derived-type dummies on a `$:GPU_ROUTINE(parallelism='[seq]')` routine. A derived-type dummy costs ~20% on gfx90a (measured; see spec).
- Line length ≤ 132 characters.
- `./mfc.sh precheck` must pass before every commit.
- Goldens are tolerance-compared. No golden may be regenerated in Tasks 1–3; Task 4 may change results, and that is the whole point of making it a separate task.
- This touches the hottest kernels in the code. Kernel-resource and wall-clock checks are part of the definition of done, not optional extras.

---

### Task 1: Add the energy operator and delete one dead assignment

**Files:**
- Modify: `src/common/m_variables_conversion.fpp` (add routine; extend `public` list)
- Modify: `src/simulation/m_cbc.fpp` (delete dead line)

**Interfaces:**
- Consumes: nothing.
- Produces: `s_compute_energy(pres, rho, gamma, pi_inf, qv, vel_sum, E)`. All arguments `real(wp)`; the first six `intent(in)`, `E` is `intent(out)`. Tasks 2 and 3 call this.

- [ ] **Step 1: Add the routine**

Insert immediately after `end subroutine s_accumulate_mixture_properties` in `src/common/m_variables_conversion.fpp`:

```fortran
    !> Total energy per unit volume of a stiffened-gas state.
    !!
    !! Thermodynamic contributions only. Magnetic energy (pres_mag) and elastic energy are added by the
    !! caller, because they are not equation-of-state terms: mixing them in here would make the operator
    !! impossible to reuse for a second equation of state.
    !!
    !! The chemistry and relativistic branches of the Riemann solvers do not use this relation at all -
    !! chemistry builds E from the mixture internal energy, and the relativistic form is unrelated - so
    !! those sites are deliberately left open-coded.
    subroutine s_compute_energy(pres, rho, gamma, pi_inf, qv, vel_sum, E)

        $:GPU_ROUTINE(function_name='s_compute_energy', parallelism='[seq]', cray_inline=True)

        real(wp), intent(in)  :: pres, rho, gamma, pi_inf, qv, vel_sum
        real(wp), intent(out) :: E

        E = gamma*pres + pi_inf + 5.e-1_wp*rho*vel_sum + qv

    end subroutine s_compute_energy
```

- [ ] **Step 2: Export it**

Add `s_compute_energy` to the `public ::` list at the top of the file, keeping every continued line ≤ 132 characters.

- [ ] **Step 3: Delete the dead CBC assignment**

In `src/simulation/m_cbc.fpp`, delete this line and nothing else:

```fortran
                            E = gamma*pres + pi_inf + 5.e-1_wp*rho*vel_K_sum
```

It sits in the `else` (non-chemistry) arm of the energy block. `E` is read only inside the `if (chemistry)` branch of the flux update further down, so on this path the value is never used.

Verify before deleting:

```bash
grep -n "E/rho" src/simulation/m_cbc.fpp
```
Expected: exactly one hit, inside the chemistry branch. If `E` is read anywhere else, **stop** — the line is not dead and this step is wrong.

After deleting, `E` is still assigned on the chemistry path and still read there, so the declaration stays.

- [ ] **Step 4: Build**

```bash
source ./mfc.sh load          # -c <slug> -m g on this hardware; slugs in toolchain/modules
./mfc.sh build --gpu mp -j 16
```
Expected: success. Nothing calls the new routine yet.

- [ ] **Step 5: Commit**

```bash
./mfc.sh precheck -j 8
git add src/common/m_variables_conversion.fpp src/simulation/m_cbc.fpp
git commit -m "Add s_compute_energy; drop a dead energy assignment in CBC"
```

---

### Task 2: Convert the ten plain stiffened-gas sites

**Files:**
- Modify: `src/simulation/m_riemann_solver_hll.fpp` (2 sites)
- Modify: `src/simulation/m_riemann_solver_hllc.fpp` (4 sites)
- Modify: `src/simulation/m_riemann_solver_hypo_hlld.fpp` (2 sites)
- Modify: `src/simulation/m_riemann_solver_lf.fpp` (2 sites)

**Interfaces:**
- Consumes: `s_compute_energy(pres, rho, gamma, pi_inf, qv, vel_sum, E)` from Task 1.
- Produces: no new names.

- [ ] **Step 1: Replace each site**

Match on text, not line number — line numbers shift as you edit. Each of these ten lines:

```fortran
E_L = gamma_L*pres_L + pi_inf_L + 5.e-1_wp*rho_L*vel_L_rms + qv_L
```

becomes:

```fortran
call s_compute_energy(pres_L, rho_L, gamma_L, pi_inf_L, qv_L, vel_L_rms, E_L)
```

and correspondingly for `_R`. The sites, by file and current line, are:

| file | lines | note |
|---|---|---|
| `m_riemann_solver_hll.fpp` | 295, 296 | literal is `5.e-1`, not `5.e-1_wp` |
| `m_riemann_solver_hllc.fpp` | 269, 270 | `5.e-1_wp` |
| `m_riemann_solver_hllc.fpp` | 1025, 1026 | `5.e-1` |
| `m_riemann_solver_hypo_hlld.fpp` | 343, 344 | `%` component form: `E%L`, `pres%L`, `vel_rms%L` |
| `m_riemann_solver_lf.fpp` | 226, 227 | `5.e-1` |

The hypo_hlld pair uses the `riemann_states` derived type, so it reads:

```fortran
call s_compute_energy(pres%L, rho%L, gamma%L, pi_inf%L, qv%L, vel_rms%L, E%L)
```

Note the literal differs between sites (`5.e-1` vs `5.e-1_wp` vs `5e-1_wp`). All three are the same value in `wp`; the operator uses `5.e-1_wp`. This is a roundoff-identical change, not a bit-identical one — see the Global Constraints note on tolerance.

- [ ] **Step 2: Confirm nothing was missed**

```bash
grep -rnE "^\s*(E_L|E_R|E%L|E%R) *= *gamma" src/simulation/m_riemann_solver_*.fpp
```
Expected: exactly six remaining hits — `hll` 288/290 and `hlld` 143/144 (the MHD sites, Task 3), and `hllc` 552/553 (Task 4). If any other hit remains, it was missed in Step 1.

- [ ] **Step 3: Build**

```bash
./mfc.sh build --gpu mp -j 16
```
Expected: success. All five solvers already `use m_variables_conversion`.

- [ ] **Step 4: Check kernel resources did not regress**

Extract the AMDGPU code object and compare against the pre-change binary:

```bash
llvm-objcopy --dump-section=.llvm.offloading=off.bin build/install/*/bin/simulation /dev/null
python3 -c "d=open('off.bin','rb').read(); i=d.find(b'\x7fELF'); open('dev.elf','wb').write(d[i:])"
llvm-readelf --notes dev.elf | grep -A30 riemann_solver_hllc
```
Compare `.private_segment_fixed_size`, `.vgpr_count` and `.agpr_count` for the HLLC kernels against the same dump taken before Task 2. Expected: unchanged. `s_compute_energy` takes only scalars, so no aggregate is materialised — but this is exactly the check that caught #1714, and "expected" is what that PR assumed.

**If scratch or VGPR rise:** stop. The likely cause is the call not being inlined. Do not accept the regression; report it.

- [ ] **Step 5: Run the affected tests**

```bash
./mfc.sh test --gpu mp -j 4
```
Expected: 0 failed, no goldens regenerated. Every case exercises a Riemann solver, so the full suite is the relevant set here — this is not a place to sample.

- [ ] **Step 6: Commit**

```bash
./mfc.sh precheck -j 8
git add src/simulation/m_riemann_solver_hll.fpp src/simulation/m_riemann_solver_hllc.fpp \
        src/simulation/m_riemann_solver_hypo_hlld.fpp src/simulation/m_riemann_solver_lf.fpp
git commit -m "Route the plain Riemann energy sites through s_compute_energy"
```

---

### Task 3: Convert the four MHD sites

**Files:**
- Modify: `src/simulation/m_riemann_solver_hll.fpp` (lines 288, 290)
- Modify: `src/simulation/m_riemann_solver_hlld.fpp` (lines 143, 144)

**Interfaces:**
- Consumes: `s_compute_energy(...)` from Task 1.
- Produces: no new names.

- [ ] **Step 1: Replace, keeping the magnetic term at the call site**

```fortran
E_L = gamma_L*pres_L + pi_inf_L + 0.5_wp*rho_L*vel_L_rms + qv_L + pres_mag%L
```

becomes:

```fortran
call s_compute_energy(pres_L, rho_L, gamma_L, pi_inf_L, qv_L, vel_L_rms, E_L)
E_L = E_L + pres_mag%L
```

and the same for `_R`, and for `hlld`'s `E%L`/`E%R` component form. Keep `hlld`'s trailing comment `! includes magnetic energy` on the `pres_mag` line, where it now belongs.

The magnetic energy is deliberately *not* pushed into the operator. It is not an equation-of-state term, and a second EOS backend must not have to know about it.

- [ ] **Step 2: Confirm only Task 4's sites remain**

```bash
grep -rnE "^\s*(E_L|E_R|E%L|E%R) *= *gamma" src/simulation/m_riemann_solver_*.fpp
```
Expected: exactly two hits, `hllc` 552 and 553.

- [ ] **Step 3: Build, check kernels, test**

Same three commands as Task 2 Steps 3–5, with the same expectations. MHD cases are a small fraction of the suite, so pay attention to whether any MHD case is exercised at all:

```bash
./mfc.sh test -l | grep -ci mhd
```
If that returns 0, say so in the commit message: the change is then unexercised by the suite and rests on review alone.

- [ ] **Step 4: Commit**

```bash
./mfc.sh precheck -j 8
git add src/simulation/m_riemann_solver_hll.fpp src/simulation/m_riemann_solver_hlld.fpp
git commit -m "Route the MHD Riemann energy sites through s_compute_energy"
```

---

### Task 4: Decide what to do about `hllc` 552/553

**This task changes behaviour. It is separated from Tasks 1–3 for that reason, and must not be folded into them.**

**Files:**
- Modify: `src/simulation/m_riemann_solver_hllc.fpp` (lines 552, 553) — only after the decision below

`hllc` 552/553 sit in the `bubbles_euler` branch and compute the energy **without** `qv`:

```fortran
E_L = gamma_L*pres_L + pi_inf_L + 5.e-1_wp*rho_L*vel_L_rms
```

Every other plain site in every solver includes `+ qv`. Converting these to `s_compute_energy` would add `qv`, changing results for any bubble case with a non-zero heat of formation.

- [ ] **Step 1: Establish whether this is reachable**

```bash
grep -rn "qv" toolchain/mfc/test/cases.py | grep -i bubble
grep -rn "fluid_pp([0-9])%qv" examples/*/case.py | head
```
Determine whether any test or example combines `bubbles_euler` with a non-zero `qv`. Record the answer.

- [ ] **Step 2: Establish whether the omission is deliberate**

`git log -L552,553:src/simulation/m_riemann_solver_hllc.fpp` and read the commit that introduced it. Bubble models commonly assume `qv = 0`, in which case the omission is harmless but inconsistent; if some commit deliberately removed `qv` here, that intent must be respected.

- [ ] **Step 3: Stop and report**

Present to your human partner: whether it is reachable, whether it appears deliberate, and which of these applies —

1. `qv` is always zero on this path → convert, results unchanged, goldens unaffected.
2. `qv` can be non-zero and the omission is a defect → convert, and treat the golden movement as a fix, with the affected cases named.
3. The omission is deliberate → leave the site open-coded with a comment recording why, and note it in the spec's related-work section.

**Do not choose between these yourself.** Options 1 and 2 differ only in whether goldens move, and option 3 is a physics judgement.

---

## Verification for the whole piece

- [ ] `./mfc.sh format -j 8` — expect "files left unchanged"
- [ ] `./mfc.sh precheck -j 8` — expect 7/7
- [ ] `./mfc.sh test --gpu mp` — expect 0 failed, 0 goldens regenerated (Tasks 1–3)
- [ ] HLLC kernel scratch / VGPR / AGPR unchanged versus the pre-change binary
- [ ] `5eq_rk3_weno3_hllc` benchmark, three runs before and three after, same node and case; expect the difference inside run-to-run spread (which has been 3–7% on this hardware, so quote the spread alongside the means rather than the delta alone)
- [ ] `git diff --stat` — expect a net reduction in `src/simulation/`

## Notes for the executor

- Read the diff before committing each task. On piece 1, a stale doc comment orphaned onto the wrong routine survived a clean build, a passing test suite and an unchanged kernel dump; only reading the diff caught it.
- Do not push `pres_mag`, elastic energy, or the chemistry `e_mix` form into `s_compute_energy`. Keeping it thermodynamic is the entire point — piece 5 adds a JWL branch to it, and that branch must not have to reason about magnetic fields.
- If Task 2's kernel check regresses, that is a finding worth more than this piece. Report it rather than working around it.
