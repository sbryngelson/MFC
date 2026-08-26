# Finishing the Equation-of-State Centralisation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the remaining 96 open-coded equation-of-state references, across thirteen files, to calls on shared operators — so that adding JWL is a change inside one module.

**Architecture:** Three operators already exist in `src/common/m_variables_conversion.fpp`:
`s_compute_mixture_coefficients`, `s_compute_energy`, `s_compute_speed_of_sound`/`_avg`. Groups A and B below convert callers to them. Groups C and D add operators for families that have none. Group E adds backend dispatch and JWL.

**Tech Stack:** Fortran 2008 + Fypp, built and tested via `./mfc.sh`. GPU offload via `GPU_*` macros.

**Spec:** `docs/superpowers/specs/2026-08-25-eos-backends-design.md`

## Global Constraints

- No `optional` dummies and no derived-type dummies on `$:GPU_ROUTINE(parallelism='[seq]')` routines. A derived-type dummy costs ~20% on gfx90a; see `.claude/rules/common-pitfalls.md`.
- Precision via `wp`/`stp` only; GPU directives via `GPU_*` macros only; lines <= 132 chars.
- `./mfc.sh precheck` must pass **before** each commit, and its result must be checked, not assumed.
- Measure speed before correctness: kernel-resource diff (seconds), then benchmark (~12 min), then tests. These conversions preserve arithmetic by construction, so codegen is the likely failure, not results.
- Compare kernel resources with `toolchain/mfc/kernel_resources.py` (branch `eos/gpu-guards`), keying on resource values. Build the baseline you compare against.
- Never edit the working tree while a build or benchmark is running. This corrupted two measurements already.

## Ordering

A before B before C. D and E depend on decisions, not code. Within A, the tasks are independent and can be done in any order or in parallel.

---

## Group A: mechanical conversions to existing operators

No new operators. Each task is: replace open-coded algebra with a call, build, kernel-diff, benchmark if the file is on a hot path, commit.

### Task A1: `m_igr.fpp` — 10 mixture accumulations, 5 blocks

**Files:** Modify `src/simulation/m_igr.fpp`

- [ ] **Step 1: Find the blocks**

```bash
grep -nE "gamma_[LR] = 0\._wp" src/simulation/m_igr.fpp
```
Expected: 5 hits, each starting a block that zeroes `rho_L/gamma_L/pi_inf_L` (and `_R`) and then accumulates.

- [ ] **Step 2: Replace each block with two calls**

```fortran
call s_compute_mixture_coefficients(alpha_rho_L, alpha_L, rho_L, gamma_L, pi_inf_L, qv_L)
call s_compute_mixture_coefficients(alpha_rho_R, alpha_R, rho_R, gamma_R, pi_inf_R, qv_R)
```
If a block has no `qv_L`, declare one and add it to that loop's `private()` clause. Check whether `m_igr` computes `qv` at all first — IGR may not carry a heat of formation, in which case the value is unused but must still be passed.

- [ ] **Step 3: Add the import if absent**

```bash
grep -c "use m_variables_conversion" src/simulation/m_igr.fpp
```
If 0, add `use m_variables_conversion, only: s_compute_mixture_coefficients`.

- [ ] **Step 4: Build and kernel-diff**

```bash
./mfc.sh build --gpu mp -j 16
python3 toolchain/mfc/kernel_resources.py build/install/*/bin/simulation --baseline <base>.json
```
Expected: no regressions. IGR kernels sit at 154-166 VGPR, well clear of any cliff, so a small rise is tolerable but should be understood, not waved through.

- [ ] **Step 5: Benchmark**

`./mfc.sh run benchmarks/igr/case.py --targets pre_process simulation -n 1 -- --gbpp 4`, three times, against three runs on the parent commit. IGR has its own benchmark; do not use the HLLC case.

- [ ] **Step 6: Commit**

```bash
./mfc.sh precheck -j 8   # check the exit code
git commit -m "Route the IGR mixture coefficients through the shared rule"
```

### Task A2: `m_ibm.fpp` — 2 energy sites

**Files:** Modify `src/simulation/m_ibm.fpp:398,400`

- [ ] **Step 1: Read both sites and their difference**

```fortran
q_cons_vf(eqn_idx%E)%sf(j, k, l) = (1 - alpha_IP(1))*(gamma*pres_IP + pi_inf + dyn_pres)
q_cons_vf(eqn_idx%E)%sf(j, k, l) = gamma*pres_IP + pi_inf + dyn_pres
```
They differ by the `(1 - alpha_IP(1))` factor and use `dyn_pres` where the operator takes `vel_sum`. `s_compute_energy` computes `0.5*rho*vel_sum`, so the caller must pass `vel_sum` such that this equals `dyn_pres`, or the conversion changes results.

- [ ] **Step 2: Establish what `dyn_pres` is here**

```bash
grep -n "dyn_pres" src/simulation/m_ibm.fpp
```
If `dyn_pres = 0.5*rho*sum(vel**2)`, pass `sum(vel**2)` as `vel_sum`. **If it is anything else, stop and report** — the sites are then not this relation and must stay open-coded.

- [ ] **Step 3: Convert, keeping the `(1 - alpha_IP(1))` factor at the call site**

The volume-fraction scaling is not an equation-of-state term, exactly as `pres_mag` was not.

- [ ] **Step 4: Build, kernel-diff, commit.** IBM has its own benchmark (`benchmarks/ibm`); use it.

### Task A3: `m_sim_helpers.fpp` — 1 energy site

**Files:** Modify `src/simulation/m_sim_helpers.fpp:100`

This is inside `s_compute_enthalpy`, whose `H` output no consumer uses any more (all three callers discard it). Two things happen here:

- [ ] **Step 1: Confirm no caller reads `H`**

```bash
grep -rn "s_compute_enthalpy" src/ | grep -v "subroutine s_compute_enthalpy"
```
For each caller, check whether the variable passed as `H` is read afterwards. Three callers are expected: `m_time_steppers`, `m_data_output`, and `m_variables_conversion`.

- [ ] **Step 2: If none read it, remove `H` and `E` from the routine** and rename it to what it computes (mixture properties), updating all three call sites. If any caller does read it, leave the signature alone and only convert line 100 to `s_compute_energy`.

- [ ] **Step 3: Build, precheck, commit.** Cold path; kernel-diff is enough, no benchmark needed.

### Task A4: pre-process and post-process — 7 references

**Files:** `src/pre_process/m_assign_variables.fpp`, `src/pre_process/m_data_output.fpp`, `src/post_process/m_derived_variables.fpp`

- [ ] **Step 1: Classify each site** as mixture coefficients, energy, or sound speed, and convert to the matching operator.
- [ ] **Step 2:** These run once and are not GPU kernels, so no kernel-diff or benchmark is required. Build and the golden suite are the gates.
- [ ] **Step 3: Commit.** Note in the message that pre-process ICs are where a wrong second EOS would be silent, since nothing downstream re-derives them.

### Task A5: `m_bubbles_EE.fpp` and `m_qbmm.fpp` — the Tait isentrope

**Files:** `src/simulation/m_bubbles_EE.fpp` (4 refs), `src/simulation/m_qbmm.fpp` (2 refs)

Both derive `n_tait` and `B_tait` from `gs_min`/`pi_infs` and use them in a bubble-dynamics
isentrope, e.g. `c = n_tait*(pres + B_tait)*(1 - alf)/rho`. Note `m_bubbles_EE:247` comments its own
conversion: `B_tait = B_tait*(n_tait - 1)/n_tait  ! make this the usual pi_inf`.

- [ ] **Step 1: Decide whether these are conversions or a fourth family.** The *coefficient*
  derivation is the mixture rule under different names and converts mechanically. The *isentrope* is
  not one of the operators that exist, and it is the same closed-form relation
  `s_equilibrate_pressure` needs (Task D2). Classify each of the six references before touching any.

- [ ] **Step 2: Convert only the coefficient derivations**, leaving the isentrope expressions alone
  and commented as belonging to the isentrope family. Do not invent an isentrope operator here — it
  should be designed once, with Task D2, not twice.

- [ ] **Step 3: Build, kernel-diff, benchmark** with a bubble case (`benchmarks/viscous_weno5_sgb_acoustic`
  exercises `bubbles_euler`), then commit.

---

## Group B: the pressure inversion operator

### Task B1: add `s_compute_pressure_from_energy`

**Files:** Modify `src/common/m_variables_conversion.fpp`

`s_compute_pressure` already exists and already branches on `mhd`, `bubbles_euler` and `hypoelasticity`. This task does **not** add a new routine; it makes the existing one the single implementation.

- [ ] **Step 1: Inventory the open-coded inversions**

```bash
grep -rniE "= *\(.*- *dyn_p.*- *pi_inf|energy.*- *pi_inf.*\)/gamma" src/
```
Expected: `m_pressure_relaxation` (per-phase, inside the Newton solve) and `m_bubbles_EL:376`. `m_data_output`'s copies were removed earlier in this work.

- [ ] **Step 2: Convert `m_bubbles_EL:376`** to `s_compute_pressure`. Check its `(1 - alf)` handling matches the bubbles branch already in the routine.

- [ ] **Step 3: Leave `m_pressure_relaxation`'s per-phase inversion alone for now** and say so in the commit. It is inside a Newton iteration that also needs the isentrope, so converting only the inversion buys little and obscures the harder problem. It belongs with Task D2.

- [ ] **Step 4: Build, kernel-diff, run the suite** — pressure relaxation is 6-equation, so the `model_eqns=3` cases are the relevant ones.

---

## Group C: families with no operator yet

### Task C1: bulk modulus — `m_rhs.fpp`, `m_hypoelastic.fpp`

Both compute `((gammas(i) + 1)*p + pi_infs(i))/gammas(i) + (4/3)*G`, which is the `alt_soundspeed` expression plus a shear term.

- [ ] **Step 1:** Add `s_compute_bulk_modulus(pres, i, blkmod)` taking a **fluid index**, not mixture coefficients — both callers want per-fluid values, and the shear term is added by the caller because it is elasticity, not equation of state.
- [ ] **Step 2:** Convert both call sites; keep `(4/3)*G` outside the operator.
- [ ] **Step 3:** Build, kernel-diff, commit. `m_rhs` is a hot path; benchmark with `hypo_hll`.

### Task C2: temperature inversion — `m_reactive_burn.fpp`

One site: `T = (pres + ps_inf(1))/((gs_min(1) - 1)*cvs(1)*rho)`.

- [ ] **Step 1:** Add `s_compute_temperature(pres, rho, i, T)`. This is the first operator to consume `cvs`, which no existing operator touches.
- [ ] **Step 2:** Convert the one site, build, commit. Cold path.
- [ ] **Step 3:** Note in the commit that `m_phase_change` computes the same quantity and is deliberately not converted here — see Task D1.

---

## Group D: phase change — decisions before code

### Task D1: scope `m_phase_change.fpp`

39 parameter references, the sole consumer of `cvs` and `qvps`, and the source of families 7 through 9 (temperature, entropy, caloric).

- [ ] **Step 1: Do not convert anything yet.** Produce an inventory: for each of `sk`, `hk`, `rhok`, `ek` and the pT-relaxation solve, record which parameters it uses and which thermodynamic relation it represents.
- [ ] **Step 2: Answer one question** — does a second EOS need to participate in phase change at all? JWL has no temperature or entropy without a caloric extension, and the spec records the decision that JWL with phase change is prohibited. If that holds, `m_phase_change` needs **no** backend dispatch: it stays stiffened-gas-only, and the work is limited to routing its mixture coefficients through the shared rule.
- [ ] **Step 3: Report the answer before writing code.** This is the difference between a small task and the largest one in the programme.

### Task D2: the relaxation isentrope

- [ ] **Step 1:** Out of scope for this plan. `s_equilibrate_pressure` needs `rho(p)` and `drho/dp` per phase in closed form; JWL has neither. Numerical-methods work, not refactoring. It gates JWL in 6-equation multi-fluid, and nothing else in this plan depends on it.

---

## Group E: dispatch and JWL

### Task E1: `eos_model(i)` plumbing

- [ ] **Step 1:** Add `eos_model` per fluid, device-resident, defaulting to stiffened gas, with named constants. Registration per `.claude/rules/common-pitfalls.md`: `_r()` + `_nv()` in `params/definitions.py`, `case_validator.py` entry, and explicit `GPU_UPDATE(device=...)` in **both** `m_global_parameters.fpp` and `m_start_up.fpp`.
- [ ] **Step 2:** Dispatch inside the per-fluid loop of each operator. The branch is wavefront-uniform, so it is nearly free — verify with a kernel-diff rather than trusting that.
- [ ] **Step 3: Two tests that can actually fail.** A `ppn = 2` case, confirmed to fail without the broadcast emitter; and a case registering stiffened gas under a second model tag, asserting bit-identical results, so a no-op dispatch is distinguishable from a real one.
- [ ] **Step 4:** Validator prohibitions: non-stiffened-gas `eos_model` requires `model_eqns = 3`; JWL with phase change is refused.

### Task E2: the JWL backend

- [ ] **Step 1:** Per-fluid parameters `A`, `B`, `R1`, `R2`, `omega`, `rho_0`, registered as above with `fluid_pp(i)%jwl_*` members hand-added to `m_derived_types.fpp`.
- [ ] **Step 2:** JWL branches in energy, pressure and sound speed. All three are closed-form in `(rho, e)`; no iteration needed.
- [ ] **Step 3:** A case exercising JWL that would fail under stiffened gas. New goldens only.

---

## Verification for every task

- [ ] `./mfc.sh format -j 8`, then `./mfc.sh precheck -j 8` — **check the exit code before committing**
- [ ] Kernel-resource diff against a freshly built baseline, for anything in `src/simulation/`
- [ ] Benchmark with the case that exercises the changed code, not a generic one
- [ ] Read the diff before committing. On this work a stale doc comment, an orphaned `qv_visc`, and two dead enthalpies all survived clean builds, passing tests and unchanged kernels; only reading the diff caught them.

## Notes for the executor

- Groups A and B are mechanical and low risk. Group C adds two small operators. Group D is a scoping question, not a coding task. Group E is the actual feature.
- The families are not four, as the spec originally said, nor five. Ten are catalogued. If you find an eleventh, add it rather than fitting it into an existing one.
- "No test exercises this" is not "this is unsupported". Several changes in this work altered unexercised paths; each was called out as a behaviour change rather than folded into a refactor.
