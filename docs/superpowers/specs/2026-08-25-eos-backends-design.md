# Equation-of-State Backends in MFC

**Date:** 2026-08-25
**Status:** design approved, not yet implemented
**Issues:** #1638 (second EOS), #1708 (open-coded EOS expressions), #1700 (selector), #1762 (sound speed, landed)

## Problem

MFC hard-codes the stiffened-gas equation of state. Adding a second one — JWL is the
immediate target, others expected — is blocked not by any single interface but by the EOS
being written out by hand in four separate expression families:

| family | where it lives | status |
|---|---|---|
| sound speed | one routine, 26 call sites | centralised by #1762 |
| energy | open-coded in every Riemann solver and in CBC | open |
| pressure inversion | four open-coded copies, one known wrong (#1709) | open |
| enthalpy | `s_compute_enthalpy` plus five open-coded sites | vestigial after #1762 |

The sound speed was the least entangled of the four. Most of the EOS knowledge that a
second backend must displace lives in the energy expression
`E = gamma*pres + pi_inf + rho*|u|^2/2 + qv`, which callers compute inline and feed back
into the solver as `E` and `H`.

## Requirement

Backends must be able to **coexist within one simulation**: fluid 1 JWL, fluid 2 stiffened
gas, in the same run. This rules out the compile-time selection MFC already uses for
chemistry (a fypp `#:if not chemistry` switch, built with `-D chemistry=False`) and forces
per-fluid runtime dispatch.

## Constraints

### Measured: no derived types in device operator signatures

A `type(...)` dummy in a routine carrying `$:GPU_ROUTINE(parallelism='[seq]')` collapses
amdflang's register allocation across the whole enclosing loop body. Measured on one MI210
(gfx90a), AFAR amdflang, OpenMP offload, `5eq_rk3_weno3_hllc`, comparing #1714 against its
own base:

| | scratch/work-item (3 HLLC kernels) | VGPR | AGPR | ns/gp/eq/rhs |
|---|---|---:|---:|---|
| master | 28 / 236 / 108 B | 134 | 2 | 2.720 |
| with `type(eos_state)` dummy | 4880 / 5072 / 4960 B | 394 | 138 | 3.266 (+20.1%) |
| same, dummy declared `value` | 4880 / 5072 / 4960 B | 394 | 138 | 3.202 (+17.7%) |

`value` does not help — the code object is byte-identical. The cost is not the constructor
call and not the size of the struct copies; it is roughly 25x larger than the aggregate
being copied. Scalars and arrays are unaffected (`adv` is an array and is fine).

Consequences for this design: operators take scalars and arrays only; parameter tables live
at module scope rather than being passed; and any future abstraction reaching for a derived
type, a procedure pointer, or an indirect call must be measured before adoption, not
assumed.

### Structural: mixture rules are a consequence of the EOS

`gamma = sum(alpha_i * gammas_i)` and `pi_inf = sum(alpha_i * pi_infs_i)` are not a
convention. They hold because the stiffened-gas form admits a linear mixture in those
variables. JWL does not. Where backends coexist there is therefore no mixture `gamma`, which
is why the operators cannot keep `gamma` and `pi_inf` in their signatures.

### Structural: the relaxation solver needs a closed-form isentrope

`s_equilibrate_pressure` (`src/simulation/m_pressure_relaxation.fpp`) already performs a
per-cell Newton-Raphson pressure equilibration inside a `[seq]` device routine, 50 iterations
maximum, tolerance 1e-10. Iterative equilibration on device is therefore an existing,
already-paid cost, not a new one.

That solver is stiffened-gas-specific in three places: the per-phase pressure inversion, the
isentrope `rho = rho_0 (p/p_0)^(1/gamma)` (Saurel et al., JFM 2009), and that isentrope's
analytic derivative. JWL's pressure inversion is closed-form and ports directly; its
isentrope is **not** analytically invertible for `rho(p)`. This is the hardest part of the
program and is deliberately scoped as separate, numerical-methods work.

## Approach

Per-fluid runtime dispatch, restricted to the six-equation model.

Rejected alternatives:

- **Compile-time selection**, as chemistry does. Cheapest and precedented, but cannot express
  mixed backends in one build. Ruled out by the coexistence requirement.
- **Procedure pointers / abstract interfaces.** The conventional answer, and unsafe here:
  indirect calls in device code are unevenly supported across nvfortran, Cray and amdflang,
  and this codebase's device codegen already fails on something far milder.

## Operator layer

A new `src/common/m_eos.fpp` owns the parameter tables, the model tag, and the operators.
`m_variables_conversion.fpp` uses it. The split exists because that file is already about
1300 lines and doing several jobs; the EOS should be readable without it and vice versa.

Operators take scalars and arrays only:

```fortran
s_compute_speed_of_sound(pres, rho, adv, c)
s_compute_energy(pres, rho, adv, vel_sum, E)
s_compute_pressure(E, rho, adv, vel_sum, pres)
```

Enthalpy does not need its own operator once energy exists: `H = (E + p)/rho` at every site
that still wants it.

Note that `s_compute_speed_of_sound` as landed in #1762 is
`(pres, rho, gamma, pi_inf, adv, c)` — it still names stiffened-gas coefficients. Dropping
`gamma` and `pi_inf` is part of this work, not already done.

## Dispatch

Dispatch happens inside the per-fluid loop the operators already run:

```fortran
do i = 1, num_fluids
    if (eos_model(i) == eos_stiffened_gas) then
        ...
    else if (eos_model(i) == eos_jwl) then
        ...
    end if
end do
```

`eos_model(i)` is a module-scope integer array, device-resident. Every lane of a wavefront is
at the same `i` at the same time, so the branch is wavefront-uniform and costs essentially
nothing. This is what makes per-fluid runtime dispatch viable on GPU at all; branching on
cell contents would not be.

## Parameters

JWL needs `A`, `B`, `R1`, `R2`, `omega` and `rho_0` per fluid, alongside the existing
`gammas`, `pi_infs`, `qvs`, `gs_min` and `ps_inf`. In this codebase that means, for each:

- a `fluid_pp(i)%jwl_*` member added by hand to `src/common/m_derived_types.fpp`
  (derived-type members are not auto-generated);
- `_r()` definition and `_nv()` registration in `toolchain/mfc/params/definitions.py`;
- a `case_validator.py` entry with a `PHYSICS_DOCS` entry, since these are
  physics-constrained;
- an explicit `$:GPU_UPDATE(device=...)` in **both** `m_global_parameters.fpp` and
  `m_start_up.fpp`. `GPU_DECLARE` alone does not make data device-resident.

## Model restrictions

Mixture `gamma` and `pi_inf` remain meaningful only when every fluid is stiffened gas. The
unified mixture-coefficient routine therefore stays stiffened-gas-only and gains a guard, and
`case_validator.py` prohibits any non-stiffened-gas `eos_model` unless `model_eqns = 3`.

This boundary is deliberate: it keeps the program out of inventing a cross-EOS mixture rule,
which has no closed form and is a research problem rather than an engineering one.

## Decomposition

This document specifies the programme. It is deliberately larger than one implementation
plan: each piece below gets its own plan, its own PR and its own verification, and the
pieces are ordered so that each is reviewable without the ones after it.

Each piece lands independently and keeps goldens neutral except where a change is deliberate.

| # | piece | notes |
|---|---|---|
| 0 | sound speed | #1762, landed |
| 1 | single home for the mixture accumulation | `s_accumulate_mixture_properties` (in `m_riemann_state.fpp`) and the `else` branch of `s_convert_species_to_mixture_variables_kernel` (in `m_variables_conversion.fpp`) contain the *same* four-line accumulation loop. Move the former into `m_variables_conversion.fpp` and have the kernel call it. Behaviour-neutral and golden-neutral. See the note below on why these two routines are **not** merged. |
| 2a | energy operator: Riemann solvers + CBC | Hot path and the densest duplication. Unblocks per-fluid dispatch in the flux. |
| 2b | energy operator: IGR | `m_igr.fpp` carries its own EOS algebra (8 sites); a separate solver family whose usage has not yet been characterised. |
| 2c | energy operator: diagnostics | `m_data_output` in all three targets, `post_process/m_derived_variables`, `m_sim_helpers`. Cold path, low risk. |
| 2f | the remaining subsystems | `m_ibm.fpp` (energy from an image point, one variant scaled by `(1 - alpha_IP(1))`) and `m_bubbles_EL.fpp:841` (energy). Neither is a solver, a diagnostic or pre-process, so both fell through the first two cuts. |
| 2e | energy operator: the conversion routines | `m_variables_conversion.fpp` itself, 17 sites including `s_convert_primitive_to_flux_variables`. Separated because operator and callers would then share a file, and the flux routine has its own chemistry branch. |
| 2d | energy operator: pre_process initial conditions | `m_assign_variables`, `m_check_patches`. Different lifecycle - runs once, no GPU - and a second EOS getting the ICs wrong is silent. |
| 3 | pressure inversion operator | Four copies collapse to one; #1709's known-wrong site is corrected here. |
| 4 | `eos_model(i)` and per-fluid dispatch | Plumbing into the centralised operators, stiffened gas as the only backend. Behaviour-neutral by construction. Drops `gamma`/`pi_inf` from the operator signatures. |
| 5 | JWL backend | Sound speed, energy, pressure. Cannot start before 4. |
| 6 | relaxation isentrope | Separate track. `rho(p)` and `drho/dp` per phase; JWL has neither in closed form, so this needs a nested sub-iteration or a reformulated relaxation. |

### Survey: the actual size of the energy piece

A grep for assignments mixing `gamma` and `pi_inf` finds roughly 69 lines of stiffened-gas algebra
across 18 files, spanning all three executables - not the six solver files this document originally
assumed. By file: `m_variables_conversion` 17, `m_riemann_solver_hllc` 9, `m_igr` 8,
`m_riemann_solver_hypo_hlld` 7, `hll` and `hlld` 4 each, `post_process/m_derived_variables` 3, and
one or two each in `m_sim_helpers`, `lf`, `m_ibm`, `m_hypoelastic`, `m_cbc`, `m_bubbles_EL`,
`m_pressure_relaxation`, `pre_process/m_assign_variables`, `pre_process/m_check_patches`,
`post_process/m_start_up` and `post_process/m_data_output`.

The families that a first, narrower search missed entirely - IGR, hypoelastic, Lagrangian bubbles,
pre-process initial conditions, and post-process derived variables - are why piece 2 is split into
2a through 2d above. Any survey of this kind should start from `pi_inf`/`ps_inf` occurrences rather
than from a guessed expression shape.

Two variants to expect while doing the work. The MHD sites add `pres_mag`; the IBM sites use
`dyn_pres` and one wraps the result in `(1 - alpha)`. Neither is a defect.

One dead assignment turned up in the survey and should simply be deleted: `m_cbc.fpp:657` computes
`E` on the non-chemistry path, but `E` is read only inside the `if (chemistry)` branch of the flux
update. Its omission of `qv` is therefore harmless - the value is never used. The non-chemistry
energy flux carries `qv` correctly through `dqv_dt`.

### Why the two mixture routines are not merged

They look like duplicates and are not. `s_accumulate_mixture_properties` is a *parameterised
subset* accumulator: callers pass `nf` as either `num_fluids` or `num_fluids - 1` (excluding
the gas phase in the bubbles path), and pass either `alpha_L` or the limited `alpha_lim_L`.
`s_convert_species_to_mixture_variables_kernel` always runs over `num_fluids`, clips and
renormalises `alpha_K` in place under `mpp_lim`, special-cases
`num_fluids == 1 .and. bubbles_euler`, and optionally emits `Re_K` and `G_K`.

Merging them would require `nf`, a clipping flag, and optional `Re_K`/`G_K` outputs on a
`[seq]` device routine — reintroducing the optional-dummy pattern that #1714's review flagged
and that was removed for backend portability. Only the shared accumulation loop is extracted;
the differing wrappers stay.

The extracted routine keeps the name `s_accumulate_mixture_properties` and lives in
`m_variables_conversion.fpp` until piece 4 relocates it to `m_eos.fpp` alongside the
operators. It is the single place where per-fluid EOS dispatch enters the stiffened-gas
mixture path.

Two cross-cutting items land **before** piece 2, because pieces 2 through 5 all touch hot
device code and a repeat of the #1714 regression would otherwise be invisible until someone
benchmarked on the right hardware:

- A GPU section entry in `.claude/rules/common-pitfalls.md` recording the derived-type
  constraint above.
- A kernel-resource CI check: dump `private_segment_fixed_size`, VGPR and AGPR counts for
  named kernels from the AMDGPU code object and fail on regression past a threshold. It needs
  a GPU to compile, not to run; it takes seconds; and it is deterministic and
  board-independent.

## Audit findings folded in

**A fifth expression family exists.** The four families listed above - sound speed, energy, pressure
inversion, enthalpy - are not exhaustive. `m_hypoelastic.fpp:629-630` computes a **bulk modulus**:

```fortran
blkmod1_K = ((gammas(1) + 1._wp)*pres_K + pi_infs(1))/gammas(1) + (4._wp/3._wp)*Gs_hypo(1)
```

That is the `alt_soundspeed` expression plus a shear term. It is EOS-dependent and needs a backend
branch, but belongs to neither the energy nor the pressure piece. It gets its own piece when someone
scopes it; it is recorded here so it is not discovered a third time.

`m_bubbles_EL.fpp:376` is a **pressure inversion**, not an energy site, and belongs to piece 3.

**Why the enumeration kept missing things.** Both re-cuts surveyed by file and then assigned by
subsystem name, so anything not obviously a solver, a diagnostic or pre-process fell through - IBM,
hypoelastic and Lagrangian bubbles were missed twice. Any future re-cut must assign every file the
survey returns, including single-site files, and classify each site by *expression family* rather
than by the subsystem its file belongs to.

**Open decision: convert once or twice.** As ordered, pieces 2a-2f route roughly 69 sites through
operators that still take `gamma` and `pi_inf`, and piece 4 then removes those arguments and edits
every one of those sites again - two rounds of edit, review and golden risk on the hottest code in
the solver, to reach a signature already decided on.

The alternative is to settle the operators' final signature first, which turns on whether the
mixture coefficients are derived inside the operator from `adv` or passed in, and convert once. That
question is not free: deriving them inside is not equivalent today, because
`s_convert_species_to_mixture_variables_kernel` special-cases `num_fluids == 1 .and. bubbles_euler`
and clips `alpha_K` in place under `mpp_lim`. Those are bounded problems, and paying them once is
probably cheaper than editing 69 hot-path lines twice. **Not yet decided.**

**Method corrections for every piece's verification:**

- Measure speed before correctness. The kernel dump takes seconds, the benchmark about twelve
  minutes and the suite over an hour - and these refactors preserve the arithmetic by construction,
  so the cheap check is also the one most likely to fail.
- Compare kernel resources by *resource profile*, not by kernel name. Names embed post-fypp line
  numbers, so an unrelated edit shifts them; a naive name-keyed diff of a neutral change produced
  forty spurious differences.
- Dump the baseline from a binary just built on the base commit. A stale install directory silently
  produced a baseline one commit older than intended.
- `./mfc.sh test -% 50` runs a uniform random sample and is a reasonable local gate; CI is the
  exhaustive one. Say in the commit when a sample rather than the full suite was run.
- The suite does exercise MHD: 37 cases match MHD or HLLD.

## Full inventory of equation-of-state usage

Surveyed by *parameter array* rather than by expression shape or subsystem. Two earlier attempts
missed files because they searched for a guessed expression, then assigned by subsystem name; this
list starts from `gammas`, `pi_infs`, `qvs`, `gs_min`, `ps_inf`, `cvs`, `qvps` and assigns every file
that consumes one.

### By expression family

| # | family | where | status |
|---|---|---|---|
| 1 | sound speed | `m_variables_conversion`, 26 solver/diagnostic sites | centralised (#1762) |
| 2 | mixture coefficients | five Riemann solvers; **`m_viscous.fpp:145`** | solvers done; `m_viscous` keeps its own copy |
| 3 | energy | five Riemann solvers; `m_cbc`; `m_ibm`; `m_bubbles_EL`; conversion routines; pre/post | solvers done; rest open |
| 4 | pressure inversion | `m_pressure_relaxation`, `m_data_output` x2, `m_bubbles_EL:376` | open (piece 3) |
| 5 | enthalpy | solvers (as `(E+p)/rho`); `m_phase_change` `hk` | closed for solvers; phase change open |
| 6 | bulk modulus | `m_hypoelastic:629-630`, `m_rhs:1075-1077` | open, unassigned |
| 7 | temperature inversion | `m_reactive_burn:56`, `m_phase_change` | open, unassigned |
| 8 | entropy | `m_phase_change` `sk` | open, unassigned |
| 9 | caloric relations | `m_phase_change` `ek`, `rhok`, `hk` (via `cvs`) | open, unassigned |
| 10 | mixture coefficients + sound speed, Tait naming | `m_acoustic_src:232-252`; also `m_qbmm`, `m_bubbles_EE`, `pre_process/m_assign_variables` | **not** a second EOS: `B_tait` is `sum(alpha*pi_infs)` and `small_gamma` is `sum(alpha*gammas)` converted to Gamma at line 251. Seventh open-coded copy of the mixture rule, plus an open-coded sound speed |

### By file, for the consumers no piece covered

| file | refs | note |
|---|---|---|
| `m_phase_change.fpp` | ~104 | the largest EOS consumer in the codebase; `gs_min` 35, `cvs` 32, `ps_inf` 24, `qvs` 10, `qvps` 3 |
| `m_viscous.fpp` | 32 | accumulates mixture coefficients itself, from `alpha_visc` |
| `m_igr.fpp` | 20 | piece 2b |
| `m_rhs.fpp` | 6 | bulk modulus with a shear term |
| `m_acoustic_src.fpp` | 6 | mixture rule + sound speed, under Tait naming |
| `m_reactive_burn.fpp` | 3 | temperature inversion |

`cvs` and `qvps` are consumed almost exclusively by `m_phase_change`. No operator built so far touches
them.

## Edge cases that need a decision, hardest first

**1. JWL may be thermodynamically incompatible with phase change.** `m_phase_change` needs entropy,
temperature and per-phase enthalpy - a thermodynamically complete equation of state. Stiffened gas
has closed forms for all of them via `cvs`, `gs_min`, `ps_inf`, `qvs`, `qvps`. JWL as normally
written is a *mechanical* EOS, `p(rho, e)`, with no temperature or entropy unless a caloric
extension is supplied. This is a physics limit, not an implementation gap, and it bounds what
"support EOS backends generally" can mean.

Prefer a validated opt-out to a hard prohibition, so unusual pairings stay reachable deliberately
rather than by accident. The default should still be that an unvalidated pairing is refused rather
than silently producing a wrong answer.

**2. The Tait-named code is stiffened gas, and converts mechanically.** `m_acoustic_src` builds
`B_tait = sum(alpha*pi_infs)` and `small_gamma = sum(alpha*gammas)`, then at line 251 converts to
`Gamma = 1/gamma + 1` and computes a sound speed. It is the mixture rule and the sound speed
open-coded under different names, both already covered by existing operators - a straightforward
conversion, not a design problem.

The `n_tait`/`B_tait` uses in `m_qbmm`, `m_bubbles_EE` and `pre_process/m_assign_variables` are
different: they are Tait-form *isentrope* relations for bubble dynamics, e.g.
`((1 + B)/(p + B))**(1/n)`. That is the same closed-form isentrope the pressure relaxation needs and
the thing JWL lacks, so they belong with edge case 3 rather than with the mechanical conversions.

**3. The relaxation isentrope.** `s_equilibrate_pressure` needs `rho(p)` and `drho/dp` per phase in
closed form. JWL's isentrope is not analytically invertible. Needs a nested iteration or a
reformulated relaxation. Numerical-methods work, tracked as piece 6.

**4. `m_viscous` accumulates from `alpha_visc`, not `adv`.** Whether it can use the shared rule
depends on whether `alpha_visc` carries the same meaning; under `bubbles_euler` the advection slot
aliases the void fraction, so this needs checking rather than assuming.

**5. `adv` is model-dependent.** Under `bubbles_euler` with `num_fluids == 1` the sole advection slot
aliases `alf`. Any operator taking `adv` must go through `s_compute_mixture_coefficients`, which
knows this. This is settled for the solvers and must not be re-derived elsewhere.

**6. Mixed backends within a cell.** Restricted to `model_eqns = 3`, where `adv` is genuinely a
composition. Unchanged from the approach section.

## Correction to scope claims

The work landed so far centralises the equation of state **for the Riemann solvers**. That is real,
but it is not the whole equation of state: `m_phase_change` alone is a comparable body of EOS
algebra, and families 6 through 10 above are untouched. Any claim that a second EOS can be added "in
one place" is premature until at least families 2 through 4 are complete and 1, 2 and 4 in the edge
case list are decided.

## Verification

| piece | goldens | additional |
|---|---|---|
| 1 | neutral — pure extraction, identical arithmetic | full suite must be bit-identical; no golden regeneration is acceptable here |
| 2 | neutral | kernel-resource check |
| 3 | neutral except #1709's corrected site | |
| 4 | neutral by construction | see both traps below |
| 5 | new goldens only | a case exercising JWL that would fail under stiffened gas |

Two traps this codebase documents, both of which apply to pieces 4 and 5:

**The broadcast trap.** `fluid_pp(i)%jwl_*` are derived-type parameters, and
`generated_bcast.fpp` covers namelist scalars only. Omitting the hand-written emitter leaves
every non-root rank holding the `dflt_real` sentinel, which a single-rank golden cannot
detect. Pieces 4 and 5 each need a `ppn = 2` case, confirmed to fail without the emitter
before it is trusted.

**The test-that-cannot-fail trap.** Piece 4 is behaviour-neutral, so a green suite proves
nothing about whether dispatch is wired at all — `eos_model` could be ignored entirely and
everything would still pass. It needs a test that fails when dispatch is broken. The simplest
is to register stiffened gas under a second model tag and assert bit-identical results, which
distinguishes a real dispatch from a no-op one.

## Known unknowns

These are carried deliberately rather than resolved here.

1. **Whether `s_compute_pressure`'s corrections commute with a non-stiffened-gas inversion.**
   The bubbles variant divides by `(1 - alf)` and the hypoelastic variant subtracts elastic
   energy before inverting. Both assume a linear inversion. Whether either is meaningful for a
   JWL phase is a physics question, to be answered in piece 3.

2. **The form of the JWL isentrope sub-iteration**, and whether a nested Newton inside
   `s_equilibrate_pressure` converges acceptably on device within the existing 50-iteration
   budget. Piece 6.

3. **Whether any target case actually needs JWL in a multi-fluid cell**, as opposed to
   per-phase under `model_eqns = 3`. If none does, the `model_eqns = 3` restriction costs
   nothing; if some does, the restriction becomes the binding limitation and the mixture
   question reopens.

## Related work not in this program

- `s_compute_enthalpy` has no consumer for its enthalpy at any of its three callers after
  #1762, and should be trimmed or renamed to what it computes.
- `c_avg` is consumed only under `wave_speeds = pressure`, so in the default configuration
  every solver computes an averaged state and an EOS call per face and discards the result.
- `s_compute_speed_of_sound_avg` mirrors the mixture-model branch list of
  `s_compute_speed_of_sound`; the two can drift. Removing the mirror means narrowing the
  averaged routine to the quantities an average legitimately has and prohibiting the
  mixture-EOS combinations at validation.
