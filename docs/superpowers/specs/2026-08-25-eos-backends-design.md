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
| 1 | reconcile the two mixture implementations | `s_accumulate_mixture_properties` and `s_convert_species_to_mixture_variables_kernel` disagree today. The second special-cases `num_fluids == 1 .and. bubbles_euler` to `gamma = gammas(1)`, `pi_inf = pi_infs(1)` (unweighted), and applies `mpp_lim` clipping and renormalisation to `alpha_K` in place before accumulating; the first does neither, always computing `sum(alpha_i*gammas_i)`. Everything downstream dispatches through one of these, so they must become one first. The narrow scope of the disagreement — single-fluid bubble cases, plus the `mpp_lim` path — bounds the golden impact. |
| 2 | energy and enthalpy operators | Moves the energy expression out of six solver files. Largest refactor; no new EOS. |
| 3 | pressure inversion operator | Four copies collapse to one; #1709's known-wrong site is corrected here. |
| 4 | `eos_model(i)` and per-fluid dispatch | Plumbing into the centralised operators, stiffened gas as the only backend. Behaviour-neutral by construction. Drops `gamma`/`pi_inf` from the operator signatures. |
| 5 | JWL backend | Sound speed, energy, pressure. Cannot start before 4. |
| 6 | relaxation isentrope | Separate track. `rho(p)` and `drho/dp` per phase; JWL has neither in closed form, so this needs a nested sub-iteration or a reformulated relaxation. |

Two cross-cutting items land **before** piece 2, because pieces 2 through 5 all touch hot
device code and a repeat of the #1714 regression would otherwise be invisible until someone
benchmarked on the right hardware:

- A GPU section entry in `.claude/rules/common-pitfalls.md` recording the derived-type
  constraint above.
- A kernel-resource CI check: dump `private_segment_fixed_size`, VGPR and AGPR counts for
  named kernels from the AMDGPU code object and fail on regression past a threshold. It needs
  a GPU to compile, not to run; it takes seconds; and it is deterministic and
  board-independent.

## Verification

| piece | goldens | additional |
|---|---|---|
| 1 | **expected to move** — it resolves a real disagreement | must state which path changed and why |
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
