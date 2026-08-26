# GPU kernel resource candidates

Static resource usage from the AMDGPU code object, gfx90a, AFAR amdflang, OpenMP offload,
recorded 2026-08-25 at commit `4b5c232c`. Read with
`toolchain/mfc/kernel_resources.py <binary>`.

**These are candidates, not findings.** Resource pressure is not time: a 21 KB kernel that runs
rarely costs less than a 236 B one that dominates. Ranking by time needs `rocprofv3 --kernel-trace`,
which has not been run. What makes the list worth keeping is the calibration point below.

**Calibration.** PR #1714 took the HLLC kernels to 4880 B scratch / 394 VGPR / 138 AGPR and cost
20% wall-clock on `5eq_rk3_weno3_hllc`. Everything above that line is worth a look.

| kernel | scratch | VGPR | AGPR | vs #1714 |
|---|---:|---:|---:|---|
| `m_riemann_solver_hlld::hlld_riemann_solver` | 21440 B | 470 | 214 | 4.4x |
| `m_viscous::compute_viscous_stress_cylindrical_boundary` | 17760 B | 400 | 144 | 3.6x |
| `m_boundary_common::populate_beta_bc_direction` | 3864 B | 400 | 144 | ~0.8x |
| `m_qbmm::mom_inv` | 3648 B | 400 | 144 | ~0.7x |
| `m_bubbles_el::enforce_el_bubbles_boundary_conditions` | 2440 B | 400 | 144 | ~0.5x |
| `m_bubbles_el::compute_bubble_el_dynamics` | 2376 B | 400 | 144 | ~0.5x |
| `m_riemann_solver_lf::lf_riemann_solver` | 2252 B | 118 | 2 | scratch only |
| `m_ibm::ibm_correct_state` | 2168 B | 250 | 2 | |
| `m_riemann_solver_hypo_hlld::hypo_hlld_riemann_solver` | 1568 B | 446 | 190 | registers |

## Notes

**`hlld` is at minimum occupancy.** 470 VGPR + 214 AGPR is 684 registers; gfx90a has 512 per SIMD
unified, so that kernel runs one wave per SIMD. If MHD cases are slow, start here.

**The 400/144 signature repeats** across five unrelated kernels - viscous, boundary, qbmm and both
Lagrangian bubble kernels. Five independent problems producing identical register counts is
unlikely; a common cause is more plausible. #1759 reports the amdflang Attributor exceeding its
`AAPointerInfo` cap and degrading ISA image-wide by 2.4-4.5x, which fits. **Measure with that PR's
`-attributor-max-pi-accesses=16384` before hand-optimising any of these** - it may move all of them
at once, and optimising around a compiler artefact would be wasted work.

**`lf` and `m_ibm` are a different shape** - high scratch, ordinary registers. That is spilled
arrays rather than register pressure, and has a different fix.

## Related

Fixed already: the averaged Riemann state (`rho_avg`, `H_avg`, `c_avg` and the Roe average's eight
square roots per face) was computed in every solver and read only under `wave_speeds = pressure`,
which is not the default. Now guarded.
