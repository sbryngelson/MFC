# AMR results on Frontier (2026-10-29/30 snapshot)

Figures, data and reports from the MFC AMR work on OLCF Frontier (MI250X, AMD AFAR compilers). Numbers were re-derived from raw
outputs by an independent audit (`reports/audit_*.txt`); corrections are applied here. Treat MFC scaling rows as provisional
until the single-build ladder (simple deck: 5-equation, 1 fluid, `mp_weno = F`, `amr_tag_abs = T`, two reps) replaces them.

## Benchmark
Blast lattice (the AMReX CNS / PeleC weak-scaling case): ideal gas, periodic, 8M base cells per GPU, one refined level, regrid every
20 coarse steps; s/step = (wall at 240 steps - wall at 40)/200. Compared codes: MFC, AMReX CNS, PeleC, and a C/HIP reference mini-app.

## Figures
| file | shows | caveats |
|---|---|---|
| `figures/amr_only_vs_gpus.png` | AMR-only seconds per coarse step (everything except each code's hydro RHS) at 8/64/128/512 GPUs | MFC np8/np64 used an earlier build than np128/np512; MFC refines more volume (43.9% vs ~30% subcycled) |
| `figures/stacked_sub.png`, `stacked_nosub.png`, `categories_*.png` | AMR-only cost by category (fill, halo/seam, reflux, restriction, regrid, RK/copies, other) per code and GPU count | MFC flux capture/copies moved out of its RHS by an estimate that is mostly idle gap (see audit) |
| `figures/coverage.png` | refined share of the domain per code | read at the grid used in the last timed steps |
| `figures/overhead_vs_gpus.png` | AMR overhead factor (total / own RHS) | flattered by a heavy RHS; secondary metric |
| `figures/waterfall.png`, `budget_bars.png` | waste budget: MFC step time today vs after each pending fix vs the necessary floor (np8) | hatched bars are estimates |
| `figures/subcycled_overhead_breakdown.png` | where MFC's subcycled AMR overhead goes, by root cause | np8, one run |
| `figures/halo_vs_load_np8.png` | halo wait vs per-rank load: the skew comes from uneven parent-data gathering, not block counts | |
| `figures/rhs_breakdown.png` | hydro RHS ns/cell/stage by kernel group, MFC vs mini-app (1 GCD, 200^3) | MFC ran MP-WENO; without it 5.16 vs 6.58 ns/cell |
| `figures/omp_vs_hip.png` | same algorithm in HIP vs Fortran OpenMP with MFC's compiler | one unroll fix closes the OpenMP gap |
| `figures/interface_error_vs_uniform.png` | coarse-fine interface error vs uniform grids, minmod vs quartic, lock-step vs subcycled | uniform fine hits its error floor at 128^2 |
| `figures/subcycle_error_map_{before,after}_seeding_fix.png` | subcycled minus lock-step error maps before/after seeding new fine cells with the quartic | |
| `figures/render_*_contact_sheet.png` | volumetric render looks (8-sphere run, 64-sphere particle curtain) | |

## Data
`data/fair_metrics.csv`, `data/categories_long.csv` (per-code, per-category AMR cost), `data/waste_budget.csv`, `data/waterfall.csv`,
`data/omp_vs_hip.csv`, `data/rhs_groups.csv`, `data/progress_data.json` (the live progress page's data).

## Reports
`reports/overhead_findings.md`, `reports/capture_and_migration_findings.md` (MFC root causes with file:line),
`reports/miniapp_findings.md` (findings F1-F5 from the reference mini-app), `reports/miniapp_scaling.md`,
`reports/waste_budget.md`, `reports/audit_*.txt` (independent re-derivation of the headline numbers).

## Movies
`movies/spheres3d_8gpu_v2_720p.mp4`: Mach 1.5 shock through eight force-driven spheres, 8 GPUs, volumetric shocks and a cutaway of
the refined blocks coloured by GPU.

## Stills
`stills/` (1600-px JPEG):
- `spheres8_{flow,cut,hero}_{400,900,1700}.jpg`: the 8-sphere run (Mach 1.5, 8 GPUs) at t = 0.32, 0.72, 1.36 in three looks:
  volumetric shock field with pressure-coloured spheres; cutaway with refined blocks outlined by GPU; close hero view.
- `spheres8_movie_f_*.jpg`: frames from the 8-sphere movie (flow pass, a callout pause, the closing shot).
- `curtain64_{flow,wake,cut,hero}_{144,440,904,1696}.jpg`: Mach 2.5 shock through a curtain of 64 force-driven spheres
  (viscous, collisions on, 80 GPUs) at t = 0.49, 1.50, 3.09, 5.80: bow shocks, vortex wakes, refinement on the cut face, close view.
- `curtain_test_midplane_slice.jpg`: mid-plane slice of the 22-sphere curtain test (8 GPUs).
