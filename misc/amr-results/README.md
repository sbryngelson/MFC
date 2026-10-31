# AMR results on Frontier (snapshot 2026-10-30)

Figures, data and reports from the MFC AMR work on OLCF Frontier (MI250X, AMD AFAR compilers). Code: branch
`spike/amr-integrate` (7273d602 = all fixes below, validated). Movies are not in git (`*.mp4` is ignored); paths below.

## Benchmark
Blast lattice (the AMReX CNS / PeleC weak-scaling case): ideal gas, periodic, 8M base cells per GPU, one refined level, regrid every
20 coarse steps; s/step = (wall at 240 steps - wall at 40)/200, 2 reps per point, one case-optimized MFC build per ladder.
MFC: WENO5 + HLLC, 5-equation single fluid (`mp_weno = F`, `amr_tag_abs = T`, `amr_place_owner = T`); AMReX CNS and PeleC:
their second-order schemes. Same grid, dt, tag rule and refinement ratio; MFC refines ~1.3x more volume (40.5% vs 30.3%
subcycled). AMR-only = wall minus each code's own hydro RHS.

## Headline (subcycled, integrated build 5d8ab069)
| GPUs | MFC s/step | AMReX CNS | PeleC | MFC efficiency | AMReX | PeleC | MFC AMR-only | AMReX | PeleC |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.765 | 0.467 | 0.600 | 1 | 1 | 1 | 0.217 | 0.281 | 0.141 |
| 64 | 0.797 | 0.600 | 0.802 | 0.96 | 0.78 | 0.75 | 0.233 | 0.408 | 0.328 |
| 128 | 0.833 | 0.613 | 0.832 | 0.92 | 0.76 | 0.72 | 0.260 | 0.421 | 0.358 |
| 512 | 0.899 | 0.678 | 0.952 | 0.85 | 0.69 | 0.63 | 0.313 | 0.486 | 0.478 |

Lock-step at 512 GPUs: MFC 0.282 s/step (efficiency 0.89) vs AMReX 0.250 (0.68). Before the fixes (651b6d88) MFC subcycled was
1.574 -> 2.183 s/step (8 -> 512 GPUs). The subcycling driver + swap packing (7273d602) cut a further ~12%: 0.699 s/step at
64 GPUs (AMR-only 0.141), see `data/scaling_simple_deck_v3.csv`. 1024-GPU rung and the 7273d602 build at 128-1024 GPUs pending.

## Sponsor figures (`pgf/`)
pgfplots sources, data tables and PDFs; PNG previews in `pgf/png/`. `*_pelec` variants compare MFC with PeleC only. Rebuild
instructions in `pgf/README.md`.
| figure | shows |
|---|---|
| `scaling_slide` | efficiency, AMR-only cost added beyond 8 GPUs, s/step vs GPUs |
| `lockstep` | lock-step efficiency and s/step, MFC vs AMReX CNS |
| `tts` | wall time to the same physical time at 128 GPUs |
| `fixes` | what each fix bought, MFC subcycled at 64 GPUs (1.95 -> 0.70 s/step) |
| `amr_only` | AMR-only cost vs GPUs, subcycled and lock-step |
| `coverage` | refined share of the domain |
| `interface_error` | coarse-fine interface error vs the uniform fine grid, minmod vs quartic |
| `rhs_models` | RHS ns/cell for the same numerics in HIP vs OpenMP variants |
| `convergence` | subcycling order of accuracy (quartic: second order) |

Rebuild everything with `pgf/build_all.sh`.

## Ladders (`data/`)
- `scaling_simple_deck_v2.csv`, `ladder_v2/` (README, VERSIONS, scripts), `figures/scaling_simple_deck_v2.png`: integrated
  build 5d8ab069 plus the 651b6d88 baseline, AMReX CNS and PeleC, 8-512 GPUs.
- `scaling_simple_deck_v3.csv`, `ladder_v3/`, `figures/scaling_simple_deck_v3.png`: subcycling driver and swap packing at
  8/64 GPUs; tile clustering findings (out of memory with subcycling: store slots sized for full blocks; post-meeting fix).
- `scaling_simple_deck.csv`, `figures/scaling_simple_deck.png`: the pre-fix single-build ladder (651b6d88).
- `progress_data.json`: data behind the live progress page.

## Pre-fix analysis (superseded by the fixes; kept for the root-cause record)
| file | shows | caveats |
|---|---|---|
| `figures/amr_only_vs_gpus.png` | AMR-only s/step per code, 8-512 GPUs | MFC np8/np64 used an earlier build than np128/np512 |
| `figures/stacked_sub.png`, `stacked_nosub.png`, `categories_*.png` | AMR-only cost by category per code and GPU count | MFC flux capture/copies moved out of its RHS by an estimate |
| `figures/coverage.png` | refined share of the domain per code | read at the grid used in the last timed steps |
| `figures/overhead_vs_gpus.png` | AMR overhead factor (total / own RHS) | flattered by a heavy RHS |
| `figures/waterfall.png`, `budget_bars.png` | waste budget: step time vs after each pending fix vs the floor (np8) | hatched bars are estimates |
| `figures/subcycled_overhead_breakdown.png` | where subcycled AMR overhead went, by root cause | np8, one run |
| `figures/halo_vs_load_np8.png` | halo wait vs per-rank load: skew from uneven parent-data gathering | |
| `figures/rhs_breakdown.png` | hydro RHS ns/cell/stage by kernel group, MFC vs mini-app | MFC ran MP-WENO |
| `figures/omp_vs_hip.png` | same algorithm in HIP vs Fortran OpenMP with MFC's compiler | |
| `figures/interface_error_vs_uniform.png` | coarse-fine interface error vs uniform grids | |
| `figures/subcycle_error_map_{before,after}_seeding_fix.png` | subcycled minus lock-step error before/after quartic seeding | |
| `figures/render_*_contact_sheet.png` | volumetric render looks | |

Data: `data/fair_metrics.csv`, `categories_long.csv`, `waste_budget.csv`, `waterfall.csv`, `omp_vs_hip.csv`, `rhs_groups.csv`.

## Reports
`reports/overhead_findings.md`, `reports/capture_and_migration_findings.md` (MFC root causes with file:line),
`reports/miniapp_findings.md`, `reports/miniapp_scaling.md` (reference C/HIP mini-app), `reports/waste_budget.md`,
`reports/audit_*.txt` (independent re-derivation of the headline numbers).

## Movies (not in git)
- 64-sphere particle curtain (Mach 2.5, 80 GPUs): `/lustre/orion/cfd154/scratch/sbryngelson/amr-report/movies/curtain64_80gpu.mp4`
  (1080p) and `_720p.mp4`.
- 8 spheres (Mach 1.5, 8 GPUs): `/lustre/orion/cfd154/scratch/sbryngelson/amr-report/movies/spheres3d_8gpu_v2.mp4`.

## Stills (`stills/`, 1600-px JPEG)
- `curtain64_hero4k_{close,wide,cut}[_caption].jpg`: 4K hero renders (full resolution in amr-report/hero/).
- `curtain64_{flow,wake,cut,hero}_{144,440,904,1696}.jpg`: Mach 2.5 shock through 64 force-driven spheres (viscous, collisions
  on, 80 GPUs) at t = 0.49, 1.50, 3.09, 5.80; `curtain64_movie_f_*.jpg`: movie frames.
- `spheres8_{flow,cut,hero}_{400,900,1700}.jpg`, `spheres8_movie_f_*.jpg`: the 8-sphere run.
- `curtain_test_midplane_slice.jpg`: mid-plane slice of the 22-sphere curtain test (8 GPUs).

## Known issues (open)
- **Quartic prolongation blowup in a dense IB cloud.** Sphere-cloud movie case (297 fixed spheres, Mach 2.5, 128 GPUs; case
  `/lustre/orion/cfd154/scratch/sbryngelson/mfc-movie/case.py`, branch `spike/amr-movie`): with `amr_prolong = 2` the run goes
  NaN at steps 363-386, 1-3 steps after a regrid, at a refined-block corner or the cloud edge. Not fixed by regrid every 4,
  a 4-cell clear ring, dt x0.75, inviscid, or the older commit d8f901dd; `amr_prolong = 1` from the step-300 restart runs
  cleanly (used for the movie). Likely cause: the unlimited 5-point quartic stencil straddles IB solid/ghost cells or a
  shock and undershoots. Proposed fix: fall back to linear where the stencil touches IB cells or a strong gradient, plus a
  positivity check on prolonged values. Reproducer: the step-300 restart.
- **Tile clustering with subcycling runs out of GPU memory** (store slots sized for full blocks); lock-step tiles are
  2-2.7x slower with no coverage gain on the blast lattice. Fix: per-block slot sizing.
- **`3D -> synthetic_turbulence` regression test** fails on `up/mega` (7273d602) at round-off level (rel 3.3e-9 vs 1e-10
  tolerance); to revisit when the branch is brought up to date with master.
- Level 2 cannot span multiple level-1 parents; no AMR for `model_eqns = 1`; restart on a different rank count segfaults.
