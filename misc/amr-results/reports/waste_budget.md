# AMR waste budget: blast lattice, L ladder, np8

Necessary floor vs actual cost per category, in seconds per coarse step. The categories add up exactly to the measured step
time (the script asserts this). Regenerate with `../../viz-venv/bin/python waste_budget.py`, which is post-processing only.
The tag floor is cached in `tag_floor.json`; pass `--retag` to recompute it (about 6 min, I/O bound).

Outputs:
- `waste_budget.csv`: config x mode x category, with necessary, waste, actual, coverage, ghost ratio, blocks, in-situ ns/cell
  and source.
- `growth.csv`: per-category s/step at np 8/64/512 from fair-metrics.
- `waterfall.csv`: both modes.
- `budget_bars.png` and `waterfall.png` (subcycled).

## Configurations (np8, steps 40->240, s240 - s40 differenced, rank means)
| config | source | total s/step lock / sub |
|---|---|---|
| MFC default (AMReX tag) | mfc-blast/L_np8_{lock,sub}_s*_abs (fair_metrics.csv row "MFC (AMReX tag)") | 0.579 / 1.988 |
| MFC tile (amr_tile_cluster, bf 8, mgs 16) | mfc-boxtest/blast/C8M16_np8_* | 2.288 / 7.318 |
| mini-app Morton | amr-miniapp/runs/blast/np8-L-*-j5792704 | 0.0949 / 0.2605 |
| mini-app owner | np8-L-sub-owner-j5794576 (no lock-step owner run exists) | - / 0.2509 |
| ref: MFC boxes build, default clustering | mfc-boxtest/blast/L_np8_lock_* (CSV only) | 1.863 / - |

## Floors (necessary)
1. **Refined volume.** The union of AMReX dengrad tags (max one-sided |rho jump| >= 0.008) dilated by amr_buf = 4 level-0 cells.
   It is computed from the level-0 density of the MFC step-240 restart: 13.5 % (lock) and 26.2 % (sub) of the domain. For
   comparison, AMReX is at 15.45 / 30.25 % and the mini-app at 15.5 / 30.0 % at step 220. The window mean is
   floor(240) x the mini-app's (window mean / step-240) coverage ratio, which gives 5.4 / 13.7 %. Cell updates per coarse step
   per GPU = (N0 + r cov 8 N0)/8, with r = 2 if subcycled.
   Necessary floor-volume work is booked under category 3, so category 1 holds only the extra volume, priced at the floor rate.
2. **Ghost work.** The floor is 1 (the fused kernel works on interior faces only).
   - MFC buff_size = weno_polyn + 2 = **4** for this inviscid case (m_helper_basic.fpp:132). The value 2*weno_polyn + 2 = 6
     applies only to viscous cases.
   - The ghost ratio is the sum of (n+8)^3 over the sum of n^3 across the step-240 block catalogs (lustre_amr_240.dat), weighted
     with level 0 (200^3 per rank, 1.125).
   - Results: MFC default 1.16 / 1.22 (64 / 216 blocks of about 106^3). MFC tile 1.47 / 1.83 (4040 / 7592 blocks of 16-32 cells
     per side).
   - Assumes every MFC RHS kernel sweeps the buffered volume, which makes this an upper bound on the ghost share.
3. **Hydro per cell.** The floor is the HIP fused kernel at 1.639 ns/cell/stage, plus the mini-app's in-situ RK update per cell.
   The Fortran OpenMP version with the unroll fix reaches 1.80. MFC's per-buffered-cell rate is 6.585/1.125 = 5.85 ns
   (rhs-profile, 200^3).
4. **C/F fill, halo, reflux, restrict.**
   - The floor is the mini-app's best measured value (minus its load-balance wait).
   - The pure bandwidth bound is 2.2 / 2.8 ms: the union's fine ghost ring read+write at 1.3 TB/s HBM, plus a level-0 halo at
     50 GB/s. The mini-app's 16 / 53 ms is therefore an *achieved* near-floor, not a physical one.
5. **Regrid.** The floor is the mini-app's best measured value.
6. **Host overhead.** The floor is 0. The actual value is the "other" phase plus the RHS time above (cells x 5.85 ns x ghost
   ratio), i.e. kernel fragmentation and launch/host gaps in situ.
7. **Load balance.** The floor is 0. The actual value is the sum of per-phase maxima minus means of the RHS (MFC, an upper bound),
   or the max-mean wait (mini-app). It is taken out of category 4, where rank-mean timers absorb it.

MFC flux capture and batch copies (counted inside MFC's RHS timer) are moved to flux/rk using the np1-trace estimate from
`fair-metrics/mfc_hidden_amr.py`. The tile runs get the same constant, which is a lower bound because they have about 30x more blocks.

## Waterfall (MFC default, subcycled np8; lock-step in waterfall.csv)
The fixes are applied in order. Each saving is computed on what the previous fix left.

| step | saving s | how | kind |
|---|---|---|---|
| A. Tile clustering coverage | 0.235 | window coverage 23.4 -> 17.0 % at today's in-situ rate, **assuming tiles are merged to today's block size** | estimated (coverage measured) |
| B. Batch per-block work | 0.789 | comm above the mini-app (Morton) + swap/other + in-RHS fragmentation (mfc-overhead FINDINGS #1-3, 5, 6) | estimated |
| C. Regrid store/migration | 0.134 | regrid down to the mini-app's; the mfc-rootcause A/B is **pending** | estimated |
| D1. RHS case-opt | 0.080 | 6.585 -> 5.753 ns/cell | measured (standalone) |
| D2. Host syncs + copies | 0.145 | removes idle 0.843 + copies 0.659 ns/cell | estimated |
| D3. MP-WENO fusion | 0.115 | 80 % of the limiter's 1.488 ns/cell (its 476 B/cell round trip) | estimated |
| E. Owner placement | 0.010 | mini-app Morton - owner at np8 (61 ms at np64) | estimated |

Today 1.988 -> after fixes 0.480 -> floor 0.209 s/step. The remaining 0.27 s is the unfused per-cell gap (about 3 vs 1.64
ns/cell), ghost cells, extra volume and load imbalance.

## Caveats
- **Tile clustering is slower today (2.29 / 7.32 s/step), not faster.** It cuts the window coverage, but its 4-8k small blocks
  multiply ghost work (1.8x) and per-block host costs: fill + halo 3.2 s and host 2.8 s subcycled. The same build at default
  clustering is already 1.86 s/step lock-step (367 blocks). Saving A therefore requires B, plus merging tiles into large boxes.
- Coverage and block lists are taken at window end or as window means. MFC per-block costs come from one run per configuration.
- Growth to 512 (growth.csv, relative-tag MFC ladder): MFC AMR-only time grows by +0.51 s/step subcycled from np8, of which
  regrid is +0.32. The mini-app grows by +0.09. Owner placement matters mostly above np8.
- mfc-rootcause-BD (flux-capture host gap, migration A/B) had no findings when this was built, so it is marked pending.
