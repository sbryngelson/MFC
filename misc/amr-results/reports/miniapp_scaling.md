# Scaling the mini-app, and the blast-lattice ladder (R10) against AMReX CNS and PeleC

Date: 2026-10-29. Frontier, partition `batch` for the ladder (g1 for development and before/after runs), 8 GCDs per node,
`-c 7 --gpus-per-task=1 --gpu-bind=closest`, GPU-aware Cray MPICH 9.1.0, `FI_MR_CACHE_MONITOR=kdreg2 MPICH_OFI_NIC_POLICY=GPU`
(as the AMReX/PeleC arms). Binary for the ladder: `bin/amr.r10` = commit 95cfb2d (md5 7a7d2f90b514).

## 1. What changed for scaling

All four known problems are fixed. With the scaling changes (03f1cc1, 95cfb2d) R0-R8 pass and **every R0-R8 output (164
dumps, body histories, regrid and change logs) is bitwise identical to the pre-change binary** (log.txt totals differ in the
last bits: they are now a tree reduction). The later conservation fix (item 6, 2e66a49) intentionally changes 33 of 120
dumps; R0-R8 pass after it and the cross-rank / per-block checks (R0 np8, R4, R5, R6) stay bitwise.

1. **Spatial index** (`src/index.c`): per level a grid of 64^3-cell buckets with CSR lists of the blocks overlapping each
   bucket. `block_at` is O(blocks in one bucket); `box_hits` returns the (block, periodic shift) pairs meeting a box, sorted;
   `box_covered` tests nesting by summed intersection volume instead of per cell.
2. **Plans from local work.** Each rank finds the (dst block, src block, shift) *pieces* that touch its own blocks (its
   blocks' ghost/cache/restriction boxes, plus the boxes its blocks feed), sorts them canonically, and pushes one **box
   segment** per piece (strided src/dst, expanded per value on the device) straight into per-peer lists. No global
   enumeration, no per-value host items; host work and device index memory scale with the number of copies. Outflow
   clamping is zero-stride sub-boxes; flux-register entries merge into runs; the reflux grouping keeps the global
   (fine block, direction, side, cell) order, so the deterministic reflux is unchanged. Fine-ghost fill lists come from a
   per-block coverage mask; new blocks are prolonged by a kernel over the flattened interior. Every plan build checks with
   one `MPI_Alltoall` that both sides of each rank pair agree on the message size.
3. **Regrid** (`src/regrid.c`): each rank tiles its own tags (dense mask per owned block), sends tile ids to the owner of
   their max_block chunk (`MPI_Alltoallv`); chunk owners clear non-nesting tiles and emit blocks; only the resulting boxes
   are gathered (`MPI_Allgatherv`), sorted by Morton key (qsort; keys are distinct) and dealt to ranks as before. The
   old-to-new data transfer uses the same piece/segment plans. Result is identical to the old global tile mask.
4. **Diagnostics**: leaf totals and rho/p extrema are a device reduction (per-thread-block partials, host sum in block
   order, then levels in order) over a per-cell "covered by level l+1" mask built from the restriction pieces;
   `log_every` controls the interval. The level-0 step change (`log_change`) is a device max. `solver_maxspeed`,
   `ib_force` and the tag buffer reuse grow-only device buffers.
5. **Found at scale, fixed**: GPU-aware MPI hang (findings F3). Plan staging buffers were freed and reallocated at every
   regrid; on the np8 blast case an intra-node exchange hung for good after the step-120 regrid (all ranks in
   `MPI_Waitall`, one inside Cray GTL `gtlt_hsa_query_request`, message sizes consistent). One shared, never-freed staging
   pool removed it.
6. **Found at scale, fixed**: non-conservative prolongation (findings F4). The quartic's fallback to MC slopes was decided
   per half-cell, so new fine cells at a regrid could miss the parent's average near fronts (leaf mass drift 4.5e-11 on
   H-sub np8, energy 1.5e-13). Both halves now fall back together: mass drift stays at the 3e-13 summation floor, same cost.

Blast-case additions: `ic = blast` (+ `ic_point = 1` for centre sampling as AMReX/PeleC), `tag = dengrad` (AMReX CNS
criterion: max one-sided |rho jump| >= tag_eps), `buf_coarse` (buffer in level-(l-1) cells before tiling, as
`n_error_buf`), `grid_eff` (bisect a chunk's bounding box until that fraction of its tiles is tagged, as AMReX's grid
efficiency), `steps`, `dump`, `regrid_log`, `regrid_detail`, setup/regrid wall times in out.txt, `grids.txt` (blocks, cells,
% of domain per level at each regrid), `minmax.txt` (leaf rho/p extrema).

### Before -> after (same case, same grids, H-sub blast to step 60 = 3 regrids; L1/L2 1028/7404 blocks at np64)

| rung | quantity | before (33c8c09 + blast case) | after (95cfb2d) |
|---|---|---|---|
| np64 (8 nodes) | regrid, 3 regrids | **339.2 s** | **0.24 s** |
| np64 | 60-step wall | 372.8 s | 2.71 s |
| np8 | regrid, 3 regrids | 8.86 s | 0.20 s |
| np8 | 60-step wall, totals logged every step / every 1000 | 25.3 s / 14.9 s | 2.45 s / 2.40 s |
| np8 | H-sub 160 steps: regrid total (8 regrids, up to 6201 L2 blocks) | (not run) | 23.6 s with per-value plans -> **2.4 s** with segments |

Setup (to the first step) is 0.2-1.5 s at every rung before and after (no refinement exists at t = 0; the first run in a job
pays HIP/MPI initialisation). The old binary was not run at np512: its halo build is O(blocks^2) and its coarse-fine build
calls a linear block search per cell of every fine block in the domain, i.e. well beyond the 2 h job cap.

## 2. The blast-lattice case (R10), matched to the AMReX/PeleC decks

3D periodic, rho = 1, u = 0, p = 1 + 9 exp(-(cos^2 pi x + cos^2 pi y + cos^2 pi z)/0.15), gamma 1.4, inviscid; 200 cells per
unit length; H: 2 refined levels, np8 400x400x200 on [0,2]x[0,2]x[0,1], doubling x, y, z to np512 1600x1600x800; L: 1 level,
np8 400^3 on [0,2]^3. Level 0 is one 200x200x100 (H) / 200^3 (L) brick per GCD. Fixed dt: finest 1e-4 (H) / 2e-4 (L);
subcycled coarse dt 4e-4; lock-step = every level at the finest dt. Regrid every 20 coarse steps; tag `dengrad` 0.008 (as
`cns.refine_dengrad` / `tagging.dengrad`); `buf_coarse = 4` (`n_error_buf 4`); blocking factor 16 (no z = 8 needed: the
mini-app's tiles are 16 level-l cells, and 200-cell level-0 extents give 400-cell level-1 extents); max block 64;
`grid_eff = 0.9`. Lock-step uses `coupling = step` (Berger-Colella reflux once per step, like CNS non-subcycled).
Inputs: `cases/blast.in` + `jobs/blast.sh` (geometry per NP), rung driver `jobs/rung.sh`, table `tools/blast_table.py`.

Scheme differences that stay: the mini-app is WENO5 (Jiang-Shu) + HLLC + SSP-RK3 with quartic conservative prolongation;
CNS is MUSCL + RK3; PeleC is unsplit CTU Godunov PLM (one hydro evaluation per level step instead of three).

**Coverage matches without tuning the threshold.** With the same tag and threshold, the mini-app's clustering (bounding box
per 64^3 chunk, bisected to grid_eff 0.9) gives the refined volume of AMReX's Berger-Rigoutsos within ~0.6 % of the domain at
every regrid (np8 H-sub, L1/L2 % of domain, mini-app vs AMReX: step 40 3.17/- vs 3.12/-, 60 5.34/2.54 vs 5.57/2.51, 100 14.14/5.36 vs
13.61/5.30, 120 16.93/10.02 vs 17.05/10.14, 140 20.34/12.18 vs 20.15/12.54, 220 30.01/19.83 vs 29.43/20.01). Without
grid_eff the mini-app refined ~15 % more.

**Physics matches.** Leaf extrema (mini-app) vs AMReX CNS / PeleC at the same time: t = 0.024 rho 0.5379/1.0660, p max
4.198 (CNS 0.5383/1.0662, 4.197); t = 0.064 rho 0.2089/1.5149, p max 2.298 (PeleC 0.2077/1.5148, 2.299; CNS 0.2091/1.5153);
t = 0.096 rho 0.1560/1.6840, p 0.7416/2.1398 (CNS 0.1561/1.6796, 0.7417/2.1367; PeleC L 0.1559/1.6816, 0.7406/2.1377).

## 3. Ladder results: seconds per coarse step

s/step = (wall of the N2-step run - wall of the N1-step run)/(N2 - N1); mini-app wall = its step loop (setup excluded, as
AMReX "Run Time advance" / PeleC "Run time w/o init"); 2 reps, rep sd <= 0.003 s unless noted. Coverage = % of domain at
the last regrid of the N2-step run of AMReX/PeleC (after step N2 - 20). AMReX/PeleC numbers from their RESULTS files and run
directories (`amrex-bench/runs_blast`, `pelec-bench/runs`; H-sub 40->160 for AMReX from `logs/bs*.out`).

| config | window | np | mini-app s/step (n) | AMReX CNS | PeleC | mini-app vs CNS | coverage L1 / L2 % (mini-app) | AMReX coverage |
|---|---|---|---|---|---|---|---|---|
| H-sub | 40->160 | 8 | **0.627** (2) | 1.447 | 1.778 | 2.31x faster | 20.34 / 12.18 | 20.15 / 12.54 (PeleC 20.15 / 12.47) |
| H-sub | 40->160 | 64 | **0.681** (2) | 1.908 | 2.444 | 2.80x faster | 20.30 / 12.18 | 20.15 / 12.54 (PeleC 20.15 / 12.47) |
| H-sub | 40->160 | 512 | **0.777** (2) | 2.256 | 2.911 | 2.90x faster | 20.30 / 12.18 | 20.15 / 12.54 (PeleC 20.15 / 12.47) |
| H-sub | 40->240 | 8 | **1.041** (2) | 2.553 | - | 2.45x faster | 30.01 / 19.83 | 29.43 / 20.01 |
| H-sub | 40->240 | 64 | **1.129** (2) | 3.118 | - | 2.76x faster | 30.02 / 19.83 | 29.43 / 20.01 |
| H-sub | 40->240 | 512 | **1.243** (2) | 3.606 | - | 2.90x faster | 30.02 / 19.83 | 29.43 / 20.01 |
| L-sub | 40->240 | 8 | **0.261** (2) | 0.467 | 0.600 | 1.79x faster | 30.02 | 30.25 |
| L-sub | 40->240 | 64 | **0.317** (2) | 0.600 | 0.802 | 1.89x faster | 30.02 | 30.25 |
| L-sub | 40->240 | 512 | **0.352** (2) | 0.678 | 0.952 | 1.93x faster | 30.02 | 30.25 |
| H-nosub | 40->240 | 8 | **0.046** (2) | 0.095 | - | 2.07x faster | 5.12 / 1.92 | 4.87 / 1.93 |
| H-nosub | 40->240 | 64 | **0.054** (2) | 0.127 | - | 2.35x faster | 5.11 / 1.92 | 4.87 / 1.93 |
| H-nosub | 40->240 | 512 | **0.060** (2) | 0.142 | - | 2.35x faster | 5.11 / 1.92 | 4.87 / 1.93 |
| L-nosub | 40->240 | 8 | **0.095** (2) | 0.169 | - | 1.78x faster | 15.52 | 15.45 |
| L-nosub | 40->240 | 64 | **0.125** (2) | 0.220 | - | 1.76x faster | 15.52 | 15.45 |
| L-nosub | 40->240 | 512 | **0.139** (2) | 0.250 | - | 1.80x faster | 15.52 | 15.45 |

Jobs: mini-app np8 5792704, np64 5792705, np512 5792706 (`runs/blast/np<NP>-<config>-j<job>/r<rep>_s<steps>/`, Slurm logs
`runs/blast/r10-np<NP>-<job>.out`); a g1 repeat of np8 (5792707) and np64 (5792829) agrees within 1.5 %. Leaf extrema of the
mini-app are identical at np8, np64 and np512 (e.g. H-sub t = 0.096: rho 0.15595/1.68399, p 0.74159/2.13975).

## 4. Phase breakdown (mini-app, per coarse step, max over ranks, differenced over the window)

| config | np | rhs | rk | halo | fill | reflux | restrict | regrid | s/step |
|---|---|---|---|---|---|---|---|---|---|
| H-sub 40->160 | 8 | 0.383 | 0.067 | 0.070 | 0.113 | 0.012 | 0.008 | 0.020 | 0.627 |
| H-sub 40->160 | 64 | 0.384 | 0.068 | 0.114 | 0.133 | 0.023 | 0.014 | 0.023 | 0.681 |
| H-sub 40->160 | 512 | 0.387 | 0.069 | 0.185 | 0.165 | 0.063 | 0.060 | 0.037 | 0.777 |
| H-sub 40->240 | 8 | 0.664 | 0.114 | 0.093 | 0.161 | 0.015 | 0.014 | 0.029 | 1.041 |
| H-sub 40->240 | 64 | 0.667 | 0.116 | 0.168 | 0.201 | 0.029 | 0.022 | 0.035 | 1.129 |
| H-sub 40->240 | 512 | 0.677 | 0.117 | 0.247 | 0.236 | 0.074 | 0.063 | 0.055 | 1.243 |
| L-sub 40->240 | 8 | 0.165 | 0.026 | 0.025 | 0.041 | 0.004 | 0.003 | 0.010 | 0.261 |
| L-sub 40->240 | 64 | 0.166 | 0.027 | 0.073 | 0.079 | 0.006 | 0.010 | 0.012 | 0.317 |
| L-sub 40->240 | 512 | 0.166 | 0.028 | 0.088 | 0.090 | 0.023 | 0.021 | 0.020 | 0.352 |
| H-nosub 40->240 | 8 | 0.029 | 0.004 | 0.006 | 0.005 | 0.001 | 0.000 | 0.003 | 0.046 |
| H-nosub 40->240 | 64 | 0.029 | 0.004 | 0.012 | 0.009 | 0.002 | 0.001 | 0.003 | 0.054 |
| H-nosub 40->240 | 512 | 0.030 | 0.004 | 0.014 | 0.012 | 0.003 | 0.002 | 0.005 | 0.060 |
| L-nosub 40->240 | 8 | 0.065 | 0.009 | 0.008 | 0.008 | 0.001 | 0.001 | 0.004 | 0.095 |
| L-nosub 40->240 | 64 | 0.066 | 0.010 | 0.032 | 0.031 | 0.006 | 0.004 | 0.006 | 0.125 |
| L-nosub 40->240 | 512 | 0.066 | 0.010 | 0.038 | 0.036 | 0.007 | 0.006 | 0.009 | 0.139 |

`rhs` = WENO5/HLLC face fluxes and divergence; `rk` = RK stage update and the step-start copy; `halo` = same-level ghost
exchange (+ outflow, unused here); `fill` = coarse-cache gather (MPI) and fine ghost prolongation every stage; `reflux` =
flux-register capture and the fine-to-coarse register exchange; `restrict` = averaging and its exchange; `regrid` = tagging,
clustering, plan builds and data transfer, amortized over the window.

For reference (TinyProfiler, exclusive, as reported in the AMReX/PeleC RESULTS files): AMReX CNS H-sub np8 240 steps
(513 s): compute_dSdt 183 s, CellConservativeLinear::interp 183 s (per-box coarse/fine ghost interpolation every RK stage,
8.6M calls), ParallelCopy_finish 52 s, RungeKutta3 27 s, regrid 0.2 s; PeleC H np8 160 steps: umeth 105 s,
construct_hydro_source 33 s, ParallelCopy_finish 30 s, umdrv 17 s.

## 5. Read

**Where the mini-app stands.** On this case, at matched dt, refined volume (within ~0.6 % of the domain per level of AMReX's)
and solution (extrema agree to 3 digits), the mini-app's coarse step costs **0.34-0.57 of AMReX CNS's** on all four
configurations and **0.27-0.44 of PeleC's** (H 40->160 and L), at np8, np64 and np512 (table above): 1.8-2.9x faster than
CNS and 2.3-3.8x faster than PeleC, while doing more work per cell (WENO5 + HLLC on primitive variables and quartic
prolongation, against MUSCL in CNS and one CTU Godunov PLM evaluation per level step in PeleC).

**Why.** (1) Batching: every kernel runs once per level and RK stage over all of a rank's blocks through a flattened cell
index; CNS and PeleC launch per box (64^3 boxes, ~10^4 L2 boxes per 8 GCDs at the end of H-sub), and CNS's per-box
coarse/fine interpolation every RK stage alone is about as expensive as its hydro (183 s vs 183 s in the H-sub np8 profile).
The mini-app's whole coarse-fine fill (cache gather + ghost prolongation every stage) is 11-26 % of its step. (2)
Communication: one message per peer per plan, packed and unpacked by one kernel from precomputed device segments, against
ParallelCopy/FillBoundary per box pair (ParallelCopy_finish alone 10-14 % in CNS and PeleC). The flux/RHS kernels (`rhs`)
are 47-68 % of the mini-app's step and are themselves not tuned (each face converts its six stencil cells to primitives; no shared-memory reuse).

**Weak scaling** (s/step at np8 over s/step at np64 / np512): mini-app H-sub 0.92 / 0.81 (40->160), 0.92 / 0.84 (40->240),
L-sub 0.82 / 0.74, H-nosub 0.85 / 0.77, L-nosub 0.76 / 0.68; AMReX CNS 0.75-0.82 / 0.64-0.71; PeleC 0.73-0.75 / 0.61-0.63.
The mini-app's loss is all communication and regrid: from np8 to np512 `rhs` is flat to 3 % and `rk` within 0.003 s, while
`halo` and `fill` grow 1.5-5x and `reflux`/`restrict` (their register and restriction exchanges cross ranks) 3-8x, because
fine blocks are dealt to ranks along a Morton curve by cell count, independently of the level-0 brick a rank owns, so most coarse data for fine ghost caches and most fine-fine halos cross ranks. A level-aware
distribution (fine blocks to the owner of the coarse cells under them where balance allows) is the obvious next step.

**What is still worse than AMReX.** Regrid: 0.003-0.055 s per coarse step amortized (3-8 % of the step; 1-1.8 s per regrid
at the end of H-sub) against AMReX's 0.2 s over a whole 240-step run (np8), because every level's plans and coarse-fine lists are
rebuilt from scratch at every regrid (host enumeration of pieces, fill masks per ghost cell, upload) and the old/new data
transfer always runs; the clustering step (tile Alltoallv, block Allgatherv, a global sort and an O(blocks) pass on every
rank) grows with the global block count: the last H-sub regrid takes 1.0 s at np8 (cluster 0.17 s, 11.8k blocks) and 1.8 s at
np512 (cluster 0.8 s, 7.5e5 blocks); halo, coarse-fine and transfer builds stay flat (O(local)). Load balance is by cell
count only.

**Caveats.** (a) Different schemes, so this is cost per coarse step at equal workload, not cost at equal accuracy (the
mini-app's higher order would favour it further). (b) The mini-app is a reduced code: ideal-gas Euler only, no EB, no
species, fixed dt (as the decks), no plot/checkpoint; AMReX/PeleC carry general machinery the mini-app does not. (c) The
blast case's refined volume grows through the window (0 % at step 40 to 20-30 % at step 240), so s/step is an average over
a ramp; the three codes see the same ramp. (d) Memory: H-sub ran to step 240 at every rung (about 51M level-2 cells per GCD
at the end) without error; PeleC runs out of its 64 GB at the step-180 regrid of H. The mini-app's peak memory was not
measured. (e) The ladder binary (95cfb2d) predates the conservative prolongation fix (2e66a49, findings F4); the fix does
not change cost (np8 H-sub 160 steps 76.30 s vs 76.23 s), grids or extrema (to 10 digits), only mass conservation (drift
4.5e-11 -> 3e-13).

## 6. np128 rung (added 2026-10-29)

16 nodes x 8 GCDs, job 5795105 (partition extended, account eng160), same binary as the ladder (`bin/amr.r10`, md5 7a7d2f90b514), `jobs/rung.sh 128 2 ...`
(geometry 1600x800x800 L / 1600x800x400 H), outputs `runs/blast/np128-<config>-j5795105/`; combined four-code job script
`/lustre/orion/cfd154/scratch/sbryngelson/amr-report/np128/rung128.sh`. s/step np8 / np64 / **np128** / np512, 2 reps:
L-sub 0.261 / 0.317 / **0.330** / 0.352; L-nosub 0.095 / 0.125 / **0.130** / 0.139; H-sub 40->160 0.627 / 0.681 / **0.709** / 0.777;
H-sub 40->240 1.041 / 1.129 / **1.164** / 1.243; H-nosub 0.046 / 0.054 / **0.056** / 0.060. Coverage identical to the other rungs.
`place = owner` at np128 (`bin/amr.place`, md5 47dc8826ebba, commit ccfd46b; dirs `np128-L-{sub,nosub}-owner-j5795105`):
L-sub **0.257**, L-nosub **0.099** s/step (vs 0.330 / 0.130 Morton), i.e. flat against np8 Morton (0.261 / 0.095).
