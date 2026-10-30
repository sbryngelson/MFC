# Flux-capture host gap and regrid migration A/B (subcycled L blast)

Caveat: baseline MFC is NOT run-to-run reproducible at np8/np16 (two identical base runs differ in 41% of the final coarse-state
words, max rel diff 1.2e-2); np1 is bitwise reproducible. Bitwise checks at np>1 are therefore not possible until that is fixed.

## B. Capture gap (np1): synchronous copies, not host compute or MPI
GPU-idle gap before a kernel = 0.08 ms + 75 us x (synchronous copies), R^2 = 0.998; each copy ~3 us on device, the rest host wait
in the OpenMP runtime's per-copy synchronize. HSA_ENABLE_SDMA=0: no change.

| one capture launch | copies | ms |
|---|---|---|
| GPU_UPDATE of 11 allocatable arrays (m_amr_registers.fpp:502), 5 copies each | 55 | 4.26 |
| of which descriptor re-copies (48 B) | 44 | ~3.3 |
| of which payload (full amr_max_blocks length, not 1:nb) | 11 | ~0.8 |
| implicit maps / MPI | 0 | 0 |
| measured gap | 55 | 4.34 |

Top copy sites, copies/step: update before capture 1,504; HLLC kernel (m_riemann_solver_hllc.fpp:919, 22 private arrays mapped
tofrom from the stack) 1,204; grid-state swap (9 grid arrays per batch) 635; fill_fine_ghosts and gather tables (amr_slab_tab,
amr_cg) 778; WENO private arrays 355; total 5,505.

Fixes: (1) pack the 11 capture tables into one, or build them on the device once per regrid (55 -> 0-5 copies per launch,
~115-200 ms/step); (2) declare HLLC/WENO private arrays inside the loop body; (3) a device table of block coordinates instead of
the per-batch grid swap; (4) one packed update per site in general. Implicit mapping of allocatables: refuted (present:allocatable
already used).

## D. Regrid migration A/B (steps 40->240, 10 regrids)
(i) reserve the per-slot device store once (MFC_AMR_STCAP); (ii) keep blocks on their previous owner (MFC_AMR_STICKY).

| run | s/step | regrid ms | GB migrated/regrid | fine max/mean |
|---|---|---|---|---|
| np8 base / base2 | 1.817 / 1.814 | 2158 / 2156 | 1.84 | 1.006 |
| np8 (i) | 1.747 | 811 | 1.84 | 1.006 |
| np8 (ii) | 1.814 | 2161 | 1.97 | 1.028 |
| np8 both | 1.740 | 831 | 1.97 | 1.028 |
| np16 base / base2 | 2.005 / 1.991 | 5069 / 5078 | 4.68 | 1.009 |
| np16 (i) | 1.800 | 905 | 4.68 | 1.009 |
| np16 (ii) | 1.974 | 4256 | 4.65 | 1.035 |
| np16 both | 1.808 | 947 | 4.65 | 1.035 |

(i) captures the gain: regrid 2.7x faster (np8), 5.6x (np16); costs 72-88 slots x 230 MB device memory reserved up front.
(ii) pure stickiness put every block on rank 0 (OOM at np16); the balance-bounded version (+-10% cut motion) does not reduce
migration because MFC ships WHOLE old blocks to every rank owning an overlapping new box. Next lever: ship only overlaps.

Files: analysis/info_b.txt, analysis/perf_b.txt, analysis/*.py, runs/b_*, runs/d{8,16}_*. Switches: worktree
/lustre/orion/cfd154/scratch/sbryngelson/mfc-rc-BD (git diff) in m_amr_store.fpp, m_amr_distribution.fpp, m_amr_regrid.fpp.
