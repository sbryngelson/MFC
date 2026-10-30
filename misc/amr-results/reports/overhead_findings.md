# MFC AMR overhead vs the mini-app (np8 subcycled blast, ms per coarse step)

MFC minus coverage-scaled mini-app; MFC non-RHS total 625 ms/step. Message count per rank per step is the same as the mini-app
(58.6 vs 58.2) and bytes are 1.28x: the gap is host-side work, kernel fragmentation and waiting, not traffic or GPU work.
MFC launches 693 kernels/step vs 81; ~8,800 small host-device copies/step vs 0.5; GPU idle 518 of 1381 ms/step at np1.

| # | ms | phase | root cause | file:line | smallest fix |
|---|---|---|---|---|---|
| 1 | 180 | fill | host: one consume + one ghost fill per block per stage (94 launches/step vs 6+13); level-0 time interpolation 11 ms; MPI wait ~17 ms per wave at np8 (lock-step 2.7) | m_amr_exchange.fpp:1175-1194, 1141-1165, 1061-1080; m_amr_stage.fpp:212-230 | cache only the coarse ring each block reads, interpolate in time inside the fill kernel, one fill kernel per level |
| 2 | 103 | other | grid-state swap 32/step at ~2.5 ms (75 ms); per-block step-start backup 31 launches/step | m_amr_advance.fpp:235,247; m_amr_frame.fpp:97-238; m_amr_store.fpp:52 | device table of block extents/coordinates built once per regrid; drop the swap |
| 3 | 100 | halo | level-0 halos of q_old and q_new 24 ms; per-stage halos absorb rank skew (mean 76, max rank 324) | m_amr_stage.fpp:48-50, 130-133 | exchange only the ring the fine patches read |
| 4 | 85 | regrid | migration 61 (41 regrowing the per-slot device store with full copies, host round trip above 4 GiB); grows with refined volume moved, not O(N^2) in blocks | m_amr_regrid.fpp:1520-1664; m_amr_store.fpp:240-357, 557 | reserve the store once or grow 2x; keep blocks on the old owner; ship overlaps only |
| 5 | 28 (lock 48) | reflux | one message per block x face x peer through host copies, fixed 786 KB | m_amr_transfer.fpp:77-145 | one device-packed message per peer |
| 6 | 16 | restrict | restrict wave | m_amr_transfer.fpp:832-888 | per-peer packing |

Hidden in "rhs": flux capture ~4.5 ms host gap per launch, ~200 ms/step at np1 subcycled (m_amr_registers.fpp:502, 516-599).

Subcycled vs lock-step at equal simulated time: 0.77x wall time (np1 and np8); the extra cost per coarse step is mostly more
refined work, not driver inefficiency. Communication growth np8 -> np64 is wait time (fill-gather MPI 101 -> 142 ms np8->np16);
owner placement drops the mini-app to 22 messages and 91 MB per rank per step; MFC needs the same placement fix.

Side finding: lock-step runs at np1 and np8 abort around step 420 with "PARTIAL transverse overlap" in s_amr_check_seam_topology
(default clustering, not investigated).

Chart: figs/subcycled_overhead_breakdown.png. Scripts: analysis/. Runs: runs/ (MFC), mini/ (mini-app). Worktree: mfc-src/.
