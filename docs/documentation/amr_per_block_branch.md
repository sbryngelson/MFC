# The `amr-per-block` branch: frozen reference

This branch is `up/mega` at `88793197` plus two reverts (`34afabf5`, `74c639ca`) that restore the AMR code paths the batched
lock-step advance retired: the per-block fine advance (`amr_batched_advance = F`), subcycling (`amr_subcycle`), the QBMM
pb/mv side state and its exchanges, and the cyl_coord, stretched-grid, Euler-bubble, phase-change and moving-particle-cloud
paths, with their 22 goldens. All of it is correct and golden-tested **at this commit** (the per-block advance measured about
1.2x slower per step than the batched advance on the ledger-166 deck, which is why it was retired).

It is a frozen reference, not a mergeable branch. After `88793197`, `up/mega` rewrote the modules this code lives in
(one exchange engine `m_amr_wave` for every wave, one gather mechanism, the stage hooks in `m_amr_stage`, the instrumentation
and the per-box gather/chunk/gsnd machinery deleted, the largest routines split): a merge conflicts in 16 files and every
conflict is a re-port, not a resolution. Re-enabling any of these paths means porting them onto the current `up/mega`, in
three largely independent stages, each gated by this branch's goldens:

1. The per-block advance loop (`m_amr_advance`, `m_amr_frame`, `m_amr_store`, `m_amr_registers`; ~1.2k lines). Mostly
   engine-independent; brings back cyl_coord and stretched grids.
2. The pb/mv side state (~1k lines): its two exchanges become two more waves on `m_amr_wave` (band 4 is now the stash
   migration; pick a free band). Brings back QBMM.
3. Subcycling (~0.8k lines): the time-lerp ghost stores and the per-level exchange sites, on the engine.

Known: the ghost fill on this branch still holds two target regions in one routine and trips the nvfortran 24.9-26.3
OpenMP-offload `fort2` crash that `up/mega` fixed in `245b3868` (split the closure kernel into its own routine); any port
starts from `up/mega`, which already carries the fix.
