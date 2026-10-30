# MFC blast lattice, simple configuration, v2 build (ladder L)

Same case, deck, rungs, windows, reps and metrics as `../mfc-simple/README.md` (v1, commit 651b6d88), rerun on a newer build with
owner-aware placement, and set beside v1, AMReX CNS and PeleC.

## What changed from v1
- Build: worktree `mfc-amr-bench2`, detached at **5d8ab069** (spike/amr-integrate-next): review fixes, quartic seeding, reflux race fix,
  owner-aware placement, per-block batching, six RHS speedups, regrid store/migration fixes. Case-optimized, built once on the login
  node; see `VERSIONS` (commit, sha256 of the three binaries; each job log prints them again at start).
- Deck: identical to v1 plus `amr_place_owner = T` (`gen.py`). A/B: np64 subcycled with `amr_place_owner = F` on the same binaries
  (`np64_sub_s{40,240}_r{1,2}_off`).
- Builds 3a0e61a1, a4eaff68 and 23b4d7aa were built and superseded during the work at the coordinator's request. No kept run used them;
  their binaries and partial runs are in `trash/<commit>/` (`trash/23b4d7aa/logs/table_23b4d7aa_partial.txt` has its np8/np64 table).

## Runs
- One directory per launch, `np<NP>_<lock|sub>_s<40|240>_r<1|2>[_off]/`. `chain.sh` runs a rung/rep's launches in one allocation;
  a `?dir` entry runs only if the previous launch failed (used for the np512 239-step subcycled fallback, which v1 needed).
- np8/np64 and the A/B ran on g1 (`-A cfd154`); np128 and np512 on batch (`-A eng160`). Each np128 rep was submitted to both queues and
  the copy that did not start first was cancelled before it ran (`sacct` shows it never started). Job IDs: `logs/sbatch_record.txt`.

## np1024 rung (128 nodes)
- Same scaling rule: domain [0,16]x[0,8]^2, 3200x1600x1600 base cells (`gen.py` RUNGS; MFC `amr_max_blocks` = 512*np as at every rung).
- AMReX CNS (nosub, sub) and PeleC use the same executables and decks as the 8-512 rows; `ref1024/` holds the np512 decks with only
  prob_hi / n_cell changed. `ref_run.sh` repeats `amrex-bench/scripts/run_blast.sh` (same env, srun line, log layout) for one rep;
  logs land in `amrex-bench/runs_blast/np1024-L-{nosub,sub}-j<job>/` and `pelec-bench/runs/np1024-L-j<job>/` so `table.py` finds them.
- One 128-node batch job (`np1024.sh`, job 5800369) runs, per rep: AMReX nosub, AMReX sub, PeleC, MFC lock-step, MFC subcycled
  (239-step MFC fallback only if the 240-step subcycled run fails); 2 reps.

## Tables and plot
`../viz-venv/bin/python -I table.py` prints v1, v2, the A/B row, AMReX and PeleC, and writes `results.csv` and `scaling.png`
(s/step and AMR-only s/step vs GPUs; dashed/open = lock-step, solid/filled = subcycled). Metric definitions are those of v1.

## Caveats
- The v1 caveats still apply: MFC refines more of the domain than AMReX/PeleC at the same tag rule (19.5 % / 40.5 % vs 15.5 % / 30.3 %
  at the step-220 regrid; identical in v1 and v2), and MFC's timed loop includes the step-240 regrid.
- v1 np128/np512 subcycled used 239-step launches (step-240 regrid out of memory). In v2, np128 and np512 complete all 240 steps
  (the np512 239-step fallback entries were skipped).
- AMR-only splits differ between codes (MFC: wall minus `coarse` + `rhs` phases; AMReX/PeleC: wall minus RHS kernel regions).
