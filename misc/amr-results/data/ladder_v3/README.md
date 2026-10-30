# MFC blast lattice, simple configuration, v3 variants (np8, np64)

Same case, deck, windows, reps and metrics as `../mfc-simple-v2/README.md` (placement on, `amr_place_owner = T`), on branch
`spike/amr-integrate-v3`, one build per commit so each change is measured on its own. Binaries and hashes: `VERSIONS`.

| suffix | build | change |
|---|---|---|
| `_ctl` | e341896a (`mfc-amr-integrate`) | control, rerun in the same session |
| `_sd` | fa3ca677 (`mfc-amr-v3-sd`) | + 572639aa lerp timer, fa3ca677 per-step coarse gather + per-stage blend |
| `_sw` | 7273d602 (`mfc-amr-v3`) | + swap packing |
| `_tiles` | 7273d602 | + `amr_tile_cluster = T` (runtime) |

Subcycled: all four. Lock-step: `_sw`, `_tiles`. `gen.py` writes the decks (imports `deck` from v2), `chain.sh` runs one rung/rep in one
allocation (rep 1 in variant order ctl, sd, sw, tiles; rep 2 reversed), `../viz-venv/bin/python -I table.py` writes `results.csv` and
`scaling.png` beside v2, AMReX CNS and PeleC. Job IDs: `logs/sbatch_record.txt`. MFC.sh files also run post_process, which aborts after
the timed simulation; it does not affect the step-loop wall.

## Validation
- `./mfc.sh test --only AMR` (non-case-optimized builds, `logs/tests-*.out`): 838 total, exit 0, at fa3ca677 and 7273d602.
- np8 base-level restart files at steps 40 and 240 are bitwise identical: `_sd` and `_sw` vs `_ctl` (subcycled), `_sw` vs v2 5d8ab069
  (lock-step).
- Tiles np1 vs np4 (`val_tiles_np{1,4}`, 200^3, 40 subcycled steps): same boxes and coverage; base fields differ at round-off
  (max abs 5.5e-14, max rel 6.8e-13).

## Tiles
- Subcycled with tiles runs out of device memory at the step-140 regrid at np8 and np64; no 240-step wall. Cause: every flat-store
  column is sized for an amr_max_grid_size (64) block, about 0.225 GiB, while a tile is 16^3; tiles put about 2x more blocks on a
  rank (479 vs 216 level-1 boxes at np8), the store doubles 32 -> 64 columns (15.8 GiB free after it vs 43.9 GiB without tiles) and
  the next doubling does not fit.
- Workaround probe (`_tilesg32`, 1 rep): `amr_max_grid_size = 32` with tiles completes 240 steps at np8 and np64 (s/step 0.998 / 1.042,
  coverage 36.4 %). `amr_max_grid_size = 16` (`np8_sub_*_tilesg16`) hits amr_max_blocks = 4096 and coverage freezes at 26.2 %: invalid.
- Lock-step: 2.0x (np8) and 2.7x (np64) the s/step of `_sw`; rhs phase 50.5 s vs 14.9 s and swap 6.3 s vs 0.8 s at np8 (16^3 tiles
  carry far more ghost cells per interior cell), and rhs/coarse imbalance rises to 1.21/1.27 at np64.
- Coverage at the step-220 regrid is 19.46 % with tiles vs 19.51 % without (lock-step), not the AMReX-like figure expected.
