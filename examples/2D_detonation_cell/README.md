# 2D Cellular Detonation (H2/O2/Ar)

A stoichiometric H2/O2 mixture heavily diluted with argon (2H2 + O2 + 7Ar) is
ignited by an overdriven driver region at the left end of a periodic-in-y
channel. Ar dilution lengthens the induction zone relative to the cell size,
producing a regular (as opposed to irregular) cellular detonation structure
that a shock-capturing, reaction-coupled solver can resolve without diffusion.
An off-center hot pocket just ahead of the driver seeds the transverse waves
that trace out the cellular pattern as the detonation self-organizes and
propagates downstream. Chemistry is on (`chemistry = T`) with reactions only
(`chem_params%diffusion = F`) since the detonation structure here is
shock-driven, not diffusion-driven.

## CLI knobs

- `--scale`: grid multiplier (transverse resolution `Ny = 200 * scale`, `Nx = 4 * Ny`); increase to resolve smaller cells.
- `--tend`: physical end time in seconds (default `200e-6`); must be long enough for the detonation to traverse several cell widths.
- `--overdrive`: ratio of driver pressure to fresh-mixture pressure (default `30`); higher overdrive gives a stronger, more stable initiation transient.

## Running on 4x A100

```bash
./mfc.sh build -j 8 --gpu acc --case-optimization -t simulation
./mfc.sh run examples/2D_detonation_cell/case.py -n 4 --gpu acc --case-optimization -- --scale 1.0
```

### Production run (calibrated for ~10 min on 4x A100)

Calibration (`--scale 1.0 --tend 15e-6`, 800x200 grid, CFL factor 0.06):
measured throughput ~3.1e7 cell-updates/s on 4x A100. Steps scale as
`Nx*Ny*NT ~ scale^3` for fixed `--tend` (dt ~ 1/scale, NT ~ scale), so a 600s
budget at `--tend 200e-6` (~13.3x the calibration `--tend`) solves to
`scale^3 = 600 / (23 * 13.3) ~= 1.96` -> `scale ~= 1.25`, giving a
1000x250 grid (`dx = 0.12 mm`, 5x finer than the `--scale 0.25` CPU smoke
test's 0.6 mm) and ~74,500 steps, predicted at ~600s wall.

```bash
./mfc.sh run examples/2D_detonation_cell/case.py -n 4 --gpu acc --case-optimization -- --scale 1.25 --tend 200e-6
```

## Correctness checks

- **CJ speed**: compare the steady-state detonation front speed (from
  successive shock-front positions in the saved output) against the
  Chapman-Jouguet speed for this mixture computed independently in Cantera.
- **Soot-foil cell size**: the transverse spacing of triple-point trajectories
  (visible as streaks in a `max(dp/dt)` or reaction-rate soot-foil-style plot)
  should match the regular cell size expected for this heavily Ar-diluted
  mixture.
