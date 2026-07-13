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
./mfc.sh build --gpu -j 8
./mfc.sh run examples/2D_detonation_cell/case.py --gpu -g 0 1 2 3 -n 4
```

## Correctness checks

- **CJ speed**: compare the steady-state detonation front speed (from
  successive shock-front positions in the saved output) against the
  Chapman-Jouguet speed for this mixture computed independently in Cantera.
- **Soot-foil cell size**: the transverse spacing of triple-point trajectories
  (visible as streaks in a `max(dp/dt)` or reaction-rate soot-foil-style plot)
  should match the regular cell size expected for this heavily Ar-diluted
  mixture.
