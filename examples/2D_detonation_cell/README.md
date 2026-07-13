# 2D Cellular Detonation (H2/O2/Ar)

A stoichiometric H2/O2 mixture heavily diluted with argon (2H2 + O2 + 7Ar) at
low pressure (6.67 kPa) detonates in a periodic-in-y channel. Ar dilution
lengthens the induction zone relative to the cell size, giving a regular
cellular structure that a shock-capturing, reaction-coupled solver can resolve
without diffusion. Chemistry is on (`chemistry = T`) with reactions only
(`chem_params%diffusion = F`) — the detonation structure here is shock-driven.

## Initiation (piston driver)

The detonation is launched by a **piston**: the left driver region holds hot,
thermodynamically-consistent burned products (constant-volume equilibrium of the
fresh mixture) moving forward at `--drivervel`. A velocity-driven shock (rather
than a stationary high-pressure jump) is what heats the fresh gas above the
~1100 K H2/O2 chain-branching crossover so it ignites promptly and the reaction
couples to the shock. Two failure modes this avoids:

- A *stationary* pressure driver strong enough to reach the crossover instead
  over-heats past the h2o2 thermo range (or over-compresses the contact) and NaNs.
- A driver too weak (post-shock T below ~1100 K) never ignites — the shock and
  flame decouple and the "detonation" decays to ~56% of the CJ speed.

Transverse cells are seeded with a **coherent** sinusoidal transverse-velocity
perturbation on the driver (`patch_icpp(2)%vel(2) = amp*sin(2*pi*kmode*y/Ly)`),
which imposes one clean cellular wavelength. A per-cell random perturbation
(`perturb_flow`) instead injects grid-scale white noise that advects into
y-striations rather than physical cells; a hot pocket ahead of the front
over-compresses on impact and NaNs.

## CLI knobs

- `--scale`: grid multiplier (`Ny = 200 * scale`, `Nx = 4 * Ny`); increase to resolve smaller cells.
- `--tend`: physical end time [s] (default `200e-6`); long enough to traverse the channel several times.
- `--drivervel`: driver piston velocity [m/s] (default `1200`); ~1200 gives a coupled detonation near CJ, higher is more overdriven.
- `--overdrive`: mild isentropic compression of the driver products (default `1.5`); kept low so the contact stays stable.
- `--seed` / `--no-seed`: toggle the `perturb_flow` cell seed (default on).

## Running on 4x A100

Build once (case-optimized OpenACC), then run:

```bash
./mfc.sh build -j 32 --gpu acc --case-optimization -t simulation --input examples/2D_detonation_cell/case.py
./mfc.sh run examples/2D_detonation_cell/case.py -n 4 --gpu acc --case-optimization -- --scale 1.25 --tend 200e-6
```

### Production run (~7 min on 4x A100)

`--scale 1.25 --tend 200e-6` gives a 1000x250 grid (`dx = 0.12 mm`), ~74,500
steps at CFL factor 0.06, ~400 s wall. Sizing: throughput ~3.1e7
cell-updates/s; steps scale as `Nx*Ny*NT ~ scale^3` at fixed `--tend`.

## Correctness checks (`analyze.py`)

- **CJ speed**: the leading-shock speed (tracked from the saved output) rises
  toward the Chapman-Jouguet speed (1617 m/s for this mixture, computed in
  Cantera) — ~86% window-averaged, ~93% locally by the domain exit — with a
  bounded, *shrinking* induction gap (coupled), the opposite of the decoupled
  failure mode.
- **Soot foil**: `max(pressure)` accumulated over time shows transverse
  triple-point streaks (cells) strengthening downstream; cell size grows from
  ~3 mm to ~5.6 mm as the pattern matures. A longer channel / higher `--scale`
  regularizes it further.
