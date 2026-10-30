# Findings for MFC from the mini-app reproducers

## F1. Coarse-fine prolongation is minmod-limited linear (R1)

MFC's `s_prolong_one_var` (m_amr_transfer.fpp) fills fine ghosts and new fine cells with a minmod-limited linear
reconstruction. It is second order away from extrema and first order at smooth extrema, so the fifth-order interior is
pulled down at every coarse-fine interface a wave crosses.

R1 (2D periodic density wave crossing one static fine block, t = 1, dt ~ dx^(5/3), leaf-cell L1 error over the uniform
coarse error at the same dt):

| base mesh | minmod (MFC) | MC linear | conservative quartic |
|---|---|---|---|
| 32^2  | 61.6 x  | 16.8 x | 0.76 x |
| 64^2  | 283 x   | 65.1 x | 0.76 x |
| 128^2 | 1243 x  | 246 x  | 0.76 x |

The quartic is the conservative half-cell average of the quartic through five coarse averages, applied one direction at a
time, falling back to MC slopes only when it leaves the neighbour range by more than the smallest local second difference.
With it the refined run is more accurate than the coarse grid alone, as it should be.

Reproduce: `jobs/ladder.sh R1` (informational lines report minmod and MC).

## F2. Initial conditions must be cell averages

The solver evolves cell averages, and prolongation/restriction are exact for cell averages. Initialising point values at
cell centres makes the coarse and fine "averages" carry different hidden factors (for a sine, sinc(pi k dx) per
direction), which shows up at every interface as a resolution-dependent mismatch: R1 converged at third order regardless
of the prolongation scheme, dt, reflux or WENO weights until the initial condition was integrated (4-point Gauss per
direction). MFC initialises point values; any MFC vs mini-app comparison, and any accuracy study across an interface,
must account for this (initialise averages, or compare against exact averages).

## F3. GPU-aware MPI buffers must not be freed and reallocated (Cray MPICH 9.1.0)

The mini-app hipFree'd and hipMalloc'd each plan's device send/receive buffers at every regrid. On the np8 blast lattice
(R10) an intra-node exchange then hung for good after the step-120 regrid: every rank in MPI_Waitall, one inside Cray GTL's
HSA IPC request (`gtlt_hsa_query_request`), with consistent message sizes on both sides (checked). MPICH caches IPC mappings
of device buffers by address; a new buffer at a reused address is the likely trigger. One shared staging pool that is never freed
(growth keeps the old buffer) removed the hang. AMReX avoids it by keeping communication buffers in its own arena. Any
device buffer MFC hands to MPI and frees at a regrid should come from a persistent pool instead.

## F4. A limiter fallback in prolongation must be decided per coarse cell, not per child

The conservative quartic prolongation fell back to MC slopes when the *requested* half-cell value left its bound, so near a
front the two children of a coarse cell could use different reconstructions and their sum was no longer twice the parent.
Ghost fills do not care, but new fine cells at a regrid do: on the blast lattice (R10, H sub, np8) leaf mass drifted by
4.5e-11 (relative) while the refined region grew (steps 40-120), with energy at 1.5e-13; R6's smooth blob never triggered it.
Testing both halves and falling back together keeps every 1D pass, and hence the tensor product, conservative: the drift
stays at the 3e-13 summation floor, identical to MC-only prolongation, at the same cost. Any limited prolongation in MFC
(the minmod-limited linear one of F1 is symmetric by construction) must keep the fallback decision symmetric in the same way.

## F5. Regridding does not cost an order; minmod ghost prolongation does (R9)

MFC's two-level regrid test (2D blob at (0.35, 0.35), u = v = 1, relative-jump tag 0.01, regrid every 0.02, t = 0.24,
subcycled at D vs lock-step at the finest dt; max |level-0 rho difference|), run in the mini-app (`jobs/case_R9.sh`):

| levels | ghosts / seeding | regrid every 0.02 | regrid only to t = 0.02 |
|---|---|---|---|
| 1 | quartic / quartic | 1.96, 1.90 | 1.99, 1.99 |
| 1 | minmod / minmod | 0.95, 0.98 | 1.09, 1.01 |
| 1 | quartic / minmod | 1.96, 1.80 | 1.99, 1.99 |
| 2 | quartic / quartic | 2.02, 2.01 | 1.91, 1.95 |
| 2 | minmod / minmod | 0.91, 0.96 | 1.08, 1.02 |
| 2 | quartic / minmod | 2.02, 2.01 | 1.91, 1.95 |

(orders for D = 2e-3 -> 1e-3 -> 5e-4; sub and lock built identical grids at every regrid.) The first-order term comes from
minmod in the *ghost* fill (time-interpolated coarse data reconstructed through a switching limiter), with or without
regrids; minmod seeding of new cells is harmless. With quartic ghosts the mini-app stays second order through 12 regrids,
so MFC's first order with quartic ghosts must come from something its regrid path does differently. The mini-app regrids
only at coarse synchronization (after the whole recursive subcycle, reflux and restriction of every level): per level l =
1.. top-down, fill level l-1 ghosts at the synchronized time, tag l-1, cluster, build the new level (arena, halo, coarse
cache, fill list, restriction and flux-register lists; registers start empty), gather the new l-1 state into the cache,
prolong every new cell from it, then copy old level-l data wherever old and new blocks overlap. The next step starts from
scratch: Uold copy, cache of the coarse step-start state (time interpolation), register capture with acc = 0.
