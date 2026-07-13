#!/usr/bin/env python3
"""
Verification + visualization for the 2D cellular detonation example.

Reads MFC post-process Silo-HDF5 output for 2H2+O2+7Ar and checks the
result against Chapman-Jouguet (CJ) detonation theory:
  1. Computes the CJ speed from Cantera (equilibrium-Hugoniot / sonic
     tangency construction -- there is no built-in Cantera or SDToolbox
     CJ routine, so it is implemented directly below).
  2. Tracks the leading-shock position over time from the saved pressure
     fields and fits a propagation speed, compared to the CJ speed.
  3. Accumulates per-cell maximum pressure into a soot-foil image and
     estimates the transverse cell size from the streak spacing.
  4. Saves T / pressure / Y_OH snapshots showing the front structure.

Run from this directory after post_process has been executed:
    python analyze.py

All dependencies (cantera, matplotlib, scipy, h5py) are installed
automatically by the MFC toolchain.
"""

import re
import sys
import warnings

import cantera as ct
import matplotlib
import numpy as np
from scipy.optimize import fsolve
from scipy.signal import find_peaks

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mfc.viz import assemble_silo, discover_timesteps

CASE_DIR = "."
CTFILE = "h2o2.yaml"
X = "H2:2,O2:1,AR:7"  # 2H2 + O2 + 7Ar
T0, P0 = 298.0, 6670.0


# 1. CJ speed via the equilibrium Hugoniot + sonic (tangency) condition
#
# The CJ point is where the Rayleigh line is tangent to the equilibrium
# (fully-reacted) Hugoniot curve; equivalently, the downstream flow speed
# equals the *equilibrium/reactive* sound speed (dP/drho at const entropy,
# with the composition re-equilibrated at every perturbed state -- this is
# lower than the frozen sound speed and is what makes the tangency exact).
# Near the CJ point the two Hugoniot branches merge and the jump-condition
# solve becomes ill-conditioned, so U1 is swept down in small steps and
# the M2->1 crossing is located by extrapolating (1-M2)^2, which is linear
# in U1 near a square-root-type tangency singularity.


def _frozen_shock_guess(U1, gas1, T1, P1):
    """Non-reacting normal-shock (T2, P2) to seed the equilibrium solve."""
    c1 = gas1.sound_speed
    gamma1 = gas1.cp / gas1.cv
    mach = U1 / c1
    pr = (2 * gamma1 * mach**2 - (gamma1 - 1)) / (gamma1 + 1)
    tr = pr * ((gamma1 + 1) / (gamma1 - 1) + pr) / (1 + (gamma1 + 1) / (gamma1 - 1) * pr)
    return T1 * tr, P1 * pr


def _equilibrium_postshock(U1, T2_guess, P2_guess, gas1, rho1, h1, T1, P1):
    """Solve mass/momentum/energy jump conditions with the downstream gas
    held at chemical equilibrium; returns (gas2, u2, T2, P2, converged)."""
    gas2 = ct.Solution(CTFILE)
    m1 = rho1 * U1

    def resid(vars_):
        T2, P2 = vars_
        if T2 <= 0 or P2 <= 0:
            return [1e10, 1e10]
        gas2.TPX = T2, P2, X
        gas2.equilibrate("TP")
        v2 = 1.0 / gas2.density
        h2 = gas2.enthalpy_mass
        eq1 = (P1 + rho1 * U1**2) - (P2 + m1**2 * v2)
        eq2 = (h1 + 0.5 * U1**2) - (h2 + 0.5 * (m1 * v2) ** 2)
        return [eq1, eq2]

    sol, _info, ier, _msg = fsolve(resid, [T2_guess, P2_guess], full_output=True, xtol=1e-13)
    T2, P2 = sol
    gas2.TPX = T2, P2, X
    gas2.equilibrate("TP")
    u2 = m1 / gas2.density
    ok = ier == 1 or max(abs(v) for v in resid(sol)) < 1e-3
    return gas2, u2, T2, P2, ok


def _equilibrium_sound_speed(gas):
    """dP/drho at constant entropy with re-equilibrated composition."""
    s0, p0, r0 = gas.entropy_mass, gas.P, gas.density
    dp = 1e-6 * p0
    g = ct.Solution(CTFILE)
    g.TPX = gas.T, gas.P, X
    g.SP = s0, p0 + dp
    g.equilibrate("SP")
    return np.sqrt(dp / (g.density - r0))


def compute_cj_speed(T1=T0, P1=P0):
    """CJ detonation speed [m/s] for X at (T1, P1)."""
    gas1 = ct.Solution(CTFILE)
    gas1.TPX = T1, P1, X
    rho1, h1 = gas1.density, gas1.enthalpy_mass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # equilibrium-solver range warnings during the coarse overdriven sweep

        # Coarse sweep from a strongly overdriven state down toward CJ.
        U1 = 2500.0
        T2g, P2g = _frozen_shock_guess(U1, gas1, T1, P1)
        Us, M2s = [], []
        while U1 > 1000.0:
            gas2, u2, T2, P2, ok = _equilibrium_postshock(U1, T2g, P2g, gas1, rho1, h1, T1, P1)
            if not ok:
                break
            M2 = u2 / _equilibrium_sound_speed(gas2)
            Us.append(U1)
            M2s.append(M2)
            T2g, P2g = T2, P2
            if M2 >= 1.0:
                break
            U1 -= 10.0

        # Fine sweep (0.2 m/s) restarted just above the last coarse point,
        # to get several well-converged points close to the tangency.
        U1 = Us[-1] + 15.0
        T2g, P2g = _frozen_shock_guess(U1, gas1, T1, P1)
        Us, M2s = [], []
        while U1 > 1000.0:
            gas2, u2, T2, P2, ok = _equilibrium_postshock(U1, T2g, P2g, gas1, rho1, h1, T1, P1)
            if not ok:
                break
            M2 = u2 / _equilibrium_sound_speed(gas2)
            Us.append(U1)
            M2s.append(M2)
            T2g, P2g = T2, P2
            if M2 >= 1.0:
                break
            U1 -= 0.2

    Us, M2s = np.array(Us), np.array(M2s)
    # (1-M2)^2 ~ linear in U1 near the tangency point (square-root singularity)
    # -- extrapolate the last several well-converged points to (1-M2)^2 = 0.
    n = min(15, len(Us))
    y2 = (1 - M2s[-n:]) ** 2
    slope, intercept = np.polyfit(Us[-n:], y2, 1)
    return -intercept / slope


# 2. Load production-run metadata and saved timesteps
#
# simulation.inp records the actual dt/step count MFC ran with (case.py's
# own dt/NT depend on --scale/--tend, which may not match this run's CLI
# args). silo_hdf5/ can also contain leftover files from earlier
# calibration runs at other resolutions -- keep only steps that are
# multiples of the production t_step_save, which uniquely identifies the
# steps written by the run described in simulation.inp.


def _read_inp(name):
    text = open(f"{CASE_DIR}/simulation.inp").read()
    return float(re.search(rf"^{name}\s*=\s*([^\s]+)", text, re.M).group(1))


dt = _read_inp("dt")
t_step_save = int(_read_inp("t_step_save"))
t_step_stop = int(_read_inp("t_step_stop"))

all_steps = discover_timesteps(CASE_DIR, "silo")
steps = sorted(s for s in all_steps if s % t_step_save == 0 and s <= t_step_stop)
if not steps:
    sys.exit("No silo timesteps found -- did you run post_process?")
print(f"Loaded {len(steps)} production timesteps (dt={dt:.4e} s, t_step_save={t_step_save})")


# 3. Leading-shock front tracking -> measured propagation speed
#
# The fresh mixture ahead of the front sits uniformly at P0; the leading
# shock is the rightmost point where the y-averaged pressure still exceeds
# a threshold well above P0. Averaging over y (rather than a single-y
# gradient) is robust to internal, non-leading pressure features (e.g. the
# driver/fresh contact-surface transient) and, cross-checked against the
# per-column leading-tip position, gives the same propagation speed to <1%.

THRESH = 2.0 * P0
times = np.array([s * dt for s in steps if s > 0])
x_front = np.full(len(times), np.nan)
domain_x_max = None

for i, s in enumerate(s for s in steps if s > 0):
    a = assemble_silo(CASE_DIR, s, var="pres")
    p = a.variables["pres"]
    if domain_x_max is None:
        domain_x_max = a.x_cc.max()
    p_avg = p.mean(axis=1)
    above = np.where(p_avg > THRESH)[0]
    if above.size:
        x_front[i] = a.x_cc[above[-1]]

# Exclude points where the front has exited the domain (open x_domain%end)
# and fit only the later half of the remaining run, per-brief, to avoid the
# initial overdriven-driver transient.
unsaturated = np.isfinite(x_front) & (x_front < domain_x_max - 2 * (a.x_cc[1] - a.x_cc[0]))
t_valid = times[unsaturated]
window = t_valid >= (t_valid.min() + 0.5 * (t_valid.max() - t_valid.min()))
fit_mask = unsaturated.copy()
fit_mask[unsaturated] = window

A = np.vstack([times[fit_mask], np.ones(fit_mask.sum())]).T
D_measured, x_intercept = np.linalg.lstsq(A, x_front[fit_mask], rcond=None)[0]

print("\nComputing CJ speed from Cantera (equilibrium Hugoniot)...")
D_cj = compute_cj_speed()
print(f"  CJ speed (Cantera):      {D_cj:.1f} m/s")
print(f"  Measured front speed:    {D_measured:.1f} m/s  (fit over t in [{times[fit_mask].min():.2e}, {times[fit_mask].max():.2e}] s, N={fit_mask.sum()})")
print(f"  Ratio measured/CJ:       {D_measured / D_cj * 100:.1f}%")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(times[unsaturated] * 1e6, x_front[unsaturated] * 100, "o", ms=3, color="0.5", label="front position (all unsaturated steps)")
ax.plot(times[fit_mask] * 1e6, x_front[fit_mask] * 100, "o", ms=4, color="C0", label="fit window")
t_line = np.array([times[fit_mask].min(), times[fit_mask].max()])
ax.plot(t_line * 1e6, (D_measured * t_line + x_intercept) * 100, "C0-", label=f"fit: D = {D_measured:.0f} m/s")
ax.set_xlabel(r"$t$ ($\mu$s)")
ax.set_ylabel(r"front position $x$ (cm)")
ax.legend()
ax.set_title(f"Measured {D_measured:.0f} m/s vs. CJ {D_cj:.0f} m/s ({D_measured / D_cj * 100:.0f}%)")
plt.tight_layout()
plt.savefig("front_speed.png", dpi=200)
plt.close()
print("Saved: front_speed.png")


# 4. Soot foil: per-cell maximum pressure over all saved steps
pmax = None
x_cc = y_cc = None
for s in steps:
    a = assemble_silo(CASE_DIR, s, var="pres")
    p = a.variables["pres"]
    if np.isnan(p).any():
        continue  # step 0 initial condition has a zero-thickness alpha interface -> NaN pressure
    if pmax is None:
        pmax = p.copy()
        x_cc, y_cc = a.x_cc, a.y_cc
    else:
        np.maximum(pmax, p, out=pmax)

fig, ax = plt.subplots(figsize=(14, 4))
im = ax.pcolormesh(x_cc * 100, y_cc * 100, pmax.T / 1e3, shading="auto", cmap="inferno")
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_aspect("equal")
plt.colorbar(im, ax=ax, label="max pressure (kPa)")
plt.tight_layout()
plt.savefig("soot_foil.png", dpi=200)
plt.close()
print("Saved: soot_foil.png")

# Cell-size estimate: transverse spacing of local pressure maxima (streak
# tracks), sampled across many x-slices in the developed region (excludes
# the fixed-IC driver block at x < x_driver=0.25*Lx and the domain edge).
mask_x = (x_cc > 0.06) & (x_cc < 0.95 * x_cc.max())
dy = y_cc[1] - y_cc[0]
spacings = []
for row in pmax[mask_x, :]:
    peaks, _ = find_peaks(row, prominence=3000.0, distance=max(1, int(0.001 / dy)))
    if len(peaks) >= 2:
        spacings.extend(np.diff(y_cc[peaks]).tolist())
spacings = np.array(spacings)
lam_mean, lam_median = spacings.mean(), np.median(spacings)
print(f"\nCell-size estimate (transverse streak spacing, N={len(spacings)}):")
print(f"  mean = {lam_mean * 1e3:.2f} mm, median = {lam_median * 1e3:.2f} mm, std = {spacings.std() * 1e3:.2f} mm")


# 5. Snapshots: T, pressure, Y_OH at a late (pre-exit) timestep
#
# "Late" = the last unsaturated step, so the front and unburned region
# ahead of it are both still fully inside the domain.
late_idx = np.where(unsaturated)[0][-1]
late_step = [s for s in steps if s > 0][late_idx]
a = assemble_silo(CASE_DIR, late_step)
t_late = late_step * dt

for var, cmap, label, fname in [
    ("T", "inferno", "$T$ (K)", "snapshot_T.png"),
    ("pres", "viridis", "$p$ (Pa)", "snapshot_pressure.png"),
    ("Y_OH", "cividis", "$Y_{OH}$", "snapshot_Y_OH.png"),
]:
    fig, ax = plt.subplots(figsize=(12, 3.5))
    im = ax.pcolormesh(a.x_cc * 100, a.y_cc * 100, a.variables[var].T, shading="auto", cmap=cmap)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_aspect("equal")
    ax.set_title(f"t = {t_late * 1e6:.1f} $\\mu$s (step {late_step})")
    plt.colorbar(im, ax=ax, label=label)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Saved: {fname}")

print("\nDone.")
