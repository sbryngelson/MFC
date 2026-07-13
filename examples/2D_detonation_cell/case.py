#!/usr/bin/env python3
# 2D cellular detonation in argon-diluted H2/O2 (2H2 + O2 + 7Ar).
# Heavy Ar dilution -> regular cells. Reactions ON, diffusion OFF (shock-driven).
import argparse
import json
import sys

import cantera as ct

parser = argparse.ArgumentParser(prog="2D_detonation_cell")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT", help="MFC toolchain state.")
parser.add_argument("--scale", type=float, default=1.0, help="Grid multiplier (calibration knob).")
parser.add_argument("--tend", type=float, default=200e-6, help="Physical end time [s].")
parser.add_argument(
    "--overdrive",
    type=float,
    default=2.5,
    help="Isentropic compression of the UV-equilibrium driver (1 = plain; >1 = overdriven, thermo-consistent). ~2.5 sits near CJ pressure and launches a detonation.",
)
args = parser.parse_args()

ctfile = "h2o2.yaml"
X = "H2:2,O2:1,AR:7"  # Ar-diluted -> regular cells
T0, P0 = 298.0, 6670.0  # low p lengthens the reaction zone (explicit-solver friendly)

# Fresh premixed reactants (fills the domain).
fresh = ct.Solution(ctfile)
fresh.TPX = T0, P0, X

# Detonation driver: constant-volume explosion products -> a thermodynamically
# self-consistent hot, high-pressure burned state (T, P, rho all from Cantera).
# --overdrive (>1) isentropically compresses the products for extra drive, scaling
# P and rho together so T stays in the thermo range. Forcing P = overdrive*P0 at
# the fresh density would imply T ~ 9000 K (past the h2o2 NASA-polynomial limit)
# and NaN on the first RHS eval.
driver = ct.Solution(ctfile)
driver.TPX = T0, P0, X
driver.equilibrate("UV")
if args.overdrive > 1.0:
    driver.SP = driver.entropy_mass, args.overdrive * driver.P
P_drive = driver.P
print(f"driver init: T={driver.T:.1f} K  P={driver.P:.1f} Pa  rho={driver.density:.5f} kg/m^3", file=sys.stderr)

# --- Domain: x = propagation, y = transverse (periodic). Nominal; refined by calibration. ---
Ly = 0.03  # channel height [m] ~ a few cells (verify/refine)
Lx = 4.0 * Ly  # run-up length
Ny = int(200 * args.scale)
Nx = int(4 * Ny)
dx = Lx / Nx

# CJ speed ~1.6 km/s for this mixture; size dt from the fastest signal.
D_cj_guess = 1600.0
c_drive = driver.sound_speed
dt = 0.02 * dx / (D_cj_guess + c_drive)  # CFL ~ 0.02 (stiff reacting shock; cf. 1D_reactive_shocktube)

NT = int(args.tend / dt)
NS = max(1, NT // 100)

x_driver = 0.25 * Lx  # driver occupies the left 25% (sustains the shock so it locks into a detonation)

case = {
    "run_time_info": "T",
    # Domain
    "x_domain%beg": 0.0,
    "x_domain%end": Lx,
    "y_domain%beg": 0.0,
    "y_domain%end": Ly,
    "m": Nx,
    "n": Ny,
    "p": 0,
    "dt": float(dt),
    "t_step_start": 0,
    "t_step_stop": NT,
    "t_step_save": NS,
    "t_step_print": NS,
    "parallel_io": "T",
    # Algorithm
    "model_eqns": "5eq",
    "num_fluids": 1,
    "num_patches": 3,
    "mpp_lim": "F",
    "mixture_err": "F",
    "weno_avg": "F",
    "time_stepper": "rk3",
    "weno_order": 5,
    "weno_eps": 1e-16,
    "mapped_weno": "T",
    "mp_weno": "T",
    "riemann_solver": "hllc",
    "wave_speeds": "direct",
    "avg_state": "arithmetic",
    # BCs: x extrapolation (open), y periodic (clean regular cells)
    "bc_x%beg": -3,
    "bc_x%end": -3,
    "bc_y%beg": -1,
    "bc_y%end": -1,
    # Chemistry
    "chemistry": "T",
    "chem_params%diffusion": "F",
    "chem_params%reactions": "T",
    "cantera_file": ctfile,
    # Output
    "format": "silo",
    "precision": "double",
    "prim_vars_wrt": "T",
    "chem_wrt_T": "T",
    # Patch 1: fresh reactants, whole domain
    "patch_icpp(1)%geometry": 3,
    "patch_icpp(1)%x_centroid": Lx / 2,
    "patch_icpp(1)%y_centroid": Ly / 2,
    "patch_icpp(1)%length_x": Lx,
    "patch_icpp(1)%length_y": Ly,
    "patch_icpp(1)%vel(1)": 0.0,
    "patch_icpp(1)%vel(2)": 0.0,
    "patch_icpp(1)%pres": fresh.P,
    "patch_icpp(1)%alpha(1)": 1.0,
    "patch_icpp(1)%alpha_rho(1)": fresh.density,
    # Patch 2: overdriven driver (hot products), left region, full height
    "patch_icpp(2)%geometry": 3,
    "patch_icpp(2)%alter_patch(1)": "T",
    "patch_icpp(2)%x_centroid": x_driver / 2,
    "patch_icpp(2)%y_centroid": Ly / 2,
    "patch_icpp(2)%length_x": x_driver,
    "patch_icpp(2)%length_y": Ly,
    "patch_icpp(2)%vel(1)": 0.0,
    "patch_icpp(2)%vel(2)": 0.0,
    "patch_icpp(2)%pres": P_drive,
    "patch_icpp(2)%alpha(1)": 1.0,
    "patch_icpp(2)%alpha_rho(1)": driver.density,
    # Patch 3: off-center hot pocket just ahead of driver -> seeds transverse cells
    "patch_icpp(3)%geometry": 2,
    "patch_icpp(3)%alter_patch(1)": "T",
    "patch_icpp(3)%x_centroid": x_driver * 1.2,
    "patch_icpp(3)%y_centroid": 0.35 * Ly,
    "patch_icpp(3)%radius": 0.04 * Ly,
    "patch_icpp(3)%vel(1)": 0.0,
    "patch_icpp(3)%vel(2)": 0.0,
    "patch_icpp(3)%pres": P_drive,
    "patch_icpp(3)%alpha(1)": 1.0,
    "patch_icpp(3)%alpha_rho(1)": driver.density,
    # Fluid EOS (ideal-gas closure is bypassed by chemistry, but gamma must be set)
    "fluid_pp(1)%gamma": 1.0 / (1.4 - 1.0),
    "fluid_pp(1)%pi_inf": 0.0,
}

# Species mass fractions per patch + per-species output
for i in range(len(fresh.Y)):
    case[f"chem_wrt_Y({i + 1})"] = "T"
    case[f"patch_icpp(1)%Y({i + 1})"] = float(fresh.Y[i])
    case[f"patch_icpp(2)%Y({i + 1})"] = float(driver.Y[i])
    case[f"patch_icpp(3)%Y({i + 1})"] = float(driver.Y[i])

if __name__ == "__main__":
    print(json.dumps(case))
