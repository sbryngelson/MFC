#!/usr/bin/env python3
# 2D methane detonation with FULL GRI-Mech 3.0 chemistry (53 species, 325
# reactions) -- a real hydrocarbon fuel, not H2. A piston-driven shock ignites
# stoichiometric CH4/O2 (Ar-diluted for larger, resolvable cells) and couples
# into a self-sustained detonation. Demonstrates MFC's Pyrometheus pipeline on a
# large mechanism. Methane's high activation energy => irregular cells.
import argparse
import json
import sys

import cantera as ct

parser = argparse.ArgumentParser(prog="2D_methane_detonation")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT", help="MFC toolchain state.")
parser.add_argument("--scale", type=float, default=1.0, help="Grid multiplier.")
parser.add_argument("--ndim", type=int, default=2, choices=(2, 3), help="Spatial dimensions.")
parser.add_argument("--ar", type=float, default=3.0, help="Ar dilution per O2.")
parser.add_argument("--drivervel", type=float, default=1500.0, help="Piston speed [m/s].")
parser.add_argument("--overdrive", type=float, default=2.0, help="Driver over-compression.")
parser.add_argument("--seedamp", type=float, default=40.0, help="Transverse cell-seed amplitude [m/s].")
parser.add_argument("--tend", type=float, default=8.0e-5, help="Physical end time [s].")
args = parser.parse_args()
is_3d = args.ndim == 3

ctfile = "gri30.yaml"
X = f"CH4:1,O2:2,AR:{args.ar}"  # stoichiometric CH4 + 2 O2
T0, P0 = 298.0, 8000.0  # low p -> longer reaction zone / larger cells

fresh = ct.Solution(ctfile)
fresh.TPX = T0, P0, X
driver = ct.Solution(ctfile)
driver.TPX = T0, P0, X
driver.equilibrate("UV")
if args.overdrive > 1.0:
    driver.SP = driver.entropy_mass, args.overdrive * driver.P
print(f"CH4 driver: T={driver.T:.0f} K P={driver.P:.3e} rho={driver.density:.4f} nsp={fresh.n_species}", file=sys.stderr)

Ly = 0.04
Lz = Ly
Lx = 4.0 * Ly
if is_3d:
    Ny = int(40 * args.scale)
    Nx = int(4 * Ny)
    Nz = Ny
else:
    Ny = int(120 * args.scale)
    Nx = int(4 * Ny)
    Nz = 0
dx = Lx / Nx
x_driver = 0.25 * Lx
geom = 9 if is_3d else 3

D_cj = 1800.0
dt = 0.035 * dx / (D_cj + driver.sound_speed)
NT = int(args.tend / dt)
NS = max(1, NT // 80)

case = {
    "run_time_info": "T",
    "x_domain%beg": 0.0,
    "x_domain%end": Lx,
    "y_domain%beg": 0.0,
    "y_domain%end": Ly,
    "m": Nx,
    "n": Ny,
    "p": Nz,
    "dt": float(dt),
    "t_step_start": 0,
    "t_step_stop": NT,
    "t_step_save": NS,
    "t_step_print": NS,
    "parallel_io": "T",
    "model_eqns": "5eq",
    "num_fluids": 1,
    "num_patches": 2,
    "mpp_lim": "F",
    "mixture_err": "T",
    "weno_avg": "F",
    "time_stepper": "rk3",
    "weno_order": 5,
    "weno_eps": 1e-16,
    "mapped_weno": "T",
    "mp_weno": "T",
    "riemann_solver": "hllc",
    "wave_speeds": "direct",
    "avg_state": "arithmetic",
    "bc_x%beg": -3,
    "bc_x%end": -3,
    "bc_y%beg": -1,
    "bc_y%end": -1,
    "chemistry": "T",
    "chem_params%diffusion": "F",
    "chem_params%reactions": "T",
    "chem_params%reaction_substeps": 20,
    "cantera_file": ctfile,
    "format": "silo",
    "precision": "double",
    "prim_vars_wrt": "T",
    "chem_wrt_T": "T",
    # Patch 1: fresh premix, whole domain
    "patch_icpp(1)%geometry": geom,
    "patch_icpp(1)%x_centroid": Lx / 2,
    "patch_icpp(1)%y_centroid": Ly / 2,
    "patch_icpp(1)%length_x": Lx,
    "patch_icpp(1)%length_y": Ly,
    "patch_icpp(1)%vel(1)": 0.0,
    "patch_icpp(1)%vel(2)": 0.0,
    "patch_icpp(1)%pres": fresh.P,
    "patch_icpp(1)%alpha(1)": 1.0,
    "patch_icpp(1)%alpha_rho(1)": fresh.density,
    # Patch 2: overdriven driver, left, piston pushing right
    "patch_icpp(2)%geometry": geom,
    "patch_icpp(2)%alter_patch(1)": "T",
    "patch_icpp(2)%x_centroid": x_driver / 2,
    "patch_icpp(2)%y_centroid": Ly / 2,
    "patch_icpp(2)%length_x": x_driver,
    "patch_icpp(2)%length_y": Ly,
    "patch_icpp(2)%vel(1)": args.drivervel,
    "patch_icpp(2)%pres": driver.P,
    "patch_icpp(2)%alpha(1)": 1.0,
    "patch_icpp(2)%alpha_rho(1)": driver.density,
    "patch_icpp(2)%vel(2)": f"{args.seedamp}*sin(2*pi*3*y/{Ly})",
    "fluid_pp(1)%gamma": 1.0 / (1.3 - 1.0),
    "fluid_pp(1)%pi_inf": 0.0,
}

if is_3d:
    case.update(
        {
            "z_domain%beg": 0.0,
            "z_domain%end": Lz,
            "bc_z%beg": -1,
            "bc_z%end": -1,
            "patch_icpp(1)%z_centroid": Lz / 2,
            "patch_icpp(1)%length_z": Lz,
            "patch_icpp(1)%vel(3)": 0.0,
            "patch_icpp(2)%z_centroid": Lz / 2,
            "patch_icpp(2)%length_z": Lz,
            "patch_icpp(2)%vel(3)": 0.0,
        }
    )
    case["patch_icpp(2)%vel(3)"] = f"{args.seedamp}*sin(2*pi*3*z/{Lz})"

for i in range(fresh.n_species):
    case[f"patch_icpp(1)%Y({i + 1})"] = float(fresh.Y[i])
    case[f"patch_icpp(2)%Y({i + 1})"] = float(driver.Y[i])
# output a couple of markers only (53 species is a lot)
for sp in ("OH", "CH", "CO"):
    if fresh.species_index(sp) >= 0:
        case[f"chem_wrt_Y({fresh.species_index(sp) + 1})"] = "T"

if __name__ == "__main__":
    print(json.dumps(case))
