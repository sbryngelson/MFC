#!/usr/bin/env python3
"""
2D axisymmetric convergence case: density sine wave in x (axial/z), cyl_coord=T.

In MFC 2D cylindrical: x=axial(z), y=radial(r).
rho = 1 + 0.2*sin(2*pi*x), u_x=1, p=1, u_y=0.
Exact solution at time T: rho(x,T) = 1 + 0.2*sin(2*pi*(x-T)).
Domain [0,5] = 5 full periods of the sine IC; periodic x-BCs make the exact
solution valid everywhere (no upstream contamination from ghost cells).
"""

import argparse
import json
import math
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--mfc", type=str, default=None)
parser.add_argument("-N", type=int, default=64, help="Grid cells in x (axial) direction")
parser.add_argument("--order", type=int, default=5)
parser.add_argument("--cfl", type=float, default=0.02)
parser.add_argument("--nr", type=int, default=6, help="Radial cells (fixed)")
parser.add_argument("--Tfinal", type=float, default=0.1, help="Final simulation time")
args = parser.parse_args()

N = args.N  # axial cells (x-direction, refined)
Nr = args.nr  # radial cells (y-direction, fixed)
# Domain [0,5] = 5 periods of sin(2*pi*x); periodic x-BCs make it exact everywhere
Lx = 5.0
Lr = 0.5  # radial domain length

dx = Lx / N
dt = args.cfl * dx
Nt = math.ceil(args.Tfinal / dt)
dt = args.Tfinal / Nt  # adjust to hit Tfinal exactly

case = {
    # x=axial, y=radial
    "m": N - 1,
    "n": Nr - 1,
    "p": 0,
    "x_domain%beg": 0.0,
    "x_domain%end": Lx,
    "y_domain%beg": 0.0,
    "y_domain%end": Lr,
    "cyl_coord": "T",
    "dt": dt,
    "t_step_start": 0,
    "t_step_stop": Nt,
    "t_step_save": Nt,
    "weno_order": args.order,
    "weno_eps": 1e-16,
    "mapped_weno": "F",
    "wenoz": "F",
    "teno": "F",
    "mp_weno": "F",
    "time_stepper": 3,
    "riemann_solver": 2,
    "wave_speeds": 1,
    "avg_state": 2,
    "model_eqns": 2,
    "num_fluids": 1,
    "fluid_pp(1)%gamma": 1.4,
    "fluid_pp(1)%pi_inf": 0.0,
    # x: periodic (domain [0,5] = 5 full periods of the sine IC; wave advects cleanly)
    # y: symmetry axis at r=0, extrapolation at outer r (same as CI tests)
    "bc_x%beg": -1,
    "bc_x%end": -1,
    "bc_y%beg": -2,
    "bc_y%end": -3,
    "num_patches": 1,
    "patch_icpp(1)%geometry": 3,
    "patch_icpp(1)%x_centroid": Lx / 2,
    "patch_icpp(1)%y_centroid": Lr / 2,
    "patch_icpp(1)%length_x": Lx,
    "patch_icpp(1)%length_y": Lr,
    "patch_icpp(1)%vel(1)": 1.0,
    "patch_icpp(1)%vel(2)": 0.0,
    "patch_icpp(1)%pres": 1.0,
    "patch_icpp(1)%alpha_rho(1)": "1.0 + 0.2*sin(2.0*acos(-1.0_wp)*x)",
    "patch_icpp(1)%alpha(1)": 1.0,
}

if args.mfc is not None:
    print(json.dumps(case))
