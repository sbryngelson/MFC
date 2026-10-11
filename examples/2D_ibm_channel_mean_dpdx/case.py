#!/usr/bin/env python3
"""
2D periodic channel between two immersed-boundary wall slabs, driven by a mean
pressure gradient (mean_dpdx) or held at a target bulk mass flux (mean_mflux).

Gap: y in [0.05, 0.25] (H = 0.1). Steady Poiseuille profile for f = -dP/dx:
    u(y) = f/(2 mu) * (H^2 - (y - 0.15)^2),  Q = rho*u_bulk = rho*f*H^2/(3 mu).
--bc_walls replaces the slabs by no-slip domain walls (bc_y = -16) at the gap faces.
--ctrl targets Q = 1/120 from f = 0; the controller should settle at f = 0.05.
See compare.py.
"""

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--ny", type=int, default=64, help="cells across the gap")
parser.add_argument("--ctrl", action="store_true", help="hold the bulk mass flux instead of dP/dx")
parser.add_argument("--bc_walls", action="store_true", help="no-slip domain walls at the gap faces instead of IB slabs")
parser.add_argument("--t_stop", type=float, default=2.5)
args, _ = parser.parse_known_args()

L_x, L_y = 0.2, 0.3
slab = 0.099  # centroids just inside the domain; gap faces on cell boundaries
mu, rho, f = 0.02, 1.0, 0.05
dy = 0.2 / args.ny
dt = 6.0e-5 * min(1.0, (dy / 0.003125) ** 2)
nt = 5 * int(round(args.t_stop / dt / 5))

case = {
    "run_time_info": "F",
    "x_domain%beg": 0.0,
    "x_domain%end": L_x,
    "y_domain%beg": 0.0,
    "y_domain%end": L_y,
    "m": 31,
    "n": int(round(L_y / dy)) - 1,
    "p": 0,
    "dt": dt,
    "t_step_start": 0,
    "t_step_stop": nt,
    "t_step_save": nt // 5,
    "num_patches": 1,
    "model_eqns": 2,
    "num_fluids": 1,
    "mpp_lim": "F",
    "mixture_err": "T",
    "time_stepper": 3,
    "weno_order": 5,
    "weno_eps": 1e-16,
    "mapped_weno": "T",
    "weno_Re_flux": "T",
    "mp_weno": "T",
    "weno_avg": "T",
    "riemann_solver": 2,
    "wave_speeds": 1,
    "avg_state": 2,
    "bc_x%beg": -1,
    "bc_x%end": -1,
    "bc_y%beg": -16,
    "bc_y%end": -16,
    "viscous": "T",
    "ib": "T",
    "num_ibs": 2,
    "ib_state_wrt": "T",
    "format": 1,
    "precision": 2,
    "prim_vars_wrt": "T",
    "fd_order": 4,
    "parallel_io": "T",
    "patch_icpp(1)%geometry": 3,
    "patch_icpp(1)%x_centroid": 0.5 * L_x,
    "patch_icpp(1)%y_centroid": 0.5 * L_y,
    "patch_icpp(1)%length_x": L_x,
    "patch_icpp(1)%length_y": L_y,
    "patch_icpp(1)%vel(1)": 0.0,
    "patch_icpp(1)%vel(2)": 0.0,
    "patch_icpp(1)%pres": 10.0,
    "patch_icpp(1)%alpha_rho(1)": rho,
    "patch_icpp(1)%alpha(1)": 1.0,
    "fluid_pp(1)%gamma": 1.0 / (1.4 - 1.0),
    "fluid_pp(1)%eos": "ideal_gas",
    "fluid_pp(1)%Re(1)": 1.0 / mu,
}

for i, y_c in ((1, 0.05 - 0.5 * slab), (2, 0.25 + 0.5 * slab)):
    case |= {
        f"patch_ib({i})%geometry": 3,
        f"patch_ib({i})%x_centroid": 0.5 * L_x,
        f"patch_ib({i})%y_centroid": y_c,
        f"patch_ib({i})%length_x": 2.0 * L_x,
        f"patch_ib({i})%length_y": slab,
        f"patch_ib({i})%slip": "F",
        f"patch_ib({i})%mass": 0.0,
    }

if args.bc_walls:
    case = {k: v for k, v in case.items() if not k.startswith("patch_ib") and k not in ("ib", "num_ibs", "ib_state_wrt")}
    case |= {"y_domain%beg": 0.05, "y_domain%end": 0.25, "n": args.ny - 1, "patch_icpp(1)%y_centroid": 0.15, "patch_icpp(1)%length_y": 0.2}

if args.ctrl:
    case |= {"mean_mflux(1)": rho * f * 0.1**2 / (3.0 * mu), "mean_dpdx_gain": 0.05}
else:
    case |= {"mean_dpdx(1)": -f}

print(json.dumps(case))
