#!/usr/bin/env python3
# 2D isentropic vortex convergence case (weak-vortex, small-epsilon formulation).
#
# Uses hcid=283: 3-pt Gauss-Legendre cell averages of conserved variables as IC.
# Vortex strength eps=0.01 (weak vortex) so the primitive→conserved covariance
# error O(eps^3 h^2) stays well below the WENO5 scheme error O(eps^2 h^5) at
# resolutions N=16..64.  With periodic BCs and a stationary vortex the comparison
# L2(rho(T) - rho(0)) isolates the scheme's accumulated spatial truncation error.
import argparse
import json
import math

parser = argparse.ArgumentParser(description="2D isentropic vortex convergence case")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
parser.add_argument("-N", type=int, default=32, help="Grid points per dim (default: 32)")
parser.add_argument("--order", type=int, default=5, help="WENO order: 1, 3, 5, or 7 (default: 5)")
parser.add_argument("--muscl", action="store_true", help="Use MUSCL instead of WENO")
parser.add_argument("--teno", action="store_true", help="Use TENO instead of WENO")
parser.add_argument("--teno-ct", type=float, default=1e-6, help="TENO CT threshold (default: 1e-6)")
parser.add_argument("--muscl-lim", type=int, default=0, help="MUSCL limiter: 0=unlimited 1=minmod ... (default: 0)")
args = parser.parse_args()

gamma = 1.4
eps_vortex = 0.01  # vortex strength: small enough that prim->cons floor is negligible
N = args.N
m = N - 1
dx = 10.0 / N

# Max wave speed: c_sound at ambient + max rotational velocity (at r~0.7 for exp(1-r^2))
c_max = math.sqrt(gamma) + eps_vortex / (2.0 * math.pi)
dt = 0.4 * dx / c_max
T_end = 2.0
Nt = max(4, math.ceil(T_end / dt))
dt = T_end / Nt  # adjust to land exactly on T_end

if args.muscl:
    scheme_params = {
        "recon_type": 2,
        "muscl_order": 2,
        "muscl_lim": args.muscl_lim,
    }
else:
    scheme_params = {
        "recon_type": 1,
        "weno_order": args.order,
        "weno_eps": 1.0e-40,
        "mapped_weno": "F" if (args.order == 1 or args.teno) else "T",
        "null_weights": "F",
        "mp_weno": "F",
        "teno": "T" if args.teno else "F",
        **({"teno_CT": args.teno_ct} if args.teno else {}),
    }

print(
    json.dumps(
        {
            "run_time_info": "F",
            "x_domain%beg": -5.0,
            "x_domain%end": 5.0,
            "y_domain%beg": -5.0,
            "y_domain%end": 5.0,
            "m": m,
            "n": m,
            "p": 0,
            "dt": dt,
            "t_step_start": 0,
            "t_step_stop": Nt,
            "t_step_save": Nt,
            "num_patches": 1,
            "model_eqns": 2,
            "alt_soundspeed": "F",
            "num_fluids": 1,
            "mpp_lim": "F",
            "mixture_err": "F",
            "time_stepper": 3,
            "riemann_solver": 2,
            "wave_speeds": 1,
            "avg_state": 2,
            "bc_x%beg": -4,
            "bc_x%end": -4,
            "bc_y%beg": -4,
            "bc_y%end": -4,
            "format": 1,
            "precision": 2,
            "prim_vars_wrt": "T",
            "parallel_io": "F",
            "patch_icpp(1)%geometry": 3,
            "patch_icpp(1)%x_centroid": 0.0,
            "patch_icpp(1)%y_centroid": 0.0,
            "patch_icpp(1)%length_x": 10.0,
            "patch_icpp(1)%length_y": 10.0,
            "patch_icpp(1)%hcid": 283,
            "patch_icpp(1)%epsilon": eps_vortex,
            "patch_icpp(1)%vel(1)": 0.0,
            "patch_icpp(1)%vel(2)": 0.0,
            "patch_icpp(1)%pres": 1.0,
            "patch_icpp(1)%alpha_rho(1)": 1.0,
            "patch_icpp(1)%alpha(1)": 1.0,
            "fluid_pp(1)%gamma": 1.0 / (gamma - 1.0),
            "fluid_pp(1)%pi_inf": 0.0,
            **scheme_params,
        }
    )
)
