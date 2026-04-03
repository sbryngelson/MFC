#!/usr/bin/env python3
import json
import math
import sys

# Grid size from environment or default
N = 128

print(
    json.dumps(
        {
            "run_time_info": "F",
            "x_domain%beg": 0,
            "x_domain%end": 1,
            "y_domain%beg": 0,
            "y_domain%end": 1,
            "z_domain%beg": 0,
            "z_domain%end": 1,
            "m": N,
            "n": N,
            "p": N,
            "dt": 1e-6,
            "t_step_start": 0,
            "t_step_stop": 500,
            "t_step_save": 500,
            "num_patches": 1,
            "model_eqns": 2,
            "num_fluids": 2,
            "time_stepper": 3,
            "weno_order": 5,
            "weno_eps": 1e-16,
            "mapped_weno": "T",
            "riemann_solver": 2,
            "wave_speeds": 1,
            "avg_state": 2,
            "bc_x%beg": -1,
            "bc_x%end": -1,
            "bc_y%beg": -1,
            "bc_y%end": -1,
            "bc_z%beg": -1,
            "bc_z%end": -1,
            "format": 1,
            "precision": 2,
            "prim_vars_wrt": "T",
            "parallel_io": "T",
            "patch_icpp(1)%geometry": 9,
            "patch_icpp(1)%x_centroid": 0.5,
            "patch_icpp(1)%y_centroid": 0.5,
            "patch_icpp(1)%z_centroid": 0.5,
            "patch_icpp(1)%length_x": 1,
            "patch_icpp(1)%length_y": 1,
            "patch_icpp(1)%length_z": 1,
            "patch_icpp(1)%vel(1)": 1.0,
            "patch_icpp(1)%vel(2)": 0.0,
            "patch_icpp(1)%vel(3)": 0.0,
            "patch_icpp(1)%pres": 1e5,
            "patch_icpp(1)%alpha_rho(1)": 1.0,
            "patch_icpp(1)%alpha_rho(2)": 0.1,
            "patch_icpp(1)%alpha(1)": 0.9,
            "patch_icpp(1)%alpha(2)": 0.1,
            "fluid_pp(1)%gamma": 0.4,
            "fluid_pp(1)%pi_inf": 0,
            "fluid_pp(2)%gamma": 0.4,
            "fluid_pp(2)%pi_inf": 0,
        }
    )
)
