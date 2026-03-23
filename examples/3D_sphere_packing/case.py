#!/usr/bin/env python3
# 3D sphere packing: ~250K uniform spheres via jittered FCC lattice
# with immersed boundary method in a cubic domain.

import json
import math

# Domain
L = 0.01  # 1 cm cube
Nx = Ny = Nz = 199

# Sphere packing
R = 4.92e-05       # sphere radius
vf = 0.50           # target solid volume fraction
min_gap = 0.1 * R   # minimum surface-to-surface gap

# Flow
gam_a = 1.4
rho_a = 1.2
p_a = 101325.0
c_a = math.sqrt(gam_a * p_a / rho_a)
Ma = 0.1
u_a = Ma * c_a

# Time stepping
dx = L / Nx
dt = 0.25 * dx / (c_a + abs(u_a))
t_stop = 500
t_save = 50

print(
    json.dumps(
        {
            # Logistics
            "run_time_info": "T",
            # Computational Domain
            "x_domain%beg": 0.0,
            "x_domain%end": L,
            "y_domain%beg": 0.0,
            "y_domain%end": L,
            "z_domain%beg": 0.0,
            "z_domain%end": L,
            "m": Nx,
            "n": Ny,
            "p": Nz,
            "cyl_coord": "F",
            "dt": dt,
            "t_step_start": 0,
            "t_step_stop": t_stop,
            "t_step_save": t_save,
            # Simulation Algorithm
            "model_eqns": 2,
            "alt_soundspeed": "F",
            "mixture_err": "T",
            "mpp_lim": "F",
            "time_stepper": 3,
            "weno_order": 5,
            "weno_eps": 1.0e-16,
            "weno_avg": "T",
            "avg_state": 2,
            "mapped_weno": "T",
            "null_weights": "F",
            "mp_weno": "F",
            "riemann_solver": 2,
            "wave_speeds": 1,
            # Boundary conditions (walls)
            "bc_x%beg": -2,
            "bc_x%end": -3,
            "bc_y%beg": -2,
            "bc_y%end": -3,
            "bc_z%beg": -2,
            "bc_z%end": -3,
            # Immersed boundaries (sphere packing)
            "ib": "T",
            "num_ibs": 0,
            "num_patches": 1,
            "num_fluids": 1,
            # Sphere packing parameters
            "sphere_pack": "T",
            "sphere_pack_radius": R,
            "sphere_pack_void_frac": 1 - vf,
            "sphere_pack_min_gap": min_gap,
            "sphere_pack_seed": 42,
            # Output
            "format": 1,
            "precision": 2,
            "prim_vars_wrt": "T",
            "parallel_io": "T",
            # Patch: uniform air filling the domain
            "patch_icpp(1)%geometry": 9,
            "patch_icpp(1)%x_centroid": L / 2,
            "patch_icpp(1)%y_centroid": L / 2,
            "patch_icpp(1)%z_centroid": L / 2,
            "patch_icpp(1)%length_x": L,
            "patch_icpp(1)%length_y": L,
            "patch_icpp(1)%length_z": L,
            "patch_icpp(1)%vel(1)": u_a,
            "patch_icpp(1)%vel(2)": 0.0,
            "patch_icpp(1)%vel(3)": 0.0,
            "patch_icpp(1)%pres": p_a,
            "patch_icpp(1)%alpha_rho(1)": rho_a,
            "patch_icpp(1)%alpha(1)": 1.0,
            # Fluid parameters (air)
            "fluid_pp(1)%gamma": 1.0 / (gam_a - 1.0),
            "fluid_pp(1)%pi_inf": 0.0,
        }
    )
)
