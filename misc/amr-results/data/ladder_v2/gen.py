#!/usr/bin/env python3
"""Simple-config MFC blast-lattice decks (ladder L of ../mfc-blast/gen.py): 5-eq model, one fluid, plain mapped WENO5 + HLLC (mp_weno off),
AMReX/PeleC tag rule (absolute density face jump >= 0.008), default prolongation, owner-aware placement (amr_place_owner).
Writes np<NP>_<lock|sub>_s<40|240>_r<1|2>[_off]/case.py; _off = placement off (A/B)."""
import json, pathlib, sys

RUNGS = {8: (2, 2, 2), 64: (4, 4, 4), 128: (8, 4, 4), 512: (8, 8, 8), 1024: (16, 8, 8)}  # = amrex-bench/inputs_blast/L-*/np*.amr prob_hi (np1024: ref1024/)
CPU = 200
B = pathlib.Path(__file__).parent


def deck(lens, steps, sub, np_, place=True):
    nx, ny, nz = (int(L * CPU) for L in lens)
    return {
        "run_time_info": "T", "rank_time_wrt": "T",
        "x_domain%beg": 0.0, "x_domain%end": lens[0], "y_domain%beg": 0.0, "y_domain%end": lens[1], "z_domain%beg": 0.0, "z_domain%end": lens[2],
        "m": nx - 1, "n": ny - 1, "p": nz - 1,
        "dt": 4.0e-4 if sub else 2.0e-4, "t_step_start": 0, "t_step_stop": steps, "t_step_save": steps,
        "num_patches": 1, "model_eqns": 2, "num_fluids": 1, "mpp_lim": "F", "mixture_err": "F", "time_stepper": 3,
        "weno_order": 5, "weno_eps": 1.0e-16, "mapped_weno": "T", "null_weights": "F", "mp_weno": "F",
        "riemann_solver": 2, "wave_speeds": 1, "avg_state": 2,
        "bc_x%beg": -1, "bc_x%end": -1, "bc_y%beg": -1, "bc_y%end": -1, "bc_z%beg": -1, "bc_z%end": -1,
        "format": 1, "precision": 2, "prim_vars_wrt": "F", "parallel_io": "T",
        "patch_icpp(1)%geometry": 9,
        "patch_icpp(1)%x_centroid": lens[0] / 2, "patch_icpp(1)%y_centroid": lens[1] / 2, "patch_icpp(1)%z_centroid": lens[2] / 2,
        "patch_icpp(1)%length_x": lens[0], "patch_icpp(1)%length_y": lens[1], "patch_icpp(1)%length_z": lens[2],
        "patch_icpp(1)%vel(1)": 0.0, "patch_icpp(1)%vel(2)": 0.0, "patch_icpp(1)%vel(3)": 0.0,
        "patch_icpp(1)%alpha_rho(1)": 1.0, "patch_icpp(1)%alpha(1)": 1.0,
        "patch_icpp(1)%pres": "1.0 + 9.0 * exp(-(cos(pi * x)**2 + cos(pi * y)**2 + cos(pi * z)**2) / 0.15)",
        "fluid_pp(1)%gamma": 1.0 / 0.4, "fluid_pp(1)%pi_inf": 0.0,
        "amr": "T", "amr_max_level": 1, "amr_ref_ratio": 2, "amr_regrid_int": 20, "amr_tag_eps": 0.008, "amr_tag_abs": "T", "amr_buf": 4,
        "amr_max_grid_size": 64, "amr_blocking_factor": 16, "amr_cluster_eff": 0.9, "amr_max_blocks": 512 * np_, "amr_prolong": 1,
        "amr_block_beg(1)": 32, "amr_block_end(1)": 47, "amr_block_beg(2)": 32, "amr_block_end(2)": 47, "amr_block_beg(3)": 32, "amr_block_end(3)": 47,
        "amr_subcycle": "T" if sub else "F", "amr_place_owner": "T" if place else "F",
    }


def write(np_, mode, steps, rep, place=True):
    d = B / f"np{np_}_{mode}_s{steps}_r{rep}{'' if place else '_off'}"; d.mkdir(exist_ok=True)
    (d / "case.py").write_text("#!/usr/bin/env python3\nimport json\nprint(json.dumps("
                               + json.dumps(deck(RUNGS[np_], steps, mode == "sub", np_, place), indent=1) + "))\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["off"]:  # placement A/B: np64 subcycled
        for steps in (40, 240):
            for rep in (1, 2): write(64, "sub", steps, rep, False)
        sys.exit()
    if len(sys.argv) > 1:  # fallback: gen.py STEPS MODE NP... (e.g. 239 sub 128 512: stop before the step-240 regrid)
        for np_ in map(int, sys.argv[3:]):
            for rep in (1, 2): write(np_, sys.argv[2], int(sys.argv[1]), rep)
        sys.exit()
    for np_, lens in RUNGS.items():
        for mode in ("lock", "sub"):
            for steps in (40, 240):
                for rep in (1, 2): write(np_, mode, steps, rep)
