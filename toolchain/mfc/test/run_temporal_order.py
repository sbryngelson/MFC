#!/usr/bin/env python3
"""
Time-integration order verification for MFC's RK1, RK2, and RK3 time steppers.

Uses the 1D single-fluid Euler advection problem (rho = 1 + 0.2*sin(2*pi*x),
u=1, p=1, L=1, T=1) with a fine spatial grid (N=512, WENO5) so the spatial
error (~4e-12) is negligible compared to the temporal error at the CFLs tested.

L2(rho(T) - rho(0)) measures total accumulated error.  By fixing N and varying
CFL (and hence dt), the spatial contribution is constant and the measured rate
reflects the time integration order.

CFL ranges are chosen to be within each stepper's stability region and keep
temporal errors well above the ~4e-12 spatial floor:
  RK1 (Euler, 1st order): CFL=[0.10, 0.05] — stable limit ~0.1 with WENO5+LF
    (nearly-imaginary eigenvalues constrain Euler more than TVD RK);
    error ~2.5e-4 and ~1.2e-4 (rate ≈ 1.0)
  RK2 (TVD Heun, 2nd order): CFL=[0.50, 0.25];
    error ~1.2e-6 and ~2.9e-7 (rate ≈ 2.0)
  RK3 (TVD Shu-Osher, 3rd order): CFL=[0.50, 0.25];
    error ~8.3e-10 and ~1.1e-10 (rate ≈ 3.0)

Usage:
    python toolchain/mfc/test/run_temporal_order.py
    python toolchain/mfc/test/run_temporal_order.py --schemes RK3/WENO5 --cfls 0.5 0.25 0.125
"""

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

CASE = "examples/1D_euler_convergence/case.py"
MFC = "./mfc.sh"

# (label, extra_args, expected_order, tolerance, cfls)
# N=512 is fixed; WENO5 keeps spatial error ~4e-12 (negligible at CFL>=0.25).
# RK1/RK2 temporal errors at CFL=0.5 are ~2e-4 and ~2e-7, both >> spatial floor.
SCHEMES = [
    # RK1 (Forward Euler): nearly-imaginary WENO5+LF eigenvalues constrain stability
    # to CFL < ~0.1; use [0.10, 0.05] which are provably stable and temporal-dominated
    ("RK1/WENO5", ["--order", "5", "--time-stepper", "1"], 1, 0.1, [0.10, 0.05]),
    # RK2/RK3 (TVD): stable to CFL~1; use [0.50, 0.25] for clean temporal dominance
    ("RK2/WENO5", ["--order", "5", "--time-stepper", "2"], 2, 0.2, [0.50, 0.25]),
    ("RK3/WENO5", ["--order", "5", "--time-stepper", "3"], 3, 0.3, [0.50, 0.25]),
]

N_SPATIAL = 512  # fixed spatial resolution


def read_cons_var(run_dir: str, step: int, var_idx: int, num_ranks: int = 1, expected_size: int = None) -> np.ndarray:
    """Read q_cons_vf{var_idx} from all MPI ranks and concatenate into one 1D array."""
    chunks = []
    for rank in range(num_ranks):
        path = os.path.join(run_dir, "p_all", f"p{rank}", str(step), f"q_cons_vf{var_idx}.dat")
        with open(path, "rb") as f:
            rec_len = struct.unpack("i", f.read(4))[0]
            data = np.frombuffer(f.read(rec_len), dtype=np.float64)
            f.read(4)
        chunks.append(data.copy())
    combined = np.concatenate(chunks)
    if expected_size is not None and combined.size != expected_size:
        raise ValueError(f"Expected {expected_size} values across {num_ranks} ranks, got {combined.size}")
    return combined


# 1D single-fluid Euler (model_eqns=2, num_fluids=1): vf1=ρ, vf2=ρu, vf3=E
CONS_VARS_1D = [("density", 1), ("x-momentum", 2), ("energy", 3)]
CONS_TOL = 1e-10


def conservation_errors(run_dir: str, Nt: int, cell_vol: float, var_list: list, num_ranks: int, expected_size: int = None) -> dict:
    """Return relative conservation error |Σq(T) - Σq(0)| / |Σq(0)| for each variable."""
    errs = {}
    for name, idx in var_list:
        q0 = read_cons_var(run_dir, 0, idx, num_ranks, expected_size)
        qT = read_cons_var(run_dir, Nt, idx, num_ranks, expected_size)
        s0 = float(np.sum(q0)) * cell_vol
        sT = float(np.sum(qT)) * cell_vol
        errs[name] = abs(sT - s0) / (abs(s0) + 1e-300)
    return errs


def l2_error(a: np.ndarray, b: np.ndarray, dx: float) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2) * dx))


def run_case(tmpdir: str, cfl: float, extra_args: list, num_ranks: int = 1):
    """Run the 1D advection case at fixed N=512 with given CFL. Returns (dt, Nt, run_dir)."""
    result = subprocess.run(
        [sys.executable, CASE, "--mfc", "{}", "-N", str(N_SPATIAL), "--cfl", str(cfl)] + extra_args,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"case.py failed:\n{result.stderr}")
    cfg = json.loads(result.stdout)
    Nt = int(cfg["t_step_stop"])
    dt = float(cfg["dt"])

    cmd = [
        MFC,
        "run",
        CASE,
        "-t",
        "pre_process",
        "simulation",
        "-n",
        str(num_ranks),
        "--",
        "-N",
        str(N_SPATIAL),
        "--cfl",
        str(cfl),
    ] + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), check=False)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr)
        raise RuntimeError(f"./mfc.sh run failed for CFL={cfl}")

    case_dir = os.path.dirname(CASE)
    src = os.path.join(case_dir, "p_all")
    cfl_tag = f"cfl{cfl:.4f}".replace(".", "p")
    dst = os.path.join(tmpdir, cfl_tag, "p_all")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    shutil.rmtree(src, ignore_errors=True)
    shutil.rmtree(os.path.join(case_dir, "D"), ignore_errors=True)

    return dt, Nt, os.path.join(tmpdir, cfl_tag)


def test_scheme(label, extra_args, expected_order, tol, cfls, num_ranks=1):
    print(f"\n{'=' * 60}")
    print(f"  {label}  N={N_SPATIAL}  (need rate >= {expected_order - tol:.1f})")
    print(f"{'=' * 60}")

    errors = []
    dts = []
    nts = []
    dx = 1.0 / N_SPATIAL
    all_cons_errs = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for cfl in cfls:
            dt, Nt, run_dir = run_case(tmpdir, cfl, extra_args, num_ranks)
            dts.append(dt)
            nts.append(Nt)
            vf0 = read_cons_var(run_dir, 0, 1, num_ranks, expected_size=N_SPATIAL)
            vfT = read_cons_var(run_dir, Nt, 1, num_ranks, expected_size=N_SPATIAL)
            err = l2_error(vfT, vf0, dx)
            errors.append(err)
            all_cons_errs.append(conservation_errors(run_dir, Nt, dx, CONS_VARS_1D, num_ranks, expected_size=N_SPATIAL))

    rates = [None]
    for i in range(1, len(cfls)):
        log_dt0 = math.log(dts[i - 1])
        log_dt1 = math.log(dts[i])
        rates.append((math.log(errors[i]) - math.log(errors[i - 1])) / (log_dt1 - log_dt0))

    print(f"  {'CFL':>7}  {'dt':>12}  {'Nt':>6}  {'L2 error':>14}  {'rate':>8}")
    print(f"  {'-' * 7}  {'-' * 12}  {'-' * 6}  {'-' * 14}  {'-' * 8}")
    for i, cfl in enumerate(cfls):
        r_str = f"{rates[i]:>8.2f}" if rates[i] is not None else f"{'---':>8}"
        print(f"  {cfl:>7.3f}  {dts[i]:>12.6e}  {nts[i]:>6}  {errors[i]:>14.6e}  {r_str}")

    if len(cfls) > 1:
        log_dt = np.log(np.array(dts, dtype=float))
        log_err = np.log(np.array(errors, dtype=float))
        overall, _ = np.polyfit(log_dt, log_err, 1)
        print(f"\n  Fitted rate: {overall:.2f}  (need >= {expected_order - tol:.1f})")
        rate_passed = overall >= expected_order - tol
    else:
        print("\n  (need >= 2 CFL values to compute rate)")
        rate_passed = True

    print(f"\n  Conservation (need rel. error < {CONS_TOL:.0e}):")
    cons_passed = True
    for name, _ in CONS_VARS_1D:
        max_err = max(ce[name] for ce in all_cons_errs)
        ok = max_err < CONS_TOL
        print(f"    {name:<14}: max = {max_err:.2e}  {'OK' if ok else 'FAIL'}")
        if not ok:
            cons_passed = False

    passed = rate_passed and cons_passed
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="MFC RK3 temporal order verification")
    parser.add_argument(
        "--cfls",
        type=float,
        nargs="+",
        default=None,
        help="CFL values to test (default: per-scheme values)",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["RK1/WENO5", "RK2/WENO5", "RK3/WENO5"],
        help="Schemes to test (default: all)",
    )
    parser.add_argument("--num-ranks", type=int, default=1, help="MPI ranks per simulation (default: 1)")
    args = parser.parse_args()

    results = {}
    for label, extra_args, expected_order, tol, default_cfls in SCHEMES:
        if label not in args.schemes:
            continue
        cfls = args.cfls if args.cfls is not None else default_cfls
        try:
            passed = test_scheme(label, extra_args, expected_order, tol, cfls, args.num_ranks)
        except Exception as e:
            import traceback

            print(f"  ERROR: {e}")
            traceback.print_exc()
            passed = False
        results[label] = passed

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    all_pass = True
    for label, passed in results.items():
        print(f"  {label:<14} {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
