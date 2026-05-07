#!/usr/bin/env python3
"""
Time-integration order verification for MFC's RK3 time stepper.

Uses the 1D single-fluid Euler advection problem (rho = 1 + 0.2*sin(2*pi*x),
u=1, p=1, L=1, T=1) with a fine spatial grid (N=512, WENO5) so the spatial
error (~4e-12) is negligible compared to the RK3 temporal error at the CFLs
tested here.

L2(rho(T) - rho(0)) measures total accumulated error.  By fixing N and varying
CFL (and hence dt), the spatial contribution is constant and the measured rate
reflects the time integration order.

CFL values [0.5, 0.25] keep the temporal error well above the spatial floor:
  CFL=0.50 → err ~8.3e-10 (>200x spatial floor)
  CFL=0.25 → err ~1.1e-10 (>25x spatial floor)
Pairwise rate ≈ 2.95, threshold ≥ 2.7.

Usage:
    python toolchain/mfc/test/run_temporal_order.py
    python toolchain/mfc/test/run_temporal_order.py --cfls 0.5 0.25 0.125
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
# All schemes here use RK3 (time_stepper=3 is default in MFC).
# N=512 is fixed; WENO5 keeps spatial error ~4e-12 (negligible at CFL>=0.25).
SCHEMES = [
    ("RK3/WENO5", ["--order", "5"], 3, 0.3, [0.5, 0.25]),
]

N_SPATIAL = 512  # fixed spatial resolution


def read_vf1_1d(run_dir: str, step: int, num_ranks: int = 1) -> np.ndarray:
    """Read q_cons_vf1 from all MPI ranks and concatenate into one 1D array."""
    chunks = []
    for rank in range(num_ranks):
        path = os.path.join(run_dir, "p_all", f"p{rank}", str(step), "q_cons_vf1.dat")
        with open(path, "rb") as f:
            rec_len = struct.unpack("i", f.read(4))[0]
            data = np.frombuffer(f.read(rec_len), dtype=np.float64)
            f.read(4)
        chunks.append(data.copy())
    return np.concatenate(chunks)


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

    with tempfile.TemporaryDirectory() as tmpdir:
        for cfl in cfls:
            dt, Nt, run_dir = run_case(tmpdir, cfl, extra_args, num_ranks)
            dts.append(dt)
            nts.append(Nt)
            vf0 = read_vf1_1d(run_dir, 0, num_ranks)
            vfT = read_vf1_1d(run_dir, Nt, num_ranks)
            err = l2_error(vfT, vf0, dx)
            errors.append(err)

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
        passed = overall >= expected_order - tol
    else:
        print("\n  (need >= 2 CFL values to compute rate)")
        passed = True

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
        default=["RK3/WENO5"],
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
            print(f"  ERROR: {e}")
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
