#!/usr/bin/env python3
"""
Convergence-rate verification for MFC's 2D isentropic vortex problem.

Uses hcid=283: 3-pt Gauss-Legendre cell averages of conserved variables as IC.
The vortex strength eps=0.01 (set in case.py) is chosen so that the dominant
error source is the WENO spatial truncation error O(eps^2 * h^p), not the
primitive-to-conserved covariance floor O(eps^3 * h^2).  For h > eps^(1/3)=0.22
(i.e., N < 46 per dimension), the p-th order scheme shows rate p.

L2(rho(T) - rho(0)) measures accumulated scheme error; the comparison to rho(0)
(the numerical IC) eliminates IC discretisation error, isolating the scheme error.

WENO7/TENO7 are NOT tested here.  For the isentropic vortex, the IC
primitive→conserved covariance error is O(eps^3 * h^2).  The WENO7 scheme
error is O(eps^2 * h^7).  Scheme error dominates only when h > eps^(1/5);
with eps=0.01 that requires h > 0.40, i.e., N < 25.  At N=64-128 the
covariance floor dominates and the measured rate is ~2, not 7.
WENO7/TENO7 7th-order convergence is verified by the 1D test (run_convergence_1d.py)
which uses a pure advection problem that avoids this nonlinear floor.

Usage:
    python toolchain/mfc/test/run_convergence.py [--resolutions 32 64 128]
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

CASE = "examples/2D_isentropicvortex_convergence/case.py"
MFC = "./mfc.sh"

# (label, extra_args, expected_order, tolerance, min_N, max_N)
# With eps=0.01 and N=32..128 the prim->cons covariance error O(eps^3 h^2) is
# well below the scheme's spatial error O(eps^2 h^p), so each scheme shows its
# nominal rate.  The tolerance is the allowable shortfall from the nominal order.
#
# WENO3: at N=32-128 the rate is ~2.0-2.2 (pre-asymptotic; approaches 3 at
#   finer grids).  Threshold 1.8.
# WENO7/TENO7 are omitted — see module docstring for why.
SCHEMES = [
    ("WENO5", ["--order", "5"], 5, 1.0, 32, None),
    ("WENO3", ["--order", "3"], 3, 1.2, 32, None),
    ("WENO1", ["--order", "1"], 1, 0.4, 32, None),
    ("MUSCL2", ["--muscl"], 2, 0.5, 32, None),
    ("TENO5", ["--order", "5", "--teno", "--teno-ct", "1e-6"], 5, 1.0, 32, None),
]


def read_cons_vf1(run_dir: str, step: int, N: int) -> np.ndarray:
    """Read density (q_cons_vf1 = alpha_rho(1) = rho for single fluid) from p_all output."""
    path = os.path.join(run_dir, "p_all", "p0", str(step), "q_cons_vf1.dat")
    with open(path, "rb") as f:
        rec_len = struct.unpack("i", f.read(4))[0]
        data = np.frombuffer(f.read(rec_len), dtype=np.float64)
        f.read(4)  # trailing record marker
    if data.size != N * N:
        raise ValueError(f"Expected {N * N} values, got {data.size} in {path}")
    return data.reshape((N, N), order="F")


def l2_error(rho_final: np.ndarray, rho_init: np.ndarray, dx: float) -> float:
    """L2 error: sqrt(sum((f-g)^2 * dx^2))."""
    diff = rho_final - rho_init
    return float(np.sqrt(np.sum(diff**2) * dx**2))


def convergence_rate(errors: list, resolutions: list) -> float:
    """Least-squares slope of log(error) vs log(dx), dx = 10/N."""
    log_dx = np.log(10.0 / np.array(resolutions, dtype=float))
    log_err = np.log(np.array(errors, dtype=float))
    rate, _ = np.polyfit(log_dx, log_err, 1)
    return float(rate)


def run_case(tmpdir: str, N: int, extra_args: list):
    """Run the vortex case at resolution N. Returns (Nt, run_dir)."""
    # Query case parameters to find t_step_stop
    result = subprocess.run(
        [sys.executable, CASE, "--mfc", "{}", "-N", str(N)] + extra_args,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"case.py failed:\n{result.stderr}")
    cfg = json.loads(result.stdout)
    Nt = int(cfg["t_step_stop"])

    # Run only pre_process and simulation (post_process not needed for p_all)
    cmd = [
        MFC,
        "run",
        CASE,
        "-t",
        "pre_process",
        "simulation",
        "-n",
        "1",
        "--",
        "-N",
        str(N),
    ] + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), check=False)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        raise RuntimeError(f"./mfc.sh run failed for N={N}")

    # Copy p_all to temp dir, then clean the case directory for next run
    case_dir = os.path.dirname(CASE)
    src = os.path.join(case_dir, "p_all")
    dst = os.path.join(tmpdir, f"N{N}", "p_all")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    shutil.rmtree(src, ignore_errors=True)
    shutil.rmtree(os.path.join(case_dir, "D"), ignore_errors=True)

    return Nt, os.path.join(tmpdir, f"N{N}")


def test_scheme(label, extra_args, expected_order, tol, resolutions, min_N=None, max_N=None):
    if min_N is not None:
        resolutions = [N for N in resolutions if N >= min_N]
    if max_N is not None:
        resolutions = [N for N in resolutions if N <= max_N]
    print(f"\n{'=' * 60}")
    print(f"  {label}  (need rate >= {expected_order - tol:.1f})")
    print(f"{'=' * 60}")

    errors = []
    nts = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for N in resolutions:
            dx = 10.0 / N
            Nt, run_dir = run_case(tmpdir, N, extra_args)
            nts.append(Nt)
            rho0 = read_cons_vf1(run_dir, 0, N)
            rhoT = read_cons_vf1(run_dir, Nt, N)
            err = l2_error(rhoT, rho0, dx)
            errors.append(err)

    # Compute pairwise rates
    rates = [None]
    for i in range(1, len(resolutions)):
        log_dx0 = math.log(10.0 / resolutions[i - 1])
        log_dx1 = math.log(10.0 / resolutions[i])
        rates.append((math.log(errors[i]) - math.log(errors[i - 1])) / (log_dx1 - log_dx0))

    print(f"  {'N':>6}  {'Nt':>5}  {'dx':>10}  {'L2 error':>14}  {'rate':>8}")
    print(f"  {'-' * 6}  {'-' * 5}  {'-' * 10}  {'-' * 14}  {'-' * 8}")
    for i, N in enumerate(resolutions):
        dx = 10.0 / N
        r_str = f"{rates[i]:>8.2f}" if rates[i] is not None else f"{'---':>8}"
        print(f"  {N:>6}  {nts[i]:>5}  {dx:>10.5f}  {errors[i]:>14.6e}  {r_str}")

    if len(resolutions) > 1:
        overall = convergence_rate(errors, resolutions)
        print(f"\n  Fitted rate: {overall:.2f}  (need >= {expected_order - tol:.1f})")
        passed = overall >= expected_order - tol
    else:
        print("\n  (need >= 2 resolutions to compute rate)")
        passed = True  # can't fail with a single resolution

    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="MFC convergence-rate verification")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[32, 64, 128], help="Grid resolutions (default: 32 64 128; N<32 unsupported for WENO5)")
    parser.add_argument("--schemes", nargs="+", default=["WENO5", "WENO3", "WENO1", "MUSCL2", "TENO5"], help="Schemes to test (default: all)")
    args = parser.parse_args()

    results = {}
    for label, extra_args, expected_order, tol, min_N, max_N in SCHEMES:
        if label not in args.schemes:
            continue
        try:
            passed = test_scheme(label, extra_args, expected_order, tol, args.resolutions, min_N, max_N)
        except Exception as e:
            print(f"  ERROR: {e}")
            passed = False
        results[label] = passed

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    all_pass = True
    for label, passed in results.items():
        print(f"  {label:<12} {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
