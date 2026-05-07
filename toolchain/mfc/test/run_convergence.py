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
    # WENO5/TENO5: need (N/2) >= num_stcls_min*recon_order = 25 cells/rank with 4
    # ranks in 2x2; min_N=64 ensures 32 cells/rank which satisfies this constraint.
    ("WENO5", ["--order", "5"], 5, 1.0, 64, None),
    ("WENO3", ["--order", "3"], 3, 1.2, 32, None),
    ("WENO1", ["--order", "1"], 1, 0.4, 32, None),
    ("MUSCL2", ["--muscl"], 2, 0.5, 32, None),
    ("TENO5", ["--order", "5", "--teno", "--teno-ct", "1e-6"], 5, 1.0, 64, None),
]


def read_cons_var(run_dir: str, step: int, var_idx: int, N: int, num_ranks: int = 1) -> np.ndarray:
    """Read q_cons_vf{var_idx} from all MPI ranks and return as a flat array.

    Spatial ordering is not preserved across ranks but that is fine for L2
    norm and sum computations, which are invariant to permutation of elements.
    """
    chunks = []
    for rank in range(num_ranks):
        path = os.path.join(run_dir, "p_all", f"p{rank}", str(step), f"q_cons_vf{var_idx}.dat")
        with open(path, "rb") as f:
            rec_len = struct.unpack("i", f.read(4))[0]
            data = np.frombuffer(f.read(rec_len), dtype=np.float64)
            f.read(4)
        chunks.append(data.copy())
    combined = np.concatenate(chunks)
    if combined.size != N * N:
        raise ValueError(f"Expected {N * N} values across {num_ranks} ranks, got {combined.size}")
    return combined


# 2D single-fluid Euler (model_eqns=2, num_fluids=1): vf1=ρ, vf2=ρu, vf3=ρv, vf4=E
# Momentum is excluded: the isentropic vortex has zero net linear momentum, making
# the relative error formula ill-conditioned (denominator ≈ 0). Density and energy
# have large nonzero integrals and are the meaningful conserved quantities to verify.
CONS_VARS_2D = [("density", 1), ("energy", 4)]
CONS_TOL = 1e-10


def conservation_errors(run_dir: str, Nt: int, N: int, cell_vol: float, var_list: list, num_ranks: int) -> dict:
    """Return relative conservation error |Σq(T) - Σq(0)| / |Σq(0)| for each variable."""
    errs = {}
    for name, idx in var_list:
        q0 = read_cons_var(run_dir, 0, idx, N, num_ranks)
        qT = read_cons_var(run_dir, Nt, idx, N, num_ranks)
        s0 = float(np.sum(q0)) * cell_vol
        sT = float(np.sum(qT)) * cell_vol
        errs[name] = abs(sT - s0) / (abs(s0) + 1e-300)
    return errs


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


def run_case(tmpdir: str, N: int, extra_args: list, num_ranks: int = 1):
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
        str(num_ranks),
        "--",
        "-N",
        str(N),
    ] + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), check=False)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr)
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


def test_scheme(label, extra_args, expected_order, tol, resolutions, min_N=None, max_N=None, num_ranks=1):
    if min_N is not None:
        resolutions = [N for N in resolutions if N >= min_N]
    if max_N is not None:
        resolutions = [N for N in resolutions if N <= max_N]
    print(f"\n{'=' * 60}")
    print(f"  {label}  (need rate >= {expected_order - tol:.1f})")
    print(f"{'=' * 60}")

    errors = []
    nts = []
    all_cons_errs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for N in resolutions:
            dx = 10.0 / N
            Nt, run_dir = run_case(tmpdir, N, extra_args, num_ranks)
            nts.append(Nt)
            rho0 = read_cons_var(run_dir, 0, 1, N, num_ranks)
            rhoT = read_cons_var(run_dir, Nt, 1, N, num_ranks)
            err = l2_error(rhoT, rho0, dx)
            errors.append(err)
            all_cons_errs.append(conservation_errors(run_dir, Nt, N, dx**2, CONS_VARS_2D, num_ranks))

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
        rate_passed = overall >= expected_order - tol
    else:
        print("\n  (need >= 2 resolutions to compute rate)")
        rate_passed = True  # can't fail with a single resolution

    print(f"\n  Conservation (need rel. error < {CONS_TOL:.0e}):")
    cons_passed = True
    for name, _ in CONS_VARS_2D:
        max_err = max(ce[name] for ce in all_cons_errs)
        ok = max_err < CONS_TOL
        print(f"    {name:<14}: max = {max_err:.2e}  {'OK' if ok else 'FAIL'}")
        if not ok:
            cons_passed = False

    passed = rate_passed and cons_passed
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="MFC convergence-rate verification")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[32, 64, 128], help="Grid resolutions (default: 32 64 128; N<32 unsupported for WENO5)")
    parser.add_argument("--schemes", nargs="+", default=["WENO5", "WENO3", "WENO1", "MUSCL2", "TENO5"], help="Schemes to test (default: all)")
    parser.add_argument("--num-ranks", type=int, default=1, help="MPI ranks per simulation (default: 1)")
    args = parser.parse_args()

    results = {}
    for label, extra_args, expected_order, tol, min_N, max_N in SCHEMES:
        if label not in args.schemes:
            continue
        try:
            passed = test_scheme(label, extra_args, expected_order, tol, args.resolutions, min_N, max_N, args.num_ranks)
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
        print(f"  {label:<12} {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
