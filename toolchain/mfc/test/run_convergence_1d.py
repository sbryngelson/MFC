#!/usr/bin/env python3
"""
Convergence-rate verification for MFC's 1D single-fluid Euler equations.

Single fluid with a density sine wave: rho = 1 + 0.2*sin(2*pi*x), u=1, p=1.
After exactly one period (T=1, u=1, L=1), the exact solution equals the IC.
L2(rho(T) - rho(0)) measures the accumulated scheme spatial truncation error.
No non-conservative alpha equation — clean benchmark for all schemes.

WENO5/TENO5 use CFL=0.02: RK3 temporal error O(dt^3) is then negligible
relative to the O(h^5) spatial error at N=128-512.

WENO7/TENO7 use CFL=0.005: at CFL=0.02 the RK3 temporal error (~3.4e-12 at
N=128) is comparable to the spatial error (~4.4e-12), giving a spurious rate
of ~3.7.  With CFL=0.005 the temporal error drops by (0.005/0.02)^3 = 1/64
to ~5.3e-14, well below spatial, and the measured rate approaches 7.
N is capped at 256 — the machine-precision floor is reached near N=512.

WENO3-JS degrades to 2nd order at smooth extrema (Henrick et al. 2005).
The expected rate for WENO3 here is therefore 2, not 3; the 2D isentropic
vortex test (run_convergence.py) verifies WENO3 rate 3.

MUSCL2 uses muscl_lim=0 (unlimited central-difference) by default.  TVD
limiters clip slopes to zero at smooth extrema and stall at 1st order on the
sine wave; the unlimited limiter preserves 2nd-order convergence everywhere.

Usage:
    python toolchain/mfc/test/run_convergence_1d.py [--resolutions 128 256 512 1024]
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

# (label, extra_args, expected_order, tolerance, min_N, max_N)
# CFL is baked into each scheme's extra_args so that WENO7/TENO7 can use a
# smaller CFL independently of all other schemes.
#
# Per-scheme resolution bounds let each scheme run over the range where its
# asymptotic order is cleanly visible:
#   WENO5  : cap at N=512 — double-precision floor kills the rate at N=1024
#            (error ~2.6e-12, rate collapses to 0.69); [128,512] gives 4.99.
#   WENO3  : start at N=256 — skips the coarsest pre-asymptotic points;
#            WENO3-JS degrades to 2nd order at smooth extrema (Henrick 2005),
#            asymptote confirmed 1.99 at N=4096; [256,1024] gives ~1.87.
#   WENO1  : full range [128,1024]; rate 0.97.
#   MUSCL2 : full range [128,1024]; unlimited slope, rate exactly 2.00.
#   TENO5  : same range as WENO5; CT=1e-6; rate matches WENO5 on smooth problems.
#   WENO7  : CFL=0.005, range [64,128] — at N=256 the spatial error (~1.7e-14)
#            falls below the round-off accumulation floor (~2.5e-12 for ~28M
#            cell-steps), so only N=64 and N=128 give a clean rate ≥6.5.
#   TENO7  : same range and CFL as WENO7; CT=1e-9.
SCHEMES = [
    ("WENO5", ["--order", "5", "--cfl", "0.02"], 5, 0.2, 128, 512),
    ("WENO3", ["--order", "3", "--cfl", "0.02"], 2, 0.2, 256, None),
    ("WENO1", ["--order", "1", "--cfl", "0.02"], 1, 0.05, 128, None),
    ("MUSCL2", ["--muscl", "--cfl", "0.02"], 2, 0.1, 128, None),
    ("TENO5", ["--order", "5", "--teno", "--teno-ct", "1e-6", "--cfl", "0.02"], 5, 0.2, 128, 512),
    ("WENO7", ["--order", "7", "--cfl", "0.005"], 7, 0.5, 64, 128),
    ("TENO7", ["--order", "7", "--teno", "--teno-ct", "1e-9", "--cfl", "0.005"], 7, 0.5, 64, 128),
]


def read_cons_var(run_dir: str, step: int, var_idx: int, num_ranks: int = 1) -> np.ndarray:
    """Read q_cons_vf{var_idx} from all MPI ranks and concatenate into one 1D array."""
    chunks = []
    for rank in range(num_ranks):
        path = os.path.join(run_dir, "p_all", f"p{rank}", str(step), f"q_cons_vf{var_idx}.dat")
        with open(path, "rb") as f:
            rec_len = struct.unpack("i", f.read(4))[0]
            data = np.frombuffer(f.read(rec_len), dtype=np.float64)
            f.read(4)
        chunks.append(data.copy())
    return np.concatenate(chunks)


# 1D single-fluid Euler (model_eqns=2, num_fluids=1): vf1=ρ, vf2=ρu, vf3=E
CONS_VARS_1D = [("density", 1), ("x-momentum", 2), ("energy", 3)]
CONS_TOL = 1e-10


def conservation_errors(run_dir: str, Nt: int, cell_vol: float, var_list: list, num_ranks: int) -> dict:
    """Return relative conservation error |Σq(T) - Σq(0)| / |Σq(0)| for each variable."""
    errs = {}
    for name, idx in var_list:
        q0 = read_cons_var(run_dir, 0, idx, num_ranks)
        qT = read_cons_var(run_dir, Nt, idx, num_ranks)
        s0 = float(np.sum(q0)) * cell_vol
        sT = float(np.sum(qT)) * cell_vol
        errs[name] = abs(sT - s0) / (abs(s0) + 1e-300)
    return errs


def l2_error(a: np.ndarray, b: np.ndarray, dx: float) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2) * dx))


def convergence_rate(errors: list, resolutions: list) -> float:
    log_dx = np.log(1.0 / np.array(resolutions, dtype=float))
    log_err = np.log(np.array(errors, dtype=float))
    rate, _ = np.polyfit(log_dx, log_err, 1)
    return float(rate)


def run_case(tmpdir: str, N: int, extra_args: list, num_ranks: int = 1):
    """Run the 1D advection case at resolution N. Returns (Nt, run_dir)."""
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
        print(result.stdout[-3000:])
        raise RuntimeError(f"./mfc.sh run failed for N={N}")

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
            dx = 1.0 / N
            Nt, run_dir = run_case(tmpdir, N, extra_args, num_ranks)
            nts.append(Nt)
            vf0 = read_cons_var(run_dir, 0, 1, num_ranks)
            vfT = read_cons_var(run_dir, Nt, 1, num_ranks)
            err = l2_error(vfT, vf0, dx)
            errors.append(err)
            all_cons_errs.append(conservation_errors(run_dir, Nt, dx, CONS_VARS_1D, num_ranks))
            print(f"  N={N}: Nt={Nt}, |vf0|={len(vf0)}, err={err:.4e}")

    rates = [None]
    for i in range(1, len(resolutions)):
        log_dx0 = math.log(1.0 / resolutions[i - 1])
        log_dx1 = math.log(1.0 / resolutions[i])
        rates.append((math.log(errors[i]) - math.log(errors[i - 1])) / (log_dx1 - log_dx0))

    print()
    print(f"  {'N':>6}  {'Nt':>6}  {'dx':>10}  {'L2 error':>14}  {'rate':>8}")
    print(f"  {'-' * 6}  {'-' * 6}  {'-' * 10}  {'-' * 14}  {'-' * 8}")
    for i, N in enumerate(resolutions):
        r_str = f"{rates[i]:>8.2f}" if rates[i] is not None else f"{'---':>8}"
        print(f"  {N:>6}  {nts[i]:>6}  {1.0 / N:>10.6f}  {errors[i]:>14.6e}  {r_str}")

    if len(resolutions) > 1:
        overall = convergence_rate(errors, resolutions)
        print(f"\n  Fitted rate: {overall:.2f}  (need >= {expected_order - tol:.1f})")
        rate_passed = overall >= expected_order - tol
    else:
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
    parser = argparse.ArgumentParser(description="MFC 1D advection convergence-rate verification")
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
        help="Grid resolutions to test (default: 64 128 256 512 1024)",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["WENO5", "WENO3", "WENO1", "MUSCL2", "TENO5", "WENO7", "TENO7"],
        help="Schemes to test",
    )
    parser.add_argument("--muscl-lim", type=int, default=0, help="MUSCL limiter (0=unlimited 1=minmod ...; default: 0)")
    parser.add_argument("--num-ranks", type=int, default=1, help="MPI ranks per simulation (default: 1)")
    args = parser.parse_args()

    muscl_extra = ["--muscl-lim", str(args.muscl_lim)]

    results = {}
    for label, extra_args, expected_order, tol, min_N, max_N in SCHEMES:
        if label not in args.schemes:
            continue
        try:
            passed = test_scheme(label, extra_args + muscl_extra, expected_order, tol, args.resolutions, min_N, max_N, args.num_ranks)
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
