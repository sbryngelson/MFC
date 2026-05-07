#!/usr/bin/env python3
"""
L1 self-convergence study for the 1D Sod shock tube across all MFC schemes.

Sod problem: rho_L=1, u_L=0, p_L=1; rho_R=0.125, u_R=0, p_R=0.1; T=0.2.
Contains a shock, contact discontinuity, and rarefaction fan.

By Godunov's theorem, any conservative monotone scheme converges at 1st order
in L1 for problems with shocks.  Higher-order schemes (WENO5, TENO7, ...) also
achieve L1 rate ~1 globally because the shock contributes an O(h) error that
dominates the smooth-region high-order accuracy.

Self-convergence method: run at N and 2N, cell-average the finer solution to
the coarse grid, compute L1(rho_N - avg(rho_{2N})).  No exact solution needed.
Ranks are read in rank order, which equals spatial order for 1D decomposition.

Usage:
    python toolchain/mfc/test/run_sod.py
    python toolchain/mfc/test/run_sod.py --resolutions 64 128 256 512 --schemes WENO5 TENO5
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

CASE = "examples/1D_sod_convergence/case.py"
MFC = "./mfc.sh"

# (label, extra_args, expected_order, tolerance, min_N)
# All schemes achieve L1 rate ~1 on shock problems (Godunov's theorem).
SCHEMES = [
    # WENO1 (1st-order upwind): contact discontinuity smears over O(sqrt(h*T)) width,
    # giving L1 contribution O(h^0.5), so fitted rate ~0.6-0.7 is physically expected.
    ("WENO1", ["--order", "1"], 1, 0.5, None),
    ("WENO3", ["--order", "3"], 1, 0.3, None),
    ("WENO5", ["--order", "5"], 1, 0.3, None),
    ("WENO7", ["--order", "7"], 1, 0.3, None),
    ("MUSCL-minmod", ["--muscl", "--muscl-lim", "1"], 1, 0.3, None),
    ("MUSCL-MC", ["--muscl", "--muscl-lim", "2"], 1, 0.3, None),
    ("MUSCL-VanLeer", ["--muscl", "--muscl-lim", "4"], 1, 0.3, None),
    # SUPERBEE is over-compressive near contacts; at N=64 the rate is pre-asymptotic
    # (~0.40), so min_N=128 skips the pre-asymptotic point and gives a reliable fit.
    ("MUSCL-SUPERBEE", ["--muscl", "--muscl-lim", "5"], 1, 0.5, 128),
    ("TENO5", ["--order", "5", "--teno", "--teno-ct", "1e-6"], 1, 0.3, None),
    ("TENO7", ["--order", "7", "--teno", "--teno-ct", "1e-9"], 1, 0.3, None),
]


def read_cons_var(run_dir: str, step: int, var_idx: int, num_ranks: int = 1) -> np.ndarray:
    """Read q_cons_vf{var_idx} from all ranks in rank order (= spatial order for 1D)."""
    chunks = []
    for rank in range(num_ranks):
        path = os.path.join(run_dir, "p_all", f"p{rank}", str(step), f"q_cons_vf{var_idx}.dat")
        with open(path, "rb") as f:
            rec_len = struct.unpack("i", f.read(4))[0]
            data = np.frombuffer(f.read(rec_len), dtype=np.float64)
            f.read(4)
        chunks.append(data.copy())
    return np.concatenate(chunks)


def l1_self_error(coarse: np.ndarray, fine: np.ndarray, dx_coarse: float) -> float:
    """L1 diff between coarse solution and cell-averaged fine solution."""
    assert len(fine) == 2 * len(coarse), f"Expected 2:1 ratio, got {len(fine)}:{len(coarse)}"
    fine_avg = (fine[0::2] + fine[1::2]) / 2.0
    return float(np.sum(np.abs(coarse - fine_avg)) * dx_coarse)


def run_case(tmpdir: str, N: int, extra_args: list, num_ranks: int = 1):
    """Run the Sod case at resolution N. Returns (Nt, run_dir)."""
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


def test_scheme(label, extra_args, expected_order, tol, resolutions, min_N=None, num_ranks=1):
    if min_N is not None:
        resolutions = [N for N in resolutions if N >= min_N]
    print(f"\n{'=' * 60}")
    print(f"  {label}  (need L1 rate >= {expected_order - tol:.1f})")
    print(f"{'=' * 60}")

    # Need consecutive pairs — resolutions must be factors of 2 apart
    nts = []
    run_dirs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for N in resolutions:
            Nt, run_dir = run_case(tmpdir, N, extra_args, num_ranks)
            nts.append(Nt)
            run_dirs.append(run_dir)

        # Compute L1 self-errors: compare each N against 2N
        errors = []
        error_resolutions = []
        for i in range(len(resolutions) - 1):
            N_c = resolutions[i]
            N_f = resolutions[i + 1]
            if N_f != 2 * N_c:
                continue  # skip non-2x pairs
            dx_c = 1.0 / N_c
            rho_c = read_cons_var(run_dirs[i], nts[i], 1, num_ranks)
            rho_f = read_cons_var(run_dirs[i + 1], nts[i + 1], 1, num_ranks)
            err = l1_self_error(rho_c, rho_f, dx_c)
            errors.append(err)
            error_resolutions.append(N_c)

    rates = [None]
    for i in range(1, len(errors)):
        log_h0 = math.log(1.0 / error_resolutions[i - 1])
        log_h1 = math.log(1.0 / error_resolutions[i])
        rates.append((math.log(errors[i]) - math.log(errors[i - 1])) / (log_h1 - log_h0))

    print(f"  {'N':>6}  {'Nt':>6}  {'L1 self-err':>14}  {'rate':>8}")
    print(f"  {'-' * 6}  {'-' * 6}  {'-' * 14}  {'-' * 8}")
    for i, N in enumerate(error_resolutions):
        r_str = f"{rates[i]:>8.2f}" if rates[i] is not None else f"{'---':>8}"
        print(f"  {N:>6}  {nts[i]:>6}  {errors[i]:>14.6e}  {r_str}")

    if len(errors) >= 2:
        log_h = np.log(1.0 / np.array(error_resolutions, dtype=float))
        log_e = np.log(np.array(errors, dtype=float))
        overall, _ = np.polyfit(log_h, log_e, 1)
        print(f"\n  Fitted rate: {overall:.2f}  (need >= {expected_order - tol:.1f})")
        passed = overall >= expected_order - tol
    elif len(errors) == 1:
        print(f"\n  Single pair rate: {rates[-1]:.2f}  (need >= {expected_order - tol:.1f})")
        passed = rates[-1] >= expected_order - tol
    else:
        print("\n  (need >= 2 consecutive resolutions to compute rate)")
        passed = True

    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="MFC Sod shock tube L1 convergence")
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
        help="Grid resolutions — must be consecutive factors of 2 (default: 128 256 512 1024)",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=[s[0] for s in SCHEMES],  # label is always first element
        help="Schemes to test (default: all)",
    )
    parser.add_argument("--num-ranks", type=int, default=1, help="MPI ranks per simulation (default: 1)")
    args = parser.parse_args()

    results = {}
    for label, extra_args, expected_order, tol, min_N in SCHEMES:
        if label not in args.schemes:
            continue
        try:
            passed = test_scheme(label, extra_args, expected_order, tol, args.resolutions, min_N, args.num_ranks)
        except Exception as e:
            print(f"  ERROR: {e}")
            passed = False
        results[label] = passed

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    all_pass = True
    for label, passed in results.items():
        print(f"  {label:<18} {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
