#!/usr/bin/env python3
"""
Convergence-rate verification for WENO5 on a 2D axisymmetric (cyl_coord=T) grid.

Density sine wave in z: rho = 1 + 0.2*sin(2*pi*z), u_z=1, p=1, u_r=0.
Exact solution at time T: rho_exact(z,T) = 1 + 0.2*sin(2*pi*(z-T)).
Nr is held fixed; Nz is refined.
L2 error is computed by averaging density over r then comparing to exact solution.
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

CASE = "examples/2D_axisym_convergence/case.py"
MFC = "./mfc.sh"
NR = 32  # fixed radial cells (n=31; WENO5 needs n+1 >= num_stcls_min*5 = 25)
TFINAL = 0.1  # short final time avoids long-time cylindrical instability


def read_field(run_dir: str, step: int, var_idx: int, nz: int, nr: int) -> np.ndarray:
    """Read 2D field. In MFC: x=axial(Nz), y=radial(Nr). Fortran col-major -> shape (Nz, Nr)."""
    path = os.path.join(run_dir, "p_all", "p0", str(step), f"q_cons_vf{var_idx}.dat")
    with open(path, "rb") as f:
        rec_len = struct.unpack("i", f.read(4))[0]
        data = np.frombuffer(f.read(rec_len), dtype=np.float64).copy()
        f.read(4)
    if data.size != nr * nz:
        raise ValueError(f"Expected {nr * nz} values, got {data.size}")
    # Fortran stores (x, y) = (axial, radial) in column-major: first index varies fastest
    return data.reshape(nz, nr, order="F")


def run_case(tmpdir: str, nz: int, extra_args: list) -> tuple:
    """Run the 2D axisym case at resolution nz. Returns (Nt, run_dir)."""
    result = subprocess.run(
        [sys.executable, CASE, "--mfc", "{}", "-N", str(nz), "--nr", str(NR), "--Tfinal", str(TFINAL)] + extra_args,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"case.py failed:\n{result.stderr}")
    cfg = json.loads(result.stdout)
    Nt = int(cfg["t_step_stop"])
    dt = float(cfg["dt"])

    cmd = [MFC, "run", CASE, "-t", "pre_process", "simulation", "-n", "1", "--", "-N", str(nz), "--nr", str(NR), "--Tfinal", str(TFINAL)] + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), check=False)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr)
        raise RuntimeError(f"./mfc.sh run failed for Nz={nz}")

    case_dir = os.path.dirname(CASE)
    src = os.path.join(case_dir, "p_all")
    dst = os.path.join(tmpdir, f"N{nz}", "p_all")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    shutil.rmtree(src, ignore_errors=True)
    shutil.rmtree(os.path.join(case_dir, "D"), ignore_errors=True)
    return Nt, dt, os.path.join(tmpdir, f"N{nz}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--cfl", type=float, default=0.02)
    parser.add_argument("--order", type=int, default=5)
    args = parser.parse_args()

    extra = ["--cfl", str(args.cfl), "--order", str(args.order)]

    print(f"\n{'=' * 60}")
    print(f"  WENO{args.order} on 2D axisymmetric (cyl_coord=T) grid")
    print(f"  Sine wave in z, Nr={NR} fixed, Nz refined, T={TFINAL}")
    print(f"{'=' * 60}")

    errors, nts = [], []
    with tempfile.TemporaryDirectory() as tmpdir:
        for nz in args.resolutions:
            Lx = 5.0  # must match case.py
            dz = Lx / nz
            Nt, dt, run_dir = run_case(tmpdir, nz, extra)
            nts.append(Nt)
            T_actual = Nt * dt  # actual final time

            # Read final density field (Nz x Nr), average over r
            rhoT = read_field(run_dir, Nt, 1, nz, NR)
            rhoT_z = rhoT.mean(axis=1)  # shape (nz,)

            # Exact solution: rho(z, T) = 1 + 0.2*sin(2*pi*(z - T))
            x_cc = (np.arange(nz) + 0.5) * dz
            rho_exact = 1.0 + 0.2 * np.sin(2.0 * np.pi * (x_cc - T_actual))

            err = float(np.sqrt(np.sum((rhoT_z - rho_exact) ** 2) * dz))
            errors.append(err)
            print(f"  Nz={nz:4d}: Nt={Nt}, err={err:.4e}")

    Lx = 5.0
    rates = [None]
    for i in range(1, len(args.resolutions)):
        nz0, nz1 = args.resolutions[i - 1], args.resolutions[i]
        rates.append((math.log(errors[i]) - math.log(errors[i - 1])) / (math.log(Lx / nz1) - math.log(Lx / nz0)))

    Lx = 5.0
    print(f"\n  {'Nz':>6}  {'Nt':>6}  {'dz':>10}  {'L2 error':>14}  {'rate':>8}")
    print(f"  {'-' * 6}  {'-' * 6}  {'-' * 10}  {'-' * 14}  {'-' * 8}")
    for i, nz in enumerate(args.resolutions):
        r = f"{rates[i]:>8.2f}" if rates[i] is not None else f"{'---':>8}"
        print(f"  {nz:>6}  {nts[i]:>6}  {Lx / nz:>10.6f}  {errors[i]:>14.6e}  {r}")

    if len(args.resolutions) > 1:
        log_dz = np.log(Lx / np.array(args.resolutions, dtype=float))
        log_err = np.log(np.array(errors, dtype=float))
        rate, _ = np.polyfit(log_dz, log_err, 1)
        expected = args.order - 0.2
        passed = rate >= expected
        print(f"\n  Fitted rate: {rate:.2f}  (need >= {expected:.1f})")
        print(f"  {'PASS' if passed else 'FAIL'}")
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
