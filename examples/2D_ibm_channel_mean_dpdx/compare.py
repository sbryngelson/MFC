#!/usr/bin/env python3
"""
Compare the last restart_data/ save of case.py against the Poiseuille solution;
report the IB wall force and, with mean_dpdx.dat, the bulk flux and dP/dx.

Usage: python3 compare.py <case_dir>
"""

import glob
import os
import sys

import numpy as np

MU, RHO, F, H, YC, L_X = 0.02, 1.0, 0.05, 0.1, 0.15, 0.2
NVAR = 5  # alpha_rho, mom_x, mom_y, E, alpha

d = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
rd = os.path.join(d, "restart_data")
step = lambda p: int(os.path.basename(p).split("_")[1].split(".")[0])
dats = sorted(glob.glob(os.path.join(rd, "lustre_[0-9]*.dat")), key=step)
ycb = np.fromfile(os.path.join(rd, "lustre_y_cb.dat"))
nx = np.fromfile(os.path.join(rd, "lustre_x_cb.dat")).size - 1
y = 0.5 * (ycb[:-1] + ycb[1:])
gap = np.abs(y - YC) < H


def u_of(path):
    q = np.fromfile(path).reshape((NVAR, y.size, nx))
    return (q[1] / q[0]).mean(axis=1)


u = u_of(dats[-1])
u_ana = F / (2 * MU) * (H**2 - (y - YC) ** 2)
e2 = np.sqrt(np.mean((u[gap] - u_ana[gap]) ** 2)) / u_ana[gap].max()
einf = np.abs(u[gap] - u_ana[gap]).max() / u_ana[gap].max()
drift = np.abs(u[gap] - u_of(dats[-2])[gap]).max() / u_ana[gap].max()
print(f"ny_gap={gap.sum()} t_step={step(dats[-1])} L2/umax={e2:.3e} Linf/umax={einf:.3e} " f"u_c={u[np.argmin(np.abs(y - YC))]:.5e} (ana {F * H**2 / (2 * MU):.5e}) last-save drift={drift:.1e}")

ibf = os.path.join(rd, f"ib_state_{step(dats[-1])}.dat")
if os.path.exists(ibf):
    rec = np.fromfile(ibf).reshape(2, 20)
    # wall shear f*H*L_x plus displacement force f*V (slab volume inside the domain: 0.05*L_x)
    print(f"IB x-force per wall: {rec[0, 1]:.5e}, {rec[1, 1]:.5e}  (ana {F * (H + 0.05) * L_X:.5e})")

hist = os.path.join(d, "mean_dpdx.dat")
if os.path.exists(hist):
    h = np.loadtxt(hist)
    print(f"final t={h[-1, 1]:.4f} Q_x={h[-1, 2]:.6e} dP/dx={h[-1, 5]:.6e}")
