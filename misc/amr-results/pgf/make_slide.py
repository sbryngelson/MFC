#!/usr/bin/env python3
"""Slide figure: MFC vs AMReX CNS vs PeleC on the blast-lattice weak-scaling case, in metrics that do not depend on each code's
scheme cost: (a) weak-scaling efficiency (s/step at 8 GPUs over s/step at N), (b) AMR-only seconds added beyond 8 GPUs,
(c) MFC before vs after the fixes. Reads mfc-simple/results.csv (v1) and, if present, mfc-simple-v2/results.csv (v2)."""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = "/lustre/orion/cfd154/scratch/sbryngelson/"
OUT = os.path.dirname(os.path.abspath(__file__))
NP = [8, 64, 128, 512]


def load(path):
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    t = {}
    for r in rows:
        try:
            t[(r["code"], r["mode"], int(r["np"]))] = (float(r["s_per_step"]), float(r["amr_only_s_per_step"]))
        except (ValueError, KeyError):
            pass
    return t


v1 = load(B + "mfc-simple/results.csv")
v2 = load(B + "mfc-simple-v2/results.csv")
lines = [  # label, table, code, mode, style
    ("MFC", v2, "MFC v2", "sub", dict(color="#1f5fbf", marker="o", lw=2.4)),
    ("AMReX CNS", v1, "AMReX CNS", "sub", dict(color="#555555", marker="s", lw=1.8)),
    ("PeleC", v1, "PeleC", "sub", dict(color="#999999", marker="^", lw=1.8)),
]


def series(tab, code, mode, k):
    xs, ys = [], []
    for n in NP:
        if (code, mode, n) in tab:
            xs.append(n); ys.append(tab[(code, mode, n)][k])
    return xs, ys


fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for lab, tab, code, mode, st in lines:
    xs, ys = series(tab, code, mode, 0)
    if len(xs) < 2 or xs[0] != 8:
        continue
    ax[0].plot(xs, [ys[0] / y for y in ys], label=lab, **st)
    xa, ya = series(tab, code, mode, 1)
    if xa and xa[0] == 8:
        ax[1].plot(xa, [y - ya[0] for y in ya], label=lab, **st)
xs, ys = series(v2, "MFC v2", "sub", 0)
ax[2].plot(xs, ys, label="MFC", color="#1f5fbf", marker="o", lw=2.4)
for code, st in (("AMReX CNS", dict(color="#555555", marker="s")), ("PeleC", dict(color="#999999", marker="^"))):
    xs, ys = series(v1, code, "sub", 0)
    ax[2].plot(xs, ys, label=code, lw=1.2, **st)
titles = ["Weak-scaling efficiency (higher is better)", "AMR-only seconds added beyond 8 GPUs (lower is better)",
          "Seconds per coarse step"]
ylab = ["s/step at 8 GPUs / s/step at N", "seconds per coarse step", "seconds"]
for a, t, y in zip(ax, titles, ylab):
    a.set_xscale("log", base=2); a.set_xticks(NP); a.set_xticklabels([str(n) for n in NP])
    a.set_xlabel("GPUs (8M base cells each)"); a.set_ylabel(y); a.set_title(t, fontsize=11)
    a.grid(alpha=0.3); a.spines[["top", "right"]].set_visible(False)
ax[0].set_ylim(0, 1.05); ax[0].legend(frameon=False, fontsize=9)
ax[2].legend(frameon=False, fontsize=9)
fig.suptitle("Blast lattice, subcycled, one refined level: MFC vs AMReX CNS vs PeleC on Frontier", fontsize=12)
fig.text(0.5, -0.02, "MFC: WENO5 + HLLC, 5-equation single fluid; AMReX CNS and PeleC: second-order schemes. Same grid, dt, "
         "tag rule and refinement ratio. MFC refines ~1.3x more volume. 2 reps per point.", ha="center", fontsize=9, color="#444")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "slide_scaling.png"), dpi=200, bbox_inches="tight")
print("wrote", os.path.join(OUT, "slide_scaling.png"), "v2 points:", len(v2))
