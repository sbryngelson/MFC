#!/usr/bin/env python3
"""Write pgfplots .dat tables from the ladder CSV and the progress data: rerun after new results land, then latexmk."""
import csv, json, math, os

B = "/lustre/orion/cfd154/scratch/sbryngelson/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NP = [8, 64, 128, 512]
CODES = {"mfc1": "MFC v1", "mfc2": "MFC v2", "amrex": "AMReX CNS", "pelec": "PeleC"}
os.makedirs(OUT, exist_ok=True)

tab = {}
for path in ("mfc-simple-v2/results.csv", "mfc-simple-v3/results.csv"):
    if not os.path.exists(B + path):
        continue
    for r in csv.DictReader(open(B + path)):
        try:
            tab[(r["code"], r["mode"], int(r["np"]))] = (float(r["s_per_step"]), float(r["amr_only_s_per_step"]),
                                                       float(r["tts_s"]))
        except (ValueError, KeyError):
            pass


def get(code, mode, n, k):
    v = tab.get((code, mode, n))
    return v[k] if v else math.nan


def write(name, rows, cols):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(" ".join(cols) + "\n")
        for r in rows:
            f.write(" ".join("nan" if isinstance(x, float) and math.isnan(x) else f"{x:.6g}" for x in r) + "\n")


for mode in ("sub", "lock"):
    cols = ["np"] + list(CODES)
    t = {key: [get(c, mode, n, 0) for n in NP] for key, c in CODES.items()}
    a = {key: [get(c, mode, n, 1) for n in NP] for key, c in CODES.items()}
    write(f"step_{mode}.dat", [[n] + [t[k][i] for k in CODES] for i, n in enumerate(NP)], cols)
    write(f"eff_{mode}.dat", [[n] + [t[k][0] / t[k][i] for k in CODES] for i, n in enumerate(NP)], cols)
    write(f"amr_{mode}.dat", [[n] + [a[k][i] for k in CODES] for i, n in enumerate(NP)], cols)
    write(f"amradd_{mode}.dat", [[n] + [a[k][i] - a[k][0] for k in CODES] for i, n in enumerate(NP)], cols)

# Time to solution at 128 GPUs (largest rung complete for every code)
with open(os.path.join(OUT, "tts_np128.tex"), "w") as f:
    for key, c in (("mfc", "MFC v2"), ("pelec", "PeleC"), ("amrex", "AMReX CNS")):
        for mode in ("sub", "lock"):
            v = get(c, mode, 128, 2)
            f.write(f"\\def\\tts{key}{mode}{{{'nan' if math.isnan(v) else f'{v:.4g}'}}}\n")

# What each fix bought, np64 subcycled; later rows appear once mfc-simple-v3 lands
STEPS = [("Before fixes", "MFC v1"), ("All fixes but placement", "MFC v2 place off"), ("+ owner placement", "MFC v2"),
         ("+ subcycling driver", "MFC v3 subdrv"), ("+ swap packing", "MFC v3 subdrv+swap"),
         ("+ tile clustering", "MFC v3 +tiles")]
rows = [(lab, get(c, "sub", 64, 0)) for lab, c in STEPS]
rows = [r for r in rows if not math.isnan(r[1])]
with open(os.path.join(OUT, "fixes_np64.dat"), "w") as f:
    f.write("i,label,s\n" + "".join(f"{i},{lab},{s:.4g}\n" for i, (lab, s) in enumerate(rows)))
with open(os.path.join(OUT, "refs_np64.tex"), "w") as f:  # reference lines for the fixes chart
    f.write(f"\\def\\amrexsub{{{get('AMReX CNS', 'sub', 64, 0):.4g}}}\\def\\pelecsub{{{get('PeleC', 'sub', 64, 0):.4g}}}"
            f"\\def\\nfix{{{len(rows) - 1}}}\n")

conv = json.load(open(B + "amr-report/progress/data.json"))["conv"]["series"]
for i, s in enumerate(conv):
    write(f"conv{i}.dat", list(zip(s["x"], s["y"])), ["dt", "err"])
print("wrote", sorted(os.listdir(OUT)))
