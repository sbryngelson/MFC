#!/usr/bin/env python3
"""MFC simple-config blast lattice, v3 variants (this dir, see VERSIONS/gen.py) beside v2 (../mfc-simple-v2) and AMReX CNS / PeleC. Writes results.csv and scaling.png, prints the table.
MFC: s/step = (w240 - w40)/200 from "[phase] step-loop wall"; time to solution = w240 (t = 0.048 lock, 0.096 sub);
AMR-only = wall - (coarse + rhs) phase means; RHS imbalance = phase-table imbalance column (240-step run); coverage = regrid at step 220."""
import csv, glob, os, re, statistics as st, sys

sys.dont_write_bytecode = True
B = os.path.dirname(os.path.abspath(__file__))
S = "/lustre/orion/cfd154/scratch/sbryngelson"
FM = f"{S}/amr-report/fair-metrics"
REF_DIRS = {("AMReX CNS", "nosub"): f"{S}/amrex-bench/runs_blast/np{{}}-L-nosub-j*", ("AMReX CNS", "sub"): f"{S}/amrex-bench/runs_blast/np{{}}-L-sub-j*",
            ("PeleC", "sub"): f"{S}/pelec-bench/runs/np{{}}-L-j*"}
NPS = (8, 64, 128, 512, 1024)
MFC = [("MFC v2", f"{S}/mfc-simple-v2", ""), ("MFC v3 ctl e341896a", B, "_ctl"), ("MFC v3 subdrv", B, "_sd"), ("MFC v3 subdrv+swap", B, "_sw"),
       ("MFC v3 +tiles", B, "_tiles"), ("MFC v3 +tiles mgs32", B, "_tilesg32")]  # (label, run dir, run-name suffix)
FAILED = {"_tiles": "failed: device out of memory at the step-140 regrid (see README)"}  # suffix -> source note for runs without a wall
TPHYS = {"lock": 0.048, "sub": 0.096}
FIELDS = ["code", "mode", "np", "window", "reps", "s_per_step", "s_per_step_sd", "tts_s", "tts_sd", "amr_only_s_per_step", "amr_only_tts_s",
          "rhs_imb", "coarse_imb", "cov_pct", "t_phys", "source"]


def mfc_run(d):
    p = os.path.join(d, "MFC.out")
    if not os.path.exists(p): return None
    t = open(p, errors="replace").read()
    w = re.search(r"\[phase\] step-loop wall =\s*([\d.]+)", t)
    if not w: return None
    ph = {k: (float(a), float(b), float(c)) for k, a, b, c in re.findall(r"\[phase\] (\w+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+%\s+([\d.]+)", t)}
    cov, step = {}, None  # progress line "t_step k" precedes the regrid done before step k + 1
    for line in t.splitlines():
        s = re.search(r"\(t_step (\d+)\)", line)
        if s: step = int(s.group(1))
        c = re.search(r"\[amr-cov\].*level coverage \(share of domain\)\s*([\d.]+)", line)
        if c and step is not None: cov[step + 1] = 100 * float(c.group(1))
    solver = sum(ph.get(k, (0, 0, 0))[0] for k in ("coarse", "rhs"))
    return dict(wall=float(w.group(1)), amr=float(w.group(1)) - solver, ph=ph, cov=cov, l0=ph.get("L0", (0,))[0])


def mfc_rows(code, base, sfx):
    rows = []
    for np_ in NPS:
        for mode in ("lock", "sub"):
            reps, n2 = [], 240
            for steps in (240, 239):  # 239: fallback when the step-240 regrid runs out of device memory; TTS adds one step
                for r in (1, 2):
                    a, b = mfc_run(f"{base}/np{np_}_{mode}_s40_r{r}{sfx}"), mfc_run(f"{base}/np{np_}_{mode}_s{steps}_r{r}{sfx}")
                    if a and b: reps.append((a, b))
                if reps: n2 = steps; break
            if not reps:
                if not sfx or sfx in FAILED: rows.append(dict(code=code, mode=mode, np=np_, reps=0, source=FAILED.get(sfx, "pending")))
                continue
            n = n2 - 40
            sps = [(b["wall"] - a["wall"]) / n for a, b in reps]
            for (a, b), s in zip(reps, sps): b["wall"] += (240 - n2) * s; b["amr"] += (240 - n2) * (b["amr"] - a["amr"]) / n
            for a, b in reps:
                if a["l0"] or b["l0"]: print(f"note: {code} np{np_} {mode} has an L0 phase ({b['l0']:.1f} s), not subtracted")
            m = lambda f: st.mean(f(a, b) for a, b in reps)
            rows.append(dict(code=code, mode=mode, np=np_, reps=len(reps), s_per_step=st.mean(sps), s_per_step_sd=st.pstdev(sps),
                             tts_s=m(lambda a, b: b["wall"]), tts_sd=st.pstdev([b["wall"] for _, b in reps]),
                             amr_only_s_per_step=m(lambda a, b: (b["amr"] - a["amr"]) / 200), amr_only_tts_s=m(lambda a, b: b["amr"]),
                             window=f"40->{n2}",
                             rhs_imb=m(lambda a, b: b["ph"]["rhs"][2]), coarse_imb=m(lambda a, b: b["ph"]["coarse"][2]),
                             cov_pct=m(lambda a, b: b["cov"].get(220, float("nan"))), t_phys=TPHYS[mode],
                             source=f"{base}/np{np_}_{mode}_s{{40,{n2}}}_r{{1,2}}{sfx}"))
    return rows


def ref_rows():
    """AMReX / PeleC from fair_metrics.csv; rungs missing there (np128) recomputed with fair_metrics.amr_profiled on the same logs.
    Time to solution = mean 'Run Time advance' of the 240-step logs."""
    sys.path.insert(0, FM)
    import fair_metrics as fm
    csvrows = {(r["code"], r["mode"], int(r["np"])): r for r in csv.DictReader(open(f"{FM}/fair_metrics.csv"))
               if r["ladder"] == "L" and r["code"] in ("AMReX CNS", "PeleC") and r["status"] == "ok"}
    rows = []
    for (code, md), pat in REF_DIRS.items():
        for np_ in NPS:
            ds = sorted(glob.glob(pat.format(np_)), key=lambda d: int(d.rsplit("-j", 1)[1]))
            if not ds: continue
            d, c = ds[-1], csvrows.get((code, md, np_))
            if c and c["source"] != d: print(f"note: {code} {md} np{np_}: csv uses {c['source']}, newest is {d}")
            if c:
                d = c["source"]; v = {k: float(c[k]) for k in ("s_per_step", "s_per_step_sd", "amr_only_s_per_step", "rhs_imbalance_max_over_mean", "cov_L1_pct")}
                src = d + " (fair_metrics.csv)"
            else:
                r = fm.amr_profiled(code, d, "L", md)
                v = {k: r[k] for k in ("s_per_step", "s_per_step_sd", "amr_only_s_per_step", "rhs_imbalance_max_over_mean", "cov_L1_pct")}
                src = d + " (fair_metrics.amr_profiled)"
            w240 = [fm.tp_wall(fm.txt(f)) for f in sorted(glob.glob(d + "/run_r*_s240.log"))]
            w240 = [w for w in w240 if w is not None]
            mode = "lock" if md == "nosub" else "sub"
            rows.append(dict(code=code, mode=mode, np=np_, window="40->240", reps=len(w240), s_per_step=v["s_per_step"], s_per_step_sd=v["s_per_step_sd"],
                             tts_s=st.mean(w240), tts_sd=st.pstdev(w240), amr_only_s_per_step=v["amr_only_s_per_step"],
                             rhs_imb=v["rhs_imbalance_max_over_mean"], cov_pct=v["cov_L1_pct"], t_phys=TPHYS[mode], source=src))
    return rows


def fmt(v):
    return "" if v is None else (f"{v:.4g}" if isinstance(v, float) else v)


def plot(rows):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sty = {"MFC v2": ("o", "0.6"), "MFC v3 ctl e341896a": ("D", "0.45"), "MFC v3 subdrv+swap": ("v", "0.2"), "MFC v3 +tiles": ("P", "0.0"), "MFC v3 +tiles mgs32": ("X", "0.0"),
           "AMReX CNS": ("s", "0.3"), "PeleC": ("^", "0.7")}
    ls = {"lock": "--", "sub": "-"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
    for ax, key, ttl in ((axs[0], "s_per_step", "Wall time per level-0 step"), (axs[1], "amr_only_s_per_step", "AMR-only time per level-0 step")):
        for code, (mk, col) in sty.items():
            for mode in ("lock", "sub"):
                pts = sorted((r["np"], r[key]) for r in rows if r["code"] == code and r["mode"] == mode and r.get(key) is not None)
                if not pts: continue
                lab = f"{code}, {'lock-step' if mode == 'lock' else 'subcycled'}"
                ax.plot(*zip(*pts), ls=ls[mode], marker=mk, color=col, mfc=col if mode == "sub" else "white", mec=col, ms=8, lw=2, label=lab)
        ax.set_xscale("log", base=2); ax.set_xticks(NPS); ax.set_xticklabels([str(n) for n in NPS])
        ax.set_xlabel("GPUs (GCDs), 8M base cells each"); ax.set_ylabel("s per level-0 step"); ax.set_title(ttl, fontsize=11)
        ax.set_ylim(bottom=0); ax.grid(axis="y", color="0.88", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("Blast lattice, ladder L weak scaling (dashed/open = lock-step, solid/filled = subcycled)", fontsize=11)
    fig.tight_layout(rect=(0, 0.11, 1, 1)); fig.savefig(f"{B}/scaling.png", dpi=160)


def main():
    rows = [r for c in MFC for r in mfc_rows(*c)] + ref_rows()
    order = {c[0]: i for i, c in enumerate(MFC + [("AMReX CNS",), ("PeleC",)])}
    rows.sort(key=lambda r: (r["mode"], order[r["code"]], r["np"]))
    with open(f"{B}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS); w.writeheader()
        for r in rows: w.writerow({k: fmt(r.get(k)) for k in FIELDS})
    print(f"{'mode':5} {'code':20} {'np':>4} {'window':>7} {'reps':>4} {'s/step':>7} {'sd':>7} {'TTS s':>7} {'AMR s/st':>8} {'AMR TTS':>8} {'rhs imb':>7} {'crs imb':>7} {'cov %':>6}")
    for r in rows:
        if not r.get("reps"): print(f"{r['mode']:5} {r['code']:20} {r['np']:>4}  {r['source']}"); continue
        print(f"{r['mode']:5} {r['code']:20} {r['np']:>4} {r['window']:>7} {r['reps']:>4} {r['s_per_step']:7.4f} {r['s_per_step_sd']:7.4f} {r['tts_s']:7.1f} "
              f"{r['amr_only_s_per_step']:8.4f} {r['amr_only_tts_s'] if r.get('amr_only_tts_s') is not None else float('nan'):8.1f} "
              f"{r['rhs_imb']:7.3f} {r['coarse_imb'] if r.get('coarse_imb') is not None else float('nan'):7.3f} {r['cov_pct']:6.2f}")
    plot([r for r in rows if r.get("reps")])


if __name__ == "__main__":
    main()
