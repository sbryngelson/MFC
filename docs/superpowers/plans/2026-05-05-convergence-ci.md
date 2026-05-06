# Convergence CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automated convergence-rate verification for WENO1/3/5 and MUSCL on the 2D isentropic vortex problem, run in CI on every PR.

**Architecture:** A new parametric example case (`2D_isentropicvortex_convergence`) is run at multiple grid resolutions by a standalone Python script (`toolchain/mfc/test/run_convergence.py`). The script reads `p_all/` binary output, computes L2 errors against the t=0 initial condition (the exact solution for this stationary vortex), fits convergence rates, and exits non-zero on failure. A new GitHub Actions workflow calls it.

**Tech Stack:** Python 3.9+, NumPy, struct (stdlib), MFC's `./mfc.sh run`, Fortran unformatted binary I/O.

---

## Background

The isentropic vortex is an exact stationary solution to the compressible Euler equations (single fluid, γ=1.4, ε=5, α=1). With periodic BCs and no background flow, the solution at any time T is identical to the initial condition. Numerical errors accumulate at rate O(hᵖ) where p is the scheme order. We measure:

```
L2_error = ||ρ(T) - ρ(0)||_L2  =  sqrt( Σ (ρ(T,i,j) - ρ(0,i,j))² * dx * dy )
```

across resolutions N=32, 64, 128 and verify slope of log-log plot ≥ expected order − 0.5.

Physical end time T=2.0 (fixed). dt = 0.4 * dx / sqrt(γ). Nt = ceil(T/dt).

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `examples/2D_isentropicvortex_convergence/case.py` | Parametric vortex case: accepts `-N`, `--order`, `--muscl` |
| Create | `toolchain/mfc/test/run_convergence.py` | Driver: runs multiple resolutions, reads output, checks rates |
| Modify | `toolchain/mfc/test/cases.py` | Add new example to `casesToSkip` |
| Create | `.github/workflows/convergence.yml` | CI job |

---

## Task 1: Parametric convergence case

**Files:**
- Create: `examples/2D_isentropicvortex_convergence/case.py`
- Modify: `toolchain/mfc/test/cases.py` (add to skip list)

### Physics

Isentropic vortex centered at (0,0) on [-5,5]²:
```
r² = (x-xc)² + (y-yc)²
f  = (ε/(2π)) * exp(α*(1 - r²))        # perturbation kernel
ρ  = [1 - (ε²*(γ-1))/(8α*π²) * exp(2α*(1-r²))]^(1/(γ-1))
p  = ρ^γ                                 # isentropic
u  = u0 + (y-yc)*f
v  = v0 - (x-xc)*f
```

With ε=5, α=1, γ=1.4, u0=v0=0, xc=yc=0.

- [ ] **Step 1.1: Create the case file**

```python
#!/usr/bin/env python3
# examples/2D_isentropicvortex_convergence/case.py
import argparse, json, math

parser = argparse.ArgumentParser(description="2D isentropic vortex convergence case")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
parser.add_argument("-N",       type=int,  default=64,  help="Grid points per dim (default: 64)")
parser.add_argument("--order",  type=int,  default=5,   help="WENO order: 1, 3, or 5 (default: 5)")
parser.add_argument("--muscl",  action="store_true",    help="Use MUSCL instead of WENO")
args = parser.parse_args()

# Physics parameters
eps   = 5.0
alpha = 1.0
gamma = 1.4
xc, yc = 0.0, 0.0

# Isentropic vortex IC strings (evaluated by MFC's analytic IC engine)
# r² = (x-xc)² + (y-yc)²
r2   = f"((x - {xc})**2 + (y - {yc})**2)"
kern = f"({eps}/(2*pi))*exp({alpha}*(1 - {r2}))"
T_fac = f"(1 - ({eps}**2*({gamma}-1))/(8*{alpha}*pi**2)*exp(2*{alpha}*(1 - {r2})))"

alpha_rho = f"{T_fac}**(1/({gamma}-1))"
pres      = f"{T_fac}**({gamma}/({gamma}-1))"
vel1      = f"(y - {yc})*{kern}"
vel2      = f"-((x - {xc})*{kern})"

# Grid
N  = args.N
m  = N - 1
Lx = 10.0
dx = Lx / N

# Time stepping: CFL=0.4, c_max ≈ sqrt(gamma) ≈ 1.18
c_max = math.sqrt(gamma)
dt    = 0.4 * dx / c_max
T_end = 2.0
Nt    = max(4, math.ceil(T_end / dt))
dt    = T_end / Nt  # adjust to land exactly on T_end

# Scheme selection
if args.muscl:
    scheme_params = {
        "recon_type": 2,
        "muscl_order": 2,
        "muscl_lim": 1,
    }
else:
    scheme_params = {
        "recon_type": 1,
        "weno_order": args.order,
        "weno_eps": 1.0e-16,
        "mapped_weno": "T",
        "null_weights": "F",
        "mp_weno": "F",
    }

print(json.dumps({
    "run_time_info": "F",
    "x_domain%beg": -5.0, "x_domain%end": 5.0,
    "y_domain%beg": -5.0, "y_domain%end": 5.0,
    "m": m, "n": m, "p": 0,
    "dt": dt,
    "t_step_start": 0, "t_step_stop": Nt, "t_step_save": Nt,
    "num_patches": 1,
    "model_eqns": 3,
    "alt_soundspeed": "F",
    "num_fluids": 1,
    "mpp_lim": "F", "mixture_err": "F",
    "time_stepper": 3,
    "riemann_solver": 2,
    "wave_speeds": 1,
    "avg_state": 2,
    "bc_x%beg": -4, "bc_x%end": -4,
    "bc_y%beg": -4, "bc_y%end": -4,
    "format": 1,
    "precision": 2,
    "prim_vars_wrt": "T",
    "parallel_io": "F",
    "patch_icpp(1)%geometry": 3,
    "patch_icpp(1)%x_centroid": xc,
    "patch_icpp(1)%y_centroid": yc,
    "patch_icpp(1)%length_x": Lx,
    "patch_icpp(1)%length_y": Lx,
    "patch_icpp(1)%vel(1)": vel1,
    "patch_icpp(1)%vel(2)": vel2,
    "patch_icpp(1)%pres": pres,
    "patch_icpp(1)%alpha_rho(1)": alpha_rho,
    "patch_icpp(1)%alpha(1)": 1.0,
    "fluid_pp(1)%gamma": 1.0 / (gamma - 1.0),
    "fluid_pp(1)%pi_inf": 0.0,
    **scheme_params,
}))
```

- [ ] **Step 1.2: Add to casesToSkip**

In `toolchain/mfc/test/cases.py`, find the `casesToSkip` list and append:
```python
"2D_isentropicvortex_convergence",
```

- [ ] **Step 1.3: Validate the case runs**

```bash
git checkout master          # or whichever branch you're on
source ./mfc.sh load -c p -m c
./mfc.sh build -t pre_process simulation -j 8
./mfc.sh run examples/2D_isentropicvortex_convergence/case.py -n 1 -- -N 32 --order 5
```

Expected: pre_process and simulation both complete with exit 0. Check that
`examples/2D_isentropicvortex_convergence/p_all/p0/0/q_prim_vf1.dat` and
`examples/2D_isentropicvortex_convergence/p_all/p0/{Nt}/q_prim_vf1.dat` exist.

- [ ] **Step 1.4: Commit**

```bash
git add examples/2D_isentropicvortex_convergence/case.py toolchain/mfc/test/cases.py
git commit -m "Add parametric 2D isentropic vortex convergence case"
```

---

## Task 2: Convergence test runner

**Files:**
- Create: `toolchain/mfc/test/run_convergence.py`

The script:
1. Builds pre_process + simulation (optionally skipped with `--no-build`)
2. For each scheme (WENO1, WENO3, WENO5, MUSCL) and each N in [32, 64, 128]:
   - Runs the case in a temp directory
   - Reads `p_all/p0/0/q_prim_vf1.dat` (initial density) and `p_all/p0/{Nt}/q_prim_vf1.dat` (final density)
   - Computes L2 error
3. Fits convergence rate (least-squares slope of log error vs log dx)
4. Checks rate >= expected_order - 0.5
5. Prints a table; exits 1 if any test fails

### Reading binary output

MFC writes Fortran unformatted sequential binary. Each file has one record:
```
[4-byte record length][N*N * 8-byte float64 values][4-byte record length]
```
For a 2D case with m=N-1, n=N-1, there are N*N values, stored in column-major (Fortran) order.

- [ ] **Step 2.1: Create `toolchain/mfc/test/run_convergence.py`**

```python
#!/usr/bin/env python3
"""
Convergence-rate verification for MFC's 2D isentropic vortex problem.
Runs WENO1/3/5 and MUSCL at multiple grid resolutions, checks that
L2 errors decrease at the expected rate.

Usage:
    python toolchain/mfc/test/run_convergence.py [--no-build] [--resolutions 32 64 128]
"""
import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

CASE = "examples/2D_isentropicvortex_convergence/case.py"
MFC  = "./mfc.sh"

# (label, extra_args, expected_order, tolerance)
SCHEMES = [
    ("WENO5",  ["--order", "5"],         5, 0.5),
    ("WENO3",  ["--order", "3"],         3, 0.5),
    ("WENO1",  ["--order", "1"],         1, 0.4),
    ("MUSCL2", ["--muscl"],              2, 0.5),
]


def read_prim_vf1(run_dir: str, step: int, N: int) -> np.ndarray:
    """Read density (q_prim_vf1) from p_all binary output at the given step."""
    path = os.path.join(run_dir, "p_all", "p0", str(step), "q_prim_vf1.dat")
    with open(path, "rb") as f:
        rec_len = struct.unpack("i", f.read(4))[0]
        data = np.frombuffer(f.read(rec_len), dtype=np.float64)
        f.read(4)  # trailing record length
    assert data.size == N * N, f"Expected {N*N} values, got {data.size}"
    return data.reshape((N, N), order="F")


def l2_error(rho_final: np.ndarray, rho_init: np.ndarray, dx: float) -> float:
    """Normalised L2 error: sqrt(sum((f-g)^2 * dx^2))."""
    diff = rho_final - rho_init
    return float(np.sqrt(np.sum(diff**2) * dx**2))


def convergence_rate(errors: list[float], resolutions: list[int]) -> float:
    """Least-squares slope of log(error) vs log(dx), dx = 10/N."""
    log_dx  = np.log(10.0 / np.array(resolutions, dtype=float))
    log_err = np.log(np.array(errors, dtype=float))
    # polyfit: log_err = rate * log_dx + const
    rate, _ = np.polyfit(log_dx, log_err, 1)
    return float(rate)


def run_case(tmpdir: str, N: int, extra_args: list[str]) -> int:
    """Run the vortex case at resolution N in tmpdir. Returns the final step number."""
    import json, math
    # Determine Nt by running case.py with --dry-run equivalent: just parse its output
    result = subprocess.run(
        [sys.executable, CASE, "--mfc", "{}", "-N", str(N)] + extra_args,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"case.py failed:\n{result.stderr}")
    cfg = json.loads(result.stdout)
    Nt = int(cfg["t_step_stop"])

    cmd = [
        MFC, "run", CASE, "--no-build", "-n", "1",
        "--", "-N", str(N),
    ] + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    if result.returncode != 0:
        print(result.stdout[-2000:])
        raise RuntimeError(f"./mfc.sh run failed for N={N}")

    # Copy p_all output to tmpdir (mfc.sh run writes into the case directory)
    src = os.path.join(os.path.dirname(CASE), "p_all")
    dst = os.path.join(tmpdir, f"N{N}", "p_all")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    # Clean up case dir for next run
    shutil.rmtree(os.path.join(os.path.dirname(CASE), "p_all"), ignore_errors=True)
    shutil.rmtree(os.path.join(os.path.dirname(CASE), "D"),     ignore_errors=True)

    return Nt, dst


def test_scheme(label, extra_args, expected_order, tol, resolutions):
    print(f"\n{'='*60}")
    print(f"  {label}  (expected order ≥ {expected_order - tol:.1f})")
    print(f"{'='*60}")
    print(f"  {'N':>6}  {'dx':>10}  {'L2 error':>14}  {'rate':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*14}  {'-'*8}")

    errors = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for N in resolutions:
            dx = 10.0 / N
            Nt, run_dir = run_case(tmpdir, N, extra_args)
            rho0 = read_prim_vf1(run_dir, 0,  N)
            rhoT = read_prim_vf1(run_dir, Nt, N)
            err = l2_error(rhoT, rho0, dx)
            errors.append(err)
            print(f"  {N:>6}  {dx:>10.5f}  {err:>14.6e}  {'---':>8}")

    # Compute rates between consecutive pairs
    rates = []
    for i in range(1, len(resolutions)):
        log_dx0 = np.log(10.0 / resolutions[i-1])
        log_dx1 = np.log(10.0 / resolutions[i])
        rate = (np.log(errors[i]) - np.log(errors[i-1])) / (log_dx1 - log_dx0)
        rates.append(rate)

    # Reprint table with rates
    print(f"\n  {'N':>6}  {'dx':>10}  {'L2 error':>14}  {'rate':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*14}  {'-'*8}")
    for i, N in enumerate(resolutions):
        dx  = 10.0 / N
        err = errors[i]
        r   = f"{rates[i-1]:>8.2f}" if i > 0 else f"{'---':>8}"
        print(f"  {N:>6}  {dx:>10.5f}  {err:>14.6e}  {r}")

    overall_rate = convergence_rate(errors, resolutions)
    print(f"\n  Overall fitted rate: {overall_rate:.2f}  (need >= {expected_order - tol:.1f})")

    passed = overall_rate >= expected_order - tol
    status = "PASS" if passed else "FAIL"
    print(f"  {status}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="MFC convergence-rate verification")
    parser.add_argument("--no-build",    action="store_true",   help="Skip build step")
    parser.add_argument("--resolutions", type=int, nargs="+",
                        default=[32, 64, 128],                  help="Grid resolutions (default: 32 64 128)")
    parser.add_argument("--schemes",     nargs="+",
                        default=["WENO5", "WENO3", "WENO1", "MUSCL2"],
                        help="Schemes to test (default: all)")
    args = parser.parse_args()

    if not args.no_build:
        print("Building pre_process and simulation...")
        result = subprocess.run([MFC, "build", "-t", "pre_process", "simulation", "-j", "8"],
                                capture_output=False)
        if result.returncode != 0:
            sys.exit(1)

    results = {}
    for label, extra_args, expected_order, tol in SCHEMES:
        if label not in args.schemes:
            continue
        try:
            passed = test_scheme(label, extra_args, expected_order, tol, args.resolutions)
        except Exception as e:
            print(f"  ERROR: {e}")
            passed = False
        results[label] = passed

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    all_pass = True
    for label, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {label:<12} {status}")
        if not passed:
            all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2: Smoke test the runner at a single resolution**

```bash
python toolchain/mfc/test/run_convergence.py --no-build --resolutions 32 --schemes WENO5
```

Expected output: table with N=32, an L2 error ~O(1e-4) or smaller, rate shown as `---`, and `PASS`.

- [ ] **Step 2.3: Run two resolutions and check the rate column appears**

```bash
python toolchain/mfc/test/run_convergence.py --no-build --resolutions 32 64 --schemes WENO5
```

Expected: rate for N=64 row is close to 5 (may vary at small N; WENO5 needs more points to reach asymptotic regime).

- [ ] **Step 2.4: Run all resolutions, all schemes**

```bash
python toolchain/mfc/test/run_convergence.py --no-build --resolutions 32 64 128
```

All four schemes should show `PASS`. If WENO1 or MUSCL2 fail at low resolution, increase to 32 64 128 256. Adjust expected tolerances if needed before committing.

- [ ] **Step 2.5: Commit**

```bash
git add toolchain/mfc/test/run_convergence.py
git commit -m "Add convergence-rate test runner for 2D isentropic vortex"
```

---

## Task 3: CI workflow

**Files:**
- Create: `.github/workflows/convergence.yml`

- [ ] **Step 3.1: Create the workflow**

```yaml
# .github/workflows/convergence.yml
name: Convergence

on:
  push:
    branches: [master]
  pull_request:
  workflow_dispatch:

jobs:
  convergence:
    name: "2D Isentropic Vortex Convergence"
    runs-on: ubuntu-latest
    timeout-minutes: 60

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install Python dependencies
        run: pip install numpy fypp

      - name: Set up MPI
        run: sudo apt-get install -y --no-install-recommends libopenmpi-dev openmpi-bin

      - name: Run convergence tests
        run: |
          python toolchain/mfc/test/run_convergence.py \
            --resolutions 32 64 128
```

- [ ] **Step 3.2: Validate the workflow file is valid YAML**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/convergence.yml'))" && echo "OK"
```

Expected: `OK`

- [ ] **Step 3.3: Commit and push to trigger CI**

```bash
git add .github/workflows/convergence.yml
git commit -m "Add convergence CI workflow (issue #305)"
git push
```

Watch the Actions tab at `https://github.com/MFlowCode/MFC/actions` for the `Convergence` job.

---

## Self-Review

**Spec coverage:**
- ✅ WENO5, WENO3, WENO1: tested via `--order 5/3/1`
- ✅ MUSCL: tested via `--muscl`
- ✅ 2D isentropic vortex problem: `examples/2D_isentropicvortex_convergence/case.py`
- ✅ CI integration: `.github/workflows/convergence.yml`
- ✅ Convergence rate verified quantitatively, not just "does it run"

**Placeholder scan:** None found — all steps have concrete code and commands.

**Type consistency:** `read_prim_vf1` returns `np.ndarray`, consumed by `l2_error`. `convergence_rate` takes `list[float]` and `list[int]`, returning `float`. `test_scheme` returns `bool`. All consistent.

**Known limitation:** `run_case` always runs in the case directory (MFC's run writes output there). Two concurrent runs of the same case would collide. The script runs sequentially so this is not an issue for CI.

**Tuning note:** Asymptotic convergence for WENO5 typically requires N≥64. At N=32 the rate may be lower. The overall fitted rate across [32,64,128] should be ≥4.5 for WENO5. If not, increase `--resolutions` to `64 128 256` and update the default in the script.
