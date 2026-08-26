"""Report GPU kernel resource usage from a built binary, and flag regressions.

Scratch size and register counts are properties of the code object, so they are
deterministic, board-independent and cheap to read - unlike wall-clock. A change that
inflates them is invisible to the golden suite and to CI, which is how a 20% regression
reached review once (see .claude/rules/common-pitfalls.md).

  ./mfc.sh kernel-resources <binary>                    # print
  ./mfc.sh kernel-resources <binary> --baseline b.json  # compare, non-zero on regression
  ./mfc.sh kernel-resources <binary> --write b.json     # record a baseline
"""

import json
import re
import shutil
import subprocess
import sys

FIELDS = ("private_segment_fixed_size", "vgpr_count", "agpr_count")
ELF_MAGIC = b"\x7fELF"


def _tool(name: str) -> str:
    """llvm-objcopy/llvm-readelf, from PATH or a ROCm install."""
    found = shutil.which(name) or shutil.which(f"/opt/rocm/llvm/bin/{name}")
    if not found:
        raise RuntimeError(f"{name} not found; it ships with ROCm's llvm")
    return found


def read(binary: str) -> dict:
    """Map kernel name -> resource dict, for every kernel in the offload image."""
    blob = subprocess.run(
        [_tool("llvm-objcopy"), "--dump-section=.llvm.offloading=/dev/stdout", binary, "/dev/null"],
        capture_output=True,
        check=True,
    ).stdout
    start = blob.find(ELF_MAGIC)
    if start < 0:
        return {}

    notes = subprocess.run([_tool("llvm-readelf"), "--notes", "/dev/stdin"], input=blob[start:], capture_output=True, check=True).stdout.decode(errors="replace")

    # Fields are emitted alphabetically, so .name is not first in a record; parse per
    # record rather than streaming. Records are top-level list items, indented exactly two
    # spaces - matching any indent would also split on the nested .args entries.
    kernels = {}
    for record in re.split(r"\n  - \.", notes):
        fields = dict(re.findall(r"\.?([a-z_]+):\s*(\S+)", record))
        name = fields.get("name")
        if name:
            kernels[name] = {f: int(fields[f]) for f in FIELDS if f in fields}
    return kernels


def _strip(name: str) -> str:
    """Normalise a kernel name to something stable across builds.

    Names carry a per-module hash (`__omp_offloading_<hex>_<hex>__`) that changes whenever the
    source file changes, and a post-fypp line number that shifts under unrelated edits. Leaving
    either in means the kernels you just modified fail to match and are silently skipped.
    """
    name = name.split("__QM")[-1]
    return re.sub(r"_l\d+(_\d+)?$", "", name)


def _values(res: dict) -> tuple:
    """Canonical sort key: dict key order differs after a JSON round-trip."""
    return tuple(res.get(f, 0) for f in FIELDS)


def compare(baseline: dict, current: dict) -> list:
    """Regressions as (kernel, field, was, now), keyed on resources not names."""

    def profile(kernels):
        counts = {}
        for name, res in kernels.items():
            counts.setdefault(_strip(name), []).append(res)
        return counts

    was, now = profile(baseline), profile(current)
    if sum(len(v) for v in was.values()) != sum(len(v) for v in now.values()):
        print("WARNING: kernel count changed; amdflang regenerates the whole device image, so " "untouched kernels may shift too (see #1759)")
    regressions = []
    for name, olds in was.items():
        news = now.get(name)
        if not news or len(news) != len(olds):
            continue
        for old, new in zip(sorted(olds, key=_values), sorted(news, key=_values)):
            for field in FIELDS:
                if new.get(field, 0) > old.get(field, 0):
                    regressions.append((name, field, old[field], new[field]))
    return regressions


def main(argv: list) -> int:
    if not argv:
        print(__doc__)
        return 2
    kernels = read(argv[0])
    if not kernels:
        print("no GPU offload image in this binary")
        return 0

    if "--write" in argv:
        path = argv[argv.index("--write") + 1]
        with open(path, "w") as handle:
            json.dump(kernels, handle, indent=1, sort_keys=True)
        print(f"wrote {len(kernels)} kernels to {path}")
        return 0

    if "--baseline" in argv:
        with open(argv[argv.index("--baseline") + 1]) as handle:
            regressions = compare(json.load(handle), kernels)
        for name, field, old, new in regressions:
            print(f"REGRESSION {name}: {field} {old} -> {new}")
        print(f"{len(regressions)} regression(s) across {len(kernels)} kernels")
        return 1 if regressions else 0

    for name in sorted(kernels):
        res = kernels[name]
        print(f"{name}  " + "  ".join(f"{f}={res.get(f, '?')}" for f in FIELDS))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
