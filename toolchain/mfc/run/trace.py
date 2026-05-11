#!/usr/bin/env python3
import collections
import os
import re
import subprocess
import sys


def _normalize_acc_line(line: str) -> str:
    line = re.sub(r"file=[^ ]*/(src/[^ ]*)", r"file=\1", line)
    line = re.sub(r"[^ ]*/(src/[^ ]*)", r"\1", line)
    line = re.sub(r"devaddr=0x[0-9A-Fa-f]+", "devaddr=<addr>", line)
    return line


def _is_acc_noise(line: str) -> bool:
    prefixes = (
        "Enter ",
        "Leave ",
        "Wait ",
        "Implicit wait",
        "upload CUDA data",
        "download CUDA data",
        "create CUDA data",
        "delete CUDA data",
        "alloc",
    )
    return line.startswith(prefixes) or (".fpp function=" in line and " device=" in line)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: trace.py <command> [args...]", file=sys.stderr)
        return 2

    command = sys.argv[1:]

    if any(arg.endswith("/syscheck") for arg in command):
        env = os.environ.copy()
        env["MFC_TRACE"] = "0"
        env.pop("MFC_TRACE_ACC_NOTIFY", None)
        env["NV_ACC_NOTIFY"] = "0"
        return subprocess.run(command, env=env, check=False).returncode

    trace_file = os.environ.get("MFC_TRACE_FILE")
    if not os.environ.get("MFC_TRACE_ACC_NOTIFY") or not trace_file:
        return subprocess.run(command, check=False).returncode

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert process.stdout is not None

    seen = set()
    repeats = collections.Counter()

    with open(trace_file, "a", encoding="utf-8") as trace:
        for output_line in process.stdout:
            line = output_line.rstrip("\n")
            if line.startswith("launch CUDA kernel"):
                short_line = _normalize_acc_line(line)
                if short_line in seen:
                    repeats[short_line] += 1
                else:
                    seen.add(short_line)
                    trace.write(f"TRACE_ACC {short_line}\n")
                    trace.flush()
            elif not _is_acc_noise(line):
                print(line, flush=True)

        for line, count in sorted(repeats.items()):
            trace.write(f"TRACE_ACC_REPEAT x{count} {line}\n")

    return process.wait()


if __name__ == "__main__":
    sys.exit(main())
