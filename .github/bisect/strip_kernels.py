"""Strip the GPU offload directives from one slice of the kernels in src/simulation/m_amr_exchange.fpp (compile-bisect helper).

    python3 strip_kernels.py CHUNK NCHUNKS      # CHUNK in 1..NCHUNKS; 0 = change nothing

Kernels are the subroutines/functions that contain $:GPU_PARALLEL_LOOP; they are split into NCHUNKS contiguous groups and the
$:GPU_PARALLEL_LOOP(...) statement and its $:END_GPU_PARALLEL_LOOP() are deleted in the chosen group, so that group runs as plain host loops."""
import re
import sys

chunk, nchunks = int(sys.argv[1]), int(sys.argv[2])
path = "src/simulation/m_amr_exchange.fpp"
text = open(path).read()
lines = text.split("\n")
start_re = re.compile(r"^\s*(?:impure |pure |elemental )*(?:integer |logical |real\(wp\) )?(subroutine|function)\s+([\w${}]+)")
end_re = re.compile(r"^\s*end\s+(subroutine|function)\s+([\w${}]+)")
spans = []
cur = None
for i, l in enumerate(lines):
    m = start_re.match(l)
    if m and cur is None:
        cur = (i, m.group(2))
    m = end_re.match(l)
    if m and cur and m.group(2) == cur[1]:
        spans.append((cur[0], i, cur[1]))
        cur = None
kern = [s for s in spans if any("$:GPU_PARALLEL_LOOP(" in lines[j] for j in range(s[0], s[1] + 1))]
print(len(spans), "procedures,", len(kern), "kernel procedures", flush=True)
if chunk == 0:
    sys.exit(0)
per = -(-len(kern) // nchunks)
mine = kern[(chunk - 1) * per : chunk * per]
print("stripping:", [k[2] for k in mine], flush=True)
drop = set()
for a, b, _ in mine:
    j = a
    while j <= b:
        if "$:GPU_PARALLEL_LOOP(" in lines[j]:
            depth = 0
            k = j
            while True:
                depth += lines[k].count("(") - lines[k].count(")")
                drop.add(k)
                if depth <= 0:
                    break
                k += 1
            j = k
        elif "$:END_GPU_PARALLEL_LOOP()" in lines[j]:
            drop.add(j)
        j += 1
open(path, "w").write("\n".join(l for i, l in enumerate(lines) if i not in drop))
print("deleted", len(drop), "lines", flush=True)
