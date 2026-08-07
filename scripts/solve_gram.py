#!/usr/bin/env python3
"""Solve a Gram matrix written by feature-gram or ordering-gram into weights.

Kept as a file rather than inlined into a Slurm --wrap: nested shell quoting
silently mangled the inline version into something that still ran and failed
obscurely.

Usage:
    solve_gram.py <gram.csv> <weights-out.txt>
"""
import sys
import numpy as np

if len(sys.argv) != 3:
    raise SystemExit(__doc__)

xtx, xty = [], None
for line in open(sys.argv[1]):
    line = line.strip()
    if line.startswith("xtx,"):
        xtx.append([float(v) for v in line.split(",")[1:]])
    elif line.startswith("xty,"):
        xty = np.array([float(v) for v in line.split(",")[1:]])

weights = np.linalg.lstsq(np.array(xtx), xty, rcond=None)[0]
with open(sys.argv[2], "w") as handle:
    handle.write("# solved by scripts/solve_gram.py\n")
    handle.write("\n".join(f"{w:.10f}" for w in weights) + "\n")
print(f"wrote {len(weights)} weights to {sys.argv[2]}")
