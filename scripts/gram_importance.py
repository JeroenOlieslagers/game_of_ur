#!/usr/bin/env python3
"""Exact feature importance over the full population of states.

Consumes the normal equations written by `royalur_analysis feature-gram`, which
are accumulated over *every* non-terminal state in the map. Because least
squares needs only `X'X` and `X'y`, this is the exact full-population fit, not a
sample: there is no sampling error, and nothing to overfit to.

Every subset of features can then be fitted by solving the corresponding
submatrix, so all 2^k subset fits come from the one pass. That makes an exact
Shapley decomposition of R^2 affordable.

R^2 is monotone under nesting -- a superset can always reproduce a subset by
zeroing a weight -- so every Shapley value here is non-negative, and they sum to
the full model's R^2.

Usage:
    gram_importance.py <gram.csv>
"""

from __future__ import annotations

import itertools
import math
import sys

import numpy as np


def load(path: str):
    names, rows, yty, xtx, xty = None, None, None, [], None
    for line in open(path):
        line = line.strip()
        if line.startswith("# columns:"):
            names = line.split(":", 1)[1].strip().split(",")
        elif line.startswith("#") or not line:
            continue
        elif line.startswith("rows,"):
            rows = int(line.split(",")[1])
        elif line.startswith("yty,"):
            yty = float(line.split(",")[1])
        elif line.startswith("xtx,"):
            xtx.append([float(v) for v in line.split(",")[1:]])
        elif line.startswith("xty,"):
            xty = np.array([float(v) for v in line.split(",")[1:]])
    return names, rows, yty, np.array(xtx), xty


def r_squared(columns, xtx, xty, yty, rows, intercept_index, total_sum_squares):
    """R^2 for a feature subset, solved from the Gram submatrix.

    The intercept is always included, so the model is never worse than
    predicting the mean.
    """
    idx = list(columns) + [intercept_index]
    a = xtx[np.ix_(idx, idx)]
    b = xty[idx]
    try:
        beta = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(a, b, rcond=None)[0]
    # Residual sum of squares = y'y - 2 b'X'y + b'X'X b, which reduces to
    # y'y - b'X'y at the least-squares solution.
    rss = yty - beta @ b
    return 1.0 - rss / total_sum_squares


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    names, rows, yty, xtx, xty = load(sys.argv[1])
    intercept_index = len(names) - 1
    feature_names = names[:intercept_index]
    count = len(feature_names)

    # Total sum of squares about the mean: y'y - n * mean^2.
    mean = xty[intercept_index] / rows
    total_sum_squares = yty - rows * mean * mean

    print(f"{rows:,} states (full population), {count} features\n")
    full = r_squared(range(count), xtx, xty, yty, rows, intercept_index, total_sum_squares)
    print(f"full model R^2 = {full:.6f}")

    beta = np.linalg.solve(
        xtx[np.ix_(list(range(count)) + [intercept_index], list(range(count)) + [intercept_index])],
        xty[list(range(count)) + [intercept_index]],
    )
    scale = abs(beta[feature_names.index("advancement_self")])
    print("\nfitted weights (win-probability points):")
    for name, weight in sorted(zip(feature_names, beta[:-1]), key=lambda p: -abs(p[1])):
        print(f"  {name:>18s} {weight:+9.4f}   {weight / scale:+8.1f} squares of advancement")
    print(f"  {'intercept':>18s} {beta[-1]:+9.4f}")

    cache = {}

    def value_of(subset):
        key = tuple(sorted(subset))
        if key not in cache:
            cache[key] = r_squared(key, xtx, xty, yty, rows, intercept_index, total_sum_squares)
        return cache[key]

    shapley = np.zeros(count)
    for i in range(count):
        rest = [j for j in range(count) if j != i]
        for size in range(len(rest) + 1):
            weight = math.factorial(size) * math.factorial(count - size - 1) / math.factorial(count)
            for subset in itertools.combinations(rest, size):
                shapley[i] += weight * (value_of(tuple(subset) + (i,)) - value_of(subset))

    print(f"\nexact Shapley decomposition of R^2 ({2 ** count} subsets):")
    for i in np.argsort(-shapley):
        share = 100 * shapley[i] / full
        print(f"  {feature_names[i]:>18s} {shapley[i]:8.5f}  {share:5.1f}% of explained variance")
    print(f"  {'sum':>18s} {shapley.sum():8.5f}  (= full model R^2 {full:.5f})")


if __name__ == "__main__":
    main()
