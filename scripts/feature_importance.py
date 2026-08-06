#!/usr/bin/env python3
"""Rank features by how much they contribute to *play strength*.

Consumes the per-move dump from `royalur_analysis dump-moves`, which holds every
candidate move of every sampled position with its features and its exact value.
That makes regret computable for an arbitrary weight vector without re-running
the engine, so thousands of feature subsets can be scored.

Two importance measures are reported, because they disagree:

  * Shapley over R^2 -- how much each feature helps *predict* the value.
  * Shapley over regret reduction -- how much each feature helps *choose moves*.

Shapley averages a feature's marginal contribution over every ordering, which is
the principled fix for the fact that nested-model contributions depend on the
order features are added when the features are correlated. Ours are strongly
correlated, so this matters.

Usage:
    feature_importance.py <moves.csv> [--subset-cap N]
"""

from __future__ import annotations

import csv
import itertools
import math
import sys
from collections import defaultdict

import numpy as np


def load(path: str):
    """Rows arrive grouped by position and in move order, so segment boundaries
    can be read off directly rather than reconstructed."""
    rows = list(csv.DictReader(open(path)))
    names = [k for k in rows[0] if k not in ("state", "move", "turn_passed", "value_mover")]
    features = np.array([[float(r[n]) for n in names] for r in rows])
    values = np.array([float(r["value_mover"]) for r in rows])
    passed = np.array([int(r["turn_passed"]) for r in rows], dtype=bool)
    state = np.array([int(r["state"]) for r in rows])
    offsets = np.flatnonzero(np.r_[True, state[1:] != state[:-1]])
    # Fit against the successor's value from the successor mover's perspective,
    # which is what the features describe.
    targets = np.where(passed, 100.0 - values, values)
    return names, features, values, passed, targets, offsets


def fit(features, targets, columns):
    if not columns:
        return np.zeros(0), float(targets.mean())
    design = np.hstack([features[:, columns], np.ones((len(features), 1))])
    solution, *_ = np.linalg.lstsq(design, targets, rcond=None)
    return solution[:-1], float(solution[-1])


def r_squared(features, targets, columns):
    weights, intercept = fit(features, targets, columns)
    predicted = (features[:, columns] @ weights + intercept) if columns else np.full(len(targets), intercept)
    return 1 - ((targets - predicted) ** 2).sum() / ((targets - targets.mean()) ** 2).sum()


def mean_regret(features, values, passed, offsets, best, columns, weights, intercept):
    """Regret of the greedy policy induced by these weights.

    Fully vectorised: one matmul plus segment reductions, so the cost is linear
    in the number of candidate moves regardless of how many positions there are.
    """
    if columns:
        score = features[:, columns] @ weights + intercept
    else:
        score = np.full(len(values), intercept)
    # A move that passes the turn hands the position to the opponent, so its
    # score must be reflected about 50 to compare with a move that keeps it.
    score = np.where(passed, 100.0 - score, score)

    counts = np.diff(np.r_[offsets, len(score)])
    group_max = np.repeat(np.maximum.reduceat(score, offsets), counts)
    hits = np.flatnonzero(score >= group_max)
    # First maximum in each segment, matching the engine's tie-breaking.
    owner = np.searchsorted(offsets, hits, side="right") - 1
    _, first = np.unique(owner, return_index=True)
    chosen = hits[first]
    return float(np.mean(best - values[chosen]))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    names, features, values, passed, targets, offsets = load(path)
    counts = np.diff(np.r_[offsets, len(values)])
    best = np.maximum.reduceat(values, offsets)
    count = len(names)
    print(f"{len(offsets)} positions, {len(features)} candidate moves, {count} features\n")

    def regret_of(cols, w, b):
        return mean_regret(features, values, passed, offsets, best, cols, w, b)

    full_cols = list(range(count))
    weights, intercept = fit(features, targets, full_cols)
    print(f"full model: R^2 = {r_squared(features, targets, full_cols):.4f}  "
          f"mean regret = {regret_of(full_cols, weights, intercept):.4f} pp")
    baseline = regret_of([], np.zeros(0), float(targets.mean()))
    print(f"no features (constant score): mean regret = {baseline:.4f} pp\n")

    # Exact Shapley over all 2^count subsets.
    cache_r2, cache_regret = {}, {}

    def value_of(subset):
        key = tuple(sorted(subset))
        if key not in cache_r2:
            cols = list(key)
            w, b = fit(features, targets, cols)
            cache_r2[key] = r_squared(features, targets, cols)
            cache_regret[key] = baseline - regret_of(cols, w, b)
        return cache_r2[key], cache_regret[key]

    shapley_r2 = np.zeros(count)
    shapley_regret = np.zeros(count)
    others = list(range(count))
    for i in range(count):
        rest = [j for j in others if j != i]
        for size in range(len(rest) + 1):
            weight = math.factorial(size) * math.factorial(count - size - 1) / math.factorial(count)
            for subset in itertools.combinations(rest, size):
                with_i = value_of(tuple(subset) + (i,))
                without = value_of(subset)
                shapley_r2[i] += weight * (with_i[0] - without[0])
                shapley_regret[i] += weight * (with_i[1] - without[1])

    print(f"{'feature':>18s} {'Shapley R^2':>12s} {'Shapley regret':>15s}")
    order = np.argsort(-shapley_regret)
    for i in order:
        print(f"{names[i]:>18s} {shapley_r2[i]:12.4f} {shapley_regret[i]:15.4f}")
    print(f"\n{'sum':>18s} {shapley_r2.sum():12.4f} {shapley_regret.sum():15.4f}")
    print("(Shapley values sum exactly to the full model's total, by construction.)")


if __name__ == "__main__":
    main()
