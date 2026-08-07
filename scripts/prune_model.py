#!/usr/bin/env python3
"""Find the simplest model that plays nearly as well: a regret/complexity frontier.

Rather than apply an information criterion, this traces the Pareto frontier
directly -- for each number of terms, the lowest regret greedy selection can
reach. Read the knee off the curve.

Why not BIC or similar: those are derived for likelihood-based selection, and
our loss is move regret, a decision loss. Penalising the squared-error fit would
regularise the wrong objective, which is the same mistake as fitting values when
ordering is what matters. There is also little in-sample optimism to correct at
these parameter counts.

Where optimism *is* real: greedily choosing the best of hundreds of candidates
at every step is itself a fit, so the selected model's regret is biased low.
Positions are therefore split, selection runs on one half, and the reported
curve is measured on the other.

Terms are main effects and pairwise products. Each candidate fit is a solve of a
submatrix of one precomputed Gram matrix, so a step over hundreds of candidates
costs no data passes.

Usage:
    prune_model.py <move_features.csv> [--terms N]
"""

from __future__ import annotations

import csv
import sys

import numpy as np

META = ("state", "move", "turn_passed", "value_mover", "occupancy")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    max_terms = 24
    if "--terms" in sys.argv:
        max_terms = int(sys.argv[sys.argv.index("--terms") + 1])

    rows = list(csv.DictReader(open(sys.argv[1])))
    names = [k for k in rows[0] if k not in META]
    raw = np.array([[float(r[n]) for n in names] for r in rows])
    values = np.array([float(r["value_mover"]) for r in rows])
    passed = np.array([int(r["turn_passed"]) for r in rows], dtype=float)
    state = np.array([int(r["state"]) for r in rows])
    offsets = np.flatnonzero(np.r_[True, state[1:] != state[:-1]])
    counts = np.diff(np.r_[offsets, len(state)])
    sign = 1 - 2 * passed
    target = values - 100 * passed

    # Candidate terms: main effects, then pairwise products.
    labels = list(names)
    columns = [raw[:, i] for i in range(len(names))]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            labels.append(f"{names[i]} x {names[j]}")
            columns.append(raw[:, i] * raw[:, j])
    design = np.array(columns).T * sign[:, None]
    design = np.hstack([design, sign[:, None]])       # trailing intercept term
    labels.append("intercept")
    intercept = len(labels) - 1
    print(f"{len(offsets)} positions, {len(rows)} moves, {len(labels)} candidate terms")

    half = len(offsets) // 2
    boundary = offsets[half]

    def make(split_offsets, first, last):
        local = split_offsets - first
        local_counts = np.diff(np.r_[local, last - first])
        owner = np.repeat(np.arange(len(local)), local_counts)
        block = design[first:last]
        mean = np.add.reduceat(block, local, axis=0) / local_counts[:, None]
        centred = block - mean[owner]
        y = target[first:last]
        y_centred = y - np.repeat(np.add.reduceat(y, local) / local_counts, local_counts)
        return {
            "design": block, "centred": centred, "y_centred": y_centred,
            "offsets": local, "counts": local_counts,
            "values": values[first:last], "passed": passed[first:last],
            "best": np.maximum.reduceat(values[first:last], local),
            # One Gram over all terms; any subset is a submatrix of it.
            "gram": centred.T @ centred, "moment": centred.T @ y_centred,
        }

    search = make(offsets[:half], 0, boundary)
    holdout = make(offsets[half:], boundary, len(values))

    def fit(part, terms):
        index = np.array(terms)
        matrix = part["gram"][np.ix_(index, index)] + 1e-8 * np.eye(len(index))
        return np.linalg.solve(matrix, part["moment"][index])

    def regret(part, terms, weights):
        score = part["design"][:, terms] @ weights + 100 * part["passed"]
        group_max = np.repeat(np.maximum.reduceat(score, part["offsets"]), part["counts"])
        hits = np.flatnonzero(score >= group_max)
        who = np.searchsorted(part["offsets"], hits, side="right") - 1
        _, first = np.unique(who, return_index=True)
        return float(np.mean(part["best"] - part["values"][hits[first]]))

    chosen = [intercept]
    print(f"\n{'terms':>5s} {'search':>9s} {'holdout':>9s}   added")
    print("-" * 72)
    frontier = []
    for step in range(max_terms):
        best = None
        for candidate in range(len(labels)):
            if candidate in chosen:
                continue
            terms = chosen + [candidate]
            try:
                weights = fit(search, terms)
            except np.linalg.LinAlgError:
                continue
            score = regret(search, terms, weights)
            if best is None or score < best[0]:
                best = (score, candidate, weights)
        if best is None:
            break
        score, candidate, weights = best
        chosen.append(candidate)
        held = regret(holdout, chosen, fit(holdout, chosen))
        frontier.append((len(chosen) - 1, score, held, labels[candidate]))
        print(f"{len(chosen) - 1:>5d} {score:>9.4f} {held:>9.4f}   {labels[candidate]}")

    print("\nfrontier (terms excluding intercept, held-out regret):")
    for terms, _, held, _ in frontier:
        print(f"  {terms:>3d} terms -> {held:.4f} pp")


if __name__ == "__main__":
    main()
