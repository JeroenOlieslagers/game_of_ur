#!/usr/bin/env python3
"""Shapley decomposition of regret reduction, by Monte Carlo over permutations.

The R^2 decomposition in gram_importance.py is exact because 14 features means
2^14 subsets. With all 36 features an exact Shapley is out of reach (2^36), so
this samples permutations instead: for each ordering, features are added one at
a time and a feature is credited with the regret it removes at the point it
enters. Averaging over enough orderings converges to the Shapley value.

Regret, not R^2, is the right currency here. The exact R^2 decomposition put
scored/hand at 81% of explained variance while threat and exposure came last --
and those last two are among the few features that discriminate between the
candidate moves of a position. Ranking by variance explained ranks features
almost backwards from what play needs.

Scoring matches the rest of the analysis: state features signed by the
reflection, move features left mover-relative, and the 100-point constant added
back before siblings are compared.

Usage:
    shapley_regret.py <move_features.csv> [--permutations N]
"""

from __future__ import annotations

import csv
import sys

import numpy as np

STATE = ["advancement_self", "advancement_opp", "scored_self", "scored_opp",
         "hand_self", "hand_opp", "safe_self", "safe_opp", "exposure_self",
         "threat_self", "centre_self", "centre_opp", "frontmost_self", "frontmost_opp"]
MOVE = ["advance", "captures", "scores", "enters", "lands_rosette", "lands_centre",
        "leaves_centre", "dest_safe", "src_was_exposed", "delta_exposure",
        "delta_threat", "keeps_turn", "capture_value", "rescue_value",
        "delta_exposure_value", "delta_threat_value", "captures_frontmost",
        "capture_gap_to_front", "moves_frontmost", "becomes_safe_forever",
        "contact_possible", "threat_count"]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    permutations = 120
    if "--permutations" in sys.argv:
        permutations = int(sys.argv[sys.argv.index("--permutations") + 1])

    rows = list(csv.DictReader(open(sys.argv[1])))
    names = STATE + MOVE
    columns = {n: np.array([float(r[n]) for r in rows]) for n in names}
    values = np.array([float(r["value_mover"]) for r in rows])
    passed = np.array([int(r["turn_passed"]) for r in rows], dtype=float)
    state = np.array([int(r["state"]) for r in rows])
    offsets = np.flatnonzero(np.r_[True, state[1:] != state[:-1]])
    counts = np.diff(np.r_[offsets, len(state)])
    owner = np.repeat(np.arange(len(offsets)), counts)
    best = np.maximum.reduceat(values, offsets)
    sign = 1 - 2 * passed
    target = values - 100 * passed

    design = np.hstack([
        np.array([columns[n] * sign for n in STATE]).T,
        np.array([columns[n] for n in MOVE]).T,
        sign[:, None],
    ])
    intercept = design.shape[1] - 1

    # Centre once; every subset fit is then a submatrix solve of one Gram.
    mean = np.add.reduceat(design, offsets, axis=0) / counts[:, None]
    centred = design - mean[owner]
    target_centred = target - np.repeat(np.add.reduceat(target, offsets) / counts, counts)
    gram = centred.T @ centred
    moment = centred.T @ target_centred

    def regret_of(terms):
        if not terms:
            terms = [intercept]
        index = np.array(sorted(set(terms + [intercept])))
        matrix = gram[np.ix_(index, index)] + 1e-8 * np.eye(len(index))
        weights = np.linalg.solve(matrix, moment[index])
        score = design[:, index] @ weights + 100 * passed
        group_max = np.repeat(np.maximum.reduceat(score, offsets), counts)
        hits = np.flatnonzero(score >= group_max)
        who = np.searchsorted(offsets, hits, side="right") - 1
        _, first = np.unique(who, return_index=True)
        return float(np.mean(best - values[hits[first]]))

    baseline = regret_of([])
    full = regret_of(list(range(len(names))))
    print(f"{len(offsets)} positions, {len(names)} features, {permutations} permutations")
    print(f"no features: {baseline:.4f} pp   all features: {full:.4f} pp   "
          f"total reduction: {baseline - full:.4f} pp\n")

    rng = np.random.default_rng(0)
    contribution = np.zeros(len(names))
    for step in range(permutations):
        order = rng.permutation(len(names))
        current = []
        previous = baseline
        for feature in order:
            current.append(int(feature))
            now = regret_of(current)
            contribution[feature] += previous - now
            previous = now
        if (step + 1) % 20 == 0:
            print(f"  {step + 1} permutations done", flush=True)
    contribution /= permutations

    print(f"\n{'feature':>22s} {'regret reduction':>18s}  share")
    for i in np.argsort(-contribution):
        share = 100 * contribution[i] / (baseline - full)
        print(f"{names[i]:>22s} {contribution[i]:>18.5f}  {share:5.1f}%")
    print(f"\n{'sum':>22s} {contribution.sum():>18.5f}  (total reduction {baseline - full:.5f})")


if __name__ == "__main__":
    main()
