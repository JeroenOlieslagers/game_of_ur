#!/usr/bin/env python3
"""Fit every linear model family on one dataset and report them side by side.

All families are scored by the same metric -- mean move regret in
win-probability points -- computed the same way, so the numbers are comparable.

Two things the scoring must get right, both of which have bitten this analysis:

  * A score estimates the mover's value MINUS 100 for a move that passes the
    turn, because the fit moves that constant to the target side. It has to be
    added back before siblings are compared: `passed` differs between the
    candidate moves of a position, so dropping it penalises every turn-passing
    move by 100 and the policy grabs rosettes.
  * Features are signed by `s = 1 - 2 * passed`, which is the reflection about 50
    that relates the two players' perspectives.

Usage:
    model_roster.py <move_features.csv> [--out table.csv]
"""

from __future__ import annotations

import csv
import sys

import numpy as np

META = ("state", "move", "turn_passed", "value_mover", "occupancy")

# Feature groups, so families can be assembled by name rather than by index.
MOVE_MAGNITUDE = ["capture_value", "rescue_value", "delta_exposure_value", "delta_threat_value"]
MOVE_STRUCTURAL = ["captures_frontmost", "capture_gap_to_front", "moves_frontmost",
                   "becomes_safe_forever", "contact_possible", "threat_count"]
MOVE_BASE = ["advance", "captures", "scores", "enters", "lands_rosette", "lands_centre",
             "leaves_centre", "dest_safe", "src_was_exposed", "delta_exposure",
             "delta_threat", "keeps_turn"]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    rows = list(csv.DictReader(open(sys.argv[1])))
    names = [k for k in rows[0] if k not in META]
    state_names = [n for n in names if n not in MOVE_BASE + MOVE_MAGNITUDE + MOVE_STRUCTURAL]

    features = {n: np.array([float(r[n]) for r in rows]) for n in names}
    values = np.array([float(r["value_mover"]) for r in rows])
    passed = np.array([int(r["turn_passed"]) for r in rows], dtype=float)
    state = np.array([int(r["state"]) for r in rows])
    offsets = np.flatnonzero(np.r_[True, state[1:] != state[:-1]])
    counts = np.diff(np.r_[offsets, len(state)])
    owner = np.repeat(np.arange(len(offsets)), counts)
    best = np.maximum.reduceat(values, offsets)
    sign = 1 - 2 * passed
    target = values - 100 * passed          # what a score estimates
    optimal = offsets + np.array([int(np.argmax(values[o:o + c]))
                                  for o, c in zip(offsets, counts)])

    def design_of(selected, interactions):
        columns = [features[n] for n in selected]
        block = [np.array(columns).T * sign[:, None]]
        if interactions:
            products = [columns[i] * columns[j]
                        for i in range(len(columns)) for j in range(i + 1, len(columns))]
            block.append(np.array(products).T * sign[:, None])
        block.append(sign[:, None])
        return np.hstack(block)

    def evaluate(design, ordering=True):
        if ordering:
            mean = np.add.reduceat(design, offsets, axis=0) / counts[:, None]
            mean_target = np.add.reduceat(target, offsets) / counts
            weights = np.linalg.lstsq(design - mean[owner], target - mean_target[owner],
                                      rcond=None)[0]
        else:
            weights = np.linalg.lstsq(design, target, rcond=None)[0]
        predicted = design @ weights
        # R^2 against the successor value, for reference.
        residual = target - predicted
        r2 = 1 - (residual ** 2).sum() / ((target - target.mean()) ** 2).sum()
        # Add the constant back before comparing siblings.
        score = predicted + 100 * passed
        group_max = np.repeat(np.maximum.reduceat(score, offsets), counts)
        hits = np.flatnonzero(score >= group_max)
        who = np.searchsorted(offsets, hits, side="right") - 1
        _, first = np.unique(who, return_index=True)
        chosen = hits[first]
        regret = best - values[chosen]
        agreement = 100.0 * np.mean(chosen == optimal)
        return design.shape[1], r2, float(regret.mean()), float(np.percentile(regret, 95)), agreement

    move_all = MOVE_BASE + MOVE_MAGNITUDE + MOVE_STRUCTURAL

    # Interaction order is capped by column count: with k features, order 3 adds
    # C(k,3) columns. Beyond ~20 features that is solved through the Gram matrix
    # rather than a dense design, so triples are restricted to the most
    # influential features, chosen by |weight| x within-position spread from the
    # additive fit.
    def top_features(selected, keep):
        design = design_of(selected, False)
        mean = np.add.reduceat(design, offsets, axis=0) / counts[:, None]
        mean_target = np.add.reduceat(target, offsets) / counts
        centred = design - mean[owner]
        weights = np.linalg.lstsq(centred, target - mean_target[owner], rcond=None)[0]
        influence = np.abs(weights[:len(selected)]) * centred[:, :len(selected)].std(axis=0)
        order = np.argsort(-influence)[:keep]
        return [selected[i] for i in order]

    roster = [
        ("state features, value fit", state_names, False, False),
        ("state features", state_names, False, True),
        ("move features (original 12)", MOVE_BASE, False, True),
        ("move features (all 22)", move_all, False, True),
        ("state + move (original)", state_names + MOVE_BASE, False, True),
        ("state + move (all)", state_names + move_all, False, True),
        ("state + move (original) + pairwise", state_names + MOVE_BASE, True, True),
        ("state + move (all) + pairwise", state_names + move_all, True, True),
    ]

    print(f"{len(offsets)} positions, {len(values)} candidate moves\n")

    # Baselines. "first legal move" is NOT neutral: the engine lists the scoring
    # move first, then entering from hand, then pieces by increasing path
    # position, so it encodes a real strategy ("score, else enter, else move the
    # least advanced piece"). Uniform random is the honest zero point.
    index = np.arange(len(values))
    first = np.minimum.reduceat(index, offsets)
    rng = np.random.default_rng(0)
    random_pick = offsets + np.array([rng.integers(c) for c in counts])
    print(f"{'baseline':<38s} {'params':>7s} {'R^2':>7s} {'regret':>8s} {'p95':>7s} {'agree%':>7s}")
    print("-" * 80)
    for label, pick in [("uniform random", random_pick), ("first legal move (score/enter/least)", first)]:
        regret = best - values[pick]
        print(f"{label:<38s} {0:>7d} {0.0:>7.4f} {regret.mean():>8.4f} "
              f"{np.percentile(regret, 95):>7.3f} {100.0 * np.mean(pick == optimal):>7.2f}")
    print()
    print(f"{'model':<38s} {'params':>7s} {'R^2':>7s} {'regret':>8s} {'p95':>7s} {'agree%':>7s}")
    print("-" * 80)
    results = []
    for label, selected, interactions, ordering in roster:
        design = design_of(selected, interactions)
        params, r2, regret, p95, agreement = evaluate(design, ordering)
        results.append((label, params, r2, regret, p95, agreement))
        print(f"{label:<38s} {params:>7d} {r2:>7.4f} {regret:>8.4f} {p95:>7.3f} {agreement:>7.2f}")

    # Interaction order, on the most influential features so the column count
    # stays tractable. This supersedes an earlier sweep that used only 10
    # features and excluded the move features entirely.
    print("\ninteraction order (on the 14 most influential of all features):")
    import itertools
    chosen = top_features(state_names + move_all, 14)
    for order in (1, 2, 3):
        columns = [np.prod([features[n] for n in combo], axis=0)
                   for size in range(1, order + 1)
                   for combo in itertools.combinations(chosen, size)]
        design = np.hstack([np.array(columns).T * sign[:, None], sign[:, None]])
        params, r2, regret, p95, agreement = evaluate(design, True)
        print(f"  {'order ' + str(order):<36s} {params:>7d} {r2:>7.4f} {regret:>8.4f} "
              f"{p95:>7.3f} {agreement:>7.2f}")

    if "--out" in sys.argv:
        path = sys.argv[sys.argv.index("--out") + 1]
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model", "params", "r2", "mean_regret", "p95_regret", "agreement_pct"])
            writer.writerows(results)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
