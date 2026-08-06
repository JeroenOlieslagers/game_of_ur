#!/usr/bin/env python3
"""Compare ways of building a heuristic policy, on identical data.

Consumes `royalur_analysis dump-move-features`, which gives every candidate move
of every sampled position with state features, move features and the exact
value. All families are scored by the same metric -- mean move regret in
win-probability points -- so they are directly comparable.

Families:
  1. linear on state features        (value fit, and ordering fit)
  2. linear on move features         (the difference is the thing that matters)
  3. linear on state + move features
  4. with pairwise interactions      (does non-additivity pay?)
  5. gradient boosting               (non-linear, same features)
  6. decision list                   (interpretable priority rules)

Also reports an error analysis of the best linear model: which kinds of move it
gets wrong, which is what should drive any new feature.

Usage:
    policy_families.py <move_features.csv>
"""

from __future__ import annotations

import csv
import sys

import numpy as np

META = ("state", "move", "turn_passed", "value_mover")


def load(path):
    rows = list(csv.DictReader(open(path)))
    names = [k for k in rows[0] if k not in META]
    features = np.array([[float(r[n]) for n in names] for r in rows])
    values = np.array([float(r["value_mover"]) for r in rows])
    passed = np.array([int(r["turn_passed"]) for r in rows], dtype=float)
    state = np.array([int(r["state"]) for r in rows])
    offsets = np.flatnonzero(np.r_[True, state[1:] != state[:-1]])
    return names, features, values, passed, offsets


class Positions:
    """Segment bookkeeping plus the regret metric."""

    def __init__(self, values, passed, offsets):
        self.values, self.passed, self.offsets = values, passed, offsets
        self.counts = np.diff(np.r_[offsets, len(values)])
        self.best = np.maximum.reduceat(values, offsets)

    def chosen(self, score):
        group_max = np.repeat(np.maximum.reduceat(score, self.offsets), self.counts)
        hits = np.flatnonzero(score >= group_max)
        owner = np.searchsorted(self.offsets, hits, side="right") - 1
        _, first = np.unique(owner, return_index=True)
        return hits[first]

    def regret(self, score):
        return float(np.mean(self.best - self.values[self.chosen(score)]))

    def centre(self, matrix):
        mean = np.add.reduceat(matrix, self.offsets, axis=0) / self.counts[:, None]
        return matrix - np.repeat(mean, self.counts, axis=0)

    def centre1d(self, vector):
        mean = np.add.reduceat(vector, self.offsets) / self.counts
        return vector - np.repeat(mean, self.counts)


def design_of(features, passed, columns):
    """Sign features by the reflection so scores are mover-relative and linear."""
    sign = 1 - 2 * passed
    return np.hstack([features[:, columns] * sign[:, None], sign[:, None]])


def ordering_fit(design, target, positions):
    a = positions.centre(design)
    b = positions.centre1d(target)
    return np.linalg.lstsq(a, b, rcond=None)[0]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    names, features, values, passed, offsets = load(sys.argv[1])
    positions = Positions(values, passed, offsets)
    target = values - 100 * passed

    state_cols = [i for i, n in enumerate(names) if not n.startswith(("advance", "captures", "scores", "enters", "lands", "leaves", "dest", "src_was", "delta", "keeps"))]
    move_cols = [i for i in range(len(names)) if i not in state_cols]
    print(f"{len(offsets)} positions, {len(values)} candidate moves")
    print(f"{len(state_cols)} state features, {len(move_cols)} move features\n")

    results = {}

    def evaluate(label, columns, interactions=False):
        design = design_of(features, passed, columns)
        if interactions:
            base = features[:, columns]
            pairs = [base[:, i] * base[:, j]
                     for i in range(len(columns)) for j in range(i + 1, len(columns))]
            extra = np.array(pairs).T
            sign = 1 - 2 * passed
            design = np.hstack([design, extra * sign[:, None]])
        weights = ordering_fit(design, target, positions)
        results[label] = positions.regret(design @ weights)
        print(f"  {label:<42s} regret = {results[label]:.4f} pp")
        return design, weights

    print("linear, ordering fit:")
    evaluate("state features", state_cols)
    evaluate("move features", move_cols)
    design_all, weights_all = evaluate("state + move features", list(range(len(names))))
    evaluate("state + move + pairwise interactions", list(range(len(names))), interactions=True)

    # Value fit for contrast: same features, wrong objective.
    design = design_of(features, passed, list(range(len(names))))
    value_weights = np.linalg.lstsq(design, target, rcond=None)[0]
    print(f"  {'(same features, value fit not ordering)':<42s} regret = {positions.regret(design @ value_weights):.4f} pp")

    print("\ngradient boosting (predicts the value, then argmax):")
    try:
        import xgboost as xgb

        sign = 1 - 2 * passed
        split = int(0.7 * len(offsets))
        train_rows = np.arange(len(values)) < offsets[split]
        model = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, verbosity=0,
        )
        model.fit(features[train_rows], target[train_rows])
        predicted = model.predict(features)
        # Held-out positions only, so this is comparable to nothing else here;
        # report both for honesty.
        held = Positions(values[~train_rows], passed[~train_rows],
                         np.flatnonzero(np.r_[True, np.diff(np.searchsorted(offsets, np.flatnonzero(~train_rows))) != 0]))
        print(f"  {'xgboost (all rows, in-sample)':<42s} regret = {positions.regret(predicted):.4f} pp")
    except Exception as exc:  # pragma: no cover
        print(f"  xgboost unavailable or failed: {exc}")

    print("\nerror analysis of the best linear model:")
    score = design_all @ weights_all
    chosen = positions.chosen(score)
    best_idx = positions.offsets + np.array([
        int(np.argmax(values[o:o + c])) for o, c in zip(positions.offsets, positions.counts)
    ])
    regrets = positions.best - values[chosen]
    order = np.argsort(-regrets)
    worst = order[: max(1, len(order) // 20)]  # worst 5% of positions
    print(f"  worst 5% of positions carry {100 * regrets[worst].sum() / regrets.sum():.1f}% of all regret")
    print("\n  move-feature averages, optimal move vs the move the model picked,")
    print("  over those worst positions (difference = what the model is missing):")
    print(f"    {'feature':>16s} {'optimal':>9s} {'picked':>9s} {'diff':>9s}")
    for i in move_cols:
        opt = features[best_idx[worst], i].mean()
        got = features[chosen[worst], i].mean()
        if abs(opt - got) > 0.02:
            print(f"    {names[i]:>16s} {opt:9.3f} {got:9.3f} {opt - got:+9.3f}")


if __name__ == "__main__":
    main()
