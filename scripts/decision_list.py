#!/usr/bin/env python3
"""Learn an interpretable priority list of rules for choosing a move.

A decision list picks a move the way a person describes strategy: try the first
rule; if any candidate move satisfies it, discard the rest and move on to the
next rule to break the tie; continue until one move remains.

Rules are learned greedily -- at each slot, append whichever predicate most
reduces mean regret. The result is a short, readable strategy whose strength can
be stated exactly in win-probability points, which is the point: unlike a weight
vector, it can be written in a paper as prose.

Also reports which pairwise interactions carry weight in the linear model,
ranked by how much they actually move the score rather than by raw coefficient.

Usage:
    decision_list.py <move_features.csv> [--rules N]
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
    state = np.array([int(r["state"]) for r in rows])
    passed = np.array([int(r["turn_passed"]) for r in rows], dtype=float)
    offsets = np.flatnonzero(np.r_[True, state[1:] != state[:-1]])
    return names, features, values, passed, offsets


def build_predicates(names, features):
    """Binary tests on a move, each of which a person could state in words."""
    predicates = []
    for name in ("scores", "captures", "lands_rosette", "lands_centre", "enters",
                 "dest_safe", "keeps_turn", "leaves_centre", "src_was_exposed"):
        if name in names:
            column = features[:, names.index(name)]
            predicates.append((f"{name}", column > 0.5))
            predicates.append((f"not {name}", column <= 0.5))
    if "advance" in names:
        column = features[:, names.index("advance")]
        for threshold in (2, 3, 4):
            predicates.append((f"advance >= {threshold}", column >= threshold))
    for name in ("delta_exposure", "delta_threat"):
        if name in names:
            column = features[:, names.index(name)]
            predicates.append((f"{name} <= 0", column <= 0))
            predicates.append((f"{name} > 0", column > 0))
    return predicates


def apply_list(rules, alive, offsets, counts):
    """Filter candidates rule by rule, keeping a rule only where it leaves some
    move alive in that position."""
    for _, mask in rules:
        candidate = alive & mask
        kept = np.repeat(np.add.reduceat(candidate.astype(np.int64), offsets), counts)
        alive = np.where(kept > 0, candidate, alive)
    return alive


def regret_of(alive, values, best, offsets, counts):
    # Ties break on the first surviving move, matching the engine.
    index = np.arange(len(values))
    big = index.max() + 1
    pick = np.minimum.reduceat(np.where(alive, index, big), offsets)
    return float(np.mean(best - values[pick]))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    limit = 6
    if "--rules" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--rules") + 1])

    names, features, values, passed, offsets = load(sys.argv[1])
    counts = np.diff(np.r_[offsets, len(values)])
    best = np.maximum.reduceat(values, offsets)
    predicates = build_predicates(names, features)

    print(f"{len(offsets)} positions, {len(values)} candidate moves, "
          f"{len(predicates)} candidate rules\n")

    alive0 = np.ones(len(values), dtype=bool)
    print(f"no rules (first legal move): regret = {regret_of(alive0, values, best, offsets, counts):.4f} pp\n")

    chosen: list = []
    for slot in range(limit):
        scored = []
        for name, mask in predicates:
            if any(name == existing for existing, _ in chosen):
                continue
            alive = apply_list(chosen + [(name, mask)], alive0, offsets, counts)
            scored.append((regret_of(alive, values, best, offsets, counts), name, mask))
        scored.sort(key=lambda item: item[0])
        regret, name, mask = scored[0]
        chosen.append((name, mask))
        print(f"  rule {slot + 1}: prefer moves where {name:<22s} -> regret = {regret:.4f} pp")

    print("\nfinal decision list, in order:")
    for index, (name, _) in enumerate(chosen, 1):
        print(f"  {index}. prefer moves where {name}")

    # Which interactions actually matter in the linear model?
    print("\npairwise interactions, ranked by how much they move the score:")
    sign = 1 - 2 * passed
    base = features
    design = [base * sign[:, None], sign[:, None]]
    labels = list(names) + ["intercept"]
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((f"{names[i]} x {names[j]}", base[:, i] * base[:, j]))
    design.append(np.array([column for _, column in pairs]).T * sign[:, None])
    labels += [name for name, _ in pairs]
    matrix = np.hstack(design)

    mean = np.add.reduceat(matrix, offsets, axis=0) / counts[:, None]
    centred = matrix - np.repeat(mean, counts, axis=0)
    target = values - 100 * passed
    target_centred = target - np.repeat(np.add.reduceat(target, offsets) / counts, counts)
    weights = np.linalg.lstsq(centred, target_centred, rcond=None)[0]

    # A coefficient is only meaningful next to the spread of its column: what
    # matters is how much the term varies between sibling moves.
    spread = centred.std(axis=0)
    influence = np.abs(weights) * spread
    order = np.argsort(-influence)
    for rank in order[:12]:
        kind = "interaction" if " x " in labels[rank] else "main effect"
        print(f"  {labels[rank]:<44s} {influence[rank]:7.3f}  ({kind})")


if __name__ == "__main__":
    main()
