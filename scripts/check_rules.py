#!/usr/bin/env python3
"""Python side of the rule-evaluator cross-check.

The DSL is implemented twice -- `rule_dsl.py` here and a recursive-descent
parser in `rust/src/main.rs` -- because a decision list has to *play* to get a
win rate, and playing happens in the engine. Two implementations of one grammar
is a place where a silent divergence produces a plausible-but-wrong number
rather than an error, so `royalur_analysis check-rules` and this script print
the same quantities over the same file and the two are diffed.

Usage:
    check_rules.py <rules.txt> <move_features.csv>
"""

from __future__ import annotations

import csv
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rule_dsl import evaluate  # noqa: E402

META = ("state", "move", "turn_passed", "value_mover", "occupancy")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    rules = [line.strip() for line in open(sys.argv[1])
             if line.strip() and not line.startswith("#")]
    rows = list(csv.DictReader(open(sys.argv[2])))
    names = [k for k in rows[0] if k not in META]
    columns = {n: np.array([float(r[n]) for r in rows]) for n in names}
    values = np.array([float(r["value_mover"]) for r in rows])
    state = np.array([int(r["state"]) for r in rows])
    offsets = np.flatnonzero(np.r_[True, state[1:] != state[:-1]])
    counts = np.diff(np.r_[offsets, len(state)])
    best = np.maximum.reduceat(values, offsets)

    print(f"{len(rows)} rows, {len(rules)} rules")
    masks = []
    for text in rules:
        mask = evaluate(text, columns)
        masks.append(mask)
        print(f"  holds={int(mask.sum()):>8}   {text}")

    alive = np.ones(len(rows), dtype=bool)
    for mask in masks:
        candidate = alive & mask
        kept = np.repeat(np.add.reduceat(candidate.astype(np.int64), offsets), counts)
        alive = np.where(kept > 0, candidate, alive)

    index = np.arange(len(values))
    pick = np.minimum.reduceat(np.where(alive, index, index.max() + 1), offsets)
    print(f"positions={len(offsets)} mean_regret={np.mean(best - values[pick]):.6f}")


if __name__ == "__main__":
    main()
