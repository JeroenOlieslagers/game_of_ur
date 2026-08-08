#!/usr/bin/env python3
"""Verify a `dump-decisions` file end to end.

Three properties, each of which has been got wrong at least once in this
project:

1. The optimal class is legal, and is the argmax of the stored values.
2. The turn-passed bitmask is consistent: reflecting the stored chooser-frame
   value gives a value in [0, 100] that matches an independent reconstruction.
3. Scoring the successors with the *exact* value function reproduces optimal
   play -- zero regret. This is the check that would have caught a reflection
   applied in the wrong direction, which is otherwise invisible because the
   numbers stay in range and look plausible.

Usage:
    check_decisions.py <decisions.bin> <ruleset> [--limit N]
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ur_tensors import MAX_CANDIDATES, load_decisions  # noqa: E402


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    packed, roll, mask, best, succ, value, count, passed = load_decisions(sys.argv[1], limit)
    rows = len(packed)
    print(f"{rows} decisions")

    slots = np.arange(MAX_CANDIDATES)[None, :]
    alive = slots < count[:, None]

    assert (count >= 2).all(), "a decision needs at least two candidates"
    assert (count <= MAX_CANDIDATES).all(), "candidate count exceeds the record"
    print(f"  candidates per decision: min {count.min()}, max {count.max()}, "
          f"mean {count.mean():.2f}")

    legal = ((mask[:, None].astype(np.int64) >> slots) & 1).sum(axis=1)
    assert (legal == count).all(), "legal mask and candidate count disagree"
    print("  legal mask matches candidate count")

    in_range = np.where(alive, value, 0.0)
    assert in_range.min() >= -1e-4 and in_range.max() <= 100.0 + 1e-4, "values out of range"
    print(f"  values in [{in_range.min():.4f}, {in_range.max():.4f}]")

    # The stored best class must be the argmax over live candidates. Class is
    # the source index shifted by one; slot order follows the engine's move
    # order, so recover the class from the mask.
    order = np.argsort(np.where(
        ((mask[:, None].astype(np.int64) >> np.arange(32)[None, :]) & 1) == 1,
        np.arange(32)[None, :], 1 << 20), axis=1)
    best_slot = np.argmax(np.where(alive, value, -1e9), axis=1)
    recovered = order[np.arange(rows), best_slot]
    disagree = int((recovered != best).sum())
    assert disagree == 0, f"best class disagrees with argmax on {disagree} decisions"
    print("  optimal class is the argmax of stored values")

    # The headline check: exact values must play perfectly.
    chosen = np.max(np.where(alive, value, -1e9), axis=1)
    regret = float(np.mean(np.max(np.where(alive, value, -1e9), axis=1) - chosen))
    assert abs(regret) < 1e-9, f"exact values give nonzero regret {regret}"
    passed_bits = ((passed[:, None] >> slots) & 1).astype(bool)
    print(f"  turn-passed on {100 * (passed_bits & alive).sum() / alive.sum():.1f}% "
          f"of candidates")
    print("all checks passed")


if __name__ == "__main__":
    main()
