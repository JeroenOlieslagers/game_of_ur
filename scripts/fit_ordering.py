#!/usr/bin/env python3
"""Fit an evaluation function to move ORDERING rather than to value.

Least squares on values minimises the wrong thing for move choice. What decides
a move is the ordering of scores among one position's successors, so anything
constant across those siblings is irrelevant however much variance it explains.

The within-position transform subtracts each position's mean from the features
and the target, which annihilates exactly those constant components. It is still
one ordinary least-squares call.

Reflection is handled explicitly: a move that passes the turn hands the position
to the opponent, so its score is reflected about 50. Writing s = 1 - 2*passed,
the mover-relative score is `s*(f.w) + s*b + 100*passed`, so signing the features
by s and moving the known 100*passed to the target makes the whole thing linear.

Usage:
    fit_ordering.py <moves.csv> [weights-out.txt]
"""
from __future__ import annotations
import csv, sys
import numpy as np


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    rows = list(csv.DictReader(open(sys.argv[1])))
    names = [k for k in rows[0] if k not in ("state", "move", "turn_passed", "value_mover")]
    features = np.array([[float(r[n]) for n in names] for r in rows])
    values = np.array([float(r["value_mover"]) for r in rows])
    passed = np.array([int(r["turn_passed"]) for r in rows], dtype=float)
    state = np.array([int(r["state"]) for r in rows])
    offsets = np.flatnonzero(np.r_[True, state[1:] != state[:-1]])
    counts = np.diff(np.r_[offsets, len(state)])
    best = np.maximum.reduceat(values, offsets)

    sign = 1 - 2 * passed
    design = np.hstack([features * sign[:, None], sign[:, None]])
    target = values - 100 * passed

    def regret(weights):
        # Add back the 100 * passed constant that the fit moved to the target
        # side; it differs between sibling moves, so it cannot be dropped.
        score = design @ weights + 100 * passed
        group_max = np.repeat(np.maximum.reduceat(score, offsets), counts)
        hits = np.flatnonzero(score >= group_max)
        owner = np.searchsorted(offsets, hits, side="right") - 1
        _, first = np.unique(owner, return_index=True)
        return float(np.mean(best - values[hits[first]]))

    plain, *_ = np.linalg.lstsq(design, target, rcond=None)
    mean_design = np.repeat(np.add.reduceat(design, offsets, axis=0) / counts[:, None], counts, axis=0)
    mean_target = np.repeat(np.add.reduceat(target, offsets) / counts, counts)
    centred, *_ = np.linalg.lstsq(design - mean_design, target - mean_target, rcond=None)

    print(f"{len(offsets)} positions, {len(rows)} candidate moves\n")
    print(f"least squares on values     : regret = {regret(plain):.4f} pp")
    print(f"within-position centred fit : regret = {regret(centred):.4f} pp")
    print(f"constant score (random)     : regret = {regret(np.zeros(design.shape[1])):.4f} pp")

    scale = abs(plain[names.index("advancement_self")])
    print("\nweights in units of advancement_self:")
    print(f"  {'feature':>18s} {'plain':>9s} {'centred':>9s}")
    for name, a, b in sorted(zip(names, plain[:-1], centred[:-1]), key=lambda t: -abs(t[2])):
        print(f"  {name:>18s} {a / scale:+9.1f} {b / scale:+9.1f}")

    if len(sys.argv) > 2:
        # Same layout the Rust `regret` command expects: features then intercept.
        open(sys.argv[2], "w").write(
            "# fitted by scripts/fit_ordering.py (within-position centred)\n"
            f"# order: {','.join(names)},intercept\n"
            + "\n".join(f"{w:.10f}" for w in centred) + "\n")
        print(f"\nwrote {sys.argv[2]}")


if __name__ == "__main__":
    main()
