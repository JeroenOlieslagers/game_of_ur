#!/usr/bin/env python3
"""N-tuple network, and higher-order interactions, scored by move regret.

An N-tuple network gives every *configuration* of a small set of board
positions its own weight, and sums those weights over a bank of such tuples. It
is still a linear model -- just in a very large one-hot feature space -- so the
same within-position ordering fit applies unchanged, and it captures piece
*interactions* that scalar features cannot express at all.

Tuples here are windows along the mover's own path. A window of k positions has
3^k configurations (empty / mine / theirs), so a bank of overlapping windows
costs a few hundred weights.

Also sweeps polynomial interaction order on the scalar features, to see how far
beyond pairwise it is worth going.

Usage:
    ntuple.py <move_features_with_occupancy.csv>
"""

from __future__ import annotations

import csv
import itertools
import sys

import numpy as np

META = ("state", "move", "turn_passed", "value_mover", "occupancy")


def load(path):
    rows = list(csv.DictReader(open(path)))
    names = [k for k in rows[0] if k not in META]
    features = np.array([[float(r[n]) for n in names] for r in rows])
    values = np.array([float(r["value_mover"]) for r in rows])
    passed = np.array([int(r["turn_passed"]) for r in rows], dtype=float)
    occupancy = np.array([[int(c) for c in r["occupancy"]] for r in rows], dtype=np.int8)
    state = np.array([int(r["state"]) for r in rows])
    offsets = np.flatnonzero(np.r_[True, state[1:] != state[:-1]])
    return names, features, values, passed, occupancy, offsets


class Positions:
    def __init__(self, values, offsets, passed):
        self.values, self.offsets, self.passed = values, offsets, passed
        self.counts = np.diff(np.r_[offsets, len(values)])
        self.best = np.maximum.reduceat(values, offsets)
        self._owner = None

    def regret(self, score):
        # See scripts/policy_families.py: the 100 * passed offset must be added
        # back before sibling moves are comparable.
        score = score + 100 * self.passed
        group_max = np.repeat(np.maximum.reduceat(score, self.offsets), self.counts)
        hits = np.flatnonzero(score >= group_max)
        owner = np.searchsorted(self.offsets, hits, side="right") - 1
        _, first = np.unique(owner, return_index=True)
        return float(np.mean(self.best - self.values[hits[first]]))

    def centre(self, matrix):
        mean = np.add.reduceat(matrix, self.offsets, axis=0) / self.counts[:, None]
        return matrix - np.repeat(mean, self.counts, axis=0)

    def position_mean(self, matrix):
        """Per-position column means: one row per position, not per move."""
        return np.add.reduceat(matrix, self.offsets, axis=0) / self.counts[:, None]

    def owner(self):
        """Which position each row belongs to."""
        if self._owner is None:
            self._owner = np.repeat(np.arange(len(self.offsets)), self.counts)
        return self._owner

    def centre1d(self, vector):
        mean = np.add.reduceat(vector, self.offsets) / self.counts
        return vector - np.repeat(mean, self.counts)


def fit_and_score(design, target, positions, ridge=1e-6):
    """Ordering fit, solved from the Gram matrix in row blocks.

    A one-hot N-tuple design is wide, so materialising a second centred copy of
    it costs gigabytes. Accumulating X'X and X'y blockwise keeps memory
    proportional to columns^2 instead of rows x columns. The ridge keeps the
    system solvable when some configurations never occur in the data.
    """
    columns = design.shape[1]
    gram = np.zeros((columns, columns))
    moment = np.zeros(columns)
    centred_target = positions.centre1d(target)
    # Per-POSITION means, so the stored array is a third the size of the design,
    # and blocks are centred by indexing rather than by materialising a second
    # full-size copy.
    means = positions.position_mean(design)
    owner = positions.owner()
    block = 20000
    for start in range(0, len(design), block):
        stop = min(start + block, len(design))
        chunk = design[start:stop] - means[owner[start:stop]]
        gram += chunk.T @ chunk
        moment += chunk.T @ centred_target[start:stop]
    gram += ridge * np.eye(columns)
    weights = np.linalg.solve(gram, moment)
    return positions.regret(design @ weights), weights


def tuple_design(occupancy, windows, sign):
    """One-hot the configuration of each window. Columns are (window, config)."""
    blocks = []
    for window in windows:
        width = len(window)
        code = np.zeros(len(occupancy), dtype=np.int64)
        for position in window:
            code = code * 3 + occupancy[:, position]
        onehot = np.zeros((len(occupancy), 3 ** width), dtype=np.float64)
        onehot[np.arange(len(occupancy)), code] = 1.0
        blocks.append(onehot * sign[:, None])
    return np.hstack(blocks)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    names, features, values, passed, occupancy, offsets = load(sys.argv[1])
    positions = Positions(values, offsets, passed)
    target = values - 100 * passed
    sign = 1 - 2 * passed
    path_len = occupancy.shape[1]
    print(f"{len(offsets)} positions, {len(values)} candidate moves, path length {path_len}\n")

    scalar = np.hstack([features * sign[:, None], sign[:, None]])
    base, _ = fit_and_score(scalar, target, positions)
    print(f"scalar features, additive           : regret = {base:.4f} pp  ({scalar.shape[1]} weights)")

    print("\nN-tuple networks (sliding windows along the path):")
    for width in (2, 3, 4):
        windows = [tuple(range(start, start + width)) for start in range(path_len - width + 1)]
        design = np.hstack([tuple_design(occupancy, windows, sign), sign[:, None]])
        regret, _ = fit_and_score(design, target, positions)
        print(f"  {len(windows)} windows of width {width:<2d}          : regret = {regret:.4f} pp  ({design.shape[1]} weights)")

    print("\nN-tuple plus the scalar features:")
    for width in (3, 4):
        windows = [tuple(range(start, start + width)) for start in range(path_len - width + 1)]
        design = np.hstack([tuple_design(occupancy, windows, sign), scalar])
        regret, _ = fit_and_score(design, target, positions)
        print(f"  width {width} + scalars                 : regret = {regret:.4f} pp  ({design.shape[1]} weights)")

    print("\npolynomial interaction order on the scalar features:")
    # Restrict to the most influential features, or the column count explodes.
    _, weights = fit_and_score(scalar, target, positions)
    spread = positions.centre(scalar).std(axis=0)
    influence = np.abs(weights) * spread
    top = np.argsort(-influence[: len(names)])[:10]
    print(f"  (using the 10 most influential: {', '.join(names[i] for i in top[:4])}, ...)")
    for order in (1, 2, 3, 4):
        columns = [features[:, list(combo)].prod(axis=1)
                   for size in range(1, order + 1)
                   for combo in itertools.combinations(top, size)]
        design = np.hstack([np.array(columns).T * sign[:, None], sign[:, None]])
        regret, _ = fit_and_score(design, target, positions)
        print(f"  order {order}                             : regret = {regret:.4f} pp  ({design.shape[1]} weights)")


if __name__ == "__main__":
    main()
