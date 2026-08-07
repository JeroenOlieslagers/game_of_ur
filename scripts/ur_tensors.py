#!/usr/bin/env python3
"""Shared plumbing for stage 2: unpack the Rust dumps into model inputs.

`royalur_analysis dump-tensors` writes 12 bytes per state -- a `u64` packed
position then an `f32` win percentage for the player to move -- and
`dump-successors` writes the evaluation set as sibling groups. Both use the same
packing, described in `pack_position`:

    bits  0..47   24 board slots, two bits each: 0 empty, 1 mover, 2 other
    bits 48..50   mover's pieces in hand
    bits 51..53   other player's pieces in hand
    bits 54..57   mover's score
    bits 58..61   other player's score

Positions are canonicalised to light-to-move before packing, so board index `t`
always means the same square of the *mover's* geometry and the self/other
symmetry costs a model nothing to represent.

The reflection rule that governs everything here: a value from the mover's
perspective converts to the other player's by `v -> 100 - v`, never by negation.
A model scores the successor from the successor mover's perspective, so the
chooser's value is `100 - score` exactly when the move passed the turn.
"""

from __future__ import annotations

import numpy as np

BOARD_LEN = 24

PATHS = {
    "blitz": ([9, 6, 3, 0, 1, 4, 7, 10, 13, 16, 19, 20, 23, 22, 21, 18],
              [11, 8, 5, 2, 1, 4, 7, 10, 13, 16, 19, 18, 21, 22, 23, 20]),
    "masters": ([9, 6, 3, 0, 1, 4, 7, 10, 13, 16, 19, 20, 23, 22, 21, 18],
                [11, 8, 5, 2, 1, 4, 7, 10, 13, 16, 19, 18, 21, 22, 23, 20]),
    "finkel": ([9, 6, 3, 0, 1, 4, 7, 10, 13, 16, 19, 22, 21, 18],
               [11, 8, 5, 2, 1, 4, 7, 10, 13, 16, 19, 22, 23, 20]),
}
PIECES = {"blitz": 5, "masters": 7, "finkel": 7}


def usable_tiles(ruleset: str) -> list[int]:
    """The 20 board slots either player can occupy, in ascending index order."""
    light, dark = PATHS[ruleset]
    return sorted(set(light) | set(dark))


def unpack(packed: np.ndarray, ruleset: str) -> np.ndarray:
    """Packed words -> float32 design matrix, one row per position.

    Layout: 20 tiles one-hot over {empty, mover, other} = 60, then both hands
    one-hot over 0..pieces. Scores are omitted because they are determined by
    the rest (`score = pieces - hand - on_board`), so including them would only
    add collinear inputs.

    Input *order* is irrelevant to a dense net -- every slot gets its own weight
    -- so tiles stay board-indexed here. Path ordering matters only to models
    that exploit adjacency, which build their own views.
    """
    packed = np.asarray(packed, dtype=np.uint64)
    tiles = usable_tiles(ruleset)
    pieces = PIECES[ruleset]
    rows = len(packed)
    hand_width = pieces + 1
    out = np.zeros((rows, 3 * len(tiles) + 2 * hand_width), dtype=np.float32)

    for slot, tile in enumerate(tiles):
        code = ((packed >> np.uint64(2 * tile)) & np.uint64(3)).astype(np.int64)
        out[np.arange(rows), 3 * slot + code] = 1.0

    base = 3 * len(tiles)
    mover_hand = ((packed >> np.uint64(48)) & np.uint64(7)).astype(np.int64)
    other_hand = ((packed >> np.uint64(51)) & np.uint64(7)).astype(np.int64)
    out[np.arange(rows), base + np.clip(mover_hand, 0, pieces)] = 1.0
    out[np.arange(rows), base + hand_width + np.clip(other_hand, 0, pieces)] = 1.0
    return out


def input_width(ruleset: str) -> int:
    return 3 * len(usable_tiles(ruleset)) + 2 * (PIECES[ruleset] + 1)


def path_occupancy(packed: np.ndarray, ruleset: str) -> np.ndarray:
    """Occupancy indexed along each player's own path, as int8 in {0,1,2}.

    Columns are the mover's path then the other player's, so column `k` means
    "path position k" for the respective player regardless of colour. This is
    the frame an N-tuple network wants, because adjacency along a path is what
    makes a tuple of tiles meaningful.
    """
    packed = np.asarray(packed, dtype=np.uint64)
    light, dark = PATHS[ruleset]
    columns = []
    for tile in list(light) + list(dark):
        columns.append(((packed >> np.uint64(2 * tile)) & np.uint64(3)).astype(np.int8))
    return np.stack(columns, axis=1)


def load_decisions(path: str, limit: int | None = None):
    """Memory-map a `dump-decisions` file: every decision in the game.

    Returns (packed position, roll, legal-source mask, optimal source class).
    This is the policy analogue of the full table: training a policy head on a
    sampled subset instead makes the rate axis meaningless, because the object
    being compressed shrinks to something a mid-sized model can memorise.
    """
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    count = len(raw) // 16
    if limit is not None:
        count = min(count, limit)
    view = raw[: 16 * count].reshape(count, 16)
    packed = view[:, :8].copy().view(np.uint64).reshape(count)
    # u32, not u16: a 16-tile path has 17 source classes.
    mask = view[:, 8:12].copy().view(np.uint32).reshape(count)
    roll = view[:, 12].copy().astype(np.int64)
    best = view[:, 13].copy().astype(np.int64)
    return packed, roll, mask, best


def load_tensors(path: str, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Memory-map a `dump-tensors` file as (packed words, values)."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    count = len(raw) // 12
    if limit is not None:
        count = min(count, limit)
    view = raw[: 12 * count].reshape(count, 12)
    packed = view[:, :8].copy().view(np.uint64).reshape(count)
    values = view[:, 8:].copy().view(np.float32).reshape(count)
    return packed, values


class Successors:
    """The evaluation set: sibling candidate moves with exact values.

    Scoring is deliberately model-agnostic. Hand it any array of scores over the
    successor rows -- from a net, a boosted ensemble, a quantised table -- and it
    applies the reflection, takes the argmax within each position, and returns
    exact regret. Nothing about the model enters the engine.
    """

    def __init__(self, path: str, ruleset: str):
        import csv

        rows = list(csv.DictReader(open(path)))
        self.ruleset = ruleset
        self.position = np.array([int(r["position"]) for r in rows])
        self.passed = np.array([int(r["passed"]) for r in rows], dtype=np.float32)
        self.terminal = np.array([int(r["terminal"]) for r in rows], dtype=bool)
        self.value = np.array([float(r["value_mover"]) for r in rows], dtype=np.float64)
        self.packed = np.array([int(r["packed"]) for r in rows], dtype=np.uint64)

        self.offsets = np.flatnonzero(np.r_[True, self.position[1:] != self.position[:-1]])
        self.counts = np.diff(np.r_[self.offsets, len(self.position)])
        self.best = np.maximum.reduceat(self.value, self.offsets)

        # Present only in dumps written after the policy-head change; a value
        # model never looks at them.
        if "parent" in rows[0]:
            self.parent = np.array([int(r["parent"]) for r in rows], dtype=np.uint64)[self.offsets]
            self.roll = np.array([int(r["roll"]) for r in rows], dtype=np.int64)[self.offsets]
            self.source = np.array([int(r["source"]) for r in rows], dtype=np.int64)
        else:
            self.parent = self.roll = self.source = None

    def path_length(self) -> int:
        return len(PATHS[self.ruleset][0])

    def policy_targets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per position: the legal-source mask, the optimal source, and a map
        from (position, source) back to the successor row.

        Sources are indices into the mover's own path, shifted by one so that
        entry from hand (-1) becomes class 0. Illegal sources are masked rather
        than merely trained against, because a policy that picks an illegal move
        has no defined regret.
        """
        if self.source is None:
            raise ValueError("this dump predates the parent/roll/source columns")
        classes = self.path_length() + 1
        rows = len(self.offsets)
        mask = np.zeros((rows, classes), dtype=bool)
        row_of = np.full((rows, classes), -1, dtype=np.int64)
        for row, (offset, count) in enumerate(zip(self.offsets, self.counts)):
            local = self.source[offset:offset + count] + 1
            mask[row, local] = True
            row_of[row, local] = np.arange(offset, offset + count)
        best_row = np.array([offset + np.argmax(self.value[offset:offset + count])
                             for offset, count in zip(self.offsets, self.counts)])
        best_source = self.source[best_row] + 1
        return mask, best_source, row_of

    def __len__(self) -> int:
        return len(self.offsets)

    def design(self) -> np.ndarray:
        return unpack(self.packed, self.ruleset)

    def regret(self, score: np.ndarray) -> dict:
        """Exact mean regret and agreement for a model's scores on successors.

        `score` is the model's estimate of the value to whoever moves *in the
        successor*. A move that passes the turn hands the position to the
        opponent, so the chooser's value is `100 - score` there -- the reflection
        about 50, applied once, in one place.
        """
        score = np.asarray(score, dtype=np.float64).copy()
        score = np.where(self.passed > 0, 100.0 - score, score)
        # A finished successor means the mover just won; nothing to estimate.
        score = np.where(self.terminal, 100.0, score)
        group_max = np.repeat(np.maximum.reduceat(score, self.offsets), self.counts)
        hits = np.flatnonzero(score >= group_max)
        who = np.searchsorted(self.offsets, hits, side="right") - 1
        _, first = np.unique(who, return_index=True)
        chosen = self.value[hits[first]]
        gap = self.best - chosen
        return {
            "regret": float(np.mean(gap)),
            "p95": float(np.percentile(gap, 95)),
            "max": float(np.max(gap)),
            "agreement": float(np.mean(gap < 1e-9)),
        }
