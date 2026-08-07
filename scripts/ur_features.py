#!/usr/bin/env python3
"""Stage 1's 14 state features, computed on the GPU from a packed position.

These are recomputed here rather than dumped from Rust for three reasons: the
Masters feature table would be 14 GB, a dumped table has to be kept aligned with
whichever record file consumes it (`tensors.bin` for value models,
`decisions.bin` for policy heads), and a feature is a pure function of the
position anyway.

The definitions must match `features()` in `rust/src/main.rs` exactly, and
`verify_against` checks that against a Rust-produced sample rather than trusting
this file. Two are non-trivial: `exposure_self` and `threat_self` are

    sum over rolls r of P(r) * 1[a capture is legally available with roll r]

which is a lookahead over the dice distribution, not a function of the current
occupancy alone. That is precisely why they are worth supplying to a network:
learning them means learning to simulate a roll. They also came top of the
stage-1 Shapley-over-regret ranking.

Positions are canonical (mover is always light), so "self" is the light path
throughout and no colour branch is needed.
"""

from __future__ import annotations

import numpy as np
import torch

from ur_tensors import PATHS, PIECES

ROSETTES = (0, 2, 10, 18, 20)
CENTRE = 10
FEATURE_NAMES = [
    "advancement_self", "advancement_opp", "scored_self", "scored_opp",
    "hand_self", "hand_opp", "safe_self", "safe_opp",
    "exposure_self", "threat_self", "centre_self", "centre_opp",
    "frontmost_self", "frontmost_opp",
]
ROLL_PROBABILITIES = {
    "blitz": [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16],
    "finkel": [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16],
    "masters": [0.0, 3 / 8, 3 / 8, 1 / 8, 1 / 8],
}
SAFE_ROSETTES = {"finkel": True, "blitz": False, "masters": False}


class StateFeatures:
    """Precomputes the board geometry once, then evaluates batches."""

    def __init__(self, ruleset: str, device: torch.device):
        light, dark = PATHS[ruleset]
        self.ruleset = ruleset
        self.device = device
        self.pieces = PIECES[ruleset]
        self.length = len(light)
        self.probabilities = ROLL_PROBABILITIES[ruleset]
        self.safe_rosettes = SAFE_ROSETTES[ruleset]

        self.self_path = torch.tensor(light, device=device)
        self.opp_path = torch.tensor(dark, device=device)
        # Private tiles are on exactly one player's path, so nothing there can
        # ever be captured.
        shared = set(light) & set(dark)
        self.self_private = torch.tensor([i for i, t in enumerate(light) if t not in shared],
                                         device=device)
        self.opp_private = torch.tensor([i for i, t in enumerate(dark) if t not in shared],
                                        device=device)
        self.self_rosette = torch.tensor([t in ROSETTES for t in light], device=device)
        self.opp_rosette = torch.tensor([t in ROSETTES for t in dark], device=device)
        self.progress = torch.arange(1, self.length + 1, device=device, dtype=torch.float32)

    def occupancy(self, packed: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
        """0 empty, 1 the player to move, 2 the other, indexed along `path`."""
        return (packed.view(-1, 1) >> (2 * path).view(1, -1)) & 3

    def capture_chance(self, mine: torch.Tensor, theirs: torch.Tensor,
                       hand: torch.Tensor, rosette: torch.Tensor) -> torch.Tensor:
        """P(the owner of `mine` can capture a piece of `theirs` next roll).

        `mine`/`theirs` are boolean occupancy along the *capturing* player's own
        path. A capture with roll r needs a destination holding an enemy piece,
        reachable from a source that the capturing player occupies -- or from
        entry off the hand, which is source -1 and therefore destination r-1.
        """
        rows, length = mine.shape
        total = torch.zeros(rows, device=mine.device)
        for roll, probability in enumerate(self.probabilities):
            if roll == 0 or probability == 0.0:
                continue
            destination = torch.arange(length, device=mine.device)
            source = destination - roll
            target = theirs.clone()
            if self.safe_rosettes:
                # A piece on a rosette cannot be taken under Finkel rules.
                target = target & ~rosette.view(1, -1)
            # Sources on the board: the capturer must occupy destination - roll.
            from_board = torch.zeros_like(target)
            valid = source >= 0
            if valid.any():
                from_board[:, valid] = mine[:, source[valid]]
            # Entry from hand reaches exactly destination roll - 1.
            from_hand = torch.zeros_like(target)
            if roll - 1 < length:
                from_hand[:, roll - 1] = hand > 0
            reachable = target & (from_board | from_hand)
            total = total + probability * reachable.any(dim=1).float()
        return total

    def __call__(self, packed: torch.Tensor) -> torch.Tensor:
        packed = packed.view(-1)
        rows = packed.shape[0]
        own = self.occupancy(packed, self.self_path)
        other = self.occupancy(packed, self.opp_path)

        mine_own = own == 1          # my pieces, indexed along my path
        theirs_own = own == 2        # their pieces, on squares my path visits
        mine_opp = other == 1        # my pieces, on squares their path visits
        theirs_opp = other == 2      # their pieces, indexed along their path

        progress = self.progress.view(1, -1)
        advancement_self = (mine_own * progress).sum(dim=1)
        advancement_opp = (theirs_opp * progress).sum(dim=1)
        frontmost_self = (mine_own * progress).max(dim=1).values
        frontmost_opp = (theirs_opp * progress).max(dim=1).values

        hand_self = ((packed >> 48) & 7).clamp(0, self.pieces).float()
        hand_opp = ((packed >> 51) & 7).clamp(0, self.pieces).float()
        scored_self = ((packed >> 54) & 15).float()
        scored_opp = ((packed >> 58) & 15).float()

        safe_self = mine_own[:, self.self_private].sum(dim=1).float()
        safe_opp = theirs_opp[:, self.opp_private].sum(dim=1).float()

        centre = (packed >> (2 * CENTRE)) & 3
        centre_self = (centre == 1).float()
        centre_opp = (centre == 2).float()

        # threat: I capture next roll, along my path.
        threat_self = self.capture_chance(mine_own, theirs_own, hand_self, self.self_rosette)
        # exposure: they capture next roll, along their path, so the roles of
        # the two occupancy codes swap.
        exposure_self = self.capture_chance(theirs_opp, mine_opp, hand_opp, self.opp_rosette)

        return torch.stack([
            advancement_self, advancement_opp, scored_self, scored_opp,
            hand_self, hand_opp, safe_self, safe_opp,
            exposure_self, threat_self, centre_self, centre_opp,
            frontmost_self, frontmost_opp,
        ], dim=1)


def verify_against(csv_path: str, ruleset: str, device=None, limit: int = 20000) -> None:
    """Check these against Rust-computed features for the same positions.

    `dump-successors --with-features` writes the 14 features beside the packed
    word, so this compares like for like. Any mismatch is a bug here, not there:
    the Rust definitions are the ones stage 1 was measured with.
    """
    import csv as csv_module

    device = device or torch.device("cpu")
    rows = []
    with open(csv_path) as handle:
        for index, row in enumerate(csv_module.DictReader(handle)):
            if index >= limit:
                break
            rows.append(row)
    if not rows or FEATURE_NAMES[0] not in rows[0]:
        raise SystemExit(f"{csv_path} has no feature columns; re-dump with --with-features")

    packed = torch.tensor([int(r["packed"]) for r in rows], dtype=torch.int64, device=device)
    expected = np.array([[float(r[n]) for n in FEATURE_NAMES] for r in rows])
    # Terminal successors have no meaningful features and Rust writes zeros.
    live = np.array([int(r["terminal"]) == 0 for r in rows])
    actual = StateFeatures(ruleset, device)(packed).cpu().numpy()

    worst = 0.0
    for column, name in enumerate(FEATURE_NAMES):
        gap = np.abs(actual[live, column] - expected[live, column]).max()
        worst = max(worst, gap)
        flag = "" if gap < 1e-6 else "   <-- MISMATCH"
        print(f"  {name:>18}  max |diff| = {gap:.3e}{flag}")
    if worst >= 1e-6:
        raise SystemExit(f"features disagree with Rust (worst {worst:.3e})")
    print(f"all 14 features match Rust on {int(live.sum())} positions")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        raise SystemExit("usage: ur_features.py <successors_with_features.csv> <ruleset>")
    verify_against(sys.argv[1], sys.argv[2])
