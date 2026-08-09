#!/usr/bin/env python3
"""Fit policies and write their weights in the layout the engine expects.

`royalur_analysis winrate` reads a weight file and plays the policy against the
optimal agent. The layout must match exactly what MovePolicy::score computes:

    [ state features (signed) , move features (unsigned) ,
      pairwise products over that combined vector in (i<j) order ,
      intercept ]

State features describe the successor from the *next* mover's perspective, so
they are reflected when the move passes the turn; move features are already
mover-relative and are not. The engine adds the 100-point constant back for a
turn-passing move, so the fit here must leave it on the target side.

Usage:
    export_policies.py <move_features.csv> <output-dir>
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

STATE = ["advancement_self", "advancement_opp", "scored_self", "scored_opp",
         "hand_self", "hand_opp", "safe_self", "safe_opp", "exposure_self",
         "threat_self", "centre_self", "centre_opp", "frontmost_self", "frontmost_opp",
         "stuck_self", "stuck_opp", "selfblock_self", "selfblock_opp",
         "rosettes_self", "rosettes_opp"]
MOVE = ["advance", "captures", "scores", "enters", "lands_rosette", "lands_centre",
        "leaves_centre", "dest_safe", "src_was_exposed", "delta_exposure",
        "delta_threat", "keeps_turn", "capture_value", "rescue_value",
        "delta_exposure_value", "delta_threat_value", "captures_frontmost",
        "capture_gap_to_front", "moves_frontmost", "becomes_safe_forever",
        "contact_possible", "threat_count"]


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    rows = list(csv.DictReader(open(sys.argv[1])))
    out = pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    columns = {n: np.array([float(r[n]) for r in rows]) for n in STATE + MOVE}
    values = np.array([float(r["value_mover"]) for r in rows])
    passed = np.array([int(r["turn_passed"]) for r in rows], dtype=float)
    state = np.array([int(r["state"]) for r in rows])
    offsets = np.flatnonzero(np.r_[True, state[1:] != state[:-1]])
    counts = np.diff(np.r_[offsets, len(state)])
    owner = np.repeat(np.arange(len(offsets)), counts)
    best = np.maximum.reduceat(values, offsets)
    sign = 1 - 2 * passed
    target = values - 100 * passed

    combined = np.array([columns[n] * sign for n in STATE] + [columns[n] for n in MOVE]).T

    def build(interactions):
        blocks = [combined]
        if interactions:
            width = combined.shape[1]
            blocks.append(np.array([combined[:, i] * combined[:, j]
                                    for i in range(width)
                                    for j in range(i + 1, width)]).T)
        blocks.append(sign[:, None])
        return np.hstack(blocks)

    for interactions, name in ((False, "additive"), (True, "pairwise")):
        design = build(interactions)
        mean = np.add.reduceat(design, offsets, axis=0) / counts[:, None]
        mean_target = np.add.reduceat(target, offsets) / counts
        weights = np.linalg.lstsq(design - mean[owner], target - mean_target[owner],
                                  rcond=None)[0]
        score = design @ weights + 100 * passed
        group_max = np.repeat(np.maximum.reduceat(score, offsets), counts)
        hits = np.flatnonzero(score >= group_max)
        who = np.searchsorted(offsets, hits, side="right") - 1
        _, first = np.unique(who, return_index=True)
        regret = float(np.mean(best - values[hits[first]]))
        path = out / f"policy_{name}.txt"
        path.write_text("\n".join(f"{w:.10f}" for w in weights) + "\n")
        print(f"{name}: {len(weights)} weights, offline regret {regret:.4f} pp -> {path}")


if __name__ == "__main__":
    main()
