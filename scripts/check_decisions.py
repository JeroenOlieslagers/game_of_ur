#!/usr/bin/env python3
"""Verify a `dump-decisions` file.

What can be checked here, and what cannot, is worth being explicit about.

Checkable from the file alone: the legal-source mask agrees with the candidate
count, the optimal class is one of the legal ones, values are in range, every
decision has at least two candidates, and turn-passed bits are set only for
slots that hold a candidate.

NOT checkable here: whether the stored value is really in the chooser's frame
and whether the turn-passed bit points the right way. Both need the lookup
table, so they are established in Rust at the point of writing -- the value
comes straight from `light_win_percent` with one reflection, and the bit is a
comparison of `is_light_turn` before and after the move. A wrong reflection
would leave every number in range and every check below passing, which is
exactly why it is called out rather than assumed.

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
    slots = np.arange(MAX_CANDIDATES)[None, :]
    alive = slots < count[:, None]
    print(f"{rows} decisions")

    assert (count >= 2).all(), "a decision needs at least two candidates"
    assert (count <= MAX_CANDIDATES).all(), "candidate count exceeds the record"
    print(f"  candidates: min {count.min()}, max {count.max()}, mean {count.mean():.2f} "
          f"(MAX_CANDIDATES={MAX_CANDIDATES})")

    # The mask is indexed by SOURCE CLASS (up to 17 of them), not by candidate
    # slot, so it needs a full popcount rather than a sum over slot positions.
    bits = np.zeros(rows, dtype=np.int64)
    wide = mask.astype(np.int64)
    for bit in range(32):
        bits += (wide >> bit) & 1
    assert (bits == count).all(), (
        f"legal mask popcount != candidate count on {int((bits != count).sum())} decisions")
    print("  legal-mask popcount matches candidate count")

    assert (((wide >> best) & 1) == 1).all(), "optimal class is not marked legal"
    print("  optimal class is legal")

    live = value[alive]
    assert live.min() >= -1e-4 and live.max() <= 100.0 + 1e-4, "values out of range"
    print(f"  values in [{live.min():.4f}, {live.max():.4f}]")

    passed_bits = ((passed[:, None] >> slots) & 1).astype(bool)
    assert not (passed_bits & ~alive).any(), "turn-passed bit set on an empty slot"
    print(f"  turn passes on {100 * passed_bits[alive].mean():.1f}% of candidates")

    dead = value[~alive]
    assert (dead == 0).all(), "unused candidate slots must be zeroed"
    print("  unused slots are zeroed")
    print("all checks passed")


if __name__ == "__main__":
    main()
