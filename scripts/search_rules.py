#!/usr/bin/env python3
"""Score a pool of candidate rules and assemble the best decision list.

Rules come from anywhere -- hand-written, enumerated, or proposed by a language
model -- and are graded exactly, because the map is solved. Two safeguards make
the reported number honest:

  * Every rule is parsed by the DSL, so malformed or unsafe text is rejected
    rather than run.
  * Positions are split in half. The list is assembled greedily on the search
    half and reported on the held-out half. Choosing the best of a thousand
    rules is itself a fit; without a holdout the winner's score is optimistic.

Usage:
    search_rules.py <move_features.csv> <rules.txt> [--length N]
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rule_dsl import Dataset, RuleError, evaluate  # noqa: E402


class Split:
    """A contiguous half of the positions, with its own regret metric."""

    def __init__(self, data, first_row, last_row, offsets):
        self.data = data
        self.rows = slice(first_row, last_row)
        self.offsets = offsets - first_row
        self.counts = np.diff(np.r_[self.offsets, last_row - first_row])
        self.values = data.values[self.rows]
        self.best = np.maximum.reduceat(self.values, self.offsets)
        self.size = last_row - first_row

    def regret(self, alive):
        index = np.arange(self.size)
        pick = np.minimum.reduceat(np.where(alive, index, index.max() + 1), self.offsets)
        return float(np.mean(self.best - self.values[pick]))

    def narrow(self, alive, mask):
        candidate = alive & mask
        kept = np.repeat(np.add.reduceat(candidate.astype(np.int64), self.offsets), self.counts)
        return np.where(kept > 0, candidate, alive)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    length = 8
    if "--length" in sys.argv:
        length = int(sys.argv[sys.argv.index("--length") + 1])

    data = Dataset(sys.argv[1])
    text = [line.strip() for line in open(sys.argv[2])]
    candidates = [line for line in text if line and not line.startswith("#")]

    masks, names, rejected = [], [], []
    for rule in candidates:
        try:
            mask = evaluate(rule, data.columns)
        except RuleError as error:
            rejected.append((rule, str(error)))
            continue
        if mask.sum() == 0 or mask.all():
            continue  # never fires, or never discriminates
        masks.append(mask)
        names.append(rule)

    print(f"{len(candidates)} candidate rules: {len(masks)} usable, "
          f"{len(rejected)} rejected, {len(candidates) - len(masks) - len(rejected)} vacuous")
    for rule, error in rejected[:5]:
        print(f"  rejected: {rule[:60]!r} -- {error}")

    # Split by position, halfway through.
    middle = len(data.offsets) // 2
    boundary = data.offsets[middle]
    search = Split(data, 0, boundary, data.offsets[:middle])
    holdout = Split(data, boundary, data.rows, data.offsets[middle:])
    print(f"\n{middle} positions to search on, {len(data.offsets) - middle} held out")

    alive_search = np.ones(search.size, dtype=bool)
    alive_holdout = np.ones(holdout.size, dtype=bool)
    print(f"\nbaseline (first legal move): search {search.regret(alive_search):.4f} pp, "
          f"holdout {holdout.regret(alive_holdout):.4f} pp\n")

    chosen = []
    for slot in range(length):
        best = None
        for index, mask in enumerate(masks):
            if index in chosen:
                continue
            regret = search.regret(search.narrow(alive_search, mask[search.rows]))
            if best is None or regret < best[0]:
                best = (regret, index)
        regret, index = best
        chosen.append(index)
        alive_search = search.narrow(alive_search, masks[index][search.rows])
        alive_holdout = holdout.narrow(alive_holdout, masks[index][holdout.rows])
        print(f"  {slot + 1}. {names[index][:64]:<64s}")
        print(f"      search {regret:.4f} pp   holdout {holdout.regret(alive_holdout):.4f} pp")

    print("\nfinal decision list:")
    for position, index in enumerate(chosen, 1):
        print(f"  {position}. prefer moves where: {names[index]}")
    print(f"\nheld-out mean regret: {holdout.regret(alive_holdout):.4f} pp")


if __name__ == "__main__":
    main()
