#!/usr/bin/env python3
"""The reference rate-distortion curve: just store the table at fewer bits.

Before any learned model is worth reporting, it has to beat the dumbest possible
compressor -- keeping the table but rounding every value to b bits. That curve
costs one pass over the map and it is the bar everything else is judged against.
Two versions are computed:

  * **uniform** bins over [0, 100];
  * **Lloyd-Max**, the optimal b-level scalar quantiser. Because the entire
    population of values is in hand, this is computed exactly rather than
    estimated from a sample, which makes it a genuine memoryless coding bound
    rather than a baseline.

Distortion is policy regret, not value error, for the reason stage 1 established:
what decides a move is the ordering of sibling values, and a quantiser that
preserves the level while collapsing near-ties can have tiny value error and
terrible play. Value error is reported alongside to show the two coming apart.

Also reported is the **policy floor**: perfect play needs only the argmax at each
decision state, not values at all, so `log2(legal moves)` bits per decision is
the true cost of playing perfectly from a table. Every learned model should be
read against that number rather than against the 96 bits per state the map
actually occupies on disk.

Usage:
    quantise_curve.py <ruleset> <successors.csv> <tensors.bin> [--max-bits 12]
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ur_tensors import Successors, load_tensors  # noqa: E402


def nearest(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Map each value to its closest code point, via the bin midpoints.

    The obvious `argmin(|values - codebook|)` allocates rows x levels, which is
    6 GB at 4096 levels on 180k successors. The codebook is sorted, so a binary
    search over midpoints gives the same answer in linear space.
    """
    midpoints = 0.5 * (codebook[1:] + codebook[:-1])
    return codebook[np.searchsorted(midpoints, values)]


def lloyd_max(values: np.ndarray, levels: int, iterations: int = 40) -> np.ndarray:
    """Optimal scalar quantiser codebook, by Lloyd's algorithm on a histogram.

    A 4096-bin histogram of the exact value population stands in for the samples
    themselves, which keeps this exact to within bin width while running in
    milliseconds instead of minutes on 500M values.
    """
    counts, edges = np.histogram(values, bins=4096, range=(0.0, 100.0))
    centres = 0.5 * (edges[:-1] + edges[1:])
    weight = counts.astype(np.float64)
    # Initialise on quantiles so empty regions do not strand code points.
    cumulative = np.cumsum(weight) / max(weight.sum(), 1.0)
    codebook = np.interp(np.linspace(0, 1, levels + 2)[1:-1], cumulative, centres)
    for _ in range(iterations):
        assignment = np.abs(centres[:, None] - codebook[None, :]).argmin(axis=1)
        for k in range(levels):
            mask = assignment == k
            mass = weight[mask].sum()
            if mass > 0:
                codebook[k] = (centres[mask] * weight[mask]).sum() / mass
        codebook.sort()
    return codebook


def argument(flag: str, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    ruleset, successors_path, tensors_path = sys.argv[1], sys.argv[2], sys.argv[3]
    max_bits = int(argument("--max-bits", 12))

    successors = Successors(successors_path, ruleset)
    # The table stores each successor from its own mover's perspective; that is
    # the quantity being quantised, so undo the reflection before rounding and
    # let `regret` put it back.
    stored = np.where(successors.passed > 0, 100.0 - successors.value, successors.value)

    _, all_values = load_tensors(tensors_path)
    total_states = len(all_values)
    print(f"{ruleset}: {len(successors)} decision positions, "
          f"{len(stored)} successors, {total_states} states in table")

    exact = successors.regret(stored)
    assert exact["regret"] < 1e-9, f"exact values must have zero regret, got {exact}"
    print("exact values reproduce optimal play (regret 0), reflection is consistent\n")

    counts = successors.counts
    policy_bits = float(np.mean(np.log2(counts)))
    print(f"policy floor: {policy_bits:.3f} bits per decision state "
          f"(mean log2 of {counts.mean():.2f} legal moves)\n")

    print(f"{'bits':>5} {'scheme':>10} {'bits/state':>11} {'MAE':>9} {'maxerr':>9} "
          f"{'regret':>9} {'agreement':>10}")
    results = []
    for bits in range(1, max_bits + 1):
        levels = 1 << bits
        for scheme in ("uniform", "lloyd-max"):
            if scheme == "uniform":
                step = 100.0 / levels
                quantised = np.clip((np.floor(stored / step) + 0.5) * step, 0.0, 100.0)
            else:
                quantised = nearest(stored, lloyd_max(all_values.astype(np.float64), levels))
            error = np.abs(quantised - stored)
            row = successors.regret(quantised)
            results.append({"bits": bits, "scheme": scheme, "mae": float(error.mean()),
                            "max_error": float(error.max()), **row})
            print(f"{bits:>5} {scheme:>10} {bits:>11} {error.mean():>9.4f} "
                  f"{error.max():>9.4f} {row['regret']:>9.4f} {row['agreement']:>10.4f}")

    out = argument("--out", f"quantise_{ruleset}.json")
    with open(out, "w") as handle:
        json.dump({"ruleset": ruleset, "states": total_states,
                   "policy_bits_per_decision": policy_bits, "curve": results}, handle, indent=1)
    print(f"\nwrote {out}")

    # Translate the curve into the units the learned models are reported in, so
    # the two land on one axis: a model of P fp32 parameters costs 32P bits in
    # total, against `bits x states` for the quantised table.
    print(f"\nequivalent model size at each rate ({total_states} states):")
    for bits in (1, 2, 4, 8):
        params = bits * total_states / 32
        print(f"  {bits:>2} bits/state = {bits * total_states / 8e6:>10.1f} MB "
              f"= a {params / 1e6:>9.1f}M-parameter fp32 model")


if __name__ == "__main__":
    main()
