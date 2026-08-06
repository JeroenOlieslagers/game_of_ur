#!/usr/bin/env python3
"""Fit a linear evaluation function to the exact win probabilities.

`royalur_analysis regret` dumps a feature matrix alongside the true value of
each sampled position. This fits least squares to it and writes the weights back
in the order `regret` expects, so they can be fed in as an extra rung of the
ladder:

    royalur_analysis regret <map> <out-dir> <samples> onpolicy <seed> <weights>

Fit on **on-policy** samples. Fitting on uniformly sampled states tunes the
weights to positions that essentially never occur in play.

Usage:
    fit_heuristic.py <features.csv> <weights.txt>
"""

from __future__ import annotations

import csv
import pathlib
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        raise SystemExit("numpy is required: pip install numpy")

    source = pathlib.Path(sys.argv[1])
    destination = pathlib.Path(sys.argv[2])

    rows = list(csv.DictReader(source.open()))
    if not rows:
        raise SystemExit(f"{source} is empty")
    names = [key for key in rows[0] if key != "value_mover"]

    features = np.array([[float(row[name]) for name in names] for row in rows])
    values = np.array([float(row["value_mover"]) for row in rows])

    # The intercept is fitted AND kept. It is tempting to drop it on the grounds
    # that a constant shifts every candidate move equally and so cannot change
    # which move is chosen -- but that is false here. A score is the *mover's*
    # win percentage, and converting to a fixed perspective reflects it about 50
    # (`v -> 100 - v`). A move onto a rosette keeps the turn while other moves
    # pass it, so the reflection applies to some successors and not others, and
    # the intercept does not cancel.
    design = np.hstack([features, np.ones((len(features), 1))])
    solution, *_ = np.linalg.lstsq(design, values, rcond=None)
    predicted = design @ solution
    r2 = 1 - ((values - predicted) ** 2).sum() / ((values - values.mean()) ** 2).sum()

    print(f"states   {len(rows)}")
    print(f"R^2      {r2:.4f}")
    print(f"residual {np.std(values - predicted):.3f} percentage points")
    print()
    scale = abs(solution[names.index("advancement_self")]) if "advancement_self" in names else 0.0
    for name, weight in sorted(zip(names, solution), key=lambda pair: -abs(pair[1])):
        squares = f"{weight / scale:+8.1f}" if scale > 0 else "      --"
        print(f"  {name:>18s} {weight:+9.3f}   {squares} squares of advancement")
    print(f"  {'intercept':>18s} {solution[-1]:+9.3f}   (kept: see comment in source)")

    destination.write_text(
        "# fitted by scripts/fit_heuristic.py, one weight per line\n"
        f"# order: {','.join(names)},intercept\n"
        + "\n".join(f"{weight:.10f}" for weight in solution)
        + "\n"
    )
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
