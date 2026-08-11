#!/usr/bin/env python3
"""Plot an aggregated long-game self-play action-count distribution."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: plot_lengths.py HISTOGRAM.csv SUMMARY.json OUTPUT.png")
    histogram_path = Path(sys.argv[1])
    summary_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    with histogram_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with summary_path.open() as handle:
        summary = json.load(handle)

    actions = np.asarray([int(row["actions"]) for row in rows], dtype=np.int64)
    counts = np.asarray([int(row["count"]) for row in rows], dtype=np.int64)
    games = int(counts.sum())
    probabilities = counts / games

    # Aggregate adjacent integer lengths into about 150 equal-width bins so the
    # PMF remains legible even if the optimal policy has a long tail.
    bin_count = min(150, max(20, int(np.sqrt(len(actions)))))
    edges = np.linspace(actions.min(), actions.max() + 1, bin_count + 1)
    binned, _ = np.histogram(actions, bins=edges, weights=probabilities)
    centres = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)

    survival = 1.0 - np.cumsum(probabilities) + probabilities
    mean = float(summary["simulation_mean_actions"])
    value = float(summary["starting_state_value_actions"])

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, (distribution, tail) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    distribution.bar(centres, binned, width=widths, color="#4169a1", alpha=0.85, edgecolor="none")
    distribution.axvline(value, color="#b33a3a", linewidth=2, label=f"LUT value: {value:.2f}")
    distribution.axvline(mean, color="#1c7c54", linewidth=1.8, linestyle="--", label=f"Monte Carlo mean: {mean:.2f}")
    distribution.set(title=f"Longest-game Finkel self-play ({games:,} games)", xlabel="Game length (rewarded legal moves)", ylabel="Probability per bin")
    distribution.legend(frameon=True)

    tail.step(actions, survival, where="post", color="#6f4e7c", linewidth=1.5)
    tail.set_yscale("log")
    tail.set(xlabel="Game length (rewarded legal moves)", ylabel="P(length ≥ x)", title="Survival function (log scale)")
    tail.set_ylim(bottom=max(0.5 / games, 1e-8), top=1.05)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    figure.savefig(output_path.with_suffix(".svg"))
    print(output_path)


if __name__ == "__main__":
    main()
