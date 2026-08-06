#!/usr/bin/env python3
"""Combine sharded Monte Carlo simulation output into final statistics.

`royalur_analysis simulate` writes one pair of CSVs per shard, holding raw
binomial counts. Summing the counts across shards is exactly equivalent to
having run one long simulation, so this recomputes the percentages from the
summed totals rather than averaging per-shard percentages (which would be wrong
whenever shards differ in size).

The output schema matches what `analyze` produces, so analysis/plot_results.jl
consumes it unchanged:

    <ruleset>_compare.csv   state,key,predicted_pct,games,light_wins,simulated_pct
    <ruleset>_epsilon.csv   epsilon,games,optimal_wins,optimal_win_pct

Usage:
    aggregate_simulations.py <shard-dir> <output-dir> [ruleset ...]
"""

from __future__ import annotations

import csv
import pathlib
import sys
from collections import defaultdict


def aggregate_compare(shards: list[pathlib.Path], destination: pathlib.Path) -> tuple[int, int]:
    # Keyed by state index; the key and prediction are properties of the state
    # and must agree across shards.
    games: dict[int, int] = defaultdict(int)
    wins: dict[int, int] = defaultdict(int)
    keys: dict[int, str] = {}
    predicted: dict[int, str] = {}

    for shard in shards:
        with shard.open() as handle:
            for row in csv.DictReader(handle):
                # A shard killed by walltime leaves a truncated final line.
                # Complete rows are still valid counts, so keep them and drop
                # only the partial one.
                if not row.get("light_wins") or row.get("games") in (None, ""):
                    print(f"{shard}: skipping incomplete final row")
                    continue
                state = int(row["state"])
                if state in keys and keys[state] != row["key"]:
                    raise SystemExit(
                        f"{shard}: state {state} has key {row['key']}, expected {keys[state]}. "
                        "Shards must sample the same states; check they used the same binary."
                    )
                keys[state] = row["key"]
                predicted[state] = row["predicted_pct"]
                games[state] += int(row["games"])
                wins[state] += int(row["light_wins"])

    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state", "key", "predicted_pct", "games", "light_wins", "simulated_pct"])
        for state in sorted(games):
            total, won = games[state], wins[state]
            writer.writerow([
                state, keys[state], predicted[state], total, won,
                f"{100.0 * won / total:.12f}",
            ])
    return len(games), sum(games.values())


def aggregate_epsilon(shards: list[pathlib.Path], destination: pathlib.Path) -> tuple[int, int]:
    games: dict[str, int] = defaultdict(int)
    wins: dict[str, int] = defaultdict(int)

    for shard in shards:
        with shard.open() as handle:
            for row in csv.DictReader(handle):
                if not row.get("optimal_wins") or row.get("games") in (None, ""):
                    print(f"{shard}: skipping incomplete final row")
                    continue
                epsilon = row["epsilon"]
                games[epsilon] += int(row["games"])
                wins[epsilon] += int(row["optimal_wins"])

    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epsilon", "games", "optimal_wins", "optimal_win_pct"])
        for epsilon in sorted(games, key=float):
            total, won = games[epsilon], wins[epsilon]
            writer.writerow([
                epsilon, total, won, f"{100.0 * won / total:.12f}",
            ])
    return len(games), sum(games.values())


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    shard_dir = pathlib.Path(sys.argv[1])
    output_dir = pathlib.Path(sys.argv[2])
    rulesets = sys.argv[3:] or ["finkel", "blitz", "masters"]
    output_dir.mkdir(parents=True, exist_ok=True)

    for ruleset in rulesets:
        for kind, aggregate in (("compare", aggregate_compare), ("epsilon", aggregate_epsilon)):
            shards = sorted(shard_dir.glob(f"{ruleset}_{kind}_*.csv"))
            if not shards:
                print(f"{ruleset} {kind}: no shards found in {shard_dir}")
                continue
            destination = output_dir / f"{ruleset}_{kind}.csv"
            rows, total_games = aggregate(shards, destination)
            print(
                f"{ruleset} {kind}: {len(shards)} shards -> {rows} rows, "
                f"{total_games:,} games -> {destination}"
            )


if __name__ == "__main__":
    main()
