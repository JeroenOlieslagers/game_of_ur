# Longest-game Finkel policy

This experiment keeps the exact Finkel state keys and move generator used by
the main solver, but changes the MDP objective from win probability to expected
remaining legal moves.

For a pre-roll state `s`,

```text
V(s) = sum_r P(r) max_a [1 + V(next(s, r, a))]
```

when roll `r` has a legal move. A zero roll or a roll with no legal move has no
action and contributes `V(next(s, r))`; a terminal state is zero. The move that
ends the game still earns one. Both players maximise the same cooperative
duration objective, so a turn change does not complement the stored value.

The resulting `.rgu` file is a lookup table over all 137,892,016 published
Finkel state keys. At play time the agent evaluates the legal successor keys and
chooses the one with greatest stored expected remaining length. Ties use the
move generator's stable order.

## Full local run

From the repository root:

```bash
bash long_game/run.sh /path/to/finkel.rgu long_game/results 10000000
```

The input must be the published Percent16 Finkel map; its values are discarded
and only its state keys are reused. The script builds the Rust binary, solves the
new Bellman equation, verifies a sample, simulates self-play, and plots the
action-count distribution. It resumes completed score layers from
`finkel_longest.checkpoint`.

Outputs:

- `finkel_longest.rgu`: f64 duration lookup table.
- `finkel_longest.policy`: explicit state-by-roll action lookup used by self-play.
- `finkel_longest.checkpoint` and `finkel_longest.layers/`: resumable solve data.
- `length_histogram.csv`: exact counts from self-play.
- `summary.json`: starting value and Monte Carlo statistics.
- `length_distribution.png` and `.svg`: distribution and survival plots.

## NYU Torch

The Slurm job is self-contained and uses the repository's shared cluster
environment:

```bash
sbatch --account=torch_pr_362_general --partition=cpu_short,cs \
  --export=ALL,UR_MODELS=/scratch/jo2229/path/to/models \
  long_game/cluster.sbatch
```

Override `UR_LONG_GAMES`, `UR_LONG_TOLERANCE`, `UR_LONG_MAX_SWEEPS`,
`UR_LONG_SEED`, or `UR_LONG_OUTPUT` as needed. The default simulation size is
10,000,000 games.
