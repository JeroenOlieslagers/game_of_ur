# Handoff: longest-game Finkel lookup policy

> **Superseded in part. See `RESULTS.md` for what was actually computed.**
>
> The solve was run on Torch and produces a policy achieving 412,543 +/- 252
> actions, but the lookup values did **not** converge to the optimum. The
> handoff's plan was sound; what it could not anticipate is that this objective
> is undiscounted with a per-sweep contraction of 0.99998, so a small Bellman
> residual does not bound the error -- the amplification is `1/(1-rho) ~ 50,000`.
> Its recommendation 4 ("consider a looser residual such as 1e-7 or 1e-8, but
> verify policy stability separately") is therefore too optimistic: no tolerance
> reachable in reasonable time certifies the value. Its recommendation 3
> (modified policy iteration with a sparse linear solve) remains the right fix
> and was not completed -- a Gauss-Seidel-plus-extrapolation approximation of it
> was tried and did not converge either.

## Status at handoff

The implementation is present and tested, but the full numerical solve is not
finished. Consequently, the requested starting-state value, 10,000,000-game
self-play distribution, and final plot do **not** exist yet. No approximate
number is presented as the requested result.

The local solve was deliberately interrupted for this handoff while working on
score layer `(3,4)`. Its last asynchronous sweep report was:

```text
long-game scores=[3,4] sweep=44300 max_delta=4.929797796649e-7
```

This is not a certified Bellman residual. Only completed layers are appended to
the checkpoint, so the unfinished `(3,4)` work was lost when the process was
stopped. Eight earlier layers remain in the ignored local checkpoint.

## Requested experiment and interpretation

The task was to make a separate experiment for the Finkel ruleset that learns
a lookup-table policy which maximizes game duration, rather than win
probability, and then:

1. report the value of the starting state;
2. run 10,000,000 self-play games under the learned policy; and
3. plot the distribution of game lengths.

I made the reward convention explicit because there are two plausible meanings
of "game length." This implementation optimizes and reports **legal moves
(actions)**, not dice rolls:

- a legal move earns `+1`, including a move that wins the game;
- a zero roll or any roll with no legal move earns `+0`;
- a terminal position has value `0`;
- both sides cooperate to maximize the same duration objective;
- changing turns therefore does not complement or negate the value.

For a pre-roll state `s`, the Bellman equation is:

```text
V(s) = sum_r P(r) max_a [1 + V(next(s, r, a))]
```

when roll `r` has at least one legal move. When it does not, that roll's term is
the value after applying the pass/turn behavior. This makes the starting value
the expected number of rewarded legal moves remaining under optimal
cooperative play. Simulation also records dice-roll counts as a secondary
diagnostic, but the plotted distribution is action count.

## Repository review and design choice

The existing Rust solver already has the important expensive infrastructure:

- the exact Finkel move generator and state encoding;
- the published set of 137,892,016 reachable lookup keys;
- score-layer decomposition;
- compact successor generation and multithreaded layer sweeps;
- generic `.rgu` lookup-table I/O.

The new experiment reuses that implementation instead of maintaining a second
game engine. It reads the published Percent16 Finkel map only for its sorted
state keys and ruleset metadata; all win-probability values are discarded and
the duration values start at zero.

The solver is isolated in `rust/src/long_game.rs` and exposed through three new
commands in `rust/src/main.rs`:

```text
train-long-game <percent16-finkel.rgu> <output-duration.rgu> [tolerance] [max-sweeps]
verify-long-game <duration.rgu> [samples]
simulate-long-game <duration.rgu> <output-dir> [games] [seed]
```

The experiment is Finkel-only and asserts the ruleset at load time.

## Files added or changed

### `rust/src/long_game.rs`

Contains the complete duration experiment:

- zero-initialized f64 value table over the published Finkel keys;
- duration-specific precomputed successor records;
- score-layer value iteration and deterministic convergence checks;
- resumable completed-layer checkpointing;
- `.rgu` writing with duration-objective metadata;
- sampled Bellman verification and starting-state reporting;
- construction of a literal state-by-roll action policy table;
- deterministic, parallel self-play;
- aggregated histogram and JSON summary output;
- unit tests for action rewards, Bellman maximization, and the acceleration
  helper.

Successors are stored in a compact CSR-like representation. A high bit marks
whether a transition corresponds to a rewarded action; a sentinel represents
a terminal successor. Before solving each score layer, up to 2,000 randomly
sampled states are compared against the on-demand Bellman implementation using
exact equality. This guards against errors in the specialized precomputation.

The main iteration is a parallel asynchronous atomic Gauss-Seidel-style sweep.
A complete deterministic Bellman pass supplies the residual used to certify a
layer. There is also an experimental guarded residual extrapolation step every
500 sweeps. It estimates a slow residual mode, proposes a correction, and only
accepts it after a full deterministic pass shows a lower maximum residual. It
did not materially help the slow `(3,4)` layer in the local run and is a good
candidate for replacement by policy evaluation or a sparse linear solve.

### `rust/src/main.rs`

Declares the new module, adds the three commands above, and documents them in
CLI usage. Existing commands and win-probability behavior are unchanged.

### `long_game/run.sh`

End-to-end local driver. It builds the release binary, trains if the final map
does not exist, verifies the finished map, runs self-play, and makes the plot.
The default game count is 10,000,000.

### `long_game/cluster.sbatch`

Slurm driver requesting one node, 16 CPUs, 48 GiB, and four hours. It sources
the repository's existing `cluster/env.sh`, serializes concurrent Cargo builds
with `flock`, and runs the same train/verify/simulate/plot pipeline. It is
requeue-enabled, and completed score layers survive restarts when the output
directory is on persistent storage.

### `long_game/plot_lengths.py`

Reads the aggregated CSV histogram and JSON summary. It produces a PNG and SVG
with a binned probability-mass view and a log-scale survival curve, marking the
lookup-table starting value and Monte Carlo mean.

### `long_game/README.md`

Short operational documentation for local and Torch runs.

### `.gitignore`

Ignores Slurm logs named `long_game/long_game_*.out`. Existing ignore rules
already exclude result directories, `.rgu` maps, checkpoints, layer files,
CSVs, and Rust build output.

## Output formats

The end-to-end run creates:

- `finkel_longest.rgu`: f64 duration value for every published Finkel state;
- `finkel_longest.policy`: direct state-by-roll action lookup for simulation;
- `finkel_longest.checkpoint`: values for completed score layers;
- `finkel_longest.layers/`: regenerated layer-index files;
- `length_histogram.csv`: nonzero `(actions, count, probability, cdf)` rows;
- `summary.json`: starting value, Monte Carlo mean/stddev/standard error,
  difference between mean and value, quantiles, maximum, mean rolls, light win
  fraction, seed, game count, and elapsed seconds;
- `length_distribution.png` and `length_distribution.svg`.

The direct policy table begins with the eight-byte magic `RGULPOL1`, a
little-endian `u64` state count, and a little-endian `u32` roll count. It then
stores five bytes per state, one for each roll. Sources `-1..13` are encoded as
`0..14`; `255` means no action. For 137,892,016 states, the payload is
689,460,080 bytes plus the 20-byte header. The file is built once in parallel
and allows the 10M-game simulation to avoid evaluating every legal successor
at every move.

Self-play randomness is reproducible across thread counts: every global game
index derives its own seed from the requested master seed.

## Commands

From the repository root (`new_code`), the complete local workflow is:

```bash
bash long_game/run.sh ../ruleset_analysis/models/finkel.rgu long_game/results 10000000
```

The equivalent stages are:

```bash
cargo build --release --manifest-path rust/Cargo.toml
rust/target/release/royalur_analysis train-long-game \
  ../ruleset_analysis/models/finkel.rgu \
  long_game/results/finkel_longest.rgu 1e-10 1000000
rust/target/release/royalur_analysis verify-long-game \
  long_game/results/finkel_longest.rgu 10000
rust/target/release/royalur_analysis simulate-long-game \
  long_game/results/finkel_longest.rgu long_game/results 10000000 \
  10047152335197783521
python3 long_game/plot_lengths.py \
  long_game/results/length_histogram.csv \
  long_game/results/summary.json \
  long_game/results/length_distribution.png
```

Useful environment overrides are:

- `UR_FINKEL_INPUT`
- `UR_LONG_OUTPUT`
- `UR_LONG_GAMES`
- `UR_LONG_TOLERANCE`
- `UR_LONG_MAX_SWEEPS`
- `UR_LONG_SEED`
- `UR_PLOT_PYTHON`

The local system `python3` did not have Matplotlib. A bundled Python with NumPy
and Matplotlib was available at:

```text
/Users/jeroen/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
```

It can be selected with `UR_PLOT_PYTHON`. The cluster environment also needs
NumPy and Matplotlib for the final plotting step; training and simulation do
not depend on Python.

## Torch/Slurm handoff

The intended cluster submission, from the repository root, is:

```bash
sbatch --account=torch_pr_362_general --partition=cpu_short,cs \
  --export=ALL,UR_MODELS=/scratch/jo2229/path/to/models \
  long_game/cluster.sbatch
```

Alternatively set the exact input explicitly:

```bash
sbatch --account=torch_pr_362_general --partition=cpu_short,cs \
  --export=ALL,UR_FINKEL_INPUT=/scratch/jo2229/path/to/finkel.rgu \
  long_game/cluster.sbatch
```

Submit from the repository root, or export `UR_ROOT` to its cluster path.
Slurm runs a spool copy of the script, so `BASH_SOURCE` alone cannot reliably
find the checkout.

I attempted to inspect `/scratch/jo2229` and use the existing `ssh -fN torch`
connection. The local execution environment required an escalation review for
SSH, and that review service failed because its organization was not verified.
This was an authorization-tool failure, not evidence that the Torch tunnel,
host, or files were unavailable. I therefore did not inspect, copy, submit, or
modify anything on the cluster.

The local checkpoint is ignored and is not part of the commit. If it is worth
preserving the eight completed layers, copy only
`long_game/results/finkel_longest.checkpoint` to the same output path on the
cluster. The layer-index files are deterministic and quick to regenerate from
the identical input state-key map; copying the roughly 528 MiB layer directory
is unnecessary. Do not reuse the checkpoint with a different state-key map.

## Validation completed

The following checks passed during development:

```text
cargo test --manifest-path rust/Cargo.toml
  3 passed; 0 failed

cargo build --release --manifest-path rust/Cargo.toml
  passed

bash -n long_game/run.sh
  passed

bash -n long_game/cluster.sbatch
  passed

python3 -m py_compile long_game/plot_lengths.py
  passed

git diff --check
  passed
```

The whole existing `rust/src/main.rs` is not `rustfmt`-clean in the baseline.
Only the new `long_game.rs` was formatted to avoid a large unrelated diff.

Verification of a finished model is intentionally separate from training. It
checks sampled Bellman residuals and prints:

```text
ruleset=finkel
objective=max_expected_legal_moves
entries=137892016
starting_state_value_actions=...
sampled_states=...
sample_max_bellman_residual=...
```

## Partial local numerical work

The local input was:

```text
/Users/jeroen/Documents/game_of_ur/ruleset_analysis/models/finkel.rgu
```

It is a roughly 789 MiB published Percent16 Finkel map. A separate f64 map also
exists locally, but the implementation deliberately uses the compact published
map because only its keys are needed.

Completed checkpoint records at handoff were:

| score layer | states | certified residual | minimum value | maximum value | mean value |
|---|---:|---:|---:|---:|---:|
| `(6,6)` | 217 | `7.4317e-11` | 1.0000 | 14.4185 | 6.4852 |
| `(5,6)` | 2,956 | `4.8804e-11` | — | 24.5105 | — |
| `(5,5)` | 9,696 | `6.0709e-11` | — | 62.7378 | — |
| `(4,6)` | 12,628 | `5.8265e-11` | — | 36.4690 | — |
| `(4,5)` | 79,760 | `7.9893e-11` | — | 145.0426 | — |
| `(4,4)` | 157,864 | `9.4474e-11` | — | 788.5432 | — |
| `(3,6)` | 38,082 | `6.6294e-11` | — | 53.7515 | — |
| `(3,5)` | 231,604 | `8.5265e-11` | — | 354.8275 | — |

Only the `(6,6)` minimum and mean were recorded during inspection; dashes are
unknown, not zero. The ignored local artifacts at handoff were approximately:

```text
long_game/results/finkel_longest.checkpoint   4.1 MiB
long_game/results/finkel_longest.layers/      528 MiB, 28 files
```

There is no final `finkel_longest.rgu`, policy table, histogram, summary, or
plot yet.

## Performance finding and main risk

The undiscounted positive-reward objective converges much more slowly than the
original win-probability objective. Captures allow long cycles, so some policy
subchains have a spectral radius very close to one. The normal solver's runtime
does not predict this experiment's runtime.

The `(3,4)` layer has 882,492 states. It required 44,300 asynchronous sweeps to
move the update magnitude into the `5e-7` range and still had not met the
requested `1e-10` deterministic residual. Larger and lower-score layers may be
substantially worse. A four-hour Slurm allocation may requeue before the solve
finishes. Completed-layer checkpointing makes that safe, provided the job is
resubmitted with the same persistent output directory.

The 10M-game simulation may also be expensive because the optimized policy is
specifically seeking long trajectories. The direct policy file removes most
per-move decision overhead, but total runtime remains unknown until the model
and its starting value are available.

## Recommended next steps

1. Put the checkout and Percent16 Finkel key map on persistent Torch storage,
   optionally copy the local completed-layer checkpoint, and submit the Slurm
   script.
2. Watch whether later layers make progress fast enough. The log reports each
   layer, sweep counts, update magnitudes, certified residuals, and elapsed
   time.
3. If plain iteration remains too slow, keep the existing state/successor
   pipeline but replace the experimental extrapolation with modified policy
   iteration: freeze the maximizing action per roll after the policy stabilizes,
   solve `(I-P_pi)V=r_pi` for that score layer using a sparse iterative solver,
   improve the policy, and repeat. This directly targets the slow recurrent
   modes.
4. If exact `1e-10` values are not essential, consider a looser residual such
   as `1e-7` or `1e-8`, but verify policy stability separately. A small value
   residual does not automatically rule out action changes when alternatives
   are nearly tied.
5. Once `finkel_longest.rgu` exists, run `verify-long-game` before simulation.
   The printed starting value is the requested exact lookup result to the
   achieved Bellman tolerance.
6. Run the 10,000,000 games and plot. Compare `simulation_mean_actions` with
   `starting_state_value_actions`; their difference should be consistent with
   the Monte Carlo standard error plus numerical solve error.
7. Archive the finished map, policy, summary, histogram, and plots. They are
   ignored by Git because of their size and generated nature.

The implementation is ready for continued numerical work, but the numerical
result requested by the original task remains outstanding.
