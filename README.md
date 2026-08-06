# Solving the Royal Game of Ur

Exact solutions for several rule sets of the Royal Game of Ur (RGU), by value
iteration over the complete state space.

The repository holds two solvers that do different jobs:

| | What it does | Scale |
| --- | --- | --- |
| [`julia/`](julia) | Readable reference implementation. Enumerates the state space from scratch and solves it. | Small piece counts, on a laptop |
| [`rust/`](rust) | Optimised solver. Refines the published Percent16 maps to full `f64` precision, keeping their exact state keys. | All three published rule sets, up to 500,981,472 states |

The Julia code is the one to read to understand the method. The Rust code is the
one that produced the high-precision maps, and it solves the largest rule set in
[minutes](#measured-runtimes).

## Quick start

### The readable solver (Julia)

Needs only [Julia](https://julialang.org/downloads/); no dependencies beyond the
standard library.

```bash
julia --project=julia julia/scripts/solve.jl        # N=3, theta=1e-3
julia --project=julia julia/scripts/solve.jl 5 1e-6 # 5 pieces, tighter threshold
julia --project=julia julia/test/runtests.jl        # test suite
```

It prints the first player's win probability. `N` is the number of pieces per
player; the 32-bit encoding supports `N <= 7`, though memory becomes binding
before that (see [Why two solvers](#why-two-solvers)).

### The optimised solver (Rust)

Needs a Rust toolchain (`cargo`) and nothing else.

```bash
python scripts/download_models.py finkel            # fetch a published map
cargo build --release --manifest-path rust/Cargo.toml

rust/target/release/royalur_analysis train-f64 \
    models/finkel.rgu models/finkel_f64_ours.rgu 3e-14 10000 precomputed-gauss-seidel
```

Rule sets are `finkel`, `blitz` and `masters`; the solver identifies which one a
map is from its embedded metadata. Verify a finished map, and compare two
independently produced maps:

```bash
rust/target/release/royalur_analysis verify models/finkel_f64_ours.rgu 10000
rust/target/release/royalur_analysis compare-checkpoint \
    models/finkel_f64_ours.checkpoint models/finkel_f64_ours.layers models/finkel_f64.rgu
```

For Slurm, see [Running on a cluster](#running-on-a-cluster).

## The method

### The game as a Markov decision process

A position plus the player to move is a state. A dice roll is a chance event, a
move choice is a decision. Both players play optimally and RGU is zero-sum, so
one number per state suffices: the probability that the player to move wins. The
optimal value function satisfies the Bellman equation

```
V(s) = sum_r P(r) * max_{s' in successors(s, r)} V(s')
```

where `P(r)` is the probability of rolling `r` and the `max` is over the moving
player's options. Because the value always means "probability that the player to
move wins", a successor in which the turn has passed contributes `1 - V(s')`.

Two things keep this tractable. The board is always read from the perspective of
the player to move, so only one value per position is stored. And terminal states
have known values, which grounds the recursion.

### Score layers

Scores never decrease. Partition states by `(min(score_l, score_d),
max(score_l, score_d))`, and each layer depends only on itself and on layers with
strictly higher scores. Solving layers in descending score order means that when
work starts on a layer, every successor *outside* it already holds its final
value, leaving only the within-layer transitions to iterate.

This is what turns one global iteration into a sequence of bounded ones, and it
is why a layer is the unit of checkpointing. Within a layer, transitions can
cycle — pieces get captured and re-enter — so a layer genuinely has to be
iterated to a fixed point rather than evaluated in one pass. It converges because
any infinite play scores with probability 1, which makes the within-layer
operator a contraction.

### Iteration strategies

Both solvers iterate a layer to a fixed point. The Rust solver offers two ways,
chosen by the last argument to `train-f64`. They share a checkpoint format, so a
run started under one can be resumed under the other, and both accept a layer
only once a deterministic residual pass shows `max |T(V) - V| <= tolerance`.

**`ondemand-jacobi`** regenerates successors from the decoded position on every
sweep and applies updates out of place. The straightforward scheme.

**`precomputed-gauss-seidel`** (the default) is about 20x faster per sweep and
needs fewer sweeps. Two changes get it there.

*Precomputed successor indices.* An on-demand sweep decodes each state and runs a
binary-search lookup for every successor. Since a layer is swept dozens of times,
it is far cheaper to materialise the successor indices once into a CSR table and
let each sweep be a gather over it. Only the *current* layer needs a table:
successors outside it have frozen values and are read by index like any other, so
the table scales with the layer, not with the whole map. Measured at about 99
bytes per state — 1.489 GB for Blitz's largest layer of 14,993,636 states, built
in 2.24 s. Building it costs roughly six sweeps against layers needing dozens, so
it repays itself several times over. A dense whole-map transition tensor would be
about 56 GB for Masters; this is what makes that unnecessary.

*In-place (Gauss-Seidel) updates.* Sweeping in place lets a sweep see values
published earlier in the same sweep, which contracts faster than Jacobi's
out-of-place update. Measured per-sweep contraction was about 0.65 for Jacobi
against 0.55 for Gauss-Seidel.

Sweeps are parallel in both schemes, split across
`thread::available_parallelism()` scoped threads. The Gauss-Seidel sweep is
therefore the *asynchronous* analogue of a serial Gauss-Seidel pass: threads
publish updates as they go, so which updates a thread observes depends on timing.
Asynchronous value iteration converges to the same fixed point regardless of
update order, given a contraction and enough updates per state, so this is sound
— but it does make sweep deltas non-reproducible. They are used only as a
progress signal; the layer is certified by a separate deterministic residual
pass, and that is the number recorded in the checkpoint. Values are accessed as
relaxed atomics so the concurrent reads and writes are well defined and cannot
tear; on x86-64 these compile to plain loads and stores.

### Precision

The tolerance is in percentage points. `3e-14` is the practical `f64` residual
floor: every score layer of every rule set converges to exactly
`2.842170943040e-14`, which is 2 units in the last place at magnitude 100. This
matches the `2.8e-14` precision reported for the published Finkel map.

### State encoding

**Julia.** States are 32-bit unsigned integers in a self-other representation, so
it is always "self's" turn and the identity of the light and dark player does not
matter. In order: 13 bits for the central column, whose 8 tiles are encoded as
[trits](https://en.wikipedia.org/wiki/Ternary_numeral_system) (0 empty, 1 self,
2 other) packed into 13 bits; 6 bits for self's safe tiles; 6 bits for other's
safe tiles; 3 bits for self's pieces still at home; 3 bits for other's.

**Rust.** The Rust solver instead reads and writes the `.rgu` format used by
[RoyalUr-Java](https://github.com/RoyalUr/RoyalUr-Java), keeping the published
maps' exact state keys so its output is a drop-in higher-precision replacement.
It reimplements that project's `SimpleGameStateEncoding`: the 20 board tiles
split into "war" tiles reachable by both players and private tiles reachable by
one; private tiles take one bit each; war tiles are packed 2 bits each and then
compressed to eliminate combinations using more pieces than a player owns. Scores
are not stored — they are recovered as `pieces - inHand - onBoard`.

The layout follows from the path pair, which is why key widths differ per rule
set:

| Rule set | Path | War tiles | Segments | Board bits | Key bits | Maps |
| --- | --- | --- | --- | --- | --- | --- |
| Finkel | Bell (14 tiles) | 8 | 1 x 13 bits | 25 | 31 | 1 |
| Blitz, Masters | Masters (16 tiles) | 12 | 2 x 10 bits | 28 | 34 | 4 |

A Finkel key fits in 32 bits so it needs a single map, while Masters keys spill
past 32 bits and are split across four.

### Rule sets

| Rule set | Pieces | Dice | Path | Rosette grants roll | Capture grants roll | Rosettes safe | States |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Finkel | 7 | 4 binary | Bell | yes | no | **yes** | 137,892,016 |
| Blitz | 5 | 4 binary | Masters | yes | **yes** | no | 41,254,034 |
| Masters | 7 | 3 binary, 0 counts as 4 | Masters | yes | no | no | 500,981,472 |

Four binary dice give `P(r) = C(4, r) / 16`. Masters' three binary dice with zero
counting as four give `P(1) = P(2) = 3/8` and `P(3) = P(4) = 1/8`, with
`P(0) = 0`, so Masters has no pass-turn roll. Safe rosettes mean a piece standing
on a rosette cannot be captured, which removes moves rather than adding them.

The solver checks each map's metadata against the rules it would apply and
refuses to run on a mismatch.

## Measured runtimes

Solving each rule set from its published Percent16 map down to a residual of
`3e-14`, using `precomputed-gauss-seidel`.

Hardware: one node of the NYU Torch cluster — 2 sockets x 64 cores, 513 GB RAM,
x86_64, glibc 2.34. Each job used **16 of the 128 cores** and a fraction of the
memory; these are not whole-node runs.

<!-- RESULTS TABLE -->

Runtime is the solve itself, excluding reading the input map and writing the
output. The solver is single-node by design: score layers are strictly sequential
and every sweep needs a global reduction, so it does not distribute across nodes.
It has not needed to.

## Validation

Correctness rests on four checks that do not simply restate each other.

**Against an independently published solution.** Finkel is solved here from the
published Percent16 map, and a published `f64` Finkel map by Padraig Lamont
exists separately. Ours is diffed against it.

<!-- FINKEL VALIDATION -->

**Two iteration schemes against each other.** Blitz was solved twice, once with
`ondemand-jacobi` and once with `precomputed-gauss-seidel`, on different
machines. Across the 26,250,145 states of the 14 layers both runs completed they
agree to `1.847e-13` percentage points — about 13 units in the last place at
magnitude 100, which is what two different update orders converging to the same
fixed point at the `f64` floor should look like.

**The precomputed table against on-demand generation.** Building a successor
table proves every successor key resolves in the map, because a missing key
aborts the run. Beyond that, each layer checks that the precomputed Bellman
update matches the on-demand one to the bit on sampled states, and `bench-layer`
asserts exact agreement over 10,000 states.

**Encoding round trips.** `verify` and `preflight-train` decode states, re-encode
them, and confirm both the key and the recovered scores survive, along with the
generated successors. This has a useful side effect: run `preflight-train` on a
*published* map and it reports the Bellman residual of the published values.
Those values are already optimal, so a residual at the Percent16 quantisation
floor (`1.335e-3`) confirms the rules are right — a wrong path, dice distribution
or capture rule gives a residual orders of magnitude larger. This is how the
Finkel implementation was validated before any solve was run.

## Why two solvers

The Julia implementation builds the state space by breadth-first search and holds
an explicit transition tensor. That is the clearest way to express the method and
is fast enough to solve small piece counts interactively, but it does not reach
the published rule sets: a dense transition tensor for Masters' 500 million
states would need tens of GB before values and index structures.

The Rust solver exists for that regime. It starts from the published maps instead
of enumerating states, holds a successor table for one score layer at a time, and
reaches the `f64` floor on the largest rule set in minutes. Keeping both is
deliberate — the Julia code documents the method, the Rust code delivers the
result.

## Repository layout

```
julia/          readable reference solver (BFS + value iteration) and tests
rust/           optimised solver: refines published maps to f64
cluster/        Slurm scripts for the Rust solver
analysis/       simulations, Elo analysis, and the manuscript figure scripts
scripts/        download published maps from Hugging Face
docs/solver.md  operational notes for the Rust solver
```

## Running on a cluster

`cluster/` holds Slurm scripts for the Rust solver. They derive every path from
the repository location, so it can live anywhere; supply your own account and
partition:

```bash
sbatch --account=<account> --partition=<partition> \
    --export=ALL,UR_RULESET=masters cluster/train.sbatch
```

Because the solver checkpoints every completed score layer, the job queues its
own successor with `--dependency=afterany` before starting work, so a restart's
queue wait overlaps with the current job's runtime; the successor exits
immediately once the final map exists. Note that `--export` carries environment
variables but not resource requests, so a chained successor uses the script's own
`#SBATCH` defaults.

Keep requests small. These are minutes of work on 16 cores, and a small short job
gets placed by the backfill scheduler far sooner than a whole-node reservation.
`cluster/bench.sbatch` runs `bench-layer` to compare the two iteration strategies
on a single score layer.

Two portability notes, learned on NYU Torch and handled by `cluster/env.sh`:
login nodes may have no C toolchain at all, and may run a *newer* glibc than the
compute nodes (2.39 against 2.34), so a binary built on the login node dies at
runtime on a compute node. The scripts therefore build on a compute node against
a conda `sysroot_linux-64=2.17` toolchain, and keep `cargo`'s many small files
off a quota-limited home directory.

## Analysis and figures

`analysis/` holds the simulation and Elo code, and `analysis/plot_results.jl`
generates the manuscript figures, taking the results, figure output and model
directories as arguments:

```bash
julia analysis/plot_results.jl <results-dir> <figure-dir> <models-dir>
```

It needs [CairoMakie](https://docs.makie.org/), MakieExtra, and the solved maps.

## Solved maps

The published Percent16 and `f64` maps live on
[Hugging Face](https://huggingface.co/sothatsit/RoyalUrModels); fetch them with
`scripts/download_models.py`. They are deliberately not committed — the Masters
map alone is 3 GB.

## Related projects

- [RoyalUr-Java](https://github.com/RoyalUr/RoyalUr-Java) — reference
  implementation of the rules and the `.rgu` map format
- [RoyalUr-Python](https://github.com/RoyalUr/RoyalUr-Python) — reading the maps
  from Python
