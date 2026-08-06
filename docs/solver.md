# High-precision Blitz and Masters solver

The Rust solver refines the public Percent16 maps into f64 maps while retaining
their exact state keys. It implements the verified Blitz and Masters rules and
processes score layers in the same dependency order as RoyalUr-Java.

## One-command run

From any directory:

```bash
bash scripts/run_solver.sh masters
```

The equivalent direct command is:

```bash
cargo build --release \
  --manifest-path rust/Cargo.toml

rust/target/release/royalur_analysis \
  train-f64 \
  models/masters3d.rgu \
  models/masters3d_f64.rgu \
  3e-14 \
  10000 \
  precomputed-gauss-seidel
```

The wrapper accepts optional model-directory, tolerance, maximum-iteration and
strategy arguments:

```text
run_f64_solver.sh {blitz|masters} [model-directory] [tolerance] [max-iterations] [strategy]
```

## Iteration strategies

Two schemes are available. Both process the same score layers in the same
dependency order, share the same checkpoint format (so a run started under one
can be resumed under the other), and accept a layer only when a deterministic
residual pass shows `max |T(v) - v| <= tolerance`.

`ondemand-jacobi` is the original scheme: successors are regenerated from the
decoded position on every sweep, and updates are applied out of place.

`precomputed-gauss-seidel` (default) materialises the successor indices for the
score layer currently being solved into a CSR table, then runs in-place sweeps
over it. Two things make this much cheaper:

- A sweep becomes a gather over precomputed indices instead of a decode plus a
  binary-search lookup per successor. Only the current layer needs a table:
  successors outside it have frozen values and are read by index, so the table
  is proportional to the layer, not to the whole map (about 99 bytes per state,
  measured 1.489 GB for the 15.0M-state Blitz layer `[0,0]`). Building it costs
  roughly six sweeps, against layers that need dozens.
- In-place updates let a sweep see values published earlier in the same sweep,
  which contracts faster than Jacobi's out-of-place update.

Measured on Blitz layer `[2,2]` (324,747 states, 8 threads), seconds per sweep:

| Strategy | s/sweep | Speedup |
| --- | --- | --- |
| `ondemand-jacobi` | 3.305 | 1x |
| precomputed successors, Jacobi | 0.237 | 14x |
| `precomputed-gauss-seidel` | 0.141 | 23x |

Observed per-sweep contraction was about 0.65 for Jacobi and 0.55 for
Gauss-Seidel, so Gauss-Seidel also needs fewer sweeps to reach the tolerance
(roughly 41 versus 57 from the Percent16 starting residual).

Sweeps are parallel in both schemes. The Gauss-Seidel sweep is the asynchronous
analogue of a serial Gauss-Seidel pass: threads publish updates as they go, so
which updates a thread observes is timing dependent. Values are accessed as
relaxed atomics so those concurrent reads and writes are well defined and cannot
tear. Because sweep deltas are therefore not reproducible, they serve only as a
progress signal; the layer is certified by a separate deterministic residual
pass, which is what gets recorded in the checkpoint.

## Benchmarking a single score layer

`bench-layer` reports preprocessing cost, seconds per sweep, the delta
trajectory and peak RSS for all three strategies on one layer, and asserts that
the precomputed Bellman update agrees exactly with the on-demand one:

```bash
royalur_analysis bench-layer <percent16-model.rgu> <min-score> <max-score> \
  [sweeps] [work-dir]
```

It writes only inside `work-dir`, so it cannot disturb a production checkpoint.

For a cluster copy of the repository, pass the directory containing
`masters3d.rgu` as the second argument. This avoids dependence on the local
absolute paths shown above.

## Masters job inputs and outputs

Input:

- `models/masters3d.rgu`: Percent16 source map with 500,981,472 stored states.

Outputs and resumable working data:

- `models/masters3d_f64.rgu`: final f64 RGU map.
- `models/masters3d_f64.checkpoint`: completed score-layer values.
- `models/masters3d_f64.layers/`: cached score-layer indices.
- `models/masters3d_f64.rgu.partial`: temporary final map while it is being
  written; it is renamed only after the write completes.

Restarting the same command restores all complete checkpoint records. An
interrupted, incomplete score layer is recomputed. The Percent16 input map is
never modified.

The stopping tolerance is in percentage points. `3e-14` reaches the practical
f64 residual floor observed in the Blitz run and is comparable with the
Finkel map's reported `2.8e-14` precision.

## Verification after completion

```bash
rust/target/release/royalur_analysis \
  verify \
  models/masters3d_f64.rgu \
  10000
```

Expected Masters state count: `500981472`. The verification checks encoding
round trips, decoded scores, key lookup, and generated successors.

## Cluster note

The solver uses all logical CPUs visible to one process through Rust scoped
threads. On Linux `thread::available_parallelism` honours the cgroup/cpuset, so
under Slurm `--cpus-per-task` alone sets the thread count; the job scripts log
`nproc` against `SLURM_CPUS_PER_TASK` to catch silent under-threading.

It is not distributed across nodes: score layers are strictly sequential, and
each sweep needs a global reduction before the layer can be committed.
Distributing it would require partitioning a layer's state indices across ranks
with a global maximum reduction per sweep. Single-node parallelism has been
sufficient since the switch to precomputed successors and Gauss-Seidel.

Multiple processes must not write the same checkpoint or output path
concurrently.

### Slurm scripts (NYU Torch)

`cluster/env.sh` holds the shared environment; the other scripts source it.

- `cluster/build.sbatch` builds the release binary on a compute node.
- `cluster/train.sbatch` runs a full solve. `UR_RULESET` selects
  `masters` (default) or `blitz`; `UR_STRATEGY` selects the iteration scheme.
- `cluster/bench.sbatch` runs `bench-layer` on one score layer.

Two environment details on this cluster:

- Login nodes have no C toolchain at all, and login nodes run glibc 2.39 while
  compute nodes run glibc 2.34. The binary is therefore built on a compute node,
  against a conda `gcc_linux-64` plus `sysroot_linux-64=2.17` toolchain created
  in `$UR_TOOLCHAIN`, so it stays runnable on compute nodes.
- `rustup`/`cargo` live under `/scratch`, not `$HOME`, because the home file
  quota is tight and cargo creates many files.

Walltime tiers are assigned by QOS: `cpu48` allows up to 2 days, `cpu168` up to
7 days but only on the `cs` partition. Because the solver checkpoints every
completed score layer, `train_masters.sbatch` queues its own successor with
`--dependency=afterany` before starting work, so a restart's queue wait overlaps
with the current job's runtime. The successor exits immediately if the final map
already exists, which is how the chain terminates.

