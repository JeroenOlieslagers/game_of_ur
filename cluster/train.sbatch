#!/usr/bin/env bash
#SBATCH --job-name=ur_solve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=logs/solve_%j.out
#SBATCH --error=logs/solve_%j.out

# Refine one published Percent16 map into an f64 map, on a single node.
#
# Submit with (from the repository root):
#   sbatch --account=<account> --partition=<partition> \
#     --export=ALL,UR_RULESET=masters cluster/train.sbatch
#
# Sizing: the whole job is minutes, not hours. Keep --cpus-per-task and --time
# small so the backfill scheduler can place the job; see the repository README for
# measured runtimes.
#
# Parallelism: the solver fans each sweep across
# thread::available_parallelism() scoped threads, which on Linux honours the
# Slurm cpuset, so --cpus-per-task alone sets the thread count. Score layers are
# strictly sequential (each depends on the higher-score layers it can reach) and
# every sweep needs a global reduction, so this does not distribute across nodes.
#
# Restart: the solver checkpoints every COMPLETED score layer, so re-running the
# same command resumes. This script queues its own successor up front with
# --dependency=afterany so a restart's queue wait overlaps with this job's
# runtime; the successor exits immediately if the final map already exists.
# Resource flags are NOT inherited by the successor (sbatch --export carries
# environment variables, not resource requests), so it uses the defaults above.

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"

RULESET=${UR_RULESET:-masters}
case $RULESET in
    masters) INPUT=$UR_MODELS/masters3d.rgu; OUTPUT=$UR_MODELS/masters3d_f64.rgu ;;
    blitz)   INPUT=$UR_MODELS/blitz.rgu;     OUTPUT=$UR_MODELS/blitz_f64.rgu ;;
    # Finkel also has a published f64 map, so our output can be diffed against it.
    finkel)  INPUT=$UR_MODELS/finkel.rgu;    OUTPUT=$UR_MODELS/finkel_f64_ours.rgu ;;
    *) echo "unknown ruleset: $RULESET" >&2; exit 2 ;;
esac
TOLERANCE=${UR_TOLERANCE:-3e-14}
MAX_ITERATIONS=${UR_MAX_ITERATIONS:-10000}
# Precomputed successor indices + in-place Gauss-Seidel: ~14x per sweep from
# precomputation and a further ~1.7x from Gauss-Seidel, measured on blitz.
STRATEGY=${UR_STRATEGY:-precomputed-gauss-seidel}
BINARY=$UR_ROOT/rust/target/release/royalur_analysis

CHAIN=${UR_CHAIN:-0}
MAX_CHAIN=${UR_MAX_CHAIN:-4}

echo "=== job $SLURM_JOB_ID chain-link $CHAIN on $(hostname) at $(date +%FT%T) ==="
echo "ruleset=$RULESET strategy=$STRATEGY tolerance=$TOLERANCE"
echo "cpus_allocated=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}MB"
# The solver sizes its thread pool from sched_getaffinity, which is what nproc
# reports, so these two numbers must agree or we are silently under-threaded.
echo "affinity_cpus=$(nproc)"
if [[ -n ${SLURM_CPUS_PER_TASK:-} && $(nproc) -ne ${SLURM_CPUS_PER_TASK} ]]; then
    echo "WARNING: nproc=$(nproc) != cpus-per-task=${SLURM_CPUS_PER_TASK}; solver will use $(nproc) threads"
fi
# Recorded so reported runtimes can be tied to specific hardware.
echo "cpu_model=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -1)"
echo "node_cpus_total=$(getconf _NPROCESSORS_ONLN) node_memory_kb=$(sed -n 's/^MemTotal:[[:space:]]*//p' /proc/meminfo)"
echo "glibc=$(ldd --version | head -1)"

# /usr/bin/time is not guaranteed to exist on the compute images.
TIME_CMD=()
if [[ -x /usr/bin/time ]]; then
    TIME_CMD=(/usr/bin/time -v)
fi

# Already finished? Nothing to do (this is how the chain terminates).
if [[ -f $OUTPUT ]]; then
    echo "final map already exists: $OUTPUT"
    ls -la "$OUTPUT"
    exit 0
fi

# Queue the successor now so its wait overlaps with this job's runtime.
if (( CHAIN < MAX_CHAIN )); then
    NEXT=$(sbatch --parsable \
        --dependency=afterany:"$SLURM_JOB_ID" \
        --export=ALL,UR_CHAIN=$((CHAIN + 1)),UR_MAX_CHAIN="$MAX_CHAIN",UR_RULESET="$RULESET",UR_STRATEGY="$STRATEGY",UR_TOLERANCE="$TOLERANCE",UR_MAX_ITERATIONS="$MAX_ITERATIONS" \
        --job-name="ur_$RULESET" \
        "$UR_CLUSTER_DIR/train.sbatch")
    echo "queued successor chain-link $((CHAIN + 1)) as job $NEXT"
else
    echo "chain limit $MAX_CHAIN reached; no successor queued"
fi

# Build if needed. Done on the compute node so the binary links against the
# compute-node glibc (2.34), not the login-node glibc (2.39).
#
# Serialised with flock: two jobs running cargo concurrently against the same
# target directory corrupt each other's intermediate objects (one deletes an
# object the other is about to link). The check is repeated inside the lock so
# the loser of the race skips the build entirely.
(
    flock 9
    if [[ ! -x $BINARY ]]; then
        echo "=== building solver ==="
        if ! command -v cc >/dev/null 2>&1 && [[ ! -x ${CC:-/nonexistent} ]]; then
            source "$UR_CONDA_SH"
            conda create -y -p "$UR_TOOLCHAIN" -c conda-forge \
                gcc_linux-64 sysroot_linux-64=2.17 binutils_linux-64
            source "$UR_CLUSTER_DIR/env.sh"
        fi
        cargo build --release --manifest-path "$UR_ROOT/rust/Cargo.toml"
    else
        echo "binary already present; skipping build"
    fi
) 9>"$UR_ROOT/rust/.build.lock"

if [[ ! -x $BINARY ]]; then
    echo "build did not produce $BINARY" >&2
    exit 1
fi
echo "binary: $(ls -la "$BINARY")"

# Run the solver directly rather than under srun: this is a single-task
# shared-memory job, and an srun step's cpu-binding can confine the thread pool.
echo "=== train-f64 start $(date +%FT%T) ==="
set +e
"${TIME_CMD[@]}" "$BINARY" train-f64 "$INPUT" "$OUTPUT" "$TOLERANCE" "$MAX_ITERATIONS" "$STRATEGY"
STATUS=$?
set -e
echo "=== train-f64 exited status=$STATUS at $(date +%FT%T) ==="

if [[ -f $OUTPUT ]]; then
    echo "=== verifying $OUTPUT ==="
    ls -la "$OUTPUT"
    "$BINARY" verify "$OUTPUT" 10000
fi

echo "=== slurm accounting ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,AveCPU,TotalCPU,State 2>/dev/null || true

exit $STATUS
