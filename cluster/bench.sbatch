#!/usr/bin/env bash
#SBATCH --job-name=ur_bench
#SBATCH --account=torch_pr_362_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.out

# Benchmarks the Masters score layer given by UR_LAYER (default "0 0", the
# largest layer) under three strategies: on-demand Jacobi (current production
# path), precomputed-successor Jacobi, and precomputed-successor in-place
# Gauss-Seidel. Reports preprocessing cost, seconds per sweep, the delta
# trajectory (convergence rate) and peak RSS.
#
# Writes only into its own bench work directory, so it cannot touch the
# production checkpoint or output map.

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"

INPUT=$UR_MODELS/masters3d.rgu
BINARY=$UR_ROOT/rust/target/release/royalur_analysis
LAYER=${UR_LAYER:-"0 0"}
SWEEPS=${UR_SWEEPS:-10}
WORKDIR=${UR_BENCH_DIR:-$UR_ROOT/bench}

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date +%FT%T) ==="
echo "cpus_allocated=${SLURM_CPUS_PER_TASK:-?} affinity_cpus=$(nproc) mem=${SLURM_MEM_PER_NODE:-?}MB"
if [[ -n ${SLURM_CPUS_PER_TASK:-} && $(nproc) -ne ${SLURM_CPUS_PER_TASK} ]]; then
    echo "WARNING: nproc=$(nproc) != cpus-per-task=${SLURM_CPUS_PER_TASK}"
fi
echo "layer=$LAYER sweeps=$SWEEPS"

if [[ ! -x $BINARY ]]; then
    echo "=== building solver ==="
    if ! command -v cc >/dev/null 2>&1 && [[ ! -x ${CC:-/nonexistent} ]]; then
        source "$UR_CONDA_SH"
        conda create -y -p "$UR_TOOLCHAIN" -c conda-forge \
            gcc_linux-64 sysroot_linux-64=2.17 binutils_linux-64
        source "$UR_CLUSTER_DIR/env.sh"
    fi
    cargo build --release --manifest-path "$UR_ROOT/rust/Cargo.toml"
fi

mkdir -p "$WORKDIR"
echo "=== bench-layer start $(date +%FT%T) ==="
# shellcheck disable=SC2086
"$BINARY" bench-layer "$INPUT" $LAYER "$SWEEPS" "$WORKDIR"
echo "=== bench-layer done $(date +%FT%T) ==="

echo "=== slurm accounting ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,AveCPU,TotalCPU,State 2>/dev/null || true
