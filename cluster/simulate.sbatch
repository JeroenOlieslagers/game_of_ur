#!/usr/bin/env bash
#SBATCH --job-name=ur_simulate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/simulate_%A_%a.out
#SBATCH --error=logs/simulate_%A_%a.out

# One shard of the Monte Carlo simulations, as a Slurm array. Each task
# simulates the same states with a different game RNG, writing raw counts;
# scripts/aggregate_simulations.py then sums them exactly.
#
# Submit, for example 100 shards of 100,000 games per state, which is
# 100 states x 100,000 games x 100 shards = 1e9 compare games:
#
#   sbatch --account=<account> --partition=<partition> --array=0-99 \
#     --export=ALL,UR_RULESET=masters,UR_GAMES_PER_STATE=100000,UR_GAMES_PER_EPSILON=100000 \
#     cluster/simulate.sbatch
#
# Then aggregate on the cluster and copy only the two small CSVs back:
#
#   python scripts/aggregate_simulations.py shards results masters
#
# Sizing: measured throughput is about 9,000 games per second per core, so
# 1e9 games is roughly 31 core-hours. Total games are
# (compare-states x games-per-state + 101 x games-per-epsilon) x array size.

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"

RULESET=${UR_RULESET:-masters}
case $RULESET in
    masters) MODEL=$UR_MODELS/masters3d_f64.rgu ;;
    blitz)   MODEL=$UR_MODELS/blitz_f64.rgu ;;
    finkel)  MODEL=$UR_MODELS/finkel_f64_ours.rgu ;;
    *) echo "unknown ruleset: $RULESET" >&2; exit 2 ;;
esac

SHARD_DIR=${UR_SHARD_DIR:-$UR_ROOT/shards}
COMPARE_STATES=${UR_COMPARE_STATES:-100}
GAMES_PER_STATE=${UR_GAMES_PER_STATE:-20000}
GAMES_PER_EPSILON=${UR_GAMES_PER_EPSILON:-20000}
LABEL=${SLURM_ARRAY_TASK_ID:-0}
BINARY=$UR_ROOT/rust/target/release/royalur_analysis

echo "=== shard $LABEL of ${SLURM_ARRAY_TASK_COUNT:-1} on $(hostname) at $(date +%FT%T) ==="
echo "ruleset=$RULESET model=$MODEL"
echo "cpus=${SLURM_CPUS_PER_TASK:-?} affinity_cpus=$(nproc)"
echo "compare_states=$COMPARE_STATES games_per_state=$GAMES_PER_STATE games_per_epsilon=$GAMES_PER_EPSILON"

if [[ ! -f $MODEL ]]; then
    echo "missing solved map: $MODEL (run cluster/train.sbatch first)" >&2
    exit 1
fi

# Only shard 0 may build: concurrent array tasks would race on the target dir.
if [[ ! -x $BINARY ]]; then
    echo "missing $BINARY; build it first with cluster/build.sbatch" >&2
    exit 1
fi

mkdir -p "$SHARD_DIR"
"$BINARY" simulate "$MODEL" "$SHARD_DIR" "$LABEL" \
    "$COMPARE_STATES" "$GAMES_PER_STATE" "$GAMES_PER_EPSILON"

echo "=== shard $LABEL done $(date +%FT%T) ==="
