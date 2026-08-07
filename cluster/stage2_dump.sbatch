#!/usr/bin/env bash
#SBATCH --job-name=ur_s2dump
#SBATCH --account=torch_pr_362_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/s2dump_%x_%j.out
#SBATCH --error=logs/s2dump_%x_%j.out

# Writes the stage-2 datasets for one rule set:
#
#   tensors.bin     the whole table as (packed position, value), for training
#   eval.csv        60k on-policy decision positions with their candidate
#                   successors, for regret
#   eval_uniform.csv  the same drawn uniformly over the table -- the compression
#                   view rather than the play view; the two disagree and stage 1
#                   showed the disagreement matters
#   train.csv       400k positions, the training set for the ordering objective
#
# Usage: sbatch --job-name=<ruleset> stage2_dump.sbatch <ruleset> <map.rgu>

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"

RULESET=${1:?ruleset required}
MAP=${2:?map path required}
OUT=${UR_STAGE2:-$UR_ROOT/stage2}/$RULESET
mkdir -p "$OUT"

BINARY=$UR_ROOT/rust/target/release/royalur_analysis
if [[ ! -x $BINARY ]]; then
    echo "binary missing at $BINARY -- run build.sbatch first" >&2
    exit 1
fi

echo "host=$(hostname) ruleset=$RULESET map=$MAP"
echo "=== evaluation sets (fast, and they gate the big dump) ==="
# On-policy first: it runs the packing symmetry assertions implicitly through
# the same code path and takes seconds, so a packing bug fails here rather than
# after an hour of streaming.
"$BINARY" dump-successors "$MAP" "$OUT/eval.csv"         60000  onpolicy 0x5c1d47a29fb30e68
"$BINARY" dump-successors "$MAP" "$OUT/eval_uniform.csv" 60000  uniform  0x1f7b3c9d20e45a16
"$BINARY" dump-successors "$MAP" "$OUT/train.csv"        400000 onpolicy 0xa3e91f7c5b28d604

echo "=== full table ==="
"$BINARY" dump-tensors "$MAP" "$OUT/tensors.bin"

ls -la "$OUT"
echo "dump complete for $RULESET"
