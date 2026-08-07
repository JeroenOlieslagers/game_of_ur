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
# Slurm copies the batch script into a spool directory, so BASH_SOURCE points
# there rather than at the repository. UR_ROOT is exported at submission time
# and is the only reliable way back to the tree.
source "${UR_ROOT:?export UR_ROOT when submitting}/cluster/env.sh"

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

echo "=== every decision in the game (policy training set) ==="
# Stride 1 is the whole space. A policy fitted to a *sample* compresses that
# sample, and since on-policy sampling revisits early-game states, a separately
# drawn on-policy evaluation set turned out to be entirely contained in a 400k
# training sample -- which made the measurement circular.
if [[ -s $OUT/decisions.bin ]] && [[ -z ${UR_FORCE_TENSORS:-} ]]; then
    echo "decisions.bin already present ($(stat -c%s "$OUT/decisions.bin") bytes); skipping"
else
    "$BINARY" dump-decisions "$MAP" "$OUT/decisions.bin" "${UR_DECISION_STRIDE:-1}" 0
fi

echo "=== full table ==="
# The successor files change whenever the CSV schema does; the table dump only
# changes when the packing does, and it is the expensive half. Skipping an
# existing one makes a schema change a thirty second job rather than an hour.
if [[ -s $OUT/tensors.bin ]] && [[ -z ${UR_FORCE_TENSORS:-} ]]; then
    echo "tensors.bin already present ($(stat -c%s "$OUT/tensors.bin") bytes); skipping"
    echo "set UR_FORCE_TENSORS=1 to rewrite it"
else
    "$BINARY" dump-tensors "$MAP" "$OUT/tensors.bin"
fi

ls -la "$OUT"
echo "dump complete for $RULESET"
