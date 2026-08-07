#!/usr/bin/env bash
#SBATCH --job-name=ur_s2nn
#SBATCH --account=torch_pr_362_general
#SBATCH --partition=l40s_public,h200_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/s2nn_%x_%j.out
#SBATCH --error=logs/s2nn_%x_%j.out

# The capacity sweep for one rule set and one objective.
#
# A single GPU on a short walltime is deliberate. The models top out around 8M
# parameters and the whole table sits in device memory, so this is bound by
# kernel launches rather than by FLOPs; a second GPU would buy nothing. The two
# hour limit is about scheduling, not about need -- the sweep takes well under
# an hour, and a short request backfills into gaps that an eight hour one will
# sit behind for days on a low fairshare.
#
# Usage: sbatch stage2_sweep.sbatch <ruleset> <value|ordering>

set -euo pipefail
# Slurm copies the batch script into a spool directory, so BASH_SOURCE points
# there rather than at the repository. UR_ROOT is exported at submission time
# and is the only reliable way back to the tree.
source "${UR_ROOT:?export UR_ROOT when submitting}/cluster/env.sh"

RULESET=${1:?ruleset required}
OBJECTIVE=${2:-value}
ARCH=${3:-mlp}
FEATURES=${4:-none}
SEED=${5:-0}
DATA=${UR_STAGE2:-$UR_ROOT/stage2}/$RULESET
RESULTS=${UR_STAGE2:-$UR_ROOT/stage2}/results
mkdir -p "$RESULTS"
cd "$RESULTS"

PYTHON=${UR_VENV:-$UR_ROOT/.venv}/bin/python
echo "host=$(hostname) ruleset=$RULESET objective=$OBJECTIVE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

SIZES=${UR_SIZES:-}
if [[ -z $SIZES ]]; then
    # Capacity-matched ladders: both families span ~3k to ~7M parameters.
    if [[ $ARCH == transformer ]]; then
        SIZES=16x1,16x2,24x2,32x2,48x2,64x2,96x2,128x2,192x2,256x2,384x3,512x3
    else
        SIZES=32x2,64x2,128x2,256x2,512x3,1024x3,2048x3
    fi
fi

EXTRA=()
if [[ $OBJECTIVE == ordering ]]; then
    EXTRA=(--train-successors "$DATA/train.csv")
elif [[ $OBJECTIVE == policy ]]; then
    EXTRA=(--decisions "$DATA/decisions.bin"
           --max-decisions "${UR_MAX_DECISIONS:-600000000}")
fi

"$PYTHON" "$UR_ROOT/scripts/nn_sweep.py" "$RULESET" \
    "$DATA/tensors.bin" "$DATA/eval.csv" \
    --objective "$OBJECTIVE" --arch "${ARCH:-mlp}" --features "$FEATURES" --seed "$SEED" \
    --sizes "$SIZES" \
    --steps 60000 \
    --out "$RESULTS/nn_${RULESET}_${ARCH:-mlp}_${OBJECTIVE}_${FEATURES}_s${SEED}.jsonl" \
    "${EXTRA[@]}"

echo "sweep complete for $RULESET/$OBJECTIVE"
