#!/usr/bin/env bash
#SBATCH --job-name=ur_s2nn
#SBATCH --account=torch_pr_362_general
#SBATCH --partition=l40s,h100,h200,rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/s2nn_%x_%j.out
#SBATCH --error=logs/s2nn_%x_%j.out

# The capacity sweep for one rule set and one objective.
#
# A single GPU is deliberate. The models top out around 8M parameters and the
# whole table sits in device memory, so this is bound by kernel launches rather
# than by FLOPs -- a second GPU would buy nothing and would schedule far worse.
#
# Usage: sbatch stage2_sweep.sbatch <ruleset> <value|ordering>

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"

RULESET=${1:?ruleset required}
OBJECTIVE=${2:-value}
DATA=${UR_STAGE2:-$UR_ROOT/stage2}/$RULESET
RESULTS=${UR_STAGE2:-$UR_ROOT/stage2}/results
mkdir -p "$RESULTS"
cd "$RESULTS"

PYTHON=${UR_VENV:-$UR_ROOT/.venv}/bin/python
echo "host=$(hostname) ruleset=$RULESET objective=$OBJECTIVE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

EXTRA=()
if [[ $OBJECTIVE == ordering ]]; then
    EXTRA=(--train-successors "$DATA/train.csv")
fi

"$PYTHON" "$UR_ROOT/scripts/nn_sweep.py" "$RULESET" \
    "$DATA/tensors.bin" "$DATA/eval.csv" \
    --objective "$OBJECTIVE" \
    --sizes 32x2,64x2,128x2,256x2,512x3,1024x3,2048x3 \
    --steps 60000 \
    --out "$RESULTS/nn_${RULESET}_${OBJECTIVE}.jsonl" \
    "${EXTRA[@]}"

echo "sweep complete for $RULESET/$OBJECTIVE"
