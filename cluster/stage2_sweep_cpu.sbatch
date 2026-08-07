#!/usr/bin/env bash
#SBATCH --job-name=ur_s2cpu
#SBATCH --account=torch_pr_362_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/s2cpu_%x_%j.out
#SBATCH --error=logs/s2cpu_%x_%j.out

# Insurance copy of the capacity sweep, on CPU.
#
# Only two GPU partitions are open to this account and the queue estimate for
# them runs to days, while `cpu_short` schedules in minutes. The small end of
# the sweep is cheap enough to train on 32 cores, so this guarantees a
# rate-distortion curve exists tomorrow even if no GPU has come free. The GPU
# jobs cover the same ground with more steps and the large sizes; whichever
# lands first, the results file records `size` and `steps` so the two never get
# confused for each other.
#
# Usage: sbatch stage2_sweep_cpu.sbatch <ruleset> <value|ordering>

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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
PYTHON=${UR_VENV:-$UR_ROOT/.venv}/bin/python
echo "host=$(hostname) ruleset=$RULESET objective=$OBJECTIVE threads=$OMP_NUM_THREADS"

SIZES=${UR_SIZES:-}
if [[ -z $SIZES ]]; then
    # Capacity-matched ladders: both families span ~3k to ~7M parameters.
    if [[ $ARCH == transformer ]]; then
        SIZES=16x1,16x2,24x2,32x2,48x2,64x2,96x2,128x2,192x2,256x2,384x3,512x3
    else
        SIZES=32x2,64x2,128x2,256x2,512x3
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
    --steps 12000 \
    --batch 16384 \
    --out "$RESULTS/nncpu_${RULESET}_${ARCH:-mlp}_${OBJECTIVE}_${FEATURES}_s${SEED}.jsonl" \
    "${EXTRA[@]}"

echo "cpu sweep complete for $RULESET/$OBJECTIVE"
