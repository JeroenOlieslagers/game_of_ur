#!/usr/bin/env bash
#SBATCH --job-name=ur_s2tune
#SBATCH --account=torch_pr_362_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/s2tune_%x_%j.out
#SBATCH --error=logs/s2tune_%x_%j.out

# Convergence control and recipe search, at one fixed capacity.
#
# This runs BEFORE the headline curves are believed, because until it is done
# every point on them says "what this recipe reached", not "what this capacity
# can do" -- and the looseness may vary with capacity, which would distort the
# shape of the curve rather than merely shifting it. The evidence that this
# matters is already in hand: the same 3,425-parameter model scored 0.1413 at
# 12k steps and 0.0841 at 60k.
#
# Coordinate descent rather than a grid: vary one axis at a time from a
# baseline. A full grid over four axes would be 100+ runs to answer a question
# that a dozen answers well enough.
#
# Usage: sbatch stage2_tune.sbatch <ruleset> [objective] [arch]

set -euo pipefail
source "${UR_ROOT:?export UR_ROOT when submitting}/cluster/env.sh"

RULESET=${1:?ruleset required}
OBJECTIVE=${2:-value}
ARCH=${3:-mlp}
DATA=${UR_STAGE2:-$UR_ROOT/stage2}/$RULESET
RESULTS=${UR_STAGE2:-$UR_ROOT/stage2}/results
mkdir -p "$RESULTS"
cd "$RESULTS"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
PYTHON=${UR_VENV:-$UR_ROOT/.venv}/bin/python
OUT=$RESULTS/tune_${RULESET}_${ARCH}_${OBJECTIVE}.jsonl

EXTRA=()
if [[ $OBJECTIVE == policy ]]; then
    EXTRA=(--decisions "$DATA/decisions.bin" --max-decisions "${UR_MAX_DECISIONS:-600000000}")
fi

run () {   # run <tag> <size> <steps> <lr> <batch> <loss>
    echo "=== tune $1: size=$2 steps=$3 lr=$4 batch=$5 loss=$6 ==="
    "$PYTHON" "$UR_ROOT/scripts/nn_sweep.py" "$RULESET" \
        "$DATA/tensors.bin" "$DATA/eval.csv" \
        --objective "$OBJECTIVE" --arch "$ARCH" \
        --sizes "$2" --steps "$3" --lr "$4" --batch "$5" --loss "$6" \
        --tag "$1" --out "$OUT" "${EXTRA[@]}"
}

# Baseline, then one axis at a time around it.
BASE_SIZE=256x2; BASE_STEPS=60000; BASE_LR=2e-3; BASE_BATCH=65536; BASE_LOSS=bce

# Axis 1 -- convergence. The point of the whole job: if 240k beats 60k by much,
# no comparison drawn at 60k is safe.
for s in 15000 60000 240000; do run steps "$BASE_SIZE" "$s" "$BASE_LR" "$BASE_BATCH" "$BASE_LOSS"; done

# Axis 2 -- learning rate.
for lr in 5e-4 1e-3 5e-3; do run lr "$BASE_SIZE" "$BASE_STEPS" "$lr" "$BASE_BATCH" "$BASE_LOSS"; done

# Axis 3 -- batch size. Interacts with steps: both scale samples seen, so the
# tag distinguishes them in the results file.
for b in 16384 262144; do run batch "$BASE_SIZE" "$BASE_STEPS" "$BASE_LR" "$b" "$BASE_LOSS"; done

# Axis 4 -- shape at roughly fixed parameter count. Depth against width is a
# free choice the sweep currently makes arbitrarily.
for shape in 176x3 128x5 108x8; do run shape "$shape" "$BASE_STEPS" "$BASE_LR" "$BASE_BATCH" "$BASE_LOSS"; done

# Axis 5 -- loss. BCE is better calibrated near 0 and 1, which is where the
# decisive positions are, but that is an argument rather than a measurement.
run loss "$BASE_SIZE" "$BASE_STEPS" "$BASE_LR" "$BASE_BATCH" mse

echo "tuning complete; results in $OUT"
