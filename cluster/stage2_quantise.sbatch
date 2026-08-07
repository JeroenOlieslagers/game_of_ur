#!/usr/bin/env bash
#SBATCH --job-name=ur_s2q
#SBATCH --account=torch_pr_362_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/s2q_%x_%j.out
#SBATCH --error=logs/s2q_%x_%j.out

# The reference rate-distortion curve: the table stored at b bits per entry,
# uniform and Lloyd-Max, scored by policy regret. No model involved, and it is
# the bar every learned model must clear.
#
# Usage: sbatch stage2_quantise.sbatch <ruleset>

set -euo pipefail
# Slurm copies the batch script into a spool directory, so BASH_SOURCE points
# there rather than at the repository. UR_ROOT is exported at submission time
# and is the only reliable way back to the tree.
source "${UR_ROOT:?export UR_ROOT when submitting}/cluster/env.sh"

RULESET=${1:?ruleset required}
DATA=${UR_STAGE2:-$UR_ROOT/stage2}/$RULESET
RESULTS=${UR_STAGE2:-$UR_ROOT/stage2}/results
mkdir -p "$RESULTS"
cd "$RESULTS"

PYTHON=${UR_VENV:-$UR_ROOT/.venv}/bin/python
echo "=== on-policy state distribution (play view) ==="
"$PYTHON" "$UR_ROOT/scripts/quantise_curve.py" "$RULESET" \
    "$DATA/eval.csv" "$DATA/tensors.bin" --max-bits 12 \
    --out "$RESULTS/quantise_${RULESET}_onpolicy.json"

echo "=== uniform state distribution (compression view) ==="
"$PYTHON" "$UR_ROOT/scripts/quantise_curve.py" "$RULESET" \
    "$DATA/eval_uniform.csv" "$DATA/tensors.bin" --max-bits 12 \
    --out "$RESULTS/quantise_${RULESET}_uniform.json"

echo "quantisation curve complete for $RULESET"
