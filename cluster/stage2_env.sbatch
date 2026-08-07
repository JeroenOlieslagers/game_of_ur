#!/usr/bin/env bash
#SBATCH --job-name=ur_s2env
#SBATCH --account=torch_pr_362_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/s2env_%j.out
#SBATCH --error=logs/s2env_%j.out

# Creates the Python environment for stage 2. Run once.
#
# The download happens inside a job rather than on a login node: login nodes are
# not the place to pull two gigabytes of CUDA wheels, and a compute node has the
# network path for it. Everything lands under /scratch, never a quota-limited
# home directory.

set -euo pipefail
# Slurm copies the batch script into a spool directory, so BASH_SOURCE points
# there rather than at the repository. UR_ROOT is exported at submission time
# and is the only reliable way back to the tree.
source "${UR_ROOT:?export UR_ROOT when submitting}/cluster/env.sh"

export UR_VENV=${UR_VENV:-$UR_ROOT/.venv}
export PIP_CACHE_DIR=$UR_ROOT/.pip-cache

if [[ ! -x $UR_VENV/bin/python ]]; then
    echo "creating venv at $UR_VENV"
    /scratch/jo2229/miniforge3/bin/python3 -m venv "$UR_VENV"
fi

"$UR_VENV/bin/python" -m pip install --upgrade pip
"$UR_VENV/bin/python" -m pip install \
    torch --index-url https://download.pytorch.org/whl/cu124
"$UR_VENV/bin/python" -m pip install numpy scikit-learn xgboost

echo "=== versions ==="
"$UR_VENV/bin/python" - <<'PY'
import numpy, torch
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda build", torch.version.cuda)
try:
    import xgboost
    print("xgboost", xgboost.__version__)
except ImportError:
    print("xgboost missing")
PY
echo "environment ready"
