#!/usr/bin/env bash
#SBATCH --job-name=ur_long_game
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=long_game/long_game_%j.out
#SBATCH --error=long_game/long_game_%j.out

set -euo pipefail
# Slurm executes a spool copy, so BASH_SOURCE does not point back into the
# repository. Submit from the repository root, or export UR_ROOT explicitly.
ROOT_HINT=${UR_ROOT:-${SLURM_SUBMIT_DIR:?submit from the repository root or export UR_ROOT}}
source "$ROOT_HINT/cluster/env.sh"

INPUT=${UR_FINKEL_INPUT:-$UR_MODELS/finkel.rgu}
OUTPUT_DIR=${UR_LONG_OUTPUT:-$UR_ROOT/long_game/results}
GAMES=${UR_LONG_GAMES:-10000000}
TOLERANCE=${UR_LONG_TOLERANCE:-1e-10}
MAX_SWEEPS=${UR_LONG_MAX_SWEEPS:-1000000}
SEED=${UR_LONG_SEED:-10047152335197783521}
PLOT_PYTHON=${UR_PLOT_PYTHON:-python3}
BINARY=$UR_ROOT/rust/target/release/royalur_analysis
MODEL=$OUTPUT_DIR/finkel_longest.rgu

echo "job=$SLURM_JOB_ID host=$(hostname) cpus=$(nproc)"
echo "input=$INPUT output=$OUTPUT_DIR games=$GAMES tolerance=$TOLERANCE"
mkdir -p "$OUTPUT_DIR" "$UR_LOGS"

if [[ ! -f $INPUT ]]; then
    echo "missing Percent16 Finkel map: $INPUT" >&2
    exit 1
fi

(
    flock 9
    cargo build --release --manifest-path "$UR_ROOT/rust/Cargo.toml"
) 9>"$UR_ROOT/rust/.build.lock"

if [[ ! -f $MODEL ]]; then
    "$BINARY" train-long-game "$INPUT" "$MODEL" "$TOLERANCE" "$MAX_SWEEPS"
fi
"$BINARY" verify-long-game "$MODEL" 10000
"$BINARY" simulate-long-game "$MODEL" "$OUTPUT_DIR" "$GAMES" "$SEED"
"$PLOT_PYTHON" "$UR_ROOT/long_game/plot_lengths.py" \
    "$OUTPUT_DIR/length_histogram.csv" \
    "$OUTPUT_DIR/summary.json" \
    "$OUTPUT_DIR/length_distribution.png"

sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,TotalCPU,State 2>/dev/null || true
