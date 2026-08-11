#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DEFAULT_INPUT=$ROOT/../ruleset_analysis/models/finkel.rgu
INPUT=${1:-${UR_FINKEL_INPUT:-$DEFAULT_INPUT}}
OUTPUT_DIR=${2:-$ROOT/long_game/results}
GAMES=${3:-10000000}
TOLERANCE=${UR_LONG_TOLERANCE:-1e-10}
MAX_SWEEPS=${UR_LONG_MAX_SWEEPS:-1000000}
SEED=${UR_LONG_SEED:-10047152335197783521}
PLOT_PYTHON=${UR_PLOT_PYTHON:-python3}
MODEL=$OUTPUT_DIR/finkel_longest.rgu
BINARY=$ROOT/rust/target/release/royalur_analysis

if [[ ! -f $INPUT ]]; then
    echo "missing Percent16 Finkel map: $INPUT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cargo build --release --manifest-path "$ROOT/rust/Cargo.toml"

if [[ ! -f $MODEL ]]; then
    "$BINARY" train-long-game "$INPUT" "$MODEL" "$TOLERANCE" "$MAX_SWEEPS"
fi

"$BINARY" verify-long-game "$MODEL" 10000
"$BINARY" simulate-long-game "$MODEL" "$OUTPUT_DIR" "$GAMES" "$SEED"
"$PLOT_PYTHON" "$ROOT/long_game/plot_lengths.py" \
    "$OUTPUT_DIR/length_histogram.csv" \
    "$OUTPUT_DIR/summary.json" \
    "$OUTPUT_DIR/length_distribution.png"
