#!/usr/bin/env bash
# Build the Rust solver and refine one published Percent16 map into an f64 map.
#
#   scripts/run_solver.sh {finkel|blitz|masters} [model-dir] [tolerance] [max-sweeps] [strategy]

set -euo pipefail

usage() {
    echo "Usage: $0 {finkel|blitz|masters} [model-dir] [tolerance] [max-sweeps] [strategy]" >&2
    echo "  strategy: precomputed-gauss-seidel (default) | ondemand-jacobi" >&2
    exit 2
}

if [[ $# -lt 1 || $# -gt 5 ]]; then
    usage
fi

RULESET=$1
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR=${2:-"$ROOT/models"}
TOLERANCE=${3:-3e-14}
MAX_SWEEPS=${4:-10000}
# Precomputed successor indices plus in-place Gauss-Seidel: about 20x faster per
# sweep than regenerating successors, and it needs fewer sweeps. ondemand-jacobi
# is the original scheme, kept for comparison; both satisfy the same
# deterministic residual criterion.
STRATEGY=${5:-precomputed-gauss-seidel}

case "$RULESET" in
    finkel)
        INPUT="$MODEL_DIR/finkel.rgu"
        # Not finkel_f64.rgu: that name belongs to the published reference map,
        # which this output is compared against.
        OUTPUT="$MODEL_DIR/finkel_f64_ours.rgu"
        ;;
    blitz)
        INPUT="$MODEL_DIR/blitz.rgu"
        OUTPUT="$MODEL_DIR/blitz_f64.rgu"
        ;;
    masters)
        INPUT="$MODEL_DIR/masters3d.rgu"
        OUTPUT="$MODEL_DIR/masters3d_f64.rgu"
        ;;
    *)
        usage
        ;;
esac

if [[ ! -f $INPUT ]]; then
    echo "Missing input map: $INPUT" >&2
    echo "Fetch it with: python scripts/download_models.py $RULESET" >&2
    exit 1
fi

echo "Ruleset:    $RULESET"
echo "Input:      $INPUT"
echo "Output:     $OUTPUT"
echo "Tolerance:  $TOLERANCE percentage points"
echo "Max sweeps per score layer: $MAX_SWEEPS"
echo "Strategy:   $STRATEGY"
echo

cargo build --release --manifest-path "$ROOT/rust/Cargo.toml"
exec "$ROOT/rust/target/release/royalur_analysis" \
    train-f64 "$INPUT" "$OUTPUT" "$TOLERANCE" "$MAX_SWEEPS" "$STRATEGY"
