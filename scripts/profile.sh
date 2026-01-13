#!/usr/bin/env bash
# Profile innr SIMD operations using flamegraph
#
# Usage: ./scripts/profile.sh [bench_name]
#   bench_name: dense, sparse, maxsim (default: dense)
#
# Requirements:
#   cargo install flamegraph
#   On macOS: needs dtrace permissions (SIP partially disabled)
#   On Linux: needs perf (sudo apt install linux-tools-generic)

set -euo pipefail

BENCH="${1:-dense}"

echo "=== Profiling innr ($BENCH) ==="
echo "Building with release profile (debug symbols enabled)..."

# Build the benchmark
cargo build --release --bench "$BENCH"

# Find the benchmark binary
BENCH_BIN=$(find target/release/deps -name "${BENCH}-*" -type f -perm +111 | head -1)

if [[ -z "$BENCH_BIN" ]]; then
    echo "Error: Could not find benchmark binary for $BENCH"
    exit 1
fi

echo "Running flamegraph on $BENCH_BIN..."

# Generate flamegraph
# --root is needed on macOS for dtrace
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Note: On macOS, you may need to run with sudo or disable SIP for dtrace"
    sudo flamegraph -o "flamegraph_${BENCH}.svg" -- "$BENCH_BIN" --bench
else
    flamegraph -o "flamegraph_${BENCH}.svg" -- "$BENCH_BIN" --bench
fi

echo ""
echo "Flamegraph saved to: flamegraph_${BENCH}.svg"
echo "Open in browser to inspect hot paths."
