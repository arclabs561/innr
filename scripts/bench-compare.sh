#!/usr/bin/env bash
# Compare benchmark results between different CPU targets
#
# Usage: ./scripts/bench-compare.sh
#
# Compares:
#   1. Native (your CPU's best features)
#   2. x86-64-v3 (AVX2+FMA, ~89% coverage)
#   3. x86-64 baseline (SSE2 only)

set -euo pipefail

echo "=== innr SIMD Benchmark Comparison ==="
echo ""

# Create results directory
mkdir -p bench_results

run_bench() {
    local target="$1"
    local name="$2"
    local flags="$3"
    
    echo "--- $name ($target) ---"
    RUSTFLAGS="$flags" cargo bench --bench dense -- --save-baseline "$name" 2>/dev/null
    echo ""
}

# Native (best available on this CPU)
echo "1. Running with native CPU features..."
run_bench "native" "native" "-C target-cpu=native"

# x86-64-v3 (AVX2+FMA)
echo "2. Running with x86-64-v3 (AVX2+FMA)..."
run_bench "x86-64-v3" "avx2" "-C target-cpu=x86-64-v3"

# x86-64 baseline (SSE2 only)
echo "3. Running with x86-64 baseline (SSE2)..."
run_bench "x86-64" "baseline" "-C target-cpu=x86-64"

echo ""
echo "=== Comparison Summary ==="
echo "Run 'cargo bench -- --baseline native' to compare against native"
echo "Run 'cargo bench -- --baseline avx2' to compare against AVX2"
