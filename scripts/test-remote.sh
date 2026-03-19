#!/usr/bin/env bash
# Run innr tests and benchmarks on a remote host via SSH.
#
# Usage:
#   ./scripts/test-remote.sh <ssh-target> [--bench]
#
# Examples:
#   ./scripts/test-remote.sh ec2-user@10.0.1.5          # tests only
#   ./scripts/test-remote.sh ubuntu@graviton.local --bench  # tests + benchmarks
#
# Prerequisites on the remote host:
#   - Rust toolchain (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh)
#   - Git
#
# The script:
#   1. Pushes the current working tree (including uncommitted changes) to the remote
#   2. Runs cargo test, captures CPU info and SIMD features
#   3. Optionally runs cargo bench and downloads results
#   4. Cleans up the remote directory

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <ssh-target> [--bench]"
    echo ""
    echo "  ssh-target: user@host or SSH config alias"
    echo "  --bench:    also run benchmarks (slow)"
    exit 1
fi

SSH_TARGET="$1"
RUN_BENCH=false
if [[ "${2:-}" == "--bench" ]]; then
    RUN_BENCH=true
fi

REMOTE_DIR="/tmp/innr-test-$$"
RESULTS_DIR="bench_results/remote"
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

echo "=== innr remote test ==="
echo "Target: $SSH_TARGET"
echo "Remote dir: $REMOTE_DIR"
echo ""

# Create remote directory and sync source
echo "--- Syncing source to remote ---"
ssh "$SSH_TARGET" "mkdir -p $REMOTE_DIR"
rsync -az --exclude target --exclude bench_results --exclude '*.svg' \
    "$REPO_ROOT/" "$SSH_TARGET:$REMOTE_DIR/"

# Build the test script to run remotely
REMOTE_SCRIPT=$(cat <<'SCRIPT'
set -euo pipefail

cd REMOTE_DIR_PLACEHOLDER

echo "=== CPU Info ==="
if [[ -f /proc/cpuinfo ]]; then
    # Linux
    grep -m1 "model name" /proc/cpuinfo || true
    echo "Flags: $(grep -m1 "flags" /proc/cpuinfo | sed 's/.*://' | tr ' ' '\n' | grep -E 'avx|sse|neon|fma|sve' | tr '\n' ' ')"
elif command -v sysctl &>/dev/null; then
    # macOS
    sysctl -n machdep.cpu.brand_string 2>/dev/null || true
    sysctl -n hw.optional.neon 2>/dev/null && echo "NEON: available" || true
fi
echo "Arch: $(uname -m)"
echo ""

# Ensure Rust is available
export PATH="$HOME/.cargo/bin:$PATH"
if ! command -v cargo &>/dev/null; then
    echo "ERROR: cargo not found. Install Rust first:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
    exit 1
fi
echo "Rust: $(rustc --version)"
echo ""

echo "=== Building ==="
cargo build --all-features 2>&1

echo ""
echo "=== Running tests ==="
cargo test --all-features 2>&1

echo ""
echo "=== SIMD differential tests ==="
cargo test simd_correctness --all-features 2>&1

echo ""
echo "=== Numerical edge cases ==="
cargo test --test numerical_edge_cases --all-features 2>&1

if [[ "RUN_BENCH_PLACEHOLDER" == "true" ]]; then
    echo ""
    echo "=== Running benchmarks ==="
    # Use native CPU features for best results
    RUSTFLAGS="-C target-cpu=native" cargo bench --all-features 2>&1

    echo ""
    echo "=== Benchmark results saved ==="
fi

echo ""
echo "=== DONE ==="
SCRIPT
)

# Substitute placeholders
REMOTE_SCRIPT="${REMOTE_SCRIPT//REMOTE_DIR_PLACEHOLDER/$REMOTE_DIR}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//RUN_BENCH_PLACEHOLDER/$RUN_BENCH}"

# Execute
echo "--- Running tests on remote ---"
echo ""
ssh "$SSH_TARGET" "$REMOTE_SCRIPT" 2>&1 | tee "${RESULTS_DIR:-/dev/null}" || true

# Download benchmark results if they exist
if [[ "$RUN_BENCH" == "true" ]]; then
    mkdir -p "$REPO_ROOT/$RESULTS_DIR"
    echo ""
    echo "--- Downloading benchmark results ---"
    rsync -az "$SSH_TARGET:$REMOTE_DIR/target/criterion/" \
        "$REPO_ROOT/$RESULTS_DIR/$(ssh "$SSH_TARGET" 'uname -m')/" 2>/dev/null || \
        echo "No criterion results found (benchmarks may have failed)"
fi

# Cleanup
echo ""
echo "--- Cleaning up remote ---"
ssh "$SSH_TARGET" "rm -rf $REMOTE_DIR"

echo ""
echo "=== Complete ==="
