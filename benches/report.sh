#!/usr/bin/env bash
# Run innr's consumer-shape benches in both builds and write a markdown table.
#
#   benches/report.sh OUT.md [--numkong] [-- CRITERION_FILTER]
#
# Build 1: --no-default-features (innr_portable).
# Build 2: default features (innr_simd, plus consumer fallbacks and, with
#          --numkong, the NumKong reference column).
# Env: INNR_BENCH_WARMUP_MS / INNR_BENCH_MEASURE_MS (see benches/support),
#      CARGO_TARGET_DIR (respected). Needs cargo and python3.
set -euo pipefail

ROOT="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"

if [[ $# -lt 1 || "$1" == -* ]]; then
  echo "usage: $0 OUT.md [--numkong] [-- CRITERION_FILTER]" >&2
  exit 2
fi
OUT="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1")"
shift

FEATURES=()
FILTER=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --numkong) FEATURES=(--features bench-numkong); shift ;;
    --) shift; FILTER=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

BENCHES=(--bench consumer_pair --bench consumer_matvec --bench consumer_scan --bench consumer_maxsim)
TARGET="${CARGO_TARGET_DIR:-$ROOT/target}"
# A fresh Criterion directory per report, so stale results never mix in.
CRITERION_HOME="$TARGET/criterion-consumer-$(date +%Y%m%d-%H%M%S)"
export CRITERION_HOME
mkdir -p "$CRITERION_HOME"

cd "$ROOT"
load() { uptime | sed 's/.*load averages*: *//'; }
load_before="$(load)"
start=$(date +%s)
echo "== portable build (--no-default-features)" >&2
cargo bench --no-default-features "${BENCHES[@]}" -- ${FILTER[@]+"${FILTER[@]}"} >&2
echo "== simd build (default features ${FEATURES[*]-})" >&2
cargo bench ${FEATURES[@]+"${FEATURES[@]}"} "${BENCHES[@]}" -- ${FILTER[@]+"${FILTER[@]}"} >&2
elapsed=$(( $(date +%s) - start ))
load_after="$(load)"

if [[ "$(uname -s)" == Darwin ]]; then
  CPU="$(sysctl -n machdep.cpu.brand_string)"
else
  CPU="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
fi

INNR_REPORT_CPU="$CPU" \
INNR_REPORT_ARCH="$(uname -m)-$(uname -s | tr '[:upper:]' '[:lower:]')" \
INNR_REPORT_RUSTC="$(rustc -V)" \
INNR_REPORT_REV="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
INNR_REPORT_ELAPSED="$elapsed" \
INNR_REPORT_LOAD="$load_before / $load_after ($(getconf _NPROCESSORS_ONLN) CPUs)" \
INNR_REPORT_SETTINGS="warm-up ${INNR_BENCH_WARMUP_MS:-300} ms, measurement ${INNR_BENCH_MEASURE_MS:-1000} ms per bench${FEATURES[0]:+, NumKong enabled}" \
  python3 "$ROOT/benches/report.py" "$CRITERION_HOME" > "$OUT"
echo "wrote $OUT (${elapsed}s of benchmarking; raw data in $CRITERION_HOME)" >&2
