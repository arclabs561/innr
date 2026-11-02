#!/usr/bin/env bash
# Run innr's consumer-shape benches in both builds and write a markdown table.
#
#   benches/report.sh OUT.md [--numkong] [-- CRITERION_FILTER]
#
# Build 1: --no-default-features (innr_portable).
# Build 2: --features simd (innr_simd, plus consumer fallbacks).
# --numkong: also runs benches/numkong (a separate unpublished crate) into the
#            same Criterion directory, adding the NumKong reference column.
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

NUMKONG=0
FILTER=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --numkong) NUMKONG=1; shift ;;
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
echo "== simd build (--features simd)" >&2
cargo bench --features simd "${BENCHES[@]}" -- ${FILTER[@]+"${FILTER[@]}"} >&2
NK_NOTE=""
if [[ $NUMKONG == 1 ]]; then
  NK_NOTE=", NumKong enabled"
  echo "== numkong reference (benches/numkong)" >&2
  cargo bench --manifest-path "$ROOT/benches/numkong/Cargo.toml" --target-dir "$TARGET" -- ${FILTER[@]+"${FILTER[@]}"} >&2
fi
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
INNR_REPORT_SETTINGS="warm-up ${INNR_BENCH_WARMUP_MS:-300} ms, measurement ${INNR_BENCH_MEASURE_MS:-1000} ms per bench${NK_NOTE}" \
  python3 "$ROOT/benches/report.py" "$CRITERION_HOME" > "$OUT"
echo "wrote $OUT (${elapsed}s of benchmarking; raw data in $CRITERION_HOME)" >&2
