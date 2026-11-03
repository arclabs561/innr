#!/usr/bin/env bash
# Build the probe and check that the portable f32 kernels still load full
# vector registers. The portable kernels rely on LLVM's auto-vectorizer, and
# small source changes have flipped them between full-width loads and two-lane
# `movsd` groups (1.4x slower at d128 on x86_64) without any test failing.
#
#   benches/probe/codegen.sh            # check
#   benches/probe/codegen.sh --print    # also print the instruction mix
#
# x86_64 counts `movups` (128-bit unaligned loads), aarch64 `ldr q` / `ldp q`.
# Each checked symbol needs MIN_WIDE of them: a vectorized main loop has 12
# to 68, while the four-lane leftover path alone has a few, so one stray wide
# load cannot pass the check. Needs objdump.
set -euo pipefail

ROOT="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
PRINT=0
[[ "${1:-}" == "--print" ]] && PRINT=1
MIN_WIDE=6

cargo build -q --release --manifest-path "$ROOT/Cargo.toml"
TARGET="$(cargo metadata -q --format-version 1 --no-deps --manifest-path "$ROOT/Cargo.toml" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"
BIN="$TARGET/release/innr-probe"
DIS="$(mktemp)"
trap 'rm -f "$DIS"' EXIT
objdump -d --no-show-raw-insn -C "$BIN" > "$DIS"

case "$(uname -m)" in
  x86_64) WIDE='movups' ;;
  arm64 | aarch64) WIDE='ldr[[:space:]]+q[0-9]|ldp[[:space:]]+q[0-9]' ;;
  *) echo "codegen.sh: no check for $(uname -m)" >&2; exit 0 ;;
esac

fail=0
for sym in probe_dot_portable probe_cosine_portable; do
  # GNU objdump prints `<innr_probe::sym>:`; macOS llvm-objdump prints `__ZN...sym...:`.
  body="$(awk -v s="$sym" '$0 ~ s && /:$/ {p=1; next} p && /^$/ {exit} p' "$DIS")"
  if [[ -z "$body" ]]; then
    echo "FAIL $sym: symbol not found in $BIN" >&2
    fail=1
    continue
  fi
  wide="$(printf '%s\n' "$body" | grep -cE "$WIDE" || true)"
  echo "$sym: $wide full-width loads"
  if [[ "$PRINT" == 1 ]]; then
    printf '%s\n' "$body" | awk '{for (i = 1; i <= NF; i++) if ($i ~ /^[a-z][a-z0-9.]+$/) {print $i; break}}' \
      | sort | uniq -c | sort -rn | head -12
  fi
  if [[ "$wide" -lt "$MIN_WIDE" ]]; then
    echo "FAIL $sym: $wide full-width loads (< $MIN_WIDE); the auto-vectorizer lost the main loop" >&2
    fail=1
  fi
done
exit "$fail"
