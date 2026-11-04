# Changelog

## 0.7.8

### Added

- `innr::scan`: `scores`, `scores_into`, `top_k`, `rerank`, `sign_dot_into`
  and `sign_dot_top_k` over a `Rows` source, so callers keep their layout:
  `[Vec<T>]`, `[&[T]]`, `[[T; N]]`, `scan::Flat` (a view of a row-major
  buffer, no copy) or `PackedRows`. One dispatch per call. Integer metrics
  rank exactly, `f32` puts NaN last.
- `rerank`: exact scores for a candidate list, the second stage after a
  cheap scan (for example `sign_dot_top_k` over binary codes of the same
  rows). On `PackedRows` and as `scan::rerank`.
- `cosine_parts(a, b)`: `(Σ a·b, Σ a², Σ b²)` in one pass, for callers with
  their own zero-norm rules.

### Changed

- `PackedRows` methods run through `scan`; byte-code scans got faster on
  x86_64 (i7-8559U, per row: 96 B 7.1 to 5.5 ns, 256 B 14.7 to 12.4 ns;
  2.0x and 1.4x per-code calls). 16 B codes stay faster per call (0.86x).

## 0.7.7

### Added

- `binary::sign_dot(query, code)`: `Σ query[i] · (bit i of code ? +1 : -1)`,
  bits LSB-first per byte, the asymmetric score of 1-bit (sign)
  quantization with an `f32` query. `PackedRows<u8>::sign_dot_into` and
  `sign_dot_top_k` scan packed codes in one dispatch. Against the usual
  per-bit loop with one accumulator (qntz 0.3's), per code: Apple M3 4.5x
  at 128 bits, 6.3 to 6.5x at 768 and 1536; i7-8559U 5.0x, 7.9 to 8.4x and
  15.8x. On x86 the packed scan is up to 1.25x faster again at 128 bits.

## 0.7.6

### Changed

- `innr::corpus::Corpus` is renamed `innr::packed::PackedRows`, matching
  `PackedBinary` and `PackedTernary`: it packs rows (float vectors or byte
  codes), and "corpus" suggested text documents. `innr::corpus::Corpus` stays
  as a deprecated alias and `innr::corpus::{Hit, PackError}` as re-exports,
  so 0.7.3 to 0.7.5 code compiles with a deprecation warning; they go in 0.8.
- `dot` documents that its summation order depends on the dispatch level
  and, on x86 from 512 elements, on the second argument's alignment, so
  `dot(a, b)` and `dot(b, a)` can differ in the last bits (within
  `(n + 2) · ε · Σ|a[i]·b[i]|`).

## 0.7.5

### Added

- `Corpus<u8>`: byte-packed codes (binary quantization, sketches) in a
  packed corpus. `scores` / `scores_into` take any byte metric; `top_k` and
  `top_k_many` rank `Hamming` (`u32`) and `Dot` (`u64`) scores exactly, ties
  by row order. `Corpus` is now `Corpus<T = f32>` and `Hit` is
  `Hit<S = f32>`, so existing code is unchanged. Byte rows pad to 16 bytes.
  On x86_64 the scan runs the SIMD Hamming kernel from 32 B, since a scan
  dispatches once rather than per row. i7-8559U, per row, against calling
  `hamming_distance` per code: 96 B 1.45 to 1.54x, 256 B 1.26 to 1.28x,
  16 B 0.75x (rows shorter than one AVX2 vector; keep per-row calls there).
  Apple M3: 16 B 1.23 to 1.37x, 96 to 256 B within 7%.

### Changed

- `Corpus::scores::<Cosine>` and `top_k::<Cosine>` keep each row's squared
  norm (computed on the first cosine scan) and compute one dot product per
  row instead of three sums. i7-8559U, per row, against per-row `cosine`:
  d128 1.9x, d768 1.2x, d1536 1.3x; corpus cosine is within 2 to 8% of a
  corpus dot scan. Results match `cosine` to 1e-5, including the f64
  fallback for overflowing inputs and 0.0 for zero rows.

## 0.7.4

### Changed

- `hamming_distance` is faster on Apple silicon (aarch64 with `dotprod`):
  the portable kernel is a byte loop that compiles to `cnt` plus `udot`,
  and the SIMD kernel takes over from 2048 B instead of 128 B. Apple M3
  Max, per call, one interleaved run (ratios held across runs, absolute times varied with machine load): 96 B 7.43 to 6.23 ns, 128 B 11.4 to 7.7 ns, 256 B 18.8 to
  11.9 ns. 0.7.0 to 0.7.3 were up to 1.6x slower than 0.6.3 at 96 to 256
  B on these targets. x86_64 and generic aarch64 Linux are unchanged.

## 0.7.3

### Added

- `corpus::Corpus`: packs `f32` rows once, each row on a 64-byte boundary
  from 16 dimensions up, and scores a query against every row inside one
  SIMD dispatch. `scores`, `scores_into`, `top_k` and `top_k_many` take the
  metric as a type parameter; `top_k` keeps the best `k` during the scan
  and returns `Hit { index, score }`, best first, NaN last. Brute-force
  scan at 128 dimensions, per row vs `Corpus`: x86_64 Linux 1.9-2.4x,
  Apple M3 Max 1.2-2.05x; exact top-k from 768 dimensions at parity.

## 0.7.2

### Fixed

- NaN scores rank last in every ranking (`TopK`, `batch_knn`,
  `batch_knn_dot`, `batch_knn_cosine`, the `u8` kNN, `distance` argmin),
  whatever the NaN's sign. `f32::total_cmp` ranked positive NaN first in
  descending rankings and negative NaN (x86's default for invalid
  operations) first in ascending ones.
- `cosine` returns the right value when entries are large enough (above
  about 1.8e19) that the `f32` sums of squares overflow; those inputs are
  recomputed in `f64`.
- `PackedTernary::new` stores the reserved code `0b11` as 0, so
  `ternary_hamming` agrees with `get` and `ternary_dot`.

## 0.7.1

### Fixed

- SIMD `cosine` (and `Metric<f32>` for `Cosine`) keeps four accumulators per
  sum instead of two from 64 `f32` (AVX2) or 32 (NEON) up. Two per sum left
  the one-pass loop latency-bound on Apple silicon, slower than 0.6.3 at
  large dimensions. Apple M3 Max, interleaved before and after: d768 84.7 to
  54.3 ns, d1536 187 to 105 ns, d16 to d128 within 6%. Ryzen 5600X: within 4%
  at every size.

## 0.7.0

### Changed

- The SIMD backend is now [`fearless_simd`](https://crates.io/crates/fearless_simd)
  1.2, behind the `simd` feature. innr's local `std::arch` unsafe kernels are
  replaced by one set of safe kernels generic over `fearless_simd::Simd`,
  dispatched at runtime to AVX-512 (Ice Lake-class), AVX2+FMA (x86-64-v3),
  NEON or WASM SIMD128. innr now forbids unsafe code. x86 CPUs with only the
  older AVX-512 subset (Skylake-X) run the AVX2+FMA kernels; x86 CPUs without
  x86-64-v3 run the portable kernels.
- The `simd` feature is on by default, so default users keep SIMD as in
  0.6.3. `default-features = false` gives a dependency-free build with the
  portable kernels; prefer it on CPUs with neither AVX2 nor NEON, where the
  `simd` build falls back to portable code behind a dispatch check (up to
  24% slower at small sizes on a Celeron J3455). The README's Performance
  section lists what `simd` buys on six measured ARM and x86 CPUs.
- The SIMD kernels sum in a different order than in 0.6.3 (native-width
  vectors, eight accumulators, two for inputs shorter than eight vectors,
  fused multiply-add), so f32 results can differ
  in the last bits (up to about 2e-6 relative on the test fixtures). Accuracy tests
  now bound every reduction against a higher-precision reference instead of
  pinning exact bits.
- The portable kernels (dot, squared L2, L1, cosine and norm for `f32` and
  `f64`, mixed `f32`-by-`u8` dot, `u8` dot and byte Hamming) keep 8 to 32
  independent accumulator lanes over `as_chunks` blocks, so LLVM vectorizes
  them without reassociating float adds. The `f32`/`f64` folds finish the
  leftover in four lanes, and their final reduction differs between x86 and
  other targets, chosen from disassembly of what LLVM generates for each.
  Portable cosine is three folds instead of one fused pass. On a Ryzen 5600X
  (baseline x86-64, no `simd`; before and after are separate runs on an idle
  host), portable dot at d128 went from about 20 to 13 ns and portable
  cosine at d768 from about 410 to 135 ns. Portable f32
  results can differ from 0.6.3 in the last bits (up to about 6e-6 relative
  on the test fixtures); integer results are unchanged.
- `slot_hamming_u64` counts mismatches with a plain loop that LLVM
  vectorizes on x86 (about 1.5x faster on a Ryzen 5600X; unchanged on NEON).
- On x86, the SIMD float reductions align the second input to the vector
  width before folding inputs of 512 `f32` (AVX2) or more. Allocators only
  guarantee 16 bytes, and a buffer 16 bytes off a 32-byte boundary splits
  half of AVX2's loads across cache lines: with both inputs off, `dot` at
  d4096 on a Ryzen 5600X went from 196 to 127 ns (aligned: 123 to 126).
- `fast_cosine_dispatch` with `simd` now returns the exact fused cosine on
  SIMD-sized inputs instead of a refined hardware rsqrt estimate.
- Byte Hamming switches to SIMD at 128 bytes (was 32 on NEON/AVX2 and 64 on
  AVX-512); below that the portable kernel, now on 64-bit words, measured as
  fast. `slot_hamming_u64` always runs the portable kernel: a SIMD kernel
  measured no faster on NEON at 4 to 1024 slots.
- Portable dense kernels take a four-lane path below 16 (`f32`) or 8 (`f64`)
  elements, measured faster than the wide lane array and than a sequential
  loop for 3 to 12 dimensions.
- MSRV is Rust 1.89, the minimum for `fearless_simd` 1.2.
- The extended property CI job now selects its named integration targets
  instead of an empty name filter; explicit test-local case limits remain in
  effect.

### Added

- `innr::capabilities()` returns `backend::Capabilities`: the SIMD level in
  use and `backend::Thresholds`, the input length at which each kernel family
  switches from the portable to the SIMD kernel. Its `Display` prints both on
  one line for benchmark output and bug reports.
  `backend::Backend::WasmSimd128` names the WASM level.
- `innr::metric`: metrics as zero-sized types (`Dot`, `Cosine`, `L2Sq`, `L1`,
  `Hamming`) implementing a sealed `Metric<T>` trait with a `DIRECTION`
  (similarity or distance), and a sealed `Element` trait carrying each element
  type's accumulator. The free functions are thin wrappers over these.

### Fixed

- Corrected the accuracy documentation of `fast_rsqrt` (maximum relative error
  about 1.75e-3, not ~23 bits), `fast_rsqrt_precise` (about 4.7e-6; it does
  not match IEEE `1.0 / x.sqrt()`) and `fast_cosine` (up to about 3.5e-3
  relative, including for normalized vectors). `fast_cosine_dispatch` now
  documents that inputs shorter than 16 elements use `fast_cosine`. Results
  are unchanged.

## 0.6.3

- `scalar::mixed_dot_u8_f32`: public SIMD-dispatched f32-by-u8 inner product
  for callers that own their own affine correction terms.
- `batch_l2_squared_into`, `batch_dot_into`, `batch_norms_into`, and
  `batch_cosine_into`: reusable-output variants for `VerticalBatch` scans.
  These preserve the allocating APIs while letting repeated-query callers keep
  distance/score buffers across scans.

## 0.6.2

Additive release. One new primitive plus README coverage for kernels that
already shipped.

- `slot_compare_counts`: the `(eq, lt, gt)` per-position comparison triple (returned as `SlotCounts`), generalizing `slot_hamming` (which is `len - eq`). SetSketch (Ertl 2021) and UltraLogLog (Ertl 2024) joint Jaccard/cardinality estimators consume the `lt`/`gt` counts, not just `eq`; innr returns the metric-agnostic triple and leaves the estimator to the consuming sketch crate. Portable and generic over `T: Ord`; a SIMD specialization is deferred until a consumer profiles it. Neither anndists nor simsimd exposes this triple.
- Docs: the README now lists `slot_hamming_u16` (b-bit MinHash at b=16) and `slot_compare_counts`, both exported since their respective releases but previously undocumented in the API summary.

## 0.6.1

Additive release. New SIMD kernels, introspection, and a substantially
widened test surface; the only behavior change is a panic-on-mismatch fix
in `fast_cosine_dispatch` (see below).

- f64 reductions are now SIMD-dispatched: `dense_f64::{dot_f64, l2_distance_squared_f64, l1_distance_f64}` (and `cosine_f64`/`norm_f64`/`l2_distance_f64` via them) run AVX-512 (8 doubles/zmm, masked tail) / AVX2 / NEON with exact FMA accumulation. Previously portable-only.
- `slot_hamming_u16` and `slot_hamming_u64`: SIMD differing-slot counts for b-bit MinHash at b=16 (u16) and 64-bit sketches (u64), alongside the existing `slot_hamming_u32`. AVX-512BW/AVX2/NEON.
- `backend::{dense_backend, slot_backend}`: report which kernel family (`Avx512`/`Avx2Fma`/`Neon`/`Portable`) a given length dispatches to on the current machine.
- `dense::normalize_with_norm`: normalize in place and return the original L2 norm.
- Fixes: `sparse_dense_dot` no longer reaches out-of-bounds on unsorted input from safe code; `fast_cosine` compares squared norms against the squared epsilon (small-norm vectors no longer collapse to 0); `TopK` uses total order so a NaN candidate cannot poison the eviction gate; `PackedBinary`/`PackedTernary::new` mask padding bits; `maxsim_cosine` pre-checks dimensions; `fast_cosine_dispatch` now panics on length mismatch regardless of input size (previously truncated silently for SIMD-sized inputs).
- Testing: AVX-512 kernels now execute in CI under Intel SDE; added differential fuzz harnesses, a native Linux-ARM job, coverage and weekly mutation jobs, an aarch64-target clippy row, and l1/mixed-dot differential coverage.
- Docs: crate-level Contracts section; `minhash_jaccard` documented as unbiased for classic MinHash / wide slots and at b>=14, with the small-b b-bit correction noted and deliberately left to the consumer.

## 0.6.0

- New optional `anndists` feature: implements `anndists::dist::Distance` for `DistCosine`, `DistDot`, `DistL2`, `DistL1`, `DistHamming`, and `DistSlotU32`, making innr's metrics drop-in distances for `hnsw_rs` (which binds to anndists's trait). The impls delegate to innr's own `Distance` trait so the two cannot drift. Verified end-to-end in `tests/anndists_interop.rs` by building real `hnsw_rs` indexes (u32 MinHash sketches with `DistSlotU32`, f32 embeddings with `DistCosine`). Closes the adapter gap documented in 0.5.1 (issue #1).
- The default build remains dependency-free; the feature pulls in `anndists 0.1.5` only when enabled.

## 0.5.1

Refinements to the 0.5.0 slot/distance surface, grounded in how the
`anndists` / `hnsw_rs` ecosystem actually consumes these metrics:

- Added `jaccard_distance`: the fraction of differing slots (`1 - minhash_jaccard`), matching the value `anndists`'s integer `DistHamming` returns. `minhash_jaccard` is a similarity (larger is closer); `jaccard_distance` is the distance form indexes expect.
- `DistSlotU32::eval` now returns the normalized differing fraction (`jaccard_distance`) instead of the raw differing count, so an index built on it sees the same distance scale as the `anndists` ecosystem. Ordering is unchanged.
- Corrected the `distance` module docs: innr's `Distance` is its own trait, not a drop-in for `hnsw_rs` (which binds to `anndists::dist::distances::Distance`). Using innr's metrics in an `hnsw_rs` index needs an adapter implementing that trait, deliberately kept out of the dependency-free core.

## 0.5.0

New modules:
- `slot`: integer-slot Hamming distance and MinHash Jaccard estimation. `slot_hamming_u32` counts differing `u32` slots with full SIMD dispatch (AVX-512F mask compare, AVX2, NEON, portable); `slot_hamming` is a generic `PartialEq` fallback for `u16`/`u64`/other widths; `minhash_jaccard` returns the standard MinHash collision-probability estimate (fraction of matching slots).
- `distance`: generic `Distance` trait (`eval(&self, &[T], &[T]) -> f32`, smaller = closer) with zero-sized metric types `DistCosine`, `DistDot`, `DistL2`, `DistL1`, `DistHamming`, `DistSlotU32`. Mirrors the `anndists` / `hnsw_rs` convention so innr's metrics can back a generic index.

These are additive; no breaking changes. Motivated by a request to support integer Hamming for MinHash collision estimation and a generic distance trait for use as a pluggable backend (issue #1).

## 0.2.0

New modules:
- `scalar`: uint8 affine quantization (4x memory compression) with SIMD-accelerated asymmetric dot product (f32 query x u8 corpus). NEON and AVX2 kernels. Includes `fit_quantile` (outlier-clipping), `fit_vectors` (corpus-wide fit), `query_context` (amortized precomputation), and `batch_knn_u8` (quantized batch search).

Breaking changes:
- `BatchKnnResult.distances` renamed to `BatchKnnResult.scores` (the field stores similarities for cosine/dot kNN, not distances)
- `PackedBinary` and `PackedTernary` fields are now private. Use `.data()` and `.dimension()` accessors. `new()` now validates data length.
- Removed `dot_portable`, `cosine_portable`, `sparse_dot_portable` from crate root re-exports (still accessible via `innr::dense::dot_portable` etc.)
- Removed `fast_cosine_distance` (trivial `1.0 - fast_cosine()` wrapper)
- Removed `batch_asymmetric_dot` (trivial `.iter().map().collect()` wrapper)
- Removed `sparse`, `maxsim`, `full` feature flags (all code compiles unconditionally)

New features:
- Fused single-pass cosine SIMD kernel: ~3x less memory bandwidth, 85-97% faster than 0.1.x (AVX-512, AVX2, NEON)
- SIMD-accelerated L1 (Manhattan) distance: NEON `vabdq_f32`, AVX-512 `_mm512_abs_ps`, AVX2 `andnot`
- `batch_knn_cosine`: top-k by cosine similarity
- `batch_knn_dot`: top-k by dot product (MIPS)
- `batch_knn_filtered`: k-NN with predicate pushdown (`Fn(usize) -> bool`)
- `batch_knn_reordered`: exact k-NN with variance-ordered dimension processing
- `VerticalBatch::from_slices`: construct from `&[&[f32]]` without Vec allocation
- `batch_dimension_variance`: per-dimension variance computation

Fixes:
- `sparse_dot` and `maxsim` dimension checks promoted from `debug_assert` to `assert` (silent wrong results in release builds)
- `sparsity()` no longer divides by zero for dimension-0 vectors
- `batch_cosine` uses `NORM_EPSILON` constant instead of hardcoded `1e-9`
- Cosine SIMD tests handle zero-norm edge case correctly on x86_64
- `MIN_DIM_SIMD` gated behind `cfg(x86_64 | aarch64)` to fix armv7 cross-compilation

Quality:
- `#[must_use]` on all pure public functions
- `PartialEq` derived for `VerticalBatch` and `BatchKnnResult`
- `MIN_DIM_SIMD` and `NORM_EPSILON` demoted to `pub(crate)` (implementation details)
- AVX-512 added to SIMD dispatch documentation
- Batch benchmark suite added

## 0.1.9

- Single-pass L2 SIMD kernel, fix numerical stability
- Promote debug_assert to assert, document batch_knn_adaptive
- Remove clifford and metric modules

## 0.1.8

- Fix length mismatch panic, add SAFETY docs, clean dead code, harden tests
