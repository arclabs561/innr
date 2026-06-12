# Changelog

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
