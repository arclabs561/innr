# Changelog

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
