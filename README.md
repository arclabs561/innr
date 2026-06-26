# innr

[![crates.io](https://img.shields.io/crates/v/innr.svg)](https://crates.io/crates/innr)
[![Documentation](https://docs.rs/innr/badge.svg)](https://docs.rs/innr)
[![CI](https://github.com/arclabs561/innr/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/innr/actions/workflows/ci.yml)

SIMD-accelerated vector similarity primitives: dot, cosine, and
Euclidean distance over `f32` / `u8`, plus binary, ternary, and
scalar quantization. Targets x86 AVX2/AVX-512 and aarch64 NEON,
with scalar fallback.

## Quickstart

```toml
[dependencies]
innr = "0.6"
```

```rust
use innr::{dot, cosine, norm};

let a = [1.0_f32, 0.0, 0.0];
let b = [0.707, 0.707, 0.0];

let d = dot(&a, &b);      // 0.707
let c = cosine(&a, &b);   // 0.707
let n = norm(&a);         // 1.0
```

### Batch search

```rust
use innr::batch::{VerticalBatch, batch_knn_dot};

// 4 vectors of dimension 3
let corpus = vec![
    vec![1.0f32, 0.0, 0.0],
    vec![0.0, 1.0, 0.0],
    vec![0.7, 0.7, 0.0],
    vec![0.0, 0.0, 1.0],
];
let batch = VerticalBatch::from_rows(&corpus);

let query = [0.8f32, 0.6, 0.0];
let result = batch_knn_dot(&query, &batch, 2);
// result.indices: top-2 nearest by dot product
// result.scores: corresponding similarity scores
```

## Operations

**Core (f32)**: `dot`, `cosine`, `norm`, `l2_distance`, `l2_distance_squared`, `l1_distance`, `angular_distance`, `normalize`, `normalize_with_norm` (normalize in place, returns the original norm). Portable fallbacks in `innr::dense` (e.g. `dot_portable`).

**Core (f64)**: `innr::dense_f64::{dot_f64, cosine_f64, norm_f64, l2_distance_f64, l2_distance_squared_f64, l1_distance_f64}` -- SIMD-dispatched (AVX-512/AVX2/NEON) with exact FMA accumulation, for iterative-solver residuals and statistical reductions.

**Backend introspection**: `innr::backend::{dense_backend, slot_backend}` report which kernel family (`Avx512` / `Avx2Fma` / `Neon` / `Portable`) a given length dispatches to on this machine.

**Matryoshka**: `matryoshka_dot`, `matryoshka_cosine` -- similarity on a prefix of the embedding.

**Binary quantization (1-bit)**: `encode_binary` to packed bits, `binary_dot`, `binary_hamming`, `binary_jaccard`. 32x memory reduction over f32.

**Ternary quantization (1.58-bit)**: `ternary::encode_ternary` to {-1, 0, +1}, `ternary_dot`, `ternary_hamming`, `asymmetric_dot` (float query x ternary doc). 16-20x compression.

**Scalar quantization (uint8)**: `scalar::QuantizationParams` (from `fit()`, `fit_quantile()`, or `from_range()`), `quantize_u8`, `asymmetric_dot_u8`. Precomputed query path via `query_context()` + `asymmetric_dot_u8_precomputed`. Batch search via `batch_knn_u8`. 4x compression.

**Fast approximate math**: `fast_cosine_dispatch` (SIMD-dispatched), `fast_cosine` (portable Quake III), `fast_rsqrt`, `fast_rsqrt_precise`.

**Batch operations**: `batch::VerticalBatch` (PDX-style columnar layout) with `batch_dot`, `batch_l2_squared`, `batch_l2_squared_pruning`, `batch_cosine`, `batch_norms`, `batch_knn`, `batch_knn_cosine`, `batch_knn_dot`, `batch_knn_filtered` (predicate pushdown), `batch_knn_reordered` (variance-ordered pruning), `batch_knn_adaptive` (approximate early-exit), `batch_dimension_variance`.

**Sparse vectors**: `sparse_dot`.

**Late interaction**: `maxsim`, `maxsim_cosine` (ColBERT-style), `sparse_maxsim` (sparse late interaction).

**Integer-slot Hamming / MinHash**: `slot_hamming_u16`, `slot_hamming_u32`, and `slot_hamming_u64` (SIMD-dispatched differing-slot count; u16 suits b-bit MinHash at b=16, the u64 path suits probminhash-style 64-bit sketches), `slot_hamming` (generic widths), `minhash_jaccard` (collision-probability estimate), `jaccard_distance` (distance form), `slot_compare_counts` (the `(eq, lt, gt)` per-position triple in `SlotCounts`, for SetSketch / UltraLogLog joint estimators that need the `lt`/`gt` counts, not just `eq`).

**Metric trait**: `distance::Distance` with zero-sized metrics `DistCosine`, `DistDot`, `DistL2`, `DistL1`, `DistHamming`, `DistSlotU32` for parameterizing generic indexes. With the optional `anndists` feature these also implement `anndists::dist::Distance`, so they work directly as `hnsw_rs` distances:

```toml
innr = { version = "0.6", features = ["anndists"] }
```

```rust
use hnsw_rs::prelude::Hnsw;
use innr::distance::DistSlotU32;

// HNSW over u32 MinHash sketches, innr's slot Hamming as the metric.
let index = Hnsw::<u32, DistSlotU32>::new(16, 10_000, 16, 200, DistSlotU32);
```

## SIMD Dispatch

| Architecture | Instructions | Detection |
|--------------|--------------|-----------|
| x86_64 | AVX-512F | Runtime |
| x86_64 | AVX2 + FMA | Runtime |
| aarch64 | NEON | Always |
| Other | Portable | LLVM auto-vec |

Short vectors use portable code (threshold 16 dims for dense f32, 32 for quantized u8, 8 for integer slots). MSRV 1.75 applies to aarch64 and portable targets; x86_64 requires Rust 1.89+ (AVX-512 intrinsic stabilization).

## Performance

![Benchmark throughput](docs/bench_throughput.png)

*Apple Silicon (NEON). Run `cargo bench` to reproduce on your hardware.*

The f64 reductions reach ~8.5 Gelem/s and `slot_hamming_u64` ~5.6 Gelem/s on Apple silicon (NEON), measured cache-resident and single-core; cold-load streaming throughput is lower. AVX-512 paths are executed and differential-tested in CI under Intel SDE.

For maximum performance, build with native CPU features:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Run benchmarks:

```bash
cargo bench
```

## Examples

[**01_basic_ops.rs**](examples/01_basic_ops.rs) -- Core similarity metrics and their mathematical relationships. Proves `L2^2(a,b) = 2(1 - cosine(a,b))` for normalized vectors.

[**batch_demo.rs**](examples/batch_demo.rs) -- PDX-style columnar layout for batch retrieval. Transposes 10K vectors (128d), runs 100 queries, verifies k-NN against brute-force.

[**binary_demo.rs**](examples/binary_demo.rs) -- Binary quantization for first-stage retrieval. 32x memory reduction, measures recall@10 against full-precision search.

More in [`examples/`](examples/):

[**fast_math_demo.rs**](examples/fast_math_demo.rs) -- Newton-Raphson `rsqrt` for 3-10x faster cosine, the speed/accuracy trade-off for first-stage scoring.

[**matryoshka_search.rs**](examples/matryoshka_search.rs) -- MRL progressive search: a coarse pass on a 128d prefix, then a fine pass at full 768d, the cheap-then-precise pattern for large indexes.

[**maxsim_colbert.rs**](examples/maxsim_colbert.rs) -- ColBERT-style MaxSim late interaction, the per-token scoring multi-vector rerankers use.

[**ternary_demo.rs**](examples/ternary_demo.rs) -- 1.58-bit ternary quantization: 16x memory and ~18x speed, with the recall trade-off measured.

## Tests

```bash
cargo test -p innr
```

## License

Dual-licensed under MIT or Apache-2.0.
