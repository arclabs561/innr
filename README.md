# innr

[![crates.io](https://img.shields.io/crates/v/innr.svg)](https://crates.io/crates/innr)
[![Documentation](https://docs.rs/innr/badge.svg)](https://docs.rs/innr)
[![CI](https://github.com/arclabs561/innr/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/innr/actions/workflows/ci.yml)

SIMD-accelerated vector similarity primitives.

## Quickstart

```toml
[dependencies]
innr = "0.7"
```

The default `simd` feature dispatches at runtime to AVX-512, AVX2+FMA, NEON
or WASM SIMD128 through [`fearless_simd`](https://crates.io/crates/fearless_simd),
and falls back to portable kernels on other CPUs. Without it, innr has no
dependencies and uses portable kernels written for the auto-vectorizer:

```toml
innr = { version = "0.7", default-features = false }
```

The portable build is the better choice on CPUs with neither AVX2 nor NEON,
where the `simd` build only adds a dispatch check (see
[Performance](#performance) for six measured CPUs).
innr forbids unsafe code in its own source and uses only `fearless_simd`'s
safe API; `fearless_simd` confines its unsafe code to its own audited
target-feature wrappers.

```rust
use innr::{dot, cosine, norm};

let a = [1.0_f32, 0.0, 0.0];
let b = [0.707, 0.707, 0.0];

let d = dot(&a, &b);
let c = cosine(&a, &b);
let n = norm(&a);
assert!((d - 0.707).abs() < 1e-6);
assert!((c - 0.707107).abs() < 1e-6);
assert_eq!(n, 1.0);
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
assert_eq!(result.indices, vec![2, 0]);
assert!((result.scores[0] - 0.98).abs() < 1e-6);
```

## API

The [API reference](https://docs.rs/innr) covers dense and sparse similarity,
quantization, batch search, and late interaction. SIMD and portable paths may
differ in floating-point rounding.

**Metric trait**: `distance::Distance` with zero-sized metrics `DistCosine`, `DistDot`, `DistL2`, `DistL1`, `DistHamming`, `DistSlotU32` for parameterizing generic indexes. With the optional `anndists` feature these also implement `anndists::dist::Distance`, so they work directly as `hnsw_rs` distances:

```toml
innr = { version = "0.7", features = ["anndists"] }
```

```rust
use hnsw_rs::prelude::Hnsw;
use innr::distance::DistSlotU32;

// HNSW over u32 MinHash sketches, innr's slot Hamming as the metric.
let index = Hnsw::<u32, DistSlotU32>::new(16, 10_000, 16, 200, DistSlotU32);
```

**Packed corpus**: `corpus::Corpus` packs `f32` rows once, each row aligned,
and scores a query against every row in one SIMD dispatch, with top-k kept
during the scan. For brute-force scans at 128 dimensions this measured
1.2-2.4x faster than calling a pair kernel per row (more rows, larger gain).
Cosine scans keep each row's norm, so they cost one dot product per row:
1.2-1.9x faster than per-row `cosine` on an i7-8559U.

```rust
use innr::corpus::Corpus;
use innr::metric::Cosine;

let rows = [[1.0_f32, 0.0], [0.0, 1.0], [0.7, 0.7]];
let corpus = Corpus::from_rows(rows.iter().map(|r| r.as_slice()), 2).unwrap();
let best = corpus.top_k::<Cosine>(&[0.1, 1.0], 1);
assert_eq!(best[0].index, 1);
```

`Corpus<u8>` holds byte-packed binary codes and ranks `Hamming` exactly.
On x86_64 it beat per-code `hamming_distance` calls by 1.3-1.5x from 96-byte
codes up; below 32 bytes keep the per-code calls.

```rust
use innr::corpus::Corpus;
use innr::metric::Hamming;

let codes = [[0b1111_0000u8, 0], [0, 0], [0xFF, 0xFF]];
let corpus = Corpus::from_rows(codes.iter().map(|c| c.as_slice()), 2).unwrap();
let nearest = corpus.top_k::<Hamming>(&[0, 0], 2);
assert_eq!((nearest[0].index, nearest[0].score), (1, 0));
assert_eq!((nearest[1].index, nearest[1].score), (0, 4));
```

## SIMD Dispatch

With the `simd` feature:

| Architecture | Level | Detection |
|--------------|-------|-----------|
| x86 / x86_64 | AVX-512 (Ice Lake-class) | Runtime |
| x86 / x86_64 | AVX2 + FMA (x86-64-v3) | Runtime |
| aarch64 | NEON | Always |
| wasm32 | SIMD128 (with `+simd128`) | Compile time |
| Other | Portable | LLVM auto-vec |

`innr::capabilities()` reports the level in use and the per-kernel length
thresholds; shorter inputs use portable code (16 dims for dense f32, 8 for
dense f64, 32 for the u8 dot, 128 bytes for bit Hamming, 8 for integer
slots). Requires Rust 1.89.

## Performance

Criterion medians, single core, inputs in cache, from
[`benches/results`](benches/results); the last column is the fastest
fallback that a consumer of innr used before adopting it, on the same
machine:

| Operation | M3 Max `simd` | M3 Max portable | Ryzen 5600X `simd` | Ryzen 5600X portable | Consumer fallback (M3 / Ryzen) |
|---|---:|---:|---:|---:|---|
| dot, d=16 | 1.9 ns | 2.0 ns | 5.0 ns | 4.9 ns | rankops 1.9 ns / rankops 4.6 ns |
| dot, d=768 | 39.5 ns | 41.2 ns | 35.0 ns | 48.6 ns | rankops 125.6 ns / rankops 73.5 ns |
| cosine, d=768 | 83.6 ns | 109.3 ns | 51.5 ns | 128.5 ns | rankops 370.9 ns / rankops 185.3 ns |
| squared L2, d=768 | 39.5 ns | 42.8 ns | 34.0 ns | 59.6 ns | vicinity 488.1 ns / clump 487.6 ns |
| bit Hamming, 96 B | 2.8 ns | 3.0 ns | 7.0 ns | 7.3 ns | sketchir 3.3 ns / sketchir 36.7 ns |
| matrix-vector, 768x768 | 31.6 µs | 32.2 µs | 25.6 µs | 38.2 µs | qntz 367.3 µs / qntz 366.6 µs |
| scan, 100k x 128 | 1.82 ms | 2.03 ms | 3.61 ms | 3.67 ms | vicinity 5.56 ms / vicinity 8.52 ms |
| MaxSim, 32x128 tokens, d=128 | 25.4 µs | 41.0 µs | 24.4 µs | 48.6 µs | rankops_flat 45.3 µs / rankops_flat 34.0 µs |

Machines: Apple M3 Max (macOS, rustc 1.99) and AMD Ryzen 5 5600X (Linux,
rustc 1.91), innr 0.7.0 kernels, 2026-11-02. The M3 run's A/A noise floor
(one kernel measured three times) was 5%; the Ryzen's was up to 32% at
small sizes, so treat differences under about a third there as ties.

What the `simd` feature buys depends on the CPU. Speedup of the `simd` build
over the portable build (`default-features = false`), each pair timed in the
same run with `benches/probe`:

| CPU | dot d=16 | dot d=768 | cosine d=16 | cosine d=768 |
|---|---:|---:|---:|---:|
| Apple M3 Max (NEON) | 1.04x | 1.01x | 1.72x | 1.26x |
| Raspberry Pi 3B+, Cortex-A53 (NEON) | 1.03x | 1.00x | 1.46x | 1.53x |
| AMD Ryzen 5 5600X (AVX2) | 0.97x | 1.15x | 2.33x | 2.52x |
| Intel i7-8559U (AVX2) | 0.90x | 1.35x | 2.11x | 2.99x |
| Intel i3-8109U (AVX2) | 1.21x | 1.36x | 2.07x | 3.00x |
| Intel Celeron J3455 (no AVX) | 0.76x | 0.98x | 0.95x | 0.99x |

On ARM the portable build already compiles to NEON, so `simd` mostly helps
cosine, MaxSim and the quantized dot. On x86 the portable build targets
SSE2 and `simd` adds AVX2. On a CPU without AVX2 or NEON the `simd` build
runs portable kernels behind a dispatch check, which cost up to 24% on the
Celeron; use `default-features = false` there. The x86 d=768 rows predate
an alignment change for long inputs in 0.7.0.

Not claimed: AVX-512 speed (the kernels run and are tested in CI under
Intel SDE, which does not measure speed), multi-threaded throughput, and
data that does not fit in cache. Reproduce with
`benches/report.sh OUT.md` on your machine.

For maximum performance, build with native CPU features:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Run benchmarks:

```bash
cargo bench
```

## Examples

[**01_basic_ops.rs**](examples/01_basic_ops.rs): Core similarity metrics and their mathematical relationships. Proves `L2^2(a,b) = 2(1 - cosine(a,b))` for normalized vectors.

[**batch_demo.rs**](examples/batch_demo.rs): PDX-style columnar layout for batch retrieval. Transposes 10K vectors (128d), runs 100 queries, verifies k-NN against brute-force.

[**binary_demo.rs**](examples/binary_demo.rs): Binary quantization for first-stage retrieval. 32x memory reduction, measures recall@10 against full-precision search.

More in [`examples/`](examples/):

[**matryoshka_search.rs**](examples/matryoshka_search.rs): MRL progressive search: a coarse pass on a 128d prefix, then a fine pass at full 768d, the cheap-then-precise pattern for large indexes.

[**maxsim_colbert.rs**](examples/maxsim_colbert.rs): ColBERT-style MaxSim late interaction, the per-token scoring multi-vector rerankers use.

[**ternary_demo.rs**](examples/ternary_demo.rs): ternary quantization in a packed 2-bit representation, with local timing and ranking comparisons.

## Tests

```bash
cargo test -p innr
```

## License

Dual-licensed under MIT or Apache-2.0.
