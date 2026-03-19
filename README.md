# innr

[![crates.io](https://img.shields.io/crates/v/innr.svg)](https://crates.io/crates/innr)
[![Documentation](https://docs.rs/innr/badge.svg)](https://docs.rs/innr)
[![CI](https://github.com/arclabs561/innr/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/innr/actions/workflows/ci.yml)

Vector similarity primitives with SIMD dispatch (AVX-512, AVX2+FMA, NEON). Pure Rust, zero dependencies, MSRV 1.75.

Computes dot product, cosine similarity, L2/L1 distance, binary/ternary quantized distances, ColBERT MaxSim, Matryoshka prefix similarity, and batch k-NN (L2, cosine, dot, filtered) over columnar layouts. Runtime CPU detection picks the widest available ISA -- no build-time flags required.

## Quickstart

```toml
[dependencies]
innr = "0.1.9"
```

```rust
use innr::{dot, cosine, norm};

let a = [1.0_f32, 0.0, 0.0];
let b = [0.707, 0.707, 0.0];

let d = dot(&a, &b);      // 0.707
let c = cosine(&a, &b);   // 0.707
let n = norm(&a);         // 1.0
```

## Operations

### Core

| Function | Description |
|----------|-------------|
| `dot`, `dot_portable` | Inner product (SIMD / portable) |
| `cosine`, `cosine_portable` | Cosine similarity (single-pass fused SIMD) |
| `norm` | L2 norm |
| `l2_distance` | Euclidean distance |
| `l2_distance_squared` | Squared Euclidean distance (avoids sqrt) |
| `l1_distance` | Manhattan distance (SIMD-accelerated) |
| `angular_distance` | Angular distance (arccos-based) |

### Matryoshka embeddings

| Function | Description |
|----------|-------------|
| `matryoshka_dot` | Dot product on a prefix of the embedding |
| `matryoshka_cosine` | Cosine similarity on a prefix of the embedding |

### Binary quantization (1-bit)

| Type / Function | Description |
|-----------------|-------------|
| `encode_binary` | Quantize `f32` vector to packed bits |
| `PackedBinary` | Packed bit-vector type |
| `binary_dot` | Dot product on packed binary vectors |
| `binary_hamming` | Hamming distance |
| `binary_jaccard` | Jaccard similarity |

### Ternary quantization (1.58-bit)

| Type / Function | Description |
|-----------------|-------------|
| `ternary::encode_ternary` | Quantize `f32` to {-1, 0, +1} |
| `ternary::PackedTernary` | Packed ternary vector type |
| `ternary::ternary_dot` | Inner product on packed ternary vectors |
| `ternary::ternary_hamming` | Hamming distance on ternary vectors |
| `ternary::asymmetric_dot` | Float query x ternary doc product |
| `ternary::sparsity` | Fraction of zero entries |

### Fast approximate math

| Function | Description |
|----------|-------------|
| `fast_cosine` | Approximate cosine via `fast_rsqrt` |
| `fast_rsqrt` | Fast inverse square root (hardware rsqrt + Newton-Raphson) |
| `fast_rsqrt_precise` | Two-iteration Newton-Raphson variant |
| `fast_cosine_distance` | `1 - fast_cosine` |

### Batch operations (PDX-style columnar layout)

| Type / Function | Description |
|-----------------|-------------|
| `batch::VerticalBatch` | Columnar (SoA) vector store |
| `batch::batch_dot` | Batch dot products against a query |
| `batch::batch_l2_squared` | Batch squared L2 distances |
| `batch::batch_cosine` | Batch cosine similarities |
| `batch::batch_norms` | Norms for all vectors in the batch |
| `batch::batch_knn` | Exact k-NN (L2) over a batch |
| `batch::batch_knn_cosine` | Top-k by cosine similarity |
| `batch::batch_knn_dot` | Top-k by dot product (MIPS) |
| `batch::batch_knn_filtered` | k-NN with predicate pushdown |
| `batch::batch_knn_reordered` | Exact k-NN with variance-ordered pruning |
| `batch::batch_knn_adaptive` | Approximate early-exit k-NN |
| `batch::batch_l2_squared_pruning` | Batch L2 with threshold pruning |
| `batch::batch_dimension_variance` | Per-dimension variance (for reordering) |

### Sparse vectors

| Function | Description |
|----------|-------------|
| `sparse_dot`, `sparse_dot_portable` | Sparse vector dot (sorted-index merge) |
| `sparse_maxsim` | Sparse MaxSim scoring |

### ColBERT late interaction

| Function | Description |
|----------|-------------|
| `maxsim` | MaxSim dot-product scoring |
| `maxsim_cosine` | MaxSim cosine scoring |

## SIMD Dispatch

| Architecture | Instructions | Detection |
|--------------|--------------|-----------|
| x86_64 | AVX-512F | Runtime |
| x86_64 | AVX2 + FMA | Runtime |
| aarch64 | NEON | Always |
| Other | Portable | LLVM auto-vec |

Vectors < 16 dimensions use portable code.

## Performance

![Benchmark throughput](docs/bench_throughput.png)

*Apple Silicon (NEON). Run `cargo bench` to reproduce on your hardware.*

For maximum performance, build with native CPU features:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Or specify a portable baseline with SIMD:

```bash
# AVX2 (89% of x86_64 CPUs)
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release

# SSE2 only (100% compatible)
RUSTFLAGS="-C target-cpu=x86-64" cargo build --release
```

Run benchmarks:

```bash
cargo bench
```

Generate flamegraphs (requires `cargo-flamegraph`):

```bash
./scripts/profile.sh dense
```

## Examples

[**01_basic_ops.rs**](examples/01_basic_ops.rs) -- The three core similarity metrics (dot product, cosine, L2 distance) and their mathematical relationships. Proves the identity `L2^2(a,b) = 2(1 - cosine(a,b))` for normalized vectors, showing that cosine and L2 are interchangeable for ranking.

[**batch_demo.rs**](examples/batch_demo.rs) -- PDX-style columnar layout for batch retrieval. Transposes 10K vectors (128d) into column-major order, runs 100 queries, and verifies k-NN results against brute-force. Demonstrates the cache-friendly memory access pattern that enables auto-vectorization.

[**binary_demo.rs**](examples/binary_demo.rs) -- Binary (1-bit) quantization for first-stage retrieval. Quantizes 384d vectors to packed bits (32x memory reduction: 150 MB vs 4.6 GB for 1M documents), computes Hamming distance and binary dot product, and measures recall@10 against full-precision search.

[**fast_math_demo.rs**](examples/fast_math_demo.rs) -- Newton-Raphson rsqrt approximation for fast cosine similarity. Benchmarks the hot path in ANN search (640 distance calls per query in HNSW at 1M scale), measures 3-10x speedup over standard cosine at `<1e-4` error, and shows architecture-specific SIMD dispatch.

[**matryoshka_search.rs**](examples/matryoshka_search.rs) -- Two-stage retrieval using Matryoshka embeddings. Uses a 128d prefix for coarse filtering (100 candidates from 10K corpus), then rescores with full 768d vectors to produce the final top-10. Measures recall and speedup vs single-stage search.

[**maxsim_colbert.rs**](examples/maxsim_colbert.rs) -- ColBERT-style late interaction scoring. Computes MaxSim (sum of per-query-token maximum similarities across document tokens) for 32 query tokens x 128 doc tokens at 128d. Demonstrates non-commutativity and batch scoring of 1000 documents.

[**ternary_demo.rs**](examples/ternary_demo.rs) -- Ternary (1.58-bit) quantization for extreme compression. Quantizes 768d vectors to `{-1, 0, +1}` (16-20x memory reduction: 90 MB vs 3.1 GB for 1M documents), measures recall trade-offs, and analyzes sparsity patterns.

```bash
cargo run --example 01_basic_ops
cargo run --example batch_demo
cargo run --example binary_demo
cargo run --example fast_math_demo
cargo run --example matryoshka_search
cargo run --example maxsim_colbert
cargo run --example ternary_demo
```

## Tests

```bash
cargo test -p innr
```

## License

Dual-licensed under MIT or Apache-2.0.
