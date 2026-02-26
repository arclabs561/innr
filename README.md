# innr

[![crates.io](https://img.shields.io/crates/v/innr.svg)](https://crates.io/crates/innr)
[![Documentation](https://docs.rs/innr/badge.svg)](https://docs.rs/innr)
[![CI](https://github.com/arclabs561/innr/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/innr/actions/workflows/ci.yml)

SIMD-accelerated vector similarity primitives. Pure Rust, zero runtime dependencies.

Unlike simsimd (C bindings) or ndarray (full linear algebra), innr is pure Rust, zero-dep, and focused on the similarity primitives that retrieval and embedding systems need.

Dual-licensed under MIT or Apache-2.0.

## Quickstart

```toml
[dependencies]
innr = "0.1.3"
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

### Core (always available)

| Function | Description |
|----------|-------------|
| `dot`, `dot_portable` | Inner product (SIMD / portable) |
| `cosine` | Cosine similarity |
| `norm` | L2 norm |
| `l2_distance` | Euclidean distance |
| `l2_distance_squared` | Squared Euclidean distance (avoids sqrt) |
| `l1_distance` | Manhattan distance |
| `angular_distance` | Angular distance (arccos-based) |
| `pool_mean` | Mean pooling over a set of vectors |

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

### Batch operations (PDX-style columnar layout)

| Type / Function | Description |
|-----------------|-------------|
| `batch::VerticalBatch` | Columnar (SoA) vector store |
| `batch::batch_dot` | Batch dot products against a query |
| `batch::batch_l2_squared` | Batch squared L2 distances |
| `batch::batch_cosine` | Batch cosine similarities |
| `batch::batch_norms` | Norms for all vectors in the batch |
| `batch::batch_knn` | Exact k-NN over a batch |
| `batch::batch_knn_adaptive` | Adaptive early-exit k-NN |

### Metric traits

| Trait | Description |
|-------|-------------|
| `SymmetricMetric` | Symmetric distance interface (`d(a,b) = d(b,a)`) |
| `Quasimetric` | Directed distance interface (`d(a,b) != d(b,a)`) |

### Clifford algebra

| Type / Function | Description |
|-----------------|-------------|
| `clifford::Rotor2D` | 2D rotor (even subalgebra of Cl(2)) |
| `clifford::wedge_2d` | 2D wedge (outer) product |
| `clifford::geometric_product_2d` | 2D geometric product (scalar + bivector) |

### Feature-gated

| Function | Feature | Description |
|----------|---------|-------------|
| `sparse_dot`, `sparse_dot_portable` | `sparse` | Sparse vector dot (sorted-index merge) |
| `sparse_maxsim` | `sparse` | Sparse MaxSim scoring |
| `maxsim`, `maxsim_cosine` | `maxsim` | ColBERT late interaction scoring |

## SIMD Dispatch

| Architecture | Instructions | Detection |
|--------------|--------------|-----------|
| x86_64 | AVX2 + FMA | Runtime |
| aarch64 | NEON | Always |
| Other | Portable | LLVM auto-vec |

Vectors < 16 dimensions use portable code.

## Features

- `sparse` -- sparse vector operations
- `maxsim` -- ColBERT late interaction scoring
- `full` -- all features

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

## Tests

```bash
cargo test -p innr
```

## License

Dual-licensed under MIT or Apache-2.0.
