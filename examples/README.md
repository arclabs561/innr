# Examples

SIMD-accelerated vector similarity primitives for ML workloads.

## Quick Start

| Example | What It Covers |
|---------|----------------|
| `01_basic_ops` | Dot, cosine, L2 with key identities |
| `fast_math_demo` | Newton-Raphson rsqrt, SIMD vs portable dispatch, search scenario |

```sh
cargo run --example 01_basic_ops --release
cargo run --example fast_math_demo --release
```

## Quantization

| Example | What It Covers | Compression |
|---------|----------------|-------------|
| `binary_demo` | 1-bit quantization: Hamming, dot, Jaccard, recall trade-off | 32x vs f32 |
| `ternary_demo` | 1.58-bit quantization: speed, dedup, ranking accuracy | 16x vs f32 |

```sh
cargo run --example binary_demo --release
cargo run --example ternary_demo --release
```

## Matryoshka (MRL)

| Example | What It Covers |
|---------|----------------|
| `matryoshka_search` | Two-stage retrieval: coarse 128d filter, fine 768d re-rank, recall and timing |

```sh
cargo run --example matryoshka_search --release
```

## Multi-Vector and Batch

| Example | What It Covers | Feature Flag |
|---------|----------------|--------------|
| `maxsim_colbert` | ColBERT-style late interaction scoring, non-commutativity | `maxsim` |
| `batch_demo` | PDX columnar layout, batch kNN, batch dot, timing | (none) |

```sh
cargo run --example maxsim_colbert --release --features maxsim
cargo run --example batch_demo --release
```

## Decision Tree

```
What do you need?

Exact similarity/distance?
  --> Standard ops: dot, cosine, l2_distance (01_basic_ops)

Faster cosine on hot paths (0.2% error OK)?
  --> fast_cosine / fast_cosine_dispatch (fast_math_demo)

32x memory compression, first-stage retrieval?
  --> Binary quantization: encode_binary, binary_hamming (binary_demo)

16x compression with rough ranking?
  --> Ternary quantization: encode_ternary, ternary_dot (ternary_demo)

Prefix-truncatable (MRL) two-stage retrieval?
  --> matryoshka_cosine coarse pass + cosine fine pass (matryoshka_search)

ColBERT / late interaction scoring?
  --> maxsim (maxsim_colbert, requires --features maxsim)

Batch kNN over a corpus?
  --> VerticalBatch + batch_knn (batch_demo)
```

## Key Insight: Distance Is the Bottleneck

In HNSW search: ~20 hops x 32 neighbors = 640 distance calls per query.
A 3x speedup in distance computation = 3x faster search.

| Operation | Standard | Accelerated | Typical Speedup |
|-----------|----------|-------------|-----------------|
| dot | portable loop | AVX2/NEON dispatch | 4-8x |
| cosine | divide by norms | fast_rsqrt (Newton-Raphson) | 2-3x |
| binary hamming | N/A | XOR + popcount | orders of magnitude vs f32 |
