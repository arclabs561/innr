# Examples

SIMD-accelerated vector operations for ML workloads.

## Quick Start

| Example | What It Teaches |
|---------|-----------------|
| `01_basic_ops` | Dot, cosine, L2 with key identities |
| `simd_benchmark` | Compare SIMD vs portable performance |

```sh
cargo run --example 01_basic_ops --release
cargo run --example simd_benchmark --release
```

## Advanced

| Example | What It Teaches | When to Use |
|---------|-----------------|-------------|
| `fast_math_demo` | Newton-Raphson rsqrt for 3x faster cosine | Hot paths in ANN search |
| `ternary_demo` | 1.58-bit quantization: 16x memory, 18x speed | Deduplication, coarse filtering |

```sh
cargo run --example fast_math_demo --release
cargo run --example ternary_demo --release
```

## Key Insight: Distance is the Bottleneck

In HNSW search: ~20 hops * 32 neighbors = 640 distance calls per query.

A 3x speedup in distance computation = 3x faster search.

| Operation | Standard | Fast | Speedup |
|-----------|----------|------|---------|
| rsqrt | `1.0/x.sqrt()` | Newton-Raphson | ~3x |
| cosine | divide by norms | fast_rsqrt | ~3x |
| dot | portable | AVX2/NEON | ~4-8x |

## When to Use Which

```
Need exact results?
  └─> Standard ops (cosine, dot, l2_distance)

Hot path, can tolerate 0.2% error?
  └─> fast_cosine

Memory-constrained, coarse filtering?
  └─> Ternary quantization

Deduplication (need rough similarity)?
  └─> Ternary + threshold
```
