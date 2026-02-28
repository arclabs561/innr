# innr User Guide

SIMD-accelerated vector primitives.

## Quick Start

```rust
use innr::{dot, cosine, l2_distance};

let a = vec![1.0, 2.0, 3.0, 4.0];
let b = vec![4.0, 3.0, 2.0, 1.0];

let similarity = cosine(&a, &b);      // [-1, 1]
let distance = l2_distance(&a, &b);   // [0, ∞)
let product = dot(&a, &b);            // 20.0
```

Run: `cargo run --example simd_benchmark --release`

---

## Contents

1. [Operations](#1-operations)
2. [SIMD](#2-simd)
3. [Ternary Quantization](#3-ternary-quantization)
4. [Common Pitfalls](#4-common-pitfalls)

---

## 1. Operations

### Core functions

| Function | Formula | Output range |
|----------|---------|--------------|
| `dot(a, b)` | Σ aᵢbᵢ | (-∞, ∞) |
| `cosine(a, b)` | dot/(‖a‖‖b‖) | [-1, 1] |
| `l2_distance(a, b)` | √Σ(aᵢ-bᵢ)² | [0, ∞) |

### Batch operations

Process N vectors against one query:

```rust
use innr::batch_dot;

let scores = batch_dot(&query, &documents_flat, dimension);
```

**Why batch?** Amortizes function call overhead; enables better cache usage.

---

## 2. SIMD

### Automatic dispatch

innr detects CPU features at runtime:

```
AVX-512  → 16 floats/instruction  (server CPUs)
AVX2+FMA → 8 floats/instruction   (most x86_64 since 2013)
NEON     → 4 floats/instruction   (all ARM64)
Scalar   → 1 float/instruction    (fallback)
```

### Performance (768-dim vectors)

| Architecture | Throughput vs scalar |
|-------------|---------------------|
| Scalar | 1x |
| NEON | 3-4x |
| AVX2+FMA | 4-6x |
| AVX-512 | 8-12x |

---

## 3. Ternary Quantization

### The idea

Reduce each dimension to {-1, 0, +1}:
- Values above threshold → +1
- Values below -threshold → -1
- Otherwise → 0

This gives **16x memory compression** (2 bits vs 32 bits per dimension).

### Usage

```rust
use innr::ternary::{encode_ternary, ternary_dot};

// Encode (threshold ~0.3 works well for normalized embeddings)
let packed = encode_ternary(&embedding, 0.3);

// Compare (uses popcount - very fast)
let similarity = ternary_dot(&packed_a, &packed_b);
```

### Memory savings

| Dimension | f32 | Ternary | Compression |
|-----------|-----|---------|-------------|
| 384 | 1.5 KB | 96 B | 16x |
| 768 | 3 KB | 192 B | 16x |
| 1536 | 6 KB | 384 B | 16x |

### When to use

- **Memory-bound**: Fit 16x more vectors in RAM
- **First-pass filtering**: Fast coarse ranking, then re-rank with f32
- **Similarity, not exact values**: Ranking preservation > absolute accuracy

### Asymmetric computation

Keep query at full precision, quantize only database:

```rust
use innr::ternary::asymmetric_dot;

let score = asymmetric_dot(&query_f32, &doc_ternary);
```

Better accuracy than symmetric (ternary × ternary).

**Example**: `cargo run --example ternary_demo --release`

---

## 4. Common Pitfalls

### 1. Using SIMD for tiny vectors

**Problem**: SIMD has setup overhead. For dim < 16, scalar may be faster.

**Solution**: Profile. innr automatically falls back when SIMD isn't beneficial.

### 2. Expecting fast_cosine to always be faster

**Problem**: `fast_cosine` uses rsqrt approximation, which helps when sqrt is the bottleneck. On Apple Silicon with fast native sqrt, the benefit is smaller.

**Solution**: Profile your actual workload. For pre-normalized vectors, just use `dot()`.

### 3. Wrong threshold for ternary

**Problem**: Threshold too high → mostly zeros → poor discrimination.
Threshold too low → mostly ±1 → loses information.

**Solution**: For normalized embeddings, threshold ≈ 0.2-0.4 works well. Use `compute_threshold()` for adaptive selection.

### 4. Expecting ternary to preserve exact similarities

**Problem**: Ternary is lossy. Don't expect identical rankings.

**Solution**: Use ternary for first-pass filtering (k=1000), then re-rank with f32 (k=10).

---

## Examples

| Example | Demonstrates |
|---------|-------------|
| `simd_benchmark` | SIMD speedup by dimension |
| `fast_math_demo` | rsqrt approximation accuracy |
| `ternary_demo` | Compression, speed, quality tradeoffs |
