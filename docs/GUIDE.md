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

Run: `cargo run --example 01_basic_ops --release`

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
use innr::batch::{batch_dot, VerticalBatch};

let documents = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
let query = [0.8, 0.6];
let batch = VerticalBatch::from_rows(&documents);
let scores = batch_dot(&query, &batch);
assert_eq!(scores, vec![0.8, 0.6]);
```

**Why batch?** Amortizes function call overhead; enables better cache usage.

---

## 2. SIMD

### Automatic dispatch

innr detects CPU features at runtime:

```text
AVX-512  → 16 floats/instruction  (server CPUs)
AVX2+FMA → 8 floats/instruction   (most x86_64 since 2013)
NEON     → 4 floats/instruction   (all ARM64)
Scalar   → 1 float/instruction    (fallback)
```

### Performance

Throughput depends on dimension, cache residency, and the selected kernel.
Run `cargo bench` on the target machine before making a deployment choice.

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

let a = [0.8, -0.5, 0.1];
let b = [0.7, 0.4, 0.2];
let packed_a = encode_ternary(&a, 0.3);
let packed_b = encode_ternary(&b, 0.3);
let similarity = ternary_dot(&packed_a, &packed_b);
assert_eq!(similarity, 0);
```

### Memory savings

| Dimension | f32 | Ternary | Compression |
|-----------|-----|---------|-------------|
| 384 | 1.5 KiB | 96 B | 16x |
| 768 | 3 KiB | 192 B | 16x |
| 1536 | 6 KiB | 384 B | 16x |

### When to use

- **Memory-bound**: Fit 16x more vectors in RAM
- **First-pass filtering**: Fast coarse ranking, then re-rank with f32
- **Similarity, not exact values**: Ranking preservation > absolute accuracy

### Asymmetric computation

Keep query at full precision, quantize only database:

```rust
use innr::ternary::{asymmetric_dot, encode_ternary};

let query = [0.8, -0.5, 0.1];
let document = encode_ternary(&[0.7, 0.4, 0.2], 0.3);
let score = asymmetric_dot(&query, &document);
assert!((score - 0.3).abs() < 1e-6);
```

This preserves the query values. Measure retrieval quality on your data to
choose between symmetric and asymmetric scoring.

**Example**: `cargo run --example ternary_demo --release`

---

## 4. Common Pitfalls

### 1. Using SIMD for tiny vectors

**Problem**: SIMD has setup overhead. For dim < 16, scalar may be faster.

**Solution**: Profile. Dense f32 operations use portable code below 16 dimensions;
other kernels have their own dispatch thresholds.

### 2. Wrong threshold for ternary

**Problem**: Threshold too high → mostly zeros → poor discrimination.
Threshold too low → mostly ±1 → loses information.

**Solution**: Choose a threshold using representative vectors and measure
retrieval quality. A fixed threshold can erase most coordinates in a normalized
embedding, especially at higher dimensions.

### 3. Expecting ternary to preserve exact similarities

**Problem**: Ternary is lossy. Don't expect identical rankings.

**Solution**: Use ternary for first-pass filtering (k=1000), then re-rank with f32 (k=10).

---

## Examples

| Example | Demonstrates |
|---------|-------------|
| `01_basic_ops` | Core similarity operations |
| `batch_demo` | Batch scan and k-NN helpers |
| `ternary_demo` | Compression, speed, quality tradeoffs |
