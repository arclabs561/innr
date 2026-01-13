# innr

SIMD-accelerated vector similarity primitives.

## What

Core operations for embedding similarity with automatic SIMD dispatch:
- `dot` — inner product
- `norm` — L2 norm
- `cosine` — cosine similarity
- `l2_distance` — Euclidean distance
- `sparse_dot` — sparse vector dot product (feature `sparse`)
- `maxsim` — ColBERT late interaction (feature `maxsim`)

## SIMD Dispatch

| Architecture | Instructions | Detection |
|--------------|--------------|-----------|
| x86_64 | AVX2 + FMA | Runtime |
| aarch64 | NEON | Always |
| Other | Portable | LLVM auto-vectorizes |

Vectors shorter than 16 dimensions use portable code.

## Usage

```rust
use innr::{dot, cosine, norm};

let a = [1.0_f32, 0.0, 0.0];
let b = [0.707, 0.707, 0.0];

let d = dot(&a, &b);      // 0.707
let c = cosine(&a, &b);   // 0.707
let n = norm(&a);         // 1.0
```

## Features

- `sparse` — sparse vector operations
- `maxsim` — ColBERT late interaction scoring
- `full` — all features

## Why "innr"

Inner product. The fundamental operation.
