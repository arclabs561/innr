# innr

[![crates.io](https://img.shields.io/crates/v/innr.svg)](https://crates.io/crates/innr)
[![Documentation](https://docs.rs/innr/badge.svg)](https://docs.rs/innr)
[![CI](https://github.com/arclabs561/innr/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/innr/actions/workflows/ci.yml)

SIMD-accelerated vector similarity primitives.

Dual-licensed under MIT or Apache-2.0.

## Why this exists

`innr` is the dependency you reach for when you need **fast, well-tested vector math** without
pulling in a full ANN index or ML framework. It is designed to sit under crates like `jin`,
retrieval pipelines, and evaluation tooling.

## Quickstart

```toml
[dependencies]
innr = "0.1.1"
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

| Function | Description |
|----------|-------------|
| `dot` | Inner product |
| `norm` | L2 norm |
| `cosine` | Cosine similarity |
| `l2_distance` | Euclidean distance |
| `sparse_dot` | Sparse vector dot (`sparse` feature) |
| `maxsim` | ColBERT late interaction (`maxsim` feature) |

## SIMD Dispatch

| Architecture | Instructions | Detection |
|--------------|--------------|-----------|
| x86_64 | AVX2 + FMA | Runtime |
| aarch64 | NEON | Always |
| Other | Portable | LLVM auto-vec |

Vectors < 16 dimensions use portable code.

## Features

- `sparse` — sparse vector operations
- `maxsim` — ColBERT late interaction scoring
- `full` — all features

## Best starting points

- **Cosine / dot / norm**: `cosine`, `dot`, `norm`
- **Distances**: `l2_distance`
- **When using cosine**: normalize once (or use an index that expects normalized vectors)

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
