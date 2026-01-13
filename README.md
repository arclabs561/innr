# innr

SIMD-accelerated vector similarity primitives.

## Usage

```rust
use innr::{dot, cosine, norm};

let a = [1.0_f32, 0.0, 0.0];
let b = [0.707, 0.707, 0.0];

let d = dot(&a, &b);
let c = cosine(&a, &b);
let n = norm(&a);

assert!((d - 0.707).abs() < 1e-3);
assert!((c - 0.707).abs() < 1e-3);
assert!((n - 1.0).abs() < 1e-6);
```

## Features

- `sparse` — sparse vector operations
- `maxsim` — ColBERT late interaction scoring
- `full` — all features

## License

MIT OR Apache-2.0
