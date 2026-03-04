# Audit Report -- innr
Date: 2026-03-04 | Scope: arch, qa, release

## Summary
14 findings: 2 structural, 4 coherence, 5 surface, 3 hygiene

## Architecture

### STRUCTURAL

1. **CI breaks with its own lint config** [`Cargo.toml:73`, `.github/workflows/ci.yml:12`]
   `[lints.rust] unsafe_code = "warn"` combined with CI's `RUSTFLAGS: -Dwarnings` promotes the 11 expected unsafe warnings to errors. Verified: `RUSTFLAGS="-Dwarnings" cargo check --all-features` fails. Either remove `unsafe_code = "warn"` from `[lints.rust]` (unsafe is intentional in a SIMD crate) or add targeted `#[allow(unsafe_code)]` on the arch modules and fast_math SIMD functions.

2. **`fast_cosine` uses `get_unchecked` without bounds justification** [`src/fast_math.rs:104-105`]
   The portable `fast_cosine` loop indexes with `get_unchecked(i)` where `i` ranges `0..n` and `n = a.len().min(b.len())`. This is technically safe since `i < n <= a.len()` and `i < n <= b.len()`, but the bounds invariant is implicit. The equivalent `dot_portable` achieves the same via iterator zip without unsafe. The `get_unchecked` here saves negligible overhead since the loop body is scalar and LLVM will eliminate the bounds check anyway. Consider using the iterator pattern for consistency with the rest of the codebase.

### COHERENCE

3. **Git tag lags published version** [repo root]
   `Cargo.toml` version: `0.1.6`. Latest git tag: `v0.1.4`. Missing tags for `v0.1.5` and `v0.1.6`. Tag-based tooling (GitHub releases, `cargo install` from git) will see stale versions.

4. **README documents subset of public API** [`README.md`]
   Three public functions in `dense` are not mentioned in the README: `bilinear`, `geometric_outer_product`, `metric_residual`. Also omitted: `fast_cosine_distance`, `batch_l2_squared_pruning`, `batch_knn_adaptive`, `BatchKnnResult`, `DEFAULT_BLOCK_SIZE`. The README tables should either list them or the functions should be `pub(crate)` if not intended for external use.

5. **`L1_ALIGNMENT_EPSILON` is public but unused** [`src/lib.rs:142`]
   Exported constant `L1_ALIGNMENT_EPSILON` is not referenced anywhere in the crate. Its doc comment mentions "cross-lingual alignment" which is domain-specific to a downstream consumer. Either use it internally or remove it from the public API (downstream can define its own constant).

6. **`DEFAULT_BLOCK_SIZE` is public but unused** [`src/batch.rs:63`]
   Exported constant `DEFAULT_BLOCK_SIZE = 8` is not referenced in any batch function. The batch functions don't parameterize block size. Either wire it into the batch implementation or make it `pub(crate)` / remove.

## Quality

### SURFACE

7. **Formatting drift** [multiple files]
   `cargo fmt --check` reports diffs in 6 examples and 4 source files. All are whitespace/line-width issues. Fix: `cargo fmt`.

8. **Two clippy suggestions** [`src/binary.rs:508`, `src/ternary.rs:818`]
   `manual_range_contains`: `j >= 0.0 && j <= 1.0` should be `(0.0..=1.0).contains(&j)` in proptest assertions. Fix: `cargo clippy --fix --lib -p innr --tests`.

9. **`geometric_outer_product` doc mentions "Foundation for 2026 Rotors"** [`src/dense.rs:403`]
   Forward-looking doc comment references a year ("2026") and an unrealized feature. This is confusing for users reading the published docs. Either remove the parenthetical or replace with a concrete statement about what the function does today.

10. **`VerticalBatch::data` field visibility** [`src/batch.rs:94`]
    The `data` field is private (good), but `VerticalBatch` does not derive `PartialEq`. Two identical batches can't be compared in tests. Not blocking, but would improve test ergonomics.

11. **No `#[must_use]` on several pure functions** [various]
    `pool_mean`, `encode_binary`, `encode_ternary`, `batch_asymmetric_dot`, `sparsity`, `binary_jaccard`, `geometric_outer_product`, `batch_l2_squared`, `batch_dot`, `batch_norms`, `batch_cosine` are pure functions without `#[must_use]`. The core dense functions (`dot`, `cosine`, `norm`, etc.) already have `#[must_use]`. Apply consistently.

## Release Readiness

### SURFACE

12. **No CHANGELOG** [repo root]
    Published crate (v0.1.6 on crates.io) has no CHANGELOG. Users must read git history to understand what changed between versions. Low priority for a 0.x crate but good practice before 1.0.

### HYGIENE

13. **No `deny.toml` config** [repo root]
    CI runs `cargo-deny` via `EmbarkStudios/cargo-deny-action@v2` but there's no `deny.toml` in the repo root, so it uses defaults. Consider adding explicit config to document license policy and banned crates.

14. **`arch` modules are `pub`** [`src/arch/mod.rs:7-10`]
    `arch::x86_64` and `arch::aarch64` are `pub mod` but they're implementation details -- all their functions are `unsafe` and the safe dispatch layer (`dot`, `maxsim`, `fast_cosine_dispatch`) is the intended API. Making them `pub(crate)` would prevent users from accidentally depending on the SIMD internals.
