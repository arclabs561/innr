# Contributing

Use Rust 1.94 or newer. From the repository root:

```sh
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo test --no-default-features
```

Keep pull requests scoped to one concern.
