//! Architecture-specific SIMD kernels, written against pulp tokens.
//!
//! Each kernel takes a pulp token that proves its CPU features at runtime, so
//! the modules here need no `unsafe`. The public API in parent modules handles
//! dispatch and the portable fallback.

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64;

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;
