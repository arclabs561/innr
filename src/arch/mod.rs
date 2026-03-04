//! Architecture-specific SIMD implementations.
//!
//! This module contains unsafe SIMD code for different CPU architectures.
//! The safe public API in parent modules handles dispatch and fallback.

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64;

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;
