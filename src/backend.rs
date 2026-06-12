//! SIMD backend introspection: which kernel family will actually run.
//!
//! The dispatchers in this crate select a kernel per call from runtime CPU
//! feature detection and the input length. That selection is invisible to
//! callers, which makes two situations needlessly hard: performance work
//! ("am I on the AVX-512 path or did my vectors fall under the dispatch
//! threshold?") and bug reports ("which kernel computed this?"). The
//! functions here answer those questions without the caller re-deriving
//! the dispatch rules.
//!
//! ```
//! use innr::backend::{dense_backend, Backend};
//!
//! // A 768-dim embedding: SIMD on any supported machine.
//! let b = dense_backend(768);
//! assert_ne!(b, Backend::Portable, "768 dims should dispatch to SIMD \
//!     on x86_64 and aarch64; got {b}");
//!
//! // 8 dims is below every dense threshold: always portable.
//! assert_eq!(dense_backend(8), Backend::Portable);
//! ```

use core::fmt;

/// A kernel family the dispatchers can select.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// AVX-512F (and, for integer kernels, AVX-512BW / AVX-512VPOPCNTDQ).
    Avx512,
    /// AVX2 + FMA.
    Avx2Fma,
    /// NEON (always available on aarch64).
    Neon,
    /// Scalar code, auto-vectorized by LLVM where possible.
    Portable,
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Backend::Avx512 => "avx512",
            Backend::Avx2Fma => "avx2+fma",
            Backend::Neon => "neon",
            Backend::Portable => "portable",
        };
        f.write_str(s)
    }
}

/// Backend the dense f32 kernels (`dot`, `cosine`, `l1_distance`,
/// `l2_distance`, ...) select for `len`-dimensional vectors on this machine.
#[must_use]
pub fn dense_backend(len: usize) -> Backend {
    #[cfg(target_arch = "x86_64")]
    {
        if len >= crate::dense::MIN_DIM_AVX512 && std::arch::is_x86_feature_detected!("avx512f") {
            return Backend::Avx512;
        }
        if len >= crate::MIN_DIM_SIMD
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            return Backend::Avx2Fma;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if len >= crate::MIN_DIM_SIMD {
            return Backend::Neon;
        }
    }
    let _ = len;
    Backend::Portable
}

/// Backend `slot_hamming_u32` selects for `len`-slot sketches on this
/// machine.
#[must_use]
pub fn slot_backend(len: usize) -> Backend {
    #[cfg(target_arch = "x86_64")]
    {
        if len >= crate::slot::MIN_SLOTS_AVX512 && std::arch::is_x86_feature_detected!("avx512f") {
            return Backend::Avx512;
        }
        if len >= crate::slot::MIN_SLOTS_SIMD && std::arch::is_x86_feature_detected!("avx2") {
            return Backend::Avx2Fma;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if len >= crate::slot::MIN_SLOTS_SIMD {
            return Backend::Neon;
        }
    }
    let _ = len;
    Backend::Portable
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_vectors_are_portable() {
        assert_eq!(dense_backend(1), Backend::Portable);
        assert_eq!(dense_backend(15), Backend::Portable);
        assert_eq!(slot_backend(1), Backend::Portable);
    }

    #[test]
    fn simd_lengths_dispatch_on_supported_archs() {
        let b = dense_backend(768);
        #[cfg(target_arch = "aarch64")]
        assert_eq!(b, Backend::Neon);
        #[cfg(target_arch = "x86_64")]
        assert_ne!(b, Backend::Neon);
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        assert_eq!(b, Backend::Portable);
    }

    #[test]
    fn display_names_are_stable() {
        // These strings appear in bug reports and logs; renaming them is a
        // breaking change for anyone parsing them.
        assert_eq!(Backend::Avx512.to_string(), "avx512");
        assert_eq!(Backend::Portable.to_string(), "portable");
    }
}
