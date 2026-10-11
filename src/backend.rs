//! SIMD backend introspection: which kernel family will actually run.
//!
//! Dispatch is per-call from CPU detection plus input length, and invisible
//! to callers. These functions report the choice so perf work and bug
//! reports don't have to re-derive it.
//!
//! ```
//! use innr::backend::{dense_backend, Backend};
//!
//! let b = dense_backend(768); // SIMD when the `simd` feature is on
//! println!("768-dim dense kernels run on {b}");
//! assert_eq!(dense_backend(8), Backend::Portable); // below every dense threshold
//! ```

use core::fmt;

/// A kernel family the dispatchers can select.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// AVX-512: the x86-64-v4 level (AVX-512F/BW/CD/DQ/VL), pulp's `V4` token.
    Avx512,
    /// AVX2 + FMA: the x86-64-v3 level (AVX2, FMA, BMI1/2, LZCNT), pulp's `V3`.
    Avx2Fma,
    /// NEON (always available on aarch64).
    Neon,
    /// Scalar code, auto-vectorized by LLVM where possible. The only backend
    /// when the `simd` feature is off.
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
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if len >= crate::dense::MIN_DIM_AVX512 && pulp::x86::V4::try_new().is_some() {
            return Backend::Avx512;
        }
        if len >= crate::MIN_DIM_SIMD && pulp::x86::V3::try_new().is_some() {
            return Backend::Avx2Fma;
        }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
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
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if len >= crate::slot::MIN_SLOTS_AVX512 && pulp::x86::V4::try_new().is_some() {
            return Backend::Avx512;
        }
        if len >= crate::slot::MIN_SLOTS_SIMD && pulp::x86::V3::try_new().is_some() {
            return Backend::Avx2Fma;
        }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
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
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        assert_eq!(b, Backend::Neon);
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        assert_ne!(b, Backend::Neon);
        #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
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
