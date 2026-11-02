//! SIMD backend introspection: which kernel family will actually run.
//!
//! The SIMD level is detected once per call block (with the `simd` feature)
//! and combined with the input length; callers never see it. These functions
//! report the choice so perf work and bug reports don't have to re-derive it.
//!
//! ```
//! use innr::backend::{dense_backend, Backend};
//!
//! println!("innr kernels run on {}", innr::capabilities());
//! let b = dense_backend(768); // SIMD when the `simd` feature is on
//! println!("768-dim dense kernels run on {b}");
//! assert_eq!(dense_backend(8), Backend::Portable); // below every dense threshold
//! ```

use core::fmt;

/// A kernel family the dispatchers can select.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// Ice Lake-class AVX-512 (AVX-512F/BW/CD/DQ/VL plus VPOPCNTDQ, BITALG,
    /// VBMI/VBMI2, VNNI, IFMA, GFNI, VAES): fearless_simd's `Avx512` level.
    /// CPUs with only the older AVX-512 subset run [`Backend::Avx2Fma`].
    Avx512,
    /// AVX2 + FMA: the x86-64-v3 level (AVX2, FMA, BMI1/2, F16C, LZCNT,
    /// MOVBE, POPCNT), fearless_simd's `Avx2` level.
    Avx2Fma,
    /// NEON (always available on aarch64).
    Neon,
    /// WebAssembly SIMD128, when compiled with `-C target-feature=+simd128`.
    WasmSimd128,
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
            Backend::WasmSimd128 => "wasm-simd128",
            Backend::Portable => "portable",
        };
        f.write_str(s)
    }
}

/// The SIMD level innr dispatches to on this machine, before any per-kernel
/// length threshold. Always [`Backend::Portable`] without the `simd` feature.
///
/// ```
/// let level = innr::capabilities();
/// println!("innr kernels run on {level}");
/// ```
#[must_use]
pub fn capabilities() -> Backend {
    #[cfg(feature = "simd")]
    {
        use crate::arch::Isa;
        match crate::arch::isa() {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Isa::Avx512(_) => Backend::Avx512,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Isa::Avx2(_) => Backend::Avx2Fma,
            #[cfg(target_arch = "aarch64")]
            Isa::Neon(_) => Backend::Neon,
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Isa::Wasm(_) => Backend::WasmSimd128,
            Isa::Portable => Backend::Portable,
        }
    }
    #[cfg(not(feature = "simd"))]
    {
        Backend::Portable
    }
}

/// Backend the dense f32 kernels (`dot`, `cosine`, `l1_distance`,
/// `l2_distance`, ...) select for `len`-dimensional vectors on this machine.
#[must_use]
pub fn dense_backend(len: usize) -> Backend {
    if len >= crate::MIN_DIM_SIMD {
        capabilities()
    } else {
        Backend::Portable
    }
}

/// Backend `slot_hamming_u32` selects for `len`-slot sketches on this
/// machine.
#[must_use]
pub fn slot_backend(len: usize) -> Backend {
    #[cfg(feature = "simd")]
    if len >= crate::slot::MIN_SLOTS_SIMD {
        return capabilities();
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
        #[cfg(not(feature = "simd"))]
        assert_eq!(b, Backend::Portable);
        assert_eq!(b, capabilities());
    }

    #[test]
    fn display_names_are_stable() {
        // These strings appear in bug reports and logs; renaming them is a
        // breaking change for anyone parsing them.
        assert_eq!(Backend::Avx512.to_string(), "avx512");
        assert_eq!(Backend::Portable.to_string(), "portable");
    }
}
