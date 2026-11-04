//! SIMD backend introspection: which kernel family will actually run.
//!
//! The SIMD level is detected once per call block (with the `simd` feature)
//! and combined with a per-kernel length threshold ([`Thresholds`](crate::backend::Thresholds)); callers
//! never see either. These functions
//! report the choice so perf work and bug reports don't have to re-derive it.
//!
//! ```
//! use innr::backend::{dense_backend, Backend};
//!
//! println!("innr: {}", innr::capabilities());
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
    /// Scalar code, auto-vectorized by LLVM where possible. The only backend
    /// when the `simd` feature is off.
    Portable,
    /// WebAssembly SIMD128, when compiled with `-C target-feature=+simd128`.
    /// Declared last so existing variants keep their discriminants.
    WasmSimd128,
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

/// Input lengths at which the `simd` feature's dispatched kernels take over
/// from the portable kernels, plus the portable small-input cutoffs.
///
/// Each SIMD threshold is the shortest input (elements, or bytes for
/// `hamming_u8`) that runs the dispatched kernel; shorter inputs run the
/// portable kernel. The values were set from consumer-shape benchmarks on an
/// Apple M3 (NEON); x86 has not been measured yet and uses the same values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Thresholds {
    /// Dense `f32` kernels: dot, cosine, L1, squared L2, norm, MaxSim (per
    /// token dimension) and `fast_cosine_dispatch`.
    pub dense_f32: usize,
    /// Dense `f64` kernels: dot, cosine, L1, squared L2, norm.
    pub dense_f64: usize,
    /// `u8` dot product.
    pub dot_u8: usize,
    /// Byte-packed bit Hamming distance, in bytes.
    pub hamming_u8: usize,
    /// Mixed `f32`-by-`u8` dot product.
    pub mixed_u8_f32: usize,
    /// `u16` and `u32` slot Hamming.
    pub slot_u16_u32: usize,
    /// `u64` slot Hamming; `None` means it always runs the portable kernel.
    pub slot_u64: Option<usize>,
    /// Portable dense `f32` kernels use a four-lane small-input path below
    /// this length and sixteen lanes from it on.
    pub small_f32: usize,
    /// The same cutoff for the portable dense `f64` kernels.
    pub small_f64: usize,
}

/// The thresholds the dispatchers use.
pub(crate) const THRESHOLDS: Thresholds = Thresholds {
    dense_f32: 16,
    dense_f64: 8,
    dot_u8: 32,
    // x86_64 (i7-8559U, AVX2): SIMD no faster than the portable u64-word
    // kernel up to 96 B, faster from 128 B. aarch64 with dotprod (Apple M3): the portable
    // byte loop compiles to cnt + udot and ties the SIMD kernel from 2 KiB,
    // beating it below (128 B 7.7 vs 11.0 ns).
    hamming_u8: if cfg!(all(target_arch = "aarch64", target_feature = "dotprod")) {
        2048
    } else {
        128
    },
    mixed_u8_f32: 16,
    slot_u16_u32: 8,
    slot_u64: None,
    small_f32: crate::portable::F32_LANES,
    small_f64: crate::portable::F64_LANES,
};

/// The SIMD level innr dispatches to on this machine and the length
/// thresholds at which it does. Returned by [`capabilities`].
///
/// `Display` prints one line suitable for benchmark output and bug reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Capabilities {
    /// The level the dispatched kernels run on, before any length threshold.
    /// Always [`Backend::Portable`] without the `simd` feature.
    pub backend: Backend,
    /// Length thresholds per kernel family.
    pub thresholds: Thresholds,
}

impl fmt::Display for Capabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let t = &self.thresholds;
        write!(f, "{}", self.backend)?;
        if self.backend != Backend::Portable {
            write!(
                f,
                " (simd from: dense_f32 {}, dense_f64 {}, dot_u8 {}, hamming_u8 {} B, \
                 mixed_u8_f32 {}, slot_u16_u32 {}, slot_u64 ",
                t.dense_f32, t.dense_f64, t.dot_u8, t.hamming_u8, t.mixed_u8_f32, t.slot_u16_u32,
            )?;
            match t.slot_u64 {
                Some(n) => write!(f, "{n})")?,
                None => f.write_str("never)")?,
            }
        }
        write!(
            f,
            "; portable small path below: f32 {}, f64 {}",
            t.small_f32, t.small_f64
        )
    }
}

/// The SIMD level and length thresholds innr uses on this machine.
///
/// ```
/// let caps = innr::capabilities();
/// println!("innr: {caps}"); // e.g. "neon (simd from: dense_f32 16, ...)"
/// assert!(caps.thresholds.dense_f32 > 0);
/// ```
#[must_use]
pub fn capabilities() -> Capabilities {
    Capabilities {
        backend: level(),
        thresholds: THRESHOLDS,
    }
}

fn level() -> Backend {
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
    if len >= THRESHOLDS.dense_f32 {
        level()
    } else {
        Backend::Portable
    }
}

/// Backend `slot_hamming_u32` selects for `len`-slot sketches on this
/// machine.
#[must_use]
pub fn slot_backend(len: usize) -> Backend {
    if len >= THRESHOLDS.slot_u16_u32 {
        level()
    } else {
        Backend::Portable
    }
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
        assert_eq!(b, capabilities().backend);
    }

    #[test]
    fn capabilities_display_names_level_and_thresholds() {
        let caps = capabilities();
        let line = caps.to_string();
        assert!(line.starts_with(&caps.backend.to_string()), "{line}");
        assert!(line.contains("small path below: f32 16, f64 8"), "{line}");
        if caps.backend != Backend::Portable {
            let hamming = if cfg!(all(target_arch = "aarch64", target_feature = "dotprod")) {
                "hamming_u8 2048 B"
            } else {
                "hamming_u8 128 B"
            };
            assert!(line.contains(hamming), "{line}");
            assert!(line.contains("slot_u64 never"), "{line}");
        }
    }

    #[test]
    fn display_names_are_stable() {
        // These strings appear in bug reports and logs; renaming them is a
        // breaking change for anyone parsing them.
        assert_eq!(Backend::Avx512.to_string(), "avx512");
        assert_eq!(Backend::Portable.to_string(), "portable");
    }
}
