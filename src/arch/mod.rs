//! Runtime SIMD dispatch on `fearless_simd` level tokens.
//!
//! [`isa`] picks the best supported level once per call block and returns it
//! as an [`Isa`] token. [`simd_dispatch!`] matches on that token and runs a
//! kernel generic over [`fearless_simd::Simd`] inside `Simd::vectorize`, which
//! enables the level's target features for the inlined kernel body. Kernels
//! live in [`kernels`]; they use only `fearless_simd`'s safe vector types, so
//! no `unsafe` code is written in or expanded into this crate.
//!
//! Levels below AVX2 on x86 (SSE2, SSE4.2) and targets without a SIMD level
//! report [`Isa::Portable`]: the portable kernels already auto-vectorize to
//! the compile target's baseline, so a separate SSE path adds code without a
//! measured win.

// On targets with no dispatched level (armv7, wasm32 without simd128) only
// `Isa::Portable` exists, so kernels reached only through `simd_dispatch!`
// are never called.
#[cfg_attr(
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code)
)]
pub(crate) mod kernels;

use fearless_simd::Level;

/// The SIMD level a call block runs on, carrying the token that proves it.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Isa {
    /// Ice Lake-class AVX-512 (fearless_simd's `Avx512` level).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx512(fearless_simd::Avx512),
    /// x86-64-v3: AVX2, FMA, BMI1/2, F16C, LZCNT, MOVBE, POPCNT.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2(fearless_simd::Avx2),
    /// NEON on aarch64.
    #[cfg(target_arch = "aarch64")]
    Neon(fearless_simd::Neon),
    /// WebAssembly SIMD128 (only when compiled with `+simd128`).
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Wasm(fearless_simd::WasmSimd128),
    /// No SIMD level worth dispatching to; use the portable kernels.
    Portable,
}

/// Detect the best SIMD level for this CPU.
///
/// On x86 `Level::new` caches CPU detection after the first call; on other
/// targets the level is known at compile time. Either way this is cheap
/// enough to call once per public-API call.
#[inline]
pub(crate) fn isa() -> Isa {
    let level = Level::new();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if let Some(t) = level.as_avx512() {
            return Isa::Avx512(t);
        }
        if let Some(t) = level.as_avx2() {
            return Isa::Avx2(t);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if let Some(t) = level.as_neon() {
            return Isa::Neon(t);
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        if let Some(t) = level.as_wasm_simd128() {
            return Isa::Wasm(t);
        }
    }
    let _ = level;
    Isa::Portable
}

/// Run `$simd` with `$s` bound to the token for `$isa`, inside that level's
/// target-feature context; run `$portable` when `$isa` is [`Isa::Portable`].
///
/// `$simd` should call `#[inline(always)]` kernels so their bodies compile
/// with the level's features. The expression is expanded once per level.
macro_rules! simd_dispatch {
    ($isa:expr, $s:ident => $simd:expr, _ => $portable:expr $(,)?) => {
        match $isa {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            $crate::arch::Isa::Avx512($s) => ::fearless_simd::Simd::vectorize(
                $s,
                #[inline(always)]
                || $simd,
            ),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            $crate::arch::Isa::Avx2($s) => ::fearless_simd::Simd::vectorize(
                $s,
                #[inline(always)]
                || $simd,
            ),
            #[cfg(target_arch = "aarch64")]
            $crate::arch::Isa::Neon($s) => ::fearless_simd::Simd::vectorize(
                $s,
                #[inline(always)]
                || $simd,
            ),
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            $crate::arch::Isa::Wasm($s) => ::fearless_simd::Simd::vectorize(
                $s,
                #[inline(always)]
                || $simd,
            ),
            $crate::arch::Isa::Portable => $portable,
        }
    };
}
pub(crate) use simd_dispatch;
