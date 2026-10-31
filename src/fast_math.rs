//! Fast math operations using hardware-aware approximations.
//!
//! # Newton-Raphson Inverse Square Root
//!
//! Modern SIMD instruction sets provide `rsqrt` (reciprocal square root) estimates:
//! - x86 SSE/AVX: `rsqrtps` with ~12 bits of accuracy (error up to 1.5e-4)
//! - x86 AVX-512: `vrsqrt14ps` with ~14 bits of accuracy
//! - ARM NEON: `vrsqrteq_f32` with ~8-12 bits of accuracy
//!
//! One Newton-Raphson iteration improves accuracy by squaring the number of correct bits:
//! - Input accuracy: ~12 bits → Output accuracy: ~24 bits (nearly full f32 precision)
//!
//! The iteration formula: `y' = y * (1.5 - 0.5 * x * y * y)`
//!
//! # Performance Impact
//!
//! For cosine similarity, we compute `dot(a,b) / sqrt(dot(a,a) * dot(b,b))`.
//! Traditional: 2 sqrt + 1 div = ~40-60 cycles
//! Fast rsqrt + NR: 2 rsqrt + 2 mul chains = ~10-15 cycles
//!
//! # References
//!
//! - SimSIMD (Vardanian 2023): demonstrates 3-10x speedups with rsqrt+NR
//! - Quake III "fast inverse square root" (Lomont 2003): classic integer bit-hack analysis

use crate::NORM_EPSILON_SQ;

// MIN_DIM_SIMD is only used on architectures with SIMD dispatch
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
use crate::MIN_DIM_SIMD;

/// Fast scalar inverse square root using the classic Quake III bit-hack.
///
/// One Newton-Raphson iteration from the 0x5f375a86 seed: maximum relative
/// error about 1.75e-3 (about 9 bits), as derived by Lomont (2003), "Fast
/// Inverse Square Root", and confirmed exhaustively over one period `[1, 4)`.
///
/// # Safety Note
///
/// Input must be positive. Zero or negative inputs return 0.0.
///
/// ```text
/// use innr::fast_rsqrt;
///
/// let r = fast_rsqrt(4.0);
/// assert!((r - 0.5).abs() < 1e-3);  // 1/sqrt(4) = 0.5
/// ```
#[inline]
#[must_use]
pub fn fast_rsqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    // Initial estimate via integer bit manipulation
    // Magic constant 0x5f375a86 is slightly better than original 0x5f3759df
    let i = x.to_bits();
    let y = f32::from_bits(0x5f375a86 - (i >> 1));
    // One Newton-Raphson iteration: y' = y * (1.5 - 0.5 * x * y * y)
    y * (1.5 - 0.5 * x * y * y)
}

/// Fast scalar inverse square root with two NR iterations.
///
/// Maximum relative error about 4.7e-6 (about 17 bits). It is not correctly
/// rounded, so results can differ from `1.0 / x.sqrt()` in the low bits.
#[inline]
#[must_use]
pub fn fast_rsqrt_precise(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let i = x.to_bits();
    let mut y = f32::from_bits(0x5f375a86 - (i >> 1));
    // Two NR iterations
    y = y * (1.5 - 0.5 * x * y * y);
    y * (1.5 - 0.5 * x * y * y)
}

/// Fast cosine similarity using rsqrt approximation.
///
/// Computes `dot(a,b) / sqrt(dot(a,a) * dot(b,b))` using fast inverse sqrt.
///
/// # Accuracy
///
/// Each of the two [`fast_rsqrt`] factors has up to about 1.75e-3 relative
/// error, so the result can be off by about 3.5e-3 relative, for normalized
/// vectors too: `fast_cosine(u, u)` for the unit vector `[1, 0, 0]` is 0.99662.
/// Use [`crate::cosine`] when that matters.
///
/// # Example
///
/// ```text
/// use innr::fast_math::fast_cosine;
///
/// let a = [1.0_f32, 0.0, 0.0];
/// let b = [0.707, 0.707, 0.0];
/// let c = fast_cosine(&a, &b);
/// assert!((c - 0.707).abs() < 0.01);
/// ```
#[inline]
#[must_use]
pub fn fast_cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "innr::fast_cosine: slice length mismatch ({} vs {})",
        a.len(),
        b.len()
    );

    // Compute three dot products in one pass for better cache locality
    let mut ab = 0.0f32;
    let mut aa = 0.0f32;
    let mut bb = 0.0f32;

    for (&ai, &bi) in a.iter().zip(b.iter()) {
        ab += ai * bi;
        aa += ai * ai;
        bb += bi * bi;
    }

    // Use rsqrt to avoid sqrt and division
    // cosine = ab / sqrt(aa * bb) = ab * rsqrt(aa) * rsqrt(bb)
    if aa > NORM_EPSILON_SQ && bb > NORM_EPSILON_SQ {
        ab * fast_rsqrt(aa) * fast_rsqrt(bb)
    } else {
        0.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD implementations
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(clippy::incompatible_msrv)]
pub(crate) mod x86_64 {
    //! AVX2/AVX-512 fast cosine using rsqrt instructions.

    use crate::arch::x86_64::{ld256_f32, ld512_f32};
    use pulp::x86::{V3, V4};
    use std::arch::x86_64::{__m256, __m512};

    /// AVX-512 fast cosine with hardware rsqrt + Newton-Raphson.
    ///
    /// Uses SSE `rsqrtps` for ~12-bit initial estimate, then one NR iteration.
    /// We use `_mm_rsqrt_ps` (SSE) rather than `_mm512_rsqrt14_ps` to avoid
    /// requiring AVX-512VL; SSE is always available when AVX-512F is present.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[inline]
    pub fn fast_cosine_avx512(simd: V4, a: &[f32], b: &[f32]) -> f32 {
        simd.vectorize(|| fast_cosine_avx512_in(simd, a, b))
    }

    #[inline(always)]
    fn fast_cosine_avx512_in(simd: V4, a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }

        let (a16, a_tail) = a[..n].as_chunks::<16>();
        let (b16, b_tail) = b[..n].as_chunks::<16>();

        let mut ab_sum: __m512 = simd.avx512f._mm512_setzero_ps();
        let mut aa_sum: __m512 = simd.avx512f._mm512_setzero_ps();
        let mut bb_sum: __m512 = simd.avx512f._mm512_setzero_ps();

        // Process 16 floats at a time
        for (qa, qb) in a16.iter().zip(b16) {
            let va = ld512_f32(qa);
            let vb = ld512_f32(qb);

            ab_sum = simd.avx512f._mm512_fmadd_ps(va, vb, ab_sum);
            aa_sum = simd.avx512f._mm512_fmadd_ps(va, va, aa_sum);
            bb_sum = simd.avx512f._mm512_fmadd_ps(vb, vb, bb_sum);
        }

        // Reduce to scalars
        let mut ab = simd.avx512f._mm512_reduce_add_ps(ab_sum);
        let mut aa = simd.avx512f._mm512_reduce_add_ps(aa_sum);
        let mut bb = simd.avx512f._mm512_reduce_add_ps(bb_sum);

        // Handle tail
        for (&ai, &bi) in a_tail.iter().zip(b_tail) {
            ab += ai * bi;
            aa += ai * ai;
            bb += bi * bi;
        }

        // Hardware rsqrt via SSE _mm_rsqrt_ps + one Newton-Raphson iteration
        if aa > super::NORM_EPSILON_SQ && bb > super::NORM_EPSILON_SQ {
            // Pack [aa, bb, 1.0, 1.0] into a 128-bit SSE register
            let vals = simd.sse._mm_set_ps(1.0, 1.0, bb, aa);
            // Initial ~12-bit rsqrt estimate
            let est = simd.sse._mm_rsqrt_ps(vals);
            // One NR iteration: y = y * (1.5 - 0.5 * x * y * y)
            let half = simd.sse._mm_set1_ps(0.5);
            let three_half = simd.sse._mm_set1_ps(1.5);
            let half_x = simd.sse._mm_mul_ps(half, vals);
            let est_sq = simd.sse._mm_mul_ps(est, est);
            let refined = simd.sse._mm_mul_ps(
                est,
                simd.sse
                    ._mm_sub_ps(three_half, simd.sse._mm_mul_ps(half_x, est_sq)),
            );
            // Extract lane 0 (rsqrt_aa) and lane 1 (rsqrt_bb)
            let rsqrt_aa = simd.sse._mm_cvtss_f32(refined);
            let rsqrt_bb = simd
                .sse
                ._mm_cvtss_f32(simd.sse._mm_shuffle_ps::<0x55>(refined, refined));
            ab * rsqrt_aa * rsqrt_bb
        } else {
            0.0
        }
    }

    /// AVX2 fast cosine with hardware rsqrt + Newton-Raphson.
    ///
    /// Uses SSE `_mm_rsqrt_ps` for ~12-bit initial estimate, then one
    /// Newton-Raphson iteration for ~24 bits of accuracy.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[inline]
    pub fn fast_cosine_avx2(simd: V3, a: &[f32], b: &[f32]) -> f32 {
        simd.vectorize(|| fast_cosine_avx2_in(simd, a, b))
    }

    #[inline(always)]
    fn fast_cosine_avx2_in(simd: V3, a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }

        let (a8, a_tail) = a[..n].as_chunks::<8>();
        let (b8, b_tail) = b[..n].as_chunks::<8>();

        let mut ab_sum: __m256 = simd.avx._mm256_setzero_ps();
        let mut aa_sum: __m256 = simd.avx._mm256_setzero_ps();
        let mut bb_sum: __m256 = simd.avx._mm256_setzero_ps();

        // Process 8 floats at a time
        for (qa, qb) in a8.iter().zip(b8) {
            let va = ld256_f32(qa);
            let vb = ld256_f32(qb);

            ab_sum = simd.fma._mm256_fmadd_ps(va, vb, ab_sum);
            aa_sum = simd.fma._mm256_fmadd_ps(va, va, aa_sum);
            bb_sum = simd.fma._mm256_fmadd_ps(vb, vb, bb_sum);
        }

        // Horizontal reduction helper
        #[inline(always)]
        fn hsum256(simd: V3, v: __m256) -> f32 {
            let hi = simd.avx._mm256_extractf128_ps::<1>(v);
            let lo = simd.avx._mm256_castps256_ps128(v);
            let sum128 = simd.sse._mm_add_ps(lo, hi);
            let sum64 = simd
                .sse
                ._mm_add_ps(sum128, simd.sse._mm_movehl_ps(sum128, sum128));
            let sum32 = simd
                .sse
                ._mm_add_ss(sum64, simd.sse._mm_shuffle_ps::<1>(sum64, sum64));
            simd.sse._mm_cvtss_f32(sum32)
        }

        let mut ab = hsum256(simd, ab_sum);
        let mut aa = hsum256(simd, aa_sum);
        let mut bb = hsum256(simd, bb_sum);

        // Handle tail
        for (&ai, &bi) in a_tail.iter().zip(b_tail) {
            ab += ai * bi;
            aa += ai * ai;
            bb += bi * bi;
        }

        // Hardware rsqrt via SSE _mm_rsqrt_ps + one Newton-Raphson iteration
        if aa > super::NORM_EPSILON_SQ && bb > super::NORM_EPSILON_SQ {
            // Pack [aa, bb, 1.0, 1.0] into a 128-bit SSE register
            let vals = simd.sse._mm_set_ps(1.0, 1.0, bb, aa);
            // Initial ~12-bit rsqrt estimate
            let est = simd.sse._mm_rsqrt_ps(vals);
            // One NR iteration: y = y * (1.5 - 0.5 * x * y * y)
            let half = simd.sse._mm_set1_ps(0.5);
            let three_half = simd.sse._mm_set1_ps(1.5);
            let half_x = simd.sse._mm_mul_ps(half, vals);
            let est_sq = simd.sse._mm_mul_ps(est, est);
            let refined = simd.sse._mm_mul_ps(
                est,
                simd.sse
                    ._mm_sub_ps(three_half, simd.sse._mm_mul_ps(half_x, est_sq)),
            );
            // Extract lane 0 (rsqrt_aa) and lane 1 (rsqrt_bb)
            let rsqrt_aa = simd.sse._mm_cvtss_f32(refined);
            let rsqrt_bb = simd
                .sse
                ._mm_cvtss_f32(simd.sse._mm_shuffle_ps::<0x55>(refined, refined));
            ab * rsqrt_aa * rsqrt_bb
        } else {
            0.0
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub(crate) mod aarch64 {
    //! NEON fast cosine using vrsqrte + Newton-Raphson.

    use crate::arch::aarch64::ld_f32;
    use pulp::aarch64::Neon;
    use std::arch::aarch64::{float32x2_t, float32x4_t};

    /// NEON fast cosine with 4-way unrolled accumulation and hardware rsqrt.
    ///
    /// Uses `vrsqrteq_f32` + two `vrsqrtsq_f32` Newton-Raphson iterations for
    /// the final normalization, avoiding scalar/SIMD transitions entirely.
    #[inline]
    pub fn fast_cosine_neon(simd: Neon, a: &[f32], b: &[f32]) -> f32 {
        simd.vectorize(|| fast_cosine_neon_in(simd, a, b))
    }

    #[inline(always)]
    fn fast_cosine_neon_in(simd: Neon, a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }

        let (a4, a_tail) = a[..n].as_chunks::<4>();
        let (b4, b_tail) = b[..n].as_chunks::<4>();
        let (a16, a_rem) = a4.as_chunks::<4>();
        let (b16, b_rem) = b4.as_chunks::<4>();

        // 4-way unrolled: process 16 floats per iteration
        let mut ab0: float32x4_t = simd.neon.vdupq_n_f32(0.0);
        let mut ab1: float32x4_t = simd.neon.vdupq_n_f32(0.0);
        let mut ab2: float32x4_t = simd.neon.vdupq_n_f32(0.0);
        let mut ab3: float32x4_t = simd.neon.vdupq_n_f32(0.0);
        let mut aa0: float32x4_t = simd.neon.vdupq_n_f32(0.0);
        let mut aa1: float32x4_t = simd.neon.vdupq_n_f32(0.0);
        let mut aa2: float32x4_t = simd.neon.vdupq_n_f32(0.0);
        let mut aa3: float32x4_t = simd.neon.vdupq_n_f32(0.0);
        let mut bb0: float32x4_t = simd.neon.vdupq_n_f32(0.0);
        let mut bb1: float32x4_t = simd.neon.vdupq_n_f32(0.0);
        let mut bb2: float32x4_t = simd.neon.vdupq_n_f32(0.0);
        let mut bb3: float32x4_t = simd.neon.vdupq_n_f32(0.0);

        for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a16.iter().zip(b16) {
            let (va0, vb0) = (ld_f32(a0), ld_f32(b0));
            let (va1, vb1) = (ld_f32(a1), ld_f32(b1));
            let (va2, vb2) = (ld_f32(a2), ld_f32(b2));
            let (va3, vb3) = (ld_f32(a3), ld_f32(b3));

            ab0 = simd.neon.vfmaq_f32(ab0, va0, vb0);
            ab1 = simd.neon.vfmaq_f32(ab1, va1, vb1);
            ab2 = simd.neon.vfmaq_f32(ab2, va2, vb2);
            ab3 = simd.neon.vfmaq_f32(ab3, va3, vb3);

            aa0 = simd.neon.vfmaq_f32(aa0, va0, va0);
            aa1 = simd.neon.vfmaq_f32(aa1, va1, va1);
            aa2 = simd.neon.vfmaq_f32(aa2, va2, va2);
            aa3 = simd.neon.vfmaq_f32(aa3, va3, va3);

            bb0 = simd.neon.vfmaq_f32(bb0, vb0, vb0);
            bb1 = simd.neon.vfmaq_f32(bb1, vb1, vb1);
            bb2 = simd.neon.vfmaq_f32(bb2, vb2, vb2);
            bb3 = simd.neon.vfmaq_f32(bb3, vb3, vb3);
        }

        // Combine the 4 accumulators
        let ab_sum = simd
            .neon
            .vaddq_f32(simd.neon.vaddq_f32(ab0, ab1), simd.neon.vaddq_f32(ab2, ab3));
        let aa_sum = simd
            .neon
            .vaddq_f32(simd.neon.vaddq_f32(aa0, aa1), simd.neon.vaddq_f32(aa2, aa3));
        let bb_sum = simd
            .neon
            .vaddq_f32(simd.neon.vaddq_f32(bb0, bb1), simd.neon.vaddq_f32(bb2, bb3));

        let mut ab = simd.neon.vaddvq_f32(ab_sum);
        let mut aa = simd.neon.vaddvq_f32(aa_sum);
        let mut bb = simd.neon.vaddvq_f32(bb_sum);

        // Handle remaining 4-float chunks
        let mut ab_tail: float32x4_t = simd.neon.vdupq_n_f32(0.0);
        let mut aa_tail: float32x4_t = simd.neon.vdupq_n_f32(0.0);
        let mut bb_tail: float32x4_t = simd.neon.vdupq_n_f32(0.0);

        for (qa, qb) in a_rem.iter().zip(b_rem) {
            let (va, vb) = (ld_f32(qa), ld_f32(qb));
            ab_tail = simd.neon.vfmaq_f32(ab_tail, va, vb);
            aa_tail = simd.neon.vfmaq_f32(aa_tail, va, va);
            bb_tail = simd.neon.vfmaq_f32(bb_tail, vb, vb);
        }

        ab += simd.neon.vaddvq_f32(ab_tail);
        aa += simd.neon.vaddvq_f32(aa_tail);
        bb += simd.neon.vaddvq_f32(bb_tail);

        // Scalar tail
        for (&ai, &bi) in a_tail.iter().zip(b_tail) {
            ab += ai * bi;
            aa += ai * ai;
            bb += bi * bi;
        }

        if aa > super::NORM_EPSILON_SQ && bb > super::NORM_EPSILON_SQ {
            // NEON hardware rsqrt: vrsqrte + two Newton-Raphson iterations via vrsqrts.
            // Pack aa and bb into a float32x2_t, compute rsqrt in-register,
            // then extract and multiply. This avoids scalar/SIMD transitions.
            let norms: float32x2_t = simd.neon.vdup_n_f32(0.0);
            // Load aa into lane 0, bb into lane 1
            let norms = simd.neon.vset_lane_f32::<0>(aa, norms);
            let norms = simd.neon.vset_lane_f32::<1>(bb, norms);

            // Initial estimate (~8-12 bits)
            let mut est: float32x2_t = simd.neon.vrsqrte_f32(norms);

            // Newton-Raphson iteration 1: est = est * vrsqrts(norms, est * est)
            est = simd.neon.vmul_f32(
                est,
                simd.neon.vrsqrts_f32(norms, simd.neon.vmul_f32(est, est)),
            );
            // Newton-Raphson iteration 2
            est = simd.neon.vmul_f32(
                est,
                simd.neon.vrsqrts_f32(norms, simd.neon.vmul_f32(est, est)),
            );

            let rsqrt_aa = simd.neon.vget_lane_f32::<0>(est);
            let rsqrt_bb = simd.neon.vget_lane_f32::<1>(est);

            ab * rsqrt_aa * rsqrt_bb
        } else {
            0.0
        }
    }
}

/// Dispatch to the fastest available `fast_cosine` implementation.
///
/// Selects AVX-512, AVX2+FMA, NEON, or portable based on runtime detection.
///
/// The SIMD paths refine a hardware rsqrt estimate to near full f32 precision.
/// Inputs shorter than 16 elements, or builds without SIMD, use the portable
/// [`fast_cosine`] and carry its error of up to about 3.5e-3 relative.
///
/// # Examples
///
/// ```text
/// use innr::fast_math::fast_cosine_dispatch;
///
/// let a = vec![1.0_f32, 0.0, 0.0];
/// let b = vec![1.0_f32, 0.0, 0.0];
/// let sim = fast_cosine_dispatch(&a, &b);
/// assert!((sim - 1.0).abs() < 0.02); // approximate, within ~1%
/// ```
#[inline]
#[must_use]
pub fn fast_cosine_dispatch(a: &[f32], b: &[f32]) -> f32 {
    // Assert up front so panic does not depend on size (short inputs hit
    // fast_cosine's assert; SIMD-sized inputs truncated silently).
    assert_eq!(
        a.len(),
        b.len(),
        "innr::fast_cosine_dispatch: slice length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    let n = a.len();

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if n >= 64 {
            if let Some(simd) = pulp::x86::V4::try_new() {
                return x86_64::fast_cosine_avx512(simd, a, b);
            }
        }
        if n >= MIN_DIM_SIMD {
            if let Some(simd) = pulp::x86::V3::try_new() {
                return x86_64::fast_cosine_avx2(simd, a, b);
            }
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if n >= MIN_DIM_SIMD {
            if let Some(simd) = pulp::aarch64::Neon::try_new() {
                return aarch64::fast_cosine_neon(simd, a, b);
            }
        }
    }

    fast_cosine(a, b)
}

#[cfg(test)]
mod tests {
    /// Moved from tests/simd_correctness.rs when fast_math became internal:
    /// the dispatched rsqrt cosine stays within 1e-2 of the exact cosine.
    #[test]
    fn fast_cosine_dispatch_matches_exact_cosine() {
        for dim in [1usize, 8, 16, 32, 64, 128, 384, 768] {
            for seed in 0..5u32 {
                let gen = |s: u32| -> Vec<f32> {
                    (0..dim)
                        .map(|i| ((i as u32 * 31 + s * 17) as f32 * 0.37).sin())
                        .collect()
                };
                let (a, b) = (gen(seed), gen(seed + 1000));
                let got = super::fast_cosine_dispatch(&a, &b);
                let want = crate::cosine(&a, &b);
                assert!(
                    (got - want).abs() < 1e-2,
                    "dim={dim} seed={seed}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn small_norm_vectors_not_zeroed() {
        // Regression: squared-norm guard vs unsquared epsilon zeroed any norm < ~3.2e-5.
        let a = [1e-6_f32, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6];
        let b = a;
        let fast = super::fast_cosine(&a, &b);
        let exact = crate::cosine(&a, &b);
        assert!(
            (fast - exact).abs() < 0.02,
            "fast {fast} vs exact {exact}: small-norm vectors must not collapse to 0"
        );
        assert!(
            fast > 0.9,
            "parallel small-norm vectors should be ~1.0, got {fast}"
        );
    }

    #[test]
    #[should_panic(expected = "slice length mismatch")]
    fn dispatch_length_mismatch_panics_for_simd_sizes() {
        // Regression: panic must not depend on size (SIMD-sized mismatches truncated silently).
        let a = vec![1.0_f32; 64];
        let b = vec![1.0_f32; 63];
        let _ = super::fast_cosine_dispatch(&a, &b);
    }

    use super::*;

    #[test]
    fn test_fast_rsqrt_accuracy() {
        // One NR iteration gives ~0.2% relative accuracy
        // This is the expected precision for the fast rsqrt approximation
        for &x in &[0.001f32, 0.1, 1.0, 10.0, 100.0, 1000.0] {
            let expected = 1.0 / x.sqrt();
            let actual = fast_rsqrt(x);
            let rel_error = (actual - expected).abs() / expected;
            assert!(
                rel_error < 0.005, // 0.5% relative error tolerance
                "rsqrt({}) = {}, expected {}, rel_error = {}",
                x,
                actual,
                expected,
                rel_error
            );
        }
    }

    #[test]
    fn test_fast_rsqrt_documented_error_bounds() {
        // The bit-hack error repeats every factor of 4 in x, so a dense sweep
        // of [1, 4) plus a few far scales covers the documented maxima
        // (1.75e-3 after one NR step, 4.7e-6 after two).
        let mut worst = (0f64, 0f64);
        for scale in [1e-30f32, 1e-3, 1.0, 1e3, 1e30] {
            for i in 0..30_000 {
                let x = scale * (1.0 + 3.0 * i as f32 / 30_000.0);
                let exact = 1.0 / f64::from(x).sqrt();
                let rel = |r: f32| (f64::from(r) - exact).abs() / exact;
                worst.0 = worst.0.max(rel(fast_rsqrt(x)));
                worst.1 = worst.1.max(rel(fast_rsqrt_precise(x)));
            }
        }
        assert!(worst.0 > 1.7e-3 && worst.0 < 1.76e-3, "{worst:?}");
        assert!(worst.1 > 4.0e-6 && worst.1 < 4.8e-6, "{worst:?}");

        let unit = [1.0f32, 0.0, 0.0];
        let c = fast_cosine(&unit, &unit);
        assert!((c - 1.0).abs() < 3.6e-3, "fast_cosine(u, u) = {c}");
    }

    #[test]
    fn test_fast_rsqrt_precise_accuracy() {
        // Two NR iterations give ~1e-5 relative accuracy
        for &x in &[0.001f32, 0.1, 1.0, 10.0, 100.0, 1000.0] {
            let expected = 1.0 / x.sqrt();
            let actual = fast_rsqrt_precise(x);
            let rel_error = (actual - expected).abs() / expected;
            assert!(
                rel_error < 1e-4, // 0.01% relative error tolerance
                "rsqrt_precise({}) = {}, expected {}, rel_error = {}",
                x,
                actual,
                expected,
                rel_error
            );
        }
    }

    #[test]
    fn test_fast_cosine_orthogonal() {
        let a = [1.0_f32, 0.0, 0.0];
        let b = [0.0_f32, 1.0, 0.0];
        let c = fast_cosine(&a, &b);
        assert!(c.abs() < 1e-5, "orthogonal vectors should have cosine ~0");
    }

    #[test]
    fn test_fast_cosine_parallel() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [2.0_f32, 4.0, 6.0]; // parallel to a
        let c = fast_cosine(&a, &b);
        // rsqrt approximation gives ~0.3% error on cosine
        assert!(
            (c - 1.0).abs() < 0.01,
            "parallel vectors should have cosine ~1, got {}",
            c
        );
    }

    #[test]
    fn test_fast_cosine_antiparallel() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [-1.0_f32, -2.0, -3.0];
        let c = fast_cosine(&a, &b);
        assert!(
            (c + 1.0).abs() < 0.01,
            "antiparallel vectors should have cosine ~-1, got {}",
            c
        );
    }

    #[test]
    fn test_fast_cosine_vs_standard() {
        use crate::cosine;

        // Compare fast_cosine against standard cosine for random-ish vectors
        // Fast cosine trades precision for speed: ~0.5% relative error expected
        for dim in [3, 16, 64, 128, 256, 512, 768, 1024, 1536] {
            let a: Vec<f32> = (0..dim).map(|i| ((i * 7) as f32).sin()).collect();
            let b: Vec<f32> = (0..dim).map(|i| ((i * 11) as f32).cos()).collect();

            let standard = cosine(&a, &b);
            let fast = fast_cosine(&a, &b);

            let diff = (standard - fast).abs();
            // Allow 1% absolute error (rsqrt approximation)
            assert!(
                diff < 0.01,
                "dim={}: standard={}, fast={}, diff={}",
                dim,
                standard,
                fast,
                diff
            );
        }
    }

    #[test]
    fn test_fast_cosine_dispatch_consistency() {
        // Portable uses scalar Quake III bit-hack rsqrt (~23 bits).
        // NEON dispatch uses hardware vrsqrte + 2 NR iterations (~24 bits).
        // Both are close to IEEE but differ slightly from each other.
        for dim in [8, 16, 32, 64, 128, 256, 512, 1024] {
            let a: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
            let b: Vec<f32> = (0..dim).map(|i| 1.0 - (i as f32 / dim as f32)).collect();

            let standard = crate::cosine(&a, &b);
            let portable = fast_cosine(&a, &b);
            let dispatched = fast_cosine_dispatch(&a, &b);

            // Both approximations should be close to the standard result
            let port_err = (portable - standard).abs();
            let disp_err = (dispatched - standard).abs();
            assert!(
                port_err < 0.01,
                "dim={}: portable={}, standard={}, err={}",
                dim,
                portable,
                standard,
                port_err
            );
            assert!(
                disp_err < 0.01,
                "dim={}: dispatched={}, standard={}, err={}",
                dim,
                dispatched,
                standard,
                disp_err
            );
            // And they should agree within 2e-3 (different rsqrt methods:
            // scalar Quake III bit-hack vs NEON vrsqrte + 2 NR iterations)
            let diff = (portable - dispatched).abs();
            assert!(
                diff < 2e-3,
                "dim={}: portable={}, dispatched={}, diff={}",
                dim,
                portable,
                dispatched,
                diff
            );
        }
    }

    #[test]
    fn test_fast_cosine_zero_vector() {
        let a = [1.0_f32, 2.0, 3.0];
        let zero = [0.0_f32, 0.0, 0.0];
        assert_eq!(fast_cosine(&a, &zero), 0.0);
        assert_eq!(fast_cosine(&zero, &a), 0.0);
    }

    #[test]
    fn test_fast_rsqrt_edge_cases() {
        assert_eq!(fast_rsqrt(0.0), 0.0);
        assert_eq!(fast_rsqrt(-1.0), 0.0);
        assert!(fast_rsqrt(f32::MAX).is_finite());
        assert!(fast_rsqrt(f32::MIN_POSITIVE).is_finite());
    }
}
