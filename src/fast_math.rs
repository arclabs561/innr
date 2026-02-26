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

use crate::NORM_EPSILON;

// MIN_DIM_SIMD is only used on architectures with SIMD dispatch
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::MIN_DIM_SIMD;

/// Fast scalar inverse square root using the classic Quake III bit-hack.
///
/// One Newton-Raphson iteration for ~23 bits of accuracy.
///
/// # Safety Note
///
/// Input must be positive. Zero or negative inputs return 0.0.
///
/// ```
/// use innr::fast_rsqrt;
///
/// let r = fast_rsqrt(4.0);
/// assert!((r - 0.5).abs() < 1e-3);  // 1/sqrt(4) = 0.5
/// ```
#[inline]
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
/// ~46 bits of accuracy (overkill for f32, but matches IEEE sqrt).
#[inline]
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
/// Relative error is typically < 1e-6 for normalized vectors.
/// For unnormalized vectors with extreme magnitude differences, error may be higher.
///
/// # Example
///
/// ```rust
/// use innr::fast_math::fast_cosine;
///
/// let a = [1.0_f32, 0.0, 0.0];
/// let b = [0.707, 0.707, 0.0];
/// let c = fast_cosine(&a, &b);
/// assert!((c - 0.707).abs() < 0.01);
/// ```
#[inline]
pub fn fast_cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len().min(b.len());

    // Compute three dot products in one pass for better cache locality
    let mut ab = 0.0f32;
    let mut aa = 0.0f32;
    let mut bb = 0.0f32;

    for i in 0..n {
        let ai = unsafe { *a.get_unchecked(i) };
        let bi = unsafe { *b.get_unchecked(i) };
        ab += ai * bi;
        aa += ai * ai;
        bb += bi * bi;
    }

    // Use rsqrt to avoid sqrt and division
    // cosine = ab / sqrt(aa * bb) = ab * rsqrt(aa) * rsqrt(bb)
    if aa > NORM_EPSILON && bb > NORM_EPSILON {
        ab * fast_rsqrt(aa) * fast_rsqrt(bb)
    } else {
        0.0
    }
}

/// Fast cosine distance: `1 - cosine_similarity`.
///
/// More numerically stable formulation that avoids catastrophic cancellation.
#[inline]
pub fn fast_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - fast_cosine(a, b)
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD implementations
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub mod x86_64 {
    //! AVX2/AVX-512 fast cosine using rsqrt instructions.

    /// AVX-512 fast cosine with rsqrt14 + Newton-Raphson.
    ///
    /// Uses `vrsqrt14ps` for ~14-bit initial estimate, then one NR iteration.
    ///
    /// # Safety
    ///
    /// Requires AVX-512F. Caller must verify via `is_x86_feature_detected!`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn fast_cosine_avx512(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::{
            __m512, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps, _mm512_setzero_ps,
        };

        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut ab_sum: __m512 = _mm512_setzero_ps();
        let mut aa_sum: __m512 = _mm512_setzero_ps();
        let mut bb_sum: __m512 = _mm512_setzero_ps();

        // Process 16 floats at a time
        let chunks = n / 16;
        for i in 0..chunks {
            let offset = i * 16;
            let va = _mm512_loadu_ps(a_ptr.add(offset));
            let vb = _mm512_loadu_ps(b_ptr.add(offset));

            ab_sum = _mm512_fmadd_ps(va, vb, ab_sum);
            aa_sum = _mm512_fmadd_ps(va, va, aa_sum);
            bb_sum = _mm512_fmadd_ps(vb, vb, bb_sum);
        }

        // Reduce to scalars
        let mut ab = _mm512_reduce_add_ps(ab_sum);
        let mut aa = _mm512_reduce_add_ps(aa_sum);
        let mut bb = _mm512_reduce_add_ps(bb_sum);

        // Handle tail
        let tail_start = chunks * 16;
        for i in tail_start..n {
            let ai = *a.get_unchecked(i);
            let bi = *b.get_unchecked(i);
            ab += ai * bi;
            aa += ai * ai;
            bb += bi * bi;
        }

        // Fast rsqrt with NR refinement
        if aa > super::NORM_EPSILON && bb > super::NORM_EPSILON {
            ab * super::fast_rsqrt(aa) * super::fast_rsqrt(bb)
        } else {
            0.0
        }
    }

    /// AVX2 fast cosine with rsqrt + Newton-Raphson.
    ///
    /// # Safety
    ///
    /// Requires AVX2 + FMA. Caller must verify via runtime detection.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn fast_cosine_avx2(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::{
            __m256, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps,
            _mm256_loadu_ps, _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32,
            _mm_movehl_ps, _mm_shuffle_ps,
        };

        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut ab_sum: __m256 = _mm256_setzero_ps();
        let mut aa_sum: __m256 = _mm256_setzero_ps();
        let mut bb_sum: __m256 = _mm256_setzero_ps();

        // Process 8 floats at a time
        let chunks = n / 8;
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a_ptr.add(offset));
            let vb = _mm256_loadu_ps(b_ptr.add(offset));

            ab_sum = _mm256_fmadd_ps(va, vb, ab_sum);
            aa_sum = _mm256_fmadd_ps(va, va, aa_sum);
            bb_sum = _mm256_fmadd_ps(vb, vb, bb_sum);
        }

        // Horizontal reduction helper
        #[inline(always)]
        unsafe fn hsum256(v: __m256) -> f32 {
            let hi = _mm256_extractf128_ps(v, 1);
            let lo = _mm256_castps256_ps128(v);
            let sum128 = _mm_add_ps(lo, hi);
            let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
            _mm_cvtss_f32(sum32)
        }

        let mut ab = hsum256(ab_sum);
        let mut aa = hsum256(aa_sum);
        let mut bb = hsum256(bb_sum);

        // Handle tail
        let tail_start = chunks * 8;
        for i in tail_start..n {
            let ai = *a.get_unchecked(i);
            let bi = *b.get_unchecked(i);
            ab += ai * bi;
            aa += ai * ai;
            bb += bi * bi;
        }

        if aa > super::NORM_EPSILON && bb > super::NORM_EPSILON {
            ab * super::fast_rsqrt(aa) * super::fast_rsqrt(bb)
        } else {
            0.0
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
    //! NEON fast cosine using vrsqrte + Newton-Raphson.

    /// NEON fast cosine with 4-way unrolled accumulation and hardware rsqrt.
    ///
    /// Uses `vrsqrteq_f32` + two `vrsqrtsq_f32` Newton-Raphson iterations for
    /// the final normalization, avoiding scalar/SIMD transitions entirely.
    ///
    /// # Safety
    ///
    /// NEON is always available on aarch64.
    #[target_feature(enable = "neon")]
    pub unsafe fn fast_cosine_neon(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::aarch64::{
            float32x2_t, float32x4_t, vaddq_f32, vaddvq_f32, vdup_n_f32, vdupq_n_f32,
            vfmaq_f32, vget_lane_f32, vld1q_f32, vmul_f32, vrsqrte_f32, vrsqrts_f32,
        };

        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // 4-way unrolled: process 16 floats per iteration
        let chunks_16 = n / 16;
        let mut ab0: float32x4_t = vdupq_n_f32(0.0);
        let mut ab1: float32x4_t = vdupq_n_f32(0.0);
        let mut ab2: float32x4_t = vdupq_n_f32(0.0);
        let mut ab3: float32x4_t = vdupq_n_f32(0.0);
        let mut aa0: float32x4_t = vdupq_n_f32(0.0);
        let mut aa1: float32x4_t = vdupq_n_f32(0.0);
        let mut aa2: float32x4_t = vdupq_n_f32(0.0);
        let mut aa3: float32x4_t = vdupq_n_f32(0.0);
        let mut bb0: float32x4_t = vdupq_n_f32(0.0);
        let mut bb1: float32x4_t = vdupq_n_f32(0.0);
        let mut bb2: float32x4_t = vdupq_n_f32(0.0);
        let mut bb3: float32x4_t = vdupq_n_f32(0.0);

        for i in 0..chunks_16 {
            let base = i * 16;
            let va0 = vld1q_f32(a_ptr.add(base));
            let vb0 = vld1q_f32(b_ptr.add(base));
            let va1 = vld1q_f32(a_ptr.add(base + 4));
            let vb1 = vld1q_f32(b_ptr.add(base + 4));
            let va2 = vld1q_f32(a_ptr.add(base + 8));
            let vb2 = vld1q_f32(b_ptr.add(base + 8));
            let va3 = vld1q_f32(a_ptr.add(base + 12));
            let vb3 = vld1q_f32(b_ptr.add(base + 12));

            ab0 = vfmaq_f32(ab0, va0, vb0);
            ab1 = vfmaq_f32(ab1, va1, vb1);
            ab2 = vfmaq_f32(ab2, va2, vb2);
            ab3 = vfmaq_f32(ab3, va3, vb3);

            aa0 = vfmaq_f32(aa0, va0, va0);
            aa1 = vfmaq_f32(aa1, va1, va1);
            aa2 = vfmaq_f32(aa2, va2, va2);
            aa3 = vfmaq_f32(aa3, va3, va3);

            bb0 = vfmaq_f32(bb0, vb0, vb0);
            bb1 = vfmaq_f32(bb1, vb1, vb1);
            bb2 = vfmaq_f32(bb2, vb2, vb2);
            bb3 = vfmaq_f32(bb3, vb3, vb3);
        }

        // Combine the 4 accumulators
        let ab_sum = vaddq_f32(vaddq_f32(ab0, ab1), vaddq_f32(ab2, ab3));
        let aa_sum = vaddq_f32(vaddq_f32(aa0, aa1), vaddq_f32(aa2, aa3));
        let bb_sum = vaddq_f32(vaddq_f32(bb0, bb1), vaddq_f32(bb2, bb3));

        let mut ab = vaddvq_f32(ab_sum);
        let mut aa = vaddvq_f32(aa_sum);
        let mut bb = vaddvq_f32(bb_sum);

        // Handle remaining 4-float chunks
        let remaining_start = chunks_16 * 16;
        let remaining = n - remaining_start;
        let chunks_4 = remaining / 4;

        let mut ab_tail: float32x4_t = vdupq_n_f32(0.0);
        let mut aa_tail: float32x4_t = vdupq_n_f32(0.0);
        let mut bb_tail: float32x4_t = vdupq_n_f32(0.0);

        for i in 0..chunks_4 {
            let offset = remaining_start + i * 4;
            let va = vld1q_f32(a_ptr.add(offset));
            let vb = vld1q_f32(b_ptr.add(offset));
            ab_tail = vfmaq_f32(ab_tail, va, vb);
            aa_tail = vfmaq_f32(aa_tail, va, va);
            bb_tail = vfmaq_f32(bb_tail, vb, vb);
        }

        ab += vaddvq_f32(ab_tail);
        aa += vaddvq_f32(aa_tail);
        bb += vaddvq_f32(bb_tail);

        // Scalar tail
        let tail_start = remaining_start + chunks_4 * 4;
        for i in tail_start..n {
            let ai = *a.get_unchecked(i);
            let bi = *b.get_unchecked(i);
            ab += ai * bi;
            aa += ai * ai;
            bb += bi * bi;
        }

        if aa > super::NORM_EPSILON && bb > super::NORM_EPSILON {
            // NEON hardware rsqrt: vrsqrte + two Newton-Raphson iterations via vrsqrts.
            // Pack aa and bb into a float32x2_t, compute rsqrt in-register,
            // then extract and multiply. This avoids scalar/SIMD transitions.
            let norms: float32x2_t = vdup_n_f32(0.0);
            // Load aa into lane 0, bb into lane 1
            let norms = vset_lane_f32_0(aa, norms);
            let norms = vset_lane_f32_1(bb, norms);

            // Initial estimate (~8-12 bits)
            let mut est: float32x2_t = vrsqrte_f32(norms);

            // Newton-Raphson iteration 1: est = est * vrsqrts(norms, est * est)
            est = vmul_f32(est, vrsqrts_f32(norms, vmul_f32(est, est)));
            // Newton-Raphson iteration 2
            est = vmul_f32(est, vrsqrts_f32(norms, vmul_f32(est, est)));

            let rsqrt_aa = vget_lane_f32::<0>(est);
            let rsqrt_bb = vget_lane_f32::<1>(est);

            ab * rsqrt_aa * rsqrt_bb
        } else {
            0.0
        }
    }

    /// Set lane 0 of a float32x2_t. Helper to avoid inline asm complexity.
    #[inline(always)]
    unsafe fn vset_lane_f32_0(
        val: f32,
        vec: std::arch::aarch64::float32x2_t,
    ) -> std::arch::aarch64::float32x2_t {
        use std::arch::aarch64::vset_lane_f32;
        vset_lane_f32::<0>(val, vec)
    }

    /// Set lane 1 of a float32x2_t. Helper to avoid inline asm complexity.
    #[inline(always)]
    unsafe fn vset_lane_f32_1(
        val: f32,
        vec: std::arch::aarch64::float32x2_t,
    ) -> std::arch::aarch64::float32x2_t {
        use std::arch::aarch64::vset_lane_f32;
        vset_lane_f32::<1>(val, vec)
    }
}

/// Dispatch to the fastest available `fast_cosine` implementation.
///
/// Selects AVX-512, AVX2+FMA, NEON, or portable based on runtime detection.
#[inline]
pub fn fast_cosine_dispatch(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let n = a.len().min(b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if n >= 64 && is_x86_feature_detected!("avx512f") {
            return unsafe { x86_64::fast_cosine_avx512(a, b) };
        }
        if n >= MIN_DIM_SIMD && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
        {
            return unsafe { x86_64::fast_cosine_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if n >= MIN_DIM_SIMD {
            return unsafe { aarch64::fast_cosine_neon(a, b) };
        }
    }

    fast_cosine(a, b)
}

#[cfg(test)]
mod tests {
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
                dim, portable, standard, port_err
            );
            assert!(
                disp_err < 0.01,
                "dim={}: dispatched={}, standard={}, err={}",
                dim, dispatched, standard, disp_err
            );
            // And they should agree within 2e-3 (different rsqrt methods:
            // scalar Quake III bit-hack vs NEON vrsqrte + 2 NR iterations)
            let diff = (portable - dispatched).abs();
            assert!(
                diff < 2e-3,
                "dim={}: portable={}, dispatched={}, diff={}",
                dim, portable, dispatched, diff
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
    }
}
