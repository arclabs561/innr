#![allow(unsafe_code)]
//! aarch64 SIMD implementations using NEON.
//!
//! NEON is always available on aarch64, so no runtime detection needed.
//! However, we still use target_feature for consistency with x86_64.
//!
//! # Performance Hierarchy
//!
//! | Implementation | Width | Throughput |
//! |----------------|-------|------------|
//! | dot_neon (4-way unrolled) | 4x4 f32 | ~4-8x vs scalar |
//! | Basic NEON | 4 f32 | ~2-4x vs scalar |

/// NEON dot product with 4-way unrolling.
///
/// Processes 16 floats per iteration (4 x 4), hiding memory latency.
///
/// # Safety
///
/// NEON is always available on aarch64, but we use `target_feature`
/// annotation for consistency and potential future optimizations.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{
        float32x4_t, vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32,
    };

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // 4-way unrolled: process 16 floats per iteration
    let chunks_16 = n / 16;
    let mut sum0: float32x4_t = vdupq_n_f32(0.0);
    let mut sum1: float32x4_t = vdupq_n_f32(0.0);
    let mut sum2: float32x4_t = vdupq_n_f32(0.0);
    let mut sum3: float32x4_t = vdupq_n_f32(0.0);

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

        sum0 = vfmaq_f32(sum0, va0, vb0);
        sum1 = vfmaq_f32(sum1, va1, vb1);
        sum2 = vfmaq_f32(sum2, va2, vb2);
        sum3 = vfmaq_f32(sum3, va3, vb3);
    }

    // Combine accumulators
    let sum01 = vaddq_f32(sum0, sum1);
    let sum23 = vaddq_f32(sum2, sum3);
    let sum_all = vaddq_f32(sum01, sum23);
    let mut result = vaddvq_f32(sum_all);

    // Handle remaining 4-float chunks
    let remaining_start = chunks_16 * 16;
    let remaining = n - remaining_start;
    let chunks_4 = remaining / 4;

    let mut sum: float32x4_t = vdupq_n_f32(0.0);
    for i in 0..chunks_4 {
        let offset = remaining_start + i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        sum = vfmaq_f32(sum, va, vb);
    }
    result += vaddvq_f32(sum);

    // Scalar tail
    let tail_start = remaining_start + chunks_4 * 4;
    for i in tail_start..n {
        // SAFETY: i is in tail_start..n where n = a.len().min(b.len()),
        // so i is always a valid index into both a and b.
        result += *a.get_unchecked(i) * *b.get_unchecked(i);
    }

    result
}

/// NEON MaxSim implementation.
///
/// Iterates over query tokens and computes max similarity against all doc tokens
/// using the unsafe dot_neon kernel directly.
///
/// # Safety
///
/// - NEON is always available on aarch64.
/// - `doc_tokens` must be non-empty; an empty slice causes `NEG_INFINITY` accumulation.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn maxsim_neon(query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
    let mut total_score = 0.0;

    for q in query_tokens {
        let mut max_score = f32::NEG_INFINITY;
        for d in doc_tokens {
            let score = dot_neon(q, d);
            if score > max_score {
                max_score = score;
            }
        }
        total_score += max_score;
    }

    total_score
}

/// NEON squared L2 distance: `Σ(a[i] - b[i])²`.
///
/// Single-pass: computes differences directly, avoiding catastrophic
/// cancellation from the expansion `||a||² + ||b||² - 2<a,b>`.
///
/// # Safety
///
/// NEON is always available on aarch64, but we use `target_feature`
/// annotation for consistency and potential future optimizations.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn l2_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{
        float32x4_t, vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vsubq_f32,
    };

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // 4-way unrolled: process 16 floats per iteration
    let chunks_16 = n / 16;
    let mut sum0: float32x4_t = vdupq_n_f32(0.0);
    let mut sum1: float32x4_t = vdupq_n_f32(0.0);
    let mut sum2: float32x4_t = vdupq_n_f32(0.0);
    let mut sum3: float32x4_t = vdupq_n_f32(0.0);

    for i in 0..chunks_16 {
        let base = i * 16;
        let va0 = vld1q_f32(a_ptr.add(base));
        let vb0 = vld1q_f32(b_ptr.add(base));
        let d0 = vsubq_f32(va0, vb0);
        let va1 = vld1q_f32(a_ptr.add(base + 4));
        let vb1 = vld1q_f32(b_ptr.add(base + 4));
        let d1 = vsubq_f32(va1, vb1);
        let va2 = vld1q_f32(a_ptr.add(base + 8));
        let vb2 = vld1q_f32(b_ptr.add(base + 8));
        let d2 = vsubq_f32(va2, vb2);
        let va3 = vld1q_f32(a_ptr.add(base + 12));
        let vb3 = vld1q_f32(b_ptr.add(base + 12));
        let d3 = vsubq_f32(va3, vb3);

        sum0 = vfmaq_f32(sum0, d0, d0);
        sum1 = vfmaq_f32(sum1, d1, d1);
        sum2 = vfmaq_f32(sum2, d2, d2);
        sum3 = vfmaq_f32(sum3, d3, d3);
    }

    // Combine accumulators
    let sum01 = vaddq_f32(sum0, sum1);
    let sum23 = vaddq_f32(sum2, sum3);
    let sum_all = vaddq_f32(sum01, sum23);
    let mut result = vaddvq_f32(sum_all);

    // Handle remaining 4-float chunks
    let remaining_start = chunks_16 * 16;
    let remaining = n - remaining_start;
    let chunks_4 = remaining / 4;

    let mut sum: float32x4_t = vdupq_n_f32(0.0);
    for i in 0..chunks_4 {
        let offset = remaining_start + i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        let d = vsubq_f32(va, vb);
        sum = vfmaq_f32(sum, d, d);
    }
    result += vaddvq_f32(sum);

    // Scalar tail
    let tail_start = remaining_start + chunks_4 * 4;
    for i in tail_start..n {
        // SAFETY: i is in tail_start..n where n = a.len().min(b.len()),
        // so i is always a valid index into both a and b.
        let d = *a.get_unchecked(i) - *b.get_unchecked(i);
        result += d * d;
    }

    result
}

/// NEON L1 (Manhattan) distance: `Σ|a[i] - b[i]|`.
///
/// Uses `vabdq_f32` (absolute difference) for fused sub+abs in one instruction.
/// 4-way unrolled for pipeline efficiency.
///
/// # Safety
///
/// NEON is always available on aarch64.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn l1_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{
        float32x4_t, vabdq_f32, vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32,
    };

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // 4-way unrolled: process 16 floats per iteration
    let chunks_16 = n / 16;
    let mut sum0: float32x4_t = vdupq_n_f32(0.0);
    let mut sum1: float32x4_t = vdupq_n_f32(0.0);
    let mut sum2: float32x4_t = vdupq_n_f32(0.0);
    let mut sum3: float32x4_t = vdupq_n_f32(0.0);

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

        // vabdq_f32: absolute difference in one instruction
        sum0 = vaddq_f32(sum0, vabdq_f32(va0, vb0));
        sum1 = vaddq_f32(sum1, vabdq_f32(va1, vb1));
        sum2 = vaddq_f32(sum2, vabdq_f32(va2, vb2));
        sum3 = vaddq_f32(sum3, vabdq_f32(va3, vb3));
    }

    // Combine accumulators
    let sum01 = vaddq_f32(sum0, sum1);
    let sum23 = vaddq_f32(sum2, sum3);
    let sum_all = vaddq_f32(sum01, sum23);
    let mut result = vaddvq_f32(sum_all);

    // Handle remaining 4-float chunks
    let remaining_start = chunks_16 * 16;
    let remaining = n - remaining_start;
    let chunks_4 = remaining / 4;

    let mut sum: float32x4_t = vdupq_n_f32(0.0);
    for i in 0..chunks_4 {
        let offset = remaining_start + i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        sum = vaddq_f32(sum, vabdq_f32(va, vb));
    }
    result += vaddvq_f32(sum);

    // Scalar tail
    let tail_start = remaining_start + chunks_4 * 4;
    for i in tail_start..n {
        // SAFETY: i is in tail_start..n where n = a.len().min(b.len()),
        // so i is always a valid index into both a and b.
        result += (*a.get_unchecked(i) - *b.get_unchecked(i)).abs();
    }

    result
}

/// NEON fused cosine similarity: single-pass dot(a,b), norm(a)^2, norm(b)^2.
///
/// Accumulates all three products in one pass over memory, then uses
/// exact sqrt/div for IEEE-correct normalization. ~3x less memory bandwidth
/// than the 3-pass approach (dot + norm + norm).
///
/// # Safety
///
/// NEON is always available on aarch64.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn cosine_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{
        float32x4_t, vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32,
    };

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // 4-way unrolled: process 16 floats per iteration, 3 accumulators each
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
        // SAFETY: i is in tail_start..n where n = a.len().min(b.len()),
        // so i is always a valid index into both a and b.
        let ai = *a.get_unchecked(i);
        let bi = *b.get_unchecked(i);
        ab += ai * bi;
        aa += ai * ai;
        bb += bi * bi;
    }

    // Exact normalization (IEEE sqrt + div)
    if aa > crate::NORM_EPSILON_SQ && bb > crate::NORM_EPSILON_SQ {
        ab / (aa.sqrt() * bb.sqrt())
    } else {
        0.0
    }
}

/// NEON mixed-precision dot product: `sum(a_f32[i] * b_u8[i] as f32)`.
///
/// Widens u8 -> u16 -> u32 -> f32, then FMA with query f32 values.
/// Processes 16 u8 elements per iteration (4 FMA operations).
///
/// # Safety
///
/// NEON is always available on aarch64.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot_u8_f32_neon(a: &[f32], b: &[u8]) -> f32 {
    use std::arch::aarch64::{
        float32x4_t, vaddq_f32, vaddvq_f32, vcvtq_f32_u32, vdupq_n_f32, vfmaq_f32, vget_high_u16,
        vget_high_u8, vget_low_u16, vget_low_u8, vld1q_f32, vld1q_u8, vmovl_u16, vmovl_u8,
    };

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // Process 16 elements per iteration (16 u8 -> 4x4 f32)
    let chunks_16 = n / 16;
    let mut sum0: float32x4_t = vdupq_n_f32(0.0);
    let mut sum1: float32x4_t = vdupq_n_f32(0.0);
    let mut sum2: float32x4_t = vdupq_n_f32(0.0);
    let mut sum3: float32x4_t = vdupq_n_f32(0.0);

    for i in 0..chunks_16 {
        let base = i * 16;

        // Load 16 x u8
        let vb = vld1q_u8(b_ptr.add(base));

        // Widen: u8x16 -> u16x8 (lo/hi halves)
        let b_lo_u16 = vmovl_u8(vget_low_u8(vb));
        let b_hi_u16 = vmovl_u8(vget_high_u8(vb));

        // Widen u16x4 -> u32x4 -> f32x4 (4 groups of 4)
        let b0_f32 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_lo_u16)));
        let b1_f32 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_lo_u16)));
        let b2_f32 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_hi_u16)));
        let b3_f32 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_hi_u16)));

        // Load 16 x f32 query values (4 loads of 4)
        let a0 = vld1q_f32(a_ptr.add(base));
        let a1 = vld1q_f32(a_ptr.add(base + 4));
        let a2 = vld1q_f32(a_ptr.add(base + 8));
        let a3 = vld1q_f32(a_ptr.add(base + 12));

        // FMA: sum += a * b_f32
        sum0 = vfmaq_f32(sum0, a0, b0_f32);
        sum1 = vfmaq_f32(sum1, a1, b1_f32);
        sum2 = vfmaq_f32(sum2, a2, b2_f32);
        sum3 = vfmaq_f32(sum3, a3, b3_f32);
    }

    // Combine accumulators
    let sum01 = vaddq_f32(sum0, sum1);
    let sum23 = vaddq_f32(sum2, sum3);
    let sum_all = vaddq_f32(sum01, sum23);
    let mut result = vaddvq_f32(sum_all);

    // Scalar tail
    let tail_start = chunks_16 * 16;
    for i in tail_start..n {
        // SAFETY: i is in tail_start..n where n = a.len().min(b.len()),
        // so i is always a valid index into both a and b.
        result += *a.get_unchecked(i) * (*b.get_unchecked(i) as f32);
    }

    result
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dot_neon_correctness() {
        use super::*;

        // Test various sizes around SIMD boundaries
        for size in [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 128, 256] {
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.2).collect();

            let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let actual = unsafe { dot_neon(&a, &b) };

            let rel_error = if expected.abs() > 1e-6 {
                (actual - expected).abs() / expected.abs()
            } else {
                (actual - expected).abs()
            };
            assert!(
                rel_error < 1e-5,
                "size={}: expected={}, actual={}, rel_error={}",
                size,
                expected,
                actual,
                rel_error
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_l1_neon_correctness() {
        use super::*;

        for size in [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 128, 256] {
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.2).collect();

            let expected: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
            let actual = unsafe { l1_neon(&a, &b) };

            let rel_error = if expected.abs() > 1e-6 {
                (actual - expected).abs() / expected.abs()
            } else {
                (actual - expected).abs()
            };
            assert!(
                rel_error < 1e-5,
                "L1 size={}: expected={}, actual={}, rel_error={}",
                size,
                expected,
                actual,
                rel_error
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_cosine_neon_correctness() {
        use super::*;

        for size in [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 128, 256] {
            let a: Vec<f32> = (0..size).map(|i| ((i * 7) as f32).sin()).collect();
            let b: Vec<f32> = (0..size).map(|i| ((i * 11) as f32).cos()).collect();

            // Reference: scalar cosine
            let ab: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let aa: f32 = a.iter().map(|x| x * x).sum();
            let bb: f32 = b.iter().map(|x| x * x).sum();
            let expected = if aa > 1e-9 && bb > 1e-9 {
                ab / (aa.sqrt() * bb.sqrt())
            } else {
                0.0
            };

            let actual = unsafe { cosine_neon(&a, &b) };

            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-5,
                "size={}: expected={}, actual={}, diff={}",
                size,
                expected,
                actual,
                diff
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_l2_squared_neon_correctness() {
        use super::*;

        for size in [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 128, 256] {
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.2).collect();

            let expected: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
            let actual = unsafe { l2_squared_neon(&a, &b) };

            let rel_error = if expected.abs() > 1e-6 {
                (actual - expected).abs() / expected.abs()
            } else {
                (actual - expected).abs()
            };
            assert!(
                rel_error < 1e-5,
                "size={}: expected={}, actual={}, rel_error={}",
                size,
                expected,
                actual,
                rel_error
            );
        }
    }
}
