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
        result += *a.get_unchecked(i) * *b.get_unchecked(i);
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
}
