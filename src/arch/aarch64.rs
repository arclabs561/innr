//! aarch64 SIMD implementations using NEON.
//!
//! NEON is always available on aarch64, so no runtime detection needed.
//! However, we still use target_feature for consistency with x86_64.

/// NEON dot product implementation.
///
/// # Safety
///
/// NEON is always available on aarch64, but we use `target_feature`
/// annotation for consistency and potential future optimizations.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{float32x4_t, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum: float32x4_t = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // Process 4 floats at a time using NEON
    // SAFETY: `vld1q_f32` is unaligned load, no alignment required.
    // offset = i*4 < chunks*4 <= n, so within bounds.
    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        // FMA: sum = va * vb + sum
        sum = vfmaq_f32(sum, va, vb);
    }

    // Horizontal reduction: reduce 4 f32s to 1
    // vaddvq_f32 adds all 4 lanes
    let mut result = vaddvq_f32(sum);

    // Handle remainder with scalar ops
    let tail_start = chunks * 4;
    for i in 0..remainder {
        // SAFETY: tail_start + i < n, so within bounds
        result += *a.get_unchecked(tail_start + i) * *b.get_unchecked(tail_start + i);
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
