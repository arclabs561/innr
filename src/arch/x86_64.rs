//! x86_64 SIMD implementations using AVX2 and FMA.
//!
//! These functions are unsafe and require runtime feature detection
//! before calling. The safe public API handles this.

/// AVX2+FMA dot product implementation.
///
/// # Safety
///
/// Caller must verify `is_x86_feature_detected!("avx2")` and
/// `is_x86_feature_detected!("fma")` before calling.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        __m256, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps,
    };

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let chunks = n / 8;
    let remainder = n % 8;

    let mut sum: __m256 = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // Process 8 floats at a time using AVX2
    // SAFETY: `_mm256_loadu_ps` is unaligned load, no alignment required.
    // offset = i*8 < chunks*8 <= n, so within bounds.
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        // FMA: sum = va * vb + sum (fused multiply-add, single rounding)
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal reduction: reduce 8 f32s to 1
    // Split 256-bit into two 128-bit halves
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    // Add the two halves: 4 f32s
    let sum128 = _mm_add_ps(lo, hi);
    // Add high 64 bits to low 64 bits: 2 f32s
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    // Add the two remaining f32s
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);

    // Handle remainder with scalar ops
    let tail_start = chunks * 8;
    for i in 0..remainder {
        // SAFETY: tail_start + i < n, so within bounds
        result += *a.get_unchecked(tail_start + i) * *b.get_unchecked(tail_start + i);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_dot_avx2_correctness() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("AVX2+FMA not available, skipping test");
            return;
        }

        // Test various sizes around SIMD boundaries
        for size in [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 128, 256] {
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.2).collect();

            let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let actual = unsafe { dot_avx2(&a, &b) };

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
