//! x86_64 SIMD implementations using AVX2/AVX-512 and FMA.
//!
//! These functions are unsafe and require runtime feature detection
//! before calling. The safe public API handles this.
//!
//! # Performance Hierarchy
//!
//! | ISA | Width | Coverage | Speedup vs scalar |
//! |-----|-------|----------|-------------------|
//! | AVX-512 | 16 f32 | ~9% | 10-20x |
//! | AVX2+FMA | 8 f32 | ~89% | 5-10x |
//! | SSE2 | 4 f32 | ~100% | 2-4x |

/// AVX-512 dot product with 4-way unrolling.
///
/// Processes 64 floats per iteration (4 x 16), hiding memory latency.
///
/// # Safety
///
/// Caller must verify `is_x86_feature_detected!("avx512f")` before calling.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        __m512, _mm512_add_ps, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps,
        _mm512_setzero_ps,
    };

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // 4-way unrolled: process 64 floats per iteration
    let chunks_64 = n / 64;
    let mut sum0: __m512 = _mm512_setzero_ps();
    let mut sum1: __m512 = _mm512_setzero_ps();
    let mut sum2: __m512 = _mm512_setzero_ps();
    let mut sum3: __m512 = _mm512_setzero_ps();

    for i in 0..chunks_64 {
        let base = i * 64;
        let va0 = _mm512_loadu_ps(a_ptr.add(base));
        let vb0 = _mm512_loadu_ps(b_ptr.add(base));
        let va1 = _mm512_loadu_ps(a_ptr.add(base + 16));
        let vb1 = _mm512_loadu_ps(b_ptr.add(base + 16));
        let va2 = _mm512_loadu_ps(a_ptr.add(base + 32));
        let vb2 = _mm512_loadu_ps(b_ptr.add(base + 32));
        let va3 = _mm512_loadu_ps(a_ptr.add(base + 48));
        let vb3 = _mm512_loadu_ps(b_ptr.add(base + 48));

        sum0 = _mm512_fmadd_ps(va0, vb0, sum0);
        sum1 = _mm512_fmadd_ps(va1, vb1, sum1);
        sum2 = _mm512_fmadd_ps(va2, vb2, sum2);
        sum3 = _mm512_fmadd_ps(va3, vb3, sum3);
    }

    // Combine accumulators
    let sum01 = _mm512_add_ps(sum0, sum1);
    let sum23 = _mm512_add_ps(sum2, sum3);
    let sum_all = _mm512_add_ps(sum01, sum23);
    let mut result = _mm512_reduce_add_ps(sum_all);

    // Handle remaining 16-float chunks
    let remaining_start = chunks_64 * 64;
    let remaining = n - remaining_start;
    let chunks_16 = remaining / 16;

    for i in 0..chunks_16 {
        let offset = remaining_start + i * 16;
        let va = _mm512_loadu_ps(a_ptr.add(offset));
        let vb = _mm512_loadu_ps(b_ptr.add(offset));
        result += _mm512_reduce_add_ps(_mm512_fmadd_ps(va, vb, _mm512_setzero_ps()));
    }

    // Scalar tail
    let tail_start = remaining_start + chunks_16 * 16;
    for i in tail_start..n {
        result += *a.get_unchecked(i) * *b.get_unchecked(i);
    }

    result
}

/// AVX2+FMA dot product with 4-way unrolling.
///
/// Processes 32 floats per iteration (4 x 8), hiding memory latency.
///
/// # Safety
///
/// Caller must verify `is_x86_feature_detected!("avx2")` and
/// `is_x86_feature_detected!("fma")` before calling.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps,
        _mm256_loadu_ps, _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehl_ps,
        _mm_shuffle_ps,
    };

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // 4-way unrolled: process 32 floats per iteration
    let chunks_32 = n / 32;
    let mut sum0: __m256 = _mm256_setzero_ps();
    let mut sum1: __m256 = _mm256_setzero_ps();
    let mut sum2: __m256 = _mm256_setzero_ps();
    let mut sum3: __m256 = _mm256_setzero_ps();

    for i in 0..chunks_32 {
        let base = i * 32;
        let va0 = _mm256_loadu_ps(a_ptr.add(base));
        let vb0 = _mm256_loadu_ps(b_ptr.add(base));
        let va1 = _mm256_loadu_ps(a_ptr.add(base + 8));
        let vb1 = _mm256_loadu_ps(b_ptr.add(base + 8));
        let va2 = _mm256_loadu_ps(a_ptr.add(base + 16));
        let vb2 = _mm256_loadu_ps(b_ptr.add(base + 16));
        let va3 = _mm256_loadu_ps(a_ptr.add(base + 24));
        let vb3 = _mm256_loadu_ps(b_ptr.add(base + 24));

        sum0 = _mm256_fmadd_ps(va0, vb0, sum0);
        sum1 = _mm256_fmadd_ps(va1, vb1, sum1);
        sum2 = _mm256_fmadd_ps(va2, vb2, sum2);
        sum3 = _mm256_fmadd_ps(va3, vb3, sum3);
    }

    // Combine accumulators
    let sum01 = _mm256_add_ps(sum0, sum1);
    let sum23 = _mm256_add_ps(sum2, sum3);
    let sum_all = _mm256_add_ps(sum01, sum23);

    // Horizontal reduction
    let hi = _mm256_extractf128_ps(sum_all, 1);
    let lo = _mm256_castps256_ps128(sum_all);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);

    // Handle remaining 8-float chunks
    let remaining_start = chunks_32 * 32;
    let remaining = n - remaining_start;
    let chunks_8 = remaining / 8;

    let mut sum: __m256 = _mm256_setzero_ps();
    for i in 0..chunks_8 {
        let offset = remaining_start + i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Reduce remaining sum
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    result += _mm_cvtss_f32(sum32);

    // Scalar tail
    let tail_start = remaining_start + chunks_8 * 8;
    for i in tail_start..n {
        result += *a.get_unchecked(i) * *b.get_unchecked(i);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_dot_avx512_correctness() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("AVX-512F not available, skipping test");
            return;
        }

        // Test various sizes around AVX-512 boundaries (16, 64)
        for size in [1, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 256, 512, 1024] {
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.2).collect();

            let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let actual = unsafe { dot_avx512(&a, &b) };

            let rel_error = if expected.abs() > 1e-6 {
                (actual - expected).abs() / expected.abs()
            } else {
                (actual - expected).abs()
            };
            assert!(
                rel_error < 1e-4, // Slightly looser for larger accumulations
                "AVX-512 size={}: expected={}, actual={}, rel_error={}",
                size,
                expected,
                actual,
                rel_error
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_dot_avx2_correctness() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("AVX2+FMA not available, skipping test");
            return;
        }

        // Test various sizes around AVX2 boundaries (8, 32)
        for size in [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 128, 256, 512] {
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
                "AVX2 size={}: expected={}, actual={}, rel_error={}",
                size,
                expected,
                actual,
                rel_error
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_vs_avx512_consistency() {
        if !is_x86_feature_detected!("avx2")
            || !is_x86_feature_detected!("fma")
            || !is_x86_feature_detected!("avx512f")
        {
            eprintln!("Need both AVX2+FMA and AVX-512F, skipping");
            return;
        }

        // Both should produce nearly identical results
        for size in [64, 128, 256, 512, 1024] {
            let a: Vec<f32> = (0..size).map(|i| ((i * 7) as f32).sin()).collect();
            let b: Vec<f32> = (0..size).map(|i| ((i * 11) as f32).cos()).collect();

            let avx2_result = unsafe { dot_avx2(&a, &b) };
            let avx512_result = unsafe { dot_avx512(&a, &b) };

            let diff = (avx2_result - avx512_result).abs();
            let max_val = avx2_result.abs().max(avx512_result.abs()).max(1e-6);
            let rel_diff = diff / max_val;

            assert!(
                rel_diff < 1e-5,
                "AVX2 vs AVX-512 mismatch at size={}: avx2={}, avx512={}, rel_diff={}",
                size,
                avx2_result,
                avx512_result,
                rel_diff
            );
        }
    }
}
