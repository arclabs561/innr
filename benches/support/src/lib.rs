//! Shared pieces for innr's consumer-shape benchmarks.
//!
//! - [`config`]: one Criterion configuration for every consumer bench, with
//!   measurement time overridable from the environment so the same command
//!   runs a quick pass or a careful one.
//! - [`data`]: seeded data generators, so every variant sees identical inputs.
//! - [`rankops`]: rankops' no-innr fallback, which uses raw SIMD intrinsics
//!   and therefore cannot live in an innr bench target (`unsafe_code` is
//!   forbidden crate-wide).

use std::time::Duration;

use criterion::Criterion;

/// Criterion settings for the consumer benches.
///
/// `INNR_BENCH_WARMUP_MS` (default 300) and `INNR_BENCH_MEASURE_MS`
/// (default 1000) trade run time for tighter intervals. Defaults keep the
/// whole suite near 20 minutes on an M3 Max for both builds combined.
pub fn config() -> Criterion {
    let ms = |var: &str, default: u64| {
        std::env::var(var)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    Criterion::default()
        .warm_up_time(Duration::from_millis(ms("INNR_BENCH_WARMUP_MS", 300)))
        .measurement_time(Duration::from_millis(ms("INNR_BENCH_MEASURE_MS", 1000)))
        .sample_size(20)
        .noise_threshold(0.03)
        .without_plots()
}

/// Seeded data generators.
pub mod data {
    use rand::prelude::*;

    /// `n` uniform values in `[-1, 1)`.
    pub fn f32s(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| rng.random_range(-1.0f32..1.0)).collect()
    }

    /// `n` uniform values in `[-1, 1)`.
    pub fn f64s(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| rng.random_range(-1.0f64..1.0)).collect()
    }

    /// `n` random bytes.
    pub fn bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| rng.random()).collect()
    }

    /// `n` random u64 values drawn from a small alphabet, so roughly half of
    /// two signatures' slots agree (MinHash slots for similar sets).
    pub fn slots_u64(n: usize, seed: u64) -> Vec<u64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| rng.random_range(0..2u64)).collect()
    }

    /// Row-major `rows x dim` matrix of unit-norm rows.
    pub fn unit_rows(rows: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut m = f32s(rows * dim, seed);
        for row in m.chunks_exact_mut(dim) {
            let n = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            row.iter_mut().for_each(|x| *x /= n);
        }
        m
    }

    /// One unit-norm vector.
    pub fn unit_vec(dim: usize, seed: u64) -> Vec<f32> {
        unit_rows(1, dim, seed)
    }

    /// Sign bits of each `dim`-row, LSB-first, `dim.div_ceil(8)` bytes per row.
    pub fn sign_codes(rows: &[f32], dim: usize) -> Vec<u8> {
        let bpr = dim.div_ceil(8);
        let n = rows.len() / dim;
        let mut out = vec![0u8; n * bpr];
        for (r, row) in rows.chunks_exact(dim).enumerate() {
            for (j, &x) in row.iter().enumerate() {
                if x > 0.0 {
                    out[r * bpr + j / 8] |= 1 << (j % 8);
                }
            }
        }
        out
    }

    /// `n` token vectors of length `dim`, each a separate allocation (the
    /// `Vec<Vec<f32>>` shape rankops' batch API takes).
    pub fn tokens(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        unit_rows(n, dim, seed)
            .chunks_exact(dim)
            .map(<[f32]>::to_vec)
            .collect()
    }
}

/// rankops' no-innr fallback (built when its `rerank` feature is off).
///
/// Copied verbatim from rankops `src/rerank/simd.rs` at a979f3c (the
/// published crate's source), except that the `#[cfg(not(feature =
/// "rerank"))]` gates are dropped because this crate has no such feature.
#[allow(clippy::all, unused_unsafe, missing_docs)]
pub mod rankops {
    /// Minimum vector dimension for SIMD to be worthwhile.
    /// Source: src/rerank/simd.rs:40-44.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    const MIN_DIM_SIMD: usize = 16;

    /// Source: src/rerank/simd.rs:47-49.
    const NORM_EPSILON: f32 = 1e-9;

    /// Source: src/rerank/simd.rs:67-103.
    #[inline]
    #[must_use]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(
            a.len(),
            b.len(),
            "dot: dimension mismatch ({} vs {})",
            a.len(),
            b.len()
        );
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        let n = a.len().min(b.len());

        #[cfg(target_arch = "x86_64")]
        {
            // Try AVX-512 first (Zen 5+, Ice Lake+): 16 floats per operation
            // _mm512_extractf32x8_ps in dot_avx512 needs AVX-512DQ, not just F.
            if n >= MIN_DIM_SIMD
                && is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512dq")
            {
                return unsafe { dot_avx512(a, b) };
            }
            // Fallback to AVX2+FMA: 8 floats per operation
            if n >= MIN_DIM_SIMD
                && is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
            {
                return unsafe { dot_avx2(a, b) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if n >= MIN_DIM_SIMD {
                return unsafe { dot_neon(a, b) };
            }
        }
        #[allow(unreachable_code)]
        dot_portable(a, b)
    }

    /// Source: src/rerank/simd.rs:109-111.
    #[inline]
    #[must_use]
    pub fn dot_portable(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Source: src/rerank/simd.rs:116-118.
    #[inline]
    #[must_use]
    pub fn norm(v: &[f32]) -> f32 {
        dot(v, v).sqrt()
    }

    /// Source: src/rerank/simd.rs:123-132.
    #[inline]
    #[must_use]
    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let d = dot(a, b);
        let na = norm(a);
        let nb = norm(b);
        if na > NORM_EPSILON && nb > NORM_EPSILON {
            d / (na * nb)
        } else {
            0.0
        }
    }

    /// Source: src/rerank/simd.rs:137-150.
    #[inline]
    #[must_use]
    pub fn maxsim(query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
        if query_tokens.is_empty() || doc_tokens.is_empty() {
            return 0.0;
        }
        query_tokens
            .iter()
            .map(|q| {
                doc_tokens
                    .iter()
                    .map(|d| dot(q, d))
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .sum()
    }

    /// Source: src/rerank/simd.rs:3224-3226.
    pub fn as_slices(tokens: &[Vec<f32>]) -> Vec<&[f32]> {
        tokens.iter().map(Vec::as_slice).collect()
    }

    /// Source: src/rerank/simd.rs:544-552 (`maxsim_batch`).
    #[must_use]
    pub fn maxsim_batch(query: &[Vec<f32>], docs: &[Vec<Vec<f32>>]) -> Vec<f32> {
        let q = as_slices(query);
        docs.iter()
            .map(|doc| {
                let d = as_slices(doc);
                maxsim(&q, &d)
            })
            .collect()
    }

    /// Source: src/rerank/colbert.rs:774-790, with `simd::dot` resolving to
    /// the fallback [`dot`] (rankops built without `rerank`).
    pub fn maxsim_flat(query: &[f32], doc: &[f32], dim: usize) -> f32 {
        debug_assert!(dim > 0);
        debug_assert!(query.len().is_multiple_of(dim));
        debug_assert!(doc.len().is_multiple_of(dim));

        if query.is_empty() || doc.is_empty() {
            return 0.0;
        }

        query
            .chunks_exact(dim)
            .map(|q| {
                doc.chunks_exact(dim)
                    .map(|d| dot(q, d))
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .sum()
    }

    /// Source: src/rerank/simd.rs:1345-1387.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512dq")]
    unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::{
            __m256, __m512, _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps,
            _mm512_castps512_ps256, _mm512_extractf32x8_ps, _mm512_fmadd_ps, _mm512_loadu_ps,
            _mm512_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehl_ps,
            _mm_shuffle_ps,
        };

        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }

        let chunks = n / 16;
        let remainder = n % 16;
        let mut sum: __m512 = _mm512_setzero_ps();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 16;
            let va = _mm512_loadu_ps(a_ptr.add(offset));
            let vb = _mm512_loadu_ps(b_ptr.add(offset));
            sum = _mm512_fmadd_ps(va, vb, sum);
        }

        // Horizontal reduction
        let sum256_lo: __m256 = _mm512_castps512_ps256(sum);
        let sum256_hi: __m256 = _mm512_extractf32x8_ps::<1>(sum);
        let sum256: __m256 = _mm256_add_ps(sum256_lo, sum256_hi);
        let hi = _mm256_extractf128_ps(sum256, 1);
        let lo = _mm256_castps256_ps128(sum256);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        let tail_start = chunks * 16;
        for i in 0..remainder {
            result += *a.get_unchecked(tail_start + i) * *b.get_unchecked(tail_start + i);
        }
        result
    }

    /// Source: src/rerank/simd.rs:1390-1427.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::{
            __m256, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps,
            _mm256_loadu_ps, _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32,
            _mm_movehl_ps, _mm_shuffle_ps,
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

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a_ptr.add(offset));
            let vb = _mm256_loadu_ps(b_ptr.add(offset));
            sum = _mm256_fmadd_ps(va, vb, sum);
        }

        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        let tail_start = chunks * 8;
        for i in 0..remainder {
            result += *a.get_unchecked(tail_start + i) * *b.get_unchecked(tail_start + i);
        }
        result
    }

    /// Source: src/rerank/simd.rs:1430-1458.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
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

        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a_ptr.add(offset));
            let vb = vld1q_f32(b_ptr.add(offset));
            sum = vfmaq_f32(sum, va, vb);
        }

        let mut result = vaddvq_f32(sum);
        let tail_start = chunks * 4;
        for i in 0..remainder {
            result += *a.get_unchecked(tail_start + i) * *b.get_unchecked(tail_start + i);
        }
        result
    }
}
