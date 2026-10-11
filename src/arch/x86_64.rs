#![allow(clippy::incompatible_msrv)]
//! x86_64 SIMD kernels for AVX2+FMA and AVX-512, written against pulp tokens.
//!
//! Each kernel takes a pulp token (`V3` for AVX2+FMA, `V4` for AVX-512), which
//! can only be obtained after runtime detection, so calling a kernel is safe.
//! Intrinsics go through the token's feature fields (`simd.avx2._mm256_...`),
//! and loads are `pulp::cast` from fixed-size arrays produced by `as_chunks`,
//! which compile to the same unaligned vector loads. This module has no
//! `unsafe`.
//!
//! # Performance Hierarchy
//!
//! | ISA | Width | Coverage | Speedup vs scalar |
//! |-----|-------|----------|-------------------|
//! | AVX-512 | 16 f32 | ~9% | 10-20x |
//! | AVX2+FMA | 8 f32 | ~89% | 5-10x |
//! | SSE2 | 4 f32 | ~100% | 2-4x |

use pulp::bytemuck::NoUninit;
use pulp::x86::{V3, V4};
use std::arch::x86_64::{__m128i, __m256, __m256d, __m256i, __m512, __m512d, __m512i};

/// Popcount of each nibble 0..=15, repeated in every 128-bit lane, for the
/// `vpshufb` popcount in the Hamming kernels.
const NIBBLE_POPCNT_LUT64: [u8; 64] = {
    let mut lut = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        lut[i] = (i as u8 & 15).count_ones() as u8;
        i += 1;
    }
    lut
};

// Loads: a `pulp::cast` from a fixed-size array (or a zero-padded copy of a
// short tail), which compiles to an unaligned vector load inside a kernel.

/// Load 8 f32 into a ymm register.
#[inline(always)]
pub(crate) fn ld256_f32(c: &[f32; 8]) -> __m256 {
    pulp::cast(*c)
}

/// Load 4 f64 into a ymm register.
#[inline(always)]
fn ld256_f64(c: &[f64; 4]) -> __m256d {
    pulp::cast(*c)
}

/// Load 32 bytes of integers into a ymm register.
#[inline(always)]
fn ld256_si<A: NoUninit>(c: &A) -> __m256i {
    pulp::cast(*c)
}

/// Load 8 bytes into the low half of an xmm register (`movq`).
#[inline(always)]
fn ld64_u8(simd: V3, c: &[u8; 8]) -> __m128i {
    simd.sse2._mm_set_epi64x(0, i64::from_ne_bytes(*c))
}

/// Load 16 f32 into a zmm register.
#[inline(always)]
pub(crate) fn ld512_f32(c: &[f32; 16]) -> __m512 {
    pulp::cast(*c)
}

/// Load 8 f64 into a zmm register.
#[inline(always)]
fn ld512_f64(c: &[f64; 8]) -> __m512d {
    pulp::cast(*c)
}

/// Load 64 bytes of integers into a zmm register.
#[inline(always)]
fn ld512_si<A: NoUninit>(c: &A) -> __m512i {
    pulp::cast(*c)
}

/// Load up to 16 f32 from `t` into a zmm register, zeroing the unused lanes.
/// Replaces the masked `vmovups {k}{z}` tail with a zero-padded copy, which
/// adds the same zeros to the accumulator.
#[inline(always)]
pub(crate) fn ld512_f32_tail(t: &[f32]) -> __m512 {
    let mut lanes = [0.0f32; 16];
    let k = t.len().min(16);
    lanes[..k].copy_from_slice(&t[..k]);
    pulp::cast(lanes)
}

/// Load up to 8 f64 from `t` into a zmm register, zeroing the unused lanes.
#[inline(always)]
fn ld512_f64_tail(t: &[f64]) -> __m512d {
    let mut lanes = [0.0f64; 8];
    let k = t.len().min(8);
    lanes[..k].copy_from_slice(&t[..k]);
    pulp::cast(lanes)
}

/// AVX-512 dot product with 4-way unrolling and masked tail handling.
///
/// Processes 64 floats per iteration (4 x 16), hiding memory latency.
/// Uses masked loads to eliminate scalar tail loop (SimSIMD optimization).
///
/// # Performance
///
/// For 1535-dim vectors (common: 1536 minus 1), masked loads provide ~84% speedup
/// over scalar tail handling by avoiding branch mispredictions.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dot_avx512(simd: V4, a: &[f32], b: &[f32]) -> f32 {
    simd.vectorize(|| dot_avx512_in(simd, a, b))
}

#[inline(always)]
fn dot_avx512_in(simd: V4, a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let (a16, a_tail) = a[..n].as_chunks::<16>();
    let (b16, b_tail) = b[..n].as_chunks::<16>();
    let (a64, a_rem) = a16.as_chunks::<4>();
    let (b64, b_rem) = b16.as_chunks::<4>();

    // 4-way unrolled: process 64 floats per iteration
    let mut sum0: __m512 = simd.avx512f._mm512_setzero_ps();
    let mut sum1: __m512 = simd.avx512f._mm512_setzero_ps();
    let mut sum2: __m512 = simd.avx512f._mm512_setzero_ps();
    let mut sum3: __m512 = simd.avx512f._mm512_setzero_ps();

    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a64.iter().zip(b64) {
        sum0 = simd
            .avx512f
            ._mm512_fmadd_ps(ld512_f32(a0), ld512_f32(b0), sum0);
        sum1 = simd
            .avx512f
            ._mm512_fmadd_ps(ld512_f32(a1), ld512_f32(b1), sum1);
        sum2 = simd
            .avx512f
            ._mm512_fmadd_ps(ld512_f32(a2), ld512_f32(b2), sum2);
        sum3 = simd
            .avx512f
            ._mm512_fmadd_ps(ld512_f32(a3), ld512_f32(b3), sum3);
    }

    // Combine accumulators
    let sum01 = simd.avx512f._mm512_add_ps(sum0, sum1);
    let sum23 = simd.avx512f._mm512_add_ps(sum2, sum3);
    let sum_all = simd.avx512f._mm512_add_ps(sum01, sum23);
    let mut result = simd.avx512f._mm512_reduce_add_ps(sum_all);

    // Handle remaining elements using masked loads (no scalar tail!)
    if !a_rem.is_empty() || !a_tail.is_empty() {
        // Process full 16-element chunks
        let mut sum_rem: __m512 = simd.avx512f._mm512_setzero_ps();
        for (qa, qb) in a_rem.iter().zip(b_rem) {
            sum_rem = simd
                .avx512f
                ._mm512_fmadd_ps(ld512_f32(qa), ld512_f32(qb), sum_rem);
        }

        // Masked load for final partial chunk (0-15 elements)
        if !a_tail.is_empty() {
            let va = ld512_f32_tail(a_tail);
            let vb = ld512_f32_tail(b_tail);
            sum_rem = simd.avx512f._mm512_fmadd_ps(va, vb, sum_rem);
        }

        result += simd.avx512f._mm512_reduce_add_ps(sum_rem);
    }

    result
}

/// AVX-512 MaxSim implementation.
///
/// Iterates over query tokens and computes max similarity against all doc tokens
/// using the dot_avx512 kernel directly to avoid dispatch overhead.
///
/// `doc_tokens` must be non-empty; an empty slice causes `NEG_INFINITY` accumulation.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn maxsim_avx512(simd: V4, query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
    simd.vectorize(|| maxsim_avx512_in(simd, query_tokens, doc_tokens))
}

#[inline(always)]
fn maxsim_avx512_in(simd: V4, query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
    let mut total_score = 0.0;

    for q in query_tokens {
        let mut max_score = f32::NEG_INFINITY;

        // We could unroll this loop or block it, but since dot_avx512 is already
        // highly optimized (4-way unrolled internally), calling it directly
        // without dispatch overhead is the primary win here.
        //
        // Optimization opportunity:
        // If we want to go faster, we should compute multiple dot products in parallel
        // (matrix-vector multiplication style) to reuse loaded `q` vectors.
        // But simply removing dispatch is a big first step.
        for d in doc_tokens {
            let score = dot_avx512_in(simd, q, d);
            if score > max_score {
                max_score = score;
            }
        }
        total_score += max_score;
    }

    total_score
}

/// AVX2 MaxSim implementation.
///
/// Iterates over query tokens and computes max similarity against all doc tokens
/// using the dot_avx2 kernel directly.
///
/// `doc_tokens` must be non-empty; an empty slice causes `NEG_INFINITY` accumulation.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn maxsim_avx2(simd: V3, query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
    simd.vectorize(|| maxsim_avx2_in(simd, query_tokens, doc_tokens))
}

#[inline(always)]
fn maxsim_avx2_in(simd: V3, query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
    let mut total_score = 0.0;

    for q in query_tokens {
        let mut max_score = f32::NEG_INFINITY;
        for d in doc_tokens {
            let score = dot_avx2_in(simd, q, d);
            if score > max_score {
                max_score = score;
            }
        }
        total_score += max_score;
    }

    total_score
}

/// AVX2+FMA dot product with 4-way unrolling.
///
/// Processes 32 floats per iteration (4 x 8), hiding memory latency.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dot_avx2(simd: V3, a: &[f32], b: &[f32]) -> f32 {
    simd.vectorize(|| dot_avx2_in(simd, a, b))
}

#[inline(always)]
fn dot_avx2_in(simd: V3, a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let (a8, a_tail) = a[..n].as_chunks::<8>();
    let (b8, b_tail) = b[..n].as_chunks::<8>();
    let (a32, a_rem) = a8.as_chunks::<4>();
    let (b32, b_rem) = b8.as_chunks::<4>();

    // 4-way unrolled: process 32 floats per iteration
    let mut sum0: __m256 = simd.avx._mm256_setzero_ps();
    let mut sum1: __m256 = simd.avx._mm256_setzero_ps();
    let mut sum2: __m256 = simd.avx._mm256_setzero_ps();
    let mut sum3: __m256 = simd.avx._mm256_setzero_ps();

    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a32.iter().zip(b32) {
        sum0 = simd.fma._mm256_fmadd_ps(ld256_f32(a0), ld256_f32(b0), sum0);
        sum1 = simd.fma._mm256_fmadd_ps(ld256_f32(a1), ld256_f32(b1), sum1);
        sum2 = simd.fma._mm256_fmadd_ps(ld256_f32(a2), ld256_f32(b2), sum2);
        sum3 = simd.fma._mm256_fmadd_ps(ld256_f32(a3), ld256_f32(b3), sum3);
    }

    // Combine accumulators
    let sum01 = simd.avx._mm256_add_ps(sum0, sum1);
    let sum23 = simd.avx._mm256_add_ps(sum2, sum3);
    let sum_all = simd.avx._mm256_add_ps(sum01, sum23);

    // Horizontal reduction
    let hi = simd.avx._mm256_extractf128_ps::<1>(sum_all);
    let lo = simd.avx._mm256_castps256_ps128(sum_all);
    let sum128 = simd.sse._mm_add_ps(lo, hi);
    let sum64 = simd
        .sse
        ._mm_add_ps(sum128, simd.sse._mm_movehl_ps(sum128, sum128));
    let sum32 = simd
        .sse
        ._mm_add_ss(sum64, simd.sse._mm_shuffle_ps::<1>(sum64, sum64));
    let mut result = simd.sse._mm_cvtss_f32(sum32);

    // Handle remaining 8-float chunks
    let mut sum: __m256 = simd.avx._mm256_setzero_ps();
    for (qa, qb) in a_rem.iter().zip(b_rem) {
        sum = simd.fma._mm256_fmadd_ps(ld256_f32(qa), ld256_f32(qb), sum);
    }

    // Reduce remaining sum
    let hi = simd.avx._mm256_extractf128_ps::<1>(sum);
    let lo = simd.avx._mm256_castps256_ps128(sum);
    let sum128 = simd.sse._mm_add_ps(lo, hi);
    let sum64 = simd
        .sse
        ._mm_add_ps(sum128, simd.sse._mm_movehl_ps(sum128, sum128));
    let sum32 = simd
        .sse
        ._mm_add_ss(sum64, simd.sse._mm_shuffle_ps::<1>(sum64, sum64));
    result += simd.sse._mm_cvtss_f32(sum32);

    // Scalar tail
    for (x, y) in a_tail.iter().zip(b_tail) {
        result += x * y;
    }

    result
}

/// AVX-512 squared L2 distance: `Σ(a[i] - b[i])²`.
///
/// Single-pass: computes differences directly, avoiding catastrophic
/// cancellation from the expansion `||a||² + ||b||² - 2<a,b>`.
/// Uses masked loads to eliminate scalar tail loop.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn l2_squared_avx512(simd: V4, a: &[f32], b: &[f32]) -> f32 {
    simd.vectorize(|| l2_squared_avx512_in(simd, a, b))
}

#[inline(always)]
fn l2_squared_avx512_in(simd: V4, a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let (a_vecs, a_tail) = a[..n].as_chunks::<16>();
    let (b_vecs, b_tail) = b[..n].as_chunks::<16>();
    let (a_quads, a_rem) = a_vecs.as_chunks::<4>();
    let (b_quads, b_rem) = b_vecs.as_chunks::<4>();

    // 4-way unrolled: process 64 floats per iteration
    let mut sum0: __m512 = simd.avx512f._mm512_setzero_ps();
    let mut sum1: __m512 = simd.avx512f._mm512_setzero_ps();
    let mut sum2: __m512 = simd.avx512f._mm512_setzero_ps();
    let mut sum3: __m512 = simd.avx512f._mm512_setzero_ps();

    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a_quads.iter().zip(b_quads) {
        let va0 = ld512_f32(a0);
        let vb0 = ld512_f32(b0);
        let d0 = simd.avx512f._mm512_sub_ps(va0, vb0);
        let va1 = ld512_f32(a1);
        let vb1 = ld512_f32(b1);
        let d1 = simd.avx512f._mm512_sub_ps(va1, vb1);
        let va2 = ld512_f32(a2);
        let vb2 = ld512_f32(b2);
        let d2 = simd.avx512f._mm512_sub_ps(va2, vb2);
        let va3 = ld512_f32(a3);
        let vb3 = ld512_f32(b3);
        let d3 = simd.avx512f._mm512_sub_ps(va3, vb3);

        sum0 = simd.avx512f._mm512_fmadd_ps(d0, d0, sum0);
        sum1 = simd.avx512f._mm512_fmadd_ps(d1, d1, sum1);
        sum2 = simd.avx512f._mm512_fmadd_ps(d2, d2, sum2);
        sum3 = simd.avx512f._mm512_fmadd_ps(d3, d3, sum3);
    }

    // Combine accumulators
    let sum01 = simd.avx512f._mm512_add_ps(sum0, sum1);
    let sum23 = simd.avx512f._mm512_add_ps(sum2, sum3);
    let sum_all = simd.avx512f._mm512_add_ps(sum01, sum23);
    let mut result = simd.avx512f._mm512_reduce_add_ps(sum_all);

    // Handle remaining elements using masked loads (no scalar tail!)
    if !a_rem.is_empty() || !a_tail.is_empty() {
        // Process full 16-element chunks
        let mut sum_rem: __m512 = simd.avx512f._mm512_setzero_ps();

        for (qa, qb) in a_rem.iter().zip(b_rem) {
            let va = ld512_f32(qa);
            let vb = ld512_f32(qb);
            let d = simd.avx512f._mm512_sub_ps(va, vb);
            sum_rem = simd.avx512f._mm512_fmadd_ps(d, d, sum_rem);
        }

        // Masked load for final partial chunk (0-15 elements)
        if !a_tail.is_empty() {
            let va = ld512_f32_tail(a_tail);
            let vb = ld512_f32_tail(b_tail);
            let d = simd.avx512f._mm512_sub_ps(va, vb);
            sum_rem = simd.avx512f._mm512_fmadd_ps(d, d, sum_rem);
        }

        result += simd.avx512f._mm512_reduce_add_ps(sum_rem);
    }

    result
}

/// AVX2+FMA squared L2 distance: `Σ(a[i] - b[i])²`.
///
/// Single-pass: computes differences directly, avoiding catastrophic
/// cancellation from the expansion `||a||² + ||b||² - 2<a,b>`.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn l2_squared_avx2(simd: V3, a: &[f32], b: &[f32]) -> f32 {
    simd.vectorize(|| l2_squared_avx2_in(simd, a, b))
}

#[inline(always)]
fn l2_squared_avx2_in(simd: V3, a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let (a_vecs, a_tail) = a[..n].as_chunks::<8>();
    let (b_vecs, b_tail) = b[..n].as_chunks::<8>();
    let (a_quads, a_rem) = a_vecs.as_chunks::<4>();
    let (b_quads, b_rem) = b_vecs.as_chunks::<4>();

    // 4-way unrolled: process 32 floats per iteration
    let mut sum0: __m256 = simd.avx._mm256_setzero_ps();
    let mut sum1: __m256 = simd.avx._mm256_setzero_ps();
    let mut sum2: __m256 = simd.avx._mm256_setzero_ps();
    let mut sum3: __m256 = simd.avx._mm256_setzero_ps();

    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a_quads.iter().zip(b_quads) {
        let va0 = ld256_f32(a0);
        let vb0 = ld256_f32(b0);
        let d0 = simd.avx._mm256_sub_ps(va0, vb0);
        let va1 = ld256_f32(a1);
        let vb1 = ld256_f32(b1);
        let d1 = simd.avx._mm256_sub_ps(va1, vb1);
        let va2 = ld256_f32(a2);
        let vb2 = ld256_f32(b2);
        let d2 = simd.avx._mm256_sub_ps(va2, vb2);
        let va3 = ld256_f32(a3);
        let vb3 = ld256_f32(b3);
        let d3 = simd.avx._mm256_sub_ps(va3, vb3);

        sum0 = simd.fma._mm256_fmadd_ps(d0, d0, sum0);
        sum1 = simd.fma._mm256_fmadd_ps(d1, d1, sum1);
        sum2 = simd.fma._mm256_fmadd_ps(d2, d2, sum2);
        sum3 = simd.fma._mm256_fmadd_ps(d3, d3, sum3);
    }

    // Combine accumulators
    let sum01 = simd.avx._mm256_add_ps(sum0, sum1);
    let sum23 = simd.avx._mm256_add_ps(sum2, sum3);
    let sum_all = simd.avx._mm256_add_ps(sum01, sum23);

    // Horizontal reduction
    let hi = simd.avx._mm256_extractf128_ps::<1>(sum_all);
    let lo = simd.avx._mm256_castps256_ps128(sum_all);
    let sum128 = simd.sse._mm_add_ps(lo, hi);
    let sum64 = simd
        .sse
        ._mm_add_ps(sum128, simd.sse._mm_movehl_ps(sum128, sum128));
    let sum32 = simd
        .sse
        ._mm_add_ss(sum64, simd.sse._mm_shuffle_ps::<1>(sum64, sum64));
    let mut result = simd.sse._mm_cvtss_f32(sum32);

    // Handle remaining 8-float chunks
    let mut sum: __m256 = simd.avx._mm256_setzero_ps();
    for (qa, qb) in a_rem.iter().zip(b_rem) {
        let va = ld256_f32(qa);
        let vb = ld256_f32(qb);
        let d = simd.avx._mm256_sub_ps(va, vb);
        sum = simd.fma._mm256_fmadd_ps(d, d, sum);
    }

    // Reduce remaining sum
    let hi = simd.avx._mm256_extractf128_ps::<1>(sum);
    let lo = simd.avx._mm256_castps256_ps128(sum);
    let sum128 = simd.sse._mm_add_ps(lo, hi);
    let sum64 = simd
        .sse
        ._mm_add_ps(sum128, simd.sse._mm_movehl_ps(sum128, sum128));
    let sum32 = simd
        .sse
        ._mm_add_ss(sum64, simd.sse._mm_shuffle_ps::<1>(sum64, sum64));
    result += simd.sse._mm_cvtss_f32(sum32);

    // Scalar tail
    for (&ai, &bi) in a_tail.iter().zip(b_tail) {
        let d = ai - bi;
        result += d * d;
    }

    result
}

/// AVX-512 L1 (Manhattan) distance: `Σ|a[i] - b[i]|`.
///
/// Uses `_mm512_abs_ps` (via bit-mask AND) for branchless absolute value.
/// 4-way unrolled with masked tail handling.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn l1_avx512(simd: V4, a: &[f32], b: &[f32]) -> f32 {
    simd.vectorize(|| l1_avx512_in(simd, a, b))
}

#[inline(always)]
fn l1_avx512_in(simd: V4, a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let (a_vecs, a_tail) = a[..n].as_chunks::<16>();
    let (b_vecs, b_tail) = b[..n].as_chunks::<16>();
    let (a_quads, a_rem) = a_vecs.as_chunks::<4>();
    let (b_quads, b_rem) = b_vecs.as_chunks::<4>();

    let mut sum0: __m512 = simd.avx512f._mm512_setzero_ps();
    let mut sum1: __m512 = simd.avx512f._mm512_setzero_ps();
    let mut sum2: __m512 = simd.avx512f._mm512_setzero_ps();
    let mut sum3: __m512 = simd.avx512f._mm512_setzero_ps();

    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a_quads.iter().zip(b_quads) {
        let d0 = simd
            .avx512f
            ._mm512_abs_ps(simd.avx512f._mm512_sub_ps(ld512_f32(a0), ld512_f32(b0)));
        let d1 = simd
            .avx512f
            ._mm512_abs_ps(simd.avx512f._mm512_sub_ps(ld512_f32(a1), ld512_f32(b1)));
        let d2 = simd
            .avx512f
            ._mm512_abs_ps(simd.avx512f._mm512_sub_ps(ld512_f32(a2), ld512_f32(b2)));
        let d3 = simd
            .avx512f
            ._mm512_abs_ps(simd.avx512f._mm512_sub_ps(ld512_f32(a3), ld512_f32(b3)));

        sum0 = simd.avx512f._mm512_add_ps(sum0, d0);
        sum1 = simd.avx512f._mm512_add_ps(sum1, d1);
        sum2 = simd.avx512f._mm512_add_ps(sum2, d2);
        sum3 = simd.avx512f._mm512_add_ps(sum3, d3);
    }

    let sum_all = simd.avx512f._mm512_add_ps(
        simd.avx512f._mm512_add_ps(sum0, sum1),
        simd.avx512f._mm512_add_ps(sum2, sum3),
    );
    let mut result = simd.avx512f._mm512_reduce_add_ps(sum_all);

    // Remaining elements with masked loads
    if !a_rem.is_empty() || !a_tail.is_empty() {
        let mut sum_rem: __m512 = simd.avx512f._mm512_setzero_ps();

        for (qa, qb) in a_rem.iter().zip(b_rem) {
            let d = simd
                .avx512f
                ._mm512_abs_ps(simd.avx512f._mm512_sub_ps(ld512_f32(qa), ld512_f32(qb)));
            sum_rem = simd.avx512f._mm512_add_ps(sum_rem, d);
        }

        if !a_tail.is_empty() {
            let va = ld512_f32_tail(a_tail);
            let vb = ld512_f32_tail(b_tail);
            let d = simd
                .avx512f
                ._mm512_abs_ps(simd.avx512f._mm512_sub_ps(va, vb));
            sum_rem = simd.avx512f._mm512_add_ps(sum_rem, d);
        }

        result += simd.avx512f._mm512_reduce_add_ps(sum_rem);
    }

    result
}

/// AVX2 L1 (Manhattan) distance: `Σ|a[i] - b[i]|`.
///
/// Uses sign-bit mask AND to compute absolute value without branching.
/// 4-way unrolled.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn l1_avx2(simd: V3, a: &[f32], b: &[f32]) -> f32 {
    simd.vectorize(|| l1_avx2_in(simd, a, b))
}

#[inline(always)]
fn l1_avx2_in(simd: V3, a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let (a_vecs, a_tail) = a[..n].as_chunks::<8>();
    let (b_vecs, b_tail) = b[..n].as_chunks::<8>();
    let (a_quads, a_rem) = a_vecs.as_chunks::<4>();
    let (b_quads, b_rem) = b_vecs.as_chunks::<4>();

    // Sign bit mask: clear bit 31 to compute abs
    let sign_mask = simd.avx._mm256_set1_ps(f32::from_bits(0x8000_0000));

    let mut sum0: __m256 = simd.avx._mm256_setzero_ps();
    let mut sum1: __m256 = simd.avx._mm256_setzero_ps();
    let mut sum2: __m256 = simd.avx._mm256_setzero_ps();
    let mut sum3: __m256 = simd.avx._mm256_setzero_ps();

    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a_quads.iter().zip(b_quads) {
        let d0 = simd.avx._mm256_andnot_ps(
            sign_mask,
            simd.avx._mm256_sub_ps(ld256_f32(a0), ld256_f32(b0)),
        );
        let d1 = simd.avx._mm256_andnot_ps(
            sign_mask,
            simd.avx._mm256_sub_ps(ld256_f32(a1), ld256_f32(b1)),
        );
        let d2 = simd.avx._mm256_andnot_ps(
            sign_mask,
            simd.avx._mm256_sub_ps(ld256_f32(a2), ld256_f32(b2)),
        );
        let d3 = simd.avx._mm256_andnot_ps(
            sign_mask,
            simd.avx._mm256_sub_ps(ld256_f32(a3), ld256_f32(b3)),
        );

        sum0 = simd.avx._mm256_add_ps(sum0, d0);
        sum1 = simd.avx._mm256_add_ps(sum1, d1);
        sum2 = simd.avx._mm256_add_ps(sum2, d2);
        sum3 = simd.avx._mm256_add_ps(sum3, d3);
    }

    // Combine accumulators
    let sum_all = simd.avx._mm256_add_ps(
        simd.avx._mm256_add_ps(sum0, sum1),
        simd.avx._mm256_add_ps(sum2, sum3),
    );

    // Horizontal reduction
    let hi = simd.avx._mm256_extractf128_ps::<1>(sum_all);
    let lo = simd.avx._mm256_castps256_ps128(sum_all);
    let sum128 = simd.sse._mm_add_ps(lo, hi);
    let sum64 = simd
        .sse
        ._mm_add_ps(sum128, simd.sse._mm_movehl_ps(sum128, sum128));
    let sum32 = simd
        .sse
        ._mm_add_ss(sum64, simd.sse._mm_shuffle_ps::<1>(sum64, sum64));
    let mut result = simd.sse._mm_cvtss_f32(sum32);

    // Handle remaining 8-float chunks
    let mut sum: __m256 = simd.avx._mm256_setzero_ps();
    for (qa, qb) in a_rem.iter().zip(b_rem) {
        let d = simd.avx._mm256_andnot_ps(
            sign_mask,
            simd.avx._mm256_sub_ps(ld256_f32(qa), ld256_f32(qb)),
        );
        sum = simd.avx._mm256_add_ps(sum, d);
    }

    let hi = simd.avx._mm256_extractf128_ps::<1>(sum);
    let lo = simd.avx._mm256_castps256_ps128(sum);
    let sum128 = simd.sse._mm_add_ps(lo, hi);
    let sum64 = simd
        .sse
        ._mm_add_ps(sum128, simd.sse._mm_movehl_ps(sum128, sum128));
    let sum32 = simd
        .sse
        ._mm_add_ss(sum64, simd.sse._mm_shuffle_ps::<1>(sum64, sum64));
    result += simd.sse._mm_cvtss_f32(sum32);

    // Scalar tail
    for (&ai, &bi) in a_tail.iter().zip(b_tail) {
        result += (ai - bi).abs();
    }

    result
}

/// AVX-512 fused cosine similarity: single-pass dot(a,b), norm(a)^2, norm(b)^2.
///
/// Accumulates all three products in one pass with 4-way unrolling and masked
/// tail handling. Uses exact sqrt/div for IEEE-correct normalization.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn cosine_avx512(simd: V4, a: &[f32], b: &[f32]) -> f32 {
    simd.vectorize(|| cosine_avx512_in(simd, a, b))
}

#[inline(always)]
fn cosine_avx512_in(simd: V4, a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let (a_vecs, a_tail) = a[..n].as_chunks::<16>();
    let (b_vecs, b_tail) = b[..n].as_chunks::<16>();
    let (a_quads, a_rem) = a_vecs.as_chunks::<4>();
    let (b_quads, b_rem) = b_vecs.as_chunks::<4>();

    // 4-way unrolled: process 64 floats per iteration, 3 accumulators each
    let mut ab0: __m512 = simd.avx512f._mm512_setzero_ps();
    let mut ab1: __m512 = simd.avx512f._mm512_setzero_ps();
    let mut ab2: __m512 = simd.avx512f._mm512_setzero_ps();
    let mut ab3: __m512 = simd.avx512f._mm512_setzero_ps();
    let mut aa0: __m512 = simd.avx512f._mm512_setzero_ps();
    let mut aa1: __m512 = simd.avx512f._mm512_setzero_ps();
    let mut aa2: __m512 = simd.avx512f._mm512_setzero_ps();
    let mut aa3: __m512 = simd.avx512f._mm512_setzero_ps();
    let mut bb0: __m512 = simd.avx512f._mm512_setzero_ps();
    let mut bb1: __m512 = simd.avx512f._mm512_setzero_ps();
    let mut bb2: __m512 = simd.avx512f._mm512_setzero_ps();
    let mut bb3: __m512 = simd.avx512f._mm512_setzero_ps();

    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a_quads.iter().zip(b_quads) {
        let va0 = ld512_f32(a0);
        let vb0 = ld512_f32(b0);
        let va1 = ld512_f32(a1);
        let vb1 = ld512_f32(b1);
        let va2 = ld512_f32(a2);
        let vb2 = ld512_f32(b2);
        let va3 = ld512_f32(a3);
        let vb3 = ld512_f32(b3);

        ab0 = simd.avx512f._mm512_fmadd_ps(va0, vb0, ab0);
        ab1 = simd.avx512f._mm512_fmadd_ps(va1, vb1, ab1);
        ab2 = simd.avx512f._mm512_fmadd_ps(va2, vb2, ab2);
        ab3 = simd.avx512f._mm512_fmadd_ps(va3, vb3, ab3);

        aa0 = simd.avx512f._mm512_fmadd_ps(va0, va0, aa0);
        aa1 = simd.avx512f._mm512_fmadd_ps(va1, va1, aa1);
        aa2 = simd.avx512f._mm512_fmadd_ps(va2, va2, aa2);
        aa3 = simd.avx512f._mm512_fmadd_ps(va3, va3, aa3);

        bb0 = simd.avx512f._mm512_fmadd_ps(vb0, vb0, bb0);
        bb1 = simd.avx512f._mm512_fmadd_ps(vb1, vb1, bb1);
        bb2 = simd.avx512f._mm512_fmadd_ps(vb2, vb2, bb2);
        bb3 = simd.avx512f._mm512_fmadd_ps(vb3, vb3, bb3);
    }

    // Combine accumulators
    let ab_all = simd.avx512f._mm512_add_ps(
        simd.avx512f._mm512_add_ps(ab0, ab1),
        simd.avx512f._mm512_add_ps(ab2, ab3),
    );
    let aa_all = simd.avx512f._mm512_add_ps(
        simd.avx512f._mm512_add_ps(aa0, aa1),
        simd.avx512f._mm512_add_ps(aa2, aa3),
    );
    let bb_all = simd.avx512f._mm512_add_ps(
        simd.avx512f._mm512_add_ps(bb0, bb1),
        simd.avx512f._mm512_add_ps(bb2, bb3),
    );

    let mut ab = simd.avx512f._mm512_reduce_add_ps(ab_all);
    let mut aa = simd.avx512f._mm512_reduce_add_ps(aa_all);
    let mut bb = simd.avx512f._mm512_reduce_add_ps(bb_all);

    // Handle remaining elements using masked loads
    if !a_rem.is_empty() || !a_tail.is_empty() {
        let mut ab_rem: __m512 = simd.avx512f._mm512_setzero_ps();
        let mut aa_rem: __m512 = simd.avx512f._mm512_setzero_ps();
        let mut bb_rem: __m512 = simd.avx512f._mm512_setzero_ps();

        for (qa, qb) in a_rem.iter().zip(b_rem) {
            let va = ld512_f32(qa);
            let vb = ld512_f32(qb);
            ab_rem = simd.avx512f._mm512_fmadd_ps(va, vb, ab_rem);
            aa_rem = simd.avx512f._mm512_fmadd_ps(va, va, aa_rem);
            bb_rem = simd.avx512f._mm512_fmadd_ps(vb, vb, bb_rem);
        }

        if !a_tail.is_empty() {
            let va = ld512_f32_tail(a_tail);
            let vb = ld512_f32_tail(b_tail);
            ab_rem = simd.avx512f._mm512_fmadd_ps(va, vb, ab_rem);
            aa_rem = simd.avx512f._mm512_fmadd_ps(va, va, aa_rem);
            bb_rem = simd.avx512f._mm512_fmadd_ps(vb, vb, bb_rem);
        }

        ab += simd.avx512f._mm512_reduce_add_ps(ab_rem);
        aa += simd.avx512f._mm512_reduce_add_ps(aa_rem);
        bb += simd.avx512f._mm512_reduce_add_ps(bb_rem);
    }

    if aa > crate::NORM_EPSILON_SQ && bb > crate::NORM_EPSILON_SQ {
        ab / (aa.sqrt() * bb.sqrt())
    } else {
        0.0
    }
}

/// AVX2+FMA fused cosine similarity: single-pass dot(a,b), norm(a)^2, norm(b)^2.
///
/// Accumulates all three products in one pass with 4-way unrolling.
/// Uses exact sqrt/div for IEEE-correct normalization.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn cosine_avx2(simd: V3, a: &[f32], b: &[f32]) -> f32 {
    simd.vectorize(|| cosine_avx2_in(simd, a, b))
}

#[inline(always)]
fn cosine_avx2_in(simd: V3, a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let (a_vecs, a_tail) = a[..n].as_chunks::<8>();
    let (b_vecs, b_tail) = b[..n].as_chunks::<8>();
    let (a_quads, a_rem) = a_vecs.as_chunks::<4>();
    let (b_quads, b_rem) = b_vecs.as_chunks::<4>();

    // 4-way unrolled: process 32 floats per iteration, 3 accumulators each
    let mut ab0: __m256 = simd.avx._mm256_setzero_ps();
    let mut ab1: __m256 = simd.avx._mm256_setzero_ps();
    let mut ab2: __m256 = simd.avx._mm256_setzero_ps();
    let mut ab3: __m256 = simd.avx._mm256_setzero_ps();
    let mut aa0: __m256 = simd.avx._mm256_setzero_ps();
    let mut aa1: __m256 = simd.avx._mm256_setzero_ps();
    let mut aa2: __m256 = simd.avx._mm256_setzero_ps();
    let mut aa3: __m256 = simd.avx._mm256_setzero_ps();
    let mut bb0: __m256 = simd.avx._mm256_setzero_ps();
    let mut bb1: __m256 = simd.avx._mm256_setzero_ps();
    let mut bb2: __m256 = simd.avx._mm256_setzero_ps();
    let mut bb3: __m256 = simd.avx._mm256_setzero_ps();

    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a_quads.iter().zip(b_quads) {
        let va0 = ld256_f32(a0);
        let vb0 = ld256_f32(b0);
        let va1 = ld256_f32(a1);
        let vb1 = ld256_f32(b1);
        let va2 = ld256_f32(a2);
        let vb2 = ld256_f32(b2);
        let va3 = ld256_f32(a3);
        let vb3 = ld256_f32(b3);

        ab0 = simd.fma._mm256_fmadd_ps(va0, vb0, ab0);
        ab1 = simd.fma._mm256_fmadd_ps(va1, vb1, ab1);
        ab2 = simd.fma._mm256_fmadd_ps(va2, vb2, ab2);
        ab3 = simd.fma._mm256_fmadd_ps(va3, vb3, ab3);

        aa0 = simd.fma._mm256_fmadd_ps(va0, va0, aa0);
        aa1 = simd.fma._mm256_fmadd_ps(va1, va1, aa1);
        aa2 = simd.fma._mm256_fmadd_ps(va2, va2, aa2);
        aa3 = simd.fma._mm256_fmadd_ps(va3, va3, aa3);

        bb0 = simd.fma._mm256_fmadd_ps(vb0, vb0, bb0);
        bb1 = simd.fma._mm256_fmadd_ps(vb1, vb1, bb1);
        bb2 = simd.fma._mm256_fmadd_ps(vb2, vb2, bb2);
        bb3 = simd.fma._mm256_fmadd_ps(vb3, vb3, bb3);
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

    // Combine accumulators
    let ab_all = simd.avx._mm256_add_ps(
        simd.avx._mm256_add_ps(ab0, ab1),
        simd.avx._mm256_add_ps(ab2, ab3),
    );
    let aa_all = simd.avx._mm256_add_ps(
        simd.avx._mm256_add_ps(aa0, aa1),
        simd.avx._mm256_add_ps(aa2, aa3),
    );
    let bb_all = simd.avx._mm256_add_ps(
        simd.avx._mm256_add_ps(bb0, bb1),
        simd.avx._mm256_add_ps(bb2, bb3),
    );

    let mut ab = hsum256(simd, ab_all);
    let mut aa = hsum256(simd, aa_all);
    let mut bb = hsum256(simd, bb_all);

    // Handle remaining 8-float chunks
    let mut ab_rem: __m256 = simd.avx._mm256_setzero_ps();
    let mut aa_rem: __m256 = simd.avx._mm256_setzero_ps();
    let mut bb_rem: __m256 = simd.avx._mm256_setzero_ps();

    for (qa, qb) in a_rem.iter().zip(b_rem) {
        let va = ld256_f32(qa);
        let vb = ld256_f32(qb);
        ab_rem = simd.fma._mm256_fmadd_ps(va, vb, ab_rem);
        aa_rem = simd.fma._mm256_fmadd_ps(va, va, aa_rem);
        bb_rem = simd.fma._mm256_fmadd_ps(vb, vb, bb_rem);
    }

    ab += hsum256(simd, ab_rem);
    aa += hsum256(simd, aa_rem);
    bb += hsum256(simd, bb_rem);

    // Scalar tail
    for (&ai, &bi) in a_tail.iter().zip(b_tail) {
        ab += ai * bi;
        aa += ai * ai;
        bb += bi * bi;
    }

    if aa > crate::NORM_EPSILON_SQ && bb > crate::NORM_EPSILON_SQ {
        ab / (aa.sqrt() * bb.sqrt())
    } else {
        0.0
    }
}

/// AVX2 mixed-precision dot product: `sum(a_f32[i] * b_u8[i] as f32)`.
///
/// Uses VPMOVZXBD to widen 8 x u8 to 8 x i32, converts to f32, then FMA.
/// 4-way unrolled: processes 32 u8 elements per outer iteration.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dot_u8_f32_avx2(simd: V3, a: &[f32], b: &[u8]) -> f32 {
    simd.vectorize(|| dot_u8_f32_avx2_in(simd, a, b))
}

#[inline(always)]
fn dot_u8_f32_avx2_in(simd: V3, a: &[f32], b: &[u8]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let (a_vecs, a_tail) = a[..n].as_chunks::<8>();
    let (b_vecs, b_tail) = b[..n].as_chunks::<8>();
    let (a_quads, a_rem) = a_vecs.as_chunks::<4>();
    let (b_quads, b_rem) = b_vecs.as_chunks::<4>();

    // Widen 8 bytes to i32, convert to f32
    let widen = |q: &[u8; 8]| {
        simd.avx
            ._mm256_cvtepi32_ps(simd.avx2._mm256_cvtepu8_epi32(ld64_u8(simd, q)))
    };

    // 4-way unrolled: process 32 elements per iteration (4 x 8)
    let mut sum0: __m256 = simd.avx._mm256_setzero_ps();
    let mut sum1: __m256 = simd.avx._mm256_setzero_ps();
    let mut sum2: __m256 = simd.avx._mm256_setzero_ps();
    let mut sum3: __m256 = simd.avx._mm256_setzero_ps();

    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a_quads.iter().zip(b_quads) {
        sum0 = simd.fma._mm256_fmadd_ps(ld256_f32(a0), widen(b0), sum0);
        sum1 = simd.fma._mm256_fmadd_ps(ld256_f32(a1), widen(b1), sum1);
        sum2 = simd.fma._mm256_fmadd_ps(ld256_f32(a2), widen(b2), sum2);
        sum3 = simd.fma._mm256_fmadd_ps(ld256_f32(a3), widen(b3), sum3);
    }

    // Combine accumulators
    let sum_all = simd.avx._mm256_add_ps(
        simd.avx._mm256_add_ps(sum0, sum1),
        simd.avx._mm256_add_ps(sum2, sum3),
    );

    // Horizontal reduction
    let hi = simd.avx._mm256_extractf128_ps::<1>(sum_all);
    let lo = simd.avx._mm256_castps256_ps128(sum_all);
    let sum128 = simd.sse._mm_add_ps(lo, hi);
    let sum64 = simd
        .sse
        ._mm_add_ps(sum128, simd.sse._mm_movehl_ps(sum128, sum128));
    let sum32 = simd
        .sse
        ._mm_add_ss(sum64, simd.sse._mm_shuffle_ps::<1>(sum64, sum64));
    let mut result = simd.sse._mm_cvtss_f32(sum32);

    // Handle remaining 8-element chunks
    let mut sum: __m256 = simd.avx._mm256_setzero_ps();
    for (qa, qb) in a_rem.iter().zip(b_rem) {
        sum = simd.fma._mm256_fmadd_ps(ld256_f32(qa), widen(qb), sum);
    }

    let hi = simd.avx._mm256_extractf128_ps::<1>(sum);
    let lo = simd.avx._mm256_castps256_ps128(sum);
    let sum128 = simd.sse._mm_add_ps(lo, hi);
    let sum64 = simd
        .sse
        ._mm_add_ps(sum128, simd.sse._mm_movehl_ps(sum128, sum128));
    let sum32 = simd
        .sse
        ._mm_add_ss(sum64, simd.sse._mm_shuffle_ps::<1>(sum64, sum64));
    result += simd.sse._mm_cvtss_f32(sum32);

    // Scalar tail
    for (&x, &y) in a_tail.iter().zip(b_tail) {
        result += x * (y as f32);
    }

    result
}

/// AVX2 u8×u8 dot product.
///
/// Loads 32 bytes per iteration, widens each 16-byte half from u8 to i16 via
/// `_mm256_cvtepu8_epi16`, then multiplies and accumulates to i32 via
/// `_mm256_madd_epi16`. Accumulates directly to i32 — no periodic drain needed
/// since each `madd` step produces at most 2 * 255^2 = 130050 < i32::MAX.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dot_u8_avx2(simd: V3, a: &[u8], b: &[u8]) -> u32 {
    simd.vectorize(|| dot_u8_avx2_in(simd, a, b))
}

#[inline(always)]
fn dot_u8_avx2_in(simd: V3, a: &[u8], b: &[u8]) -> u32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0;
    }

    let (a32, a_tail) = a[..n].as_chunks::<32>();
    let (b32, b_tail) = b[..n].as_chunks::<32>();

    let mut acc32: __m256i = simd.avx._mm256_setzero_si256();

    for (qa, qb) in a32.iter().zip(b32) {
        // Load 32 x u8 each
        let va = ld256_si(qa);
        let vb = ld256_si(qb);

        // Widen low 16 bytes (u8 -> i16) from each vector
        let va_lo = simd
            .avx2
            ._mm256_cvtepu8_epi16(simd.avx._mm256_castsi256_si128(va));
        let vb_lo = simd
            .avx2
            ._mm256_cvtepu8_epi16(simd.avx._mm256_castsi256_si128(vb));
        // Widen high 16 bytes (u8 -> i16)
        let va_hi = simd
            .avx2
            ._mm256_cvtepu8_epi16(simd.avx2._mm256_extracti128_si256::<1>(va));
        let vb_hi = simd
            .avx2
            ._mm256_cvtepu8_epi16(simd.avx2._mm256_extracti128_si256::<1>(vb));

        // madd_epi16: multiply 16xi16 pairs and add adjacent -> 8xi32
        let prod_lo = simd.avx2._mm256_madd_epi16(va_lo, vb_lo);
        let prod_hi = simd.avx2._mm256_madd_epi16(va_hi, vb_hi);
        acc32 = simd.avx2._mm256_add_epi32(acc32, prod_lo);
        acc32 = simd.avx2._mm256_add_epi32(acc32, prod_hi);
    }

    // Horizontal sum of acc32 (8 x i32)
    let hi128 = simd.avx2._mm256_permute2x128_si256::<0x01>(acc32, acc32);
    let sum128 = simd.avx2._mm256_add_epi32(acc32, hi128);
    let result: u32 = (simd.avx2._mm256_extract_epi32::<0>(sum128) as u32)
        .wrapping_add(simd.avx2._mm256_extract_epi32::<1>(sum128) as u32)
        .wrapping_add(simd.avx2._mm256_extract_epi32::<2>(sum128) as u32)
        .wrapping_add(simd.avx2._mm256_extract_epi32::<3>(sum128) as u32);

    // Scalar tail
    let tail: u32 = a_tail
        .iter()
        .zip(b_tail)
        .map(|(&x, &y)| x as u32 * y as u32)
        .sum();

    result.wrapping_add(tail)
}

/// AVX-512BW u8×u8 dot product.
///
/// Loads 64 bytes per iteration. Splits each 512-bit register into two 256-bit
/// halves, widens u8->i16 via `_mm512_cvtepu8_epi16`, multiplies and
/// accumulates to i32 via `_mm512_madd_epi16`.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dot_u8_avx512(simd: V4, a: &[u8], b: &[u8]) -> u32 {
    simd.vectorize(|| dot_u8_avx512_in(simd, a, b))
}

#[inline(always)]
fn dot_u8_avx512_in(simd: V4, a: &[u8], b: &[u8]) -> u32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0;
    }

    let (a64, a_tail) = a[..n].as_chunks::<64>();
    let (b64, b_tail) = b[..n].as_chunks::<64>();

    let mut acc32: __m512i = simd.avx512f._mm512_setzero_si512();

    for (qa, qb) in a64.iter().zip(b64) {
        // Load 64 x u8
        let va = ld512_si(qa);
        let vb = ld512_si(qb);

        // Split 512-bit -> two 256-bit halves, then widen u8->i16 (256->512)
        let va_lo16 = simd
            .avx512bw
            ._mm512_cvtepu8_epi16(simd.avx512f._mm512_extracti64x4_epi64::<0>(va));
        let vb_lo16 = simd
            .avx512bw
            ._mm512_cvtepu8_epi16(simd.avx512f._mm512_extracti64x4_epi64::<0>(vb));
        let va_hi16 = simd
            .avx512bw
            ._mm512_cvtepu8_epi16(simd.avx512f._mm512_extracti64x4_epi64::<1>(va));
        let vb_hi16 = simd
            .avx512bw
            ._mm512_cvtepu8_epi16(simd.avx512f._mm512_extracti64x4_epi64::<1>(vb));

        // madd_epi16: 32xi16 * 32xi16 -> 16xi32 (adjacent pairs summed)
        let prod_lo = simd.avx512bw._mm512_madd_epi16(va_lo16, vb_lo16);
        let prod_hi = simd.avx512bw._mm512_madd_epi16(va_hi16, vb_hi16);
        acc32 = simd.avx512f._mm512_add_epi32(acc32, prod_lo);
        acc32 = simd.avx512f._mm512_add_epi32(acc32, prod_hi);
    }

    let mut result = simd.avx512f._mm512_reduce_add_epi32(acc32) as u32;

    // Scalar tail
    for (&x, &y) in a_tail.iter().zip(b_tail) {
        result = result.wrapping_add(x as u32 * y as u32);
    }

    result
}

/// AVX2 Hamming distance using VPSHUFB nibble-LUT popcount.
///
/// XORs 32 bytes at a time, then counts set bits using the Muła/Harley-Seal
/// VPSHUFB-based popcount: split each byte into high/low nibbles, look up a
/// 16-entry table, sum the counts.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn hamming_avx2(simd: V3, a: &[u8], b: &[u8]) -> u32 {
    simd.vectorize(|| hamming_avx2_in(simd, a, b))
}

#[inline(always)]
fn hamming_avx2_in(simd: V3, a: &[u8], b: &[u8]) -> u32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0;
    }

    let (a32, a_tail) = a[..n].as_chunks::<32>();
    let (b32, b_tail) = b[..n].as_chunks::<32>();

    // Popcount LUT: popcount of nibble 0..=15
    let lut = simd.avx._mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, // lo 128-bit lane
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, // hi 128-bit lane (same)
    );
    let lo_mask = simd.avx._mm256_set1_epi8(0x0F_u8 as i8);

    let mut acc: __m256i = simd.avx._mm256_setzero_si256();

    for (qa, qb) in a32.iter().zip(b32) {
        let xored = simd.avx2._mm256_xor_si256(ld256_si(qa), ld256_si(qb));

        // Count bits: popcount via nibble LUT
        let lo_nibbles = simd.avx2._mm256_and_si256(xored, lo_mask);
        let hi_nibbles = simd
            .avx2
            ._mm256_and_si256(simd.avx2._mm256_srli_epi16::<4>(xored), lo_mask);
        let lo_cnt = simd.avx2._mm256_shuffle_epi8(lut, lo_nibbles);
        let hi_cnt = simd.avx2._mm256_shuffle_epi8(lut, hi_nibbles);

        // SAD (sum of absolute differences vs 0) accumulates byte-wise counts into u64s
        let byte_cnt = simd.avx2._mm256_add_epi64(
            simd.avx2
                ._mm256_sad_epu8(lo_cnt, simd.avx._mm256_setzero_si256()),
            simd.avx2
                ._mm256_sad_epu8(hi_cnt, simd.avx._mm256_setzero_si256()),
        );
        acc = simd.avx2._mm256_add_epi64(acc, byte_cnt);
    }

    // Horizontal sum of 4 x i64 (extract requires const index)
    let lanes: [u64; 4] = pulp::cast(acc);
    let mut result: u32 = (lanes[0] as u32)
        .wrapping_add(lanes[1] as u32)
        .wrapping_add(lanes[2] as u32)
        .wrapping_add(lanes[3] as u32);

    // Scalar tail
    for (&x, &y) in a_tail.iter().zip(b_tail) {
        result += (x ^ y).count_ones();
    }

    result
}

/// AVX-512BW Hamming distance: the AVX2 nibble-LUT popcount widened to zmm.
///
/// XORs 64 bytes at a time, looks up the popcount of each nibble with
/// `vpshufb`, and sums bytes with `vpsadbw`. pulp has no safe wrapper for
/// VPOPCNTDQ's `_mm512_popcnt_epi64`, which would need 3 ops per 64 bytes
/// instead of 9.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn hamming_avx512(simd: V4, a: &[u8], b: &[u8]) -> u32 {
    simd.vectorize(|| hamming_avx512_in(simd, a, b))
}

#[inline(always)]
fn hamming_avx512_in(simd: V4, a: &[u8], b: &[u8]) -> u32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0;
    }

    let (a64, a_tail) = a[..n].as_chunks::<64>();
    let (b64, b_tail) = b[..n].as_chunks::<64>();
    // Popcount LUT: popcount of nibble 0..=15, repeated in each 128-bit lane.
    let lut: __m512i = pulp::cast(NIBBLE_POPCNT_LUT64);
    let lo_mask = simd.avx512f._mm512_set1_epi8(0x0F);
    let zero = simd.avx512f._mm512_setzero_si512();
    let mut acc: __m512i = simd.avx512f._mm512_setzero_si512();

    for (qa, qb) in a64.iter().zip(b64) {
        let xored = simd.avx512f._mm512_xor_si512(ld512_si(qa), ld512_si(qb));
        let lo_nibbles = simd.avx512f._mm512_and_si512(xored, lo_mask);
        let hi_nibbles = simd
            .avx512f
            ._mm512_and_si512(simd.avx512bw._mm512_srli_epi16::<4>(xored), lo_mask);
        let byte_cnt = simd.avx512f._mm512_add_epi64(
            simd.avx512bw
                ._mm512_sad_epu8(simd.avx512bw._mm512_shuffle_epi8(lut, lo_nibbles), zero),
            simd.avx512bw
                ._mm512_sad_epu8(simd.avx512bw._mm512_shuffle_epi8(lut, hi_nibbles), zero),
        );
        acc = simd.avx512f._mm512_add_epi64(acc, byte_cnt);
    }

    let mut result = simd.avx512f._mm512_reduce_add_epi64(acc) as u32;

    // Scalar tail
    for (&x, &y) in a_tail.iter().zip(b_tail) {
        result += (x ^ y).count_ones();
    }

    result
}

/// Integer-slot Hamming distance over `u32` slots: count of positions where
/// `a[i] != b[i]`. AVX-512F path (16 lanes per 512-bit register).
#[inline]
pub fn slot_hamming_u32_avx512(simd: V4, a: &[u32], b: &[u32]) -> u32 {
    simd.vectorize(|| slot_hamming_u32_avx512_in(simd, a, b))
}

#[inline(always)]
fn slot_hamming_u32_avx512_in(simd: V4, a: &[u32], b: &[u32]) -> u32 {
    let n = a.len().min(b.len());
    let (a16, a_tail) = a[..n].as_chunks::<16>();
    let (b16, b_tail) = b[..n].as_chunks::<16>();
    let mut diff: u32 = 0;

    for (qa, qb) in a16.iter().zip(b16) {
        // Bit j set = lane j equal. Differing lanes = 16 - popcount(equal mask).
        let eq_mask = simd
            .avx512f
            ._mm512_cmpeq_epi32_mask(ld512_si(qa), ld512_si(qb));
        diff += 16 - eq_mask.count_ones();
    }

    // Scalar tail
    for (x, y) in a_tail.iter().zip(b_tail) {
        diff += u32::from(x != y);
    }

    diff
}

/// Integer-slot Hamming distance over `u32` slots. AVX2 path (8 lanes per
/// 256-bit register).
#[inline]
pub fn slot_hamming_u32_avx2(simd: V3, a: &[u32], b: &[u32]) -> u32 {
    simd.vectorize(|| slot_hamming_u32_avx2_in(simd, a, b))
}

#[inline(always)]
fn slot_hamming_u32_avx2_in(simd: V3, a: &[u32], b: &[u32]) -> u32 {
    let n = a.len().min(b.len());
    let (a8, a_tail) = a[..n].as_chunks::<8>();
    let (b8, b_tail) = b[..n].as_chunks::<8>();
    let mut diff: u32 = 0;

    for (qa, qb) in a8.iter().zip(b8) {
        // cmpeq -> 0xFFFFFFFF per equal lane; movemask_ps -> one bit per lane.
        let cmp = simd.avx2._mm256_cmpeq_epi32(ld256_si(qa), ld256_si(qb));
        let eq_mask = simd
            .avx
            ._mm256_movemask_ps(simd.avx._mm256_castsi256_ps(cmp)) as u32;
        diff += 8 - (eq_mask & 0xFF).count_ones();
    }

    // Scalar tail
    for (x, y) in a_tail.iter().zip(b_tail) {
        diff += u32::from(x != y);
    }

    diff
}

// ---------------------------------------------------------------------------
// u16 integer-slot Hamming: count differing u16 slots (b-bit MinHash at b=16,
// the SIMD-friendly truncation width). AVX-512BW: 32 slots/zmm via epi16 mask;
// AVX2: 16/ymm via epi16 compare + movemask_epi8 (2 mask bits per lane).
// ---------------------------------------------------------------------------

#[inline]
pub fn slot_hamming_u16_avx512(simd: V4, a: &[u16], b: &[u16]) -> u32 {
    simd.vectorize(|| slot_hamming_u16_avx512_in(simd, a, b))
}

#[inline(always)]
fn slot_hamming_u16_avx512_in(simd: V4, a: &[u16], b: &[u16]) -> u32 {
    let n = a.len().min(b.len());
    let (a32, a_tail) = a[..n].as_chunks::<32>();
    let (b32, b_tail) = b[..n].as_chunks::<32>();
    let mut diff: u32 = 0;
    for (qa, qb) in a32.iter().zip(b32) {
        // __mmask32: bit j = lane j equal; differing = 32 - popcount.
        diff += 32
            - simd
                .avx512bw
                ._mm512_cmpeq_epi16_mask(ld512_si(qa), ld512_si(qb))
                .count_ones();
    }
    for (x, y) in a_tail.iter().zip(b_tail) {
        diff += u32::from(x != y);
    }
    diff
}

#[inline]
pub fn slot_hamming_u16_avx2(simd: V3, a: &[u16], b: &[u16]) -> u32 {
    simd.vectorize(|| slot_hamming_u16_avx2_in(simd, a, b))
}

#[inline(always)]
fn slot_hamming_u16_avx2_in(simd: V3, a: &[u16], b: &[u16]) -> u32 {
    let n = a.len().min(b.len());
    let (a16, a_tail) = a[..n].as_chunks::<16>();
    let (b16, b_tail) = b[..n].as_chunks::<16>();
    let mut diff: u32 = 0;
    for (qa, qb) in a16.iter().zip(b16) {
        let (va, vb) = (ld256_si(qa), ld256_si(qb));
        // Each equal epi16 lane sets 2 bits in the byte movemask, so
        // equal lanes = popcount / 2; differing = 16 - equal.
        let eq_bytes = (simd
            .avx2
            ._mm256_movemask_epi8(simd.avx2._mm256_cmpeq_epi16(va, vb))
            as u32)
            .count_ones();
        diff += 16 - eq_bytes / 2;
    }
    for (x, y) in a_tail.iter().zip(b_tail) {
        diff += u32::from(x != y);
    }
    diff
}

// ---------------------------------------------------------------------------
// u64 integer-slot Hamming: count differing u64 slots (probminhash u64
// sketches). AVX-512: 8 slots/zmm via epi64 mask compare; AVX2: 4/ymm via
// epi64 compare + movemask_pd. Mirror slot_hamming_u32.
// ---------------------------------------------------------------------------

#[inline]
pub fn slot_hamming_u64_avx512(simd: V4, a: &[u64], b: &[u64]) -> u64 {
    simd.vectorize(|| slot_hamming_u64_avx512_in(simd, a, b))
}

#[inline(always)]
fn slot_hamming_u64_avx512_in(simd: V4, a: &[u64], b: &[u64]) -> u64 {
    let n = a.len().min(b.len());
    let (a8, a_tail) = a[..n].as_chunks::<8>();
    let (b8, b_tail) = b[..n].as_chunks::<8>();
    let mut diff: u64 = 0;
    for (qa, qb) in a8.iter().zip(b8) {
        // 8-bit mask, bit j = lane j equal; differing = 8 - popcount.
        let eq = simd
            .avx512f
            ._mm512_cmpeq_epi64_mask(ld512_si(qa), ld512_si(qb));
        diff += u64::from(8 - eq.count_ones());
    }
    for (x, y) in a_tail.iter().zip(b_tail) {
        diff += u64::from(x != y);
    }
    diff
}

#[inline]
pub fn slot_hamming_u64_avx2(simd: V3, a: &[u64], b: &[u64]) -> u64 {
    simd.vectorize(|| slot_hamming_u64_avx2_in(simd, a, b))
}

#[inline(always)]
fn slot_hamming_u64_avx2_in(simd: V3, a: &[u64], b: &[u64]) -> u64 {
    let n = a.len().min(b.len());
    let (a4, a_tail) = a[..n].as_chunks::<4>();
    let (b4, b_tail) = b[..n].as_chunks::<4>();
    let mut diff: u64 = 0;
    for (qa, qb) in a4.iter().zip(b4) {
        let (va, vb) = (ld256_si(qa), ld256_si(qb));
        // Equal lanes become all-ones (sign bit set); movemask_pd gives a
        // 4-bit equal mask, so differing = 4 - popcount.
        let eq = simd.avx._mm256_movemask_pd(
            simd.avx
                ._mm256_castsi256_pd(simd.avx2._mm256_cmpeq_epi64(va, vb)),
        );
        diff += u64::from(4 - (eq as u32).count_ones());
    }
    for (x, y) in a_tail.iter().zip(b_tail) {
        diff += u64::from(x != y);
    }
    diff
}

// ---------------------------------------------------------------------------
// f64 reductions. AVX-512: 8 doubles/zmm, masked tail (no scalar remainder).
// AVX2: 4 doubles/ymm + extract-add cascade. Mirror the f32 kernels.
// ---------------------------------------------------------------------------

#[inline]
pub fn dot_f64_avx512(simd: V4, a: &[f64], b: &[f64]) -> f64 {
    simd.vectorize(|| dot_f64_avx512_in(simd, a, b))
}

#[inline(always)]
fn dot_f64_avx512_in(simd: V4, a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (a_vecs, a_tail) = a[..n].as_chunks::<8>();
    let (b_vecs, b_tail) = b[..n].as_chunks::<8>();
    let (a_quads, a_rem) = a_vecs.as_chunks::<4>();
    let (b_quads, b_rem) = b_vecs.as_chunks::<4>();
    let mut s0: __m512d = simd.avx512f._mm512_setzero_pd();
    let mut s1: __m512d = simd.avx512f._mm512_setzero_pd();
    let mut s2: __m512d = simd.avx512f._mm512_setzero_pd();
    let mut s3: __m512d = simd.avx512f._mm512_setzero_pd();
    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a_quads.iter().zip(b_quads) {
        s0 = simd
            .avx512f
            ._mm512_fmadd_pd(ld512_f64(a0), ld512_f64(b0), s0);
        s1 = simd
            .avx512f
            ._mm512_fmadd_pd(ld512_f64(a1), ld512_f64(b1), s1);
        s2 = simd
            .avx512f
            ._mm512_fmadd_pd(ld512_f64(a2), ld512_f64(b2), s2);
        s3 = simd
            .avx512f
            ._mm512_fmadd_pd(ld512_f64(a3), ld512_f64(b3), s3);
    }
    let mut acc = simd.avx512f._mm512_add_pd(
        simd.avx512f._mm512_add_pd(s0, s1),
        simd.avx512f._mm512_add_pd(s2, s3),
    );
    for (qa, qb) in a_rem.iter().zip(b_rem) {
        acc = simd
            .avx512f
            ._mm512_fmadd_pd(ld512_f64(qa), ld512_f64(qb), acc);
    }
    if !a_tail.is_empty() {
        let va = ld512_f64_tail(a_tail);
        let vb = ld512_f64_tail(b_tail);
        acc = simd.avx512f._mm512_fmadd_pd(va, vb, acc);
    }
    simd.avx512f._mm512_reduce_add_pd(acc)
}

#[inline]
pub fn l2_squared_f64_avx512(simd: V4, a: &[f64], b: &[f64]) -> f64 {
    simd.vectorize(|| l2_squared_f64_avx512_in(simd, a, b))
}

#[inline(always)]
fn l2_squared_f64_avx512_in(simd: V4, a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (a_vecs, a_tail) = a[..n].as_chunks::<8>();
    let (b_vecs, b_tail) = b[..n].as_chunks::<8>();
    let (a_quads, a_rem) = a_vecs.as_chunks::<4>();
    let (b_quads, b_rem) = b_vecs.as_chunks::<4>();
    let mut s0: __m512d = simd.avx512f._mm512_setzero_pd();
    let mut s1: __m512d = simd.avx512f._mm512_setzero_pd();
    let mut s2: __m512d = simd.avx512f._mm512_setzero_pd();
    let mut s3: __m512d = simd.avx512f._mm512_setzero_pd();
    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a_quads.iter().zip(b_quads) {
        let d0 = simd.avx512f._mm512_sub_pd(ld512_f64(a0), ld512_f64(b0));
        let d1 = simd.avx512f._mm512_sub_pd(ld512_f64(a1), ld512_f64(b1));
        let d2 = simd.avx512f._mm512_sub_pd(ld512_f64(a2), ld512_f64(b2));
        let d3 = simd.avx512f._mm512_sub_pd(ld512_f64(a3), ld512_f64(b3));
        s0 = simd.avx512f._mm512_fmadd_pd(d0, d0, s0);
        s1 = simd.avx512f._mm512_fmadd_pd(d1, d1, s1);
        s2 = simd.avx512f._mm512_fmadd_pd(d2, d2, s2);
        s3 = simd.avx512f._mm512_fmadd_pd(d3, d3, s3);
    }
    let mut acc = simd.avx512f._mm512_add_pd(
        simd.avx512f._mm512_add_pd(s0, s1),
        simd.avx512f._mm512_add_pd(s2, s3),
    );
    for (qa, qb) in a_rem.iter().zip(b_rem) {
        let d = simd.avx512f._mm512_sub_pd(ld512_f64(qa), ld512_f64(qb));
        acc = simd.avx512f._mm512_fmadd_pd(d, d, acc);
    }
    if !a_tail.is_empty() {
        let d = simd
            .avx512f
            ._mm512_sub_pd(ld512_f64_tail(a_tail), ld512_f64_tail(b_tail));
        acc = simd.avx512f._mm512_fmadd_pd(d, d, acc);
    }
    simd.avx512f._mm512_reduce_add_pd(acc)
}

#[inline]
pub fn l1_f64_avx512(simd: V4, a: &[f64], b: &[f64]) -> f64 {
    simd.vectorize(|| l1_f64_avx512_in(simd, a, b))
}

#[inline(always)]
fn l1_f64_avx512_in(simd: V4, a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (a_vecs, a_tail) = a[..n].as_chunks::<8>();
    let (b_vecs, b_tail) = b[..n].as_chunks::<8>();
    let (a_quads, a_rem) = a_vecs.as_chunks::<4>();
    let (b_quads, b_rem) = b_vecs.as_chunks::<4>();
    let mut s0: __m512d = simd.avx512f._mm512_setzero_pd();
    let mut s1: __m512d = simd.avx512f._mm512_setzero_pd();
    let mut s2: __m512d = simd.avx512f._mm512_setzero_pd();
    let mut s3: __m512d = simd.avx512f._mm512_setzero_pd();
    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a_quads.iter().zip(b_quads) {
        s0 = simd.avx512f._mm512_add_pd(
            s0,
            simd.avx512f
                ._mm512_abs_pd(simd.avx512f._mm512_sub_pd(ld512_f64(a0), ld512_f64(b0))),
        );
        s1 = simd.avx512f._mm512_add_pd(
            s1,
            simd.avx512f
                ._mm512_abs_pd(simd.avx512f._mm512_sub_pd(ld512_f64(a1), ld512_f64(b1))),
        );
        s2 = simd.avx512f._mm512_add_pd(
            s2,
            simd.avx512f
                ._mm512_abs_pd(simd.avx512f._mm512_sub_pd(ld512_f64(a2), ld512_f64(b2))),
        );
        s3 = simd.avx512f._mm512_add_pd(
            s3,
            simd.avx512f
                ._mm512_abs_pd(simd.avx512f._mm512_sub_pd(ld512_f64(a3), ld512_f64(b3))),
        );
    }
    let mut acc = simd.avx512f._mm512_add_pd(
        simd.avx512f._mm512_add_pd(s0, s1),
        simd.avx512f._mm512_add_pd(s2, s3),
    );
    for (qa, qb) in a_rem.iter().zip(b_rem) {
        let d = simd.avx512f._mm512_sub_pd(ld512_f64(qa), ld512_f64(qb));
        acc = simd
            .avx512f
            ._mm512_add_pd(acc, simd.avx512f._mm512_abs_pd(d));
    }
    if !a_tail.is_empty() {
        let d = simd
            .avx512f
            ._mm512_sub_pd(ld512_f64_tail(a_tail), ld512_f64_tail(b_tail));
        acc = simd
            .avx512f
            ._mm512_add_pd(acc, simd.avx512f._mm512_abs_pd(d));
    }
    simd.avx512f._mm512_reduce_add_pd(acc)
}

/// Horizontal sum of a `__m256d` (4 doubles).
#[inline(always)]
fn hsum256_pd(simd: V3, v: __m256d) -> f64 {
    let lo = simd.avx._mm256_castpd256_pd128(v);
    let hi = simd.avx._mm256_extractf128_pd::<1>(v);
    let sum = simd.sse2._mm_add_pd(lo, hi);
    simd.sse2._mm_cvtsd_f64(
        simd.sse2
            ._mm_add_pd(sum, simd.sse2._mm_unpackhi_pd(sum, sum)),
    )
}

#[inline]
pub fn dot_f64_avx2(simd: V3, a: &[f64], b: &[f64]) -> f64 {
    simd.vectorize(|| dot_f64_avx2_in(simd, a, b))
}

#[inline(always)]
fn dot_f64_avx2_in(simd: V3, a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (a_vecs, a_tail) = a[..n].as_chunks::<4>();
    let (b_vecs, b_tail) = b[..n].as_chunks::<4>();
    let (a_quads, a_rem) = a_vecs.as_chunks::<4>();
    let (b_quads, b_rem) = b_vecs.as_chunks::<4>();
    let mut s0: __m256d = simd.avx._mm256_setzero_pd();
    let mut s1: __m256d = simd.avx._mm256_setzero_pd();
    let mut s2: __m256d = simd.avx._mm256_setzero_pd();
    let mut s3: __m256d = simd.avx._mm256_setzero_pd();
    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a_quads.iter().zip(b_quads) {
        s0 = simd.fma._mm256_fmadd_pd(ld256_f64(a0), ld256_f64(b0), s0);
        s1 = simd.fma._mm256_fmadd_pd(ld256_f64(a1), ld256_f64(b1), s1);
        s2 = simd.fma._mm256_fmadd_pd(ld256_f64(a2), ld256_f64(b2), s2);
        s3 = simd.fma._mm256_fmadd_pd(ld256_f64(a3), ld256_f64(b3), s3);
    }
    let acc = simd.avx._mm256_add_pd(
        simd.avx._mm256_add_pd(s0, s1),
        simd.avx._mm256_add_pd(s2, s3),
    );
    let mut result = hsum256_pd(simd, acc);
    for (qa, qb) in a_rem.iter().zip(b_rem) {
        result += hsum256_pd(
            simd,
            simd.fma
                ._mm256_fmadd_pd(ld256_f64(qa), ld256_f64(qb), simd.avx._mm256_setzero_pd()),
        );
    }
    for (x, y) in a_tail.iter().zip(b_tail) {
        result += x * y;
    }
    result
}

#[inline]
pub fn l2_squared_f64_avx2(simd: V3, a: &[f64], b: &[f64]) -> f64 {
    simd.vectorize(|| l2_squared_f64_avx2_in(simd, a, b))
}

#[inline(always)]
fn l2_squared_f64_avx2_in(simd: V3, a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (a_vecs, a_tail) = a[..n].as_chunks::<4>();
    let (b_vecs, b_tail) = b[..n].as_chunks::<4>();
    let (a_quads, a_rem) = a_vecs.as_chunks::<4>();
    let (b_quads, b_rem) = b_vecs.as_chunks::<4>();
    let mut s0: __m256d = simd.avx._mm256_setzero_pd();
    let mut s1: __m256d = simd.avx._mm256_setzero_pd();
    let mut s2: __m256d = simd.avx._mm256_setzero_pd();
    let mut s3: __m256d = simd.avx._mm256_setzero_pd();
    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a_quads.iter().zip(b_quads) {
        let d0 = simd.avx._mm256_sub_pd(ld256_f64(a0), ld256_f64(b0));
        let d1 = simd.avx._mm256_sub_pd(ld256_f64(a1), ld256_f64(b1));
        let d2 = simd.avx._mm256_sub_pd(ld256_f64(a2), ld256_f64(b2));
        let d3 = simd.avx._mm256_sub_pd(ld256_f64(a3), ld256_f64(b3));
        s0 = simd.fma._mm256_fmadd_pd(d0, d0, s0);
        s1 = simd.fma._mm256_fmadd_pd(d1, d1, s1);
        s2 = simd.fma._mm256_fmadd_pd(d2, d2, s2);
        s3 = simd.fma._mm256_fmadd_pd(d3, d3, s3);
    }
    let acc = simd.avx._mm256_add_pd(
        simd.avx._mm256_add_pd(s0, s1),
        simd.avx._mm256_add_pd(s2, s3),
    );
    let mut result = hsum256_pd(simd, acc);
    for (qa, qb) in a_rem.iter().zip(b_rem) {
        let d = simd.avx._mm256_sub_pd(ld256_f64(qa), ld256_f64(qb));
        result += hsum256_pd(
            simd,
            simd.fma._mm256_fmadd_pd(d, d, simd.avx._mm256_setzero_pd()),
        );
    }
    for (x, y) in a_tail.iter().zip(b_tail) {
        let d = x - y;
        result += d * d;
    }
    result
}

#[inline]
pub fn l1_f64_avx2(simd: V3, a: &[f64], b: &[f64]) -> f64 {
    simd.vectorize(|| l1_f64_avx2_in(simd, a, b))
}

#[inline(always)]
fn l1_f64_avx2_in(simd: V3, a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (a_vecs, a_tail) = a[..n].as_chunks::<4>();
    let (b_vecs, b_tail) = b[..n].as_chunks::<4>();
    let (a_quads, a_rem) = a_vecs.as_chunks::<4>();
    let (b_quads, b_rem) = b_vecs.as_chunks::<4>();
    // abs via andnot with the sign-bit mask.
    let signmask = simd.avx._mm256_set1_pd(-0.0);
    let absdiff = |x, y| {
        simd.avx
            ._mm256_andnot_pd(signmask, simd.avx._mm256_sub_pd(x, y))
    };
    let mut s0: __m256d = simd.avx._mm256_setzero_pd();
    let mut s1: __m256d = simd.avx._mm256_setzero_pd();
    let mut s2: __m256d = simd.avx._mm256_setzero_pd();
    let mut s3: __m256d = simd.avx._mm256_setzero_pd();
    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a_quads.iter().zip(b_quads) {
        s0 = simd
            .avx
            ._mm256_add_pd(s0, absdiff(ld256_f64(a0), ld256_f64(b0)));
        s1 = simd
            .avx
            ._mm256_add_pd(s1, absdiff(ld256_f64(a1), ld256_f64(b1)));
        s2 = simd
            .avx
            ._mm256_add_pd(s2, absdiff(ld256_f64(a2), ld256_f64(b2)));
        s3 = simd
            .avx
            ._mm256_add_pd(s3, absdiff(ld256_f64(a3), ld256_f64(b3)));
    }
    let acc = simd.avx._mm256_add_pd(
        simd.avx._mm256_add_pd(s0, s1),
        simd.avx._mm256_add_pd(s2, s3),
    );
    let mut result = hsum256_pd(simd, acc);
    for (qa, qb) in a_rem.iter().zip(b_rem) {
        result += hsum256_pd(simd, absdiff(ld256_f64(qa), ld256_f64(qb)));
    }
    for (x, y) in a_tail.iter().zip(b_tail) {
        result += (x - y).abs();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every x86 kernel against a scalar reference at odd and boundary
    /// lengths. Each ISA group runs only where the CPU reports it (CI runs the
    /// AVX-512 groups under SDE). 64 * 4 + 33 covers the 4-way unrolled body,
    /// the single-vector remainder, and the scalar or masked tail together.
    #[test]
    fn x86_kernels_match_scalar_on_odd_lengths() {
        const LENS: [usize; 16] = [
            0,
            1,
            2,
            3,
            7,
            8,
            9,
            15,
            16,
            17,
            31,
            33,
            63,
            65,
            127,
            64 * 4 + 33,
        ];
        let close = |got: f64, want: f64, tol: f64| (got - want).abs() <= tol * want.abs().max(1.0);
        let f = |x: f32| f64::from(x);

        for n in LENS {
            let fa: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) as f32).sin()).collect();
            let fb: Vec<f32> = (0..n).map(|i| ((i * 11 + 5) as f32).cos()).collect();
            let da: Vec<f64> = fa.iter().map(|&x| f64::from(x) * 1.5).collect();
            let db: Vec<f64> = fb.iter().map(|&x| f64::from(x) - 0.25).collect();
            let ua: Vec<u8> = (0..n).map(|i| (i * 37 + 11) as u8).collect();
            let ub: Vec<u8> = (0..n).map(|i| (i * 101 + 7) as u8).collect();
            let s32a: Vec<u32> = (0..n as u32).collect();
            let s32b: Vec<u32> = (0..n as u32)
                .map(|i| if i % 3 == 0 { i + 1 } else { i })
                .collect();
            let s16a: Vec<u16> = s32a.iter().map(|&x| x as u16).collect();
            let s16b: Vec<u16> = s32b.iter().map(|&x| x as u16).collect();
            let s64a: Vec<u64> = s32a.iter().map(|&x| u64::from(x) << 33).collect();
            let s64b: Vec<u64> = s32b.iter().map(|&x| u64::from(x) << 33).collect();

            let dot: f64 = fa.iter().zip(&fb).map(|(&x, &y)| f(x) * f(y)).sum();
            let aa: f64 = fa.iter().map(|&x| f(x) * f(x)).sum();
            let bb: f64 = fb.iter().map(|&y| f(y) * f(y)).sum();
            let l2: f64 = fa
                .iter()
                .zip(&fb)
                .map(|(&x, &y)| (f(x) - f(y)).powi(2))
                .sum();
            let l1: f64 = fa.iter().zip(&fb).map(|(&x, &y)| (f(x) - f(y)).abs()).sum();
            let cos = if n == 0 {
                0.0
            } else {
                dot / (aa.sqrt() * bb.sqrt())
            };
            let dot_u8f: f64 = fa.iter().zip(&ub).map(|(&x, &y)| f(x) * f64::from(y)).sum();
            let dot_u8: u64 = ua
                .iter()
                .zip(&ub)
                .map(|(&x, &y)| u64::from(x) * u64::from(y))
                .sum();
            let ham: u32 = ua
                .iter()
                .zip(&ub)
                .map(|(&x, &y)| (x ^ y).count_ones())
                .sum();
            let slot = s32a.iter().zip(&s32b).filter(|(x, y)| x != y).count();
            let slot16 = s16a.iter().zip(&s16b).filter(|(x, y)| x != y).count();
            let ddot: f64 = da.iter().zip(&db).map(|(x, y)| x * y).sum();
            let dl2: f64 = da.iter().zip(&db).map(|(x, y)| (x - y) * (x - y)).sum();
            let dl1: f64 = da.iter().zip(&db).map(|(x, y)| (x - y).abs()).sum();

            // MaxSim: min(n, 9) query tokens of dim 33 against 3 doc tokens.
            let toks = |seed: usize, count: usize| -> Vec<Vec<f32>> {
                (0..count)
                    .map(|t| {
                        (0..33)
                            .map(|i| ((i * 13 + t * 7 + seed) as f32).sin())
                            .collect()
                    })
                    .collect()
            };
            let q = toks(1, n.min(9));
            let d = toks(2, 3);
            let qr: Vec<&[f32]> = q.iter().map(Vec::as_slice).collect();
            let dr: Vec<&[f32]> = d.iter().map(Vec::as_slice).collect();
            let maxsim: f64 = qr
                .iter()
                .map(|q| {
                    dr.iter()
                        .map(|d| q.iter().zip(*d).map(|(&x, &y)| f(x) * f(y)).sum::<f64>())
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .sum();

            let t = 1e-4;
            if let Some(v3) = V3::try_new() {
                // The V3 token enables the AVX2/FMA kernels.
                {
                    assert!(close(f(dot_avx2(v3, &fa, &fb)), dot, t), "dot_avx2 n={n}");
                    assert!(
                        close(f(l2_squared_avx2(v3, &fa, &fb)), l2, t),
                        "l2_avx2 n={n}"
                    );
                    assert!(close(f(l1_avx2(v3, &fa, &fb)), l1, t), "l1_avx2 n={n}");
                    assert!(
                        close(f(cosine_avx2(v3, &fa, &fb)), cos, t),
                        "cos_avx2 n={n}"
                    );
                    let fc = crate::fast_math::x86_64::fast_cosine_avx2(v3, &fa, &fb);
                    assert!(close(f(fc), cos, 2e-3), "fast_cos_avx2 n={n}");
                    let got = dot_u8_f32_avx2(v3, &fa, &ub);
                    assert!(close(f(got), dot_u8f, t), "dot_u8_f32_avx2 n={n}");
                    assert_eq!(
                        u64::from(dot_u8_avx2(v3, &ua, &ub)),
                        dot_u8,
                        "dot_u8_avx2 n={n}"
                    );
                    assert_eq!(hamming_avx2(v3, &ua, &ub), ham, "hamming_avx2 n={n}");
                    assert_eq!(
                        slot_hamming_u32_avx2(v3, &s32a, &s32b) as usize,
                        slot,
                        "s32 n={n}"
                    );
                    assert_eq!(
                        slot_hamming_u16_avx2(v3, &s16a, &s16b) as usize,
                        slot16,
                        "s16 n={n}"
                    );
                    assert_eq!(
                        slot_hamming_u64_avx2(v3, &s64a, &s64b) as usize,
                        slot,
                        "s64 n={n}"
                    );
                    assert!(
                        close(dot_f64_avx2(v3, &da, &db), ddot, 1e-12),
                        "dot_f64_avx2 n={n}"
                    );
                    assert!(
                        close(l2_squared_f64_avx2(v3, &da, &db), dl2, 1e-12),
                        "l2f64 n={n}"
                    );
                    assert!(
                        close(l1_f64_avx2(v3, &da, &db), dl1, 1e-12),
                        "l1_f64_avx2 n={n}"
                    );
                    assert!(
                        close(f(maxsim_avx2(v3, &qr, &dr)), maxsim, t),
                        "maxsim_avx2 n={n}"
                    );
                }
            }
            if let Some(v4) = V4::try_new() {
                // The V4 token enables the AVX-512 kernels.
                {
                    assert!(
                        close(f(dot_avx512(v4, &fa, &fb)), dot, t),
                        "dot_avx512 n={n}"
                    );
                    assert!(
                        close(f(l2_squared_avx512(v4, &fa, &fb)), l2, t),
                        "l2_avx512 n={n}"
                    );
                    assert!(close(f(l1_avx512(v4, &fa, &fb)), l1, t), "l1_avx512 n={n}");
                    assert!(
                        close(f(cosine_avx512(v4, &fa, &fb)), cos, t),
                        "cos_avx512 n={n}"
                    );
                    let fc = crate::fast_math::x86_64::fast_cosine_avx512(v4, &fa, &fb);
                    assert!(close(f(fc), cos, 2e-3), "fast_cos_avx512 n={n}");
                    let got = slot_hamming_u32_avx512(v4, &s32a, &s32b) as usize;
                    assert_eq!(got, slot, "s32_avx512 n={n}");
                    let got = slot_hamming_u64_avx512(v4, &s64a, &s64b) as usize;
                    assert_eq!(got, slot, "s64_avx512 n={n}");
                    assert!(
                        close(dot_f64_avx512(v4, &da, &db), ddot, 1e-12),
                        "dot_f64 n={n}"
                    );
                    assert!(
                        close(l2_squared_f64_avx512(v4, &da, &db), dl2, 1e-12),
                        "l2f64 n={n}"
                    );
                    assert!(
                        close(l1_f64_avx512(v4, &da, &db), dl1, 1e-12),
                        "l1_f64 n={n}"
                    );
                    assert!(
                        close(f(maxsim_avx512(v4, &qr, &dr)), maxsim, t),
                        "maxsim512 n={n}"
                    );
                }
            }
            if let Some(v4) = V4::try_new() {
                // V4 includes AVX-512BW for byte and word operations.
                {
                    assert_eq!(
                        u64::from(dot_u8_avx512(v4, &ua, &ub)),
                        dot_u8,
                        "dot_u8_512 n={n}"
                    );
                    let got = slot_hamming_u16_avx512(v4, &s16a, &s16b) as usize;
                    assert_eq!(got, slot16, "s16_avx512 n={n}");
                }
            }
            if let Some(v4) = V4::try_new() {
                // The nibble-LUT popcount uses AVX-512BW, included in V4.
                assert_eq!(hamming_avx512(v4, &ua, &ub), ham, "hamming_512 n={n}");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_dot_avx512_correctness() {
        let Some(v4) = V4::try_new() else {
            eprintln!("AVX-512F not available, skipping test");
            return;
        };

        // Test various sizes around AVX-512 boundaries (16, 64)
        for size in [1, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 256, 512, 1024] {
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.2).collect();

            let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let actual = dot_avx512(v4, &a, &b);

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
        let Some(v3) = V3::try_new() else {
            eprintln!("AVX2+FMA not available, skipping test");
            return;
        };

        // Test various sizes around AVX2 boundaries (8, 32)
        for size in [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 128, 256, 512] {
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.2).collect();

            let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let actual = dot_avx2(v3, &a, &b);

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
    fn test_l2_squared_avx512_correctness() {
        let Some(v4) = V4::try_new() else {
            eprintln!("AVX-512F not available, skipping test");
            return;
        };

        for size in [1, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 256, 512, 1024] {
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.2).collect();

            let expected: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
            let actual = l2_squared_avx512(v4, &a, &b);

            let rel_error = if expected.abs() > 1e-6 {
                (actual - expected).abs() / expected.abs()
            } else {
                (actual - expected).abs()
            };
            assert!(
                rel_error < 1e-4,
                "AVX-512 L2 size={}: expected={}, actual={}, rel_error={}",
                size,
                expected,
                actual,
                rel_error
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_l2_squared_avx2_correctness() {
        let Some(v3) = V3::try_new() else {
            eprintln!("AVX2+FMA not available, skipping test");
            return;
        };

        for size in [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 128, 256, 512] {
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.2).collect();

            let expected: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
            let actual = l2_squared_avx2(v3, &a, &b);

            let rel_error = if expected.abs() > 1e-6 {
                (actual - expected).abs() / expected.abs()
            } else {
                (actual - expected).abs()
            };
            assert!(
                rel_error < 1e-5,
                "AVX2 L2 size={}: expected={}, actual={}, rel_error={}",
                size,
                expected,
                actual,
                rel_error
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_cosine_avx512_correctness() {
        let Some(v4) = V4::try_new() else {
            eprintln!("AVX-512F not available, skipping test");
            return;
        };

        for size in [1, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 256, 512, 1024] {
            let a: Vec<f32> = (0..size).map(|i| ((i * 7) as f32).sin()).collect();
            let b: Vec<f32> = (0..size).map(|i| ((i * 11) as f32).cos()).collect();

            let ab: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let aa: f32 = a.iter().map(|x| x * x).sum();
            let bb: f32 = b.iter().map(|x| x * x).sum();
            let expected = if aa > crate::NORM_EPSILON_SQ && bb > crate::NORM_EPSILON_SQ {
                ab / (aa.sqrt() * bb.sqrt())
            } else {
                0.0
            };

            let actual = cosine_avx512(v4, &a, &b);

            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-4,
                "AVX-512 cosine size={}: expected={}, actual={}, diff={}",
                size,
                expected,
                actual,
                diff
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_cosine_avx2_correctness() {
        let Some(v3) = V3::try_new() else {
            eprintln!("AVX2+FMA not available, skipping test");
            return;
        };

        for size in [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 128, 256, 512] {
            let a: Vec<f32> = (0..size).map(|i| ((i * 7) as f32).sin()).collect();
            let b: Vec<f32> = (0..size).map(|i| ((i * 11) as f32).cos()).collect();

            let ab: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let aa: f32 = a.iter().map(|x| x * x).sum();
            let bb: f32 = b.iter().map(|x| x * x).sum();
            let expected = if aa > crate::NORM_EPSILON_SQ && bb > crate::NORM_EPSILON_SQ {
                ab / (aa.sqrt() * bb.sqrt())
            } else {
                0.0
            };

            let actual = cosine_avx2(v3, &a, &b);

            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-5,
                "AVX2 cosine size={}: expected={}, actual={}, diff={}",
                size,
                expected,
                actual,
                diff
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_vs_avx512_consistency() {
        let (Some(v3), Some(v4)) = (V3::try_new(), V4::try_new()) else {
            eprintln!("Need both AVX2+FMA and AVX-512F, skipping");
            return;
        };

        // Both should produce nearly identical results
        for size in [64, 128, 256, 512, 1024] {
            let a: Vec<f32> = (0..size).map(|i| ((i * 7) as f32).sin()).collect();
            let b: Vec<f32> = (0..size).map(|i| ((i * 11) as f32).cos()).collect();

            let avx2_result = dot_avx2(v3, &a, &b);
            let avx512_result = dot_avx512(v4, &a, &b);

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
