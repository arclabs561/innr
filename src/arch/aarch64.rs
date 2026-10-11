//! aarch64 SIMD implementations using NEON.
//!
//! NEON is always available on aarch64, so no runtime detection needed.
//! However, we still use target_feature for consistency with x86_64.
//!
//! The kernels are safe `#[target_feature(enable = "neon")]` functions: NEON
//! value intrinsics are safe inside them, and every memory read goes through
//! the `ld_*` helpers below. Those take fixed-size array references (from
//! `as_chunks`), so the bound of each load is in the type and the helpers hold
//! the only `unsafe` in this module. Calling a kernel from a function without
//! `neon` enabled still needs `unsafe`; the dispatch sites carry that.
//!
//! # Performance Hierarchy
//!
//! | Implementation | Width | Throughput |
//! |----------------|-------|------------|
//! | dot_neon (4-way unrolled) | 4x4 f32 | ~4-8x vs scalar |
//! | Basic NEON | 4 f32 | ~2-4x vs scalar |

use std::arch::aarch64::{
    float32x4_t, float64x2_t, uint16x8_t, uint32x4_t, uint64x2_t, uint8x16_t, vld1q_f32, vld1q_f64,
    vld1q_u16, vld1q_u32, vld1q_u64, vld1q_u8,
};

// Loads. Each reads exactly one 128-bit q-register from an array of that
// size; vld1q only requires element alignment, which the reference has.

/// Load 4 f32 into one q-register.
#[inline]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
pub(crate) fn ld_f32(c: &[f32; 4]) -> float32x4_t {
    // SAFETY: `c` is 16 readable bytes, exactly what vld1q_f32 reads.
    unsafe { vld1q_f32(c.as_ptr()) }
}

/// Load 2 f64 into one q-register.
#[inline]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
fn ld_f64(c: &[f64; 2]) -> float64x2_t {
    // SAFETY: `c` is 16 readable bytes, exactly what vld1q_f64 reads.
    unsafe { vld1q_f64(c.as_ptr()) }
}

/// Load 16 u8 into one q-register.
#[inline]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
fn ld_u8(c: &[u8; 16]) -> uint8x16_t {
    // SAFETY: `c` is 16 readable bytes, exactly what vld1q_u8 reads.
    unsafe { vld1q_u8(c.as_ptr()) }
}

/// Load 8 u16 into one q-register.
#[inline]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
fn ld_u16(c: &[u16; 8]) -> uint16x8_t {
    // SAFETY: `c` is 16 readable bytes, exactly what vld1q_u16 reads.
    unsafe { vld1q_u16(c.as_ptr()) }
}

/// Load 4 u32 into one q-register.
#[inline]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
fn ld_u32(c: &[u32; 4]) -> uint32x4_t {
    // SAFETY: `c` is 16 readable bytes, exactly what vld1q_u32 reads.
    unsafe { vld1q_u32(c.as_ptr()) }
}

/// Load 2 u64 into one q-register.
#[inline]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
fn ld_u64(c: &[u64; 2]) -> uint64x2_t {
    // SAFETY: `c` is 16 readable bytes, exactly what vld1q_u64 reads.
    unsafe { vld1q_u64(c.as_ptr()) }
}

/// NEON dot product with 4-way unrolling.
///
/// Processes 16 floats per iteration (4 x 4), hiding memory latency.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32};

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let (a4, a_tail) = a[..n].as_chunks::<4>();
    let (b4, b_tail) = b[..n].as_chunks::<4>();
    let (a16, a_rem) = a4.as_chunks::<4>();
    let (b16, b_rem) = b4.as_chunks::<4>();

    // 4-way unrolled: process 16 floats per iteration
    let mut sum0: float32x4_t = vdupq_n_f32(0.0);
    let mut sum1: float32x4_t = vdupq_n_f32(0.0);
    let mut sum2: float32x4_t = vdupq_n_f32(0.0);
    let mut sum3: float32x4_t = vdupq_n_f32(0.0);

    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a16.iter().zip(b16) {
        sum0 = vfmaq_f32(sum0, ld_f32(a0), ld_f32(b0));
        sum1 = vfmaq_f32(sum1, ld_f32(a1), ld_f32(b1));
        sum2 = vfmaq_f32(sum2, ld_f32(a2), ld_f32(b2));
        sum3 = vfmaq_f32(sum3, ld_f32(a3), ld_f32(b3));
    }

    // Combine accumulators
    let sum01 = vaddq_f32(sum0, sum1);
    let sum23 = vaddq_f32(sum2, sum3);
    let sum_all = vaddq_f32(sum01, sum23);
    let mut result = vaddvq_f32(sum_all);

    // Handle remaining 4-float chunks
    let mut sum: float32x4_t = vdupq_n_f32(0.0);
    for (qa, qb) in a_rem.iter().zip(b_rem) {
        sum = vfmaq_f32(sum, ld_f32(qa), ld_f32(qb));
    }
    result += vaddvq_f32(sum);

    // Scalar tail
    for (x, y) in a_tail.iter().zip(b_tail) {
        result += x * y;
    }

    result
}

/// NEON MaxSim implementation.
///
/// Iterates over query tokens and computes max similarity against all doc tokens
/// using the dot_neon kernel directly.
///
/// `doc_tokens` must be non-empty; an empty slice causes `NEG_INFINITY`
/// accumulation.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn maxsim_neon(query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
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
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn l2_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vsubq_f32};

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let (a4, a_tail) = a[..n].as_chunks::<4>();
    let (b4, b_tail) = b[..n].as_chunks::<4>();
    let (a16, a_rem) = a4.as_chunks::<4>();
    let (b16, b_rem) = b4.as_chunks::<4>();

    // 4-way unrolled: process 16 floats per iteration
    let mut sum0: float32x4_t = vdupq_n_f32(0.0);
    let mut sum1: float32x4_t = vdupq_n_f32(0.0);
    let mut sum2: float32x4_t = vdupq_n_f32(0.0);
    let mut sum3: float32x4_t = vdupq_n_f32(0.0);

    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a16.iter().zip(b16) {
        let d0 = vsubq_f32(ld_f32(a0), ld_f32(b0));
        let d1 = vsubq_f32(ld_f32(a1), ld_f32(b1));
        let d2 = vsubq_f32(ld_f32(a2), ld_f32(b2));
        let d3 = vsubq_f32(ld_f32(a3), ld_f32(b3));

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
    let mut sum: float32x4_t = vdupq_n_f32(0.0);
    for (qa, qb) in a_rem.iter().zip(b_rem) {
        let d = vsubq_f32(ld_f32(qa), ld_f32(qb));
        sum = vfmaq_f32(sum, d, d);
    }
    result += vaddvq_f32(sum);

    // Scalar tail
    for (x, y) in a_tail.iter().zip(b_tail) {
        let d = x - y;
        result += d * d;
    }

    result
}

/// NEON L1 (Manhattan) distance: `Σ|a[i] - b[i]|`.
///
/// Uses `vabdq_f32` (absolute difference) for fused sub+abs in one instruction.
/// 4-way unrolled for pipeline efficiency.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn l1_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{vabdq_f32, vaddq_f32, vaddvq_f32, vdupq_n_f32};

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let (a4, a_tail) = a[..n].as_chunks::<4>();
    let (b4, b_tail) = b[..n].as_chunks::<4>();
    let (a16, a_rem) = a4.as_chunks::<4>();
    let (b16, b_rem) = b4.as_chunks::<4>();

    // 4-way unrolled: process 16 floats per iteration
    let mut sum0: float32x4_t = vdupq_n_f32(0.0);
    let mut sum1: float32x4_t = vdupq_n_f32(0.0);
    let mut sum2: float32x4_t = vdupq_n_f32(0.0);
    let mut sum3: float32x4_t = vdupq_n_f32(0.0);

    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a16.iter().zip(b16) {
        // vabdq_f32: absolute difference in one instruction
        sum0 = vaddq_f32(sum0, vabdq_f32(ld_f32(a0), ld_f32(b0)));
        sum1 = vaddq_f32(sum1, vabdq_f32(ld_f32(a1), ld_f32(b1)));
        sum2 = vaddq_f32(sum2, vabdq_f32(ld_f32(a2), ld_f32(b2)));
        sum3 = vaddq_f32(sum3, vabdq_f32(ld_f32(a3), ld_f32(b3)));
    }

    // Combine accumulators
    let sum01 = vaddq_f32(sum0, sum1);
    let sum23 = vaddq_f32(sum2, sum3);
    let sum_all = vaddq_f32(sum01, sum23);
    let mut result = vaddvq_f32(sum_all);

    // Handle remaining 4-float chunks
    let mut sum: float32x4_t = vdupq_n_f32(0.0);
    for (qa, qb) in a_rem.iter().zip(b_rem) {
        sum = vaddq_f32(sum, vabdq_f32(ld_f32(qa), ld_f32(qb)));
    }
    result += vaddvq_f32(sum);

    // Scalar tail
    for (x, y) in a_tail.iter().zip(b_tail) {
        result += (x - y).abs();
    }

    result
}

/// NEON fused cosine similarity: single-pass dot(a,b), norm(a)^2, norm(b)^2.
///
/// Accumulates all three products in one pass over memory, then uses
/// exact sqrt/div for IEEE-correct normalization. ~3x less memory bandwidth
/// than the 3-pass approach (dot + norm + norm).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn cosine_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32};

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let (a4, a_tail) = a[..n].as_chunks::<4>();
    let (b4, b_tail) = b[..n].as_chunks::<4>();
    let (a16, a_rem) = a4.as_chunks::<4>();
    let (b16, b_rem) = b4.as_chunks::<4>();

    // 4-way unrolled: process 16 floats per iteration, 3 accumulators each
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

    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a16.iter().zip(b16) {
        let (va0, vb0) = (ld_f32(a0), ld_f32(b0));
        let (va1, vb1) = (ld_f32(a1), ld_f32(b1));
        let (va2, vb2) = (ld_f32(a2), ld_f32(b2));
        let (va3, vb3) = (ld_f32(a3), ld_f32(b3));

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
    let mut ab_tail: float32x4_t = vdupq_n_f32(0.0);
    let mut aa_tail: float32x4_t = vdupq_n_f32(0.0);
    let mut bb_tail: float32x4_t = vdupq_n_f32(0.0);

    for (qa, qb) in a_rem.iter().zip(b_rem) {
        let (va, vb) = (ld_f32(qa), ld_f32(qb));
        ab_tail = vfmaq_f32(ab_tail, va, vb);
        aa_tail = vfmaq_f32(aa_tail, va, va);
        bb_tail = vfmaq_f32(bb_tail, vb, vb);
    }

    ab += vaddvq_f32(ab_tail);
    aa += vaddvq_f32(aa_tail);
    bb += vaddvq_f32(bb_tail);

    // Scalar tail
    for (&ai, &bi) in a_tail.iter().zip(b_tail) {
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
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn dot_u8_f32_neon(a: &[f32], b: &[u8]) -> f32 {
    use std::arch::aarch64::{
        vaddq_f32, vaddvq_f32, vcvtq_f32_u32, vdupq_n_f32, vfmaq_f32, vget_high_u16, vget_high_u8,
        vget_low_u16, vget_low_u8, vmovl_u16, vmovl_u8,
    };

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let (a4, _) = a[..n].as_chunks::<4>();
    let (a16, _) = a4.as_chunks::<4>();
    let (b16, _) = b[..n].as_chunks::<16>();

    // Process 16 elements per iteration (16 u8 -> 4x4 f32)
    let mut sum0: float32x4_t = vdupq_n_f32(0.0);
    let mut sum1: float32x4_t = vdupq_n_f32(0.0);
    let mut sum2: float32x4_t = vdupq_n_f32(0.0);
    let mut sum3: float32x4_t = vdupq_n_f32(0.0);

    for ([qa0, qa1, qa2, qa3], qb) in a16.iter().zip(b16) {
        // Load 16 x u8
        let vb = ld_u8(qb);

        // Widen: u8x16 -> u16x8 (lo/hi halves)
        let b_lo_u16 = vmovl_u8(vget_low_u8(vb));
        let b_hi_u16 = vmovl_u8(vget_high_u8(vb));

        // Widen u16x4 -> u32x4 -> f32x4 (4 groups of 4)
        let b0_f32 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_lo_u16)));
        let b1_f32 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_lo_u16)));
        let b2_f32 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_hi_u16)));
        let b3_f32 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_hi_u16)));

        // Load 16 x f32 query values (4 loads of 4)
        let a0 = ld_f32(qa0);
        let a1 = ld_f32(qa1);
        let a2 = ld_f32(qa2);
        let a3 = ld_f32(qa3);

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
    let tail_start = b16.len() * 16;
    for (&x, &y) in a[tail_start..n].iter().zip(&b[tail_start..n]) {
        result += x * (y as f32);
    }

    result
}

/// NEON u8×u8 dot product using `vmull_u8` + `vpadalq_u16` + `vpadalq_u32`.
///
/// Processes 16 bytes per iteration: u8×u8 -> u16 (8 lanes), then pairwise
/// accumulate to u32 (4 lanes), then u64 (2 lanes) every 256 iterations to
/// prevent overflow.
///
/// Overflow analysis: each u16 can hold at most 255*255=65025. With 16 elements
/// per iteration widened to 8 x u16, and vpadalq_u16 summing adjacent pairs to
/// u32 (4 lanes), each u32 lane accumulates at most 2 * 65025 per iteration.
/// Draining to u64 every 256 iterations keeps each u32 lane below
/// 256 * 2 * 65025 ≈ 33.3M, well under u32::MAX.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn dot_u8_neon(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::aarch64::{
        vaddlvq_u32, vdupq_n_u32, vget_high_u8, vget_low_u8, vmull_u8, vpadalq_u16,
    };

    let n = a.len().min(b.len());
    if n == 0 {
        return 0;
    }

    let (a16, a_tail) = a[..n].as_chunks::<16>();
    let (b16, b_tail) = b[..n].as_chunks::<16>();
    let mut total: u64 = 0;

    for (a_block, b_block) in a16.chunks(256).zip(b16.chunks(256)) {
        let mut acc32: uint32x4_t = vdupq_n_u32(0);

        for (qa, qb) in a_block.iter().zip(b_block) {
            let va = ld_u8(qa);
            let vb = ld_u8(qb);
            // vmull_u8: 8 x u8 * 8 x u8 -> 8 x u16 (low half)
            let prod_lo = vmull_u8(vget_low_u8(va), vget_low_u8(vb));
            let prod_hi = vmull_u8(vget_high_u8(va), vget_high_u8(vb));
            // vpadalq_u16: pairwise add adjacent u16 -> u32
            acc32 = vpadalq_u16(acc32, prod_lo);
            acc32 = vpadalq_u16(acc32, prod_hi);
        }

        // vaddlvq_u32: horizontal sum to u64
        total += vaddlvq_u32(acc32) as u64;
    }

    // Scalar tail
    for (&x, &y) in a_tail.iter().zip(b_tail) {
        total += x as u64 * y as u64;
    }

    total as u32
}

/// NEON Hamming distance using `veorq_u8` + `vcntq_u8` (per-byte popcount).
///
/// `vcntq_u8` counts set bits in each byte in a single instruction.
/// Uses `vpaddlq_u8` -> `vpaddlq_u16` -> `vpaddlq_u32` to widen the per-byte
/// counts to u32 before accumulating.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn hamming_neon(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::aarch64::{
        vaddlvq_u32, vaddq_u32, vcntq_u8, vdupq_n_u32, veorq_u8, vpaddlq_u16, vpaddlq_u8,
    };

    let n = a.len().min(b.len());
    if n == 0 {
        return 0;
    }

    let (a16, a_tail) = a[..n].as_chunks::<16>();
    let (b16, b_tail) = b[..n].as_chunks::<16>();
    let mut acc32: uint32x4_t = vdupq_n_u32(0);

    for (qa, qb) in a16.iter().zip(b16) {
        let xored = veorq_u8(ld_u8(qa), ld_u8(qb));
        // vcntq_u8: per-byte popcount (0..=8 per lane)
        let cnt8 = vcntq_u8(xored);
        // Widen: u8x16 -> u16x8 -> u32x4
        let cnt16 = vpaddlq_u8(cnt8);
        let cnt32 = vpaddlq_u16(cnt16);
        acc32 = vaddq_u32(acc32, cnt32);
    }

    let mut result = vaddlvq_u32(acc32) as u32;

    // Scalar tail
    for (&x, &y) in a_tail.iter().zip(b_tail) {
        result += (x ^ y).count_ones();
    }

    result
}

/// Integer-slot Hamming distance over `u32` slots: count of positions where
/// `a[i] != b[i]`. NEON path (4 lanes per 128-bit register).
#[target_feature(enable = "neon")]
pub fn slot_hamming_u32_neon(a: &[u32], b: &[u32]) -> u32 {
    use std::arch::aarch64::{vaddvq_u32, vceqq_u32, vshrq_n_u32};

    let n = a.len().min(b.len());
    let (a4, a_tail) = a[..n].as_chunks::<4>();
    let (b4, b_tail) = b[..n].as_chunks::<4>();

    let mut equal: u32 = 0;

    for (qa, qb) in a4.iter().zip(b4) {
        let va = ld_u32(qa);
        let vb = ld_u32(qb);
        // vceqq_u32 -> 0xFFFFFFFF per equal lane; shift right 31 -> 1 per equal
        // lane; horizontal add -> equal-lane count for this chunk.
        let cmp = vceqq_u32(va, vb);
        equal += vaddvq_u32(vshrq_n_u32(cmp, 31));
    }

    let simd_slots = (a4.len() * 4) as u32;
    let mut diff = simd_slots - equal;

    // Scalar tail
    for (x, y) in a_tail.iter().zip(b_tail) {
        diff += u32::from(x != y);
    }

    diff
}

// ---------------------------------------------------------------------------
// f64 reductions (2 doubles per q-register). Mirror the f32 kernels one width
// down; 4-way unrolled = 8 doubles per iteration.
// ---------------------------------------------------------------------------

/// Count differing u16 slots (8 per q-register). b-bit MinHash at b=16.
#[target_feature(enable = "neon")]
pub fn slot_hamming_u16_neon(a: &[u16], b: &[u16]) -> u32 {
    use std::arch::aarch64::{vaddvq_u16, vceqq_u16, vshrq_n_u16};
    let n = a.len().min(b.len());
    let (a8, a_tail) = a[..n].as_chunks::<8>();
    let (b8, b_tail) = b[..n].as_chunks::<8>();
    let mut equal: u32 = 0;
    for (qa, qb) in a8.iter().zip(b8) {
        // vceqq sets all-ones per equal lane; >>15 maps to 0/1, then sum.
        let eq = vceqq_u16(ld_u16(qa), ld_u16(qb));
        equal += u32::from(vaddvq_u16(vshrq_n_u16::<15>(eq)));
    }
    let mut diff = (a8.len() as u32) * 8 - equal;
    for (x, y) in a_tail.iter().zip(b_tail) {
        diff += u32::from(x != y);
    }
    diff
}

/// Count differing u64 slots (2 per q-register).
#[target_feature(enable = "neon")]
pub fn slot_hamming_u64_neon(a: &[u64], b: &[u64]) -> u64 {
    use std::arch::aarch64::{vaddvq_u64, vceqq_u64, vshrq_n_u64};
    let n = a.len().min(b.len());
    let (a2, a_tail) = a[..n].as_chunks::<2>();
    let (b2, b_tail) = b[..n].as_chunks::<2>();
    let mut equal: u64 = 0;
    for (qa, qb) in a2.iter().zip(b2) {
        // vceqq sets all-ones per equal lane; shift to 0/1 then horizontal-add.
        let eq = vceqq_u64(ld_u64(qa), ld_u64(qb));
        equal += vaddvq_u64(vshrq_n_u64::<63>(eq));
    }
    let mut diff = (a2.len() as u64) * 2 - equal;
    for (x, y) in a_tail.iter().zip(b_tail) {
        diff += u64::from(x != y);
    }
    diff
}

/// Dot product over f64. NEON is baseline on aarch64.
#[target_feature(enable = "neon")]
pub fn dot_f64_neon(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::aarch64::{vaddvq_f64, vdupq_n_f64, vfmaq_f64};
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (a_pairs, a_tail) = a[..n].as_chunks::<2>();
    let (b_pairs, b_tail) = b[..n].as_chunks::<2>();
    let (a8, a_rem) = a_pairs.as_chunks::<4>();
    let (b8, b_rem) = b_pairs.as_chunks::<4>();
    let mut s0: float64x2_t = vdupq_n_f64(0.0);
    let mut s1: float64x2_t = vdupq_n_f64(0.0);
    let mut s2: float64x2_t = vdupq_n_f64(0.0);
    let mut s3: float64x2_t = vdupq_n_f64(0.0);
    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a8.iter().zip(b8) {
        s0 = vfmaq_f64(s0, ld_f64(a0), ld_f64(b0));
        s1 = vfmaq_f64(s1, ld_f64(a1), ld_f64(b1));
        s2 = vfmaq_f64(s2, ld_f64(a2), ld_f64(b2));
        s3 = vfmaq_f64(s3, ld_f64(a3), ld_f64(b3));
    }
    let mut result = vaddvq_f64(s0) + vaddvq_f64(s1) + vaddvq_f64(s2) + vaddvq_f64(s3);
    for (qa, qb) in a_rem.iter().zip(b_rem) {
        result += vaddvq_f64(vfmaq_f64(vdupq_n_f64(0.0), ld_f64(qa), ld_f64(qb)));
    }
    for (x, y) in a_tail.iter().zip(b_tail) {
        result += x * y;
    }
    result
}

/// Squared L2 distance over f64.
#[target_feature(enable = "neon")]
pub fn l2_squared_f64_neon(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::aarch64::{vaddvq_f64, vdupq_n_f64, vfmaq_f64, vsubq_f64};
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (a_pairs, a_tail) = a[..n].as_chunks::<2>();
    let (b_pairs, b_tail) = b[..n].as_chunks::<2>();
    let (a8, a_rem) = a_pairs.as_chunks::<4>();
    let (b8, b_rem) = b_pairs.as_chunks::<4>();
    let mut s0: float64x2_t = vdupq_n_f64(0.0);
    let mut s1: float64x2_t = vdupq_n_f64(0.0);
    let mut s2: float64x2_t = vdupq_n_f64(0.0);
    let mut s3: float64x2_t = vdupq_n_f64(0.0);
    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a8.iter().zip(b8) {
        let d0 = vsubq_f64(ld_f64(a0), ld_f64(b0));
        let d1 = vsubq_f64(ld_f64(a1), ld_f64(b1));
        let d2 = vsubq_f64(ld_f64(a2), ld_f64(b2));
        let d3 = vsubq_f64(ld_f64(a3), ld_f64(b3));
        s0 = vfmaq_f64(s0, d0, d0);
        s1 = vfmaq_f64(s1, d1, d1);
        s2 = vfmaq_f64(s2, d2, d2);
        s3 = vfmaq_f64(s3, d3, d3);
    }
    let mut result = vaddvq_f64(s0) + vaddvq_f64(s1) + vaddvq_f64(s2) + vaddvq_f64(s3);
    for (qa, qb) in a_rem.iter().zip(b_rem) {
        let d = vsubq_f64(ld_f64(qa), ld_f64(qb));
        result += vaddvq_f64(vfmaq_f64(vdupq_n_f64(0.0), d, d));
    }
    for (x, y) in a_tail.iter().zip(b_tail) {
        let d = x - y;
        result += d * d;
    }
    result
}

/// L1 (Manhattan) distance over f64.
#[target_feature(enable = "neon")]
pub fn l1_f64_neon(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::aarch64::{vabdq_f64, vaddq_f64, vaddvq_f64, vdupq_n_f64};
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (a_pairs, a_tail) = a[..n].as_chunks::<2>();
    let (b_pairs, b_tail) = b[..n].as_chunks::<2>();
    let (a8, a_rem) = a_pairs.as_chunks::<4>();
    let (b8, b_rem) = b_pairs.as_chunks::<4>();
    let mut s0: float64x2_t = vdupq_n_f64(0.0);
    let mut s1: float64x2_t = vdupq_n_f64(0.0);
    let mut s2: float64x2_t = vdupq_n_f64(0.0);
    let mut s3: float64x2_t = vdupq_n_f64(0.0);
    for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in a8.iter().zip(b8) {
        s0 = vaddq_f64(s0, vabdq_f64(ld_f64(a0), ld_f64(b0)));
        s1 = vaddq_f64(s1, vabdq_f64(ld_f64(a1), ld_f64(b1)));
        s2 = vaddq_f64(s2, vabdq_f64(ld_f64(a2), ld_f64(b2)));
        s3 = vaddq_f64(s3, vabdq_f64(ld_f64(a3), ld_f64(b3)));
    }
    let mut result = vaddvq_f64(s0) + vaddvq_f64(s1) + vaddvq_f64(s2) + vaddvq_f64(s3);
    for (qa, qb) in a_rem.iter().zip(b_rem) {
        result += vaddvq_f64(vabdq_f64(ld_f64(qa), ld_f64(qb)));
    }
    for (x, y) in a_tail.iter().zip(b_tail) {
        result += (x - y).abs();
    }
    result
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    /// Odd and boundary lengths for every NEON kernel against a scalar
    /// reference. 2 * 4096 + 33 crosses the u8-dot u32 drain (256 blocks of
    /// 16 bytes) twice and leaves a ragged tail.
    const LENS: [usize; 15] = [
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
        2 * 4096 + 33,
    ];

    fn close(got: f64, want: f64, tol: f64) -> bool {
        (got - want).abs() <= tol * want.abs().max(1.0)
    }

    #[test]
    fn neon_kernels_match_scalar_on_odd_lengths() {
        use super::*;
        for n in LENS {
            let fa: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) as f32).sin()).collect();
            let fb: Vec<f32> = (0..n).map(|i| ((i * 11 + 5) as f32).cos()).collect();
            let da: Vec<f64> = fa.iter().map(|&x| f64::from(x) * 1.5).collect();
            let db: Vec<f64> = fb.iter().map(|&x| f64::from(x) - 0.25).collect();
            let ua: Vec<u8> = (0..n).map(|i| (i * 37 + 11) as u8).collect();
            let ub: Vec<u8> = (0..n).map(|i| (i * 101 + 7) as u8).collect();
            // Slots: equal on most positions, different on every 3rd.
            let s32a: Vec<u32> = (0..n as u32).collect();
            let s32b: Vec<u32> = (0..n as u32)
                .map(|i| if i % 3 == 0 { i + 1 } else { i })
                .collect();
            let s16a: Vec<u16> = s32a.iter().map(|&x| x as u16).collect();
            let s16b: Vec<u16> = s32b.iter().map(|&x| x as u16).collect();
            let s64a: Vec<u64> = s32a.iter().map(|&x| u64::from(x) << 33).collect();
            let s64b: Vec<u64> = s32b.iter().map(|&x| u64::from(x) << 33).collect();

            let f = |x: f32| f64::from(x);
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

            // f32 kernels accumulate in f32; tolerance is relative to max(|want|, 1).
            let t = 1e-4;
            unsafe {
                assert!(close(f(dot_neon(&fa, &fb)), dot, t), "dot n={n}");
                assert!(close(f(l2_squared_neon(&fa, &fb)), l2, t), "l2 n={n}");
                assert!(close(f(l1_neon(&fa, &fb)), l1, t), "l1 n={n}");
                assert!(close(f(cosine_neon(&fa, &fb)), cos, t), "cos n={n}");
                let fc = crate::fast_math::aarch64::fast_cosine_neon(&fa, &fb);
                assert!(close(f(fc), cos, t), "fast_cos n={n}");
                assert!(
                    close(f(dot_u8_f32_neon(&fa, &ub)), dot_u8f, t),
                    "dot_u8_f32 n={n}"
                );
                assert_eq!(u64::from(dot_u8_neon(&ua, &ub)), dot_u8, "dot_u8 n={n}");
                assert_eq!(hamming_neon(&ua, &ub), ham, "hamming n={n}");
                assert_eq!(
                    slot_hamming_u32_neon(&s32a, &s32b) as usize,
                    slot,
                    "slot32 n={n}"
                );
                assert_eq!(
                    slot_hamming_u16_neon(&s16a, &s16b) as usize,
                    slot16,
                    "slot16 n={n}"
                );
                assert_eq!(
                    slot_hamming_u64_neon(&s64a, &s64b) as usize,
                    slot,
                    "slot64 n={n}"
                );
                assert!(close(dot_f64_neon(&da, &db), ddot, 1e-12), "dot_f64 n={n}");
                assert!(
                    close(l2_squared_f64_neon(&da, &db), dl2, 1e-12),
                    "l2_f64 n={n}"
                );
                assert!(close(l1_f64_neon(&da, &db), dl1, 1e-12), "l1_f64 n={n}");
            }

            // MaxSim: n query tokens of dim 33 against 3 doc tokens.
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
            let want: f64 = qr
                .iter()
                .map(|q| {
                    dr.iter()
                        .map(|d| q.iter().zip(*d).map(|(&x, &y)| f(x) * f(y)).sum::<f64>())
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .sum();
            let got = unsafe { maxsim_neon(&qr, &dr) };
            assert!(close(f(got), want, t), "maxsim n={n}");
        }
    }

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
            let expected = if aa > crate::NORM_EPSILON_SQ && bb > crate::NORM_EPSILON_SQ {
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
