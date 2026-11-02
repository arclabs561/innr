//! Portable kernels shaped for the auto-vectorizer.
//!
//! Each reduction keeps a fixed-size array of independent accumulators and
//! walks the input in `as_chunks` blocks of the same width, so LLVM keeps the
//! accumulators in vector registers without needing permission to reassociate
//! floating-point adds. Sixteen `f32` lanes fill four 128-bit registers (NEON,
//! SSE2) or two 256-bit ones, enough independent add chains to cover FP add
//! latency. The leftover `n % LANES` elements are summed in a scalar tail and
//! the lanes are combined pairwise.
//!
//! Results agree with a sequential loop to rounding, not bit for bit; the
//! crate's accuracy tests bound the difference against a higher-precision
//! reference.

/// Accumulator lanes for `f32` reductions.
pub(crate) const F32_LANES: usize = 16;
/// Accumulator lanes for `f64` reductions (the same register footprint).
pub(crate) const F64_LANES: usize = 8;

macro_rules! tree_sum {
    ($name:ident, $t:ty) => {
        /// Pairwise sum of the lanes: halve the array until one value is left.
        #[inline(always)]
        fn $name<const N: usize>(mut v: [$t; N]) -> $t {
            let mut w = N;
            while w > 1 {
                w /= 2;
                for i in 0..w {
                    v[i] += v[i + w];
                }
            }
            v[0]
        }
    };
}
tree_sum!(tree_sum_f32, f32);
tree_sum!(tree_sum_f64, f64);

macro_rules! fold_pairs {
    ($name:ident, $t:ty, $lanes:expr, $tree:ident) => {
        /// `Σ term(a[i], b[i])` over the common prefix of `a` and `b`.
        #[inline(always)]
        pub(crate) fn $name(a: &[$t], b: &[$t], term: impl Fn($t, $t) -> $t) -> $t {
            let n = a.len().min(b.len());
            // Four lanes and a short scalar tail: used for small dimensions
            // (2-d points, short sketches), where it measured faster than the
            // wide lane array and than a sequential loop, and for the
            // leftover after the wide blocks, where a scalar loop over up to
            // LANES - 1 elements was one long dependent add chain (Apple M3
            // Max, d24: 4.4 ns against 2.6 ns at d32).
            let narrow = |a: &[$t], b: &[$t]| -> $t {
                let (ca, ra) = a.as_chunks::<4>();
                let (cb, rb) = b.as_chunks::<4>();
                let mut acc = [0.0; 4];
                for (x, y) in ca.iter().zip(cb) {
                    for i in 0..4 {
                        acc[i] += term(x[i], y[i]);
                    }
                }
                let mut tail = 0.0;
                for (&x, &y) in ra.iter().zip(rb) {
                    tail += term(x, y);
                }
                (acc[0] + acc[2]) + (acc[1] + acc[3]) + tail
            };
            if n < $lanes {
                return narrow(&a[..n], &b[..n]);
            }
            let (ca, ra) = a[..n].as_chunks::<{ $lanes }>();
            let (cb, rb) = b[..n].as_chunks::<{ $lanes }>();
            let mut acc = [0.0; $lanes];
            for (x, y) in ca.iter().zip(cb) {
                for i in 0..$lanes {
                    acc[i] += term(x[i], y[i]);
                }
            }
            $tree(acc) + narrow(ra, rb)
        }
    };
}
fold_pairs!(fold_f32, f32, F32_LANES, tree_sum_f32);
fold_pairs!(fold_f64, f64, F64_LANES, tree_sum_f64);

/// `(Σ ab, Σ a², Σ b²)` as three [`fold_f32`] passes over the common prefix.
///
/// A fused single pass with three accumulator arrays was SLP-vectorized
/// across the sums instead of across lanes (two-wide shuffles and stack
/// spills on x86_64), and three passes measured 1.4x to 2x faster from d32
/// up on both AVX2-class x86 and Apple NEON. The re-reads hit L1 at
/// embedding sizes.
#[inline(always)]
pub(crate) fn cosine_parts_f32(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    (
        fold_f32(a, b, |x, y| x * y),
        fold_f32(a, a, |x, y| x * y),
        fold_f32(b, b, |x, y| x * y),
    )
}

/// `Σ a[i] * (b[i] as f32)` over the common prefix.
#[inline(always)]
pub(crate) fn mixed_dot_u8_f32(a: &[f32], b: &[u8]) -> f32 {
    let n = a.len().min(b.len());
    let (ca, ra) = a[..n].as_chunks::<F32_LANES>();
    let (cb, rb) = b[..n].as_chunks::<F32_LANES>();
    let mut acc = [0.0f32; F32_LANES];
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..F32_LANES {
            acc[i] += x[i] * f32::from(y[i]);
        }
    }
    let mut tail = 0.0f32;
    for (&x, &y) in ra.iter().zip(rb) {
        tail += x * f32::from(y);
    }
    tree_sum_f32(acc) + tail
}

/// Exact `Σ a[i] * b[i]` over `u8`, in `u32` lanes flushed to a `u64` total.
#[inline(always)]
pub(crate) fn dot_u8_wide(a: &[u8], b: &[u8]) -> u64 {
    const L: usize = 16;
    // A lane gains at most 255² = 65025 per chunk; 65536 chunks keep it
    // below u32::MAX (65536 * 65025 < 2³²).
    const FLUSH: usize = 65536;
    let n = a.len().min(b.len());
    let (ca, ra) = a[..n].as_chunks::<L>();
    let (cb, rb) = b[..n].as_chunks::<L>();
    let mut total: u64 = 0;
    for (xs, ys) in ca.chunks(FLUSH).zip(cb.chunks(FLUSH)) {
        let mut acc = [0u32; L];
        for (x, y) in xs.iter().zip(ys) {
            for i in 0..L {
                acc[i] += u32::from(x[i]) * u32::from(y[i]);
            }
        }
        total += acc.iter().map(|&v| u64::from(v)).sum::<u64>();
    }
    for (&x, &y) in ra.iter().zip(rb) {
        total += u64::from(x) * u64::from(y);
    }
    total
}

/// Number of differing bits between byte-packed bit vectors.
///
/// XORs 64-bit words, four independent popcount accumulators per 32-byte
/// block; `count_ones` on `u64` lowers to one `popcnt` (x86 with POPCNT) or
/// `cnt` plus a pairwise add chain (NEON).
#[inline(always)]
pub(crate) fn hamming(a: &[u8], b: &[u8]) -> u32 {
    const W: usize = 4;
    let n = a.len().min(b.len());
    let (ca, ra) = a[..n].as_chunks::<{ 8 * W }>();
    let (cb, rb) = b[..n].as_chunks::<{ 8 * W }>();
    let mut acc = [0u64; W];
    for (x, y) in ca.iter().zip(cb) {
        let (xw, _) = x.as_chunks::<8>();
        let (yw, _) = y.as_chunks::<8>();
        for k in 0..W {
            let d = u64::from_le_bytes(xw[k]) ^ u64::from_le_bytes(yw[k]);
            acc[k] += u64::from(d.count_ones());
        }
    }
    let (xw, xr) = ra.as_chunks::<8>();
    let (yw, yr) = rb.as_chunks::<8>();
    let mut tail: u64 = 0;
    for (x, y) in xw.iter().zip(yw) {
        tail += u64::from((u64::from_le_bytes(*x) ^ u64::from_le_bytes(*y)).count_ones());
    }
    for (&x, &y) in xr.iter().zip(yr) {
        tail += u64::from((x ^ y).count_ones());
    }
    // At most 8 bits per input byte, so the total fits u32 for any input the
    // u32-returning API accepts without overflowing.
    (acc.iter().sum::<u64>() + tail) as u32
}

/// Number of positions where `a[i] != b[i]`.
#[inline(always)]
pub(crate) fn slot_ne_u64(a: &[u64], b: &[u64]) -> u64 {
    // A plain filter-count loop: LLVM vectorizes it (SSE2 emulates the 64-bit
    // compare with pcmpeqd + pshufd). An eight-lane accumulator array here
    // compiled to eight scalar compare chains on x86_64 and ran 1.5x slower.
    a.iter().zip(b).filter(|(x, y)| x != y).count() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_sum_adds_every_lane() {
        let v: [f32; 16] = core::array::from_fn(|i| (i + 1) as f32);
        assert_eq!(tree_sum_f32(v), 136.0);
        let v: [f64; 8] = core::array::from_fn(|i| (i + 1) as f64);
        assert_eq!(tree_sum_f64(v), 36.0);
    }

    #[test]
    fn integer_kernels_cover_tails_and_flush() {
        for n in [0usize, 1, 15, 16, 17, 31, 32, 33, 100] {
            let a: Vec<u8> = (0..n).map(|i| (i * 37 + 11) as u8).collect();
            let b: Vec<u8> = (0..n).map(|i| (i * 91 + 3) as u8).collect();
            let want: u64 = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| u64::from(x) * u64::from(y))
                .sum();
            assert_eq!(dot_u8_wide(&a, &b), want, "dot n={n}");
            let want: u32 = a.iter().zip(&b).map(|(&x, &y)| (x ^ y).count_ones()).sum();
            assert_eq!(hamming(&a, &b), want, "hamming n={n}");
        }
        // Past one u32 flush block: 16 * 65536 maximal products.
        let n = 16 * 65536 + 5;
        let a = vec![u8::MAX; n];
        assert_eq!(dot_u8_wide(&a, &a), n as u64 * 65025);
    }
}
