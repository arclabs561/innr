//! SIMD kernels written once, generic over [`fearless_simd::Simd`].
//!
//! Every kernel is `#[inline(always)]` so that it compiles inside the
//! `Simd::vectorize` context that [`super::simd_dispatch!`] establishes, with
//! that level's target features. Vector width comes from the level's native
//! types (`S::f32s` is 16 lanes on AVX-512, 8 on AVX2, 4 on NEON and WASM).
//!
//! Floating-point reductions keep eight independent vector accumulators,
//! spread leftover whole vectors over them, and finish the last `LEN - 1`
//! elements in scalar code. `mul_add` is fused on every dispatched
//! level (AVX2+FMA, AVX-512, NEON); the order of summation differs from the
//! portable kernels, so results agree to rounding, not bit for bit.

use fearless_simd::{
    Simd, SimdBase, SimdCvtFloat, SimdFloat, SimdInt, SimdIntWiden, SimdMask, SimdWiden,
};

/// Independent vector accumulators in [`fold_float`]. Eight covers FMA
/// latency times throughput on the dispatched levels (four cycles by two FMA
/// ports on x86, by four pipes on Apple NEON cores, which still need half of
/// them); four measured slower than the portable kernels on NEON at 384+.
const ACC: usize = 8;

/// Main-loop blocks from which [`fold_float`] aligns its input on x86 (512
/// `f32` on AVX2, 1024 on AVX-512). Gains were clear at d768 and mixed at
/// d256; d512 was not measured.
const PEEL_BLOCKS: usize = 8;

/// Fold `step` over equal-length float slices with [`ACC`] vector
/// accumulators.
///
/// `step(acc, x, y)` updates one accumulator from one pair of vectors;
/// `tail(acc, x, y)` does the same for one pair of scalars.
#[inline(always)]
fn fold_float<S, V>(
    simd: S,
    a: &[V::Element],
    b: &[V::Element],
    zero: V::Element,
    step: impl Fn(V, V, V) -> V,
    tail: impl Fn(V::Element, V::Element, V::Element) -> V::Element,
) -> V::Element
where
    S: Simd,
    V: SimdFloat<S>,
{
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let l = V::LEN;
    if n < ACC * l {
        return fold_float_short(simd, a, b, zero, step, tail);
    }
    // On x86, from PEEL_BLOCKS main-loop blocks up, peel scalars until `b`
    // starts on a vector-width boundary. Allocators only guarantee 16 bytes
    // (glibc alternates 0 and 16 mod 32 for consecutive equal-sized
    // allocations), and on AVX2 a buffer 16 bytes off a 32-byte boundary
    // splits every other load across cache lines. Ryzen 5600X, d4096: both
    // inputs 16 off 196 -> 127 ns; only one off about neutral (158 -> 162,
    // 168 -> 161); both aligned 123 -> 126. Aligning `b` rather than `a`
    // measured better when only one is off. NEON's 16-byte loads do not
    // split, and the peel cost 7 to 11% at d128 to d256 on an Apple M3 Max,
    // so other targets skip it. At d128 to d256 the scalar head cost about
    // as much as the splits, hence the length gate.
    let mut head_sum = zero;
    let peel = cfg!(any(target_arch = "x86", target_arch = "x86_64"));
    let (a, b) = if peel && n >= PEEL_BLOCKS * ACC * l {
        let width = l * core::mem::size_of::<V::Element>();
        let misalign = b.as_ptr() as usize % width;
        let head = (width - misalign) % width / core::mem::size_of::<V::Element>();
        for (&x, &y) in a[..head].iter().zip(&b[..head]) {
            head_sum = tail(head_sum, x, y);
        }
        (&a[head..], &b[head..])
    } else {
        (a, b)
    };
    let mut s = [V::splat(simd, zero); ACC];

    let mut ca = a.chunks_exact(ACC * l);
    let mut cb = b.chunks_exact(ACC * l);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for (k, acc) in s.iter_mut().enumerate() {
            let r = k * l..(k + 1) * l;
            *acc = step(
                *acc,
                V::from_slice(simd, &x[r.clone()]),
                V::from_slice(simd, &y[r]),
            );
        }
    }

    // Spread the leftover whole vectors (fewer than ACC) over the
    // accumulators. The loop has a constant trip count so it unrolls with
    // constant indices; a data-dependent index would spill `s` to the stack.
    let mut ra = ca.remainder().chunks_exact(l);
    let mut rb = cb.remainder().chunks_exact(l);
    for acc in s.iter_mut().take(ACC - 1) {
        let (Some(x), Some(y)) = (ra.next(), rb.next()) else {
            break;
        };
        *acc = step(*acc, V::from_slice(simd, x), V::from_slice(simd, y));
    }

    let mut w = ACC;
    while w > 1 {
        w /= 2;
        for k in 0..w {
            s[k] += s[k + w];
        }
    }
    let mut sum = s[0].reduce_sum() + head_sum;
    for (&x, &y) in ra.remainder().iter().zip(rb.remainder()) {
        sum = tail(sum, x, y);
    }
    sum
}

/// [`fold_float`] for inputs shorter than one main-loop block (`ACC`
/// vectors). With eight accumulators the result waits on three dependent
/// vector adds even when only one or two were used; at d16 on AVX2 this path
/// took the call from 5.9 to 5.2 ns. Two accumulators end in a single add.
#[inline(always)]
fn fold_float_short<S, V>(
    simd: S,
    a: &[V::Element],
    b: &[V::Element],
    zero: V::Element,
    step: impl Fn(V, V, V) -> V,
    tail: impl Fn(V::Element, V::Element, V::Element) -> V::Element,
) -> V::Element
where
    S: Simd,
    V: SimdFloat<S>,
{
    let l = V::LEN;
    let mut s0 = V::splat(simd, zero);
    let mut s1 = V::splat(simd, zero);
    let mut ca = a.chunks_exact(2 * l);
    let mut cb = b.chunks_exact(2 * l);
    for (x, y) in (&mut ca).zip(&mut cb) {
        s0 = step(
            s0,
            V::from_slice(simd, &x[..l]),
            V::from_slice(simd, &y[..l]),
        );
        s1 = step(
            s1,
            V::from_slice(simd, &x[l..]),
            V::from_slice(simd, &y[l..]),
        );
    }
    let (ra, rb) = (ca.remainder(), cb.remainder());
    let (ra, rb) = if ra.len() >= l {
        s0 = step(
            s0,
            V::from_slice(simd, &ra[..l]),
            V::from_slice(simd, &rb[..l]),
        );
        (&ra[l..], &rb[l..])
    } else {
        (ra, rb)
    };
    let mut sum = (s0 + s1).reduce_sum();
    for (&x, &y) in ra.iter().zip(rb) {
        sum = tail(sum, x, y);
    }
    sum
}

/// `Σ a[i] * b[i]` over `f32`.
#[inline(always)]
pub(crate) fn dot_f32<S: Simd>(simd: S, a: &[f32], b: &[f32]) -> f32 {
    fold_float::<S, S::f32s>(
        simd,
        a,
        b,
        0.0,
        |acc, x, y| x.mul_add(y, acc),
        |acc, x, y| acc + x * y,
    )
}

/// `Σ (a[i] - b[i])²` over `f32`.
#[inline(always)]
pub(crate) fn l2_squared_f32<S: Simd>(simd: S, a: &[f32], b: &[f32]) -> f32 {
    fold_float::<S, S::f32s>(
        simd,
        a,
        b,
        0.0,
        |acc, x, y| {
            let d = x - y;
            d.mul_add(d, acc)
        },
        |acc, x, y| {
            let d = x - y;
            acc + d * d
        },
    )
}

/// `Σ |a[i] - b[i]|` over `f32`.
#[inline(always)]
pub(crate) fn l1_f32<S: Simd>(simd: S, a: &[f32], b: &[f32]) -> f32 {
    fold_float::<S, S::f32s>(
        simd,
        a,
        b,
        0.0,
        |acc, x, y| acc + (x - y).abs(),
        |acc, x, y| acc + (x - y).abs(),
    )
}

/// `Σ a[i] * b[i]` over `f64`.
#[inline(always)]
pub(crate) fn dot_f64<S: Simd>(simd: S, a: &[f64], b: &[f64]) -> f64 {
    fold_float::<S, S::f64s>(
        simd,
        a,
        b,
        0.0,
        |acc, x, y| x.mul_add(y, acc),
        |acc, x, y| acc + x * y,
    )
}

/// `Σ (a[i] - b[i])²` over `f64`.
#[inline(always)]
pub(crate) fn l2_squared_f64<S: Simd>(simd: S, a: &[f64], b: &[f64]) -> f64 {
    fold_float::<S, S::f64s>(
        simd,
        a,
        b,
        0.0,
        |acc, x, y| {
            let d = x - y;
            d.mul_add(d, acc)
        },
        |acc, x, y| {
            let d = x - y;
            acc + d * d
        },
    )
}

/// `Σ |a[i] - b[i]|` over `f64`.
#[inline(always)]
pub(crate) fn l1_f64<S: Simd>(simd: S, a: &[f64], b: &[f64]) -> f64 {
    fold_float::<S, S::f64s>(
        simd,
        a,
        b,
        0.0,
        |acc, x, y| acc + (x - y).abs(),
        |acc, x, y| acc + (x - y).abs(),
    )
}

/// Single pass over `a` and `b` returning `(Σ ab, Σ a², Σ b²)`.
///
/// [`cosine_parts_f32`] for inputs shorter than two of its blocks: two
/// accumulators per quantity. Twelve accumulators cost more to set up and
/// reduce than they save there (d16 on an Apple M3 Max: 2.9 vs 3.3 ns).
#[inline(always)]
fn cosine_parts_short_f32<S: Simd>(simd: S, a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let l = S::f32s::LEN;
    let z = S::f32s::splat(simd, 0.0);
    let (mut ab0, mut ab1, mut aa0, mut aa1, mut bb0, mut bb1) = (z, z, z, z, z, z);

    let mut ca = a.chunks_exact(2 * l);
    let mut cb = b.chunks_exact(2 * l);
    for (x, y) in (&mut ca).zip(&mut cb) {
        let x0 = S::f32s::from_slice(simd, &x[..l]);
        let y0 = S::f32s::from_slice(simd, &y[..l]);
        let x1 = S::f32s::from_slice(simd, &x[l..]);
        let y1 = S::f32s::from_slice(simd, &y[l..]);
        ab0 = x0.mul_add(y0, ab0);
        aa0 = x0.mul_add(x0, aa0);
        bb0 = y0.mul_add(y0, bb0);
        ab1 = x1.mul_add(y1, ab1);
        aa1 = x1.mul_add(x1, aa1);
        bb1 = y1.mul_add(y1, bb1);
    }

    let mut ra = ca.remainder().chunks_exact(l);
    let mut rb = cb.remainder().chunks_exact(l);
    for (x, y) in (&mut ra).zip(&mut rb) {
        let x0 = S::f32s::from_slice(simd, x);
        let y0 = S::f32s::from_slice(simd, y);
        ab0 = x0.mul_add(y0, ab0);
        aa0 = x0.mul_add(x0, aa0);
        bb0 = y0.mul_add(y0, bb0);
    }

    let mut ab = (ab0 + ab1).reduce_sum();
    let mut aa = (aa0 + aa1).reduce_sum();
    let mut bb = (bb0 + bb1).reduce_sum();
    for (&x, &y) in ra.remainder().iter().zip(rb.remainder()) {
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    (ab, aa, bb)
}

/// Four accumulators per quantity (twelve vectors). Two per quantity left the
/// one-pass loop latency-bound: on an Apple M3 Max it ran at 2.2 dot
/// products' cost at d768, slower than innr 0.6's kernel. Twelve accumulators
/// plus two loaded vectors still fit AVX2's sixteen registers.
#[inline(always)]
pub(crate) fn cosine_parts_f32<S: Simd>(simd: S, a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    const Q: usize = 4;
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let l = S::f32s::LEN;
    if n < 2 * Q * l {
        return cosine_parts_short_f32(simd, a, b);
    }
    let z = S::f32s::splat(simd, 0.0);
    let (mut ab, mut aa, mut bb) = ([z; Q], [z; Q], [z; Q]);

    let mut ca = a.chunks_exact(Q * l);
    let mut cb = b.chunks_exact(Q * l);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for k in 0..Q {
            let r = k * l..(k + 1) * l;
            let xk = S::f32s::from_slice(simd, &x[r.clone()]);
            let yk = S::f32s::from_slice(simd, &y[r]);
            ab[k] = xk.mul_add(yk, ab[k]);
            aa[k] = xk.mul_add(xk, aa[k]);
            bb[k] = yk.mul_add(yk, bb[k]);
        }
    }

    // Leftover whole vectors (fewer than Q), one per accumulator; constant
    // trip count so the indices stay constant and nothing spills.
    let mut ra = ca.remainder().chunks_exact(l);
    let mut rb = cb.remainder().chunks_exact(l);
    for k in 0..Q - 1 {
        let (Some(x), Some(y)) = (ra.next(), rb.next()) else {
            break;
        };
        let xk = S::f32s::from_slice(simd, x);
        let yk = S::f32s::from_slice(simd, y);
        ab[k] = xk.mul_add(yk, ab[k]);
        aa[k] = xk.mul_add(xk, aa[k]);
        bb[k] = yk.mul_add(yk, bb[k]);
    }

    let sum = |v: [S::f32s; Q]| ((v[0] + v[1]) + (v[2] + v[3])).reduce_sum();
    let (mut ab, mut aa, mut bb) = (sum(ab), sum(aa), sum(bb));
    for (&x, &y) in ra.remainder().iter().zip(rb.remainder()) {
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    (ab, aa, bb)
}

/// `Σ a[i] * b[i]` over `u8`, exact in `u32`.
///
/// The caller bounds the length so the total fits in `u32` (at most
/// `u32::MAX / 255²` elements); each `u32` lane then holds a partial sum no
/// larger than the total, so no lane overflows either.
#[inline(always)]
pub(crate) fn dot_u8<S: Simd>(simd: S, a: &[u8], b: &[u8]) -> u32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let l = S::u8s::LEN;
    let mut acc = S::u32s::splat(simd, 0);

    let mut ca = a.chunks_exact(l);
    let mut cb = b.chunks_exact(l);
    for (x, y) in (&mut ca).zip(&mut cb) {
        let x = S::u8s::from_slice(simd, x);
        let y = S::u8s::from_slice(simd, y);
        // u8 x u8 fits u16; the sum of two adjacent products fits u32.
        let (lo, hi) = x.widening_mul(y);
        acc = acc + lo.pairwise_widening_add() + hi.pairwise_widening_add();
    }

    let mut sum = acc.reduce_sum();
    for (&x, &y) in ca.remainder().iter().zip(cb.remainder()) {
        sum += u32::from(x) * u32::from(y);
    }
    sum
}

/// Number of differing bits between two byte-packed bit vectors.
#[inline(always)]
pub(crate) fn hamming_u8<S: Simd>(simd: S, a: &[u8], b: &[u8]) -> u32 {
    // Per-byte popcounts are at most 8, so a u8 lane absorbs 31 of them.
    const BLOCK: usize = 31;
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let l = S::u8s::LEN;
    let mut total = S::u32s::splat(simd, 0);

    let mut ca = a.chunks_exact(l * BLOCK);
    let mut cb = b.chunks_exact(l * BLOCK);
    for (x, y) in (&mut ca).zip(&mut cb) {
        let mut bytes = S::u8s::splat(simd, 0);
        for (xs, ys) in x.chunks_exact(l).zip(y.chunks_exact(l)) {
            let d = S::u8s::from_slice(simd, xs) ^ S::u8s::from_slice(simd, ys);
            bytes += d.count_ones();
        }
        total += bytes.pairwise_widening_add().pairwise_widening_add();
    }

    // Fewer than BLOCK whole vectors remain: one more u8 accumulation.
    let mut ra = ca.remainder().chunks_exact(l);
    let mut rb = cb.remainder().chunks_exact(l);
    let mut bytes = S::u8s::splat(simd, 0);
    for (x, y) in (&mut ra).zip(&mut rb) {
        let d = S::u8s::from_slice(simd, x) ^ S::u8s::from_slice(simd, y);
        bytes += d.count_ones();
    }
    total += bytes.pairwise_widening_add().pairwise_widening_add();

    let mut sum = total.reduce_sum();
    for (&x, &y) in ra.remainder().iter().zip(rb.remainder()) {
        sum += (x ^ y).count_ones();
    }
    sum
}

/// Mixed-precision `Σ a[i] * (b[i] as f32)`.
#[inline(always)]
pub(crate) fn mixed_dot_u8_f32<S: Simd>(simd: S, a: &[f32], b: &[u8]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    // One u8 vector widens to four u32 vectors, each as wide as S::f32s.
    let lb = S::u8s::LEN;
    let lf = S::f32s::LEN;
    let z = S::f32s::splat(simd, 0.0);
    let (mut s0, mut s1, mut s2, mut s3) = (z, z, z, z);

    let mut cb = b.chunks_exact(lb);
    let mut ca = a.chunks_exact(lb);
    for (x, y) in (&mut ca).zip(&mut cb) {
        let (w0, w1) = S::u8s::from_slice(simd, y).widen();
        let (q0, q1) = w0.widen();
        let (q2, q3) = w1.widen();
        let f0 = S::f32s::float_from(q0);
        let f1 = S::f32s::float_from(q1);
        let f2 = S::f32s::float_from(q2);
        let f3 = S::f32s::float_from(q3);
        s0 = S::f32s::from_slice(simd, &x[..lf]).mul_add(f0, s0);
        s1 = S::f32s::from_slice(simd, &x[lf..2 * lf]).mul_add(f1, s1);
        s2 = S::f32s::from_slice(simd, &x[2 * lf..3 * lf]).mul_add(f2, s2);
        s3 = S::f32s::from_slice(simd, &x[3 * lf..]).mul_add(f3, s3);
    }

    let mut sum = ((s0 + s1) + (s2 + s3)).reduce_sum();
    for (&x, &y) in ca.remainder().iter().zip(cb.remainder()) {
        sum += x * f32::from(y);
    }
    sum
}

/// Number of positions where `a[i] != b[i]` over `u32` slots.
#[inline(always)]
pub(crate) fn slot_ne_u32<S: Simd>(simd: S, a: &[u32], b: &[u32]) -> u32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let l = S::u32s::LEN;
    // Mask lanes are -1 when true, so subtracting counts differing slots.
    let mut acc = S::i32s::splat(simd, 0);
    let mut ca = a.chunks_exact(l);
    let mut cb = b.chunks_exact(l);
    for (x, y) in (&mut ca).zip(&mut cb) {
        let ne = S::u32s::from_slice(simd, x).simd_ne(S::u32s::from_slice(simd, y));
        acc -= ne.to_vector();
    }
    let mut sum = acc.reduce_sum() as u32;
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        sum += u32::from(x != y);
    }
    sum
}

/// Number of positions where `a[i] != b[i]` over `u16` slots.
#[inline(always)]
pub(crate) fn slot_ne_u16<S: Simd>(simd: S, a: &[u16], b: &[u16]) -> u32 {
    // An i16 lane counts at most 32767; flush well before that.
    const BLOCK: usize = 8192;
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let l = S::u16s::LEN;
    let mut total = S::i32s::splat(simd, 0);
    for (ba, bb) in a.chunks(l * BLOCK).zip(b.chunks(l * BLOCK)) {
        let mut acc = S::i16s::splat(simd, 0);
        for (x, y) in ba.chunks_exact(l).zip(bb.chunks_exact(l)) {
            let ne = S::u16s::from_slice(simd, x).simd_ne(S::u16s::from_slice(simd, y));
            acc -= ne.to_vector();
        }
        total += acc.pairwise_widening_add();
    }
    let mut sum = total.reduce_sum() as u32;
    let whole = n - n % l;
    for (x, y) in a[whole..].iter().zip(&b[whole..]) {
        sum += u32::from(x != y);
    }
    sum
}

/// ColBERT MaxSim: `Σ_q max_d dot(q, d)`, every pair on one level token.
#[inline(always)]
pub(crate) fn maxsim<S: Simd>(simd: S, query: &[&[f32]], doc: &[&[f32]]) -> f32 {
    query
        .iter()
        .map(|q| {
            doc.iter()
                .map(|d| dot_f32(simd, q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    //! Run every kernel on fearless_simd's scalar `Fallback` level (always
    //! available) and on the native level, against scalar references, at
    //! lengths that hit the empty, unrolled, single-vector and tail paths.
    use super::*;
    use fearless_simd::{Fallback, Level};

    const LENS: &[usize] = &[
        0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257,
        1000, 4099,
    ];

    fn f32s(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| (((i as u32).wrapping_mul(2654435761) ^ seed) % 2001) as f32 / 1000.0 - 1.0)
            .collect()
    }

    fn u8s(n: usize, seed: u32) -> Vec<u8> {
        (0..n)
            .map(|i| ((i as u32).wrapping_mul(2654435761) ^ seed) as u8)
            .collect()
    }

    fn close(got: f64, want: f64, scale: f64, n: usize) -> bool {
        (got - want).abs() <= (n as f64 + 4.0) * f64::from(f32::EPSILON) * scale.max(1.0)
    }

    /// The main-loop path peels scalars until `a` is vector-aligned. Slice
    /// both inputs at every element offset up to 15 (64 bytes), separately
    /// for `a` and `b`, so each peel length and unequal offsets are covered.
    fn check_offsets<S: Simd>(s: S) {
        let (ba, bb) = (f32s(1100, 3), f32s(1100, 9));
        for n in [128usize, 129, 257, 1000, 1100 - 16] {
            for oa in 0..16 {
                for ob in [0usize, 1, 3, 4, 7] {
                    let (a, b) = (&ba[oa..oa + n], &bb[ob..ob + n]);
                    let pairs = || a.iter().zip(b).map(|(&x, &y)| (f64::from(x), f64::from(y)));
                    let r_dot: f64 = pairs().map(|(x, y)| x * y).sum();
                    let abs: f64 = pairs().map(|(x, y)| (x * y).abs()).sum();
                    let got = s.vectorize(|| dot_f32(s, a, b));
                    assert!(
                        close(got.into(), r_dot, abs, n),
                        "dot n={n} oa={oa} ob={ob}: {got} vs {r_dot}"
                    );
                    let r_l2: f64 = pairs().map(|(x, y)| (x - y).powi(2)).sum();
                    let got = s.vectorize(|| l2_squared_f32(s, a, b));
                    assert!(close(got.into(), r_l2, r_l2, n), "l2 n={n} oa={oa} ob={ob}");
                    let r_l1: f64 = pairs().map(|(x, y)| (x - y).abs()).sum();
                    let got = s.vectorize(|| l1_f32(s, a, b));
                    assert!(close(got.into(), r_l1, r_l1, n), "l1 n={n} oa={oa} ob={ob}");
                }
            }
        }
    }

    #[test]
    fn kernels_match_reference_at_every_offset() {
        check_offsets(Fallback::new());
        fearless_simd::dispatch!(Level::new(), s => check_offsets(s));
    }

    fn check_level<S: Simd>(s: S) {
        for &n in LENS {
            let a = f32s(n, 1);
            let b = f32s(n, 7);
            let abs: f64 = a.iter().zip(&b).map(|(x, y)| f64::from(x * y).abs()).sum();
            let r_dot: f64 = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum();
            let got = s.vectorize(|| dot_f32(s, &a, &b));
            assert!(
                close(got.into(), r_dot, abs, n),
                "dot n={n}: {got} vs {r_dot}"
            );

            let r_l2: f64 = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
                .sum();
            let got = s.vectorize(|| l2_squared_f32(s, &a, &b));
            assert!(close(got.into(), r_l2, r_l2, n), "l2 n={n}");

            let r_l1: f64 = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
                .sum();
            let got = s.vectorize(|| l1_f32(s, &a, &b));
            assert!(close(got.into(), r_l1, r_l1, n), "l1 n={n}");

            let (ab, aa, bb) = s.vectorize(|| cosine_parts_f32(s, &a, &b));
            let r_aa: f64 = a.iter().map(|&x| f64::from(x).powi(2)).sum();
            let r_bb: f64 = b.iter().map(|&x| f64::from(x).powi(2)).sum();
            assert!(close(ab.into(), r_dot, abs, n), "cos ab n={n}");
            assert!(close(aa.into(), r_aa, r_aa, n), "cos aa n={n}");
            assert!(close(bb.into(), r_bb, r_bb, n), "cos bb n={n}");

            let da: Vec<f64> = a.iter().map(|&x| f64::from(x)).collect();
            let db: Vec<f64> = b.iter().map(|&x| f64::from(x)).collect();
            let got = s.vectorize(|| dot_f64(s, &da, &db));
            assert!((got - r_dot).abs() <= 1e-12 * abs.max(1.0), "dot_f64 n={n}");
            let got = s.vectorize(|| l2_squared_f64(s, &da, &db));
            assert!((got - r_l2).abs() <= 1e-12 * r_l2.max(1.0), "l2_f64 n={n}");
            let got = s.vectorize(|| l1_f64(s, &da, &db));
            assert!((got - r_l1).abs() <= 1e-12 * r_l1.max(1.0), "l1_f64 n={n}");

            let ua = u8s(n, 3);
            let ub = u8s(n, 11);
            let r: u32 = ua
                .iter()
                .zip(&ub)
                .map(|(&x, &y)| u32::from(x) * u32::from(y))
                .sum();
            assert_eq!(s.vectorize(|| dot_u8(s, &ua, &ub)), r, "dot_u8 n={n}");
            let r: u32 = ua
                .iter()
                .zip(&ub)
                .map(|(&x, &y)| (x ^ y).count_ones())
                .sum();
            assert_eq!(s.vectorize(|| hamming_u8(s, &ua, &ub)), r, "hamming n={n}");

            let r_mixed: f64 = a
                .iter()
                .zip(&ub)
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum();
            let scale: f64 = a
                .iter()
                .zip(&ub)
                .map(|(&x, &y)| f64::from(x).abs() * f64::from(y))
                .sum();
            let got = s.vectorize(|| mixed_dot_u8_f32(s, &a, &ub));
            assert!(close(got.into(), r_mixed, scale, n), "mixed n={n}");

            let sa: Vec<u32> = ua.iter().map(|&x| u32::from(x % 4)).collect();
            let sb: Vec<u32> = ub.iter().map(|&x| u32::from(x % 4)).collect();
            let r = sa.iter().zip(&sb).filter(|(x, y)| x != y).count();
            assert_eq!(
                s.vectorize(|| slot_ne_u32(s, &sa, &sb)) as usize,
                r,
                "slot32 n={n}"
            );
            let ha: Vec<u16> = sa.iter().map(|&x| x as u16).collect();
            let hb: Vec<u16> = sb.iter().map(|&x| x as u16).collect();
            assert_eq!(
                s.vectorize(|| slot_ne_u16(s, &ha, &hb)) as usize,
                r,
                "slot16 n={n}"
            );
        }
    }

    #[test]
    fn kernels_match_reference_on_fallback_level() {
        check_level(Fallback::new());
    }

    #[test]
    fn kernels_match_reference_on_native_level() {
        fearless_simd::dispatch!(Level::new(), s => check_level(s));
    }

    #[test]
    fn hamming_and_slot16_cross_flush_boundaries() {
        // Long enough to cross the u8 popcount block and the i16 flush block
        // on every level.
        let n = 16 * 8192 * 2 + 77;
        let a: Vec<u8> = vec![0xff; n];
        let b: Vec<u8> = vec![0x00; n];
        let s = Level::new();
        let got = fearless_simd::dispatch!(s, s => hamming_u8(s, &a, &b));
        assert_eq!(got as usize, 8 * n);
        let ha: Vec<u16> = vec![1; n];
        let hb: Vec<u16> = vec![2; n];
        let got = fearless_simd::dispatch!(s, s => slot_ne_u16(s, &ha, &hb));
        assert_eq!(got as usize, n);
        let got = slot_ne_u16(Fallback::new(), &ha, &hb);
        assert_eq!(got as usize, n);
    }
}
