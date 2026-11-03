//! Accuracy bounds against a higher-precision reference.
//!
//! The SIMD and portable kernels sum in different orders (vector lanes,
//! several accumulators, FMA on the SIMD levels), so they agree to rounding,
//! not bit for bit, and a change of summation order is a legitimate change.
//! These tests bound each kernel's error instead: every reduction is checked
//! against an `f64` (for `f32` kernels) or compensated-`f64` (for `f64`
//! kernels) reference, with the textbook worst-case bound for any summation
//! order, `|error| <= (n + 2) * eps * Σ|term|`. Dropping, duplicating or
//! misaligning even one term exceeds the bound on these fixtures.
//!
//! Both the dispatched public functions and the portable kernels are checked,
//! so the bound holds with and without the `simd` feature.

use crate::*;

/// Fixed pseudo-random fixtures in [-1, 1), the same generator the earlier
/// bit-exact pins used.
fn rng() -> impl FnMut() -> f32 {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

const LENS: &[usize] = &[17, 33, 100, 257, 1000, 4099];

/// Neumaier-compensated sum: error about `eps²·Σ|x|`, a reference for `f64`.
fn sum_compensated(xs: impl Iterator<Item = f64>) -> f64 {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for x in xs {
        let t = s + x;
        c += if s.abs() >= x.abs() {
            (s - t) + x
        } else {
            (x - t) + s
        };
        s = t;
    }
    s + c
}

#[track_caller]
fn assert_bounded(what: &str, n: usize, got: f64, want: f64, scale: f64, eps: f64) {
    let bound = (n as f64 + 2.0) * eps * scale;
    assert!(
        (got - want).abs() <= bound,
        "{what} n={n}: got {got:e}, reference {want:e}, error {:e} > bound {bound:e}",
        (got - want).abs()
    );
}

#[test]
fn f32_reductions_are_within_worst_case_bound() {
    let mut rnd = rng();
    let eps = f64::from(f32::EPSILON);
    for &n in LENS {
        let a: Vec<f32> = (0..n).map(|_| rnd()).collect();
        let b: Vec<f32> = (0..n).map(|_| rnd()).collect();
        let pairs = || {
            a.iter()
                .zip(&b)
                .map(|(&x, &y)| (f64::from(x), f64::from(y)))
        };

        let r_dot: f64 = pairs().map(|(x, y)| x * y).sum();
        let s_dot: f64 = pairs().map(|(x, y)| (x * y).abs()).sum();
        assert_bounded("dot", n, dot(&a, &b).into(), r_dot, s_dot, eps);
        assert_bounded(
            "dot_portable",
            n,
            dense::dot_portable(&a, &b).into(),
            r_dot,
            s_dot,
            eps,
        );

        let r_l2: f64 = pairs().map(|(x, y)| (x - y) * (x - y)).sum();
        assert_bounded(
            "l2sq",
            n,
            l2_distance_squared(&a, &b).into(),
            r_l2,
            r_l2,
            eps,
        );
        let got = dense::l2_distance_squared_portable(&a, &b);
        assert_bounded("l2sq_portable", n, got.into(), r_l2, r_l2, eps);

        let r_l1: f64 = pairs().map(|(x, y)| (x - y).abs()).sum();
        assert_bounded("l1", n, l1_distance(&a, &b).into(), r_l1, r_l1, eps);
        let got = dense::l1_distance_portable(&a, &b);
        assert_bounded("l1_portable", n, got.into(), r_l1, r_l1, eps);

        let r_aa: f64 = a.iter().map(|&x| f64::from(x).powi(2)).sum();
        let r_bb: f64 = b.iter().map(|&x| f64::from(x).powi(2)).sum();
        // norm = sqrt(Σx²): relative error of the sum halves under sqrt.
        let got = f64::from(norm(&a));
        assert_bounded("norm", n, got * got, r_aa, r_aa, eps);

        // Cosine: first-order error from the three sums, then the divide.
        let (na, nb) = (r_aa.sqrt(), r_bb.sqrt());
        let r_cos = r_dot / (na * nb);
        let s_cos = s_dot / (na * nb) + r_cos.abs() + 4.0 / (n as f64 + 2.0);
        assert_bounded("cosine", n, cosine(&a, &b).into(), r_cos, s_cos, eps);
        let got = dense::cosine_portable(&a, &b);
        assert_bounded("cosine_portable", n, got.into(), r_cos, s_cos, eps);
        // Without SIMD the dispatched fast cosine is the rsqrt estimate.
        if cfg!(feature = "simd") && n >= 16 {
            let got = fast_cosine_dispatch(&a, &b);
            assert_bounded("fast_cosine_dispatch", n, got.into(), r_cos, s_cos, eps);
        }

        let u: Vec<u8> = (0..n).map(|_| ((rnd() + 1.0) * 127.0) as u8).collect();
        let r_mixed: f64 = a
            .iter()
            .zip(&u)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        let s_mixed: f64 = a
            .iter()
            .zip(&u)
            .map(|(&x, &y)| (f64::from(x) * f64::from(y)).abs())
            .sum();
        let got = scalar::mixed_dot_u8_f32(&a, &u);
        assert_bounded("mixed_u8", n, got.into(), r_mixed, s_mixed, eps);
        let got = scalar::mixed_dot_u8_f32_portable(&a, &u);
        assert_bounded("mixed_u8_portable", n, got.into(), r_mixed, s_mixed, eps);
    }
}

#[test]
fn maxsim_is_within_worst_case_bound() {
    let mut rnd = rng();
    let eps = f64::from(f32::EPSILON);
    for dim in [8usize, 16, 33, 128] {
        let q: Vec<Vec<f32>> = (0..5).map(|_| (0..dim).map(|_| rnd()).collect()).collect();
        let d: Vec<Vec<f32>> = (0..7).map(|_| (0..dim).map(|_| rnd()).collect()).collect();
        let qr: Vec<&[f32]> = q.iter().map(Vec::as_slice).collect();
        let dr: Vec<&[f32]> = d.iter().map(Vec::as_slice).collect();
        let exact = |x: &[f32], y: &[f32]| -> f64 {
            x.iter()
                .zip(y)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum()
        };
        let scale = |x: &[f32], y: &[f32]| -> f64 {
            x.iter()
                .zip(y)
                .map(|(&a, &b)| (f64::from(a) * f64::from(b)).abs())
                .sum()
        };
        // Each per-query max is off by at most that pair's bound, plus the
        // final sum over queries.
        let want: f64 = qr
            .iter()
            .map(|x| dr.iter().map(|y| exact(x, y)).fold(f64::MIN, f64::max))
            .sum();
        let s: f64 = qr
            .iter()
            .map(|x| dr.iter().map(|y| scale(x, y)).fold(0.0, f64::max))
            .sum::<f64>()
            + want.abs();
        assert_bounded("maxsim", dim, maxsim(&qr, &dr).into(), want, s, eps);
        let got = crate::maxsim::maxsim_portable(&qr, &dr);
        assert_bounded("maxsim_portable", dim, got.into(), want, s, eps);
    }
}

#[test]
fn f64_reductions_are_within_worst_case_bound() {
    let mut rnd = rng();
    let eps = f64::EPSILON;
    for &n in LENS {
        let a: Vec<f64> = (0..n).map(|_| f64::from(rnd()) * 1.25).collect();
        let b: Vec<f64> = (0..n).map(|_| f64::from(rnd()) - 0.5).collect();
        let pairs = || a.iter().zip(&b).map(|(&x, &y)| (x, y));

        let r_dot = sum_compensated(pairs().map(|(x, y)| x * y));
        let s_dot: f64 = pairs().map(|(x, y)| (x * y).abs()).sum();
        assert_bounded("dot_f64", n, dense_f64::dot_f64(&a, &b), r_dot, s_dot, eps);
        let got = dense_f64::dot_f64_portable(&a, &b);
        assert_bounded("dot_f64_portable", n, got, r_dot, s_dot, eps);

        let r_l2 = sum_compensated(pairs().map(|(x, y)| (x - y) * (x - y)));
        let got = dense_f64::l2_distance_squared_f64(&a, &b);
        assert_bounded("l2sq_f64", n, got, r_l2, r_l2, eps);
        let got = dense_f64::l2_distance_squared_f64_portable(&a, &b);
        assert_bounded("l2sq_f64_portable", n, got, r_l2, r_l2, eps);

        let r_l1 = sum_compensated(pairs().map(|(x, y)| (x - y).abs()));
        let got = dense_f64::l1_distance_f64(&a, &b);
        assert_bounded("l1_f64", n, got, r_l1, r_l1, eps);
        let got = dense_f64::l1_distance_f64_portable(&a, &b);
        assert_bounded("l1_f64_portable", n, got, r_l1, r_l1, eps);

        let r_aa = sum_compensated(a.iter().map(|x| x * x));
        let r_bb = sum_compensated(b.iter().map(|x| x * x));
        let r_cos = r_dot / (r_aa.sqrt() * r_bb.sqrt());
        let s_cos = s_dot / (r_aa.sqrt() * r_bb.sqrt()) + r_cos.abs() + 4.0 / (n as f64 + 2.0);
        let got = dense_f64::cosine_f64(&a, &b);
        assert_bounded("cosine_f64", n, got, r_cos, s_cos, eps);
    }
}

/// Integer kernels have no rounding: they must equal the reference exactly.
#[test]
fn integer_kernels_are_exact() {
    let mut rnd = rng();
    for &n in LENS {
        let a: Vec<u8> = (0..n).map(|_| ((rnd() + 1.0) * 127.9) as u8).collect();
        let b: Vec<u8> = (0..n).map(|_| ((rnd() + 1.0) * 127.9) as u8).collect();
        let r: u64 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| u64::from(x) * u64::from(y))
            .sum();
        assert_eq!(dot_u8_wide(&a, &b), r, "dot_u8_wide n={n}");
        assert_eq!(
            quant::dot_u8_wide_portable(&a, &b),
            r,
            "dot_u8_wide_portable n={n}"
        );
        let r: u32 = a.iter().zip(&b).map(|(&x, &y)| (x ^ y).count_ones()).sum();
        assert_eq!(hamming_distance(&a, &b), r, "hamming n={n}");
        assert_eq!(quant::hamming_portable(&a, &b), r, "hamming_portable n={n}");
    }
}
