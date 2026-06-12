//! Differential fuzz: SIMD l2/l1 (f32 and f64) vs scalar references over
//! arbitrary byte-decoded floats. Targets the masked-tail and unrolled-chunk
//! boundaries with adversarial bit patterns.
#![no_main]

use libfuzzer_sys::fuzz_target;

fn ref_l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}
fn ref_l1(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}
fn ref_l2sq64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}
fn ref_l1_64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

fuzz_target!(|data: &[u8]| {
    let f32s: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if f32s.len() >= 2 {
        let n = f32s.len() / 2;
        let (a, b) = (&f32s[..n], &f32s[n..2 * n]);
        // l2sq/l1 sum non-negative terms, so the result IS the magnitude:
        // result-scaled tolerance is the condition-aware bound here.
        let (s, r) = (innr::l2_distance_squared(a, b), ref_l2sq(a, b));
        if s.is_finite() && r.is_finite() {
            assert!((s - r).abs() <= 1e-3 * r.abs().max(1.0) + 1e-6, "l2sq f32: {s} vs {r}");
        }
        let (s, r) = (innr::l1_distance(a, b), ref_l1(a, b));
        if s.is_finite() && r.is_finite() {
            assert!((s - r).abs() <= 1e-3 * r.abs().max(1.0) + 1e-6, "l1 f32: {s} vs {r}");
        }
    }

    let f64s: Vec<f64> = data
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();
    if f64s.len() >= 2 {
        let n = f64s.len() / 2;
        let (a, b) = (&f64s[..n], &f64s[n..2 * n]);
        let (s, r) = (innr::dense_f64::l2_distance_squared_f64(a, b), ref_l2sq64(a, b));
        if s.is_finite() && r.is_finite() {
            assert!((s - r).abs() <= 1e-9 * r.abs().max(1.0), "l2sq f64: {s} vs {r}");
        }
        let (s, r) = (innr::dense_f64::l1_distance_f64(a, b), ref_l1_64(a, b));
        if s.is_finite() && r.is_finite() {
            assert!((s - r).abs() <= 1e-9 * r.abs().max(1.0), "l1 f64: {s} vs {r}");
        }
    }
});
