//! Differential fuzz: dispatched SIMD `dot`/`cosine` vs a scalar reference,
//! over arbitrary byte-decoded f32 (so the corpus reaches NaN, +/-inf,
//! subnormals, and -0.0 that the value-range proptest generators never make).
#![no_main]

use libfuzzer_sys::fuzz_target;

fn decode(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn ref_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fuzz_target!(|data: &[u8]| {
    let v = decode(data);
    if v.len() < 2 {
        return;
    }
    let n = v.len() / 2;
    let (a, b) = (&v[..n], &v[n..2 * n]);

    let simd = innr::dot(a, b);
    let scalar = ref_dot(a, b);
    // Only compare when both are finite: SIMD reorders the reduction, so NaN/inf
    // propagation order can legitimately differ; the contract is finite-equals.
    // Tolerance scales by the sum of product magnitudes, not the result: that
    // is the dot product's condition number, so near-cancellation inputs (where
    // both f32 reductions are legitimately imprecise) don't false-positive,
    // while a real kernel bug still blows past it.
    if simd.is_finite() && scalar.is_finite() {
        let mag: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x * y).abs()).sum();
        if mag.is_finite() {
            let tol = 1e-3 * mag + 1e-6;
            assert!(
                (simd - scalar).abs() <= tol,
                "dot diverged: simd={simd} scalar={scalar} mag={mag} n={n}"
            );
        }
    }
    // cosine must never panic or return out of [-1, 1] for finite inputs.
    let c = innr::cosine(a, b);
    assert!(
        !c.is_finite() || (-1.0001..=1.0001).contains(&c),
        "cosine out of range: {c}"
    );
});
