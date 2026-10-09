//! Fuzz the u8 scalar-quantization path over arbitrary byte-decoded f32
//! (NaN, +/-inf, subnormals): `quantize_u8` must not panic, finite in-range
//! values must dequantize to within half a step, and the dispatched SIMD
//! `mixed_dot_u8_f32` must match a scalar reference.
#![no_main]

use innr::scalar::{mixed_dot_u8_f32, quantize_u8, QuantizationParams};
use libfuzzer_sys::fuzz_target;

fn decode(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fuzz_target!(|data: &[u8]| {
    let v = decode(data);
    if v.len() < 2 {
        return;
    }
    let n = v.len() / 2;
    let (query, doc) = (&v[..n], &v[n..2 * n]);

    let params = QuantizationParams::fit(doc);
    let q = quantize_u8(doc, &params);
    assert_eq!(q.data().len(), n);

    // Round-trip: a finite value inside the fitted range lands within half a
    // quantization step (alpha / 255 / 2) of its reconstruction. Skipped when
    // the step is not a normal f32 or 255 / alpha overflows: for ranges that
    // small, codes saturate and the step underflows to 0 (a known limit of the
    // f32 affine map, not checked here).
    let step = params.alpha / 255.0;
    if params.alpha.is_finite()
        && params.offset.is_finite()
        && step.is_normal()
        && (255.0 / params.alpha).is_finite()
    {
        for (&x, &code) in doc.iter().zip(q.data()) {
            let lo = params.offset;
            let hi = params.offset + params.alpha;
            if x.is_finite() && x >= lo && x <= hi && step.is_finite() {
                let back = params.offset + f32::from(code) * step;
                // Half a step, plus f32 rounding of the affine map, which
                // scales with the magnitudes involved, not with the step.
                let ulp = 4.0 * f32::EPSILON * (lo.abs() + x.abs() + params.alpha);
                let tol = 0.5 * step * (1.0 + 1e-3) + ulp;
                assert!(
                    (back - x).abs() <= tol,
                    "round trip: x={x} code={code} back={back} step={step}"
                );
            }
        }
    }

    // SIMD mixed dot vs scalar reference, compared when both are finite.
    let simd = mixed_dot_u8_f32(query, q.data());
    let scalar: f32 = query
        .iter()
        .zip(q.data())
        .map(|(&a, &b)| a * f32::from(b))
        .sum();
    if simd.is_finite() && scalar.is_finite() {
        let mag: f32 = query
            .iter()
            .zip(q.data())
            .map(|(&a, &b)| (a * f32::from(b)).abs())
            .sum();
        if mag.is_finite() {
            let tol = 1e-3 * mag + 1e-6;
            assert!(
                (simd - scalar).abs() <= tol,
                "mixed dot diverged: simd={simd} scalar={scalar} mag={mag} n={n}"
            );
        }
    }
});
