#![cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]

//! Bit-exact pins: outputs of the 917749d kernels (the last std::arch
//! version) on fixed fixtures. These pins check deterministic outputs on the
//! recorded backend. SIMD fast cosine uses hardware reciprocal-square-root
//! estimates, whose bits need not agree across CPUs or emulation; it is checked
//! against the documented accuracy bound instead. These fixtures do not prove
//! bit identity for every kernel on every platform.
//!
//! Pins were produced on aarch64 (NEON) and on x86_64 under Rosetta
//! (AVX2+FMA, no AVX-512). AVX-512 paths are not pinned: on a machine without
//! V3 or with V4 only the portable pins are checked. `p_*` entries are portable kernels,
//! which is what the public functions run with the `simd` feature off.

#[cfg(target_arch = "aarch64")]
const PINS_AARCH64: &[(&str, usize, u64)] = &[
    ("dot", 17, 0xbcf41e58),
    ("cosine", 17, 0xbb9ee6cd),
    ("l1", 17, 0x41358ea6),
    ("l2sq", 17, 0x4145df06),
    ("mixed_u8", 17, 0xc3aa2d8c),
    ("maxsim", 17, 0x3fc3a3a3),
    ("fast_cosine", 17, 0xbb9ee6cc),
    ("dot_f64", 17, 0x40006254d9d45238),
    ("cosine_f64", 17, 0x3fc9578837249123),
    ("l2sq_f64", 17, 0x4030c298d46b48c0),
    ("l1_f64", 17, 0x402b1720b4000000),
    ("p_dot", 17, 0xbcf41e78),
    ("p_cosine", 17, 0xbb9ee6e2),
    ("p_l1", 17, 0x41358ea5),
    ("p_l2sq", 17, 0x4145df06),
    ("p_mixed_u8", 17, 0xc3aa2d8b),
    ("p_maxsim", 17, 0x3fc3a3a3),
    ("p_fast_cosine", 17, 0xbb9e7422),
    ("p_dot_f64", 17, 0x40006254d9d45238),
    ("p_l2sq_f64", 17, 0x4030c298d46b48c0),
    ("p_l1_f64", 17, 0x402b1720b4000000),
    ("dot", 33, 0xc01bf5b9),
    ("cosine", 33, 0xbe629cca),
    ("l1", 33, 0x41c5be41),
    ("l2sq", 33, 0x41d84549),
    ("mixed_u8", 33, 0xc43e5f4f),
    ("maxsim", 33, 0x3f2732d0),
    ("fast_cosine", 33, 0xbe629cc9),
    ("dot_f64", 33, 0xbfeebbbfd9b799e0),
    ("cosine_f64", 33, 0xbfac1185773b3db9),
    ("l2sq_f64", 33, 0x40428e5707f46e1c),
    ("l1_f64", 33, 0x403c29ffce000000),
    ("p_dot", 33, 0xc01bf5b8),
    ("p_cosine", 33, 0xbe629cc9),
    ("p_l1", 33, 0x41c5be40),
    ("p_l2sq", 33, 0x41d84548),
    ("p_mixed_u8", 33, 0xc43e5f4e),
    ("p_maxsim", 33, 0x3f2732d0),
    ("p_fast_cosine", 33, 0xbe62182c),
    ("p_dot_f64", 33, 0xbfeebbbfd9b799e0),
    ("p_l2sq_f64", 33, 0x40428e5707f46e1c),
    ("p_l1_f64", 33, 0x403c29ffce000000),
    ("dot", 100, 0x3f6631b0),
    ("cosine", 100, 0x3cd10088),
    ("l1", 100, 0x42886a3c),
    ("l2sq", 100, 0x428979aa),
    ("mixed_u8", 100, 0x43a61558),
    ("maxsim", 100, 0x40eec59c),
    ("fast_cosine", 100, 0x3cd10087),
    ("dot_f64", 100, 0x3ff759bfd3b6b3a0),
    ("cosine_f64", 100, 0x3f998a3e10a3897c),
    ("l2sq_f64", 100, 0x405caa3a7cecff19),
    ("l1_f64", 100, 0x40561aee1b200000),
    ("p_dot", 100, 0x3f6631ba),
    ("p_cosine", 100, 0x3cd10091),
    ("p_l1", 100, 0x42886a3d),
    ("p_l2sq", 100, 0x428979ab),
    ("p_mixed_u8", 100, 0x43a6155c),
    ("p_maxsim", 100, 0x40eec59c),
    ("p_fast_cosine", 100, 0x3cd098f4),
    ("p_dot_f64", 100, 0x3ff759bfd3b6b3a0),
    ("p_l2sq_f64", 100, 0x405caa3a7cecff17),
    ("p_l1_f64", 100, 0x40561aee1b200000),
    ("dot", 257, 0x409cc0e5),
    ("cosine", 257, 0x3d693755),
    ("l1", 257, 0x43281084),
    ("l2sq", 257, 0x43226141),
    ("mixed_u8", 257, 0x44b2a1c2),
    ("maxsim", 257, 0x4116dad2),
    ("fast_cosine", 257, 0x3d693755),
    ("dot_f64", 257, 0xc00560a74788be90),
    ("cosine_f64", 257, 0xbf92dbdad3765c8f),
    ("l2sq_f64", 257, 0x40727c67ad3813d8),
    ("l1_f64", 257, 0x406c5a9843a00000),
    ("p_dot", 257, 0x409cc0e5),
    ("p_cosine", 257, 0x3d693755),
    ("p_l1", 257, 0x43281086),
    ("p_l2sq", 257, 0x4322613f),
    ("p_mixed_u8", 257, 0x44b2a1c3),
    ("p_maxsim", 257, 0x4116dad2),
    ("p_fast_cosine", 257, 0x3d68ead1),
    ("p_dot_f64", 257, 0xc00560a74788be90),
    ("p_l2sq_f64", 257, 0x40727c67ad3813d9),
    ("p_l1_f64", 257, 0x406c5a9843a00000),
    ("dot", 1000, 0x3f43e32f),
    ("cosine", 1000, 0x3b14ed7b),
    ("l1", 1000, 0x442791b0),
    ("l2sq", 1000, 0x44281eee),
    ("mixed_u8", 1000, 0x43ddfd9e),
    ("maxsim", 1000, 0x4176bcc4),
    ("fast_cosine", 1000, 0x3b14ed7b),
    ("dot_f64", 1000, 0xc00b84d348438530),
    ("cosine_f64", 1000, 0xbf78dc7851a185b7),
    ("l2sq_f64", 1000, 0x4091d3e0581cd226),
    ("l1_f64", 1000, 0x408b177589900000),
    ("p_dot", 1000, 0x3f43e2fc),
    ("p_cosine", 1000, 0x3b14ed54),
    ("p_l1", 1000, 0x442791ae),
    ("p_l2sq", 1000, 0x44281ef1),
    ("p_mixed_u8", 1000, 0x43ddfd82),
    ("p_maxsim", 1000, 0x4176bcc4),
    ("p_fast_cosine", 1000, 0x3b14c728),
    ("p_dot_f64", 1000, 0xc00b84d348438530),
    ("p_l2sq_f64", 1000, 0x4091d3e0581cd226),
    ("p_l1_f64", 1000, 0x408b177589900000),
];

#[cfg(target_arch = "x86_64")]
const PINS_X86_AVX2: &[(&str, usize, u64)] = &[
    ("dot", 17, 0xbcf41e58),
    ("cosine", 17, 0xbb9ee6cd),
    ("l1", 17, 0x41358ea5),
    ("l2sq", 17, 0x4145df05),
    ("mixed_u8", 17, 0xc3aa2d8c),
    ("maxsim", 17, 0x3fc3a3a3),
    ("fast_cosine", 17, 0xbb9ee6cd),
    ("dot_f64", 17, 0x40006254d9d45238),
    ("cosine_f64", 17, 0x3fc9578837249123),
    ("l2sq_f64", 17, 0x4030c298d46b48c1),
    ("l1_f64", 17, 0x402b1720b4000000),
    ("p_dot", 17, 0xbcf41e78),
    ("p_cosine", 17, 0xbb9ee6e2),
    ("p_l1", 17, 0x41358ea5),
    ("p_l2sq", 17, 0x4145df06),
    ("p_mixed_u8", 17, 0xc3aa2d8b),
    ("p_maxsim", 17, 0x3fc3a3a3),
    ("p_fast_cosine", 17, 0xbb9e7422),
    ("p_dot_f64", 17, 0x40006254d9d45238),
    ("p_l2sq_f64", 17, 0x4030c298d46b48c0),
    ("p_l1_f64", 17, 0x402b1720b4000000),
    ("dot", 33, 0xc01bf5b9),
    ("cosine", 33, 0xbe629cca),
    ("l1", 33, 0x41c5be40),
    ("l2sq", 33, 0x41d84548),
    ("mixed_u8", 33, 0xc43e5f4f),
    ("maxsim", 33, 0x3f2732d0),
    ("fast_cosine", 33, 0xbe629ccb),
    ("dot_f64", 33, 0xbfeebbbfd9b799e0),
    ("cosine_f64", 33, 0xbfac1185773b3db9),
    ("l2sq_f64", 33, 0x40428e5707f46e1c),
    ("l1_f64", 33, 0x403c29ffce000000),
    ("p_dot", 33, 0xc01bf5b8),
    ("p_cosine", 33, 0xbe629cc9),
    ("p_l1", 33, 0x41c5be40),
    ("p_l2sq", 33, 0x41d84548),
    ("p_mixed_u8", 33, 0xc43e5f4e),
    ("p_maxsim", 33, 0x3f2732d0),
    ("p_fast_cosine", 33, 0xbe62182c),
    ("p_dot_f64", 33, 0xbfeebbbfd9b799e0),
    ("p_l2sq_f64", 33, 0x40428e5707f46e1c),
    ("p_l1_f64", 33, 0x403c29ffce000000),
    ("dot", 100, 0x3f6631b2),
    ("cosine", 100, 0x3cd1008b),
    ("l1", 100, 0x42886a3d),
    ("l2sq", 100, 0x428979ab),
    ("mixed_u8", 100, 0x43a61556),
    ("maxsim", 100, 0x40eec59e),
    ("fast_cosine", 100, 0x3cd10089),
    ("dot_f64", 100, 0x3ff759bfd3b6b3a0),
    ("cosine_f64", 100, 0x3f998a3e10a3897d),
    ("l2sq_f64", 100, 0x405caa3a7cecff1a),
    ("l1_f64", 100, 0x40561aee1b200000),
    ("p_dot", 100, 0x3f6631ba),
    ("p_cosine", 100, 0x3cd10091),
    ("p_l1", 100, 0x42886a3d),
    ("p_l2sq", 100, 0x428979ab),
    ("p_mixed_u8", 100, 0x43a6155c),
    ("p_maxsim", 100, 0x40eec59e),
    ("p_fast_cosine", 100, 0x3cd098f4),
    ("p_dot_f64", 100, 0x3ff759bfd3b6b3a0),
    ("p_l2sq_f64", 100, 0x405caa3a7cecff17),
    ("p_l1_f64", 100, 0x40561aee1b200000),
    ("dot", 257, 0x409cc0e5),
    ("cosine", 257, 0x3d693755),
    ("l1", 257, 0x43281085),
    ("l2sq", 257, 0x43226141),
    ("mixed_u8", 257, 0x44b2a1c2),
    ("maxsim", 257, 0x4116dad4),
    ("fast_cosine", 257, 0x3d693755),
    ("dot_f64", 257, 0xc00560a74788be90),
    ("cosine_f64", 257, 0xbf92dbdad3765c8f),
    ("l2sq_f64", 257, 0x40727c67ad3813d9),
    ("l1_f64", 257, 0x406c5a9843a00000),
    ("p_dot", 257, 0x409cc0e5),
    ("p_cosine", 257, 0x3d693755),
    ("p_l1", 257, 0x43281086),
    ("p_l2sq", 257, 0x4322613f),
    ("p_mixed_u8", 257, 0x44b2a1c3),
    ("p_maxsim", 257, 0x4116dad4),
    ("p_fast_cosine", 257, 0x3d68ead1),
    ("p_dot_f64", 257, 0xc00560a74788be90),
    ("p_l2sq_f64", 257, 0x40727c67ad3813d9),
    ("p_l1_f64", 257, 0x406c5a9843a00000),
    ("dot", 1000, 0x3f43e31f),
    ("cosine", 1000, 0x3b14ed6e),
    ("l1", 1000, 0x442791af),
    ("l2sq", 1000, 0x44281eee),
    ("mixed_u8", 1000, 0x43ddfda0),
    ("maxsim", 1000, 0x4176bcc7),
    ("fast_cosine", 1000, 0x3b14ed69),
    ("dot_f64", 1000, 0xc00b84d348438530),
    ("cosine_f64", 1000, 0xbf78dc7851a185b7),
    ("l2sq_f64", 1000, 0x4091d3e0581cd223),
    ("l1_f64", 1000, 0x408b177589900000),
    ("p_dot", 1000, 0x3f43e2fc),
    ("p_cosine", 1000, 0x3b14ed54),
    ("p_l1", 1000, 0x442791ae),
    ("p_l2sq", 1000, 0x44281ef1),
    ("p_mixed_u8", 1000, 0x43ddfd82),
    ("p_maxsim", 1000, 0x4176bcc7),
    ("p_fast_cosine", 1000, 0x3b14c728),
    ("p_dot_f64", 1000, 0xc00b84d348438530),
    ("p_l2sq_f64", 1000, 0x4091d3e0581cd226),
    ("p_l1_f64", 1000, 0x408b177589900000),
];

fn fixtures() -> Vec<(&'static str, usize, u64)> {
    use crate::*;
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let mut out = Vec::new();
    for n in [17usize, 33, 100, 257, 1000] {
        let a: Vec<f32> = (0..n).map(|_| rnd()).collect();
        let b: Vec<f32> = (0..n).map(|_| rnd()).collect();
        let u: Vec<u8> = (0..n).map(|_| ((rnd() + 1.0) * 127.0) as u8).collect();
        let da: Vec<f64> = a.iter().map(|&x| f64::from(x) * 1.25).collect();
        let db: Vec<f64> = b.iter().map(|&x| f64::from(x) - 0.5).collect();
        let q: Vec<&[f32]> = a.chunks(n.div_ceil(3)).collect();
        let d: Vec<&[f32]> = b.chunks(n.div_ceil(3)).collect();
        let dim = q.iter().chain(&d).map(|s| s.len()).min().unwrap();
        let q: Vec<&[f32]> = q.iter().map(|s| &s[..dim]).collect();
        let d: Vec<&[f32]> = d.iter().map(|s| &s[..dim]).collect();
        out.push(("dot", n, u64::from(dot(&a, &b).to_bits())));
        out.push(("cosine", n, u64::from(cosine(&a, &b).to_bits())));
        out.push(("l1", n, u64::from(l1_distance(&a, &b).to_bits())));
        out.push(("l2sq", n, u64::from(l2_distance_squared(&a, &b).to_bits())));
        out.push((
            "mixed_u8",
            n,
            u64::from(scalar::mixed_dot_u8_f32(&a, &u).to_bits()),
        ));
        out.push(("maxsim", n, u64::from(maxsim(&q, &d).to_bits())));
        let fast_cosine = crate::fast_math::fast_cosine_dispatch(&a, &b);
        // Independent mathematical reference, not another Innr cosine kernel.
        // These fixed nonzero inputs exercise whichever backend dispatch selects,
        // including the portable fallback. Preserve the existing absolute 0.01
        // fast-cosine accuracy contract even where exact SIMD pins do not apply.
        let dot: f64 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        let aa: f64 = a.iter().map(|&x| f64::from(x).powi(2)).sum();
        let bb: f64 = b.iter().map(|&x| f64::from(x).powi(2)).sum();
        let reference = dot / (aa * bb).sqrt();
        assert!(
            (f64::from(fast_cosine) - reference).abs() < 0.01,
            "fast_cosine accuracy n={n}: {fast_cosine} vs {reference}"
        );
        out.push(("fast_cosine", n, u64::from(fast_cosine.to_bits())));
        out.push(("dot_f64", n, dense_f64::dot_f64(&da, &db).to_bits()));
        out.push(("cosine_f64", n, dense_f64::cosine_f64(&da, &db).to_bits()));
        out.push((
            "l2sq_f64",
            n,
            dense_f64::l2_distance_squared_f64(&da, &db).to_bits(),
        ));
        out.push(("l1_f64", n, dense_f64::l1_distance_f64(&da, &db).to_bits()));
        out.push(("p_dot", n, u64::from(dense::dot_portable(&a, &b).to_bits())));
        out.push((
            "p_cosine",
            n,
            u64::from(dense::cosine_portable(&a, &b).to_bits()),
        ));
        out.push((
            "p_l1",
            n,
            u64::from(dense::l1_distance_portable(&a, &b).to_bits()),
        ));
        out.push((
            "p_l2sq",
            n,
            u64::from(dense::l2_distance_squared_portable(&a, &b).to_bits()),
        ));
        out.push((
            "p_mixed_u8",
            n,
            u64::from(scalar::mixed_dot_u8_f32_portable(&a, &u).to_bits()),
        ));
        out.push((
            "p_maxsim",
            n,
            u64::from(crate::maxsim::maxsim_portable(&q, &d).to_bits()),
        ));
        out.push((
            "p_fast_cosine",
            n,
            u64::from(crate::fast_math::fast_cosine(&a, &b).to_bits()),
        ));
        out.push((
            "p_dot_f64",
            n,
            dense_f64::dot_f64_portable(&da, &db).to_bits(),
        ));
        out.push((
            "p_l2sq_f64",
            n,
            dense_f64::l2_distance_squared_f64_portable(&da, &db).to_bits(),
        ));
        out.push((
            "p_l1_f64",
            n,
            dense_f64::l1_distance_f64_portable(&da, &db).to_bits(),
        ));
    }
    out
}

#[test]
fn outputs_match_917749d_bit_for_bit() {
    #[cfg(target_arch = "aarch64")]
    let pins = PINS_AARCH64;
    #[cfg(target_arch = "x86_64")]
    let pins = PINS_X86_AVX2;
    // The x86 dispatch pins were recorded with V3, without V4. Portable
    // kernels remain checkable on other x86 CPUs, except maxsim which
    // dispatches internally.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let dispatch_matches = pulp::x86::V3::try_new().is_some() && pulp::x86::V4::try_new().is_none();
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    let dispatch_matches = true;

    let mut checked = 0;
    for (name, n, got) in fixtures() {
        // Keep the historical values as provenance, but hardware rsqrt estimates
        // are not a cross-machine bit oracle. Portable fast cosine remains pinned.
        if cfg!(feature = "simd") && name == "fast_cosine" {
            continue;
        }
        if !dispatch_matches && (!name.starts_with("p_") || name == "p_maxsim") {
            continue;
        }
        // With `simd` off the public functions are the portable kernels, so
        // compare them against the portable pins.
        let key = if cfg!(feature = "simd") || name.starts_with("p_") {
            name.to_string()
        } else {
            format!("p_{name}")
        };
        // maxsim and cosine_f64 call the dispatching dot/norms, which were
        // SIMD at 917749d, so they have no portable-only pin.
        if !cfg!(feature = "simd") && matches!(name, "maxsim" | "p_maxsim" | "cosine_f64") {
            continue;
        }
        let Some(&(_, _, want)) = pins.iter().find(|(k, m, _)| *k == key && *m == n) else {
            panic!("no pin for {key} n={n}");
        };
        assert_eq!(got, want, "{name} n={n}: {got:#x} vs pinned {want:#x}");
        checked += 1;
    }
    assert_eq!(
        checked,
        if !dispatch_matches {
            45
        } else if cfg!(feature = "simd") {
            100
        } else {
            90
        }
    );
}
