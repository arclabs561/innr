//! NumKong reference column for the pair groups in innr's
//! `benches/consumer_pair.rs`. Group names, parameters and data seeds match
//! that file, so `benches/report.py` places these results in the same rows.

use std::fmt::Display;
use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion};
use innr_bench_support::data;

/// Must match `F32_DIMS` in `benches/consumer_pair.rs`.
const F32_DIMS: [usize; 8] = [16, 32, 64, 128, 384, 768, 1024, 1536];

fn run<O>(g: &mut BenchmarkGroup<'_, WallTime>, param: impl Display, mut f: impl FnMut() -> O) {
    g.bench_function(BenchmarkId::new("numkong", param), |b| b.iter(&mut f));
}

fn pair_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("pair_f32_dot");
    for d in F32_DIMS {
        let (a, b) = (data::f32s(d, 1), data::f32s(d, 2));
        run(&mut g, d, || {
            <f32 as numkong::Dot>::dot(black_box(&a), black_box(&b))
        });
    }
    g.finish();

    let mut g = c.benchmark_group("pair_f32_cosine");
    for d in F32_DIMS {
        let (a, b) = (data::f32s(d, 1), data::f32s(d, 2));
        run(&mut g, d, || {
            <f32 as numkong::Angular>::cosine(black_box(&a), black_box(&b))
        });
    }
    g.finish();

    let mut g = c.benchmark_group("pair_f32_l2sq");
    for d in F32_DIMS {
        let (a, b) = (data::f32s(d, 1), data::f32s(d, 2));
        run(&mut g, d, || {
            <f32 as numkong::Euclidean>::sqeuclidean(black_box(&a), black_box(&b))
        });
    }
    g.finish();
}

fn pair_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("pair_f64_l2sq");
    for n in [16usize, 784, 1_000, 100_000] {
        let (a, b) = (data::f64s(n, 1), data::f64s(n, 2));
        run(&mut g, n, || {
            <f64 as numkong::Euclidean>::sqeuclidean(black_box(&a), black_box(&b))
        });
    }
    g.finish();
}

fn hamming(c: &mut Criterion) {
    let mut g = c.benchmark_group("hamming_bits");
    for n in [8usize, 16, 32, 96] {
        let (a, b) = (data::bytes(n, 1), data::bytes(n, 2));
        let a1: Vec<numkong::u1x8> = a.iter().map(|&x| numkong::u1x8(x)).collect();
        let b1: Vec<numkong::u1x8> = b.iter().map(|&x| numkong::u1x8(x)).collect();
        run(&mut g, n, || {
            <numkong::u1x8 as numkong::Hamming>::hamming(black_box(&a1), black_box(&b1))
        });
    }
    g.finish();
}

criterion_group! {
    name = benches;
    config = innr_bench_support::config();
    targets = pair_f32, pair_f64, hamming
}
criterion_main!(benches);
