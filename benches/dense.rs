//! Benchmarks for dense vector operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use innr::{cosine, dot, l2_distance, norm};
use rand::prelude::*;

fn random_vec(n: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot");

    for dim in [16, 64, 128, 256, 384, 512, 768, 1024, 1536] {
        let a = random_vec(dim);
        let b = random_vec(dim);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::new("dot", dim), &dim, |bench, _| {
            bench.iter(|| dot(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

fn bench_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine");

    for dim in [128, 384, 768, 1536] {
        let a = random_vec(dim);
        let b = random_vec(dim);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::new("cosine", dim), &dim, |bench, _| {
            bench.iter(|| cosine(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

fn bench_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("norm");

    for dim in [128, 384, 768, 1536] {
        let v = random_vec(dim);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::new("norm", dim), &dim, |bench, _| {
            bench.iter(|| norm(black_box(&v)))
        });
    }

    group.finish();
}

fn bench_l2_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_distance");

    for dim in [128, 384, 768, 1536] {
        let a = random_vec(dim);
        let b = random_vec(dim);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::new("l2", dim), &dim, |bench, _| {
            bench.iter(|| l2_distance(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_dot,
    bench_cosine,
    bench_norm,
    bench_l2_distance
);
criterion_main!(benches);
