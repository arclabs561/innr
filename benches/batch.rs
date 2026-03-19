#![allow(missing_docs)]
//! Benchmarks for batch vector operations (PDX-style columnar layout).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use innr::batch::{
    batch_cosine, batch_dot, batch_knn, batch_knn_cosine, batch_l2_squared, batch_norms,
    VerticalBatch,
};
use rand::prelude::*;

fn random_vecs(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect())
        .collect()
}

fn random_vec(dim: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(99);
    (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect()
}

fn bench_batch_l2(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_l2");

    for (n, dim) in [(1000, 128), (1000, 768), (10000, 128), (10000, 768)] {
        let vectors = random_vecs(n, dim);
        let batch = VerticalBatch::from_rows(&vectors);
        let query = random_vec(dim);

        group.throughput(Throughput::Elements((n * dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("l2_squared", format!("{n}x{dim}")),
            &(),
            |bench, _| bench.iter(|| batch_l2_squared(black_box(&query), black_box(&batch))),
        );
    }

    group.finish();
}

fn bench_batch_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_dot");

    for (n, dim) in [(1000, 768), (10000, 768)] {
        let vectors = random_vecs(n, dim);
        let batch = VerticalBatch::from_rows(&vectors);
        let query = random_vec(dim);

        group.throughput(Throughput::Elements((n * dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("dot", format!("{n}x{dim}")),
            &(),
            |bench, _| bench.iter(|| batch_dot(black_box(&query), black_box(&batch))),
        );
    }

    group.finish();
}

fn bench_batch_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_cosine");

    for (n, dim) in [(1000, 768), (10000, 768)] {
        let vectors = random_vecs(n, dim);
        let batch = VerticalBatch::from_rows(&vectors);
        let norms = batch_norms(&batch);
        let query = random_vec(dim);

        group.throughput(Throughput::Elements((n * dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("cosine", format!("{n}x{dim}")),
            &(),
            |bench, _| {
                bench.iter(|| batch_cosine(black_box(&query), black_box(&batch), black_box(&norms)))
            },
        );
    }

    group.finish();
}

fn bench_batch_knn(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_knn");

    let n = 10000;
    let dim = 128;
    let vectors = random_vecs(n, dim);
    let batch = VerticalBatch::from_rows(&vectors);
    let query = random_vec(dim);

    for k in [1, 10, 100] {
        group.throughput(Throughput::Elements((n * dim) as u64));
        group.bench_with_input(BenchmarkId::new("l2", format!("k={k}")), &k, |bench, &k| {
            bench.iter(|| batch_knn(black_box(&query), black_box(&batch), k))
        });
    }

    group.finish();
}

fn bench_batch_knn_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_knn_cosine");

    let n = 10000;
    let dim = 128;
    let vectors = random_vecs(n, dim);
    let batch = VerticalBatch::from_rows(&vectors);
    let query = random_vec(dim);

    for k in [1, 10, 100] {
        group.throughput(Throughput::Elements((n * dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("cosine", format!("k={k}")),
            &k,
            |bench, &k| bench.iter(|| batch_knn_cosine(black_box(&query), black_box(&batch), k)),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_batch_l2,
    bench_batch_dot,
    bench_batch_cosine,
    bench_batch_knn,
    bench_batch_knn_cosine,
);
criterion_main!(benches);
