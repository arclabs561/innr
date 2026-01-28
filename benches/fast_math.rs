//! Benchmarks comparing fast_math approximations vs standard implementations.
//!
//! These benchmarks demonstrate the speedup from Newton-Raphson rsqrt
//! and other fast math techniques.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use innr::{cosine, fast_math};
use rand::prelude::*;

fn random_vec(n: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n).map(|_| rng.random_range(-1.0..1.0)).collect()
}

/// Benchmark fast_rsqrt vs standard 1/sqrt
fn bench_rsqrt(c: &mut Criterion) {
    let mut group = c.benchmark_group("rsqrt");

    // Test values
    let values: Vec<f32> = (1..=1000).map(|i| i as f32 * 0.01).collect();

    group.bench_function("standard_rsqrt", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for &x in black_box(&values) {
                sum += 1.0 / x.sqrt();
            }
            sum
        })
    });

    group.bench_function("fast_rsqrt", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for &x in black_box(&values) {
                sum += fast_math::fast_rsqrt(x);
            }
            sum
        })
    });

    group.bench_function("fast_rsqrt_precise", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for &x in black_box(&values) {
                sum += fast_math::fast_rsqrt_precise(x);
            }
            sum
        })
    });

    group.finish();
}

/// Benchmark fast_cosine vs standard cosine
fn bench_cosine_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_comparison");

    for dim in [128, 384, 768, 1024, 1536] {
        let a = random_vec(dim);
        let b = random_vec(dim);

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("standard", dim), &dim, |bench, _| {
            bench.iter(|| cosine(black_box(&a), black_box(&b)))
        });

        group.bench_with_input(BenchmarkId::new("fast_portable", dim), &dim, |bench, _| {
            bench.iter(|| fast_math::fast_cosine(black_box(&a), black_box(&b)))
        });

        group.bench_with_input(BenchmarkId::new("fast_dispatch", dim), &dim, |bench, _| {
            bench.iter(|| fast_math::fast_cosine_dispatch(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

/// Benchmark batch cosine computations
fn bench_batch_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_cosine");

    let dim = 768; // Common embedding dimension
    let num_vectors = 100;

    let query = random_vec(dim);
    let vectors: Vec<Vec<f32>> = (0..num_vectors).map(|_| random_vec(dim)).collect();

    group.throughput(Throughput::Elements((num_vectors * dim) as u64));

    group.bench_function("standard_batch", |b| {
        b.iter(|| {
            let mut scores = Vec::with_capacity(num_vectors);
            for v in black_box(&vectors) {
                scores.push(cosine(black_box(&query), v));
            }
            scores
        })
    });

    group.bench_function("fast_batch", |b| {
        b.iter(|| {
            let mut scores = Vec::with_capacity(num_vectors);
            for v in black_box(&vectors) {
                scores.push(fast_math::fast_cosine(black_box(&query), v));
            }
            scores
        })
    });

    group.bench_function("fast_dispatch_batch", |b| {
        b.iter(|| {
            let mut scores = Vec::with_capacity(num_vectors);
            for v in black_box(&vectors) {
                scores.push(fast_math::fast_cosine_dispatch(black_box(&query), v));
            }
            scores
        })
    });

    group.finish();
}

/// Benchmark edge cases (non-aligned dimensions)
fn bench_non_aligned(c: &mut Criterion) {
    let mut group = c.benchmark_group("non_aligned");

    // Common non-aligned dimensions (OpenAI: 1536, but often truncated to 1535, etc.)
    for dim in [127, 255, 383, 511, 767, 1023, 1535] {
        let a = random_vec(dim);
        let b = random_vec(dim);

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("standard", dim), &dim, |bench, _| {
            bench.iter(|| cosine(black_box(&a), black_box(&b)))
        });

        group.bench_with_input(BenchmarkId::new("fast_dispatch", dim), &dim, |bench, _| {
            bench.iter(|| fast_math::fast_cosine_dispatch(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_rsqrt,
    bench_cosine_comparison,
    bench_batch_cosine,
    bench_non_aligned,
);
criterion_main!(benches);
