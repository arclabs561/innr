//! Benchmarks for ternary quantization operations.
//!
//! Demonstrates the compression ratio and speed of ternary vectors
//! compared to full-precision f32 vectors.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use innr::ternary::{asymmetric_dot, encode_ternary, ternary_dot, PackedTernary};
use rand::prelude::*;

fn random_vec(n: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

/// Benchmark encoding f32 vectors to ternary
fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("ternary_encode");

    for dim in [384, 768, 1536, 3072] {
        let vec = random_vec(dim);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::new("encode", dim), &dim, |bench, _| {
            bench.iter(|| encode_ternary(black_box(&vec), 0.3))
        });
    }

    group.finish();
}

/// Benchmark ternary dot product (symmetric)
fn bench_ternary_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("ternary_dot");

    for dim in [384, 768, 1536, 3072] {
        let a = encode_ternary(&random_vec(dim), 0.3);
        let b = encode_ternary(&random_vec(dim), 0.3);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::new("ternary", dim), &dim, |bench, _| {
            bench.iter(|| ternary_dot(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

/// Benchmark asymmetric dot (f32 query, ternary document)
fn bench_asymmetric_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("asymmetric_dot");

    for dim in [384, 768, 1536, 3072] {
        let query = random_vec(dim);
        let doc = encode_ternary(&random_vec(dim), 0.3);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::new("asymmetric", dim), &dim, |bench, _| {
            bench.iter(|| asymmetric_dot(black_box(&query), black_box(&doc)))
        });
    }

    group.finish();
}

/// Compare ternary vs f32 dot product
fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("ternary_vs_f32");

    for dim in [768, 1536] {
        let a_f32 = random_vec(dim);
        let b_f32 = random_vec(dim);
        let a_ternary = encode_ternary(&a_f32, 0.3);
        let b_ternary = encode_ternary(&b_f32, 0.3);

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("f32_dot", dim), &dim, |bench, _| {
            bench.iter(|| innr::dot(black_box(&a_f32), black_box(&b_f32)))
        });

        group.bench_with_input(BenchmarkId::new("ternary_dot", dim), &dim, |bench, _| {
            bench.iter(|| ternary_dot(black_box(&a_ternary), black_box(&b_ternary)))
        });
    }

    group.finish();
}

/// Benchmark batch operations
fn bench_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_ternary");

    let dim = 768;
    let num_docs = 1000;

    let query = random_vec(dim);
    let docs: Vec<PackedTernary> = (0..num_docs)
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(i as u64);
            let vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            encode_ternary(&vec, 0.3)
        })
        .collect();

    group.throughput(Throughput::Elements((num_docs * dim) as u64));

    group.bench_function("batch_asymmetric", |b| {
        b.iter(|| {
            let mut scores = Vec::with_capacity(num_docs);
            for doc in black_box(&docs) {
                scores.push(asymmetric_dot(black_box(&query), doc));
            }
            scores
        })
    });

    // For comparison: f32 batch
    let docs_f32: Vec<Vec<f32>> = (0..num_docs)
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(i as u64);
            (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
        })
        .collect();

    group.bench_function("batch_f32", |b| {
        b.iter(|| {
            let mut scores = Vec::with_capacity(num_docs);
            for doc in black_box(&docs_f32) {
                scores.push(innr::dot(black_box(&query), doc));
            }
            scores
        })
    });

    group.finish();
}

/// Report memory compression
fn bench_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory");

    for dim in [384, 768, 1536, 3072] {
        let vec = random_vec(dim);
        let ternary = encode_ternary(&vec, 0.3);

        let f32_bytes = dim * 4;
        let ternary_bytes = ternary.memory_bytes();
        let ratio = f32_bytes as f32 / ternary_bytes as f32;

        println!(
            "dim={}: f32={} bytes, ternary={} bytes, compression={:.1}x",
            dim, f32_bytes, ternary_bytes, ratio
        );

        // Minimal benchmark just to include in report
        group.bench_with_input(BenchmarkId::new("ternary_size", dim), &dim, |bench, _| {
            bench.iter(|| black_box(ternary.memory_bytes()))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_encode,
    bench_ternary_dot,
    bench_asymmetric_dot,
    bench_comparison,
    bench_batch,
    bench_memory,
);
criterion_main!(benches);
