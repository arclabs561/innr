#![allow(missing_docs)]
//! Benchmarks for dense vector operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use innr::{
    binary::encode_binary, cosine, dot, dot_u8, hamming_distance, l1_distance, l2_distance, norm,
    TopK,
};
use rand::prelude::*;

fn random_vec(n: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n).map(|_| rng.random_range(-1.0..1.0)).collect()
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

fn bench_l1_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("l1_distance");

    for dim in [128, 384, 768, 1536] {
        let a = random_vec(dim);
        let b = random_vec(dim);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::new("l1", dim), &dim, |bench, _| {
            bench.iter(|| l1_distance(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

fn random_u8_vec(n: usize) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n).map(|_| rng.random::<u8>()).collect()
}

fn bench_hamming_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("hamming_distance");

    for len in [128usize, 768] {
        let a = random_u8_vec(len);
        let b = random_u8_vec(len);

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("hamming", len), &len, |bench, _| {
            bench.iter(|| hamming_distance(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

fn bench_dot_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_u8");

    for len in [128usize, 768] {
        let a = random_u8_vec(len);
        let b = random_u8_vec(len);

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("dot_u8", len), &len, |bench, _| {
            bench.iter(|| dot_u8(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

fn bench_binary(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary");

    for len in [128usize, 768] {
        let fa: Vec<f32> = random_vec(len);
        let fb: Vec<f32> = random_vec(len);
        let a = encode_binary(&fa, 0.0);
        let b = encode_binary(&fb, 0.0);

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("binary_dot", len), &len, |bench, _| {
            bench.iter(|| innr::binary_dot(black_box(&a), black_box(&b)))
        });
        group.bench_with_input(BenchmarkId::new("binary_hamming", len), &len, |bench, _| {
            bench.iter(|| innr::binary_hamming(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

fn bench_topk(c: &mut Criterion) {
    let mut group = c.benchmark_group("topk_insert");

    let mut rng = StdRng::seed_from_u64(42);
    let items: Vec<f32> = (0..10_000).map(|_| rng.random::<f32>()).collect();

    for k in [10usize, 100] {
        group.throughput(Throughput::Elements(items.len() as u64));
        group.bench_with_input(BenchmarkId::new("insert_10k", k), &k, |bench, &k| {
            bench.iter(|| {
                let mut topk = TopK::new(k);
                for (i, &dist) in items.iter().enumerate() {
                    topk.insert(black_box(i as u32), black_box(dist));
                }
                black_box(topk.len())
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_dot,
    bench_cosine,
    bench_norm,
    bench_l2_distance,
    bench_l1_distance,
    bench_hamming_distance,
    bench_dot_u8,
    bench_binary,
    bench_topk,
);
criterion_main!(benches);
