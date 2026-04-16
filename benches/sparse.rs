#![allow(missing_docs)]
//! Benchmarks for sparse vector operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use innr::sparse_dot;
use rand::prelude::*;

fn random_sparse(nnz: usize, vocab_size: u32) -> (Vec<u32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(42);

    // Generate unique sorted indices
    let mut indices: Vec<u32> = (0..vocab_size).collect();
    indices.shuffle(&mut rng);
    indices.truncate(nnz);
    indices.sort_unstable();

    // Generate values
    let values: Vec<f32> = (0..nnz).map(|_| rng.random_range(-1.0..1.0)).collect();

    (indices, values)
}

fn bench_sparse_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_dot");

    let vocab_size = 30000u32; // Typical vocabulary size

    // Different sparsity levels
    for (nnz_a, nnz_b) in [(10, 10), (50, 50), (100, 100), (100, 1000), (500, 500)] {
        let (a_idx, a_val) = random_sparse(nnz_a, vocab_size);
        let (b_idx, b_val) = random_sparse(nnz_b, vocab_size);

        group.throughput(Throughput::Elements((nnz_a + nnz_b) as u64));
        group.bench_with_input(
            BenchmarkId::new("sparse_dot", format!("{}x{}", nnz_a, nnz_b)),
            &(nnz_a, nnz_b),
            |bench, _| {
                bench.iter(|| {
                    sparse_dot(
                        black_box(&a_idx),
                        black_box(&a_val),
                        black_box(&b_idx),
                        black_box(&b_val),
                    )
                })
            },
        );
    }

    group.finish();
}

fn random_sparse_tuples(nnz: usize, vocab_size: u32) -> Vec<(u32, f32)> {
    let mut rng = StdRng::seed_from_u64(99);
    let mut indices: Vec<u32> = (0..vocab_size).collect();
    indices.shuffle(&mut rng);
    indices.truncate(nnz);
    indices.sort_unstable();
    let values: Vec<f32> = (0..nnz).map(|_| rng.random_range(0.0f32..1.0)).collect();
    indices.into_iter().zip(values).collect()
}

fn random_dense(dim: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(77);
    (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect()
}

fn bench_sparse_ext(c: &mut Criterion) {
    use innr::sparse_ext::{sparse_dense_dot, sparse_dot as ext_sparse_dot};

    let vocab_size = 30000u32;
    let dense_dim = 30000usize;
    let dense = random_dense(dense_dim);

    // sparse_ext::sparse_dot at various sparsity levels
    {
        let mut group = c.benchmark_group("sparse_ext_dot");
        for nnz in [10, 50, 120, 500] {
            let a = random_sparse_tuples(nnz, vocab_size);
            let b = random_sparse_tuples(nnz, vocab_size);
            group.throughput(Throughput::Elements((nnz * 2) as u64));
            group.bench_with_input(
                BenchmarkId::new("sparse_dot", nnz),
                &nnz,
                |bench, _| {
                    bench.iter(|| ext_sparse_dot(black_box(&a), black_box(&b)))
                },
            );
        }
        group.finish();
    }

    // sparse_ext::sparse_dense_dot at various sparsity levels
    {
        let mut group = c.benchmark_group("sparse_ext_dense_dot");
        for nnz in [10, 50, 120, 500] {
            let sparse = random_sparse_tuples(nnz, vocab_size);
            group.throughput(Throughput::Elements(nnz as u64));
            group.bench_with_input(
                BenchmarkId::new("sparse_dense_dot", nnz),
                &nnz,
                |bench, _| {
                    bench.iter(|| sparse_dense_dot(black_box(&sparse), black_box(&dense)))
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches, bench_sparse_dot, bench_sparse_ext);
criterion_main!(benches);
