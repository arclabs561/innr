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

criterion_group!(benches, bench_sparse_dot);
criterion_main!(benches);
