//! Benchmarks for MaxSim late interaction scoring.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use innr::{maxsim, maxsim_cosine};
use rand::prelude::*;

fn random_tokens(num_tokens: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..num_tokens)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect())
        .collect()
}

fn bench_maxsim(c: &mut Criterion) {
    let mut group = c.benchmark_group("maxsim");

    let dim = 128; // ColBERT dimension

    // Different query/doc lengths
    for (q_len, d_len) in [(32, 128), (32, 256), (32, 512), (64, 256), (64, 512)] {
        let query_vecs = random_tokens(q_len, dim);
        let doc_vecs = random_tokens(d_len, dim);

        let query_refs: Vec<&[f32]> = query_vecs.iter().map(|v| v.as_slice()).collect();
        let doc_refs: Vec<&[f32]> = doc_vecs.iter().map(|v| v.as_slice()).collect();

        group.throughput(Throughput::Elements((q_len * d_len) as u64));
        group.bench_with_input(
            BenchmarkId::new("maxsim_dot", format!("{}x{}", q_len, d_len)),
            &(q_len, d_len),
            |bench, _| bench.iter(|| maxsim(black_box(&query_refs), black_box(&doc_refs))),
        );
    }

    group.finish();
}

fn bench_maxsim_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("maxsim_cosine");

    let dim = 128;

    for (q_len, d_len) in [(32, 128), (32, 256)] {
        let query_vecs = random_tokens(q_len, dim);
        let doc_vecs = random_tokens(d_len, dim);

        let query_refs: Vec<&[f32]> = query_vecs.iter().map(|v| v.as_slice()).collect();
        let doc_refs: Vec<&[f32]> = doc_vecs.iter().map(|v| v.as_slice()).collect();

        group.throughput(Throughput::Elements((q_len * d_len) as u64));
        group.bench_with_input(
            BenchmarkId::new("maxsim_cosine", format!("{}x{}", q_len, d_len)),
            &(q_len, d_len),
            |bench, _| bench.iter(|| maxsim_cosine(black_box(&query_refs), black_box(&doc_refs))),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_maxsim, bench_maxsim_cosine);
criterion_main!(benches);
