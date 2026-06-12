#![allow(missing_docs)]
//! Benchmarks for the SIMD f64 reductions and the u64-slot Hamming kernel.
//!
//! Each compares the dispatched SIMD path against the portable scalar loop so
//! the speedup is visible on the same machine. Numbers are cache-resident,
//! single-core; report the host CPU alongside them.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use innr::dense_f64::{dot_f64, l1_distance_f64, l2_distance_squared_f64};
use innr::slot_hamming_u64;
use rand::prelude::*;

fn vec_f64(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n).map(|_| rng.random_range(-1.0..1.0)).collect()
}

fn slots_u64(n: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(7);
    (0..n).map(|_| rng.random()).collect()
}

fn bench_f64_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64_reductions");
    for dim in [64, 256, 768, 1536, 4096] {
        let a = vec_f64(dim);
        let b = vec_f64(dim);
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::new("dot_f64", dim), &dim, |bn, _| {
            bn.iter(|| dot_f64(black_box(&a), black_box(&b)))
        });
        group.bench_with_input(BenchmarkId::new("l2sq_f64", dim), &dim, |bn, _| {
            bn.iter(|| l2_distance_squared_f64(black_box(&a), black_box(&b)))
        });
        group.bench_with_input(BenchmarkId::new("l1_f64", dim), &dim, |bn, _| {
            bn.iter(|| l1_distance_f64(black_box(&a), black_box(&b)))
        });
    }
    group.finish();
}

fn bench_slot_hamming_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("slot_hamming_u64");
    // Sketch sizes spanning the MinHash range (BinDash uses thousands).
    for slots in [128, 1024, 4096, 12288] {
        let a = slots_u64(slots);
        let b = slots_u64(slots);
        group.throughput(Throughput::Elements(slots as u64));
        group.bench_with_input(BenchmarkId::new("u64", slots), &slots, |bn, _| {
            bn.iter(|| slot_hamming_u64(black_box(&a), black_box(&b)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_f64_reductions, bench_slot_hamming_u64);
criterion_main!(benches);
