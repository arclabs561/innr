#![allow(missing_docs)]
//! Query-vs-corpus scans over horizontal (row-major) storage.
//!
//! Consumer shapes (vicinity 798f4d0), innr against vicinity's no-innr
//! fallback:
//! - `scan_vicinity_brute_force`: one row of the brute-force kNN graph build,
//!   src/fresh_graph/mod.rs:887-896 (distance to every other vector, full
//!   sort, keep `max_degree`).
//! - `scan_vicinity_centroids`: IVF coarse search, src/ivf_pq/search.rs:1577-1591
//!   (distance to every centroid, partial select of `nprobe`).
//!
//! Baselines for the planned packed-corpus API (current innr only):
//! - `topk_exact`: full scan plus `innr::TopK`, the baseline a pruned
//!   dimension-major scan must beat.
//! - `topk_binary_rescore`: Hamming scan over sign codes, keep
//!   `oversample * k` candidates, rescore them with f32 dot.
//! - `topk_multi_query`: Q queries against one corpus, looped per query.

use std::fmt::Display;
use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion};
use innr::batch::{batch_dot, VerticalBatch};
use innr::corpus::Corpus;
use innr::metric::Dot;
use innr::TopK;
use innr_bench_support::data;

const INNR: &str = if cfg!(feature = "simd") {
    "innr_simd"
} else {
    "innr_portable"
};
const OTHERS: bool = cfg!(feature = "simd");

/// Neighbors kept per row in the kNN graph build (a typical `max_degree`).
const MAX_DEGREE: usize = 32;
/// Centroids probed per IVF query.
const NPROBE: usize = 16;
/// Results per top-k query.
const K: usize = 10;

fn run<O>(
    g: &mut BenchmarkGroup<'_, WallTime>,
    variant: &str,
    param: impl Display,
    mut f: impl FnMut() -> O,
) {
    g.bench_function(BenchmarkId::new(variant, param), |b| b.iter(&mut f));
}

// --- vicinity src/distance.rs:104-109 (`cosine_distance_normalized`), with
// `simd::dot` as innr (default `innr` feature) or the fallback
// (src/simd.rs:89-91). ---

#[inline]
fn cosine_distance_normalized_innr(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    1.0 - innr::dot(a, b)
}

#[inline]
fn vicinity_fallback_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn cosine_distance_normalized_local(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    1.0 - vicinity_fallback_dot(a, b)
}

/// Body of src/fresh_graph/mod.rs:888-895 for one row `i`, with the metric
/// passed in. `get_vector(j)` is a row slice of the flat store.
fn brute_force_row(
    data: &[f32],
    dim: usize,
    n: usize,
    i: usize,
    k: usize,
    dist: impl Fn(&[f32], &[f32]) -> f32,
) -> Vec<u32> {
    let get_vector = |j: usize| &data[j * dim..(j + 1) * dim];
    let vi = get_vector(i);
    let mut dists: Vec<(u32, f32)> = (0..n)
        .filter(|&j| j != i)
        .map(|j| (j as u32, dist(vi, get_vector(j))))
        .collect();
    dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    dists.truncate(k);
    dists.iter().map(|(id, _)| *id).collect()
}

/// The same row through innr's current batch API (dimension-major
/// `VerticalBatch`), for comparison with the per-row loop.
fn brute_force_row_vertical(batch: &VerticalBatch, query: &[f32], i: usize, k: usize) -> Vec<u32> {
    let dots = batch_dot(query, batch);
    let mut dists: Vec<(u32, f32)> = dots
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != i)
        .map(|(j, d)| (j as u32, 1.0 - d))
        .collect();
    dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    dists.truncate(k);
    dists.iter().map(|(id, _)| *id).collect()
}

/// src/ivf_pq/search.rs:1577-1591 (`find_nearest_centroids`), metric passed in.
fn find_nearest_centroids(
    centroids: &[f32],
    dimension: usize,
    query: &[f32],
    nprobe: usize,
    dist: impl Fn(&[f32], &[f32]) -> f32,
) -> Vec<(usize, f32)> {
    let get_centroid = |idx: usize| &centroids[idx * dimension..(idx + 1) * dimension];
    let num_centroids = centroids.len() / dimension;
    let mut dists: Vec<(usize, f32)> = (0..num_centroids)
        .map(|idx| {
            let c = get_centroid(idx);
            (idx, dist(query, c))
        })
        .collect();
    let nprobe = nprobe.min(dists.len());
    if nprobe < dists.len() {
        dists.select_nth_unstable_by(nprobe, |a, b| a.1.total_cmp(&b.1));
        dists.truncate(nprobe);
    }
    dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    dists
}

/// Exact top-k by full scan over horizontal storage.
fn topk_exact(corpus: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut top = TopK::new(k);
    for (i, row) in corpus.chunks_exact(dim).enumerate() {
        top.insert(i as u32, 1.0 - innr::dot(query, row));
    }
    top.into_sorted()
}

/// Binary candidate scan, then f32 rescoring of `oversample * k` candidates.
fn topk_binary_rescore(
    corpus: &[f32],
    codes: &[u8],
    dim: usize,
    query: &[f32],
    query_code: &[u8],
    k: usize,
    oversample: usize,
) -> Vec<(u32, f32)> {
    let bpr = dim.div_ceil(8);
    let mut ham: Vec<(u32, u32)> = codes
        .chunks_exact(bpr)
        .enumerate()
        .map(|(i, c)| (innr::hamming_distance(query_code, c), i as u32))
        .collect();
    let m = (oversample * k).min(ham.len());
    ham.select_nth_unstable(m - 1);
    let mut top = TopK::new(k);
    for &(_, id) in &ham[..m] {
        let row = &corpus[id as usize * dim..(id as usize + 1) * dim];
        top.insert(id, 1.0 - innr::dot(query, row));
    }
    top.into_sorted()
}

fn vicinity(c: &mut Criterion) {
    let dim = 128;
    let corpus = data::unit_rows(100_000, dim, 1);
    let query_row = 0usize;

    let mut g = c.benchmark_group("scan_vicinity_brute_force");
    g.sample_size(10);
    for n in [1_000usize, 10_000, 100_000] {
        let store = &corpus[..n * dim];
        let p = format!("{n}x{dim}");
        run(&mut g, INNR, &p, || {
            brute_force_row(
                black_box(store),
                dim,
                n,
                query_row,
                MAX_DEGREE,
                cosine_distance_normalized_innr,
            )
        });
        let packed = Corpus::from_flat(store, dim).unwrap();
        let q = &store[..dim];
        run(&mut g, &format!("{INNR}_corpus"), &p, || {
            // Dot ranks unit rows like the 1 - dot distance; skip the row itself.
            packed
                .top_k::<Dot>(black_box(q), MAX_DEGREE + 1)
                .into_iter()
                .filter(|h| h.index != query_row)
                .take(MAX_DEGREE)
                .map(|h| h.index as u32)
                .collect::<Vec<_>>()
        });
        let batch = VerticalBatch::from_flat(store, n, dim);
        run(&mut g, &format!("{INNR}_vertical_batch"), &p, || {
            brute_force_row_vertical(black_box(&batch), black_box(q), query_row, MAX_DEGREE)
        });
        if OTHERS {
            run(&mut g, "local_vicinity", &p, || {
                brute_force_row(
                    black_box(store),
                    dim,
                    n,
                    query_row,
                    MAX_DEGREE,
                    cosine_distance_normalized_local,
                )
            });
        }
    }
    g.finish();

    let query = data::unit_vec(dim, 7);
    let mut g = c.benchmark_group("scan_vicinity_centroids");
    g.sample_size(10);
    for n in [1_000usize, 10_000, 100_000] {
        let centroids = &corpus[..n * dim];
        let p = format!("{n}x{dim}");
        run(&mut g, INNR, &p, || {
            find_nearest_centroids(
                black_box(centroids),
                dim,
                black_box(&query),
                NPROBE,
                cosine_distance_normalized_innr,
            )
        });
        if OTHERS {
            run(&mut g, "local_vicinity", &p, || {
                find_nearest_centroids(
                    black_box(centroids),
                    dim,
                    black_box(&query),
                    NPROBE,
                    cosine_distance_normalized_local,
                )
            });
        }
    }
    g.finish();
}

fn topk_baselines(c: &mut Criterion) {
    for dim in [128usize, 768, 1536] {
        // One 100k corpus per dimension; N = 10k uses its prefix.
        let corpus = data::unit_rows(100_000, dim, 11);
        let query = data::unit_vec(dim, 12);

        let mut g = c.benchmark_group("topk_exact");
        g.sample_size(10);
        for n in [10_000usize, 100_000] {
            let store = &corpus[..n * dim];
            run(&mut g, INNR, format!("{n}x{dim}"), || {
                topk_exact(black_box(store), dim, black_box(&query), K)
            });
            let packed = Corpus::from_flat(store, dim).unwrap();
            run(
                &mut g,
                &format!("{INNR}_corpus"),
                format!("{n}x{dim}"),
                || packed.top_k::<Dot>(black_box(&query), K),
            );
        }
        g.finish();

        let codes = data::sign_codes(&corpus, dim);
        let query_code = data::sign_codes(&query, dim);
        let mut g = c.benchmark_group("topk_binary_rescore");
        g.sample_size(10);
        for oversample in [4usize, 10] {
            run(
                &mut g,
                &format!("{INNR}_os{oversample}"),
                format!("100000x{dim}"),
                || {
                    topk_binary_rescore(
                        black_box(&corpus),
                        black_box(&codes),
                        dim,
                        black_box(&query),
                        black_box(&query_code),
                        K,
                        oversample,
                    )
                },
            );
        }
        // The Hamming stage alone, to split the two costs.
        run(
            &mut g,
            &format!("{INNR}_hamming_only"),
            format!("100000x{dim}"),
            || {
                let bpr = dim.div_ceil(8);
                black_box(&codes)
                    .chunks_exact(bpr)
                    .map(|c| innr::hamming_distance(&query_code, c))
                    .collect::<Vec<u32>>()
            },
        );
        g.finish();

        if dim == 768 {
            let n = 10_000;
            let store = &corpus[..n * dim];
            let queries = data::unit_rows(64, dim, 13);
            let mut g = c.benchmark_group("topk_multi_query");
            g.sample_size(10);
            for q in [1usize, 8, 64] {
                let qs = &queries[..q * dim];
                run(&mut g, INNR, format!("Q{q}_{n}x{dim}"), || {
                    qs.chunks_exact(dim)
                        .map(|query| topk_exact(black_box(store), dim, query, K))
                        .collect::<Vec<_>>()
                });
                let packed = Corpus::from_flat(store, dim).unwrap();
                run(
                    &mut g,
                    &format!("{INNR}_corpus"),
                    format!("Q{q}_{n}x{dim}"),
                    || packed.top_k_many::<Dot, _>(black_box(qs).chunks_exact(dim), K),
                );
            }
            g.finish();
        }
    }
}

criterion_group! {
    name = benches;
    config = innr_bench_support::config();
    targets = vicinity, topk_baselines
}
criterion_main!(benches);
