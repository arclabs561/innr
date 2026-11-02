#![allow(missing_docs)]
//! ColBERT MaxSim at rankops' shape: 32 query tokens, 128-512 document
//! tokens, dimension 128.
//!
//! rankops (a979f3c) scores MaxSim two ways, both measured here:
//! - token slices: `rerank::simd::maxsim` (re-export of `innr::maxsim` with
//!   the default `rerank` feature) and `maxsim_batch` over a corpus;
//! - flat row-major buffers: `rerank::colbert::maxsim_flat`
//!   (src/rerank/colbert.rs:774-790), which calls `simd::dot` per token pair.
//!
//! `local_rankops*` are the same functions built without `rerank`, where
//! `dot` is rankops' own NEON/AVX2/AVX-512 kernel (copied verbatim into
//! `innr-bench-support` because innr forbids unsafe code in its targets).

use std::fmt::Display;
use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion};
use innr_bench_support::{data, rankops};

const INNR: &str = if cfg!(feature = "simd") {
    "innr_simd"
} else {
    "innr_portable"
};
const OTHERS: bool = cfg!(feature = "simd");

const DIM: usize = 128;
const QUERY_TOKENS: usize = 32;

fn run<O>(
    g: &mut BenchmarkGroup<'_, WallTime>,
    variant: &str,
    param: impl Display,
    mut f: impl FnMut() -> O,
) {
    g.bench_function(BenchmarkId::new(variant, param), |b| b.iter(&mut f));
}

/// rankops src/rerank/colbert.rs:774-790 with `simd::dot` = `innr::dot`.
fn maxsim_flat_innr(query: &[f32], doc: &[f32], dim: usize) -> f32 {
    if query.is_empty() || doc.is_empty() {
        return 0.0;
    }
    query
        .chunks_exact(dim)
        .map(|q| {
            doc.chunks_exact(dim)
                .map(|d| innr::dot(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

/// rankops src/rerank/simd.rs:544-552 (`maxsim_batch`) with `maxsim` =
/// `innr::maxsim`.
fn maxsim_batch_innr(query: &[Vec<f32>], docs: &[Vec<Vec<f32>>]) -> Vec<f32> {
    let q = rankops::as_slices(query);
    docs.iter()
        .map(|doc| {
            let d = rankops::as_slices(doc);
            innr::maxsim(&q, &d)
        })
        .collect()
}

fn per_doc(c: &mut Criterion) {
    let query = data::tokens(QUERY_TOKENS, DIM, 1);
    let query_flat: Vec<f32> = query.concat();
    let q = rankops::as_slices(&query);

    let mut g = c.benchmark_group("maxsim_per_doc");
    for doc_tokens in [128usize, 256, 512] {
        let doc = data::tokens(doc_tokens, DIM, 2);
        let doc_flat: Vec<f32> = doc.concat();
        let d = rankops::as_slices(&doc);
        let p = format!("{QUERY_TOKENS}x{doc_tokens}x{DIM}");
        run(&mut g, INNR, &p, || {
            innr::maxsim(black_box(&q), black_box(&d))
        });
        run(&mut g, &format!("{INNR}_flat"), &p, || {
            maxsim_flat_innr(black_box(&query_flat), black_box(&doc_flat), DIM)
        });
        if OTHERS {
            run(&mut g, "local_rankops", &p, || {
                rankops::maxsim(black_box(&q), black_box(&d))
            });
            run(&mut g, "local_rankops_flat", &p, || {
                rankops::maxsim_flat(black_box(&query_flat), black_box(&doc_flat), DIM)
            });
        }
    }
    g.finish();
}

fn per_corpus(c: &mut Criterion) {
    // Rerank 1000 candidates of 128 tokens each: the batch and flat paths
    // rankops exposes for a reranking stage.
    let n_docs = 1000;
    let doc_tokens = 128;
    let query = data::tokens(QUERY_TOKENS, DIM, 1);
    let query_flat: Vec<f32> = query.concat();
    let docs: Vec<Vec<Vec<f32>>> = (0..n_docs)
        .map(|i| data::tokens(doc_tokens, DIM, 100 + i as u64))
        .collect();
    let docs_flat: Vec<Vec<f32>> = docs.iter().map(|d| d.concat()).collect();
    let p = format!("{n_docs}docs_{QUERY_TOKENS}x{doc_tokens}x{DIM}");

    let mut g = c.benchmark_group("maxsim_per_corpus");
    g.sample_size(10);
    run(&mut g, INNR, &p, || {
        maxsim_batch_innr(black_box(&query), black_box(&docs))
    });
    run(&mut g, &format!("{INNR}_flat"), &p, || {
        docs_flat
            .iter()
            .map(|d| maxsim_flat_innr(black_box(&query_flat), d, DIM))
            .collect::<Vec<f32>>()
    });
    if OTHERS {
        run(&mut g, "local_rankops", &p, || {
            rankops::maxsim_batch(black_box(&query), black_box(&docs))
        });
        run(&mut g, "local_rankops_flat", &p, || {
            docs_flat
                .iter()
                .map(|d| rankops::maxsim_flat(black_box(&query_flat), d, DIM))
                .collect::<Vec<f32>>()
        });
    }
    g.finish();
}

criterion_group! {
    name = benches;
    config = innr_bench_support::config();
    targets = per_doc, per_corpus
}
criterion_main!(benches);
