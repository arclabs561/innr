#![allow(missing_docs)]
//! Pair kernels at the dimensions and lengths innr's consumers call them with.
//!
//! Each group compares innr (this build's path: `innr_simd` with the default
//! `simd` feature, `innr_portable` with `--no-default-features`) against the
//! consumer's own no-innr fallback, copied verbatim with its source location.
//! Fallback variants run only in the `simd` build so the two-build suite
//! measures them once; the NumKong reference column lives in the separate
//! `benches/numkong` crate. `benches/report.sh` runs both builds.
//!
//! Sources are the consumers' published repositories at these commits:
//! vicinity 798f4d0, clump f38dc53, rankops a979f3c, qntz 23aba39,
//! sketchir a56239e, statskit 5d35dd5, graphops 61b4e82, hopfield 4c1cece.

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
/// Fallbacks and references are build-independent; measure them once.
const OTHERS: bool = cfg!(feature = "simd");

/// f32 dimensions from the consumer survey: clump d16-d128, qntz and vicinity
/// 128/768, common embedding sizes 384/1024/1536.
const F32_DIMS: [usize; 8] = [16, 32, 64, 128, 384, 768, 1024, 1536];

fn run<O>(
    g: &mut BenchmarkGroup<'_, WallTime>,
    variant: &str,
    param: impl Display,
    mut f: impl FnMut() -> O,
) {
    g.bench_function(BenchmarkId::new(variant, param), |b| b.iter(&mut f));
}

// ---------------------------------------------------------------------------
// Consumer fallbacks, verbatim.
// ---------------------------------------------------------------------------

/// vicinity src/simd.rs:89-91 (`fallback::dot`).
#[inline]
fn vicinity_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// vicinity src/simd.rs:96-98 (`fallback::norm`).
#[inline]
fn vicinity_norm(v: &[f32]) -> f32 {
    vicinity_dot(v, v).sqrt()
}

/// vicinity src/simd.rs:103-113 (`fallback::cosine`).
#[inline]
fn vicinity_cosine(a: &[f32], b: &[f32]) -> f32 {
    const NORM_EPSILON: f32 = 1e-9;
    let d = vicinity_dot(a, b);
    let na = vicinity_norm(a);
    let nb = vicinity_norm(b);
    if na > NORM_EPSILON && nb > NORM_EPSILON {
        d / (na * nb)
    } else {
        0.0
    }
}

/// vicinity src/simd.rs:124-126 (`fallback::l2_distance_squared`).
#[inline]
fn vicinity_l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// clump src/cluster/distance.rs:143-157 (`CosineDistance::distance`
/// scalar branch, taken for d < 16 or without `simd`).
#[inline]
fn clump_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = (norm_a * norm_b).sqrt();
    // Use 1e-9 threshold to match innr SIMD path behavior.
    if denom < 1e-9 {
        return 0.0;
    }
    1.0 - (dot / denom)
}

/// clump src/cluster/distance.rs:72-84 (`SquaredEuclidean::distance`;
/// clump never routes L2 through innr).
#[inline]
fn clump_l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// clump's `CosineDistance` with `simd` (src/cluster/distance.rs:132-142):
/// three `innr::dot` calls rather than `innr::cosine`.
#[inline]
fn clump_cosine_distance_innr(a: &[f32], b: &[f32]) -> f32 {
    let dot = innr::dot(a, b);
    let norm_a = innr::dot(a, a);
    let norm_b = innr::dot(b, b);
    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-9 {
        return 0.0;
    }
    1.0 - dot / denom
}

/// graphops src/pagerank.rs:196-200 (L1 residual without `simd`).
#[inline]
fn graphops_l1(scores: &[f64], new_scores: &[f64]) -> f64 {
    scores
        .iter()
        .zip(new_scores.iter())
        .map(|(old, new)| (old - new).abs())
        .sum()
}

/// statskit src/regression.rs:12-16 (MSE numerator without `simd`); hopfield
/// src/lib.rs:64-73 (`l2_sq`) has the identical body.
#[inline]
fn statskit_l2_squared(y_true: &[f64], y_pred: &[f64]) -> f64 {
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).powi(2))
        .sum()
}

/// qntz src/simd_ops.rs:155-190 (`hamming_distance_portable`).
#[inline]
fn qntz_hamming(a: &[u8], b: &[u8]) -> u32 {
    let mut dist = 0u32;
    let len = a.len().min(b.len());

    let chunks = len / 8;
    for i in 0..chunks {
        let base = i * 8;
        let a_u64 = u64::from_le_bytes([
            a[base],
            a[base + 1],
            a[base + 2],
            a[base + 3],
            a[base + 4],
            a[base + 5],
            a[base + 6],
            a[base + 7],
        ]);
        let b_u64 = u64::from_le_bytes([
            b[base],
            b[base + 1],
            b[base + 2],
            b[base + 3],
            b[base + 4],
            b[base + 5],
            b[base + 6],
            b[base + 7],
        ]);
        dist += (a_u64 ^ b_u64).count_ones();
    }

    for i in (chunks * 8)..len {
        dist += (a[i] ^ b[i]).count_ones();
    }

    dist
}

/// sketchir src/hyperplane.rs:118-125 (`HyperplaneHasher::distance`
/// without `innr`).
#[inline]
fn sketchir_hamming(sig_a: &[u8], sig_b: &[u8]) -> u32 {
    sig_a
        .iter()
        .zip(sig_b.iter())
        .map(|(a, b)| (a ^ b).count_ones())
        .sum()
}

/// sketchir src/minhash.rs:108-113 (differing MinHash slots without `innr`).
#[inline]
fn sketchir_slot_hamming(values: &[u64], other: &[u64]) -> usize {
    values
        .iter()
        .zip(other.iter())
        .filter(|(a, b)| a != b)
        .count()
}

/// qntz src/adaptive.rs:527-540 (`mixed_dot_f32_u8` without `simd`).
#[inline]
fn qntz_mixed_dot(query: &[f32], codes: &[u8]) -> f32 {
    query
        .iter()
        .zip(codes.iter())
        .map(|(&q, &c)| q * c as f32)
        .sum()
}

// ---------------------------------------------------------------------------
// Benches.
// ---------------------------------------------------------------------------

fn pair_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("pair_f32_dot");
    for d in F32_DIMS {
        let (a, b) = (data::f32s(d, 1), data::f32s(d, 2));
        run(&mut g, INNR, d, || innr::dot(black_box(&a), black_box(&b)));
        if OTHERS {
            run(&mut g, "local_vicinity", d, || {
                vicinity_dot(black_box(&a), black_box(&b))
            });
            run(&mut g, "local_rankops", d, || {
                rankops::dot(black_box(&a), black_box(&b))
            });
        }
    }
    g.finish();

    let mut g = c.benchmark_group("pair_f32_cosine");
    for d in F32_DIMS {
        let (a, b) = (data::f32s(d, 1), data::f32s(d, 2));
        run(&mut g, INNR, d, || {
            innr::cosine(black_box(&a), black_box(&b))
        });
        // How clump calls innr today: three dots, not innr::cosine.
        run(&mut g, "innr_3dot_clump", d, || {
            clump_cosine_distance_innr(black_box(&a), black_box(&b))
        });
        if OTHERS {
            run(&mut g, "local_vicinity", d, || {
                vicinity_cosine(black_box(&a), black_box(&b))
            });
            run(&mut g, "local_clump", d, || {
                clump_cosine_distance(black_box(&a), black_box(&b))
            });
            run(&mut g, "local_rankops", d, || {
                rankops::cosine(black_box(&a), black_box(&b))
            });
        }
    }
    g.finish();

    let mut g = c.benchmark_group("pair_f32_l2sq");
    for d in F32_DIMS {
        let (a, b) = (data::f32s(d, 1), data::f32s(d, 2));
        run(&mut g, INNR, d, || {
            innr::l2_distance_squared(black_box(&a), black_box(&b))
        });
        if OTHERS {
            run(&mut g, "local_vicinity", d, || {
                vicinity_l2_squared(black_box(&a), black_box(&b))
            });
            run(&mut g, "local_clump", d, || {
                clump_l2_squared(black_box(&a), black_box(&b))
            });
        }
    }
    g.finish();
}

fn pair_f64(c: &mut Criterion) {
    // graphops: one L1 residual per power iteration over all n nodes
    // (graphops' own bench uses n = 10_000).
    let mut g = c.benchmark_group("pair_f64_l1");
    for n in [1_000usize, 10_000, 100_000] {
        let (a, b) = (data::f64s(n, 1), data::f64s(n, 2));
        run(&mut g, INNR, n, || {
            innr::dense_f64::l1_distance_f64(black_box(&a), black_box(&b))
        });
        if OTHERS {
            run(&mut g, "local_graphops", n, || {
                graphops_l1(black_box(&a), black_box(&b))
            });
        }
    }
    g.finish();

    // hopfield: one squared L2 per stored pattern at the pattern dimension
    // (no fixed dimension in the crate; 16 and 784 bracket small and
    // MNIST-sized patterns). statskit: one per metric call over n samples.
    let mut g = c.benchmark_group("pair_f64_l2sq");
    for n in [16usize, 784, 1_000, 100_000] {
        let (a, b) = (data::f64s(n, 1), data::f64s(n, 2));
        run(&mut g, INNR, n, || {
            innr::dense_f64::l2_distance_squared_f64(black_box(&a), black_box(&b))
        });
        if OTHERS {
            run(&mut g, "local_statskit", n, || {
                statskit_l2_squared(black_box(&a), black_box(&b))
            });
        }
    }
    g.finish();
}

fn hamming(c: &mut Criterion) {
    // Bytes per code: sketchir hyperplane signatures of 64 and 256 bits
    // (8, 32 B); qntz binary codes at d = 128 and 768 (16, 96 B).
    let mut g = c.benchmark_group("hamming_bits");
    for n in [8usize, 16, 32, 96] {
        let (a, b) = (data::bytes(n, 1), data::bytes(n, 2));
        run(&mut g, INNR, n, || {
            innr::hamming_distance(black_box(&a), black_box(&b))
        });
        if OTHERS {
            run(&mut g, "local_qntz", n, || {
                qntz_hamming(black_box(&a), black_box(&b))
            });
            run(&mut g, "local_sketchir", n, || {
                sketchir_hamming(black_box(&a), black_box(&b))
            });
        }
    }
    g.finish();

    // sketchir MinHash signatures: 128 and 256 u64 slots.
    let mut g = c.benchmark_group("slot_hamming_u64");
    for n in [128usize, 256] {
        let (a, b) = (data::slots_u64(n, 1), data::slots_u64(n, 2));
        run(&mut g, INNR, n, || {
            innr::slot_hamming_u64(black_box(&a), black_box(&b))
        });
        if OTHERS {
            run(&mut g, "local_sketchir", n, || {
                sketchir_slot_hamming(black_box(&a), black_box(&b))
            });
        }
    }
    g.finish();

    // qntz adaptive scalar quantizer: f32 query against u8 codes, d = 128, 768.
    let mut g = c.benchmark_group("mixed_dot_u8_f32");
    for d in [128usize, 768] {
        let (q, codes) = (data::f32s(d, 1), data::bytes(d, 2));
        run(&mut g, INNR, d, || {
            innr::scalar::mixed_dot_u8_f32(black_box(&q), black_box(&codes))
        });
        if OTHERS {
            run(&mut g, "local_qntz", d, || {
                qntz_mixed_dot(black_box(&q), black_box(&codes))
            });
        }
    }
    g.finish();
}

/// A/A noise floor: the same kernel and inputs under different names.
/// `aa_1` and `aa_2` run back to back at the start of this binary, `aa_3`
/// at its end, so the spread covers both short-term noise and drift across
/// the run (thermal state, other load). report.sh prints it beside the
/// results and uses it as the tie band.
fn aa_copies(c: &mut Criterion, copies: &[&str]) {
    let mut g = c.benchmark_group("aa_noise");
    for d in [128usize, 768] {
        let (a, b) = (data::f32s(d, 1), data::f32s(d, 2));
        for copy in copies {
            run(&mut g, &format!("{INNR}_{copy}"), d, || {
                innr::dot(black_box(&a), black_box(&b))
            });
        }
    }
    g.finish();
}

fn aa_noise_first(c: &mut Criterion) {
    aa_copies(c, &["aa_1", "aa_2"]);
}

fn aa_noise_last(c: &mut Criterion) {
    aa_copies(c, &["aa_3"]);
}

criterion_group! {
    name = benches;
    config = innr_bench_support::config();
    targets = aa_noise_first, pair_f32, pair_f64, hamming, aa_noise_last
}
criterion_main!(benches);
