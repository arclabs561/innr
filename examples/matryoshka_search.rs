//! Matryoshka (MRL) Progressive Search
//!
//! Two-stage retrieval: coarse pass at 128d prefix, fine pass at full 768d.
//! MRL embeddings preserve ranking quality in leading dimensions, so a short
//! prefix discards most of the corpus cheaply before full-dimension scoring.
//!
//! ```bash
//! cargo run --example matryoshka_search --release
//! ```

use innr::{cosine, matryoshka_cosine, norm};
use std::time::Instant;

const FULL_DIM: usize = 768;
const PREFIX_DIM: usize = 128;
const CORPUS_SIZE: usize = 10_000;
const COARSE_K: usize = 100;
const FINAL_K: usize = 10;

fn main() {
    println!("Matryoshka Progressive Search");
    println!("=============================\n");
    println!("Corpus: {} vectors, {}d (prefix {}d)", CORPUS_SIZE, FULL_DIM, PREFIX_DIM);
    println!("Pipeline: coarse top-{} at {}d -> fine top-{} at {}d\n", COARSE_K, PREFIX_DIM, FINAL_K, FULL_DIM);

    let corpus: Vec<Vec<f32>> = (0..CORPUS_SIZE)
        .map(|i| normalize(&generate_vec(FULL_DIM, i as u64)))
        .collect();
    let query = normalize(&generate_vec(FULL_DIM, 0xDEAD));

    // Exact brute-force at full dimension (ground truth)
    let t0 = Instant::now();
    let mut exact: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine(&query, v)))
        .collect();
    exact.sort_by(|a, b| b.1.total_cmp(&a.1));
    let exact_time = t0.elapsed();

    let exact_top_k: Vec<usize> = exact.iter().take(FINAL_K).map(|(i, _)| *i).collect();

    // Stage 1: coarse pass at PREFIX_DIM
    let t1 = Instant::now();
    let mut coarse: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, matryoshka_cosine(&query, v, PREFIX_DIM)))
        .collect();
    coarse.sort_by(|a, b| b.1.total_cmp(&a.1));
    coarse.truncate(COARSE_K);
    let coarse_time = t1.elapsed();

    let coarse_indices: Vec<usize> = coarse.iter().map(|(i, _)| *i).collect();

    // Stage 2: fine pass on coarse candidates at FULL_DIM
    let t2 = Instant::now();
    let mut fine: Vec<(usize, f32)> = coarse_indices
        .iter()
        .map(|&i| (i, cosine(&query, &corpus[i])))
        .collect();
    fine.sort_by(|a, b| b.1.total_cmp(&a.1));
    fine.truncate(FINAL_K);
    let fine_time = t2.elapsed();

    let fine_top_k: Vec<usize> = fine.iter().take(FINAL_K).map(|(i, _)| *i).collect();

    let coarse_recall = recall(&coarse_indices, &exact_top_k);
    let final_recall = recall(&fine_top_k, &exact_top_k);

    let speedup = exact_time.as_nanos() as f64 / (coarse_time + fine_time).as_nanos().max(1) as f64;

    println!("Timing");
    println!("------");
    println!("  Exact brute-force ({}d):          {:?}", FULL_DIM, exact_time);
    println!("  Coarse pass ({}d):                {:?}", PREFIX_DIM, coarse_time);
    println!("  Fine pass ({}d, {} candidates):  {:?}", FULL_DIM, COARSE_K, fine_time);
    println!("  Two-stage total:                  {:?}", coarse_time + fine_time);
    println!("  Speedup vs brute-force:           {:.2}x\n", speedup);

    println!("Recall");
    println!("------");
    println!("  Coarse recall@{} (top-{} at {}d): {:.1}%", FINAL_K, COARSE_K, PREFIX_DIM, coarse_recall * 100.0);
    println!("  Final recall@{}:                  {:.1}%\n", FINAL_K, final_recall * 100.0);

    println!("Top-{} (exact vs two-stage)", FINAL_K);
    println!("--------------------------");
    for rank in 0..FINAL_K {
        let (ei, es) = exact[rank];
        let (fi, fs) = fine[rank];
        let tag = if ei == fi { " " } else { "*" };
        println!("  #{:>2} exact: idx={:>5} sim={:.6}  | two-stage: idx={:>5} sim={:.6} {}", rank + 1, ei, es, fi, fs, tag);
    }
    println!("\n(* = rank differs from exact)");
}

/// Fraction of `ground_truth` items present in `retrieved`.
fn recall(retrieved: &[usize], ground_truth: &[usize]) -> f64 {
    let hits = ground_truth
        .iter()
        .filter(|gt| retrieved.contains(gt))
        .count();
    hits as f64 / ground_truth.len() as f64
}

/// Deterministic pseudo-random vector (xorshift-style, no deps).
fn generate_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed ^ 0x517cc1b727220a95;
    (0..dim)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// L2-normalize a vector.
fn normalize(v: &[f32]) -> Vec<f32> {
    let n = norm(v);
    if n < 1e-9 {
        return v.to_vec();
    }
    v.iter().map(|x| x / n).collect()
}
