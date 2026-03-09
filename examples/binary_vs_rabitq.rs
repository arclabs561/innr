//! Comparison: innr binary encoding vs qntz RaBitQ.
//!
//! Both reduce vectors to 1-bit representations for fast distance computation.
//! innr::encode_binary uses a simple sign/threshold test.
//! qntz RaBitQ applies a random rotation before binarizing, preserving more
//! angular information.
//!
//! Measures: how well each method's binary distances correlate with the true
//! cosine distances on the original vectors.
//!
//! Run: cargo run --example binary_vs_rabitq

use innr::binary::{binary_hamming, encode_binary};
use innr::dense::cosine;
use qntz::rabitq::{RaBitQConfig, RaBitQQuantizer};

fn main() {
    let dim = 32;
    let n = 30;
    let seed = 0xDEAD_BEEF;

    // Generate deterministic pseudo-random vectors.
    let vectors = gen_vectors(n, dim, seed);

    // --- innr: threshold binarization ---
    let binary_vecs: Vec<_> = vectors.iter().map(|v| encode_binary(v, 0.0)).collect();

    // --- qntz: RaBitQ 1-bit ---
    let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
    let config = RaBitQConfig {
        total_bits: 1,
        t_const: None,
    };
    let mut quantizer = RaBitQQuantizer::with_config(dim, seed, config).unwrap();
    quantizer.fit(&flat, n).unwrap();
    let rabitq_codes: Vec<_> = vectors
        .iter()
        .map(|v| quantizer.quantize(v).unwrap())
        .collect();

    // --- Compute all pairwise distances ---
    let n_pairs = n * (n - 1) / 2;
    let mut true_dists = Vec::with_capacity(n_pairs);
    let mut innr_dists = Vec::with_capacity(n_pairs);
    let mut rabitq_dists = Vec::with_capacity(n_pairs);

    for i in 0..n {
        for j in (i + 1)..n {
            true_dists.push((1.0 - cosine(&vectors[i], &vectors[j])) as f64);
            innr_dists.push(binary_hamming(&binary_vecs[i], &binary_vecs[j]) as f64);
            // RaBitQ: hamming on packed codes
            let h: u32 = rabitq_codes[i]
                .codes
                .iter()
                .zip(rabitq_codes[j].codes.iter())
                .map(|(&a, &b)| if a != b { 1u32 } else { 0u32 })
                .sum();
            rabitq_dists.push(h as f64);
        }
    }

    // --- Correlation (Spearman rank correlation) ---
    let rho_innr = spearman_rho(&true_dists, &innr_dists);
    let rho_rabitq = spearman_rho(&true_dists, &rabitq_dists);

    println!("=== Binary Encoding Comparison ===\n");
    println!("  {n} vectors, dim={dim}, {n_pairs} pairs\n");
    println!("  Method           Spearman rho");
    println!("  ---------------  -----------");
    println!("  innr::binary     {rho_innr:11.4}");
    println!("  qntz::RaBitQ     {rho_rabitq:11.4}");
    println!();
    if rho_rabitq > rho_innr {
        let pct = (rho_rabitq - rho_innr) / rho_innr.abs().max(1e-10) * 100.0;
        println!("  RaBitQ preserves distance ranking {pct:.1}% better than simple binarization.");
    } else {
        println!("  Simple binarization matches or beats RaBitQ here (small dim, easy data).");
    }
}

/// Deterministic pseudo-random vectors via xorshift.
fn gen_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed.max(1);
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
                })
                .collect()
        })
        .collect()
}

/// Spearman rank correlation.
fn spearman_rho(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let rx = ranks(x);
    let ry = ranks(y);
    let mean_r = (n as f64 + 1.0) / 2.0;
    let mut num = 0.0;
    let mut dx2 = 0.0;
    let mut dy2 = 0.0;
    for i in 0..n {
        let di = rx[i] - mean_r;
        let dj = ry[i] - mean_r;
        num += di * dj;
        dx2 += di * di;
        dy2 += dj * dj;
    }
    num / (dx2.sqrt() * dy2.sqrt())
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut result = vec![0.0; values.len()];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        result[*idx] = rank as f64 + 1.0;
    }
    result
}
