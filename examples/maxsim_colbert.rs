//! ColBERT-style MaxSim Late Interaction Scoring
//!
//! MaxSim computes the sum of per-query-token maximum similarities across
//! document tokens. This is the scoring function used by ColBERT and its
//! descendants (ColBERTv2, PLAID, ColPali).
//!
//! The key trade-off: MaxSim retains token-level expressiveness (like
//! cross-encoders) but uses simple aggregation (like bi-encoders), enabling
//! precomputation of document token embeddings.
//!
//! ```bash
//! cargo run --example maxsim_colbert --release --features maxsim
//! ```

use innr::{dot, maxsim};
use std::time::Instant;

fn main() {
    println!("ColBERT MaxSim Late Interaction Scoring");
    println!("=======================================\n");

    // 1. Basic correctness demonstration
    demo_basic();

    // 2. Non-commutativity (a common source of bugs)
    demo_non_commutative();

    // 3. Realistic-scale scoring with timing
    demo_realistic_scale();

    println!("Done!");
}

// =============================================================================
// Demos
// =============================================================================

fn demo_basic() {
    println!("1. Basic MaxSim Scoring");
    println!("   --------------------\n");

    println!("   MaxSim(Q, D) = sum_i max_j (q_i . d_j)\n");

    // Small example: 2 query tokens, 3 doc tokens, dim=4
    let q0 = [1.0f32, 0.0, 0.0, 0.0]; // "selects" docs with high dim-0
    let q1 = [0.0f32, 1.0, 0.0, 0.0]; // "selects" docs with high dim-1

    let d0 = [0.9f32, 0.1, 0.0, 0.0]; // high dim-0 -> matches q0
    let d1 = [0.1f32, 0.8, 0.0, 0.0]; // high dim-1 -> matches q1
    let d2 = [0.5f32, 0.5, 0.0, 0.0]; // moderate on both

    let query: Vec<&[f32]> = vec![&q0, &q1];
    let doc: Vec<&[f32]> = vec![&d0, &d1, &d2];

    let score = maxsim(&query, &doc);

    // Manual computation:
    //   q0 best match: max(0.9, 0.1, 0.5) = 0.9 (d0)
    //   q1 best match: max(0.1, 0.8, 0.5) = 0.8 (d1)
    //   total = 0.9 + 0.8 = 1.7
    println!("   Query tokens: 2, Doc tokens: 3, Dim: 4");
    println!("   MaxSim score: {:.4}", score);
    println!("   Expected:     1.7000");
    assert!((score - 1.7).abs() < 1e-5, "basic MaxSim mismatch");
    println!();
}

fn demo_non_commutative() {
    println!("2. MaxSim Is Not Commutative");
    println!("   -------------------------\n");

    println!("   maxsim(Q, D) != maxsim(D, Q) in general.");
    println!("   The first argument is always the \"query\" side.\n");

    // 1 query token vs 3 doc tokens
    let q0 = [1.0f32, 0.0, 0.0, 0.0];

    let d0 = [0.5f32, 0.5, 0.0, 0.0];
    let d1 = [0.3f32, 0.7, 0.0, 0.0];
    let d2 = [0.8f32, 0.2, 0.0, 0.0];

    let query: Vec<&[f32]> = vec![&q0];
    let doc: Vec<&[f32]> = vec![&d0, &d1, &d2];

    let score_qd = maxsim(&query, &doc);
    let score_dq = maxsim(&doc, &query);

    // maxsim(Q, D): 1 query token, picks best of 3 docs
    //   q0 best: max(0.5, 0.3, 0.8) = 0.8
    //   total = 0.8
    //
    // maxsim(D, Q): 3 "query" tokens, each picks best of 1 doc
    //   d0 best: 0.5
    //   d1 best: 0.3
    //   d2 best: 0.8
    //   total = 0.5 + 0.3 + 0.8 = 1.6
    println!("   maxsim(Q[1], D[3]) = {:.4}", score_qd);
    println!("   maxsim(D[3], Q[1]) = {:.4}", score_dq);
    println!("   Difference:          {:.4}\n", (score_qd - score_dq).abs());

    assert!((score_qd - 0.8).abs() < 1e-5);
    assert!((score_dq - 1.6).abs() < 1e-5);
}

fn demo_realistic_scale() {
    println!("3. Realistic Scale: 32 Query Tokens x 128 Doc Tokens x 128d");
    println!("   ----------------------------------------------------------\n");

    let dim = 128;
    let n_query_tokens = 32;
    let n_doc_tokens = 128;

    // Generate synthetic normalized embeddings
    let query_vecs: Vec<Vec<f32>> = (0..n_query_tokens)
        .map(|i| generate_normalized(dim, i as u64))
        .collect();
    let doc_vecs: Vec<Vec<f32>> = (0..n_doc_tokens)
        .map(|i| generate_normalized(dim, (i + 1000) as u64))
        .collect();

    let query_refs: Vec<&[f32]> = query_vecs.iter().map(|v| v.as_slice()).collect();
    let doc_refs: Vec<&[f32]> = doc_vecs.iter().map(|v| v.as_slice()).collect();

    // -- innr::maxsim --
    let start = Instant::now();
    let score = maxsim(&query_refs, &doc_refs);
    let innr_time = start.elapsed();

    // -- Naive loop for verification --
    let start = Instant::now();
    let naive_score = naive_maxsim(&query_refs, &doc_refs);
    let naive_time = start.elapsed();

    println!("   innr::maxsim score:  {:.6}", score);
    println!("   Naive loop score:    {:.6}", naive_score);
    println!(
        "   Match:               {}",
        if (score - naive_score).abs() < 1e-3 {
            "yes"
        } else {
            "NO -- divergence"
        }
    );
    assert!(
        (score - naive_score).abs() < 1e-3,
        "maxsim vs naive diverged: {} vs {}",
        score,
        naive_score
    );
    println!();
    println!("   Single-pair timing:");
    println!("     innr::maxsim: {:?}", innr_time);
    println!("     Naive loop:   {:?}", naive_time);
    println!();

    // -- Batch timing: score query against many documents --
    let n_docs = 1000;
    let all_docs: Vec<Vec<Vec<f32>>> = (0..n_docs)
        .map(|doc_id| {
            (0..n_doc_tokens)
                .map(|tok| generate_normalized(dim, (doc_id * n_doc_tokens + tok + 5000) as u64))
                .collect()
        })
        .collect();

    let start = Instant::now();
    let mut scores: Vec<f32> = Vec::with_capacity(n_docs);
    for doc in &all_docs {
        let refs: Vec<&[f32]> = doc.iter().map(|v| v.as_slice()).collect();
        scores.push(maxsim(&query_refs, &refs));
    }
    let batch_time = start.elapsed();
    std::hint::black_box(&scores);

    println!(
        "   Batch: {} docs scored in {:?} ({:.1} us/doc)",
        n_docs,
        batch_time,
        batch_time.as_micros() as f64 / n_docs as f64
    );

    // Show top-5
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("   Top-5 scores:");
    for (rank, &(idx, score)) in indexed.iter().take(5).enumerate() {
        println!("     #{}: doc {} = {:.4}", rank + 1, idx, score);
    }
    println!();
}

// =============================================================================
// Helpers
// =============================================================================

/// Naive O(|Q| * |D| * dim) MaxSim using innr::dot.
fn naive_maxsim(query: &[&[f32]], doc: &[&[f32]]) -> f32 {
    query
        .iter()
        .map(|q| {
            doc.iter()
                .map(|d| dot(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

/// Deterministic pseudo-random normalized embedding.
fn generate_normalized(dim: usize, seed: u64) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim)
        .map(|i| {
            let x = seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64 * 1442695040888963407);
            ((x >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        })
        .collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}
