//! Binary (1-Bit) Quantization
//!
//! Binary quantization maps each embedding dimension to a single bit (1 if above
//! threshold, 0 otherwise). This yields 32x memory reduction vs f32, at the cost
//! of significant information loss.
//!
//! Operations on packed binary vectors use bitwise AND/OR/XOR + popcount, which
//! run at memory bandwidth on any CPU with a popcount instruction.
//!
//! Use case: first-stage candidate retrieval over very large corpora, followed
//! by reranking with full-precision vectors.
//!
//! ```bash
//! cargo run --example binary_demo --release
//! ```

use innr::binary::{binary_dot, binary_hamming, binary_jaccard, encode_binary};
use innr::{cosine, dot};
use std::time::Instant;

fn main() {
    println!("Binary (1-Bit) Quantization");
    println!("===========================\n");

    // 1. Encoding and memory
    demo_encoding();

    // 2. Distance/similarity operations
    demo_operations();

    // 3. Memory reduction
    demo_memory();

    // 4. Recall trade-off
    demo_recall();

    println!("Done!");
}

// =============================================================================
// Demos
// =============================================================================

fn demo_encoding() {
    println!("1. Encoding: f32 -> Binary");
    println!("   -----------------------\n");

    let embedding = [0.5f32, -0.3, 0.9, 0.0, -0.7, 0.1, 0.0, 0.8];
    let packed = encode_binary(&embedding, 0.0);

    println!("   f32 values:  {:?}", embedding);
    print!("   Binary bits: [");
    for i in 0..embedding.len() {
        if i > 0 {
            print!(", ");
        }
        print!("{}", if packed.get(i) { 1 } else { 0 });
    }
    println!("]");
    println!("   Rule: 1 if value > threshold (0.0), else 0");
    println!();

    // Verify
    assert!(packed.get(0)); //  0.5 > 0
    assert!(!packed.get(1)); // -0.3 <= 0
    assert!(packed.get(2)); //  0.9 > 0
    assert!(!packed.get(3)); //  0.0 <= 0 (not strictly above)
    assert!(!packed.get(4)); // -0.7 <= 0
    assert!(packed.get(5)); //  0.1 > 0
    assert!(!packed.get(6)); //  0.0 <= 0
    assert!(packed.get(7)); //  0.8 > 0
}

fn demo_operations() {
    println!("2. Binary Similarity Operations");
    println!("   ----------------------------\n");

    let a_f32 = [1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    let b_f32 = [1.0f32, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0];

    let a = encode_binary(&a_f32, 0.0);
    let b = encode_binary(&b_f32, 0.0);

    println!("   a bits: 1 0 1 0 1 0 1 0");
    println!("   b bits: 1 1 0 0 1 1 0 0\n");

    let hamming = binary_hamming(&a, &b);
    let dot_val = binary_dot(&a, &b);
    let jaccard = binary_jaccard(&a, &b);

    // Hamming: positions that differ -> {1, 2, 5, 6} = 4
    println!("   Hamming distance: {} (bits that differ)", hamming);
    assert_eq!(hamming, 4);

    // Dot (intersection): positions both 1 -> {0, 4} = 2
    println!("   Binary dot:       {} (bits both 1)", dot_val);
    assert_eq!(dot_val, 2);

    // Jaccard: |intersection| / |union| = 2 / 6
    // Union: positions with at least one 1 -> {0, 1, 2, 4, 5, 6} = 6
    println!("   Jaccard:          {:.4} (|A & B| / |A | B|)", jaccard);
    assert!((jaccard - 2.0 / 6.0).abs() < 1e-6);
    println!();
}

fn demo_memory() {
    println!("3. Memory Reduction: 32x Compression");
    println!("   ----------------------------------\n");

    let configs: &[(&str, usize, usize)] = &[
        ("384d, 1M docs", 384, 1_000_000),
        ("768d, 1M docs", 768, 1_000_000),
        ("768d, 10M docs", 768, 10_000_000),
        ("1536d, 10M docs", 1536, 10_000_000),
    ];

    println!(
        "   {:25} {:>10} {:>10} {:>7}",
        "Config", "f32", "Binary", "Ratio"
    );
    println!("   {}", "-".repeat(55));

    for &(name, dim, n) in configs {
        let f32_bytes = n as u64 * dim as u64 * 4;
        let binary_bytes = n as u64 * (dim as u64).div_ceil(64) * 8;
        let ratio = f32_bytes as f64 / binary_bytes as f64;

        println!(
            "   {:25} {:>10} {:>10} {:>6.1}x",
            name,
            format_bytes(f32_bytes),
            format_bytes(binary_bytes),
            ratio
        );
    }
    println!();
}

fn demo_recall() {
    println!("4. Recall Trade-off: Binary vs Exact");
    println!("   ---------------------------------\n");

    let dim = 768;
    let n_docs = 5000;
    let n_queries = 50;
    let k = 10;

    // Generate corpus
    let docs_f32: Vec<Vec<f32>> = (0..n_docs)
        .map(|i| generate_normalized(dim, i as u64))
        .collect();
    let docs_binary: Vec<_> = docs_f32.iter().map(|v| encode_binary(v, 0.0)).collect();

    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|i| generate_normalized(dim, (i + 100_000) as u64))
        .collect();

    let mut total_recall = 0.0f64;

    // Compute recall: exact top-k vs binary top-k
    for query in &queries {
        let query_binary = encode_binary(query, 0.0);

        // Exact top-k by cosine
        let mut exact_scores: Vec<(usize, f32)> = docs_f32
            .iter()
            .enumerate()
            .map(|(i, d)| (i, cosine(query, d)))
            .collect();
        exact_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        let exact_topk: Vec<usize> = exact_scores.iter().take(k).map(|(i, _)| *i).collect();

        // Binary top-k by Hamming (ascending = most similar)
        let mut binary_scores: Vec<(usize, u32)> = docs_binary
            .iter()
            .enumerate()
            .map(|(i, d)| (i, binary_hamming(&query_binary, d)))
            .collect();
        binary_scores.sort_by_key(|(_, h)| *h);
        let binary_topk: Vec<usize> = binary_scores.iter().take(k).map(|(i, _)| *i).collect();

        let overlap = exact_topk.iter().filter(|i| binary_topk.contains(i)).count();
        total_recall += overlap as f64 / k as f64;
    }

    // Time binary scoring only
    let start = Instant::now();
    for query in &queries {
        let query_binary = encode_binary(query, 0.0);
        let mut binary_scores: Vec<(usize, u32)> = docs_binary
            .iter()
            .enumerate()
            .map(|(i, d)| (i, binary_hamming(&query_binary, d)))
            .collect();
        binary_scores.sort_by_key(|(_, h)| *h);
        std::hint::black_box(&binary_scores);
    }
    let binary_time = start.elapsed();

    // Time exact scoring only
    let start = Instant::now();
    for query in &queries {
        let mut scores: Vec<(usize, f32)> = docs_f32
            .iter()
            .enumerate()
            .map(|(i, d)| (i, dot(query, d)))
            .collect();
        scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        std::hint::black_box(&scores);
    }
    let exact_time = start.elapsed();

    let avg_recall = total_recall / n_queries as f64;

    println!(
        "   Corpus: {} docs x {}d, {} queries, k={}\n",
        n_docs, dim, n_queries, k
    );
    println!("   Recall@{}: {:.1}%", k, avg_recall * 100.0);
    println!("   (overlap between binary top-k and exact top-k)\n");
    println!(
        "   Binary scoring: {:?} ({:.1} us/query)",
        binary_time,
        binary_time.as_micros() as f64 / n_queries as f64
    );
    println!(
        "   Exact scoring:  {:?} ({:.1} us/query)",
        exact_time,
        exact_time.as_micros() as f64 / n_queries as f64
    );
    println!();
    println!("   Binary quantization trades recall for speed and 32x less memory.");
    println!("   Typical usage: binary retrieves top-1000 candidates, then rerank");
    println!("   with full-precision vectors for the final top-k.");
    println!();
}

// =============================================================================
// Helpers
// =============================================================================

fn generate_normalized(dim: usize, seed: u64) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim)
        .map(|i| {
            let x = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(i as u64 * 1442695040888963407);
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

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1e3)
    } else {
        format!("{} B", bytes)
    }
}
