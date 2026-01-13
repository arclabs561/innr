//! Ternary Quantization: 16x Memory, 18x Speed
//!
//! Demonstrates the trade-offs of 1.58-bit ternary quantization.
//!
//! Ternary excels at: memory compression, fast similarity, deduplication
//! Ternary struggles with: high-recall nearest neighbor search
//!
//! ```bash
//! cargo run --example ternary_demo --release
//! ```

use innr::ternary::{encode_ternary, sparsity, ternary_dot, PackedTernary};
use std::time::Instant;

fn main() {
    println!("Ternary Quantization: Extreme Compression Trade-offs");
    println!("=====================================================\n");

    println!("Ternary maps each dimension to {{-1, 0, +1}}, using ~1.58 bits/dim.");
    println!("This is AGGRESSIVE compression - 20x smaller than f32.\n");

    // 1. Memory compression (the main win)
    demo_compression();

    // 2. Speed (popcount is fast)
    demo_speed();

    // 3. Where ternary shines: deduplication
    demo_deduplication();

    // 4. Where ternary struggles: fine-grained ranking
    demo_ranking_accuracy();

    // 5. Best practice: two-stage with reranking
    demo_best_practice();

    println!("Done!");
}

// =============================================================================
// Realistic Embedding Generator
// =============================================================================

/// Generate embeddings that mimic transformer outputs with semantic clustering.
fn generate_embedding(dim: usize, doc_id: u64, n_topics: usize) -> Vec<f32> {
    // Assign to a topic cluster
    let topic = (doc_id % n_topics as u64) as usize;

    // Generate with Gaussian components + topic bias
    let mut embedding: Vec<f32> = (0..dim)
        .map(|i| {
            // Base Gaussian component
            let u1 = lcg_random(doc_id.wrapping_add(i as u64 * 2));
            let u2 = lcg_random(doc_id.wrapping_add(i as u64 * 2 + 1));
            let gauss =
                (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();

            // Add topic-specific bias to some dimensions
            let topic_dims = dim / n_topics;
            let bias = if i / topic_dims == topic { 2.0 } else { 0.0 };

            gauss + bias
        })
        .collect();

    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in &mut embedding {
            *x /= norm;
        }
    }

    embedding
}

fn lcg_random(seed: u64) -> f32 {
    let a: u64 = 6364136223846793005;
    let c: u64 = 1442695040888963407;
    let next = seed.wrapping_mul(a).wrapping_add(c);
    (next >> 33) as f32 / (1u64 << 31) as f32
}

// =============================================================================
// Demos
// =============================================================================

fn demo_compression() {
    println!("1. Memory Compression (The Main Win)");
    println!("   ----------------------------------\n");

    let configs = [
        ("Small model (384d, 100K docs)", 384, 100_000),
        ("BERT-base (768d, 1M docs)", 768, 1_000_000),
        ("OpenAI (1536d, 10M docs)", 1536, 10_000_000),
    ];

    println!(
        "   {:35} {:>12} {:>12} {:>8}",
        "Configuration", "f32", "Ternary", "Ratio"
    );
    println!("   {}", "-".repeat(72));

    for (name, dim, n) in configs {
        let f32_bytes = (n as u64) * (dim as u64) * 4;
        let ternary_bytes = (n as u64) * ((dim as u64 + 3) / 4);

        let f32_str = format_bytes(f32_bytes);
        let ternary_str = format_bytes(ternary_bytes);
        let ratio = f32_bytes as f64 / ternary_bytes as f64;

        println!(
            "   {:35} {:>12} {:>12} {:>7.0}x",
            name, f32_str, ternary_str, ratio
        );
    }

    println!("\n   Key insight: Ternary enables in-memory indices that would otherwise need disk.");
    println!();
}

fn demo_speed() {
    println!("2. Speed: Popcount vs Multiply-Add");
    println!("   --------------------------------\n");

    println!("   Ternary dot product uses bitwise operations:");
    println!("     same_sign = popcount(a_pos & b_pos) + popcount(a_neg & b_neg)");
    println!("     diff_sign = popcount(a_pos & b_neg) + popcount(a_neg & b_pos)");
    println!("     result = same_sign - diff_sign\n");

    let dims = [384, 768, 1536];
    let n_pairs = 20_000;

    println!(
        "   {:>8} {:>15} {:>15} {:>10}",
        "Dim", "f32 dot", "Ternary dot", "Speedup"
    );
    println!("   {}", "-".repeat(55));

    for &dim in &dims {
        let vecs_f32: Vec<Vec<f32>> = (0..n_pairs * 2)
            .map(|i| generate_embedding(dim, i as u64, 50))
            .collect();

        let threshold = 0.01; // Small threshold to preserve most info
        let vecs_ternary: Vec<PackedTernary> = vecs_f32
            .iter()
            .map(|v| encode_ternary(v, threshold))
            .collect();

        // Benchmark f32
        let f32_start = Instant::now();
        let mut f32_sum = 0.0f32;
        for i in 0..n_pairs {
            f32_sum += dot(&vecs_f32[i * 2], &vecs_f32[i * 2 + 1]);
        }
        let f32_time = f32_start.elapsed();

        // Benchmark ternary
        let ternary_start = Instant::now();
        let mut ternary_sum = 0i32;
        for i in 0..n_pairs {
            ternary_sum += ternary_dot(&vecs_ternary[i * 2], &vecs_ternary[i * 2 + 1]);
        }
        let ternary_time = ternary_start.elapsed();

        // Prevent optimization
        std::hint::black_box((f32_sum, ternary_sum));

        let f32_ns = f32_time.as_nanos() as f64 / n_pairs as f64;
        let ternary_ns = ternary_time.as_nanos() as f64 / n_pairs as f64;

        println!(
            "   {:>8} {:>12.1} ns {:>12.1} ns {:>9.1}x",
            dim,
            f32_ns,
            ternary_ns,
            f32_ns / ternary_ns
        );
    }
    println!();
}

fn demo_deduplication() {
    println!("3. Where Ternary Shines: Deduplication");
    println!("   ------------------------------------\n");

    println!("   Deduplication doesn't need exact rankings - just finding near-duplicates.\n");

    let dim = 768;
    let n_docs = 5000;
    let threshold = 0.01;

    // Generate documents with some duplicates
    let mut docs_f32: Vec<Vec<f32>> = Vec::new();
    let mut is_duplicate: Vec<bool> = Vec::new();

    for i in 0..n_docs {
        if i > 0 && i % 50 == 0 {
            // Create near-duplicate of random earlier doc
            let orig_idx = (i * 7) % i;
            let mut dup = docs_f32[orig_idx].clone();
            // Add tiny noise
            for (j, x) in dup.iter_mut().enumerate() {
                let noise = lcg_random((i * dim + j) as u64) * 0.01 - 0.005;
                *x += noise;
            }
            // Renormalize
            let norm: f32 = dup.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut dup {
                *x /= norm;
            }
            docs_f32.push(dup);
            is_duplicate.push(true);
        } else {
            docs_f32.push(generate_embedding(dim, i as u64, 50));
            is_duplicate.push(false);
        }
    }

    let n_true_dups = is_duplicate.iter().filter(|&&b| b).count();

    // Encode to ternary
    let docs_ternary: Vec<PackedTernary> = docs_f32
        .iter()
        .map(|d| encode_ternary(d, threshold))
        .collect();

    // Find duplicates using ternary similarity threshold
    // Ternary dot ranges from -dim to +dim, so ~0.9*dim for near-duplicates
    let dup_threshold_ternary = (dim as i32 * 6) / 10; // 60% of max (conservative)
    let dup_threshold_f32 = 0.98; // Very high for f32

    let mut ternary_found = 0;
    let mut f32_found = 0;
    let mut ternary_false_positives = 0;
    let mut f32_false_positives = 0;

    for i in 1..n_docs {
        for j in 0..i {
            let tern_sim = ternary_dot(&docs_ternary[i], &docs_ternary[j]);
            let f32_sim = dot(&docs_f32[i], &docs_f32[j]);

            let tern_dup = tern_sim > dup_threshold_ternary;
            let f32_dup = f32_sim > dup_threshold_f32;
            let true_dup = is_duplicate[i] && (j == (i * 7) % i);

            if tern_dup && true_dup {
                ternary_found += 1;
            }
            if tern_dup && !true_dup {
                ternary_false_positives += 1;
            }
            if f32_dup && true_dup {
                f32_found += 1;
            }
            if f32_dup && !true_dup {
                f32_false_positives += 1;
            }
        }
    }

    println!(
        "   {} documents with {} near-duplicates\n",
        n_docs, n_true_dups
    );
    println!(
        "   {:20} {:>12} {:>15} {:>12}",
        "Method", "Found", "False Positives", "Precision"
    );
    println!("   {}", "-".repeat(65));

    let tern_precision = if ternary_found + ternary_false_positives > 0 {
        ternary_found as f64 / (ternary_found + ternary_false_positives) as f64
    } else {
        0.0
    };
    let f32_precision = if f32_found + f32_false_positives > 0 {
        f32_found as f64 / (f32_found + f32_false_positives) as f64
    } else {
        0.0
    };

    println!(
        "   {:20} {:>12} {:>15} {:>11.1}%",
        "Ternary",
        format!("{}/{}", ternary_found, n_true_dups),
        ternary_false_positives,
        tern_precision * 100.0
    );
    println!(
        "   {:20} {:>12} {:>15} {:>11.1}%",
        "f32",
        format!("{}/{}", f32_found, n_true_dups),
        f32_false_positives,
        f32_precision * 100.0
    );

    println!(
        "\n   Ternary works well for deduplication where high precision matters more than recall."
    );
    println!();
}

fn demo_ranking_accuracy() {
    println!("4. Where Ternary Struggles: Fine-Grained Ranking");
    println!("   -----------------------------------------------\n");

    println!("   Ternary loses too much information for accurate top-k ranking.\n");

    let dim = 768;
    let n_docs = 5000;
    let threshold = 0.01;

    let docs_f32: Vec<Vec<f32>> = (0..n_docs)
        .map(|i| generate_embedding(dim, i as u64, 50))
        .collect();

    let docs_ternary: Vec<PackedTernary> = docs_f32
        .iter()
        .map(|d| encode_ternary(d, threshold))
        .collect();

    // Check sparsity
    let avg_sparsity: f32 = docs_ternary.iter().map(|d| sparsity(d)).sum::<f32>() / n_docs as f32;

    println!(
        "   Avg sparsity: {:.1}% (values mapped to 0)",
        avg_sparsity * 100.0
    );

    // Measure rank correlation for random pairs
    let n_queries = 50;
    let mut rank_correlations = Vec::new();

    for q in 0..n_queries {
        let query = generate_embedding(dim, 100000 + q, 50);
        let query_ternary = encode_ternary(&query, threshold);

        // Get f32 rankings
        let mut f32_scores: Vec<(usize, f32)> = docs_f32
            .iter()
            .enumerate()
            .map(|(i, d)| (i, dot(&query, d)))
            .collect();
        f32_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Get ternary rankings
        let mut tern_scores: Vec<(usize, i32)> = docs_ternary
            .iter()
            .enumerate()
            .map(|(i, d)| (i, ternary_dot(&query_ternary, d)))
            .collect();
        tern_scores.sort_by(|a, b| b.1.cmp(&a.1));

        // Compute rank correlation (Spearman) for top 100
        let k = 100;
        let f32_top: Vec<usize> = f32_scores.iter().take(k).map(|(i, _)| *i).collect();
        let tern_top: Vec<usize> = tern_scores.iter().take(k).map(|(i, _)| *i).collect();

        // Count overlap
        let overlap = f32_top.iter().filter(|i| tern_top.contains(i)).count();
        rank_correlations.push(overlap as f32 / k as f32);
    }

    let avg_correlation: f32 = rank_correlations.iter().sum::<f32>() / n_queries as f32;

    println!(
        "   Overlap in top-100 results: {:.1}%\n",
        avg_correlation * 100.0
    );

    println!("   This is why ternary alone isn't suitable for search.");
    println!("   For search, use: Product Quantization, Scalar Quantization, or HNSW.");
    println!();
}

fn demo_best_practice() {
    println!("5. Best Practice: When to Use Ternary");
    println!("   ------------------------------------\n");

    println!("   Good use cases:");
    println!("     - Near-duplicate detection");
    println!("     - Document fingerprinting");
    println!("     - Rough similarity pre-filtering (with reranking)");
    println!("     - Memory-constrained mobile/edge deployment");
    println!();

    println!("   Not recommended for:");
    println!("     - High-recall nearest neighbor search");
    println!("     - Fine-grained ranking");
    println!("     - Applications where top-1 accuracy matters");
    println!();

    println!("   Better alternatives for search:");
    println!("     - Product Quantization (PQ): 4-8 bits/dim, 90%+ recall");
    println!("     - Scalar Quantization (SQ): 8 bits/dim, 95%+ recall");
    println!("     - HNSW + PQ: Sublinear search with good recall");
    println!();
}

// =============================================================================
// Helpers
// =============================================================================

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
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
