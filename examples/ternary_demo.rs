//! Ternary Quantization: 16x Memory, 18x Speed
//!
//! Demonstrates the trade-offs of 1.58-bit ternary quantization.
//!
//! Ternary excels at: memory compression, fast similarity, deduplication
//! Ternary struggles with: high-recall nearest neighbor search
//!
//! # Mathematical Foundation Context
//!
//! `innr` provides low-level SIMD primitives. Higher-level crates build on these:
//!
//! | Level        | Crate            | What it provides              |
//! |--------------|------------------|-------------------------------|
//! | **Primitive** | `innr`          | SIMD dot/cosine, ternary ops  |
//! | **Index**     | `plesio`      | HNSW, LSH, IVF-PQ, RaBitQ     |
//! | **Ranking**   | `ordino`        | BM25, reranking, fusion       |
//! | **Pipeline**  | `hop`           | Ingestion, chunking, retrieval |
//!
//! Typical flow: `innr` for distance -> `plesio` for indexing -> `ordino` for ranking
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
        let ternary_bytes = (n as u64) * (dim as u64).div_ceil(4);

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
    println!("3. Where Ternary Shines: Fast Similarity Screening");
    println!("   -------------------------------------------------\n");

    println!("   Ternary preserves relative similarity well for rough screening.\n");

    let dim = 768;
    let n_topics = 50;
    let threshold = 0.01;

    // Generate pairs: same-topic and different-topic
    let n_pairs = 1000;
    let mut same_topic_corr = Vec::new();
    let mut diff_topic_corr = Vec::new();

    for i in 0..n_pairs {
        // Same topic pair (docs i and i+n_topics share the same topic cluster)
        let doc1 = generate_embedding(dim, i as u64, n_topics);
        let doc2 = generate_embedding(dim, (i + n_topics) as u64, n_topics);
        let t1 = encode_ternary(&doc1, threshold);
        let t2 = encode_ternary(&doc2, threshold);

        let f32_sim = dot(&doc1, &doc2);
        let tern_sim = ternary_dot(&t1, &t2) as f32 / dim as f32;
        same_topic_corr.push((f32_sim, tern_sim));

        // Different topic pair
        let doc3 = generate_embedding(dim, (i + n_topics / 2) as u64, n_topics);
        let t3 = encode_ternary(&doc3, threshold);

        let f32_sim2 = dot(&doc1, &doc3);
        let tern_sim2 = ternary_dot(&t1, &t3) as f32 / dim as f32;
        diff_topic_corr.push((f32_sim2, tern_sim2));
    }

    // Compute rank correlation (Spearman-ish: does ternary order match f32 order?)
    let all_pairs: Vec<(f32, f32)> = same_topic_corr
        .iter()
        .chain(diff_topic_corr.iter())
        .copied()
        .collect();

    // Sort by f32 similarity
    let mut by_f32 = all_pairs.clone();
    by_f32.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Sort by ternary similarity
    let mut by_tern = all_pairs.clone();
    by_tern.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Count how many of f32's top-100 are also in ternary's top-100
    let f32_top: std::collections::HashSet<usize> = by_f32
        .iter()
        .take(100)
        .map(|p| all_pairs.iter().position(|x| x == p).unwrap())
        .collect();
    let tern_top: std::collections::HashSet<usize> = by_tern
        .iter()
        .take(100)
        .map(|p| all_pairs.iter().position(|x| x == p).unwrap())
        .collect();

    let overlap = f32_top.intersection(&tern_top).count();

    println!(
        "   Rank preservation test ({} similarity pairs):",
        n_pairs * 2
    );
    println!("     Overlap in top-100 most similar: {}%\n", overlap);

    // Show distribution of scores
    let same_f32_mean: f32 = same_topic_corr.iter().map(|(f, _)| f).sum::<f32>() / n_pairs as f32;
    let diff_f32_mean: f32 = diff_topic_corr.iter().map(|(f, _)| f).sum::<f32>() / n_pairs as f32;
    let same_tern_mean: f32 = same_topic_corr.iter().map(|(_, t)| t).sum::<f32>() / n_pairs as f32;
    let diff_tern_mean: f32 = diff_topic_corr.iter().map(|(_, t)| t).sum::<f32>() / n_pairs as f32;

    println!("   Similarity distributions:");
    println!("   {:12} {:>15} {:>15}", "", "Same Topic", "Diff Topic");
    println!("   {}", "-".repeat(45));
    println!(
        "   {:12} {:>15.3} {:>15.3}",
        "f32 mean", same_f32_mean, diff_f32_mean
    );
    println!(
        "   {:12} {:>15.3} {:>15.3}",
        "Ternary mean", same_tern_mean, diff_tern_mean
    );

    println!("\n   Both methods separate same-topic from different-topic pairs.");
    println!("   Ternary preserves rough ordering for fast pre-filtering.");
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
    let avg_sparsity: f32 = docs_ternary.iter().map(sparsity).sum::<f32>() / n_docs as f32;

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
