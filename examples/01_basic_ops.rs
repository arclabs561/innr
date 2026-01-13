//! Basic Vector Operations
//!
//! The minimal example: dot, cosine, L2.
//!
//! # Metric Selection Guide
//!
//! | Metric     | Range      | Use Case                           |
//! |------------|------------|------------------------------------|
//! | Dot        | (-inf,inf) | Unnormalized similarity, MaxSim    |
//! | Cosine     | [-1, 1]    | Direction similarity (normalized)  |
//! | L2         | [0, inf)   | Euclidean distance, cluster radius |
//! | L2^2       | [0, inf)   | Comparison-only (avoids sqrt)      |
//!
//! # Key Identity (for normalized vectors)
//!
//! ```text
//! L2^2(a, b) = 2 * (1 - cosine(a, b))
//! ```
//!
//! This means cosine similarity and L2 distance are equivalent for
//! normalized vectors - they rank results identically.
//!
//! ```bash
//! cargo run --example 01_basic_ops --release
//! ```

use innr::{cosine, dot, l2_distance, l2_distance_squared};

fn main() {
    // Two simple vectors
    let a = vec![1.0_f32, 2.0, 3.0, 4.0];
    let b = vec![4.0_f32, 3.0, 2.0, 1.0];

    // Dot product: sum of element-wise products
    // 1*4 + 2*3 + 3*2 + 4*1 = 4 + 6 + 6 + 4 = 20
    let d = dot(&a, &b);
    println!("dot(a, b) = {}", d);
    assert!((d - 20.0).abs() < 1e-6);

    // Cosine similarity: dot / (||a|| * ||b||)
    // Range: [-1, 1], where 1 = same direction, -1 = opposite
    let c = cosine(&a, &b);
    println!("cosine(a, b) = {:.4}", c);

    // L2 (Euclidean) distance: sqrt(sum((a-b)^2))
    let l2 = l2_distance(&a, &b);
    println!("l2_distance(a, b) = {:.4}", l2);

    // Squared L2: avoids sqrt, useful for comparisons
    let l2_sq = l2_distance_squared(&a, &b);
    println!("l2_distance_squared(a, b) = {:.4}", l2_sq);
    assert!((l2_sq - l2 * l2).abs() < 1e-6);

    // Normalized vectors
    let a_norm = normalize(&a);
    let b_norm = normalize(&b);

    // For normalized vectors: dot == cosine
    let dot_norm = dot(&a_norm, &b_norm);
    let cos_norm = cosine(&a_norm, &b_norm);
    println!("\nFor normalized vectors:");
    println!("  dot = {:.4}, cosine = {:.4}", dot_norm, cos_norm);
    assert!((dot_norm - cos_norm).abs() < 1e-6);

    // For normalized vectors: L2^2 = 2(1 - cosine)
    let l2_sq_norm = l2_distance_squared(&a_norm, &b_norm);
    let expected = 2.0 * (1.0 - cos_norm);
    println!("  l2^2 = {:.4}, 2(1-cos) = {:.4}", l2_sq_norm, expected);
    assert!((l2_sq_norm - expected).abs() < 1e-5);
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / norm).collect()
}
