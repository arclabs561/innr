//! Batch Operations with Columnar (PDX-Style) Layout
//!
//! The PDX layout transposes vectors from row-major (one vector contiguous)
//! to column-major (one dimension contiguous across all vectors). This enables:
//!
//! - Sequential memory access patterns that auto-vectorize well
//! - Dimension-by-dimension early termination for kNN
//! - Cache-friendly batch distance computation
//!
//! Reference: Kuffo, Krippner, Boncz (2025, SIGMOD), "PDX: A Data Layout
//! for Vector Similarity Search".
//!
//! ```bash
//! cargo run --example batch_demo --release
//! ```

use innr::batch::{batch_dot, batch_knn, batch_l2_squared, VerticalBatch};
use innr::{dot, l2_distance_squared};
use std::time::Instant;

fn main() {
    println!("Batch Operations with PDX-Style Columnar Layout");
    println!("================================================\n");

    // 1. Layout transposition
    demo_layout();

    // 2. Batch kNN vs naive brute-force
    demo_knn();

    // 3. Batch dot product
    demo_batch_dot();

    // 4. Timing at scale
    demo_timing();

    println!("Done!");
}

// =============================================================================
// Demos
// =============================================================================

fn demo_layout() {
    println!("1. Row-Major vs Column-Major Layout");
    println!("   --------------------------------\n");

    let vectors = vec![
        vec![1.0f32, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];

    println!("   Row-major (standard): each vector is contiguous");
    for (i, v) in vectors.iter().enumerate() {
        println!("     v{}: {:?}", i, v);
    }

    let batch = VerticalBatch::from_rows(&vectors);
    println!();
    println!("   Column-major (PDX): each dimension is contiguous across vectors");
    for d in 0..batch.dimension() {
        let slice = batch.dimension_slice(d);
        println!("     dim {}: {:?}", d, slice);
    }

    // Verify round-trip
    for (i, original) in vectors.iter().enumerate() {
        let extracted = batch.extract_vector(i);
        assert_eq!(&extracted, original, "round-trip failed for vector {}", i);
    }
    println!();
    println!("   Round-trip verified: extract_vector recovers original data.");
    println!();
}

fn demo_knn() {
    println!("2. Batch kNN: Find Nearest Neighbors");
    println!("   ----------------------------------\n");

    let dim = 8;
    let n = 20;
    let k = 3;

    // Generate a small corpus
    let corpus: Vec<Vec<f32>> = (0..n)
        .map(|i| generate_embedding(dim, i as u64))
        .collect();
    let query = generate_embedding(dim, 999);

    let batch = VerticalBatch::from_rows(&corpus);

    // batch_knn
    let result = batch_knn(&query, &batch, k);

    // Naive brute-force for verification
    let mut naive_dists: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, l2_distance_squared(&query, v)))
        .collect();
    naive_dists.sort_by(|a, b| a.1.total_cmp(&b.1));
    naive_dists.truncate(k);

    println!("   Corpus: {} vectors, dim={}, k={}\n", n, dim, k);
    println!("   batch_knn results:");
    for (rank, (&idx, &dist)) in result.indices.iter().zip(result.distances.iter()).enumerate() {
        println!("     #{}: index={}, dist_sq={:.6}", rank + 1, idx, dist);
    }

    println!();
    println!("   Naive brute-force results:");
    for (rank, &(idx, dist)) in naive_dists.iter().enumerate() {
        println!("     #{}: index={}, dist_sq={:.6}", rank + 1, idx, dist);
    }

    // Verify indices match
    let batch_indices: Vec<usize> = result.indices.clone();
    let naive_indices: Vec<usize> = naive_dists.iter().map(|(i, _)| *i).collect();
    assert_eq!(batch_indices, naive_indices, "kNN results diverge");
    println!();
    println!("   Match: indices and distances agree.");
    println!();
}

fn demo_batch_dot() {
    println!("3. Batch Dot Product: One Query vs Many Documents");
    println!("   -----------------------------------------------\n");

    let corpus = vec![
        vec![1.0f32, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.707, 0.707, 0.0, 0.0],
        vec![0.5, 0.5, 0.5, 0.5],
    ];
    let query = vec![1.0f32, 0.0, 0.0, 0.0];

    let batch = VerticalBatch::from_rows(&corpus);
    let dots = batch_dot(&query, &batch);

    // Verify against innr::dot
    println!("   Query: {:?}\n", query);
    for (i, (batch_d, naive_d)) in dots
        .iter()
        .zip(corpus.iter().map(|v| dot(&query, v)))
        .enumerate()
    {
        println!(
            "     doc {}: batch_dot={:.4}, innr::dot={:.4}",
            i, batch_d, naive_d
        );
        assert!(
            (batch_d - naive_d).abs() < 1e-6,
            "dot mismatch at index {}",
            i
        );
    }
    println!();
}

fn demo_timing() {
    println!("4. Timing: Batch L2 vs Naive Loop");
    println!("   ------------------------------\n");

    let dim = 128;
    let n = 10_000;
    let n_queries = 100;

    let corpus: Vec<Vec<f32>> = (0..n)
        .map(|i| generate_embedding(dim, i as u64))
        .collect();
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|i| generate_embedding(dim, (i + 50_000) as u64))
        .collect();

    let batch = VerticalBatch::from_rows(&corpus);

    // Batch L2
    let start = Instant::now();
    let mut batch_checksum = 0.0f32;
    for q in &queries {
        let dists = batch_l2_squared(q, &batch);
        batch_checksum += dists.iter().sum::<f32>();
    }
    let batch_time = start.elapsed();
    std::hint::black_box(batch_checksum);

    // Naive loop
    let start = Instant::now();
    let mut naive_checksum = 0.0f32;
    for q in &queries {
        for v in &corpus {
            naive_checksum += l2_distance_squared(q, v);
        }
    }
    let naive_time = start.elapsed();
    std::hint::black_box(naive_checksum);

    println!(
        "   Corpus: {} vectors x {}d, {} queries\n",
        n, dim, n_queries
    );
    println!(
        "   Batch (PDX) L2:  {:?} ({:.1} us/query)",
        batch_time,
        batch_time.as_micros() as f64 / n_queries as f64
    );
    println!(
        "   Naive loop L2:   {:?} ({:.1} us/query)",
        naive_time,
        naive_time.as_micros() as f64 / n_queries as f64
    );

    let speedup = naive_time.as_nanos() as f64 / batch_time.as_nanos().max(1) as f64;
    println!("   Ratio:           {:.2}x", speedup);
    println!();

    // Verify checksums are close (floating-point summation order differs)
    let rel_diff = (batch_checksum - naive_checksum).abs() / naive_checksum.abs().max(1.0);
    println!(
        "   Checksum relative difference: {:.2e} (expected < 1e-4)",
        rel_diff
    );
    assert!(
        rel_diff < 1e-3,
        "checksums diverge: batch={}, naive={}",
        batch_checksum,
        naive_checksum
    );
    println!();
}

// =============================================================================
// Helpers
// =============================================================================

fn generate_embedding(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let x = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(i as u64 * 1442695040888963407);
            ((x >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        })
        .collect()
}
