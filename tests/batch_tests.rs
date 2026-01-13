//! Integration tests for batch vector operations.
//!
//! Tests the PDX-style columnar layout and batch distance computations.

use innr::batch::{
    batch_cosine, batch_dot, batch_knn, batch_knn_adaptive, batch_l2_squared,
    batch_l2_squared_pruning, batch_norms, VerticalBatch,
};

// =============================================================================
// VerticalBatch construction tests
// =============================================================================

#[test]
fn empty_batch() {
    let vectors: Vec<Vec<f32>> = vec![];
    let batch = VerticalBatch::from_rows(&vectors);

    assert_eq!(batch.num_vectors(), 0);
    assert_eq!(batch.dimension(), 0);
}

#[test]
fn single_vector_batch() {
    let vectors = vec![vec![1.0, 2.0, 3.0, 4.0]];
    let batch = VerticalBatch::from_rows(&vectors);

    assert_eq!(batch.num_vectors(), 1);
    assert_eq!(batch.dimension(), 4);

    for d in 0..4 {
        assert_eq!(batch.get(d, 0), (d + 1) as f32);
    }
}

#[test]
fn from_flat_matches_from_rows() {
    let vectors = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];

    let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

    let batch_rows = VerticalBatch::from_rows(&vectors);
    let batch_flat = VerticalBatch::from_flat(&flat, 3, 3);

    for d in 0..3 {
        for i in 0..3 {
            assert_eq!(
                batch_rows.get(d, i),
                batch_flat.get(d, i),
                "Mismatch at ({}, {})",
                d,
                i
            );
        }
    }
}

#[test]
fn get_unchecked_matches_get() {
    let vectors = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let batch = VerticalBatch::from_rows(&vectors);

    for d in 0..batch.dimension() {
        for i in 0..batch.num_vectors() {
            let safe = batch.get(d, i);
            let unchecked = unsafe { batch.get_unchecked(d, i) };
            assert_eq!(safe, unchecked, "Mismatch at ({}, {})", d, i);
        }
    }
}

#[test]
fn dimension_slice_correct() {
    let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let batch = VerticalBatch::from_rows(&vectors);

    // dimension 0: [1.0, 3.0, 5.0]
    let dim0 = batch.dimension_slice(0);
    assert_eq!(dim0, &[1.0, 3.0, 5.0]);

    // dimension 1: [2.0, 4.0, 6.0]
    let dim1 = batch.dimension_slice(1);
    assert_eq!(dim1, &[2.0, 4.0, 6.0]);
}

#[test]
fn extract_vector_roundtrip() {
    let vectors = vec![
        vec![1.5, 2.5, 3.5],
        vec![4.5, 5.5, 6.5],
        vec![7.5, 8.5, 9.5],
    ];
    let batch = VerticalBatch::from_rows(&vectors);

    for (i, original) in vectors.iter().enumerate() {
        let extracted = batch.extract_vector(i);
        assert_eq!(original, &extracted, "Vector {} mismatch", i);
    }
}

// =============================================================================
// Distance computation tests
// =============================================================================

#[test]
fn l2_squared_identity() {
    let vectors = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let batch = VerticalBatch::from_rows(&vectors);

    // Distance from v0 to itself should be 0
    let distances = batch_l2_squared(&vectors[0], &batch);
    assert!(distances[0].abs() < 1e-6, "Self-distance should be 0");
}

#[test]
fn l2_squared_symmetric() {
    let v1 = vec![1.0, 2.0, 3.0];
    let v2 = vec![4.0, 5.0, 6.0];

    let batch1 = VerticalBatch::from_rows(&[v1.clone()]);
    let batch2 = VerticalBatch::from_rows(&[v2.clone()]);

    let d12 = batch_l2_squared(&v1, &batch2)[0];
    let d21 = batch_l2_squared(&v2, &batch1)[0];

    assert!(
        (d12 - d21).abs() < 1e-6,
        "L2 squared should be symmetric: {} vs {}",
        d12,
        d21
    );
}

#[test]
fn l2_squared_known_value() {
    let vectors = vec![vec![0.0, 0.0, 0.0]];
    let batch = VerticalBatch::from_rows(&vectors);
    let query = vec![3.0, 4.0, 0.0];

    let distances = batch_l2_squared(&query, &batch);
    // Distance should be 5^2 = 25
    assert!(
        (distances[0] - 25.0).abs() < 1e-6,
        "Expected 25, got {}",
        distances[0]
    );
}

#[test]
fn dot_product_orthogonal() {
    let vectors = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let batch = VerticalBatch::from_rows(&vectors);

    // Query along x-axis
    let query = vec![1.0, 0.0, 0.0];
    let dots = batch_dot(&query, &batch);

    assert!((dots[0] - 1.0).abs() < 1e-6, "Parallel should be 1");
    assert!(dots[1].abs() < 1e-6, "Orthogonal should be 0");
    assert!(dots[2].abs() < 1e-6, "Orthogonal should be 0");
}

#[test]
fn cosine_normalized() {
    let vectors = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![-1.0, 0.0],
    ];
    let batch = VerticalBatch::from_rows(&vectors);
    let norms = batch_norms(&batch);
    let query = vec![1.0, 0.0];

    let cosines = batch_cosine(&query, &batch, &norms);

    assert!((cosines[0] - 1.0).abs() < 1e-6, "Parallel: {}", cosines[0]);
    assert!(cosines[1].abs() < 1e-6, "Orthogonal: {}", cosines[1]);
    let sqrt2_inv = 1.0 / 2.0_f32.sqrt();
    assert!(
        (cosines[2] - sqrt2_inv).abs() < 1e-5,
        "45 deg: expected {}, got {}",
        sqrt2_inv,
        cosines[2]
    );
    assert!(
        (cosines[3] - (-1.0)).abs() < 1e-6,
        "Opposite: {}",
        cosines[3]
    );
}

// =============================================================================
// kNN tests
// =============================================================================

#[test]
fn knn_returns_k_results() {
    let vectors: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32, 0.0]).collect();
    let batch = VerticalBatch::from_rows(&vectors);
    let query = vec![50.0, 0.0];

    for k in [1, 5, 10, 50, 100] {
        let result = batch_knn(&query, &batch, k);
        assert_eq!(result.indices.len(), k, "Should return {} results", k);
        assert_eq!(result.distances.len(), k);
    }
}

#[test]
fn knn_results_sorted() {
    let vectors: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32, (i as f32).sin()]).collect();
    let batch = VerticalBatch::from_rows(&vectors);
    let query = vec![25.0, 0.0];

    let result = batch_knn(&query, &batch, 20);

    for i in 1..result.distances.len() {
        assert!(
            result.distances[i] >= result.distances[i - 1],
            "Results not sorted: {} > {} at position {}",
            result.distances[i - 1],
            result.distances[i],
            i
        );
    }
}

#[test]
fn knn_finds_exact_match() {
    let vectors = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let batch = VerticalBatch::from_rows(&vectors);

    // Query exactly matches vector 2
    let query = vec![0.0, 1.0];
    let result = batch_knn(&query, &batch, 1);

    assert_eq!(result.indices[0], 2);
    assert!(result.distances[0] < 1e-6);
}

#[test]
fn knn_adaptive_matches_basic() {
    let vectors: Vec<Vec<f32>> = (0..100)
        .map(|i| vec![i as f32, (i as f32 * 0.1).sin(), (i as f32 * 0.1).cos()])
        .collect();
    let batch = VerticalBatch::from_rows(&vectors);
    let query = vec![50.0, 0.0, 1.0];

    let basic = batch_knn(&query, &batch, 10);
    let adaptive = batch_knn_adaptive(&query, &batch, 10, 1);

    // Adaptive should find the same or very similar results
    // (may differ slightly due to pruning estimation)
    for idx in &basic.indices[..5] {
        assert!(
            adaptive.indices.contains(idx),
            "Adaptive missing index {} from basic top-5",
            idx
        );
    }
}

// =============================================================================
// Pruning tests
// =============================================================================

#[test]
fn pruning_filters_far_vectors() {
    let vectors = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![100.0, 100.0], // Very far
        vec![2.0, 0.0],
    ];
    let batch = VerticalBatch::from_rows(&vectors);
    let query = vec![0.0, 0.0];

    let survivors = batch_l2_squared_pruning(&query, &batch, 5.0);

    let indices: Vec<usize> = survivors.iter().map(|(i, _)| *i).collect();
    assert!(indices.contains(&0), "Origin should survive");
    assert!(indices.contains(&1), "Close vector should survive");
    assert!(!indices.contains(&2), "Far vector should be pruned");
    assert!(indices.contains(&3), "Close vector should survive");
}

#[test]
fn pruning_returns_correct_distances() {
    let vectors = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 2.0]];
    let batch = VerticalBatch::from_rows(&vectors);
    let query = vec![0.0, 0.0];

    let survivors = batch_l2_squared_pruning(&query, &batch, 5.0);

    for (idx, dist) in &survivors {
        let expected = batch_l2_squared(&query, &batch)[*idx];
        assert!(
            (dist - expected).abs() < 1e-6,
            "Distance mismatch for idx {}: {} vs {}",
            idx,
            dist,
            expected
        );
    }
}

#[test]
fn pruning_tight_threshold() {
    let vectors: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32, 0.0]).collect();
    let batch = VerticalBatch::from_rows(&vectors);
    let query = vec![50.0, 0.0];

    // Only vectors within distance 2 (squared: 4)
    let survivors = batch_l2_squared_pruning(&query, &batch, 4.0);

    // Should only include 48, 49, 50, 51, 52
    let indices: Vec<usize> = survivors.iter().map(|(i, _)| *i).collect();
    assert!(indices.len() <= 5, "Should only have ~5 survivors");
    assert!(indices.contains(&50), "Exact match should survive");
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
fn knn_k_larger_than_batch() {
    let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let batch = VerticalBatch::from_rows(&vectors);
    let query = vec![0.0, 0.0];

    let result = batch_knn(&query, &batch, 100);

    assert_eq!(result.indices.len(), 2, "Should return all vectors");
}

#[test]
fn knn_k_zero() {
    let vectors = vec![vec![1.0, 2.0]];
    let batch = VerticalBatch::from_rows(&vectors);
    let query = vec![0.0, 0.0];

    let result = batch_knn(&query, &batch, 0);

    assert!(result.indices.is_empty());
    assert!(result.distances.is_empty());
}

#[test]
fn batch_norms_correct() {
    let vectors = vec![
        vec![3.0, 4.0], // norm = 5
        vec![1.0, 0.0], // norm = 1
        vec![0.0, 0.0], // norm = 0
    ];
    let batch = VerticalBatch::from_rows(&vectors);
    let norms = batch_norms(&batch);

    assert!((norms[0] - 5.0).abs() < 1e-6);
    assert!((norms[1] - 1.0).abs() < 1e-6);
    assert!(norms[2].abs() < 1e-6);
}

#[test]
fn cosine_with_zero_norm() {
    let vectors = vec![
        vec![1.0, 0.0],
        vec![0.0, 0.0], // Zero vector
    ];
    let batch = VerticalBatch::from_rows(&vectors);
    let norms = batch_norms(&batch);
    let query = vec![1.0, 0.0];

    let cosines = batch_cosine(&query, &batch, &norms);

    assert!((cosines[0] - 1.0).abs() < 1e-6);
    assert!(cosines[1].abs() < 1e-6, "Zero vector should give 0 cosine");
}

#[test]
fn cosine_with_zero_query() {
    let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let batch = VerticalBatch::from_rows(&vectors);
    let norms = batch_norms(&batch);
    let query = vec![0.0, 0.0]; // Zero query

    let cosines = batch_cosine(&query, &batch, &norms);

    for &c in &cosines {
        assert!(c.abs() < 1e-6, "Zero query should give 0 cosine");
    }
}
