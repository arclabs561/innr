//! Property-based tests for SIMD correctness.
//!
//! These tests verify that SIMD implementations produce identical results
//! to portable implementations across various input sizes and value ranges.

use proptest::prelude::*;

// Reference portable implementations for comparison
fn dot_reference(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_sq_reference(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn norm_reference(v: &[f32]) -> f32 {
    dot_reference(v, v).sqrt()
}

/// Generate vectors of specific sizes to test SIMD boundaries.
fn arb_vec_pair(len: usize) -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    (
        proptest::collection::vec(-100.0f32..100.0, len),
        proptest::collection::vec(-100.0f32..100.0, len),
    )
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 500,
        ..ProptestConfig::default()
    })]

    // ─────────────────────────────────────────────────────────────────────────
    // Dot product tests
    // ─────────────────────────────────────────────────────────────────────────

    /// Dot product matches reference for small vectors (below SIMD threshold).
    #[test]
    fn dot_small_matches_reference((a, b) in arb_vec_pair(8)) {
        let result = innr::dot(&a, &b);
        let expected = dot_reference(&a, &b);
        let tolerance = expected.abs() * 1e-5 + 1e-6;
        prop_assert!(
            (result - expected).abs() < tolerance,
            "Small dot mismatch: {} vs {} (tolerance: {})",
            result, expected, tolerance
        );
    }

    /// Dot product matches reference for medium vectors (SIMD active).
    ///
    /// Note: SIMD implementations may accumulate in different orders than
    /// sequential code, leading to different rounding errors.
    #[test]
    fn dot_medium_matches_reference((a, b) in arb_vec_pair(64)) {
        let result = innr::dot(&a, &b);
        let expected = dot_reference(&a, &b);
        // Tolerance scales with sum of |products| to handle cancellation
        let sum_abs_products: f32 = a.iter().zip(b.iter())
            .map(|(x, y)| (x * y).abs())
            .sum();
        let tolerance = sum_abs_products * 1e-5 + 1e-3;
        prop_assert!(
            (result - expected).abs() < tolerance,
            "Medium dot mismatch: {} vs {} (diff: {}, tol: {})",
            result, expected, (result - expected).abs(), tolerance
        );
    }

    /// Dot product matches reference for large vectors.
    ///
    /// With 256 elements, accumulation order differences become more significant.
    /// The tolerance must account for accumulated error across many operations.
    /// With values in [-100, 100] and 256 elements, max possible |dot| is ~2.56M.
    /// We use an absolute tolerance that scales with the magnitude of inputs.
    #[test]
    fn dot_large_matches_reference((a, b) in arb_vec_pair(256)) {
        let result = innr::dot(&a, &b);
        let expected = dot_reference(&a, &b);
        // Estimate the magnitude of intermediate products
        let sum_abs_products: f32 = a.iter().zip(b.iter())
            .map(|(x, y)| (x * y).abs())
            .sum();
        // Tolerance scales with sum of |products|, not the potentially small |expected|
        let tolerance = sum_abs_products * 1e-5 + 1e-2;
        prop_assert!(
            (result - expected).abs() < tolerance,
            "Large dot mismatch: {} vs {} (diff: {}, tol: {})",
            result, expected, (result - expected).abs(), tolerance
        );
    }

    /// Dot product is commutative.
    #[test]
    fn dot_commutative((a, b) in arb_vec_pair(128)) {
        let ab = innr::dot(&a, &b);
        let ba = innr::dot(&b, &a);
        prop_assert!(
            (ab - ba).abs() < 1e-6,
            "Dot not commutative: {} != {}",
            ab, ba
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // L2 distance squared tests
    // ─────────────────────────────────────────────────────────────────────────

    /// L2 squared matches reference for small vectors.
    #[test]
    fn l2_sq_small_matches_reference((a, b) in arb_vec_pair(8)) {
        let result = innr::l2_distance_squared(&a, &b);
        let expected = l2_sq_reference(&a, &b);
        let tolerance = expected.abs() * 1e-5 + 1e-6;
        prop_assert!(
            (result - expected).abs() < tolerance,
            "Small L2sq mismatch: {} vs {}",
            result, expected
        );
    }

    /// L2 squared matches reference for medium vectors.
    #[test]
    fn l2_sq_medium_matches_reference((a, b) in arb_vec_pair(64)) {
        let result = innr::l2_distance_squared(&a, &b);
        let expected = l2_sq_reference(&a, &b);
        let tolerance = expected.abs() * 1e-4 + 1e-5;
        prop_assert!(
            (result - expected).abs() < tolerance,
            "Medium L2sq mismatch: {} vs {}",
            result, expected
        );
    }

    /// L2 squared matches reference for large vectors.
    #[test]
    fn l2_sq_large_matches_reference((a, b) in arb_vec_pair(256)) {
        let result = innr::l2_distance_squared(&a, &b);
        let expected = l2_sq_reference(&a, &b);
        let tolerance = expected.abs() * 1e-4 + 1e-4;
        prop_assert!(
            (result - expected).abs() < tolerance,
            "Large L2sq mismatch: {} vs {}",
            result, expected
        );
    }

    /// L2 squared is symmetric.
    #[test]
    fn l2_sq_symmetric((a, b) in arb_vec_pair(128)) {
        let ab = innr::l2_distance_squared(&a, &b);
        let ba = innr::l2_distance_squared(&b, &a);
        prop_assert!(
            (ab - ba).abs() < 1e-6,
            "L2sq not symmetric: {} != {}",
            ab, ba
        );
    }

    /// L2 squared is non-negative.
    #[test]
    fn l2_sq_nonnegative((a, b) in arb_vec_pair(128)) {
        let result = innr::l2_distance_squared(&a, &b);
        prop_assert!(
            result >= 0.0,
            "L2sq should be non-negative, got {}",
            result
        );
    }

    /// L2 squared to self is zero.
    #[test]
    fn l2_sq_self_is_zero(v in proptest::collection::vec(-100.0f32..100.0, 128)) {
        let result = innr::l2_distance_squared(&v, &v);
        prop_assert!(
            result.abs() < 1e-6,
            "L2sq to self should be 0, got {}",
            result
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Cosine similarity tests
    // ─────────────────────────────────────────────────────────────────────────

    /// Cosine similarity is bounded [-1, 1].
    #[test]
    fn cosine_bounded(
        (a, b) in arb_vec_pair(128).prop_filter("non-zero", |(a, b)| {
            a.iter().any(|x| x.abs() > 1e-6) && b.iter().any(|x| x.abs() > 1e-6)
        })
    ) {
        let result = innr::cosine(&a, &b);
        prop_assert!(
            (-1.0 - 1e-5..=1.0 + 1e-5).contains(&result),
            "Cosine out of bounds: {}",
            result
        );
    }

    /// Cosine similarity is symmetric.
    #[test]
    fn cosine_symmetric((a, b) in arb_vec_pair(128)) {
        let ab = innr::cosine(&a, &b);
        let ba = innr::cosine(&b, &a);
        prop_assert!(
            (ab - ba).abs() < 1e-5,
            "Cosine not symmetric: {} != {}",
            ab, ba
        );
    }

    /// Cosine of vector with itself is 1.
    #[test]
    fn cosine_self_is_one(
        v in proptest::collection::vec(-100.0f32..100.0, 128)
            .prop_filter("non-zero", |v| v.iter().any(|x| x.abs() > 1e-6))
    ) {
        let result = innr::cosine(&v, &v);
        prop_assert!(
            (result - 1.0).abs() < 1e-5,
            "Cosine of self should be 1.0, got {}",
            result
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Norm tests
    // ─────────────────────────────────────────────────────────────────────────

    /// Norm is non-negative.
    #[test]
    fn norm_nonnegative(v in proptest::collection::vec(-100.0f32..100.0, 128)) {
        let result = innr::norm(&v);
        prop_assert!(result >= 0.0, "Norm should be non-negative, got {}", result);
    }

    /// Norm matches reference.
    #[test]
    fn norm_matches_reference(v in proptest::collection::vec(-100.0f32..100.0, 128)) {
        let result = innr::norm(&v);
        let expected = norm_reference(&v);
        let tolerance = expected.abs() * 1e-5 + 1e-6;
        prop_assert!(
            (result - expected).abs() < tolerance,
            "Norm mismatch: {} vs {}",
            result, expected
        );
    }

    /// Norm scales linearly with scalar multiplication.
    #[test]
    fn norm_scales_with_scalar(
        v in proptest::collection::vec(-10.0f32..10.0, 64),
        scale in 0.1f32..10.0
    ) {
        let scaled: Vec<f32> = v.iter().map(|x| x * scale).collect();
        let norm_v = innr::norm(&v);
        let norm_scaled = innr::norm(&scaled);
        let expected = norm_v * scale;
        let tolerance = expected.abs() * 1e-4 + 1e-5;
        prop_assert!(
            (norm_scaled - expected).abs() < tolerance,
            "Norm scaling violated: {} != {} * {} = {}",
            norm_scaled, norm_v, scale, expected
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD boundary tests (specific sizes that stress SIMD implementations)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_dot_at_simd_boundaries() {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Test sizes that are: exact SIMD width, SIMD width - 1, SIMD width + 1
    // AVX-512: 16 floats, AVX2: 8 floats, NEON: 4 floats
    let sizes = [
        1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 47, 48, 49, 63, 64, 65, 127, 128,
        129, 255, 256, 257,
    ];

    for &size in &sizes {
        let a: Vec<f32> = (0..size).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let b: Vec<f32> = (0..size).map(|_| rng.gen_range(-10.0..10.0)).collect();

        let result = innr::dot(&a, &b);
        let expected = dot_reference(&a, &b);

        let tolerance = expected.abs() * 1e-4 + 1e-5;
        assert!(
            (result - expected).abs() < tolerance,
            "Dot at size {}: {} vs {} (diff: {})",
            size,
            result,
            expected,
            (result - expected).abs()
        );
    }
}

#[test]
fn test_l2_sq_at_simd_boundaries() {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let sizes = [
        1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 47, 48, 49, 63, 64, 65, 127, 128,
        129, 255, 256, 257,
    ];

    for &size in &sizes {
        let a: Vec<f32> = (0..size).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let b: Vec<f32> = (0..size).map(|_| rng.gen_range(-10.0..10.0)).collect();

        let result = innr::l2_distance_squared(&a, &b);
        let expected = l2_sq_reference(&a, &b);

        let tolerance = expected.abs() * 1e-4 + 1e-5;
        assert!(
            (result - expected).abs() < tolerance,
            "L2sq at size {}: {} vs {} (diff: {})",
            size,
            result,
            expected,
            (result - expected).abs()
        );
    }
}

// =============================================================================
// Batch operation property tests
// =============================================================================

mod batch_props {
    use super::*;
    use innr::batch::{
        batch_cosine, batch_dot, batch_knn, batch_l2_squared, batch_norms, VerticalBatch,
    };

    /// Generate a batch of vectors.
    fn arb_batch(
        num_vectors: usize,
        dim: usize,
    ) -> impl Strategy<Value = (Vec<Vec<f32>>, Vec<f32>)> {
        (
            proptest::collection::vec(
                proptest::collection::vec(-10.0f32..10.0, dim),
                num_vectors,
            ),
            proptest::collection::vec(-10.0f32..10.0, dim),
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// Batch L2 squared matches individual computations.
        #[test]
        fn batch_l2_matches_individual((vectors, query) in arb_batch(20, 32)) {
            let batch = VerticalBatch::from_rows(&vectors);
            let batch_results = batch_l2_squared(&query, &batch);

            for (i, vec) in vectors.iter().enumerate() {
                let individual = innr::l2_distance_squared(&query, vec);
                let tolerance = individual.abs() * 1e-4 + 1e-5;
                prop_assert!(
                    (batch_results[i] - individual).abs() < tolerance,
                    "Batch L2 mismatch at {}: {} vs {}",
                    i, batch_results[i], individual
                );
            }
        }

        /// Batch dot matches individual computations.
        /// Note: Batch and individual may accumulate in different orders,
        /// leading to small floating-point differences.
        #[test]
        fn batch_dot_matches_individual((vectors, query) in arb_batch(20, 32)) {
            let batch = VerticalBatch::from_rows(&vectors);
            let batch_results = batch_dot(&query, &batch);

            for (i, vec) in vectors.iter().enumerate() {
                let individual = innr::dot(&query, vec);
                // Tolerance accounts for accumulation order differences
                let sum_abs: f32 = query.iter().zip(vec.iter())
                    .map(|(a, b)| (a * b).abs())
                    .sum();
                let tolerance = sum_abs * 1e-4 + 1e-4;
                prop_assert!(
                    (batch_results[i] - individual).abs() < tolerance,
                    "Batch dot mismatch at {}: {} vs {} (tol: {})",
                    i, batch_results[i], individual, tolerance
                );
            }
        }

        /// Batch norms are non-negative.
        #[test]
        fn batch_norms_nonnegative((vectors, _) in arb_batch(20, 32)) {
            let batch = VerticalBatch::from_rows(&vectors);
            let norms = batch_norms(&batch);

            for (i, &n) in norms.iter().enumerate() {
                prop_assert!(n >= 0.0, "Norm at {} should be non-negative: {}", i, n);
            }
        }

        /// Batch cosine is bounded [-1, 1].
        #[test]
        fn batch_cosine_bounded(
            (vectors, query) in arb_batch(20, 32).prop_filter("non-zero", |(vecs, q)| {
                q.iter().any(|x| x.abs() > 1e-6) &&
                vecs.iter().all(|v| v.iter().any(|x| x.abs() > 1e-6))
            })
        ) {
            let batch = VerticalBatch::from_rows(&vectors);
            let norms = batch_norms(&batch);
            let cosines = batch_cosine(&query, &batch, &norms);

            for (i, &c) in cosines.iter().enumerate() {
                prop_assert!(
                    (-1.0 - 1e-4..=1.0 + 1e-4).contains(&c),
                    "Cosine at {} out of bounds: {}",
                    i, c
                );
            }
        }

        /// kNN returns sorted results.
        #[test]
        fn batch_knn_sorted((vectors, query) in arb_batch(50, 16)) {
            let batch = VerticalBatch::from_rows(&vectors);
            let result = batch_knn(&query, &batch, 10);

            for i in 1..result.distances.len() {
                prop_assert!(
                    result.distances[i] >= result.distances[i - 1] - 1e-6,
                    "kNN not sorted: {} > {} at {}",
                    result.distances[i - 1], result.distances[i], i
                );
            }
        }

        /// kNN indices are unique.
        #[test]
        fn batch_knn_unique_indices((vectors, query) in arb_batch(50, 16)) {
            let batch = VerticalBatch::from_rows(&vectors);
            let result = batch_knn(&query, &batch, 20);

            let unique: std::collections::HashSet<_> = result.indices.iter().collect();
            prop_assert_eq!(
                unique.len(),
                result.indices.len(),
                "kNN indices not unique"
            );
        }

        /// Vertical batch roundtrip preserves data.
        #[test]
        fn batch_roundtrip((vectors, _) in arb_batch(10, 16)) {
            let batch = VerticalBatch::from_rows(&vectors);

            for (i, original) in vectors.iter().enumerate() {
                let extracted = batch.extract_vector(i);
                for (j, (&orig, &ext)) in original.iter().zip(extracted.iter()).enumerate() {
                    prop_assert!(
                        (orig - ext).abs() < 1e-6,
                        "Roundtrip mismatch at [{}, {}]: {} vs {}",
                        i, j, orig, ext
                    );
                }
            }
        }
    }
}

// =============================================================================
// Triangle inequality tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// L2 distance satisfies triangle inequality.
    #[test]
    fn l2_triangle_inequality(
        a in proptest::collection::vec(-10.0f32..10.0, 32),
        b in proptest::collection::vec(-10.0f32..10.0, 32),
        c in proptest::collection::vec(-10.0f32..10.0, 32),
    ) {
        let ab = innr::l2_distance(&a, &b);
        let bc = innr::l2_distance(&b, &c);
        let ac = innr::l2_distance(&a, &c);

        // ac <= ab + bc (with tolerance for floating point)
        prop_assert!(
            ac <= ab + bc + 1e-4,
            "Triangle inequality violated: {} > {} + {} = {}",
            ac, ab, bc, ab + bc
        );
    }
}
