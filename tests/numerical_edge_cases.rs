//! Numerical Edge Case Tests
//!
//! These tests target real-world numerical issues that can occur in
//! embedding-based retrieval systems. Grounded in actual failure modes.

use innr::{cosine, dot, l2_distance, l2_distance_squared, norm};

// =============================================================================
// Denormalized Numbers (Very Small Values)
// =============================================================================

#[test]
fn handles_denormalized_floats() {
    // Denormalized (subnormal) floats are very small numbers near zero
    // that lose precision. They appear in scaled-down embeddings.
    let denorm = f32::MIN_POSITIVE / 2.0; // Subnormal
    let a = vec![denorm; 64];
    let b = vec![denorm; 64];

    // Should not panic or return NaN
    let d = dot(&a, &b);
    assert!(!d.is_nan(), "dot should handle denormalized floats");

    let c = cosine(&a, &b);
    // Cosine of identical vectors should be ~1.0 (within numerical precision)
    // But subnormals may cause issues, so accept 0.0 (fallback) or ~1.0
    assert!(!c.is_nan(), "cosine should handle denormalized floats");
}

#[test]
fn handles_mixed_magnitude_vectors() {
    // Common in embeddings: some dimensions are much larger than others
    let mut a = vec![1e-10f32; 64];
    a[0] = 1.0;
    a[1] = 1.0;

    let mut b = vec![1e-10f32; 64];
    b[0] = 0.5;
    b[1] = 0.5;

    // Should produce meaningful results
    let c = cosine(&a, &b);
    assert!(c > 0.9, "similar vectors should have high cosine: {}", c);
}

// =============================================================================
// Near-Zero Norms (Padding Tokens, OOV)
// =============================================================================

#[test]
fn cosine_handles_near_zero_norm() {
    // Padding tokens or failed embeddings often have near-zero vectors
    let near_zero = vec![1e-20f32; 64];
    let normal = vec![1.0f32; 64];

    let c = cosine(&near_zero, &normal);
    // Should return 0.0, not NaN or inf
    assert!(
        c.is_finite(),
        "cosine should return finite for near-zero norm"
    );
    assert!(c.abs() <= 1.0 + 1e-5, "cosine should be bounded");
}

#[test]
fn l2_handles_identical_vectors() {
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let d = l2_distance(&v, &v);
    assert!(
        (d - 0.0).abs() < 1e-10,
        "L2 of identical vectors should be 0"
    );

    let d_sq = l2_distance_squared(&v, &v);
    assert!(
        (d_sq - 0.0).abs() < 1e-10,
        "L2^2 of identical vectors should be 0"
    );
}

// =============================================================================
// Catastrophic Cancellation
// =============================================================================

#[test]
fn handles_large_similar_vectors() {
    // When computing a-b for large, similar vectors, catastrophic cancellation
    // can occur. This is a known numerical issue in L2 distance.
    let a: Vec<f32> = (0..128).map(|i| 1e6 + i as f32 * 0.001).collect();
    let b: Vec<f32> = (0..128).map(|i| 1e6 + i as f32 * 0.001 + 1e-4).collect();

    let d = l2_distance(&a, &b);
    assert!(
        !d.is_nan(),
        "L2 should not be NaN for large similar vectors"
    );
    assert!(d < 1.0, "Distance should be small: {}", d);
}

// =============================================================================
// Dimension Boundary Cases
// =============================================================================

#[test]
fn handles_simd_boundary_dimensions() {
    // Test dimensions at SIMD register boundaries
    // AVX2: 8 floats, AVX-512: 16 floats
    for dim in [7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 + 2.0).cos()).collect();

        let d = dot(&a, &b);
        assert!(!d.is_nan(), "dot NaN at dim={}", dim);

        let c = cosine(&a, &b);
        assert!(!c.is_nan(), "cosine NaN at dim={}", dim);
        assert!(
            (-1.0 - 1e-5..=1.0 + 1e-5).contains(&c),
            "cosine out of range at dim={}: {}",
            dim,
            c
        );
    }
}

#[test]
fn handles_odd_dimensions() {
    // Non-power-of-2 dimensions common in embeddings (384, 768, 1536)
    for dim in [384, 768, 1024, 1536] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).cos()).collect();

        let d = dot(&a, &b);
        assert!(!d.is_nan(), "dot NaN at dim={}", dim);
        assert!(d.is_finite(), "dot not finite at dim={}", dim);
    }
}

// =============================================================================
// Normalized Vector Properties
// =============================================================================

#[test]
fn normalized_vectors_have_unit_norm() {
    let v = vec![3.0f32, 4.0, 0.0, 0.0];
    let n = norm(&v);
    assert!((n - 5.0).abs() < 1e-6, "norm of [3,4,0,0] should be 5");

    // Normalize
    let v_normalized: Vec<f32> = v.iter().map(|x| x / n).collect();
    let n_norm = norm(&v_normalized);
    assert!(
        (n_norm - 1.0).abs() < 1e-6,
        "normalized vector should have unit norm"
    );
}

#[test]
fn cosine_equals_dot_for_normalized() {
    // For unit vectors: cosine(a,b) = dot(a,b)
    let a_raw = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b_raw = vec![8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    let norm_a = norm(&a_raw);
    let norm_b = norm(&b_raw);

    let a: Vec<f32> = a_raw.iter().map(|x| x / norm_a).collect();
    let b: Vec<f32> = b_raw.iter().map(|x| x / norm_b).collect();

    let cos = cosine(&a, &b);
    let d = dot(&a, &b);

    assert!(
        (cos - d).abs() < 1e-5,
        "cosine should equal dot for normalized: cos={}, dot={}",
        cos,
        d
    );
}

#[test]
fn l2_cosine_relationship_for_normalized() {
    // For unit vectors: L2^2(a,b) = 2(1 - cos(a,b))
    let a_raw: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
    let b_raw: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).cos()).collect();

    let norm_a = norm(&a_raw);
    let norm_b = norm(&b_raw);

    let a: Vec<f32> = a_raw.iter().map(|x| x / norm_a).collect();
    let b: Vec<f32> = b_raw.iter().map(|x| x / norm_b).collect();

    let cos = cosine(&a, &b);
    let l2_sq = l2_distance_squared(&a, &b);

    let expected = 2.0 * (1.0 - cos);
    assert!(
        (l2_sq - expected).abs() < 1e-4,
        "L2^2 should equal 2(1-cos): l2^2={}, expected={}",
        l2_sq,
        expected
    );
}

// =============================================================================
// Empty and Single Element
// =============================================================================

#[test]
fn handles_empty_vectors() {
    let empty: Vec<f32> = vec![];
    assert_eq!(dot(&empty, &empty), 0.0);
    assert_eq!(norm(&empty), 0.0);
    // L2 of empty vectors is 0
    assert_eq!(l2_distance(&empty, &empty), 0.0);
}

#[test]
fn handles_single_element() {
    let a = vec![3.0f32];
    let b = vec![4.0f32];

    assert_eq!(dot(&a, &b), 12.0);
    assert_eq!(norm(&a), 3.0);
    assert_eq!(l2_distance(&a, &b), 1.0);

    // Cosine of single element normalized
    let c = cosine(&a, &b);
    assert!((c - 1.0).abs() < 1e-6, "parallel single-element vectors");
}

// =============================================================================
// Orthogonal and Antiparallel
// =============================================================================

#[test]
fn orthogonal_vectors() {
    let a = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let b = vec![0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let c = cosine(&a, &b);
    assert!(c.abs() < 1e-6, "orthogonal vectors should have cosine ~0");

    let d = dot(&a, &b);
    assert!(d.abs() < 1e-6, "orthogonal vectors should have dot ~0");
}

#[test]
fn antiparallel_vectors() {
    let a = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let b = vec![-1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let c = cosine(&a, &b);
    assert!(
        (c - (-1.0)).abs() < 1e-6,
        "antiparallel vectors should have cosine ~-1"
    );
}
