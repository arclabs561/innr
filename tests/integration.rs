//! Integration tests verifying `innr` works correctly across the mathematical foundation.
//!
//! These tests ensure that the SIMD primitives produce consistent results
//! when used by different downstream crates.

use std::time::Instant;

/// Test that basic operations work correctly on realistic embeddings.
#[test]
fn test_realistic_embedding_dimensions() {
    // Common embedding dimensions in production systems
    let dims = [64, 128, 256, 384, 512, 768, 1024, 1536];

    for dim in dims {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.002).cos()).collect();

        // Dot product should be finite
        let dot = innr::dot(&a, &b);
        assert!(
            dot.is_finite(),
            "Dot product not finite for dim={}: {}",
            dim,
            dot
        );

        // Cosine should be in [-1, 1]
        let cos = innr::cosine(&a, &b);
        assert!(
            (-1.0..=1.0).contains(&cos) || (cos - 1.0).abs() < 1e-5 || (cos + 1.0).abs() < 1e-5,
            "Cosine out of range for dim={}: {}",
            dim,
            cos
        );

        // L2 distance should be non-negative
        let l2 = innr::l2_distance_squared(&a, &b);
        assert!(
            l2 >= 0.0,
            "L2 squared should be non-negative for dim={}: {}",
            dim,
            l2
        );

        // Norm should be positive for non-zero vectors
        let norm_a = innr::norm(&a);
        assert!(
            norm_a > 0.0,
            "Norm should be positive for dim={}: {}",
            dim,
            norm_a
        );
    }
}

/// Test MaxSim behavior with ColBERT-style token embeddings.
#[cfg(feature = "maxsim")]
#[test]
fn test_colbert_style_maxsim() {
    // Simulate ColBERT: query has few tokens, document has many
    let query_len = 8; // typical query token count
    let doc_len = 128; // typical document token count
    let dim = 128; // ColBERT dimension

    let query_tokens: Vec<Vec<f32>> = (0..query_len)
        .map(|i| {
            (0..dim)
                .map(|j| ((i * dim + j) as f32 * 0.01).sin())
                .collect()
        })
        .collect();

    let doc_tokens: Vec<Vec<f32>> = (0..doc_len)
        .map(|i| {
            (0..dim)
                .map(|j| ((i * dim + j) as f32 * 0.01).cos())
                .collect()
        })
        .collect();

    let q_refs: Vec<&[f32]> = query_tokens.iter().map(|v| v.as_slice()).collect();
    let d_refs: Vec<&[f32]> = doc_tokens.iter().map(|v| v.as_slice()).collect();

    // MaxSim should be sum of max similarities
    let score = innr::maxsim(&q_refs, &d_refs);
    assert!(score.is_finite(), "MaxSim should be finite: {}", score);

    // MaxSim-cosine should be bounded by query token count
    let cos_score = innr::maxsim_cosine(&q_refs, &d_refs);
    assert!(
        cos_score <= query_len as f32 + 0.01,
        "MaxSim-cosine {} should be bounded by query tokens {}",
        cos_score,
        query_len
    );
}

/// Test that SIMD operations have reasonable performance.
#[test]
fn test_performance_sanity() {
    let dim = 768;
    let iterations = 10_000;

    let a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.001).collect();
    let b: Vec<f32> = (0..dim).map(|i| (dim - i) as f32 * 0.001).collect();

    let start = Instant::now();
    let mut sum = 0.0f32;
    for _ in 0..iterations {
        sum += innr::dot(&a, &b);
    }
    let elapsed = start.elapsed();

    // Sanity check: 10k dot products of 768-dim vectors should be fast
    // Even on slow hardware, this should complete in < 1 second
    assert!(
        elapsed.as_millis() < 1000,
        "10k dot products took too long: {:?}",
        elapsed
    );

    // Prevent optimization
    assert!(sum.is_finite());
}

/// Test edge cases that might cause issues in production.
#[test]
fn test_edge_cases() {
    // Zero vector handling
    let zero = vec![0.0f32; 128];
    let nonzero: Vec<f32> = (0..128).map(|i| i as f32).collect();

    assert_eq!(innr::cosine(&zero, &nonzero), 0.0);
    assert_eq!(innr::cosine(&nonzero, &zero), 0.0);
    assert_eq!(innr::cosine(&zero, &zero), 0.0);

    // Very small values (near subnormal)
    let tiny: Vec<f32> = vec![1e-38; 128];
    let result = innr::dot(&tiny, &tiny);
    assert!(result.is_finite());

    // Very large values
    let large: Vec<f32> = vec![1e18; 128];
    let result = innr::dot(&large, &large);
    // May overflow, but should not panic
    let _ = result;

    // Mixed positive/negative
    let mixed: Vec<f32> = (0..128)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let result = innr::dot(&mixed, &mixed);
    assert!((result - 128.0).abs() < 0.01);
}

/// Test that operations are deterministic.
#[test]
fn test_determinism() {
    let a: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..1024).map(|i| (i as f32).cos()).collect();

    // Same inputs should always produce same outputs
    let dot1 = innr::dot(&a, &b);
    let dot2 = innr::dot(&a, &b);
    assert_eq!(dot1, dot2, "Dot product should be deterministic");

    let cos1 = innr::cosine(&a, &b);
    let cos2 = innr::cosine(&a, &b);
    assert_eq!(cos1, cos2, "Cosine should be deterministic");

    let l2_1 = innr::l2_distance_squared(&a, &b);
    let l2_2 = innr::l2_distance_squared(&a, &b);
    assert_eq!(l2_1, l2_2, "L2 distance should be deterministic");
}
