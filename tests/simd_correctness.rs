//! SIMD Correctness Tests: Differential Testing Approach
//!
//! Strategy: Model each SIMD operation in pure scalar Rust, then verify
//! the SIMD implementation matches across random inputs.
//!
//! Based on Cryspen's approach: https://cryspen.com/post/specify-rust-simd/
//! They verified 384 x86 intrinsics + 181 AArch64 intrinsics this way.

#![allow(clippy::float_cmp)]

use innr::{cosine, dot, l2_distance, l2_distance_squared};

// =============================================================================
// Reference Implementations (Pure Scalar)
// =============================================================================

/// Reference dot product - simple, obviously correct
fn ref_dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Reference L2 squared distance
fn ref_l2_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// Reference L2 distance
fn ref_l2(a: &[f32], b: &[f32]) -> f32 {
    ref_l2_squared(a, b).sqrt()
}

/// Reference cosine similarity
fn ref_cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot = ref_dot(a, b);
    let norm_a = ref_l2_squared(a, &vec![0.0; a.len()]).sqrt();
    let norm_b = ref_l2_squared(b, &vec![0.0; b.len()]).sqrt();
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

// =============================================================================
// Test Helpers
// =============================================================================

/// Generate deterministic test vectors
fn test_vec(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let x = (seed.wrapping_mul(31).wrapping_add(i as u64 * 17)) as f32;
            (x * 0.001).sin()
        })
        .collect()
}

/// Check if two floats are approximately equal
fn approx_eq(a: f32, b: f32, rel_eps: f32) -> bool {
    let diff = (a - b).abs();
    let max_val = a.abs().max(b.abs()).max(1.0);
    diff < rel_eps * max_val
}

// =============================================================================
// Differential Tests
// =============================================================================

#[test]
fn simd_correctness_dot_small_dims() {
    // Test dimensions that exercise different SIMD code paths
    // AVX2: 8 floats/register, AVX-512: 16 floats/register
    for dim in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
        for seed in 0..10 {
            let a = test_vec(dim, seed);
            let b = test_vec(dim, seed + 1000);

            let simd_result = dot(&a, &b);
            let ref_result = ref_dot(&a, &b);

            assert!(
                approx_eq(simd_result, ref_result, 1e-5),
                "dot mismatch at dim={}, seed={}: simd={}, ref={}",
                dim,
                seed,
                simd_result,
                ref_result
            );
        }
    }
}

#[test]
fn simd_correctness_dot_large_dims() {
    // Typical embedding dimensions
    for dim in [64, 128, 256, 384, 512, 768, 1024, 1536] {
        for seed in 0..5 {
            let a = test_vec(dim, seed);
            let b = test_vec(dim, seed + 1000);

            let simd_result = dot(&a, &b);
            let ref_result = ref_dot(&a, &b);

            assert!(
                approx_eq(simd_result, ref_result, 1e-4),
                "dot mismatch at dim={}: simd={}, ref={}",
                dim,
                simd_result,
                ref_result
            );
        }
    }
}

#[test]
fn simd_correctness_l2_squared() {
    for dim in [1, 7, 8, 15, 16, 31, 32, 64, 128, 384, 768] {
        for seed in 0..5 {
            let a = test_vec(dim, seed);
            let b = test_vec(dim, seed + 1000);

            let simd_result = l2_distance_squared(&a, &b);
            let ref_result = ref_l2_squared(&a, &b);

            assert!(
                approx_eq(simd_result, ref_result, 1e-4),
                "l2_squared mismatch at dim={}: simd={}, ref={}",
                dim,
                simd_result,
                ref_result
            );
        }
    }
}

#[test]
fn simd_correctness_l2() {
    for dim in [1, 8, 16, 32, 64, 128, 384, 768] {
        for seed in 0..5 {
            let a = test_vec(dim, seed);
            let b = test_vec(dim, seed + 1000);

            let simd_result = l2_distance(&a, &b);
            let ref_result = ref_l2(&a, &b);

            assert!(
                approx_eq(simd_result, ref_result, 1e-4),
                "l2 mismatch at dim={}: simd={}, ref={}",
                dim,
                simd_result,
                ref_result
            );
        }
    }
}

#[test]
fn simd_correctness_cosine() {
    for dim in [1, 8, 16, 32, 64, 128, 384, 768] {
        for seed in 0..5 {
            let a = test_vec(dim, seed);
            let b = test_vec(dim, seed + 1000);

            let simd_result = cosine(&a, &b);
            let ref_result = ref_cosine(&a, &b);

            assert!(
                approx_eq(simd_result, ref_result, 1e-4),
                "cosine mismatch at dim={}: simd={}, ref={}",
                dim,
                simd_result,
                ref_result
            );
        }
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn simd_correctness_edge_cases() {
    // Empty vectors
    assert_eq!(dot(&[], &[]), 0.0);

    // Single element
    assert_eq!(dot(&[2.0], &[3.0]), 6.0);

    // Zeros
    let zeros = vec![0.0f32; 100];
    let ones = vec![1.0f32; 100];
    assert_eq!(dot(&zeros, &ones), 0.0);
    assert_eq!(l2_distance_squared(&zeros, &zeros), 0.0);

    // Identical vectors
    let v = test_vec(64, 42);
    assert!(approx_eq(l2_distance(&v, &v), 0.0, 1e-6));
    assert!(approx_eq(cosine(&v, &v), 1.0, 1e-5));
}

#[test]
fn simd_correctness_special_values() {
    // Very small values (underflow potential)
    let small: Vec<f32> = (0..64).map(|i| 1e-20 * (i as f32 + 1.0)).collect();
    let ref_result = ref_dot(&small, &small);
    let simd_result = dot(&small, &small);
    // Both should be very small, just check same order of magnitude
    assert!(
        (simd_result - ref_result).abs() < 1e-30,
        "small value mismatch"
    );

    // Large values (overflow potential)
    let large: Vec<f32> = (0..64).map(|i| 1e10 * (i as f32 + 1.0)).collect();
    let ref_result = ref_dot(&large, &large);
    let simd_result = dot(&large, &large);
    assert!(
        approx_eq(simd_result, ref_result, 1e-3),
        "large value mismatch: simd={}, ref={}",
        simd_result,
        ref_result
    );

    // Mixed signs
    let mixed: Vec<f32> = (0..64)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let ref_result = ref_dot(&mixed, &mixed);
    let simd_result = dot(&mixed, &mixed);
    assert_eq!(simd_result, ref_result);
}

// =============================================================================
// Invariant Tests
// =============================================================================

#[test]
fn simd_invariant_dot_commutative() {
    for dim in [32, 64, 128, 384] {
        let a = test_vec(dim, 1);
        let b = test_vec(dim, 2);
        assert!(
            approx_eq(dot(&a, &b), dot(&b, &a), 1e-6),
            "dot should be commutative"
        );
    }
}

#[test]
fn simd_invariant_l2_symmetric() {
    for dim in [32, 64, 128, 384] {
        let a = test_vec(dim, 1);
        let b = test_vec(dim, 2);
        assert!(
            approx_eq(l2_distance(&a, &b), l2_distance(&b, &a), 1e-6),
            "l2 should be symmetric"
        );
    }
}

#[test]
fn simd_invariant_l2_nonnegative() {
    for dim in [32, 64, 128, 384] {
        let a = test_vec(dim, 1);
        let b = test_vec(dim, 2);
        assert!(l2_distance(&a, &b) >= 0.0, "l2 should be non-negative");
        assert!(
            l2_distance_squared(&a, &b) >= 0.0,
            "l2_squared should be non-negative"
        );
    }
}

#[test]
fn simd_invariant_cosine_range() {
    for dim in [32, 64, 128, 384] {
        for seed in 0..10 {
            let a = test_vec(dim, seed);
            let b = test_vec(dim, seed + 100);
            let sim = cosine(&a, &b);
            assert!(
                sim >= -1.0 - 1e-5 && sim <= 1.0 + 1e-5,
                "cosine should be in [-1, 1], got {}",
                sim
            );
        }
    }
}
