//! Dense vector operations with SIMD acceleration.
//!
//! Core operations: dot product, norm, cosine similarity, L2 distance.
//!
//! # Performance Hierarchy
//!
//! Runtime dispatch selects the fastest available implementation:
//!
//! | ISA | Min dim | Typical speedup |
//! |-----|---------|-----------------|
//! | AVX-512 | 64 | ~10-20x vs scalar |
//! | AVX2+FMA | 16 | 5-10x vs scalar |
//! | NEON | 16 | 4-8x vs scalar |
//! | Portable | any | 1x (baseline) |

// arch is only used on architectures with SIMD dispatch
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::arch;

// MIN_DIM_SIMD is only used on architectures with SIMD dispatch
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::MIN_DIM_SIMD;

/// Minimum dimension for AVX-512 (64 floats = one unrolled iteration).
#[cfg(target_arch = "x86_64")]
const MIN_DIM_AVX512: usize = 64;

/// Dot product of two vectors: `Σ(a[i] * b[i])`.
///
/// Returns 0.0 for empty vectors.
///
/// # SIMD Acceleration
///
/// Automatically dispatches to (in order of preference):
/// - AVX-512 on x86_64 (runtime detection, n >= 64)
/// - AVX2+FMA on x86_64 (runtime detection, n >= 16)
/// - NEON on aarch64 (always available, n >= 16)
/// - Portable fallback otherwise
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
///
/// # Example
///
/// ```rust
/// use innr::dot;
///
/// let a = [1.0_f32, 2.0, 3.0];
/// let b = [4.0_f32, 5.0, 6.0];
/// assert!((dot(&a, &b) - 32.0).abs() < 1e-6);
/// ```
#[inline]
#[must_use]
#[allow(unsafe_code)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "innr::dot: slice length mismatch ({} vs {})",
        a.len(),
        b.len()
    );

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let n = a.len();

    #[cfg(target_arch = "x86_64")]
    {
        // Prefer AVX-512 for large vectors (16 floats/register, 4-way unrolled)
        if n >= MIN_DIM_AVX512 && is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F verified via runtime detection.
            return unsafe { arch::x86_64::dot_avx512(a, b) };
        }

        // Fall back to AVX2+FMA (8 floats/register, 4-way unrolled)
        if n >= MIN_DIM_SIMD && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
        {
            // SAFETY: AVX2 and FMA verified via runtime detection.
            return unsafe { arch::x86_64::dot_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if n >= MIN_DIM_SIMD {
            // SAFETY: NEON is always available on aarch64.
            return unsafe { arch::aarch64::dot_neon(a, b) };
        }
    }

    #[allow(unreachable_code)]
    dot_portable(a, b)
}

/// Portable (non-SIMD) dot product.
///
/// LLVM typically auto-vectorizes this for common architectures,
/// but explicit SIMD implementations are faster for dimension >= 16.
#[inline]
#[must_use]
pub fn dot_portable(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm (Euclidean norm) of a vector: `sqrt(Σ(v[i]²))`.
///
/// # Example
///
/// ```rust
/// use innr::norm;
///
/// let v = [3.0_f32, 4.0];
/// assert!((norm(&v) - 5.0).abs() < 1e-6);
/// ```
#[inline]
#[must_use]
pub fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Normalize a vector to unit length (in-place).
///
/// After normalization, `norm(v) == 1.0` (within floating-point precision).
/// Zero vectors are left unchanged (no division by zero).
///
/// # Example
///
/// ```rust
/// use innr::dense::normalize;
/// use innr::norm;
///
/// let mut v = vec![3.0_f32, 4.0];
/// normalize(&mut v);
/// assert!((norm(&v) - 1.0).abs() < 1e-6);
/// ```
pub fn normalize(v: &mut [f32]) {
    let n = norm(v);
    if n > crate::NORM_EPSILON {
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Cosine similarity between two vectors.
///
/// `cosine(a, b) = dot(a, b) / (norm(a) * norm(b))`
///
/// # Single-Pass Fusion
///
/// Computes dot(a,b), ||a||^2, and ||b||^2 in a single pass over memory,
/// reading each vector only once. This is ~3x faster than the naive
/// 3-pass approach for memory-bound workloads (typical for dim >= 128).
///
/// # Zero Vector Handling
///
/// Returns `0.0` if either vector has effectively-zero norm.
/// This avoids division by zero and provides a sensible default for
/// padding tokens, OOV embeddings, or failed inference.
///
/// # Result Range
///
/// Result is in `[-1, 1]` for valid input. Floating-point error can push
/// slightly outside this range; clamp if strict bounds are required.
///
/// # SIMD Acceleration
///
/// Automatically dispatches to (in order of preference):
/// - AVX-512 on x86_64 (runtime detection, n >= 64)
/// - AVX2+FMA on x86_64 (runtime detection, n >= 16)
/// - NEON on aarch64 (always available, n >= 16)
/// - Portable fallback otherwise
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
///
/// # Example
///
/// ```rust
/// use innr::cosine;
///
/// // Orthogonal vectors
/// let a = [1.0_f32, 0.0];
/// let b = [0.0_f32, 1.0];
/// assert!(cosine(&a, &b).abs() < 1e-6);
///
/// // Parallel vectors
/// let c = [1.0_f32, 0.0];
/// let d = [2.0_f32, 0.0];
/// assert!((cosine(&c, &d) - 1.0).abs() < 1e-6);
///
/// // Zero vector returns 0.0
/// let zero = [0.0_f32, 0.0];
/// assert_eq!(cosine(&a, &zero), 0.0);
/// ```
#[inline]
#[must_use]
#[allow(unsafe_code)]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "innr::cosine: slice length mismatch ({} vs {})",
        a.len(),
        b.len()
    );

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let n = a.len();

    #[cfg(target_arch = "x86_64")]
    {
        if n >= MIN_DIM_AVX512 && is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F verified via runtime detection.
            return unsafe { arch::x86_64::cosine_avx512(a, b) };
        }

        if n >= MIN_DIM_SIMD && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
        {
            // SAFETY: AVX2 and FMA verified via runtime detection.
            return unsafe { arch::x86_64::cosine_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if n >= MIN_DIM_SIMD {
            // SAFETY: NEON is always available on aarch64.
            return unsafe { arch::aarch64::cosine_neon(a, b) };
        }
    }

    #[allow(unreachable_code)]
    cosine_portable(a, b)
}

/// Portable (non-SIMD) cosine similarity.
///
/// Single-pass: accumulates dot(a,b), ||a||^2, ||b||^2 simultaneously.
#[inline]
#[must_use]
pub fn cosine_portable(a: &[f32], b: &[f32]) -> f32 {
    let mut ab = 0.0f32;
    let mut aa = 0.0f32;
    let mut bb = 0.0f32;

    for (&ai, &bi) in a.iter().zip(b.iter()) {
        ab += ai * bi;
        aa += ai * ai;
        bb += bi * bi;
    }

    if aa > crate::NORM_EPSILON_SQ && bb > crate::NORM_EPSILON_SQ {
        ab / (aa.sqrt() * bb.sqrt())
    } else {
        0.0
    }
}

/// Angular distance: `acos(cosine_similarity) / π`.
///
/// Unlike cosine similarity, angular distance is a **true metric**:
/// 1. d(x, y) >= 0
/// 2. d(x, y) = 0 iff x = y
/// 3. d(x, y) = d(y, x)
/// 4. d(x, z) <= d(x, y) + d(y, z) (Triangle Inequality)
///
/// Range: `[0, 1]`.
/// - 0: Identical direction
/// - 0.5: Orthogonal
/// - 1: Opposite direction
///
/// # Example
///
/// ```
/// use innr::angular_distance;
///
/// let a = [1.0_f32, 0.0];
/// let b = [0.0_f32, 1.0];
/// // Orthogonal vectors -> angular distance = 0.5
/// assert!((angular_distance(&a, &b) - 0.5).abs() < 1e-5);
/// ```
///
/// # References
/// - [Angular distance](https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity)
#[inline]
#[must_use]
pub fn angular_distance(a: &[f32], b: &[f32]) -> f32 {
    let sim = cosine(a, b).clamp(-1.0, 1.0);
    sim.acos() / std::f32::consts::PI
}

/// Matryoshka-optimized dot product for nested embeddings.
///
/// Computes the dot product only on the first `prefix_len` dimensions.
/// MRL embeddings allow variable-length scoring for adaptive retrieval:
/// index at full dimension for quality, re-rank with a short prefix for speed,
/// or use a prefix as a first-stage filter before exact scoring.
///
/// # Research Context
///
/// Matryoshka Representation Learning (MRL) optimizes representations by training
/// a single high-dimensional vector such that its prefixes are explicitly supervised.
/// This enables "train-once, deploy-everywhere" flexibility -- the same model
/// checkpoint serves 64-dim mobile retrieval and 768-dim server-side ranking.
///
/// The 2D extension generalizes truncation to both the layer axis (early exit)
/// and the dimension axis (prefix truncation), yielding a grid of quality/cost
/// tradeoffs from a single forward pass.
///
/// # Examples
///
/// ```
/// use innr::dense::{matryoshka_dot, dot};
///
/// let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
/// let b = vec![5.0_f32, 4.0, 3.0, 2.0, 1.0];
///
/// // Use first 3 dimensions only (coarse, fast retrieval)
/// let coarse = matryoshka_dot(&a, &b, 3);
/// assert_eq!(coarse, dot(&a[..3], &b[..3]));
///
/// // Use all 5 dimensions (fine, precise re-ranking)
/// let fine = matryoshka_dot(&a, &b, 5);
/// assert_eq!(fine, dot(&a, &b));
/// ```
///
/// # References
///
/// - Kusupati et al. (2022). "Matryoshka Representation Learning" (NeurIPS) --
///   the foundational paper for prefix-truncatable embeddings; shows that
///   explicit multi-granularity supervision preserves ranking quality at
///   2-8x dimension reduction.
/// - Li et al. (2024). "2D Matryoshka Sentence Embeddings" -- extends MRL
///   to both layer and dimension axes, enabling joint early-exit and
///   prefix-truncation from a single trained model.
#[inline]
#[must_use]
pub fn matryoshka_dot(a: &[f32], b: &[f32], prefix_len: usize) -> f32 {
    let end = prefix_len.min(a.len()).min(b.len());
    dot(&a[..end], &b[..end])
}

/// Matryoshka-optimized cosine similarity on the first `prefix_len` dimensions.
///
/// See [`matryoshka_dot`] for background on prefix-truncatable embeddings.
///
/// # Examples
///
/// ```
/// use innr::dense::{matryoshka_cosine, cosine};
///
/// let a = vec![1.0_f32, 0.0, 0.0, 1.0];
/// let b = vec![0.0_f32, 1.0, 0.0, 0.0];
///
/// // Cosine on first 2 dims: orthogonal -> 0.0
/// let sim_2d = matryoshka_cosine(&a, &b, 2);
/// assert!((sim_2d - 0.0).abs() < 1e-6);
/// ```
#[inline]
#[must_use]
pub fn matryoshka_cosine(a: &[f32], b: &[f32], prefix_len: usize) -> f32 {
    let end = prefix_len.min(a.len()).min(b.len());
    cosine(&a[..end], &b[..end])
}

/// L2 (Euclidean) distance: `sqrt(Σ(a[i] - b[i])²)`.
///
/// # Example
///
/// ```rust
/// use innr::l2_distance;
///
/// let a = [0.0_f32, 0.0];
/// let b = [3.0_f32, 4.0];
/// assert!((l2_distance(&a, &b) - 5.0).abs() < 1e-6);
/// ```
#[inline]
#[must_use]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    l2_distance_squared(a, b).sqrt()
}

/// L1 (Manhattan) distance: `Σ|a[i] - b[i]|`.
///
/// # SIMD Acceleration
///
/// Automatically dispatches to (in order of preference):
/// - AVX-512 on x86_64 (runtime detection, n >= 64)
/// - AVX2 on x86_64 (runtime detection, n >= 16)
/// - NEON on aarch64 (always available, n >= 16)
/// - Portable fallback otherwise
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
///
/// # Example
///
/// ```rust
/// use innr::l1_distance;
///
/// let a = [1.0_f32, 2.0];
/// let b = [4.0_f32, 0.0];
/// // |1-4| + |2-0| = 3 + 2 = 5
/// assert!((l1_distance(&a, &b) - 5.0).abs() < 1e-6);
/// ```
#[inline]
#[must_use]
#[allow(unsafe_code)]
pub fn l1_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "innr::l1_distance: slice length mismatch ({} vs {})",
        a.len(),
        b.len()
    );

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let n = a.len();

    #[cfg(target_arch = "x86_64")]
    {
        if n >= MIN_DIM_AVX512 && is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F verified via runtime detection.
            return unsafe { arch::x86_64::l1_avx512(a, b) };
        }

        if n >= MIN_DIM_SIMD && is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 verified via runtime detection.
            return unsafe { arch::x86_64::l1_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if n >= MIN_DIM_SIMD {
            // SAFETY: NEON is always available on aarch64.
            return unsafe { arch::aarch64::l1_neon(a, b) };
        }
    }

    #[allow(unreachable_code)]
    l1_distance_portable(a, b)
}

/// Portable (non-SIMD) L1 distance.
#[inline]
#[must_use]
pub fn l1_distance_portable(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Squared L2 distance: `Σ(a[i] - b[i])²`.
///
/// More efficient than [`l2_distance`] when only comparing distances
/// (no need for sqrt). Computed in a single pass over both vectors,
/// avoiding catastrophic cancellation that occurs with the expansion
/// `||a||² + ||b||² - 2<a,b>` for close vectors.
///
/// # SIMD Acceleration
///
/// Automatically dispatches to (in order of preference):
/// - AVX-512 on x86_64 (runtime detection, n >= 64)
/// - AVX2+FMA on x86_64 (runtime detection, n >= 16)
/// - NEON on aarch64 (always available, n >= 16)
/// - Portable fallback otherwise
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
///
/// # Example
///
/// ```rust
/// use innr::l2_distance_squared;
///
/// let a = [0.0_f32, 0.0];
/// let b = [3.0_f32, 4.0];
/// assert!((l2_distance_squared(&a, &b) - 25.0).abs() < 1e-6);
/// ```
#[inline]
#[must_use]
#[allow(unsafe_code)]
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "innr::l2_distance_squared: slice length mismatch ({} vs {})",
        a.len(),
        b.len()
    );

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let n = a.len();

    #[cfg(target_arch = "x86_64")]
    {
        if n >= MIN_DIM_AVX512 && is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F verified via runtime detection.
            return unsafe { arch::x86_64::l2_squared_avx512(a, b) };
        }

        if n >= MIN_DIM_SIMD && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
        {
            // SAFETY: AVX2 and FMA verified via runtime detection.
            return unsafe { arch::x86_64::l2_squared_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if n >= MIN_DIM_SIMD {
            // SAFETY: NEON is always available on aarch64.
            return unsafe { arch::aarch64::l2_squared_neon(a, b) };
        }
    }

    #[allow(unreachable_code)]
    l2_distance_squared_portable(a, b)
}

/// Portable (non-SIMD) squared L2 distance.
#[inline]
#[must_use]
pub fn l2_distance_squared_portable(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matryoshka_ranking_preservation() {
        // Create a query and several documents
        let query = [1.0, 0.5, 0.2, 0.1];
        let doc1 = [0.9, 0.4, 0.1, 0.05]; // Closest
        let doc2 = [0.1, 0.1, 0.1, 0.1]; // Farther
        let doc3 = [-0.5, -0.2, 0.0, 0.0]; // Farthest

        // Ranking at full dimension (4)
        let sim1_full = cosine(&query, &doc1);
        let sim2_full = cosine(&query, &doc2);
        let sim3_full = cosine(&query, &doc3);
        assert!(sim1_full > sim2_full);
        assert!(sim2_full > sim3_full);

        // Ranking at Matryoshka prefix (2)
        let sim1_prefix = matryoshka_cosine(&query, &doc1, 2);
        let sim2_prefix = matryoshka_cosine(&query, &doc2, 2);
        let sim3_prefix = matryoshka_cosine(&query, &doc3, 2);

        // The relative order should be preserved
        assert!(sim1_prefix > sim2_prefix);
        assert!(sim2_prefix > sim3_prefix);
    }

    #[test]
    fn test_dot_simd_threshold() {
        // Below SIMD threshold
        let small_a: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let small_b: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let result_small = dot(&small_a, &small_b);

        // Above SIMD threshold
        let large_a: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let large_b: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let result_large = dot(&large_a, &large_b);

        // Verify correctness
        let expected_small: f32 = (0..8).map(|i| (i * i) as f32).sum();
        let expected_large: f32 = (0..32).map(|i| (i * i) as f32).sum();

        assert!((result_small - expected_small).abs() < 1e-3);
        assert!((result_large - expected_large).abs() < 1e-1);
    }

    #[test]
    fn test_l2_distance_triangle_inequality() {
        let a = [0.0_f32, 0.0];
        let b = [1.0_f32, 0.0];
        let c = [0.0_f32, 1.0];

        let ab = l2_distance(&a, &b);
        let bc = l2_distance(&b, &c);
        let ac = l2_distance(&a, &c);

        // Triangle inequality: ac <= ab + bc
        assert!(ac <= ab + bc + 1e-6);
    }

    // =========================================================================
    // Edge case tests
    // =========================================================================

    #[test]
    fn test_dot_empty() {
        assert_eq!(dot(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_empty() {
        // Both vectors empty: norms are 0.0, which is below NORM_EPSILON -> returns 0.0.
        let result = cosine(&[], &[]);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_dot_single() {
        assert_eq!(dot(&[3.0], &[4.0]), 12.0);
    }

    #[test]
    fn test_dot_exactly_16_elements() {
        // dim=16 is the SIMD threshold boundary.
        let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
        let result = dot(&a, &b);
        let expected: f32 = (0..16).map(|i| (i * (i + 1)) as f32).sum();
        assert!(
            (result - expected).abs() < 1e-3,
            "dot at dim=16: got {result}, expected {expected}"
        );
    }

    #[test]
    fn test_cosine_exactly_16_elements() {
        let a: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=16).map(|i| i as f32 * 2.0).collect();
        // Parallel vectors -> cosine = 1.0.
        let result = cosine(&a, &b);
        assert!(
            (result - 1.0).abs() < 1e-5,
            "cosine of parallel vectors at dim=16: got {result}"
        );
    }

    #[test]
    fn test_norm_exactly_16_elements() {
        let v: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let result = norm(&v);
        let expected = dot(&v, &v).sqrt();
        assert!(
            (result - expected).abs() < 1e-5,
            "norm at dim=16: got {result}, expected {expected}"
        );
    }

    #[test]
    fn test_dot_large_values() {
        // Large but not overflowing: f32::MAX is ~3.4e38, so 1e18 * 1e18 = 1e36 is fine.
        let a = [1e18_f32, 1e18];
        let b = [1e18_f32, 1e18];
        let result = dot(&a, &b);
        assert!(result.is_finite(), "dot with large values should be finite");
        assert!(result > 0.0);
    }

    #[test]
    fn test_norm_large_vector() {
        let v = [1e19_f32, 1e19];
        let result = norm(&v);
        assert!(result.is_finite(), "norm of large vector should be finite");
        assert!(result > 0.0);
    }

    #[test]
    fn test_cosine_zero_vector_both() {
        // Both zero vectors -> 0.0 (not panic, not NaN).
        let zero = [0.0_f32, 0.0, 0.0];
        let result = cosine(&zero, &zero);
        assert_eq!(result, 0.0, "cosine of two zero vectors should be 0.0");
    }

    #[test]
    fn test_norm_zero_vector() {
        let zero = [0.0_f32, 0.0, 0.0];
        assert_eq!(norm(&zero), 0.0);
    }

    #[test]
    fn test_dot_all_negatives() {
        let a = [-1.0_f32, -2.0, -3.0];
        let b = [-4.0_f32, -5.0, -6.0];
        // (-1)*(-4) + (-2)*(-5) + (-3)*(-6) = 4 + 10 + 18 = 32
        let result = dot(&a, &b);
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "innr::dot: slice length mismatch")]
    fn dot_panics_on_length_mismatch() {
        let _ = dot(&[1.0, 2.0], &[1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "innr::l1_distance: slice length mismatch")]
    fn l1_distance_panics_on_length_mismatch() {
        let _ = l1_distance(&[1.0], &[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "innr::l2_distance_squared: slice length mismatch")]
    fn l2_distance_squared_panics_on_length_mismatch() {
        let _ = l2_distance_squared(&[1.0], &[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "innr::l2_distance_squared: slice length mismatch")]
    fn l2_distance_panics_on_length_mismatch() {
        let _ = l2_distance(&[1.0], &[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "innr::cosine: slice length mismatch")]
    fn cosine_panics_on_length_mismatch() {
        let _ = cosine(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn test_cosine_mixed_signs() {
        // Antiparallel vectors -> cosine = -1.0.
        let a = [1.0_f32, 2.0, 3.0];
        let b = [-1.0_f32, -2.0, -3.0];
        let result = cosine(&a, &b);
        assert!(
            (result - (-1.0)).abs() < 1e-5,
            "cosine of antiparallel vectors: got {result}, expected -1.0"
        );
    }

    // =========================================================================
    // normalize tests
    // =========================================================================

    #[test]
    fn test_normalize_unit_norm() {
        let mut v = vec![3.0_f32, 4.0];
        normalize(&mut v);
        let n = norm(&v);
        assert!(
            (n - 1.0).abs() < 1e-6,
            "norm(normalize(v)) should be ~1.0, got {n}"
        );
    }

    #[test]
    fn test_normalize_direction_preserved() {
        let mut v = vec![1.0_f32, 0.0, 0.0];
        normalize(&mut v);
        // Already unit length; values should be unchanged.
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!(v[1].abs() < 1e-6);
        assert!(v[2].abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector_unchanged() {
        // Zero vector must not be divided (would produce NaN).
        let mut v = vec![0.0_f32, 0.0, 0.0];
        normalize(&mut v);
        // Zero vector left unchanged.
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_various_dims() {
        for dim in [1, 8, 16, 64, 128] {
            let mut v: Vec<f32> = (1..=dim).map(|i| i as f32).collect();
            normalize(&mut v);
            let n = norm(&v);
            assert!(
                (n - 1.0).abs() < 1e-5,
                "norm after normalize should be ~1.0 for dim={dim}, got {n}"
            );
        }
    }

    // =========================================================================
    // matryoshka_dot tests
    // =========================================================================

    #[test]
    fn test_matryoshka_dot_equals_prefix_dot() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0_f32, 4.0, 3.0, 2.0, 1.0];

        for prefix in [1, 2, 3, 4, 5] {
            let mrd = matryoshka_dot(&a, &b, prefix);
            let expected = dot(&a[..prefix], &b[..prefix]);
            assert!(
                (mrd - expected).abs() < 1e-6,
                "matryoshka_dot(prefix={prefix}): got {mrd}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_matryoshka_dot_full_prefix_equals_dot() {
        let a = vec![1.0_f32, 0.0, -1.0];
        let b = vec![2.0_f32, 3.0, 4.0];
        let full = matryoshka_dot(&a, &b, a.len());
        let expected = dot(&a, &b);
        assert!((full - expected).abs() < 1e-6);
    }

    #[test]
    fn test_matryoshka_dot_prefix_longer_than_vec_clips() {
        // prefix_len > vec length: clips to vec length.
        let a = vec![1.0_f32, 2.0];
        let b = vec![3.0_f32, 4.0];
        let result = matryoshka_dot(&a, &b, 100);
        let expected = dot(&a, &b);
        assert!((result - expected).abs() < 1e-6);
    }

    // =========================================================================
    // matryoshka_cosine tests
    // =========================================================================

    #[test]
    fn test_matryoshka_cosine_equals_prefix_cosine() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![4.0_f32, 3.0, 2.0, 1.0];

        for prefix in [1, 2, 3, 4] {
            let mrc = matryoshka_cosine(&a, &b, prefix);
            let expected = cosine(&a[..prefix], &b[..prefix]);
            assert!(
                (mrc - expected).abs() < 1e-6,
                "matryoshka_cosine(prefix={prefix}): got {mrc}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_matryoshka_cosine_full_prefix_equals_cosine() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let full = matryoshka_cosine(&a, &b, 2);
        let expected = cosine(&a, &b);
        // Orthogonal: both should be ~0.0.
        assert!((full - expected).abs() < 1e-6);
        assert!(full.abs() < 1e-6);
    }

    #[test]
    fn test_matryoshka_cosine_prefix_one() {
        // prefix=1: cosine of scalars is sign(a[0]*b[0]).
        let a = vec![3.0_f32, -99.0, -99.0];
        let b = vec![5.0_f32, 1.0, 1.0];
        let result = matryoshka_cosine(&a, &b, 1);
        // cosine([3], [5]) = 1.0 (parallel single-element vectors).
        assert!((result - 1.0).abs() < 1e-5, "got {result}");
    }

    // =========================================================================
    // angular_distance tests
    // =========================================================================

    #[test]
    fn test_angular_distance_range() {
        // angular_distance must be in [0, 1] for any pair of vectors.
        let pairs: &[(&[f32], &[f32])] = &[
            (&[1.0, 0.0], &[0.0, 1.0]),  // orthogonal -> 0.5
            (&[1.0, 0.0], &[1.0, 0.0]),  // identical  -> 0.0
            (&[1.0, 0.0], &[-1.0, 0.0]), // opposite   -> 1.0
            (&[1.0, 1.0], &[1.0, -1.0]), // 90 deg     -> 0.5
        ];
        for (a, b) in pairs {
            let d = angular_distance(a, b);
            assert!(
                (0.0..=1.0).contains(&d),
                "angular_distance out of [0,1]: {d}"
            );
        }
    }

    #[test]
    fn test_angular_distance_orthogonal_is_half() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        let d = angular_distance(&a, &b);
        assert!(
            (d - 0.5).abs() < 1e-5,
            "orthogonal angular_distance: got {d}"
        );
    }

    #[test]
    fn test_angular_distance_identical_is_zero() {
        let a = [1.0_f32, 2.0, 3.0];
        let d = angular_distance(&a, &a);
        // cosine(a,a) may be slightly above 1.0 due to fp rounding, clamped to 1.0
        // before acos, so acos(1.0) / pi is at most a small fp epsilon.
        assert!(d < 1e-3, "identical vectors angular_distance: got {d}");
    }

    #[test]
    fn test_angular_distance_opposite_is_one() {
        let a = [1.0_f32, 0.0];
        let b = [-1.0_f32, 0.0];
        let d = angular_distance(&a, &b);
        assert!(
            (d - 1.0).abs() < 1e-5,
            "opposite vectors angular_distance: got {d}"
        );
    }

    #[test]
    fn test_angular_distance_cosine_relationship() {
        // angular_distance(a, b) == acos(cosine(a, b)) / pi
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, -1.0, 2.0];
        let c = cosine(&a, &b).clamp(-1.0, 1.0);
        let expected = c.acos() / std::f32::consts::PI;
        let result = angular_distance(&a, &b);
        assert!(
            (result - expected).abs() < 1e-6,
            "angular_distance mismatch: got {result}, expected {expected}"
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Dimensions that cross SIMD dispatch thresholds:
    /// 1 (scalar), 8 (below SIMD), 15 (boundary-1), 16 (NEON/AVX2 threshold),
    /// 17 (boundary+1), 32, 64 (AVX-512 threshold), 128, 768 (typical embedding dim).
    const DIMS: &[usize] = &[1, 8, 15, 16, 17, 32, 64, 128, 768];

    /// Bounded finite f32 in [-1000, 1000]. Avoids overflow in dot/norm accumulations
    /// while still exercising a wide range of magnitudes.
    fn bounded_f32() -> impl Strategy<Value = f32> {
        -1000.0_f32..1000.0_f32
    }

    /// Bounded f32 vector of a given length.
    fn bounded_vec(len: usize) -> impl Strategy<Value = Vec<f32>> {
        prop::collection::vec(bounded_f32(), len)
    }

    /// Strategy that picks a dimension from the SIMD-threshold set.
    fn simd_dim() -> impl Strategy<Value = usize> {
        prop::sample::select(DIMS)
    }

    /// Non-zero bounded vector. Uses values in [0.01, 100] so the norm is always
    /// well above NORM_EPSILON, avoiding the zero-vector guard in cosine().
    fn nonzero_vec(len: usize) -> impl Strategy<Value = Vec<f32>> {
        prop::collection::vec(
            prop_oneof![0.01_f32..100.0_f32, -100.0_f32..-0.01_f32,],
            len,
        )
    }

    /// Generate (dim, vec_a, vec_b) where dim comes from DIMS.
    fn dim_and_two_vecs() -> impl Strategy<Value = (usize, Vec<f32>, Vec<f32>)> {
        simd_dim().prop_flat_map(|d| (Just(d), bounded_vec(d), bounded_vec(d)))
    }

    /// Generate (dim, vec) where dim comes from DIMS.
    fn dim_and_vec() -> impl Strategy<Value = (usize, Vec<f32>)> {
        simd_dim().prop_flat_map(|d| (Just(d), bounded_vec(d)))
    }

    /// Generate (dim, vec_a, vec_b) with non-zero vectors.
    fn dim_and_two_nonzero_vecs() -> impl Strategy<Value = (usize, Vec<f32>, Vec<f32>)> {
        simd_dim().prop_flat_map(|d| (Just(d), nonzero_vec(d), nonzero_vec(d)))
    }

    /// Generate (dim, vec) with non-zero vector.
    fn dim_and_nonzero_vec() -> impl Strategy<Value = (usize, Vec<f32>)> {
        simd_dim().prop_flat_map(|d| (Just(d), nonzero_vec(d)))
    }

    /// Generate (dim, vec_a, vec_b, vec_c) for three-vector properties.
    fn dim_and_three_vecs() -> impl Strategy<Value = (usize, Vec<f32>, Vec<f32>, Vec<f32>)> {
        simd_dim().prop_flat_map(|d| (Just(d), bounded_vec(d), bounded_vec(d), bounded_vec(d)))
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        // -----------------------------------------------------------------
        // 1. Dot product commutativity: dot(a, b) == dot(b, a)
        // -----------------------------------------------------------------
        #[test]
        fn dot_commutative((dim, a, b) in dim_and_two_vecs()) {
            let ab = dot(&a, &b);
            let ba = dot(&b, &a);
            let tol = 1e-4 * ab.abs().max(ba.abs()).max(1.0);
            prop_assert!(
                (ab - ba).abs() <= tol,
                "dot commutativity failed: dot(a,b)={ab}, dot(b,a)={ba}, dim={dim}"
            );
        }

        // -----------------------------------------------------------------
        // 2. Norm is non-negative for any vector
        // -----------------------------------------------------------------
        #[test]
        fn norm_nonnegative((dim, v) in dim_and_vec()) {
            let n = norm(&v);
            prop_assert!(
                n >= 0.0,
                "norm must be >= 0, got {n} for dim={dim}"
            );
            prop_assert!(
                n.is_finite(),
                "norm must be finite for bounded input, got {n} for dim={dim}"
            );
        }

        // -----------------------------------------------------------------
        // 3. Cosine similarity in [-1, 1] for non-zero vectors
        // -----------------------------------------------------------------
        #[test]
        fn cosine_range((dim, a, b) in dim_and_two_nonzero_vecs()) {
            let c = cosine(&a, &b);
            prop_assert!(
                (-1.0 - 1e-5..=1.0 + 1e-5).contains(&c),
                "cosine out of range: {c}, dim={dim}"
            );
        }

        // -----------------------------------------------------------------
        // 4. Cosine self-similarity ~= 1.0 for non-zero vectors
        // -----------------------------------------------------------------
        #[test]
        fn cosine_self_similarity((dim, v) in dim_and_nonzero_vec()) {
            let c = cosine(&v, &v);
            prop_assert!(
                (c - 1.0).abs() < 1e-4,
                "cosine(v, v) should be ~1.0, got {c}, dim={dim}"
            );
        }

        // -----------------------------------------------------------------
        // 5. L2 self-distance ~= 0.0
        // -----------------------------------------------------------------
        #[test]
        fn l2_self_distance_zero((dim, v) in dim_and_vec()) {
            let d = l2_distance(&v, &v);
            prop_assert!(
                d.abs() < 1e-5,
                "l2_distance(v, v) should be ~0, got {d}, dim={dim}"
            );
        }

        // -----------------------------------------------------------------
        // 6. L2 triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        // -----------------------------------------------------------------
        #[test]
        fn l2_triangle_inequality((dim, a, b, c) in dim_and_three_vecs()) {
            let ab = l2_distance(&a, &b);
            let bc = l2_distance(&b, &c);
            let ac = l2_distance(&a, &c);
            // Floating-point epsilon proportional to magnitude.
            let eps = 1e-4 * (ab + bc).max(1.0);
            prop_assert!(
                ac <= ab + bc + eps,
                "triangle inequality violated: d(a,c)={ac} > d(a,b)={ab} + d(b,c)={bc}, dim={dim}"
            );
        }

        // -----------------------------------------------------------------
        // 7. L2 direct formula vs expansion formula consistency
        //
        // The direct formula `Σ(a_i - b_i)²` should agree with the
        // expansion `||a||² + ||b||² - 2<a,b>` for well-separated vectors.
        // For close vectors the expansion suffers catastrophic cancellation,
        // so we only check agreement within a loose tolerance scaled by
        // the magnitude of the norms.
        // -----------------------------------------------------------------
        #[test]
        fn l2_direct_vs_expansion((dim, a, b) in dim_and_two_vecs()) {
            let direct = l2_distance_squared(&a, &b);
            let aa: f32 = a.iter().map(|x| x * x).sum();
            let bb: f32 = b.iter().map(|x| x * x).sum();
            let ab: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let expansion = (aa + bb - 2.0 * ab).max(0.0);

            // Both should be non-negative
            prop_assert!(direct >= 0.0, "direct L2² should be >= 0, got {direct}");

            // Agree within tolerance proportional to norm magnitudes
            let scale = (aa + bb).max(1.0);
            let tol = 1e-3 * scale;
            prop_assert!(
                (direct - expansion).abs() <= tol,
                "direct={direct} vs expansion={expansion}, diff={}, scale={scale}, dim={dim}",
                (direct - expansion).abs()
            );
        }

        // -----------------------------------------------------------------
        // 8. L1 commutativity: l1(a, b) == l1(b, a)
        // -----------------------------------------------------------------
        #[test]
        fn l1_commutative((dim, a, b) in dim_and_two_vecs()) {
            let ab = l1_distance(&a, &b);
            let ba = l1_distance(&b, &a);
            let tol = 1e-4 * ab.abs().max(ba.abs()).max(1.0);
            prop_assert!(
                (ab - ba).abs() <= tol,
                "L1 commutativity failed: l1(a,b)={ab}, l1(b,a)={ba}, dim={dim}"
            );
        }

        // -----------------------------------------------------------------
        // 9. L1 non-negativity
        // -----------------------------------------------------------------
        #[test]
        fn l1_nonnegative((dim, a, b) in dim_and_two_vecs()) {
            let d = l1_distance(&a, &b);
            prop_assert!(
                d >= 0.0,
                "L1 must be >= 0, got {d} for dim={dim}"
            );
        }

        // -----------------------------------------------------------------
        // 10. L1 self-distance ~= 0.0
        // -----------------------------------------------------------------
        #[test]
        fn l1_self_distance_zero((dim, v) in dim_and_vec()) {
            let d = l1_distance(&v, &v);
            prop_assert!(
                d.abs() < 1e-5,
                "l1_distance(v, v) should be ~0, got {d}, dim={dim}"
            );
        }

        // -----------------------------------------------------------------
        // 12. SIMD dot matches f64 reference within ULP bounds
        // -----------------------------------------------------------------
        #[test]
        fn dot_matches_f64_reference((dim, a, b) in dim_and_two_vecs()) {
            let f64_ref: f64 = a.iter().zip(&b).map(|(&x, &y)| x as f64 * y as f64).sum();
            let f32_result = dot(&a, &b);

            // f32 accumulation error is O(n * eps * Σ|a_i * b_i|), not O(n * eps * |Σ a_i*b_i|).
            // When positive and negative terms cancel, the final sum can be much smaller
            // than the intermediate partial sums, so we scale by the sum of absolute products.
            let abs_product_sum: f64 = a.iter().zip(&b).map(|(&x, &y)| (x as f64 * y as f64).abs()).sum();
            let tol = (dim as f64) * f64::from(f32::EPSILON) * abs_product_sum.max(1.0);
            let diff = (f64::from(f32_result) - f64_ref).abs();

            prop_assert!(
                diff <= tol,
                "f64 reference mismatch: f32={f32_result}, f64_ref={f64_ref}, diff={diff}, tol={tol}, dim={dim}"
            );
        }

        // -----------------------------------------------------------------
        // 11. L2 direct formula accuracy for close vectors
        //
        // For vectors that differ only slightly, the direct formula should
        // produce results close to the portable reference.
        // -----------------------------------------------------------------
        #[test]
        fn l2_close_vectors_accuracy((dim, a) in dim_and_vec()) {
            // Create b = a + small perturbation
            let b: Vec<f32> = a.iter().map(|x| x + 1e-4).collect();
            let direct = l2_distance_squared(&a, &b);
            let reference = l2_distance_squared_portable(&a, &b);

            let tol = 1e-4 * reference.max(1e-6);
            prop_assert!(
                (direct - reference).abs() <= tol,
                "close vectors: direct={direct} vs reference={reference}, dim={dim}"
            );
        }
    }
}
