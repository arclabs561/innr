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
//! | AVX-512 | 64 | 10-20x vs scalar |
//! | AVX2+FMA | 16 | 5-10x vs scalar |
//! | NEON | 16 | 4-8x vs scalar |
//! | Portable | any | 1x (baseline) |

// arch is only used on architectures with SIMD dispatch
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::arch;
use crate::NORM_EPSILON;

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
/// # Debug Assertions
///
/// In debug builds, panics if vector lengths differ. In release builds,
/// mismatched lengths silently use the shorter length (for performance).
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
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "dot: dimension mismatch ({} vs {})",
        a.len(),
        b.len()
    );

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let n = a.len().min(b.len());

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

/// Cosine similarity between two vectors.
///
/// `cosine(a, b) = dot(a, b) / (norm(a) * norm(b))`
///
/// # Zero Vector Handling
///
/// Returns `0.0` if either vector has effectively-zero norm (< 1e-9).
/// This avoids division by zero and provides a sensible default for
/// padding tokens, OOV embeddings, or failed inference.
///
/// # Result Range
///
/// Result is in `[-1, 1]` for valid input. Floating-point error can push
/// slightly outside this range; clamp if strict bounds are required.
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
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d = dot(a, b);
    let na = norm(a);
    let nb = norm(b);
    if na > NORM_EPSILON && nb > NORM_EPSILON {
        d / (na * nb)
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
/// Latest embeddings (MRL) allow variable-length scoring for adaptive retrieval.
///
/// # Research Context
/// Matryoshka Representation Learning (MRL) optimizes representations by training
/// a single high-dimensional vector such that its prefixes are explicitly supervised.
/// This enables "train-once, deploy-everywhere" flexibility.
///
/// # References
/// - Kusupati et al. (2022). "Matryoshka Representation Learning" (NeurIPS)
#[inline]
#[must_use]
pub fn matryoshka_dot(a: &[f32], b: &[f32], prefix_len: usize) -> f32 {
    let end = prefix_len.min(a.len()).min(b.len());
    dot(&a[..end], &b[..end])
}

/// Matryoshka-optimized cosine similarity on the first `prefix_len` dimensions.
#[inline]
#[must_use]
pub fn matryoshka_cosine(a: &[f32], b: &[f32], prefix_len: usize) -> f32 {
    let end = prefix_len.min(a.len()).min(b.len());
    cosine(&a[..end], &b[..end])
}

/// Compute mean pooling of multiple vectors.
///
/// Result is written into `out`.
pub fn pool_mean(vectors: &[&[f32]], out: &mut [f32]) {
    if vectors.is_empty() {
        return;
    }
    let n = vectors.len() as f32;
    out.fill(0.0);
    for v in vectors {
        for (o, &vi) in out.iter_mut().zip(v.iter()) {
            *o += vi;
        }
    }
    for o in out.iter_mut() {
        *o /= n;
    }
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
pub fn l1_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "l1_distance: dimension mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Squared L2 distance: `Σ(a[i] - b[i])²`.
///
/// More efficient than [`l2_distance`] when only comparing distances
/// (no need for sqrt).
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
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "l2_distance_squared: dimension mismatch");

    // Expanded form: ||a - b||² = ||a||² + ||b||² - 2<a,b>
    // Direct computation avoids allocation for (a - b).
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

/// Bilinear product: `phi^T * psi / sqrt(d)`.
///
/// Used in Dual Goal Representations for value estimation.
#[inline]
#[must_use]
pub fn bilinear(phi: &[f32], psi: &[f32]) -> f32 {
    let d = phi.len();
    if d == 0 {
        return 0.0;
    }
    dot(phi, psi) / (d as f32).sqrt()
}

/// Clifford/Geometric Algebra Product (Foundation for 2026 Rotors).
///
/// Computes the outer product part of the Clifford product for small bivectors.
#[inline]
pub fn geometric_outer_product(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut res = Vec::with_capacity(a.len() * b.len());
    for &ai in a {
        for &bi in b {
            res.push(ai * bi);
        }
    }
    res
}

/// Metric Residual (MRN) distance: `sqrt(Σ(s[i] - g[i])² + eps) + quasi(s, g)`.
///
/// Where `quasi(s, g) = max(0, Σ max(0, s_asym[i] - g_asym[i]))` or similar asymmetric part.
/// This implementation assumes `s` and `g` are already split or handles the full vector.
#[inline]
#[must_use]
pub fn metric_residual(s: &[f32], g: &[f32], eps: f32) -> f32 {
    let d = s.len();
    let half = d / 2;
    let sym_dist = l2_distance(&s[..half], &g[..half]);

    let mut asym_max = 0.0_f32;
    for i in half..d {
        let diff = s[i] - g[i];
        if diff > asym_max {
            asym_max = diff;
        }
    }

    (sym_dist * sym_dist + eps).sqrt() + asym_max
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

    #[test]
    fn test_bilinear() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        // dot = 11, d = 2, sqrt(2) = 1.414...
        let res = bilinear(&a, &b);
        assert!((res - 11.0 / 2.0_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_metric_residual() {
        let s = [0.0, 0.0, 1.0, 0.0]; // half symmetric, half asymmetric
        let g = [0.0, 0.0, 0.0, 0.0];
        // sym_dist = 0, asym_max = 1
        let res = metric_residual(&s, &g, 1e-6);
        assert!((res - (1e-6_f32.sqrt() + 1.0)).abs() < 1e-6);
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
            prop_oneof![
                0.01_f32..100.0_f32,
                -100.0_f32..-0.01_f32,
            ],
            len,
        )
    }

    /// Generate (dim, vec_a, vec_b) where dim comes from DIMS.
    fn dim_and_two_vecs() -> impl Strategy<Value = (usize, Vec<f32>, Vec<f32>)> {
        simd_dim().prop_flat_map(|d| {
            (Just(d), bounded_vec(d), bounded_vec(d))
        })
    }

    /// Generate (dim, vec) where dim comes from DIMS.
    fn dim_and_vec() -> impl Strategy<Value = (usize, Vec<f32>)> {
        simd_dim().prop_flat_map(|d| (Just(d), bounded_vec(d)))
    }

    /// Generate (dim, vec_a, vec_b) with non-zero vectors.
    fn dim_and_two_nonzero_vecs() -> impl Strategy<Value = (usize, Vec<f32>, Vec<f32>)> {
        simd_dim().prop_flat_map(|d| {
            (Just(d), nonzero_vec(d), nonzero_vec(d))
        })
    }

    /// Generate (dim, vec) with non-zero vector.
    fn dim_and_nonzero_vec() -> impl Strategy<Value = (usize, Vec<f32>)> {
        simd_dim().prop_flat_map(|d| (Just(d), nonzero_vec(d)))
    }

    /// Generate (dim, vec_a, vec_b, vec_c) for three-vector properties.
    fn dim_and_three_vecs() -> impl Strategy<Value = (usize, Vec<f32>, Vec<f32>, Vec<f32>)> {
        simd_dim().prop_flat_map(|d| {
            (Just(d), bounded_vec(d), bounded_vec(d), bounded_vec(d))
        })
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
    }
}
