//! Dense vector operations with SIMD acceleration.
//!
//! Core operations: dot product, norm, cosine similarity, L2 distance.

use crate::arch;
use crate::{MIN_DIM_SIMD, NORM_EPSILON};

/// Dot product of two vectors: `Σ(a[i] * b[i])`.
///
/// Returns 0.0 for empty vectors.
///
/// # SIMD Acceleration
///
/// Automatically dispatches to:
/// - AVX2+FMA on x86_64 (runtime detection)
/// - NEON on aarch64 (always available)
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
    let n = a.len().min(b.len());

    #[cfg(target_arch = "x86_64")]
    {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
