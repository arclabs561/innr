//! Portable `f64` vector primitives.
//!
//! Mirrors the `f32` API in [`crate::dense`] for code paths that need higher
//! precision (scientific computing, PageRank-style accumulation, statistical
//! reductions). Portable implementations only -- explicit SIMD acceleration
//! for these is tracked as a follow-up.
//!
//! The 4-way unrolled accumulator pattern from the `f32` portable path is
//! preserved so LLVM auto-vectorization still kicks in on most targets.

/// Dot product of two `f64` vectors: `Σ(a[i] * b[i])`.
///
/// Returns 0.0 for empty slices.
///
/// # Example
///
/// ```rust
/// use innr::dense_f64::dot_f64;
///
/// let a = [1.0, 2.0, 3.0];
/// let b = [4.0, 5.0, 6.0];
/// assert!((dot_f64(&a, &b) - 32.0).abs() < 1e-12);
/// ```
#[inline]
#[must_use]
pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let chunks = n / 4;

    let mut s0 = 0.0f64;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    let mut s3 = 0.0f64;

    for i in 0..chunks {
        let base = i * 4;
        s0 += a[base] * b[base];
        s1 += a[base + 1] * b[base + 1];
        s2 += a[base + 2] * b[base + 2];
        s3 += a[base + 3] * b[base + 3];
    }

    let mut result = s0 + s1 + s2 + s3;
    for i in (chunks * 4)..n {
        result += a[i] * b[i];
    }
    result
}

/// L2 norm of an `f64` vector: `sqrt(Σ(v[i]²))`.
///
/// # Example
///
/// ```rust
/// use innr::dense_f64::norm_f64;
///
/// let v = [3.0, 4.0];
/// assert!((norm_f64(&v) - 5.0).abs() < 1e-12);
/// ```
#[inline]
#[must_use]
pub fn norm_f64(v: &[f64]) -> f64 {
    dot_f64(v, v).sqrt()
}

/// Normalize an `f64` vector to unit length in place.
///
/// Zero vectors are left unchanged (no division by zero). Returns the original
/// norm so callers can detect a zero-vector case if needed.
pub fn normalize_f64(v: &mut [f64]) -> f64 {
    let n = norm_f64(v);
    if n > f64::EPSILON {
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
    n
}

/// Cosine similarity between two `f64` vectors.
///
/// Returns 0.0 if either vector has zero norm (no division by zero).
///
/// # Example
///
/// ```rust
/// use innr::dense_f64::cosine_f64;
///
/// let a = [1.0, 0.0];
/// let b = [0.0, 1.0];
/// assert!(cosine_f64(&a, &b).abs() < 1e-12);
///
/// let c = [1.0, 1.0];
/// assert!((cosine_f64(&a, &c) - (1.0 / 2.0_f64.sqrt())).abs() < 1e-12);
/// ```
#[inline]
#[must_use]
pub fn cosine_f64(a: &[f64], b: &[f64]) -> f64 {
    let na = norm_f64(a);
    let nb = norm_f64(b);
    if na <= f64::EPSILON || nb <= f64::EPSILON {
        return 0.0;
    }
    dot_f64(a, b) / (na * nb)
}

/// Squared Euclidean distance: `Σ((a[i] - b[i])²)`.
///
/// Cheaper than `l2_distance_f64` when only relative comparisons are needed
/// (skips the square root).
#[inline]
#[must_use]
pub fn l2_distance_squared_f64(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let chunks = n / 4;

    let mut s0 = 0.0f64;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    let mut s3 = 0.0f64;

    for i in 0..chunks {
        let base = i * 4;
        let d0 = a[base] - b[base];
        let d1 = a[base + 1] - b[base + 1];
        let d2 = a[base + 2] - b[base + 2];
        let d3 = a[base + 3] - b[base + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }

    let mut result = s0 + s1 + s2 + s3;
    for i in (chunks * 4)..n {
        let d = a[i] - b[i];
        result += d * d;
    }
    result
}

/// L2 (Euclidean) distance: `sqrt(Σ((a[i] - b[i])²))`.
///
/// # Example
///
/// ```rust
/// use innr::dense_f64::l2_distance_f64;
///
/// let a = [0.0, 0.0];
/// let b = [3.0, 4.0];
/// assert!((l2_distance_f64(&a, &b) - 5.0).abs() < 1e-12);
/// ```
#[inline]
#[must_use]
pub fn l2_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    l2_distance_squared_f64(a, b).sqrt()
}

/// L1 (Manhattan / sum-of-absolute-differences) distance.
///
/// Useful as the convergence criterion for iterative methods (PageRank, k-means).
#[inline]
#[must_use]
pub fn l1_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let chunks = n / 4;

    let mut s0 = 0.0f64;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    let mut s3 = 0.0f64;

    for i in 0..chunks {
        let base = i * 4;
        s0 += (a[base] - b[base]).abs();
        s1 += (a[base + 1] - b[base + 1]).abs();
        s2 += (a[base + 2] - b[base + 2]).abs();
        s3 += (a[base + 3] - b[base + 3]).abs();
    }

    let mut result = s0 + s1 + s2 + s3;
    for i in (chunks * 4)..n {
        result += (a[i] - b[i]).abs();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_basic() {
        assert!((dot_f64(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-12);
        assert_eq!(dot_f64(&[], &[]), 0.0);
        assert_eq!(dot_f64(&[1.0], &[]), 0.0);
    }

    #[test]
    fn dot_unrolled_tail() {
        // Length not divisible by 4 -- exercise the tail loop.
        let a: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..10).map(|i| (i * 2) as f64).collect();
        let expected: f64 = (0..10).map(|i| (i as f64) * (i * 2) as f64).sum();
        assert!((dot_f64(&a, &b) - expected).abs() < 1e-12);
    }

    #[test]
    fn norm_basic() {
        assert!((norm_f64(&[3.0, 4.0]) - 5.0).abs() < 1e-12);
        assert_eq!(norm_f64(&[]), 0.0);
        assert_eq!(norm_f64(&[0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn normalize_basic() {
        let mut v = vec![3.0, 4.0];
        let n = normalize_f64(&mut v);
        assert!((n - 5.0).abs() < 1e-12);
        assert!((norm_f64(&v) - 1.0).abs() < 1e-12);

        // Zero vector left unchanged.
        let mut zero = vec![0.0, 0.0, 0.0];
        let zn = normalize_f64(&mut zero);
        assert_eq!(zn, 0.0);
        assert_eq!(zero, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn cosine_orthogonal_and_parallel() {
        assert!(cosine_f64(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-12);
        assert!((cosine_f64(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-12);
        assert!((cosine_f64(&[1.0, 2.0, 3.0], &[-1.0, -2.0, -3.0]) - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn cosine_zero_vector() {
        assert_eq!(cosine_f64(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine_f64(&[1.0, 0.0], &[0.0, 0.0]), 0.0);
    }

    #[test]
    fn l2_distance_basic() {
        assert!((l2_distance_f64(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 1e-12);
        assert_eq!(l2_distance_f64(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn l2_distance_squared_matches_dot() {
        // ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        let a = [1.5, -2.3, 7.1, 0.4];
        let b = [-0.2, 3.1, 4.0, 1.9];
        let na2 = dot_f64(&a, &a);
        let nb2 = dot_f64(&b, &b);
        let ab = dot_f64(&a, &b);
        let expected = na2 + nb2 - 2.0 * ab;
        assert!((l2_distance_squared_f64(&a, &b) - expected).abs() < 1e-10);
    }

    #[test]
    fn l1_distance_basic() {
        assert!((l1_distance_f64(&[0.0, 0.0, 0.0], &[1.0, -2.0, 3.0]) - 6.0).abs() < 1e-12);
        assert_eq!(l1_distance_f64(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    }
}
