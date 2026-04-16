//! Sparse vector primitives for learned sparse retrieval.
//!
//! Sparse vectors are represented as sorted `(dimension, weight)` pairs.
//! All operations assume inputs are sorted by dimension.

#![allow(unsafe_code)]

/// Sparse dot product between two sorted sparse vectors.
/// Both inputs must be sorted by dimension (first element of tuple).
///
/// Uses a branch-restructured merge-join: integer comparisons feed directly
/// into pointer advances without an intermediate enum, which eliminates the
/// match-dispatch overhead and lets the CPU speculate on the more common
/// non-equal branch. A four-way independent accumulator hides FP latency.
#[inline]
pub fn sparse_dot(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    let mut i = 0;
    let mut j = 0;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut acc_idx = 0usize;

    // SAFETY: bounds are checked by the while condition and the index advances below.
    while i < a.len() && j < b.len() {
        // SAFETY: i < a.len() and j < b.len() are guaranteed by the while condition.
        let ai = unsafe { a.get_unchecked(i) };
        let bj = unsafe { b.get_unchecked(j) };
        let ai_dim = ai.0;
        let bj_dim = bj.0;

        // Use branch-free pointer advances: advance the smaller index (or both on equal).
        // The multiply is speculative (produces 0 on mismatch when we don't add it),
        // but we only accumulate on a match to keep semantics correct.
        if ai_dim == bj_dim {
            let prod = ai.1 * bj.1;
            // Round-robin into 4 independent accumulators to break FP latency chain.
            match acc_idx & 3 {
                0 => s0 += prod,
                1 => s1 += prod,
                2 => s2 += prod,
                _ => s3 += prod,
            }
            acc_idx += 1;
            i += 1;
            j += 1;
        } else if ai_dim < bj_dim {
            i += 1;
        } else {
            j += 1;
        }
    }

    s0 + s1 + s2 + s3
}

/// Sparse dot product with a dense vector.
/// Sparse input must be sorted by dimension. Dense vector is indexed directly.
///
/// Uses four independent accumulators to hide FP latency. When all sparse
/// dimensions are within bounds (the common case), the fast path skips the
/// per-element bounds check by pre-verifying the maximum dimension once.
#[inline]
pub fn sparse_dense_dot(sparse: &[(u32, f32)], dense: &[f32]) -> f32 {
    if sparse.is_empty() || dense.is_empty() {
        return 0.0;
    }

    let dense_len = dense.len();

    // Fast path: if the largest dimension in the sparse vector fits in dense,
    // no per-element bounds check is needed.
    // SAFETY: sparse is non-empty (checked above), so last() is always Some.
    let max_dim = unsafe { sparse.last().unwrap_unchecked() }.0 as usize;

    if max_dim < dense_len {
        // All indices are in-bounds: skip the filter, use unsafe indexing.
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let chunks = sparse.len() / 4;

        for c in 0..chunks {
            let base = c * 4;
            // SAFETY: base + 3 < sparse.len() because chunks = sparse.len() / 4.
            let (d0, w0) = unsafe { *sparse.get_unchecked(base) };
            let (d1, w1) = unsafe { *sparse.get_unchecked(base + 1) };
            let (d2, w2) = unsafe { *sparse.get_unchecked(base + 2) };
            let (d3, w3) = unsafe { *sparse.get_unchecked(base + 3) };
            // SAFETY: max_dim < dense_len and all dims <= max_dim (sorted input).
            s0 += w0 * unsafe { *dense.get_unchecked(d0 as usize) };
            s1 += w1 * unsafe { *dense.get_unchecked(d1 as usize) };
            s2 += w2 * unsafe { *dense.get_unchecked(d2 as usize) };
            s3 += w3 * unsafe { *dense.get_unchecked(d3 as usize) };
        }

        // Scalar tail.
        let tail_start = chunks * 4;
        let mut tail = 0.0f32;
        for k in tail_start..sparse.len() {
            let (dim, weight) = unsafe { *sparse.get_unchecked(k) };
            tail += weight * unsafe { *dense.get_unchecked(dim as usize) };
        }

        s0 + s1 + s2 + s3 + tail
    } else {
        // Slow path: at least one dimension might be out of bounds.
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;

        let chunks = sparse.len() / 4;
        for c in 0..chunks {
            let base = c * 4;
            // SAFETY: base + 3 < sparse.len() because chunks = sparse.len() / 4.
            let (d0, w0) = unsafe { *sparse.get_unchecked(base) };
            let (d1, w1) = unsafe { *sparse.get_unchecked(base + 1) };
            let (d2, w2) = unsafe { *sparse.get_unchecked(base + 2) };
            let (d3, w3) = unsafe { *sparse.get_unchecked(base + 3) };
            if (d0 as usize) < dense_len {
                s0 += w0 * unsafe { *dense.get_unchecked(d0 as usize) };
            }
            if (d1 as usize) < dense_len {
                s1 += w1 * unsafe { *dense.get_unchecked(d1 as usize) };
            }
            if (d2 as usize) < dense_len {
                s2 += w2 * unsafe { *dense.get_unchecked(d2 as usize) };
            }
            if (d3 as usize) < dense_len {
                s3 += w3 * unsafe { *dense.get_unchecked(d3 as usize) };
            }
        }

        let tail_start = chunks * 4;
        let mut tail = 0.0f32;
        for k in tail_start..sparse.len() {
            let (dim, weight) = unsafe { *sparse.get_unchecked(k) };
            if (dim as usize) < dense_len {
                tail += weight * unsafe { *dense.get_unchecked(dim as usize) };
            }
        }

        s0 + s1 + s2 + s3 + tail
    }
}

/// L2 norm of a sparse vector.
pub fn sparse_l2_norm(v: &[(u32, f32)]) -> f32 {
    v.iter().map(|(_, w)| w * w).sum::<f32>().sqrt()
}

/// Normalize a sparse vector to unit L2 norm in-place.
pub fn sparse_normalize(v: &mut [(u32, f32)]) {
    let norm = sparse_l2_norm(v);
    if norm > 0.0 {
        for (_, w) in v.iter_mut() {
            *w /= norm;
        }
    }
}

/// Keep only the top-k entries by absolute weight.
/// Returns a new vector sorted by dimension.
pub fn sparse_top_k(v: &[(u32, f32)], k: usize) -> Vec<(u32, f32)> {
    if v.len() <= k {
        return v.to_vec();
    }
    let mut by_weight: Vec<_> = v.to_vec();
    by_weight.sort_by(|a, b| {
        b.1.abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    by_weight.truncate(k);
    by_weight.sort_by_key(|(dim, _)| *dim);
    by_weight
}

/// Maximum weight in a sparse vector.
pub fn sparse_max_weight(v: &[(u32, f32)]) -> f32 {
    v.iter().map(|(_, w)| *w).fold(0.0f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_dot_no_overlap() {
        let a = [(0u32, 1.0f32), (2, 2.0)];
        let b = [(1u32, 3.0f32), (3, 4.0)];
        assert_eq!(sparse_dot(&a, &b), 0.0);
    }

    #[test]
    fn test_sparse_dot_full_overlap() {
        let a = [(0u32, 1.0f32), (1, 2.0), (2, 3.0)];
        let b = [(0u32, 4.0f32), (1, 5.0), (2, 6.0)];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((sparse_dot(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_dot_partial_overlap() {
        let a = [(0u32, 1.0f32), (2, 2.0), (4, 3.0)];
        let b = [(1u32, 4.0f32), (2, 5.0), (3, 6.0)];
        // Only index 2 overlaps: 2*5 = 10
        assert!((sparse_dot(&a, &b) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_dot_empty() {
        let a = [(0u32, 1.0f32), (1, 2.0)];
        assert_eq!(sparse_dot(&[], &a), 0.0);
        assert_eq!(sparse_dot(&a, &[]), 0.0);
        assert_eq!(sparse_dot(&[] as &[(u32, f32)], &[]), 0.0);
    }

    #[test]
    fn test_sparse_dense_dot_basic() {
        let sparse = [(0u32, 2.0f32), (2, 3.0)];
        let dense = [1.0f32, 0.0, 4.0, 0.0];
        // 2*1 + 3*4 = 2 + 12 = 14
        assert!((sparse_dense_dot(&sparse, &dense) - 14.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_dense_dot_out_of_bounds_dim() {
        // Dimension 10 is out of bounds for a length-4 dense vector; should be skipped.
        let sparse = [(1u32, 1.0f32), (10, 99.0)];
        let dense = [0.0f32, 5.0, 0.0, 0.0];
        assert!((sparse_dense_dot(&sparse, &dense) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_dense_dot_empty_sparse() {
        let dense = [1.0f32, 2.0, 3.0];
        assert_eq!(sparse_dense_dot(&[], &dense), 0.0);
    }

    #[test]
    fn test_sparse_dense_dot_empty_dense() {
        let sparse = [(0u32, 1.0f32)];
        assert_eq!(sparse_dense_dot(&sparse, &[]), 0.0);
    }

    #[test]
    fn test_sparse_l2_norm_basic() {
        let v = [(0u32, 3.0f32), (1, 4.0)];
        assert!((sparse_l2_norm(&v) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_l2_norm_empty() {
        assert_eq!(sparse_l2_norm(&[]), 0.0);
    }

    #[test]
    fn test_sparse_normalize_unit() {
        let mut v = [(0u32, 3.0f32), (1, 4.0)];
        sparse_normalize(&mut v);
        assert!((sparse_l2_norm(&v) - 1.0).abs() < 1e-6);
        assert!((v[0].1 - 0.6).abs() < 1e-6);
        assert!((v[1].1 - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_normalize_zero_vector() {
        let mut v = [(0u32, 0.0f32), (1, 0.0)];
        sparse_normalize(&mut v); // should not divide by zero
        assert_eq!(v[0].1, 0.0);
        assert_eq!(v[1].1, 0.0);
    }

    #[test]
    fn test_sparse_top_k_fewer_than_k() {
        let v = [(0u32, 1.0f32), (1, 2.0)];
        let result = sparse_top_k(&v, 5);
        assert_eq!(result, v.to_vec());
    }

    #[test]
    fn test_sparse_top_k_basic() {
        let v = [(0u32, 0.5f32), (1, 3.0), (2, 1.0), (3, 2.5)];
        let result = sparse_top_k(&v, 2);
        // top-2 by abs weight: (1, 3.0) and (3, 2.5), sorted by dim
        assert_eq!(result, vec![(1u32, 3.0f32), (3, 2.5)]);
    }

    #[test]
    fn test_sparse_top_k_negative_weights() {
        let v = [(0u32, -4.0f32), (1, 1.0), (2, -2.0)];
        let result = sparse_top_k(&v, 2);
        // top-2 by abs: (0, -4.0) and (2, -2.0), sorted by dim
        assert_eq!(result, vec![(0u32, -4.0f32), (2, -2.0)]);
    }

    #[test]
    fn test_sparse_top_k_empty() {
        let result = sparse_top_k(&[], 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sparse_max_weight_basic() {
        let v = [(0u32, 1.0f32), (1, 3.0), (2, 2.0)];
        assert!((sparse_max_weight(&v) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_max_weight_empty() {
        // Empty vector: fold starts at 0.0
        assert_eq!(sparse_max_weight(&[]), 0.0);
    }

    #[test]
    fn test_sparse_max_weight_all_negative() {
        let v = [(0u32, -1.0f32), (1, -2.0)];
        // max of negative weights with 0.0 seed: returns 0.0
        assert_eq!(sparse_max_weight(&v), 0.0);
    }
}
