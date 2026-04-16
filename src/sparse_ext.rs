//! Sparse vector primitives for learned sparse retrieval.
//!
//! Sparse vectors are represented as sorted `(dimension, weight)` pairs.
//! All operations assume inputs are sorted by dimension.

/// Sparse dot product between two sorted sparse vectors.
/// Both inputs must be sorted by dimension (first element of tuple).
pub fn sparse_dot(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    let (mut i, mut j) = (0, 0);
    let mut sum = 0.0f32;
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                sum += a[i].1 * b[j].1;
                i += 1;
                j += 1;
            }
        }
    }
    sum
}

/// Sparse dot product with a dense vector.
/// Sparse input must be sorted by dimension. Dense vector is indexed directly.
pub fn sparse_dense_dot(sparse: &[(u32, f32)], dense: &[f32]) -> f32 {
    sparse
        .iter()
        .filter(|(dim, _)| (*dim as usize) < dense.len())
        .map(|(dim, weight)| weight * dense[*dim as usize])
        .sum()
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
