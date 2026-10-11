//! Sparse vector primitives for learned sparse retrieval.
//!
//! Sparse vectors are represented as sorted `(dimension, weight)` pairs.
//! All operations assume inputs are sorted by dimension.

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

    while let (Some(ai), Some(bj)) = (a.get(i), b.get(j)) {
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
/// per-element filter by pre-verifying the maximum dimension once.
#[inline]
pub fn sparse_dense_dot(sparse: &[(u32, f32)], dense: &[f32]) -> f32 {
    if sparse.is_empty() || dense.is_empty() {
        return 0.0;
    }

    let dense_len = dense.len();

    // True max over all entries, not last(): input may be unsorted.
    let max_dim = sparse.iter().map(|&(d, _)| d).max().unwrap() as usize;

    let (quads, tail) = sparse.as_chunks::<4>();
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut rest = 0.0f32;

    if max_dim < dense_len {
        // Every dim is in bounds, so the indexing below never panics and the
        // loop needs no per-element filter.
        for [(d0, w0), (d1, w1), (d2, w2), (d3, w3)] in quads {
            s0 += w0 * dense[*d0 as usize];
            s1 += w1 * dense[*d1 as usize];
            s2 += w2 * dense[*d2 as usize];
            s3 += w3 * dense[*d3 as usize];
        }
        for (dim, weight) in tail {
            rest += weight * dense[*dim as usize];
        }
    } else {
        // Slow path: at least one dimension is out of bounds; skip those.
        let at = |d: u32| dense.get(d as usize).copied();
        for [(d0, w0), (d1, w1), (d2, w2), (d3, w3)] in quads {
            if let Some(x) = at(*d0) {
                s0 += w0 * x;
            }
            if let Some(x) = at(*d1) {
                s1 += w1 * x;
            }
            if let Some(x) = at(*d2) {
                s2 += w2 * x;
            }
            if let Some(x) = at(*d3) {
                s3 += w3 * x;
            }
        }
        for (dim, weight) in tail {
            if let Some(x) = at(*dim) {
                rest += weight * x;
            }
        }
    }

    s0 + s1 + s2 + s3 + rest
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
    by_weight.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
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
    /// NaN weights must not panic the top-k sort (Rust >= 1.81 may panic on
    /// a comparator that is not a total order).
    #[test]
    fn sparse_top_k_tolerates_nan() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for _ in 0..50 {
            let v: Vec<(u32, f32)> = (0..200u32)
                .map(|i| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let w = if state.is_multiple_of(10) {
                        f32::NAN
                    } else {
                        (state % 1000) as f32 / 100.0 - 5.0
                    };
                    (i, w)
                })
                .collect();
            let top = super::sparse_top_k(&v, 20);
            assert_eq!(top.len(), 20);
        }
    }

    #[test]
    fn unsorted_sparse_input_is_safe_and_correct() {
        // Regression: fast path took last() as max, sound only for sorted input;
        // unsorted reached out-of-bounds unchecked indexing from safe code.
        let sparse = [(10u32, 1.0f32), (0u32, 2.0f32)];
        let dense = [3.0f32];
        // dim 10 is out of bounds for dense: slow path must skip it.
        assert_eq!(super::sparse_dense_dot(&sparse, &dense), 6.0);

        // Unsorted but fully in-bounds: fast path must compute the true max.
        let sparse = [(3u32, 1.0f32), (0u32, 2.0f32)];
        let dense = [1.0f32, 0.0, 0.0, 4.0];
        assert_eq!(super::sparse_dense_dot(&sparse, &dense), 6.0);
    }

    use super::*;

    /// Both paths of `sparse_dense_dot`, and `sparse_dot`, against a naive
    /// reference at lengths that hit the 4-wide body and the tail.
    #[test]
    fn sparse_kernels_match_naive_on_odd_lengths() {
        for n in [0usize, 1, 7, 8, 9, 31, 33] {
            let a: Vec<(u32, f32)> = (0..n as u32)
                .map(|i| (i * 3, i as f32 * 0.5 - 2.0))
                .collect();
            let b: Vec<(u32, f32)> = (0..n as u32)
                .map(|i| (i * 2, i as f32 * 0.25 + 1.0))
                .collect();
            let dense: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
            // Every third dim of `a` is 3*i < 3*33, so dense of len 64 forces
            // the slow path once n > 22; dense of len 128 always takes the fast path.
            let big: Vec<f32> = (0..128).map(|i| (i as f32).cos()).collect();

            let naive_sd = |d: &[f32]| -> f32 {
                a.iter()
                    .filter_map(|&(i, w)| d.get(i as usize).map(|x| w * x))
                    .sum()
            };
            let naive_ss: f32 = a
                .iter()
                .filter_map(|&(i, w)| b.iter().find(|p| p.0 == i).map(|p| w * p.1))
                .sum();

            for d in [&dense, &big] {
                let got = sparse_dense_dot(&a, d);
                assert!(
                    (got - naive_sd(d)).abs() < 1e-4,
                    "dense len={} n={n}",
                    d.len()
                );
            }
            assert!(
                (sparse_dot(&a, &b) - naive_ss).abs() < 1e-4,
                "sparse_dot n={n}"
            );
        }
    }

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
