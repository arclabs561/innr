//! Sparse vector operations.
//!
//! Sparse vectors are represented as parallel arrays of sorted indices and values.
//! The merge-join algorithm computes dot products in O(|a| + |b|) time.

/// Sparse dot product for sorted index arrays.
///
/// Computes the inner product of two sparse vectors represented as
/// (indices, values) pairs. Indices must be sorted in ascending order.
///
/// # Algorithm
///
/// Uses merge-join: two pointers advance through sorted indices,
/// accumulating products when indices match. Time complexity O(|a| + |b|).
///
/// # Arguments
///
/// * `a_indices` - Sorted indices for vector a
/// * `a_values` - Values corresponding to a_indices
/// * `b_indices` - Sorted indices for vector b
/// * `b_values` - Values corresponding to b_indices
///
/// # Example
///
/// ```rust
/// use innr::sparse_dot;
///
/// // Sparse vectors: a = [1.0 at index 0, 2.0 at index 2]
/// //                 b = [3.0 at index 0, 4.0 at index 3]
/// // dot(a, b) = 1.0 * 3.0 = 3.0 (only index 0 overlaps)
/// let a_idx = [0u32, 2];
/// let a_val = [1.0f32, 2.0];
/// let b_idx = [0u32, 3];
/// let b_val = [3.0f32, 4.0];
///
/// let result = sparse_dot(&a_idx, &a_val, &b_idx, &b_val);
/// assert!((result - 3.0).abs() < 1e-6);
/// ```
#[inline]
#[must_use]
pub fn sparse_dot(a_indices: &[u32], a_values: &[f32], b_indices: &[u32], b_values: &[f32]) -> f32 {
    debug_assert_eq!(
        a_indices.len(),
        a_values.len(),
        "sparse_dot: a indices/values length mismatch"
    );
    debug_assert_eq!(
        b_indices.len(),
        b_values.len(),
        "sparse_dot: b indices/values length mismatch"
    );

    // For small vectors, use portable implementation directly
    // (SIMD overhead not worthwhile)
    sparse_dot_portable(a_indices, a_values, b_indices, b_values)
}

/// Portable sparse dot product (merge-join algorithm).
///
/// Time complexity: O(|a| + |b|)
/// Space complexity: O(1)
#[inline]
#[must_use]
pub fn sparse_dot_portable(
    a_indices: &[u32],
    a_values: &[f32],
    b_indices: &[u32],
    b_values: &[f32],
) -> f32 {
    let mut i = 0;
    let mut j = 0;
    let mut result = 0.0;

    // Merge-join: advance pointers based on index comparison
    while i < a_indices.len() && j < b_indices.len() {
        match a_indices[i].cmp(&b_indices[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                result += a_values[i] * b_values[j];
                i += 1;
                j += 1;
            }
        }
    }

    result
}

/// Sparse MaxSim (SPLADE-style) scoring.
///
/// Computes `Σᵢ max(w_q[i] * w_d[i])` or similar aggregation for sparse vectors.
///
/// For SPLADE, the score is typically just dot product of expanded vectors.
/// But for "Sparse ColBERT" (late interaction over sparse vectors), we need maxsim.
///
/// # Arguments
/// * `query_tokens` - List of sparse vectors for query tokens
/// * `doc_tokens` - List of sparse vectors for doc tokens
///
/// # Returns
/// Sum of max similarities.
pub fn sparse_maxsim(query_tokens: &[(&[u32], &[f32])], doc_tokens: &[(&[u32], &[f32])]) -> f32 {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }

    query_tokens
        .iter()
        .map(|(q_idx, q_val)| {
            doc_tokens
                .iter()
                .map(|(d_idx, d_val)| sparse_dot(q_idx, q_val, d_idx, d_val))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_dot_no_overlap() {
        let a_idx = [0u32, 2, 4];
        let a_val = [1.0f32, 2.0, 3.0];
        let b_idx = [1u32, 3, 5];
        let b_val = [4.0f32, 5.0, 6.0];

        let result = sparse_dot(&a_idx, &a_val, &b_idx, &b_val);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_sparse_dot_full_overlap() {
        let a_idx = [0u32, 1, 2];
        let a_val = [1.0f32, 2.0, 3.0];
        let b_idx = [0u32, 1, 2];
        let b_val = [4.0f32, 5.0, 6.0];

        // dot = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let result = sparse_dot(&a_idx, &a_val, &b_idx, &b_val);
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_dot_partial_overlap() {
        let a_idx = [0u32, 2, 4];
        let a_val = [1.0f32, 2.0, 3.0];
        let b_idx = [1u32, 2, 3];
        let b_val = [4.0f32, 5.0, 6.0];

        // Only index 2 overlaps: 2*5 = 10
        let result = sparse_dot(&a_idx, &a_val, &b_idx, &b_val);
        assert!((result - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_dot_empty() {
        let empty_idx: [u32; 0] = [];
        let empty_val: [f32; 0] = [];
        let a_idx = [0u32, 1];
        let a_val = [1.0f32, 2.0];

        assert_eq!(sparse_dot(&empty_idx, &empty_val, &a_idx, &a_val), 0.0);
        assert_eq!(sparse_dot(&a_idx, &a_val, &empty_idx, &empty_val), 0.0);
    }

    #[test]
    fn test_sparse_dot_different_lengths() {
        let a_idx = [0u32, 1, 2, 3, 4];
        let a_val = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b_idx = [2u32];
        let b_val = [10.0f32];

        // Only index 2 overlaps: 3*10 = 30
        let result = sparse_dot(&a_idx, &a_val, &b_idx, &b_val);
        assert!((result - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_maxsim_basic() {
        // Query: 2 tokens, Doc: 2 tokens
        let q1_idx = [0u32, 1];
        let q1_val = [1.0f32, 2.0];
        let q2_idx = [2u32, 3];
        let q2_val = [3.0f32, 4.0];

        let d1_idx = [0u32, 2];
        let d1_val = [0.5f32, 1.5];
        let d2_idx = [1u32, 3];
        let d2_val = [2.5f32, 3.5];

        let query = vec![(&q1_idx[..], &q1_val[..]), (&q2_idx[..], &q2_val[..])];
        let doc = vec![(&d1_idx[..], &d1_val[..]), (&d2_idx[..], &d2_val[..])];

        let result = sparse_maxsim(&query, &doc);

        // q1 vs d1: 1.0*0.5 = 0.5, q1 vs d2: 2.0*2.5 = 5.0 -> max = 5.0
        // q2 vs d1: 3.0*1.5 = 4.5, q2 vs d2: 4.0*3.5 = 14.0 -> max = 14.0
        // sum = 5.0 + 14.0 = 19.0
        assert!((result - 19.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_maxsim_empty_query() {
        let doc: Vec<(&[u32], &[f32])> = vec![(&[0u32][..], &[1.0f32][..])];
        let query: Vec<(&[u32], &[f32])> = vec![];
        assert_eq!(sparse_maxsim(&query, &doc), 0.0);
    }

    #[test]
    fn test_sparse_maxsim_empty_doc() {
        let query: Vec<(&[u32], &[f32])> = vec![(&[0u32][..], &[1.0f32][..])];
        let doc: Vec<(&[u32], &[f32])> = vec![];
        assert_eq!(sparse_maxsim(&query, &doc), 0.0);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Generate a sorted sparse vector with bounded values to avoid overflow.
    /// Values are in [-1000, 1000] to prevent multiplication overflow.
    fn arb_sparse_vec_bounded(
        max_len: usize,
        max_idx: u32,
    ) -> impl Strategy<Value = (Vec<u32>, Vec<f32>)> {
        prop::collection::vec(0..max_idx, 0..=max_len).prop_flat_map(move |mut indices| {
            // Sort and dedup to get unique sorted indices
            indices.sort_unstable();
            indices.dedup();
            let n = indices.len();
            // Use bounded floats to avoid overflow
            prop::collection::vec(-1000.0f32..1000.0f32, n)
                .prop_map(move |values| (indices.clone(), values))
        })
    }

    proptest! {
        /// Sparse dot is commutative: dot(a, b) == dot(b, a)
        #[test]
        fn sparse_dot_commutative(
            (a_idx, a_val) in arb_sparse_vec_bounded(20, 1000),
            (b_idx, b_val) in arb_sparse_vec_bounded(20, 1000)
        ) {
            let ab = sparse_dot(&a_idx, &a_val, &b_idx, &b_val);
            let ba = sparse_dot(&b_idx, &b_val, &a_idx, &a_val);

            // Commutative within tolerance, or both overflow in same direction
            let is_equal = (ab - ba).abs() < 1e-3 * ab.abs().max(ba.abs()).max(1.0);
            let both_inf = ab.is_infinite() && ba.is_infinite() && ab.signum() == ba.signum();
            let both_nan = ab.is_nan() && ba.is_nan();
            prop_assert!(is_equal || both_inf || both_nan,
                "ab={}, ba={}", ab, ba);
        }

        /// Sparse dot with self equals sum of squared values.
        #[test]
        fn sparse_dot_self_is_norm_squared(
            (idx, val) in arb_sparse_vec_bounded(50, 10000)
        ) {
            let result = sparse_dot(&idx, &val, &idx, &val);
            let expected: f32 = val.iter().map(|v| v * v).sum();

            // Allow for floating-point error proportional to magnitude
            let tolerance = 1e-4 * expected.abs().max(1.0);
            prop_assert!(
                (result - expected).abs() < tolerance,
                "result={}, expected={}, tolerance={}",
                result,
                expected,
                tolerance
            );
        }

        /// Sparse dot result is finite when inputs are bounded.
        #[test]
        fn sparse_dot_finite_result(
            (a_idx, a_val) in arb_sparse_vec_bounded(20, 1000),
            (b_idx, b_val) in arb_sparse_vec_bounded(20, 1000)
        ) {
            let result = sparse_dot(&a_idx, &a_val, &b_idx, &b_val);
            // With bounded inputs, result should be finite
            prop_assert!(result.is_finite(), "result was not finite: {}", result);
        }

        /// Sparse dot with disjoint indices is zero.
        #[test]
        fn sparse_dot_disjoint_is_zero(
            (idx, val) in arb_sparse_vec_bounded(20, 500)
        ) {
            // Shift b's indices by max(a) + 1 to ensure disjoint
            let shift = idx.iter().max().copied().unwrap_or(0) + 1;
            let b_idx: Vec<u32> = idx.iter().map(|i| i + shift).collect();

            let result = sparse_dot(&idx, &val, &b_idx, &val);
            prop_assert_eq!(result, 0.0);
        }

        /// Sparse maxsim is non-negative when all values are positive.
        #[test]
        fn sparse_maxsim_nonnegative_for_positive_values(
            n_query in 1usize..5,
            n_doc in 1usize..5
        ) {
            // Generate positive-only values
            let mut query_tokens = Vec::new();
            let mut doc_tokens = Vec::new();

            for i in 0..n_query {
                query_tokens.push((vec![i as u32], vec![1.0f32]));
            }
            for i in 0..n_doc {
                doc_tokens.push((vec![i as u32], vec![1.0f32]));
            }

            let query: Vec<(&[u32], &[f32])> = query_tokens
                .iter()
                .map(|(idx, val)| (idx.as_slice(), val.as_slice()))
                .collect();
            let doc: Vec<(&[u32], &[f32])> = doc_tokens
                .iter()
                .map(|(idx, val)| (idx.as_slice(), val.as_slice()))
                .collect();

            let result = sparse_maxsim(&query, &doc);
            prop_assert!(result >= 0.0, "sparse_maxsim should be non-negative for positive values");
        }
    }
}
