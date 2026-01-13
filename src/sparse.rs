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
}
