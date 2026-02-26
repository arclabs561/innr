//! Batch vector operations with columnar (PDX-style) data layout.
//!
//! # The Data Layout Problem
//!
//! Traditional "horizontal" layouts store vectors contiguously:
//!
//! ```text
//! Horizontal (AoS - Array of Structs):
//! [v0_d0, v0_d1, v0_d2, v0_d3] [v1_d0, v1_d1, v1_d2, v1_d3] ...
//!        vector 0                      vector 1
//! ```
//!
//! This is natural but suboptimal for batch operations. When computing
//! distances to many vectors, we access memory non-sequentially.
//!
//! # Vertical (PDX) Layout
//!
//! Columnar layouts store dimensions contiguously across vectors:
//!
//! ```text
//! Vertical (SoA - Struct of Arrays):
//! [v0_d0, v1_d0, v2_d0, ...] [v0_d1, v1_d1, v2_d1, ...] ...
//!        dimension 0                dimension 1
//! ```
//!
//! # References
//!
//! - Kuffo, Krippner, Boncz (2025, SIGMOD), "PDX: A Data Layout for Vector
//!   Similarity Search" -- the canonical paper for this columnar layout. Shows
//!   that transposing storage from row-major vectors to column-major dimensions
//!   enables better SIMD utilization and early termination for batch comparisons.
//!
//! # Why This Matters
//!
//! 1. **Cache efficiency**: Processing dimension-by-dimension keeps working
//!    set small and predictable.
//!
//! 2. **Auto-vectorization**: Tight loops over contiguous memory vectorize
//!    cleanly without gather operations.
//!
//! 3. **Early termination**: Partial distances accumulate dimension-by-dimension,
//!    enabling pruning before computing all dimensions.
//!
//! # Research Context
//!
//! This pattern appears in:
//! - PDX (SIGMOD 2025): "Partition Dimensions Across"
//! - FINGER: Angle estimation via low-rank projections
//! - ADSampling: Adaptive dimension sampling for pruning
//!
//! All share the insight that dimension-at-a-time processing enables
//! optimizations impossible with vector-at-a-time.
//!
//! # Block Size
//!
//! We process vectors in blocks (typically 8-32). This balances:
//! - Register pressure (too many → spills)
//! - Loop overhead (too few → iteration cost)
//! - SIMD width (AVX2 = 8 floats, AVX-512 = 16 floats)

/// Default block size for batch operations.
/// 8 floats = one AVX2 register, good balance for most CPUs.
pub const DEFAULT_BLOCK_SIZE: usize = 8;

/// Vertical (columnar) storage for a batch of vectors.
///
/// Stores vectors in dimension-major order for efficient batch processing.
///
/// # Memory Layout
///
/// For N vectors of dimension D:
/// ```text
/// data[d * N + i] = vector i, dimension d
/// ```
///
/// # Example
///
/// ```rust
/// use innr::batch::VerticalBatch;
///
/// let vectors = vec![
///     vec![1.0, 2.0, 3.0],  // v0
///     vec![4.0, 5.0, 6.0],  // v1
/// ];
/// let batch = VerticalBatch::from_rows(&vectors);
///
/// // Access is (dimension, vector_index)
/// assert_eq!(batch.get(0, 0), 1.0);  // v0[0]
/// assert_eq!(batch.get(0, 1), 4.0);  // v1[0]
/// ```
#[derive(Clone, Debug)]
pub struct VerticalBatch {
    /// Data in dimension-major order: data[d * num_vectors + i]
    data: Vec<f32>,
    /// Number of vectors in the batch
    num_vectors: usize,
    /// Dimension of each vector
    dimension: usize,
}

impl VerticalBatch {
    /// Create from row-major (horizontal) vectors.
    ///
    /// # Panics
    ///
    /// Panics if vectors have inconsistent dimensions.
    pub fn from_rows(vectors: &[Vec<f32>]) -> Self {
        if vectors.is_empty() {
            return Self {
                data: Vec::new(),
                num_vectors: 0,
                dimension: 0,
            };
        }

        let dimension = vectors[0].len();
        let num_vectors = vectors.len();

        // Pre-allocate with exact size
        let mut data = vec![0.0f32; dimension * num_vectors];

        // Transpose: row-major → column-major
        for (i, vec) in vectors.iter().enumerate() {
            debug_assert_eq!(vec.len(), dimension, "Inconsistent vector dimension");
            for (d, &val) in vec.iter().enumerate() {
                data[d * num_vectors + i] = val;
            }
        }

        Self {
            data,
            num_vectors,
            dimension,
        }
    }

    /// Create from flat row-major data.
    pub fn from_flat(data: &[f32], num_vectors: usize, dimension: usize) -> Self {
        debug_assert_eq!(data.len(), num_vectors * dimension);

        let mut vertical = vec![0.0f32; dimension * num_vectors];

        for i in 0..num_vectors {
            for d in 0..dimension {
                vertical[d * num_vectors + i] = data[i * dimension + d];
            }
        }

        Self {
            data: vertical,
            num_vectors,
            dimension,
        }
    }

    /// Get value at (dimension, vector_index).
    #[inline]
    pub fn get(&self, dim: usize, vec_idx: usize) -> f32 {
        self.data[dim * self.num_vectors + vec_idx]
    }

    /// Get value at (dimension, vector_index) without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller must ensure `dim < self.dimension` and `vec_idx < self.num_vectors`.
    #[inline]
    pub unsafe fn get_unchecked(&self, dim: usize, vec_idx: usize) -> f32 {
        *self.data.get_unchecked(dim * self.num_vectors + vec_idx)
    }

    /// Get slice for a single dimension across all vectors.
    #[inline]
    pub fn dimension_slice(&self, dim: usize) -> &[f32] {
        let start = dim * self.num_vectors;
        &self.data[start..start + self.num_vectors]
    }

    /// Number of vectors.
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    /// Dimension of vectors.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Extract a single vector (allocates).
    pub fn extract_vector(&self, vec_idx: usize) -> Vec<f32> {
        (0..self.dimension).map(|d| self.get(d, vec_idx)).collect()
    }
}

/// Compute squared L2 distances from query to all vectors in batch.
///
/// Uses vertical layout for efficient dimension-by-dimension processing.
///
/// # Algorithm
///
/// ```text
/// for each dimension d:
///     for each vector i:
///         partial_dist[i] += (query[d] - batch[d][i])^2
/// ```
///
/// This processes memory sequentially and auto-vectorizes well.
pub fn batch_l2_squared(query: &[f32], batch: &VerticalBatch) -> Vec<f32> {
    debug_assert_eq!(query.len(), batch.dimension);

    let mut distances = vec![0.0f32; batch.num_vectors];

    // Process dimension-by-dimension (the key insight)
    for (d, &q_d) in query.iter().enumerate().take(batch.dimension) {
        let dim_slice = batch.dimension_slice(d);

        // This loop auto-vectorizes cleanly
        for (dist, &v_d) in distances.iter_mut().zip(dim_slice.iter()) {
            let diff = q_d - v_d;
            *dist += diff * diff;
        }
    }

    distances
}

/// Compute dot products from query to all vectors in batch.
pub fn batch_dot(query: &[f32], batch: &VerticalBatch) -> Vec<f32> {
    debug_assert_eq!(query.len(), batch.dimension);

    let mut products = vec![0.0f32; batch.num_vectors];

    for (d, &q_d) in query.iter().enumerate().take(batch.dimension) {
        let dim_slice = batch.dimension_slice(d);

        for (prod, &v_d) in products.iter_mut().zip(dim_slice.iter()) {
            *prod += q_d * v_d;
        }
    }

    products
}

/// Compute squared L2 distances with early termination.
///
/// Stops computing distance to a vector once it exceeds `threshold`.
/// Returns indices of vectors that survived (distance <= threshold).
///
/// # Why This Works
///
/// Distance accumulates monotonically as we process dimensions:
/// ```text
/// d=0: partial_dist[i] = (q[0] - v[0])^2
/// d=1: partial_dist[i] += (q[1] - v[1])^2
/// ...
/// ```
///
/// If `partial_dist[i] > threshold` at dimension d, the final distance
/// can only be larger, so we can skip remaining dimensions for vector i.
///
/// # Returns
///
/// Vector of (index, squared_distance) pairs for vectors within threshold.
pub fn batch_l2_squared_pruning(
    query: &[f32],
    batch: &VerticalBatch,
    threshold: f32,
) -> Vec<(usize, f32)> {
    debug_assert_eq!(query.len(), batch.dimension);

    let mut distances = vec![0.0f32; batch.num_vectors];
    let mut alive: Vec<bool> = vec![true; batch.num_vectors];
    let mut num_alive = batch.num_vectors;

    // Process dimension-by-dimension with pruning
    for (d, &q_d) in query.iter().enumerate().take(batch.dimension) {
        if num_alive == 0 {
            break;
        }

        let dim_slice = batch.dimension_slice(d);

        for (&v_d, (dist, is_alive)) in dim_slice
            .iter()
            .zip(distances.iter_mut().zip(alive.iter_mut()))
        {
            if !*is_alive {
                continue;
            }

            let diff = q_d - v_d;
            *dist += diff * diff;

            // Prune if exceeded threshold
            if *dist > threshold {
                *is_alive = false;
                num_alive -= 1;
            }
        }
    }

    // Collect survivors
    alive
        .iter()
        .enumerate()
        .filter(|(_, &a)| a)
        .map(|(i, _)| (i, distances[i]))
        .collect()
}

/// Result of batch k-nearest neighbor search.
#[derive(Clone, Debug)]
pub struct BatchKnnResult {
    /// Indices of the k nearest vectors, sorted by distance.
    pub indices: Vec<usize>,
    /// Squared distances to the k nearest vectors.
    pub distances: Vec<f32>,
}

/// Find k nearest neighbors in a batch using PDX-style processing.
///
/// Uses incremental distance computation with pruning: once we have k
/// candidates, we can skip vectors whose partial distance exceeds the
/// current k-th best.
pub fn batch_knn(query: &[f32], batch: &VerticalBatch, k: usize) -> BatchKnnResult {
    debug_assert_eq!(query.len(), batch.dimension);

    if batch.num_vectors == 0 || k == 0 {
        return BatchKnnResult {
            indices: Vec::new(),
            distances: Vec::new(),
        };
    }

    let k = k.min(batch.num_vectors);

    // Full distance computation first
    let distances = batch_l2_squared(query, batch);

    // Find k smallest
    let mut indexed: Vec<(usize, f32)> = distances.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    indexed.truncate(k);

    BatchKnnResult {
        indices: indexed.iter().map(|(i, _)| *i).collect(),
        distances: indexed.iter().map(|(_, d)| *d).collect(),
    }
}

/// Adaptive batch kNN with early termination.
///
/// Uses a two-phase approach:
/// 1. **Warmup**: Process first `warmup_dims` dimensions fully
/// 2. **Prune**: Use partial distances to eliminate candidates
///
/// This is more efficient than full pruning when the warmup phase
/// establishes a good distance bound quickly.
pub fn batch_knn_adaptive(
    query: &[f32],
    batch: &VerticalBatch,
    k: usize,
    warmup_dims: usize,
) -> BatchKnnResult {
    debug_assert_eq!(query.len(), batch.dimension);

    if batch.num_vectors == 0 || k == 0 {
        return BatchKnnResult {
            indices: Vec::new(),
            distances: Vec::new(),
        };
    }

    let k = k.min(batch.num_vectors);
    let warmup_dims = warmup_dims.min(batch.dimension);

    let mut distances = vec![0.0f32; batch.num_vectors];
    let mut alive: Vec<bool> = vec![true; batch.num_vectors];

    // Phase 1: Warmup - process first dimensions fully
    for (d, &q_d) in query.iter().enumerate().take(warmup_dims) {
        let dim_slice = batch.dimension_slice(d);

        for (dist, &v_d) in distances.iter_mut().zip(dim_slice.iter()) {
            let diff = q_d - v_d;
            *dist += diff * diff;
        }
    }

    // Find initial k-th best distance (threshold for pruning)
    let mut partial_indexed: Vec<(usize, f32)> = distances.iter().copied().enumerate().collect();
    partial_indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut threshold = if k <= partial_indexed.len() {
        partial_indexed[k - 1].1 * (batch.dimension as f32 / warmup_dims as f32)
    } else {
        f32::MAX
    };

    // Mark candidates beyond threshold as dead
    for (i, &dist) in distances.iter().enumerate() {
        // Scale partial distance to estimate full distance
        let estimated_full = dist * (batch.dimension as f32 / warmup_dims as f32);
        if estimated_full > threshold * 1.5 {
            // Conservative margin
            alive[i] = false;
        }
    }

    // Phase 2: Process remaining dimensions with pruning
    for (d, &q_d) in query
        .iter()
        .enumerate()
        .skip(warmup_dims)
        .take(batch.dimension - warmup_dims)
    {
        let dim_slice = batch.dimension_slice(d);

        for ((&v_d, dist), is_alive) in dim_slice
            .iter()
            .zip(distances.iter_mut())
            .zip(alive.iter_mut())
        {
            if !*is_alive {
                continue;
            }

            let diff = q_d - v_d;
            *dist += diff * diff;

            // Prune if definitely beyond k-th best
            if *dist > threshold {
                *is_alive = false;
            }
        }

        // Update threshold periodically
        if d % 32 == 0 {
            let mut current_best: Vec<f32> = alive
                .iter()
                .zip(distances.iter())
                .filter(|(&a, _)| a)
                .map(|(_, &d)| d)
                .collect();

            if current_best.len() >= k {
                current_best.sort_by(|a, b| a.total_cmp(b));
                threshold = current_best[k - 1];
            }
        }
    }

    // Collect final results
    let mut results: Vec<(usize, f32)> = alive
        .iter()
        .enumerate()
        .filter(|(_, &a)| a)
        .map(|(i, _)| (i, distances[i]))
        .collect();

    results.sort_by(|a, b| a.1.total_cmp(&b.1));
    results.truncate(k);

    BatchKnnResult {
        indices: results.iter().map(|(i, _)| *i).collect(),
        distances: results.iter().map(|(_, d)| *d).collect(),
    }
}

/// Norms for batch of vectors (precomputed for cosine distance).
pub fn batch_norms(batch: &VerticalBatch) -> Vec<f32> {
    let mut norms = vec![0.0f32; batch.num_vectors];

    for d in 0..batch.dimension {
        let dim_slice = batch.dimension_slice(d);
        for (norm, &v_d) in norms.iter_mut().zip(dim_slice.iter()) {
            *norm += v_d * v_d;
        }
    }

    for norm in &mut norms {
        *norm = norm.sqrt();
    }

    norms
}

/// Compute cosine similarities from query to all vectors.
pub fn batch_cosine(query: &[f32], batch: &VerticalBatch, norms: &[f32]) -> Vec<f32> {
    debug_assert_eq!(norms.len(), batch.num_vectors);

    let dots = batch_dot(query, batch);
    let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();

    if query_norm < 1e-9 {
        return vec![0.0; batch.num_vectors];
    }

    dots.into_iter()
        .zip(norms.iter())
        .map(|(dot, &norm)| {
            if norm > 1e-9 {
                dot / (query_norm * norm)
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertical_batch_creation() {
        let vectors = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let batch = VerticalBatch::from_rows(&vectors);

        assert_eq!(batch.num_vectors(), 2);
        assert_eq!(batch.dimension(), 3);

        // Check dimension-major ordering
        assert_eq!(batch.get(0, 0), 1.0); // v0[0]
        assert_eq!(batch.get(0, 1), 4.0); // v1[0]
        assert_eq!(batch.get(1, 0), 2.0); // v0[1]
        assert_eq!(batch.get(2, 1), 6.0); // v1[2]
    }

    #[test]
    fn test_batch_l2_squared() {
        let vectors = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let batch = VerticalBatch::from_rows(&vectors);
        let query = vec![1.0, 1.0, 0.0];

        let distances = batch_l2_squared(&query, &batch);

        assert!((distances[0] - 2.0).abs() < 1e-6); // sqrt(2)^2 = 2
        assert!((distances[1] - 1.0).abs() < 1e-6); // dist to (1,0,0) = 1
        assert!((distances[2] - 1.0).abs() < 1e-6); // dist to (0,1,0) = 1
    }

    #[test]
    fn test_batch_dot() {
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let batch = VerticalBatch::from_rows(&vectors);
        let query = vec![1.0, 2.0];

        let dots = batch_dot(&query, &batch);

        assert!((dots[0] - 1.0).abs() < 1e-6); // 1*1 + 0*2 = 1
        assert!((dots[1] - 2.0).abs() < 1e-6); // 0*1 + 1*2 = 2
        assert!((dots[2] - 3.0).abs() < 1e-6); // 1*1 + 1*2 = 3
    }

    #[test]
    fn test_batch_knn() {
        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![3.0, 0.0],
        ];
        let batch = VerticalBatch::from_rows(&vectors);
        let query = vec![0.5, 0.0];

        let result = batch_knn(&query, &batch, 2);

        assert_eq!(result.indices.len(), 2);
        // Closest are v0 (dist=0.25) and v1 (dist=0.25)
        assert!(result.indices.contains(&0));
        assert!(result.indices.contains(&1));
    }

    #[test]
    fn test_batch_pruning() {
        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![10.0, 0.0], // Far away
        ];
        let batch = VerticalBatch::from_rows(&vectors);
        let query = vec![0.0, 0.0];

        let survivors = batch_l2_squared_pruning(&query, &batch, 2.0);

        // Only vectors within distance sqrt(2) should survive
        assert_eq!(survivors.len(), 2);
        let indices: Vec<usize> = survivors.iter().map(|(i, _)| *i).collect();
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
        assert!(!indices.contains(&2));
    }

    #[test]
    fn test_extract_vector() {
        let vectors = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let batch = VerticalBatch::from_rows(&vectors);

        let extracted = batch.extract_vector(1);
        assert_eq!(extracted, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_batch_cosine() {
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let batch = VerticalBatch::from_rows(&vectors);
        let norms = batch_norms(&batch);
        let query = vec![1.0, 0.0];

        let cosines = batch_cosine(&query, &batch, &norms);

        assert!((cosines[0] - 1.0).abs() < 1e-6); // Parallel
        assert!(cosines[1].abs() < 1e-6); // Orthogonal
        assert!((cosines[2] - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.01); // 45 degrees
    }

    // =========================================================================
    // Edge cases: empty and single-element batches
    // =========================================================================

    #[test]
    fn test_empty_batch() {
        let batch = VerticalBatch::from_rows(&[]);
        assert_eq!(batch.num_vectors(), 0);
        assert_eq!(batch.dimension(), 0);
    }

    #[test]
    fn test_single_vector_batch() {
        let vectors = vec![vec![1.0, 2.0, 3.0]];
        let batch = VerticalBatch::from_rows(&vectors);
        assert_eq!(batch.num_vectors(), 1);
        assert_eq!(batch.dimension(), 3);
        assert_eq!(batch.get(0, 0), 1.0);
        assert_eq!(batch.get(1, 0), 2.0);
        assert_eq!(batch.get(2, 0), 3.0);
        assert_eq!(batch.extract_vector(0), vec![1.0, 2.0, 3.0]);
    }

    // =========================================================================
    // from_flat: round-trip and equivalence with from_rows
    // =========================================================================

    #[test]
    fn test_from_flat_matches_from_rows() {
        let vectors = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]];
        let flat: Vec<f32> = vectors.iter().flatten().copied().collect();

        let batch_rows = VerticalBatch::from_rows(&vectors);
        let batch_flat = VerticalBatch::from_flat(&flat, 3, 3);

        for d in 0..3 {
            for v in 0..3 {
                assert_eq!(
                    batch_rows.get(d, v),
                    batch_flat.get(d, v),
                    "mismatch at dim={d}, vec={v}"
                );
            }
        }
    }

    #[test]
    fn test_from_flat_single_vector() {
        let flat = [10.0, 20.0];
        let batch = VerticalBatch::from_flat(&flat, 1, 2);
        assert_eq!(batch.num_vectors(), 1);
        assert_eq!(batch.dimension(), 2);
        assert_eq!(batch.extract_vector(0), vec![10.0, 20.0]);
    }

    // =========================================================================
    // dimension_slice correctness
    // =========================================================================

    #[test]
    fn test_dimension_slice() {
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let batch = VerticalBatch::from_rows(&vectors);

        // Dimension 0 across all vectors: [1.0, 3.0, 5.0]
        assert_eq!(batch.dimension_slice(0), &[1.0, 3.0, 5.0]);
        // Dimension 1 across all vectors: [2.0, 4.0, 6.0]
        assert_eq!(batch.dimension_slice(1), &[2.0, 4.0, 6.0]);
    }

    // =========================================================================
    // get_unchecked safety-checked test
    // =========================================================================

    #[test]
    fn test_get_unchecked_matches_get() {
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let batch = VerticalBatch::from_rows(&vectors);

        for d in 0..2 {
            for v in 0..2 {
                let safe = batch.get(d, v);
                let unchecked = unsafe { batch.get_unchecked(d, v) };
                assert_eq!(safe, unchecked, "mismatch at dim={d}, vec={v}");
            }
        }
    }

    // =========================================================================
    // batch_norms correctness
    // =========================================================================

    #[test]
    fn test_batch_norms() {
        let vectors = vec![vec![3.0, 4.0], vec![0.0, 0.0], vec![1.0, 0.0]];
        let batch = VerticalBatch::from_rows(&vectors);
        let norms = batch_norms(&batch);

        assert!((norms[0] - 5.0).abs() < 1e-6); // sqrt(9+16)
        assert!(norms[1].abs() < 1e-6); // zero vector
        assert!((norms[2] - 1.0).abs() < 1e-6); // unit vector
    }

    // =========================================================================
    // batch_l2_squared: query equal to one of the vectors
    // =========================================================================

    #[test]
    fn test_batch_l2_squared_exact_match() {
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let batch = VerticalBatch::from_rows(&vectors);
        let query = vec![3.0, 4.0]; // matches v1 exactly

        let distances = batch_l2_squared(&query, &batch);
        assert!(distances[1].abs() < 1e-9, "exact match should have distance ~0");
        assert!(distances[0] > 0.0);
        assert!(distances[2] > 0.0);
    }

    // =========================================================================
    // batch_dot: zero query
    // =========================================================================

    #[test]
    fn test_batch_dot_zero_query() {
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let batch = VerticalBatch::from_rows(&vectors);
        let query = vec![0.0, 0.0];

        let dots = batch_dot(&query, &batch);
        assert_eq!(dots, vec![0.0, 0.0]);
    }

    // =========================================================================
    // batch_cosine: zero-norm query and zero-norm vectors
    // =========================================================================

    #[test]
    fn test_batch_cosine_zero_query() {
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let batch = VerticalBatch::from_rows(&vectors);
        let norms = batch_norms(&batch);
        let query = vec![0.0, 0.0];

        let cosines = batch_cosine(&query, &batch, &norms);
        // Zero query norm -> all cosines 0.0
        assert_eq!(cosines, vec![0.0, 0.0]);
    }

    #[test]
    fn test_batch_cosine_zero_norm_vector() {
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 0.0]];
        let batch = VerticalBatch::from_rows(&vectors);
        let norms = batch_norms(&batch);
        let query = vec![1.0, 0.0];

        let cosines = batch_cosine(&query, &batch, &norms);
        assert!((cosines[0] - 1.0).abs() < 1e-6); // parallel
        assert_eq!(cosines[1], 0.0); // zero-norm vector
    }

    // =========================================================================
    // batch_knn edge cases
    // =========================================================================

    #[test]
    fn test_batch_knn_k_zero() {
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let batch = VerticalBatch::from_rows(&vectors);
        let query = vec![1.0, 0.0];

        let result = batch_knn(&query, &batch, 0);
        assert!(result.indices.is_empty());
        assert!(result.distances.is_empty());
    }

    #[test]
    fn test_batch_knn_empty_batch() {
        let batch = VerticalBatch::from_rows(&[]);
        let result = batch_knn(&[], &batch, 5);
        assert!(result.indices.is_empty());
    }

    #[test]
    fn test_batch_knn_k_larger_than_n() {
        let vectors = vec![vec![1.0], vec![2.0]];
        let batch = VerticalBatch::from_rows(&vectors);
        let query = vec![1.5];

        let result = batch_knn(&query, &batch, 10);
        // k is clamped to num_vectors
        assert_eq!(result.indices.len(), 2);
    }

    #[test]
    fn test_batch_knn_sorted_by_distance() {
        let vectors = vec![
            vec![10.0, 0.0],
            vec![1.0, 0.0],
            vec![5.0, 0.0],
            vec![0.0, 0.0],
        ];
        let batch = VerticalBatch::from_rows(&vectors);
        let query = vec![0.0, 0.0];

        let result = batch_knn(&query, &batch, 4);
        // Distances should be sorted ascending
        for w in result.distances.windows(2) {
            assert!(w[0] <= w[1], "distances not sorted: {:?}", result.distances);
        }
        // Closest is v3 (origin)
        assert_eq!(result.indices[0], 3);
    }

    // =========================================================================
    // batch_l2_squared_pruning edge cases
    // =========================================================================

    #[test]
    fn test_pruning_threshold_zero() {
        let vectors = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let batch = VerticalBatch::from_rows(&vectors);
        let query = vec![0.0, 0.0];

        let survivors = batch_l2_squared_pruning(&query, &batch, 0.0);
        // Only the origin (distance 0) survives with threshold 0
        assert_eq!(survivors.len(), 1);
        assert_eq!(survivors[0].0, 0);
        assert!(survivors[0].1.abs() < 1e-9);
    }

    #[test]
    fn test_pruning_all_survive() {
        let vectors = vec![vec![0.1, 0.0], vec![0.0, 0.1]];
        let batch = VerticalBatch::from_rows(&vectors);
        let query = vec![0.0, 0.0];

        let survivors = batch_l2_squared_pruning(&query, &batch, 100.0);
        assert_eq!(survivors.len(), 2);
    }

    #[test]
    fn test_pruning_none_survive() {
        let vectors = vec![vec![10.0, 0.0], vec![0.0, 10.0]];
        let batch = VerticalBatch::from_rows(&vectors);
        let query = vec![0.0, 0.0];

        // Threshold below the smallest distance (100.0)
        let survivors = batch_l2_squared_pruning(&query, &batch, 0.5);
        assert!(survivors.is_empty());
    }

    // =========================================================================
    // batch_knn_adaptive: correctness check vs. exact knn
    // =========================================================================

    #[test]
    fn test_batch_knn_adaptive_empty() {
        let batch = VerticalBatch::from_rows(&[]);
        let result = batch_knn_adaptive(&[], &batch, 5, 2);
        assert!(result.indices.is_empty());
    }

    #[test]
    fn test_batch_knn_adaptive_k_zero() {
        let vectors = vec![vec![1.0, 2.0]];
        let batch = VerticalBatch::from_rows(&vectors);
        let result = batch_knn_adaptive(&[1.0, 2.0], &batch, 0, 1);
        assert!(result.indices.is_empty());
    }

    #[test]
    fn test_batch_knn_adaptive_finds_nearest() {
        // The adaptive version should find the true nearest neighbor.
        // We use well-separated vectors to avoid pruning the correct answer.
        let vectors = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![100.0, 100.0, 100.0, 100.0],
            vec![0.1, 0.1, 0.1, 0.1],
        ];
        let batch = VerticalBatch::from_rows(&vectors);
        let query = vec![0.0, 0.0, 0.0, 0.0];

        let exact = batch_knn(&query, &batch, 1);
        let adaptive = batch_knn_adaptive(&query, &batch, 1, 2);

        // Both should find v0 (the origin) as nearest
        assert_eq!(exact.indices[0], 0);
        assert_eq!(adaptive.indices[0], 0);
    }

    // =========================================================================
    // Larger batch: 16+ vectors to exercise loop iterations
    // =========================================================================

    #[test]
    fn test_batch_l2_squared_large() {
        let n = 32;
        let dim = 8;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..dim).map(|d| (i * dim + d) as f32).collect())
            .collect();
        let batch = VerticalBatch::from_rows(&vectors);
        let query: Vec<f32> = vectors[0].clone();

        let distances = batch_l2_squared(&query, &batch);
        assert!(distances[0].abs() < 1e-9, "self-distance should be ~0");
        // All other distances should be positive
        for (i, &d) in distances.iter().enumerate().skip(1) {
            assert!(d > 0.0, "distance to vector {i} should be positive, got {d}");
        }
    }

    #[test]
    fn test_extract_all_vectors_roundtrip() {
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let batch = VerticalBatch::from_rows(&vectors);

        for (i, original) in vectors.iter().enumerate() {
            assert_eq!(batch.extract_vector(i), *original, "roundtrip failed for vector {i}");
        }
    }
}
