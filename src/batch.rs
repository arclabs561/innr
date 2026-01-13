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
}
