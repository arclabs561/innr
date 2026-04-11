//! Fixed-capacity top-K nearest neighbor tracker.
//!
//! Maintains the K smallest distances seen so far during ANN graph search.
//! Designed for the inner loop of greedy graph traversal where the vast
//! majority of candidates are rejected (threshold check fails), making
//! branch prediction near-perfect for the fast path.
//!
//! # Design
//!
//! The internal buffer stores entries sorted descending by distance so the
//! worst (largest) distance sits at index 0 for O(1) threshold access and
//! immediate rejection of non-improving candidates.
//!
//! When a candidate does improve the set, `copy_within` shifts the relevant
//! slice -- this compiles to `memmove`, which is SIMD-accelerated on most
//! platforms and avoids element-by-element branching during the shift.
//!
//! # Performance Characteristics
//!
//! - Fast path (reject): one comparison + return. Branch is well-predicted
//!   because most candidates are above threshold in a converged search.
//! - Slow path (insert): binary search to find insertion point, then
//!   `copy_within` shift -- one SIMD `memmove` over at most K elements.
//! - `threshold()`: O(1), no memory access beyond index 0.
//! - `into_sorted()`: reverse of the internal buffer -- O(K).

/// Fixed-capacity min-tracker for top-K nearest neighbor search.
///
/// Maintains the K smallest `(id, distance)` pairs seen so far.
/// The internal buffer is kept sorted descending so the worst element
/// (largest distance) is always at index 0.
///
/// # Example
///
/// ```rust
/// use innr::TopK;
///
/// let mut top = TopK::new(3);
/// for (id, dist) in [(0u32, 1.5), (1, 0.3), (2, 2.0), (3, 0.8)] {
///     top.insert(id, dist);
/// }
/// // top-3: ids 1 (0.3), 3 (0.8), 0 (1.5)
/// assert_eq!(top.len(), 3);
/// let results = top.into_sorted();
/// assert_eq!(results[0].0, 1); // closest
/// ```
pub struct TopK {
    k: usize,
    /// Distances sorted descending: `distances[0]` is the current worst.
    distances: Vec<f32>,
    /// Parallel ID array, same ordering as `distances`.
    ids: Vec<u32>,
    /// Number of items currently held (never exceeds `k`).
    count: usize,
}

impl TopK {
    /// Create a new `TopK` tracker with capacity `k`.
    ///
    /// # Panics
    ///
    /// Panics if `k == 0`.
    #[must_use]
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "innr::TopK: k must be >= 1");
        Self {
            k,
            distances: Vec::with_capacity(k),
            ids: Vec::with_capacity(k),
            count: 0,
        }
    }

    /// Return the current threshold (worst distance in the set).
    ///
    /// Returns `f32::INFINITY` when fewer than `k` items have been inserted,
    /// ensuring all candidates are accepted until the set is full.
    #[inline]
    #[must_use]
    pub fn threshold(&self) -> f32 {
        if self.count < self.k {
            f32::INFINITY
        } else {
            // Safety: count == k > 0, buffer has exactly k elements.
            self.distances[0]
        }
    }

    /// Try to insert a candidate.
    ///
    /// Accepts the candidate if `distance < threshold()`, replacing the
    /// current worst entry (or filling an empty slot). The common case
    /// (rejection) is a single comparison and return, making it
    /// branch-predictor-friendly when most candidates are above threshold.
    #[inline]
    pub fn insert(&mut self, id: u32, distance: f32) {
        if self.count < self.k {
            // Buffer not yet full: always insert.
            self.insert_sorted(id, distance);
            self.count += 1;
        } else if distance < self.distances[0] {
            // Better than current worst: evict index 0 and re-insert.
            // Remove slot 0 (worst) by shifting left, then insert at the
            // correct sorted position. copy_within -> memmove (SIMD).
            self.distances.copy_within(1.., 0);
            self.ids.copy_within(1.., 0);
            // Now the last slot is a duplicate; overwrite it during the
            // sorted insertion below.
            // Treat buffer as length k-1 for finding the insertion point,
            // then write into the freed trailing slot.
            let pos = self.find_insert_pos(distance, self.k - 1);
            // Shift [pos..k-1] right by one to open slot at pos.
            self.distances.copy_within(pos..self.k - 1, pos + 1);
            self.ids.copy_within(pos..self.k - 1, pos + 1);
            self.distances[pos] = distance;
            self.ids[pos] = id;
        }
        // else: distance >= threshold, reject. No branch needed beyond this.
    }

    /// Number of items currently held (min of total insertions and k).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if no items have been inserted yet.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Consume the tracker, returning results sorted ascending by distance
    /// (best/closest first).
    #[must_use]
    pub fn into_sorted(mut self) -> Vec<(u32, f32)> {
        // Internal buffer is descending; reverse for ascending output.
        self.distances.reverse();
        self.ids.reverse();
        self.ids.into_iter().zip(self.distances).collect()
    }

    // --- private helpers ---------------------------------------------------

    /// Insert `(id, distance)` into the sorted descending buffer when it has
    /// `count` live entries (with capacity already reserved).
    ///
    /// Called only when `count < k`, so `push` is safe.
    #[inline]
    fn insert_sorted(&mut self, id: u32, distance: f32) {
        let pos = self.find_insert_pos(distance, self.count);
        // Extend by one; then shift [pos..count] right.
        self.distances.push(distance);
        self.ids.push(id);
        let len = self.distances.len();
        self.distances.copy_within(pos..len - 1, pos + 1);
        self.ids.copy_within(pos..len - 1, pos + 1);
        self.distances[pos] = distance;
        self.ids[pos] = id;
    }

    /// Find the insertion index for `distance` in a descending-sorted buffer
    /// of length `len`, using binary search.
    ///
    /// Returns the leftmost index where `self.distances[i] <= distance`
    /// (i.e., where the new element should be placed to maintain descending
    /// order with equal elements pushed toward higher indices).
    #[inline]
    fn find_insert_pos(&self, distance: f32, len: usize) -> usize {
        // Binary search over self.distances[..len] (sorted descending).
        // We want the first index where distances[i] <= distance.
        let slice = &self.distances[..len];
        match slice.binary_search_by(|&d| {
            // Reverse the standard ordering: treat descending as ascending.
            d.partial_cmp(&distance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .reverse()
        }) {
            Ok(i) | Err(i) => i,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_top3() {
        let mut top = TopK::new(3);
        top.insert(0, 1.5);
        top.insert(1, 0.3);
        top.insert(2, 2.0);
        top.insert(3, 0.8);
        top.insert(4, 5.0);
        assert_eq!(top.len(), 3);
        let results = top.into_sorted();
        // Expected: (1, 0.3), (3, 0.8), (0, 1.5)
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], (1, 0.3));
        assert_eq!(results[1], (3, 0.8));
        assert_eq!(results[2], (0, 1.5));
    }

    #[test]
    fn threshold_tracking() {
        let mut top = TopK::new(3);
        assert_eq!(top.threshold(), f32::INFINITY);

        top.insert(0, 1.0);
        assert_eq!(top.threshold(), f32::INFINITY); // not full yet

        top.insert(1, 2.0);
        assert_eq!(top.threshold(), f32::INFINITY); // still not full

        top.insert(2, 3.0);
        assert_eq!(top.threshold(), 3.0); // now full

        top.insert(3, 1.5); // replaces 3.0
        assert_eq!(top.threshold(), 2.0);

        top.insert(4, 0.5); // replaces 2.0
        assert_eq!(top.threshold(), 1.5);

        top.insert(5, 10.0); // rejected
        assert_eq!(top.threshold(), 1.5);
    }

    #[test]
    fn duplicate_distances() {
        let mut top = TopK::new(3);
        top.insert(0, 1.0);
        top.insert(1, 1.0);
        top.insert(2, 1.0);
        top.insert(3, 1.0); // rejected (not strictly less)
        assert_eq!(top.len(), 3);
        let results = top.into_sorted();
        assert_eq!(results.len(), 3);
        for (_, d) in &results {
            assert_eq!(*d, 1.0);
        }
    }

    #[test]
    fn k1_edge_case() {
        let mut top = TopK::new(1);
        assert_eq!(top.threshold(), f32::INFINITY);
        top.insert(0, 5.0);
        assert_eq!(top.threshold(), 5.0);
        top.insert(1, 3.0);
        assert_eq!(top.threshold(), 3.0);
        top.insert(2, 10.0);
        assert_eq!(top.threshold(), 3.0);
        top.insert(3, 1.0);
        assert_eq!(top.threshold(), 1.0);

        let results = top.into_sorted();
        assert_eq!(results, vec![(3, 1.0)]);
    }

    #[test]
    fn large_n_k10() {
        let k = 10;
        let mut top = TopK::new(k);
        // Insert 10000 items; the k smallest are 0..10 with distances 0.0..9.0.
        for i in 0u32..10_000 {
            top.insert(i, i as f32);
        }
        assert_eq!(top.len(), k);
        let results = top.into_sorted();
        assert_eq!(results.len(), k);
        for (rank, (id, dist)) in results.iter().enumerate() {
            assert_eq!(*id, rank as u32);
            assert!((*dist - rank as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn sorted_output_ascending() {
        let mut top = TopK::new(5);
        // Insert in reverse order.
        for i in (0u32..5).rev() {
            top.insert(i, i as f32);
        }
        let results = top.into_sorted();
        for i in 0..results.len() - 1 {
            assert!(
                results[i].1 <= results[i + 1].1,
                "not ascending at index {i}"
            );
        }
    }

    #[test]
    fn is_empty_and_len() {
        let mut top = TopK::new(4);
        assert!(top.is_empty());
        assert_eq!(top.len(), 0);
        top.insert(0, 1.0);
        assert!(!top.is_empty());
        assert_eq!(top.len(), 1);
        top.insert(1, 2.0);
        top.insert(2, 3.0);
        top.insert(3, 4.0);
        assert_eq!(top.len(), 4);
        top.insert(4, 5.0); // rejected
        assert_eq!(top.len(), 4);
    }

    #[test]
    fn insert_in_sorted_order() {
        // Insert already-sorted ascending -- exercises all shift paths.
        let mut top = TopK::new(4);
        top.insert(0, 1.0);
        top.insert(1, 2.0);
        top.insert(2, 3.0);
        top.insert(3, 4.0);
        top.insert(4, 0.5); // replaces 4.0, inserts at end (best)
        let results = top.into_sorted();
        assert_eq!(results[0], (4, 0.5));
        assert_eq!(results[3], (2, 3.0));
    }
}
