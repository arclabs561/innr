//! Generic [`Distance`](crate::distance::Distance) trait for using innr's
//! metrics as a pluggable backend.
//!
//! The free functions elsewhere in this crate ([`crate::cosine`], [`crate::dot`],
//! [`crate::l2_distance`], ...) are the fast path when you know the metric at the
//! call site. This module wraps them behind a trait so a generic index (a custom
//! HNSW, a brute-force searcher, a clustering routine) can be parameterized over
//! the metric instead of hard-coding one.
//!
//! The trait shape intentionally mirrors the convention used by `anndists` /
//! `hnsw_rs` (`eval(&self, &[T], &[T]) -> f32`, smaller = closer).
//!
//! This is innr's *own* trait, not a re-export. `hnsw_rs` binds specifically to
//! `anndists::dist::distances::Distance` (it does `pub use anndists`), so these
//! metric types are not automatically usable as `hnsw_rs` distances: a thin
//! adapter that implements `anndists`'s trait for them would be required, and is
//! deliberately left out of the dependency-free core. Use this trait for
//! parameterizing innr's own generic code over a metric.
//!
//! # Convention
//!
//! `eval` returns a **distance**: smaller means more similar. Similarity metrics
//! are converted accordingly:
//!
//! - [`DistCosine`](crate::distance::DistCosine) returns `1 - cosine_similarity` (range `[0, 2]`).
//! - [`DistDot`](crate::distance::DistDot) returns `-dot` so that larger dot products sort first.
//! - [`DistL2`](crate::distance::DistL2), [`DistL1`](crate::distance::DistL1) are already distances.
//! - [`DistHamming`](crate::distance::DistHamming) returns the bit-Hamming distance as `f32`.
//! - [`DistSlotU32`](crate::distance::DistSlotU32) returns the normalized integer-slot Hamming distance (fraction of differing slots).
//!
//! # Example
//!
//! ```rust
//! use innr::distance::{Distance, DistCosine, DistL2};
//!
//! fn nearest<D: Distance<f32>>(metric: &D, query: &[f32], corpus: &[Vec<f32>]) -> usize {
//!     corpus
//!         .iter()
//!         .enumerate()
//!         .min_by(|(_, a), (_, b)| {
//!             metric
//!                 .eval(query, a)
//!                 .total_cmp(&metric.eval(query, b))
//!         })
//!         .map(|(i, _)| i)
//!         .unwrap()
//! }
//!
//! let corpus = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
//! assert_eq!(nearest(&DistCosine, &[1.0, 0.1], &corpus), 0);
//! assert_eq!(nearest(&DistL2, &[0.9, 0.9], &corpus), 2);
//! ```

use crate::{cosine, dot, hamming_distance, l1_distance, l2_distance, slot::jaccard_distance};

/// A distance metric over slices of `T`. `eval` returns a distance: smaller is
/// more similar.
///
/// Mirrors the `anndists` / `hnsw_rs` trait shape so innr's metrics can back a
/// generic index written against that convention.
pub trait Distance<T> {
    /// Distance between `a` and `b`. Smaller means more similar.
    fn eval(&self, a: &[T], b: &[T]) -> f32;
}

/// Cosine distance: `1 - cosine_similarity`. Range `[0, 2]`.
#[derive(Debug, Clone, Copy, Default)]
pub struct DistCosine;

impl Distance<f32> for DistCosine {
    #[inline]
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        1.0 - cosine(a, b)
    }
}

/// Negated dot product, so that larger inner products sort first. Use with
/// normalized vectors for a maximum-inner-product search.
#[derive(Debug, Clone, Copy, Default)]
pub struct DistDot;

impl Distance<f32> for DistDot {
    #[inline]
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        -dot(a, b)
    }
}

/// Euclidean (L2) distance.
#[derive(Debug, Clone, Copy, Default)]
pub struct DistL2;

impl Distance<f32> for DistL2 {
    #[inline]
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        l2_distance(a, b)
    }
}

/// Manhattan (L1) distance.
#[derive(Debug, Clone, Copy, Default)]
pub struct DistL1;

impl Distance<f32> for DistL1 {
    #[inline]
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        l1_distance(a, b)
    }
}

/// Bit-Hamming distance over byte-packed binary vectors (see
/// [`crate::hamming_distance`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct DistHamming;

impl Distance<u8> for DistHamming {
    #[inline]
    fn eval(&self, a: &[u8], b: &[u8]) -> f32 {
        hamming_distance(a, b) as f32
    }
}

/// Normalized integer-slot Hamming distance over `u32` slots: the fraction of
/// differing slots (see [`crate::jaccard_distance`]). The natural metric for
/// MinHash sketches.
///
/// Returns `differing / len` rather than the raw count, matching the value the
/// `anndists` integer `DistHamming` produces, so an index built on this metric
/// sees the same distance scale as that ecosystem.
#[derive(Debug, Clone, Copy, Default)]
pub struct DistSlotU32;

impl Distance<u32> for DistSlotU32 {
    #[inline]
    fn eval(&self, a: &[u32], b: &[u32]) -> f32 {
        jaccard_distance(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_distance_zero_for_parallel() {
        let d = DistCosine.eval(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]);
        assert!(
            d.abs() < 1e-6,
            "parallel vectors should have cosine distance 0, got {d}"
        );
    }

    #[test]
    fn dot_distance_orders_by_inner_product() {
        // larger dot product -> smaller (more negative) distance
        let near = DistDot.eval(&[1.0, 0.0], &[1.0, 0.0]);
        let far = DistDot.eval(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(near < far);
    }

    #[test]
    fn l2_matches_free_function() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 0.0, 3.0];
        assert_eq!(DistL2.eval(&a, &b), l2_distance(&a, &b));
    }

    #[test]
    fn l1_matches_free_function() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 0.0, 3.0];
        assert_eq!(DistL1.eval(&a, &b), l1_distance(&a, &b));
    }

    #[test]
    fn hamming_matches_free_function() {
        let a = [0b1111_0000u8, 0xFF];
        let b = [0b1010_1010u8, 0x00];
        assert_eq!(DistHamming.eval(&a, &b), hamming_distance(&a, &b) as f32);
    }

    #[test]
    fn slot_distance_is_normalized_differing_fraction() {
        let a = [1u32, 2, 3, 4];
        let b = [1u32, 0, 3, 9];
        // 2 of 4 slots differ -> 0.5
        assert_eq!(DistSlotU32.eval(&a, &b), 0.5);
    }

    // A generic consumer compiles and runs against any Distance impl.
    fn closest<T, D: Distance<T>>(metric: &D, q: &[T], corpus: &[Vec<T>]) -> usize {
        corpus
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| metric.eval(q, a).total_cmp(&metric.eval(q, b)))
            .map(|(i, _)| i)
            .unwrap()
    }

    #[test]
    fn generic_index_over_metric() {
        let corpus = vec![vec![1.0f32, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        assert_eq!(closest(&DistCosine, &[1.0, 0.05], &corpus), 0);
        assert_eq!(closest(&DistL2, &[0.95, 0.95], &corpus), 2);

        let sketches = vec![vec![1u32, 2, 3, 4], vec![1, 2, 3, 9], vec![9, 9, 9, 9]];
        assert_eq!(closest(&DistSlotU32, &[1, 2, 3, 4], &sketches), 0);
    }
}
