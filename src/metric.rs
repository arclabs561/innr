//! Metric and quasimetric trait surfaces.
//!
//! This module is intentionally minimal:
//! - It defines *interfaces*, not implementations.
//! - It is dependency-free (L0-friendly).
//! - It keeps scalar type generic (works for `f32` and `f64`).
//!
//! Rationale:
//! - We want downstream crates (L1+) to be able to specialize metric/space pairs
//!   without routing through an extra "law" crate boundary.
//! - By keeping the trait here, every vector-space component can be "metric-aware"
//!   without introducing higher-layer dependencies.
//!
//! Note: Any "axiom checking" utilities (triangle inequality checks, projections, etc.)
//! are better housed in higher layers where richer numeric bounds/deps are acceptable.
//! This file stays small by design.
/// A symmetric distance metric.
///
/// This is an interface only; implementations decide their own numeric behavior.
pub trait SymmetricMetric<T> {
    /// Compute the (symmetric) distance between `a` and `b`.
    fn distance(&self, a: &[T], b: &[T]) -> T;
}

/// A directed distance (quasimetric), where symmetry need not hold.
///
/// Use this for "reachability"-style geometry:
/// \(d(x, y)\) can differ from \(d(y, x)\).
pub trait Quasimetric<T> {
    /// Compute the directed distance from `source` to `target`.
    fn reachability(&self, source: &[T], target: &[T]) -> T;
}
