//! SIMD-accelerated vector similarity primitives.
//!
//! `innr` (from "inner product") provides building blocks for embedding similarity:
//!
//! - **Dense**: [`dot`], [`cosine`], [`norm`], [`l2_distance`]
//! - **Sparse**: `sparse_dot` (feature `sparse`)
//! - **Late interaction**: `maxsim` (feature `maxsim`)
//!
//! # SIMD Dispatch
//!
//! All functions automatically dispatch to the fastest available instruction set:
//!
//! | Architecture | Instructions | Detection |
//! |--------------|--------------|-----------|
//! | x86_64 | AVX2 + FMA | Runtime |
//! | aarch64 | NEON | Always available |
//! | Other | Portable | LLVM auto-vectorizes |
//!
//! Vectors shorter than 16 dimensions use portable code (SIMD overhead not worthwhile).
//!
//! # Historical Context
//!
//! The inner product (dot product) dates to Grassmann's 1844 "Ausdehnungslehre" and
//! Hamilton's quaternions, formalized in Gibbs and Heaviside's vector calculus (~1880s).
//! Modern embedding similarity (Word2Vec 2013, BERT 2018) relies on inner products
//! in high-dimensional spaces where SIMD acceleration is essential.
//!
//! ColBERT's MaxSim (Khattab & Zaharia, 2020) extends this to token-level late
//! interaction, requiring O(|Q| * |D|) inner products per query-document pair.
//!
//! # Example
//!
//! ```rust
//! use innr::{dot, cosine, norm};
//!
//! let a = [1.0_f32, 0.0, 0.0];
//! let b = [0.707, 0.707, 0.0];
//!
//! // Dot product
//! let d = dot(&a, &b);
//! assert!((d - 0.707).abs() < 0.01);
//!
//! // Cosine similarity (normalized dot product)
//! let c = cosine(&a, &b);
//! assert!((c - 0.707).abs() < 0.01);
//!
//! // L2 norm
//! let n = norm(&a);
//! assert!((n - 1.0).abs() < 1e-6);
//! ```
//!
//! # References
//!
//! - Gibbs, J.W. (1881). "Elements of Vector Analysis"
//! - Mikolov et al. (2013). "Efficient Estimation of Word Representations" (Word2Vec)
//! - Khattab & Zaharia (2020). "ColBERT: Efficient and Effective Passage Search"

#![warn(missing_docs)]
#![warn(clippy::all)]

mod arch;
mod dense;

#[cfg(feature = "sparse")]
mod sparse;

#[cfg(feature = "maxsim")]
mod maxsim;

// Re-export core operations
pub use dense::{cosine, dot, dot_portable, l2_distance, l2_distance_squared, norm};

#[cfg(feature = "sparse")]
pub use sparse::{sparse_dot, sparse_dot_portable};

#[cfg(feature = "maxsim")]
pub use maxsim::{maxsim, maxsim_cosine};

/// Minimum vector dimension for SIMD to be worthwhile.
///
/// Below this threshold, function call overhead outweighs SIMD benefits.
/// Matches qdrant's MIN_DIM_SIZE_SIMD threshold.
pub const MIN_DIM_SIMD: usize = 16;

/// Threshold for treating a norm as "effectively zero".
///
/// Chosen to be larger than `f32::EPSILON` (~1.19e-7) to provide numerical
/// headroom while remaining small enough to only catch degenerate cases.
///
/// Used by [`cosine`] to avoid division by zero.
pub const NORM_EPSILON: f32 = 1e-9;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_basic() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let result = dot(&a, &b);
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_empty() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        assert_eq!(dot(&a, &b), 0.0);
    }

    #[test]
    fn test_norm() {
        let v = [3.0_f32, 4.0];
        assert!((norm(&v) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        assert!(cosine(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_parallel() {
        let a = [1.0_f32, 0.0];
        let b = [2.0_f32, 0.0];
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = [1.0_f32, 2.0];
        let zero = [0.0_f32, 0.0];
        assert_eq!(cosine(&a, &zero), 0.0);
    }

    #[test]
    fn test_l2_distance() {
        let a = [0.0_f32, 0.0];
        let b = [3.0_f32, 4.0];
        assert!((l2_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance_same_point() {
        let a = [1.0_f32, 2.0, 3.0];
        assert!(l2_distance(&a, &a) < 1e-9);
    }
}
