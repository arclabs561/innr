//! SIMD-accelerated vector similarity primitives.
//!
//! Fast building blocks for embedding similarity with automatic hardware dispatch.
//!
//! # Which Function Should I Use?
//!
//! | Task | Function | Notes |
//! |------|----------|-------|
//! | **Similarity (normalized)** | [`cosine`] | Most embeddings are normalized |
//! | **Similarity (raw)** | [`dot`] | When you know norms |
//! | **Distance (L2)** | [`l2_distance`] | For k-NN, clustering |
//! | **Token-level matching** | [`maxsim`] | ColBERT-style late interaction |
//! | **Sparse vectors** | [`sparse_dot`] | BM25 scores, SPLADE |
//!
//! # SIMD Dispatch
//!
//! All functions automatically dispatch to the fastest available instruction set:
//!
//! | Architecture | Instructions | Detection |
//! |--------------|--------------|-----------|
//! | x86_64 | AVX-512F | Runtime |
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
//! interaction, requiring O(|Q| x |D|) inner products per query-document pair.
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

/// Binary (1-bit) quantization: encode, Hamming distance, dot product, Jaccard.
pub mod binary;

/// Dense vector primitives: dot, cosine, norm, L2/L1 distance, matryoshka.
pub mod dense;

/// Fast math operations using hardware-aware approximations (rsqrt, NR iteration).
pub mod fast_math;

/// Batch vector operations with columnar (PDX-style) layout.
pub mod batch;

/// Sparse vector dot product via sorted-index merge join.
mod sparse;

/// ColBERT MaxSim late interaction scoring for multi-vector retrieval.
mod maxsim;

// Re-export core operations
pub use dense::{
    angular_distance, cosine, dot, l1_distance, l2_distance, l2_distance_squared,
    matryoshka_cosine, matryoshka_dot, norm, normalize,
};

// Re-export binary operations
pub use binary::{binary_dot, binary_hamming, binary_jaccard, encode_binary, PackedBinary};

// Re-export fast math (rsqrt-based approximations)
pub use fast_math::{fast_cosine, fast_cosine_dispatch, fast_rsqrt, fast_rsqrt_precise};

/// Ternary quantization (1.58-bit) for ultra-compressed embeddings.
pub mod ternary;

/// Scalar quantization (uint8) for memory-efficient asymmetric similarity.
pub mod scalar;

pub use sparse::{sparse_dot, sparse_maxsim};

pub use maxsim::{maxsim, maxsim_cosine};

/// Minimum vector dimension for SIMD to be worthwhile.
///
/// Below this, function call overhead outweighs SIMD benefits.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) const MIN_DIM_SIMD: usize = 16;

/// Threshold for treating a norm as "effectively zero".
///
/// Chosen to be larger than `f32::EPSILON` (~1.19e-7) to provide numerical
/// headroom while remaining small enough to only catch degenerate cases.
///
/// Used by [`cosine`] to avoid division by zero.
pub(crate) const NORM_EPSILON: f32 = 1e-9;

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

    #[test]
    fn test_l1_distance() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 0.0, 1.0];
        // |1-4| + |2-0| + |3-1| = 3 + 2 + 2 = 7
        assert!((l1_distance(&a, &b) - 7.0).abs() < 1e-6);
    }
}
