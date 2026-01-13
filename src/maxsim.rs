//! ColBERT MaxSim late interaction scoring.
//!
//! MaxSim computes similarity between multi-vector representations
//! (like token embeddings) by summing the maximum similarity each
//! query token achieves with any document token.
//!
//! # Historical Context
//!
//! ColBERT (Khattab & Zaharia, SIGIR 2020) introduced late interaction
//! as a middle ground between:
//!
//! - **Bi-encoders**: Fast (single vector per doc), but limited expressiveness
//! - **Cross-encoders**: Expressive (full attention), but O(|Q| * |D|) per pair
//!
//! Late interaction keeps token-level representations (like cross-encoders)
//! but uses simple max-pool aggregation (fast like bi-encoders).
//!
//! # Mathematical Formulation
//!
//! ```text
//! MaxSim(Q, D) = Σᵢ maxⱼ(Qᵢ · Dⱼ)
//! ```
//!
//! For query Q = [q₁, q₂, ...] and document D = [d₁, d₂, ...]:
//! 1. For each query token qᵢ, find its best match in D: maxⱼ(qᵢ · dⱼ)
//! 2. Sum these maximum similarities across all query tokens
//!
//! # Not Commutative
//!
//! **Warning**: `maxsim(Q, D) ≠ maxsim(D, Q)`
//!
//! The first argument is always interpreted as the query. Swapping arguments
//! changes the semantics: which side's tokens "select" their best matches.
//!
//! # References
//!
//! - Khattab & Zaharia (2020). "ColBERT: Efficient and Effective Passage Search"
//! - Santhanam et al. (2022). "ColBERTv2: Effective and Efficient Retrieval"

use crate::dense::{cosine, dot};

/// MaxSim: sum over query tokens of max dot product with any doc token.
///
/// # Arguments
///
/// * `query_tokens` - Query token embeddings (first argument = query)
/// * `doc_tokens` - Document token embeddings
///
/// # Returns
///
/// Sum of maximum similarities. Returns 0.0 if either input is empty.
///
/// # Complexity
///
/// - Time: O(|Q| * |D| * dim)
/// - Space: O(1)
///
/// # Example
///
/// ```rust
/// use innr::maxsim;
///
/// // Query: two tokens [1,0] and [0,1]
/// // Doc: three tokens, best matches are doc[0] for q[0], doc[1] for q[1]
/// let q1 = [1.0f32, 0.0];
/// let q2 = [0.0f32, 1.0];
/// let d1 = [0.9f32, 0.1];  // best match for q1
/// let d2 = [0.1f32, 0.9];  // best match for q2
/// let d3 = [0.5f32, 0.5];
///
/// let query: &[&[f32]] = &[&q1, &q2];
/// let doc: &[&[f32]] = &[&d1, &d2, &d3];
///
/// let score = maxsim(query, doc);
/// // score = max(0.9, 0.1, 0.5) + max(0.1, 0.9, 0.5) = 0.9 + 0.9 = 1.8
/// assert!((score - 1.8).abs() < 0.01);
/// ```
#[inline]
#[must_use]
pub fn maxsim(query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }

    query_tokens
        .iter()
        .map(|q| {
            doc_tokens
                .iter()
                .map(|d| dot(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

/// MaxSim with cosine similarity instead of dot product.
///
/// Use this when embeddings are not pre-normalized.
///
/// # Arguments
///
/// * `query_tokens` - Query token embeddings (first argument = query)
/// * `doc_tokens` - Document token embeddings
///
/// # Returns
///
/// Sum of maximum cosine similarities. Returns 0.0 if either input is empty.
#[inline]
#[must_use]
pub fn maxsim_cosine(query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }

    query_tokens
        .iter()
        .map(|q| {
            doc_tokens
                .iter()
                .map(|d| cosine(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxsim_basic() {
        let q1 = [1.0f32, 0.0];
        let q2 = [0.0f32, 1.0];
        let d1 = [0.9f32, 0.1];
        let d2 = [0.1f32, 0.9];

        let query: &[&[f32]] = &[&q1, &q2];
        let doc: &[&[f32]] = &[&d1, &d2];

        let score = maxsim(query, doc);
        // q1 best match: d1 (0.9*1 + 0.1*0 = 0.9)
        // q2 best match: d2 (0.1*0 + 0.9*1 = 0.9)
        // total = 0.9 + 0.9 = 1.8
        assert!((score - 1.8).abs() < 1e-6);
    }

    #[test]
    fn test_maxsim_empty() {
        let q1 = [1.0f32, 0.0];
        let query: &[&[f32]] = &[&q1];
        let empty: &[&[f32]] = &[];

        assert_eq!(maxsim(query, empty), 0.0);
        assert_eq!(maxsim(empty, query), 0.0);
    }

    #[test]
    fn test_maxsim_not_commutative() {
        // Different number of query vs doc tokens shows non-commutativity
        let q1 = [1.0f32, 0.0];
        let d1 = [0.5f32, 0.5];
        let d2 = [0.5f32, 0.5];

        let query: &[&[f32]] = &[&q1];
        let doc: &[&[f32]] = &[&d1, &d2];

        let score_qd = maxsim(query, doc);
        let score_dq = maxsim(doc, query);

        // With 1 query token vs 2 doc tokens, results differ
        // score_qd = max(0.5, 0.5) = 0.5 (sum over 1 query token)
        // score_dq = max(0.5) + max(0.5) = 1.0 (sum over 2 "query" tokens)
        assert!((score_qd - 0.5).abs() < 1e-6);
        assert!((score_dq - 1.0).abs() < 1e-6);
        assert!((score_qd - score_dq).abs() > 0.4); // Not equal
    }

    #[test]
    fn test_maxsim_cosine_normalized() {
        // For normalized vectors, maxsim and maxsim_cosine should be similar
        let q1 = [1.0f32, 0.0]; // already normalized
        let d1 = [1.0f32, 0.0]; // already normalized

        let query: &[&[f32]] = &[&q1];
        let doc: &[&[f32]] = &[&d1];

        let dot_score = maxsim(query, doc);
        let cos_score = maxsim_cosine(query, doc);

        assert!((dot_score - cos_score).abs() < 1e-6);
    }

    #[test]
    fn test_maxsim_cosine_unnormalized() {
        // For unnormalized vectors, cosine handles the normalization
        let q1 = [2.0f32, 0.0]; // not normalized
        let d1 = [3.0f32, 0.0]; // not normalized

        let query: &[&[f32]] = &[&q1];
        let doc: &[&[f32]] = &[&d1];

        let cos_score = maxsim_cosine(query, doc);
        // cosine([2,0], [3,0]) = (2*3) / (2 * 3) = 1.0
        assert!((cos_score - 1.0).abs() < 1e-6);
    }
}
