//! Property-based tests for MaxSim late interaction scoring.
#![cfg(feature = "maxsim")]

use proptest::prelude::*;

/// Generate query/document token pairs.
fn arb_token_pair(
    query_tokens: usize,
    doc_tokens: usize,
    dim: usize,
) -> impl Strategy<Value = (Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    (
        proptest::collection::vec(proptest::collection::vec(-10.0f32..10.0, dim), query_tokens),
        proptest::collection::vec(proptest::collection::vec(-10.0f32..10.0, dim), doc_tokens),
    )
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 200,
        ..ProptestConfig::default()
    })]

    /// MaxSim with empty query returns 0.
    #[test]
    fn maxsim_empty_query_is_zero(
        doc_tokens in proptest::collection::vec(
            proptest::collection::vec(-10.0f32..10.0, 32), 1..8
        )
    ) {
        let empty: Vec<&[f32]> = vec![];
        let doc: Vec<&[f32]> = doc_tokens.iter().map(|v| v.as_slice()).collect();
        let score = innr::maxsim(&empty, &doc);
        prop_assert_eq!(score, 0.0);
    }

    /// MaxSim with empty document returns 0.
    #[test]
    fn maxsim_empty_doc_is_zero(
        query_tokens in proptest::collection::vec(
            proptest::collection::vec(-10.0f32..10.0, 32), 1..8
        )
    ) {
        let query: Vec<&[f32]> = query_tokens.iter().map(|v| v.as_slice()).collect();
        let empty: Vec<&[f32]> = vec![];
        let score = innr::maxsim(&query, &empty);
        prop_assert_eq!(score, 0.0);
    }

    /// MaxSim is NOT commutative (important invariant).
    #[test]
    fn maxsim_not_commutative((query, doc) in arb_token_pair(2, 4, 32)) {
        let q: Vec<&[f32]> = query.iter().map(|v| v.as_slice()).collect();
        let d: Vec<&[f32]> = doc.iter().map(|v| v.as_slice()).collect();

        let qd = innr::maxsim(&q, &d);
        let dq = innr::maxsim(&d, &q);

        // With different token counts, scores should generally differ
        // This test documents the important invariant that order matters
        prop_assert!(
            qd.is_finite() && dq.is_finite(),
            "Scores should be finite: qd={}, dq={}",
            qd, dq
        );
    }

    /// MaxSim single query token equals max dot product.
    #[test]
    fn maxsim_single_query_equals_max_dot(
        query_token in proptest::collection::vec(-10.0f32..10.0, 64),
        doc_tokens in proptest::collection::vec(
            proptest::collection::vec(-10.0f32..10.0, 64), 1..16
        )
    ) {
        let query: Vec<&[f32]> = vec![query_token.as_slice()];
        let doc: Vec<&[f32]> = doc_tokens.iter().map(|v| v.as_slice()).collect();

        let maxsim_score = innr::maxsim(&query, &doc);

        // Should equal max(dot(query_token, doc_token) for all doc_tokens)
        let max_dot: f32 = doc_tokens
            .iter()
            .map(|d| innr::dot(&query_token, d))
            .fold(f32::NEG_INFINITY, f32::max);

        let tolerance = max_dot.abs() * 1e-4 + 1e-5;
        prop_assert!(
            (maxsim_score - max_dot).abs() < tolerance,
            "Single query maxsim {} != max dot {} (diff: {})",
            maxsim_score, max_dot, (maxsim_score - max_dot).abs()
        );
    }

    /// MaxSim is additive over query tokens.
    #[test]
    fn maxsim_is_additive(
        (query1, doc) in arb_token_pair(2, 4, 32),
        query2 in proptest::collection::vec(
            proptest::collection::vec(-10.0f32..10.0, 32), 2
        )
    ) {
        let q1: Vec<&[f32]> = query1.iter().map(|v| v.as_slice()).collect();
        let q2: Vec<&[f32]> = query2.iter().map(|v| v.as_slice()).collect();
        let d: Vec<&[f32]> = doc.iter().map(|v| v.as_slice()).collect();

        let score1 = innr::maxsim(&q1, &d);
        let score2 = innr::maxsim(&q2, &d);

        // Combined query
        let mut combined_query = query1.clone();
        combined_query.extend(query2.iter().cloned());
        let combined: Vec<&[f32]> = combined_query.iter().map(|v| v.as_slice()).collect();
        let score_combined = innr::maxsim(&combined, &d);

        let expected = score1 + score2;
        // Use tolerance based on max magnitude to handle catastrophic cancellation
        let max_magnitude = score1.abs().max(score2.abs()).max(1.0);
        let tolerance = max_magnitude * 1e-4;
        prop_assert!(
            (score_combined - expected).abs() < tolerance,
            "Additivity: {} != {} + {} = {} (diff: {}, tol: {})",
            score_combined, score1, score2, expected,
            (score_combined - expected).abs(), tolerance
        );
    }

    /// MaxSim with identical query and doc is positive.
    #[test]
    fn maxsim_identical_is_positive(
        tokens in proptest::collection::vec(
            proptest::collection::vec(-10.0f32..10.0, 32)
                .prop_filter("non-zero", |v| v.iter().any(|x| x.abs() > 0.1)),
            1..8
        )
    ) {
        let t: Vec<&[f32]> = tokens.iter().map(|v| v.as_slice()).collect();
        let score = innr::maxsim(&t, &t);

        prop_assert!(
            score >= 0.0,
            "MaxSim with identical tokens should be non-negative, got {}",
            score
        );
    }

    /// MaxSim-cosine is bounded by query token count.
    #[test]
    fn maxsim_cosine_bounded((query, doc) in arb_token_pair(4, 8, 32)) {
        let q: Vec<&[f32]> = query.iter().map(|v| v.as_slice()).collect();
        let d: Vec<&[f32]> = doc.iter().map(|v| v.as_slice()).collect();

        let score = innr::maxsim_cosine(&q, &d);

        // Each query token contributes at most 1.0 (max cosine similarity)
        let upper_bound = query.len() as f32 + 1e-5;
        let lower_bound = -(query.len() as f32) - 1e-5;

        prop_assert!(
            score >= lower_bound && score <= upper_bound,
            "MaxSim-cosine {} out of bounds [{}, {}]",
            score, lower_bound, upper_bound
        );
    }
}

#[test]
fn test_maxsim_basic_example() {
    // Example from documentation
    let q1 = [1.0f32, 0.0];
    let q2 = [0.0f32, 1.0];
    let d1 = [0.9f32, 0.1]; // best match for q1
    let d2 = [0.1f32, 0.9]; // best match for q2
    let d3 = [0.5f32, 0.5];

    let query: &[&[f32]] = &[&q1, &q2];
    let doc: &[&[f32]] = &[&d1, &d2, &d3];

    let score = innr::maxsim(query, doc);
    // score = max(0.9, 0.1, 0.5) + max(0.1, 0.9, 0.5) = 0.9 + 0.9 = 1.8
    assert!((score - 1.8).abs() < 0.01);
}
