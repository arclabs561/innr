#[cfg(all(test, any(target_arch = "x86_64", target_arch = "aarch64")))]
mod maxsim_simd_props {
    use innr::dense::dot;
    use innr::maxsim;
    use proptest::prelude::*;

    // Portable implementation for reference
    fn maxsim_portable(query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
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

    proptest! {
        #[test]
        fn maxsim_simd_matches_portable(
            dim in 16usize..128,
            query_data in proptest::collection::vec(-1.0f32..1.0, 16..1280), // ample data
            doc_data in proptest::collection::vec(-1.0f32..1.0, 16..2560)
        ) {
            // Construct query vectors of length `dim`
            let mut query_vecs = Vec::new();
            for chunk in query_data.chunks(dim) {
                if chunk.len() == dim {
                    query_vecs.push(chunk.to_vec());
                }
            }
            if query_vecs.is_empty() {
                // Ensure at least one
                query_vecs.push(vec![0.0; dim]);
            }

            // Construct doc vectors of length `dim`
            let mut doc_vecs = Vec::new();
            for chunk in doc_data.chunks(dim) {
                if chunk.len() == dim {
                    doc_vecs.push(chunk.to_vec());
                }
            }
            if doc_vecs.is_empty() {
                doc_vecs.push(vec![0.0; dim]);
            }

            let query_refs: Vec<&[f32]> = query_vecs.iter().map(|v| v.as_slice()).collect();
            let doc_refs: Vec<&[f32]> = doc_vecs.iter().map(|v| v.as_slice()).collect();

            let expected = maxsim_portable(&query_refs, &doc_refs);
            let actual = maxsim(&query_refs, &doc_refs);

            // Allow some floating point drift due to SIMD association
            // Summing many products can drift.
            prop_assert!(
                (expected - actual).abs() < 1e-3,
                "maxsim mismatch: dim={}, Q={}, D={}, expected {}, got {}, diff {}",
                dim, query_vecs.len(), doc_vecs.len(), expected, actual, (expected - actual).abs()
            );
        }
    }
}
