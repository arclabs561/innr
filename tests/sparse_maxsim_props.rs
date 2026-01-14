#[cfg(all(test, feature = "sparse"))]
mod sparse_maxsim_props {
    use innr::sparse_maxsim;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn sparse_maxsim_sanity(
            // Generate query: list of sparse vectors
            query_raw in proptest::collection::vec(
                proptest::collection::vec((0u32..100, -1.0f32..1.0), 1..10), 1..5
            ),
            // Generate doc: list of sparse vectors
            doc_raw in proptest::collection::vec(
                proptest::collection::vec((0u32..100, -1.0f32..1.0), 1..10), 1..10
            )
        ) {
            // Convert to sorted sparse format
            let query_vecs: Vec<(Vec<u32>, Vec<f32>)> = query_raw.into_iter().map(|raw| {
                let mut sorted = raw;
                sorted.sort_by_key(|(idx, _)| *idx);
                sorted.dedup_by_key(|(idx, _)| *idx); // Ensure unique indices
                let (indices, values): (Vec<u32>, Vec<f32>) = sorted.into_iter().unzip();
                (indices, values)
            }).collect();

            let doc_vecs: Vec<(Vec<u32>, Vec<f32>)> = doc_raw.into_iter().map(|raw| {
                let mut sorted = raw;
                sorted.sort_by_key(|(idx, _)| *idx);
                sorted.dedup_by_key(|(idx, _)| *idx);
                let (indices, values): (Vec<u32>, Vec<f32>) = sorted.into_iter().unzip();
                (indices, values)
            }).collect();

            // Prepare slices
            let query_slices: Vec<(&[u32], &[f32])> = query_vecs.iter()
                .map(|(i, v)| (i.as_slice(), v.as_slice()))
                .collect();

            let doc_slices: Vec<(&[u32], &[f32])> = doc_vecs.iter()
                .map(|(i, v)| (i.as_slice(), v.as_slice()))
                .collect();

            let score = sparse_maxsim(&query_slices, &doc_slices);

            // Basic sanity check: should not be NaN
            prop_assert!(!score.is_nan());

            // If any query vector matches any doc vector exactly, score should be positive
            // (assuming non-zero vectors)
        }
    }
}
