#![allow(missing_docs)]
#[cfg(test)]
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

        /// sparse_dot should match dense dot on equivalent representations.
        #[test]
        fn sparse_dot_matches_dense(
            dim in 10u32..50,
            nnz in 1usize..10,
        ) {
            use innr::{sparse_dot, dot};

            // Build a sparse vector with `nnz` non-zero entries
            let indices: Vec<u32> = (0..nnz as u32).collect();
            let values: Vec<f32> = (0..nnz).map(|i| (i as f32 + 1.0) * 0.1).collect();

            // Build equivalent dense vector
            let mut dense = vec![0.0f32; dim as usize];
            for (&idx, &val) in indices.iter().zip(values.iter()) {
                if (idx as usize) < dense.len() {
                    dense[idx as usize] = val;
                }
            }

            // sparse_dot(a, a) should equal dense dot(a, a)
            let sparse_result = sparse_dot(&indices, &values, &indices, &values);
            let dense_result = dot(&dense, &dense);

            prop_assert!(
                (sparse_result - dense_result).abs() < 1e-5,
                "sparse_dot={} vs dense dot={}", sparse_result, dense_result
            );
        }

        /// sparse_dot should be symmetric.
        #[test]
        fn sparse_dot_symmetric(
            a_raw in proptest::collection::vec((0u32..50, -1.0f32..1.0), 1..8),
            b_raw in proptest::collection::vec((0u32..50, -1.0f32..1.0), 1..8),
        ) {
            use innr::sparse_dot;

            let mut a = a_raw;
            a.sort_by_key(|(idx, _)| *idx);
            a.dedup_by_key(|(idx, _)| *idx);
            let (ai, av): (Vec<u32>, Vec<f32>) = a.into_iter().unzip();

            let mut b = b_raw;
            b.sort_by_key(|(idx, _)| *idx);
            b.dedup_by_key(|(idx, _)| *idx);
            let (bi, bv): (Vec<u32>, Vec<f32>) = b.into_iter().unzip();

            let ab = sparse_dot(&ai, &av, &bi, &bv);
            let ba = sparse_dot(&bi, &bv, &ai, &av);

            prop_assert!(
                (ab - ba).abs() < 1e-6,
                "sparse_dot not symmetric: ab={}, ba={}", ab, ba
            );
        }
    }
}
