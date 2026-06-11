//! End-to-end proof that the `anndists` feature makes innr's metric types
//! drop-in distances for `hnsw_rs`, which binds to `anndists::dist::Distance`
//! (not to a structurally similar trait). The index-construction tests here
//! guard the actual consumer binding, not just the trait signature.
#![cfg(feature = "anndists")]

use hnsw_rs::prelude::Hnsw;
use innr::distance::{DistCosine, DistDot, DistHamming, DistL1, DistL2, DistSlotU32};

/// The anndists impls delegate to the innr trait; assert the two never drift.
#[test]
fn anndists_eval_matches_innr_eval() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [4.0f32, 3.0, 2.0, 1.0];
    for metric_pair in [
        (
            <DistCosine as innr::distance::Distance<f32>>::eval(&DistCosine, &a, &b),
            <DistCosine as anndists::dist::Distance<f32>>::eval(&DistCosine, &a, &b),
        ),
        (
            <DistDot as innr::distance::Distance<f32>>::eval(&DistDot, &a, &b),
            <DistDot as anndists::dist::Distance<f32>>::eval(&DistDot, &a, &b),
        ),
        (
            <DistL2 as innr::distance::Distance<f32>>::eval(&DistL2, &a, &b),
            <DistL2 as anndists::dist::Distance<f32>>::eval(&DistL2, &a, &b),
        ),
        (
            <DistL1 as innr::distance::Distance<f32>>::eval(&DistL1, &a, &b),
            <DistL1 as anndists::dist::Distance<f32>>::eval(&DistL1, &a, &b),
        ),
    ] {
        assert_eq!(metric_pair.0, metric_pair.1);
    }

    let p = [0b1111_0000u8, 0xFF, 0x00];
    let q = [0b1010_1010u8, 0x0F, 0x00];
    assert_eq!(
        <DistHamming as innr::distance::Distance<u8>>::eval(&DistHamming, &p, &q),
        <DistHamming as anndists::dist::Distance<u8>>::eval(&DistHamming, &p, &q),
    );

    let s = [1u32, 2, 3, 4];
    let t = [1u32, 9, 3, 7];
    assert_eq!(
        <DistSlotU32 as innr::distance::Distance<u32>>::eval(&DistSlotU32, &s, &t),
        <DistSlotU32 as anndists::dist::Distance<u32>>::eval(&DistSlotU32, &s, &t),
    );
}

/// The jianshu93 use case (innr#1): an hnsw_rs index over MinHash-style u32
/// sketches using innr's integer-slot Hamming as the distance.
#[test]
fn hnsw_index_over_minhash_sketches_with_slot_u32() {
    const SLOTS: usize = 64;
    // Base sketch plus perturbed copies; sketch i differs from base in i slots.
    let sketches: Vec<Vec<u32>> = (0..16)
        .map(|i| {
            let mut v: Vec<u32> = (0..SLOTS as u32).collect();
            for (s, slot) in v.iter_mut().enumerate().take(i) {
                *slot = 1_000_000 + s as u32 + (i as u32) * SLOTS as u32;
            }
            v
        })
        .collect();

    let hnsw = Hnsw::<u32, DistSlotU32>::new(16, sketches.len(), 16, 200, DistSlotU32);
    for (id, sketch) in sketches.iter().enumerate() {
        hnsw.insert((sketch.as_slice(), id));
    }

    // A query one slot away from sketch 0 must rank sketch 0 first.
    let mut query: Vec<u32> = (0..SLOTS as u32).collect();
    query[SLOTS - 1] = 999;
    let neighbours = hnsw.search(&query, 3, 64);
    assert!(!neighbours.is_empty());
    assert_eq!(neighbours[0].d_id, 0);
    // Distance scale is the fraction of differing slots: 1 of 64.
    assert!((neighbours[0].distance - 1.0 / SLOTS as f32).abs() < 1e-6);
}

/// Cosine over f32 embeddings, the other common hnsw_rs pairing.
#[test]
fn hnsw_index_over_f32_with_cosine() {
    let corpus: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![0.7, 0.7, 0.0],
    ];
    let hnsw = Hnsw::<f32, DistCosine>::new(16, corpus.len(), 16, 200, DistCosine);
    for (id, v) in corpus.iter().enumerate() {
        hnsw.insert((v.as_slice(), id));
    }
    let neighbours = hnsw.search(&[0.9, 0.1, 0.0], 2, 32);
    assert_eq!(neighbours[0].d_id, 0);
}
