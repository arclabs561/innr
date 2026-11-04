//! The pre-0.7.6 path `innr::corpus::Corpus` still compiles and behaves as
//! `innr::packed::PackedRows`, for float rows and byte codes.
#![allow(deprecated)]

use innr::corpus::{Corpus, Hit, PackError};
use innr::metric::{Dot, Hamming};

#[test]
fn old_name_still_packs_and_scans() {
    let rows = [[1.0_f32, 0.0], [0.0, 1.0]];
    let c: Corpus = Corpus::from_rows(rows.iter().map(|r| r.as_slice()), 2).unwrap();
    let best: Vec<Hit> = c.top_k::<Dot>(&[0.0, 2.0], 1);
    assert_eq!(best[0].index, 1);

    let codes = [[0u8, 0], [0xFF, 0]];
    let b = Corpus::from_rows(codes.iter().map(|r| r.as_slice()), 2).unwrap();
    assert_eq!(b.top_k::<Hamming>(&[0xFF, 0], 1)[0].index, 1);

    assert_eq!(
        Corpus::<f32>::from_flat(&[1.0; 3], 2).unwrap_err(),
        PackError::NotMultipleOfDimension { len: 3, dim: 2 }
    );
}
