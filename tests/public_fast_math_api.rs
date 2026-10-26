//! Compile-time and behavior checks for the public fast-math API.

use innr::fast_math::{
    fast_cosine as module_fast_cosine, fast_cosine_dispatch as module_fast_cosine_dispatch,
    fast_rsqrt as module_fast_rsqrt, fast_rsqrt_precise as module_fast_rsqrt_precise,
};
use innr::{fast_cosine, fast_cosine_dispatch, fast_rsqrt, fast_rsqrt_precise};

#[test]
fn root_and_module_paths_remain_callable() {
    let a = [1.0_f32, 0.0, 0.0];
    let b = [0.707, 0.707, 0.0];

    assert!((fast_rsqrt(4.0) - 0.5).abs() < 1e-3);
    assert!((fast_rsqrt_precise(4.0) - 0.5).abs() < 1e-5);
    assert!((fast_cosine(&a, &b) - 0.707).abs() < 0.01);
    assert!((fast_cosine_dispatch(&a, &b) - 0.707).abs() < 0.02);

    assert!((module_fast_rsqrt(4.0) - 0.5).abs() < 1e-3);
    assert!((module_fast_rsqrt_precise(4.0) - 0.5).abs() < 1e-5);
    assert!((module_fast_cosine(&a, &b) - 0.707).abs() < 0.01);
    assert!((module_fast_cosine_dispatch(&a, &b) - 0.707).abs() < 0.02);
}

#[test]
#[should_panic(expected = "innr::fast_cosine_dispatch: slice length mismatch")]
fn root_dispatch_keeps_mismatched_length_panic() {
    let _ = fast_cosine_dispatch(&[1.0], &[1.0, 2.0]);
}

#[test]
#[should_panic(expected = "innr::fast_cosine: slice length mismatch")]
fn root_cosine_keeps_mismatched_length_panic() {
    let _ = fast_cosine(&[1.0], &[1.0, 2.0]);
}

#[test]
#[should_panic(expected = "innr::fast_cosine: slice length mismatch")]
fn module_cosine_keeps_mismatched_length_panic() {
    let _ = module_fast_cosine(&[1.0], &[1.0, 2.0]);
}

#[test]
#[should_panic(expected = "innr::fast_cosine_dispatch: slice length mismatch")]
fn module_dispatch_keeps_mismatched_length_panic() {
    let _ = module_fast_cosine_dispatch(&[1.0], &[1.0, 2.0]);
}
