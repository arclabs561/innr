#![allow(missing_docs)]
//! Matrix-vector shapes: a row-major `rows x dim` matrix times one vector,
//! as innr's consumers write it today (one `innr::dot` per row).
//!
//! Each consumer function is copied twice from its source: the `simd`/`innr`
//! branch (labelled with this build's innr path) and the no-innr branch
//! (`local_*`, measured only in the `simd` build). Sources at qntz 23aba39,
//! vicinity 798f4d0, wass 0d708c4.

use std::fmt::Display;
use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion};
use innr_bench_support::data;
use ndarray::{Array1, Array2};

const INNR: &str = if cfg!(feature = "simd") {
    "innr_simd"
} else {
    "innr_portable"
};
const OTHERS: bool = cfg!(feature = "simd");

fn run<O>(
    g: &mut BenchmarkGroup<'_, WallTime>,
    variant: &str,
    param: impl Display,
    mut f: impl FnMut() -> O,
) {
    g.bench_function(BenchmarkId::new(variant, param), |b| b.iter(&mut f));
}

// --- qntz src/rabitq.rs:1585-1603 (`apply_rotation`), square rotation. ---

fn qntz_rabitq_rotation_innr(vector: &[f32], rotation: &[f32], dimension: usize) -> Vec<f32> {
    let nrows = rotation.len() / dimension;
    let mut result = vec![0.0f32; nrows];
    for (i, out) in result.iter_mut().enumerate() {
        let row = &rotation[i * dimension..(i + 1) * dimension];
        *out = innr::dot(row, vector);
    }
    result
}

#[allow(clippy::needless_range_loop)]
fn qntz_rabitq_rotation_local(vector: &[f32], rotation: &[f32], dimension: usize) -> Vec<f32> {
    let nrows = rotation.len() / dimension;
    let mut result = vec![0.0f32; nrows];
    for (i, out) in result.iter_mut().enumerate() {
        let row = &rotation[i * dimension..(i + 1) * dimension];
        {
            let mut sum = 0.0f32;
            for j in 0..dimension {
                sum += row[j] * vector[j];
            }
            *out = sum;
        }
    }
    result
}

// --- qntz src/binary.rs:305-339 (`apply_rotation_rect`). ---

fn qntz_binary_rotation_innr(
    rotation: &[f32],
    vector: &[f32],
    dim: usize,
    projected_dim: usize,
) -> Vec<f32> {
    let mut result = vec![0.0f32; projected_dim];
    for (i, out) in result.iter_mut().enumerate() {
        let row = &rotation[i * dim..(i + 1) * dim];
        *out = innr::dot(row, vector);
    }
    result
}

fn qntz_binary_rotation_local(
    rotation: &[f32],
    vector: &[f32],
    dim: usize,
    projected_dim: usize,
) -> Vec<f32> {
    let mut result = vec![0.0f32; projected_dim];
    for (i, out) in result.iter_mut().enumerate() {
        let row = &rotation[i * dim..(i + 1) * dim];
        {
            let chunks = dim / 4;
            let mut s0 = 0.0f32;
            let mut s1 = 0.0f32;
            let mut s2 = 0.0f32;
            let mut s3 = 0.0f32;
            for c in 0..chunks {
                let b = c * 4;
                s0 += row[b] * vector[b];
                s1 += row[b + 1] * vector[b + 1];
                s2 += row[b + 2] * vector[b + 2];
                s3 += row[b + 3] * vector[b + 3];
            }
            let mut sum = (s0 + s1) + (s2 + s3);
            for j in (chunks * 4)..dim {
                sum += row[j] * vector[j];
            }
            *out = sum;
        }
    }
    result
}

// --- vicinity src/adsampling.rs:437-444 (`rotate_vector`). With the default
// `innr` feature `crate::simd::dot` is `innr::dot`; without it, the fallback
// at src/simd.rs:89-91. ---

fn vicinity_rotate_innr(v: &[f32], rotation_t: &[f32], dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dim];
    for i in 0..dim {
        let row = &rotation_t[i * dim..(i + 1) * dim];
        result[i] = innr::dot(row, v);
    }
    result
}

#[inline]
fn vicinity_fallback_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn vicinity_rotate_local(v: &[f32], rotation_t: &[f32], dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dim];
    for i in 0..dim {
        let row = &rotation_t[i * dim..(i + 1) * dim];
        result[i] = vicinity_fallback_dot(row, v);
    }
    result
}

// --- wass src/lib.rs:1843-1868 (`project_and_sort`), projection only: the
// trailing sort is identical in both branches and excluded. ---

fn wass_project_innr(points: &Array2<f32>, direction: &Array1<f32>, n_points: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let row = points.row(i);
        let projection = match row.as_slice() {
            Some(values) => innr::dense::dot(values, direction.as_slice().unwrap()),
            None => row.dot(direction),
        };
        v.push(projection);
    }
    v
}

fn wass_project_local(points: &Array2<f32>, direction: &Array1<f32>) -> Vec<f32> {
    points.dot(direction).to_vec()
}

fn qntz(c: &mut Criterion) {
    // qntz benches RaBitQ and binary quantizers at d = 128 and 768, with
    // projected_dim = dim for the binary quantizer.
    let mut g = c.benchmark_group("matvec_qntz_rabitq_rotation");
    for d in [128usize, 768] {
        let (m, v) = (data::f32s(d * d, 1), data::f32s(d, 2));
        let p = format!("{d}x{d}");
        run(&mut g, INNR, &p, || {
            qntz_rabitq_rotation_innr(black_box(&v), black_box(&m), d)
        });
        if OTHERS {
            run(&mut g, "local_qntz", &p, || {
                qntz_rabitq_rotation_local(black_box(&v), black_box(&m), d)
            });
        }
    }
    g.finish();

    let mut g = c.benchmark_group("matvec_qntz_binary_rotation");
    for d in [128usize, 768] {
        let (m, v) = (data::f32s(d * d, 1), data::f32s(d, 2));
        let p = format!("{d}x{d}");
        run(&mut g, INNR, &p, || {
            qntz_binary_rotation_innr(black_box(&m), black_box(&v), d, d)
        });
        if OTHERS {
            run(&mut g, "local_qntz", &p, || {
                qntz_binary_rotation_local(black_box(&m), black_box(&v), d, d)
            });
        }
    }
    g.finish();
}

fn vicinity(c: &mut Criterion) {
    // ADSampling rotates each query by a dim x dim orthogonal matrix.
    let mut g = c.benchmark_group("matvec_vicinity_adsampling");
    for d in [128usize, 768] {
        let (m, v) = (data::f32s(d * d, 1), data::f32s(d, 2));
        let p = format!("{d}x{d}");
        run(&mut g, INNR, &p, || {
            vicinity_rotate_innr(black_box(&v), black_box(&m), d)
        });
        if OTHERS {
            run(&mut g, "local_vicinity", &p, || {
                vicinity_rotate_local(black_box(&v), black_box(&m), d)
            });
        }
    }
    g.finish();
}

fn wass(c: &mut Criterion) {
    // Sliced Wasserstein projects n points onto one random direction per
    // slice. wass's examples use 2-d point clouds; 64 and 768 cover
    // embedding clouds.
    let mut g = c.benchmark_group("matvec_wass_projection");
    for (n, d) in [(1000usize, 2usize), (1000, 64), (1000, 768)] {
        let points = Array2::from_shape_vec((n, d), data::f32s(n * d, 1)).unwrap();
        let direction = Array1::from_vec(data::unit_vec(d, 2));
        let p = format!("{n}x{d}");
        run(&mut g, INNR, &p, || {
            wass_project_innr(black_box(&points), black_box(&direction), n)
        });
        if OTHERS {
            run(&mut g, "local_wass_ndarray", &p, || {
                wass_project_local(black_box(&points), black_box(&direction))
            });
        }
    }
    g.finish();
}

criterion_group! {
    name = benches;
    config = innr_bench_support::config();
    targets = qntz, vicinity, wass
}
criterion_main!(benches);
