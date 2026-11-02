//! Metrics as types.
//!
//! Each metric is a zero-sized type ([`Dot`], [`Cosine`], [`L2Sq`], [`L1`],
//! [`Hamming`]) implementing [`Metric<T>`] for the element types it supports.
//! Generic code takes `M: Metric<T>`; the call monomorphizes to one kernel,
//! and [`Metric::DIRECTION`] says whether a larger score is closer or farther,
//! so callers do not hand-write `1.0 - dot(..)` conversions.
//!
//! The free functions ([`crate::dot`], [`crate::cosine`], ...) are thin
//! wrappers over these types and remain the shortest way to score one pair.
//!
//! ```
//! use innr::metric::{Cosine, Direction, L2Sq, Metric};
//!
//! fn nearest<M: Metric<f32, Output = f32>>(query: &[f32], rows: &[Vec<f32>]) -> usize {
//!     let better = |a: f32, b: f32| match M::DIRECTION {
//!         Direction::Similarity => a > b,
//!         Direction::Distance => a < b,
//!     };
//!     let mut best = 0;
//!     for (i, row) in rows.iter().enumerate().skip(1) {
//!         if better(M::eval(query, row), M::eval(query, &rows[best])) {
//!             best = i;
//!         }
//!     }
//!     best
//! }
//!
//! let rows = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.7, 0.7]];
//! assert_eq!(nearest::<Cosine>(&[0.1, 1.0], &rows), 1);
//! assert_eq!(nearest::<L2Sq>(&[0.6, 0.6], &rows), 2);
//! ```

use core::fmt::Debug;

/// Element types the kernels read, each with the type its sums accumulate in.
///
/// Integer kernels accumulate in a wider type than the element so per-lane
/// sums cannot overflow for the lengths the kernel accepts; floats accumulate
/// in their own type. This trait is sealed.
pub trait Element: Copy + Send + Sync + Debug + 'static + sealed::Sealed {
    /// Accumulator type for sums of products of this element.
    type Acc: Copy + Send + Sync + Debug + 'static;
}

impl Element for f32 {
    type Acc = f32;
}

impl Element for f64 {
    type Acc = f64;
}

/// `u8` products are at most `255² = 65025`, so a `u32` lane holds the sum of
/// at least 66051 of them. Wider totals are reported as `u64`.
impl Element for u8 {
    type Acc = u32;
}

/// Whether a larger metric value means the vectors are closer or farther.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Larger is closer (dot product, cosine similarity).
    Similarity,
    /// Smaller is closer (L1, squared L2, Hamming).
    Distance,
}

/// A similarity or distance between two equal-length slices of `T`.
///
/// Implemented by the zero-sized metric types in this module. The trait is
/// sealed: its kernels are chosen per CPU at runtime and are not meant to be
/// replaced from outside the crate.
pub trait Metric<T: Element>:
    sealed::Kernel<T> + Copy + Default + Debug + Send + Sync + 'static
{
    /// Whether larger values mean closer or farther.
    const DIRECTION: Direction;

    /// Score `a` against `b` on the best kernel for this CPU.
    ///
    /// # Panics
    ///
    /// Panics if `a.len() != b.len()`.
    #[inline]
    #[must_use]
    fn eval(a: &[T], b: &[T]) -> Self::Output {
        assert_eq!(
            a.len(),
            b.len(),
            "innr::metric: slice length mismatch ({} vs {})",
            a.len(),
            b.len()
        );
        eval_unchecked::<T, Self>(a, b)
    }
}

/// Dispatch one call: SIMD kernel when the `simd` feature is on, a SIMD level
/// is available, and the input is long enough; otherwise the portable kernel.
///
/// Callers have already checked or truncated lengths.
#[inline]
pub(crate) fn eval_unchecked<T, K: sealed::Kernel<T>>(a: &[T], b: &[T]) -> K::Output {
    #[cfg(feature = "simd")]
    if a.len() >= K::MIN_SIMD_LEN {
        return crate::arch::simd_dispatch!(
            crate::arch::isa(),
            s => K::simd(s, a, b),
            _ => K::portable(a, b),
        );
    }
    K::portable(a, b)
}

pub(crate) mod sealed {
    /// Seals [`super::Element`].
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for u8 {}

    /// The kernels behind a metric. Crate-internal code generic over a metric
    /// (MaxSim, batch scans) calls `simd` inside one dispatch for many pairs.
    pub trait Kernel<T> {
        /// The score type.
        type Output: Copy + Send + Sync + core::fmt::Debug + 'static;

        /// Shortest input for which the SIMD kernel beats the portable one.
        const MIN_SIMD_LEN: usize;

        /// Portable kernel; compares the common prefix of `a` and `b`.
        fn portable(a: &[T], b: &[T]) -> Self::Output;

        /// SIMD kernel for level `simd`. Implementations are
        /// `#[inline(always)]` so they compile with the level's features.
        #[cfg(feature = "simd")]
        fn simd<S: fearless_simd::Simd>(simd: S, a: &[T], b: &[T]) -> Self::Output;
    }
}

use sealed::Kernel;

macro_rules! metric_type {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
        pub struct $name;
    };
}

metric_type!(
    /// Dot product `Σ a[i]·b[i]`. Similarity.
    ///
    /// Output: `f32` for `f32`, `f64` for `f64`, exact `u64` for `u8`.
    Dot
);
metric_type!(
    /// Cosine similarity, `0.0` when either norm is effectively zero.
    Cosine
);
metric_type!(
    /// Squared Euclidean distance `Σ (a[i] - b[i])²`.
    L2Sq
);
metric_type!(
    /// Manhattan distance `Σ |a[i] - b[i]|`.
    L1
);
metric_type!(
    /// Bit-level Hamming distance between byte-packed bit vectors.
    Hamming
);

#[cfg(feature = "simd")]
use crate::arch::kernels;
use crate::{dense, dense_f64, quant};

use crate::backend::THRESHOLDS;

const MIN_F32: usize = THRESHOLDS.dense_f32;
const MIN_F64: usize = THRESHOLDS.dense_f64;

macro_rules! float_metric {
    ($metric:ty, $dir:expr, $t:ty, $min:expr, $portable:path, $simd:ident) => {
        impl Kernel<$t> for $metric {
            type Output = $t;
            const MIN_SIMD_LEN: usize = $min;
            #[inline]
            fn portable(a: &[$t], b: &[$t]) -> $t {
                $portable(a, b)
            }
            #[cfg(feature = "simd")]
            #[inline(always)]
            fn simd<S: fearless_simd::Simd>(simd: S, a: &[$t], b: &[$t]) -> $t {
                kernels::$simd(simd, a, b)
            }
        }
        impl Metric<$t> for $metric {
            const DIRECTION: Direction = $dir;
        }
    };
}

float_metric!(
    Dot,
    Direction::Similarity,
    f32,
    MIN_F32,
    dense::dot_portable,
    dot_f32
);
float_metric!(
    L2Sq,
    Direction::Distance,
    f32,
    MIN_F32,
    dense::l2_distance_squared_portable,
    l2_squared_f32
);
float_metric!(
    L1,
    Direction::Distance,
    f32,
    MIN_F32,
    dense::l1_distance_portable,
    l1_f32
);
float_metric!(
    Dot,
    Direction::Similarity,
    f64,
    MIN_F64,
    dense_f64::dot_f64_portable,
    dot_f64
);
float_metric!(
    L2Sq,
    Direction::Distance,
    f64,
    MIN_F64,
    dense_f64::l2_distance_squared_f64_portable,
    l2_squared_f64
);
float_metric!(
    L1,
    Direction::Distance,
    f64,
    MIN_F64,
    dense_f64::l1_distance_f64_portable,
    l1_f64
);

impl Kernel<f32> for Cosine {
    type Output = f32;
    const MIN_SIMD_LEN: usize = MIN_F32;
    #[inline]
    fn portable(a: &[f32], b: &[f32]) -> f32 {
        dense::cosine_portable(a, b)
    }
    #[cfg(feature = "simd")]
    #[inline(always)]
    fn simd<S: fearless_simd::Simd>(simd: S, a: &[f32], b: &[f32]) -> f32 {
        let (ab, aa, bb) = kernels::cosine_parts_f32(simd, a, b);
        dense::finish_cosine(ab, aa, bb)
    }
}
impl Metric<f32> for Cosine {
    const DIRECTION: Direction = Direction::Similarity;
}

impl Kernel<f64> for Cosine {
    type Output = f64;
    const MIN_SIMD_LEN: usize = MIN_F64;
    #[inline]
    fn portable(a: &[f64], b: &[f64]) -> f64 {
        use dense_f64::dot_f64_portable as d;
        dense_f64::finish_cosine_f64(d(a, b), d(a, a), d(b, b))
    }
    #[cfg(feature = "simd")]
    #[inline(always)]
    fn simd<S: fearless_simd::Simd>(simd: S, a: &[f64], b: &[f64]) -> f64 {
        let d = |x, y| kernels::dot_f64(simd, x, y);
        dense_f64::finish_cosine_f64(d(a, b), d(a, a), d(b, b))
    }
}
impl Metric<f64> for Cosine {
    const DIRECTION: Direction = Direction::Similarity;
}

impl Kernel<u8> for Dot {
    type Output = u64;
    const MIN_SIMD_LEN: usize = THRESHOLDS.dot_u8;
    #[inline]
    fn portable(a: &[u8], b: &[u8]) -> u64 {
        quant::dot_u8_wide_portable(a, b)
    }
    #[cfg(feature = "simd")]
    #[inline(always)]
    fn simd<S: fearless_simd::Simd>(simd: S, a: &[u8], b: &[u8]) -> u64 {
        // Each chunk's total fits u32, so the kernel's u32 lanes cannot wrap.
        a.chunks(quant::MAX_U32_DOT_DIM)
            .zip(b.chunks(quant::MAX_U32_DOT_DIM))
            .map(|(a, b)| u64::from(kernels::dot_u8(simd, a, b)))
            .try_fold(0_u64, u64::checked_add)
            .expect("innr::dot_u8_wide: result exceeds u64::MAX")
    }
}
impl Metric<u8> for Dot {
    const DIRECTION: Direction = Direction::Similarity;
}

impl Kernel<u8> for Hamming {
    type Output = u32;
    const MIN_SIMD_LEN: usize = THRESHOLDS.hamming_u8;
    #[inline]
    fn portable(a: &[u8], b: &[u8]) -> u32 {
        quant::hamming_portable(a, b)
    }
    #[cfg(feature = "simd")]
    #[inline(always)]
    fn simd<S: fearless_simd::Simd>(simd: S, a: &[u8], b: &[u8]) -> u32 {
        kernels::hamming_u8(simd, a, b)
    }
}
impl Metric<u8> for Hamming {
    const DIRECTION: Direction = Direction::Distance;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_metrics_match_free_functions() {
        let a: Vec<f32> = (0..100).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..100).map(|i| (i as f32 * 0.11).cos()).collect();
        assert_eq!(Dot::eval(&a, &b), crate::dot(&a, &b));
        assert_eq!(Cosine::eval(&a, &b), crate::cosine(&a, &b));
        assert_eq!(L2Sq::eval(&a, &b), crate::l2_distance_squared(&a, &b));
        assert_eq!(L1::eval(&a, &b), crate::l1_distance(&a, &b));

        let ua: Vec<u8> = (0..100).map(|i| (i * 7) as u8).collect();
        let ub: Vec<u8> = (0..100).map(|i| (i * 13) as u8).collect();
        assert_eq!(Dot::eval(&ua, &ub), crate::dot_u8_wide(&ua, &ub));
        assert_eq!(Hamming::eval(&ua, &ub), crate::hamming_distance(&ua, &ub));
    }

    #[test]
    fn directions() {
        assert_eq!(<Dot as Metric<f32>>::DIRECTION, Direction::Similarity);
        assert_eq!(<Cosine as Metric<f64>>::DIRECTION, Direction::Similarity);
        assert_eq!(<L2Sq as Metric<f32>>::DIRECTION, Direction::Distance);
        assert_eq!(<Hamming as Metric<u8>>::DIRECTION, Direction::Distance);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn eval_rejects_length_mismatch() {
        let _ = <Dot as Metric<f32>>::eval(&[1.0, 2.0], &[1.0]);
    }
}
