//! Scans over rows in the caller's own layout.
//!
//! Every function here takes a [`Rows`] source: a slice of `Vec`s, a slice
//! of slices, a slice of arrays, a [`Flat`] view of a row-major buffer, or a
//! [`PackedRows`]. Each call dispatches to the SIMD level once and loops over
//! the rows inside it, so a scan costs one dispatch rather than one per row
//! (on x86 a dispatch is a real function call).
//!
//! [`PackedRows`] is the fast source, not a requirement: its rows are aligned
//! (misaligned AVX2 loads cost up to 1.6x) and it keeps cosine row norms once
//! computed. Its methods call these functions.
//!
//! ```
//! use innr::metric::Dot;
//! use innr::scan::{self, Flat};
//!
//! // Rows stored the caller's way: one flat row-major buffer.
//! let data = [1.0_f32, 0.0, 0.0, 1.0, 0.6, 0.8];
//! let rows = Flat::new(&data, 2).unwrap();
//! let best = scan::top_k::<Dot, _, _>(&[0.0, 1.0], &rows, 2);
//! assert_eq!(best.iter().map(|h| h.index).collect::<Vec<_>>(), [1, 2]);
//!
//! // Or a Vec of Vecs, unchanged.
//! let nested = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0]];
//! assert_eq!(scan::top_k::<Dot, _, _>(&[0.0, 1.0], nested.as_slice(), 1)[0].index, 1);
//! ```

use crate::metric::{sealed::Kernel, Direction, Element, Metric};
use crate::packed::{Hit, PackError, PackedRows};

/// A borrowed source of equal-length rows.
///
/// Implemented for `[Vec<T>]`, `[&[T]]`, `[[T; N]]`, [`Flat`] and
/// [`PackedRows`]. The scan functions check each row's length against
/// [`Rows::dim`] and panic on a mismatch.
pub trait Rows<T> {
    /// Number of rows.
    fn len(&self) -> usize;

    /// Whether there are no rows.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Length every row must have. For an empty source any value is
    /// accepted.
    fn dim(&self) -> usize;

    /// Row `i`; may panic if `i >= self.len()`.
    fn row(&self, i: usize) -> &[T];

    /// The rows in order with their indices. Sources that can hold rows of
    /// different lengths check each against [`Rows::dim`] here; the default
    /// calls [`Rows::row`].
    fn iter_rows<'a>(&'a self) -> impl Iterator<Item = (usize, &'a [T])>
    where
        T: 'a,
    {
        (0..self.len()).map(move |i| (i, self.row(i)))
    }

    /// Per-row values a metric can reuse (cosine: squared norms), if the
    /// source keeps them. [`PackedRows`] returns them once a cosine scan has
    /// computed them.
    fn norms(&self) -> Option<&[T]> {
        None
    }
}

/// A row of a source that can be ragged, checked against the first row.
#[inline]
fn checked<T>(r: &[T], i: usize, dim: usize) -> &[T] {
    assert_eq!(
        r.len(),
        dim,
        "innr::scan: row {i} has {} elements, expected {dim}",
        r.len()
    );
    r
}

impl<T> Rows<T> for [Vec<T>] {
    fn len(&self) -> usize {
        <[Vec<T>]>::len(self)
    }
    fn dim(&self) -> usize {
        self.first().map_or(0, Vec::len)
    }
    fn row(&self, i: usize) -> &[T] {
        checked(&self[i], i, Rows::dim(self))
    }
    fn iter_rows<'a>(&'a self) -> impl Iterator<Item = (usize, &'a [T])>
    where
        T: 'a,
    {
        let dim = Rows::dim(self);
        self.iter()
            .enumerate()
            .map(move |(i, r)| (i, checked(r, i, dim)))
    }
}

impl<T> Rows<T> for [&[T]] {
    fn len(&self) -> usize {
        <[&[T]]>::len(self)
    }
    fn dim(&self) -> usize {
        self.first().map_or(0, |r| r.len())
    }
    fn row(&self, i: usize) -> &[T] {
        checked(self[i], i, Rows::dim(self))
    }
    fn iter_rows<'a>(&'a self) -> impl Iterator<Item = (usize, &'a [T])>
    where
        T: 'a,
    {
        let dim = Rows::dim(self);
        self.iter()
            .enumerate()
            .map(move |(i, r)| (i, checked(r, i, dim)))
    }
}

impl<T, const N: usize> Rows<T> for [[T; N]] {
    fn len(&self) -> usize {
        <[[T; N]]>::len(self)
    }
    fn dim(&self) -> usize {
        N
    }
    fn row(&self, i: usize) -> &[T] {
        &self[i]
    }
    fn iter_rows<'a>(&'a self) -> impl Iterator<Item = (usize, &'a [T])>
    where
        T: 'a,
    {
        self.iter().map(<[T; N]>::as_slice).enumerate()
    }
}

impl<T: Element + Default> Rows<T> for PackedRows<T> {
    fn len(&self) -> usize {
        PackedRows::len(self)
    }
    fn dim(&self) -> usize {
        PackedRows::dim(self)
    }
    fn row(&self, i: usize) -> &[T] {
        PackedRows::row(self, i)
    }
    fn iter_rows<'a>(&'a self) -> impl Iterator<Item = (usize, &'a [T])>
    where
        T: 'a,
    {
        self.strided_rows().enumerate()
    }
    fn norms(&self) -> Option<&[T]> {
        self.stored_norms()
    }
}

/// A row-major buffer viewed as rows of `dim` elements, without copying.
#[derive(Debug, Clone, Copy)]
pub struct Flat<'a, T> {
    data: &'a [T],
    dim: usize,
}

impl<'a, T> Flat<'a, T> {
    /// View `data` as `data.len() / dim` rows.
    ///
    /// # Errors
    ///
    /// [`PackError::ZeroDimension`] if `dim == 0`;
    /// [`PackError::NotMultipleOfDimension`] if `data.len() % dim != 0`.
    pub fn new(data: &'a [T], dim: usize) -> Result<Self, PackError> {
        if dim == 0 {
            return Err(PackError::ZeroDimension);
        }
        if !data.len().is_multiple_of(dim) {
            return Err(PackError::NotMultipleOfDimension {
                len: data.len(),
                dim,
            });
        }
        Ok(Self { data, dim })
    }
}

impl<T> Rows<T> for Flat<'_, T> {
    fn len(&self) -> usize {
        self.data.len() / self.dim
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn row(&self, i: usize) -> &[T] {
        &self.data[i * self.dim..][..self.dim]
    }
    fn iter_rows<'a>(&'a self) -> impl Iterator<Item = (usize, &'a [T])>
    where
        T: 'a,
    {
        self.data.chunks_exact(self.dim).enumerate()
    }
}

/// A metric output that [`top_k`] and [`rerank`] can rank: `f32` (NaN last)
/// and the exact integer outputs `u32` and `u64` (ties by row order). Sealed.
pub trait Score: Copy + sealed::Select {}
impl Score for f32 {}
impl Score for u32 {}
impl Score for u64 {}

pub(crate) mod sealed {
    use super::{Direction, Hit};
    use crate::topk::TopK;
    use core::cmp::Reverse;
    use std::collections::BinaryHeap;

    /// Streaming best-`k` selection for one score type.
    pub trait Select: Sized {
        type Sel;
        fn start(k: usize, dir: Direction) -> Self::Sel;
        fn push(sel: &mut Self::Sel, row: usize, score: Self);
        fn finish(sel: Self::Sel) -> Vec<Hit<Self>>;
    }

    pub struct FloatSel {
        top: TopK,
        flip: bool,
    }

    impl Select for f32 {
        type Sel = FloatSel;
        fn start(k: usize, dir: Direction) -> FloatSel {
            // TopK keeps the smallest keys; negate similarities.
            FloatSel {
                top: TopK::new(k),
                flip: matches!(dir, Direction::Similarity),
            }
        }
        #[inline]
        fn push(sel: &mut FloatSel, row: usize, score: f32) {
            let id = u32::try_from(row).expect("innr::scan: more than u32::MAX rows");
            sel.top.insert(id, if sel.flip { -score } else { score });
        }
        fn finish(sel: FloatSel) -> Vec<Hit> {
            let flip = sel.flip;
            sel.top
                .into_sorted()
                .into_iter()
                .map(|(i, s)| Hit {
                    index: i as usize,
                    score: if flip { -s } else { s },
                })
                .collect()
        }
    }

    /// A max-heap of the `k` best seen, whose top is the worst kept hit; the
    /// row breaks ties toward the earlier row.
    pub enum IntSel<S: Ord> {
        Smallest(usize, BinaryHeap<(S, usize)>),
        Largest(usize, BinaryHeap<(Reverse<S>, usize)>),
    }

    macro_rules! int_select {
        ($t:ty) => {
            impl Select for $t {
                type Sel = IntSel<$t>;
                fn start(k: usize, dir: Direction) -> IntSel<$t> {
                    match dir {
                        Direction::Distance => {
                            IntSel::Smallest(k, BinaryHeap::with_capacity(k + 1))
                        }
                        Direction::Similarity => {
                            IntSel::Largest(k, BinaryHeap::with_capacity(k + 1))
                        }
                    }
                }
                #[inline]
                fn push(sel: &mut IntSel<$t>, row: usize, s: $t) {
                    match sel {
                        IntSel::Smallest(k, heap) => {
                            if heap.len() < *k {
                                heap.push((s, row));
                            } else if heap.peek().is_some_and(|&(worst, _)| s < worst) {
                                heap.pop();
                                heap.push((s, row));
                            }
                        }
                        IntSel::Largest(k, heap) => {
                            if heap.len() < *k {
                                heap.push((Reverse(s), row));
                            } else if heap.peek().is_some_and(|&(Reverse(worst), _)| s > worst) {
                                heap.pop();
                                heap.push((Reverse(s), row));
                            }
                        }
                    }
                }
                fn finish(sel: IntSel<$t>) -> Vec<Hit<$t>> {
                    // Best first; equal scores by row.
                    match sel {
                        IntSel::Smallest(_, heap) => {
                            let mut v = heap.into_vec();
                            v.sort_unstable();
                            v.into_iter()
                                .map(|(score, index)| Hit { index, score })
                                .collect()
                        }
                        IntSel::Largest(_, heap) => {
                            let mut v = heap.into_vec();
                            v.sort_unstable();
                            v.into_iter()
                                .map(|(Reverse(score), index)| Hit { index, score })
                                .collect()
                        }
                    }
                }
            }
        };
    }
    int_select!(u32);
    int_select!(u64);
}

/// Call `f(row, score)` for the listed rows (all rows if `ids` is `None`),
/// in order, with one SIMD dispatch.
///
/// The loops are written out per case (SIMD or portable, stored norms or
/// not, all rows or a list) rather than through a scoring closure: a closure
/// holding SIMD code can compile out of line without the level's target
/// features, which made every vector op a call in an earlier kernel.
fn for_each<M, T, R>(q: &[T], rows: &R, ids: Option<&[usize]>, mut f: impl FnMut(usize, M::Output))
where
    T: Element,
    M: Kernel<T>,
    R: Rows<T> + ?Sized,
{
    let dim = rows.dim();
    assert!(
        rows.is_empty() || q.len() == dim,
        "innr::scan: query has {} elements, rows have {dim}",
        q.len()
    );
    let norms = if M::NORMED { rows.norms() } else { None };
    let nq = norms.map(|_| M::row_norm(q).expect("innr: NORMED metric without row_norm"));
    // `$i` and `$r` name the row index and slice inside `$score`, which is
    // expanded into the loop body, not wrapped in a closure.
    macro_rules! scan {
        ($i:ident, $r:ident => $score:expr) => {
            match ids {
                None => {
                    for ($i, $r) in rows.iter_rows() {
                        f($i, $score);
                    }
                }
                Some(ids) => {
                    for &$i in ids {
                        let $r = rows.row($i);
                        f($i, $score);
                    }
                }
            }
        };
    }
    #[cfg(feature = "simd")]
    if dim >= M::MIN_SCAN_LEN {
        crate::arch::simd_dispatch!(
            crate::arch::isa(),
            s => match (norms, nq) {
                (Some(nr), Some(nq)) => scan!(i, r => M::simd_normed(s, q, r, nq, nr[i])),
                _ => scan!(i, r => M::simd(s, q, r)),
            },
            _ => match (norms, nq) {
                (Some(nr), Some(nq)) => scan!(i, r => M::portable_normed(q, r, nq, nr[i])),
                _ => scan!(i, r => M::portable(q, r)),
            },
        );
        return;
    }
    match (norms, nq) {
        (Some(nr), Some(nq)) => scan!(i, r => M::portable_normed(q, r, nq, nr[i])),
        _ => scan!(i, r => M::portable(q, r)),
    }
}

/// Score `query` against every row, in row order, into `out` (cleared first).
///
/// # Panics
///
/// Panics if `query.len()` or a row's length differs from `rows.dim()`.
pub fn scores_into<M, T, R>(query: &[T], rows: &R, out: &mut Vec<M::Output>)
where
    T: Element,
    M: Metric<T>,
    R: Rows<T> + ?Sized,
{
    out.clear();
    out.reserve(rows.len());
    for_each::<M, T, R>(query, rows, None, |_, s| out.push(s));
}

/// [`scores_into`] into a new vector.
///
/// # Panics
///
/// As [`scores_into`].
#[must_use]
pub fn scores<M, T, R>(query: &[T], rows: &R) -> Vec<M::Output>
where
    T: Element,
    M: Metric<T>,
    R: Rows<T> + ?Sized,
{
    let mut out = Vec::new();
    scores_into::<M, T, R>(query, rows, &mut out);
    out
}

/// The `k` best rows for `query`, best first by the metric's [`Direction`].
///
/// `f32` scores rank NaN last; integer scores (`Hamming`, `Dot` on `u8`)
/// rank exactly, ties by row order. Fewer than `k` hits only when there are
/// fewer rows.
///
/// # Panics
///
/// As [`scores_into`].
#[must_use]
pub fn top_k<M, T, R>(query: &[T], rows: &R, k: usize) -> Vec<Hit<M::Output>>
where
    T: Element,
    M: Metric<T>,
    M::Output: Score,
    R: Rows<T> + ?Sized,
{
    select::<M, T, R>(query, rows, None, k)
}

/// The `k` best of `candidates` (row indices) for `query`, best first, with
/// the rows' own indices: the exact second stage after a cheap scan such as
/// [`sign_dot_top_k`] over codes of the same rows. An index listed twice is
/// scored twice.
///
/// # Panics
///
/// As [`scores_into`], or if a candidate is out of range.
#[must_use]
pub fn rerank<M, T, R, I>(query: &[T], rows: &R, candidates: I, k: usize) -> Vec<Hit<M::Output>>
where
    T: Element,
    M: Metric<T>,
    M::Output: Score,
    R: Rows<T> + ?Sized,
    I: IntoIterator<Item = usize>,
{
    let ids: Vec<usize> = candidates.into_iter().collect();
    if let Some(&bad) = ids.iter().find(|&&i| i >= rows.len()) {
        panic!(
            "innr::scan::rerank: candidate {bad} out of range ({})",
            rows.len()
        );
    }
    select::<M, T, R>(query, rows, Some(&ids), k)
}

fn select<M, T, R>(q: &[T], rows: &R, ids: Option<&[usize]>, k: usize) -> Vec<Hit<M::Output>>
where
    T: Element,
    M: Metric<T>,
    M::Output: Score,
    R: Rows<T> + ?Sized,
{
    let k = k.min(ids.map_or(rows.len(), <[usize]>::len));
    if k == 0 {
        return Vec::new();
    }
    let mut sel = <M::Output as sealed::Select>::start(k, M::DIRECTION);
    for_each::<M, T, R>(q, rows, ids, |i, s| {
        <M::Output as sealed::Select>::push(&mut sel, i, s)
    });
    <M::Output as sealed::Select>::finish(sel)
}

/// [`sign_dot`](crate::binary::sign_dot) of `query` against every code, in
/// row order, into `out` (cleared first), with one dispatch.
///
/// # Panics
///
/// Panics if a code is shorter than `query.len().div_ceil(8)` bytes.
pub fn sign_dot_into<R>(query: &[f32], codes: &R, out: &mut Vec<f32>)
where
    R: Rows<u8> + ?Sized,
{
    out.clear();
    out.reserve(codes.len());
    for_each_sign_dot(query, codes, |_, s| out.push(s));
}

/// The `k` codes with the largest [`sign_dot`](crate::binary::sign_dot)
/// against `query`, best first; NaN last.
///
/// # Panics
///
/// As [`sign_dot_into`].
#[must_use]
pub fn sign_dot_top_k<R>(query: &[f32], codes: &R, k: usize) -> Vec<Hit>
where
    R: Rows<u8> + ?Sized,
{
    let k = k.min(codes.len());
    if k == 0 {
        return Vec::new();
    }
    let mut sel = <f32 as sealed::Select>::start(k, Direction::Similarity);
    for_each_sign_dot(query, codes, |i, s| {
        <f32 as sealed::Select>::push(&mut sel, i, s)
    });
    <f32 as sealed::Select>::finish(sel)
}

fn for_each_sign_dot<R>(query: &[f32], codes: &R, mut f: impl FnMut(usize, f32))
where
    R: Rows<u8> + ?Sized,
{
    let dim = codes.dim();
    assert!(
        codes.is_empty() || dim * 8 >= query.len(),
        "innr::scan::sign_dot: codes have {dim} bytes, query needs {}",
        query.len().div_ceil(8)
    );
    let row = |i: usize| {
        let r = codes.row(i);
        assert_eq!(
            r.len(),
            dim,
            "innr::scan: row {i} has {} bytes, expected {dim}",
            r.len()
        );
        r
    };
    #[cfg(feature = "simd")]
    if query.len() >= 8 {
        crate::arch::simd_dispatch!(
            crate::arch::isa(),
            s => {
                for i in 0..codes.len() {
                    f(i, crate::arch::kernels::sign_dot_f32(s, query, row(i)));
                }
            },
            _ => {
                for i in 0..codes.len() {
                    f(i, crate::portable::sign_dot_f32(query, row(i)));
                }
            },
        );
        return;
    }
    for i in 0..codes.len() {
        f(i, crate::portable::sign_dot_f32(query, row(i)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{Cosine, Dot, Hamming, L2Sq};

    fn data(n: usize, dim: usize, seed: u32) -> Vec<Vec<f32>> {
        (0..n)
            .map(|r| {
                (0..dim)
                    .map(|i| {
                        ((((r * dim + i) as u32).wrapping_mul(2654435761) ^ seed) % 2001) as f32
                            / 1000.0
                            - 1.0
                    })
                    .collect()
            })
            .collect()
    }

    /// Every layout gives the same scores, top-k and rerank as PackedRows.
    #[test]
    fn layouts_agree_with_packed_rows() {
        let (n, dim) = (30, 20);
        let nested = data(n, dim, 9);
        let flat: Vec<f32> = nested.concat();
        let slices: Vec<&[f32]> = nested.iter().map(Vec::as_slice).collect();
        let arrays: Vec<[f32; 20]> = nested
            .iter()
            .map(|r| r.as_slice().try_into().unwrap())
            .collect();
        let packed = PackedRows::from_flat(&flat, dim).unwrap();
        let flat_view = Flat::new(&flat, dim).unwrap();
        let q = &data(1, dim, 4)[0];
        let ids = [3usize, 17, 8, 29, 0];

        let want_dot = packed.scores::<Dot>(q);
        let want_top = packed.top_k::<L2Sq>(q, 5);
        let want_cos = packed.top_k::<Cosine>(q, 4);
        let want_re = packed.rerank::<Dot, _>(q, ids, 3);
        let close = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .all(|(x, y)| (x - y).abs() <= 1e-5 * x.abs().max(1.0))
        };
        let idx = |h: &[Hit]| h.iter().map(|h| h.index).collect::<Vec<_>>();

        macro_rules! check {
            ($rows:expr) => {
                assert!(close(&scores::<Dot, _, _>(q, $rows), &want_dot));
                assert_eq!(idx(&top_k::<L2Sq, _, _>(q, $rows, 5)), idx(&want_top));
                assert_eq!(idx(&top_k::<Cosine, _, _>(q, $rows, 4)), idx(&want_cos));
                assert_eq!(
                    idx(&rerank::<Dot, _, _, _>(q, $rows, ids, 3)),
                    idx(&want_re)
                );
            };
        }
        check!(nested.as_slice());
        check!(slices.as_slice());
        check!(arrays.as_slice());
        check!(&flat_view);
        check!(&packed);
    }

    /// Byte codes in a flat buffer rank like PackedRows<u8>, for Hamming
    /// (exact integers) and sign_dot.
    #[test]
    fn flat_byte_codes_agree_with_packed_rows() {
        let codes: Vec<u8> = (0..40 * 12)
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 7) as u8)
            .collect();
        let flat = Flat::new(&codes, 12).unwrap();
        let packed = PackedRows::from_flat(&codes, 12).unwrap();
        let q = &codes[5 * 12..6 * 12];
        assert_eq!(
            top_k::<Hamming, _, _>(q, &flat, 6),
            packed.top_k::<Hamming>(q, 6)
        );
        let fq = &data(1, 96, 2)[0];
        let mut a = Vec::new();
        let mut b = Vec::new();
        sign_dot_into(fq, &flat, &mut a);
        packed.sign_dot_into(fq, &mut b);
        assert_eq!(a, b);
        assert_eq!(sign_dot_top_k(fq, &flat, 5), packed.sign_dot_top_k(fq, 5));
    }

    #[test]
    #[should_panic(expected = "row 1 has 2 elements, expected 3")]
    fn ragged_rows_panic() {
        let rows = vec![vec![1.0_f32; 3], vec![1.0; 2]];
        let _ = scores::<Dot, _, _>(&[1.0; 3], rows.as_slice());
    }

    #[test]
    fn flat_rejects_bad_shapes() {
        assert_eq!(
            Flat::new(&[1.0_f32; 4], 0).unwrap_err(),
            PackError::ZeroDimension
        );
        assert_eq!(
            Flat::new(&[1.0_f32; 5], 2).unwrap_err(),
            PackError::NotMultipleOfDimension { len: 5, dim: 2 }
        );
        let empty: [Vec<f32>; 0] = [];
        assert!(top_k::<Dot, _, _>(&[1.0], empty.as_slice(), 3).is_empty());
    }
}
