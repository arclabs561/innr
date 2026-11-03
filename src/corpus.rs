//! A packed corpus: pack the rows once, then score many queries against it.
//!
//! [`Corpus`] stores `f32` rows contiguously, each row starting on a 64-byte
//! boundary (from 16 dimensions up), and scores a query against every row
//! inside one SIMD dispatch. Compared with calling a pair kernel per row:
//!
//! - the backend is chosen once per call, not once per row (on x86 the
//!   dispatch is a real call, which is most of the cost at small dimensions);
//! - rows and the query are aligned, so wide loads never split across cache
//!   lines (on AVX2 a buffer 16 bytes off a 32-byte boundary was 1.5x slower);
//! - [`Corpus::top_k`] keeps the best `k` as it scans, without building a
//!   score vector.
//!
//! The metric is a type parameter ([`crate::metric`]); its
//! [`Direction`] decides whether larger or smaller scores rank first.
//!
//! ```
//! use innr::corpus::Corpus;
//! use innr::metric::{Cosine, L2Sq};
//!
//! let rows = [[1.0_f32, 0.0], [0.0, 1.0], [0.7, 0.7]];
//! let corpus = Corpus::from_rows(rows.iter().map(|r| r.as_slice()), 2).unwrap();
//!
//! let best = corpus.top_k::<Cosine>(&[0.1, 1.0], 1);
//! assert_eq!(best[0].index, 1);
//!
//! let nearest = corpus.top_k::<L2Sq>(&[0.6, 0.6], 2);
//! assert_eq!(nearest[0].index, 2);
//! assert!(nearest[0].score < nearest[1].score);
//! ```

use crate::metric::{sealed::Kernel, Direction, Metric};
use crate::topk::TopK;

/// One result of [`Corpus::top_k`]: a row and its score under the metric.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// Index of the row in the corpus.
    pub index: usize,
    /// The metric's value for the query and this row.
    pub score: f32,
}

/// Row stride and buffer alignment, in `f32` elements (64 bytes).
const ALIGN: usize = 16;

/// Why [`Corpus::from_flat`] or [`Corpus::from_rows`] rejected its input.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PackError {
    /// The dimension was zero.
    ZeroDimension,
    /// A flat buffer's length is not a multiple of the dimension.
    NotMultipleOfDimension {
        /// Length of the flat buffer.
        len: usize,
        /// The dimension it was split by.
        dim: usize,
    },
    /// A row's length differs from the dimension.
    RaggedRow {
        /// Index of the offending row.
        row: usize,
        /// The dimension every row must have.
        expected: usize,
        /// The row's actual length.
        got: usize,
    },
}

impl core::fmt::Display for PackError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "innr::Corpus: dimension must be at least 1"),
            Self::NotMultipleOfDimension { len, dim } => write!(
                f,
                "innr::Corpus: flat length {len} is not a multiple of dimension {dim}"
            ),
            Self::RaggedRow { row, expected, got } => write!(
                f,
                "innr::Corpus: row {row} has {got} elements, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for PackError {}

/// `f32` rows packed for repeated scoring. See the [module docs](self).
#[derive(Debug, Clone)]
pub struct Corpus {
    /// Backing storage; rows start at `start` and are `stride` apart.
    buf: Vec<f32>,
    start: usize,
    stride: usize,
    dim: usize,
    len: usize,
}

/// Elements between consecutive rows: the dimension rounded up to 64 bytes
/// from 16 dimensions up. Below that the SIMD kernels do not run and padding
/// would multiply the memory for little gain.
fn stride_for(dim: usize) -> usize {
    if dim >= ALIGN {
        dim.div_ceil(ALIGN) * ALIGN
    } else {
        dim
    }
}

/// A zeroed buffer of `n` elements whose element `start` is 64-byte aligned.
fn aligned_zeroed(n: usize) -> (Vec<f32>, usize) {
    let buf = vec![0.0_f32; n + ALIGN];
    let misalign = buf.as_ptr() as usize % (ALIGN * 4);
    let start = (ALIGN * 4 - misalign) % (ALIGN * 4) / 4;
    (buf, start)
}

impl Corpus {
    /// Pack `len / dim` rows from a row-major buffer.
    ///
    /// # Errors
    ///
    /// [`PackError::ZeroDimension`] if `dim == 0`;
    /// [`PackError::NotMultipleOfDimension`] if `data.len() % dim != 0`.
    pub fn from_flat(data: &[f32], dim: usize) -> Result<Self, PackError> {
        if dim == 0 {
            return Err(PackError::ZeroDimension);
        }
        if !data.len().is_multiple_of(dim) {
            return Err(PackError::NotMultipleOfDimension {
                len: data.len(),
                dim,
            });
        }
        Self::from_rows(data.chunks_exact(dim), dim)
    }

    /// Pack rows of length `dim`.
    ///
    /// # Errors
    ///
    /// [`PackError::ZeroDimension`] if `dim == 0`; [`PackError::RaggedRow`]
    /// for the first row whose length is not `dim`.
    pub fn from_rows<'a, I>(rows: I, dim: usize) -> Result<Self, PackError>
    where
        I: IntoIterator<Item = &'a [f32]>,
    {
        if dim == 0 {
            return Err(PackError::ZeroDimension);
        }
        let rows: Vec<&[f32]> = rows.into_iter().collect();
        if let Some((row, r)) = rows.iter().enumerate().find(|(_, r)| r.len() != dim) {
            return Err(PackError::RaggedRow {
                row,
                expected: dim,
                got: r.len(),
            });
        }
        let stride = stride_for(dim);
        let (mut buf, start) = aligned_zeroed(rows.len() * stride);
        for (i, r) in rows.iter().enumerate() {
            let at = start + i * stride;
            buf[at..at + dim].copy_from_slice(r);
        }
        Ok(Self {
            buf,
            start,
            stride,
            dim,
            len: rows.len(),
        })
    }

    /// Number of rows.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the corpus has no rows.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Length of every row.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Row `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.len()`.
    #[inline]
    #[must_use]
    pub fn row(&self, i: usize) -> &[f32] {
        assert!(
            i < self.len,
            "innr::Corpus: row {i} out of range ({})",
            self.len
        );
        let at = self.start + i * self.stride;
        &self.buf[at..at + self.dim]
    }

    /// The rows in order.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = &[f32]> + '_ {
        (0..self.len).map(|i| self.row(i))
    }

    /// Score `query` against every row, in row order.
    ///
    /// # Panics
    ///
    /// Panics if `query.len() != self.dim()`.
    #[must_use]
    pub fn scores<M>(&self, query: &[f32]) -> Vec<f32>
    where
        M: Metric<f32, Output = f32>,
    {
        let mut out = Vec::with_capacity(self.len);
        self.scores_into::<M>(query, &mut out);
        out
    }

    /// [`Corpus::scores`] into a caller-owned buffer, which is cleared first, so
    /// repeated scans reuse its allocation.
    ///
    /// # Panics
    ///
    /// Panics if `query.len() != self.dim()`.
    pub fn scores_into<M>(&self, query: &[f32], out: &mut Vec<f32>)
    where
        M: Metric<f32, Output = f32>,
    {
        out.clear();
        out.reserve(self.len);
        let q = self.aligned_query(query);
        self.for_each_score::<M>(q.as_slice(), |_, s| out.push(s));
    }

    /// The `k` best rows for `query`, best first.
    ///
    /// "Best" follows the metric's [`Direction`]: largest score for a
    /// similarity, smallest for a distance. Returns fewer than `k` hits only
    /// when the corpus has fewer than `k` rows, and nothing when `k == 0`.
    /// NaN scores rank last.
    ///
    /// # Panics
    ///
    /// Panics if `query.len() != self.dim()`.
    #[must_use]
    pub fn top_k<M>(&self, query: &[f32], k: usize) -> Vec<Hit>
    where
        M: Metric<f32, Output = f32>,
    {
        let q = self.aligned_query(query);
        self.top_k_aligned::<M>(q.as_slice(), k)
    }

    /// [`Corpus::top_k`] for each query, in order.
    ///
    /// # Panics
    ///
    /// Panics if any query's length differs from `self.dim()`.
    #[must_use]
    pub fn top_k_many<'a, M, I>(&self, queries: I, k: usize) -> Vec<Vec<Hit>>
    where
        M: Metric<f32, Output = f32>,
        I: IntoIterator<Item = &'a [f32]>,
    {
        let mut q = AlignedQuery::new(self.dim);
        queries
            .into_iter()
            .map(|query| {
                q.load(query);
                self.top_k_aligned::<M>(q.as_slice(), k)
            })
            .collect()
    }

    fn top_k_aligned<M>(&self, q: &[f32], k: usize) -> Vec<Hit>
    where
        M: Metric<f32, Output = f32>,
    {
        if k == 0 || self.len == 0 {
            return Vec::new();
        }
        // TopK keeps the smallest keys; negate similarities so the largest
        // score has the smallest key. Row indices fit u32 for any corpus
        // whose rows fit in memory at 16+ dimensions; larger counts are
        // rejected below rather than truncated.
        let flip = matches!(M::DIRECTION, Direction::Similarity);
        let mut top = TopK::new(k.min(self.len));
        assert!(
            u32::try_from(self.len).is_ok(),
            "innr::Corpus::top_k: more than u32::MAX rows"
        );
        self.for_each_score::<M>(q, |i, s| {
            top.insert(i as u32, if flip { -s } else { s });
        });
        top.into_sorted()
            .into_iter()
            .map(|(i, s)| Hit {
                index: i as usize,
                score: if flip { -s } else { s },
            })
            .collect()
    }

    /// Copy `query` into 64-byte-aligned storage after checking its length.
    fn aligned_query(&self, query: &[f32]) -> AlignedQuery {
        let mut q = AlignedQuery::new(self.dim);
        q.load(query);
        q
    }

    /// Call `f(row, score)` for every row, with one SIMD dispatch for the
    /// whole corpus.
    #[inline]
    fn for_each_score<M>(&self, q: &[f32], mut f: impl FnMut(usize, f32))
    where
        M: Kernel<f32, Output = f32>,
    {
        debug_assert_eq!(q.len(), self.dim);
        #[cfg(feature = "simd")]
        if self.dim >= M::MIN_SIMD_LEN {
            crate::arch::simd_dispatch!(
                crate::arch::isa(),
                s => {
                    for i in 0..self.len {
                        let at = self.start + i * self.stride;
                        f(i, M::simd(s, q, &self.buf[at..at + self.dim]));
                    }
                },
                _ => self.for_each_portable::<M>(q, &mut f),
            );
            return;
        }
        self.for_each_portable::<M>(q, &mut f);
    }

    fn for_each_portable<M>(&self, q: &[f32], f: &mut impl FnMut(usize, f32))
    where
        M: Kernel<f32, Output = f32>,
    {
        for i in 0..self.len {
            let at = self.start + i * self.stride;
            f(i, M::portable(q, &self.buf[at..at + self.dim]));
        }
    }
}

/// A query copied into 64-byte-aligned storage.
struct AlignedQuery {
    buf: Vec<f32>,
    start: usize,
    dim: usize,
}

impl AlignedQuery {
    fn new(dim: usize) -> Self {
        let (buf, start) = aligned_zeroed(dim);
        Self { buf, start, dim }
    }

    fn load(&mut self, query: &[f32]) {
        assert_eq!(
            query.len(),
            self.dim,
            "innr::Corpus: query has {} elements, corpus dimension is {}",
            query.len(),
            self.dim
        );
        self.buf[self.start..self.start + self.dim].copy_from_slice(query);
    }

    fn as_slice(&self) -> &[f32] {
        &self.buf[self.start..self.start + self.dim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{Cosine, Dot, L2Sq, L1};

    fn rows(n: usize, dim: usize, seed: u32) -> Vec<Vec<f32>> {
        (0..n)
            .map(|r| {
                (0..dim)
                    .map(|i| {
                        let x = ((r * dim + i) as u32).wrapping_mul(2654435761) ^ seed;
                        (x % 2001) as f32 / 1000.0 - 1.0
                    })
                    .collect()
            })
            .collect()
    }

    fn pack(rows: &[Vec<f32>], dim: usize) -> Corpus {
        Corpus::from_rows(rows.iter().map(Vec::as_slice), dim).unwrap()
    }

    /// `scan` must equal the metric applied per row (same kernels, same
    /// thresholds), for every metric and the dimensions around each path.
    fn check_scores<M: Metric<f32, Output = f32>>() {
        for dim in [1, 3, 15, 16, 17, 31, 64, 100, 768] {
            for n in [0, 1, 5, 37] {
                let data = rows(n, dim, 7);
                let corpus = pack(&data, dim);
                let q = &rows(1, dim, 99)[0];
                let got = corpus.scores::<M>(q);
                let want: Vec<f32> = data.iter().map(|r| M::eval(q, r)).collect();
                assert_eq!(got.len(), n);
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    let tol = 1e-5 * w.abs().max(1.0);
                    assert!((g - w).abs() <= tol, "dim={dim} n={n} row={i}: {g} vs {w}");
                }
            }
        }
    }

    #[test]
    fn scores_match_per_row_metric() {
        check_scores::<Dot>();
        check_scores::<Cosine>();
        check_scores::<L2Sq>();
        check_scores::<L1>();
    }

    /// `top_k` must return the same rows as sorting the full scan by the
    /// metric's direction, best first.
    fn check_top_k<M: Metric<f32, Output = f32>>() {
        for dim in [2, 16, 100] {
            let data = rows(50, dim, 3);
            let corpus = pack(&data, dim);
            let q = &rows(1, dim, 41)[0];
            let scores = corpus.scores::<M>(q);
            let mut order: Vec<usize> = (0..scores.len()).collect();
            order.sort_by(|&a, &b| match M::DIRECTION {
                Direction::Similarity => scores[b].total_cmp(&scores[a]),
                Direction::Distance => scores[a].total_cmp(&scores[b]),
            });
            for k in [1, 5, 50, 80] {
                let got = corpus.top_k::<M>(q, k);
                assert_eq!(got.len(), k.min(50), "dim={dim} k={k}");
                for (rank, hit) in got.iter().enumerate() {
                    assert_eq!(
                        hit.score, scores[order[rank]],
                        "dim={dim} k={k} rank={rank}"
                    );
                    assert_eq!(hit.score, scores[hit.index], "score belongs to its row");
                }
            }
        }
    }

    #[test]
    fn top_k_matches_sorted_scores() {
        check_top_k::<Dot>();
        check_top_k::<Cosine>();
        check_top_k::<L2Sq>();
        check_top_k::<L1>();
    }

    #[test]
    fn top_k_edge_cases() {
        let data = rows(3, 8, 1);
        let corpus = pack(&data, 8);
        assert!(corpus.top_k::<Dot>(&data[0], 0).is_empty());
        let empty = Corpus::from_rows(std::iter::empty::<&[f32]>(), 8).unwrap();
        assert!(empty.is_empty());
        assert!(empty.top_k::<Dot>(&data[0], 3).is_empty());
        assert!(empty.scores::<Dot>(&data[0]).is_empty());
    }

    /// NaN rows rank after every finite row whatever the NaN's sign bit; x86
    /// arithmetic produces negative NaN, which `total_cmp` would sort first.
    fn check_nan_last<M: Metric<f32, Output = f32>>() {
        for dim in [4, 32] {
            let mut data = rows(10, dim, 11);
            data[2][1] = f32::NAN;
            data[7][0] = -f32::NAN;
            let corpus = pack(&data, dim);
            let q = &rows(1, dim, 5)[0];
            let all = corpus.top_k::<M>(q, 10);
            assert_eq!(all.len(), 10);
            assert!(all[..8].iter().all(|h| h.score.is_finite()), "dim={dim}");
            assert!(all[8..].iter().all(|h| h.score.is_nan()), "dim={dim}");
            let few = corpus.top_k::<M>(q, 3);
            assert!(few.iter().all(|h| h.score.is_finite()), "dim={dim}");
        }
    }

    #[test]
    fn top_k_ranks_nan_last() {
        check_nan_last::<Dot>();
        check_nan_last::<L2Sq>();
        check_nan_last::<L1>();
    }

    /// Cosine maps NaN input to 0.0 rather than NaN; the corpus scan must
    /// keep that per-row contract.
    #[test]
    fn cosine_scores_nan_rows_as_per_row_metric() {
        let mut data = rows(4, 32, 11);
        data[1][3] = f32::NAN;
        data[2][0] = -f32::NAN;
        let q = &rows(1, 32, 5)[0];
        let got = pack(&data, 32).scores::<Cosine>(q);
        assert_eq!(got[1], Cosine::eval(q, &data[1]));
        assert_eq!(got[2], Cosine::eval(q, &data[2]));
        assert_eq!(got[1], 0.0);
    }

    #[test]
    fn top_k_many_matches_top_k() {
        let data = rows(40, 32, 5);
        let corpus = pack(&data, 32);
        let queries = rows(4, 32, 77);
        let many = corpus.top_k_many::<L2Sq, _>(queries.iter().map(Vec::as_slice), 3);
        for (q, got) in queries.iter().zip(&many) {
            assert_eq!(got, &corpus.top_k::<L2Sq>(q, 3));
        }
    }

    #[test]
    fn rows_are_aligned_from_16_dimensions() {
        for dim in [16, 17, 100, 768] {
            let corpus = pack(&rows(5, dim, 2), dim);
            for r in corpus.rows() {
                assert_eq!(r.as_ptr() as usize % 64, 0, "dim={dim}");
                assert_eq!(r.len(), dim);
            }
        }
    }

    #[test]
    fn row_round_trips() {
        let data = rows(6, 20, 9);
        let corpus = pack(&data, 20);
        assert_eq!(corpus.len(), 6);
        assert_eq!(corpus.dim(), 20);
        for (i, r) in data.iter().enumerate() {
            assert_eq!(corpus.row(i), r.as_slice());
        }
        let flat: Vec<f32> = data.concat();
        let from_flat = Corpus::from_flat(&flat, 20).unwrap();
        assert!(from_flat.rows().eq(corpus.rows()));
    }

    #[test]
    fn pack_rejects_bad_input() {
        assert_eq!(
            Corpus::from_flat(&[1.0; 6], 0).unwrap_err(),
            PackError::ZeroDimension
        );
        assert_eq!(
            Corpus::from_flat(&[1.0; 7], 3).unwrap_err(),
            PackError::NotMultipleOfDimension { len: 7, dim: 3 }
        );
        let ragged: [&[f32]; 2] = [&[1.0, 2.0], &[1.0]];
        assert_eq!(
            Corpus::from_rows(ragged, 2).unwrap_err(),
            PackError::RaggedRow {
                row: 1,
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    #[should_panic(expected = "query has 3 elements")]
    fn scores_rejects_wrong_query_length() {
        let corpus = pack(&rows(2, 4, 1), 4);
        let _ = corpus.scores::<Dot>(&[1.0, 2.0, 3.0]);
    }
}
