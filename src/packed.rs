//! Packed rows: pack the rows once, then score many queries against them.
//!
//! [`PackedRows`] stores rows contiguously (`f32` by default, `f64`, or `u8`
//! binary codes), each float row starting on a 64-byte boundary from 16
//! dimensions up and each byte row on a 16-byte boundary from 16 bytes, and
//! scores a query against every row inside one SIMD dispatch. Compared with
//! calling a pair kernel per row:
//!
//! - the backend is chosen once per call, not once per row (on x86 the
//!   dispatch is a real call, which is most of the cost at small dimensions);
//! - rows and the query are aligned, so wide loads never split across cache
//!   lines (on AVX2 a buffer 16 bytes off a 32-byte boundary was 1.5x slower);
//! - [`PackedRows::top_k`] keeps the best `k` as it scans, without building a
//!   score vector.
//!
//! The metric is a type parameter ([`crate::metric`]); its
//! [`Direction`] decides whether larger or smaller scores rank first.
//!
//! ```
//! use innr::packed::PackedRows;
//! use innr::metric::{Cosine, L2Sq};
//!
//! let rows = [[1.0_f32, 0.0], [0.0, 1.0], [0.7, 0.7]];
//! let corpus = PackedRows::from_rows(rows.iter().map(|r| r.as_slice()), 2).unwrap();
//!
//! let best = corpus.top_k::<Cosine>(&[0.1, 1.0], 1);
//! assert_eq!(best[0].index, 1);
//!
//! let nearest = corpus.top_k::<L2Sq>(&[0.6, 0.6], 2);
//! assert_eq!(nearest[0].index, 2);
//! assert!(nearest[0].score < nearest[1].score);
//! ```
//!
//! Two-stage search: score cheap sign-bit codes of every row, then rerank the
//! best candidates exactly.
//!
//! ```
//! use innr::metric::Cosine;
//! use innr::packed::PackedRows;
//!
//! let rows = [[0.9_f32, 0.1, -0.2, 0.4], [-0.5, 0.8, 0.1, -0.3], [0.7, 0.2, -0.1, 0.6]];
//! let sign_code = |r: &[f32]| {
//!     let mut c = 0u8;
//!     for (i, &x) in r.iter().enumerate() {
//!         if x > 0.0 { c |= 1 << i; }
//!     }
//!     [c]
//! };
//! let codes: Vec<[u8; 1]> = rows.iter().map(|r| sign_code(r)).collect();
//! let codes = PackedRows::from_rows(codes.iter().map(|c| c.as_slice()), 1).unwrap();
//! let exact = PackedRows::from_rows(rows.iter().map(|r| r.as_slice()), 4).unwrap();
//!
//! // Rows 0 and 2 share a sign code, so the coarse stage keeps both; the
//! // exact stage picks row 2, which equals the query.
//! let q = [0.7_f32, 0.2, -0.1, 0.6];
//! let candidates = codes.sign_dot_top_k(&q, 2);
//! let best = exact.rerank::<Cosine, _>(&q, candidates.iter().map(|h| h.index), 1);
//! assert_eq!(best[0].index, 2);
//! ```

use crate::metric::{sealed::Kernel, Direction, Element, Metric};
use crate::topk::TopK;

/// One result of [`PackedRows::top_k`]: a row and its score under the metric.
///
/// `S` is the metric's output: `f32` for `f32` rows, `u32` for Hamming and
/// `u64` for the dot product over `u8` rows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit<S = f32> {
    /// Index of the row in the corpus.
    pub index: usize,
    /// The metric's value for the query and this row.
    pub score: S,
}

/// Row and buffer alignment in bytes. 64 (a cache line; an AVX-512 vector)
/// for float rows. 16 for byte rows: binary codes are often 16 to 128 bytes,
/// below the byte kernels' SIMD thresholds, where 64-byte padding would
/// multiply memory for no gain.
const fn align_bytes<T>() -> usize {
    if core::mem::size_of::<T>() == 1 {
        16
    } else {
        64
    }
}

/// Why [`PackedRows::from_flat`] or [`PackedRows::from_rows`] rejected its input.
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
            Self::ZeroDimension => write!(f, "innr::PackedRows: dimension must be at least 1"),
            Self::NotMultipleOfDimension { len, dim } => write!(
                f,
                "innr::PackedRows: flat length {len} is not a multiple of dimension {dim}"
            ),
            Self::RaggedRow { row, expected, got } => write!(
                f,
                "innr::PackedRows: row {row} has {got} elements, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for PackError {}

/// Rows packed for repeated scoring: `f32` by default, or `f64` or `u8`
/// (byte-packed binary codes, scored with `Hamming`). See the
/// [module docs](self).
#[derive(Debug, Clone)]
pub struct PackedRows<T = f32> {
    /// Backing storage; rows start at `start` and are `stride` apart.
    buf: Vec<T>,
    start: usize,
    stride: usize,
    dim: usize,
    len: usize,
    /// Per-row values for metrics that use them (cosine: squared norms),
    /// filled on the first such scan.
    norms: std::sync::OnceLock<Vec<T>>,
}

/// Elements between consecutive rows: the row rounded up to the alignment
/// once it is at least that long (16 `f32`, 16 bytes). Shorter rows stay
/// unpadded: the SIMD kernels do not run on them and padding would multiply
/// the memory.
fn stride_for<T>(dim: usize) -> usize {
    let align = align_bytes::<T>() / core::mem::size_of::<T>();
    if dim >= align {
        dim.div_ceil(align) * align
    } else {
        dim
    }
}

/// A zeroed buffer of `n` elements whose element `start` is aligned to
/// [`align_bytes`].
fn aligned_zeroed<T: Copy + Default>(n: usize) -> (Vec<T>, usize) {
    let (bytes, size) = (align_bytes::<T>(), core::mem::size_of::<T>());
    let buf = vec![T::default(); n + bytes / size];
    let misalign = buf.as_ptr() as usize % bytes;
    let start = (bytes - misalign) % bytes / size;
    (buf, start)
}

impl<T: Element + Default> PackedRows<T> {
    /// Pack `len / dim` rows from a row-major buffer.
    ///
    /// # Errors
    ///
    /// [`PackError::ZeroDimension`] if `dim == 0`;
    /// [`PackError::NotMultipleOfDimension`] if `data.len() % dim != 0`.
    pub fn from_flat(data: &[T], dim: usize) -> Result<Self, PackError> {
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
        I: IntoIterator<Item = &'a [T]>,
    {
        if dim == 0 {
            return Err(PackError::ZeroDimension);
        }
        let rows: Vec<&[T]> = rows.into_iter().collect();
        if let Some((row, r)) = rows.iter().enumerate().find(|(_, r)| r.len() != dim) {
            return Err(PackError::RaggedRow {
                row,
                expected: dim,
                got: r.len(),
            });
        }
        let stride = stride_for::<T>(dim);
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
            norms: std::sync::OnceLock::new(),
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
    pub fn row(&self, i: usize) -> &[T] {
        assert!(
            i < self.len,
            "innr::PackedRows: row {i} out of range ({})",
            self.len
        );
        let at = self.start + i * self.stride;
        &self.buf[at..at + self.dim]
    }

    /// The rows in order, walked by stride (no per-row range check).
    pub(crate) fn strided_rows(&self) -> impl Iterator<Item = &[T]> + '_ {
        let dim = self.dim;
        self.buf[self.start..]
            .chunks(self.stride)
            .take(self.len)
            .map(move |c| &c[..dim])
    }

    /// Cosine row norms, if a cosine scan has computed them.
    pub(crate) fn stored_norms(&self) -> Option<&[T]> {
        self.norms.get().map(Vec::as_slice)
    }

    /// The rows in order.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = &[T]> + '_ {
        (0..self.len).map(|i| self.row(i))
    }

    /// Score `query` against every row, in row order.
    ///
    /// # Panics
    ///
    /// Panics if `query.len() != self.dim()`.
    #[must_use]
    pub fn scores<M>(&self, query: &[T]) -> Vec<M::Output>
    where
        M: Metric<T>,
    {
        let mut out = Vec::with_capacity(self.len);
        self.scores_into::<M>(query, &mut out);
        out
    }

    /// [`PackedRows::scores`] into a caller-owned buffer, which is cleared first, so
    /// repeated scans reuse its allocation.
    ///
    /// # Panics
    ///
    /// Panics if `query.len() != self.dim()`.
    pub fn scores_into<M>(&self, query: &[T], out: &mut Vec<M::Output>)
    where
        M: Metric<T>,
    {
        out.clear();
        out.reserve(self.len);
        let q = self.aligned_query(query);
        self.for_each_score::<M>(q.as_slice(), |_, s| out.push(s));
    }

    /// Copy `query` into aligned storage after checking its length.
    fn aligned_query(&self, query: &[T]) -> AlignedQuery<T> {
        let mut q = AlignedQuery::new(self.dim);
        q.load(query);
        q
    }

    /// Call `f(row, score)` for every row, with one SIMD dispatch for the
    /// whole corpus.
    #[inline]
    fn for_each_score<M>(&self, q: &[T], mut f: impl FnMut(usize, M::Output))
    where
        M: Kernel<T>,
    {
        debug_assert_eq!(q.len(), self.dim);
        if M::NORMED {
            return self.for_each_normed::<M>(q, f);
        }
        #[cfg(feature = "simd")]
        if self.dim >= M::MIN_SCAN_LEN {
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

    /// `f(row, score)` for the listed rows only, in list order, with one
    /// dispatch. Uses stored norms for `NORMED` metrics only if already
    /// computed. Indices are checked by the caller.
    fn for_each_score_at<M>(&self, q: &[T], ids: &[usize], mut f: impl FnMut(usize, M::Output))
    where
        M: Kernel<T>,
    {
        let row = |i: usize| &self.buf[self.start + i * self.stride..][..self.dim];
        let norms = if M::NORMED { self.norms.get() } else { None };
        let nq = norms.map(|_| M::row_norm(q).expect("innr: NORMED metric without row_norm"));
        #[cfg(feature = "simd")]
        if self.dim >= M::MIN_SCAN_LEN {
            crate::arch::simd_dispatch!(
                crate::arch::isa(),
                s => {
                    for &i in ids {
                        let score = match (norms, nq) {
                            (Some(n), Some(nq)) => M::simd_normed(s, q, row(i), nq, n[i]),
                            _ => M::simd(s, q, row(i)),
                        };
                        f(i, score);
                    }
                },
                _ => {
                    for &i in ids {
                        let score = match (norms, nq) {
                            (Some(n), Some(nq)) => M::portable_normed(q, row(i), nq, n[i]),
                            _ => M::portable(q, row(i)),
                        };
                        f(i, score);
                    }
                },
            );
            return;
        }
        for &i in ids {
            let score = match (norms, nq) {
                (Some(n), Some(nq)) => M::portable_normed(q, row(i), nq, n[i]),
                _ => M::portable(q, row(i)),
            };
            f(i, score);
        }
    }

    /// [`PackedRows::for_each_score`] for metrics that take per-row norms: the
    /// rows' are computed once per corpus, the query's once per scan.
    fn for_each_normed<M>(&self, q: &[T], mut f: impl FnMut(usize, M::Output))
    where
        M: Kernel<T>,
    {
        let norm = |r: &[T]| M::row_norm(r).expect("innr: NORMED metric without row_norm");
        let norms = self.norms.get_or_init(|| self.rows().map(norm).collect());
        let nq = norm(q);
        #[cfg(feature = "simd")]
        if self.dim >= M::MIN_SCAN_LEN {
            crate::arch::simd_dispatch!(
                crate::arch::isa(),
                s => {
                    for (i, &nr) in norms.iter().enumerate() {
                        let at = self.start + i * self.stride;
                        f(i, M::simd_normed(s, q, &self.buf[at..at + self.dim], nq, nr));
                    }
                },
                _ => {
                    for (i, &nr) in norms.iter().enumerate() {
                        let at = self.start + i * self.stride;
                        f(i, M::portable_normed(q, &self.buf[at..at + self.dim], nq, nr));
                    }
                },
            );
            return;
        }
        for (i, &nr) in norms.iter().enumerate() {
            let at = self.start + i * self.stride;
            f(
                i,
                M::portable_normed(q, &self.buf[at..at + self.dim], nq, nr),
            );
        }
    }

    fn for_each_portable<M>(&self, q: &[T], f: &mut impl FnMut(usize, M::Output))
    where
        M: Kernel<T>,
    {
        for i in 0..self.len {
            let at = self.start + i * self.stride;
            f(i, M::portable(q, &self.buf[at..at + self.dim]));
        }
    }
}

impl PackedRows<f32> {
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

    /// [`PackedRows::top_k`] for each query, in order.
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

    /// The `k` best of `candidates` (row indices) for `query`, best first,
    /// with the rows' own indices in the hits.
    ///
    /// The second stage of a two-stage search: take candidates from a cheap
    /// scan (for example [`PackedRows::sign_dot_top_k`] over binary codes of
    /// the same rows), then score only those exactly here, in one dispatch.
    /// An index listed twice is scored twice. For cosine, stored row norms
    /// are used if a full scan already computed them; a rerank alone does not
    /// compute norms for every row.
    ///
    /// # Panics
    ///
    /// Panics if `query.len() != self.dim()` or a candidate is out of range.
    #[must_use]
    pub fn rerank<M, I>(&self, query: &[f32], candidates: I, k: usize) -> Vec<Hit>
    where
        M: Metric<f32, Output = f32>,
        I: IntoIterator<Item = usize>,
    {
        let ids: Vec<usize> = candidates.into_iter().collect();
        if let Some(&bad) = ids.iter().find(|&&i| i >= self.len) {
            panic!(
                "innr::PackedRows::rerank: candidate {bad} out of range ({})",
                self.len
            );
        }
        let k = k.min(ids.len());
        if k == 0 {
            return Vec::new();
        }
        let q = self.aligned_query(query);
        let flip = matches!(M::DIRECTION, Direction::Similarity);
        let mut top = TopK::new(k);
        self.for_each_score_at::<M>(q.as_slice(), &ids, |i, s| {
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
            "innr::PackedRows::top_k: more than u32::MAX rows"
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
}

impl PackedRows<u8> {
    /// The `k` best rows for `query` under a byte metric (`Hamming`, or `Dot`
    /// on `u8`), best first.
    ///
    /// Scores are integers and rank exactly, ties by row order. Returns fewer
    /// than `k` hits only when the corpus has fewer than `k` rows, and
    /// nothing when `k == 0`.
    ///
    /// # Panics
    ///
    /// Panics if `query.len() != self.dim()`.
    #[must_use]
    pub fn top_k<M>(&self, query: &[u8], k: usize) -> Vec<Hit<M::Output>>
    where
        M: Metric<u8>,
        M::Output: Ord,
    {
        let q = self.aligned_query(query);
        self.top_k_aligned_int::<M>(q.as_slice(), k)
    }

    /// [`PackedRows::top_k`] for each query, in order.
    ///
    /// # Panics
    ///
    /// Panics if any query's length differs from `self.dim()`.
    #[must_use]
    pub fn top_k_many<'a, M, I>(&self, queries: I, k: usize) -> Vec<Vec<Hit<M::Output>>>
    where
        M: Metric<u8>,
        M::Output: Ord,
        I: IntoIterator<Item = &'a [u8]>,
    {
        let mut q = AlignedQuery::new(self.dim);
        queries
            .into_iter()
            .map(|query| {
                q.load(query);
                self.top_k_aligned_int::<M>(q.as_slice(), k)
            })
            .collect()
    }

    /// [`sign_dot`](crate::binary::sign_dot) of `query` against every code,
    /// in row order, into a caller-owned buffer (cleared first), with one
    /// SIMD dispatch for the whole scan.
    ///
    /// # Panics
    ///
    /// Panics if a code is shorter than `query.len().div_ceil(8)` bytes,
    /// that is, if `8 * self.dim() < query.len()`.
    pub fn sign_dot_into(&self, query: &[f32], out: &mut Vec<f32>) {
        out.clear();
        out.reserve(self.len);
        self.for_each_sign_dot(query, |_, s| out.push(s));
    }

    /// The `k` codes with the largest [`sign_dot`](crate::binary::sign_dot)
    /// against `query`, best first; NaN scores rank last.
    ///
    /// # Panics
    ///
    /// Panics if `8 * self.dim() < query.len()`.
    #[must_use]
    pub fn sign_dot_top_k(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let k = k.min(self.len);
        if k == 0 {
            return Vec::new();
        }
        assert!(
            u32::try_from(self.len).is_ok(),
            "innr::PackedRows::sign_dot_top_k: more than u32::MAX rows"
        );
        // TopK keeps the smallest keys: negate so the largest score ranks first.
        let mut top = TopK::new(k);
        self.for_each_sign_dot(query, |i, s| top.insert(i as u32, -s));
        top.into_sorted()
            .into_iter()
            .map(|(i, s)| Hit {
                index: i as usize,
                score: -s,
            })
            .collect()
    }

    fn for_each_sign_dot(&self, query: &[f32], mut f: impl FnMut(usize, f32)) {
        assert!(
            self.dim * 8 >= query.len(),
            "innr::PackedRows::sign_dot: codes have {} bytes, query needs {}",
            self.dim,
            query.len().div_ceil(8)
        );
        let row = |i: usize| &self.buf[self.start + i * self.stride..][..self.dim];
        #[cfg(feature = "simd")]
        if query.len() >= 8 {
            crate::arch::simd_dispatch!(
                crate::arch::isa(),
                s => {
                    for i in 0..self.len {
                        f(i, crate::arch::kernels::sign_dot_f32(s, query, row(i)));
                    }
                },
                _ => {
                    for i in 0..self.len {
                        f(i, crate::portable::sign_dot_f32(query, row(i)));
                    }
                },
            );
            return;
        }
        for i in 0..self.len {
            f(i, crate::portable::sign_dot_f32(query, row(i)));
        }
    }

    fn top_k_aligned_int<M>(&self, q: &[u8], k: usize) -> Vec<Hit<M::Output>>
    where
        M: Metric<u8>,
        M::Output: Ord,
    {
        let k = k.min(self.len);
        if k == 0 {
            return Vec::new();
        }
        // A max-heap of the k best seen so far, keyed so its top is the
        // worst kept hit: (score, row) for a distance, (Reverse(score), row)
        // for a similarity. The row breaks ties toward the earlier row.
        use core::cmp::Reverse;
        use std::collections::BinaryHeap;
        let mut hits: Vec<Hit<M::Output>> = match M::DIRECTION {
            Direction::Distance => {
                let mut heap = BinaryHeap::with_capacity(k + 1);
                self.for_each_score::<M>(q, |i, s| {
                    if heap.len() < k {
                        heap.push((s, i));
                    } else if let Some(&(worst, _)) = heap.peek() {
                        if s < worst {
                            heap.pop();
                            heap.push((s, i));
                        }
                    }
                });
                heap.into_iter()
                    .map(|(score, index)| Hit { index, score })
                    .collect()
            }
            Direction::Similarity => {
                let mut heap = BinaryHeap::with_capacity(k + 1);
                self.for_each_score::<M>(q, |i, s| {
                    if heap.len() < k {
                        heap.push((Reverse(s), i));
                    } else if let Some(&(Reverse(worst), _)) = heap.peek() {
                        if s > worst {
                            heap.pop();
                            heap.push((Reverse(s), i));
                        }
                    }
                });
                heap.into_iter()
                    .map(|(Reverse(score), index)| Hit { index, score })
                    .collect()
            }
        };
        match M::DIRECTION {
            Direction::Distance => {
                hits.sort_by(|a, b| a.score.cmp(&b.score).then(a.index.cmp(&b.index)))
            }
            Direction::Similarity => {
                hits.sort_by(|a, b| b.score.cmp(&a.score).then(a.index.cmp(&b.index)))
            }
        }
        hits
    }
}

/// A query copied into storage aligned like the corpus rows.
struct AlignedQuery<T> {
    buf: Vec<T>,
    start: usize,
    dim: usize,
}

impl<T: Copy + Default> AlignedQuery<T> {
    fn new(dim: usize) -> Self {
        let (buf, start) = aligned_zeroed(dim);
        Self { buf, start, dim }
    }

    fn load(&mut self, query: &[T]) {
        assert_eq!(
            query.len(),
            self.dim,
            "innr::PackedRows: query has {} elements, corpus dimension is {}",
            query.len(),
            self.dim
        );
        self.buf[self.start..self.start + self.dim].copy_from_slice(query);
    }

    fn as_slice(&self) -> &[T] {
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

    fn pack(rows: &[Vec<f32>], dim: usize) -> PackedRows {
        PackedRows::from_rows(rows.iter().map(Vec::as_slice), dim).unwrap()
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
                    // Packed rows are aligned and per-row inputs may not be,
                    // so x86 sums them in different orders; bound by Σ|q·r|
                    // (cosine's terms are at most 1 after normalization).
                    let mag: f32 = q.iter().zip(&data[i]).map(|(x, y)| (x * y).abs()).sum();
                    let tol = 1e-5 * w.abs().max(1.0) + (dim as f32 + 2.0) * f32::EPSILON * mag;
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
        let empty = PackedRows::from_rows(std::iter::empty::<&[f32]>(), 8).unwrap();
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
        let from_flat = PackedRows::from_flat(&flat, 20).unwrap();
        assert!(from_flat.rows().eq(corpus.rows()));
    }

    #[test]
    fn pack_rejects_bad_input() {
        assert_eq!(
            PackedRows::from_flat(&[1.0; 6], 0).unwrap_err(),
            PackError::ZeroDimension
        );
        assert_eq!(
            PackedRows::from_flat(&[1.0; 7], 3).unwrap_err(),
            PackError::NotMultipleOfDimension { len: 7, dim: 3 }
        );
        let ragged: [&[f32]; 2] = [&[1.0, 2.0], &[1.0]];
        assert_eq!(
            PackedRows::from_rows(ragged, 2).unwrap_err(),
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

    /// Cosine scans reuse the stored row norms; overflowing squared norms
    /// still reach the f64 fallback; zero rows still score 0.
    #[test]
    fn cosine_with_stored_norms_matches_per_row() {
        for dim in [8, 64, 300] {
            let mut data = rows(12, dim, 5);
            data[3].iter_mut().for_each(|x| *x *= 1e20); // squares overflow f32
            data[7].iter_mut().for_each(|x| *x = 0.0);
            let corpus = pack(&data, dim);
            let q = &rows(1, dim, 77)[0];
            let want: Vec<f32> = data.iter().map(|r| Cosine::eval(q, r)).collect();
            for _ in 0..2 {
                let got = corpus.scores::<Cosine>(q);
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!((g - w).abs() <= 1e-5, "dim={dim} row={i}: {g} vs {w}");
                }
            }
            assert!(want[3].is_finite() && want[3].abs() <= 1.0);
            assert_eq!(corpus.scores::<Cosine>(q)[7], 0.0);
            let _ = corpus.scores::<Dot>(q);
            assert_eq!(corpus.top_k::<Cosine>(q, 3), corpus.top_k::<Cosine>(q, 3));
        }
    }

    fn codes(n: usize, len: usize, seed: u32) -> Vec<Vec<u8>> {
        (0..n)
            .map(|r| {
                (0..len)
                    .map(|i| (((r * len + i) as u32).wrapping_mul(2654435761) ^ seed) as u8)
                    .collect()
            })
            .collect()
    }

    /// Byte rows score exactly like the per-row functions, across the
    /// padding (16 B) and SIMD thresholds (128 B, 2048 B on Apple).
    #[test]
    fn byte_scores_match_per_row_functions() {
        use crate::metric::Hamming;
        for len in [1, 3, 15, 16, 17, 96, 128, 300, 2100] {
            for n in [0, 1, 5, 37] {
                let data = codes(n, len, 7);
                let corpus = PackedRows::from_rows(data.iter().map(Vec::as_slice), len).unwrap();
                let q = &codes(1, len, 99)[0];
                let ham: Vec<u32> = data.iter().map(|r| crate::hamming_distance(q, r)).collect();
                assert_eq!(corpus.scores::<Hamming>(q), ham, "len={len} n={n}");
                let dot: Vec<u64> = data.iter().map(|r| crate::dot_u8_wide(q, r)).collect();
                assert_eq!(corpus.scores::<Dot>(q), dot, "len={len} n={n}");
            }
        }
    }

    /// Byte top_k equals sorting every score by (score, row): ascending for
    /// Hamming, descending for the dot product.
    #[test]
    fn byte_top_k_matches_sorted_scores() {
        use crate::metric::Hamming;
        for len in [2, 16, 96] {
            let data = codes(60, len, 3);
            let corpus = PackedRows::from_rows(data.iter().map(Vec::as_slice), len).unwrap();
            let q = &codes(1, len, 41)[0];
            let ham = corpus.scores::<Hamming>(q);
            let mut by_ham: Vec<usize> = (0..ham.len()).collect();
            by_ham.sort_by_key(|&i| (ham[i], i));
            let dot = corpus.scores::<Dot>(q);
            let mut by_dot: Vec<usize> = (0..dot.len()).collect();
            by_dot.sort_by_key(|&i| (core::cmp::Reverse(dot[i]), i));
            for k in [0, 1, 7, 60, 90] {
                let got: Vec<usize> = corpus
                    .top_k::<Hamming>(q, k)
                    .iter()
                    .map(|h| h.index)
                    .collect();
                assert_eq!(got, by_ham[..k.min(60)], "hamming len={len} k={k}");
                let got: Vec<usize> = corpus.top_k::<Dot>(q, k).iter().map(|h| h.index).collect();
                assert_eq!(got, by_dot[..k.min(60)], "dot len={len} k={k}");
            }
        }
    }

    /// Equal scores keep the earliest rows, in row order.
    #[test]
    fn byte_top_k_breaks_ties_by_row() {
        use crate::metric::Hamming;
        let corpus = PackedRows::from_rows([[0u8; 4]; 9].iter().map(|r| r.as_slice()), 4).unwrap();
        let hits = corpus.top_k::<Hamming>(&[0xFF; 4], 3);
        assert_eq!(
            hits.iter().map(|h| (h.index, h.score)).collect::<Vec<_>>(),
            [(0, 32), (1, 32), (2, 32)]
        );
        let many =
            corpus.top_k_many::<Hamming, _>([[0xFFu8; 4].as_slice(), [0u8; 4].as_slice()], 2);
        assert_eq!(many[0], hits[..2]);
        assert_eq!(many[1].iter().map(|h| h.score).collect::<Vec<_>>(), [0, 0]);
    }

    /// A sign-dot scan equals per-code `sign_dot`; its top-k equals sorting
    /// those scores descending.
    #[test]
    fn sign_dot_scan_matches_per_code_calls() {
        for bits in [5usize, 64, 100, 768] {
            let len = bits.div_ceil(8);
            let data = codes(50, len, 13);
            let packed = PackedRows::from_rows(data.iter().map(Vec::as_slice), len).unwrap();
            let q = &rows(1, bits, 21)[0];
            let want: Vec<f32> = data.iter().map(|c| crate::binary::sign_dot(q, c)).collect();
            let mut got = Vec::new();
            packed.sign_dot_into(q, &mut got);
            for (g, w) in got.iter().zip(&want) {
                assert!(
                    (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                    "bits={bits}: {g} vs {w}"
                );
            }
            // `codes` repeats rows at some lengths, so compare ranked scores
            // (the f32 top-k does not order ties) and that each hit's score
            // is its row's.
            let mut sorted = got.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let top = packed.sign_dot_top_k(q, 7);
            let scores: Vec<f32> = top.iter().map(|h| h.score).collect();
            assert_eq!(scores, sorted[..7], "bits={bits}");
            assert!(top.iter().all(|h| h.score == got[h.index]), "bits={bits}");
        }
    }

    /// Reranking every row is top_k; reranking a subset is top_k over a
    /// corpus of just those rows, with indices mapped back; cosine gives the
    /// same hits before and after a full scan stores the row norms.
    #[test]
    fn rerank_matches_top_k_over_the_candidates() {
        let dim = 48;
        let data = rows(40, dim, 17);
        let packed = pack(&data, dim);
        let q = &rows(1, dim, 3)[0];
        let all = packed.rerank::<L2Sq, _>(q, 0..40, 6);
        assert_eq!(all, packed.top_k::<L2Sq>(q, 6));

        let ids = [31usize, 2, 17, 8, 25, 39, 11];
        let sub: Vec<Vec<f32>> = ids.iter().map(|&i| data[i].clone()).collect();
        let want: Vec<(usize, f32)> = pack(&sub, dim)
            .top_k::<Dot>(q, 4)
            .iter()
            .map(|h| (ids[h.index], h.score))
            .collect();
        let got: Vec<(usize, f32)> = packed
            .rerank::<Dot, _>(q, ids, 4)
            .iter()
            .map(|h| (h.index, h.score))
            .collect();
        assert_eq!(got, want);

        let before = packed.rerank::<Cosine, _>(q, ids, 4);
        let _ = packed.scores::<Cosine>(q); // stores the norms
        let after = packed.rerank::<Cosine, _>(q, ids, 4);
        for (b, a) in before.iter().zip(&after) {
            assert_eq!(b.index, a.index);
            assert!((b.score - a.score).abs() <= 1e-5);
        }
        assert!(packed.rerank::<Dot, _>(q, [], 3).is_empty());
    }

    #[test]
    #[should_panic(expected = "candidate 40 out of range (40)")]
    fn rerank_rejects_out_of_range_candidates() {
        let packed = pack(&rows(40, 8, 1), 8);
        let _ = packed.rerank::<Dot, _>(&[0.0; 8], [3, 40], 2);
    }

    #[test]
    fn byte_rows_are_aligned_from_16_bytes() {
        for len in [16, 17, 96, 100] {
            let data = codes(5, len, 2);
            let corpus = PackedRows::from_rows(data.iter().map(Vec::as_slice), len).unwrap();
            for (r, want) in corpus.rows().zip(&data) {
                assert_eq!(r.as_ptr() as usize % 16, 0, "len={len}");
                assert_eq!(r, want.as_slice());
            }
        }
    }
}
