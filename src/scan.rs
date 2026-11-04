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

use crate::metric::Element;
use crate::packed::{PackError, PackedRows};

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

#[cfg(test)]
mod tests {
    use super::*;

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
        let f = Flat::new(&[1.0_f32, 2.0, 3.0, 4.0], 2).unwrap();
        assert_eq!(
            (Rows::len(&f), Rows::dim(&f), f.row(1)),
            (2, 2, &[3.0, 4.0][..])
        );
        let rows: Vec<(usize, &[f32])> = f.iter_rows().collect();
        assert_eq!(rows, [(0, &[1.0, 2.0][..]), (1, &[3.0, 4.0][..])]);
    }

    #[test]
    #[should_panic(expected = "row 1 has 2 elements, expected 3")]
    fn ragged_rows_panic() {
        let rows = vec![vec![1.0_f32; 3], vec![1.0; 2]];
        let _ = rows.as_slice().iter_rows().count();
    }
}
