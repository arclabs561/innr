//! Integer-slot Hamming distance and MinHash Jaccard estimation.
//!
//! Where [`crate::hamming_distance`] counts differing *bits* in byte-packed
//! binary vectors, these functions count differing *slots* in vectors of
//! fixed-width integers (`u16`, `u32`, `u64`, ...). The slot is the unit of
//! comparison: two slots either match or they don't.
//!
//! # MinHash
//!
//! The primary use case is MinHash sketches. Given two sketches `a` and `b`
//! of equal length, the fraction of matching slots is an unbiased estimator
//! of the Jaccard similarity of the original sets:
//!
//! ```text
//! J_hat(A, B) = matches / len = 1 - slot_hamming(a, b) / len
//! ```
//!
//! [`minhash_jaccard`] returns this estimate directly. The same primitive is
//! sometimes called "integer Hamming" or, confusingly, "Jaccard" in other
//! libraries.
//!
//! # Similarity vs distance
//!
//! Be deliberate about direction. [`minhash_jaccard`] returns a *similarity*
//! (fraction of matching slots, larger is closer); [`jaccard_distance`] returns
//! a *distance* (fraction of differing slots, smaller is closer). Index
//! libraries in the `anndists` / `hnsw_rs` ecosystem index on the distance form,
//! so use [`jaccard_distance`] when interoperating with them. [`slot_hamming`]
//! and [`slot_hamming_u32`] return the raw differing-slot *count*, not a
//! normalized fraction.
//!
//! # Dispatch
//!
//! [`slot_hamming_u32`] uses the same runtime-dispatch pattern as the rest of
//! the crate (AVX-512 > AVX2 > NEON > portable). For other widths, the generic
//! [`slot_hamming`] relies on LLVM auto-vectorization of the integer-compare
//! loop, which is effective for `u16`/`u64`.

// arch is only used on architectures with SIMD dispatch
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::arch;

/// Minimum slot count for the AVX2 / NEON paths to be worthwhile.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) const MIN_SLOTS_SIMD: usize = 8;

/// Minimum slot count for the AVX-512 path (16 lanes).
#[cfg(target_arch = "x86_64")]
pub(crate) const MIN_SLOTS_AVX512: usize = 16;

/// Integer-slot Hamming distance over `u32` slots: the number of positions
/// where `a[i] != b[i]`.
///
/// This is the SIMD-accelerated path for the most common MinHash sketch width.
/// For other integer widths use the generic [`slot_hamming`].
///
/// Returns 0 for empty slices.
///
/// # SIMD Acceleration
///
/// Automatically dispatches to (in order of preference):
/// - AVX-512F on x86_64 (runtime detection, n >= 16)
/// - AVX2 on x86_64 (runtime detection, n >= 8)
/// - NEON on aarch64 (always available, n >= 8)
/// - Portable fallback otherwise
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
///
/// # Example
///
/// ```rust
/// use innr::slot_hamming_u32;
///
/// let a = [1u32, 2, 3, 4];
/// let b = [1u32, 0, 3, 9];
/// // positions 1 and 3 differ
/// assert_eq!(slot_hamming_u32(&a, &b), 2);
/// ```
#[inline]
#[must_use]
#[allow(unsafe_code)]
pub fn slot_hamming_u32(a: &[u32], b: &[u32]) -> u32 {
    assert_eq!(
        a.len(),
        b.len(),
        "innr::slot_hamming_u32: slice length mismatch ({} vs {})",
        a.len(),
        b.len()
    );

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let n = a.len();

    #[cfg(target_arch = "x86_64")]
    {
        if n >= MIN_SLOTS_AVX512 && is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F verified via runtime detection.
            return unsafe { arch::x86_64::slot_hamming_u32_avx512(a, b) };
        }

        if n >= MIN_SLOTS_SIMD && is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 verified via runtime detection.
            return unsafe { arch::x86_64::slot_hamming_u32_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if n >= MIN_SLOTS_SIMD {
            // SAFETY: NEON is always available on aarch64.
            return unsafe { arch::aarch64::slot_hamming_u32_neon(a, b) };
        }
    }

    #[allow(unreachable_code)]
    slot_hamming_u32_portable(a, b)
}

/// Portable (non-SIMD) `u32`-slot Hamming distance.
#[inline]
#[must_use]
pub fn slot_hamming_u32_portable(a: &[u32], b: &[u32]) -> u32 {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count() as u32
}

/// Generic integer-slot Hamming distance: the number of positions where
/// `a[i] != b[i]`, for slots of any [`PartialEq`] type.
///
/// Use this for `u16`, `u64`, `u128`, or any custom slot type. For `u32`
/// prefer [`slot_hamming_u32`], which is SIMD-accelerated. The generic path
/// is a tight compare-and-count loop that LLVM auto-vectorizes well for the
/// fixed-width integer types.
///
/// Compares over `a.len().min(b.len())` positions; trailing elements of the
/// longer slice are ignored (unlike [`slot_hamming_u32`], which panics on a
/// length mismatch). Returns 0 for empty slices.
///
/// # Example
///
/// ```rust
/// use innr::slot_hamming;
///
/// let a = [10u16, 20, 30];
/// let b = [10u16, 99, 30];
/// assert_eq!(slot_hamming(&a, &b), 1);
/// ```
#[inline]
#[must_use]
pub fn slot_hamming<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

/// MinHash Jaccard-similarity estimate from two `u32` sketches: the fraction
/// of matching slots, `1 - slot_hamming_u32(a, b) / len`.
///
/// This is the standard MinHash collision-probability estimator: with `k`
/// independent hash slots, the expected fraction of matches equals the
/// Jaccard similarity of the original sets.
///
/// Returns `1.0` for two empty sketches (vacuously identical).
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
///
/// # Example
///
/// ```rust
/// use innr::minhash_jaccard;
///
/// let a = [1u32, 2, 3, 4];
/// let b = [1u32, 2, 3, 9];
/// // 3 of 4 slots match -> 0.75
/// assert_eq!(minhash_jaccard(&a, &b), 0.75);
/// ```
#[inline]
#[must_use]
pub fn minhash_jaccard(a: &[u32], b: &[u32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "innr::minhash_jaccard: slice length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    let len = a.len();
    if len == 0 {
        return 1.0;
    }
    let diff = slot_hamming_u32(a, b);
    let matches = len as u32 - diff;
    matches as f32 / len as f32
}

/// MinHash Jaccard *distance* between two `u32` sketches: the fraction of
/// differing slots, `slot_hamming_u32(a, b) / len`, i.e. `1 - minhash_jaccard`.
///
/// This is the complement of [`minhash_jaccard`] and the form most index
/// libraries expect (smaller is closer). It matches the value `anndists`
/// returns from its integer `DistHamming` (normalized differing count), so use
/// this when porting from or interoperating with that ecosystem; use
/// [`minhash_jaccard`] when you want a similarity in `[0, 1]`.
///
/// Returns `0.0` for two empty sketches (vacuously identical, distance 0).
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
///
/// # Example
///
/// ```rust
/// use innr::jaccard_distance;
///
/// let a = [1u32, 2, 3, 4];
/// let b = [1u32, 2, 3, 9];
/// // 1 of 4 slots differs -> 0.25
/// assert_eq!(jaccard_distance(&a, &b), 0.25);
/// ```
#[inline]
#[must_use]
pub fn jaccard_distance(a: &[u32], b: &[u32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "innr::jaccard_distance: slice length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    let len = a.len();
    if len == 0 {
        return 0.0;
    }
    slot_hamming_u32(a, b) as f32 / len as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_hamming_u32_basic() {
        let a = [1u32, 2, 3, 4];
        let b = [1u32, 0, 3, 9];
        assert_eq!(slot_hamming_u32(&a, &b), 2);
    }

    #[test]
    fn test_slot_hamming_u32_empty() {
        assert_eq!(slot_hamming_u32(&[], &[]), 0);
    }

    #[test]
    fn test_slot_hamming_u32_identical() {
        let v = [7u32, 11, 13, 17, 19];
        assert_eq!(slot_hamming_u32(&v, &v), 0);
    }

    #[test]
    fn test_slot_hamming_u32_all_differ() {
        let a = [1u32; 32];
        let b = [2u32; 32];
        assert_eq!(slot_hamming_u32(&a, &b), 32);
    }

    #[test]
    fn test_slot_hamming_u32_symmetric() {
        let a = [1u32, 5, 9, 13, 2, 6, 10, 14, 3];
        let b = [1u32, 0, 9, 0, 2, 0, 10, 0, 3];
        assert_eq!(slot_hamming_u32(&a, &b), slot_hamming_u32(&b, &a));
    }

    #[test]
    fn test_slot_hamming_u32_boundary_sizes() {
        for size in [1usize, 7, 8, 15, 16, 17, 31, 32, 33, 64, 128] {
            let a: Vec<u32> = (0..size as u32).collect();
            // flip every third slot
            let b: Vec<u32> = (0..size as u32)
                .map(|i| if i % 3 == 0 { i + 1000 } else { i })
                .collect();
            let expected = a.iter().zip(&b).filter(|(x, y)| x != y).count() as u32;
            assert_eq!(
                slot_hamming_u32(&a, &b),
                expected,
                "slot_hamming_u32 mismatch at size={size}"
            );
        }
    }

    #[test]
    fn test_slot_hamming_u32_matches_portable() {
        let a: Vec<u32> = (0..100u32).map(|i| i.wrapping_mul(2654435761)).collect();
        let b: Vec<u32> = (0..100u32).map(|i| i.wrapping_mul(40503)).collect();
        assert_eq!(slot_hamming_u32(&a, &b), slot_hamming_u32_portable(&a, &b));
    }

    #[test]
    #[should_panic(expected = "innr::slot_hamming_u32: slice length mismatch")]
    fn test_slot_hamming_u32_length_mismatch() {
        let _ = slot_hamming_u32(&[1, 2], &[1, 2, 3]);
    }

    #[test]
    fn test_slot_hamming_generic_u16() {
        let a = [10u16, 20, 30];
        let b = [10u16, 99, 30];
        assert_eq!(slot_hamming(&a, &b), 1);
    }

    #[test]
    fn test_slot_hamming_generic_u64() {
        let a = [1u64, 2, 3, 4];
        let b = [1u64, 9, 3, 9];
        assert_eq!(slot_hamming(&a, &b), 2);
    }

    #[test]
    fn test_slot_hamming_generic_agrees_with_u32() {
        let a: Vec<u32> = (0..50u32).map(|i| i % 7).collect();
        let b: Vec<u32> = (0..50u32).map(|i| i % 5).collect();
        assert_eq!(slot_hamming(&a, &b) as u32, slot_hamming_u32(&a, &b));
    }

    #[test]
    fn test_minhash_jaccard_basic() {
        let a = [1u32, 2, 3, 4];
        let b = [1u32, 2, 3, 9];
        assert_eq!(minhash_jaccard(&a, &b), 0.75);
    }

    #[test]
    fn test_minhash_jaccard_identical() {
        let v = [5u32, 6, 7, 8];
        assert_eq!(minhash_jaccard(&v, &v), 1.0);
    }

    #[test]
    fn test_minhash_jaccard_disjoint() {
        let a = [1u32, 2, 3, 4];
        let b = [5u32, 6, 7, 8];
        assert_eq!(minhash_jaccard(&a, &b), 0.0);
    }

    #[test]
    fn test_minhash_jaccard_empty() {
        assert_eq!(minhash_jaccard(&[], &[]), 1.0);
    }

    #[test]
    fn test_jaccard_distance_basic() {
        let a = [1u32, 2, 3, 4];
        let b = [1u32, 2, 3, 9];
        assert_eq!(jaccard_distance(&a, &b), 0.25);
    }

    #[test]
    fn test_jaccard_distance_complements_similarity() {
        let a = [1u32, 5, 9, 2, 6, 10, 3, 7];
        let b = [1u32, 0, 9, 0, 6, 0, 3, 0];
        assert_eq!(jaccard_distance(&a, &b), 1.0 - minhash_jaccard(&a, &b));
    }

    #[test]
    fn test_jaccard_distance_identical() {
        let v = [5u32, 6, 7, 8];
        assert_eq!(jaccard_distance(&v, &v), 0.0);
    }

    #[test]
    fn test_jaccard_distance_empty() {
        assert_eq!(jaccard_distance(&[], &[]), 0.0);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    const SIZES: &[usize] = &[0, 1, 7, 8, 15, 16, 31, 32, 64, 96, 128];

    fn arb_u32_pair(len: usize) -> impl Strategy<Value = (Vec<u32>, Vec<u32>)> {
        // small value domain so collisions (matches) actually happen
        (
            prop::collection::vec(0u32..8, len),
            prop::collection::vec(0u32..8, len),
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        #[test]
        fn slot_hamming_u32_matches_portable(
            (a, b) in prop::sample::select(SIZES).prop_flat_map(arb_u32_pair)
        ) {
            prop_assert_eq!(slot_hamming_u32(&a, &b), slot_hamming_u32_portable(&a, &b));
        }

        #[test]
        fn slot_hamming_u32_symmetric(
            (a, b) in prop::sample::select(SIZES).prop_flat_map(arb_u32_pair)
        ) {
            prop_assert_eq!(slot_hamming_u32(&a, &b), slot_hamming_u32(&b, &a));
        }

        #[test]
        fn slot_hamming_u32_self_is_zero(
            a in prop::sample::select(SIZES)
                .prop_flat_map(|s| prop::collection::vec(0u32..8, s))
        ) {
            prop_assert_eq!(slot_hamming_u32(&a, &a), 0);
        }

        // differing slots never exceed length
        #[test]
        fn slot_hamming_u32_bounded(
            (a, b) in prop::sample::select(SIZES).prop_flat_map(arb_u32_pair)
        ) {
            prop_assert!(slot_hamming_u32(&a, &b) <= a.len() as u32);
        }

        // generic path agrees with the u32-specialized path
        #[test]
        fn generic_agrees_with_u32(
            (a, b) in prop::sample::select(SIZES).prop_flat_map(arb_u32_pair)
        ) {
            prop_assert_eq!(slot_hamming(&a, &b) as u32, slot_hamming_u32(&a, &b));
        }

        // jaccard estimate stays in [0, 1]
        #[test]
        fn minhash_jaccard_in_unit_interval(
            (a, b) in prop::sample::select(SIZES).prop_flat_map(arb_u32_pair)
        ) {
            let j = minhash_jaccard(&a, &b);
            prop_assert!((0.0..=1.0).contains(&j), "jaccard {} out of range", j);
        }
    }
}
