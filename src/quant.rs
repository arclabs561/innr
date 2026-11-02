//! Integer quantization primitives: u8 dot product and Hamming distance.
//!
//! These are the hot inner loops for quantized vector search:
//!
//! - [`dot_u8_wide`]: symmetric u8×u8 dot product for INT8-quantized embeddings
//! - [`dot_u8`]: the same operation with a checked `u32` result
//! - `hamming_distance`: byte-packed Hamming distance for binary-quantized embeddings
//!
//! # Dispatch
//!
//! Both functions use the same runtime-dispatch pattern as the rest of this crate:
//! with the `simd` feature, AVX-512 > AVX2 > NEON > WASM SIMD128 > portable.

use crate::metric::{eval_unchecked, Dot, Hamming};

/// Largest number of maximal u8 products whose sum fits in a `u32`.
#[cfg_attr(not(any(feature = "simd", test)), allow(dead_code))]
pub(crate) const MAX_U32_DOT_DIM: usize = u32::MAX as usize / (u8::MAX as usize * u8::MAX as usize);

/// Unsigned 8-bit integer dot product: `Σ(a[i] * b[i])`.
///
/// Returns 0 for empty slices. For vectors whose result may not fit in `u32`,
/// use [`dot_u8_wide`] or [`checked_dot_u8`].
///
/// # SIMD Acceleration
///
/// With the `simd` feature and `n >= 32`, dispatches at runtime to the best
/// available level (AVX-512, AVX2+FMA, NEON, WASM SIMD128); otherwise runs
/// the portable kernel.
///
/// # Panics
///
/// Panics if `a.len() != b.len()` or the result exceeds [`u32::MAX`].
///
/// # Example
///
/// ```rust
/// use innr::dot_u8;
///
/// let a = [1u8, 2, 3];
/// let b = [4u8, 5, 6];
/// // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
/// assert_eq!(dot_u8(&a, &b), 32);
/// ```
#[inline]
#[must_use]
pub fn dot_u8(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(
        a.len(),
        b.len(),
        "innr::dot_u8: slice length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    checked_dot_u8(a, b).expect("innr::dot_u8: result exceeds u32::MAX; use dot_u8_wide")
}

/// Unsigned 8-bit integer dot product with an exact `u64` result.
///
/// This is the preferred API when vector dimensions or input magnitudes are
/// not known to keep the result within `u32`. SIMD kernels are evaluated in
/// bounded chunks so their `u32` partial sums cannot overflow.
///
/// # Panics
///
/// Panics if `a.len() != b.len()` or the exact result exceeds `u64::MAX`.
#[inline]
#[must_use]
pub fn dot_u8_wide(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(
        a.len(),
        b.len(),
        "innr::dot_u8_wide: slice length mismatch ({} vs {})",
        a.len(),
        b.len()
    );

    eval_unchecked::<u8, Dot>(a, b)
}

/// Unsigned 8-bit integer dot product if its result fits in `u32`.
///
/// Returns `None` when the exact result exceeds [`u32::MAX`]. Length mismatch
/// remains a programming error and panics, as it does for [`dot_u8`].
#[inline]
#[must_use]
pub fn checked_dot_u8(a: &[u8], b: &[u8]) -> Option<u32> {
    u32::try_from(dot_u8_wide(a, b)).ok()
}

/// Portable (non-SIMD) u8 dot product with sixteen `u32` accumulator lanes
/// that LLVM vectorizes.
#[inline]
#[must_use]
pub fn dot_u8_portable(a: &[u8], b: &[u8]) -> u32 {
    u32::try_from(dot_u8_wide_portable(a, b))
        .expect("innr::dot_u8_portable: result exceeds u32::MAX; use dot_u8_wide_portable")
}

/// Portable u8 dot product with an exact `u64` result.
///
/// Unlike the dispatched public APIs, this low-level implementation processes
/// the common prefix when the slice lengths differ.
#[inline]
#[must_use]
pub fn dot_u8_wide_portable(a: &[u8], b: &[u8]) -> u64 {
    crate::portable::dot_u8_wide(a, b)
}

/// Hamming distance between two byte-packed bit vectors.
///
/// Each byte stores 8 bits, so `a` and `b` represent `a.len() * 8`-bit vectors.
/// Returns the count of bit positions where the two vectors differ.
///
/// Returns 0 for empty slices.
///
/// # SIMD Acceleration
///
/// With the `simd` feature and `n >= 32`, dispatches at runtime to the best
/// available level (AVX-512, AVX2+FMA, NEON, WASM SIMD128); otherwise runs
/// the portable kernel.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
///
/// # Example
///
/// ```rust
/// use innr::hamming_distance;
///
/// // 0b11110000 XOR 0b10101010 = 0b01011010 -> 4 bits set
/// let a = [0b1111_0000u8];
/// let b = [0b1010_1010u8];
/// assert_eq!(hamming_distance(&a, &b), 4);
/// ```
#[inline]
#[must_use]
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(
        a.len(),
        b.len(),
        "innr::hamming_distance: slice length mismatch ({} vs {})",
        a.len(),
        b.len()
    );

    eval_unchecked::<u8, Hamming>(a, b)
}

/// Portable (non-SIMD) Hamming distance: XOR bytes then count bits.
#[inline]
#[must_use]
pub fn hamming_portable(a: &[u8], b: &[u8]) -> u32 {
    crate::portable::hamming(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot_u8_oracle(a: &[u8], b: &[u8]) -> u64 {
        let exact: u128 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| u128::from(x) * u128::from(y))
            .sum();
        u64::try_from(exact).expect("test vector dot product fits in u64")
    }

    // =========================================================================
    // dot_u8 tests
    // =========================================================================

    #[test]
    fn test_dot_u8_basic() {
        let a = [1u8, 2, 3];
        let b = [4u8, 5, 6];
        // 4 + 10 + 18 = 32
        assert_eq!(dot_u8(&a, &b), 32);
    }

    #[test]
    fn test_dot_u8_empty() {
        assert_eq!(dot_u8(&[], &[]), 0);
    }

    #[test]
    fn test_dot_u8_single() {
        assert_eq!(dot_u8(&[255], &[255]), 255 * 255);
    }

    #[test]
    fn test_dot_u8_all_zeros() {
        let a = [0u8; 64];
        let b = [0u8; 64];
        assert_eq!(dot_u8(&a, &b), 0);
    }

    #[test]
    fn test_dot_u8_all_max() {
        // 255 * 255 = 65025; sum over 4 elements = 260100
        let a = [255u8; 4];
        let b = [255u8; 4];
        assert_eq!(dot_u8(&a, &b), 4 * 65025);
    }

    #[test]
    fn test_dot_u8_commutative() {
        let a = [10u8, 20, 30, 40, 50];
        let b = [1u8, 2, 3, 4, 5];
        assert_eq!(dot_u8(&a, &b), dot_u8(&b, &a));
    }

    #[test]
    fn test_dot_u8_large_simd() {
        // dim=128: exercises the SIMD kernels when `simd` is on
        let a: Vec<u8> = (0..128u8).collect();
        let b: Vec<u8> = (0..128u8).collect();
        let expected: u32 = (0..128u32).map(|i| i * i).sum();
        assert_eq!(dot_u8(&a, &b), expected);
    }

    #[test]
    fn test_dot_u8_boundary_sizes() {
        // Test around SIMD dispatch thresholds
        for size in [1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 128] {
            let a: Vec<u8> = (0..size).map(|i| (i % 16) as u8).collect();
            let b: Vec<u8> = (0..size).map(|i| ((i + 1) % 16) as u8).collect();
            let expected: u32 = a.iter().zip(&b).map(|(&x, &y)| x as u32 * y as u32).sum();
            assert_eq!(dot_u8(&a, &b), expected, "dot_u8 mismatch at size={size}");
        }
    }

    #[test]
    fn dot_u8_overflow_boundary_is_explicit() {
        for size in [MAX_U32_DOT_DIM, MAX_U32_DOT_DIM + 1] {
            let a = vec![u8::MAX; size];
            let b = vec![u8::MAX; size];
            let expected = dot_u8_oracle(&a, &b);

            assert_eq!(dot_u8_wide(&a, &b), expected);
            assert_eq!(dot_u8_wide_portable(&a, &b), expected);
            assert_eq!(checked_dot_u8(&a, &b), u32::try_from(expected).ok());
        }

        let a = vec![u8::MAX; MAX_U32_DOT_DIM];
        assert_eq!(
            dot_u8(&a, &a),
            u32::try_from(dot_u8_oracle(&a, &a)).unwrap()
        );

        let a = vec![u8::MAX; MAX_U32_DOT_DIM + 1];
        let panic = std::panic::catch_unwind(|| dot_u8(&a, &a));
        assert!(panic.is_err(), "legacy u32 API must not silently wrap");
    }

    #[test]
    fn dot_u8_wide_matches_independent_oracle_on_deterministic_inputs() {
        for size in [0, 1, 31, 32, 63, 64, 65, 4097, MAX_U32_DOT_DIM + 1] {
            let a: Vec<u8> = (0..size)
                .map(|i| ((i as u64 * 73 + 19) % 256) as u8)
                .collect();
            let b: Vec<u8> = (0..size)
                .map(|i| ((i as u64 * 151 + 41) % 256) as u8)
                .collect();
            assert_eq!(dot_u8_wide(&a, &b), dot_u8_oracle(&a, &b));
        }
    }

    /// The SIMD kernel, on the native level and on fearless_simd's scalar
    /// fallback level, at the u32 chunk boundary and on a mixed pattern.
    #[cfg(feature = "simd")]
    #[test]
    fn simd_dot_u8_kernel_matches_oracle_at_overflow_boundary() {
        use crate::metric::{sealed::Kernel, Dot};
        fn check<S: fearless_simd::Simd>(s: S) {
            let kernel = |x: &[u8], y: &[u8]| s.vectorize(|| <Dot as Kernel<u8>>::simd(s, x, y));
            for size in [MAX_U32_DOT_DIM, MAX_U32_DOT_DIM + 1] {
                let a = vec![u8::MAX; size];
                assert_eq!(kernel(&a, &a), dot_u8_oracle(&a, &a), "size {size}");
            }
            let size = 4097;
            let a: Vec<u8> = (0..size).map(|i| ((i * 73 + 19) % 256) as u8).collect();
            let b: Vec<u8> = (0..size).map(|i| ((i * 151 + 41) % 256) as u8).collect();
            assert_eq!(kernel(&a, &b), dot_u8_oracle(&a, &b), "random pattern");
        }
        fearless_simd::dispatch!(fearless_simd::Level::new(), s => check(s));
        check(fearless_simd::Fallback::new());
    }

    #[test]
    #[should_panic(expected = "innr::dot_u8: slice length mismatch")]
    fn test_dot_u8_length_mismatch() {
        let _ = dot_u8(&[1, 2], &[1, 2, 3]);
    }

    // =========================================================================
    // hamming_distance tests
    // =========================================================================

    #[test]
    fn test_hamming_basic() {
        // 0b11110000 XOR 0b10101010 = 0b01011010 -> 4 bits
        let a = [0b1111_0000u8];
        let b = [0b1010_1010u8];
        assert_eq!(hamming_distance(&a, &b), 4);
    }

    #[test]
    fn test_hamming_empty() {
        assert_eq!(hamming_distance(&[], &[]), 0);
    }

    #[test]
    fn test_hamming_identical() {
        let v = [0xDE, 0xAD, 0xBE, 0xEF];
        assert_eq!(hamming_distance(&v, &v), 0);
    }

    #[test]
    fn test_hamming_complement() {
        // All bits flipped -> every bit differs
        let a = [0xFF, 0xFF];
        let b = [0x00, 0x00];
        assert_eq!(hamming_distance(&a, &b), 16);
    }

    #[test]
    fn test_hamming_single_bit() {
        let a = [0b0000_0001u8];
        let b = [0b0000_0000u8];
        assert_eq!(hamming_distance(&a, &b), 1);
    }

    #[test]
    fn test_hamming_all_same() {
        let a = [0u8; 64];
        let b = [0u8; 64];
        assert_eq!(hamming_distance(&a, &b), 0);
    }

    #[test]
    fn test_hamming_all_differ() {
        let a = [0xFFu8; 16];
        let b = [0x00u8; 16];
        assert_eq!(hamming_distance(&a, &b), 128); // 16 bytes * 8 bits
    }

    #[test]
    fn test_hamming_symmetric() {
        let a = [0b1010_1010u8, 0b1100_1100, 0b1111_0000];
        let b = [0b0101_0101u8, 0b0011_0011, 0b0000_1111];
        assert_eq!(hamming_distance(&a, &b), hamming_distance(&b, &a));
    }

    #[test]
    fn test_hamming_boundary_sizes() {
        // Test around SIMD dispatch thresholds (in bytes)
        for size in [1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 128] {
            let a: Vec<u8> = (0..size).map(|i| (i * 3) as u8).collect();
            let b: Vec<u8> = (0..size).map(|i| (i * 5) as u8).collect();
            let expected: u32 = a.iter().zip(&b).map(|(&x, &y)| (x ^ y).count_ones()).sum();
            assert_eq!(
                hamming_distance(&a, &b),
                expected,
                "hamming_distance mismatch at size={size}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "innr::hamming_distance: slice length mismatch")]
    fn test_hamming_length_mismatch() {
        let _ = hamming_distance(&[1, 2], &[1, 2, 3]);
    }

    // =========================================================================
    // Portable implementations directly (ISA-independent correctness)
    // =========================================================================

    #[test]
    fn test_dot_u8_portable_matches_dispatch() {
        let a: Vec<u8> = (0..64).map(|i| (i * 7) as u8).collect();
        let b: Vec<u8> = (0..64).map(|i| (i * 3) as u8).collect();
        assert_eq!(dot_u8(&a, &b), dot_u8_portable(&a, &b));
    }

    #[test]
    fn test_hamming_portable_matches_dispatch() {
        let a: Vec<u8> = (0..64).map(|i| (i * 7) as u8).collect();
        let b: Vec<u8> = (0..64).map(|i| (i * 3) as u8).collect();
        assert_eq!(hamming_distance(&a, &b), hamming_portable(&a, &b));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Byte-vector lengths that exercise scalar, partial-SIMD, and full-SIMD paths.
    const SIZES: &[usize] = &[0, 1, 7, 16, 31, 32, 33, 64, 96, 128, 256];

    fn arb_u8_pair(len: usize) -> impl Strategy<Value = (Vec<u8>, Vec<u8>)> {
        (
            prop::collection::vec(any::<u8>(), len),
            prop::collection::vec(any::<u8>(), len),
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        // dot_u8 matches portable reference for all sizes
        #[test]
        fn dot_u8_matches_portable(
            size in prop::sample::select(SIZES),
            (a, b) in prop::sample::select(SIZES).prop_flat_map(arb_u8_pair)
        ) {
            let _ = size; // size is selected via arb_u8_pair above
            prop_assert_eq!(dot_u8(&a, &b), dot_u8_portable(&a, &b));
        }

        // dot_u8 is commutative
        #[test]
        fn dot_u8_commutative(
            (a, b) in prop::sample::select(SIZES).prop_flat_map(arb_u8_pair)
        ) {
            prop_assert_eq!(dot_u8(&a, &b), dot_u8(&b, &a));
        }

        // hamming_distance matches portable reference for all sizes
        #[test]
        fn hamming_matches_portable(
            (a, b) in prop::sample::select(SIZES).prop_flat_map(arb_u8_pair)
        ) {
            prop_assert_eq!(hamming_distance(&a, &b), hamming_portable(&a, &b));
        }

        // hamming_distance is symmetric
        #[test]
        fn hamming_symmetric(
            (a, b) in prop::sample::select(SIZES).prop_flat_map(arb_u8_pair)
        ) {
            prop_assert_eq!(hamming_distance(&a, &b), hamming_distance(&b, &a));
        }

        // hamming_distance is 0 for identical inputs
        #[test]
        fn hamming_self_is_zero(
            a in prop::sample::select(SIZES).prop_flat_map(|s| prop::collection::vec(any::<u8>(), s))
        ) {
            prop_assert_eq!(hamming_distance(&a, &a), 0);
        }

        // hamming_distance <= 8 * len (one bit per byte position at most)
        #[test]
        fn hamming_bounded(
            (a, b) in prop::sample::select(SIZES).prop_flat_map(arb_u8_pair)
        ) {
            let h = hamming_distance(&a, &b);
            prop_assert!(h <= (a.len() * 8) as u32,
                "hamming {} > max {} for len {}", h, a.len() * 8, a.len());
        }
    }
}
