//! Integer quantization primitives: u8 dot product and Hamming distance.
//!
//! These are the hot inner loops for quantized vector search:
//!
//! - `dot_u8`: symmetric u8×u8 dot product for INT8-quantized embeddings
//! - `hamming_distance`: byte-packed Hamming distance for binary-quantized embeddings
//!
//! # Dispatch
//!
//! Both functions use the same runtime-dispatch pattern as the rest of this crate:
//! AVX-512 > AVX2 > NEON > portable.

// arch is only used on architectures with SIMD dispatch
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::arch;

/// Minimum vector length for SIMD to be worthwhile on these integer paths.
const MIN_DIM_SIMD: usize = 32;

/// Minimum dimension for the AVX-512 path (64-byte chunks).
#[cfg(target_arch = "x86_64")]
const MIN_DIM_AVX512: usize = 64;

/// Unsigned 8-bit integer dot product: `Σ(a[i] * b[i])`.
///
/// Returns 0 for empty slices. Computes in a `u32` accumulator to avoid overflow
/// (max value per element = 255 * 255 = 65025; max total at dim 65535 ≈ 4.26e9 < u32::MAX).
///
/// # SIMD Acceleration
///
/// Automatically dispatches to (in order of preference):
/// - AVX-512 on x86_64 (runtime detection, n >= 64)
/// - AVX2 on x86_64 (runtime detection, n >= 32)
/// - NEON on aarch64 (always available, n >= 32)
/// - Portable fallback otherwise
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
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
#[allow(unsafe_code)]
pub fn dot_u8(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(
        a.len(),
        b.len(),
        "innr::dot_u8: slice length mismatch ({} vs {})",
        a.len(),
        b.len()
    );

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let n = a.len();

    #[cfg(target_arch = "x86_64")]
    {
        if n >= MIN_DIM_AVX512
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512f")
        {
            // SAFETY: AVX-512BW verified via runtime detection.
            return unsafe { arch::x86_64::dot_u8_avx512(a, b) };
        }

        if n >= MIN_DIM_SIMD && is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 verified via runtime detection.
            return unsafe { arch::x86_64::dot_u8_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if n >= MIN_DIM_SIMD {
            // SAFETY: NEON is always available on aarch64.
            return unsafe { arch::aarch64::dot_u8_neon(a, b) };
        }
    }

    #[allow(unreachable_code)]
    dot_u8_portable(a, b)
}

/// Portable (non-SIMD) u8 dot product with 4-way unrolled accumulators.
///
/// Uses four independent u32 accumulators to avoid data-dependency stalls
/// on pipelined CPUs. LLVM typically vectorizes this further.
#[inline]
#[must_use]
pub fn dot_u8_portable(a: &[u8], b: &[u8]) -> u32 {
    let n = a.len().min(b.len());
    let chunks = n / 4;

    let mut s0: u32 = 0;
    let mut s1: u32 = 0;
    let mut s2: u32 = 0;
    let mut s3: u32 = 0;

    for i in 0..chunks {
        let base = i * 4;
        s0 += a[base] as u32 * b[base] as u32;
        s1 += a[base + 1] as u32 * b[base + 1] as u32;
        s2 += a[base + 2] as u32 * b[base + 2] as u32;
        s3 += a[base + 3] as u32 * b[base + 3] as u32;
    }

    let mut result = s0.wrapping_add(s1).wrapping_add(s2).wrapping_add(s3);

    for i in (chunks * 4)..n {
        result += a[i] as u32 * b[i] as u32;
    }

    result
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
/// Automatically dispatches to (in order of preference):
/// - AVX-512 with VPOPCNTDQ on x86_64 (runtime detection, n >= 64)
/// - AVX2 with VPSHUFB LUT on x86_64 (runtime detection, n >= 32)
/// - NEON on aarch64 (vcntq_u8, always available, n >= 32)
/// - Portable fallback otherwise
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
#[allow(unsafe_code)]
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(
        a.len(),
        b.len(),
        "innr::hamming_distance: slice length mismatch ({} vs {})",
        a.len(),
        b.len()
    );

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let n = a.len();

    #[cfg(target_arch = "x86_64")]
    {
        if n >= MIN_DIM_AVX512
            && is_x86_feature_detected!("avx512vpopcntdq")
            && is_x86_feature_detected!("avx512f")
        {
            // SAFETY: AVX-512VPOPCNTDQ verified via runtime detection.
            return unsafe { arch::x86_64::hamming_avx512(a, b) };
        }

        if n >= MIN_DIM_SIMD && is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 verified via runtime detection.
            return unsafe { arch::x86_64::hamming_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if n >= MIN_DIM_SIMD {
            // SAFETY: NEON is always available on aarch64.
            return unsafe { arch::aarch64::hamming_neon(a, b) };
        }
    }

    #[allow(unreachable_code)]
    hamming_portable(a, b)
}

/// Portable (non-SIMD) Hamming distance: XOR bytes then count bits.
#[inline]
#[must_use]
pub fn hamming_portable(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // dim=128: exercises AVX2 and NEON paths
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
