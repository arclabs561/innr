//! SIMD-accelerated binary (1-bit) vector operations.
//!
//! # Binary Quantization
//!
//! Binary vectors use only two values: {0, 1}. This yields 1 bit per dimension,
//! providing the highest possible compression (32x vs f32). Despite the extreme
//! quantization, binary embeddings retain enough signal for first-stage retrieval
//! when used as a re-ranking filter or in combination with asymmetric scoring.
//!
//! Binary quantization composes well with matryoshka (prefix-truncatable)
//! embeddings: truncate first to reduce dimension, then binarize. This yields
//! compounding compression along both the dimension and precision axes.
//!
//! # Hamming Distance
//!
//! For binary vectors, the distance is the number of positions where bits differ.
//! This is computed using XOR followed by popcount.
//!
//! # Inner Product
//!
//! The "dot product" of binary vectors is the number of positions where both are 1.
//! This is computed using AND followed by popcount.
//!
//! # References
//!
//! - Gan et al. (2023). "Binary Embedding-based Retrieval at Tencent" (KDD) --
//!   production validation of binary embeddings for retrieval over billions of
//!   documents with 32x compression, using Hamming distance as the primary metric.
//! - Ahemad (2025). "Quantization Aware Matryoshka Adaptation" (CIKM) -- shows
//!   that matryoshka + binary quantization compose well: models trained with
//!   joint MRL + binary awareness retain quality at extreme compression ratios.

/// Packed binary vector as array of u64.
///
/// Each u64 stores 64 binary values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedBinary {
    /// Packed data: 64 values per u64
    pub data: Vec<u64>,
    /// Original dimension
    pub dimension: usize,
}

impl PackedBinary {
    /// Create from raw data.
    pub fn new(data: Vec<u64>, dimension: usize) -> Self {
        Self { data, dimension }
    }

    /// Create zero-initialized vector.
    pub fn zeros(dimension: usize) -> Self {
        let num_u64s = dimension.div_ceil(64);
        Self {
            data: vec![0; num_u64s],
            dimension,
        }
    }

    /// Set value at index.
    pub fn set(&mut self, idx: usize, val: bool) {
        if idx >= self.dimension {
            return;
        }
        let word = idx / 64;
        let bit = idx % 64;
        if val {
            self.data[word] |= 1u64 << bit;
        } else {
            self.data[word] &= !(1u64 << bit);
        }
    }

    /// Get value at index.
    pub fn get(&self, idx: usize) -> bool {
        if idx >= self.dimension {
            return false;
        }
        let word = idx / 64;
        let bit = idx % 64;
        ((self.data[word] >> bit) & 1) != 0
    }

    /// Memory size in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * 8
    }
}

/// Encode f32 slice as packed binary.
///
/// Values above `threshold` become 1, others become 0.
///
/// ```
/// use innr::binary::encode_binary;
///
/// let v = [0.5_f32, -0.1, 0.9, 0.0];
/// let packed = encode_binary(&v, 0.0);
/// assert!(packed.get(0));   // 0.5 > 0.0
/// assert!(!packed.get(1));  // -0.1 <= 0.0
/// assert!(packed.get(2));   // 0.9 > 0.0
/// ```
pub fn encode_binary(values: &[f32], threshold: f32) -> PackedBinary {
    let mut result = PackedBinary::zeros(values.len());
    for (i, &v) in values.iter().enumerate() {
        if v > threshold {
            result.set(i, true);
        }
    }
    result
}

/// Compute Hamming distance between two binary vectors.
///
/// ```
/// use innr::binary::{encode_binary, binary_hamming};
///
/// let a = encode_binary(&[1.0, -1.0, 1.0, -1.0], 0.0);
/// let b = encode_binary(&[1.0, 1.0, -1.0, -1.0], 0.0);
/// assert_eq!(binary_hamming(&a, &b), 2); // positions 1 and 2 differ
/// ```
#[inline]
pub fn binary_hamming(a: &PackedBinary, b: &PackedBinary) -> u32 {
    debug_assert_eq!(a.dimension, b.dimension);
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(&wa, &wb)| (wa ^ wb).count_ones())
        .sum()
}

/// Compute binary dot product (intersection count).
///
/// ```
/// use innr::binary::{encode_binary, binary_dot};
///
/// let a = encode_binary(&[1.0, -1.0, 1.0, -1.0], 0.0);
/// let b = encode_binary(&[1.0, 1.0, -1.0, -1.0], 0.0);
/// assert_eq!(binary_dot(&a, &b), 1); // only position 0 is 1 in both
/// ```
#[inline]
pub fn binary_dot(a: &PackedBinary, b: &PackedBinary) -> u32 {
    debug_assert_eq!(a.dimension, b.dimension);
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(&wa, &wb)| (wa & wb).count_ones())
        .sum()
}

/// Compute Jaccard similarity: `|A intersection B| / `|A union B|`.
///
/// ```
/// use innr::binary::{encode_binary, binary_jaccard};
///
/// let a = encode_binary(&[1.0, -1.0, 1.0, -1.0], 0.0);
/// let b = encode_binary(&[1.0, 1.0, -1.0, -1.0], 0.0);
/// // intersection=1, union=3 -> jaccard = 1/3
/// let j = binary_jaccard(&a, &b);
/// assert!((j - 1.0 / 3.0).abs() < 1e-6);
/// ```
pub fn binary_jaccard(a: &PackedBinary, b: &PackedBinary) -> f32 {
    let intersection = binary_dot(a, b);
    let union = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&wa, &wb)| (wa | wb).count_ones())
        .sum::<u32>();

    if union == 0 {
        1.0
    } else {
        intersection as f32 / union as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_ops() {
        let mut a = PackedBinary::zeros(4);
        let mut b = PackedBinary::zeros(4);

        a.set(0, true);
        a.set(1, true);

        b.set(1, true);
        b.set(2, true);

        // a: 1100, b: 0110
        assert_eq!(binary_hamming(&a, &b), 2); // indices 0 and 2 differ
        assert_eq!(binary_dot(&a, &b), 1); // only index 1 matches
        assert!((binary_jaccard(&a, &b) - 1.0 / 3.0).abs() < 1e-6);
    }

    // =========================================================================
    // PackedBinary construction and accessors
    // =========================================================================

    #[test]
    fn test_zeros() {
        let v = PackedBinary::zeros(128);
        assert_eq!(v.dimension, 128);
        assert_eq!(v.data.len(), 2); // 128 / 64 = 2
        for i in 0..128 {
            assert!(!v.get(i), "bit {i} should be 0");
        }
    }

    #[test]
    fn test_new() {
        let data = vec![0xFF_u64]; // first 8 bits set
        let v = PackedBinary::new(data, 8);
        for i in 0..8 {
            assert!(v.get(i), "bit {i} should be 1");
        }
    }

    #[test]
    fn test_memory_bytes() {
        let v = PackedBinary::zeros(256);
        // 256 / 64 = 4 u64s = 32 bytes
        assert_eq!(v.memory_bytes(), 32);
    }

    // =========================================================================
    // set/get boundary and out-of-bounds behavior
    // =========================================================================

    #[test]
    fn test_set_and_clear() {
        let mut v = PackedBinary::zeros(64);
        v.set(0, true);
        assert!(v.get(0));
        v.set(0, false);
        assert!(!v.get(0));
    }

    #[test]
    fn test_set_out_of_bounds_is_noop() {
        let mut v = PackedBinary::zeros(4);
        v.set(100, true); // should not panic
        assert!(!v.get(100)); // out-of-bounds returns false
    }

    #[test]
    fn test_get_out_of_bounds() {
        let v = PackedBinary::zeros(4);
        assert!(!v.get(4));
        assert!(!v.get(1000));
    }

    #[test]
    fn test_set_last_bit_in_word() {
        let mut v = PackedBinary::zeros(64);
        v.set(63, true);
        assert!(v.get(63));
        assert!(!v.get(62));
    }

    // =========================================================================
    // Multi-word operations (dimension > 64)
    // =========================================================================

    #[test]
    fn test_multi_word_hamming() {
        // 128-bit vectors spanning two u64 words
        let mut a = PackedBinary::zeros(128);
        let mut b = PackedBinary::zeros(128);

        // Set bits in first word
        a.set(0, true);
        b.set(0, true); // same

        // Set bits in second word
        a.set(64, true);
        b.set(65, true); // different positions

        // Differences: bit 64 (a has, b doesn't) and bit 65 (b has, a doesn't) = 2
        assert_eq!(binary_hamming(&a, &b), 2);
    }

    #[test]
    fn test_multi_word_dot() {
        let mut a = PackedBinary::zeros(128);
        let mut b = PackedBinary::zeros(128);

        a.set(0, true);
        a.set(64, true);
        a.set(65, true);

        b.set(0, true);
        b.set(64, true);
        b.set(100, true);

        // Intersection: bits 0 and 64 -> dot = 2
        assert_eq!(binary_dot(&a, &b), 2);
    }

    #[test]
    fn test_multi_word_jaccard() {
        let mut a = PackedBinary::zeros(128);
        let mut b = PackedBinary::zeros(128);

        // a: bits {0, 64, 65}
        a.set(0, true);
        a.set(64, true);
        a.set(65, true);

        // b: bits {0, 64, 100}
        b.set(0, true);
        b.set(64, true);
        b.set(100, true);

        // intersection = {0, 64} -> 2
        // union = {0, 64, 65, 100} -> 4
        // jaccard = 2/4 = 0.5
        let j = binary_jaccard(&a, &b);
        assert!((j - 0.5).abs() < 1e-6);
    }

    // =========================================================================
    // encode_binary edge cases
    // =========================================================================

    #[test]
    fn test_encode_binary_all_above() {
        let v = [1.0, 2.0, 3.0, 4.0];
        let packed = encode_binary(&v, 0.0);
        for i in 0..4 {
            assert!(packed.get(i), "all values > 0, bit {i} should be set");
        }
    }

    #[test]
    fn test_encode_binary_all_below() {
        let v = [-1.0, -2.0, -3.0, -4.0];
        let packed = encode_binary(&v, 0.0);
        for i in 0..4 {
            assert!(!packed.get(i), "all values <= 0, bit {i} should be clear");
        }
    }

    #[test]
    fn test_encode_binary_at_threshold() {
        // Values exactly at threshold are NOT above it (> not >=)
        let v = [0.0_f32];
        let packed = encode_binary(&v, 0.0);
        assert!(!packed.get(0), "value exactly at threshold should be 0");
    }

    #[test]
    fn test_encode_binary_empty() {
        let v: [f32; 0] = [];
        let packed = encode_binary(&v, 0.0);
        assert_eq!(packed.dimension, 0);
    }

    #[test]
    fn test_encode_binary_large() {
        // 768 dimensions (typical embedding size)
        let v: Vec<f32> = (0..768).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let packed = encode_binary(&v, 0.0);
        assert_eq!(packed.dimension, 768);
        assert_eq!(packed.data.len(), 12); // ceil(768/64)

        for i in 0..768 {
            if i % 2 == 0 {
                assert!(packed.get(i), "even index {i} should be 1");
            } else {
                assert!(!packed.get(i), "odd index {i} should be 0");
            }
        }
    }

    // =========================================================================
    // Hamming/dot/Jaccard identity and edge cases
    // =========================================================================

    #[test]
    fn test_hamming_identical() {
        let v = encode_binary(&[1.0, -1.0, 1.0, -1.0], 0.0);
        assert_eq!(binary_hamming(&v, &v), 0);
    }

    #[test]
    fn test_hamming_complement() {
        let a = encode_binary(&[1.0, 1.0, 1.0, 1.0], 0.0);
        let b = encode_binary(&[-1.0, -1.0, -1.0, -1.0], 0.0);
        assert_eq!(binary_hamming(&a, &b), 4);
    }

    #[test]
    fn test_dot_self() {
        let v = encode_binary(&[1.0, -1.0, 1.0, -1.0, 1.0], 0.0);
        // Self-dot = popcount = number of 1-bits = 3
        assert_eq!(binary_dot(&v, &v), 3);
    }

    #[test]
    fn test_jaccard_identical() {
        let v = encode_binary(&[1.0, -1.0, 1.0], 0.0);
        let j = binary_jaccard(&v, &v);
        assert!((j - 1.0).abs() < 1e-6, "jaccard(v, v) should be 1.0");
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a = encode_binary(&[1.0, -1.0], 0.0); // bits: {0}
        let b = encode_binary(&[-1.0, 1.0], 0.0); // bits: {1}
        // intersection = 0, union = 2 -> jaccard = 0
        let j = binary_jaccard(&a, &b);
        assert!(j.abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_both_empty() {
        let a = PackedBinary::zeros(4);
        let b = PackedBinary::zeros(4);
        // No bits set in either -> union=0 -> returns 1.0 by convention
        assert!((binary_jaccard(&a, &b) - 1.0).abs() < 1e-6);
    }

    // =========================================================================
    // Non-zero threshold encoding
    // =========================================================================

    #[test]
    fn test_encode_binary_nonzero_threshold() {
        let v = [0.1, 0.5, 0.9, 1.5];
        let packed = encode_binary(&v, 0.5);
        assert!(!packed.get(0)); // 0.1 <= 0.5
        assert!(!packed.get(1)); // 0.5 <= 0.5 (not strictly above)
        assert!(packed.get(2)); // 0.9 > 0.5
        assert!(packed.get(3)); // 1.5 > 0.5
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Dimensions that cross u64 word boundaries.
    const DIMS: &[usize] = &[31, 32, 33, 63, 64, 65, 128];

    /// Strategy: pick a dimension from DIMS, then generate a bool vec of that length.
    fn arb_packed_binary() -> impl Strategy<Value = PackedBinary> {
        prop::sample::select(DIMS).prop_flat_map(|dim| {
            prop::collection::vec(any::<bool>(), dim).prop_map(move |bits| {
                let mut pb = PackedBinary::zeros(dim);
                for (i, &b) in bits.iter().enumerate() {
                    pb.set(i, b);
                }
                pb
            })
        })
    }

    /// Strategy: pair of PackedBinary with the same dimension.
    fn arb_packed_binary_pair() -> impl Strategy<Value = (PackedBinary, PackedBinary)> {
        prop::sample::select(DIMS).prop_flat_map(|dim| {
            let a = prop::collection::vec(any::<bool>(), dim).prop_map(move |bits| {
                let mut pb = PackedBinary::zeros(dim);
                for (i, &b) in bits.iter().enumerate() {
                    pb.set(i, b);
                }
                pb
            });
            let b = prop::collection::vec(any::<bool>(), dim).prop_map(move |bits| {
                let mut pb = PackedBinary::zeros(dim);
                for (i, &b) in bits.iter().enumerate() {
                    pb.set(i, b);
                }
                pb
            });
            (a, b)
        })
    }

    proptest! {
        // 1. Hamming symmetry: hamming(a, b) == hamming(b, a)
        #[test]
        fn proptest_hamming_symmetry((a, b) in arb_packed_binary_pair()) {
            prop_assert_eq!(binary_hamming(&a, &b), binary_hamming(&b, &a));
        }

        // 2. Hamming self-distance is zero: hamming(a, a) == 0
        #[test]
        fn proptest_hamming_self_zero(a in arb_packed_binary()) {
            prop_assert_eq!(binary_hamming(&a, &a), 0);
        }

        // 3. Hamming range: hamming(a, b) <= dimension
        #[test]
        fn proptest_hamming_range((a, b) in arb_packed_binary_pair()) {
            let h = binary_hamming(&a, &b);
            prop_assert!(h as usize <= a.dimension,
                "hamming {} > dimension {}", h, a.dimension);
        }

        // 4. Jaccard range: jaccard(a, b) in [0.0, 1.0]
        #[test]
        fn proptest_jaccard_range((a, b) in arb_packed_binary_pair()) {
            let j = binary_jaccard(&a, &b);
            prop_assert!(j >= 0.0 && j <= 1.0,
                "jaccard {} not in [0, 1]", j);
        }

        // 5. Jaccard self-similarity: jaccard(a, a) == 1.0 for non-zero vectors
        #[test]
        fn proptest_jaccard_self(a in arb_packed_binary()) {
            let has_bits = a.data.iter().any(|&w| w != 0);
            let j = binary_jaccard(&a, &a);
            if has_bits {
                prop_assert!((j - 1.0).abs() < 1e-6,
                    "jaccard(a, a) = {} but expected 1.0 for non-zero vector", j);
            } else {
                // Convention: jaccard of two zero vectors is 1.0
                prop_assert!((j - 1.0).abs() < 1e-6,
                    "jaccard(zero, zero) = {} but expected 1.0", j);
            }
        }

        // 6. Dot product commutativity: dot(a, b) == dot(b, a)
        #[test]
        fn proptest_dot_commutativity((a, b) in arb_packed_binary_pair()) {
            prop_assert_eq!(binary_dot(&a, &b), binary_dot(&b, &a));
        }

        // 7. Encode deterministic: encode(v, t) == encode(v, t) for same inputs
        #[test]
        fn proptest_encode_deterministic(
            dim in prop::sample::select(DIMS),
            threshold in -10.0f32..10.0,
        ) {
            let values: Vec<f32> = (0..dim).map(|i| (i as f32 - dim as f32 / 2.0) * 0.1).collect();
            let a = encode_binary(&values, threshold);
            let b = encode_binary(&values, threshold);
            prop_assert_eq!(a, b);
        }
    }
}
