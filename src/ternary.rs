//! SIMD-accelerated ternary vector operations.
//!
//! # Ternary Quantization
//!
//! Ternary vectors use only three values per dimension: {-1, 0, +1}.
//! This yields ~1.58 bits per dimension (log2(3)), providing massive
//! compression while maintaining surprising accuracy.
//!
//! # Representation
//!
//! We use two bits per value:
//! - 00 = 0
//! - 01 = +1
//! - 10 = -1
//! - 11 = reserved
//!
//! This packs 4 values per byte, 32 values per u64.
//!
//! # Inner Product Computation
//!
//! For ternary vectors a, b, the inner product is:
//!
//! ```text
//! <a, b> = Σ a[i] * b[i]
//!        = count(a=+1 AND b=+1) + count(a=-1 AND b=-1)
//!        - count(a=+1 AND b=-1) - count(a=-1 AND b=+1)
//!        = count(same_sign) - count(different_sign)
//! ```
//!
//! This can be computed efficiently using bit manipulation:
//! 1. Extract "positive" bits (01 patterns) and "negative" bits (10 patterns)
//! 2. Use AND/XOR to find agreements/disagreements
//! 3. Use popcount to count matching positions
//!
//! # SIMD Acceleration
//!
//! Popcount is highly efficient on modern CPUs:
//! - x86_64: POPCNT instruction (1 cycle throughput)
//! - AVX-512 VPOPCNT: 64 bytes per operation
//! - ARM NEON: CNT instruction
//!
//! With 32 values per u64, a 768-dim vector requires only 24 u64s (192 bytes).

/// Packed ternary vector as array of u64.
///
/// Each u64 stores 32 ternary values (2 bits each).
/// Format: bits[2i..2i+2] encode value at position i.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedTernary {
    /// Packed data: 32 values per u64
    pub data: Vec<u64>,
    /// Original dimension
    pub dimension: usize,
}

impl PackedTernary {
    /// Create from raw data.
    pub fn new(data: Vec<u64>, dimension: usize) -> Self {
        Self { data, dimension }
    }

    /// Create zero-initialized vector.
    pub fn zeros(dimension: usize) -> Self {
        let num_u64s = dimension.div_ceil(32);
        Self {
            data: vec![0; num_u64s],
            dimension,
        }
    }

    /// Set value at index.
    ///
    /// # Arguments
    /// * `idx` - Index (0..dimension)
    /// * `val` - Value (-1, 0, or 1)
    pub fn set(&mut self, idx: usize, val: i8) {
        if idx >= self.dimension {
            return;
        }
        let word = idx / 32;
        let bit = (idx % 32) * 2;

        // Clear existing bits
        self.data[word] &= !(0b11u64 << bit);

        // Set new value
        let bits: u64 = match val {
            1 => 0b01,
            -1 => 0b10,
            _ => 0b00,
        };
        self.data[word] |= bits << bit;
    }

    /// Get value at index.
    pub fn get(&self, idx: usize) -> i8 {
        if idx >= self.dimension {
            return 0;
        }
        let word = idx / 32;
        let bit = (idx % 32) * 2;
        let bits = (self.data[word] >> bit) & 0b11;
        match bits {
            0b01 => 1,
            0b10 => -1,
            _ => 0,
        }
    }

    /// Count non-zero elements.
    pub fn nnz(&self) -> usize {
        let mut count = 0;
        for i in 0..self.dimension {
            if self.get(i) != 0 {
                count += 1;
            }
        }
        count
    }

    /// Memory size in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * 8
    }
}

/// Encode f32 slice as packed ternary.
///
/// Values above `threshold` become +1, below `-threshold` become -1,
/// values in between become 0.
pub fn encode_ternary(values: &[f32], threshold: f32) -> PackedTernary {
    let mut result = PackedTernary::zeros(values.len());
    for (i, &v) in values.iter().enumerate() {
        if v > threshold {
            result.set(i, 1);
        } else if v < -threshold {
            result.set(i, -1);
        }
    }
    result
}

/// Compute ternary inner product using popcount.
///
/// This is the core operation, computing:
/// `<a, b> = count(same_sign) - count(different_sign)`
///
/// # Algorithm
///
/// For each u64 word:
/// 1. Extract positive bits (where 2-bit pattern = 01)
/// 2. Extract negative bits (where 2-bit pattern = 10)
/// 3. Same-sign matches: (pos_a & pos_b) | (neg_a & neg_b)
/// 4. Different-sign matches: (pos_a & neg_b) | (neg_a & pos_b)
/// 5. Contribution = popcount(same) - popcount(diff)
#[inline]
pub fn ternary_dot(a: &PackedTernary, b: &PackedTernary) -> i32 {
    debug_assert_eq!(a.dimension, b.dimension);
    debug_assert_eq!(a.data.len(), b.data.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("popcnt") {
            // SAFETY: POPCNT verified via runtime detection
            return unsafe { ternary_dot_popcnt(&a.data, &b.data) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON always available on aarch64
        return ternary_dot_portable(&a.data, &b.data);
    }

    #[allow(unreachable_code)]
    ternary_dot_portable(&a.data, &b.data)
}

/// Portable implementation of ternary dot product.
fn ternary_dot_portable(a: &[u64], b: &[u64]) -> i32 {
    let mut same_count: u32 = 0;
    let mut diff_count: u32 = 0;

    // Mask for extracting odd bits (bit 0, 2, 4, ...)
    const ODD_MASK: u64 = 0x5555555555555555;
    // Mask for extracting even bits (bit 1, 3, 5, ...)
    const EVEN_MASK: u64 = 0xAAAAAAAAAAAAAAAA;

    for (&wa, &wb) in a.iter().zip(b.iter()) {
        // Extract positive bits (pattern 01: bit0=1, bit1=0)
        // positive_a[i] = 1 if bits[2i..2i+2] == 01
        let pos_a = wa & !((wa & EVEN_MASK) >> 1) & ODD_MASK;
        let pos_b = wb & !((wb & EVEN_MASK) >> 1) & ODD_MASK;

        // Extract negative bits (pattern 10: bit0=0, bit1=1)
        // negative_a[i] = 1 if bits[2i..2i+2] == 10
        let neg_a = !wa & ((wa & EVEN_MASK) >> 1) & ODD_MASK;
        let neg_b = !wb & ((wb & EVEN_MASK) >> 1) & ODD_MASK;

        // Same sign: both positive or both negative
        let same = (pos_a & pos_b) | (neg_a & neg_b);

        // Different sign: one positive, one negative
        let diff = (pos_a & neg_b) | (neg_a & pos_b);

        same_count += same.count_ones();
        diff_count += diff.count_ones();
    }

    same_count as i32 - diff_count as i32
}

/// x86_64 POPCNT implementation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
unsafe fn ternary_dot_popcnt(a: &[u64], b: &[u64]) -> i32 {
    use std::arch::x86_64::_popcnt64;

    let mut same_count: i64 = 0;
    let mut diff_count: i64 = 0;

    const ODD_MASK: u64 = 0x5555555555555555;
    const EVEN_MASK: u64 = 0xAAAAAAAAAAAAAAAA;

    for (&wa, &wb) in a.iter().zip(b.iter()) {
        let pos_a = wa & !((wa & EVEN_MASK) >> 1) & ODD_MASK;
        let pos_b = wb & !((wb & EVEN_MASK) >> 1) & ODD_MASK;
        let neg_a = !wa & ((wa & EVEN_MASK) >> 1) & ODD_MASK;
        let neg_b = !wb & ((wb & EVEN_MASK) >> 1) & ODD_MASK;

        let same = (pos_a & pos_b) | (neg_a & neg_b);
        let diff = (pos_a & neg_b) | (neg_a & pos_b);

        same_count += i64::from(_popcnt64(same as i64));
        diff_count += i64::from(_popcnt64(diff as i64));
    }

    (same_count - diff_count) as i32
}

/// Asymmetric dot product: f32 query against ternary vector.
///
/// More accurate than symmetric ternary comparison since query
/// retains full precision.
#[inline]
pub fn asymmetric_dot(query: &[f32], ternary: &PackedTernary) -> f32 {
    debug_assert_eq!(query.len(), ternary.dimension);

    let mut sum = 0.0f32;
    for (i, &q) in query.iter().enumerate() {
        let t = ternary.get(i) as f32;
        sum += q * t;
    }
    sum
}

/// Batch asymmetric dot products.
///
/// Computes query against multiple ternary vectors efficiently.
pub fn batch_asymmetric_dot(query: &[f32], vectors: &[PackedTernary]) -> Vec<f32> {
    vectors.iter().map(|v| asymmetric_dot(query, v)).collect()
}

/// Hamming distance for ternary vectors.
///
/// Counts positions where values differ (ignoring zeros).
pub fn ternary_hamming(a: &PackedTernary, b: &PackedTernary) -> u32 {
    debug_assert_eq!(a.dimension, b.dimension);

    let mut diff_count: u32 = 0;

    const ODD_MASK: u64 = 0x5555555555555555;
    const EVEN_MASK: u64 = 0xAAAAAAAAAAAAAAAA;

    for (&wa, &wb) in a.data.iter().zip(b.data.iter()) {
        // Extract non-zero positions
        let nz_a = (wa & ODD_MASK) | ((wa & EVEN_MASK) >> 1);
        let nz_b = (wb & ODD_MASK) | ((wb & EVEN_MASK) >> 1);

        // XOR to find differences, mask to only count where both non-zero
        let both_nz = nz_a & nz_b;
        let xor = wa ^ wb;
        let diff = (xor & ODD_MASK) | ((xor & EVEN_MASK) >> 1);

        diff_count += (diff & both_nz).count_ones();
    }

    diff_count
}

/// Sparsity (fraction of zeros) in ternary vector.
pub fn sparsity(v: &PackedTernary) -> f32 {
    let nnz = v.nnz();
    1.0 - (nnz as f32 / v.dimension as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let values = vec![0.5, -0.5, 0.1, -0.1, 0.8, -0.8];
        let packed = encode_ternary(&values, 0.3);

        assert_eq!(packed.get(0), 1); // 0.5 > 0.3
        assert_eq!(packed.get(1), -1); // -0.5 < -0.3
        assert_eq!(packed.get(2), 0); // 0.1 in [-0.3, 0.3]
        assert_eq!(packed.get(3), 0); // -0.1 in [-0.3, 0.3]
        assert_eq!(packed.get(4), 1); // 0.8 > 0.3
        assert_eq!(packed.get(5), -1); // -0.8 < -0.3
    }

    #[test]
    fn test_ternary_dot_same() {
        let mut a = PackedTernary::zeros(4);
        a.set(0, 1);
        a.set(1, -1);
        a.set(2, 0);
        a.set(3, 1);

        // Dot with itself: 1*1 + (-1)*(-1) + 0*0 + 1*1 = 3
        let dot = ternary_dot(&a, &a);
        assert_eq!(dot, 3);
    }

    #[test]
    fn test_ternary_dot_opposite() {
        let mut a = PackedTernary::zeros(4);
        let mut b = PackedTernary::zeros(4);

        a.set(0, 1);
        a.set(1, -1);
        b.set(0, -1);
        b.set(1, 1);

        // Opposite signs: 1*(-1) + (-1)*1 = -2
        let dot = ternary_dot(&a, &b);
        assert_eq!(dot, -2);
    }

    #[test]
    fn test_ternary_dot_orthogonal() {
        let mut a = PackedTernary::zeros(4);
        let mut b = PackedTernary::zeros(4);

        a.set(0, 1);
        a.set(1, 0);
        b.set(0, 0);
        b.set(1, 1);

        // Orthogonal (no overlap): 1*0 + 0*1 = 0
        let dot = ternary_dot(&a, &b);
        assert_eq!(dot, 0);
    }

    #[test]
    fn test_large_vector() {
        // Test with 768 dimensions (typical embedding size)
        let values: Vec<f32> = (0..768)
            .map(|i| {
                let x = (i as f32 / 768.0) - 0.5;
                if i % 3 == 0 {
                    x * 2.0
                } else {
                    x * 0.5
                }
            })
            .collect();

        let packed = encode_ternary(&values, 0.3);

        // Should compress to 768/32 = 24 u64s = 192 bytes
        assert_eq!(packed.data.len(), 24);
        assert_eq!(packed.memory_bytes(), 192);

        // Self-dot should equal nnz
        let dot = ternary_dot(&packed, &packed);
        assert_eq!(dot as usize, packed.nnz());
    }

    #[test]
    fn test_asymmetric_dot() {
        let mut t = PackedTernary::zeros(4);
        t.set(0, 1);
        t.set(1, -1);
        t.set(2, 0);
        t.set(3, 1);

        let query = vec![0.5, 0.5, 0.5, 0.5];

        // 0.5*1 + 0.5*(-1) + 0.5*0 + 0.5*1 = 0.5
        let dot = asymmetric_dot(&query, &t);
        assert!((dot - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_hamming() {
        let mut a = PackedTernary::zeros(4);
        let mut b = PackedTernary::zeros(4);

        a.set(0, 1);
        a.set(1, -1);
        a.set(2, 1);
        b.set(0, 1);
        b.set(1, 1); // differs
        b.set(2, -1); // differs

        let hamming = ternary_hamming(&a, &b);
        assert_eq!(hamming, 2);
    }

    // =========================================================================
    // PackedTernary: construction and accessors
    // =========================================================================

    #[test]
    fn test_zeros_all_zero() {
        let v = PackedTernary::zeros(100);
        for i in 0..100 {
            assert_eq!(v.get(i), 0, "index {i} should be 0");
        }
        assert_eq!(v.nnz(), 0);
    }

    #[test]
    fn test_set_get_all_values() {
        let mut v = PackedTernary::zeros(3);
        v.set(0, 1);
        v.set(1, -1);
        v.set(2, 0);
        assert_eq!(v.get(0), 1);
        assert_eq!(v.get(1), -1);
        assert_eq!(v.get(2), 0);
    }

    #[test]
    fn test_set_overwrite() {
        let mut v = PackedTernary::zeros(1);
        v.set(0, 1);
        assert_eq!(v.get(0), 1);
        v.set(0, -1);
        assert_eq!(v.get(0), -1);
        v.set(0, 0);
        assert_eq!(v.get(0), 0);
    }

    #[test]
    fn test_set_out_of_bounds_is_noop() {
        let mut v = PackedTernary::zeros(4);
        v.set(100, 1); // should not panic
    }

    #[test]
    fn test_get_out_of_bounds_returns_zero() {
        let v = PackedTernary::zeros(4);
        assert_eq!(v.get(4), 0);
        assert_eq!(v.get(1000), 0);
    }

    // =========================================================================
    // Word boundary: values at positions 31, 32 (crossing u64 word boundary)
    // =========================================================================

    #[test]
    fn test_word_boundary() {
        let mut v = PackedTernary::zeros(64);
        // Last position in first word
        v.set(31, 1);
        // First position in second word
        v.set(32, -1);

        assert_eq!(v.get(31), 1);
        assert_eq!(v.get(32), -1);
        assert_eq!(v.get(30), 0);
        assert_eq!(v.get(33), 0);
    }

    // =========================================================================
    // nnz and sparsity
    // =========================================================================

    #[test]
    fn test_nnz() {
        let mut v = PackedTernary::zeros(10);
        v.set(0, 1);
        v.set(3, -1);
        v.set(7, 1);
        assert_eq!(v.nnz(), 3);
    }

    #[test]
    fn test_sparsity_all_zero() {
        let v = PackedTernary::zeros(100);
        assert!((sparsity(&v) - 1.0).abs() < 1e-6, "all-zero vector has sparsity 1.0");
    }

    #[test]
    fn test_sparsity_all_nonzero() {
        let values: Vec<f32> = vec![1.0, -1.0, 1.0, -1.0];
        let packed = encode_ternary(&values, 0.0);
        assert!(sparsity(&packed).abs() < 1e-6, "all-nonzero vector has sparsity 0.0");
    }

    #[test]
    fn test_sparsity_half() {
        let values: Vec<f32> = vec![1.0, 0.0, -1.0, 0.0];
        let packed = encode_ternary(&values, 0.5);
        // 1.0 > 0.5 -> +1, 0.0 in range -> 0, -1.0 < -0.5 -> -1, 0.0 -> 0
        // nnz = 2 out of 4
        assert!((sparsity(&packed) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_memory_bytes() {
        let v = PackedTernary::zeros(768);
        // 768 / 32 = 24 u64s = 192 bytes
        assert_eq!(v.memory_bytes(), 192);
    }

    // =========================================================================
    // encode_ternary edge cases
    // =========================================================================

    #[test]
    fn test_encode_ternary_empty() {
        let packed = encode_ternary(&[], 0.5);
        assert_eq!(packed.dimension, 0);
        assert_eq!(packed.nnz(), 0);
    }

    #[test]
    fn test_encode_ternary_at_threshold() {
        // Values exactly at +/- threshold are NOT encoded as +/-1
        // (v > threshold, not >=; v < -threshold, not <=)
        let packed = encode_ternary(&[0.5, -0.5], 0.5);
        assert_eq!(packed.get(0), 0, "value exactly at +threshold should be 0");
        assert_eq!(packed.get(1), 0, "value exactly at -threshold should be 0");
    }

    #[test]
    fn test_encode_ternary_zero_threshold() {
        // threshold=0: positive -> +1, negative -> -1, zero -> 0
        let packed = encode_ternary(&[1.0, -1.0, 0.0], 0.0);
        assert_eq!(packed.get(0), 1);
        assert_eq!(packed.get(1), -1);
        assert_eq!(packed.get(2), 0);
    }

    // =========================================================================
    // ternary_dot: larger vectors and mixed scenarios
    // =========================================================================

    #[test]
    fn test_ternary_dot_all_zeros() {
        let a = PackedTernary::zeros(100);
        let b = PackedTernary::zeros(100);
        assert_eq!(ternary_dot(&a, &b), 0);
    }

    #[test]
    fn test_ternary_dot_mixed_large() {
        // 64 dimensions (exactly 2 u64 words)
        let mut a = PackedTernary::zeros(64);
        let mut b = PackedTernary::zeros(64);

        // Set some values in both words
        for i in (0..64).step_by(3) {
            a.set(i, 1);
        }
        for i in (0..64).step_by(3) {
            b.set(i, 1);
        }

        // All matching -> dot = count of set positions
        let count = (0..64).step_by(3).count() as i32;
        assert_eq!(ternary_dot(&a, &b), count);
    }

    // =========================================================================
    // asymmetric_dot edge cases
    // =========================================================================

    #[test]
    fn test_asymmetric_dot_zero_query() {
        let mut t = PackedTernary::zeros(3);
        t.set(0, 1);
        t.set(1, -1);

        let query = vec![0.0, 0.0, 0.0];
        assert!(asymmetric_dot(&query, &t).abs() < 1e-9);
    }

    #[test]
    fn test_asymmetric_dot_zero_ternary() {
        let t = PackedTernary::zeros(3);
        let query = vec![1.0, 2.0, 3.0];
        assert!(asymmetric_dot(&query, &t).abs() < 1e-9);
    }

    #[test]
    fn test_asymmetric_dot_negative_query() {
        let mut t = PackedTernary::zeros(2);
        t.set(0, 1);
        t.set(1, -1);

        let query = vec![-3.0, -4.0];
        // -3*1 + -4*(-1) = -3 + 4 = 1
        let result = asymmetric_dot(&query, &t);
        assert!((result - 1.0).abs() < 1e-6);
    }

    // =========================================================================
    // batch_asymmetric_dot
    // =========================================================================

    #[test]
    fn test_batch_asymmetric_dot() {
        let mut t1 = PackedTernary::zeros(2);
        t1.set(0, 1);

        let mut t2 = PackedTernary::zeros(2);
        t2.set(1, -1);

        let query = vec![1.0, 2.0];
        let results = batch_asymmetric_dot(&query, &[t1, t2]);

        // t1: 1*1 + 0*2 = 1
        assert!((results[0] - 1.0).abs() < 1e-6);
        // t2: 0*1 + (-1)*2 = -2
        assert!((results[1] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_batch_asymmetric_dot_empty() {
        let query = vec![1.0, 2.0];
        let results = batch_asymmetric_dot(&query, &[]);
        assert!(results.is_empty());
    }

    // =========================================================================
    // ternary_hamming edge cases
    // =========================================================================

    #[test]
    fn test_hamming_identical() {
        let mut v = PackedTernary::zeros(10);
        v.set(0, 1);
        v.set(3, -1);
        v.set(7, 1);
        assert_eq!(ternary_hamming(&v, &v), 0, "hamming(v, v) should be 0");
    }

    #[test]
    fn test_hamming_all_opposite() {
        let mut a = PackedTernary::zeros(4);
        let mut b = PackedTernary::zeros(4);

        a.set(0, 1);
        a.set(1, -1);
        a.set(2, 1);
        a.set(3, -1);

        b.set(0, -1);
        b.set(1, 1);
        b.set(2, -1);
        b.set(3, 1);

        assert_eq!(ternary_hamming(&a, &b), 4);
    }

    #[test]
    fn test_hamming_zeros_ignored() {
        // Hamming only counts positions where both are non-zero and differ
        let mut a = PackedTernary::zeros(4);
        let mut b = PackedTernary::zeros(4);

        a.set(0, 1);
        // b[0] = 0 -> not counted (both must be non-zero)
        b.set(1, -1);
        // a[1] = 0 -> not counted

        assert_eq!(ternary_hamming(&a, &b), 0);
    }

    // =========================================================================
    // Encode + dot round-trip consistency
    // =========================================================================

    #[test]
    fn test_encode_then_dot_consistency() {
        let values_a: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 10.0).collect();
        let values_b: Vec<f32> = (0..32).map(|i| ((i * 3) as f32 - 48.0) / 10.0).collect();

        let a = encode_ternary(&values_a, 0.5);
        let b = encode_ternary(&values_b, 0.5);

        // Compute expected dot product from decoded values
        let mut expected = 0i32;
        for i in 0..32 {
            expected += (a.get(i) as i32) * (b.get(i) as i32);
        }

        assert_eq!(ternary_dot(&a, &b), expected);
    }
}
