//! SIMD-accelerated binary (1-bit) vector operations.
//!
//! # Binary Quantization
//!
//! Binary vectors use only two values: {0, 1}. This yields 1 bit per dimension,
//! providing the highest possible compression (32x vs f32).
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
/// Values above `threshold` (default 0.0) become 1, others become 0.
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
#[inline]
pub fn binary_dot(a: &PackedBinary, b: &PackedBinary) -> u32 {
    debug_assert_eq!(a.dimension, b.dimension);
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(&wa, &wb)| (wa & wb).count_ones())
        .sum()
}

/// Compute Jaccard similarity: `|A ∩ B| / |A ∪ B|`.
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
}
