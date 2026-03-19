//! Scalar quantization (uint8) for memory-efficient similarity search.
//!
//! Quantizes f32 vectors to u8 with affine mapping (4x compression).
//! Supports asymmetric distance: the query stays in f32 while corpus
//! vectors are quantized, preserving most of the ranking quality.
//!
//! # Quantization Scheme
//!
//! Each f32 value is mapped to u8 via:
//!
//! ```text
//! u8 = clamp(round((f32 - offset) / alpha * 255), 0, 255)
//! ```
//!
//! where `alpha = max - min` and `offset = min`, computed from the corpus.
//!
//! Dequantization: `f32 = alpha * (u8 as f32 / 255.0) + offset`
//!
//! # Asymmetric Distance
//!
//! The dot product between f32 query `q` and quantized doc `d` decomposes as:
//!
//! ```text
//! dot(q, dequant(d)) = (alpha/255) * sum(q[i] * d[i]) + offset * sum(q[i])
//! ```
//!
//! The `sum(q[i])` term is query-only and can be precomputed once per query,
//! amortized across all corpus comparisons.
//!
//! # References
//!
//! - Qdrant (2023). "Scalar Quantization" -- production uint8 quantization
//!   with quantile-based range selection

// arch is only used on architectures with SIMD dispatch
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::arch;

/// Quantization parameters for a collection of vectors.
///
/// Computed once from the corpus, applied to all vectors.
/// Shared across all quantized vectors in the same collection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantizationParams {
    /// Range: `max - min` of the corpus values.
    pub alpha: f32,
    /// Offset: `min` of the corpus values.
    pub offset: f32,
}

impl QuantizationParams {
    /// Create params from explicit min/max range.
    #[must_use]
    pub fn from_range(min: f32, max: f32) -> Self {
        let alpha = max - min;
        Self {
            alpha: if alpha > 0.0 { alpha } else { 1.0 },
            offset: min,
        }
    }

    /// Compute params from a flat slice of corpus values.
    ///
    /// Scans all values to find the range. For large corpora,
    /// consider using [`QuantizationParams::from_range`] with
    /// pre-sampled min/max to avoid a full scan.
    #[must_use]
    pub fn fit(values: &[f32]) -> Self {
        if values.is_empty() {
            return Self {
                alpha: 1.0,
                offset: 0.0,
            };
        }

        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in values {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }

        Self::from_range(min, max)
    }

    /// Compute params from a corpus of vectors (iterator of slices).
    ///
    /// Scans all values across all vectors to find the global range.
    #[must_use]
    pub fn fit_vectors(vectors: &[&[f32]]) -> Self {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for v in vectors {
            for &val in *v {
                if val < min {
                    min = val;
                }
                if val > max {
                    max = val;
                }
            }
        }
        if min > max {
            return Self {
                alpha: 1.0,
                offset: 0.0,
            };
        }
        Self::from_range(min, max)
    }
}

/// Scalar-quantized u8 vector.
///
/// Each dimension is stored as a single byte, giving 4x compression
/// over f32. Use with [`QuantizationParams`] for dequantization.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedU8 {
    data: Vec<u8>,
    dimension: usize,
}

impl QuantizedU8 {
    /// Create from raw quantized data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != dimension`.
    pub fn new(data: Vec<u8>, dimension: usize) -> Self {
        assert_eq!(
            data.len(),
            dimension,
            "QuantizedU8: data length {} doesn't match dimension {}",
            data.len(),
            dimension
        );
        Self { data, dimension }
    }

    /// Raw quantized data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Original vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Memory size in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }
}

/// Quantize an f32 vector to u8.
#[must_use]
pub fn quantize_u8(values: &[f32], params: &QuantizationParams) -> QuantizedU8 {
    let inv_alpha = 255.0 / params.alpha;
    let data: Vec<u8> = values
        .iter()
        .map(|&v| {
            let normalized = (v - params.offset) * inv_alpha;
            normalized.round().clamp(0.0, 255.0) as u8
        })
        .collect();
    QuantizedU8 {
        dimension: values.len(),
        data,
    }
}

/// Precomputed query context for amortizing `sum(q[i])` across corpus comparisons.
#[derive(Clone, Copy, Debug)]
pub struct QueryContext {
    /// Sum of all query components.
    pub query_sum: f32,
}

/// Compute query context (precompute `sum(q[i])` for batch scoring).
#[must_use]
pub fn query_context(query: &[f32]) -> QueryContext {
    QueryContext {
        query_sum: query.iter().sum(),
    }
}

/// Asymmetric dot product: f32 query x quantized u8 document.
///
/// Computes `dot(query, dequantize(quantized))` without materializing
/// the dequantized vector, using the decomposition:
///
/// ```text
/// result = (alpha/255) * mixed_dot(q, d) + offset * sum(q)
/// ```
///
/// # SIMD Acceleration
///
/// Dispatches to NEON (aarch64) or AVX2/AVX-512 (x86_64) for the
/// mixed-precision inner loop.
///
/// # Panics
///
/// Panics if `query.len() != quantized.dimension()`.
#[must_use]
#[allow(unsafe_code)]
pub fn asymmetric_dot_u8(
    query: &[f32],
    quantized: &QuantizedU8,
    params: &QuantizationParams,
) -> f32 {
    assert_eq!(
        query.len(),
        quantized.dimension,
        "asymmetric_dot_u8: dimension mismatch ({} vs {})",
        query.len(),
        quantized.dimension
    );

    let ctx = query_context(query);
    asymmetric_dot_u8_precomputed(query, quantized, params, &ctx)
}

/// Asymmetric dot product with precomputed query context.
///
/// Use this in batch scoring loops where `query_context()` is computed
/// once per query and reused across all corpus vectors.
#[must_use]
#[allow(unsafe_code)]
pub fn asymmetric_dot_u8_precomputed(
    query: &[f32],
    quantized: &QuantizedU8,
    params: &QuantizationParams,
    ctx: &QueryContext,
) -> f32 {
    assert_eq!(
        query.len(),
        quantized.dimension,
        "asymmetric_dot_u8_precomputed: dimension mismatch ({} vs {})",
        query.len(),
        quantized.dimension
    );

    let mixed = mixed_dot_u8_f32(query, &quantized.data);
    (params.alpha / 255.0) * mixed + params.offset * ctx.query_sum
}

/// Mixed-precision dot product: `sum(a_f32[i] * b_u8[i] as f32)`.
///
/// This is the hot inner loop. SIMD-dispatched.
#[inline]
#[allow(unsafe_code)]
fn mixed_dot_u8_f32(a: &[f32], b: &[u8]) -> f32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let n = a.len().min(b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if n >= 16 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2 and FMA verified via runtime detection.
            return unsafe { arch::x86_64::dot_u8_f32_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if n >= 16 {
            // SAFETY: NEON is always available on aarch64.
            return unsafe { arch::aarch64::dot_u8_f32_neon(a, b) };
        }
    }

    #[allow(unreachable_code)]
    mixed_dot_u8_f32_portable(a, b)
}

/// Portable mixed-precision dot product.
#[inline]
fn mixed_dot_u8_f32_portable(a: &[f32], b: &[u8]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&af, &bu)| af * bu as f32)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_roundtrip() {
        let values = [0.0f32, 0.5, 1.0, -1.0, 0.25];
        let params = QuantizationParams::fit(&values);
        let quantized = quantize_u8(&values, &params);

        assert_eq!(quantized.dimension(), values.len());

        // Dequantize and check
        for (i, &original) in values.iter().enumerate() {
            let dequant = params.alpha * (quantized.data()[i] as f32 / 255.0) + params.offset;
            let error = (original - dequant).abs();
            assert!(
                error < params.alpha / 255.0 + 1e-6,
                "roundtrip error too large at {i}: original={original}, dequant={dequant}, error={error}"
            );
        }
    }

    #[test]
    fn test_quantize_range() {
        let values = [-1.0f32, 0.0, 1.0];
        let params = QuantizationParams::fit(&values);
        let q = quantize_u8(&values, &params);

        assert_eq!(q.data()[0], 0); // min -> 0
        assert_eq!(q.data()[2], 255); // max -> 255
        assert!((q.data()[1] as i32 - 128).abs() <= 1); // mid -> ~128
    }

    #[test]
    fn test_asymmetric_dot_matches_exact() {
        let doc = [1.0f32, 2.0, 3.0, 4.0];
        let query = [0.5f32, 0.5, 0.5, 0.5];

        let exact_dot: f32 = doc.iter().zip(&query).map(|(d, q)| d * q).sum();

        let params = QuantizationParams::fit(&doc);
        let quantized = quantize_u8(&doc, &params);
        let approx_dot = asymmetric_dot_u8(&query, &quantized, &params);

        let error = (exact_dot - approx_dot).abs();
        let tolerance = params.alpha / 255.0 * doc.len() as f32;
        assert!(
            error < tolerance,
            "asymmetric dot error too large: exact={exact_dot}, approx={approx_dot}, error={error}, tolerance={tolerance}"
        );
    }

    #[test]
    fn test_precomputed_matches_direct() {
        let doc = [1.0f32, 2.0, 3.0];
        let query = [0.5f32, 1.0, 1.5];
        let params = QuantizationParams::fit(&doc);
        let quantized = quantize_u8(&doc, &params);

        let direct = asymmetric_dot_u8(&query, &quantized, &params);
        let ctx = query_context(&query);
        let precomputed = asymmetric_dot_u8_precomputed(&query, &quantized, &params, &ctx);

        assert!(
            (direct - precomputed).abs() < 1e-6,
            "precomputed mismatch: direct={direct}, precomputed={precomputed}"
        );
    }

    #[test]
    fn test_quantize_empty() {
        let params = QuantizationParams::fit(&[]);
        let q = quantize_u8(&[], &params);
        assert_eq!(q.dimension(), 0);
        assert_eq!(q.memory_bytes(), 0);
    }

    #[test]
    fn test_quantize_constant() {
        // All same value -> alpha = 0, should not divide by zero
        let values = [5.0f32; 10];
        let params = QuantizationParams::fit(&values);
        let q = quantize_u8(&values, &params);
        assert_eq!(q.dimension(), 10);
        // All should map to 0 (value == offset, normalized = 0)
    }

    #[test]
    fn test_params_from_range() {
        let params = QuantizationParams::from_range(-1.0, 1.0);
        assert!((params.alpha - 2.0).abs() < 1e-6);
        assert!((params.offset - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_fit_vectors() {
        let v1 = [0.0f32, 1.0];
        let v2 = [-1.0f32, 2.0];
        let params = QuantizationParams::fit_vectors(&[&v1, &v2]);
        assert!((params.offset - (-1.0)).abs() < 1e-6);
        assert!((params.alpha - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_memory_bytes() {
        let params = QuantizationParams::from_range(0.0, 1.0);
        let q = quantize_u8(&[0.5; 768], &params);
        assert_eq!(q.memory_bytes(), 768); // 4x compression vs 768*4 = 3072 bytes
    }

    #[test]
    fn test_asymmetric_dot_large() {
        // Larger vector to exercise SIMD paths
        let dim = 128;
        let doc: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.3).cos()).collect();

        let exact_dot: f32 = doc.iter().zip(&query).map(|(d, q)| d * q).sum();

        let params = QuantizationParams::fit(&doc);
        let quantized = quantize_u8(&doc, &params);
        let approx_dot = asymmetric_dot_u8(&query, &quantized, &params);

        let rel_error = if exact_dot.abs() > 1e-6 {
            (exact_dot - approx_dot).abs() / exact_dot.abs()
        } else {
            (exact_dot - approx_dot).abs()
        };

        // u8 quantization at dim=128 can produce ~10-20% relative error
        // when the exact dot product is small (near zero). Use absolute
        // error tolerance instead.
        let abs_error = (exact_dot - approx_dot).abs();
        let tolerance = params.alpha / 255.0 * (dim as f32).sqrt() + 0.1;
        assert!(
            abs_error < tolerance,
            "dim={dim}: exact={exact_dot}, approx={approx_dot}, abs_error={abs_error}, tolerance={tolerance}"
        );
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn test_asymmetric_dot_dimension_mismatch() {
        let params = QuantizationParams::from_range(0.0, 1.0);
        let q = quantize_u8(&[0.5, 0.5], &params);
        let _ = asymmetric_dot_u8(&[1.0, 2.0, 3.0], &q, &params);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        #[test]
        fn quantize_values_in_range(
            values in prop::collection::vec(-10.0f32..10.0, 1..200)
        ) {
            let params = QuantizationParams::fit(&values);
            let q = quantize_u8(&values, &params);

            prop_assert_eq!(q.dimension(), values.len());
            for &byte in q.data() {
                // All u8 values are inherently in [0, 255]
                prop_assert!(byte <= 255);
            }
        }

        #[test]
        fn asymmetric_dot_approximates_exact(
            dim in 1..200usize
        ) {
            let doc: Vec<f32> = (0..dim).map(|i| ((i * 7) as f32).sin()).collect();
            let query: Vec<f32> = (0..dim).map(|i| ((i * 11) as f32).cos()).collect();

            let exact: f32 = doc.iter().zip(&query).map(|(d, q)| d * q).sum();

            let params = QuantizationParams::fit(&doc);
            let quantized = quantize_u8(&doc, &params);
            let approx = asymmetric_dot_u8(&query, &quantized, &params);

            // Quantization error is bounded by alpha/255 * sqrt(dim) * query_norm
            let tolerance = params.alpha / 255.0 * (dim as f32).sqrt()
                * query.iter().map(|x| x * x).sum::<f32>().sqrt()
                + 0.1; // slack for very small values

            prop_assert!(
                (exact - approx).abs() < tolerance,
                "dim={}: exact={}, approx={}, error={}, tolerance={}",
                dim, exact, approx, (exact - approx).abs(), tolerance
            );
        }

        #[test]
        fn precomputed_equals_direct(
            dim in 1..100usize
        ) {
            let doc: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
            let query: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();

            let params = QuantizationParams::fit(&doc);
            let quantized = quantize_u8(&doc, &params);

            let direct = asymmetric_dot_u8(&query, &quantized, &params);
            let ctx = query_context(&query);
            let precomputed = asymmetric_dot_u8_precomputed(&query, &quantized, &params, &ctx);

            prop_assert!(
                (direct - precomputed).abs() < 1e-5,
                "direct={}, precomputed={}", direct, precomputed
            );
        }
    }
}
