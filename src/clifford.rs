//! Clifford Algebra (Geometric Algebra) for steerable embeddings.
//!
//! # Rotors
//!
//! Geometric algebra generalizes complex numbers and quaternions to arbitrary
//! dimensions. A **Rotor** represents a rotation in a plane defined by a bivector.
//!
//! # References
//!
//! - Ruhe, Brandstetter, Forre (2023, NeurIPS), "Clifford Group Equivariant Neural
//!   Networks" -- rotors as core primitives for O(n)/E(n)-equivariant architectures
//! - Kamdem Teyou et al. (2024), "Embedding Knowledge Graphs in Degenerate Clifford
//!   Algebras" -- Clifford algebras for KG embeddings, showing gains over quaternions
//!
//! ## Geometric Product
//!
//! The fundamental operation is the geometric product:
//! `ab = a · b + a ∧ b`
//! where `a · b` is the symmetric inner product (scalar) and `a ∧ b` is the
//! antisymmetric outer product (bivector).
//!
//! ## Rotors
//!
//! A rotor in the plane `B` is defined as:
//! `R = exp(θB/2) = cos(θ/2) + B sin(θ/2)`
//!
//! Rotors act on vectors via the sandwich product: `v' = R v R†`.

use crate::dot;

/// A simple Rotor in a 2D plane (even subalgebra of Cl(2)).
/// Equivalent to a complex number or a 2D rotation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rotor2D {
    /// Scalar part: cos(theta/2)
    pub s: f32,
    /// Bivector part (e12): sin(theta/2)
    pub b: f32,
}

impl Rotor2D {
    /// Create rotor from angle theta.
    pub fn from_angle(theta: f32) -> Self {
        let (sin, cos) = (theta / 2.0).sin_cos();
        Self { s: cos, b: sin }
    }

    /// Rotate a 2D vector.
    pub fn rotate(&self, v: [f32; 2]) -> [f32; 2] {
        // v' = R v R*
        // For 2D, this simplifies to standard 2D rotation matrix multiplication.
        // But we implement it via the geometric product for rigor.
        let x = v[0];
        let y = v[1];

        // R = s + b*e12
        // R* = s - b*e12
        // v' = (s + b*e12) (x*e1 + y*e2) (s - b*e12)
        // ... simplifies to:
        let cos_theta = self.s * self.s - self.b * self.b;
        let sin_theta = 2.0 * self.s * self.b;

        [x * cos_theta - y * sin_theta, x * sin_theta + y * cos_theta]
    }
}

/// Compute the bivector part of the geometric product (the wedge product).
///
/// For 2D: `a ∧ b = (a1*b2 - a2*b1) * e12`.
#[inline]
pub fn wedge_2d(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[1] - a[1] * b[0]
}

/// Full geometric product of two 2D vectors.
///
/// Result is a Multivector with scalar and bivector parts.
#[inline]
pub fn geometric_product_2d(a: [f32; 2], b: [f32; 2]) -> (f32, f32) {
    (dot(&a, &b), wedge_2d(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotor_2d() {
        let r = Rotor2D::from_angle(std::f32::consts::FRAC_PI_2); // 90 degrees
        let v = [1.0, 0.0];
        let v_rot = r.rotate(v);

        assert!((v_rot[0] - 0.0).abs() < 1e-6);
        assert!((v_rot[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_geometric_product_2d() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let (s, b_part) = geometric_product_2d(a, b);

        assert_eq!(s, 0.0); // Orthogonal
        assert_eq!(b_part, 1.0); // Full wedge
    }

    // =========================================================================
    // Rotor2D: more rotation angles
    // =========================================================================

    #[test]
    fn test_rotor_identity() {
        // theta = 0 -> no rotation
        let r = Rotor2D::from_angle(0.0);
        let v = [3.0, 7.0];
        let v_rot = r.rotate(v);
        assert!((v_rot[0] - 3.0).abs() < 1e-6);
        assert!((v_rot[1] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_rotor_180() {
        // 180 degrees: (x, y) -> (-x, -y)
        let r = Rotor2D::from_angle(std::f32::consts::PI);
        let v = [1.0, 0.0];
        let v_rot = r.rotate(v);
        assert!((v_rot[0] - (-1.0)).abs() < 1e-5);
        assert!(v_rot[1].abs() < 1e-5);
    }

    #[test]
    fn test_rotor_360() {
        // Full rotation should return to original
        let r = Rotor2D::from_angle(2.0 * std::f32::consts::PI);
        let v = [2.0, 5.0];
        let v_rot = r.rotate(v);
        assert!((v_rot[0] - 2.0).abs() < 1e-4);
        assert!((v_rot[1] - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_rotor_negative_angle() {
        // -90 degrees: (1, 0) -> (0, -1)
        let r = Rotor2D::from_angle(-std::f32::consts::FRAC_PI_2);
        let v = [1.0, 0.0];
        let v_rot = r.rotate(v);
        assert!(v_rot[0].abs() < 1e-6);
        assert!((v_rot[1] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_rotor_preserves_magnitude() {
        let r = Rotor2D::from_angle(1.23); // arbitrary angle
        let v = [3.0, 4.0];
        let v_rot = r.rotate(v);
        let mag_orig = (v[0] * v[0] + v[1] * v[1]).sqrt();
        let mag_rot = (v_rot[0] * v_rot[0] + v_rot[1] * v_rot[1]).sqrt();
        assert!(
            (mag_orig - mag_rot).abs() < 1e-5,
            "rotation should preserve magnitude: {mag_orig} vs {mag_rot}"
        );
    }

    #[test]
    fn test_rotor_composition() {
        // Two 45-degree rotations = one 90-degree rotation
        let r45 = Rotor2D::from_angle(std::f32::consts::FRAC_PI_4);
        let r90 = Rotor2D::from_angle(std::f32::consts::FRAC_PI_2);

        let v = [1.0, 0.0];
        let composed = r45.rotate(r45.rotate(v));
        let direct = r90.rotate(v);

        assert!((composed[0] - direct[0]).abs() < 1e-5);
        assert!((composed[1] - direct[1]).abs() < 1e-5);
    }

    #[test]
    fn test_rotor_rotate_y_axis() {
        // 90 degrees: (0, 1) -> (-1, 0)
        let r = Rotor2D::from_angle(std::f32::consts::FRAC_PI_2);
        let v = [0.0, 1.0];
        let v_rot = r.rotate(v);
        assert!((v_rot[0] - (-1.0)).abs() < 1e-6);
        assert!(v_rot[1].abs() < 1e-6);
    }

    // =========================================================================
    // Wedge product properties
    // =========================================================================

    #[test]
    fn test_wedge_antisymmetric() {
        let a = [3.0, 7.0];
        let b = [2.0, 5.0];
        let ab = wedge_2d(a, b);
        let ba = wedge_2d(b, a);
        assert!((ab + ba).abs() < 1e-6, "wedge should be antisymmetric");
    }

    #[test]
    fn test_wedge_self_is_zero() {
        let a = [3.0, 7.0];
        assert!(wedge_2d(a, a).abs() < 1e-6, "a ^ a should be 0");
    }

    #[test]
    fn test_wedge_parallel_is_zero() {
        let a = [1.0, 2.0];
        let b = [2.0, 4.0]; // parallel to a
        assert!(wedge_2d(a, b).abs() < 1e-6, "parallel vectors have zero wedge");
    }

    #[test]
    fn test_wedge_unit_basis() {
        let e1 = [1.0, 0.0];
        let e2 = [0.0, 1.0];
        assert!((wedge_2d(e1, e2) - 1.0).abs() < 1e-6, "e1 ^ e2 = 1");
    }

    // =========================================================================
    // Geometric product properties
    // =========================================================================

    #[test]
    fn test_geometric_product_parallel() {
        // Parallel vectors: wedge = 0, scalar = dot
        let a = [1.0, 0.0];
        let b = [3.0, 0.0];
        let (s, w) = geometric_product_2d(a, b);
        assert!((s - 3.0).abs() < 1e-6);
        assert!(w.abs() < 1e-6);
    }

    #[test]
    fn test_geometric_product_self() {
        // a * a = |a|^2 (pure scalar)
        let a = [3.0, 4.0];
        let (s, w) = geometric_product_2d(a, a);
        assert!((s - 25.0).abs() < 1e-6, "a*a scalar should be |a|^2");
        assert!(w.abs() < 1e-6, "a*a wedge should be 0");
    }
}
