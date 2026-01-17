//! Clifford Algebra (Geometric Algebra) for steerable embeddings.
//!
//! # 2026 Rotors
//!
//! Geometric algebra generalizes complex numbers and quaternions to arbitrary
//! dimensions. A **Rotor** represents a rotation in a plane defined by a bivector.
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
        
        [
            x * cos_theta - y * sin_theta,
            x * sin_theta + y * cos_theta
        ]
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
        
        assert_eq!(s, 0.0);      // Orthogonal
        assert_eq!(b_part, 1.0); // Full wedge
    }
}
