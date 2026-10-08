//! 95%-confidence error ellipses, in logical px, from a gaze point's covariance.

use eye_core::ScreenModel;
use nalgebra::{Matrix2, Point2, Vector2};

/// `k` for a 95% confidence region of a 2-dof chi-square distribution:
/// `sqrt(-2 * ln(0.05))`.
pub const K95: f64 = 2.447_746_830_680_816;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceEllipse {
    pub center: Point2<f64>,
    pub semi_axes: (f64, f64),
    pub angle: f64,
}

/// Logical px per mm along x and y, from the rig's screen model.
pub fn logical_px_per_mm(screen: &ScreenModel) -> Vector2<f64> {
    Vector2::new(
        f64::from(screen.size_px.0) / screen.scale / screen.size_mm.x,
        f64::from(screen.size_px.1) / screen.scale / screen.size_mm.y,
    )
}

pub fn cov_mm_to_logical_px(cov_mm: &Matrix2<f64>, px_per_mm: &Vector2<f64>) -> Matrix2<f64> {
    let d = Matrix2::from_diagonal(px_per_mm);
    d * cov_mm * d
}

/// `None` if `cov` has a non-finite entry or a negative trace.
pub fn confidence_ellipse(
    center: Point2<f64>,
    cov_px: &Matrix2<f64>,
    k: f64,
    max_axis: f64,
) -> Option<ConfidenceEllipse> {
    let (a, b, c) = (cov_px[(0, 0)], cov_px[(0, 1)], cov_px[(1, 1)]);
    if !(a.is_finite() && b.is_finite() && c.is_finite()) || a + c < 0.0 {
        return None;
    }
    let m = 0.5 * (a + c);
    let d = (0.25 * (a - c).powi(2) + b * b).sqrt();
    let (l1, l2) = (m + d, (m - d).max(0.0));
    let axis = |l: f64| (k * l.sqrt()).clamp(0.5, max_axis);
    Some(ConfidenceEllipse {
        center,
        semi_axes: (axis(l1), axis(l2)),
        angle: 0.5 * (2.0 * b).atan2(a - c),
    })
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_4;

    use approx::assert_relative_eq;
    use eye_core::OutputId;
    use nalgebra::{Rotation2, Vector2};
    use proptest::prelude::*;

    use super::*;

    fn edp1() -> ScreenModel {
        ScreenModel {
            output: OutputId::from("eDP-1"),
            size_mm: Vector2::new(310.0, 170.0),
            size_px: (3840, 2160),
            scale: 2.0,
        }
    }

    #[test]
    fn test_logical_px_per_mm_edp1() {
        let px_per_mm = logical_px_per_mm(&edp1());
        assert_relative_eq!(px_per_mm.x, 1920.0 / 310.0, epsilon = 1e-12);
        assert_relative_eq!(px_per_mm.y, 1080.0 / 170.0, epsilon = 1e-12);
    }

    #[test]
    fn test_cov_mm_to_logical_px_is_s_c_s() {
        let cov = Matrix2::new(1.0, 0.5, 0.5, 2.0);
        let s = Vector2::new(2.0, 3.0);
        let result = cov_mm_to_logical_px(&cov, &s);
        assert_relative_eq!(result, Matrix2::new(4.0, 3.0, 3.0, 18.0), epsilon = 1e-12);
    }

    #[test]
    fn test_confidence_ellipse_axis_aligned() {
        let cov = Matrix2::new(4.0, 0.0, 0.0, 1.0);
        let e = confidence_ellipse(Point2::new(0.0, 0.0), &cov, K95, 1e6).expect("finite");
        assert_relative_eq!(e.semi_axes.0, 2.0 * K95, epsilon = 1e-12);
        assert_relative_eq!(e.semi_axes.1, K95, epsilon = 1e-12);
        assert_relative_eq!(e.angle, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_confidence_ellipse_rotated_45_degrees() {
        let cov = Matrix2::new(5.0, 4.0, 4.0, 5.0);
        let e = confidence_ellipse(Point2::new(0.0, 0.0), &cov, K95, 1e6).expect("finite");
        assert_relative_eq!(e.semi_axes.0, 3.0 * K95, epsilon = 1e-9);
        assert_relative_eq!(e.semi_axes.1, K95, epsilon = 1e-9);
        assert_relative_eq!(e.angle, FRAC_PI_4, epsilon = 1e-12);
    }

    #[test]
    fn test_confidence_ellipse_isotropic_angle_zero() {
        let cov = Matrix2::new(3.0, 0.0, 0.0, 3.0);
        let e = confidence_ellipse(Point2::new(0.0, 0.0), &cov, K95, 1e6).expect("finite");
        assert_relative_eq!(e.semi_axes.0, e.semi_axes.1, epsilon = 1e-12);
        assert_relative_eq!(e.angle, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_confidence_ellipse_rejects_nonfinite() {
        let cov = Matrix2::new(f64::NAN, 0.0, 0.0, 1.0);
        assert_eq!(
            confidence_ellipse(Point2::new(0.0, 0.0), &cov, K95, 1e6),
            None
        );
    }

    #[test]
    fn test_confidence_ellipse_clamps_degenerate_minor_axis() {
        let cov = Matrix2::new(4.0, 0.0, 0.0, -1e-12);
        let e = confidence_ellipse(Point2::new(0.0, 0.0), &cov, K95, 1e6).expect("finite");
        assert_relative_eq!(e.semi_axes.1, 0.5, epsilon = 1e-12);
    }

    proptest! {
        #[test]
        fn test_confidence_ellipse_reconstructs_covariance(
            theta in -std::f64::consts::FRAC_PI_2..std::f64::consts::FRAC_PI_2,
            la in 0.05..1e3f64,
            lb in 0.05..1e3f64,
        ) {
            let (l1, l2) = (la.max(lb), la.min(lb));
            let r = Rotation2::new(theta);
            let d = Matrix2::new(l1, 0.0, 0.0, l2);
            let cov = r.matrix() * d * r.matrix().transpose();

            let max_axis = 1e6;
            let e = confidence_ellipse(Point2::new(0.0, 0.0), &cov, K95, max_axis).expect("finite");

            let r2 = Rotation2::new(e.angle);
            let g = Matrix2::new((e.semi_axes.0 / K95).powi(2), 0.0, 0.0, (e.semi_axes.1 / K95).powi(2));
            let reconstructed = r2.matrix() * g * r2.matrix().transpose();

            let norm = cov.norm();
            for i in 0..2 {
                for j in 0..2 {
                    prop_assert!(
                        (reconstructed[(i, j)] - cov[(i, j)]).abs()
                            <= 1e-9 * norm.max(1.0) + 1e-9 * cov[(i, j)].abs()
                    );
                }
            }
        }
    }
}
