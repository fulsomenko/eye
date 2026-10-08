//! Direct least-squares ellipse fitting (Fitzgibbon, Pilu, Fisher 1999), in the numerically
//! stable form of Halir and Flusser 1998.

use eye_core::Ellipse2;
use nalgebra::{Matrix2, Matrix3, Point2, Vector2, Vector3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EllipseFit {
    pub ellipse: Ellipse2,
    pub rms_residual: f64,
}

fn null_vector(m: &Matrix3<f64>) -> Vector3<f64> {
    let rows = [
        m.row(0).transpose(),
        m.row(1).transpose(),
        m.row(2).transpose(),
    ];
    [(0, 1), (0, 2), (1, 2)]
        .into_iter()
        .map(|(i, j)| rows[i].cross(&rows[j]))
        .fold(Vector3::zeros(), |best, v| {
            if v.norm_squared() > best.norm_squared() {
                v
            } else {
                best
            }
        })
}

/// Direct least-squares ellipse fit (Fitzgibbon, Pilu, Fisher 1999) in the numerically
/// stable form of Halir and Flusser 1998. Needs >= 6 points; `None` if degenerate.
pub fn fit_ellipse(points: &[Point2<f64>]) -> Option<EllipseFit> {
    if points.len() < 6 {
        return None;
    }
    let n = points.len() as f64;

    let mean = points
        .iter()
        .fold(Vector2::zeros(), |acc, p| acc + p.coords)
        / n;
    let mean_point = Point2::from(mean);

    let mean_sq_dist = points
        .iter()
        .map(|p| (p - mean_point).norm_squared())
        .sum::<f64>()
        / n;
    let s = mean_sq_dist.sqrt();
    if !(s.is_finite() && s > 0.0) {
        return None;
    }

    let normalized: Vec<Point2<f64>> = points
        .iter()
        .map(|p| Point2::from((p - mean_point) / s))
        .collect();

    let mut s1 = Matrix3::<f64>::zeros();
    let mut s2 = Matrix3::<f64>::zeros();
    let mut s3 = Matrix3::<f64>::zeros();
    for p in &normalized {
        let (x, y) = (p.x, p.y);
        let d1 = Vector3::new(x * x, x * y, y * y);
        let d2 = Vector3::new(x, y, 1.0);
        s1 += d1 * d1.transpose();
        s2 += d1 * d2.transpose();
        s3 += d2 * d2.transpose();
    }

    let s3_inv = s3.try_inverse()?;
    let t = -(s3_inv * s2.transpose());
    let c1_inv = Matrix3::new(0.0, 0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0, 0.0);
    let m = c1_inv * (s1 + s2 * t);
    // `Matrix::eigenvalues` bails out whenever the Schur form leaves any 2x2 block
    // undecoupled, even one with a real (possibly repeated) root, which this matrix often
    // has. `complex_eigenvalues` always succeeds; keep only the roots whose imaginary part
    // is numerical noise.
    let a1 = m
        .complex_eigenvalues()
        .iter()
        .filter(|lambda| lambda.im.abs() <= 1e-9 * lambda.re.abs().max(1.0))
        .map(|lambda| null_vector(&(m - Matrix3::identity() * lambda.re)))
        .find(|v| 4.0 * v[0] * v[2] - v[1] * v[1] > 0.0)?;
    let a2 = t * a1;
    let (a, b, c, d, e, f) = (a1[0], a1[1], a1[2], a2[0], a2[1], a2[2]);
    let am = Matrix2::new(a, b / 2.0, b / 2.0, c);
    let p0 = -0.5 * am.try_inverse()? * Vector2::new(d, e);
    let f0 = a * p0.x * p0.x + b * p0.x * p0.y + c * p0.y * p0.y + d * p0.x + e * p0.y + f;
    let eig = am.symmetric_eigen();
    let (q0, q1) = (-f0 / eig.eigenvalues[0], -f0 / eig.eigenvalues[1]);
    if q0 <= 0.0 || q1 <= 0.0 {
        return None;
    }
    let v0 = eig.eigenvectors.column(0);
    let ellipse = Ellipse2::new(
        Point2::from(p0 * s + mean),
        q0.sqrt() * s,
        q1.sqrt() * s,
        v0[1].atan2(v0[0]),
    )
    .ok()?;

    let residuals_sq: f64 = normalized
        .iter()
        .map(|p| {
            let (x, y) = (p.x, p.y);
            let fval = a * x * x + b * x * y + c * y * y + d * x + e * y + f;
            let gx = 2.0 * a * x + b * y + d;
            let gy = b * x + 2.0 * c * y + e;
            let grad_norm = (gx * gx + gy * gy).sqrt();
            let sampson = if grad_norm > 0.0 {
                fval / grad_norm
            } else {
                0.0
            } * s;
            sampson * sampson
        })
        .sum();
    let rms_residual = (residuals_sq / n).sqrt();

    Some(EllipseFit {
        ellipse,
        rms_residual,
    })
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    use super::*;

    fn ellipse_points(
        center: Point2<f64>,
        semi_major: f64,
        semi_minor: f64,
        angle: f64,
        n: usize,
    ) -> Vec<Point2<f64>> {
        (0..n)
            .map(|i| {
                let t = 2.0 * PI * i as f64 / n as f64;
                let (x, y) = (semi_major * t.cos(), semi_minor * t.sin());
                let (ca, sa) = (angle.cos(), angle.sin());
                Point2::new(center.x + x * ca - y * sa, center.y + x * sa + y * ca)
            })
            .collect()
    }

    #[test]
    fn test_fit_exact_ellipse_recovers_parameters() {
        let center = Point2::new(3.2, -1.1);
        let points = ellipse_points(center, 5.0, 3.0, 0.4, 20);
        let fit = fit_ellipse(&points).expect("fit should succeed");
        assert_abs_diff_eq!(fit.ellipse.center().x, center.x, epsilon = 1e-9);
        assert_abs_diff_eq!(fit.ellipse.center().y, center.y, epsilon = 1e-9);
        assert_abs_diff_eq!(fit.ellipse.semi_major(), 5.0, epsilon = 1e-9);
        assert_abs_diff_eq!(fit.ellipse.semi_minor(), 3.0, epsilon = 1e-9);
        assert_abs_diff_eq!(fit.ellipse.angle(), 0.4, epsilon = 1e-9);
        assert!(fit.rms_residual < 1e-9);
    }

    #[test]
    fn test_fit_circle_reports_equal_axes() {
        let center = Point2::new(290.3, 180.7);
        let points = ellipse_points(center, 3.0, 3.0, 0.0, 16);
        let fit = fit_ellipse(&points).expect("fit should succeed");
        assert_abs_diff_eq!(
            fit.ellipse.semi_major(),
            fit.ellipse.semi_minor(),
            epsilon = 1e-9
        );
        assert_eq!(fit.ellipse.angle(), 0.0);
    }

    #[test]
    fn test_fit_collinear_points_returns_none() {
        let points: Vec<Point2<f64>> = (0..6)
            .map(|i| Point2::new(i as f64, 2.0 * i as f64))
            .collect();
        assert!(fit_ellipse(&points).is_none());
    }

    #[test]
    fn test_fit_five_points_returns_none() {
        let points = ellipse_points(Point2::new(0.0, 0.0), 3.0, 2.0, 0.0, 5);
        assert!(fit_ellipse(&points).is_none());
    }

    proptest! {
        #[test]
        fn prop_fit_recovers_random_ellipses(
            cx in -50.0f64..50.0,
            cy in -50.0f64..50.0,
            major in 1.0f64..20.0,
            ratio in 0.3f64..1.0,
            angle in -std::f64::consts::FRAC_PI_2..std::f64::consts::FRAC_PI_2,
            n in 12usize..40,
        ) {
            let minor = major * ratio;
            let center = Point2::new(cx, cy);
            let points = ellipse_points(center, major, minor, angle, n);
            let fit = fit_ellipse(&points);
            prop_assert!(fit.is_some());
            let fit = fit.unwrap();
            prop_assert!((fit.ellipse.center().x - cx).abs() < 1e-6);
            prop_assert!((fit.ellipse.center().y - cy).abs() < 1e-6);
            prop_assert!((fit.ellipse.semi_major() - major).abs() < 1e-6);
            prop_assert!((fit.ellipse.semi_minor() - minor).abs() < 1e-6);
            if ratio <= 0.99 {
                let diff = (fit.ellipse.angle() - angle).rem_euclid(PI);
                let diff = diff.min(PI - diff);
                prop_assert!(diff < 1e-6);
            }
        }
    }
}
