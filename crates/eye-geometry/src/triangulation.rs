//! Two-view triangulation of a physical point with its 3D covariance.
//!
//! Hartley & Zisserman, *Multiple View Geometry*, 2nd ed., §12.1 (midpoint) and §4.2
//! (reprojection-error minimisation).

use eye_core::{CameraModel, Measured};
use nalgebra::{DVector, Point2, Point3, Unit, Vector3};

use crate::GeometryError;
use crate::camera::Intrinsics;
use crate::lsq::{self, ResidualModel, SolveOptions};
use crate::uncertainty::Cov3;

pub const MIN_PARALLAX_RAD: f64 = 0.005;

#[derive(Debug, Clone, Copy)]
pub struct View<'a> {
    pub camera: &'a CameraModel,
    pub pixel: Measured<Point2<f64>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangulated {
    pub point: Point3<f64>,
    pub cov: Cov3,
    /// Unwhitened reprojection RMS per view: `sqrt(Σ_k |u_k − û_k|² / 2)`.
    pub rms_px: f64,
    pub parallax_rad: f64,
}

fn parallax_rad(d1: &Vector3<f64>, d2: &Vector3<f64>) -> f64 {
    d1.cross(d2).norm().atan2(d1.dot(d2))
}

/// Midpoint of the common perpendicular of two rays; `None` when (near) parallel.
pub fn midpoint(
    o1: &Point3<f64>,
    d1: &Unit<Vector3<f64>>,
    o2: &Point3<f64>,
    d2: &Unit<Vector3<f64>>,
) -> Option<Point3<f64>> {
    let w = o1 - o2;
    let b = d1.dot(d2);
    let denom = 1.0 - b * b;
    if denom < 1e-12 {
        return None;
    }
    let (d, e) = (d1.dot(&w), d2.dot(&w));
    let s = (b * e - d) / denom;
    let t = (e - b * d) / denom;
    Some(Point3::from(
        ((o1 + d1.into_inner() * s).coords + (o2 + d2.into_inner() * t).coords) * 0.5,
    ))
}

struct TwoView<'a> {
    views: [(&'a CameraModel, Intrinsics, Measured<Point2<f64>>); 2],
}

impl ResidualModel for TwoView<'_> {
    fn num_residuals(&self) -> usize {
        4
    }

    fn residuals(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let p = Point3::new(x[0], x[1], x[2]);
        let mut r = DVector::zeros(4);
        for (k, (cam, intr, m)) in self.views.iter().enumerate() {
            let u = intr
                .project(&cam.screen_from_camera.inverse_transform_point(&p))
                .ok()?;
            r[2 * k] = (u.x - m.value().x) / m.sigma();
            r[2 * k + 1] = (u.y - m.value().y) / m.sigma();
        }
        Some(r)
    }
}

pub fn triangulate(a: &View<'_>, b: &View<'_>) -> Result<Triangulated, GeometryError> {
    if a.pixel.sigma() <= 0.0 || b.pixel.sigma() <= 0.0 {
        return Err(GeometryError::Degenerate("zero sigma"));
    }

    let (o1, d1) = crate::camera::pixel_ray(a.camera, a.pixel.value())?;
    let (o2, d2) = crate::camera::pixel_ray(b.camera, b.pixel.value())?;

    let parallax_rad = parallax_rad(&d1, &d2);
    if parallax_rad < MIN_PARALLAX_RAD {
        return Err(GeometryError::Degenerate("parallax"));
    }
    let x0 = midpoint(&o1, &d1, &o2, &d2).ok_or(GeometryError::Degenerate("parallax"))?;

    if a.camera.screen_from_camera.inverse_transform_point(&x0).z <= 0.0
        || b.camera.screen_from_camera.inverse_transform_point(&x0).z <= 0.0
    {
        return Err(GeometryError::BehindCamera);
    }

    let problem = TwoView {
        views: [
            (a.camera, Intrinsics::from_camera_model(a.camera), a.pixel),
            (b.camera, Intrinsics::from_camera_model(b.camera), b.pixel),
        ],
    };
    let fit = lsq::solve(
        &problem,
        DVector::from_vec(vec![x0.x, x0.y, x0.z]),
        &SolveOptions {
            inflate_by_chi2: false,
            ..Default::default()
        },
    )?;
    let point = Point3::new(fit.x[0], fit.x[1], fit.x[2]);
    let cov = Cov3::from_iterator(fit.cov.iter().copied());

    let mut sq_sum = 0.0;
    for (cam, intr, m) in &problem.views {
        let u = intr.project(&cam.screen_from_camera.inverse_transform_point(&point))?;
        sq_sum += (u - m.value()).norm_squared();
    }
    let rms_px: f64 = (sq_sum / 2.0).sqrt();

    Ok(Triangulated {
        point,
        cov,
        rms_px,
        parallax_rad,
    })
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use approx::assert_abs_diff_eq;
    use eye_core::CameraId;
    use nalgebra::{Isometry3, Matrix3, Translation3, UnitQuaternion, Vector3};

    use super::*;
    use crate::synth::SplitMix64;

    #[test]
    fn test_parallax_atan2_is_exact_at_1e_8_rad() {
        let theta = 1e-8_f64;
        let d1 = Vector3::new(0.0, 0.0, 1.0);
        let d2 = Vector3::new(theta.sin(), 0.0, theta.cos());
        let p = parallax_rad(&d1, &d2);
        assert_abs_diff_eq!(p, theta, epsilon = 1e-12);
    }

    #[test]
    fn test_midpoint_intersecting_rays_returns_intersection() {
        let o1 = Point3::new(0.0, 0.0, 0.0);
        let o2 = Point3::new(10.0, 0.0, 0.0);
        let target = Point3::new(5.0, 3.0, -400.0);
        let d1 = Unit::new_normalize(target - o1);
        let d2 = Unit::new_normalize(target - o2);
        let m = midpoint(&o1, &d1, &o2, &d2).unwrap();
        assert_abs_diff_eq!(m.x, target.x, epsilon = 1e-9);
        assert_abs_diff_eq!(m.y, target.y, epsilon = 1e-9);
        assert_abs_diff_eq!(m.z, target.z, epsilon = 1e-9);
    }

    #[test]
    fn test_midpoint_skew_rays_returns_centre_of_common_perpendicular() {
        let o1 = Point3::new(0.0, 0.0, 0.0);
        let d1 = Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0));
        let o2 = Point3::new(0.0, 0.0, 2.0);
        let d2 = Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0));
        let m = midpoint(&o1, &d1, &o2, &d2).unwrap();
        assert_abs_diff_eq!(m.x, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(m.y, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(m.z, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_midpoint_parallel_rays_returns_none() {
        let o1 = Point3::new(0.0, 0.0, 0.0);
        let o2 = Point3::new(1.0, 0.0, 0.0);
        let d = Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0));
        assert_eq!(midpoint(&o1, &d, &o2, &d), None);
    }

    fn cam(intr: Intrinsics, at: Point3<f64>, id: &str) -> CameraModel {
        let mut model = CameraModel {
            id: CameraId::new(id),
            width: intr.width,
            height: intr.height,
            fx: intr.fx,
            fy: intr.fy,
            cx: intr.cx,
            cy: intr.cy,
            distortion: [0.0; 5],
            screen_from_camera: Isometry3::from_parts(
                Translation3::from(at),
                UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI),
            ),
        };
        intr.apply_to(&mut model);
        model
    }

    fn ir_intr() -> Intrinsics {
        Intrinsics {
            width: 640,
            height: 360,
            fx: 457.0,
            fy: 457.0,
            cx: 320.0,
            cy: 180.0,
            distortion: crate::camera::Distortion {
                k1: 0.08,
                k2: -0.15,
                p1: 0.001,
                p2: -0.0005,
                k3: 0.05,
            },
        }
    }

    fn rgb_intr() -> Intrinsics {
        Intrinsics {
            width: 1280,
            height: 720,
            fx: 914.0,
            fy: 914.0,
            cx: 640.0,
            cy: 360.0,
            distortion: crate::camera::Distortion {
                k1: 0.05,
                ..Default::default()
            },
        }
    }

    fn stereo_rig() -> (CameraModel, CameraModel) {
        let ir = cam(ir_intr(), Point3::new(167.5, -7.0, 0.0), "ir");
        let rgb = cam(rgb_intr(), Point3::new(142.5, -7.0, 0.0), "rgb");
        (ir, rgb)
    }

    fn pixel_of(c: &CameraModel, x: &Point3<f64>) -> Point2<f64> {
        Intrinsics::from_camera_model(c)
            .project(&c.screen_from_camera.inverse_transform_point(x))
            .unwrap()
    }

    #[test]
    fn test_triangulate_noise_free_recovers_point() {
        let (ir, rgb) = stereo_rig();
        let x = Point3::new(155.0, 40.0, -500.0);
        let a = View {
            camera: &rgb,
            pixel: Measured::new(pixel_of(&rgb, &x), 0.5).unwrap(),
        };
        let b = View {
            camera: &ir,
            pixel: Measured::new(pixel_of(&ir, &x), 0.5).unwrap(),
        };
        let t = triangulate(&a, &b).unwrap();
        assert_abs_diff_eq!((t.point - x).norm(), 0.0, epsilon = 1e-8);
        assert!(t.rms_px < 1e-8, "rms_px = {}", t.rms_px);
    }

    #[test]
    fn test_triangulate_cov_matches_monte_carlo() {
        let (ir, rgb) = stereo_rig();
        let x = Point3::new(155.0, 40.0, -500.0);
        let sigma = 0.5;
        let rgb_px = pixel_of(&rgb, &x);
        let ir_px = pixel_of(&ir, &x);

        let a = View {
            camera: &rgb,
            pixel: Measured::new(rgb_px, sigma).unwrap(),
        };
        let b = View {
            camera: &ir,
            pixel: Measured::new(ir_px, sigma).unwrap(),
        };
        let predicted = triangulate(&a, &b).unwrap();

        let depth_sigma = predicted.cov[(2, 2)].sqrt();
        assert!(
            (9.0..=13.0).contains(&depth_sigma),
            "depth sigma = {depth_sigma}"
        );

        let n = 5_000;
        let mut rng = SplitMix64::new(5);
        let mut sum = Vector3::zeros();
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let noisy_rgb = Point2::new(
                rgb_px.x + sigma * rng.gaussian(),
                rgb_px.y + sigma * rng.gaussian(),
            );
            let noisy_ir = Point2::new(
                ir_px.x + sigma * rng.gaussian(),
                ir_px.y + sigma * rng.gaussian(),
            );
            let a = View {
                camera: &rgb,
                pixel: Measured::new(noisy_rgb, sigma).unwrap(),
            };
            let b = View {
                camera: &ir,
                pixel: Measured::new(noisy_ir, sigma).unwrap(),
            };
            let t = triangulate(&a, &b).unwrap();
            let v = t.point.coords;
            sum += v;
            samples.push(v);
        }
        let mean = sum / n as f64;
        let mut empirical = Matrix3::zeros();
        for v in &samples {
            let d = v - mean;
            empirical += d * d.transpose();
        }
        empirical /= (n - 1) as f64;

        for k in 0..3 {
            let tol = 0.10 * predicted.cov[(k, k)].abs();
            assert!(
                (empirical[(k, k)] - predicted.cov[(k, k)]).abs() <= tol,
                "diag {k}: empirical {} predicted {}",
                empirical[(k, k)],
                predicted.cov[(k, k)]
            );
        }
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    continue;
                }
                let tol = 0.05 * (predicted.cov[(i, i)] * predicted.cov[(j, j)]).sqrt();
                assert!(
                    (empirical[(i, j)] - predicted.cov[(i, j)]).abs() <= tol,
                    "offdiag ({i},{j}): empirical {} predicted {}",
                    empirical[(i, j)],
                    predicted.cov[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_triangulate_small_baseline_is_degenerate() {
        let ir = cam(ir_intr(), Point3::new(143.5, -7.0, 0.0), "ir");
        let rgb = cam(rgb_intr(), Point3::new(142.5, -7.0, 0.0), "rgb");
        let x = Point3::new(155.0, 40.0, -800.0);
        let a = View {
            camera: &rgb,
            pixel: Measured::new(pixel_of(&rgb, &x), 0.5).unwrap(),
        };
        let b = View {
            camera: &ir,
            pixel: Measured::new(pixel_of(&ir, &x), 0.5).unwrap(),
        };
        assert_eq!(
            triangulate(&a, &b),
            Err(GeometryError::Degenerate("parallax"))
        );
    }

    #[test]
    fn test_triangulate_point_behind_cameras_errors() {
        let (ir, rgb) = stereo_rig();
        let x = Point3::new(155.0, 40.0, 300.0);
        let mirrored = |c: &CameraModel| {
            let p_cam = c.screen_from_camera.inverse_transform_point(&x);
            Intrinsics::from_camera_model(c)
                .project(&Point3::from(-p_cam.coords))
                .unwrap()
        };
        let a = View {
            camera: &rgb,
            pixel: Measured::new(mirrored(&rgb), 0.5).unwrap(),
        };
        let b = View {
            camera: &ir,
            pixel: Measured::new(mirrored(&ir), 0.5).unwrap(),
        };
        assert_eq!(triangulate(&a, &b), Err(GeometryError::BehindCamera));
    }

    #[test]
    fn test_triangulate_zero_sigma_is_degenerate() {
        let (ir, rgb) = stereo_rig();
        let x = Point3::new(155.0, 40.0, -500.0);
        let a = View {
            camera: &rgb,
            pixel: Measured::new(pixel_of(&rgb, &x), 0.0).unwrap(),
        };
        let b = View {
            camera: &ir,
            pixel: Measured::new(pixel_of(&ir, &x), 0.5).unwrap(),
        };
        assert_eq!(
            triangulate(&a, &b),
            Err(GeometryError::Degenerate("zero sigma"))
        );
    }

    #[test]
    fn test_triangulate_mixed_resolutions() {
        let (ir, rgb_src) = stereo_rig();
        let mut rgb = rgb_src;
        rgb_intr().scaled(640, 360).apply_to(&mut rgb);
        let x = Point3::new(155.0, 40.0, -500.0);
        let a = View {
            camera: &rgb,
            pixel: Measured::new(pixel_of(&rgb, &x), 0.5).unwrap(),
        };
        let b = View {
            camera: &ir,
            pixel: Measured::new(pixel_of(&ir, &x), 0.5).unwrap(),
        };
        let t = triangulate(&a, &b).unwrap();
        assert_abs_diff_eq!((t.point - x).norm(), 0.0, epsilon = 1e-8);
    }
}
