//! Pinhole camera model with OpenCV-order Brown-Conrady distortion.
//!
//! Camera frame: x right, y down, z forward, mm. Pixel centres are at +0.5 (the image
//! origin is the top-left pixel corner); intrinsics imported from OpenCV need
//! `cx += 0.5, cy += 0.5`.

use eye_core::{CameraModel, Measured};
use nalgebra::{Matrix2, Point2, Point3, Unit, Vector3};

use crate::GeometryError;
use crate::uncertainty::{Cov2, isotropic2, propagate};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Distortion {
    pub k1: f64,
    pub k2: f64,
    pub p1: f64,
    pub p2: f64,
    pub k3: f64,
}

impl Distortion {
    pub fn from_opencv(d: [f64; 5]) -> Self {
        Self {
            k1: d[0],
            k2: d[1],
            p1: d[2],
            p2: d[3],
            k3: d[4],
        }
    }

    pub fn to_opencv(&self) -> [f64; 5] {
        [self.k1, self.k2, self.p1, self.p2, self.k3]
    }

    pub fn is_zero(&self) -> bool {
        self.to_opencv().iter().all(|k| *k == 0.0)
    }
}

/// Pixel centres are at +0.5 (image origin at the top-left pixel CORNER). Intrinsics imported
/// from OpenCV need `cx += 0.5, cy += 0.5`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Intrinsics {
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub distortion: Distortion,
}

impl Intrinsics {
    /// Square pixels, principal point at the image centre, no distortion; `hfov_rad` is the
    /// horizontal field of view.
    pub fn from_hfov(width: u32, height: u32, hfov_rad: f64) -> Self {
        let (w, h) = (f64::from(width), f64::from(height));
        let f = (w / 2.0) / (hfov_rad / 2.0).tan();
        Self {
            width,
            height,
            fx: f,
            fy: f,
            cx: w / 2.0,
            cy: h / 2.0,
            distortion: Distortion::default(),
        }
    }

    /// Same, from the DIAGONAL field of view (how webcam datasheets state it):
    /// `f = (hypot(W, H) / 2) / tan(dfov / 2)`.
    pub fn from_diagonal_fov(width: u32, height: u32, dfov_rad: f64) -> Self {
        let (w, h) = (f64::from(width), f64::from(height));
        let f = (w.hypot(h) / 2.0) / (dfov_rad / 2.0).tan();
        Self {
            width,
            height,
            fx: f,
            fy: f,
            cx: w / 2.0,
            cy: h / 2.0,
            distortion: Distortion::default(),
        }
    }

    pub fn from_camera_model(m: &CameraModel) -> Self {
        Self {
            width: m.width,
            height: m.height,
            fx: m.fx,
            fy: m.fy,
            cx: m.cx,
            cy: m.cy,
            distortion: Distortion::from_opencv(m.distortion),
        }
    }

    pub fn apply_to(&self, m: &mut CameraModel) {
        m.width = self.width;
        m.height = self.height;
        m.fx = self.fx;
        m.fy = self.fy;
        m.cx = self.cx;
        m.cy = self.cy;
        m.distortion = self.distortion.to_opencv();
    }

    /// Same lens at another resolution of the same sensor (e.g. 1280x720 -> 640x360).
    ///
    /// # Panics
    ///
    /// Panics if the new resolution does not have the same aspect ratio as this one; an
    /// aspect-changing mode needs its own calibration.
    pub fn scaled(&self, width: u32, height: u32) -> Self {
        let sx = f64::from(width) / f64::from(self.width);
        let sy = f64::from(height) / f64::from(self.height);
        assert!(
            (sx - sy).abs() < 1e-9,
            "scaled: aspect ratio must not change"
        );
        Self {
            width,
            height,
            fx: self.fx * sx,
            fy: self.fy * sy,
            cx: self.cx * sx,
            cy: self.cy * sy,
            distortion: self.distortion,
        }
    }

    /// Normalized -> distorted normalized.
    pub fn distort(&self, n: &Point2<f64>) -> Point2<f64> {
        let Distortion { k1, k2, p1, p2, k3 } = self.distortion;
        let (x, y) = (n.x, n.y);
        let r2 = x * x + y * y;
        let radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
        Point2::new(
            x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
            y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
        )
    }

    fn distort_jacobian(&self, n: &Point2<f64>) -> Matrix2<f64> {
        let Distortion { k1, k2, p1, p2, k3 } = self.distortion;
        let (x, y) = (n.x, n.y);
        let r2 = x * x + y * y;
        let radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
        let dradial_dr2 = k1 + r2 * (2.0 * k2 + 3.0 * k3 * r2);
        Matrix2::new(
            radial + 2.0 * x * x * dradial_dr2 + 2.0 * p1 * y + 6.0 * p2 * x,
            2.0 * x * y * dradial_dr2 + 2.0 * p1 * x + 2.0 * p2 * y,
            2.0 * x * y * dradial_dr2 + 2.0 * p1 * x + 2.0 * p2 * y,
            radial + 2.0 * y * y * dradial_dr2 + 6.0 * p1 * y + 2.0 * p2 * x,
        )
    }

    /// Inverse of [`Self::distort`], via Newton's method.
    pub fn undistort(&self, d: &Point2<f64>) -> Result<Point2<f64>, GeometryError> {
        if self.distortion.is_zero() {
            return Ok(*d);
        }
        let mut n = *d;
        for _ in 0..20 {
            let g = self.distort(&n) - d;
            let step = self
                .distort_jacobian(&n)
                .try_inverse()
                .ok_or(GeometryError::Degenerate("distortion not invertible here"))?
                * g;
            n -= step;
            if step.norm() < 1e-14 {
                return Ok(n);
            }
        }
        Err(GeometryError::NotConverged("undistort".into()))
    }

    /// Camera-frame point (mm) -> pixel (px).
    pub fn project(&self, p_cam: &Point3<f64>) -> Result<Point2<f64>, GeometryError> {
        if p_cam.z <= 0.0 {
            return Err(GeometryError::BehindCamera);
        }
        let d = self.distort(&Point2::new(p_cam.x / p_cam.z, p_cam.y / p_cam.z));
        Ok(Point2::new(
            self.fx * d.x + self.cx,
            self.fy * d.y + self.cy,
        ))
    }

    pub fn pixel_to_normalized(&self, px: &Point2<f64>) -> Result<Point2<f64>, GeometryError> {
        self.undistort(&Point2::new(
            (px.x - self.cx) / self.fx,
            (px.y - self.cy) / self.fy,
        ))
    }

    /// Camera-frame ray through a pixel.
    pub fn unproject(&self, px: &Point2<f64>) -> Result<Unit<Vector3<f64>>, GeometryError> {
        let n = self.pixel_to_normalized(px)?;
        Ok(Unit::new_normalize(Vector3::new(n.x, n.y, 1.0)))
    }

    pub fn unproject_measured(
        &self,
        px: &Measured<Point2<f64>>,
    ) -> Result<(Point2<f64>, Cov2), GeometryError> {
        let n = self.pixel_to_normalized(px.value())?;
        let jd = self
            .distort_jacobian(&n)
            .try_inverse()
            .ok_or(GeometryError::Degenerate("distortion not invertible here"))?;
        let j = jd * Matrix2::new(1.0 / self.fx, 0.0, 0.0, 1.0 / self.fy);
        Ok((n, propagate(&j, &isotropic2(px.sigma()))))
    }
}

/// Ray in the reference (screen) frame through a pixel of a rig camera.
pub fn pixel_ray(
    model: &CameraModel,
    px: &Point2<f64>,
) -> Result<(Point3<f64>, Unit<Vector3<f64>>), GeometryError> {
    let dir = Intrinsics::from_camera_model(model).unproject(px)?;
    let origin = Point3::from(model.screen_from_camera.translation.vector);
    Ok((
        origin,
        Unit::new_normalize(model.screen_from_camera.transform_vector(&dir)),
    ))
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use approx::assert_abs_diff_eq;
    use eye_core::CameraId;
    use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3, vector};
    use proptest::prelude::*;

    use super::*;
    use crate::synth::SplitMix64;
    use crate::uncertainty::numeric_jacobian;

    fn ir_test_intrinsics() -> Intrinsics {
        Intrinsics {
            width: 640,
            height: 360,
            fx: 457.0,
            fy: 457.0,
            cx: 320.0,
            cy: 180.0,
            distortion: Distortion {
                k1: 0.08,
                k2: -0.15,
                p1: 0.001,
                p2: -0.0005,
                k3: 0.05,
            },
        }
    }

    fn nominal_camera_model() -> CameraModel {
        CameraModel {
            id: CameraId::new("ir"),
            width: 640,
            height: 360,
            fx: 457.0,
            fy: 457.0,
            cx: 320.0,
            cy: 180.0,
            distortion: [0.0; 5],
            screen_from_camera: Isometry3::identity(),
        }
    }

    fn distinct_camera_model() -> CameraModel {
        CameraModel {
            id: CameraId::new("ir"),
            width: 640,
            height: 360,
            fx: 457.25,
            fy: 456.5,
            cx: 320.125,
            cy: 180.375,
            distortion: [0.08, -0.15, 0.001, -0.0005, 0.05],
            screen_from_camera: Isometry3::from_parts(
                Translation3::new(155.0, -7.0, 0.0),
                UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI),
            ),
        }
    }

    #[test]
    fn test_from_hfov_matches_formula() {
        let i = Intrinsics::from_hfov(640, 360, 70f64.to_radians());
        let expected = 320.0 / 35f64.to_radians().tan();
        assert_abs_diff_eq!(i.fx, expected, epsilon = 1e-9);
        assert_abs_diff_eq!(i.fy, expected, epsilon = 1e-9);
        assert_eq!(i.cx, 320.0);
        assert_eq!(i.cy, 180.0);
    }

    #[test]
    fn test_from_diagonal_fov_matches_dell_spec() {
        let a = Intrinsics::from_diagonal_fov(1280, 720, 75.8f64.to_radians());
        assert_abs_diff_eq!(a.fx, 943.253, epsilon = 1e-3);
        assert_abs_diff_eq!(a.fy, 943.253, epsilon = 1e-3);
        assert_eq!(a.cx, 640.0);
        assert_eq!(a.cy, 360.0);
        assert!(a.distortion.is_zero());

        let b = Intrinsics::from_diagonal_fov(1280, 720, 87f64.to_radians());
        assert_abs_diff_eq!(b.fx, 773.793, epsilon = 1e-3);

        let c = Intrinsics::from_diagonal_fov(640, 360, 75.8f64.to_radians());
        assert_abs_diff_eq!(c.fx, 471.626, epsilon = 1e-3);

        let d = Intrinsics::from_diagonal_fov(640, 360, 87f64.to_radians());
        assert_abs_diff_eq!(d.fx, 386.897, epsilon = 1e-3);
    }

    #[test]
    fn test_project_optical_axis_hits_principal_point() {
        let i = ir_test_intrinsics();
        let p = i.project(&Point3::new(0.0, 0.0, 500.0)).unwrap();
        assert_abs_diff_eq!(p.x, i.cx, epsilon = 1e-12);
        assert_abs_diff_eq!(p.y, i.cy, epsilon = 1e-12);
    }

    #[test]
    fn test_project_behind_camera_errors() {
        let i = ir_test_intrinsics();
        assert_eq!(
            i.project(&Point3::new(0.0, 0.0, 0.0)),
            Err(GeometryError::BehindCamera)
        );
        assert_eq!(
            i.project(&Point3::new(0.0, 0.0, -1.0)),
            Err(GeometryError::BehindCamera)
        );
    }

    #[test]
    fn test_zero_distortion_project_is_pinhole() {
        let mut i = ir_test_intrinsics();
        i.distortion = Distortion::default();
        let p = i.project(&Point3::new(100.0, -50.0, 500.0)).unwrap();
        assert_abs_diff_eq!(p.x, 320.0 + 457.0 * 0.2, epsilon = 1e-9);
        assert_abs_diff_eq!(p.y, 180.0 - 457.0 * 0.1, epsilon = 1e-9);
    }

    proptest! {
        #[test]
        fn prop_distort_undistort_round_trip(
            x in -0.75f64..0.75,
            y in -0.75f64..0.75,
        ) {
            prop_assume!(x * x + y * y <= 0.75 * 0.75);
            let i = ir_test_intrinsics();
            let n = Point2::new(x, y);
            let d = i.distort(&n);
            let back = i.undistort(&d).unwrap();
            prop_assert!((back.x - n.x).abs() < 1e-10);
            prop_assert!((back.y - n.y).abs() < 1e-10);
        }

        #[test]
        fn prop_project_unproject_round_trip(
            px in 0.0f64..640.0,
            py in 0.0f64..360.0,
            depth in 200.0f64..1000.0,
        ) {
            let i = ir_test_intrinsics();
            let pixel = Point2::new(px, py);
            let dir = i.unproject(&pixel).unwrap();
            let p_cam = Point3::from(dir.into_inner() * (depth / dir.z));
            let back = i.project(&p_cam).unwrap();
            prop_assert!((back.x - pixel.x).abs() < 1e-8);
            prop_assert!((back.y - pixel.y).abs() < 1e-8);
        }
    }

    #[test]
    fn test_distort_jacobian_matches_numeric() {
        let i = ir_test_intrinsics();
        let points = [
            (0.0, 0.0),
            (0.1, 0.0),
            (0.0, 0.1),
            (0.3, 0.2),
            (-0.3, 0.2),
            (0.3, -0.2),
            (-0.3, -0.2),
            (0.5, 0.5),
            (-0.5, -0.5),
            (0.7, 0.1),
        ];
        for (x, y) in points {
            let n = Point2::new(x, y);
            let analytic = i.distort_jacobian(&n);
            let f = |v: &nalgebra::SVector<f64, 2>| -> Option<nalgebra::SVector<f64, 2>> {
                let d = i.distort(&Point2::new(v[0], v[1]));
                Some(vector![d.x, d.y])
            };
            let numeric = numeric_jacobian(f, &vector![x, y]).unwrap();
            assert_abs_diff_eq!(analytic, numeric, epsilon = 1e-7);
        }
    }

    #[test]
    fn test_unproject_measured_covariance_matches_monte_carlo() {
        let i = ir_test_intrinsics();
        let pixel = Point2::new(639.0, 359.0);
        let sigma = 0.5;
        let measured = Measured::new(pixel, sigma).unwrap();
        let (_, predicted) = i.unproject_measured(&measured).unwrap();

        let n = 20_000;
        let mut rng = SplitMix64::new(11);
        let mut samples = Vec::with_capacity(n);
        let mut sum = Vector3::zeros().xy();
        for _ in 0..n {
            let noisy = Point2::new(
                pixel.x + sigma * rng.gaussian(),
                pixel.y + sigma * rng.gaussian(),
            );
            let normalized = i.pixel_to_normalized(&noisy).unwrap();
            let v = vector![normalized.x, normalized.y];
            sum += v;
            samples.push(v);
        }
        let mean = sum / n as f64;
        let mut empirical = Matrix2::zeros();
        for v in &samples {
            let d = v - mean;
            empirical += d * d.transpose();
        }
        empirical /= (n - 1) as f64;

        for k in 0..2 {
            let tol = 0.05 * predicted[(k, k)].abs();
            assert!(
                (empirical[(k, k)] - predicted[(k, k)]).abs() <= tol,
                "diag {k}: empirical {} predicted {}",
                empirical[(k, k)],
                predicted[(k, k)]
            );
        }
        for a in 0..2 {
            for b in 0..2 {
                if a == b {
                    continue;
                }
                let tol = 0.05 * (predicted[(a, a)] * predicted[(b, b)]).sqrt();
                assert!(
                    (empirical[(a, b)] - predicted[(a, b)]).abs() <= tol,
                    "offdiag ({a},{b}): empirical {} predicted {}",
                    empirical[(a, b)],
                    predicted[(a, b)]
                );
            }
        }
    }

    #[test]
    fn test_undistort_non_invertible_point_errors() {
        let i = Intrinsics {
            width: 640,
            height: 360,
            fx: 457.0,
            fy: 457.0,
            cx: 320.0,
            cy: 180.0,
            distortion: Distortion {
                k1: -0.5,
                ..Default::default()
            },
        };
        let result = i.undistort(&Point2::new(0.7, 0.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_scaled_halves_focal_and_principal_point() {
        let i = Intrinsics {
            width: 1280,
            height: 720,
            fx: 914.0,
            fy: 914.0,
            cx: 640.0,
            cy: 360.0,
            distortion: Distortion::default(),
        };
        let s = i.scaled(640, 360);
        assert_abs_diff_eq!(s.fx, 457.0, epsilon = 1e-12);
        assert_abs_diff_eq!(s.cx, 320.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "aspect ratio must not change")]
    fn test_scaled_rejects_aspect_change() {
        let i = Intrinsics {
            width: 1280,
            height: 720,
            fx: 914.0,
            fy: 914.0,
            cx: 640.0,
            cy: 360.0,
            distortion: Distortion::default(),
        };
        i.scaled(640, 480);
    }

    #[test]
    fn test_camera_model_round_trip() {
        let original = distinct_camera_model();
        let mut clone = original.clone();
        let intrinsics = Intrinsics::from_camera_model(&original);
        intrinsics.apply_to(&mut clone);
        assert_eq!(clone.width, original.width);
        assert_eq!(clone.height, original.height);
        assert_eq!(clone.fx, original.fx);
        assert_eq!(clone.fy, original.fy);
        assert_eq!(clone.cx, original.cx);
        assert_eq!(clone.cy, original.cy);
        assert_eq!(clone.distortion, original.distortion);
        assert_eq!(clone.id, original.id);
        assert_eq!(clone.screen_from_camera, original.screen_from_camera);
    }

    #[test]
    fn test_pixel_ray_uses_extrinsics() {
        let mut model = nominal_camera_model();
        model.screen_from_camera = Isometry3::from_parts(
            Translation3::new(155.0, -7.0, 0.0),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI),
        );
        let px = Point2::new(model.cx, model.cy);
        let (origin, dir) = pixel_ray(&model, &px).unwrap();
        assert_abs_diff_eq!(origin.x, 155.0, epsilon = 1e-12);
        assert_abs_diff_eq!(origin.y, -7.0, epsilon = 1e-12);
        assert_abs_diff_eq!(origin.z, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dir.x, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dir.y, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dir.z, -1.0, epsilon = 1e-12);
    }
}
