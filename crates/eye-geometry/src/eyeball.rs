//! Eyeball model: anatomical priors and the pupil-centre + eyeball-model gaze ray, after
//! Guestrin & Eizenman 2006 ("General theory of remote gaze estimation using the pupil center
//! and corneal reflections", IEEE TBME 53(6)) without corneal refraction.

use std::f64::consts::PI;

use eye_core::{CameraModel, GazeRay, Measured, Side};
use nalgebra::{Point2, Point3, Unit, UnitQuaternion, Vector2, Vector3, Vector5};

use crate::GeometryError;
use crate::angles::{direction_from_yaw_pitch, yaw_pitch_from_direction};
use crate::camera::pixel_ray;
use crate::uncertainty::{Cov3, block_diag, isotropic2, propagate_fn};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Kappa {
    pub alpha_rad: f64,
    pub beta_rad: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyeParams {
    pub globe_radius_mm: f64,
    pub rotation_to_apex_mm: f64,
    pub rotation_to_pupil_mm: f64,
    pub iris_radius_mm: f64,
    pub cornea_radius_mm: f64,
    pub corner_midpoint_to_rotation_mm: f64,
    pub kappa: Kappa,
}

impl Default for EyeParams {
    fn default() -> Self {
        Self {
            globe_radius_mm: 12.0,
            rotation_to_apex_mm: 13.5,
            rotation_to_pupil_mm: 10.46,
            iris_radius_mm: 5.85,
            cornea_radius_mm: 7.8,
            corner_midpoint_to_rotation_mm: 4.5,
            kappa: Kappa {
                alpha_rad: 5f64.to_radians(),
                beta_rad: 1.5f64.to_radians(),
            },
        }
    }
}

impl EyeParams {
    pub fn rotation_to_cornea_mm(&self) -> f64 {
        self.rotation_to_apex_mm - self.cornea_radius_mm
    }

    pub fn cornea_to_pupil_mm(&self) -> f64 {
        self.rotation_to_pupil_mm - self.rotation_to_cornea_mm()
    }

    pub fn rotation_to_iris_mm(&self) -> f64 {
        (self.globe_radius_mm.powi(2) - self.iris_radius_mm.powi(2)).sqrt()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyeCentre {
    pub position: Point3<f64>,
    pub cov: Cov3,
}

pub fn eyeball_centre_in_head(
    inner: &Point3<f64>,
    outer: &Point3<f64>,
    params: &EyeParams,
) -> Point3<f64> {
    Point3::from(
        (inner.coords + outer.coords) / 2.0
            + Vector3::new(0.0, 0.0, params.corner_midpoint_to_rotation_mm),
    )
}

pub fn screen_from_viewer(
    screen_from_camera: &UnitQuaternion<f64>,
    camera_from_head: &UnitQuaternion<f64>,
) -> UnitQuaternion<f64> {
    screen_from_camera * camera_from_head * UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI)
}

pub fn ray_sphere_near(
    o: &Point3<f64>,
    d: &Unit<Vector3<f64>>,
    c: &Point3<f64>,
    r: f64,
) -> Point3<f64> {
    let oc = o - c;
    let b = d.dot(&oc);
    let disc = b * b - (oc.norm_squared() - r * r);
    if disc >= 0.0 {
        return o + d.into_inner() * (-b - disc.sqrt());
    }
    let closest = o + d.into_inner() * (-b);
    c + (closest - c).normalize() * r
}

pub fn optical_axis(
    rotation_centre: &Point3<f64>,
    pupil: &Point3<f64>,
) -> Option<Unit<Vector3<f64>>> {
    Unit::try_new(pupil - rotation_centre, 1e-12)
}

pub fn visual_axis(
    optical: &Unit<Vector3<f64>>,
    kappa: &Kappa,
    side: Side,
    screen_from_viewer: &UnitQuaternion<f64>,
) -> Unit<Vector3<f64>> {
    let local = Unit::new_normalize(screen_from_viewer.inverse_transform_vector(optical));
    let a = yaw_pitch_from_direction(&local);
    let s = match side {
        Side::Right => -1.0,
        Side::Left => 1.0,
    };
    let shifted = direction_from_yaw_pitch(&Vector2::new(
        a.x + s * kappa.alpha_rad,
        a.y + kappa.beta_rad,
    ));
    Unit::new_normalize(screen_from_viewer.transform_vector(&shifted))
}

pub fn cornea_centre_from_coaxial_glint(
    glint_origin: &Point3<f64>,
    glint_dir: &Unit<Vector3<f64>>,
    centre: &Point3<f64>,
    params: &EyeParams,
) -> Point3<f64> {
    ray_sphere_near(
        glint_origin,
        glint_dir,
        centre,
        params.rotation_to_cornea_mm(),
    )
}

pub fn gaze_ray(
    side: Side,
    centre: &EyeCentre,
    camera: &CameraModel,
    pupil_px: &Measured<Point2<f64>>,
    params: &EyeParams,
    screen_from_viewer: &UnitQuaternion<f64>,
) -> Result<GazeRay, GeometryError> {
    let g = |th: &Vector5<f64>| -> Option<Vector2<f64>> {
        let e = Point3::new(th[0], th[1], th[2]);
        let (o, d) = pixel_ray(camera, &Point2::new(th[3], th[4])).ok()?;
        let p = ray_sphere_near(&o, &d, &e, params.rotation_to_pupil_mm);
        let opt = optical_axis(&e, &p)?;
        Some(yaw_pitch_from_direction(&visual_axis(
            &opt,
            &params.kappa,
            side,
            screen_from_viewer,
        )))
    };
    let px = pupil_px.value();
    let th = Vector5::new(
        centre.position.x,
        centre.position.y,
        centre.position.z,
        px.x,
        px.y,
    );
    let cov_in = block_diag::<3, 2, 5>(&centre.cov, &isotropic2(pupil_px.sigma()));
    let (angles, angular_cov) =
        propagate_fn(g, &th, &cov_in).ok_or(GeometryError::Degenerate("gaze ray"))?;
    Ok(GazeRay {
        side: Some(side),
        origin: centre.position,
        direction: direction_from_yaw_pitch(&angles),
        angular_cov,
        origin_cov: centre.cov,
    })
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use approx::assert_abs_diff_eq;
    use eye_core::CameraId;
    use nalgebra::{Isometry3, Matrix3, Translation3};

    use super::*;
    use crate::camera::Intrinsics;
    use crate::synth::SplitMix64;

    fn fixture_camera() -> CameraModel {
        CameraModel {
            id: CameraId::new("ir"),
            width: 640,
            height: 360,
            fx: 457.0,
            fy: 457.0,
            cx: 320.0,
            cy: 180.0,
            distortion: [0.08, -0.15, 0.001, -0.0005, 0.05],
            screen_from_camera: Isometry3::from_parts(
                Translation3::new(167.5, -7.0, 0.0),
                UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI),
            ),
        }
    }

    fn zero_kappa_params() -> EyeParams {
        EyeParams {
            kappa: Kappa {
                alpha_rad: 0.0,
                beta_rad: 0.0,
            },
            ..Default::default()
        }
    }

    #[test]
    fn test_defaults_match_table() {
        let p = EyeParams::default();
        assert_abs_diff_eq!(p.globe_radius_mm, 12.0, epsilon = 1e-12);
        assert_abs_diff_eq!(p.rotation_to_apex_mm, 13.5, epsilon = 1e-12);
        assert_abs_diff_eq!(p.rotation_to_pupil_mm, 10.46, epsilon = 1e-12);
        assert_abs_diff_eq!(p.iris_radius_mm, 5.85, epsilon = 1e-12);
        assert_abs_diff_eq!(p.cornea_radius_mm, 7.8, epsilon = 1e-12);
        assert_abs_diff_eq!(p.corner_midpoint_to_rotation_mm, 4.5, epsilon = 1e-12);
        assert_abs_diff_eq!(p.kappa.alpha_rad, 5f64.to_radians(), epsilon = 1e-12);
        assert_abs_diff_eq!(p.kappa.beta_rad, 1.5f64.to_radians(), epsilon = 1e-12);
        assert_abs_diff_eq!(p.rotation_to_cornea_mm(), 5.7, epsilon = 1e-12);
        assert_abs_diff_eq!(p.cornea_to_pupil_mm(), 4.76, epsilon = 1e-12);
        assert_abs_diff_eq!(p.rotation_to_iris_mm(), 10.48, epsilon = 0.005);
    }

    #[test]
    fn test_eye_constants_are_consistent() {
        let p = EyeParams::default();
        assert_abs_diff_eq!(
            p.rotation_to_cornea_mm() + p.cornea_to_pupil_mm(),
            p.rotation_to_pupil_mm,
            epsilon = 1e-12
        );
        assert!((p.rotation_to_iris_mm() - p.rotation_to_pupil_mm).abs() < 0.05);
    }

    #[test]
    fn test_ray_sphere_near_hits_front_of_sphere() {
        let o = Point3::new(0.0, 0.0, 0.0);
        let d = Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0));
        let c = Point3::new(0.0, 0.0, 500.0);
        let p = ray_sphere_near(&o, &d, &c, 10.0);
        assert_abs_diff_eq!(p, Point3::new(0.0, 0.0, 490.0), epsilon = 1e-12);
    }

    #[test]
    fn test_ray_sphere_near_is_continuous_at_tangency() {
        let d = Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0));
        let c = Point3::new(0.0, 0.0, 500.0);
        let p1 = ray_sphere_near(&Point3::new(9.9999, 0.0, 0.0), &d, &c, 10.0);
        let p2 = ray_sphere_near(&Point3::new(10.0001, 0.0, 0.0), &d, &c, 10.0);
        assert!((p1 - p2).norm() < 0.05);
    }

    #[test]
    fn test_visual_axis_right_eye_frontal_is_nasal_and_up() {
        let optical = Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0));
        let params = EyeParams::default();
        let r_v = UnitQuaternion::identity();

        let right = visual_axis(&optical, &params.kappa, Side::Right, &r_v);
        let right_angles = yaw_pitch_from_direction(&right);
        assert_abs_diff_eq!(right_angles.x, -5f64.to_radians(), epsilon = 1e-12);
        assert_abs_diff_eq!(right_angles.y, 1.5f64.to_radians(), epsilon = 1e-12);

        let left = visual_axis(&optical, &params.kappa, Side::Left, &r_v);
        let left_angles = yaw_pitch_from_direction(&left);
        assert_abs_diff_eq!(left_angles.x, 5f64.to_radians(), epsilon = 1e-12);
        assert_abs_diff_eq!(left_angles.y, 1.5f64.to_radians(), epsilon = 1e-12);
    }

    #[test]
    fn test_visual_axis_follows_head_roll() {
        let optical = Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0));
        let params = EyeParams::default();
        let r_v = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), PI / 2.0);

        let axis = visual_axis(&optical, &params.kappa, Side::Right, &r_v);
        let angles = yaw_pitch_from_direction(&axis);
        assert_abs_diff_eq!(angles.x, 1.5f64.to_radians(), epsilon = 1e-3);
        assert_abs_diff_eq!(angles.y, 5f64.to_radians(), epsilon = 1e-3);
    }

    #[test]
    fn test_screen_from_viewer_is_identity_for_frontal_head_and_nominal_camera() {
        let screen_from_camera = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI);
        let camera_from_head = UnitQuaternion::identity();
        let r_v = screen_from_viewer(&screen_from_camera, &camera_from_head);
        assert!(r_v.angle() < 1e-12);
    }

    #[test]
    fn test_gaze_ray_recovers_true_axis_noise_free() {
        let cam = fixture_camera();
        let params = zero_kappa_params();
        let e = Point3::new(185.0, 60.0, -500.0);
        let target = Point3::new(100.0, 50.0, 0.0);
        let axis = Unit::new_normalize(target - e);
        let p = e + axis.into_inner() * params.rotation_to_pupil_mm;
        let pupil_cam = cam.screen_from_camera.inverse_transform_point(&p);
        let pupil_pixel = Intrinsics::from_camera_model(&cam)
            .project(&pupil_cam)
            .unwrap();

        let centre = EyeCentre {
            position: e,
            cov: Matrix3::zeros(),
        };
        let pupil_px = Measured::new(pupil_pixel, 0.3).unwrap();
        let r_v = UnitQuaternion::identity();

        let ray = gaze_ray(Side::Right, &centre, &cam, &pupil_px, &params, &r_v).unwrap();

        assert_abs_diff_eq!(ray.direction, axis, epsilon = 1e-9);

        let t = -ray.origin.z / ray.direction.z;
        let hit = ray.origin + ray.direction.into_inner() * t;
        assert_abs_diff_eq!(hit, target, epsilon = 1e-6);
    }

    #[test]
    fn test_gaze_ray_cov_matches_monte_carlo() {
        let cam = fixture_camera();
        let params = zero_kappa_params();
        let e = Point3::new(185.0, 60.0, -500.0);
        let target = Point3::new(100.0, 50.0, 0.0);
        let axis = Unit::new_normalize(target - e);
        let p = e + axis.into_inner() * params.rotation_to_pupil_mm;
        let pupil_cam = cam.screen_from_camera.inverse_transform_point(&p);
        let pupil_pixel = Intrinsics::from_camera_model(&cam)
            .project(&pupil_cam)
            .unwrap();
        let r_v = UnitQuaternion::identity();

        let sigma_px = 0.3;
        let e_cov = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, 25.0));
        let centre = EyeCentre {
            position: e,
            cov: e_cov,
        };
        let pupil_px = Measured::new(pupil_pixel, sigma_px).unwrap();
        let predicted = gaze_ray(Side::Right, &centre, &cam, &pupil_px, &params, &r_v)
            .unwrap()
            .angular_cov;

        let n = 5_000;
        let mut rng = SplitMix64::new(9);
        let mut samples = Vec::with_capacity(n);
        let mut sum = Vector2::zeros();
        for _ in 0..n {
            let noisy_e = Point3::new(
                e.x + 1.0 * rng.gaussian(),
                e.y + 1.0 * rng.gaussian(),
                e.z + 5.0 * rng.gaussian(),
            );
            let noisy_px = Point2::new(
                pupil_pixel.x + sigma_px * rng.gaussian(),
                pupil_pixel.y + sigma_px * rng.gaussian(),
            );
            let trial_centre = EyeCentre {
                position: noisy_e,
                cov: Matrix3::zeros(),
            };
            let trial_px = Measured::new(noisy_px, 1e-12).unwrap();
            let ray = gaze_ray(Side::Right, &trial_centre, &cam, &trial_px, &params, &r_v).unwrap();
            let angles = yaw_pitch_from_direction(&ray.direction);
            sum += angles;
            samples.push(angles);
        }
        let mean = sum / n as f64;
        let mut empirical = nalgebra::Matrix2::zeros();
        for a in &samples {
            let d = a - mean;
            empirical += d * d.transpose();
        }
        empirical /= (n - 1) as f64;

        for k in 0..2 {
            let tol = 0.10 * predicted[(k, k)].abs();
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
    fn test_gaze_ray_lateral_sensitivity_is_one_over_radius() {
        let cam = fixture_camera();
        let params = zero_kappa_params();
        let e = Point3::new(185.0, 60.0, -500.0);
        let target = Point3::new(100.0, 50.0, 0.0);
        let axis = Unit::new_normalize(target - e);
        let p = e + axis.into_inner() * params.rotation_to_pupil_mm;
        let pupil_cam = cam.screen_from_camera.inverse_transform_point(&p);
        let pupil_pixel = Intrinsics::from_camera_model(&cam)
            .project(&pupil_cam)
            .unwrap();
        let r_v = UnitQuaternion::identity();

        let centre = EyeCentre {
            position: e,
            cov: Matrix3::from_diagonal(&Vector3::new(1.0, 0.0, 0.0)),
        };
        let pupil_px = Measured::new(pupil_pixel, 1e-9).unwrap();
        let ray = gaze_ray(Side::Right, &centre, &cam, &pupil_px, &params, &r_v).unwrap();
        let sigma_yaw = ray.angular_cov[(0, 0)].sqrt();
        assert_abs_diff_eq!(sigma_yaw, 1.0 / 10.46, epsilon = 0.15 * (1.0 / 10.46));
    }

    #[test]
    fn test_gaze_ray_fills_origin_cov() {
        let cam = fixture_camera();
        let params = zero_kappa_params();
        let e = Point3::new(185.0, 60.0, -500.0);
        let target = Point3::new(100.0, 50.0, 0.0);
        let axis = Unit::new_normalize(target - e);
        let p = e + axis.into_inner() * params.rotation_to_pupil_mm;
        let pupil_cam = cam.screen_from_camera.inverse_transform_point(&p);
        let pupil_pixel = Intrinsics::from_camera_model(&cam)
            .project(&pupil_cam)
            .unwrap();
        let r_v = UnitQuaternion::identity();

        let e_cov = Matrix3::from_diagonal(&Vector3::new(1.0, 2.0, 3.0));
        let centre = EyeCentre {
            position: e,
            cov: e_cov,
        };
        let pupil_px = Measured::new(pupil_pixel, 0.3).unwrap();
        let ray = gaze_ray(Side::Right, &centre, &cam, &pupil_px, &params, &r_v).unwrap();
        assert_eq!(ray.origin_cov, e_cov);
        assert_eq!(ray.side, Some(Side::Right));
    }

    #[test]
    fn test_eyeball_centre_in_head_is_behind_corner_midpoint() {
        let inner = Point3::new(30.0, 0.0, 0.0);
        let outer = Point3::new(60.0, 0.0, 0.0);
        let params = EyeParams::default();
        let centre = eyeball_centre_in_head(&inner, &outer, &params);
        assert_abs_diff_eq!(centre, Point3::new(45.0, 0.0, 4.5), epsilon = 1e-12);
    }

    #[test]
    fn test_cornea_centre_from_glint_lies_between_camera_and_rotation_centre() {
        let cam = fixture_camera();
        let e = Point3::new(185.0, 60.0, -500.0);
        let params = EyeParams::default();
        let cam_centre = Point3::from(cam.screen_from_camera.translation.vector);
        let dir = Unit::new_normalize(e - cam_centre);
        let c = cornea_centre_from_coaxial_glint(&cam_centre, &dir, &e, &params);
        assert_abs_diff_eq!(
            (c - e).norm(),
            params.rotation_to_cornea_mm(),
            epsilon = 1e-9
        );
        assert!((c - cam_centre).norm() < (e - cam_centre).norm());
    }
}
