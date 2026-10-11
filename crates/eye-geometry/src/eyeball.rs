//! Eyeball model: anatomical priors and the pupil-centre + eyeball-model gaze ray, after
//! Guestrin & Eizenman 2006 ("General theory of remote gaze estimation using the pupil center
//! and corneal reflections", IEEE TBME 53(6)) without corneal refraction.

use std::f64::consts::PI;

use eye_core::angles::{direction_from_yaw_pitch, yaw_pitch_from_direction};
use eye_core::{CameraModel, GazeRay, Measured, Side, Timestamp};
use nalgebra::{
    Matrix2x3, Point2, Point3, SVector, Unit, UnitQuaternion, Vector2, Vector3, Vector5,
};

use crate::GeometryError;
use crate::camera::pixel_ray;
use crate::uncertainty::{Cov3, block_diag, isotropic2, numeric_jacobian, propagate_fn};

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

    /// `self` with kappa zeroed: the visual axis coincides with the optical axis.
    pub fn without_kappa(self) -> Self {
        Self {
            kappa: Kappa {
                alpha_rad: 0.0,
                beta_rad: 0.0,
            },
            ..self
        }
    }

    /// `self` when `apply_kappa`, else `without_kappa()`.
    pub fn effective(self, apply_kappa: bool) -> Self {
        if apply_kappa {
            self
        } else {
            self.without_kappa()
        }
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
    let s = match side {
        Side::Right => -1.0,
        Side::Left => 1.0,
    };
    let primary = Vector3::z();
    let listing =
        UnitQuaternion::rotation_between(&primary, &local).unwrap_or_else(UnitQuaternion::identity);
    let kappa_dir = direction_from_yaw_pitch(&Vector2::new(s * kappa.alpha_rad, kappa.beta_rad));
    let shifted = listing * kappa_dir.into_inner();
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
    screen_from_viewer: Option<&UnitQuaternion<f64>>,
    at: Timestamp,
) -> Result<GazeRay, GeometryError> {
    let viewer = screen_from_viewer
        .copied()
        .unwrap_or_else(UnitQuaternion::identity);
    let g = |th: &Vector5<f64>| -> Option<Vector2<f64>> {
        let e = Point3::new(th[0], th[1], th[2]);
        let (o, d) = pixel_ray(camera, &Point2::new(th[3], th[4])).ok()?;
        let p = ray_sphere_near(&o, &d, &e, params.rotation_to_pupil_mm);
        let opt = optical_axis(&e, &p)?;
        Some(yaw_pitch_from_direction(&visual_axis(
            &opt,
            &params.kappa,
            side,
            &viewer,
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
        timestamp: at,
        origin: centre.position,
        direction: direction_from_yaw_pitch(&angles),
        angular_cov,
        origin_cov: centre.cov,
        head_rotation: screen_from_viewer.copied(),
    })
}

/// Jacobian of the pupil-sphere ray's `(yaw, pitch)` with respect to `centre.position`, holding
/// `pupil_px` fixed. Lets a caller that builds two rays from the same measured eye centre (an
/// ir-pupil ray seeded from the landmark frame) account for the centre error they share.
pub fn gaze_ray_centre_jacobian(
    side: Side,
    centre: &EyeCentre,
    camera: &CameraModel,
    pupil_px: &Measured<Point2<f64>>,
    params: &EyeParams,
    screen_from_viewer: Option<&UnitQuaternion<f64>>,
) -> Option<Matrix2x3<f64>> {
    let viewer = screen_from_viewer
        .copied()
        .unwrap_or_else(UnitQuaternion::identity);
    let (o, d) = pixel_ray(camera, pupil_px.value()).ok()?;
    let g = |e: &Vector3<f64>| -> Option<Vector2<f64>> {
        let e = Point3::from(*e);
        let p = ray_sphere_near(&o, &d, &e, params.rotation_to_pupil_mm);
        let opt = optical_axis(&e, &p)?;
        Some(yaw_pitch_from_direction(&visual_axis(
            &opt,
            &params.kappa,
            side,
            &viewer,
        )))
    };
    numeric_jacobian(g, &centre.position.coords)
}

/// Glint-referenced ray: the cornea centre lies on the glint ray at `rotation_to_cornea_mm()`
/// from `centre`, the pupil on the pupil ray at `cornea_to_pupil_mm()` from the cornea centre.
/// If the glint ray misses the cornea sphere, `ray_sphere_near` returns the nearest point on it
/// instead, so the ray degrades rather than failing.
#[allow(clippy::too_many_arguments)]
pub fn gaze_ray_pccr(
    side: Side,
    centre: &EyeCentre,
    camera: &CameraModel,
    pupil_px: &Measured<Point2<f64>>,
    glint_px: &Measured<Point2<f64>>,
    params: &EyeParams,
    screen_from_viewer: Option<&UnitQuaternion<f64>>,
    at: Timestamp,
) -> Result<GazeRay, GeometryError> {
    let viewer = screen_from_viewer
        .copied()
        .unwrap_or_else(UnitQuaternion::identity);
    let g = |th: &SVector<f64, 7>| -> Option<Vector2<f64>> {
        let e = Point3::new(th[0], th[1], th[2]);
        let (o, u) = pixel_ray(camera, &Point2::new(th[3], th[4])).ok()?;
        let (_, v) = pixel_ray(camera, &Point2::new(th[5], th[6])).ok()?;
        let c = cornea_centre_from_coaxial_glint(&o, &v, &e, params);
        let p = ray_sphere_near(&o, &u, &c, params.cornea_to_pupil_mm());
        let opt = optical_axis(&c, &p)?;
        Some(yaw_pitch_from_direction(&visual_axis(
            &opt,
            &params.kappa,
            side,
            &viewer,
        )))
    };
    let pupil = pupil_px.value();
    let glint = glint_px.value();
    let th = SVector::<f64, 7>::from_row_slice(&[
        centre.position.x,
        centre.position.y,
        centre.position.z,
        pupil.x,
        pupil.y,
        glint.x,
        glint.y,
    ]);
    let cov5 = block_diag::<3, 2, 5>(&centre.cov, &isotropic2(pupil_px.sigma()));
    let cov_in = block_diag::<5, 2, 7>(&cov5, &isotropic2(glint_px.sigma()));
    let (angles, angular_cov) =
        propagate_fn(g, &th, &cov_in).ok_or(GeometryError::Degenerate("gaze ray pccr"))?;
    Ok(GazeRay {
        side: Some(side),
        timestamp: at,
        origin: centre.position,
        direction: direction_from_yaw_pitch(&angles),
        angular_cov,
        origin_cov: centre.cov,
        head_rotation: screen_from_viewer.copied(),
    })
}

/// Jacobian of the glint-referenced ray's `(yaw, pitch)` with respect to `centre.position`,
/// holding `pupil_px` and `glint_px` fixed. See [`gaze_ray_centre_jacobian`].
#[allow(clippy::too_many_arguments)]
pub fn gaze_ray_pccr_centre_jacobian(
    side: Side,
    centre: &EyeCentre,
    camera: &CameraModel,
    pupil_px: &Measured<Point2<f64>>,
    glint_px: &Measured<Point2<f64>>,
    params: &EyeParams,
    screen_from_viewer: Option<&UnitQuaternion<f64>>,
) -> Option<Matrix2x3<f64>> {
    let viewer = screen_from_viewer
        .copied()
        .unwrap_or_else(UnitQuaternion::identity);
    let (o, u) = pixel_ray(camera, pupil_px.value()).ok()?;
    let (_, v) = pixel_ray(camera, glint_px.value()).ok()?;
    let g = |e: &Vector3<f64>| -> Option<Vector2<f64>> {
        let e = Point3::from(*e);
        let c = cornea_centre_from_coaxial_glint(&o, &v, &e, params);
        let p = ray_sphere_near(&o, &u, &c, params.cornea_to_pupil_mm());
        let opt = optical_axis(&c, &p)?;
        Some(yaw_pitch_from_direction(&visual_axis(
            &opt,
            &params.kappa,
            side,
            &viewer,
        )))
    };
    numeric_jacobian(g, &centre.position.coords)
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
    fn test_without_kappa_zeroes_only_kappa() {
        let p = EyeParams {
            globe_radius_mm: 13.0,
            ..EyeParams::default()
        }
        .without_kappa();

        assert_eq!(
            p.kappa,
            Kappa {
                alpha_rad: 0.0,
                beta_rad: 0.0
            }
        );
        assert_abs_diff_eq!(p.globe_radius_mm, 13.0, epsilon = 1e-12);
        let default = EyeParams::default();
        assert_abs_diff_eq!(
            p.rotation_to_apex_mm,
            default.rotation_to_apex_mm,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            p.rotation_to_pupil_mm,
            default.rotation_to_pupil_mm,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(p.iris_radius_mm, default.iris_radius_mm, epsilon = 1e-12);
        assert_abs_diff_eq!(
            p.cornea_radius_mm,
            default.cornea_radius_mm,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            p.corner_midpoint_to_rotation_mm,
            default.corner_midpoint_to_rotation_mm,
            epsilon = 1e-12
        );

        assert_eq!(default.effective(true), default);
        assert_eq!(default.effective(false), default.without_kappa());
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

    fn additive_visual_axis(
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

    #[test]
    fn test_visual_axis_listing_matches_additive_on_yaw_axis() {
        let params = EyeParams::default();
        let r_v = UnitQuaternion::identity();
        let optical = direction_from_yaw_pitch(&Vector2::new(20f64.to_radians(), 0.0));

        let listing = visual_axis(&optical, &params.kappa, Side::Right, &r_v);
        let additive = additive_visual_axis(&optical, &params.kappa, Side::Right, &r_v);

        let angle_diff = listing.dot(&additive).clamp(-1.0, 1.0).acos();
        assert_abs_diff_eq!(angle_diff, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_visual_axis_listing_differs_from_additive_on_pitch_axis() {
        let params = EyeParams::default();
        let r_v = UnitQuaternion::identity();
        let optical = direction_from_yaw_pitch(&Vector2::new(0.0, 20f64.to_radians()));

        let listing = visual_axis(&optical, &params.kappa, Side::Right, &r_v);
        let additive = additive_visual_axis(&optical, &params.kappa, Side::Right, &r_v);

        let angle_diff = listing.dot(&additive).clamp(-1.0, 1.0).acos();
        assert_abs_diff_eq!(
            angle_diff,
            0.354f64.to_radians(),
            epsilon = 0.02f64.to_radians()
        );
    }

    #[test]
    fn test_visual_axis_listing_differs_from_additive_off_axes() {
        let params = EyeParams::default();
        let r_v = UnitQuaternion::identity();

        let optical =
            direction_from_yaw_pitch(&Vector2::new(25f64.to_radians(), 25f64.to_radians()));
        let listing = visual_axis(&optical, &params.kappa, Side::Right, &r_v);
        let additive = additive_visual_axis(&optical, &params.kappa, Side::Right, &r_v);
        let angle_diff = listing.dot(&additive).clamp(-1.0, 1.0).acos();
        assert_abs_diff_eq!(
            angle_diff,
            0.534f64.to_radians(),
            epsilon = 0.05f64.to_radians()
        );

        let optical_swapped =
            direction_from_yaw_pitch(&Vector2::new(-25f64.to_radians(), 25f64.to_radians()));
        let listing_swapped = visual_axis(&optical_swapped, &params.kappa, Side::Right, &r_v);
        let additive_swapped =
            additive_visual_axis(&optical_swapped, &params.kappa, Side::Right, &r_v);
        let angle_diff_swapped = listing_swapped
            .dot(&additive_swapped)
            .clamp(-1.0, 1.0)
            .acos();
        assert_abs_diff_eq!(
            angle_diff_swapped,
            0.869f64.to_radians(),
            epsilon = 0.05f64.to_radians()
        );
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

        let ray = gaze_ray(
            Side::Right,
            &centre,
            &cam,
            &pupil_px,
            &params,
            Some(&r_v),
            Timestamp::from_nanos(0),
        )
        .unwrap();

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
        let predicted = gaze_ray(
            Side::Right,
            &centre,
            &cam,
            &pupil_px,
            &params,
            Some(&r_v),
            Timestamp::from_nanos(0),
        )
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
            let ray = gaze_ray(
                Side::Right,
                &trial_centre,
                &cam,
                &trial_px,
                &params,
                Some(&r_v),
                Timestamp::from_nanos(0),
            )
            .unwrap();
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
        let ray = gaze_ray(
            Side::Right,
            &centre,
            &cam,
            &pupil_px,
            &params,
            Some(&r_v),
            Timestamp::from_nanos(0),
        )
        .unwrap();
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
        let ray = gaze_ray(
            Side::Right,
            &centre,
            &cam,
            &pupil_px,
            &params,
            Some(&r_v),
            Timestamp::from_nanos(0),
        )
        .unwrap();
        assert_eq!(ray.origin_cov, e_cov);
        assert_eq!(ray.side, Some(Side::Right));
    }

    #[test]
    fn test_gaze_ray_carries_timestamp_and_optional_pose() {
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
        let rot = UnitQuaternion::identity();

        let with_pose = gaze_ray(
            Side::Right,
            &centre,
            &cam,
            &pupil_px,
            &params,
            Some(&rot),
            Timestamp::from_nanos(42),
        )
        .unwrap();
        assert_eq!(with_pose.timestamp, Timestamp::from_nanos(42));
        assert_eq!(with_pose.head_rotation, Some(rot));

        let without_pose = gaze_ray(
            Side::Right,
            &centre,
            &cam,
            &pupil_px,
            &params,
            None,
            Timestamp::from_nanos(42),
        )
        .unwrap();
        assert_eq!(without_pose.head_rotation, None);
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

    fn synth_pccr_pixels(
        cam: &CameraModel,
        params: &EyeParams,
        e: &Point3<f64>,
        axis: &Unit<Vector3<f64>>,
    ) -> (Point2<f64>, Point2<f64>) {
        let o_cam = Point3::from(cam.screen_from_camera.translation.vector);
        let p = e + axis.into_inner() * params.rotation_to_pupil_mm;
        let k = e + axis.into_inner() * params.rotation_to_cornea_mm();
        let glint_screen = k + (o_cam - k).normalize() * params.cornea_radius_mm;

        let pupil_cam = cam.screen_from_camera.inverse_transform_point(&p);
        let glint_cam = cam
            .screen_from_camera
            .inverse_transform_point(&glint_screen);
        let intrinsics = Intrinsics::from_camera_model(cam);
        (
            intrinsics.project(&pupil_cam).unwrap(),
            intrinsics.project(&glint_cam).unwrap(),
        )
    }

    #[test]
    fn test_gaze_ray_pccr_recovers_true_axis_noise_free() {
        let cam = fixture_camera();
        let params = zero_kappa_params();
        let e = Point3::new(185.0, 60.0, -500.0);
        let target = Point3::new(100.0, 50.0, 0.0);
        let axis = Unit::new_normalize(target - e);
        let (pupil_pixel, glint_pixel) = synth_pccr_pixels(&cam, &params, &e, &axis);

        let centre = EyeCentre {
            position: e,
            cov: Matrix3::zeros(),
        };
        let pupil_px = Measured::new(pupil_pixel, 1e-9).unwrap();
        let glint_px = Measured::new(glint_pixel, 1e-9).unwrap();

        let ray = gaze_ray_pccr(
            Side::Right,
            &centre,
            &cam,
            &pupil_px,
            &glint_px,
            &params,
            None,
            Timestamp::from_nanos(42),
        )
        .unwrap();

        assert_abs_diff_eq!(ray.direction, axis, epsilon = 1e-9);

        let t = -ray.origin.z / ray.direction.z;
        let hit = ray.origin + ray.direction.into_inner() * t;
        assert_abs_diff_eq!(hit, target, epsilon = 1e-6);
        assert_eq!(ray.timestamp, Timestamp::from_nanos(42));
        assert_eq!(ray.head_rotation, None);
    }

    #[test]
    fn test_gaze_ray_pccr_lateral_centre_sensitivity_is_second_order() {
        let cam = fixture_camera();
        let params = zero_kappa_params();
        let e = Point3::new(185.0, 60.0, -500.0);
        let target = Point3::new(100.0, 50.0, 0.0);
        let axis = Unit::new_normalize(target - e);
        let (pupil_pixel, glint_pixel) = synth_pccr_pixels(&cam, &params, &e, &axis);
        let r_v = UnitQuaternion::identity();

        let lateral_cov = Matrix3::from_diagonal(&Vector3::new(1.0, 0.0, 0.0));
        let centre = EyeCentre {
            position: e,
            cov: lateral_cov,
        };
        let pupil_px = Measured::new(pupil_pixel, 1e-9).unwrap();
        let glint_px = Measured::new(glint_pixel, 1e-9).unwrap();

        let pccr_ray = gaze_ray_pccr(
            Side::Right,
            &centre,
            &cam,
            &pupil_px,
            &glint_px,
            &params,
            Some(&r_v),
            Timestamp::from_nanos(0),
        )
        .unwrap();
        let pccr_sigma_yaw = pccr_ray.angular_cov[(0, 0)].sqrt();
        assert!(
            pccr_sigma_yaw < 0.1f64.to_radians(),
            "pccr sigma_yaw {} rad",
            pccr_sigma_yaw
        );

        let pupil_sphere_ray = gaze_ray(
            Side::Right,
            &centre,
            &cam,
            &pupil_px,
            &params,
            Some(&r_v),
            Timestamp::from_nanos(0),
        )
        .unwrap();
        let pupil_sphere_sigma_yaw = pupil_sphere_ray.angular_cov[(0, 0)].sqrt();
        assert_abs_diff_eq!(
            pupil_sphere_sigma_yaw,
            1.0 / 10.46,
            epsilon = 0.15 * (1.0 / 10.46)
        );
    }

    #[test]
    fn test_gaze_ray_pccr_depth_sensitivity_below_0_3_deg() {
        let cam = fixture_camera();
        let params = zero_kappa_params();
        let e = Point3::new(185.0, 60.0, -500.0);
        let target = Point3::new(100.0, 50.0, 0.0);
        let axis = Unit::new_normalize(target - e);
        let (pupil_pixel, glint_pixel) = synth_pccr_pixels(&cam, &params, &e, &axis);

        let depth_cov = Matrix3::from_diagonal(&Vector3::new(0.0, 0.0, 121.0));
        let centre = EyeCentre {
            position: e,
            cov: depth_cov,
        };
        let pupil_px = Measured::new(pupil_pixel, 1e-9).unwrap();
        let glint_px = Measured::new(glint_pixel, 1e-9).unwrap();

        let ray = gaze_ray_pccr(
            Side::Right,
            &centre,
            &cam,
            &pupil_px,
            &glint_px,
            &params,
            None,
            Timestamp::from_nanos(0),
        )
        .unwrap();
        let sigma = (ray.angular_cov[(0, 0)] + ray.angular_cov[(1, 1)]).sqrt();
        assert!(sigma < 0.3f64.to_radians(), "sigma {} rad", sigma);
    }

    #[test]
    fn test_gaze_ray_pccr_pixel_sensitivity_is_one_over_k_d() {
        let cam = fixture_camera();
        let params = zero_kappa_params();
        let e = Point3::new(185.0, 60.0, -500.0);
        let target = Point3::new(100.0, 50.0, 0.0);
        let axis = Unit::new_normalize(target - e);
        let (pupil_pixel, glint_pixel) = synth_pccr_pixels(&cam, &params, &e, &axis);

        let centre = EyeCentre {
            position: e,
            cov: Matrix3::zeros(),
        };
        let expected = (500.0 / 457.0) / params.cornea_to_pupil_mm();

        let pupil_px = Measured::new(pupil_pixel, 1.0).unwrap();
        let glint_px = Measured::new(glint_pixel, 1e-9).unwrap();
        let ray = gaze_ray_pccr(
            Side::Right,
            &centre,
            &cam,
            &pupil_px,
            &glint_px,
            &params,
            None,
            Timestamp::from_nanos(0),
        )
        .unwrap();
        let sigma_yaw = ray.angular_cov[(0, 0)].sqrt();
        assert_abs_diff_eq!(sigma_yaw, expected, epsilon = 0.15 * expected);

        let pupil_px_quiet = Measured::new(pupil_pixel, 1e-9).unwrap();
        let glint_px_noisy = Measured::new(glint_pixel, 1.0).unwrap();
        let ray_swapped = gaze_ray_pccr(
            Side::Right,
            &centre,
            &cam,
            &pupil_px_quiet,
            &glint_px_noisy,
            &params,
            None,
            Timestamp::from_nanos(0),
        )
        .unwrap();
        let sigma_yaw_swapped = ray_swapped.angular_cov[(0, 0)].sqrt();
        assert_abs_diff_eq!(sigma_yaw_swapped, expected, epsilon = 0.15 * expected);
    }

    #[test]
    fn test_gaze_ray_pccr_cov_matches_monte_carlo() {
        let cam = fixture_camera();
        let params = zero_kappa_params();
        let e = Point3::new(185.0, 60.0, -500.0);
        let target = Point3::new(100.0, 50.0, 0.0);
        let axis = Unit::new_normalize(target - e);
        let (pupil_pixel, glint_pixel) = synth_pccr_pixels(&cam, &params, &e, &axis);

        let pupil_sigma_px = 0.3;
        let glint_sigma_px = 0.3;
        let e_cov = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, 25.0));
        let centre = EyeCentre {
            position: e,
            cov: e_cov,
        };
        let pupil_px = Measured::new(pupil_pixel, pupil_sigma_px).unwrap();
        let glint_px = Measured::new(glint_pixel, glint_sigma_px).unwrap();
        let predicted = gaze_ray_pccr(
            Side::Right,
            &centre,
            &cam,
            &pupil_px,
            &glint_px,
            &params,
            None,
            Timestamp::from_nanos(0),
        )
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
            let noisy_pupil = Point2::new(
                pupil_pixel.x + pupil_sigma_px * rng.gaussian(),
                pupil_pixel.y + pupil_sigma_px * rng.gaussian(),
            );
            let noisy_glint = Point2::new(
                glint_pixel.x + glint_sigma_px * rng.gaussian(),
                glint_pixel.y + glint_sigma_px * rng.gaussian(),
            );
            let trial_centre = EyeCentre {
                position: noisy_e,
                cov: Matrix3::zeros(),
            };
            let trial_pupil_px = Measured::new(noisy_pupil, 1e-12).unwrap();
            let trial_glint_px = Measured::new(noisy_glint, 1e-12).unwrap();
            let ray = gaze_ray_pccr(
                Side::Right,
                &trial_centre,
                &cam,
                &trial_pupil_px,
                &trial_glint_px,
                &params,
                None,
                Timestamp::from_nanos(0),
            )
            .unwrap();
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
}
