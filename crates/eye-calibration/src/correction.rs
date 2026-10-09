use std::collections::BTreeMap;

use eye_core::log::field;
use eye_core::stage::GazeCorrection;
use eye_core::{GazeRay, Rig, Side};
use eye_geometry::angles::{direction_from_yaw_pitch, yaw_pitch_from_direction};
use eye_geometry::uncertainty::{block_diag, numeric_jacobian, propagate};
use nalgebra::{Matrix2, SMatrix, SVector, Vector2};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum EyeKey {
    Left,
    Right,
    Cyclopean,
}

pub(crate) fn eye_label(k: EyeKey) -> &'static str {
    match k {
        EyeKey::Left => "left",
        EyeKey::Right => "right",
        EyeKey::Cyclopean => "cyclopean",
    }
}

impl From<Option<Side>> for EyeKey {
    fn from(side: Option<Side>) -> Self {
        match side {
            Some(Side::Left) => Self::Left,
            Some(Side::Right) => Self::Right,
            None => Self::Cyclopean,
        }
    }
}

/// `yaw'   = yaw   + a0 + a1*yaw + a2*pitch`
/// `pitch' = pitch + b0 + b1*yaw + b2*pitch`          (radians; theta = [a0, a1, a2, b0, b1, b2])
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AngularCorrection {
    pub theta: [f64; 6],
    pub cov: [[f64; 6]; 6],
    /// `Quadratic` only: `[c0, c1, c2, d0, d1, d2]` added to `theta`'s map as
    /// `+ c0*yaw^2 + c1*pitch^2 + c2*yaw*pitch` (resp. `d.. ` for pitch'). Zero for every other model.
    #[serde(default)]
    pub quad: [f64; 6],
    #[serde(default)]
    pub quad_cov: [[f64; 6]; 6],
    /// `Quadratic` only: the `cov[r][k] = Cov(theta_r, quad_k)` cross block of the joint 12x12
    /// fit covariance. Zero for every other model.
    #[serde(default)]
    pub quad_cross_cov: [[f64; 6]; 6],
    pub model: CorrectionModel,
    pub targets_used: u32,
    pub rms_after_rad: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CorrectionModel {
    Affine,
    OffsetOnly,
    /// Second-order polynomial in `(yaw, pitch)`, screen frame: `theta`'s affine map plus `quad`.
    Quadratic,
    /// `theta`'s affine map applied in the head frame (`GazeRay::head_rotation`), not the screen
    /// frame: rotate the ray into the head frame, apply, rotate back.
    HeadFrame,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UserProfile {
    pub version: u32,
    pub name: String,
    pub created_unix_s: u64,
    pub rig_fingerprint: String,
    pub estimator: String,
    pub eyes: BTreeMap<EyeKey, AngularCorrection>,
}

pub(crate) fn design(o: &Vector2<f64>) -> SMatrix<f64, 2, 6> {
    SMatrix::<f64, 2, 6>::new(1.0, o.x, o.y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, o.x, o.y)
}

pub(crate) fn design_quad(o: &Vector2<f64>) -> SMatrix<f64, 2, 6> {
    let (yy, pp, yp) = (o.x * o.x, o.y * o.y, o.x * o.y);
    SMatrix::<f64, 2, 6>::new(yy, pp, yp, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, yy, pp, yp)
}

pub(crate) fn design12(o: &Vector2<f64>) -> SMatrix<f64, 2, 12> {
    let mut d = SMatrix::<f64, 2, 12>::zeros();
    d.fixed_view_mut::<2, 6>(0, 0).copy_from(&design(o));
    d.fixed_view_mut::<2, 6>(0, 6).copy_from(&design_quad(o));
    d
}

fn head_frame_correct(
    rot: &nalgebra::UnitQuaternion<f64>,
    a: &Vector2<f64>,
    th: &SVector<f64, 6>,
) -> Vector2<f64> {
    let dir_head = rot.inverse() * direction_from_yaw_pitch(a);
    let a_head = yaw_pitch_from_direction(&dir_head);
    let corrected_head = a_head + design(&a_head) * th;
    yaw_pitch_from_direction(&(*rot * direction_from_yaw_pitch(&corrected_head)))
}

impl GazeCorrection for UserProfile {
    fn correct(&self, ray: &GazeRay) -> GazeRay {
        let eye = EyeKey::from(ray.side);
        let Some(c) = self.eyes.get(&eye) else {
            tracing::trace!(
                eye = eye_label(eye),
                { field::REASON } = "no_profile_entry",
                "ray uncorrected"
            );
            return ray.clone();
        };
        let a = yaw_pitch_from_direction(&ray.direction);
        let (corrected, angular_cov) = match c.model {
            CorrectionModel::Affine | CorrectionModel::OffsetOnly => {
                let th = SVector::<f64, 6>::from(c.theta);
                let b = design(&a);
                let j = Matrix2::new(1.0 + th[1], th[2], th[4], 1.0 + th[5]);
                let cov_theta = SMatrix::<f64, 6, 6>::from_fn(|r, k| c.cov[r][k]);
                (
                    a + b * th,
                    propagate(&j, &ray.angular_cov) + propagate(&b, &cov_theta),
                )
            }
            CorrectionModel::Quadratic => {
                let th = SVector::<f64, 6>::from(c.theta);
                let q = SVector::<f64, 6>::from(c.quad);
                let b = design(&a);
                let bq = design_quad(&a);
                let j = Matrix2::new(
                    1.0 + th[1] + 2.0 * q[0] * a.x + q[2] * a.y,
                    th[2] + 2.0 * q[1] * a.y + q[2] * a.x,
                    th[4] + 2.0 * q[3] * a.x + q[5] * a.y,
                    1.0 + th[5] + 2.0 * q[4] * a.y + q[5] * a.x,
                );
                let b12 = design12(&a);
                let cov_theta = SMatrix::<f64, 6, 6>::from_fn(|r, k| c.cov[r][k]);
                let cov_quad = SMatrix::<f64, 6, 6>::from_fn(|r, k| c.quad_cov[r][k]);
                let cross = SMatrix::<f64, 6, 6>::from_fn(|r, k| c.quad_cross_cov[r][k]);
                let mut cov12 = SMatrix::<f64, 12, 12>::zeros();
                cov12.fixed_view_mut::<6, 6>(0, 0).copy_from(&cov_theta);
                cov12.fixed_view_mut::<6, 6>(6, 6).copy_from(&cov_quad);
                cov12.fixed_view_mut::<6, 6>(0, 6).copy_from(&cross);
                cov12
                    .fixed_view_mut::<6, 6>(6, 0)
                    .copy_from(&cross.transpose());
                (
                    a + b * th + bq * q,
                    propagate(&j, &ray.angular_cov) + propagate(&b12, &cov12),
                )
            }
            CorrectionModel::HeadFrame => {
                let th = SVector::<f64, 6>::from(c.theta);
                let cov_theta = SMatrix::<f64, 6, 6>::from_fn(|r, k| c.cov[r][k]);
                let rot = ray.head_rotation;
                let x = SVector::<f64, 8>::from_fn(|i, _| if i < 2 { a[i] } else { th[i - 2] });
                let cov8 = block_diag::<2, 6, 8>(&ray.angular_cov, &cov_theta);
                let f = move |x: &SVector<f64, 8>| -> Option<SVector<f64, 2>> {
                    let ao = Vector2::new(x[0], x[1]);
                    let theta = SVector::<f64, 6>::from_fn(|i, _| x[2 + i]);
                    Some(head_frame_correct(&rot, &ao, &theta))
                };
                let j = numeric_jacobian::<2, 8>(f, &x).unwrap_or_else(SMatrix::zeros);
                (head_frame_correct(&rot, &a, &th), propagate(&j, &cov8))
            }
        };
        tracing::trace!(
            eye = eye_label(eye),
            model = ?c.model,
            yaw_in_deg = a.x.to_degrees(),
            pitch_in_deg = a.y.to_degrees(),
            yaw_out_deg = corrected.x.to_degrees(),
            pitch_out_deg = corrected.y.to_degrees(),
            "ray corrected"
        );
        GazeRay {
            direction: direction_from_yaw_pitch(&corrected),
            angular_cov,
            ..ray.clone()
        }
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv_feed_bytes(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn fnv_feed_f64(hash: &mut u64, v: f64) {
    fnv_feed_bytes(hash, &v.to_bits().to_le_bytes());
}

fn fnv_feed_u32(hash: &mut u64, v: u32) {
    fnv_feed_bytes(hash, &v.to_le_bytes());
}

pub fn rig_fingerprint(rig: &Rig) -> String {
    let mut hash = FNV_OFFSET;

    let screen = rig.screen();
    fnv_feed_bytes(&mut hash, screen.output.as_str().as_bytes());
    fnv_feed_bytes(&mut hash, &[0xFF]);
    fnv_feed_f64(&mut hash, screen.size_mm.x);
    fnv_feed_f64(&mut hash, screen.size_mm.y);
    fnv_feed_u32(&mut hash, screen.size_px.0);
    fnv_feed_u32(&mut hash, screen.size_px.1);
    fnv_feed_f64(&mut hash, screen.scale);

    for camera in rig.cameras() {
        fnv_feed_bytes(&mut hash, camera.id.as_str().as_bytes());
        fnv_feed_bytes(&mut hash, &[0xFF]);
        fnv_feed_u32(&mut hash, camera.width);
        fnv_feed_u32(&mut hash, camera.height);
        fnv_feed_f64(&mut hash, camera.fx);
        fnv_feed_f64(&mut hash, camera.fy);
        fnv_feed_f64(&mut hash, camera.cx);
        fnv_feed_f64(&mut hash, camera.cy);
        for d in camera.distortion {
            fnv_feed_f64(&mut hash, d);
        }
        let pose = &camera.screen_from_camera;
        fnv_feed_f64(&mut hash, pose.rotation.coords.x);
        fnv_feed_f64(&mut hash, pose.rotation.coords.y);
        fnv_feed_f64(&mut hash, pose.rotation.coords.z);
        fnv_feed_f64(&mut hash, pose.rotation.coords.w);
        fnv_feed_f64(&mut hash, pose.translation.vector.x);
        fnv_feed_f64(&mut hash, pose.translation.vector.y);
        fnv_feed_f64(&mut hash, pose.translation.vector.z);
    }

    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use approx::assert_abs_diff_eq;
    use eye_core::{CameraId, CameraModel, OutputId, ScreenModel};
    use nalgebra::{Point3, SMatrix, Unit, UnitQuaternion, Vector2, Vector3};

    use super::*;
    use crate::nominal::nominal_screen_from_camera;

    const RIG_FINGERPRINT_FIXTURE: &str = "768e389a57cfc942";

    fn fixture_rig() -> Rig {
        let screen = ScreenModel {
            output: OutputId::from("eDP-1"),
            size_mm: Vector2::new(310.0, 174.375),
            size_px: (3840, 2160),
            scale: 2.0,
        };
        let rgb = CameraModel {
            id: CameraId::from("rgb"),
            width: 1280,
            height: 720,
            fx: 858.5,
            fy: 858.5,
            cx: 640.0,
            cy: 360.0,
            distortion: [0.0; 5],
            screen_from_camera: nominal_screen_from_camera(&Point3::new(155.0, -7.0, 0.0)),
        };
        let ir = CameraModel {
            id: CameraId::from("ir"),
            width: 640,
            height: 360,
            fx: 429.25,
            fy: 429.25,
            cx: 320.0,
            cy: 180.0,
            distortion: [0.08, -0.15, 0.001, -0.0005, 0.0],
            screen_from_camera: nominal_screen_from_camera(&Point3::new(155.0, -7.0, 0.0)),
        };
        Rig::new(vec![rgb, ir], screen).unwrap()
    }

    fn fixture_profile() -> UserProfile {
        let mut eyes = BTreeMap::new();
        eyes.insert(
            EyeKey::Right,
            AngularCorrection {
                theta: [0.01, -0.02, 0.0, 0.03, 0.0, -0.01],
                cov: [[0.0; 6]; 6],
                quad: [0.0; 6],
                quad_cov: [[0.0; 6]; 6],
                quad_cross_cov: [[0.0; 6]; 6],
                model: CorrectionModel::Affine,
                targets_used: 9,
                rms_after_rad: 0.001,
            },
        );
        UserProfile {
            version: 1,
            name: "max".into(),
            created_unix_s: 1_700_000_000,
            rig_fingerprint: "deadbeefcafef00d".into(),
            estimator: "ir-pupil".into(),
            eyes,
        }
    }

    fn straight_ray(side: Option<eye_core::Side>) -> GazeRay {
        GazeRay {
            side,
            origin: Point3::new(155.0, 85.0, -500.0),
            direction: Unit::new_normalize(Vector3::new(0.1, -0.05, 1.0)),
            angular_cov: Matrix2::identity() * 1e-4,
            origin_cov: nalgebra::Matrix3::zeros(),
            head_rotation: UnitQuaternion::identity(),
        }
    }

    #[test]
    fn test_missing_eye_passes_through() {
        let profile = fixture_profile();
        let ray = straight_ray(Some(eye_core::Side::Left));
        let corrected = profile.correct(&ray);
        assert_eq!(corrected, ray);
    }

    #[test]
    fn test_logs_ray_corrected_at_trace() {
        let profile = fixture_profile();
        let ray = straight_ray(Some(eye_core::Side::Right));
        let expected_yaw_in = yaw_pitch_from_direction(&ray.direction).x.to_degrees();

        let (corrected, records) =
            eye_log::testing::capture_logs(tracing::Level::TRACE, || profile.correct(&ray));

        let expected_yaw_out = yaw_pitch_from_direction(&corrected.direction)
            .x
            .to_degrees();
        let rec = records
            .iter()
            .find(|r| r.message == "ray corrected")
            .expect("no 'ray corrected' record");
        assert_eq!(rec.level, eye_log::Level::Trace);
        assert_eq!(
            rec.fields.get("eye"),
            Some(&eye_log::Value::Str("right".to_string()))
        );
        assert_eq!(
            rec.fields.get("model"),
            Some(&eye_log::Value::Str("Affine".to_string()))
        );
        match rec.fields.get("yaw_in_deg") {
            Some(eye_log::Value::F64(v)) => {
                assert_abs_diff_eq!(*v, expected_yaw_in, epsilon = 1e-9)
            }
            other => panic!("expected yaw_in_deg F64, got {other:?}"),
        }
        match rec.fields.get("yaw_out_deg") {
            Some(eye_log::Value::F64(v)) => {
                assert_abs_diff_eq!(*v, expected_yaw_out, epsilon = 1e-9)
            }
            other => panic!("expected yaw_out_deg F64, got {other:?}"),
        }
    }

    #[test]
    fn test_logs_ray_uncorrected_at_trace() {
        let profile = fixture_profile();
        let ray = straight_ray(Some(eye_core::Side::Left));

        let (_, records) =
            eye_log::testing::capture_logs(tracing::Level::TRACE, || profile.correct(&ray));

        let rec = records
            .iter()
            .find(|r| r.message == "ray uncorrected")
            .expect("no 'ray uncorrected' record");
        assert_eq!(rec.level, eye_log::Level::Trace);
        assert_eq!(
            rec.fields.get("eye"),
            Some(&eye_log::Value::Str("left".to_string()))
        );
        assert_eq!(
            rec.fields.get(field::REASON),
            Some(&eye_log::Value::Str("no_profile_entry".to_string()))
        );
    }

    #[test]
    fn test_corrected_cov_includes_parameter_uncertainty() {
        let mut eyes = BTreeMap::new();
        let diag = [1e-6, 1e-4, 1e-4, 2e-6, 1e-4, 1e-4];
        let mut cov = [[0.0; 6]; 6];
        for (i, v) in diag.into_iter().enumerate() {
            cov[i][i] = v;
        }
        eyes.insert(
            EyeKey::Right,
            AngularCorrection {
                theta: [0.0; 6],
                cov,
                quad: [0.0; 6],
                quad_cov: [[0.0; 6]; 6],
                quad_cross_cov: [[0.0; 6]; 6],
                model: CorrectionModel::Affine,
                targets_used: 9,
                rms_after_rad: 0.0,
            },
        );
        let profile = UserProfile {
            version: 1,
            name: "max".into(),
            created_unix_s: 0,
            rig_fingerprint: String::new(),
            estimator: "ir-pupil".into(),
            eyes,
        };
        let ray = straight_ray(Some(eye_core::Side::Right));
        let corrected = profile.correct(&ray);

        let a = yaw_pitch_from_direction(&ray.direction);
        let b = design(&a);
        let cov_theta = SMatrix::<f64, 6, 6>::from_fn(|r, k| cov[r][k]);
        let expected = ray.angular_cov + propagate(&b, &cov_theta);
        assert_abs_diff_eq!(corrected.angular_cov, expected, epsilon = 1e-15);
    }

    #[test]
    fn test_rig_fingerprint_stable_and_sensitive() {
        let rig = fixture_rig();
        assert_eq!(rig_fingerprint(&rig), RIG_FINGERPRINT_FIXTURE);

        let mut cameras = rig.cameras().to_vec();
        cameras[1].fx += 1e-9;
        let perturbed = Rig::new(cameras, rig.screen().clone()).unwrap();
        assert_ne!(rig_fingerprint(&perturbed), RIG_FINGERPRINT_FIXTURE);

        let mut cameras = rig.cameras().to_vec();
        cameras[1].id = CameraId::from("ir2");
        let renamed = Rig::new(cameras, rig.screen().clone()).unwrap();
        assert_ne!(rig_fingerprint(&renamed), RIG_FINGERPRINT_FIXTURE);
    }

    #[test]
    fn test_profile_toml_round_trip() {
        let profile = fixture_profile();
        let text = toml::to_string(&profile).unwrap();
        let back: UserProfile = toml::from_str(&text).unwrap();
        assert_eq!(back, profile);

        let mut with_extra = text;
        with_extra.push_str("\nfoo = 1\n");
        assert!(toml::from_str::<UserProfile>(&with_extra).is_err());
    }

    fn single_eye_profile(model: CorrectionModel, theta: [f64; 6]) -> UserProfile {
        let mut eyes = BTreeMap::new();
        eyes.insert(
            EyeKey::Right,
            AngularCorrection {
                theta,
                cov: [[0.0; 6]; 6],
                quad: [0.0; 6],
                quad_cov: [[0.0; 6]; 6],
                quad_cross_cov: [[0.0; 6]; 6],
                model,
                targets_used: 9,
                rms_after_rad: 0.0,
            },
        );
        UserProfile {
            version: 1,
            name: "x".into(),
            created_unix_s: 0,
            rig_fingerprint: String::new(),
            estimator: "landmark".into(),
            eyes,
        }
    }

    #[test]
    fn test_head_frame_equals_affine_at_identity_head_rotation() {
        let theta = [0.02, -0.01, 0.0, -0.015, 0.0, 0.02];
        let head_profile = single_eye_profile(CorrectionModel::HeadFrame, theta);
        let affine_profile = single_eye_profile(CorrectionModel::Affine, theta);

        let ray = straight_ray(Some(eye_core::Side::Right));
        let a = yaw_pitch_from_direction(&head_profile.correct(&ray).direction);
        let b = yaw_pitch_from_direction(&affine_profile.correct(&ray).direction);
        assert_abs_diff_eq!(a.x, b.x, epsilon = 1e-9);
        assert_abs_diff_eq!(a.y, b.y, epsilon = 1e-9);
    }

    #[test]
    fn test_head_frame_tracks_head_rotation() {
        let theta = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0];
        let profile = single_eye_profile(CorrectionModel::HeadFrame, theta);

        let mut ray = straight_ray(Some(eye_core::Side::Right));
        ray.head_rotation =
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 60.0_f64.to_radians());
        let corrected = yaw_pitch_from_direction(&profile.correct(&ray).direction);

        let th = SVector::<f64, 6>::from(theta);
        let dir_head = ray.head_rotation.inverse() * ray.direction;
        let a_head = yaw_pitch_from_direction(&dir_head);
        let corrected_head = a_head + design(&a_head) * th;
        let expected = yaw_pitch_from_direction(
            &(ray.head_rotation * direction_from_yaw_pitch(&corrected_head)),
        );

        assert_abs_diff_eq!(corrected.x, expected.x, epsilon = 1e-9);
        assert_abs_diff_eq!(corrected.y, expected.y, epsilon = 1e-9);

        let a = yaw_pitch_from_direction(&ray.direction);
        let naive = a + design(&a) * th;
        assert!(
            (corrected - naive).norm() > 0.02,
            "{corrected:?} vs {naive:?}"
        );
    }

    #[test]
    fn test_quadratic_correction_includes_cross_covariance_term() {
        let mut cov = [[0.0; 6]; 6];
        cov[0][0] = 1e-4;
        cov[3][3] = 1e-4;
        let mut quad_cov = [[0.0; 6]; 6];
        quad_cov[0][0] = 1e-3;
        quad_cov[3][3] = 1e-3;
        let mut quad_cross_cov = [[0.0; 6]; 6];
        quad_cross_cov[0][0] = 5e-4;
        quad_cross_cov[3][3] = -5e-4;

        let mut eyes = BTreeMap::new();
        eyes.insert(
            EyeKey::Right,
            AngularCorrection {
                theta: [0.01, 0.0, 0.0, -0.01, 0.0, 0.0],
                cov,
                quad: [0.05, -0.03, 0.0, 0.02, 0.0, 0.0],
                quad_cov,
                quad_cross_cov,
                model: CorrectionModel::Quadratic,
                targets_used: 16,
                rms_after_rad: 0.0,
            },
        );
        let profile = UserProfile {
            version: 1,
            name: "x".into(),
            created_unix_s: 0,
            rig_fingerprint: String::new(),
            estimator: "landmark".into(),
            eyes,
        };

        let mut ray = straight_ray(Some(eye_core::Side::Right));
        ray.angular_cov = Matrix2::zeros();
        let corrected = profile.correct(&ray);

        let a = yaw_pitch_from_direction(&ray.direction);
        let b12 = design12(&a);
        let cov_theta = SMatrix::<f64, 6, 6>::from_fn(|r, k| cov[r][k]);
        let cov_quad = SMatrix::<f64, 6, 6>::from_fn(|r, k| quad_cov[r][k]);
        let cross = SMatrix::<f64, 6, 6>::from_fn(|r, k| quad_cross_cov[r][k]);
        let mut cov12 = SMatrix::<f64, 12, 12>::zeros();
        cov12.fixed_view_mut::<6, 6>(0, 0).copy_from(&cov_theta);
        cov12.fixed_view_mut::<6, 6>(6, 6).copy_from(&cov_quad);
        cov12.fixed_view_mut::<6, 6>(0, 6).copy_from(&cross);
        cov12
            .fixed_view_mut::<6, 6>(6, 0)
            .copy_from(&cross.transpose());
        let expected = b12 * cov12 * b12.transpose();

        assert_abs_diff_eq!(corrected.angular_cov, expected, epsilon = 1e-15);

        let block_diag_only =
            propagate(&design(&a), &cov_theta) + propagate(&design_quad(&a), &cov_quad);
        assert!(
            (corrected.angular_cov - block_diag_only).abs().sum() > 1e-9,
            "cross term had no effect"
        );
    }
}
