use std::collections::BTreeMap;

use eye_core::log::field;
use eye_core::stage::GazeCorrection;
use eye_core::{GazeRay, RaySource, Rig, Side};
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
    /// RMS of the post-fit residual over the fitted targets, radians. Added as an isotropic
    /// floor to every corrected ray's `angular_cov`.
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
pub struct CalibrationPose {
    /// Normalised mean of the fitted samples' head rotations, unit quaternion `[x, y, z, w]`
    /// (screen frame); `None` when no fitted sample carried a pose.
    pub head_rotation: Option<[f64; 4]>,
    pub samples_with_head_pose: u32,
    pub samples_total: u32,
    /// Mean ray origin per eye, screen frame mm.
    pub eye_origin_mm: BTreeMap<EyeKey, [f64; 3]>,
}

#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Provenance {
    pub session_id: Option<String>,
    pub git_rev: Option<String>,
    /// `text_fingerprint` of the serialized `[estimate]` table the rays were produced with.
    pub estimator_fingerprint: Option<String>,
    pub protocol: Option<eye_core::session::ProtocolConfig>,
    pub fit: Option<crate::user_fit::FitConfig>,
    /// Bench leave-one-target-out mean angular error on the fitted session, degrees.
    pub expected_loto_mean_deg: Option<f64>,
}

pub const PROFILE_VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UserProfile {
    pub version: u32,
    pub name: String,
    pub created_unix_s: u64,
    pub rig_fingerprint: String,
    pub estimator: String,
    /// `[corrections.<source>.<eye>]`
    pub corrections: BTreeMap<RaySource, BTreeMap<EyeKey, AngularCorrection>>,
    #[serde(default)]
    pub calibration_pose: Option<CalibrationPose>,
    #[serde(default)]
    pub provenance: Provenance,
}

impl UserProfile {
    pub fn correction(&self, source: RaySource, eye: EyeKey) -> Option<&AngularCorrection> {
        self.corrections.get(&source)?.get(&eye)
    }

    /// Number of (source, eye) entries.
    pub fn correction_count(&self) -> usize {
        self.corrections.values().map(BTreeMap::len).sum()
    }

    /// The lowest-ordered source present (`RaySource::RgbOnly` when present).
    pub fn primary_source(&self) -> Option<RaySource> {
        self.corrections.keys().min().copied()
    }

    /// The ray corrected by the (source, eye) entry. `None` when there is no entry, or the entry
    /// is `HeadFrame` and the ray carries no head pose: the ray cannot be put in true-angle units.
    pub fn correct_source(&self, source: RaySource, ray: &GazeRay) -> Option<GazeRay> {
        let eye = EyeKey::from(ray.side);
        let Some(c) = self.correction(source, eye) else {
            tracing::trace!(
                source = source.as_str(),
                eye = eye_label(eye),
                { field::REASON } = "no_profile_entry",
                "ray uncorrected"
            );
            return None;
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
                let Some(rot) = ray.head_rotation else {
                    tracing::trace!(
                        source = source.as_str(),
                        eye = eye_label(eye),
                        { field::REASON } = "no_head_pose",
                        "ray uncorrected"
                    );
                    return None;
                };
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
        let residual = Matrix2::identity() * (c.rms_after_rad * c.rms_after_rad);
        let angular_cov = angular_cov + residual;
        tracing::trace!(
            source = source.as_str(),
            eye = eye_label(eye),
            model = ?c.model,
            yaw_in_deg = a.x.to_degrees(),
            pitch_in_deg = a.y.to_degrees(),
            yaw_out_deg = corrected.x.to_degrees(),
            pitch_out_deg = corrected.y.to_degrees(),
            "ray corrected"
        );
        Some(GazeRay {
            direction: direction_from_yaw_pitch(&corrected),
            angular_cov,
            ..ray.clone()
        })
    }
}

/// The source a v1 profile's `eyes` (fitted on the estimator's output rays) describes.
pub fn legacy_source(estimator: &str) -> RaySource {
    match estimator {
        "pccr" | "ir-pupil" => RaySource::IrOnly,
        _ => RaySource::RgbOnly,
    }
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
    fn correct(&self, source: RaySource, ray: &GazeRay) -> Option<GazeRay> {
        self.correct_source(source, ray)
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

pub fn text_fingerprint(text: &str) -> String {
    let mut hash = FNV_OFFSET;
    fnv_feed_bytes(&mut hash, text.as_bytes());
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
            version: PROFILE_VERSION,
            name: "max".into(),
            created_unix_s: 1_700_000_000,
            rig_fingerprint: "deadbeefcafef00d".into(),
            estimator: "ir-pupil".into(),
            corrections: BTreeMap::from([(legacy_source("ir-pupil"), eyes)]),
            calibration_pose: None,
            provenance: Provenance::default(),
        }
    }

    fn straight_ray(side: Option<eye_core::Side>) -> GazeRay {
        GazeRay {
            side,
            timestamp: eye_core::Timestamp::from_nanos(0),
            origin: Point3::new(155.0, 85.0, -500.0),
            direction: Unit::new_normalize(Vector3::new(0.1, -0.05, 1.0)),
            angular_cov: Matrix2::identity() * 1e-4,
            origin_cov: nalgebra::Matrix3::zeros(),
            head_rotation: Some(UnitQuaternion::identity()),
        }
    }

    #[test]
    fn test_missing_eye_returns_none() {
        let profile = fixture_profile();
        let ray = straight_ray(Some(eye_core::Side::Left));
        assert_eq!(profile.correct(RaySource::IrOnly, &ray), None);
    }

    #[test]
    fn test_logs_ray_corrected_at_trace() {
        let profile = fixture_profile();
        let ray = straight_ray(Some(eye_core::Side::Right));
        let expected_yaw_in = yaw_pitch_from_direction(&ray.direction).x.to_degrees();

        let (corrected, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            profile.correct(RaySource::IrOnly, &ray).unwrap()
        });

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

        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            profile.correct(RaySource::IrOnly, &ray)
        });

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
            version: PROFILE_VERSION,
            name: "max".into(),
            created_unix_s: 0,
            rig_fingerprint: String::new(),
            estimator: "ir-pupil".into(),
            corrections: BTreeMap::from([(legacy_source("ir-pupil"), eyes)]),
            calibration_pose: None,
            provenance: Provenance::default(),
        };
        let ray = straight_ray(Some(eye_core::Side::Right));
        let corrected = profile.correct(RaySource::IrOnly, &ray).unwrap();

        let a = yaw_pitch_from_direction(&ray.direction);
        let b = design(&a);
        let cov_theta = SMatrix::<f64, 6, 6>::from_fn(|r, k| cov[r][k]);
        let expected = ray.angular_cov + propagate(&b, &cov_theta);
        assert_abs_diff_eq!(corrected.angular_cov, expected, epsilon = 1e-15);
    }

    #[test]
    fn test_corrected_cov_includes_fit_residual_floor() {
        let mut eyes = BTreeMap::new();
        eyes.insert(
            EyeKey::Right,
            AngularCorrection {
                theta: [0.0; 6],
                cov: [[0.0; 6]; 6],
                quad: [0.0; 6],
                quad_cov: [[0.0; 6]; 6],
                quad_cross_cov: [[0.0; 6]; 6],
                model: CorrectionModel::Affine,
                targets_used: 9,
                rms_after_rad: 0.05,
            },
        );
        let profile = UserProfile {
            version: PROFILE_VERSION,
            name: "max".into(),
            created_unix_s: 0,
            rig_fingerprint: String::new(),
            estimator: "ir-pupil".into(),
            corrections: BTreeMap::from([(legacy_source("ir-pupil"), eyes)]),
            calibration_pose: None,
            provenance: Provenance::default(),
        };
        let mut ray = straight_ray(Some(eye_core::Side::Right));
        ray.angular_cov = Matrix2::identity() * 1e-4;
        let corrected = profile.correct(RaySource::IrOnly, &ray).unwrap();

        let expected = Matrix2::identity() * 1e-4 + Matrix2::identity() * 0.0025;
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
        let mut profile = fixture_profile();
        profile.corrections.insert(
            RaySource::RgbOnly,
            BTreeMap::from([(
                EyeKey::Left,
                AngularCorrection {
                    theta: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    cov: [[0.0; 6]; 6],
                    quad: [0.0; 6],
                    quad_cov: [[0.0; 6]; 6],
                    quad_cross_cov: [[0.0; 6]; 6],
                    model: CorrectionModel::Affine,
                    targets_used: 6,
                    rms_after_rad: 0.002,
                },
            )]),
        );
        profile.calibration_pose = Some(CalibrationPose {
            head_rotation: Some([0.0, 0.0, 0.0, 1.0]),
            samples_with_head_pose: 42,
            samples_total: 50,
            eye_origin_mm: BTreeMap::from([(EyeKey::Right, [155.0, 85.0, -500.0])]),
        });
        profile.provenance = Provenance {
            session_id: Some("sess-1".into()),
            git_rev: Some("abc123".into()),
            estimator_fingerprint: Some(text_fingerprint("kind = \"fused\"\n")),
            protocol: None,
            fit: None,
            expected_loto_mean_deg: Some(0.42),
        };
        let text = toml::to_string(&profile).unwrap();
        let back: UserProfile = toml::from_str(&text).unwrap();
        assert_eq!(back, profile);

        let mut with_extra = text;
        with_extra.push_str("\nfoo = 1\n");
        assert!(toml::from_str::<UserProfile>(&with_extra).is_err());
    }

    #[test]
    fn test_text_fingerprint_stable_and_sensitive() {
        const FIXTURE: &str = "8a277ef72b61e921";
        assert_eq!(text_fingerprint("kind = \"fused\"\n"), FIXTURE);
        assert_ne!(text_fingerprint("kind = \"fusee\"\n"), FIXTURE);
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
            version: PROFILE_VERSION,
            name: "x".into(),
            created_unix_s: 0,
            rig_fingerprint: String::new(),
            estimator: "landmark".into(),
            corrections: BTreeMap::from([(legacy_source("landmark"), eyes)]),
            calibration_pose: None,
            provenance: Provenance::default(),
        }
    }

    #[test]
    fn test_head_frame_equals_affine_at_identity_head_rotation() {
        let theta = [0.02, -0.01, 0.0, -0.015, 0.0, 0.02];
        let head_profile = single_eye_profile(CorrectionModel::HeadFrame, theta);
        let affine_profile = single_eye_profile(CorrectionModel::Affine, theta);

        let ray = straight_ray(Some(eye_core::Side::Right));
        let a = yaw_pitch_from_direction(
            &head_profile
                .correct(RaySource::RgbOnly, &ray)
                .unwrap()
                .direction,
        );
        let b = yaw_pitch_from_direction(
            &affine_profile
                .correct(RaySource::RgbOnly, &ray)
                .unwrap()
                .direction,
        );
        assert_abs_diff_eq!(a.x, b.x, epsilon = 1e-9);
        assert_abs_diff_eq!(a.y, b.y, epsilon = 1e-9);
    }

    #[test]
    fn test_head_frame_tracks_head_rotation() {
        let theta = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0];
        let profile = single_eye_profile(CorrectionModel::HeadFrame, theta);

        let mut ray = straight_ray(Some(eye_core::Side::Right));
        let head_rotation =
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 60.0_f64.to_radians());
        ray.head_rotation = Some(head_rotation);
        let corrected =
            yaw_pitch_from_direction(&profile.correct(RaySource::RgbOnly, &ray).unwrap().direction);

        let th = SVector::<f64, 6>::from(theta);
        let dir_head = head_rotation.inverse() * ray.direction;
        let a_head = yaw_pitch_from_direction(&dir_head);
        let corrected_head = a_head + design(&a_head) * th;
        let expected =
            yaw_pitch_from_direction(&(head_rotation * direction_from_yaw_pitch(&corrected_head)));

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
            version: PROFILE_VERSION,
            name: "x".into(),
            created_unix_s: 0,
            rig_fingerprint: String::new(),
            estimator: "landmark".into(),
            corrections: BTreeMap::from([(legacy_source("landmark"), eyes)]),
            calibration_pose: None,
            provenance: Provenance::default(),
        };

        let mut ray = straight_ray(Some(eye_core::Side::Right));
        ray.angular_cov = Matrix2::zeros();
        let corrected = profile.correct(RaySource::RgbOnly, &ray).unwrap();

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

    fn affine_entry(theta: [f64; 6]) -> AngularCorrection {
        AngularCorrection {
            theta,
            cov: [[0.0; 6]; 6],
            quad: [0.0; 6],
            quad_cov: [[0.0; 6]; 6],
            quad_cross_cov: [[0.0; 6]; 6],
            model: CorrectionModel::Affine,
            targets_used: 9,
            rms_after_rad: 0.0,
        }
    }

    #[test]
    fn test_correct_source_uses_matching_source_entry() {
        let rgb_theta = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0];
        let ir_theta = [-0.02, 0.0, 0.0, 0.0, 0.0, 0.0];
        let profile = UserProfile {
            version: PROFILE_VERSION,
            name: "x".into(),
            created_unix_s: 0,
            rig_fingerprint: String::new(),
            estimator: String::new(),
            corrections: BTreeMap::from([
                (
                    RaySource::RgbOnly,
                    BTreeMap::from([(EyeKey::Right, affine_entry(rgb_theta))]),
                ),
                (
                    RaySource::IrOnly,
                    BTreeMap::from([(EyeKey::Right, affine_entry(ir_theta))]),
                ),
            ]),
            calibration_pose: None,
            provenance: Provenance::default(),
        };
        let ray = straight_ray(Some(eye_core::Side::Right));
        let a = yaw_pitch_from_direction(&ray.direction);

        let rgb_corrected = profile.correct_source(RaySource::RgbOnly, &ray).unwrap();
        let rgb_yaw = yaw_pitch_from_direction(&rgb_corrected.direction).x;
        assert_abs_diff_eq!(rgb_yaw, a.x + 0.01, epsilon = 1e-12);

        let ir_corrected = profile.correct_source(RaySource::IrOnly, &ray).unwrap();
        let ir_yaw = yaw_pitch_from_direction(&ir_corrected.direction).x;
        assert_abs_diff_eq!(ir_yaw, a.x - 0.02, epsilon = 1e-12);
    }

    #[test]
    fn test_correct_source_returns_none_without_entry() {
        let profile = fixture_profile();
        let ray = straight_ray(Some(eye_core::Side::Left));

        let (result, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            profile.correct_source(RaySource::IrOnly, &ray)
        });

        assert_eq!(result, None);
        let rec = records
            .iter()
            .find(|r| r.message == "ray uncorrected")
            .expect("no 'ray uncorrected' record");
        assert_eq!(
            rec.fields.get("source"),
            Some(&eye_log::Value::Str("ir-only".to_string()))
        );
        assert_eq!(
            rec.fields.get(field::REASON),
            Some(&eye_log::Value::Str("no_profile_entry".to_string()))
        );
    }

    #[test]
    fn test_correct_source_returns_none_for_head_frame_without_pose() {
        let profile = UserProfile {
            version: PROFILE_VERSION,
            name: "x".into(),
            created_unix_s: 0,
            rig_fingerprint: String::new(),
            estimator: String::new(),
            corrections: BTreeMap::from([(
                RaySource::IrOnly,
                BTreeMap::from([(
                    EyeKey::Right,
                    AngularCorrection {
                        model: CorrectionModel::HeadFrame,
                        ..affine_entry([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
                    },
                )]),
            )]),
            calibration_pose: None,
            provenance: Provenance::default(),
        };
        let mut ray = straight_ray(Some(eye_core::Side::Right));
        ray.head_rotation = None;

        assert_eq!(profile.correct_source(RaySource::IrOnly, &ray), None);
    }

    #[test]
    fn test_correct_source_scales_cov_by_inverse_gain_squared() {
        let profile = UserProfile {
            version: PROFILE_VERSION,
            name: "x".into(),
            created_unix_s: 0,
            rig_fingerprint: String::new(),
            estimator: String::new(),
            corrections: BTreeMap::from([(
                RaySource::IrOnly,
                BTreeMap::from([(EyeKey::Right, affine_entry([0.0, 1.5, 0.0, 0.0, 0.0, 1.5]))]),
            )]),
            calibration_pose: None,
            provenance: Provenance::default(),
        };
        let mut ray = straight_ray(Some(eye_core::Side::Right));
        ray.angular_cov = Matrix2::new(1e-4, 0.0, 0.0, 2e-4);

        let corrected = profile.correct_source(RaySource::IrOnly, &ray).unwrap();
        let expected = Matrix2::new(6.25e-4, 0.0, 0.0, 12.5e-4);
        assert_abs_diff_eq!(corrected.angular_cov, expected, epsilon = 1e-15);
    }

    #[test]
    fn test_legacy_source_maps_estimator_names() {
        assert_eq!(legacy_source("pccr"), RaySource::IrOnly);
        assert_eq!(legacy_source("ir-pupil"), RaySource::IrOnly);
        for estimator in ["fused", "landmark", "", "test-kappa-ray"] {
            assert_eq!(legacy_source(estimator), RaySource::RgbOnly);
        }
    }
}
