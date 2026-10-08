use std::collections::BTreeMap;

use eye_core::stage::GazeCorrection;
use eye_core::{GazeRay, Rig, Side};
use eye_geometry::angles::{direction_from_yaw_pitch, yaw_pitch_from_direction};
use eye_geometry::uncertainty::propagate;
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
    pub model: CorrectionModel,
    pub targets_used: u32,
    pub rms_after_rad: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CorrectionModel {
    Affine,
    OffsetOnly,
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

impl GazeCorrection for UserProfile {
    fn correct(&self, ray: &GazeRay) -> GazeRay {
        let Some(c) = self.eyes.get(&EyeKey::from(ray.side)) else {
            return ray.clone();
        };
        let th = SVector::<f64, 6>::from(c.theta);
        let a = yaw_pitch_from_direction(&ray.direction);
        let b = design(&a);
        let j = Matrix2::new(1.0 + th[1], th[2], th[4], 1.0 + th[5]);
        let cov_theta = SMatrix::<f64, 6, 6>::from_fn(|r, k| c.cov[r][k]);
        GazeRay {
            direction: direction_from_yaw_pitch(&(a + b * th)),
            angular_cov: propagate(&j, &ray.angular_cov) + propagate(&b, &cov_theta),
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
    use nalgebra::{Point3, SMatrix, Unit, Vector2, Vector3};

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
}
