use eye_core::log::field;
use eye_core::observation::SCHEME_IR_PUPIL_PAIR;
use eye_core::{
    CameraId, CameraModel, EyeObservation, FaceObservation, Measured, Observations, Side, Timestamp,
};
use eye_geometry::camera::pixel_ray;
use nalgebra::{Point2, Point3};

use crate::log::side_str;

/// Pixel-space pupil centres from one `ir-pupil-pair` observation.
#[derive(Debug, Clone, PartialEq)]
pub struct PupilPair {
    pub camera: CameraId,
    pub timestamp: Timestamp,
    pub right: Measured<Point2<f64>>,
    pub left: Measured<Point2<f64>>,
}

impl PupilPair {
    /// Every `ir-pupil-pair` observation with both pupils, sorted by timestamp.
    pub fn all_from_observations(obs: &[Observations]) -> Vec<PupilPair> {
        let mut pairs: Vec<PupilPair> = obs
            .iter()
            .filter_map(|o| {
                let face = o.face.as_ref()?;
                if face.scheme != SCHEME_IR_PUPIL_PAIR {
                    return None;
                }
                let (Some(right), Some(left)) = (
                    pupil_centre(face, Side::Right),
                    pupil_centre(face, Side::Left),
                ) else {
                    tracing::debug!(
                        { field::REASON } = "missing_pupil",
                        { field::CAMERA } = %o.camera,
                        obs_ts_ns = o.timestamp.as_nanos(),
                        eyes = face.eyes.len() as u64,
                        "ir observation without both pupils"
                    );
                    return None;
                };
                Some(PupilPair {
                    camera: o.camera.clone(),
                    timestamp: o.timestamp,
                    right,
                    left,
                })
            })
            .collect();
        pairs.sort_by_key(|pair| pair.timestamp);
        pairs
    }

    /// The latest of those (dual mode delivers the bracketing previous and next lit frames, R30).
    pub fn from_observations(obs: &[Observations]) -> Option<PupilPair> {
        let mut pairs = Self::all_from_observations(obs);
        pairs.pop()
    }
}

fn pupil_centre(face: &FaceObservation, side: Side) -> Option<Measured<Point2<f64>>> {
    Some(face.eye(side)?.pupil?.map(|ellipse| ellipse.center()))
}

/// The first glint within `max_px` of `pupil`, if any.
pub(crate) fn glint_near(
    eye: &EyeObservation,
    side: Side,
    pupil: &Point2<f64>,
    max_px: f64,
) -> Option<Measured<Point2<f64>>> {
    let found = eye
        .glints
        .iter()
        .find(|g| (*g.value() - pupil).norm() <= max_px)
        .copied();
    if found.is_none() {
        let nearest_px = eye
            .glints
            .iter()
            .map(|g| (*g.value() - pupil).norm())
            .min_by(f64::total_cmp);
        tracing::debug!(
            { field::REASON } = "no_glint",
            side = side_str(side),
            glints = eye.glints.len() as u64,
            nearest_px,
            max_glint_offset_px = max_px,
            "glint rejected"
        );
    }
    found
}

/// 3D pupil centres `[right, left]` in the reference frame under the equal-range assumption.
pub fn binocular_pupils(
    cam: &CameraModel,
    pair: &PupilPair,
    ipd_mm: f64,
) -> Option<[Point3<f64>; 2]> {
    let log_pixel_ray_failed = |e: &eye_geometry::GeometryError| {
        tracing::debug!(
            { field::REASON } = "pixel_ray_failed",
            error = %e,
            "binocular pupils failed"
        );
    };
    let (origin, u_right) = pixel_ray(cam, pair.right.value())
        .inspect_err(log_pixel_ray_failed)
        .ok()?;
    let (_, u_left) = pixel_ray(cam, pair.left.value())
        .inspect_err(log_pixel_ray_failed)
        .ok()?;
    let separation = (u_right.into_inner() - u_left.into_inner()).norm();
    if separation <= f64::EPSILON {
        tracing::debug!(
            { field::REASON } = "zero_separation",
            separation,
            "binocular pupils failed"
        );
        return None;
    }
    let range = ipd_mm / separation;
    Some([
        origin + u_right.into_inner() * range,
        origin + u_left.into_inner() * range,
    ])
}

#[cfg(test)]
mod tests {
    use eye_core::observation::SCHEME_MEDIAPIPE_478;
    use eye_core::{Ellipse2, EyeObservation};
    use eye_geometry::eyeball::EyeParams;
    use eye_log::testing::capture_logs;
    use eye_log::{Level, Value};
    use nalgebra::Vector3;

    use super::*;
    use crate::testutil::{EYE_CENTRES, synthetic_ir_observation, test_rig};

    #[test]
    fn test_binocular_pupils_recovers_range_within_3_percent() {
        let rig = test_rig();
        let cam = rig.camera("ir").expect("rig has an ir camera");
        let target = Point2::new(155.0, 85.0);
        let obs = synthetic_ir_observation(&rig, target, Vector3::zeros(), 0.0, 1);
        let pair = PupilPair::from_observations(&[obs]).expect("synthetic observation is a pair");
        let x = binocular_pupils(cam, &pair, 63.0).expect("pupils are separated");

        let r = EyeParams::default().rotation_to_pupil_mm;
        let origin = Point3::from(cam.screen_from_camera.translation.vector);
        let target3 = Point3::new(target.x, target.y, 0.0);
        for (i, centre) in EYE_CENTRES.into_iter().enumerate() {
            let true_pupil = centre + (target3 - centre).normalize() * r;
            let true_range = (true_pupil - origin).norm();
            let estimated_range = (x[i] - origin).norm();
            let relative_error = (estimated_range - true_range).abs() / true_range;
            assert!(
                relative_error < 0.03,
                "eye {i}: relative error {relative_error}"
            );
        }
    }

    #[test]
    fn test_binocular_pupils_identical_pixels_returns_none() {
        let rig = test_rig();
        let cam = rig.camera("ir").expect("rig has an ir camera");
        let pixel = Measured::new(Point2::new(320.0, 180.0), 0.0).expect("valid sigma");
        let pair = PupilPair {
            camera: CameraId::new("ir"),
            timestamp: Timestamp::from_nanos(0),
            right: pixel,
            left: pixel,
        };
        assert!(binocular_pupils(cam, &pair, 63.0).is_none());
    }

    #[test]
    fn test_logs_zero_separation_at_debug() {
        let rig = test_rig();
        let cam = rig.camera("ir").expect("rig has an ir camera");
        let pixel = Measured::new(Point2::new(320.0, 180.0), 0.0).expect("valid sigma");
        let pair = PupilPair {
            camera: CameraId::new("ir"),
            timestamp: Timestamp::from_nanos(0),
            right: pixel,
            left: pixel,
        };

        let (result, logs) =
            capture_logs(tracing::Level::DEBUG, || binocular_pupils(cam, &pair, 63.0));

        assert!(result.is_none());
        let rec = logs
            .iter()
            .find(|r| r.message == "binocular pupils failed")
            .expect("zero_separation logged");
        assert_eq!(rec.level, Level::Debug);
        assert_eq!(
            rec.fields[field::REASON],
            Value::Str("zero_separation".into())
        );
        assert_eq!(rec.fields["separation"], Value::F64(0.0));
    }

    #[test]
    fn test_logs_missing_pupil_at_debug() {
        let mut right_eye = EyeObservation::new(Side::Right);
        right_eye.pupil = Some(
            Measured::new(
                Ellipse2::circle(Point2::new(300.0, 180.0), 3.0).expect("valid ellipse"),
                0.2,
            )
            .expect("valid sigma"),
        );
        let obs = Observations {
            camera: CameraId::new("ir"),
            timestamp: Timestamp::from_nanos(0),
            face: Some(FaceObservation {
                scheme: SCHEME_IR_PUPIL_PAIR,
                landmarks: Vec::new(),
                eyes: vec![right_eye],
            }),
        };

        let (pairs, logs) = capture_logs(tracing::Level::DEBUG, || {
            PupilPair::all_from_observations(&[obs])
        });

        assert!(pairs.is_empty());
        let rec = logs
            .iter()
            .find(|r| r.message == "ir observation without both pupils")
            .expect("missing_pupil logged");
        assert_eq!(rec.level, Level::Debug);
        assert_eq!(
            rec.fields[field::REASON],
            Value::Str("missing_pupil".into())
        );
        assert_eq!(rec.fields[field::CAMERA], Value::Str("ir".into()));
        assert_eq!(rec.fields["obs_ts_ns"], Value::U64(0));
        assert_eq!(rec.fields["eyes"], Value::U64(1));
    }

    #[test]
    fn test_from_observations_picks_latest_pair() {
        let rig = test_rig();
        let target = Point2::new(155.0, 85.0);
        let first = synthetic_ir_observation(&rig, target, Vector3::zeros(), 0.0, 1);
        let mut second = synthetic_ir_observation(&rig, target, Vector3::zeros(), 0.0, 2);
        second.timestamp = Timestamp::from_nanos(136_000_000);
        let rgb = Observations {
            camera: CameraId::new("rgb"),
            timestamp: Timestamp::from_nanos(50_000_000),
            face: Some(FaceObservation {
                scheme: SCHEME_MEDIAPIPE_478,
                landmarks: Vec::new(),
                eyes: Vec::new(),
            }),
        };
        let obs = [second.clone(), rgb, first.clone()];

        let all = PupilPair::all_from_observations(&obs);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].timestamp, first.timestamp);
        assert_eq!(all[1].timestamp, second.timestamp);

        let latest = PupilPair::from_observations(&obs).expect("a pair is present");
        assert_eq!(latest.timestamp, second.timestamp);
    }
}
