use eye_core::observation::SCHEME_IR_PUPIL_PAIR;
use eye_core::{CameraId, CameraModel, FaceObservation, Measured, Observations, Side, Timestamp};
use eye_geometry::camera::pixel_ray;
use nalgebra::{Point2, Point3};

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
                Some(PupilPair {
                    camera: o.camera.clone(),
                    timestamp: o.timestamp,
                    right: pupil_centre(face, Side::Right)?,
                    left: pupil_centre(face, Side::Left)?,
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

/// 3D pupil centres `[right, left]` in the reference frame under the equal-range assumption.
pub fn binocular_pupils(
    cam: &CameraModel,
    pair: &PupilPair,
    ipd_mm: f64,
) -> Option<[Point3<f64>; 2]> {
    let (origin, u_right) = pixel_ray(cam, pair.right.value()).ok()?;
    let (_, u_left) = pixel_ray(cam, pair.left.value()).ok()?;
    let separation = (u_right.into_inner() - u_left.into_inner()).norm();
    if separation <= f64::EPSILON {
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
    use eye_geometry::eyeball::EyeParams;
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
