//! Synthetic round trip: landmark rays at one head pose fit a user profile, and the corrected
//! screen hit must stay on target as the head translates and rotates.

use std::f64::consts::PI;

use eye_calibration::correction::{CorrectionModel, UserProfile};
use eye_calibration::user_fit::{DotSessionFit, FitConfig, FitSample, ProfileMeta};
use eye_core::{Ellipse2, GazeRay, Measured, RaySource, Rig};
use eye_geometry::camera::Intrinsics;
use eye_geometry::eyeball::EyeParams;
use eye_geometry::screen::intersect_gaze;
use nalgebra::{Isometry3, Point2, Translation3, UnitQuaternion, Vector2, Vector3};

use crate::landmark::{IrisResponse, LandmarkEstimator, LandmarkOptions};
use crate::testutil::{synthetic_eye_centres, synthetic_rgb_observation_with_response, test_rig};

const TARGETS: [(f64, f64); 9] = [
    (20.0, 15.0),
    (155.0, 15.0),
    (290.0, 15.0),
    (20.0, 85.0),
    (155.0, 85.0),
    (290.0, 85.0),
    (20.0, 155.0),
    (155.0, 155.0),
    (290.0, 155.0),
];

/// Rendered iris offset from the eye centre's image, as a fraction of the true offset.
const IRIS_GAIN: f64 = 0.45;

fn head(dx: f64, dz: f64, yaw_deg: f64) -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::new(155.0 + dx, 40.0, -500.0 + dz),
        UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI + yaw_deg.to_radians()),
    )
}

fn rays(
    rig: &Rig,
    screen_from_head: &Isometry3<f64>,
    target: Point2<f64>,
    gain: f64,
    seed: u64,
) -> Vec<GazeRay> {
    let mut obs = synthetic_rgb_observation_with_response(
        rig,
        screen_from_head,
        1.0,
        target,
        0.0,
        0.2,
        seed,
        &IrisResponse::IDENTITY,
    );
    let cam = rig.camera("rgb").expect("test rig has an rgb camera");
    let intrinsics = Intrinsics::from_camera_model(cam);
    let centres = synthetic_eye_centres(screen_from_head, 1.0, &EyeParams::default());
    let face = obs.face.as_mut().expect("face is present");
    for (eye, centre) in face.eyes.iter_mut().zip(centres) {
        let centre_px = intrinsics
            .project(&cam.screen_from_camera.inverse_transform_point(&centre))
            .expect("eye centre projects in front of the camera");
        let iris = eye.iris.as_mut().expect("iris is present");
        let compressed = centre_px + (iris.value().center() - centre_px) * gain;
        *iris = Measured::new(
            Ellipse2::circle(compressed, 11.85).expect("iris ellipse is valid"),
            iris.sigma(),
        )
        .expect("sigma is valid");
    }
    let mut estimator = LandmarkEstimator::new(LandmarkOptions {
        apply_kappa: false,
        iris_response: IrisResponse::IDENTITY,
        ..Default::default()
    });
    estimator
        .estimate_frame(&obs, rig)
        .expect("estimate succeeds")
        .expect("frame is recovered")
        .eyes
        .into_iter()
        .map(|e| e.ray)
        .collect()
}

fn fit(rig: &Rig, model: Option<CorrectionModel>, gain: f64) -> UserProfile {
    let calibration_head = head(0.0, 0.0, 0.0);
    let mut samples = Vec::new();
    for (i, &(x, y)) in TARGETS.iter().enumerate() {
        let target_mm = Point2::new(x, y);
        for k in 0..8 {
            for ray in rays(rig, &calibration_head, target_mm, gain, 100 * i as u64 + k) {
                samples.push(FitSample {
                    ray,
                    target_mm,
                    source: RaySource::RgbOnly,
                });
            }
        }
    }
    let cfg = FitConfig {
        model_override: model,
        ..FitConfig::default()
    };
    DotSessionFit::fit_with(&samples, rig, &cfg, ProfileMeta::default())
        .expect("profile fits")
        .profile
}

/// Mean (corrected hit - target) over both eyes and every target, screen mm.
fn mean_error_mm(
    profile: &UserProfile,
    rig: &Rig,
    screen_from_head: &Isometry3<f64>,
    gain: f64,
) -> Vector2<f64> {
    let mut sum = Vector2::zeros();
    let mut n = 0.0;
    for (i, &(x, y)) in TARGETS.iter().enumerate() {
        let target_mm = Point2::new(x, y);
        for ray in rays(rig, screen_from_head, target_mm, gain, 9000 + i as u64) {
            let corrected = profile
                .correct_source(RaySource::RgbOnly, &ray)
                .expect("profile has an entry for both eyes");
            let hit = intersect_gaze(&corrected).expect("corrected ray hits the screen");
            sum += hit.mm - target_mm;
            n += 1.0;
        }
    }
    sum / n
}

fn poses() -> [(&'static str, Isometry3<f64>); 6] {
    [
        ("left 76 mm", head(-76.0, 0.0, 0.0)),
        ("right 100 mm", head(100.0, 0.0, 0.0)),
        ("back 130 mm", head(0.0, -130.0, 0.0)),
        ("forward 170 mm", head(0.0, 170.0, 0.0)),
        ("yaw 15 deg", head(0.0, 0.0, 15.0)),
        ("left 76 mm + yaw -7.6 deg", head(-76.0, 0.0, -7.6)),
    ]
}

#[test]
fn test_exact_iris_stays_on_target_under_head_motion() {
    let rig = test_rig();
    for model in [CorrectionModel::HeadFrame, CorrectionModel::CameraFrame] {
        let profile = fit(&rig, Some(model), 1.0);
        for (name, pose) in poses() {
            let e = mean_error_mm(&profile, &rig, &pose, 1.0);
            assert!(e.norm() < 6.0, "{model:?} {name}: mean error {e:?} mm");
        }
    }
}

#[test]
fn test_head_frame_profile_moves_gaze_against_head_translation() {
    let rig = test_rig();
    let profile = fit(&rig, Some(CorrectionModel::HeadFrame), IRIS_GAIN);
    let expected_factor = -(1.0 - IRIS_GAIN) / IRIS_GAIN;

    let e = mean_error_mm(&profile, &rig, &head(-76.0, 0.0, 0.0), IRIS_GAIN);
    let factor = e.x / -76.0;
    assert!(
        (factor - expected_factor).abs() < 0.25,
        "lateral error {:.1} mm for a -76 mm shift: factor {factor:.2}, expected {expected_factor:.2}",
        e.x
    );
}

#[test]
fn test_camera_frame_profile_stays_on_target_under_head_motion() {
    let rig = test_rig();
    let profile = fit(&rig, None, IRIS_GAIN);
    let model = profile
        .correction(
            RaySource::RgbOnly,
            eye_calibration::correction::EyeKey::Right,
        )
        .expect("right eye fitted")
        .model;
    assert_eq!(model, CorrectionModel::CameraFrame);
    for (name, pose) in poses() {
        let e = mean_error_mm(&profile, &rig, &pose, IRIS_GAIN);
        assert!(e.norm() < 10.0, "{name}: mean error {e:?} mm");
    }
}
