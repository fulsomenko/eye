use std::f64::consts::PI;

use eye_core::observation::SCHEME_IR_PUPIL_PAIR;
use eye_core::{
    CameraId, CameraModel, Ellipse2, EyeObservation, FaceObservation, Measured, Observations,
    OutputId, Rig, ScreenModel, Side, Timestamp,
};
use eye_geometry::camera::Intrinsics;
use eye_geometry::eyeball::EyeParams;
use eye_geometry::synth::SplitMix64;
use nalgebra::{Isometry3, Point2, Point3, Translation3, UnitQuaternion, Vector2, Vector3};

pub(crate) fn test_rig() -> Rig {
    let ir = CameraModel {
        id: CameraId::new("ir"),
        width: 640,
        height: 360,
        fx: 457.0,
        fy: 457.0,
        cx: 320.0,
        cy: 180.0,
        distortion: [0.0; 5],
        screen_from_camera: Isometry3::from_parts(
            Translation3::new(167.5, -7.0, 0.0),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI),
        ),
    };
    let rgb = CameraModel {
        id: CameraId::new("rgb"),
        width: 1280,
        height: 720,
        fx: 914.0,
        fy: 914.0,
        cx: 640.0,
        cy: 360.0,
        distortion: [0.0; 5],
        screen_from_camera: Isometry3::from_parts(
            Translation3::new(142.5, -7.0, 0.0),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI),
        ),
    };
    let screen = ScreenModel {
        output: OutputId::new("eDP-1"),
        size_mm: Vector2::new(310.0, 170.0),
        size_px: (3840, 2160),
        scale: 2.0,
    };
    Rig::new(vec![ir, rgb], screen).expect("test rig is valid")
}

pub(crate) const EYE_CENTRES: [Point3<f64>; 2] = [
    Point3::new(186.5, 40.0, -500.0),
    Point3::new(123.5, 40.0, -500.0),
];

pub(crate) fn synthetic_ir_observation_at(
    rig: &Rig,
    centres: [Point3<f64>; 2],
    target_mm: Point2<f64>,
    sigma_px: f64,
    seed: u64,
) -> Observations {
    let cam = rig.camera("ir").expect("test rig has an ir camera");
    let intrinsics = Intrinsics::from_camera_model(cam);
    let r = EyeParams::default().rotation_to_pupil_mm;
    let target = Point3::new(target_mm.x, target_mm.y, 0.0);
    let mut rng = SplitMix64::new(seed);

    let eyes = [Side::Right, Side::Left]
        .into_iter()
        .zip(centres)
        .map(|(side, centre)| {
            let direction = (target - centre).normalize();
            let pupil_screen = centre + direction * r;
            let pupil_cam = cam
                .screen_from_camera
                .inverse_transform_point(&pupil_screen);
            let pixel = intrinsics
                .project(&pupil_cam)
                .expect("synthetic pupil projects in front of the camera");
            let noisy = Point2::new(
                pixel.x + sigma_px * rng.gaussian(),
                pixel.y + sigma_px * rng.gaussian(),
            );
            let mut eye = EyeObservation::new(side);
            eye.pupil = Some(
                Measured::new(
                    Ellipse2::circle(noisy, 3.0).expect("synthetic pupil ellipse is valid"),
                    sigma_px,
                )
                .expect("sigma_px is a valid Measured sigma"),
            );
            eye
        })
        .collect();

    Observations {
        camera: CameraId::new("ir"),
        timestamp: Timestamp::from_nanos(0),
        face: Some(FaceObservation {
            scheme: SCHEME_IR_PUPIL_PAIR,
            landmarks: Vec::new(),
            eyes,
        }),
    }
}

pub(crate) fn synthetic_ir_observation(
    rig: &Rig,
    target_mm: Point2<f64>,
    head_offset_mm: Vector3<f64>,
    sigma_px: f64,
    seed: u64,
) -> Observations {
    let centres = [
        EYE_CENTRES[0] + head_offset_mm,
        EYE_CENTRES[1] + head_offset_mm,
    ];
    synthetic_ir_observation_at(rig, centres, target_mm, sigma_px, seed)
}
