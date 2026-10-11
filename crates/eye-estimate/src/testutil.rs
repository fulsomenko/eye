use std::f64::consts::PI;

use eye_core::observation::{LandmarkScheme, mediapipe478};
use eye_core::{
    CameraId, CameraModel, Ellipse2, EyeCorners, EyeObservation, FaceObservation, Measured,
    Observations, OutputId, Rig, ScreenModel, Side, Timestamp,
};
use eye_geometry::camera::Intrinsics;
use eye_geometry::eyeball::{EyeParams, eyeball_centre_in_head};
use eye_geometry::face_template::MEDIAPIPE_RIGID;
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
            scheme: LandmarkScheme::IR_PUPIL_PAIR,
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

/// Renders the ir-pupil-pair observation for `centres` gazing at `target_mm`: pupils at `C + r_p *
/// g`, `r_p = rotation_to_pupil_mm`, plus one glint per eye: `K = C + rotation_to_cornea_mm() * g`,
/// `glint = K + cornea_radius_mm * normalize(o_ir - K)` (the corneal point whose normal points at
/// the camera), projected into "ir", `N(0, glint_sigma_px)` noise, `Measured` sigma =
/// `glint_sigma_px`.
fn synthetic_pccr_observation_from_centres(
    rig: &Rig,
    target_mm: Point2<f64>,
    centres: [Point3<f64>; 2],
    pupil_sigma_px: f64,
    glint_sigma_px: f64,
    seed: u64,
) -> Observations {
    let cam = rig.camera("ir").expect("test rig has an ir camera");
    let intrinsics = Intrinsics::from_camera_model(cam);
    let params = EyeParams::default();
    let r_p = params.rotation_to_pupil_mm;
    let r_k = params.rotation_to_cornea_mm();
    let target = Point3::new(target_mm.x, target_mm.y, 0.0);
    let o_ir = Point3::from(cam.screen_from_camera.translation.vector);
    let mut rng = SplitMix64::new(seed);

    let eyes = [Side::Right, Side::Left]
        .into_iter()
        .zip(centres)
        .map(|(side, centre)| {
            let g = (target - centre).normalize();
            let pupil_screen = centre + g * r_p;
            let pupil_cam = cam
                .screen_from_camera
                .inverse_transform_point(&pupil_screen);
            let pupil_px = intrinsics
                .project(&pupil_cam)
                .expect("synthetic pupil projects in front of the camera");
            let noisy_pupil = Point2::new(
                pupil_px.x + pupil_sigma_px * rng.gaussian(),
                pupil_px.y + pupil_sigma_px * rng.gaussian(),
            );

            let k = centre + g * r_k;
            let glint_screen = k + (o_ir - k).normalize() * params.cornea_radius_mm;
            let glint_cam = cam
                .screen_from_camera
                .inverse_transform_point(&glint_screen);
            let glint_px = intrinsics
                .project(&glint_cam)
                .expect("synthetic glint projects in front of the camera");
            let noisy_glint = Point2::new(
                glint_px.x + glint_sigma_px * rng.gaussian(),
                glint_px.y + glint_sigma_px * rng.gaussian(),
            );

            let mut eye = EyeObservation::new(side);
            eye.pupil = Some(
                Measured::new(
                    Ellipse2::circle(noisy_pupil, 3.0).expect("synthetic pupil ellipse is valid"),
                    pupil_sigma_px,
                )
                .expect("pupil_sigma_px is a valid Measured sigma"),
            );
            eye.glints = vec![
                Measured::new(noisy_glint, glint_sigma_px)
                    .expect("glint_sigma_px is a valid Measured sigma"),
            ];
            eye
        })
        .collect();

    Observations {
        camera: CameraId::new("ir"),
        timestamp: Timestamp::from_nanos(0),
        face: Some(FaceObservation {
            scheme: LandmarkScheme::IR_PUPIL_PAIR,
            landmarks: Vec::new(),
            eyes,
        }),
    }
}

/// `synthetic_pccr_observation_from_centres` with eye centres `EYE_CENTRES + head_offset_mm`.
pub(crate) fn synthetic_pccr_observation(
    rig: &Rig,
    target_mm: Point2<f64>,
    head_offset_mm: Vector3<f64>,
    pupil_sigma_px: f64,
    glint_sigma_px: f64,
    seed: u64,
) -> Observations {
    let centres = [
        EYE_CENTRES[0] + head_offset_mm,
        EYE_CENTRES[1] + head_offset_mm,
    ];
    synthetic_pccr_observation_from_centres(
        rig,
        target_mm,
        centres,
        pupil_sigma_px,
        glint_sigma_px,
        seed,
    )
}

/// `synthetic_pccr_observation_from_centres` geometry at an arbitrary (translated and/or rotated)
/// head pose: eye centres from `synthetic_eye_centres(screen_from_head, 1.0,
/// &EyeParams::default())` rather than `EYE_CENTRES + head_offset_mm`.
pub(crate) fn synthetic_pccr_observation_posed(
    rig: &Rig,
    target_mm: Point2<f64>,
    screen_from_head: &Isometry3<f64>,
    pupil_sigma_px: f64,
    glint_sigma_px: f64,
    seed: u64,
) -> Observations {
    let centres = synthetic_eye_centres(screen_from_head, 1.0, &EyeParams::default());
    synthetic_pccr_observation_from_centres(
        rig,
        target_mm,
        centres,
        pupil_sigma_px,
        glint_sigma_px,
        seed,
    )
}

/// Rotation centres [right, left] of the synthetic head: `screen_from_head * (scale *
/// eyeball_centre_in_head(T(inner), T(outer), params))`.
pub(crate) fn synthetic_eye_centres(
    screen_from_head: &Isometry3<f64>,
    scale: f64,
    params: &EyeParams,
) -> [Point3<f64>; 2] {
    let template_point = |landmark: usize| {
        MEDIAPIPE_RIGID
            .point(landmark)
            .expect("landmark is in the rigid template")
    };
    let right = eyeball_centre_in_head(&template_point(133), &template_point(33), params);
    let left = eyeball_centre_in_head(&template_point(362), &template_point(263), params);
    [right, left]
        .map(|centre| screen_from_head.transform_point(&Point3::from(centre.coords * scale)))
}

/// Poses `MEDIAPIPE_RIGID` scaled by `scale` about the head origin at `screen_from_head`,
/// projects into the "rgb" camera, fills 478 landmarks (template indices exact plus
/// `N(0, landmark_noise_px)`; every other index at the projected head origin), and both eyes:
/// corners at the projected template corners, iris at the pupil-sphere crossing towards
/// `target_mm` plus `N(0, iris_noise_px)`. Scheme `LandmarkScheme::MEDIAPIPE_478`, camera "rgb",
/// timestamp 0. Noise from `SplitMix64::new(seed)`.
pub(crate) fn synthetic_rgb_observation(
    rig: &Rig,
    screen_from_head: &Isometry3<f64>,
    scale: f64,
    target_mm: Point2<f64>,
    landmark_noise_px: f64,
    iris_noise_px: f64,
    seed: u64,
) -> Observations {
    let cam = rig.camera("rgb").expect("test rig has an rgb camera");
    let intrinsics = Intrinsics::from_camera_model(cam);
    let camera_from_screen = cam.screen_from_camera.inverse();
    let params = EyeParams::default();
    let mut rng = SplitMix64::new(seed);

    let project = |p_screen: &Point3<f64>| -> Point2<f64> {
        intrinsics
            .project(&camera_from_screen.transform_point(p_screen))
            .expect("synthetic point projects in front of the camera")
    };

    let head_origin_px = project(&screen_from_head.transform_point(&Point3::origin()));
    let mut landmarks = vec![head_origin_px; mediapipe478::COUNT];
    for &(i, p) in MEDIAPIPE_RIGID.points {
        let scaled_head_point = Point3::from(Point3::new(p[0], p[1], p[2]).coords * scale);
        let pixel = project(&screen_from_head.transform_point(&scaled_head_point));
        landmarks[i] = Point2::new(
            pixel.x + landmark_noise_px * rng.gaussian(),
            pixel.y + landmark_noise_px * rng.gaussian(),
        );
    }

    let target = Point3::new(target_mm.x, target_mm.y, 0.0);
    let centres = synthetic_eye_centres(screen_from_head, scale, &params);

    let eyes = [
        (
            Side::Right,
            mediapipe478::RIGHT_EYE_LATERAL,
            mediapipe478::RIGHT_EYE_MEDIAL,
        ),
        (
            Side::Left,
            mediapipe478::LEFT_EYE_LATERAL,
            mediapipe478::LEFT_EYE_MEDIAL,
        ),
    ]
    .into_iter()
    .zip(centres)
    .map(|((side, lateral_idx, medial_idx), centre)| {
        let direction = (target - centre).normalize();
        let pupil_screen = centre + direction * params.rotation_to_pupil_mm;
        let pupil_pixel = project(&pupil_screen);
        let noisy_pupil = Point2::new(
            pupil_pixel.x + iris_noise_px * rng.gaussian(),
            pupil_pixel.y + iris_noise_px * rng.gaussian(),
        );
        let mut eye = EyeObservation::new(side);
        eye.corners = Some(EyeCorners {
            lateral: Measured::new(landmarks[lateral_idx], landmark_noise_px)
                .expect("synthetic sigma is valid"),
            medial: Measured::new(landmarks[medial_idx], landmark_noise_px)
                .expect("synthetic sigma is valid"),
        });
        eye.iris = Some(
            Measured::new(
                Ellipse2::circle(noisy_pupil, 11.85).expect("synthetic iris ellipse is valid"),
                iris_noise_px,
            )
            .expect("synthetic sigma is valid"),
        );
        eye
    })
    .collect();

    Observations {
        camera: CameraId::new("rgb"),
        timestamp: Timestamp::from_nanos(0),
        face: Some(FaceObservation {
            scheme: LandmarkScheme::MEDIAPIPE_478,
            landmarks,
            eyes,
        }),
    }
}
