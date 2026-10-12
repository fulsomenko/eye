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
use nalgebra::{
    Isometry3, Matrix2, Point2, Point3, Translation3, UnitQuaternion, Vector2, Vector3,
};

use crate::landmark::{
    IrisResponse, LEFT_LOWER_LID, LEFT_UPPER_LID, RIGHT_LOWER_LID, RIGHT_UPPER_LID,
};

const LOWER_LID_BELOW_CORNERS_MM: f64 = 3.0;
const UPPER_LID_ABOVE_CORNERS_MM: f64 = 5.0;

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

/// What a detector with `response` reports for a true pupil at `pupil_px` (see
/// [`IrisResponse`]): the forward map that `LandmarkEstimator` inverts, given the noise-free
/// lower lid.
#[allow(clippy::too_many_arguments)]
fn detected_iris(
    response: &IrisResponse,
    project: &impl Fn(&Point3<f64>) -> Point2<f64>,
    screen_from_head: &Isometry3<f64>,
    scale: f64,
    side: Side,
    rotation_centre: &Point3<f64>,
    lower_lid_px: &Point2<f64>,
    pupil_px: &Point2<f64>,
) -> Point2<f64> {
    let (medial, lateral) = match side {
        Side::Right => (
            mediapipe478::RIGHT_EYE_MEDIAL,
            mediapipe478::RIGHT_EYE_LATERAL,
        ),
        Side::Left => (
            mediapipe478::LEFT_EYE_MEDIAL,
            mediapipe478::LEFT_EYE_LATERAL,
        ),
    };
    let corner = |i| {
        MEDIAPIPE_RIGID
            .point(i)
            .expect("corners are in the rigid template")
    };
    let mid_head = Point3::from((corner(medial).coords + corner(lateral).coords) * (scale / 2.0));
    let lateral_px = project(rotation_centre);
    let vertical_px = project(&screen_from_head.transform_point(&mid_head));
    let axis_px = |axis: Vector3<f64>| {
        project(&(rotation_centre + screen_from_head.rotation * axis)) - lateral_px
    };
    let head_px = Matrix2::from_columns(&[axis_px(Vector3::x()), axis_px(Vector3::y())]);
    let px_head = head_px.try_inverse().expect("head axes span the image");
    let temporal_x = match side {
        Side::Right => -1.0,
        Side::Left => 1.0,
    };
    let offset = Vector2::new(temporal_x * response.offset_mm[0], response.offset_mm[1]);
    let truth = Vector2::new(
        (px_head * (pupil_px - lateral_px)).x,
        (px_head * (pupil_px - vertical_px)).y,
    );
    let lower_lid_y = (px_head * (lower_lid_px - vertical_px)).y;
    let measured = offset
        + Vector2::from(response.gain).component_mul(&truth)
        + Vector2::new(0.0, response.lower_lid_weight * lower_lid_y);
    let vertical_shift = (px_head * (vertical_px - lateral_px)).y;
    lateral_px + head_px * Vector2::new(measured.x, measured.y + vertical_shift)
}

/// A MediaPipe RGB observation: the iris follows [`IrisResponse::MEDIAPIPE_478`].
pub(crate) fn synthetic_rgb_observation(
    rig: &Rig,
    screen_from_head: &Isometry3<f64>,
    scale: f64,
    target_mm: Point2<f64>,
    landmark_noise_px: f64,
    iris_noise_px: f64,
    seed: u64,
) -> Observations {
    synthetic_rgb_observation_with_response(
        rig,
        screen_from_head,
        scale,
        target_mm,
        landmark_noise_px,
        iris_noise_px,
        seed,
        &IrisResponse::MEDIAPIPE_478,
    )
}

/// Poses `MEDIAPIPE_RIGID` scaled by `scale` about the head origin at `screen_from_head`,
/// projects into the "rgb" camera, fills 478 landmarks (template indices exact plus
/// `N(0, landmark_noise_px)`; the lower-lid landmarks 3 mm below and the upper-lid landmarks
/// 5 mm above the eye-corner midpoint, with the same noise; every other index at the projected
/// head origin), and both eyes: corners at the projected template corners, iris where a detector
/// with `response` reports the pupil-sphere crossing towards `target_mm`, plus
/// `N(0, iris_noise_px)`. Scheme `LandmarkScheme::MEDIAPIPE_478`, camera "rgb", timestamp 0.
#[allow(clippy::too_many_arguments)]
pub(crate) fn synthetic_rgb_observation_with_response(
    rig: &Rig,
    screen_from_head: &Isometry3<f64>,
    scale: f64,
    target_mm: Point2<f64>,
    landmark_noise_px: f64,
    iris_noise_px: f64,
    seed: u64,
    response: &IrisResponse,
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

    let lower_lid_px = [
        (
            mediapipe478::RIGHT_EYE_MEDIAL,
            mediapipe478::RIGHT_EYE_LATERAL,
            RIGHT_LOWER_LID,
        ),
        (
            mediapipe478::LEFT_EYE_MEDIAL,
            mediapipe478::LEFT_EYE_LATERAL,
            LEFT_LOWER_LID,
        ),
    ]
    .map(|(medial, lateral, lid)| {
        let corner = |i| {
            MEDIAPIPE_RIGID
                .point(i)
                .expect("corners are in the rigid template")
        };
        let mid = (corner(medial).coords + corner(lateral).coords) / 2.0;
        let lid_head =
            Point3::from((mid + Vector3::new(0.0, LOWER_LID_BELOW_CORNERS_MM, 0.0)) * scale);
        let pixel = project(&screen_from_head.transform_point(&lid_head));
        for i in lid {
            landmarks[i] = Point2::new(
                pixel.x + landmark_noise_px * rng.gaussian(),
                pixel.y + landmark_noise_px * rng.gaussian(),
            );
        }
        pixel
    });
    // A separate noise stream keeps every other landmark's noise what it was before the upper lid.
    let mut upper_lid_rng = SplitMix64::new(seed ^ 0x9e37_79b9_7f4a_7c15);
    for (medial, lateral, lid) in [
        (
            mediapipe478::RIGHT_EYE_MEDIAL,
            mediapipe478::RIGHT_EYE_LATERAL,
            RIGHT_UPPER_LID,
        ),
        (
            mediapipe478::LEFT_EYE_MEDIAL,
            mediapipe478::LEFT_EYE_LATERAL,
            LEFT_UPPER_LID,
        ),
    ] {
        let corner = |i| {
            MEDIAPIPE_RIGID
                .point(i)
                .expect("corners are in the rigid template")
        };
        let mid = (corner(medial).coords + corner(lateral).coords) / 2.0;
        let lid_head =
            Point3::from((mid - Vector3::new(0.0, UPPER_LID_ABOVE_CORNERS_MM, 0.0)) * scale);
        let pixel = project(&screen_from_head.transform_point(&lid_head));
        for i in lid {
            landmarks[i] = Point2::new(
                pixel.x + landmark_noise_px * upper_lid_rng.gaussian(),
                pixel.y + landmark_noise_px * upper_lid_rng.gaussian(),
            );
        }
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
    .zip(lower_lid_px)
    .map(
        |(((side, lateral_idx, medial_idx), centre), lower_lid_px)| {
            let direction = (target - centre).normalize();
            let pupil_screen = centre + direction * params.rotation_to_pupil_mm;
            let pupil_pixel = detected_iris(
                response,
                &project,
                screen_from_head,
                scale,
                side,
                &centre,
                &lower_lid_px,
                &project(&pupil_screen),
            );
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
        },
    )
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
