//! MediaPipe landmarks to gaze rays: PnP head pose against a rigid 3D face template, then the
//! eyeball model per eye.

use std::collections::HashMap;

use eye_core::angles::{direction_from_yaw_pitch, yaw_pitch_from_direction};
use eye_core::log::field;
use eye_core::observation::{LandmarkScheme, mediapipe478};
use eye_core::stage::{GazeEstimator, StageError};
use eye_core::{
    CameraId, CameraModel, FaceObservation, GazeRay, Measured, Observations, RaySource, Rig, Side,
    SourcedRay, Timestamp,
};
use eye_geometry::camera::{Intrinsics, pixel_ray};
use eye_geometry::eyeball::{
    EyeCentre, EyeParams, eyeball_centre_in_head, optical_axis, ray_sphere_near,
    screen_from_viewer, visual_axis,
};
use eye_geometry::face_template::MEDIAPIPE_RIGID;
use eye_geometry::pnp::{Pose, solve_pnp};
use eye_geometry::uncertainty::{block_diag, isotropic2, propagate_fn};
use nalgebra::{
    Isometry3, Matrix2, Point2, Point3, SVector, Translation3, UnitQuaternion, Vector2, Vector3,
    Vector6,
};
use serde::Deserialize;

use crate::EstimateError;
use crate::log::trace_ray;
use crate::options::parse_options;

/// How the detected iris centre responds to the true pupil: `measured = offset_mm + gain * true`
/// per head axis `[lateral (temporal +), vertical (down +)]`, where `true` is the true pupil's
/// image offset in head-frame mm, laterally from the eyeball rotation centre's image and
/// vertically from the eye-corner midpoint's image. The vertical anchor sits on the face surface
/// so that PnP pitch error does not move it. The vertical `measured` is the iris minus
/// `lower_lid_weight` times the lower lid, both about that midpoint. The estimator inverts this
/// before the eyeball model.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IrisResponse {
    pub gain: [f64; 2],
    pub offset_mm: [f64; 2],
    #[serde(default)]
    pub lower_lid_weight: f64,
}

impl IrisResponse {
    /// The detected iris is the true pupil.
    pub const IDENTITY: Self = Self {
        gain: [1.0, 1.0],
        offset_mm: [0.0, 0.0],
        lower_lid_weight: 0.0,
    };

    /// MediaPipe Face Landmarker (478) iris, fitted on 26 recorded dot sessions of one user
    /// (notes/eye/compress-20261011). `offset_mm` is for the optical axis (kappa applied after).
    pub const MEDIAPIPE_478: Self = Self {
        gain: [0.493, 0.175],
        offset_mm: [0.63, -3.29],
        lower_lid_weight: 0.5,
    };
}

pub(crate) const RIGHT_LOWER_LID: [usize; 3] = [144, 145, 153];
pub(crate) const LEFT_LOWER_LID: [usize; 3] = [373, 374, 380];
/// Pairwise opposite the `*_LOWER_LID` landmarks.
pub(crate) const RIGHT_UPPER_LID: [usize; 3] = [160, 159, 158];
pub(crate) const LEFT_UPPER_LID: [usize; 3] = [387, 386, 385];

/// One frame's eye geometry the [`IrisResponse`] is inverted about: points in the camera frame.
struct EyeAnchors {
    rotation_centre: Point3<f64>,
    corner_midpoint: Point3<f64>,
    camera_from_head: UnitQuaternion<f64>,
    lower_lid_px: Point2<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct LandmarkOptions {
    pub landmark_sigma_px: f64,
    pub max_reprojection_px: f64,
    pub max_iris_miss_mm: f64,
    pub apply_kappa: bool,
    pub iris_response: IrisResponse,
    /// An eye whose aperture (lid gap over corner distance, image px) falls below this fraction
    /// of its running reference is dropped as closed. 0 disables the gate.
    pub min_aperture_ratio: f64,
    /// Time constant of the running aperture reference, an exponential average over all frames.
    pub aperture_time_constant_s: f64,
}

impl Default for LandmarkOptions {
    fn default() -> Self {
        Self {
            landmark_sigma_px: 1.5,
            max_reprojection_px: 6.0,
            max_iris_miss_mm: 2.0,
            apply_kappa: true,
            iris_response: IrisResponse::MEDIAPIPE_478,
            min_aperture_ratio: 0.75,
            aperture_time_constant_s: 2.0,
        }
    }
}

#[derive(Debug)]
pub struct LandmarkEstimator {
    options: LandmarkOptions,
    params: EyeParams,
    last_pose: HashMap<CameraId, Isometry3<f64>>,
    aperture_reference: HashMap<(CameraId, Side), ApertureReference>,
}

#[derive(Debug, Clone, Copy)]
struct ApertureReference {
    at: Timestamp,
    aperture: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LandmarkEye {
    pub side: Side,
    pub centre: EyeCentre,
    /// As detected.
    pub iris_px: Measured<Point2<f64>>,
    /// `iris_px` with the [`IrisResponse`] inverted: the pupil the eyeball model uses.
    pub pupil_px: Point2<f64>,
    /// Covariance of `pupil_px` from the iris noise and the head-pose error, px².
    pub pupil_cov: Matrix2<f64>,
    pub ray: GazeRay,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LandmarkFrame {
    pub camera: CameraId,
    pub timestamp: Timestamp,
    pub pose: Pose,
    pub viewer: UnitQuaternion<f64>,
    pub eyes: Vec<LandmarkEye>,
}

impl LandmarkEstimator {
    pub const NAME: &'static str = "landmark";

    pub fn new(options: LandmarkOptions) -> Self {
        Self {
            options,
            params: EyeParams::default(),
            last_pose: HashMap::new(),
            aperture_reference: HashMap::new(),
        }
    }

    pub fn from_config(table: &toml::Table, _rig: &eye_core::Rig) -> Result<Self, StageError> {
        Ok(Self::new(parse_options(Self::NAME, table)?))
    }

    /// Overrides the anatomical priors.
    pub fn with_params(self, params: EyeParams) -> Self {
        Self { params, ..self }
    }

    /// `self.params` with kappa zeroed per `self.options.apply_kappa`.
    pub fn effective_params(&self) -> EyeParams {
        self.params.effective(self.options.apply_kappa)
    }

    /// Detailed result for one RGB observation; reused by `FusedEstimator`.
    pub fn estimate_frame(
        &mut self,
        obs: &Observations,
        rig: &Rig,
    ) -> Result<Option<LandmarkFrame>, EstimateError> {
        let Some(face) = obs
            .face
            .as_ref()
            .filter(|f| f.scheme == LandmarkScheme::MEDIAPIPE_478)
        else {
            return Ok(None);
        };
        let cam = rig
            .camera(obs.camera.as_str())
            .ok_or_else(|| EstimateError::UnknownCamera(obs.camera.to_string()))?;
        let Some(pose) = self.head_pose(&obs.camera, cam, face) else {
            self.last_pose.remove(&obs.camera);
            return Ok(None);
        };
        let warm_start = self.last_pose.contains_key(&obs.camera);
        tracing::trace!(
            rms_px = pose.rms_px,
            tx_mm = pose.camera_from_object.translation.vector.x,
            ty_mm = pose.camera_from_object.translation.vector.y,
            tz_mm = pose.camera_from_object.translation.vector.z,
            rotation_rad = pose.camera_from_object.rotation.angle(),
            warm_start,
            "head pose"
        );
        self.last_pose
            .insert(obs.camera.clone(), pose.camera_from_object);
        let viewer = screen_from_viewer(
            &cam.screen_from_camera.rotation,
            &pose.camera_from_object.rotation,
        );
        let mut eyes = Vec::new();
        for side in [Side::Right, Side::Left] {
            if self.eye_open(&obs.camera, side, face, obs.timestamp)
                && let Some(eye) = self.eye(side, face, &pose, cam, &viewer, obs.timestamp)
            {
                eyes.push(eye);
            }
        }
        for eye in &eyes {
            trace_ray(Self::NAME, &eye.ray);
        }
        Ok(Some(LandmarkFrame {
            camera: obs.camera.clone(),
            timestamp: obs.timestamp,
            pose,
            viewer,
            eyes,
        }))
    }

    fn head_pose(&self, id: &CameraId, cam: &CameraModel, face: &FaceObservation) -> Option<Pose> {
        if face.landmarks.len() != mediapipe478::COUNT {
            tracing::debug!(
                { field::REASON } = "landmark_count",
                landmarks = face.landmarks.len() as u64,
                "head pose rejected"
            );
            return None;
        }
        let (object, image) = MEDIAPIPE_RIGID
            .correspondences(&face.landmarks, self.options.landmark_sigma_px)
            .inspect_err(|e| {
                tracing::debug!(
                    { field::REASON } = "correspondences_failed",
                    error = %e,
                    "head pose rejected"
                );
            })
            .ok()?;
        let pose = solve_pnp(
            &Intrinsics::from_camera_model(cam),
            &object,
            &image,
            self.last_pose.get(id),
        )
        .inspect_err(|e| {
            tracing::debug!(
                { field::REASON } = "pnp_failed",
                error = %e,
                "head pose rejected"
            );
        })
        .ok()?;
        if pose.rms_px > self.options.max_reprojection_px {
            tracing::debug!(
                { field::REASON } = "rms_too_high",
                rms_px = pose.rms_px,
                max_reprojection_px = self.options.max_reprojection_px,
                "head pose rejected"
            );
            return None;
        }
        Some(pose)
    }

    /// Updates `side`'s running aperture reference with this frame and reports whether the eye
    /// was open relative to the reference before the update. A missing reference, or one later
    /// than `at`, restarts from this frame.
    fn eye_open(
        &mut self,
        camera: &CameraId,
        side: Side,
        face: &FaceObservation,
        at: Timestamp,
    ) -> bool {
        let Some(aperture) = eye_aperture(face, side) else {
            return true;
        };
        let key = (camera.clone(), side);
        let previous = self
            .aperture_reference
            .get(&key)
            .filter(|reference| at >= reference.at);
        let (ratio, updated) = match previous {
            Some(reference) => {
                let dt_s = at.nanos_since(reference.at) as f64 / 1e9;
                let alpha = 1.0 - (-dt_s / self.options.aperture_time_constant_s).exp();
                (
                    aperture / reference.aperture,
                    reference.aperture + alpha * (aperture - reference.aperture),
                )
            }
            None => (1.0, aperture),
        };
        self.aperture_reference.insert(
            key,
            ApertureReference {
                at,
                aperture: updated,
            },
        );
        if ratio < self.options.min_aperture_ratio {
            tracing::debug!(
                { field::REASON } = "eye_closed",
                side = side.as_str(),
                aperture_ratio = ratio,
                min_aperture_ratio = self.options.min_aperture_ratio,
                "eye dropped"
            );
            return false;
        }
        true
    }

    fn eye(
        &self,
        side: Side,
        face: &FaceObservation,
        pose: &Pose,
        cam: &CameraModel,
        viewer: &UnitQuaternion<f64>,
        at: Timestamp,
    ) -> Option<LandmarkEye> {
        let (inner_idx, outer_idx) = match side {
            Side::Right => (
                mediapipe478::RIGHT_EYE_MEDIAL,
                mediapipe478::RIGHT_EYE_LATERAL,
            ),
            Side::Left => (
                mediapipe478::LEFT_EYE_MEDIAL,
                mediapipe478::LEFT_EYE_LATERAL,
            ),
        };
        let e_head = eyeball_centre_in_head(
            &MEDIAPIPE_RIGID.point(inner_idx)?,
            &MEDIAPIPE_RIGID.point(outer_idx)?,
            &self.params,
        );

        let corner_mid_head = corner_midpoint(
            &MEDIAPIPE_RIGID.point(inner_idx)?,
            &MEDIAPIPE_RIGID.point(outer_idx)?,
        );

        let (r0, t0) = (
            pose.camera_from_object.rotation,
            pose.camera_from_object.translation.vector,
        );
        let screen_from_camera = cam.screen_from_camera;
        let perturbed = |x: &[f64]| -> Isometry3<f64> {
            Isometry3::from_parts(
                Translation3::from(t0 + Vector3::new(x[3], x[4], x[5])),
                UnitQuaternion::from_scaled_axis(Vector3::new(x[0], x[1], x[2])) * r0,
            )
        };
        let e_of = |x: &Vector6<f64>| -> Option<Vector3<f64>> {
            Some((screen_from_camera * perturbed(x.as_slice()) * e_head).coords)
        };
        let Some((e, cov)) = propagate_fn::<3, 6>(e_of, &Vector6::zeros(), &pose.cov) else {
            tracing::debug!(
                { field::REASON } = "propagate_failed",
                side = side.as_str(),
                "eye dropped"
            );
            return None;
        };
        let centre = EyeCentre {
            position: Point3::from(e),
            cov,
        };

        let Some(eye) = face.eye(side) else {
            tracing::debug!(
                { field::REASON } = "eye_missing",
                side = side.as_str(),
                "eye dropped"
            );
            return None;
        };
        let Some(iris_px) = eye.iris.map(|m| m.map(|ellipse| ellipse.center())) else {
            tracing::debug!(
                { field::REASON } = "no_iris",
                side = side.as_str(),
                "eye dropped"
            );
            return None;
        };

        let intr = Intrinsics::from_camera_model(cam);
        let response = self.options.iris_response;
        let lid = match side {
            Side::Right => RIGHT_LOWER_LID,
            Side::Left => LEFT_LOWER_LID,
        };
        let lower_lid_px = Point2::from(
            lid.iter()
                .map(|&i| face.landmarks.get(i).map(|p| p.coords))
                .sum::<Option<Vector2<f64>>>()?
                / lid.len() as f64,
        );
        let anchors = |camera_from_head: &Isometry3<f64>, lower_lid_px: Point2<f64>| EyeAnchors {
            rotation_centre: camera_from_head * e_head,
            corner_midpoint: camera_from_head * corner_mid_head,
            camera_from_head: camera_from_head.rotation,
            lower_lid_px,
        };
        let Some(pupil_px) = invert_iris_response(
            &response,
            side,
            &intr,
            &anchors(&pose.camera_from_object, lower_lid_px),
            iris_px.value(),
        ) else {
            tracing::debug!(
                { field::REASON } = "iris_response_degenerate",
                side = side.as_str(),
                "eye dropped"
            );
            return None;
        };

        let (o, u) = pixel_ray(cam, &pupil_px)
            .inspect_err(|err| {
                tracing::debug!(
                    { field::REASON } = "pixel_ray_failed",
                    side = side.as_str(),
                    error = %err,
                    "eye dropped"
                );
            })
            .ok()?;
        let miss =
            (centre.position - o).cross(&u.into_inner()).norm() - self.params.rotation_to_pupil_mm;
        if miss > self.options.max_iris_miss_mm {
            tracing::debug!(
                { field::REASON } = "iris_miss",
                side = side.as_str(),
                miss_mm = miss,
                max_iris_miss_mm = self.options.max_iris_miss_mm,
                "eye dropped"
            );
            return None;
        }

        let params = self.effective_params();
        let angles_of = |response: &IrisResponse, x: &SVector<f64, 10>| -> Option<Vector2<f64>> {
            let camera_from_head = perturbed(x.as_slice());
            let e = screen_from_camera * camera_from_head * e_head;
            let pupil = invert_iris_response(
                response,
                side,
                &intr,
                &anchors(&camera_from_head, Point2::new(x[8], x[9])),
                &Point2::new(x[6], x[7]),
            )?;
            let (o, d) = pixel_ray(cam, &pupil).ok()?;
            let p = ray_sphere_near(&o, &d, &e, params.rotation_to_pupil_mm);
            let opt = optical_axis(&e, &p)?;
            Some(yaw_pitch_from_direction(&visual_axis(
                &opt,
                &params.kappa,
                side,
                viewer,
            )))
        };
        let x0 = SVector::<f64, 10>::from_row_slice(&[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            iris_px.value().x,
            iris_px.value().y,
            lower_lid_px.x,
            lower_lid_px.y,
        ]);
        let lid_sigma = self.options.landmark_sigma_px / (lid.len() as f64).sqrt();
        let cov_in = block_diag::<8, 2, 10>(
            &block_diag::<6, 2, 8>(&pose.cov, &isotropic2(iris_px.sigma())),
            &isotropic2(lid_sigma),
        );
        // Deliberately the covariance of the ray read from the uncorrected iris: the inverted ray's
        // own (about 1/gain larger) makes the fused selector prefer less accurate IR candidates.
        let uncorrected = |x: &SVector<f64, 10>| angles_of(&IrisResponse::IDENTITY, x);
        let (Some(angles), Some((_, angular_cov))) = (
            angles_of(&response, &x0),
            propagate_fn::<2, 10>(uncorrected, &x0, &cov_in),
        ) else {
            tracing::debug!(
                { field::REASON } = "gaze_ray_failed",
                side = side.as_str(),
                "eye dropped"
            );
            return None;
        };
        let ray = GazeRay {
            side: Some(side),
            timestamp: at,
            origin: centre.position,
            direction: direction_from_yaw_pitch(&angles),
            angular_cov,
            origin_cov: centre.cov,
            head_rotation: Some(*viewer),
        };

        let pupil_of = |x: &SVector<f64, 10>| -> Option<Vector2<f64>> {
            let camera_from_head = perturbed(x.as_slice());
            invert_iris_response(
                &response,
                side,
                &intr,
                &anchors(&camera_from_head, Point2::new(x[8], x[9])),
                &Point2::new(x[6], x[7]),
            )
            .map(|p| p.coords)
        };
        let (_, pupil_cov) = propagate_fn::<2, 10>(pupil_of, &x0, &cov_in)?;
        Some(LandmarkEye {
            side,
            centre,
            iris_px,
            pupil_px,
            pupil_cov,
            ray,
        })
    }
}

/// Mean upper-to-lower lid distance over the eye-corner distance, in image px. `None` when a
/// landmark is missing or the corners coincide.
fn eye_aperture(face: &FaceObservation, side: Side) -> Option<f64> {
    let (upper, lower, medial, lateral) = match side {
        Side::Right => (
            RIGHT_UPPER_LID,
            RIGHT_LOWER_LID,
            mediapipe478::RIGHT_EYE_MEDIAL,
            mediapipe478::RIGHT_EYE_LATERAL,
        ),
        Side::Left => (
            LEFT_UPPER_LID,
            LEFT_LOWER_LID,
            mediapipe478::LEFT_EYE_MEDIAL,
            mediapipe478::LEFT_EYE_LATERAL,
        ),
    };
    let landmark = |i: usize| face.landmarks.get(i);
    let width = (landmark(medial)? - landmark(lateral)?).norm();
    if width <= f64::EPSILON {
        return None;
    }
    let gap = upper
        .iter()
        .zip(lower)
        .map(|(&u, l)| Some((landmark(u)? - landmark(l)?).norm()))
        .sum::<Option<f64>>()?
        / upper.len() as f64;
    Some(gap / width)
}

fn corner_midpoint(inner: &Point3<f64>, outer: &Point3<f64>) -> Point3<f64> {
    Point3::from((inner.coords + outer.coords) / 2.0)
}

/// The pupil pixel the eyeball model expects for a detected `iris_px`: inverts `response` with
/// lateral offsets about the rotation centre and vertical offsets about the corner midpoint, along
/// the head's x and y axes.
fn invert_iris_response(
    response: &IrisResponse,
    side: Side,
    intr: &Intrinsics,
    anchors: &EyeAnchors,
    iris_px: &Point2<f64>,
) -> Option<Point2<f64>> {
    if *response == IrisResponse::IDENTITY {
        return Some(*iris_px);
    }
    let lateral_px = intr.project(&anchors.rotation_centre).ok()?;
    let vertical_px = intr.project(&anchors.corner_midpoint).ok()?;
    let axis_px = |axis: Vector3<f64>| -> Option<Vector2<f64>> {
        Some(
            intr.project(&(anchors.rotation_centre + anchors.camera_from_head * axis))
                .ok()?
                - lateral_px,
        )
    };
    let head_px = Matrix2::from_columns(&[axis_px(Vector3::x())?, axis_px(Vector3::y())?]);
    let px_head = head_px.try_inverse()?;
    let temporal_x = match side {
        Side::Right => -1.0,
        Side::Left => 1.0,
    };
    let offset = Vector2::new(temporal_x * response.offset_mm[0], response.offset_mm[1]);
    let lower_lid_y = (px_head * (anchors.lower_lid_px - vertical_px)).y;
    let measured = Vector2::new(
        (px_head * (iris_px - lateral_px)).x,
        (px_head * (iris_px - vertical_px)).y - response.lower_lid_weight * lower_lid_y,
    );
    let truth = (measured - offset).component_div(&Vector2::from(response.gain));
    let vertical_shift = (px_head * (vertical_px - lateral_px)).y;
    Some(lateral_px + head_px * Vector2::new(truth.x, truth.y + vertical_shift))
}

impl GazeEstimator for LandmarkEstimator {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn estimate(&mut self, obs: &[Observations], rig: &Rig) -> Result<Vec<SourcedRay>, StageError> {
        let mut rays = Vec::new();
        for o in obs {
            if let Some(frame) = self.estimate_frame(o, rig)? {
                rays.extend(frame.eyes.into_iter().map(|e| e.ray));
            }
        }
        Ok(SourcedRay::tag(RaySource::RgbOnly, rays))
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use approx::assert_abs_diff_eq;
    use eye_geometry::angles::yaw_pitch_from_direction;
    use eye_geometry::eyeball::visual_axis;
    use eye_log::testing::capture_logs;
    use eye_log::{Level, Value};
    use nalgebra::{Matrix2, Translation3, Unit, Vector2};

    use super::*;
    use crate::testutil::{
        synthetic_eye_centres, synthetic_ir_observation, synthetic_rgb_observation,
        synthetic_rgb_observation_with_response, test_rig,
    };

    fn frontal_screen_from_head() -> Isometry3<f64> {
        Isometry3::from_parts(
            Translation3::new(155.0, 40.0, -500.0),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI),
        )
    }

    fn angle_deg(a: &Unit<Vector3<f64>>, b: &Unit<Vector3<f64>>) -> f64 {
        a.dot(b).clamp(-1.0, 1.0).acos().to_degrees()
    }

    #[test]
    fn test_frontal_head_pose_is_recovered() {
        let rig = test_rig();
        let cam = rig.camera("rgb").expect("rig has an rgb camera");
        let screen_from_head = frontal_screen_from_head();
        let obs = synthetic_rgb_observation(
            &rig,
            &screen_from_head,
            1.0,
            Point2::new(155.0, 85.0),
            0.0,
            0.0,
            1,
        );
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());

        let frame = estimator
            .estimate_frame(&obs, &rig)
            .expect("estimate succeeds")
            .expect("frame is recovered");

        let truth = cam.screen_from_camera.inverse() * screen_from_head;
        let dt =
            (frame.pose.camera_from_object.translation.vector - truth.translation.vector).norm();
        assert!(dt < 0.5, "translation error {dt} mm");
        let dr = (frame.pose.camera_from_object.rotation * truth.rotation.inverse()).angle();
        assert!(
            dr.to_degrees() < 0.05,
            "rotation error {} deg",
            dr.to_degrees()
        );
    }

    #[test]
    fn test_noise_free_face_recovers_optical_axis_within_0_1_deg() {
        let rig = test_rig();
        let heads = [
            frontal_screen_from_head(),
            Isometry3::from_parts(
                Translation3::new(155.0, 40.0, -500.0),
                UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI + 15f64.to_radians()),
            ),
        ];
        let targets = [
            Point2::new(0.0, 0.0),
            Point2::new(310.0, 0.0),
            Point2::new(0.0, 170.0),
            Point2::new(310.0, 170.0),
            Point2::new(155.0, 85.0),
        ];
        for screen_from_head in &heads {
            let centres = synthetic_eye_centres(screen_from_head, 1.0, &EyeParams::default());
            for target in targets {
                let obs =
                    synthetic_rgb_observation(&rig, screen_from_head, 1.0, target, 0.0, 0.0, 1);
                let mut estimator = LandmarkEstimator::new(LandmarkOptions {
                    apply_kappa: false,
                    ..Default::default()
                });
                let frame = estimator
                    .estimate_frame(&obs, &rig)
                    .expect("estimate succeeds")
                    .expect("frame is recovered");
                assert_eq!(frame.eyes.len(), 2, "target {target:?}: expected two eyes");
                for (eye, centre_true) in frame.eyes.iter().zip(centres) {
                    let true_direction =
                        Unit::new_normalize(Point3::new(target.x, target.y, 0.0) - centre_true);
                    let error_deg = angle_deg(&eye.ray.direction, &true_direction);
                    assert!(
                        error_deg <= 0.1,
                        "target {target:?} side {:?}: error {error_deg} deg",
                        eye.side
                    );
                }
            }
        }
    }

    #[test]
    fn test_origin_is_template_rotation_centre() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &EyeParams::default());
        let obs = synthetic_rgb_observation(
            &rig,
            &screen_from_head,
            1.0,
            Point2::new(155.0, 85.0),
            0.0,
            0.0,
            1,
        );
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());

        let frame = estimator
            .estimate_frame(&obs, &rig)
            .expect("estimate succeeds")
            .expect("frame is recovered");

        for (eye, centre_true) in frame.eyes.iter().zip(centres) {
            let d = (eye.ray.origin - centre_true).norm();
            assert!(d < 0.5, "side {:?}: origin off by {d} mm", eye.side);
        }
    }

    #[test]
    fn test_head_translation_keeps_gaze_on_target() {
        let rig = test_rig();
        let screen_from_head = Isometry3::from_parts(
            Translation3::new(205.0, 40.0, -450.0),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI),
        );
        let target = Point2::new(100.0, 50.0);
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &EyeParams::default());
        let obs = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 1);
        let mut estimator = LandmarkEstimator::new(LandmarkOptions {
            apply_kappa: false,
            ..Default::default()
        });

        let frame = estimator
            .estimate_frame(&obs, &rig)
            .expect("estimate succeeds")
            .expect("frame is recovered");

        for (eye, centre_true) in frame.eyes.iter().zip(centres) {
            let true_direction =
                Unit::new_normalize(Point3::new(target.x, target.y, 0.0) - centre_true);
            let error_deg = angle_deg(&eye.ray.direction, &true_direction);
            assert!(
                error_deg <= 0.1,
                "side {:?}: error {error_deg} deg",
                eye.side
            );
        }
    }

    #[test]
    fn test_iris_far_outside_sphere_drops_eye() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let mut obs = synthetic_rgb_observation(
            &rig,
            &screen_from_head,
            1.0,
            Point2::new(155.0, 85.0),
            0.0,
            0.0,
            1,
        );
        {
            let face = obs.face.as_mut().expect("face is present");
            let right = face
                .eyes
                .iter_mut()
                .find(|e| e.side == Side::Right)
                .expect("right eye is present");
            let iris = right.iris.as_mut().expect("right iris is present");
            let ellipse = *iris.value();
            let shifted = eye_core::Ellipse2::circle(
                Point2::new(ellipse.center().x - 100.0, ellipse.center().y),
                ellipse.semi_major(),
            )
            .expect("shifted ellipse is valid");
            *iris = Measured::new(shifted, iris.sigma()).expect("sigma is valid");
        }
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());

        let frame = estimator
            .estimate_frame(&obs, &rig)
            .expect("estimate succeeds")
            .expect("frame is recovered");

        assert!(frame.eyes.iter().all(|e| e.side != Side::Right));
        assert!(frame.eyes.iter().any(|e| e.side == Side::Left));
    }

    #[test]
    fn test_bad_pose_rms_returns_none_and_clears_warm_start() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let good = synthetic_rgb_observation(
            &rig,
            &screen_from_head,
            1.0,
            Point2::new(155.0, 85.0),
            0.0,
            0.0,
            1,
        );
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());
        estimator
            .estimate_frame(&good, &rig)
            .expect("estimate succeeds")
            .expect("frame is recovered");
        assert!(estimator.last_pose.contains_key(&CameraId::new("rgb")));

        let mut bad = good;
        {
            let face = bad.face.as_mut().expect("face is present");
            let indices: Vec<usize> = MEDIAPIPE_RIGID.points.iter().map(|&(i, _)| i).collect();
            let values: Vec<Point2<f64>> = indices.iter().map(|&i| face.landmarks[i]).collect();
            for (i, v) in indices.iter().zip(values.iter().rev()) {
                face.landmarks[*i] = *v;
            }
        }

        let frame = estimator
            .estimate_frame(&bad, &rig)
            .expect("estimate succeeds");
        assert!(frame.is_none());
        assert!(!estimator.last_pose.contains_key(&CameraId::new("rgb")));
    }

    #[test]
    fn test_ir_observation_is_ignored() {
        let rig = test_rig();
        let obs =
            synthetic_ir_observation(&rig, Point2::new(155.0, 85.0), Vector3::zeros(), 0.0, 1);
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());

        let frame = estimator
            .estimate_frame(&obs, &rig)
            .expect("estimate succeeds");
        assert!(frame.is_none());

        let rays = estimator.estimate(&[obs], &rig).expect("estimate succeeds");
        assert!(rays.is_empty());
    }

    #[test]
    fn test_angular_cov_matches_monte_carlo() {
        assert_angular_cov_matches_monte_carlo(&IrisResponse::IDENTITY, 1.5, 1.0);
    }

    #[test]
    fn test_angular_cov_is_that_of_the_uncorrected_ray() {
        let rig = test_rig();
        let obs = synthetic_rgb_observation(
            &rig,
            &frontal_screen_from_head(),
            1.0,
            Point2::new(100.0, 50.0),
            0.5,
            0.3,
            7,
        );
        let corrected = right_eye_ray(LandmarkOptions::default(), &obs, &rig);
        let uncorrected = right_eye_ray(
            LandmarkOptions {
                iris_response: IrisResponse::IDENTITY,
                ..Default::default()
            },
            &obs,
            &rig,
        );

        assert!(angle_deg(&corrected.ray.direction, &uncorrected.ray.direction) > 1.0);
        assert_abs_diff_eq!(
            corrected.ray.angular_cov,
            uncorrected.ray.angular_cov,
            epsilon = 1e-15
        );
    }

    fn assert_angular_cov_matches_monte_carlo(
        response: &IrisResponse,
        landmark_sigma_px: f64,
        iris_sigma_px: f64,
    ) {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let options = LandmarkOptions {
            landmark_sigma_px,
            apply_kappa: false,
            iris_response: *response,
            ..Default::default()
        };
        let observe = |landmark_noise_px, iris_noise_px, seed| {
            synthetic_rgb_observation_with_response(
                &rig,
                &screen_from_head,
                1.0,
                target,
                landmark_noise_px,
                iris_noise_px,
                seed,
                response,
            )
        };

        let mut noise_free = observe(0.0, 0.0, 0);
        for eye in noise_free
            .face
            .as_mut()
            .expect("face is present")
            .eyes
            .iter_mut()
        {
            eye.iris = eye
                .iris
                .map(|m| Measured::new(m.into_value(), iris_sigma_px).expect("valid sigma"));
        }
        let mut reference_estimator = LandmarkEstimator::new(options.clone());
        let reference_frame = reference_estimator
            .estimate_frame(&noise_free, &rig)
            .expect("estimate succeeds")
            .expect("frame is recovered");
        let predicted = reference_frame.eyes[0].ray.angular_cov;

        let n = 2000;
        let mut samples = Vec::with_capacity(n);
        for seed in 0..n as u64 {
            let obs = observe(landmark_sigma_px, iris_sigma_px, seed);
            let mut estimator = LandmarkEstimator::new(options.clone());
            let frame = estimator
                .estimate_frame(&obs, &rig)
                .expect("estimate succeeds")
                .expect("frame is recovered");
            samples.push(yaw_pitch_from_direction(&frame.eyes[0].ray.direction));
        }

        let mut sum = Vector2::zeros();
        for s in &samples {
            sum += s;
        }
        let mean = sum / n as f64;
        let mut empirical = Matrix2::zeros();
        for s in &samples {
            let d = s - mean;
            empirical += d * d.transpose();
        }
        empirical /= (n - 1) as f64;

        for k in 0..2 {
            let predicted_std = predicted[(k, k)].sqrt();
            let empirical_std = empirical[(k, k)].sqrt();
            let tol = 0.2 * predicted_std;
            assert!(
                (empirical_std - predicted_std).abs() <= tol,
                "k={k}: empirical std {empirical_std} predicted std {predicted_std}"
            );
        }
        let tol_offdiag = 0.1 * (predicted[(0, 0)] * predicted[(1, 1)]).sqrt();
        assert!(
            (empirical[(0, 1)] - predicted[(0, 1)]).abs() <= tol_offdiag,
            "empirical offdiag {} predicted offdiag {}",
            empirical[(0, 1)],
            predicted[(0, 1)]
        );
    }

    #[test]
    fn test_origin_cov_reflects_pose_cov() {
        let rig = test_rig();
        let cam = rig.camera("rgb").expect("rig has an rgb camera");
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let obs = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 0);

        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());
        let frame = estimator
            .estimate_frame(&obs, &rig)
            .expect("estimate succeeds")
            .expect("frame is recovered");

        let params = EyeParams::default();
        let e_head = eyeball_centre_in_head(
            &MEDIAPIPE_RIGID.point(133).expect("landmark in template"),
            &MEDIAPIPE_RIGID.point(33).expect("landmark in template"),
            &params,
        );
        let pose = frame.pose;
        let (r0, t0) = (
            pose.camera_from_object.rotation,
            pose.camera_from_object.translation.vector,
        );
        let screen_from_camera = cam.screen_from_camera;
        let e_of = |x: &Vector6<f64>| -> Option<Vector3<f64>> {
            let rotation = UnitQuaternion::from_scaled_axis(Vector3::new(x[0], x[1], x[2])) * r0;
            let camera_from_head = Isometry3::from_parts(
                Translation3::from(t0 + Vector3::new(x[3], x[4], x[5])),
                rotation,
            );
            Some((screen_from_camera * camera_from_head * e_head).coords)
        };
        let (_, expected_cov) =
            propagate_fn::<3, 6>(e_of, &Vector6::zeros(), &pose.cov).expect("propagation succeeds");

        assert_abs_diff_eq!(frame.eyes[0].centre.cov, expected_cov, epsilon = 1e-12);
        assert_abs_diff_eq!(frame.eyes[0].ray.origin_cov, expected_cov, epsilon = 1e-12);

        let trace1 = expected_cov.trace();

        let doubled = LandmarkOptions {
            landmark_sigma_px: LandmarkOptions::default().landmark_sigma_px * 2.0,
            ..Default::default()
        };
        let mut estimator2 = LandmarkEstimator::new(doubled);
        let frame2 = estimator2
            .estimate_frame(&obs, &rig)
            .expect("estimate succeeds")
            .expect("frame is recovered");
        let trace2 = frame2.eyes[0].centre.cov.trace();

        let ratio = trace2 / trace1;
        assert!((3.96..=4.04).contains(&ratio), "trace ratio {ratio}");
    }

    #[test]
    fn test_apply_kappa_follows_head_rotation() {
        let rig = test_rig();
        let screen_from_head = Isometry3::from_parts(
            Translation3::new(155.0, 40.0, -500.0),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI)
                * UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 30f64.to_radians()),
        );
        let target = Point2::new(155.0, 85.0);
        let obs = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 1);

        let mut with_kappa = LandmarkEstimator::new(LandmarkOptions {
            apply_kappa: true,
            ..Default::default()
        });
        let mut without_kappa = LandmarkEstimator::new(LandmarkOptions {
            apply_kappa: false,
            ..Default::default()
        });

        let frame_kappa = with_kappa
            .estimate_frame(&obs, &rig)
            .expect("estimate succeeds")
            .expect("frame is recovered");
        let frame_plain = without_kappa
            .estimate_frame(&obs, &rig)
            .expect("estimate succeeds")
            .expect("frame is recovered");

        for (eye_kappa, eye_plain) in frame_kappa.eyes.iter().zip(&frame_plain.eyes) {
            let expected = visual_axis(
                &eye_plain.ray.direction,
                &EyeParams::default().kappa,
                eye_plain.side,
                &frame_plain.viewer,
            );
            assert_abs_diff_eq!(eye_kappa.ray.direction, expected, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_eye_corner_indices_resolve_in_rigid_template() {
        for idx in [
            mediapipe478::RIGHT_EYE_MEDIAL,
            mediapipe478::RIGHT_EYE_LATERAL,
            mediapipe478::LEFT_EYE_MEDIAL,
            mediapipe478::LEFT_EYE_LATERAL,
        ] {
            assert!(
                MEDIAPIPE_RIGID.point(idx).is_some(),
                "index {idx} missing from rigid template"
            );
        }
    }

    #[test]
    fn test_head_pose_rejects_non_mediapipe_count() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let mut obs = synthetic_rgb_observation(
            &rig,
            &screen_from_head,
            1.0,
            Point2::new(155.0, 85.0),
            0.0,
            0.0,
            1,
        );
        obs.face
            .as_mut()
            .expect("face is present")
            .landmarks
            .truncate(mediapipe478::COUNT - 1);
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());

        let (frame, logs) = capture_logs(tracing::Level::DEBUG, || {
            estimator
                .estimate_frame(&obs, &rig)
                .expect("estimate succeeds")
        });

        assert!(frame.is_none());
        let rec = logs
            .iter()
            .find(|r| r.message == "head pose rejected")
            .expect("landmark_count logged");
        assert_eq!(
            rec.fields[field::REASON],
            Value::Str("landmark_count".into())
        );
    }

    #[test]
    fn test_short_landmark_vector_yields_none() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let mut obs = synthetic_rgb_observation(
            &rig,
            &screen_from_head,
            1.0,
            Point2::new(155.0, 85.0),
            0.0,
            0.0,
            1,
        );
        obs.face
            .as_mut()
            .expect("face is present")
            .landmarks
            .truncate(468);
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());

        let frame = estimator
            .estimate_frame(&obs, &rig)
            .expect("estimate succeeds");
        assert!(frame.is_none());
    }

    #[test]
    fn test_unknown_camera_is_error() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let mut obs = synthetic_rgb_observation(
            &rig,
            &screen_from_head,
            1.0,
            Point2::new(155.0, 85.0),
            0.0,
            0.0,
            1,
        );
        obs.camera = CameraId::new("cam9");
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());

        let err = estimator
            .estimate_frame(&obs, &rig)
            .expect_err("unknown camera is an error");
        match err {
            EstimateError::UnknownCamera(ref camera) => assert_eq!(camera, "cam9"),
            other => panic!("expected UnknownCamera, got {other:?}"),
        }

        let stage_err = estimator
            .estimate(&[obs], &rig)
            .expect_err("unknown camera is an error");
        assert!(matches!(stage_err, StageError::Failed(_)));
    }

    fn right_eye_ray(options: LandmarkOptions, obs: &Observations, rig: &Rig) -> LandmarkEye {
        LandmarkEstimator::new(options)
            .estimate_frame(obs, rig)
            .expect("estimate succeeds")
            .expect("frame is recovered")
            .eyes
            .into_iter()
            .find(|e| e.side == Side::Right)
            .expect("right eye present")
    }

    #[test]
    fn test_compressed_iris_read_as_true_pupil_compresses_gaze() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let left = Point2::new(0.0, 85.0);
        let right = Point2::new(310.0, 85.0);
        let identity = LandmarkOptions {
            apply_kappa: false,
            iris_response: IrisResponse::IDENTITY,
            ..Default::default()
        };
        let observe =
            |target| synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 1);

        let span = |options: &LandmarkOptions| {
            let a = right_eye_ray(options.clone(), &observe(left), &rig);
            let b = right_eye_ray(options.clone(), &observe(right), &rig);
            angle_deg(&a.ray.direction, &b.ray.direction)
        };
        let centre = synthetic_eye_centres(&screen_from_head, 1.0, &EyeParams::default())[0];
        let true_span = angle_deg(
            &Unit::new_normalize(Point3::new(left.x, left.y, 0.0) - centre),
            &Unit::new_normalize(Point3::new(right.x, right.y, 0.0) - centre),
        );

        let compressed = span(&identity) / true_span;
        assert!(
            (0.4..0.6).contains(&compressed),
            "identity response span ratio {compressed}"
        );
        let inverted = span(&LandmarkOptions {
            apply_kappa: false,
            ..Default::default()
        }) / true_span;
        assert!(
            (inverted - 1.0).abs() < 0.01,
            "inverted response span ratio {inverted}"
        );
    }

    #[test]
    fn test_identity_response_uses_detected_iris_as_pupil() {
        let rig = test_rig();
        let obs = synthetic_rgb_observation_with_response(
            &rig,
            &frontal_screen_from_head(),
            1.0,
            Point2::new(100.0, 50.0),
            0.0,
            0.0,
            1,
            &IrisResponse::IDENTITY,
        );
        let eye = right_eye_ray(
            LandmarkOptions {
                iris_response: IrisResponse::IDENTITY,
                ..Default::default()
            },
            &obs,
            &rig,
        );

        assert_eq!(eye.pupil_px, *eye.iris_px.value());
    }

    #[test]
    fn test_iris_response_scales_pupil_offset_per_head_axis() {
        let rig = test_rig();
        let obs = synthetic_rgb_observation_with_response(
            &rig,
            &frontal_screen_from_head(),
            1.0,
            Point2::new(0.0, 170.0),
            0.0,
            0.0,
            1,
            &IrisResponse::IDENTITY,
        );
        let options = |gain: [f64; 2]| LandmarkOptions {
            iris_response: IrisResponse {
                gain,
                offset_mm: [0.0, 0.0],
                lower_lid_weight: 0.0,
            },
            ..Default::default()
        };
        let base = right_eye_ray(options([1.0, 0.999_999]), &obs, &rig);
        let halved_x = right_eye_ray(options([0.5, 0.999_999]), &obs, &rig);
        let halved_y = right_eye_ray(options([0.999_999, 0.5]), &obs, &rig);

        let shift_x = halved_x.pupil_px - base.pupil_px;
        let shift_y = halved_y.pupil_px - base.pupil_px;
        assert!(shift_x.x.abs() > 1.0, "lateral shift {shift_x:?}");
        assert!(
            shift_x.y.abs() < 0.05 * shift_x.x.abs(),
            "lateral shift {shift_x:?}"
        );
        assert!(shift_y.y.abs() > 1.0, "vertical shift {shift_y:?}");
        assert!(
            shift_y.x.abs() < 0.05 * shift_y.y.abs(),
            "vertical shift {shift_y:?}"
        );
    }

    #[test]
    fn test_iris_response_parses_from_config_without_lower_lid_weight() {
        let table: toml::Table =
            toml::from_str("iris_response = { gain = [0.5, 0.25], offset_mm = [0.5, -2.0] }")
                .expect("valid toml");
        let options: LandmarkOptions =
            parse_options(LandmarkEstimator::NAME, &table).expect("options parse");

        assert_eq!(
            options.iris_response,
            IrisResponse {
                gain: [0.5, 0.25],
                offset_mm: [0.5, -2.0],
                lower_lid_weight: 0.0,
            }
        );
    }

    #[test]
    fn test_from_config_rejects_unknown_option() {
        let rig = test_rig();
        let mut table = toml::Table::new();
        table.insert("bogus".into(), 1.into());

        let err = LandmarkEstimator::from_config(&table, &rig).expect_err("unknown option errors");
        assert!(matches!(err, StageError::Config(_)));
    }

    #[test]
    fn test_logs_head_pose_at_trace() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let obs = synthetic_rgb_observation(
            &rig,
            &screen_from_head,
            1.0,
            Point2::new(155.0, 85.0),
            0.0,
            0.0,
            1,
        );
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());

        let (_, logs1) = capture_logs(tracing::Level::TRACE, || {
            estimator
                .estimate_frame(&obs, &rig)
                .expect("estimate succeeds")
                .expect("frame is recovered")
        });
        let rec1 = logs1
            .iter()
            .find(|r| r.message == "head pose")
            .expect("head pose logged");
        assert_eq!(rec1.level, Level::Trace);
        assert_eq!(rec1.target, "eye_estimate::landmark");
        assert!(matches!(rec1.fields["rms_px"], Value::F64(v) if v < 1.0));
        assert!(matches!(
            rec1.fields["tz_mm"],
            Value::F64(v) if (v - 500.0).abs() < 10.0
        ));
        assert_eq!(rec1.fields["warm_start"], Value::Bool(false));

        let (_, logs2) = capture_logs(tracing::Level::TRACE, || {
            estimator
                .estimate_frame(&obs, &rig)
                .expect("estimate succeeds")
                .expect("frame is recovered")
        });
        let rec2 = logs2
            .iter()
            .find(|r| r.message == "head pose")
            .expect("head pose logged");
        assert_eq!(rec2.fields["warm_start"], Value::Bool(true));
    }

    #[test]
    fn test_logs_head_pose_rejected_at_debug() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let mut obs = synthetic_rgb_observation(
            &rig,
            &screen_from_head,
            1.0,
            Point2::new(155.0, 85.0),
            0.0,
            0.0,
            1,
        );
        obs.face
            .as_mut()
            .expect("face is present")
            .landmarks
            .truncate(468);
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());

        let (frame, logs) = capture_logs(tracing::Level::DEBUG, || {
            estimator
                .estimate_frame(&obs, &rig)
                .expect("estimate succeeds")
        });

        assert!(frame.is_none());
        let rec = logs
            .iter()
            .find(|r| r.message == "head pose rejected")
            .expect("landmark_count logged");
        assert_eq!(
            rec.fields[field::REASON],
            Value::Str("landmark_count".into())
        );
        assert_eq!(rec.fields["landmarks"], Value::U64(468));
    }

    #[test]
    fn test_logs_reversed_landmarks_rejected_at_debug() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let good = synthetic_rgb_observation(
            &rig,
            &screen_from_head,
            1.0,
            Point2::new(155.0, 85.0),
            0.0,
            0.0,
            1,
        );
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());
        estimator
            .estimate_frame(&good, &rig)
            .expect("estimate succeeds")
            .expect("frame is recovered");

        let mut bad = good;
        {
            let face = bad.face.as_mut().expect("face is present");
            let indices: Vec<usize> = MEDIAPIPE_RIGID.points.iter().map(|&(i, _)| i).collect();
            let values: Vec<Point2<f64>> = indices.iter().map(|&i| face.landmarks[i]).collect();
            for (i, v) in indices.iter().zip(values.iter().rev()) {
                face.landmarks[*i] = *v;
            }
        }

        let (frame, logs) = capture_logs(tracing::Level::DEBUG, || {
            estimator
                .estimate_frame(&bad, &rig)
                .expect("estimate succeeds")
        });

        assert!(frame.is_none());
        let rec = logs
            .iter()
            .find(|r| r.message == "head pose rejected")
            .expect("rms_too_high logged");
        assert_eq!(rec.fields[field::REASON], Value::Str("rms_too_high".into()));
        assert!(matches!(rec.fields["rms_px"], Value::F64(v) if v > 6.0));
    }

    #[test]
    fn test_logs_eye_dropped_at_debug() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let mut obs = synthetic_rgb_observation(
            &rig,
            &screen_from_head,
            1.0,
            Point2::new(155.0, 85.0),
            0.0,
            0.0,
            1,
        );
        {
            let face = obs.face.as_mut().expect("face is present");
            let right = face
                .eyes
                .iter_mut()
                .find(|e| e.side == Side::Right)
                .expect("right eye is present");
            let iris = right.iris.as_mut().expect("right iris is present");
            let ellipse = *iris.value();
            let shifted = eye_core::Ellipse2::circle(
                Point2::new(ellipse.center().x - 100.0, ellipse.center().y),
                ellipse.semi_major(),
            )
            .expect("shifted ellipse is valid");
            *iris = Measured::new(shifted, iris.sigma()).expect("sigma is valid");
        }
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());

        let (frame, logs) = capture_logs(tracing::Level::DEBUG, || {
            estimator
                .estimate_frame(&obs, &rig)
                .expect("estimate succeeds")
                .expect("frame is recovered")
        });

        assert!(frame.eyes.iter().all(|e| e.side != Side::Right));
        let rec = logs
            .iter()
            .find(|r| {
                r.message == "eye dropped"
                    && r.fields[field::REASON] == Value::Str("iris_miss".into())
            })
            .expect("iris_miss logged");
        assert_eq!(rec.fields["side"], Value::Str("right".into()));
        assert!(matches!(rec.fields["miss_mm"], Value::F64(v) if v > 2.0));

        let mut obs2 = synthetic_rgb_observation(
            &rig,
            &screen_from_head,
            1.0,
            Point2::new(155.0, 85.0),
            0.0,
            0.0,
            1,
        );
        {
            let face = obs2.face.as_mut().expect("face is present");
            let right = face
                .eyes
                .iter_mut()
                .find(|e| e.side == Side::Right)
                .expect("right eye is present");
            right.iris = None;
        }
        let mut estimator2 = LandmarkEstimator::new(LandmarkOptions::default());
        let (_, logs2) = capture_logs(tracing::Level::DEBUG, || {
            estimator2
                .estimate_frame(&obs2, &rig)
                .expect("estimate succeeds")
        });
        let rec2 = logs2
            .iter()
            .find(|r| {
                r.message == "eye dropped"
                    && r.fields[field::REASON] == Value::Str("no_iris".into())
            })
            .expect("no_iris logged");
        assert_eq!(rec2.fields["side"], Value::Str("right".into()));
    }

    const FRAME_NS: u64 = 133_000_000;

    /// A frontal face at `frame` * 133 ms with each eye's upper lid at `apertures` [right, left]
    /// times its rendered gap from the lower lid.
    fn lidded_frame(rig: &Rig, frame: u64, apertures: [f64; 2]) -> Observations {
        let mut obs = synthetic_rgb_observation(
            rig,
            &frontal_screen_from_head(),
            1.0,
            Point2::new(155.0, 85.0),
            0.0,
            0.0,
            1,
        );
        obs.timestamp = Timestamp::from_nanos(frame * FRAME_NS);
        let face = obs.face.as_mut().expect("face is present");
        let lids = [
            (RIGHT_UPPER_LID, RIGHT_LOWER_LID),
            (LEFT_UPPER_LID, LEFT_LOWER_LID),
        ];
        for (aperture, (upper, lower)) in apertures.into_iter().zip(lids) {
            for (u, l) in upper.into_iter().zip(lower) {
                let lower_px = face.landmarks[l];
                face.landmarks[u] = lower_px + (face.landmarks[u] - lower_px) * aperture;
            }
        }
        obs
    }

    fn sides_kept(estimator: &mut LandmarkEstimator, obs: &Observations, rig: &Rig) -> Vec<Side> {
        estimator
            .estimate_frame(obs, rig)
            .expect("estimate succeeds")
            .expect("frame is recovered")
            .eyes
            .iter()
            .map(|e| e.side)
            .collect()
    }

    fn open_for(estimator: &mut LandmarkEstimator, rig: &Rig, frames: std::ops::Range<u64>) {
        for frame in frames {
            assert_eq!(
                sides_kept(estimator, &lidded_frame(rig, frame, [1.0, 1.0]), rig),
                [Side::Right, Side::Left],
                "open frame {frame}"
            );
        }
    }

    #[test]
    fn test_blink_after_open_frames_drops_both_eyes() {
        let rig = test_rig();
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());
        open_for(&mut estimator, &rig, 0..8);

        let kept = sides_kept(&mut estimator, &lidded_frame(&rig, 8, [0.4, 0.4]), &rig);

        assert!(kept.is_empty(), "kept {kept:?}");
    }

    #[test]
    fn test_open_eyes_after_blink_are_kept() {
        let rig = test_rig();
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());
        open_for(&mut estimator, &rig, 0..8);
        sides_kept(&mut estimator, &lidded_frame(&rig, 8, [0.4, 0.4]), &rig);

        let kept = sides_kept(&mut estimator, &lidded_frame(&rig, 9, [1.0, 1.0]), &rig);

        assert_eq!(kept, [Side::Right, Side::Left]);
    }

    #[test]
    fn test_wink_drops_only_the_closed_eye() {
        let rig = test_rig();
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());
        open_for(&mut estimator, &rig, 0..8);

        let kept = sides_kept(&mut estimator, &lidded_frame(&rig, 8, [1.0, 0.4]), &rig);

        assert_eq!(kept, [Side::Right]);
    }

    #[test]
    fn test_slightly_narrowed_eyes_are_kept() {
        let rig = test_rig();
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());
        open_for(&mut estimator, &rig, 0..8);

        let kept = sides_kept(&mut estimator, &lidded_frame(&rig, 8, [0.85, 0.85]), &rig);

        assert_eq!(kept, [Side::Right, Side::Left]);
    }

    #[test]
    fn test_narrow_first_frame_is_kept() {
        let rig = test_rig();
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());

        let kept = sides_kept(&mut estimator, &lidded_frame(&rig, 0, [0.4, 0.4]), &rig);

        assert_eq!(kept, [Side::Right, Side::Left]);
    }

    #[test]
    fn test_persistent_narrowing_is_readmitted_within_2_s() {
        let rig = test_rig();
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());
        open_for(&mut estimator, &rig, 0..15);
        let narrowed = |estimator: &mut LandmarkEstimator, frame| {
            sides_kept(estimator, &lidded_frame(&rig, frame, [0.6, 0.6]), &rig)
        };

        assert!(narrowed(&mut estimator, 15).is_empty());
        let readmitted = (16..30).find(|&frame| narrowed(&mut estimator, frame).len() == 2);

        assert!(
            readmitted.is_some_and(|frame| (frame - 15) * FRAME_NS <= 2_000_000_000),
            "readmitted at frame {readmitted:?}"
        );
    }

    #[test]
    fn test_zero_min_aperture_ratio_keeps_closed_eyes() {
        let rig = test_rig();
        let mut estimator = LandmarkEstimator::new(LandmarkOptions {
            min_aperture_ratio: 0.0,
            ..Default::default()
        });
        open_for(&mut estimator, &rig, 0..8);

        let kept = sides_kept(&mut estimator, &lidded_frame(&rig, 8, [0.4, 0.4]), &rig);

        assert_eq!(kept, [Side::Right, Side::Left]);
    }

    #[test]
    fn test_earlier_timestamp_restarts_aperture_reference() {
        let rig = test_rig();
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());
        open_for(&mut estimator, &rig, 10..18);

        let kept = sides_kept(&mut estimator, &lidded_frame(&rig, 0, [0.4, 0.4]), &rig);

        assert_eq!(kept, [Side::Right, Side::Left]);
    }

    #[test]
    fn test_aperture_gate_parses_from_config() {
        let table: toml::Table =
            toml::from_str("min_aperture_ratio = 0.6\naperture_time_constant_s = 4.0")
                .expect("valid toml");
        let options: LandmarkOptions =
            parse_options(LandmarkEstimator::NAME, &table).expect("options parse");

        assert_eq!(options.min_aperture_ratio, 0.6);
        assert_eq!(options.aperture_time_constant_s, 4.0);
    }

    #[test]
    fn test_logs_eye_closed_at_debug() {
        let rig = test_rig();
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());
        open_for(&mut estimator, &rig, 0..8);

        let (_, logs) = capture_logs(tracing::Level::DEBUG, || {
            sides_kept(&mut estimator, &lidded_frame(&rig, 8, [0.4, 1.0]), &rig)
        });

        let rec = logs
            .iter()
            .find(|r| {
                r.message == "eye dropped"
                    && r.fields[field::REASON] == Value::Str("eye_closed".into())
            })
            .expect("eye_closed logged");
        assert_eq!(rec.fields["side"], Value::Str("right".into()));
        assert!(matches!(rec.fields["aperture_ratio"], Value::F64(v) if v < 0.5));
    }

    #[test]
    fn test_landmark_ray_timestamp_is_observation_timestamp() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let mut obs = synthetic_rgb_observation(
            &rig,
            &screen_from_head,
            1.0,
            Point2::new(155.0, 85.0),
            0.0,
            0.0,
            1,
        );
        obs.timestamp = Timestamp::from_nanos(68_000_000);
        let mut estimator = LandmarkEstimator::new(LandmarkOptions::default());

        let frame = estimator
            .estimate_frame(&obs, &rig)
            .expect("estimate succeeds")
            .expect("frame is recovered");

        assert!(!frame.eyes.is_empty());
        for eye in &frame.eyes {
            assert_eq!(eye.ray.timestamp, obs.timestamp);
            assert_eq!(eye.ray.head_rotation, Some(frame.viewer));
        }
    }
}
