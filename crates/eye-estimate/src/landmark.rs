//! MediaPipe landmarks to gaze rays: PnP head pose against a rigid 3D face template, then the
//! eyeball model per eye.

use std::collections::HashMap;

use eye_core::log::field;
use eye_core::observation::{LandmarkScheme, mediapipe478};
use eye_core::stage::{GazeEstimator, StageError};
use eye_core::{
    CameraId, CameraModel, FaceObservation, GazeRay, Measured, Observations, RaySource, Rig, Side,
    SourcedRay, Timestamp,
};
use eye_geometry::camera::{Intrinsics, pixel_ray};
use eye_geometry::eyeball::{
    EyeCentre, EyeParams, eyeball_centre_in_head, gaze_ray, screen_from_viewer,
};
use eye_geometry::face_template::MEDIAPIPE_RIGID;
use eye_geometry::pnp::{Pose, solve_pnp};
use eye_geometry::uncertainty::propagate_fn;
use nalgebra::{Isometry3, Point2, Point3, Translation3, UnitQuaternion, Vector3, Vector6};
use serde::Deserialize;

use crate::EstimateError;
use crate::log::{side_str, trace_ray};
use crate::options::parse_options;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct LandmarkOptions {
    pub landmark_sigma_px: f64,
    pub max_reprojection_px: f64,
    pub max_iris_miss_mm: f64,
    pub apply_kappa: bool,
}

impl Default for LandmarkOptions {
    fn default() -> Self {
        Self {
            landmark_sigma_px: 1.5,
            max_reprojection_px: 6.0,
            max_iris_miss_mm: 2.0,
            apply_kappa: true,
        }
    }
}

#[derive(Debug)]
pub struct LandmarkEstimator {
    options: LandmarkOptions,
    params: EyeParams,
    last_pose: HashMap<CameraId, Isometry3<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LandmarkEye {
    pub side: Side,
    pub centre: EyeCentre,
    pub iris_px: Measured<Point2<f64>>,
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
        let eyes: Vec<LandmarkEye> = [Side::Right, Side::Left]
            .into_iter()
            .filter_map(|side| self.eye(side, face, &pose, cam, &viewer, obs.timestamp))
            .collect();
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
        let Some((e, cov)) = propagate_fn::<3, 6>(e_of, &Vector6::zeros(), &pose.cov) else {
            tracing::debug!(
                { field::REASON } = "propagate_failed",
                side = side_str(side),
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
                side = side_str(side),
                "eye dropped"
            );
            return None;
        };
        let Some(iris_px) = eye.iris.map(|m| m.map(|ellipse| ellipse.center())) else {
            tracing::debug!(
                { field::REASON } = "no_iris",
                side = side_str(side),
                "eye dropped"
            );
            return None;
        };

        let (o, u) = pixel_ray(cam, iris_px.value())
            .inspect_err(|err| {
                tracing::debug!(
                    { field::REASON } = "pixel_ray_failed",
                    side = side_str(side),
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
                side = side_str(side),
                miss_mm = miss,
                max_iris_miss_mm = self.options.max_iris_miss_mm,
                "eye dropped"
            );
            return None;
        }

        let params = self.effective_params();
        let ray = gaze_ray(side, &centre, cam, &iris_px, &params, Some(viewer), at)
            .inspect_err(|e| {
                tracing::debug!(
                    { field::REASON } = "gaze_ray_failed",
                    side = side_str(side),
                    error = %e,
                    "eye dropped"
                );
            })
            .ok()?;

        Some(LandmarkEye {
            side,
            centre,
            iris_px,
            ray,
        })
    }
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
        synthetic_eye_centres, synthetic_ir_observation, synthetic_rgb_observation, test_rig,
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
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let options = LandmarkOptions {
            landmark_sigma_px: 1.5,
            apply_kappa: false,
            ..Default::default()
        };

        let mut noise_free =
            synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 0);
        for eye in noise_free
            .face
            .as_mut()
            .expect("face is present")
            .eyes
            .iter_mut()
        {
            eye.iris = eye
                .iris
                .map(|m| Measured::new(m.into_value(), 1.0).expect("valid sigma"));
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
            let obs =
                synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 1.5, 1.0, seed);
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
