use eye_core::log::field;
use eye_core::stage::{GazeEstimator, StageError};
use eye_core::{CameraModel, GazeRay, Measured, Observations, Rig, ScreenModel, Side};
use eye_geometry::camera::pixel_ray;
use eye_geometry::eyeball::{EyeCentre, EyeParams, Kappa, gaze_ray, optical_axis, ray_sphere_near};
use eye_geometry::screen::intersect_plane;
use nalgebra::{Matrix3, Point2, Point3, Unit, Vector3};
use serde::Deserialize;

use crate::EstimateError;
use crate::ir::{PupilPair, binocular_pupils};
use crate::log::{side_str, trace_ray};
use crate::options::parse_options;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct IrPupilOptions {
    pub ipd_mm: f64,
    pub anchor_sigma_mm: f64,
    pub reanchor_distance_mm: f64,
    pub reanchor_margin: f64,
    pub apply_kappa: bool,
}

impl Default for IrPupilOptions {
    fn default() -> Self {
        Self {
            ipd_mm: 63.0,
            anchor_sigma_mm: 0.5,
            reanchor_distance_mm: 5.0,
            reanchor_margin: 0.2,
            apply_kappa: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Trigger {
    Initial,
    Distance,
    OffScreen,
}

/// IR-only baseline. Assumes a still, frontal head between re-anchors: head translation and eye
/// rotation are indistinguishable without corners, head pose or glints.
#[derive(Debug)]
pub struct IrPupilEstimator {
    options: IrPupilOptions,
    params: EyeParams,
    anchor: Option<[Point3<f64>; 2]>,
    last_optical: Option<[Unit<Vector3<f64>>; 2]>,
    pub(crate) last_reanchor: Option<Trigger>,
}

impl IrPupilEstimator {
    pub const NAME: &'static str = "ir-pupil";

    pub fn new(options: IrPupilOptions) -> Self {
        Self {
            options,
            params: EyeParams::default(),
            anchor: None,
            last_optical: None,
            last_reanchor: None,
        }
    }

    pub fn from_config(table: &toml::Table, _rig: &eye_core::Rig) -> Result<Self, StageError> {
        Ok(Self::new(parse_options(Self::NAME, table)?))
    }

    /// Shared with `FusedEstimator`'s IR chain.
    pub(crate) fn estimate_rays(
        &mut self,
        obs: &[Observations],
        rig: &Rig,
    ) -> Result<Vec<GazeRay>, EstimateError> {
        self.last_reanchor = None;
        let Some(pair) = PupilPair::from_observations(obs) else {
            tracing::debug!(
                { field::REASON } = "no_pupil_pair",
                observations = obs.len() as u64,
                "no pupil pair"
            );
            return Ok(vec![]);
        };
        let cam = rig
            .camera(pair.camera.as_str())
            .ok_or_else(|| EstimateError::UnknownCamera(pair.camera.to_string()))?;
        let Some(x) = binocular_pupils(cam, &pair, self.options.ipd_mm) else {
            return Ok(vec![]);
        };
        let r = self.params.rotation_to_pupil_mm;
        let mut anchor = match self.anchor {
            None => {
                tracing::debug!({ field::REASON } = "initial", "re-anchor");
                self.last_reanchor = Some(Trigger::Initial);
                self.anchor_from(&x, rig.screen())
            }
            Some(c) => {
                let drift = ((x[0] - c[0]).norm() - r).abs() + ((x[1] - c[1]).norm() - r).abs();
                let max_drift = 2.0 * self.options.reanchor_distance_mm;
                if drift > max_drift {
                    tracing::debug!(
                        { field::REASON } = "distance",
                        drift_mm = drift,
                        max_drift_mm = max_drift,
                        "re-anchor"
                    );
                    self.last_reanchor = Some(Trigger::Distance);
                    self.anchor_from(&x, rig.screen())
                } else {
                    c
                }
            }
        };

        if let Some(g) = self.optical_pair(cam, &pair, &anchor) {
            let hit = screen_hit(&anchor, &g);
            let margin = self.options.reanchor_margin;
            if !hit.is_some_and(|h| within_grown_screen(&h, margin, rig.screen())) {
                tracing::debug!(
                    { field::REASON } = "off_screen",
                    hit_x_mm = hit.map(|h| h.x),
                    hit_y_mm = hit.map(|h| h.y),
                    margin,
                    "re-anchor"
                );
                self.last_reanchor = Some(Trigger::OffScreen);
                anchor = self.anchor_from(&x, rig.screen());
            }
        }

        self.anchor = Some(anchor);
        if let Some(g) = self.optical_pair(cam, &pair, &anchor) {
            self.last_optical = Some(g);
        }

        let rays: Vec<GazeRay> = self
            .rays(&pair, cam, &anchor)
            .into_iter()
            .flatten()
            .collect();
        for ray in &rays {
            trace_ray(Self::NAME, ray);
        }
        Ok(rays)
    }

    fn anchor_from(&self, x: &[Point3<f64>; 2], screen: &ScreenModel) -> [Point3<f64>; 2] {
        let r = self.params.rotation_to_pupil_mm;
        let t0 = Point3::new(screen.size_mm.x / 2.0, screen.size_mm.y / 2.0, 0.0);
        let dir = |i: usize| match self.last_optical {
            Some(g) => g[i].into_inner(),
            None => (t0 - x[i]).normalize(),
        };
        [x[0] - dir(0) * r, x[1] - dir(1) * r]
    }

    fn optical(
        &self,
        cam: &CameraModel,
        px: &Point2<f64>,
        centre: &Point3<f64>,
    ) -> Option<Unit<Vector3<f64>>> {
        let (o, d) = pixel_ray(cam, px).ok()?;
        optical_axis(
            centre,
            &ray_sphere_near(&o, &d, centre, self.params.rotation_to_pupil_mm),
        )
    }

    fn optical_pair(
        &self,
        cam: &CameraModel,
        pair: &PupilPair,
        anchor: &[Point3<f64>; 2],
    ) -> Option<[Unit<Vector3<f64>>; 2]> {
        Some([
            self.optical(cam, pair.right.value(), &anchor[0])?,
            self.optical(cam, pair.left.value(), &anchor[1])?,
        ])
    }

    fn rays(
        &self,
        pair: &PupilPair,
        cam: &CameraModel,
        anchor: &[Point3<f64>; 2],
    ) -> [Option<GazeRay>; 2] {
        let params = if self.options.apply_kappa {
            self.params
        } else {
            EyeParams {
                kappa: Kappa {
                    alpha_rad: 0.0,
                    beta_rad: 0.0,
                },
                ..self.params
            }
        };
        let a = self.options.anchor_sigma_mm;
        let cov = Matrix3::from_diagonal(&Vector3::new(a * a, a * a, 9.0 * a * a));
        let ray_for = |side: Side, position: Point3<f64>, px: &Measured<Point2<f64>>| {
            gaze_ray(
                side,
                &EyeCentre { position, cov },
                cam,
                px,
                &params,
                None,
                pair.timestamp,
            )
            .inspect_err(|e| {
                tracing::debug!(
                    { field::REASON } = "gaze_ray_failed",
                    side = side_str(side),
                    error = %e,
                    "gaze ray failed"
                );
            })
            .ok()
        };
        [
            ray_for(Side::Right, anchor[0], &pair.right),
            ray_for(Side::Left, anchor[1], &pair.left),
        ]
    }
}

fn screen_hit(anchor: &[Point3<f64>; 2], g: &[Unit<Vector3<f64>>; 2]) -> Option<Point2<f64>> {
    let origin = Point3::from((anchor[0].coords + anchor[1].coords) / 2.0);
    let direction = Unit::try_new(g[0].into_inner() + g[1].into_inner(), 1e-12)?;
    intersect_plane(&origin, &direction)
}

fn within_grown_screen(hit: &Point2<f64>, margin: f64, screen: &ScreenModel) -> bool {
    let (w, h) = (screen.size_mm.x, screen.size_mm.y);
    (-margin * w..=(1.0 + margin) * w).contains(&hit.x)
        && (-margin * h..=(1.0 + margin) * h).contains(&hit.y)
}

impl GazeEstimator for IrPupilEstimator {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn estimate(&mut self, obs: &[Observations], rig: &Rig) -> Result<Vec<GazeRay>, StageError> {
        Ok(self.estimate_rays(obs, rig)?)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use eye_core::observation::{SCHEME_IR_PUPIL_PAIR, SCHEME_MEDIAPIPE_478};
    use eye_core::{CameraId, Ellipse2, EyeObservation, FaceObservation, Measured, Timestamp};
    use eye_geometry::angles::yaw_pitch_from_direction;
    use eye_geometry::camera::Intrinsics;
    use eye_geometry::synth::SplitMix64;
    use eye_log::testing::capture_logs;
    use eye_log::{Level, Value};
    use nalgebra::Vector2;

    use super::*;
    use crate::testutil::{
        EYE_CENTRES, synthetic_ir_observation, synthetic_ir_observation_at, test_rig,
    };

    fn angle_deg(a: &Unit<Vector3<f64>>, b: &Unit<Vector3<f64>>) -> f64 {
        a.dot(b).clamp(-1.0, 1.0).acos().to_degrees()
    }

    #[test]
    fn test_first_frame_at_screen_centre_points_at_centre() {
        let rig = test_rig();
        let mut estimator = IrPupilEstimator::new(IrPupilOptions {
            apply_kappa: false,
            ..Default::default()
        });
        let obs =
            synthetic_ir_observation(&rig, Point2::new(155.0, 85.0), Vector3::zeros(), 0.0, 1);

        let rays = estimator
            .estimate_rays(&[obs], &rig)
            .expect("estimate succeeds");

        assert_eq!(rays.len(), 2);
        assert_eq!(estimator.last_reanchor, Some(Trigger::Initial));
        for ray in &rays {
            let hit = intersect_plane(&ray.origin, &ray.direction).expect("ray hits the screen");
            assert_abs_diff_eq!(hit, Point2::new(155.0, 85.0), epsilon = 1e-6);
        }
    }

    #[test]
    fn test_still_head_recovers_targets_within_0_8_deg() {
        let rig = test_rig();
        let mut estimator = IrPupilEstimator::new(IrPupilOptions {
            apply_kappa: false,
            ..Default::default()
        });
        let centre_obs =
            synthetic_ir_observation(&rig, Point2::new(155.0, 85.0), Vector3::zeros(), 0.0, 1);
        estimator
            .estimate_rays(&[centre_obs], &rig)
            .expect("anchoring frame succeeds");

        let targets = [
            Point2::new(0.0, 0.0),
            Point2::new(310.0, 0.0),
            Point2::new(0.0, 170.0),
            Point2::new(310.0, 170.0),
            Point2::new(155.0, 85.0),
        ];
        for target in targets {
            let obs = synthetic_ir_observation(&rig, target, Vector3::zeros(), 0.0, 2);
            let rays = estimator
                .estimate_rays(&[obs], &rig)
                .expect("estimate succeeds");
            assert_eq!(rays.len(), 2, "target {target:?}: expected two rays");
            for (ray, centre_true) in rays.iter().zip(EYE_CENTRES) {
                let true_direction =
                    Unit::new_normalize(Point3::new(target.x, target.y, 0.0) - centre_true);
                let error_deg = angle_deg(&ray.direction, &true_direction);
                println!("target {target:?}: error {error_deg:.4} deg");
                assert!(
                    error_deg <= 0.8,
                    "target {target:?}: error {error_deg} deg exceeds 0.8 deg"
                );
            }
        }
    }

    #[test]
    fn test_small_lateral_move_triggers_offscreen_reanchor() {
        let rig = test_rig();
        let mut estimator = IrPupilEstimator::new(IrPupilOptions {
            apply_kappa: false,
            ..Default::default()
        });
        let centre_obs =
            synthetic_ir_observation(&rig, Point2::new(155.0, 85.0), Vector3::zeros(), 0.0, 1);
        estimator
            .estimate_rays(&[centre_obs], &rig)
            .expect("anchoring frame succeeds");
        let anchor_before = estimator.anchor.expect("anchor is set");

        let moved = synthetic_ir_observation(
            &rig,
            Point2::new(155.0, 85.0),
            Vector3::new(5.0, 0.0, 0.0),
            0.0,
            2,
        );
        let rays = estimator
            .estimate_rays(&[moved], &rig)
            .expect("estimate succeeds");

        assert_eq!(estimator.last_reanchor, Some(Trigger::OffScreen));
        assert_ne!(estimator.anchor.expect("anchor is set"), anchor_before);
        assert_eq!(rays.len(), 2, "expected two rays after the re-anchor");
        for ray in &rays {
            let hit = intersect_plane(&ray.origin, &ray.direction).expect("ray hits the screen");
            assert!(
                (hit - Point2::new(155.0, 85.0)).norm() < 10.0,
                "hit {hit:?} too far from target"
            );
        }
    }

    #[test]
    fn test_depth_head_move_reanchors_without_jump() {
        let rig = test_rig();
        let mut estimator = IrPupilEstimator::new(IrPupilOptions {
            apply_kappa: false,
            ..Default::default()
        });
        let centre_obs =
            synthetic_ir_observation(&rig, Point2::new(155.0, 85.0), Vector3::zeros(), 0.0, 1);
        estimator
            .estimate_rays(&[centre_obs], &rig)
            .expect("anchoring frame succeeds");

        let mut previous_right: Option<Unit<Vector3<f64>>> = None;
        for k in 1_u64..=10 {
            let offset = Vector3::new(0.0, 0.0, 2.0 * k as f64);
            let obs = synthetic_ir_observation(&rig, Point2::new(155.0, 85.0), offset, 0.0, 10 + k);
            let rays = estimator
                .estimate_rays(&[obs], &rig)
                .expect("estimate succeeds");
            let right = rays[0].direction;

            let true_centre = EYE_CENTRES[0] + offset;
            let true_direction = Unit::new_normalize(Point3::new(155.0, 85.0, 0.0) - true_centre);
            let error_deg = angle_deg(&right, &true_direction);
            println!("k={k}: error {error_deg:.4} deg");

            if k % 3 == 0 {
                assert_eq!(
                    estimator.last_reanchor,
                    Some(Trigger::Distance),
                    "k={k} should trigger a distance re-anchor"
                );
                if let Some(prev) = previous_right {
                    let continuity = angle_deg(&right, &prev);
                    assert!(
                        continuity < 0.05,
                        "k={k}: continuity {continuity} deg exceeds 0.05 deg"
                    );
                }
            } else {
                assert_eq!(estimator.last_reanchor, None, "k={k} should not re-anchor");
            }
            previous_right = Some(right);
        }
    }

    #[test]
    fn test_angular_cov_matches_monte_carlo() {
        let rig = test_rig();
        let cam = rig.camera("ir").expect("rig has an ir camera");
        let estimator = IrPupilEstimator::new(IrPupilOptions {
            anchor_sigma_mm: 0.5,
            apply_kappa: false,
            ..Default::default()
        });

        let target = Point3::new(100.0, 50.0, 0.0);
        let r = EyeParams::default().rotation_to_pupil_mm;
        let intrinsics = Intrinsics::from_camera_model(cam);
        let pixel_for = |centre: Point3<f64>| -> Point2<f64> {
            let direction = (target - centre).normalize();
            let pupil_screen = centre + direction * r;
            let pupil_cam = cam
                .screen_from_camera
                .inverse_transform_point(&pupil_screen);
            intrinsics
                .project(&pupil_cam)
                .expect("synthetic pupil projects in front of the camera")
        };
        let sigma_px = 0.2;
        let right_px = pixel_for(EYE_CENTRES[0]);
        let left_px = pixel_for(EYE_CENTRES[1]);
        let base_pair = PupilPair {
            camera: CameraId::new("ir"),
            timestamp: Timestamp::from_nanos(0),
            right: Measured::new(right_px, sigma_px).expect("valid sigma"),
            left: Measured::new(left_px, sigma_px).expect("valid sigma"),
        };

        let n = 2000;
        let mut rng = SplitMix64::new(11);
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let noisy_right = Point2::new(
                right_px.x + sigma_px * rng.gaussian(),
                right_px.y + sigma_px * rng.gaussian(),
            );
            let noisy_left = Point2::new(
                left_px.x + sigma_px * rng.gaussian(),
                left_px.y + sigma_px * rng.gaussian(),
            );
            let noisy_pair = PupilPair {
                camera: base_pair.camera.clone(),
                timestamp: base_pair.timestamp,
                right: Measured::new(noisy_right, sigma_px).expect("valid sigma"),
                left: Measured::new(noisy_left, sigma_px).expect("valid sigma"),
            };
            let noisy_anchor = [
                EYE_CENTRES[0]
                    + Vector3::new(
                        0.5 * rng.gaussian(),
                        0.5 * rng.gaussian(),
                        1.5 * rng.gaussian(),
                    ),
                EYE_CENTRES[1]
                    + Vector3::new(
                        0.5 * rng.gaussian(),
                        0.5 * rng.gaussian(),
                        1.5 * rng.gaussian(),
                    ),
            ];
            let rays = estimator.rays(&noisy_pair, cam, &noisy_anchor);
            let right_ray = rays[0].clone().expect("right ray computed");
            samples.push(yaw_pitch_from_direction(&right_ray.direction));
        }

        let predicted = estimator.rays(&base_pair, cam, &EYE_CENTRES)[0]
            .clone()
            .expect("right ray computed")
            .angular_cov;

        let mut sum = Vector2::zeros();
        for s in &samples {
            sum += s;
        }
        let mean = sum / n as f64;
        let mut empirical = nalgebra::Matrix2::zeros();
        for s in &samples {
            let d = s - mean;
            empirical += d * d.transpose();
        }
        empirical /= (n - 1) as f64;

        for k in 0..2 {
            let predicted_std = predicted[(k, k)].sqrt();
            let empirical_std = empirical[(k, k)].sqrt();
            let tol = 0.15 * predicted_std;
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
    fn test_origin_cov_is_anchor_cov() {
        let rig = test_rig();
        let cam = rig.camera("ir").expect("rig has an ir camera");
        let estimator = IrPupilEstimator::new(IrPupilOptions {
            anchor_sigma_mm: 0.5,
            apply_kappa: false,
            ..Default::default()
        });
        let obs = synthetic_ir_observation_at(&rig, EYE_CENTRES, Point2::new(100.0, 50.0), 0.0, 1);
        let pair = PupilPair::from_observations(&[obs]).expect("a pair is present");

        let rays = estimator.rays(&pair, cam, &EYE_CENTRES);
        let expected = Matrix3::from_diagonal(&Vector3::new(0.25, 0.25, 2.25));
        for ray in rays.map(|r| r.expect("ray computed")) {
            assert_abs_diff_eq!(ray.origin_cov, expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_no_ir_observation_returns_empty() {
        let rig = test_rig();
        let mut estimator = IrPupilEstimator::new(IrPupilOptions::default());
        let rays = estimator
            .estimate_rays(&[], &rig)
            .expect("estimate succeeds");
        assert!(rays.is_empty());
    }

    #[test]
    fn test_single_pupil_face_returns_empty() {
        let rig = test_rig();
        let mut estimator = IrPupilEstimator::new(IrPupilOptions::default());
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

        let rays = estimator
            .estimate_rays(&[obs], &rig)
            .expect("estimate succeeds");
        assert!(rays.is_empty());
    }

    #[test]
    fn test_rgb_observation_is_ignored() {
        let rig = test_rig();
        let mut estimator = IrPupilEstimator::new(IrPupilOptions::default());
        let obs = Observations {
            camera: CameraId::new("rgb"),
            timestamp: Timestamp::from_nanos(0),
            face: Some(FaceObservation {
                scheme: SCHEME_MEDIAPIPE_478,
                landmarks: Vec::new(),
                eyes: Vec::new(),
            }),
        };

        let rays = estimator
            .estimate_rays(&[obs], &rig)
            .expect("estimate succeeds");
        assert!(rays.is_empty());
    }

    #[test]
    fn test_unknown_camera_is_error() {
        let rig = test_rig();
        let mut estimator = IrPupilEstimator::new(IrPupilOptions::default());
        let mut obs =
            synthetic_ir_observation(&rig, Point2::new(155.0, 85.0), Vector3::zeros(), 0.0, 1);
        obs.camera = CameraId::new("cam9");

        let err = estimator
            .estimate_rays(std::slice::from_ref(&obs), &rig)
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
    fn test_apply_kappa_rotates_nasally() {
        let rig = test_rig();
        let target = Point2::new(155.0, 85.0);
        let obs = synthetic_ir_observation(&rig, target, Vector3::zeros(), 0.0, 1);

        let mut with_kappa = IrPupilEstimator::new(IrPupilOptions {
            apply_kappa: true,
            ..Default::default()
        });
        let mut without_kappa = IrPupilEstimator::new(IrPupilOptions {
            apply_kappa: false,
            ..Default::default()
        });

        let rays_kappa = with_kappa
            .estimate_rays(std::slice::from_ref(&obs), &rig)
            .expect("estimate succeeds");
        let rays_plain = without_kappa
            .estimate_rays(&[obs], &rig)
            .expect("estimate succeeds");

        let angles_kappa: Vec<Vector2<f64>> = rays_kappa
            .iter()
            .map(|r| yaw_pitch_from_direction(&r.direction))
            .collect();
        let angles_plain: Vec<Vector2<f64>> = rays_plain
            .iter()
            .map(|r| yaw_pitch_from_direction(&r.direction))
            .collect();

        assert_abs_diff_eq!(
            angles_kappa[0].x - angles_plain[0].x,
            -5f64.to_radians(),
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            angles_kappa[1].x - angles_plain[1].x,
            5f64.to_radians(),
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            angles_kappa[0].y - angles_plain[0].y,
            1.5f64.to_radians(),
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            angles_kappa[1].y - angles_plain[1].y,
            1.5f64.to_radians(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_from_config_rejects_unknown_option() {
        let rig = test_rig();
        let mut table = toml::Table::new();
        table.insert("bogus".into(), 1.into());

        let err = IrPupilEstimator::from_config(&table, &rig).expect_err("unknown option errors");
        assert!(matches!(err, StageError::Config(_)));
    }

    #[test]
    fn test_from_config_empty_uses_defaults() {
        let rig = test_rig();
        let estimator = IrPupilEstimator::from_config(&toml::Table::new(), &rig)
            .expect("empty config uses defaults");
        assert_abs_diff_eq!(estimator.options.ipd_mm, 63.0, epsilon = 1e-12);
    }

    #[test]
    fn test_boxed_gaze_estimator_emits_two_rays_per_frame() {
        let rig = test_rig();
        let mut estimator: Box<dyn GazeEstimator> =
            Box::new(IrPupilEstimator::from_config(&toml::Table::new(), &rig).expect("defaults"));
        let obs =
            synthetic_ir_observation(&rig, Point2::new(155.0, 85.0), Vector3::zeros(), 0.0, 1);

        let rays = estimator
            .estimate(&[obs], &rig)
            .expect("boxed estimator succeeds");

        assert_eq!(rays.len(), 2);
        assert_eq!(rays[0].side, Some(Side::Right));
        assert_eq!(rays[1].side, Some(Side::Left));
        for ray in &rays {
            ray.validate().expect("ray is valid");
        }
    }

    #[test]
    fn test_logs_gaze_ray_at_trace() {
        let rig = test_rig();
        let mut estimator = IrPupilEstimator::new(IrPupilOptions {
            apply_kappa: false,
            ..Default::default()
        });
        let obs =
            synthetic_ir_observation(&rig, Point2::new(155.0, 85.0), Vector3::zeros(), 0.0, 1);

        let (rays, logs) = capture_logs(tracing::Level::TRACE, || {
            estimator
                .estimate_rays(&[obs], &rig)
                .expect("estimate succeeds")
        });

        assert_eq!(rays.len(), 2);
        let recs: Vec<_> = logs.iter().filter(|r| r.message == "gaze ray").collect();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].level, Level::Trace);
        assert_eq!(recs[0].target, "eye_estimate::log");
        assert_eq!(recs[0].fields["source"], Value::Str("ir-pupil".into()));
        assert_eq!(recs[0].fields["side"], Value::Str("right".into()));
        assert_eq!(recs[1].fields["side"], Value::Str("left".into()));
        assert!(matches!(
            recs[0].fields["origin_z_mm"],
            Value::F64(z) if (z - (-500.0)).abs() < 20.0
        ));
        assert!(matches!(
            recs[0].fields["yaw_sigma_rad"],
            Value::F64(s) if s > 0.0
        ));
        assert!(matches!(
            recs[0].fields["angular_cov_det"],
            Value::F64(d) if d > 0.0
        ));
    }

    #[test]
    fn test_logs_no_pupil_pair_at_debug() {
        let rig = test_rig();
        let mut estimator = IrPupilEstimator::new(IrPupilOptions::default());

        let (rays, logs) = capture_logs(tracing::Level::DEBUG, || {
            estimator
                .estimate_rays(&[], &rig)
                .expect("estimate succeeds")
        });

        assert!(rays.is_empty());
        let rec = logs
            .iter()
            .find(|r| r.message == "no pupil pair")
            .expect("no_pupil_pair logged");
        assert_eq!(rec.target, "eye_estimate::ir_pupil");
        assert_eq!(rec.fields["observations"], Value::U64(0));
        assert_eq!(
            rec.fields[field::REASON],
            Value::Str("no_pupil_pair".into())
        );
        let keys: Vec<&str> = rec.fields.keys().map(String::as_str).collect();
        assert_eq!(keys, vec!["observations", field::REASON]);
    }

    #[test]
    fn test_logs_reanchor_at_debug() {
        let rig = test_rig();
        let mut estimator = IrPupilEstimator::new(IrPupilOptions {
            apply_kappa: false,
            ..Default::default()
        });
        let centre_obs =
            synthetic_ir_observation(&rig, Point2::new(155.0, 85.0), Vector3::zeros(), 0.0, 1);

        let (_, logs_initial) = capture_logs(tracing::Level::DEBUG, || {
            estimator
                .estimate_rays(std::slice::from_ref(&centre_obs), &rig)
                .expect("estimate succeeds")
        });
        let rec_initial = logs_initial
            .iter()
            .find(|r| r.message == "re-anchor")
            .expect("initial re-anchor logged");
        assert_eq!(
            rec_initial.fields[field::REASON],
            Value::Str("initial".into())
        );

        let mut logs_distance = Vec::new();
        for k in 1_u64..=3 {
            let offset = Vector3::new(0.0, 0.0, 2.0 * k as f64);
            let obs = synthetic_ir_observation(&rig, Point2::new(155.0, 85.0), offset, 0.0, 10 + k);
            let (_, logs) = capture_logs(tracing::Level::DEBUG, || {
                estimator
                    .estimate_rays(&[obs], &rig)
                    .expect("estimate succeeds")
            });
            logs_distance = logs;
        }
        let rec_distance = logs_distance
            .iter()
            .find(|r| r.message == "re-anchor")
            .expect("distance re-anchor logged");
        assert_eq!(
            rec_distance.fields[field::REASON],
            Value::Str("distance".into())
        );
        let drift = match rec_distance.fields["drift_mm"] {
            Value::F64(v) => v,
            ref other => panic!("expected F64 drift_mm, got {other:?}"),
        };
        let max_drift = match rec_distance.fields["max_drift_mm"] {
            Value::F64(v) => v,
            ref other => panic!("expected F64 max_drift_mm, got {other:?}"),
        };
        assert_abs_diff_eq!(max_drift, 10.0, epsilon = 1e-12);
        assert!(drift > max_drift);

        let mut estimator2 = IrPupilEstimator::new(IrPupilOptions {
            apply_kappa: false,
            ..Default::default()
        });
        estimator2
            .estimate_rays(&[centre_obs], &rig)
            .expect("anchoring frame succeeds");
        let moved = synthetic_ir_observation(
            &rig,
            Point2::new(155.0, 85.0),
            Vector3::new(5.0, 0.0, 0.0),
            0.0,
            2,
        );
        let (_, logs_offscreen) = capture_logs(tracing::Level::DEBUG, || {
            estimator2
                .estimate_rays(&[moved], &rig)
                .expect("estimate succeeds")
        });
        let rec_offscreen = logs_offscreen
            .iter()
            .find(|r| r.message == "re-anchor")
            .expect("off_screen re-anchor logged");
        assert_eq!(
            rec_offscreen.fields[field::REASON],
            Value::Str("off_screen".into())
        );
        assert!(matches!(rec_offscreen.fields["hit_x_mm"], Value::F64(_)));
    }

    #[test]
    fn test_ir_pupil_rays_have_pair_timestamp_and_no_pose() {
        let rig = test_rig();
        let mut obs =
            synthetic_ir_observation(&rig, Point2::new(155.0, 85.0), Vector3::zeros(), 0.0, 1);
        obs.timestamp = eye_core::Timestamp::from_nanos(68_000_000);
        let mut estimator = IrPupilEstimator::new(IrPupilOptions::default());

        let rays = estimator
            .estimate_rays(std::slice::from_ref(&obs), &rig)
            .expect("estimate succeeds");

        assert_eq!(rays.len(), 2);
        for ray in &rays {
            assert_eq!(ray.timestamp, obs.timestamp);
            assert_eq!(ray.head_rotation, None);
        }
    }
}
