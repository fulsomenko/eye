use eye_core::stage::{GazeEstimator, StageError};
use eye_core::{GazeRay, Measured, Observations, Rig, Side};
use eye_geometry::camera::pixel_ray;
use eye_geometry::eyeball::{EyeParams, visual_axis};
use eye_geometry::uncertainty::propagate_fn;
use nalgebra::{Matrix6, Point2, Point3, Unit, UnitQuaternion, Vector3, Vector6};
use serde::Deserialize;

use crate::EstimateError;
use crate::ir::PupilPair;
use crate::options::parse_options;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PccrOptions {
    pub ipd_mm: f64,
    pub max_glint_offset_px: f64,
    pub apply_kappa: bool,
}

impl Default for PccrOptions {
    fn default() -> Self {
        Self {
            ipd_mm: 63.0,
            max_glint_offset_px: 3.0,
            apply_kappa: true,
        }
    }
}

#[derive(Debug)]
pub struct PccrEstimator {
    options: PccrOptions,
    params: EyeParams,
}

impl PccrEstimator {
    pub const NAME: &'static str = "pccr";

    pub fn new(options: PccrOptions) -> Self {
        Self {
            options,
            params: EyeParams::default(),
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
        let Some(pair) = PupilPair::from_observations(obs) else {
            return Ok(vec![]);
        };
        let cam = rig
            .camera(pair.camera.as_str())
            .ok_or_else(|| EstimateError::UnknownCamera(pair.camera.to_string()))?;
        let face = obs
            .iter()
            .find(|o| o.timestamp == pair.timestamp && o.camera == pair.camera)
            .and_then(|o| o.face.as_ref());
        let Some(face) = face else {
            return Ok(vec![]);
        };

        let mut rays = Vec::new();
        for side in [Side::Right, Side::Left] {
            let pupil = match side {
                Side::Right => &pair.right,
                Side::Left => &pair.left,
            };
            let Some(glint) = self.glint_for(face, side, pupil) else {
                continue;
            };
            let z = Vector6::new(
                pair.right.value().x,
                pair.right.value().y,
                pair.left.value().x,
                pair.left.value().y,
                glint.value().x,
                glint.value().y,
            );
            let sigmas = [
                pair.right.sigma(),
                pair.right.sigma(),
                pair.left.sigma(),
                pair.left.sigma(),
                glint.sigma(),
                glint.sigma(),
            ];
            let cov =
                Matrix6::from_diagonal(&Vector6::from_iterator(sigmas.into_iter().map(|s| s * s)));
            if let Some(ray) = self.ray(side, cam, &z, &cov) {
                rays.push(ray);
            }
        }
        Ok(rays)
    }

    /// The first glint within `max_glint_offset_px` of `pupil`, if any.
    fn glint_for(
        &self,
        face: &eye_core::FaceObservation,
        side: Side,
        pupil: &Measured<Point2<f64>>,
    ) -> Option<Measured<Point2<f64>>> {
        let eye = face.eye(side)?;
        eye.glints
            .iter()
            .find(|g| (*g.value() - pupil.value()).norm() <= self.options.max_glint_offset_px)
            .copied()
    }

    fn solve(
        &self,
        side: Side,
        cam: &eye_core::CameraModel,
        z: &Vector6<f64>,
    ) -> Option<(Point3<f64>, Unit<Vector3<f64>>)> {
        let ray = |i: usize| pixel_ray(cam, &Point2::new(z[i], z[i + 1])).ok();
        let ((o, u_r), (_, u_l), (_, v)) = (ray(0)?, ray(2)?, ray(4)?);
        let s = self.options.ipd_mm / (u_r.into_inner() - u_l.into_inner()).norm();
        let u = if side == Side::Right { u_r } else { u_l };
        let (k, g) = pccr_axis(&o, &u, &v, s, self.params.cornea_to_pupil_mm())?;
        let origin = k - g.into_inner() * self.params.rotation_to_cornea_mm();
        let direction = if self.options.apply_kappa {
            visual_axis(&g, &self.params.kappa, side, &UnitQuaternion::identity())
        } else {
            g
        };
        Some((origin, direction))
    }

    fn ray(
        &self,
        side: Side,
        cam: &eye_core::CameraModel,
        z: &Vector6<f64>,
        cov: &Matrix6<f64>,
    ) -> Option<GazeRay> {
        let (angles, angular_cov) = propagate_fn::<2, 6>(
            |z| {
                self.solve(side, cam, z)
                    .map(|(_, d)| eye_geometry::angles::yaw_pitch_from_direction(&d))
            },
            z,
            cov,
        )?;
        let (origin, origin_cov) =
            propagate_fn::<3, 6>(|z| self.solve(side, cam, z).map(|(o, _)| o.coords), z, cov)?;
        Some(GazeRay {
            side: Some(side),
            origin: Point3::from(origin),
            direction: eye_geometry::angles::direction_from_yaw_pitch(&angles),
            angular_cov,
            origin_cov,
        })
    }
}

impl GazeEstimator for PccrEstimator {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn estimate(&mut self, obs: &[Observations], rig: &Rig) -> Result<Vec<GazeRay>, StageError> {
        Ok(self.estimate_rays(obs, rig)?)
    }
}

/// Core solve: pupil and glint unit rays (reference frame), shared camera centre `o`, pupil
/// range `s`, cornea-centre-to-pupil distance `k_d`. Returns `(cornea centre K, optical axis
/// toward the screen)`.
pub fn pccr_axis(
    o: &Point3<f64>,
    u_pupil: &Unit<Vector3<f64>>,
    v_glint: &Unit<Vector3<f64>>,
    s: f64,
    k_d: f64,
) -> Option<(Point3<f64>, Unit<Vector3<f64>>)> {
    let c = u_pupil.dot(v_glint);
    let disc = k_d * k_d - s * s * (1.0 - c * c);
    if disc < 0.0 {
        return None;
    }
    let t = s * c + disc.sqrt();
    let k = o + v_glint.into_inner() * t;
    let g = Unit::try_new(u_pupil.into_inner() * s - v_glint.into_inner() * t, 1e-12)?;
    Some((k, g))
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use eye_geometry::synth::SplitMix64;
    use nalgebra::{Matrix2, Vector2};

    use super::*;
    use crate::fused::{FusedEstimator, FusedOptions, FusedSource, IrChainKind};
    use crate::testutil::{EYE_CENTRES, synthetic_pccr_observation, test_rig};

    fn angle_deg(a: &Unit<Vector3<f64>>, b: &Unit<Vector3<f64>>) -> f64 {
        a.dot(b).clamp(-1.0, 1.0).acos().to_degrees()
    }

    #[test]
    fn test_pccr_axis_exact_range_recovers_optical_axis_within_0_01_deg() {
        let rig = test_rig();
        let cam = rig.camera("ir").expect("rig has an ir camera");
        let params = EyeParams::default();
        let centre = EYE_CENTRES[0];
        let target = Point3::new(0.0, 0.0, 0.0);
        let g_true = Unit::new_normalize(target - centre);

        let o = Point3::from(cam.screen_from_camera.translation.vector);
        let p = centre + g_true.into_inner() * params.rotation_to_pupil_mm;
        let k_true = centre + g_true.into_inner() * params.rotation_to_cornea_mm();
        let glint = k_true + (o - k_true).normalize() * params.cornea_radius_mm;

        let u = Unit::new_normalize(p - o);
        let v = Unit::new_normalize(glint - o);
        let s = (p - o).norm();

        let (k, g) = pccr_axis(&o, &u, &v, s, params.cornea_to_pupil_mm())
            .expect("exact range has a real root");

        let error_deg = angle_deg(&g, &g_true);
        assert!(error_deg < 0.01, "error {error_deg} deg");
        assert_abs_diff_eq!(k, k_true, epsilon = 1e-6);
    }

    #[test]
    fn test_pccr_axis_no_real_root_returns_none() {
        let o = Point3::origin();
        let u = Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0));
        let s = 500.0;
        let k_d = 4.76;

        let v_far = Unit::new_normalize(direction_from_angle(6.6_f64 / s));
        assert!(pccr_axis(&o, &u, &v_far, s, k_d).is_none());

        let v_near = Unit::new_normalize(direction_from_angle(4.0_f64 / s));
        assert!(pccr_axis(&o, &u, &v_near, s, k_d).is_some());
    }

    fn direction_from_angle(tan_theta: f64) -> Vector3<f64> {
        let theta = tan_theta.atan();
        Vector3::new(theta.sin(), 0.0, theta.cos())
    }

    #[test]
    fn test_pccr_recovers_synthetic_optical_axis_within_0_8_deg() {
        let rig = test_rig();
        let mut estimator = PccrEstimator::new(PccrOptions {
            apply_kappa: false,
            ..Default::default()
        });
        let targets = [
            Point2::new(0.0, 0.0),
            Point2::new(310.0, 0.0),
            Point2::new(0.0, 170.0),
            Point2::new(310.0, 170.0),
            Point2::new(155.0, 85.0),
        ];
        for target in targets {
            let obs = synthetic_pccr_observation(&rig, target, Vector3::zeros(), 0.0, 0.0, 1);
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
    fn test_head_translation_keeps_pccr_error_small() {
        let rig = test_rig();
        let mut still = PccrEstimator::new(PccrOptions {
            apply_kappa: false,
            ..Default::default()
        });
        let mut moved_estimator = PccrEstimator::new(PccrOptions {
            apply_kappa: false,
            ..Default::default()
        });
        let head_offset = Vector3::new(40.0, 0.0, 50.0);
        let targets = [
            Point2::new(0.0, 0.0),
            Point2::new(310.0, 0.0),
            Point2::new(0.0, 170.0),
            Point2::new(310.0, 170.0),
            Point2::new(155.0, 85.0),
        ];
        for target in targets {
            let still_obs = synthetic_pccr_observation(&rig, target, Vector3::zeros(), 0.0, 0.0, 1);
            let still_rays = still
                .estimate_rays(&[still_obs], &rig)
                .expect("estimate succeeds");

            let moved_obs = synthetic_pccr_observation(&rig, target, head_offset, 0.0, 0.0, 1);
            let moved_rays = moved_estimator
                .estimate_rays(&[moved_obs], &rig)
                .expect("estimate succeeds");

            for i in 0..2 {
                let centre_moved = EYE_CENTRES[i] + head_offset;
                let true_still =
                    Unit::new_normalize(Point3::new(target.x, target.y, 0.0) - EYE_CENTRES[i]);
                let true_moved =
                    Unit::new_normalize(Point3::new(target.x, target.y, 0.0) - centre_moved);

                let error_still = angle_deg(&still_rays[i].direction, &true_still);
                let error_moved = angle_deg(&moved_rays[i].direction, &true_moved);
                println!(
                    "target {target:?} eye {i}: still {error_still:.4} deg moved {error_moved:.4} deg"
                );
                assert!(
                    error_moved < 1.0,
                    "target {target:?} eye {i}: moved error {error_moved} deg"
                );
                let delta = (error_moved - error_still).abs();
                assert!(
                    delta < 0.5,
                    "target {target:?} eye {i}: error delta {delta} deg"
                );
            }
        }
    }

    #[test]
    fn test_glint_too_far_from_pupil_drops_eye() {
        let rig = test_rig();
        let cam = rig.camera("ir").expect("rig has an ir camera");
        let mut estimator = PccrEstimator::new(PccrOptions {
            apply_kappa: false,
            ..Default::default()
        });
        let target = Point2::new(155.0, 85.0);
        let base_obs = synthetic_pccr_observation(&rig, target, Vector3::zeros(), 0.0, 0.0, 1);
        let base_rays = estimator
            .estimate_rays(std::slice::from_ref(&base_obs), &rig)
            .expect("estimate succeeds");
        assert_eq!(
            base_rays.len(),
            2,
            "unshifted glints must yield two rays so the shift below is what drops the eye"
        );

        let base_face = base_obs.face.as_ref().expect("face present");
        let right_pupil = base_face
            .eye(Side::Right)
            .and_then(|e| e.pupil)
            .expect("right pupil present")
            .value()
            .center();
        let left_pupil = base_face
            .eye(Side::Left)
            .and_then(|e| e.pupil)
            .expect("left pupil present")
            .value()
            .center();
        let right_glint = *base_face
            .eye(Side::Right)
            .expect("right eye present")
            .glints[0]
            .value();
        let shifted = right_glint + Vector2::new(3.5, 0.0);
        assert!(
            (shifted - right_pupil).norm() > estimator.options.max_glint_offset_px,
            "shift must exceed max_glint_offset_px for the filter to apply"
        );
        let z = Vector6::new(
            right_pupil.x,
            right_pupil.y,
            left_pupil.x,
            left_pupil.y,
            shifted.x,
            shifted.y,
        );
        assert!(
            estimator.solve(Side::Right, cam, &z).is_some(),
            "shifted glint must still have a real pccr root, so only the \
             max_glint_offset_px filter (not geometry) can be dropping the eye"
        );

        let mut obs = base_obs;
        {
            let face = obs.face.as_mut().expect("face present");
            let right = face
                .eyes
                .iter_mut()
                .find(|e| e.side == Side::Right)
                .expect("right eye present");
            right.glints = vec![Measured::new(shifted, 0.0).expect("valid sigma")];
        }

        let rays = estimator
            .estimate_rays(&[obs], &rig)
            .expect("estimate succeeds");
        assert_eq!(rays.len(), 1);
        assert_eq!(rays[0].side, Some(Side::Left));
    }

    #[test]
    fn test_no_glints_returns_empty() {
        let rig = test_rig();
        let mut estimator = PccrEstimator::new(PccrOptions::default());
        let mut obs = synthetic_pccr_observation(
            &rig,
            Point2::new(155.0, 85.0),
            Vector3::zeros(),
            0.0,
            0.0,
            1,
        );
        for eye in obs.face.as_mut().expect("face present").eyes.iter_mut() {
            eye.glints.clear();
        }

        let rays = estimator
            .estimate_rays(&[obs], &rig)
            .expect("estimate succeeds");
        assert!(rays.is_empty());
    }

    #[test]
    fn test_covariance_matches_monte_carlo() {
        let rig = test_rig();
        let cam = rig.camera("ir").expect("rig has an ir camera");
        let estimator = PccrEstimator::new(PccrOptions {
            apply_kappa: false,
            ..Default::default()
        });
        let target = Point2::new(100.0, 50.0);
        let pupil_sigma = 0.2;
        let glint_sigma = 0.3;

        let base_obs = synthetic_pccr_observation(&rig, target, Vector3::zeros(), 0.0, 0.0, 1);
        let base_face = base_obs.face.as_ref().expect("face present");
        let base_right = base_face.eye(Side::Right).expect("right eye present");
        let base_left = base_face.eye(Side::Left).expect("left eye present");
        let base_right_pupil = base_right.pupil.expect("pupil present").value().center();
        let base_left_pupil = base_left.pupil.expect("pupil present").value().center();
        let base_z = Vector6::new(
            base_right_pupil.x,
            base_right_pupil.y,
            base_left_pupil.x,
            base_left_pupil.y,
            base_right.glints[0].value().x,
            base_right.glints[0].value().y,
        );
        let sigmas = [
            pupil_sigma,
            pupil_sigma,
            pupil_sigma,
            pupil_sigma,
            glint_sigma,
            glint_sigma,
        ];
        let cov =
            Matrix6::from_diagonal(&Vector6::from_iterator(sigmas.into_iter().map(|s| s * s)));
        let predicted = estimator
            .ray(Side::Right, cam, &base_z, &cov)
            .expect("ray computed")
            .angular_cov;

        let n = 2000;
        let mut rng = SplitMix64::new(13);
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let noisy_z = Vector6::new(
                base_z[0] + pupil_sigma * rng.gaussian(),
                base_z[1] + pupil_sigma * rng.gaussian(),
                base_z[2] + pupil_sigma * rng.gaussian(),
                base_z[3] + pupil_sigma * rng.gaussian(),
                base_z[4] + glint_sigma * rng.gaussian(),
                base_z[5] + glint_sigma * rng.gaussian(),
            );
            let Some((_, direction)) = estimator.solve(Side::Right, cam, &noisy_z) else {
                continue;
            };
            samples.push(eye_geometry::angles::yaw_pitch_from_direction(&direction));
        }

        let n = samples.len();
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
            let tol = 0.15 * predicted_std;
            println!(
                "k={k}: empirical std {} deg predicted std {} deg",
                empirical_std.to_degrees(),
                predicted_std.to_degrees()
            );
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
    fn test_fused_with_ir_pccr_uses_pccr_chain() {
        let rig = test_rig();
        let mut table = toml::Table::new();
        table.insert("ir".into(), "pccr".into());
        let mut fused = FusedEstimator::from_config(&table, &rig).expect("config parses");

        let obs = synthetic_pccr_observation(
            &rig,
            Point2::new(155.0, 85.0),
            Vector3::zeros(),
            0.2,
            0.3,
            1,
        );

        let selected = fused
            .estimate_detailed(std::slice::from_ref(&obs), &rig)
            .expect("estimate succeeds");

        let mut reference = PccrEstimator::new(PccrOptions::default());
        let expected = reference
            .estimate_rays(&[obs], &rig)
            .expect("estimate succeeds");

        assert_eq!(
            expected.len(),
            2,
            "both eyes must resolve for this check to be meaningful"
        );
        assert_eq!(selected.len(), expected.len());
        for (source, ray) in &selected {
            assert_eq!(*source, FusedSource::IrOnly);
            let matching = expected
                .iter()
                .find(|r| r.side == ray.side)
                .expect("matching ray");
            assert_abs_diff_eq!(ray.origin, matching.origin, epsilon = 1e-12);
            assert_abs_diff_eq!(ray.direction, matching.direction, epsilon = 1e-12);
            assert_abs_diff_eq!(ray.angular_cov, matching.angular_cov, epsilon = 1e-12);
            assert_abs_diff_eq!(ray.origin_cov, matching.origin_cov, epsilon = 1e-12);
        }

        assert_eq!(
            FusedOptions::default().ir,
            IrChainKind::Pupil,
            "default ir chain stays Pupil"
        );
    }

    #[test]
    fn test_from_config_rejects_unknown_option() {
        let rig = test_rig();
        let mut table = toml::Table::new();
        table.insert("bogus".into(), 1.into());

        let err = PccrEstimator::from_config(&table, &rig).expect_err("unknown option errors");
        assert!(matches!(err, StageError::Config(_)));
    }
}
