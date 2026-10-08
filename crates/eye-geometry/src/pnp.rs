//! Perspective-n-Point head/board pose solver with pose covariance.

use eye_core::Measured;
use nalgebra::{
    DVector, Isometry3, Matrix6, Point2, Point3, Translation3, UnitQuaternion, Vector3,
};

use crate::GeometryError;
use crate::camera::Intrinsics;
use crate::lsq::{self, ResidualModel, SolveOptions};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    pub camera_from_object: Isometry3<f64>,
    /// Covariance of [δω (rad, rotation vector applied on the LEFT, camera frame), t (mm)],
    /// linearised at the returned pose: R_true ≈ exp([δω]×) · R_est.
    pub cov: Matrix6<f64>,
    /// Unwhitened RMS reprojection distance per point, px.
    pub rms_px: f64,
}

struct PnpProblem<'a> {
    intr: &'a Intrinsics,
    object: &'a [Point3<f64>],
    image: &'a [Measured<Point2<f64>>],
    r0: UnitQuaternion<f64>,
}

impl PnpProblem<'_> {
    fn pose(&self, x: &DVector<f64>) -> Isometry3<f64> {
        let dw = Vector3::new(x[0], x[1], x[2]);
        Isometry3::from_parts(
            Translation3::new(x[3], x[4], x[5]),
            UnitQuaternion::from_scaled_axis(dw) * self.r0,
        )
    }
}

impl ResidualModel for PnpProblem<'_> {
    fn num_residuals(&self) -> usize {
        2 * self.object.len()
    }

    fn residuals(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let pose = self.pose(x);
        let mut r = DVector::zeros(self.num_residuals());
        for (i, (p, m)) in self.object.iter().zip(self.image).enumerate() {
            let u = self.intr.project(&pose.transform_point(p)).ok()?;
            r[2 * i] = (u.x - m.value().x) / m.sigma();
            r[2 * i + 1] = (u.y - m.value().y) / m.sigma();
        }
        Some(r)
    }
}

/// One LM run from `start`; returns the pose and χ².
fn solve_from(
    intr: &Intrinsics,
    object: &[Point3<f64>],
    image: &[Measured<Point2<f64>>],
    start: &Isometry3<f64>,
) -> Result<(Pose, f64), GeometryError> {
    let problem = PnpProblem {
        intr,
        object,
        image,
        r0: start.rotation,
    };
    let t = start.translation.vector;
    let fit = lsq::solve(
        &problem,
        DVector::from_vec(vec![0.0, 0.0, 0.0, t.x, t.y, t.z]),
        &SolveOptions::default(),
    )?;
    let pose = problem.pose(&fit.x);
    let at_estimate = PnpProblem {
        r0: pose.rotation,
        ..problem
    };
    let t = pose.translation.vector;
    let j = lsq::jacobian(
        &at_estimate,
        &DVector::from_vec(vec![0.0, 0.0, 0.0, t.x, t.y, t.z]),
    )
    .ok_or(GeometryError::Degenerate("jacobian at optimum"))?;
    let cov = lsq::covariance_from_jacobian(&j, fit.chi2, fit.dof)?;
    let mut ss = 0.0;
    for (p, m) in object.iter().zip(image) {
        ss += (intr.project(&pose.transform_point(p))? - m.value()).norm_squared();
    }
    let pose = Pose {
        camera_from_object: pose,
        cov: Matrix6::from_iterator(cov.iter().copied()),
        rms_px: (ss / object.len() as f64).sqrt(),
    };
    Ok((pose, fit.chi2))
}

/// Identity rotation, depth from the ratio of 3D to 2D spread, translation through the 2D centroid.
pub fn frontal_init(
    intr: &Intrinsics,
    object: &[Point3<f64>],
    image: &[Measured<Point2<f64>>],
) -> Result<Isometry3<f64>, GeometryError> {
    let n = object.len() as f64;
    let f_bar = (intr.fx + intr.fy) / 2.0;

    let c3 = object
        .iter()
        .fold(Vector3::zeros(), |acc, p| acc + p.coords)
        / n;
    let mut normalized_sum = nalgebra::Vector2::zeros();
    for m in image {
        let nrm = intr.pixel_to_normalized(m.value())?;
        normalized_sum += nrm.coords;
    }
    let n_bar = normalized_sum / n;

    let u_bar = image
        .iter()
        .fold(nalgebra::Vector2::zeros(), |acc, m| acc + m.value().coords)
        / n;

    let s3_sq = object
        .iter()
        .map(|p| (p.coords.xy() - c3.xy()).norm_squared())
        .sum::<f64>()
        / n;
    let s3 = s3_sq.sqrt();

    let s2_sq = image
        .iter()
        .map(|m| (m.value().coords - u_bar).norm_squared())
        .sum::<f64>()
        / n;
    let s2 = s2_sq.sqrt() / f_bar;

    if s2 <= 0.0 || !s2.is_finite() {
        return Err(GeometryError::Degenerate("frontal_init: zero image spread"));
    }

    let z0 = s3 / s2;
    let t0 = Vector3::new(z0 * n_bar.x, z0 * n_bar.y, z0) - c3;

    Ok(Isometry3::from_parts(
        Translation3::from(t0),
        UnitQuaternion::identity(),
    ))
}

fn validate(object: &[Point3<f64>], image: &[Measured<Point2<f64>>]) -> Result<(), GeometryError> {
    if image.iter().any(|m| m.sigma() <= 0.0) {
        return Err(GeometryError::Degenerate("zero sigma"));
    }
    if object.len() != image.len() || object.len() < 4 {
        return Err(GeometryError::TooFewObservations {
            need: 4,
            got: object.len().min(image.len()),
        });
    }
    Ok(())
}

pub fn solve_pnp(
    intr: &Intrinsics,
    object: &[Point3<f64>],
    image: &[Measured<Point2<f64>>],
    init: Option<&Isometry3<f64>>,
) -> Result<Pose, GeometryError> {
    validate(object, image)?;

    if let Some(init) = init {
        return solve_from(intr, object, image, init).map(|(pose, _)| pose);
    }

    let t0 = frontal_init(intr, object, image)?;
    let frontal_result = solve_from(intr, object, image, &t0);

    let mut median_sigma_sorted: Vec<f64> = image.iter().map(|m| m.sigma()).collect();
    median_sigma_sorted.sort_by(|a, b| a.total_cmp(b));
    let median_sigma = median_sigma_sorted[median_sigma_sorted.len() / 2];

    let needs_alt_starts = match &frontal_result {
        Ok((pose, _)) => pose.rms_px > 3.0 * median_sigma,
        Err(_) => true,
    };

    if !needs_alt_starts {
        return frontal_result.map(|(pose, _)| pose);
    }

    let mut best: Option<(Pose, f64)> = frontal_result.ok();
    for sign in [1.0, -1.0] {
        let dr = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), sign * 30f64.to_radians());
        let start = Isometry3::from_parts(t0.translation, dr);
        if let Ok((pose, chi2)) = solve_from(intr, object, image, &start)
            && best.as_ref().is_none_or(|(_, best_chi2)| chi2 < *best_chi2)
        {
            best = Some((pose, chi2));
        }
    }

    match best {
        Some((pose, _)) => Ok(pose),
        None => solve_from(intr, object, image, &t0).map(|(pose, _)| pose),
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3, Vector6};

    use super::*;
    use crate::camera::Distortion;
    use crate::face_template::MEDIAPIPE_RIGID;
    use crate::synth::SplitMix64;

    fn intr() -> Intrinsics {
        Intrinsics {
            width: 1280,
            height: 720,
            fx: 914.0,
            fy: 914.0,
            cx: 640.0,
            cy: 360.0,
            distortion: Distortion {
                k1: 0.05,
                k2: -0.1,
                ..Default::default()
            },
        }
    }

    fn object() -> Vec<Point3<f64>> {
        MEDIAPIPE_RIGID
            .points
            .iter()
            .map(|(_, p)| Point3::new(p[0], p[1], p[2]))
            .collect()
    }

    fn head_pose(yaw: f64, pitch: f64, roll: f64, t: Vector3<f64>) -> Isometry3<f64> {
        Isometry3::from_parts(
            Translation3::from(t),
            UnitQuaternion::from_euler_angles(pitch, yaw, roll),
        )
    }

    fn image_of(
        object: &[Point3<f64>],
        pose: &Isometry3<f64>,
        sigma: f64,
        mut rng: Option<&mut SplitMix64>,
    ) -> Vec<Measured<Point2<f64>>> {
        let intr = intr();
        object
            .iter()
            .map(|p| {
                let mut u = intr.project(&pose.transform_point(p)).unwrap();
                if let Some(rng) = rng.as_deref_mut() {
                    u.x += sigma * rng.gaussian();
                    u.y += sigma * rng.gaussian();
                }
                Measured::new(u, sigma).unwrap()
            })
            .collect()
    }

    fn image(
        pose: &Isometry3<f64>,
        sigma: f64,
        rng: Option<&mut SplitMix64>,
    ) -> Vec<Measured<Point2<f64>>> {
        image_of(&object(), pose, sigma, rng)
    }

    fn pose_error(est: &Isometry3<f64>, truth: &Isometry3<f64>) -> Vector6<f64> {
        let dr = est.rotation * truth.rotation.inverse();
        let dw = dr.scaled_axis();
        let dt = est.translation.vector - truth.translation.vector;
        Vector6::new(dw.x, dw.y, dw.z, dt.x, dt.y, dt.z)
    }

    #[test]
    fn test_correspondences_then_solve_recovers_pose() {
        let truth = head_pose(
            10f64.to_radians(),
            -5f64.to_radians(),
            0.0,
            Vector3::new(0.0, 20.0, 500.0),
        );
        let intr = intr();
        let mut lm = vec![Point2::new(0.0, 0.0); 478];
        for (i, p) in MEDIAPIPE_RIGID.points {
            let obj = Point3::new(p[0], p[1], p[2]);
            lm[*i] = intr.project(&truth.transform_point(&obj)).unwrap();
        }

        let (object, image) = MEDIAPIPE_RIGID.correspondences(&lm, 1.0).unwrap();
        let pose = solve_pnp(&intr, &object, &image, None).unwrap();

        let err = pose_error(&pose.camera_from_object, &truth);
        assert!(err.fixed_rows::<3>(0).norm() < 1e-6);
        assert!(err.fixed_rows::<3>(3).norm() < 1e-6);
    }

    #[test]
    fn test_pnp_noise_free_frontal_recovers_pose_exactly() {
        let truth = head_pose(0.0, 0.0, 0.0, Vector3::new(0.0, 20.0, 500.0));
        let intr = intr();
        let object = object();
        let image = image(&truth, 1.0, None);

        let pose = solve_pnp(&intr, &object, &image, None).unwrap();
        let err = pose_error(&pose.camera_from_object, &truth);
        assert!(err.fixed_rows::<3>(0).norm() < 1e-6);
        assert!(err.fixed_rows::<3>(3).norm() < 1e-6);
        assert!(pose.rms_px < 1e-6);
    }

    #[test]
    fn test_pnp_converges_from_frontal_init_over_laptop_domain() {
        let intr = intr();
        let object = object();
        for &yaw_deg in &[-40.0f64, -20.0, 0.0, 20.0, 40.0] {
            for &pitch_deg in &[-25.0f64, 0.0, 25.0] {
                for &roll_deg in &[-15.0f64, 15.0] {
                    for &depth in &[300.0f64, 550.0, 800.0] {
                        let truth = head_pose(
                            yaw_deg.to_radians(),
                            pitch_deg.to_radians(),
                            roll_deg.to_radians(),
                            Vector3::new(0.0, 20.0, depth),
                        );
                        let image = image(&truth, 1.0, None);
                        let pose = solve_pnp(&intr, &object, &image, None).unwrap_or_else(|e| {
                            panic!(
                                "yaw={yaw_deg} pitch={pitch_deg} roll={roll_deg} depth={depth}: {e}"
                            )
                        });
                        let err = pose_error(&pose.camera_from_object, &truth);
                        assert!(
                            err.fixed_rows::<3>(0).norm() < 1e-5,
                            "yaw={yaw_deg} pitch={pitch_deg} roll={roll_deg} depth={depth}: rot err {}",
                            err.fixed_rows::<3>(0).norm()
                        );
                        assert!(
                            err.fixed_rows::<3>(3).norm() < 1e-4,
                            "yaw={yaw_deg} pitch={pitch_deg} roll={roll_deg} depth={depth}: trans err {}",
                            err.fixed_rows::<3>(3).norm()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_pnp_noisy_error_within_predicted_covariance() {
        let intr = intr();
        let object = object();
        let truth = head_pose(15f64.to_radians(), 0.0, 0.0, Vector3::new(0.0, 20.0, 550.0));
        let sigma = 1.5;

        let noise_free_image = image(&truth, sigma, None);
        let reference = solve_pnp(&intr, &object, &noise_free_image, None).unwrap();
        let reference_cov = reference.cov;

        let n_trials = 500;
        let mut rng = SplitMix64::new(21);
        let mut sum_outer = Matrix6::zeros();
        let mut nees_sum = 0.0;
        for _ in 0..n_trials {
            let noisy_image = image(&truth, sigma, Some(&mut rng));
            let pose = solve_pnp(&intr, &object, &noisy_image, None).unwrap();
            let err = pose_error(&pose.camera_from_object, &truth);
            sum_outer += err * err.transpose();
            let cov_inv = pose.cov.try_inverse().unwrap();
            nees_sum += (err.transpose() * cov_inv * err)[(0, 0)];
        }
        let empirical = sum_outer / n_trials as f64;
        let mean_nees = nees_sum / n_trials as f64;

        for k in 0..6 {
            let ratio = empirical[(k, k)] / reference_cov[(k, k)];
            assert!(
                (0.8..=1.2).contains(&ratio),
                "diag {k}: ratio {ratio} empirical {} reference {}",
                empirical[(k, k)],
                reference_cov[(k, k)]
            );
        }
        assert!((4.8..=6.6).contains(&mean_nees), "mean NEES = {mean_nees}");
    }

    #[test]
    fn test_pnp_with_tracking_init_uses_it() {
        let intr = intr();
        let object = object();
        let truth = head_pose(15f64.to_radians(), 0.0, 0.0, Vector3::new(0.0, 20.0, 550.0));
        let image = image(&truth, 1.5, None);

        let without_init = solve_pnp(&intr, &object, &image, None).unwrap();

        let init = Isometry3::from_parts(
            Translation3::new(20.0, 0.0, 0.0),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 5f64.to_radians()),
        ) * truth;
        let with_init = solve_pnp(&intr, &object, &image, Some(&init)).unwrap();

        let err = pose_error(
            &with_init.camera_from_object,
            &without_init.camera_from_object,
        );
        assert!(err.norm() < 1e-9, "err norm = {}", err.norm());
    }

    #[test]
    fn test_pnp_too_few_points_errors() {
        let intr = intr();
        let object = object()[..3].to_vec();
        let truth = head_pose(0.0, 0.0, 0.0, Vector3::new(0.0, 20.0, 500.0));
        let image = image(&truth, 1.0, None)[..3].to_vec();

        let err = solve_pnp(&intr, &object, &image, None).unwrap_err();
        assert_eq!(err, GeometryError::TooFewObservations { need: 4, got: 3 });
    }

    #[test]
    fn test_pnp_zero_sigma_is_degenerate() {
        let intr = intr();
        let object = object();
        let truth = head_pose(0.0, 0.0, 0.0, Vector3::new(0.0, 20.0, 500.0));
        let mut image = image(&truth, 1.0, None);
        image[0] = Measured::new(*image[0].value(), 0.0).unwrap();

        let err = solve_pnp(&intr, &object, &image, None).unwrap_err();
        assert_eq!(err, GeometryError::Degenerate("zero sigma"));
    }

    #[test]
    fn test_pnp_planar_board_with_init() {
        let intr = intr();
        let mut object = Vec::new();
        for j in 0..6 {
            for i in 0..9 {
                object.push(Point3::new(
                    25.0 * i as f64 - 100.0,
                    25.0 * j as f64 - 62.5,
                    0.0,
                ));
            }
        }
        let truth = Isometry3::from_parts(
            Translation3::new(0.0, 0.0, 400.0),
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 30f64.to_radians()),
        );
        let mut rng = SplitMix64::new(22);
        let image = image_of(&object, &truth, 0.2, Some(&mut rng));

        let init = Isometry3::from_parts(
            Translation3::new(10.0, 0.0, 0.0),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 3f64.to_radians()),
        ) * truth;

        let pose = solve_pnp(&intr, &object, &image, Some(&init)).unwrap();
        let err = pose_error(&pose.camera_from_object, &truth);
        assert!(
            err.fixed_rows::<3>(0).norm().to_degrees() < 0.1,
            "rot err deg = {}",
            err.fixed_rows::<3>(0).norm().to_degrees()
        );
        assert!(
            err.fixed_rows::<3>(3).norm() < 0.5,
            "trans err mm = {}",
            err.fixed_rows::<3>(3).norm()
        );
    }
}
