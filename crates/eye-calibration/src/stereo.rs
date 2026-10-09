//! IR-from-RGB stereo extrinsics: the rigid transform between two cameras recovered from
//! boards seen simultaneously by both, refined jointly with all board poses.

use eye_core::log::field;
use eye_core::{CameraId, Measured, Rig};
use eye_geometry::camera::Intrinsics;
use eye_geometry::lsq::{self, ResidualModel, SolveOptions};
use eye_geometry::pnp::solve_pnp;
use nalgebra::{
    DVector, Isometry3, Matrix3, Matrix6, Point2, Point3, Quaternion, Translation3, UnitQuaternion,
    Vector3,
};

use crate::checkerboard::BoardObservation;
use crate::error::CalibrationError;
use crate::intrinsics::{homography_dlt, pose_from_homography, reject_views};

#[derive(Debug, Clone)]
pub struct StereoView {
    pub a: BoardObservation,
    pub b: BoardObservation,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StereoConfig {
    pub corner_sigma_px: f64,
    pub min_views: usize,
    pub view_outlier_factor: f64,
}

impl Default for StereoConfig {
    fn default() -> Self {
        Self {
            corner_sigma_px: 0.1,
            min_views: 5,
            view_outlier_factor: 3.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StereoFit {
    pub b_from_a: Isometry3<f64>,
    /// Covariance of [dw (rad), t (mm)] of `b_from_a` (rotation increment applied on the left, frame b).
    pub cov: Matrix6<f64>,
    /// Per-axis RMS over both cameras and used views: sqrt(sum(dx^2 + dy^2) / (2N)), N = corner observations.
    pub rms_px: f64,
    pub views_used: Vec<usize>,
}

/// Object point plus its pixel observation in each camera.
type ViewCorr = Vec<(Point3<f64>, Point2<f64>, Point2<f64>)>;

fn validate_views(views: &[StereoView], min_views: usize) -> Result<(), CalibrationError> {
    for v in views {
        if v.a.spec != v.b.spec || v.a.corners.len() != v.b.corners.len() {
            return Err(CalibrationError::Param {
                name: "views",
                reason: "a and b observations of a view must share spec and corner count"
                    .to_string(),
            });
        }
    }
    if views.len() < min_views {
        return Err(CalibrationError::InsufficientData {
            what: "views",
            need: min_views,
            got: views.len(),
        });
    }
    Ok(())
}

fn solve_view_pose(
    intr: &Intrinsics,
    obs: &BoardObservation,
    sigma: f64,
) -> Result<Isometry3<f64>, CalibrationError> {
    let obj: Vec<Point2<f64>> = obs
        .corners
        .iter()
        .map(|c| Point2::new(c.object_mm.x, c.object_mm.y))
        .collect();
    let mut undistorted_px = Vec::with_capacity(obs.corners.len());
    for c in &obs.corners {
        let n = intr.pixel_to_normalized(&c.image_px)?;
        undistorted_px.push(Point2::new(
            intr.fx * n.x + intr.cx,
            intr.fy * n.y + intr.cy,
        ));
    }
    let h = homography_dlt(&obj, &undistorted_px)?;
    let k = Matrix3::new(intr.fx, 0.0, intr.cx, 0.0, intr.fy, intr.cy, 0.0, 0.0, 1.0);
    let init = pose_from_homography(&k, &h);

    let object: Vec<Point3<f64>> = obs.corners.iter().map(|c| c.object_mm).collect();
    let image: Vec<Measured<Point2<f64>>> = obs
        .corners
        .iter()
        .map(|c| Measured::new(c.image_px, sigma))
        .collect::<Result<_, _>>()?;

    let pose = solve_pnp(intr, &object, &image, Some(&init))?;
    Ok(pose.camera_from_object)
}

/// Sign-aligned quaternion mean (Markley et al., small-dispersion case): flips each `q` so it
/// is on the same hemisphere as `qs[0]` before summing, since `q` and `-q` are the same rotation.
fn mean_rotation(qs: &[UnitQuaternion<f64>]) -> UnitQuaternion<f64> {
    let q0 = qs[0].into_inner();
    let sum = qs
        .iter()
        .fold(Quaternion::new(0.0, 0.0, 0.0, 0.0), |acc, q| {
            let q = q.into_inner();
            if q.dot(&q0) < 0.0 { acc - q } else { acc + q }
        });
    UnitQuaternion::new_normalize(sum)
}

struct StereoProblem<'a> {
    intr_a: &'a Intrinsics,
    intr_b: &'a Intrinsics,
    corners: &'a [ViewCorr],
    r0_x: UnitQuaternion<f64>,
    r0_v: &'a [UnitQuaternion<f64>],
    sigma: f64,
}

impl StereoProblem<'_> {
    fn b_from_a(&self, x: &DVector<f64>) -> Isometry3<f64> {
        let dw = Vector3::new(x[0], x[1], x[2]);
        let t = Vector3::new(x[3], x[4], x[5]);
        Isometry3::from_parts(
            Translation3::from(t),
            UnitQuaternion::from_scaled_axis(dw) * self.r0_x,
        )
    }

    fn a_from_board(&self, x: &DVector<f64>, v: usize) -> Isometry3<f64> {
        let base = 6 + 6 * v;
        let dw = Vector3::new(x[base], x[base + 1], x[base + 2]);
        let t = Vector3::new(x[base + 3], x[base + 4], x[base + 5]);
        Isometry3::from_parts(
            Translation3::from(t),
            UnitQuaternion::from_scaled_axis(dw) * self.r0_v[v],
        )
    }
}

impl ResidualModel for StereoProblem<'_> {
    fn num_residuals(&self) -> usize {
        4 * self.corners.iter().map(Vec::len).sum::<usize>()
    }

    fn residuals(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let b_from_a = self.b_from_a(x);
        let mut r = DVector::zeros(self.num_residuals());
        let mut idx = 0;
        for (v, pts) in self.corners.iter().enumerate() {
            let a_from_board = self.a_from_board(x, v);
            let b_from_board = b_from_a * a_from_board;
            for (obj, ua, ub) in pts {
                let proj_a = self
                    .intr_a
                    .project(&a_from_board.transform_point(obj))
                    .ok()?;
                let proj_b = self
                    .intr_b
                    .project(&b_from_board.transform_point(obj))
                    .ok()?;
                r[idx] = (proj_a.x - ua.x) / self.sigma;
                r[idx + 1] = (proj_a.y - ua.y) / self.sigma;
                r[idx + 2] = (proj_b.x - ub.x) / self.sigma;
                r[idx + 3] = (proj_b.y - ub.y) / self.sigma;
                idx += 4;
            }
        }
        Some(r)
    }
}

struct JointResult {
    b_from_a: Isometry3<f64>,
    board_poses: Vec<Isometry3<f64>>,
    cov: Matrix6<f64>,
}

struct StereoCams<'a> {
    intr_a: &'a Intrinsics,
    intr_b: &'a Intrinsics,
    sigma: f64,
}

fn solve_joint(
    cams: &StereoCams<'_>,
    corners: &[ViewCorr],
    r0_x: UnitQuaternion<f64>,
    r0_v: &[UnitQuaternion<f64>],
    x0: DVector<f64>,
    opts: &SolveOptions,
) -> Result<(JointResult, DVector<f64>), CalibrationError> {
    let problem = StereoProblem {
        intr_a: cams.intr_a,
        intr_b: cams.intr_b,
        corners,
        r0_x,
        r0_v,
        sigma: cams.sigma,
    };
    let fit = lsq::solve(&problem, x0, opts)?;
    tracing::trace!(
        views = corners.len() as u64,
        params = fit.x.len() as u64,
        residuals = problem.num_residuals() as u64,
        chi2 = fit.chi2,
        dof = fit.dof as u64,
        evaluations = fit.evaluations as u64,
        chi2_reduced = fit.chi2_reduced(),
        "lm solve"
    );
    let b_from_a = problem.b_from_a(&fit.x);
    let board_poses = (0..corners.len())
        .map(|v| problem.a_from_board(&fit.x, v))
        .collect();
    let cov = Matrix6::from_iterator(fit.cov.view((0, 0), (6, 6)).iter().copied());
    Ok((
        JointResult {
            b_from_a,
            board_poses,
            cov,
        },
        fit.x,
    ))
}

fn initial_params(t0: Vector3<f64>, poses_a: &[Isometry3<f64>]) -> DVector<f64> {
    let mut x0 = DVector::zeros(6 + 6 * poses_a.len());
    x0[3] = t0.x;
    x0[4] = t0.y;
    x0[5] = t0.z;
    for (v, pose) in poses_a.iter().enumerate() {
        let base = 6 + 6 * v;
        let t = pose.translation.vector;
        x0[base + 3] = t.x;
        x0[base + 4] = t.y;
        x0[base + 5] = t.z;
    }
    x0
}

fn subset_params(x: &DVector<f64>, kept: &[usize]) -> DVector<f64> {
    let mut x0 = DVector::zeros(6 + 6 * kept.len());
    x0.rows_mut(0, 6).copy_from(&x.rows(0, 6));
    for (new_v, &orig_v) in kept.iter().enumerate() {
        let src = 6 + 6 * orig_v;
        let dst = 6 + 6 * new_v;
        for off in 0..6 {
            x0[dst + off] = x[src + off];
        }
    }
    x0
}

/// Per-axis RMS over both cameras for one view: sqrt(sum(dx^2 + dy^2) / (2N)), N = corner
/// observations (each camera's observation of each corner counts once).
fn view_rms(
    intr_a: &Intrinsics,
    intr_b: &Intrinsics,
    b_from_a: &Isometry3<f64>,
    a_from_board: &Isometry3<f64>,
    pts: &ViewCorr,
) -> f64 {
    let b_from_board = b_from_a * a_from_board;
    let mut sq = 0.0;
    let mut n = 0usize;
    for (obj, ua, ub) in pts {
        if let Ok(pa) = intr_a.project(&a_from_board.transform_point(obj)) {
            sq += (pa.x - ua.x).powi(2) + (pa.y - ua.y).powi(2);
            n += 1;
        }
        if let Ok(pb) = intr_b.project(&b_from_board.transform_point(obj)) {
            sq += (pb.x - ub.x).powi(2) + (pb.y - ub.y).powi(2);
            n += 1;
        }
    }
    (sq / (2.0 * n as f64)).sqrt()
}

pub fn calibrate_stereo(
    intr_a: &Intrinsics,
    intr_b: &Intrinsics,
    views: &[StereoView],
    cfg: &StereoConfig,
) -> Result<StereoFit, CalibrationError> {
    validate_views(views, cfg.min_views)?;

    let corners: Vec<ViewCorr> = views
        .iter()
        .map(|v| {
            v.a.corners
                .iter()
                .zip(&v.b.corners)
                .map(|(ca, cb)| (ca.object_mm, ca.image_px, cb.image_px))
                .collect()
        })
        .collect();

    let mut poses_a = Vec::with_capacity(views.len());
    let mut poses_b = Vec::with_capacity(views.len());
    for v in views {
        poses_a.push(solve_view_pose(intr_a, &v.a, cfg.corner_sigma_px)?);
        poses_b.push(solve_view_pose(intr_b, &v.b, cfg.corner_sigma_px)?);
    }

    let x_v: Vec<Isometry3<f64>> = poses_a
        .iter()
        .zip(&poses_b)
        .map(|(pa, pb)| pb * pa.inverse())
        .collect();
    let mean_t = x_v
        .iter()
        .fold(Vector3::zeros(), |acc, x| acc + x.translation.vector)
        / x_v.len() as f64;
    let qs: Vec<UnitQuaternion<f64>> = x_v.iter().map(|x| x.rotation).collect();
    let r0_x = mean_rotation(&qs);
    tracing::trace!(
        tx_mm = mean_t.x,
        ty_mm = mean_t.y,
        tz_mm = mean_t.z,
        rotation_deg = r0_x.angle().to_degrees(),
        "stereo initial estimate"
    );

    let r0_v: Vec<UnitQuaternion<f64>> = poses_a.iter().map(|p| p.rotation).collect();
    let x0 = initial_params(mean_t, &poses_a);

    let opts = SolveOptions::default();
    let cams = StereoCams {
        intr_a,
        intr_b,
        sigma: cfg.corner_sigma_px,
    };
    let (fit1, x1) = solve_joint(&cams, &corners, r0_x, &r0_v, x0, &opts)?;

    let per_view_rms1: Vec<f64> = corners
        .iter()
        .zip(&fit1.board_poses)
        .map(|(pts, pose)| view_rms(intr_a, intr_b, &fit1.b_from_a, pose, pts))
        .collect();

    let (kept, threshold_px) = reject_views(&per_view_rms1, cfg.view_outlier_factor);
    for (view, &rms_px) in per_view_rms1.iter().enumerate() {
        if !kept.contains(&view) {
            tracing::debug!(
                view = view as u64,
                rms_px,
                threshold_px,
                { field::REASON } = "rms_above_threshold",
                "view rejected"
            );
        }
    }
    if kept.len() < cfg.min_views {
        return Err(CalibrationError::InsufficientData {
            what: "views",
            need: cfg.min_views,
            got: kept.len(),
        });
    }

    let final_fit = if kept.len() == views.len() {
        fit1
    } else {
        let kept_corners: Vec<ViewCorr> = kept.iter().map(|&i| corners[i].clone()).collect();
        let kept_r0_v: Vec<UnitQuaternion<f64>> = kept.iter().map(|&i| r0_v[i]).collect();
        let x0_2 = subset_params(&x1, &kept);
        let (fit2, _) = solve_joint(&cams, &kept_corners, r0_x, &kept_r0_v, x0_2, &opts)?;
        fit2
    };

    let mut sq_sum = 0.0;
    let mut n_total = 0usize;
    for (new_v, &orig_v) in kept.iter().enumerate() {
        let a_from_board = &final_fit.board_poses[new_v];
        let b_from_board = final_fit.b_from_a * a_from_board;
        for (obj, ua, ub) in &corners[orig_v] {
            if let Ok(pa) = intr_a.project(&a_from_board.transform_point(obj)) {
                sq_sum += (pa.x - ua.x).powi(2) + (pa.y - ua.y).powi(2);
                n_total += 1;
            }
            if let Ok(pb) = intr_b.project(&b_from_board.transform_point(obj)) {
                sq_sum += (pb.x - ub.x).powi(2) + (pb.y - ub.y).powi(2);
                n_total += 1;
            }
        }
    }
    let rms_px = (sq_sum / (2.0 * n_total as f64)).sqrt();

    let t = final_fit.b_from_a.translation.vector;
    tracing::info!(
        views = views.len() as u64,
        views_used = kept.len() as u64,
        rms_px,
        tx_mm = t.x,
        ty_mm = t.y,
        tz_mm = t.z,
        rotation_deg = final_fit.b_from_a.rotation.angle().to_degrees(),
        "stereo fitted"
    );

    Ok(StereoFit {
        b_from_a: final_fit.b_from_a,
        cov: final_fit.cov,
        rms_px,
        views_used: kept,
    })
}

/// `screen_from_b = screen_from_a * b_from_a^-1`; the anchor's pose and every intrinsic stay untouched.
pub fn apply_stereo(
    rig: Rig,
    anchor: &CameraId,
    other: &CameraId,
    fit: &StereoFit,
) -> Result<Rig, CalibrationError> {
    let screen_from_a = rig
        .camera(anchor.as_str())
        .ok_or_else(|| CalibrationError::UnknownCamera(anchor.to_string()))?
        .screen_from_camera;
    let (mut cameras, screen) = rig.into_parts();
    let b = cameras
        .iter_mut()
        .find(|c| &c.id == other)
        .ok_or_else(|| CalibrationError::UnknownCamera(other.to_string()))?;
    b.screen_from_camera = screen_from_a * fit.b_from_a.inverse();
    Ok(Rig::new(cameras, screen)?)
}

/// True when both observations have the same spec and every corner moved less than `tol_px`.
pub fn is_static(prev: &BoardObservation, cur: &BoardObservation, tol_px: f64) -> bool {
    prev.spec == cur.spec
        && prev.corners.len() == cur.corners.len()
        && prev
            .corners
            .iter()
            .zip(&cur.corners)
            .all(|(p, c)| (c.image_px - p.image_px).norm() < tol_px)
}

#[cfg(test)]
mod tests {
    use eye_core::{CameraModel, OutputId, ScreenModel};
    use eye_geometry::synth::SplitMix64;
    use nalgebra::Vector2;

    use super::*;
    use crate::checkerboard::BoardCorner;
    use crate::corners::BoardSpec;
    use crate::nominal::nominal_screen_from_camera;
    use crate::testutil::{fixture_ir_intrinsics, fixture_rgb_intrinsics};

    fn board_spec() -> BoardSpec {
        BoardSpec {
            inner_cols: 9,
            inner_rows: 6,
            square_mm: 25.0,
        }
    }

    fn rgb_screen_from_camera() -> Isometry3<f64> {
        nominal_screen_from_camera(&Point3::new(142.5, -7.0, 0.0))
    }

    fn ir_screen_from_camera() -> Isometry3<f64> {
        let tilt = Isometry3::from_parts(
            Translation3::identity(),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.5f64.to_radians()),
        );
        nominal_screen_from_camera(&Point3::new(167.5, -7.0, 0.0)) * tilt
    }

    fn truth_b_from_a() -> Isometry3<f64> {
        ir_screen_from_camera().inverse() * rgb_screen_from_camera()
    }

    fn observation_from_pose(
        intr: &Intrinsics,
        pose: &Isometry3<f64>,
        spec: &BoardSpec,
        mut noise_rng: Option<&mut SplitMix64>,
    ) -> BoardObservation {
        let corners = (0..spec.inner_rows)
            .flat_map(|j| (0..spec.inner_cols).map(move |i| (i, j)))
            .map(|(i, j)| {
                let object_mm = Point3::new(
                    f64::from(i) * spec.square_mm,
                    f64::from(j) * spec.square_mm,
                    0.0,
                );
                let mut px = intr.project(&pose.transform_point(&object_mm)).unwrap();
                if let Some(rng) = noise_rng.as_mut() {
                    px.x += 0.2 * rng.gaussian();
                    px.y += 0.2 * rng.gaussian();
                }
                BoardCorner {
                    grid: (i, j),
                    object_mm,
                    image_px: px,
                }
            })
            .collect();
        BoardObservation {
            spec: *spec,
            corners,
        }
    }

    fn in_bounds(
        intr: &Intrinsics,
        spec: &BoardSpec,
        pose: &Isometry3<f64>,
        margin_px: f64,
    ) -> bool {
        let (w, h) = (f64::from(intr.width), f64::from(intr.height));
        (0..spec.inner_rows).all(|j| {
            (0..spec.inner_cols).all(|i| {
                let object_mm = Point3::new(
                    f64::from(i) * spec.square_mm,
                    f64::from(j) * spec.square_mm,
                    0.0,
                );
                match intr.project(&pose.transform_point(&object_mm)) {
                    Ok(p) => {
                        (margin_px..=w - margin_px).contains(&p.x)
                            && (margin_px..=h - margin_px).contains(&p.y)
                    }
                    Err(_) => false,
                }
            })
        })
    }

    fn accept_both(
        intr_a: &Intrinsics,
        intr_b: &Intrinsics,
        spec: &BoardSpec,
        b_from_a: Isometry3<f64>,
        margin_px: f64,
    ) -> impl Fn(&Isometry3<f64>) -> bool {
        let intr_a = *intr_a;
        let intr_b = *intr_b;
        let spec = *spec;
        move |pose_a: &Isometry3<f64>| {
            let pose_b = b_from_a * pose_a;
            in_bounds(&intr_a, &spec, pose_a, margin_px)
                && in_bounds(&intr_b, &spec, &pose_b, margin_px)
        }
    }

    fn synthetic_stereo_views(
        n: usize,
        seed: u64,
        noise_a_seed: u64,
        noise_b_seed: u64,
    ) -> (Isometry3<f64>, Vec<Isometry3<f64>>, Vec<StereoView>) {
        let intr_a = fixture_rgb_intrinsics();
        let intr_b = fixture_ir_intrinsics();
        let spec = board_spec();
        let truth = truth_b_from_a();
        let mut pose_rng = SplitMix64::new(seed);
        let poses_a = crate::intrinsics::synthetic_board_poses(
            &mut pose_rng,
            n,
            accept_both(&intr_a, &intr_b, &spec, truth, 10.0),
        );
        let mut noise_a = SplitMix64::new(noise_a_seed);
        let mut noise_b = SplitMix64::new(noise_b_seed);
        let views = poses_a
            .iter()
            .map(|pose_a| {
                let pose_b = truth * pose_a;
                StereoView {
                    a: observation_from_pose(&intr_a, pose_a, &spec, Some(&mut noise_a)),
                    b: observation_from_pose(&intr_b, &pose_b, &spec, Some(&mut noise_b)),
                }
            })
            .collect();
        (truth, poses_a, views)
    }

    #[test]
    fn test_stereo_recovers_relative_pose() {
        let (truth, _, views) = synthetic_stereo_views(12, 23, 100, 200);
        let intr_a = fixture_rgb_intrinsics();
        let intr_b = fixture_ir_intrinsics();

        let fit = calibrate_stereo(&intr_a, &intr_b, &views, &StereoConfig::default()).unwrap();

        let t_err = (fit.b_from_a.translation.vector - truth.translation.vector).norm();
        assert!(t_err < 0.5, "t_err={t_err}");
        let rot_err = (fit.b_from_a.rotation.inverse() * truth.rotation)
            .angle()
            .to_degrees();
        assert!(rot_err < 0.1, "rot_err={rot_err}");
        assert!((0.15..=0.25).contains(&fit.rms_px), "rms_px={}", fit.rms_px);
        for k in 0..3 {
            let err = (fit.b_from_a.translation.vector[k] - truth.translation.vector[k]).abs();
            let bound = 3.0 * fit.cov[(3 + k, 3 + k)].sqrt();
            assert!(err < bound, "axis {k}: err={err} bound={bound}");
        }
    }

    #[test]
    fn test_initial_mean_rotation_of_identical_rotations_is_exact() {
        let q = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.4);
        let neg_q = UnitQuaternion::new_unchecked(-q.into_inner());
        let qs = vec![q, q, neg_q, q, neg_q];
        let mean = mean_rotation(&qs);
        let angle = (mean.inverse() * q).angle();
        assert!(angle < 1e-12, "angle={angle}");
    }

    #[test]
    fn test_outlier_view_rejected() {
        let (_, _, mut views) = synthetic_stereo_views(6, 0, 11, 12);
        for c in &mut views[3].b.corners {
            c.image_px.x += 3.0;
        }
        let intr_a = fixture_rgb_intrinsics();
        let intr_b = fixture_ir_intrinsics();

        let fit = calibrate_stereo(&intr_a, &intr_b, &views, &StereoConfig::default()).unwrap();
        assert!(
            !fit.views_used.contains(&3),
            "views_used={:?}",
            fit.views_used
        );
    }

    #[test]
    fn test_logs_stereo_fitted_at_info() {
        let (truth, _, views) = synthetic_stereo_views(12, 23, 100, 200);
        let intr_a = fixture_rgb_intrinsics();
        let intr_b = fixture_ir_intrinsics();

        let (_fit, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            calibrate_stereo(&intr_a, &intr_b, &views, &StereoConfig::default()).unwrap()
        });

        let fitted: Vec<_> = records
            .iter()
            .filter(|r| r.message == "stereo fitted")
            .collect();
        assert_eq!(fitted.len(), 1);
        let rec = fitted[0];
        assert_eq!(rec.level, eye_log::Level::Info);
        assert_eq!(rec.fields["views"], eye_log::Value::U64(12));
        assert_eq!(rec.fields["views_used"], eye_log::Value::U64(12));
        match rec.fields["rms_px"] {
            eye_log::Value::F64(v) => assert!((0.15..=0.25).contains(&v), "rms_px={v}"),
            ref other => panic!("expected F64 rms_px, got {other:?}"),
        }
        for (field_name, truth_val) in [
            ("tx_mm", truth.translation.vector.x),
            ("ty_mm", truth.translation.vector.y),
            ("tz_mm", truth.translation.vector.z),
        ] {
            match rec.fields[field_name] {
                eye_log::Value::F64(v) => assert!((v - truth_val).abs() < 0.5, "{field_name}={v}"),
                ref other => panic!("expected F64 {field_name}, got {other:?}"),
            }
        }

        let initial: Vec<_> = records
            .iter()
            .filter(|r| r.message == "stereo initial estimate")
            .collect();
        assert_eq!(initial.len(), 1);

        let lm_solve: Vec<_> = records
            .iter()
            .filter(|r| r.message == "lm solve" && r.target == "eye_calibration::stereo")
            .collect();
        assert_eq!(lm_solve.len(), 1);
    }

    #[test]
    fn test_logs_stereo_view_rejected_at_debug() {
        let (_, _, mut views) = synthetic_stereo_views(6, 0, 11, 12);
        for c in &mut views[3].b.corners {
            c.image_px.x += 3.0;
        }
        let intr_a = fixture_rgb_intrinsics();
        let intr_b = fixture_ir_intrinsics();

        let (fit, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            calibrate_stereo(&intr_a, &intr_b, &views, &StereoConfig::default()).unwrap()
        });
        assert!(!fit.views_used.contains(&3));

        let rejected: Vec<_> = records
            .iter()
            .filter(|r| r.message == "view rejected" && r.target == "eye_calibration::stereo")
            .collect();
        assert_eq!(rejected.len(), 1);
        let rec = rejected[0];
        assert_eq!(rec.level, eye_log::Level::Debug);
        assert_eq!(rec.fields["view"], eye_log::Value::U64(3));
        assert!(rec.fields.contains_key("threshold_px"));
        assert_eq!(
            rec.fields[field::REASON],
            eye_log::Value::Str("rms_above_threshold".to_string())
        );

        let lm_solve: Vec<_> = records
            .iter()
            .filter(|r| r.message == "lm solve" && r.target == "eye_calibration::stereo")
            .collect();
        assert_eq!(lm_solve.len(), 2);
    }

    #[test]
    fn test_too_few_views_errors() {
        let (_, _, views) = synthetic_stereo_views(3, 41, 13, 14);
        let intr_a = fixture_rgb_intrinsics();
        let intr_b = fixture_ir_intrinsics();

        let err = calibrate_stereo(&intr_a, &intr_b, &views, &StereoConfig::default()).unwrap_err();
        assert!(matches!(
            err,
            CalibrationError::InsufficientData {
                what: "views",
                need: 5,
                got: 3,
            }
        ));
    }

    #[test]
    fn test_mismatched_corner_counts_error() {
        let (truth, poses_a, mut views) = synthetic_stereo_views(5, 51, 15, 16);
        let other_spec = BoardSpec {
            inner_cols: 7,
            inner_rows: 4,
            square_mm: 25.0,
        };
        let intr_b = fixture_ir_intrinsics();
        views[0].b = observation_from_pose(&intr_b, &(truth * poses_a[0]), &other_spec, None);

        let intr_a = fixture_rgb_intrinsics();
        let err = calibrate_stereo(&intr_a, &intr_b, &views, &StereoConfig::default()).unwrap_err();
        assert!(matches!(err, CalibrationError::Param { name: "views", .. }));
    }

    fn edp1_screen() -> ScreenModel {
        ScreenModel {
            output: OutputId::from("eDP-1"),
            size_mm: Vector2::new(310.0, 174.375),
            size_px: (3840, 2160),
            scale: 2.0,
        }
    }

    fn camera_model(
        id: &str,
        intr: &Intrinsics,
        screen_from_camera: Isometry3<f64>,
    ) -> CameraModel {
        CameraModel {
            id: CameraId::from(id),
            width: intr.width,
            height: intr.height,
            fx: intr.fx,
            fy: intr.fy,
            cx: intr.cx,
            cy: intr.cy,
            distortion: intr.distortion.to_opencv(),
            screen_from_camera,
        }
    }

    fn truth_rig() -> Rig {
        let rgb = camera_model("rgb", &fixture_rgb_intrinsics(), rgb_screen_from_camera());
        let ir = camera_model("ir", &fixture_ir_intrinsics(), ir_screen_from_camera());
        Rig::new(vec![rgb, ir], edp1_screen()).unwrap()
    }

    fn project_in(rig: &Rig, cam_id: &str, p: &Point3<f64>) -> Point2<f64> {
        let cam = rig.camera(cam_id).unwrap();
        let intr = Intrinsics::from_camera_model(cam);
        let p_cam = cam.screen_from_camera.inverse_transform_point(p);
        intr.project(&p_cam).unwrap()
    }

    #[test]
    fn test_apply_stereo_moves_only_other_camera() {
        let truth = truth_b_from_a();
        let rgb = camera_model("rgb", &fixture_rgb_intrinsics(), rgb_screen_from_camera());
        let ir_nominal = camera_model(
            "ir",
            &fixture_ir_intrinsics(),
            nominal_screen_from_camera(&Point3::new(155.0, -7.0, 0.0)),
        );
        let rig = Rig::new(vec![rgb.clone(), ir_nominal], edp1_screen()).unwrap();

        let fit = StereoFit {
            b_from_a: truth,
            cov: Matrix6::zeros(),
            rms_px: 0.0,
            views_used: vec![],
        };

        let updated =
            apply_stereo(rig, &CameraId::from("rgb"), &CameraId::from("ir"), &fit).unwrap();

        let rgb_after = updated.camera("rgb").unwrap();
        assert_eq!(*rgb_after, rgb);

        let ir_after = updated.camera("ir").unwrap();
        let ir_truth = ir_screen_from_camera();
        assert!(
            (ir_after.screen_from_camera.translation.vector - ir_truth.translation.vector).norm()
                < 1e-9
        );
        let rot_err = (ir_after.screen_from_camera.rotation.inverse() * ir_truth.rotation).angle();
        assert!(rot_err < 1e-12, "rot_err={rot_err}");

        let p = Point3::new(155.0, 85.0, -500.0);
        let truth_rig = truth_rig();
        for cam in ["rgb", "ir"] {
            let a = project_in(&truth_rig, cam, &p);
            let b = project_in(&updated, cam, &p);
            assert!((a - b).norm() < 1e-6, "{cam}: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn test_apply_stereo_unknown_camera_errors() {
        let rig = truth_rig();
        let fit = StereoFit {
            b_from_a: Isometry3::identity(),
            cov: Matrix6::zeros(),
            rms_px: 0.0,
            views_used: vec![],
        };
        let err =
            apply_stereo(rig, &CameraId::from("depth"), &CameraId::from("ir"), &fit).unwrap_err();
        assert!(matches!(err, CalibrationError::UnknownCamera(id) if id == "depth"));
    }

    #[test]
    fn test_is_static_thresholds() {
        let spec = board_spec();
        let pose = Isometry3::from_parts(
            Translation3::new(-90.0, -50.0, 400.0),
            UnitQuaternion::identity(),
        );
        let prev = observation_from_pose(&fixture_ir_intrinsics(), &pose, &spec, None);

        let mut cur_small = prev.clone();
        for c in &mut cur_small.corners {
            c.image_px.x += 0.3;
        }
        assert!(is_static(&prev, &cur_small, 0.5));

        let mut cur_large = prev.clone();
        for c in &mut cur_large.corners {
            c.image_px.x += 0.8;
        }
        assert!(!is_static(&prev, &cur_large, 0.5));

        let mut cur_other_spec = prev.clone();
        cur_other_spec.spec = BoardSpec {
            inner_cols: 7,
            inner_rows: 4,
            square_mm: 25.0,
        };
        assert!(!is_static(&prev, &cur_other_spec, 0.5));
    }

    #[test]
    fn test_end_to_end_rendered_stereo() {
        use crate::checkerboard::{DetectorConfig, detect_board};
        use crate::testutil::render_board;

        let intr_a = fixture_rgb_intrinsics();
        let intr_b = fixture_ir_intrinsics();
        let spec = board_spec();
        let truth = truth_b_from_a();

        let mut pose_rng = SplitMix64::new(23);
        let poses_a = crate::intrinsics::synthetic_board_poses(
            &mut pose_rng,
            6,
            accept_both(&intr_a, &intr_b, &spec, truth, 70.0),
        );

        let views: Vec<StereoView> = poses_a
            .iter()
            .enumerate()
            .map(|(v, pose_a)| {
                let pose_b = truth * pose_a;
                let (img_a, _) = render_board(&intr_a, pose_a, &spec, 100 + v as u64);
                let (img_b, _) = render_board(&intr_b, &pose_b, &spec, 200 + v as u64);
                let a = detect_board(&img_a.view(), &spec, &DetectorConfig::default())
                    .unwrap()
                    .unwrap_or_else(|| panic!("v={v}: rgb board not found"));
                let b = detect_board(&img_b.view(), &spec, &DetectorConfig::default())
                    .unwrap()
                    .unwrap_or_else(|| panic!("v={v}: ir board not found"));
                StereoView { a, b }
            })
            .collect();

        let cfg = StereoConfig {
            min_views: 5,
            ..StereoConfig::default()
        };
        let fit = calibrate_stereo(&intr_a, &intr_b, &views, &cfg).unwrap();

        let t_err = (fit.b_from_a.translation.vector - truth.translation.vector).norm();
        assert!(t_err < 1.0, "t_err={t_err}");
    }
}
