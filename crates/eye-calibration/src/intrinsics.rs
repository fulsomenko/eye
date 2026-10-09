//! Zhang's camera calibration: per-view homographies, the closed-form intrinsics solution,
//! pose recovery, and a joint Levenberg-Marquardt refinement over all views at once.

use eye_core::log::field;
use eye_core::{CameraId, Rig};
use eye_geometry::GeometryError;
use eye_geometry::camera::{Distortion, Intrinsics};
use eye_geometry::lsq::{self, ResidualModel, SolveOptions};
use nalgebra::{
    DMatrix, DVector, Isometry3, Matrix3, Point2, Point3, Rotation3, SMatrix, SVector,
    Translation3, UnitQuaternion, Vector2, Vector3,
};

use crate::checkerboard::BoardObservation;
use crate::error::CalibrationError;
use crate::nominal::focal_prior;

/// The two Dell Latitude 7420 webcam lens modules' published diagonal FOV, same as `identify_module`.
const DELL_DIAG_FOV_DEG: [f64; 2] = [75.8, 87.0];

#[cfg(test)]
use eye_geometry::synth::SplitMix64;
#[cfg(test)]
use nalgebra::Unit;

#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IntrinsicsConfig {
    pub fit_k3: bool,
    pub fit_tangential: bool,
    pub corner_sigma_px: f64,
    pub min_views: usize,
    pub view_outlier_factor: f64,
}

impl Default for IntrinsicsConfig {
    fn default() -> Self {
        Self {
            fit_k3: false,
            fit_tangential: true,
            corner_sigma_px: 0.1,
            min_views: 10,
            view_outlier_factor: 3.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LensModule {
    Fov75_8,
    Fov87,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct IntrinsicsFit {
    pub intrinsics: Intrinsics,
    /// Covariance of [fx, fy, cx, cy, k1, k2, p1, p2, k3]; rows/cols of fixed parameters are zero.
    pub cov: SMatrix<f64, 9, 9>,
    /// Per-axis reprojection RMS over used views: sqrt(sum(dx^2 + dy^2) / (2N)), N = corners.
    pub rms_px: f64,
    pub per_view_rms_px: Vec<f64>,
    pub views_used: Vec<usize>,
    pub camera_from_board: Vec<Isometry3<f64>>,
    pub module: LensModule,
    /// (fx - focal_prior.f_px) / focal_prior.sigma_px; |prior_z| > 2 logs a warning.
    pub prior_z: f64,
}

fn normalizer(pts: &[Point2<f64>]) -> Matrix3<f64> {
    let n = pts.len() as f64;
    let centroid = pts.iter().fold(Vector2::zeros(), |acc, p| acc + p.coords) / n;
    let mean_dist = pts
        .iter()
        .map(|p| (p.coords - centroid).norm())
        .sum::<f64>()
        / n;
    let s = std::f64::consts::SQRT_2 / mean_dist;
    Matrix3::new(
        s,
        0.0,
        -s * centroid.x,
        0.0,
        s,
        -s * centroid.y,
        0.0,
        0.0,
        1.0,
    )
}

fn apply_h(h: &Matrix3<f64>, p: &Point2<f64>) -> Point2<f64> {
    let v = h * Vector3::new(p.x, p.y, 1.0);
    Point2::new(v.x / v.z, v.y / v.z)
}

pub fn homography_dlt(
    obj: &[Point2<f64>],
    img: &[Point2<f64>],
) -> Result<Matrix3<f64>, CalibrationError> {
    if obj.len() < 5 || obj.len() != img.len() {
        return Err(CalibrationError::InsufficientData {
            what: "homography points",
            need: 5,
            got: obj.len().min(img.len()),
        });
    }
    let (t_obj, t_img) = (normalizer(obj), normalizer(img));
    let mut a = DMatrix::<f64>::zeros(2 * obj.len(), 9);
    for (k, (p, q)) in obj.iter().zip(img).enumerate() {
        let p = apply_h(&t_obj, p);
        let q = apply_h(&t_img, q);
        let (x, y, u, v) = (p.x, p.y, q.x, q.y);
        a.row_mut(2 * k)
            .copy_from_slice(&[0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]);
        a.row_mut(2 * k + 1)
            .copy_from_slice(&[x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u]);
    }
    let svd = a.svd(false, true);
    let v_t = svd.v_t.ok_or(GeometryError::Degenerate("svd"))?;
    let h = v_t.row(v_t.nrows() - 1);
    let hn = Matrix3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]);
    let t_img_inv = t_img
        .try_inverse()
        .ok_or(GeometryError::Degenerate("normalizer"))?;
    let hm = t_img_inv * hn * t_obj;
    Ok(hm / hm[(2, 2)])
}

fn v_ij(h: &Matrix3<f64>, i: usize, j: usize) -> SVector<f64, 6> {
    let (hi, hj) = (h.column(i), h.column(j));
    SVector::<f64, 6>::from([
        hi[0] * hj[0],
        hi[0] * hj[1] + hi[1] * hj[0],
        hi[1] * hj[1],
        hi[2] * hj[0] + hi[0] * hj[2],
        hi[2] * hj[1] + hi[1] * hj[2],
        hi[2] * hj[2],
    ])
}

pub fn zhang_closed_form(
    homographies: &[Matrix3<f64>],
    width: u32,
    height: u32,
) -> Result<Matrix3<f64>, CalibrationError> {
    let n = homographies.len();
    if n < 3 {
        return Err(CalibrationError::InsufficientData {
            what: "views",
            need: 3,
            got: n,
        });
    }
    let mut a = DMatrix::<f64>::zeros(2 * n + 1, 6);
    for (k, h) in homographies.iter().enumerate() {
        let v01 = v_ij(h, 0, 1);
        let diff = v_ij(h, 0, 0) - v_ij(h, 1, 1);
        a.row_mut(2 * k)
            .copy_from_slice(&[v01[0], v01[1], v01[2], v01[3], v01[4], v01[5]]);
        a.row_mut(2 * k + 1)
            .copy_from_slice(&[diff[0], diff[1], diff[2], diff[3], diff[4], diff[5]]);
    }
    a.row_mut(2 * n)
        .copy_from_slice(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

    let svd = a.svd(false, true);
    let s = &svd.singular_values;
    if s.len() < 5 || s[4] <= 1e-9 * s[0] {
        return Err(GeometryError::Degenerate("views").into());
    }
    let v_t = svd.v_t.ok_or(GeometryError::Degenerate("svd"))?;
    let b = v_t.row(v_t.nrows() - 1);
    let (b11, b12, b22, b13, b23, b33) = (b[0], b[1], b[2], b[3], b[4], b[5]);

    let denom = b11 * b22 - b12 * b12;
    if denom <= 0.0 {
        return Err(GeometryError::Degenerate("views").into());
    }
    let v0 = (b12 * b13 - b11 * b23) / denom;
    let lambda = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11;
    if lambda / b11 <= 0.0 || lambda * b11 <= 0.0 {
        return Err(GeometryError::Degenerate("views").into());
    }

    let fx = (lambda / b11).sqrt();
    let fy = (lambda * b11 / denom).sqrt();
    let cx = -b13 * fx * fx / lambda;
    let cy = v0;
    if ![fx, fy, cx, cy].iter().all(|v| v.is_finite()) {
        return Err(GeometryError::Degenerate("views").into());
    }
    if !(0.0..=f64::from(width)).contains(&cx) || !(0.0..=f64::from(height)).contains(&cy) {
        return Err(GeometryError::Degenerate("views").into());
    }

    Ok(Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0))
}

pub fn pose_from_homography(k: &Matrix3<f64>, h: &Matrix3<f64>) -> Isometry3<f64> {
    let k_inv = k
        .try_inverse()
        .expect("K from zhang_closed_form has positive fx, fy and is invertible");
    let kh0 = k_inv * h.column(0).into_owned();
    let kh1 = k_inv * h.column(1).into_owned();
    let kh2 = k_inv * h.column(2).into_owned();
    let mu = 1.0 / kh0.norm();
    let mut r1 = kh0 * mu;
    let mut r2 = kh1 * mu;
    let r3 = r1.cross(&r2);
    let mut t = kh2 * mu;
    if t.z < 0.0 {
        r1 = -r1;
        r2 = -r2;
        t = -t;
    }
    let r_raw = Matrix3::from_columns(&[r1, r2, r3]);
    let svd = r_raw.svd(true, true);
    let u = svd.u.expect("computed with compute_u = true");
    let v_t = svd.v_t.expect("computed with compute_v = true");
    let d = (u * v_t).determinant().signum();
    let fix = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, d));
    let r = u * fix * v_t;
    let rotation = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r));
    Isometry3::from_parts(Translation3::from(t), rotation)
}

/// Which of the 9 parameters (intrinsics plus Brown-Conrady distortion terms) are free in the
/// fit, and where each lands in the canonical [fx, fy, cx, cy, k1, k2, p1, p2, k3] layout.
#[derive(Debug, Clone, Copy)]
struct Mask {
    fit_tangential: bool,
    fit_k3: bool,
}

impl Mask {
    fn n_intrinsic(&self) -> usize {
        6 + usize::from(self.fit_tangential) * 2 + usize::from(self.fit_k3)
    }

    /// Canonical index (0..9) that each local parameter (in `x` order) maps to.
    fn full_indices(&self) -> Vec<usize> {
        let mut idxs = vec![0, 1, 2, 3, 4, 5];
        if self.fit_tangential {
            idxs.push(6);
            idxs.push(7);
        }
        if self.fit_k3 {
            idxs.push(8);
        }
        idxs
    }

    fn unpack_distortion(&self, x: &[f64]) -> Distortion {
        let k1 = x[4];
        let k2 = x[5];
        let mut i = 6;
        let (p1, p2) = if self.fit_tangential {
            let v = (x[i], x[i + 1]);
            i += 2;
            v
        } else {
            (0.0, 0.0)
        };
        let k3 = if self.fit_k3 { x[i] } else { 0.0 };
        Distortion { k1, k2, p1, p2, k3 }
    }
}

type ViewCorners = Vec<(Point3<f64>, Point2<f64>)>;

struct CalibProblem<'a> {
    width: u32,
    height: u32,
    mask: Mask,
    corners: &'a [ViewCorners],
    r0: &'a [UnitQuaternion<f64>],
    sigma: f64,
}

impl CalibProblem<'_> {
    fn n_intr(&self) -> usize {
        self.mask.n_intrinsic()
    }

    fn intrinsics(&self, x: &DVector<f64>) -> Intrinsics {
        Intrinsics {
            width: self.width,
            height: self.height,
            fx: x[0],
            fy: x[1],
            cx: x[2],
            cy: x[3],
            distortion: self.mask.unpack_distortion(x.as_slice()),
        }
    }

    fn pose(&self, x: &DVector<f64>, v: usize) -> Isometry3<f64> {
        let base = self.n_intr() + 6 * v;
        let dw = Vector3::new(x[base], x[base + 1], x[base + 2]);
        let t = Vector3::new(x[base + 3], x[base + 4], x[base + 5]);
        Isometry3::from_parts(
            Translation3::from(t),
            UnitQuaternion::from_scaled_axis(dw) * self.r0[v],
        )
    }
}

impl ResidualModel for CalibProblem<'_> {
    fn num_residuals(&self) -> usize {
        2 * self.corners.iter().map(Vec::len).sum::<usize>()
    }

    fn residuals(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let intr = self.intrinsics(x);
        let mut r = DVector::zeros(self.num_residuals());
        let mut idx = 0;
        for (v, pts) in self.corners.iter().enumerate() {
            let pose = self.pose(x, v);
            for (obj, img) in pts {
                let u = intr.project(&pose.transform_point(obj)).ok()?;
                r[idx] = (u.x - img.x) / self.sigma;
                r[idx + 1] = (u.y - img.y) / self.sigma;
                idx += 2;
            }
        }
        Some(r)
    }
}

struct SolveResult {
    intrinsics: Intrinsics,
    poses: Vec<Isometry3<f64>>,
    cov_intr: DMatrix<f64>,
}

#[derive(Debug, Clone, Copy)]
struct CalibContext {
    width: u32,
    height: u32,
    mask: Mask,
    sigma: f64,
}

fn solve_joint(
    ctx: CalibContext,
    corners: &[ViewCorners],
    r0: &[UnitQuaternion<f64>],
    x0: DVector<f64>,
    opts: &SolveOptions,
) -> Result<(SolveResult, DVector<f64>), CalibrationError> {
    let problem = CalibProblem {
        width: ctx.width,
        height: ctx.height,
        mask: ctx.mask,
        corners,
        r0,
        sigma: ctx.sigma,
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
    let intrinsics = problem.intrinsics(&fit.x);
    let poses = (0..corners.len())
        .map(|v| problem.pose(&fit.x, v))
        .collect();
    let n_intr = ctx.mask.n_intrinsic();
    let cov_intr = fit.cov.view((0, 0), (n_intr, n_intr)).into_owned();
    Ok((
        SolveResult {
            intrinsics,
            poses,
            cov_intr,
        },
        fit.x,
    ))
}

fn initial_params(n_intr: usize, k: &Matrix3<f64>, poses: &[Isometry3<f64>]) -> DVector<f64> {
    let mut x0 = DVector::zeros(n_intr + 6 * poses.len());
    x0[0] = k[(0, 0)];
    x0[1] = k[(1, 1)];
    x0[2] = k[(0, 2)];
    x0[3] = k[(1, 2)];
    for (v, pose) in poses.iter().enumerate() {
        let base = n_intr + 6 * v;
        let t = pose.translation.vector;
        x0[base + 3] = t.x;
        x0[base + 4] = t.y;
        x0[base + 5] = t.z;
    }
    x0
}

fn subset_params(n_intr: usize, x: &DVector<f64>, kept: &[usize]) -> DVector<f64> {
    let mut x0 = DVector::zeros(n_intr + 6 * kept.len());
    x0.rows_mut(0, n_intr).copy_from(&x.rows(0, n_intr));
    for (new_v, &orig_v) in kept.iter().enumerate() {
        let src = n_intr + 6 * orig_v;
        let dst = n_intr + 6 * new_v;
        for off in 0..6 {
            x0[dst + off] = x[src + off];
        }
    }
    x0
}

/// Per-axis RMS reprojection error for one view: sqrt(sum(dx^2 + dy^2) / (2N)).
fn view_rms(intr: &Intrinsics, pose: &Isometry3<f64>, pts: &ViewCorners) -> f64 {
    let mut sq = 0.0;
    for (obj, img) in pts {
        if let Ok(u) = intr.project(&pose.transform_point(obj)) {
            sq += (u.x - img.x).powi(2) + (u.y - img.y).powi(2);
        }
    }
    (sq / (2.0 * pts.len() as f64)).sqrt()
}

/// Drops views whose RMS exceeds `factor * median`, returns (kept indices, threshold).
pub(crate) fn reject_views(per_view_rms: &[f64], factor: f64) -> (Vec<usize>, f64) {
    let mut sorted = per_view_rms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("RMS values are finite"));
    let mid = sorted.len() / 2;
    let median = if sorted.is_empty() {
        0.0
    } else if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    };
    let threshold = factor * median;
    let kept = per_view_rms
        .iter()
        .enumerate()
        .filter(|&(_, &r)| r <= threshold)
        .map(|(i, _)| i)
        .collect();
    (kept, threshold)
}

pub fn calibrate_intrinsics(
    width: u32,
    height: u32,
    views: &[BoardObservation],
    cfg: &IntrinsicsConfig,
) -> Result<IntrinsicsFit, CalibrationError> {
    if views.len() < cfg.min_views {
        return Err(CalibrationError::InsufficientData {
            what: "views",
            need: cfg.min_views,
            got: views.len(),
        });
    }

    let mask = Mask {
        fit_tangential: cfg.fit_tangential,
        fit_k3: cfg.fit_k3,
    };

    let corners: Vec<ViewCorners> = views
        .iter()
        .map(|v| {
            v.corners
                .iter()
                .map(|c| (c.object_mm, c.image_px))
                .collect()
        })
        .collect();

    let homographies: Vec<Matrix3<f64>> = corners
        .iter()
        .map(|pts| {
            let obj: Vec<Point2<f64>> = pts.iter().map(|(o, _)| Point2::new(o.x, o.y)).collect();
            let img: Vec<Point2<f64>> = pts.iter().map(|(_, i)| *i).collect();
            homography_dlt(&obj, &img)
        })
        .collect::<Result<_, _>>()?;

    let k = zhang_closed_form(&homographies, width, height)?;
    tracing::trace!(
        fx = k[(0, 0)],
        fy = k[(1, 1)],
        cx = k[(0, 2)],
        cy = k[(1, 2)],
        "closed-form intrinsics"
    );
    let init_poses: Vec<Isometry3<f64>> = homographies
        .iter()
        .map(|h| pose_from_homography(&k, h))
        .collect();
    let r0: Vec<UnitQuaternion<f64>> = init_poses.iter().map(|p| p.rotation).collect();

    let n_intr = mask.n_intrinsic();
    let x0 = initial_params(n_intr, &k, &init_poses);

    let ctx = CalibContext {
        width,
        height,
        mask,
        sigma: cfg.corner_sigma_px,
    };
    let opts = SolveOptions::default();
    let (fit1, x1) = solve_joint(ctx, &corners, &r0, x0, &opts)?;

    let per_view_rms1: Vec<f64> = corners
        .iter()
        .zip(&fit1.poses)
        .map(|(pts, pose)| view_rms(&fit1.intrinsics, pose, pts))
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

    let (final_fit, per_view_rms_px) = if kept.len() == views.len() {
        (fit1, per_view_rms1)
    } else {
        let kept_corners: Vec<ViewCorners> = kept.iter().map(|&i| corners[i].clone()).collect();
        let kept_r0: Vec<UnitQuaternion<f64>> = kept.iter().map(|&i| r0[i]).collect();
        let x0_2 = subset_params(n_intr, &x1, &kept);
        let (fit2, _) = solve_joint(ctx, &kept_corners, &kept_r0, x0_2, &opts)?;

        let mut rms_all = vec![0.0; views.len()];
        for (new_v, &orig_v) in kept.iter().enumerate() {
            rms_all[orig_v] = view_rms(&fit2.intrinsics, &fit2.poses[new_v], &corners[orig_v]);
        }
        for (i, slot) in rms_all.iter_mut().enumerate() {
            if !kept.contains(&i) {
                *slot = view_rms(&fit2.intrinsics, &fit1.poses[i], &corners[i]);
            }
        }
        (fit2, rms_all)
    };
    for (view, &rms_px) in per_view_rms_px.iter().enumerate() {
        tracing::trace!(
            view = view as u64,
            rms_px,
            kept = kept.contains(&view),
            "view residual"
        );
    }

    let mut cov = SMatrix::<f64, 9, 9>::zeros();
    let idxs = mask.full_indices();
    for (li, &fi) in idxs.iter().enumerate() {
        for (lj, &fj) in idxs.iter().enumerate() {
            cov[(fi, fj)] = final_fit.cov_intr[(li, lj)];
        }
    }

    let mut sq_sum = 0.0;
    let mut n_total = 0usize;
    for (new_v, &orig_v) in kept.iter().enumerate() {
        let pose = &final_fit.poses[new_v];
        for (obj, img) in &corners[orig_v] {
            if let Ok(u) = final_fit.intrinsics.project(&pose.transform_point(obj)) {
                sq_sum += (u.x - img.x).powi(2) + (u.y - img.y).powi(2);
                n_total += 1;
            }
        }
    }
    let rms_px = (sq_sum / (2.0 * n_total as f64)).sqrt();

    let module = identify_module(&final_fit.intrinsics);
    let prior_z = warn_if_outside_priors(&final_fit.intrinsics, width, height)?;

    tracing::info!(
        width = u64::from(width),
        height = u64::from(height),
        views = views.len() as u64,
        views_used = kept.len() as u64,
        rms_px,
        fx = final_fit.intrinsics.fx,
        fy = final_fit.intrinsics.fy,
        cx = final_fit.intrinsics.cx,
        cy = final_fit.intrinsics.cy,
        k1 = final_fit.intrinsics.distortion.k1,
        k2 = final_fit.intrinsics.distortion.k2,
        p1 = final_fit.intrinsics.distortion.p1,
        p2 = final_fit.intrinsics.distortion.p2,
        k3 = final_fit.intrinsics.distortion.k3,
        module = ?module,
        prior_z,
        "intrinsics fitted"
    );

    Ok(IntrinsicsFit {
        intrinsics: final_fit.intrinsics,
        cov,
        rms_px,
        per_view_rms_px,
        views_used: kept,
        camera_from_board: final_fit.poses,
        module,
        prior_z,
    })
}

fn warn_if_outside_priors(
    intr: &Intrinsics,
    width: u32,
    height: u32,
) -> Result<f64, CalibrationError> {
    let prior = focal_prior(width, height, &DELL_DIAG_FOV_DEG)?;
    let prior_z = (intr.fx - prior.f_px) / prior.sigma_px;
    if prior_z.abs() > 2.0 {
        tracing::warn!(
            prior_z,
            fx = intr.fx,
            prior_f_px = prior.f_px,
            prior_sigma_px = prior.sigma_px,
            "fitted focal length is outside both Dell lens modules' priors"
        );
    }
    Ok(prior_z)
}

/// Nearest Dell module: the module whose focal is within 5 % of `(fx + fy) / 2`; the two
/// modules differ by 20 %, so at most one matches; else `Unknown`.
pub fn identify_module(intr: &Intrinsics) -> LensModule {
    let f_mean = (intr.fx + intr.fy) / 2.0;
    for (module, deg) in [(LensModule::Fov75_8, 75.8_f64), (LensModule::Fov87, 87.0)] {
        let f = Intrinsics::from_diagonal_fov(intr.width, intr.height, deg.to_radians()).fx;
        if (f_mean - f).abs() <= 0.05 * f {
            return module;
        }
    }
    LensModule::Unknown
}

/// Writes the fit into camera `camera` of `rig` (other cameras and all extrinsics untouched).
pub fn apply_intrinsics(
    rig: Rig,
    camera: &CameraId,
    fit: &IntrinsicsFit,
) -> Result<Rig, CalibrationError> {
    let (mut cameras, screen) = rig.into_parts();
    let m = cameras
        .iter_mut()
        .find(|c| &c.id == camera)
        .ok_or_else(|| CalibrationError::UnknownCamera(camera.to_string()))?;
    if (m.width, m.height) != (fit.intrinsics.width, fit.intrinsics.height) {
        return Err(CalibrationError::Param {
            name: "camera",
            reason: format!(
                "{camera} streams {}x{}, the fit is for {}x{}",
                m.width, m.height, fit.intrinsics.width, fit.intrinsics.height
            ),
        });
    }
    fit.intrinsics.apply_to(m);
    Ok(Rig::new(cameras, screen)?)
}

/// Draws poses until `n` are accepted. Each draw, in this order of `uniform()` calls:
/// `d = 250 + 250u`; tilt axis `(cos a, sin a, 0)` with `a = 2 pi u`; tilt angle `40 deg * u`;
/// in-plane roll `2 pi u`; board centre at camera `(0.25 d (2u - 1), 0.15 d (2u - 1), d)`.
/// Rotation = tilt * roll (roll about z); translation places the 9x6 x 25 mm board centre
/// (object (100, 62.5, 0)) at the drawn centre.
#[cfg(test)]
pub(crate) fn synthetic_board_poses(
    rng: &mut SplitMix64,
    n: usize,
    accept: impl Fn(&Isometry3<f64>) -> bool,
) -> Vec<Isometry3<f64>> {
    let board_centre_obj = Vector3::new(100.0, 62.5, 0.0);
    let mut poses = Vec::with_capacity(n);
    while poses.len() < n {
        let d = 250.0 + 250.0 * rng.uniform();
        let a = std::f64::consts::TAU * rng.uniform();
        let axis = Vector3::new(a.cos(), a.sin(), 0.0);
        let tilt_deg = 40.0 * rng.uniform();
        let roll = std::f64::consts::TAU * rng.uniform();
        let cx = 0.25 * d * (2.0 * rng.uniform() - 1.0);
        let cy = 0.15 * d * (2.0 * rng.uniform() - 1.0);
        let centre = Vector3::new(cx, cy, d);

        let tilt =
            UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis), tilt_deg.to_radians());
        let roll_q = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), roll);
        let rotation = tilt * roll_q;
        let translation = centre - rotation * board_centre_obj;
        let pose = Isometry3::from_parts(Translation3::from(translation), rotation);
        if accept(&pose) {
            poses.push(pose);
        }
    }
    poses
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use eye_core::{CameraId, CameraModel, OutputId, Rig, ScreenModel};
    use nalgebra::{Translation3, Vector3};

    use super::*;
    use crate::checkerboard::BoardCorner;
    use crate::corners::BoardSpec;

    fn board_spec() -> BoardSpec {
        BoardSpec {
            inner_cols: 9,
            inner_rows: 6,
            square_mm: 25.0,
        }
    }

    fn truth_intrinsics() -> Intrinsics {
        Intrinsics {
            width: 640,
            height: 360,
            fx: 457.0,
            fy: 457.0,
            cx: 320.0,
            cy: 180.0,
            distortion: Distortion {
                k1: 0.08,
                k2: -0.15,
                p1: 0.001,
                p2: -0.0005,
                k3: 0.0,
            },
        }
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

    fn accept_within_margin(
        intr: &Intrinsics,
        spec: &BoardSpec,
        margin_px: f64,
    ) -> impl Fn(&Isometry3<f64>) -> bool {
        let intr = *intr;
        let spec = *spec;
        let (w, h) = (f64::from(intr.width), f64::from(intr.height));
        move |pose: &Isometry3<f64>| {
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
    }

    fn accept_in_bounds(intr: &Intrinsics, spec: &BoardSpec) -> impl Fn(&Isometry3<f64>) -> bool {
        accept_within_margin(intr, spec, 10.0)
    }

    fn synthetic_views(n: usize) -> Vec<BoardObservation> {
        let intr = truth_intrinsics();
        let spec = board_spec();
        let mut pose_rng = SplitMix64::new(17);
        let poses = synthetic_board_poses(&mut pose_rng, n, accept_in_bounds(&intr, &spec));
        let mut noise_rng = SplitMix64::new(18);
        poses
            .iter()
            .map(|pose| observation_from_pose(&intr, pose, &spec, Some(&mut noise_rng)))
            .collect()
    }

    #[test]
    fn test_homography_exact_on_pinhole_data() {
        let spec = board_spec();
        let intr = Intrinsics {
            distortion: Distortion::default(),
            ..truth_intrinsics()
        };
        let pose = Isometry3::from_parts(
            Translation3::new(-90.0, -50.0, 400.0),
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.2),
        );
        let obj: Vec<Point2<f64>> = (0..spec.inner_rows)
            .flat_map(|j| (0..spec.inner_cols).map(move |i| (i, j)))
            .map(|(i, j)| Point2::new(f64::from(i) * spec.square_mm, f64::from(j) * spec.square_mm))
            .collect();
        let img: Vec<Point2<f64>> = obj
            .iter()
            .map(|p| {
                intr.project(&pose.transform_point(&Point3::new(p.x, p.y, 0.0)))
                    .unwrap()
            })
            .collect();

        let h = homography_dlt(&obj, &img).unwrap();
        for (p, truth) in obj.iter().zip(&img) {
            let mapped = apply_h(&h, p);
            assert_abs_diff_eq!(mapped.x, truth.x, epsilon = 1e-9);
            assert_abs_diff_eq!(mapped.y, truth.y, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_homography_too_few_points_errors() {
        let pts = [
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
        ];
        let err = homography_dlt(&pts, &pts).unwrap_err();
        assert!(matches!(
            err,
            CalibrationError::InsufficientData {
                need: 5,
                got: 4,
                ..
            }
        ));
    }

    fn pinhole_homographies(n: usize) -> (Intrinsics, Vec<Matrix3<f64>>) {
        let spec = board_spec();
        let intr = Intrinsics {
            distortion: Distortion::default(),
            ..truth_intrinsics()
        };
        let mut rng = SplitMix64::new(42);
        let poses = synthetic_board_poses(&mut rng, n, accept_in_bounds(&intr, &spec));
        let hs = poses
            .iter()
            .map(|pose| {
                let obj: Vec<Point2<f64>> = (0..spec.inner_rows)
                    .flat_map(|j| (0..spec.inner_cols).map(move |i| (i, j)))
                    .map(|(i, j)| {
                        Point2::new(f64::from(i) * spec.square_mm, f64::from(j) * spec.square_mm)
                    })
                    .collect();
                let img: Vec<Point2<f64>> = obj
                    .iter()
                    .map(|p| {
                        intr.project(&pose.transform_point(&Point3::new(p.x, p.y, 0.0)))
                            .unwrap()
                    })
                    .collect();
                homography_dlt(&obj, &img).unwrap()
            })
            .collect();
        (intr, hs)
    }

    #[test]
    fn test_closed_form_recovers_k_noise_free_pinhole() {
        let (intr, hs) = pinhole_homographies(5);
        let k = zhang_closed_form(&hs, intr.width, intr.height).unwrap();
        assert_relative_eq!(k[(0, 0)], intr.fx, max_relative = 1e-6);
        assert_relative_eq!(k[(1, 1)], intr.fy, max_relative = 1e-6);
        assert_relative_eq!(k[(0, 2)], intr.cx, max_relative = 1e-6);
        assert_relative_eq!(k[(1, 2)], intr.cy, max_relative = 1e-6);
    }

    #[test]
    fn test_closed_form_two_views_errors() {
        let (intr, hs) = pinhole_homographies(2);
        let err = zhang_closed_form(&hs, intr.width, intr.height).unwrap_err();
        assert!(matches!(
            err,
            CalibrationError::InsufficientData {
                what: "views",
                need: 3,
                got: 2,
            }
        ));
    }

    #[test]
    fn test_closed_form_parallel_boards_degenerate() {
        let spec = board_spec();
        let intr = Intrinsics {
            distortion: Distortion::default(),
            ..truth_intrinsics()
        };

        let fronto_parallel: Vec<Isometry3<f64>> = (0..4)
            .map(|k| {
                let k = f64::from(k);
                Isometry3::from_parts(
                    Translation3::new(-100.0 + 20.0 * k, -50.0 + 5.0 * k, 300.0 + 30.0 * k),
                    UnitQuaternion::identity(),
                )
            })
            .collect();

        let shared_tilt: Vec<Isometry3<f64>> = (0..4)
            .map(|k| {
                let k = f64::from(k);
                Isometry3::from_parts(
                    Translation3::new(-100.0 + 20.0 * k, -50.0 + 5.0 * k, 300.0 + 30.0 * k),
                    UnitQuaternion::from_scaled_axis(Vector3::new(0.3, 0.2, 0.0)),
                )
            })
            .collect();

        for poses in [fronto_parallel, shared_tilt] {
            let hs: Vec<Matrix3<f64>> = poses
                .iter()
                .map(|pose| {
                    let obj: Vec<Point2<f64>> = (0..spec.inner_rows)
                        .flat_map(|j| (0..spec.inner_cols).map(move |i| (i, j)))
                        .map(|(i, j)| {
                            Point2::new(
                                f64::from(i) * spec.square_mm,
                                f64::from(j) * spec.square_mm,
                            )
                        })
                        .collect();
                    let img: Vec<Point2<f64>> = obj
                        .iter()
                        .map(|p| {
                            intr.project(&pose.transform_point(&Point3::new(p.x, p.y, 0.0)))
                                .unwrap()
                        })
                        .collect();
                    homography_dlt(&obj, &img).unwrap()
                })
                .collect();
            let result = zhang_closed_form(&hs, intr.width, intr.height);
            assert!(
                matches!(
                    result,
                    Err(CalibrationError::Geometry(GeometryError::Degenerate(_)))
                ),
                "expected degenerate, got {result:?}"
            );
        }
    }

    #[test]
    fn test_pose_from_homography_recovers_pose() {
        let (intr, hs) = pinhole_homographies(1);
        let spec = board_spec();
        let mut rng = SplitMix64::new(42);
        let truth_pose = synthetic_board_poses(&mut rng, 1, accept_in_bounds(&intr, &spec))[0];

        let k = Matrix3::new(intr.fx, 0.0, intr.cx, 0.0, intr.fy, intr.cy, 0.0, 0.0, 1.0);
        let recovered = pose_from_homography(&k, &hs[0]);

        let rot_err = (recovered.rotation.inverse() * truth_pose.rotation)
            .angle()
            .abs();
        assert!(rot_err < 1e-9, "rot_err={rot_err}");
        let t_err = (recovered.translation.vector - truth_pose.translation.vector).norm();
        assert!(t_err < 1e-6, "t_err={t_err}");
    }

    #[test]
    fn test_full_calibration_recovers_ir_fixture() {
        let views = synthetic_views(15);
        let fit = calibrate_intrinsics(640, 360, &views, &IntrinsicsConfig::default()).unwrap();

        let truth = truth_intrinsics();
        assert_relative_eq!(fit.intrinsics.fx, truth.fx, max_relative = 0.005);
        assert_relative_eq!(fit.intrinsics.fy, truth.fy, max_relative = 0.005);
        assert!((fit.intrinsics.cx - truth.cx).abs() < 1.5);
        assert!((fit.intrinsics.cy - truth.cy).abs() < 1.5);
        assert!((fit.intrinsics.distortion.k1 - truth.distortion.k1).abs() < 0.02);
        assert!((fit.intrinsics.distortion.k2 - truth.distortion.k2).abs() < 0.05);
        assert!((0.15..=0.25).contains(&fit.rms_px), "rms_px={}", fit.rms_px);
        assert!(
            (fit.intrinsics.fx - 457.0).abs() < 3.0 * fit.cov[(0, 0)].sqrt(),
            "fx={} cov00={}",
            fit.intrinsics.fx,
            fit.cov[(0, 0)]
        );
        assert_eq!(fit.views_used.len(), 15);
    }

    #[test]
    fn test_outlier_view_is_rejected() {
        let mut views = synthetic_views(15);
        let intr = truth_intrinsics();
        let mut pose_rng = SplitMix64::new(17);
        let poses =
            synthetic_board_poses(&mut pose_rng, 15, accept_in_bounds(&intr, &board_spec()));
        views[7] = observation_from_pose(&intr, &poses[7], &board_spec(), None);
        let mut noise_rng = SplitMix64::new(99);
        for corner in &mut views[7].corners {
            corner.image_px.x += 2.0 * noise_rng.gaussian();
            corner.image_px.y += 2.0 * noise_rng.gaussian();
        }

        let fit = calibrate_intrinsics(640, 360, &views, &IntrinsicsConfig::default()).unwrap();
        assert!(
            !fit.views_used.contains(&7),
            "views_used={:?}",
            fit.views_used
        );
        assert_relative_eq!(fit.intrinsics.fx, intr.fx, max_relative = 0.005);
    }

    #[test]
    fn test_too_few_views_errors() {
        let views = synthetic_views(6);
        let cfg = IntrinsicsConfig {
            min_views: 10,
            ..Default::default()
        };
        let err = calibrate_intrinsics(640, 360, &views, &cfg).unwrap_err();
        assert!(matches!(
            err,
            CalibrationError::InsufficientData {
                what: "views",
                need: 10,
                got: 6,
            }
        ));
    }

    #[test]
    fn test_k3_fixed_by_default() {
        let views = synthetic_views(15);
        let fit = calibrate_intrinsics(640, 360, &views, &IntrinsicsConfig::default()).unwrap();
        assert_abs_diff_eq!(fit.intrinsics.distortion.k3, 0.0, epsilon = 0.0);
        for i in 0..9 {
            assert_eq!(fit.cov[(8, i)], 0.0);
            assert_eq!(fit.cov[(i, 8)], 0.0);
        }
    }

    #[test]
    fn test_identify_module_75_8() {
        let mk = |fx: f64| Intrinsics {
            width: 640,
            height: 360,
            fx,
            fy: fx,
            cx: 320.0,
            cy: 180.0,
            distortion: Distortion::default(),
        };
        assert_eq!(identify_module(&mk(471.63)), LensModule::Fov75_8);
        assert_eq!(identify_module(&mk(457.0)), LensModule::Fov75_8);

        let rgb = Intrinsics {
            width: 1280,
            height: 720,
            fx: 943.25,
            fy: 943.25,
            cx: 640.0,
            cy: 360.0,
            distortion: Distortion::default(),
        };
        assert_eq!(identify_module(&rgb), LensModule::Fov75_8);
    }

    #[test]
    fn test_identify_module_87() {
        let intr = Intrinsics {
            width: 640,
            height: 360,
            fx: 386.90,
            fy: 386.90,
            cx: 320.0,
            cy: 180.0,
            distortion: Distortion::default(),
        };
        assert_eq!(identify_module(&intr), LensModule::Fov87);
    }

    #[test]
    fn test_identify_module_ambiguous_unknown() {
        let intr = Intrinsics {
            width: 640,
            height: 360,
            fx: 429.26,
            fy: 429.26,
            cx: 320.0,
            cy: 180.0,
            distortion: Distortion::default(),
        };
        assert_eq!(identify_module(&intr), LensModule::Unknown);
    }

    #[test]
    fn test_prior_z_of_fixture() {
        let views = synthetic_views(15);
        let fit = calibrate_intrinsics(640, 360, &views, &IntrinsicsConfig::default()).unwrap();
        assert!(
            (fit.prior_z - 0.655).abs() < 0.05,
            "prior_z={}",
            fit.prior_z
        );
    }

    #[test]
    fn test_logs_intrinsics_fitted_at_info() {
        let views = synthetic_views(15);

        let (_fit, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            calibrate_intrinsics(640, 360, &views, &IntrinsicsConfig::default()).unwrap()
        });

        let fitted: Vec<_> = records
            .iter()
            .filter(|r| r.message == "intrinsics fitted")
            .collect();
        assert_eq!(fitted.len(), 1);
        let rec = fitted[0];
        assert_eq!(rec.level, eye_log::Level::Info);
        assert_eq!(rec.fields["views"], eye_log::Value::U64(15));
        assert_eq!(rec.fields["views_used"], eye_log::Value::U64(15));
        match rec.fields["rms_px"] {
            eye_log::Value::F64(v) => assert!((0.15..=0.25).contains(&v), "rms_px={v}"),
            ref other => panic!("expected F64 rms_px, got {other:?}"),
        }
        match rec.fields["module"] {
            eye_log::Value::Str(_) => {}
            ref other => panic!("expected Str module, got {other:?}"),
        }
        match rec.fields["prior_z"] {
            eye_log::Value::F64(v) => assert!((v - 0.655).abs() < 0.05, "prior_z={v}"),
            ref other => panic!("expected F64 prior_z, got {other:?}"),
        }

        assert!(
            !records.iter().any(|r| r.level == eye_log::Level::Warn),
            "unexpected warn records: {records:?}"
        );

        let residuals: Vec<_> = records
            .iter()
            .filter(|r| r.message == "view residual")
            .collect();
        assert_eq!(residuals.len(), 15);
        for r in &residuals {
            assert_eq!(r.fields["kept"], eye_log::Value::Bool(true));
        }

        let rejected: Vec<_> = records
            .iter()
            .filter(|r| r.message == "view rejected")
            .collect();
        assert_eq!(rejected.len(), 0);

        let lm_solve: Vec<_> = records.iter().filter(|r| r.message == "lm solve").collect();
        assert_eq!(lm_solve.len(), 1);
        match lm_solve[0].fields["evaluations"] {
            eye_log::Value::U64(v) => assert!(v > 0),
            ref other => panic!("expected U64 evaluations, got {other:?}"),
        }
        let (residuals, params) = match (
            &lm_solve[0].fields["residuals"],
            &lm_solve[0].fields["params"],
        ) {
            (eye_log::Value::U64(r), eye_log::Value::U64(p)) => (*r, *p),
            other => panic!("expected U64 residuals/params, got {other:?}"),
        };
        assert_eq!(
            lm_solve[0].fields["dof"],
            eye_log::Value::U64(residuals - params)
        );

        let closed_form: Vec<_> = records
            .iter()
            .filter(|r| r.message == "closed-form intrinsics")
            .collect();
        assert_eq!(closed_form.len(), 1);
    }

    #[test]
    fn test_logs_view_rejected_at_debug() {
        let mut views = synthetic_views(15);
        let intr = truth_intrinsics();
        let mut pose_rng = SplitMix64::new(17);
        let poses =
            synthetic_board_poses(&mut pose_rng, 15, accept_in_bounds(&intr, &board_spec()));
        views[7] = observation_from_pose(&intr, &poses[7], &board_spec(), None);
        let mut noise_rng = SplitMix64::new(99);
        for corner in &mut views[7].corners {
            corner.image_px.x += 2.0 * noise_rng.gaussian();
            corner.image_px.y += 2.0 * noise_rng.gaussian();
        }

        let (fit, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            calibrate_intrinsics(640, 360, &views, &IntrinsicsConfig::default()).unwrap()
        });
        assert!(!fit.views_used.contains(&7));

        let rejected: Vec<_> = records
            .iter()
            .filter(|r| r.message == "view rejected")
            .collect();
        assert_eq!(rejected.len(), 1);
        let rec = rejected[0];
        assert_eq!(rec.level, eye_log::Level::Debug);
        assert_eq!(rec.fields["view"], eye_log::Value::U64(7));
        match (&rec.fields["rms_px"], &rec.fields["threshold_px"]) {
            (eye_log::Value::F64(rms), eye_log::Value::F64(threshold)) => {
                assert!(rms > threshold, "rms={rms} threshold={threshold}");
            }
            other => panic!("expected F64 rms_px/threshold_px, got {other:?}"),
        }
        assert_eq!(
            rec.fields[field::REASON],
            eye_log::Value::Str("rms_above_threshold".to_string())
        );

        let lm_solve: Vec<_> = records.iter().filter(|r| r.message == "lm solve").collect();
        assert_eq!(lm_solve.len(), 2);

        let residuals: Vec<_> = records
            .iter()
            .filter(|r| r.message == "view residual")
            .collect();
        assert_eq!(residuals.len(), 15);
        for r in &residuals {
            let kept = r.fields["kept"] == eye_log::Value::Bool(true);
            let is_view_7 = r.fields["view"] == eye_log::Value::U64(7);
            assert_eq!(
                kept, !is_view_7,
                "view={:?} kept={:?}",
                r.fields["view"], r.fields["kept"]
            );
        }
    }

    #[test]
    fn test_logs_focal_prior_warning_at_warn() {
        let (prior_z, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            warn_if_outside_priors(
                &Intrinsics {
                    fx: 300.0,
                    ..truth_intrinsics()
                },
                640,
                360,
            )
            .unwrap()
        });

        let warnings: Vec<_> = records
            .iter()
            .filter(|r| r.level == eye_log::Level::Warn)
            .collect();
        assert_eq!(warnings.len(), 1);
        let rec = warnings[0];
        match rec.fields["prior_z"] {
            eye_log::Value::F64(v) => assert!(v < -2.0, "prior_z={v}"),
            ref other => panic!("expected F64 prior_z, got {other:?}"),
        }
        assert_eq!(rec.fields["fx"], eye_log::Value::F64(300.0));
        assert!(rec.fields.contains_key("prior_f_px"));
        assert!(rec.fields.contains_key("prior_sigma_px"));
        assert_eq!(rec.fields["prior_z"], eye_log::Value::F64(prior_z));
    }

    fn edp1_rig() -> Rig {
        let screen = ScreenModel {
            output: OutputId::from("eDP-1"),
            size_mm: Vector2::new(310.0, 174.375),
            size_px: (3840, 2160),
            scale: 2.0,
        };
        let rgb = CameraModel {
            id: CameraId::from("rgb"),
            width: 1280,
            height: 720,
            fx: 858.52,
            fy: 858.52,
            cx: 640.0,
            cy: 360.0,
            distortion: [0.0; 5],
            screen_from_camera: Isometry3::identity(),
        };
        let ir = CameraModel {
            id: CameraId::from("ir"),
            width: 640,
            height: 360,
            fx: 429.26,
            fy: 429.26,
            cx: 320.0,
            cy: 180.0,
            distortion: [0.0; 5],
            screen_from_camera: Isometry3::identity(),
        };
        Rig::new(vec![rgb, ir], screen).unwrap()
    }

    #[test]
    fn test_apply_intrinsics_writes_only_named_camera() {
        let views = synthetic_views(15);
        let fit = calibrate_intrinsics(640, 360, &views, &IntrinsicsConfig::default()).unwrap();

        let rig = edp1_rig();
        let rgb_before = rig.camera("rgb").unwrap().clone();
        let ir_before_pose = rig.camera("ir").unwrap().screen_from_camera;

        let updated = apply_intrinsics(rig, &CameraId::from("ir"), &fit).unwrap();
        let ir_after = updated.camera("ir").unwrap();
        assert_relative_eq!(ir_after.fx, fit.intrinsics.fx, epsilon = 1e-12);
        assert_relative_eq!(ir_after.fy, fit.intrinsics.fy, epsilon = 1e-12);
        assert_relative_eq!(ir_after.cx, fit.intrinsics.cx, epsilon = 1e-12);
        assert_relative_eq!(ir_after.cy, fit.intrinsics.cy, epsilon = 1e-12);
        assert_eq!(ir_after.distortion, fit.intrinsics.distortion.to_opencv());
        assert_eq!(ir_after.screen_from_camera, ir_before_pose);

        let rgb_after = updated.camera("rgb").unwrap();
        assert_eq!(*rgb_after, rgb_before);

        let rig2 = edp1_rig();
        let unknown_err = apply_intrinsics(rig2, &CameraId::from("depth"), &fit).unwrap_err();
        assert!(matches!(unknown_err, CalibrationError::UnknownCamera(id) if id == "depth"));

        let rig3 = edp1_rig();
        let wrong_size_err = apply_intrinsics(rig3, &CameraId::from("rgb"), &fit).unwrap_err();
        assert!(matches!(
            wrong_size_err,
            CalibrationError::Param { name: "camera", .. }
        ));
    }

    #[test]
    fn test_end_to_end_rendered_boards() {
        use crate::checkerboard::{DetectorConfig, detect_board};
        use crate::testutil::render_board;

        let intr = truth_intrinsics();
        let spec = board_spec();
        let mut pose_rng = SplitMix64::new(23);
        let poses =
            synthetic_board_poses(&mut pose_rng, 12, accept_within_margin(&intr, &spec, 70.0));

        let views: Vec<BoardObservation> = poses
            .iter()
            .enumerate()
            .map(|(seed, pose)| {
                let (img, _truth) = render_board(&intr, pose, &spec, seed as u64);
                detect_board(&img.view(), &spec, &DetectorConfig::default())
                    .unwrap()
                    .unwrap_or_else(|| panic!("seed={seed} pose={pose:?}: board not found"))
            })
            .collect();

        let fit = calibrate_intrinsics(640, 360, &views, &IntrinsicsConfig::default()).unwrap();
        assert_relative_eq!(fit.intrinsics.fx, intr.fx, max_relative = 0.01);
        assert!(fit.rms_px < 0.3, "rms_px={}", fit.rms_px);

        let rig = edp1_rig();
        let updated = apply_intrinsics(rig, &CameraId::from("ir"), &fit).unwrap();
        let ir_after = updated.camera("ir").unwrap();
        assert_relative_eq!(ir_after.fx, 457.0, max_relative = 0.01);
        assert_eq!(fit.module, LensModule::Fov75_8);
    }
}
