use std::collections::{BTreeMap, HashMap};

use eye_core::log::field;
use eye_core::{GazeRay, RaySource, Rig};
use eye_geometry::angles::yaw_pitch_from_direction;
use nalgebra::{
    Cholesky, Matrix2, Point2, Point3, Quaternion, SMatrix, SVector, Unit, UnitQuaternion, Vector2,
};

use crate::correction::{
    AngularCorrection, CalibrationPose, CorrectionModel, EyeKey, PROFILE_VERSION, Provenance,
    UserProfile, design, design_quad, design12, eye_label,
};
use crate::error::CalibrationError;

#[derive(Debug, Clone)]
pub struct FitSample {
    pub ray: GazeRay,
    pub target_mm: Point2<f64>,
    pub source: RaySource,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FitConfig {
    pub sample_outlier_mads: f64,
    pub min_samples_per_target: usize,
    pub fixation_jitter_deg: f64,
    pub huber_k: f64,
    pub target_outlier_factor: f64,
    pub slope_prior_sigma: f64,
    /// Ridge prior sigma on the four slope terms for every source except `rgb-only`
    /// (`slope_prior_sigma` keeps applying to `rgb-only`).
    pub ir_slope_prior_sigma: f64,
    pub offset_prior_sigma_deg: f64,
    pub quad_prior_sigma: f64,
    pub min_targets_affine: usize,
    pub min_targets_offset: usize,
    pub min_targets_quadratic: usize,
    /// Skips model selection and fits this model directly, as long as `min_targets_offset` is met.
    /// `None` fits `HeadFrame` when any sample carries a head pose, and otherwise walks the
    /// target-count ladder (`min_targets_*`, ending in `Quadratic`).
    pub model_override: Option<CorrectionModel>,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            sample_outlier_mads: 3.0,
            min_samples_per_target: 5,
            fixation_jitter_deg: 0.25,
            huber_k: 1.345,
            target_outlier_factor: 3.0,
            slope_prior_sigma: 0.15,
            ir_slope_prior_sigma: 0.5,
            offset_prior_sigma_deg: 10.0,
            quad_prior_sigma: 2.0,
            min_targets_affine: 6,
            min_targets_offset: 3,
            min_targets_quadratic: 12,
            model_override: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ProfileMeta {
    pub name: String,
    pub created_unix_s: u64,
    pub estimator: String,
    pub provenance: Provenance,
}

#[derive(Debug, Clone)]
pub struct FitOutcome {
    pub profile: UserProfile,
    pub reports: Vec<EyeFitReport>,
}

impl FitOutcome {
    /// One verdict per target index, merged across the primary source's eyes (the lowest
    /// `source` among `reports`): an index is `rejected` if any such eye's diagnostics rejected
    /// it. When rejected, the reported `samples`/`residual_deg`/`reason` come from whichever
    /// rejecting eye has the most samples; when not rejected, they come from whichever eye has
    /// the most samples.
    pub fn verdicts(&self) -> Vec<TargetVerdict> {
        let Some(primary) = self.reports.iter().map(|r| r.source).min() else {
            return Vec::new();
        };
        let mut merged: BTreeMap<u32, TargetVerdict> = BTreeMap::new();
        for report in self.reports.iter().filter(|r| r.source == primary) {
            for v in &report.diagnostics {
                match merged.entry(v.index) {
                    std::collections::btree_map::Entry::Vacant(e) => {
                        e.insert(v.clone());
                    }
                    std::collections::btree_map::Entry::Occupied(mut e) => {
                        let existing = e.get_mut();
                        *existing = merge_verdicts(existing, v);
                    }
                }
            }
        }
        merged.into_values().collect()
    }
}

/// Merges two eyes' verdicts for the same target index. A rejected result always carries its
/// `reason`/`residual_deg`/`samples` from a rejecting eye (the one with more samples, if both
/// reject), never from an eye that accepted the target.
fn merge_verdicts(a: &TargetVerdict, b: &TargetVerdict) -> TargetVerdict {
    match (a.rejected, b.rejected) {
        (true, false) => a.clone(),
        (false, true) => b.clone(),
        _ if b.samples > a.samples => b.clone(),
        _ => a.clone(),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EyeFitReport {
    pub source: RaySource,
    pub key: EyeKey,
    pub model: Option<CorrectionModel>,
    pub targets_used: Vec<u32>,
    pub targets_rejected: Vec<u32>,
    pub rms_before_deg: f64,
    pub rms_after_deg: f64,
    /// Leave-one-target-out RMS over per-target means, in-fit (no filter, no screen
    /// intersection): a fit-stability diagnostic, not the accuracy estimate (the bench LOTO is).
    pub loo_target_mean_deg: f64,
    /// Samples skipped because the HeadFrame model was requested and the ray carried no head pose.
    pub samples_without_head_pose: usize,
    pub diagnostics: Vec<TargetVerdict>,
}

/// Why `DotSessionFit` excluded a target from the fit. The limit named by each variant is the
/// same threshold the fitter itself applies; this is a label on an existing decision, not a new one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TargetReject {
    TooFewSamples { have: usize, need: usize },
    Jitter { deg: f64, limit: f64 },
    Residual { deg: f64, limit: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct TargetVerdict {
    pub index: u32,
    pub samples: usize,
    /// Diagnostic magnitude in the unit the matching `reason` was decided in: degrees of raw
    /// angular spread for a `TooFewSamples` diagnostic (computed before any model is fit), or the
    /// fitter's unitless covariance-weighted residual `rho` for a `Residual` diagnostic (computed
    /// after fitting). The two are not comparable across variants.
    pub residual_deg: f64,
    pub rejected: bool,
    pub reason: Option<TargetReject>,
}

#[derive(Debug, Clone)]
struct TargetAgg {
    index: u32,
    samples: usize,
    observed: Vector2<f64>,
    desired: Vector2<f64>,
    cov_inv: Matrix2<f64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DotSessionFit;

impl DotSessionFit {
    pub fn fit(samples: &[FitSample], rig: &Rig) -> Result<UserProfile, CalibrationError> {
        Self::fit_with(samples, rig, &FitConfig::default(), ProfileMeta::default())
            .map(|outcome| outcome.profile)
    }

    pub fn fit_with(
        samples: &[FitSample],
        rig: &Rig,
        cfg: &FitConfig,
        meta: ProfileMeta,
    ) -> Result<FitOutcome, CalibrationError> {
        let primary = samples.iter().map(|s| s.source).min();
        let primary_samples: Vec<&FitSample> = samples
            .iter()
            .filter(|s| Some(s.source) == primary)
            .collect();
        let calibration_pose = calibration_pose(&primary_samples);

        let mut target_index_of: HashMap<(u64, u64), u32> = HashMap::new();
        let mut target_mm_of: Vec<Point2<f64>> = Vec::new();
        for s in samples {
            let key = (s.target_mm.x.to_bits(), s.target_mm.y.to_bits());
            target_index_of.entry(key).or_insert_with(|| {
                let idx = target_mm_of.len() as u32;
                target_mm_of.push(s.target_mm);
                idx
            });
        }

        let mut by_source: BTreeMap<RaySource, Vec<&FitSample>> = BTreeMap::new();
        for s in samples {
            by_source.entry(s.source).or_default().push(s);
        }

        if by_source.is_empty() {
            return Err(CalibrationError::InsufficientData {
                what: "targets",
                need: cfg.min_targets_offset,
                got: 0,
            });
        }

        let mut corrections = BTreeMap::new();
        let mut reports = Vec::new();
        let mut primary_err: Option<CalibrationError> = None;

        for (source, source_samples) in &by_source {
            let source = *source;
            match fit_source(source, source_samples, &target_index_of, &target_mm_of, cfg) {
                Ok(source_fit) => {
                    if source_fit.eyes.is_empty() {
                        if Some(source) == primary {
                            let (_, reason, got) = source_fit
                                .omitted
                                .iter()
                                .min_by_key(|(_, _, got)| *got)
                                .copied()
                                .unwrap_or((
                                    EyeKey::Right,
                                    "too_few_targets",
                                    source_fit.max_usable,
                                ));
                            primary_err = Some(CalibrationError::InsufficientData {
                                what: if reason == "too_few_targets_after_rejection" {
                                    "targets after robust rejection"
                                } else {
                                    "targets"
                                },
                                need: cfg.min_targets_offset,
                                got,
                            });
                        } else {
                            tracing::debug!(
                                source = source.as_str(),
                                { field::REASON } = "no_eye_fitted",
                                "source omitted"
                            );
                            reports.extend(source_fit.reports);
                        }
                    } else {
                        corrections.insert(source, source_fit.eyes);
                        reports.extend(source_fit.reports);
                    }
                }
                Err(e) => {
                    if Some(source) == primary {
                        primary_err = Some(e);
                    } else {
                        tracing::debug!(source = source.as_str(), reason = %e, "source omitted");
                    }
                }
            }
        }

        if let Some(e) = primary_err {
            return Err(e);
        }

        let rig_fingerprint = crate::correction::rig_fingerprint(rig);
        tracing::info!(
            sources = corrections.len() as u64,
            corrections = corrections.values().map(BTreeMap::len).sum::<usize>() as u64,
            targets = target_mm_of.len() as u64,
            samples = samples.len() as u64,
            rig_fingerprint = %rig_fingerprint,
            name = %meta.name,
            "profile fitted"
        );

        Ok(FitOutcome {
            profile: UserProfile {
                version: PROFILE_VERSION,
                name: meta.name,
                created_unix_s: meta.created_unix_s,
                rig_fingerprint,
                estimator: meta.estimator,
                corrections,
                calibration_pose: Some(calibration_pose),
                provenance: meta.provenance,
            },
            reports,
        })
    }
}

struct SourceFit {
    eyes: BTreeMap<EyeKey, AngularCorrection>,
    reports: Vec<EyeFitReport>,
    max_usable: usize,
    omitted: Vec<(EyeKey, &'static str, usize)>,
}

/// Fits every eye present in `samples` (all one `source`) against the shared `target_index_of`
/// global indexing. `Err` only for the `HeadFrame` precondition (no sample of this source carries
/// a head pose); an eye (or every eye) fitting too few targets is reported in `SourceFit::eyes`
/// being partial or empty, not as an `Err` (the caller decides whether that is fatal, based on
/// whether `source` is the primary one).
fn fit_source(
    source: RaySource,
    samples: &[&FitSample],
    target_index_of: &HashMap<(u64, u64), u32>,
    target_mm_of: &[Point2<f64>],
    cfg: &FitConfig,
) -> Result<SourceFit, CalibrationError> {
    let model_override = cfg.model_override.or_else(|| {
        samples
            .iter()
            .any(|s| s.ray.head_rotation.is_some())
            .then_some(CorrectionModel::HeadFrame)
    });
    let use_head_frame = model_override == Some(CorrectionModel::HeadFrame);
    let (samples, without_pose): (Vec<&FitSample>, Vec<&FitSample>) = samples
        .iter()
        .copied()
        .partition(|s| !use_head_frame || s.ray.head_rotation.is_some());
    if use_head_frame && samples.is_empty() {
        return Err(CalibrationError::NoHeadPose {
            samples: without_pose.len(),
        });
    }
    if !without_pose.is_empty() {
        tracing::debug!(
            source = source.as_str(),
            { field::REASON } = "no_head_pose",
            dropped = without_pose.len() as u64,
            "samples skipped"
        );
    }

    let mut by_eye_target: HashMap<(EyeKey, u32), Vec<&FitSample>> = HashMap::new();
    for s in samples.iter().copied() {
        let key = (s.target_mm.x.to_bits(), s.target_mm.y.to_bits());
        let idx = target_index_of[&key];
        let eye = EyeKey::from(s.ray.side);
        by_eye_target.entry((eye, idx)).or_default().push(s);
    }

    let mut eyes_present: Vec<EyeKey> = by_eye_target.keys().map(|(e, _)| *e).collect();
    eyes_present.sort();
    eyes_present.dedup();

    let jitter_rad = cfg.fixation_jitter_deg.to_radians();
    let s_off = cfg.offset_prior_sigma_deg.to_radians();
    let s_slope = if source == RaySource::RgbOnly {
        cfg.slope_prior_sigma
    } else {
        cfg.ir_slope_prior_sigma
    };
    let prior6 = SMatrix::<f64, 6, 6>::from_diagonal(&SVector::<f64, 6>::from([
        1.0 / (s_off * s_off),
        1.0 / (s_slope * s_slope),
        1.0 / (s_slope * s_slope),
        1.0 / (s_off * s_off),
        1.0 / (s_slope * s_slope),
        1.0 / (s_slope * s_slope),
    ]));
    let s_quad = cfg.quad_prior_sigma;
    let prior12 = {
        let mut p = SMatrix::<f64, 12, 12>::zeros();
        p.fixed_view_mut::<6, 6>(0, 0).copy_from(&prior6);
        p.fixed_view_mut::<6, 6>(6, 6)
            .copy_from(&SMatrix::<f64, 6, 6>::from_diagonal_element(
                1.0 / (s_quad * s_quad),
            ));
        p
    };

    let mut eyes = BTreeMap::new();
    let mut reports = Vec::new();
    let mut max_usable = 0usize;
    let mut omitted: Vec<(EyeKey, &'static str, usize)> = Vec::new();

    for eye in eyes_present {
        let mut sample_rejected: Vec<u32> = Vec::new();
        let mut usable: Vec<TargetAgg> = Vec::new();
        let mut diagnostics: Vec<TargetVerdict> = Vec::new();

        let mut target_indices: Vec<u32> = (0..target_mm_of.len() as u32)
            .filter(|i| by_eye_target.contains_key(&(eye, *i)))
            .collect();
        target_indices.sort_unstable();

        for idx in target_indices {
            let group = &by_eye_target[&(eye, idx)];
            let target_mm = target_mm_of[idx as usize];
            let target_point = Point3::new(target_mm.x, target_mm.y, 0.0);

            let mut os = Vec::with_capacity(group.len());
            let mut ds = Vec::with_capacity(group.len());
            let mut es = Vec::with_capacity(group.len());
            for s in group.iter() {
                let desired_dir = Unit::new_normalize(target_point - s.ray.origin);
                let (o, d) = if use_head_frame {
                    let inv = s.ray.head_rotation.expect("partitioned above").inverse();
                    (
                        yaw_pitch_from_direction(&(inv * s.ray.direction)),
                        yaw_pitch_from_direction(&(inv * desired_dir)),
                    )
                } else {
                    (
                        yaw_pitch_from_direction(&s.ray.direction),
                        yaw_pitch_from_direction(&desired_dir),
                    )
                };
                os.push(o);
                ds.push(d);
                es.push(o - d);
            }

            let keep = reject_sample_outliers(&es, cfg);
            let n_keep = keep.iter().filter(|&&k| k).count();
            if n_keep < cfg.min_samples_per_target {
                tracing::debug!(
                    source = source.as_str(),
                    eye = eye_label(eye),
                    target = idx as u64,
                    { field::REASON } = "too_few_samples",
                    samples = group.len() as u64,
                    kept = n_keep as u64,
                    min_samples_per_target = cfg.min_samples_per_target as u64,
                    "target rejected"
                );
                sample_rejected.push(idx);
                let spread_deg = sample_spread_deg(&es);
                diagnostics.push(TargetVerdict {
                    index: idx,
                    samples: n_keep,
                    residual_deg: spread_deg,
                    rejected: true,
                    reason: Some(TargetReject::TooFewSamples {
                        have: n_keep,
                        need: cfg.min_samples_per_target,
                    }),
                });
                continue;
            }

            let kept_o: Vec<Vector2<f64>> = os
                .iter()
                .zip(&keep)
                .filter(|(_, k)| **k)
                .map(|(o, _)| *o)
                .collect();
            let kept_d: Vec<Vector2<f64>> = ds
                .iter()
                .zip(&keep)
                .filter(|(_, k)| **k)
                .map(|(d, _)| *d)
                .collect();
            let kept_e: Vec<Vector2<f64>> = es
                .iter()
                .zip(&keep)
                .filter(|(_, k)| **k)
                .map(|(e, _)| *e)
                .collect();

            let n = kept_o.len() as f64;
            let o_k = kept_o.iter().fold(Vector2::zeros(), |a, b| a + b) / n;
            let d_k = kept_d.iter().fold(Vector2::zeros(), |a, b| a + b) / n;
            let mean_e = kept_e.iter().fold(Vector2::zeros(), |a, b| a + b) / n;
            let mut s_k = Matrix2::zeros();
            for e in &kept_e {
                let diff = e - mean_e;
                s_k += diff * diff.transpose();
            }
            if kept_e.len() > 1 {
                s_k /= (kept_e.len() - 1) as f64;
            }
            let c_k = s_k / n + Matrix2::identity() * (jitter_rad * jitter_rad);
            let cov_inv = c_k
                .try_inverse()
                .unwrap_or_else(|| Matrix2::identity() / (jitter_rad * jitter_rad));

            tracing::trace!(
                source = source.as_str(),
                eye = eye_label(eye),
                target = idx as u64,
                samples = kept_o.len() as u64,
                err_yaw_deg = mean_e.x.to_degrees(),
                err_pitch_deg = mean_e.y.to_degrees(),
                "target aggregated"
            );

            usable.push(TargetAgg {
                index: idx,
                samples: n_keep,
                observed: o_k,
                desired: d_k,
                cov_inv,
            });
        }

        max_usable = max_usable.max(usable.len());

        if usable.len() < cfg.min_targets_offset {
            tracing::debug!(
                source = source.as_str(),
                eye = eye_label(eye),
                { field::REASON } = "too_few_targets",
                usable = usable.len() as u64,
                min_targets_offset = cfg.min_targets_offset as u64,
                "eye omitted"
            );
            omitted.push((eye, "too_few_targets", usable.len()));
            reports.push(EyeFitReport {
                source,
                key: eye,
                model: None,
                targets_used: Vec::new(),
                targets_rejected: sample_rejected,
                rms_before_deg: 0.0,
                rms_after_deg: 0.0,
                loo_target_mean_deg: 0.0,
                samples_without_head_pose: without_pose.len(),
                diagnostics,
            });
            continue;
        }

        let mut weights = vec![1.0; usable.len()];
        let mut theta_prev: Option<SVector<f64, 6>> = None;
        let mut rho = vec![0.0; usable.len()];
        for iteration in 0..10u32 {
            let (theta, _cov, _chi2) = solve_weighted(&usable, &weights, &prior6)
                .expect("ridge prior keeps the normal matrix positive definite");
            rho = usable
                .iter()
                .map(|t| {
                    let r = t.desired - (t.observed + design(&t.observed) * theta);
                    (r.dot(&(t.cov_inv * r))).sqrt()
                })
                .collect();
            let step = theta_prev.map(|p| (theta - p).norm());
            let converged = step.map(|s| s < 1e-9).unwrap_or(false);
            tracing::trace!(
                source = source.as_str(),
                eye = eye_label(eye),
                iteration = u64::from(iteration),
                step = step.unwrap_or(0.0),
                converged,
                "irls iteration"
            );
            theta_prev = Some(theta);
            if converged {
                break;
            }
            weights = rho
                .iter()
                .map(|&r| {
                    if r > 0.0 {
                        (cfg.huber_k / r).min(1.0)
                    } else {
                        1.0
                    }
                })
                .collect();
        }

        let mut rho_sorted = rho.clone();
        let median_rho = median(&mut rho_sorted);
        let residual_limit = (cfg.target_outlier_factor * median_rho).max(3.0);
        let mut huber_rejected = Vec::new();
        let mut remaining = Vec::new();
        for (t, &r) in usable.iter().zip(&rho) {
            let rejected = r > residual_limit;
            if rejected {
                tracing::debug!(
                    source = source.as_str(),
                    eye = eye_label(eye),
                    target = t.index as u64,
                    { field::REASON } = "huber_outlier",
                    rho = r,
                    median_rho,
                    target_outlier_factor = cfg.target_outlier_factor,
                    "target rejected"
                );
                huber_rejected.push(t.index);
            } else {
                tracing::debug!(
                    source = source.as_str(),
                    eye = eye_label(eye),
                    target = t.index as u64,
                    rho = r,
                    median_rho,
                    "target residual"
                );
                remaining.push(t.clone());
            }
            diagnostics.push(TargetVerdict {
                index: t.index,
                samples: t.samples,
                residual_deg: r,
                rejected,
                reason: rejected.then_some(TargetReject::Residual {
                    deg: r,
                    limit: residual_limit,
                }),
            });
        }

        let mut targets_rejected = sample_rejected.clone();
        targets_rejected.extend(huber_rejected);
        targets_rejected.sort_unstable();

        if remaining.len() < cfg.min_targets_offset {
            tracing::debug!(
                source = source.as_str(),
                eye = eye_label(eye),
                { field::REASON } = "too_few_targets_after_rejection",
                remaining = remaining.len() as u64,
                min_targets_offset = cfg.min_targets_offset as u64,
                "eye omitted"
            );
            omitted.push((eye, "too_few_targets_after_rejection", remaining.len()));
            reports.push(EyeFitReport {
                source,
                key: eye,
                model: None,
                targets_used: Vec::new(),
                targets_rejected,
                rms_before_deg: 0.0,
                rms_after_deg: 0.0,
                loo_target_mean_deg: 0.0,
                samples_without_head_pose: without_pose.len(),
                diagnostics,
            });
            continue;
        }

        let model = match model_override {
            Some(m) => m,
            None if remaining.len() >= cfg.min_targets_quadratic => CorrectionModel::Quadratic,
            None if remaining.len() >= cfg.min_targets_affine => CorrectionModel::Affine,
            None => CorrectionModel::OffsetOnly,
        };

        let ones = vec![1.0; remaining.len()];
        let (
            theta_final,
            quad_final,
            cov_final_raw,
            quad_cov_final_raw,
            cross_final_raw,
            chi2_final,
            dof,
        ) = match model {
            CorrectionModel::Affine | CorrectionModel::HeadFrame => {
                let (theta, cov, chi2) = solve_weighted(&remaining, &ones, &prior6)
                    .expect("ridge prior keeps the normal matrix positive definite");
                (
                    theta,
                    SVector::<f64, 6>::zeros(),
                    cov,
                    SMatrix::<f64, 6, 6>::zeros(),
                    SMatrix::<f64, 6, 6>::zeros(),
                    chi2,
                    2.0 * remaining.len() as f64 - 6.0,
                )
            }
            CorrectionModel::OffsetOnly => {
                let (theta2, cov2, chi2) = fit_offset_only(&remaining, s_off);
                let theta = SVector::<f64, 6>::from([theta2.x, 0.0, 0.0, theta2.y, 0.0, 0.0]);
                let mut cov = SMatrix::<f64, 6, 6>::zeros();
                cov[(0, 0)] = cov2[(0, 0)];
                cov[(0, 3)] = cov2[(0, 1)];
                cov[(3, 0)] = cov2[(1, 0)];
                cov[(3, 3)] = cov2[(1, 1)];
                (
                    theta,
                    SVector::<f64, 6>::zeros(),
                    cov,
                    SMatrix::<f64, 6, 6>::zeros(),
                    SMatrix::<f64, 6, 6>::zeros(),
                    chi2,
                    2.0 * remaining.len() as f64 - 2.0,
                )
            }
            CorrectionModel::Quadratic => {
                let (theta12, cov12, chi2) = solve_weighted12(&remaining, &ones, &prior12)
                    .expect("ridge prior keeps the normal matrix positive definite");
                let theta = SVector::<f64, 6>::from_fn(|i, _| theta12[i]);
                let quad = SVector::<f64, 6>::from_fn(|i, _| theta12[6 + i]);
                let cov = SMatrix::<f64, 6, 6>::from_fn(|r, k| cov12[(r, k)]);
                let quad_cov = SMatrix::<f64, 6, 6>::from_fn(|r, k| cov12[(6 + r, 6 + k)]);
                let cross = SMatrix::<f64, 6, 6>::from_fn(|r, k| cov12[(r, 6 + k)]);
                (
                    theta,
                    quad,
                    cov,
                    quad_cov,
                    cross,
                    chi2,
                    2.0 * remaining.len() as f64 - 12.0,
                )
            }
        };
        let scale = if dof > 0.0 {
            (chi2_final / dof).max(1.0)
        } else {
            1.0
        };
        let cov_final = cov_final_raw * scale;
        let quad_cov_final = quad_cov_final_raw * scale;
        let cross_final = cross_final_raw * scale;

        let mut targets_used: Vec<u32> = remaining.iter().map(|t| t.index).collect();
        targets_used.sort_unstable();

        let rms_before = rms_deg(remaining.iter().map(|t| (t.desired, t.observed)));
        let rms_after = rms_deg(remaining.iter().map(|t| {
            (
                t.desired,
                t.observed
                    + design(&t.observed) * theta_final
                    + design_quad(&t.observed) * quad_final,
            )
        }));

        let loo_target_mean = loo_target_mean_deg(eye, &remaining, model, s_off, &prior6, &prior12);

        eyes.insert(
            eye,
            AngularCorrection {
                theta: theta_to_array(&theta_final),
                cov: cov_to_array(&cov_final),
                quad: theta_to_array(&quad_final),
                quad_cov: cov_to_array(&quad_cov_final),
                quad_cross_cov: cov_to_array(&cross_final),
                model,
                targets_used: targets_used.len() as u32,
                rms_after_rad: rms_after.to_radians(),
            },
        );

        let chi2_reduced = if dof > 0.0 { chi2_final / dof } else { 0.0 };
        tracing::info!(
            source = source.as_str(),
            eye = eye_label(eye),
            model = ?model,
            targets_used = targets_used.len() as u64,
            targets_rejected = targets_rejected.len() as u64,
            rms_before_deg = rms_before,
            rms_after_deg = rms_after,
            loo_target_mean_deg = loo_target_mean,
            chi2_reduced,
            "eye fitted"
        );

        reports.push(EyeFitReport {
            source,
            key: eye,
            model: Some(model),
            targets_used,
            targets_rejected,
            rms_before_deg: rms_before,
            rms_after_deg: rms_after,
            loo_target_mean_deg: loo_target_mean,
            samples_without_head_pose: without_pose.len(),
            diagnostics,
        });
    }

    Ok(SourceFit {
        eyes,
        reports,
        max_usable,
        omitted,
    })
}

/// Mean origin per eye over ALL input samples; quaternion mean over the samples that carry a
/// pose, with each quaternion sign-aligned to the first so antipodal representations do not
/// cancel.
fn calibration_pose(samples: &[&FitSample]) -> CalibrationPose {
    let mut origin_sum: BTreeMap<EyeKey, ([f64; 3], u32)> = BTreeMap::new();
    for s in samples {
        let e = origin_sum
            .entry(EyeKey::from(s.ray.side))
            .or_insert(([0.0; 3], 0));
        for k in 0..3 {
            e.0[k] += s.ray.origin[k];
        }
        e.1 += 1;
    }
    let eye_origin_mm = origin_sum
        .into_iter()
        .map(|(k, (sum, n))| (k, sum.map(|v| v / f64::from(n))))
        .collect();

    let mut first: Option<Quaternion<f64>> = None;
    let mut q_sum = nalgebra::Vector4::<f64>::zeros();
    let mut with_pose = 0u32;
    for q in samples.iter().filter_map(|s| s.ray.head_rotation) {
        let q0 = *first.get_or_insert(*q);
        let sign = if q.coords.dot(&q0.coords) < 0.0 {
            -1.0
        } else {
            1.0
        };
        q_sum += sign * q.coords;
        with_pose += 1;
    }
    let head_rotation = (with_pose > 0).then(|| {
        let mean = UnitQuaternion::from_quaternion(Quaternion::from(q_sum));
        [mean.i, mean.j, mean.k, mean.w]
    });
    CalibrationPose {
        head_rotation,
        samples_with_head_pose: with_pose,
        samples_total: samples.len() as u32,
        eye_origin_mm,
    }
}

fn reject_sample_outliers(es: &[Vector2<f64>], cfg: &FitConfig) -> Vec<bool> {
    let xs: Vec<f64> = es.iter().map(|e| e.x).collect();
    let ys: Vec<f64> = es.iter().map(|e| e.y).collect();
    let (med_x, mad_x) = median_mad(&xs);
    let (med_y, mad_y) = median_mad(&ys);
    let floor = 0.1_f64.to_radians();
    let thresh_x = cfg.sample_outlier_mads * 1.4826 * mad_x.max(floor);
    let thresh_y = cfg.sample_outlier_mads * 1.4826 * mad_y.max(floor);
    es.iter()
        .map(|e| (e.x - med_x).abs() <= thresh_x && (e.y - med_y).abs() <= thresh_y)
        .collect()
}

fn sample_spread_deg(es: &[Vector2<f64>]) -> f64 {
    let xs: Vec<f64> = es.iter().map(|e| e.x).collect();
    let ys: Vec<f64> = es.iter().map(|e| e.y).collect();
    let (_, mad_x) = median_mad(&xs);
    let (_, mad_y) = median_mad(&ys);
    (1.4826 * mad_x).hypot(1.4826 * mad_y).to_degrees()
}

fn median_mad(values: &[f64]) -> (f64, f64) {
    let mut sorted = values.to_vec();
    let med = median(&mut sorted);
    let mut abs_dev: Vec<f64> = values.iter().map(|v| (v - med).abs()).collect();
    let mad = median(&mut abs_dev);
    (med, mad)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

fn solve_weighted(
    targets: &[TargetAgg],
    weights: &[f64],
    prior: &SMatrix<f64, 6, 6>,
) -> Option<(SVector<f64, 6>, SMatrix<f64, 6, 6>, f64)> {
    let mut n = *prior;
    let mut g = SVector::<f64, 6>::zeros();
    for (t, &w) in targets.iter().zip(weights) {
        let a = design(&t.observed);
        n += a.transpose() * t.cov_inv * a * w;
        g += a.transpose() * t.cov_inv * (t.desired - t.observed) * w;
    }
    let chol = Cholesky::new(n)?;
    let theta = chol.solve(&g);
    let chi2: f64 = targets
        .iter()
        .map(|t| {
            let r = t.observed + design(&t.observed) * theta - t.desired;
            r.dot(&(t.cov_inv * r))
        })
        .sum();
    Some((theta, chol.inverse(), chi2))
}

fn solve_weighted12(
    targets: &[TargetAgg],
    weights: &[f64],
    prior: &SMatrix<f64, 12, 12>,
) -> Option<(SVector<f64, 12>, SMatrix<f64, 12, 12>, f64)> {
    let mut n = *prior;
    let mut g = SVector::<f64, 12>::zeros();
    for (t, &w) in targets.iter().zip(weights) {
        let a = design12(&t.observed);
        n += a.transpose() * t.cov_inv * a * w;
        g += a.transpose() * t.cov_inv * (t.desired - t.observed) * w;
    }
    let chol = Cholesky::new(n)?;
    let theta = chol.solve(&g);
    let chi2: f64 = targets
        .iter()
        .map(|t| {
            let r = t.observed + design12(&t.observed) * theta - t.desired;
            r.dot(&(t.cov_inv * r))
        })
        .sum();
    Some((theta, chol.inverse(), chi2))
}

fn fit_offset_only(targets: &[TargetAgg], prior_off_rad: f64) -> (Vector2<f64>, Matrix2<f64>, f64) {
    let prior = Matrix2::identity() * (1.0 / (prior_off_rad * prior_off_rad));
    let mut n = prior;
    let mut g = Vector2::zeros();
    for t in targets {
        n += t.cov_inv;
        g += t.cov_inv * (t.desired - t.observed);
    }
    let chol = Cholesky::new(n).expect("ridge prior keeps the normal matrix positive definite");
    let theta = chol.solve(&g);
    let chi2: f64 = targets
        .iter()
        .map(|t| {
            let r = t.desired - (t.observed + theta);
            r.dot(&(t.cov_inv * r))
        })
        .sum();
    (theta, chol.inverse(), chi2)
}

fn rms_deg(pairs: impl Iterator<Item = (Vector2<f64>, Vector2<f64>)>) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for (a, b) in pairs {
        sum += (a - b).norm_squared();
        n += 1;
    }
    (sum / n as f64).sqrt().to_degrees()
}

fn loo_target_mean_deg(
    eye: EyeKey,
    targets: &[TargetAgg],
    model: CorrectionModel,
    s_off: f64,
    prior6: &SMatrix<f64, 6, 6>,
    prior12: &SMatrix<f64, 12, 12>,
) -> f64 {
    let mut sum = 0.0;
    for (i, held_out) in targets.iter().enumerate() {
        let rest: Vec<TargetAgg> = targets
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, t)| t.clone())
            .collect();
        let predicted = match model {
            CorrectionModel::Affine | CorrectionModel::HeadFrame => {
                let ones = vec![1.0; rest.len()];
                let (theta, _, _) = solve_weighted(&rest, &ones, prior6)
                    .expect("ridge prior keeps the normal matrix positive definite");
                held_out.observed + design(&held_out.observed) * theta
            }
            CorrectionModel::OffsetOnly => {
                let (theta2, _, _) = fit_offset_only(&rest, s_off);
                held_out.observed + theta2
            }
            CorrectionModel::Quadratic => {
                let ones = vec![1.0; rest.len()];
                let (theta12, _, _) = solve_weighted12(&rest, &ones, prior12)
                    .expect("ridge prior keeps the normal matrix positive definite");
                held_out.observed + design12(&held_out.observed) * theta12
            }
        };
        let residual_deg = (held_out.desired - predicted).norm().to_degrees();
        tracing::debug!(
            eye = eye_label(eye),
            held_out = held_out.index as u64,
            residual_deg,
            model = ?model,
            "loo fold"
        );
        sum += (held_out.desired - predicted).norm_squared();
    }
    (sum / targets.len() as f64).sqrt().to_degrees()
}

fn theta_to_array(v: &SVector<f64, 6>) -> [f64; 6] {
    let mut out = [0.0; 6];
    for i in 0..6 {
        out[i] = v[i];
    }
    out
}

fn cov_to_array(m: &SMatrix<f64, 6, 6>) -> [[f64; 6]; 6] {
    let mut out = [[0.0; 6]; 6];
    for r in 0..6 {
        for c in 0..6 {
            out[r][c] = m[(r, c)];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use eye_core::stage::GazeCorrection;
    use eye_core::{CameraId, CameraModel, OutputId, RaySource, ScreenModel, Side};
    use eye_geometry::angles::direction_from_yaw_pitch;
    use eye_geometry::synth::SplitMix64;
    use nalgebra::{Matrix3, UnitQuaternion, Vector3};

    use super::*;
    use crate::correction::rig_fingerprint;
    use crate::nominal::nominal_screen_from_camera;
    use crate::protocol::{ProtocolConfig, TargetProtocol};

    #[derive(Clone, Copy)]
    struct Bias {
        gain_y: f64,
        offset_y_deg: f64,
        gain_p: f64,
        offset_p_deg: f64,
    }

    const BIASED: Bias = Bias {
        gain_y: 1.05,
        offset_y_deg: 2.0,
        gain_p: 0.97,
        offset_p_deg: -1.0,
    };
    const UNBIASED: Bias = Bias {
        gain_y: 1.0,
        offset_y_deg: 0.0,
        gain_p: 1.0,
        offset_p_deg: 0.0,
    };

    #[derive(Clone, Copy)]
    struct SessionConfig {
        grid: [u32; 2],
        samples_per_target: usize,
        side: Option<Side>,
        base_xy: (f64, f64),
        sway_mm: f64,
        bias: Bias,
        seed: u64,
        noise_deg: f64,
        source: RaySource,
    }

    impl Default for SessionConfig {
        fn default() -> Self {
            Self {
                grid: [3, 3],
                samples_per_target: 24,
                side: Some(Side::Right),
                base_xy: (185.0, 60.0),
                sway_mm: 10.0,
                bias: BIASED,
                seed: 31,
                noise_deg: 1.0,
                source: RaySource::RgbOnly,
            }
        }
    }

    fn screen() -> ScreenModel {
        ScreenModel {
            output: OutputId::from("eDP-1"),
            size_mm: Vector2::new(310.0, 174.375),
            size_px: (3840, 2160),
            scale: 2.0,
        }
    }

    fn fixture_rig() -> Rig {
        let camera = CameraModel {
            id: CameraId::from("ir"),
            width: 640,
            height: 360,
            fx: 429.25,
            fy: 429.25,
            cx: 320.0,
            cy: 180.0,
            distortion: [0.0; 5],
            screen_from_camera: nominal_screen_from_camera(&Point3::new(155.0, -7.0, 0.0)),
        };
        Rig::new(vec![camera], screen()).unwrap()
    }

    fn generate_session(cfg: &SessionConfig) -> Vec<FitSample> {
        let proto = TargetProtocol::new(ProtocolConfig {
            grid: cfg.grid,
            ..ProtocolConfig::default()
        })
        .unwrap();
        let screen = screen();
        let targets = proto.sequence(0);
        let mut rng = SplitMix64::new(cfg.seed);
        let mut out = Vec::with_capacity(targets.len() * cfg.samples_per_target);
        let mut n: u64 = 0;
        for t in &targets {
            let target_mm = proto.target_mm(t, &screen);
            for _ in 0..cfg.samples_per_target {
                let a = cfg.sway_mm;
                let phase = n as f64;
                let origin = Point3::new(
                    cfg.base_xy.0 + a * (2.0 * std::f64::consts::PI * phase / 50.0).sin(),
                    cfg.base_xy.1 + 0.5 * a * (2.0 * std::f64::consts::PI * phase / 37.0).sin(),
                    -500.0,
                );
                let true_dir =
                    Unit::new_normalize(Point3::new(target_mm.x, target_mm.y, 0.0) - origin);
                let true_angles = yaw_pitch_from_direction(&true_dir);
                let obs_yaw = cfg.bias.gain_y * true_angles.x
                    + cfg.bias.offset_y_deg.to_radians()
                    + cfg.noise_deg.to_radians() * rng.gaussian();
                let obs_pitch = cfg.bias.gain_p * true_angles.y
                    + cfg.bias.offset_p_deg.to_radians()
                    + cfg.noise_deg.to_radians() * rng.gaussian();
                let direction = direction_from_yaw_pitch(&Vector2::new(obs_yaw, obs_pitch));
                let ray = GazeRay {
                    side: cfg.side,
                    timestamp: eye_core::Timestamp::from_nanos(0),
                    origin,
                    direction,
                    angular_cov: Matrix2::identity() * 1.0_f64.to_radians().powi(2),
                    origin_cov: Matrix3::identity(),
                    head_rotation: None,
                };
                out.push(FitSample {
                    ray,
                    target_mm,
                    source: cfg.source,
                });
                n += 1;
            }
        }
        out
    }

    fn add_yaw_bias_deg(sample: &mut FitSample, delta_deg: f64) {
        let angles = yaw_pitch_from_direction(&sample.ray.direction);
        let shifted = Vector2::new(angles.x + delta_deg.to_radians(), angles.y);
        sample.ray.direction = direction_from_yaw_pitch(&shifted);
    }

    fn deg(rad: f64) -> f64 {
        rad.to_degrees()
    }

    #[test]
    fn test_recovers_offset_and_gain_3x3() {
        let samples = generate_session(&SessionConfig::default());
        let outcome = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap();
        let right = outcome
            .profile
            .correction(RaySource::RgbOnly, EyeKey::Right)
            .unwrap();
        assert_eq!(right.model, CorrectionModel::Affine);
        let th = right.theta;
        assert_abs_diff_eq!(deg(th[0]), -1.9048, epsilon = 0.25);
        assert_abs_diff_eq!(deg(th[3]), 1.0309, epsilon = 0.25);
        assert_abs_diff_eq!(th[1], -0.04762, epsilon = 0.025);
        assert_abs_diff_eq!(th[5], 0.03093, epsilon = 0.04);
        assert_abs_diff_eq!(th[2], 0.0, epsilon = 0.04);
        assert_abs_diff_eq!(th[4], 0.0, epsilon = 0.04);
        let report = outcome
            .reports
            .iter()
            .find(|r| r.key == EyeKey::Right)
            .unwrap();
        assert!(report.rms_after_deg < 0.5, "{}", report.rms_after_deg);
    }

    #[test]
    fn test_fit_groups_by_source_and_eye() {
        let rgb_cfg = SessionConfig::default();
        let ir_cfg = SessionConfig {
            bias: Bias {
                gain_y: 0.5,
                offset_y_deg: 0.0,
                gain_p: 0.5,
                offset_p_deg: 0.0,
            },
            source: RaySource::IrOnly,
            seed: 97,
            ..SessionConfig::default()
        };
        let mut samples = generate_session(&rgb_cfg);
        samples.extend(generate_session(&ir_cfg));

        let outcome = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap();

        assert_eq!(outcome.profile.version, PROFILE_VERSION);
        assert_eq!(
            outcome.profile.corrections.keys().collect::<Vec<_>>(),
            vec![&RaySource::RgbOnly, &RaySource::IrOnly]
        );
        let rgb = outcome
            .profile
            .correction(RaySource::RgbOnly, EyeKey::Right)
            .unwrap();
        assert_abs_diff_eq!(rgb.theta[1], -0.04762, epsilon = 0.025);
        let ir = outcome
            .profile
            .correction(RaySource::IrOnly, EyeKey::Right)
            .unwrap();
        assert_abs_diff_eq!(ir.theta[1], 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_ir_source_fit_recovers_low_gain_without_prior_shrinkage() {
        let base_cfg = SessionConfig {
            bias: Bias {
                gain_y: 0.4,
                offset_y_deg: 0.0,
                gain_p: 0.4,
                offset_p_deg: 0.0,
            },
            noise_deg: 5.0,
            seed: 83,
            ..SessionConfig::default()
        };
        let mut samples = generate_session(&SessionConfig {
            source: RaySource::IrOnly,
            ..base_cfg
        });
        samples.extend(generate_session(&SessionConfig {
            source: RaySource::RgbOnly,
            ..base_cfg
        }));

        let outcome = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap();

        let ir_theta1 = outcome
            .profile
            .correction(RaySource::IrOnly, EyeKey::Right)
            .unwrap()
            .theta[1];
        assert_abs_diff_eq!(ir_theta1, 1.5, epsilon = 0.2);

        let rgb_theta1 = outcome
            .profile
            .correction(RaySource::RgbOnly, EyeKey::Right)
            .unwrap()
            .theta[1];
        assert!(
            rgb_theta1 < ir_theta1 - 0.1,
            "rgb={rgb_theta1} ir={ir_theta1}"
        );
    }

    #[test]
    fn test_source_without_enough_targets_is_omitted_other_source_fitted() {
        let rgb_samples = generate_session(&SessionConfig::default());
        let ir_cfg = SessionConfig {
            source: RaySource::IrOnly,
            seed: 53,
            ..SessionConfig::default()
        };
        let per_target = ir_cfg.samples_per_target;
        let ir_samples: Vec<FitSample> = generate_session(&ir_cfg)
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i / per_target < 2)
            .map(|(_, s)| s)
            .collect();

        let mut samples = rgb_samples;
        samples.extend(ir_samples);

        let outcome = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap();

        assert!(
            outcome
                .profile
                .correction(RaySource::IrOnly, EyeKey::Right)
                .is_none()
        );
        assert!(
            outcome
                .profile
                .correction(RaySource::RgbOnly, EyeKey::Right)
                .is_some()
        );
        let ir_report = outcome
            .reports
            .iter()
            .find(|r| r.source == RaySource::IrOnly)
            .expect("the omitted source still reports");
        assert_eq!(ir_report.model, None);
    }

    #[test]
    fn test_calibration_pose_uses_primary_source_samples() {
        let rgb_cfg = SessionConfig::default();
        let ir_cfg = SessionConfig {
            base_xy: (250.0, 60.0),
            source: RaySource::IrOnly,
            seed: 61,
            ..SessionConfig::default()
        };
        let mut samples = generate_session(&rgb_cfg);
        samples.extend(generate_session(&ir_cfg));

        let outcome = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap();
        let pose = outcome.profile.calibration_pose.expect("pose recorded");
        let x = pose.eye_origin_mm[&EyeKey::Right][0];
        assert_abs_diff_eq!(x, 185.0, epsilon = 15.0);
    }

    #[test]
    fn test_verdicts_come_from_primary_source_only() {
        let rgb_samples = generate_session(&SessionConfig::default());
        let ir_cfg = SessionConfig {
            source: RaySource::IrOnly,
            seed: 71,
            ..SessionConfig::default()
        };
        let per_target = ir_cfg.samples_per_target;
        let mut ir_samples = generate_session(&ir_cfg);
        for i in 0..per_target {
            let idx = 4 * per_target + i;
            add_yaw_bias_deg(&mut ir_samples[idx], 10.0);
        }

        let mut samples = rgb_samples;
        samples.extend(ir_samples);

        let outcome = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap();
        let verdicts = outcome.verdicts();
        let v = verdicts.iter().find(|v| v.index == 4).unwrap();
        assert!(!v.rejected, "{v:?}");
    }

    #[test]
    fn test_fit_config_without_ir_slope_prior_parses_with_default() {
        let toml = "slope_prior_sigma = 0.2\n";
        let cfg: FitConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.slope_prior_sigma, 0.2);
        assert_eq!(cfg.ir_slope_prior_sigma, 0.5);
    }

    #[test]
    fn test_recovers_with_4x4() {
        let cfg = SessionConfig {
            grid: [4, 4],
            ..SessionConfig::default()
        };
        let samples = generate_session(&cfg);
        let affine_cfg = FitConfig {
            model_override: Some(CorrectionModel::Affine),
            ..FitConfig::default()
        };
        let outcome = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &affine_cfg,
            ProfileMeta::default(),
        )
        .unwrap();
        let right = outcome
            .profile
            .correction(RaySource::RgbOnly, EyeKey::Right)
            .unwrap();
        let th = right.theta;
        assert_abs_diff_eq!(deg(th[0]), -1.9048, epsilon = 0.15);
        assert_abs_diff_eq!(deg(th[3]), 1.0309, epsilon = 0.15);
        assert_abs_diff_eq!(th[1], -0.04762, epsilon = 0.02);
        assert_abs_diff_eq!(th[5], 0.03093, epsilon = 0.03);
    }

    #[test]
    fn test_head_sway_is_handled_by_per_sample_desired_angles() {
        let cfg = SessionConfig {
            sway_mm: 30.0,
            bias: UNBIASED,
            ..SessionConfig::default()
        };
        let samples = generate_session(&cfg);
        let outcome = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap();
        let right = outcome
            .profile
            .correction(RaySource::RgbOnly, EyeKey::Right)
            .unwrap();
        let th = right.theta;
        assert_abs_diff_eq!(deg(th[0]), 0.0, epsilon = 0.2);
        assert_abs_diff_eq!(deg(th[3]), 0.0, epsilon = 0.2);
        assert_abs_diff_eq!(th[1], 0.0, epsilon = 0.04);
        assert_abs_diff_eq!(th[5], 0.0, epsilon = 0.04);
    }

    #[test]
    fn test_blink_samples_rejected() {
        let mut samples = generate_session(&SessionConfig::default());
        let per_target = SessionConfig::default().samples_per_target;
        for target in 0..9 {
            for i in 0..3 {
                let idx = target * per_target + i;
                add_yaw_bias_deg(&mut samples[idx], 15.0);
            }
        }
        let outcome = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap();
        let right = outcome
            .profile
            .correction(RaySource::RgbOnly, EyeKey::Right)
            .unwrap();
        let th = right.theta;
        assert_abs_diff_eq!(deg(th[0]), -1.9048, epsilon = 0.25);
        assert_abs_diff_eq!(deg(th[3]), 1.0309, epsilon = 0.25);
        assert_abs_diff_eq!(th[1], -0.04762, epsilon = 0.025);
        assert_abs_diff_eq!(th[5], 0.03093, epsilon = 0.04);
    }

    #[test]
    fn test_wrong_target_is_rejected() {
        let mut samples = generate_session(&SessionConfig::default());
        let per_target = SessionConfig::default().samples_per_target;
        for i in 0..per_target {
            let idx = 4 * per_target + i;
            add_yaw_bias_deg(&mut samples[idx], 8.0);
        }
        let outcome = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap();
        let report = outcome
            .reports
            .iter()
            .find(|r| r.key == EyeKey::Right)
            .unwrap();
        assert!(
            report.targets_rejected.contains(&4),
            "{:?}",
            report.targets_rejected
        );
        let right = outcome
            .profile
            .correction(RaySource::RgbOnly, EyeKey::Right)
            .unwrap();
        let th = right.theta;
        assert_abs_diff_eq!(deg(th[0]), -1.9048, epsilon = 0.3);
        assert_abs_diff_eq!(deg(th[3]), 1.0309, epsilon = 0.3);
    }

    #[test]
    fn test_verdicts_match_fit_outcome_rejections() {
        let mut samples = generate_session(&SessionConfig::default());
        let per_target = SessionConfig::default().samples_per_target;
        for i in 0..per_target {
            let idx = 4 * per_target + i;
            add_yaw_bias_deg(&mut samples[idx], 8.0);
        }
        let outcome = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap();
        let report = outcome
            .reports
            .iter()
            .find(|r| r.key == EyeKey::Right)
            .unwrap();
        let verdicts = outcome.verdicts();
        for &idx in &report.targets_rejected {
            let v = verdicts.iter().find(|v| v.index == idx).unwrap();
            assert!(v.rejected, "target {idx}");
            assert!(v.reason.is_some(), "target {idx}");
        }
        for &idx in &report.targets_used {
            let v = verdicts.iter().find(|v| v.index == idx).unwrap();
            assert!(!v.rejected, "target {idx}");
            assert!(v.reason.is_none(), "target {idx}");
        }
    }

    #[test]
    fn test_verdict_residual_limit_equals_target_outlier_factor_times_spread() {
        let mut samples = generate_session(&SessionConfig::default());
        let per_target = SessionConfig::default().samples_per_target;
        for i in 0..per_target {
            let idx = 4 * per_target + i;
            add_yaw_bias_deg(&mut samples[idx], 8.0);
        }
        let cfg = FitConfig::default();
        let outcome =
            DotSessionFit::fit_with(&samples, &fixture_rig(), &cfg, ProfileMeta::default())
                .unwrap();
        let verdicts = outcome.verdicts();
        let rejected = verdicts
            .iter()
            .find(|v| v.index == 4)
            .expect("target 4 is rejected");
        let Some(TargetReject::Residual { deg, limit }) = rejected.reason else {
            panic!("expected a Residual reason, got {:?}", rejected.reason);
        };
        assert_eq!(deg, rejected.residual_deg);

        let mut spread: Vec<f64> = verdicts
            .iter()
            .filter(|v| !matches!(v.reason, Some(TargetReject::TooFewSamples { .. })))
            .map(|v| v.residual_deg)
            .collect();
        spread.sort_by(f64::total_cmp);
        let median_rho = median(&mut spread);
        assert_abs_diff_eq!(
            limit,
            (cfg.target_outlier_factor * median_rho).max(3.0),
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_verdicts_merge_takes_reason_from_rejecting_eye_not_accepting_eye() {
        let right_cfg = SessionConfig {
            side: Some(Side::Right),
            bias: UNBIASED,
            seed: 11,
            ..SessionConfig::default()
        };
        let right_samples = generate_session(&right_cfg);

        let left_cfg = SessionConfig {
            side: Some(Side::Left),
            bias: UNBIASED,
            base_xy: (125.0, 60.0),
            seed: 23,
            ..SessionConfig::default()
        };
        let per_target = left_cfg.samples_per_target;
        let starved_target = 4u32;
        let left_samples: Vec<FitSample> = generate_session(&left_cfg)
            .into_iter()
            .enumerate()
            .filter(|(i, _)| {
                let target = i / per_target;
                target != starved_target as usize || i % per_target < 2
            })
            .map(|(_, s)| s)
            .collect();

        let mut samples = right_samples;
        samples.extend(left_samples);

        let outcome = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap();

        let right_report = outcome
            .reports
            .iter()
            .find(|r| r.key == EyeKey::Right)
            .unwrap();
        let left_report = outcome
            .reports
            .iter()
            .find(|r| r.key == EyeKey::Left)
            .unwrap();
        assert!(
            right_report.targets_used.contains(&starved_target),
            "setup: the right eye should accept target {starved_target}: {:?}",
            right_report.targets_used
        );
        assert!(
            left_report.targets_rejected.contains(&starved_target),
            "setup: the left eye should reject target {starved_target}: {:?}",
            left_report.targets_rejected
        );

        let merged = outcome.verdicts();
        let v = merged
            .iter()
            .find(|v| v.index == starved_target)
            .expect("a verdict exists for the starved target");
        assert!(v.rejected, "{v:?}");
        assert!(
            matches!(v.reason, Some(TargetReject::TooFewSamples { .. })),
            "a merged rejected verdict must carry a rejecting eye's reason: {v:?}"
        );
    }

    #[test]
    fn test_sample_rejection_reports_too_few_samples_not_jitter() {
        let samples = generate_session(&SessionConfig::default());
        let per_target = SessionConfig::default().samples_per_target;
        let kept_for_target4 = 2;
        let samples: Vec<FitSample> = samples
            .into_iter()
            .enumerate()
            .filter(|(i, _)| {
                let target = i / per_target;
                target != 4 || i % per_target < kept_for_target4
            })
            .map(|(_, s)| s)
            .collect();
        let outcome = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap();
        let verdicts = outcome.verdicts();
        let v = verdicts.iter().find(|v| v.index == 4).unwrap();
        assert!(v.rejected);
        assert!(
            matches!(v.reason, Some(TargetReject::TooFewSamples { have, .. }) if have == kept_for_target4),
            "{:?}",
            v.reason
        );
    }

    #[test]
    fn test_few_targets_fall_back_to_offset_only() {
        let per_target = SessionConfig::default().samples_per_target;
        let samples: Vec<FitSample> = generate_session(&SessionConfig::default())
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i / per_target < 4)
            .map(|(_, s)| s)
            .collect();
        let outcome = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap();
        let right = outcome
            .profile
            .correction(RaySource::RgbOnly, EyeKey::Right)
            .unwrap();
        assert_eq!(right.model, CorrectionModel::OffsetOnly);
        assert_abs_diff_eq!(right.theta[1], 0.0, epsilon = 0.0);
        assert_abs_diff_eq!(right.theta[2], 0.0, epsilon = 0.0);
        assert_abs_diff_eq!(right.theta[4], 0.0, epsilon = 0.0);
        assert_abs_diff_eq!(right.theta[5], 0.0, epsilon = 0.0);
    }

    #[test]
    fn test_two_targets_error() {
        let per_target = SessionConfig::default().samples_per_target;
        let samples: Vec<FitSample> = generate_session(&SessionConfig::default())
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i / per_target < 2)
            .map(|(_, s)| s)
            .collect();
        let err = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            CalibrationError::InsufficientData {
                what: "targets",
                need: 3,
                got: 2,
            }
        ));
    }

    /// Pins the first-gate path, unchanged by the second-gate fix below.
    #[test]
    fn test_fit_reports_first_gate_count_when_too_few_usable() {
        let per_target = SessionConfig::default().samples_per_target;
        let samples: Vec<FitSample> = generate_session(&SessionConfig::default())
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i / per_target < 2)
            .map(|(_, s)| s)
            .collect();
        let err = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "need at least 3 targets, got 2");
    }

    #[test]
    fn test_fit_reports_post_rejection_count_when_robust_rejection_empties_fit() {
        // Keep only targets 0-3 of the 3x3 grid, so after target 0 is starved
        // below min_samples_per_target (TooFewSamples), exactly 3 targets reach
        // the robust pass; the bias on target 1 then empties it to 2.
        let cfg = SessionConfig {
            grid: [3, 3],
            bias: UNBIASED,
            ..SessionConfig::default()
        };
        let per_target = cfg.samples_per_target;
        let starved_target = 0usize;
        let outlier_target = 1usize;
        let samples: Vec<FitSample> = generate_session(&cfg)
            .into_iter()
            .enumerate()
            .filter(|(i, _)| {
                let target = i / per_target;
                (target != starved_target || i % per_target < 3) && target < 4
            })
            .map(|(i, mut s)| {
                if i / per_target == outlier_target {
                    add_yaw_bias_deg(&mut s, 45.0);
                }
                s
            })
            .collect();

        let err = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "need at least 3 targets after robust rejection, got 2"
        );
    }

    #[test]
    fn test_starved_eye_is_omitted_other_eye_fitted() {
        let mut samples = generate_session(&SessionConfig::default());
        let left_cfg = SessionConfig {
            grid: [3, 3],
            samples_per_target: 24,
            side: Some(Side::Left),
            base_xy: (125.0, 60.0),
            sway_mm: 0.0,
            bias: UNBIASED,
            seed: 97,
            noise_deg: 1.0,
            source: RaySource::RgbOnly,
        };
        let per_target = left_cfg.samples_per_target;
        let left_samples: Vec<FitSample> = generate_session(&left_cfg)
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i / per_target < 2)
            .map(|(_, s)| s)
            .collect();
        samples.extend(left_samples);

        let outcome = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap();
        assert!(
            outcome
                .profile
                .correction(RaySource::RgbOnly, EyeKey::Right)
                .is_some()
        );
        assert!(
            !outcome
                .profile
                .correction(RaySource::RgbOnly, EyeKey::Left)
                .is_some()
        );
        let left_report = outcome
            .reports
            .iter()
            .find(|r| r.key == EyeKey::Left)
            .unwrap();
        assert_eq!(left_report.model, None);
        assert!(left_report.targets_used.is_empty());
    }

    #[test]
    fn test_correction_inverts_synthetic_bias() {
        let samples = generate_session(&SessionConfig::default());
        let profile = DotSessionFit::fit(&samples, &fixture_rig()).unwrap();
        let boxed: Box<dyn GazeCorrection> = Box::new(profile);

        type TargetSums = (Vector2<f64>, Vector2<f64>, u32);
        let mut by_target: HashMap<(u64, u64), TargetSums> = HashMap::new();
        let mut sum_signed = Vector2::zeros();
        let mut n_total = 0usize;
        for s in &samples {
            let corrected = boxed.correct(&s.ray);
            let corrected_angles = yaw_pitch_from_direction(&corrected.direction);
            let target_point = Point3::new(s.target_mm.x, s.target_mm.y, 0.0);
            let desired_dir = Unit::new_normalize(target_point - s.ray.origin);
            let desired_angles = yaw_pitch_from_direction(&desired_dir);
            let key = (s.target_mm.x.to_bits(), s.target_mm.y.to_bits());
            let entry = by_target
                .entry(key)
                .or_insert((Vector2::zeros(), Vector2::zeros(), 0));
            entry.0 += corrected_angles;
            entry.1 += desired_angles;
            entry.2 += 1;
            sum_signed += corrected_angles - desired_angles;
            n_total += 1;
        }
        let mut sum_sq = 0.0;
        for (sum_c, sum_d, n) in by_target.values() {
            let mean_c = sum_c / f64::from(*n);
            let mean_d = sum_d / f64::from(*n);
            sum_sq += (mean_c - mean_d).norm_squared();
        }
        let rms_deg = (sum_sq / by_target.len() as f64).sqrt().to_degrees();
        assert!(rms_deg < 0.4, "rms = {rms_deg}");

        let mean_signed = sum_signed / n_total as f64;
        assert!(mean_signed.x.to_degrees().abs() < 0.1, "{mean_signed:?}");
        assert!(mean_signed.y.to_degrees().abs() < 0.1, "{mean_signed:?}");
    }

    #[test]
    fn test_cyclopean_rays_fit_separately() {
        let cfg = SessionConfig {
            side: None,
            ..SessionConfig::default()
        };
        let samples = generate_session(&cfg);
        let outcome = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap();
        assert!(
            outcome
                .profile
                .correction(RaySource::RgbOnly, EyeKey::Cyclopean)
                .is_some()
        );
    }

    #[test]
    fn test_fit_records_rig_fingerprint() {
        let samples = generate_session(&SessionConfig::default());
        let rig = fixture_rig();
        let profile = DotSessionFit::fit(&samples, &rig).unwrap();
        assert_eq!(profile.rig_fingerprint, rig_fingerprint(&rig));
    }

    fn generate_quadratic_session(curvature_yaw: f64, seed: u64) -> Vec<FitSample> {
        let proto = TargetProtocol::new(ProtocolConfig {
            grid: [4, 4],
            ..ProtocolConfig::default()
        })
        .unwrap();
        let screen = screen();
        let origin = Point3::new(185.0, 60.0, -500.0);
        let targets = proto.sequence(0);
        let mut rng = SplitMix64::new(seed);
        let mut out = Vec::new();
        for t in &targets {
            let target_mm = proto.target_mm(t, &screen);
            for _ in 0..24 {
                let true_dir =
                    Unit::new_normalize(Point3::new(target_mm.x, target_mm.y, 0.0) - origin);
                let true_angles = yaw_pitch_from_direction(&true_dir);
                let obs_yaw = true_angles.x
                    + curvature_yaw * true_angles.x * true_angles.x
                    + 0.3_f64.to_radians() * rng.gaussian();
                let obs_pitch = true_angles.y + 0.3_f64.to_radians() * rng.gaussian();
                let direction = direction_from_yaw_pitch(&Vector2::new(obs_yaw, obs_pitch));
                let ray = GazeRay {
                    side: Some(Side::Right),
                    timestamp: eye_core::Timestamp::from_nanos(0),
                    origin,
                    direction,
                    angular_cov: Matrix2::identity() * 1.0_f64.to_radians().powi(2),
                    origin_cov: Matrix3::identity(),
                    head_rotation: None,
                };
                out.push(FitSample {
                    ray,
                    target_mm,
                    source: RaySource::RgbOnly,
                });
            }
        }
        out
    }

    #[test]
    fn test_quadratic_recovers_synthetic_curvature() {
        let curvature_yaw = -0.6;
        let samples = generate_quadratic_session(curvature_yaw, 71);
        let cfg = FitConfig {
            model_override: Some(CorrectionModel::Quadratic),
            ..FitConfig::default()
        };
        let outcome =
            DotSessionFit::fit_with(&samples, &fixture_rig(), &cfg, ProfileMeta::default())
                .unwrap();
        let right = outcome
            .profile
            .correction(RaySource::RgbOnly, EyeKey::Right)
            .unwrap();
        assert_eq!(right.model, CorrectionModel::Quadratic);
        assert_abs_diff_eq!(right.quad[0], -curvature_yaw, epsilon = 0.2);
        let report = outcome
            .reports
            .iter()
            .find(|r| r.key == EyeKey::Right)
            .unwrap();
        assert!(report.rms_after_deg < 1.0, "{}", report.rms_after_deg);
    }

    #[test]
    fn test_quadratic_needs_twelve_targets() {
        let cfg16 = SessionConfig {
            grid: [4, 4],
            ..SessionConfig::default()
        };
        let per_target = cfg16.samples_per_target;
        let all16 = generate_session(&cfg16);

        let samples11: Vec<FitSample> = all16
            .iter()
            .cloned()
            .enumerate()
            .filter(|(i, _)| i / per_target < 11)
            .map(|(_, s)| s)
            .collect();
        let outcome11 = DotSessionFit::fit_with(
            &samples11,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap();
        let right11 = outcome11
            .profile
            .correction(RaySource::RgbOnly, EyeKey::Right)
            .unwrap();
        assert_eq!(right11.model, CorrectionModel::Affine);
        assert_eq!(right11.targets_used, 11);

        let samples12: Vec<FitSample> = all16
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i / per_target < 12)
            .map(|(_, s)| s)
            .collect();
        let outcome12 = DotSessionFit::fit_with(
            &samples12,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap();
        let right12 = outcome12
            .profile
            .correction(RaySource::RgbOnly, EyeKey::Right)
            .unwrap();
        assert_eq!(right12.model, CorrectionModel::Quadratic);
        assert_eq!(right12.targets_used, 12);
    }

    #[test]
    fn test_quadratic_cross_covariance_is_populated() {
        let curvature_yaw = -0.6;
        let samples = generate_quadratic_session(curvature_yaw, 71);
        let cfg = FitConfig {
            model_override: Some(CorrectionModel::Quadratic),
            ..FitConfig::default()
        };
        let outcome =
            DotSessionFit::fit_with(&samples, &fixture_rig(), &cfg, ProfileMeta::default())
                .unwrap();
        let right = outcome
            .profile
            .correction(RaySource::RgbOnly, EyeKey::Right)
            .unwrap();
        let has_nonzero_cross = right
            .quad_cross_cov
            .iter()
            .flatten()
            .any(|v| v.abs() > 1e-12);
        assert!(has_nonzero_cross, "{:?}", right.quad_cross_cov);
    }

    #[test]
    fn test_head_frame_fit_recovers_known_offset_under_rotation() {
        let bias = [0.05, 0.0, 0.0, -0.03, 0.0, 0.0];
        let theta_expected = bias.map(|v| -v);
        let proto = TargetProtocol::new(ProtocolConfig {
            grid: [3, 3],
            ..ProtocolConfig::default()
        })
        .unwrap();
        let screen = screen();
        let origin = Point3::new(185.0, 60.0, -500.0);
        let targets = proto.sequence(0);
        let mut rng = SplitMix64::new(13);
        let mut samples = Vec::new();
        for (ti, t) in targets.iter().enumerate() {
            let target_mm = proto.target_mm(t, &screen);
            let rot_deg = -20.0 + 5.0 * ti as f64;
            let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), rot_deg.to_radians());
            for _ in 0..24 {
                let true_dir =
                    Unit::new_normalize(Point3::new(target_mm.x, target_mm.y, 0.0) - origin);
                let head_angles = yaw_pitch_from_direction(&(rot.inverse() * true_dir));
                let th = SVector::<f64, 6>::from(bias);
                let observed_head = head_angles
                    + design(&head_angles) * th
                    + Vector2::new(
                        1.0_f64.to_radians() * rng.gaussian(),
                        1.0_f64.to_radians() * rng.gaussian(),
                    );
                let direction = rot * direction_from_yaw_pitch(&observed_head);
                let ray = GazeRay {
                    side: Some(Side::Right),
                    timestamp: eye_core::Timestamp::from_nanos(0),
                    origin,
                    direction,
                    angular_cov: Matrix2::identity() * 1.0_f64.to_radians().powi(2),
                    origin_cov: Matrix3::identity(),
                    head_rotation: Some(rot),
                };
                samples.push(FitSample {
                    ray,
                    target_mm,
                    source: RaySource::RgbOnly,
                });
            }
        }

        let cfg = FitConfig {
            model_override: Some(CorrectionModel::HeadFrame),
            ..FitConfig::default()
        };
        let outcome =
            DotSessionFit::fit_with(&samples, &fixture_rig(), &cfg, ProfileMeta::default())
                .unwrap();
        let right = outcome
            .profile
            .correction(RaySource::RgbOnly, EyeKey::Right)
            .unwrap();
        assert_eq!(right.model, CorrectionModel::HeadFrame);
        for (fitted, expected) in right.theta.iter().zip(theta_expected.iter()) {
            assert_abs_diff_eq!(fitted, expected, epsilon = 0.03);
        }

        let profile = outcome.profile.clone();
        let probe_rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 35.0_f64.to_radians());
        let probe_target = proto.target_mm(&targets[0], &screen);
        let true_dir =
            Unit::new_normalize(Point3::new(probe_target.x, probe_target.y, 0.0) - origin);
        let head_angles = yaw_pitch_from_direction(&(probe_rot.inverse() * true_dir));
        let th = SVector::<f64, 6>::from(bias);
        let observed_head = head_angles + design(&head_angles) * th;
        let probe_ray = GazeRay {
            side: Some(Side::Right),
            timestamp: eye_core::Timestamp::from_nanos(0),
            origin,
            direction: probe_rot * direction_from_yaw_pitch(&observed_head),
            angular_cov: Matrix2::identity() * 1.0_f64.to_radians().powi(2),
            origin_cov: Matrix3::identity(),
            head_rotation: Some(probe_rot),
        };
        let corrected = profile.correct(&probe_ray);
        let corrected_angles = yaw_pitch_from_direction(&corrected.direction);
        let desired_angles = yaw_pitch_from_direction(&true_dir);
        let err_deg = (corrected_angles - desired_angles).norm().to_degrees();
        assert!(err_deg < 0.15, "{err_deg}");
    }

    #[test]
    fn test_head_frame_fit_errors_when_no_sample_has_head_pose() {
        let samples = generate_session(&SessionConfig::default());
        assert!(samples.iter().all(|s| s.ray.head_rotation.is_none()));
        let cfg = FitConfig {
            model_override: Some(CorrectionModel::HeadFrame),
            ..FitConfig::default()
        };
        let err = DotSessionFit::fit_with(&samples, &fixture_rig(), &cfg, ProfileMeta::default())
            .unwrap_err();
        assert!(
            matches!(err, CalibrationError::NoHeadPose { samples } if samples == 216),
            "{err:?}"
        );
    }

    fn head_frame_rotation_session() -> (Vec<FitSample>, Vec<Point2<f64>>, Point3<f64>) {
        let bias = [0.05, 0.0, 0.0, -0.03, 0.0, 0.0];
        let proto = TargetProtocol::new(ProtocolConfig {
            grid: [3, 3],
            ..ProtocolConfig::default()
        })
        .unwrap();
        let screen = screen();
        let origin = Point3::new(185.0, 60.0, -500.0);
        let targets = proto.sequence(0);
        let mut rng = SplitMix64::new(13);
        let mut samples = Vec::new();
        let mut target_mms = Vec::new();
        for (ti, t) in targets.iter().enumerate() {
            let target_mm = proto.target_mm(t, &screen);
            target_mms.push(target_mm);
            let rot_deg = -20.0 + 5.0 * ti as f64;
            let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), rot_deg.to_radians());
            for _ in 0..24 {
                let true_dir =
                    Unit::new_normalize(Point3::new(target_mm.x, target_mm.y, 0.0) - origin);
                let head_angles = yaw_pitch_from_direction(&(rot.inverse() * true_dir));
                let th = SVector::<f64, 6>::from(bias);
                let observed_head = head_angles
                    + design(&head_angles) * th
                    + Vector2::new(
                        1.0_f64.to_radians() * rng.gaussian(),
                        1.0_f64.to_radians() * rng.gaussian(),
                    );
                let direction = rot * direction_from_yaw_pitch(&observed_head);
                let ray = GazeRay {
                    side: Some(Side::Right),
                    timestamp: eye_core::Timestamp::from_nanos(0),
                    origin,
                    direction,
                    angular_cov: Matrix2::identity() * 1.0_f64.to_radians().powi(2),
                    origin_cov: Matrix3::identity(),
                    head_rotation: Some(rot),
                };
                samples.push(FitSample {
                    ray,
                    target_mm,
                    source: RaySource::RgbOnly,
                });
            }
        }
        (samples, target_mms, origin)
    }

    #[test]
    fn test_default_fit_picks_head_frame_when_samples_carry_pose() {
        let (samples, _, _) = head_frame_rotation_session();
        let outcome = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap();
        let right = outcome
            .profile
            .correction(RaySource::RgbOnly, EyeKey::Right)
            .unwrap();
        assert_eq!(right.model, CorrectionModel::HeadFrame);
        assert_eq!(right.targets_used, 9);
    }

    #[test]
    fn test_default_fit_keeps_ladder_without_head_pose() {
        let samples = generate_session(&SessionConfig::default());
        assert!(samples.iter().all(|s| s.ray.head_rotation.is_none()));
        let outcome = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap();
        let right = outcome
            .profile
            .correction(RaySource::RgbOnly, EyeKey::Right)
            .unwrap();
        assert_ne!(right.model, CorrectionModel::HeadFrame);
    }

    #[test]
    fn test_explicit_override_beats_head_frame_default() {
        let (samples, _, _) = head_frame_rotation_session();
        let cfg = FitConfig {
            model_override: Some(CorrectionModel::Affine),
            ..FitConfig::default()
        };
        let outcome =
            DotSessionFit::fit_with(&samples, &fixture_rig(), &cfg, ProfileMeta::default())
                .unwrap();
        let right = outcome
            .profile
            .correction(RaySource::RgbOnly, EyeKey::Right)
            .unwrap();
        assert_eq!(right.model, CorrectionModel::Affine);
    }

    #[test]
    fn test_head_frame_fit_drops_samples_without_pose_and_reports() {
        let (samples, target_mms, origin) = head_frame_rotation_session();
        let cfg = FitConfig {
            model_override: Some(CorrectionModel::HeadFrame),
            ..FitConfig::default()
        };

        let outcome_a =
            DotSessionFit::fit_with(&samples, &fixture_rig(), &cfg, ProfileMeta::default())
                .unwrap();
        let theta_a = outcome_a
            .profile
            .correction(RaySource::RgbOnly, EyeKey::Right)
            .unwrap()
            .theta;

        let mut samples_b = samples.clone();
        let off_direction = Unit::new_normalize(Vector3::new(1.0, 1.0, 1.0));
        for target_mm in &target_mms {
            for _ in 0..3 {
                let ray = GazeRay {
                    side: Some(Side::Right),
                    timestamp: eye_core::Timestamp::from_nanos(0),
                    origin,
                    direction: off_direction,
                    angular_cov: Matrix2::identity() * 1.0_f64.to_radians().powi(2),
                    origin_cov: Matrix3::identity(),
                    head_rotation: None,
                };
                samples_b.push(FitSample {
                    ray,
                    target_mm: *target_mm,
                    source: RaySource::RgbOnly,
                });
            }
        }

        let outcome_b =
            DotSessionFit::fit_with(&samples_b, &fixture_rig(), &cfg, ProfileMeta::default())
                .unwrap();
        let theta_b = outcome_b
            .profile
            .correction(RaySource::RgbOnly, EyeKey::Right)
            .unwrap()
            .theta;

        for (a, b) in theta_a.iter().zip(theta_b.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-9);
        }
        assert_eq!(
            outcome_b.reports[0].samples_without_head_pose,
            3 * target_mms.len()
        );
    }

    #[test]
    fn test_eye_fit_report_exposes_loo_target_mean_deg() {
        let samples = generate_session(&SessionConfig::default());
        let (outcome, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            DotSessionFit::fit_with(
                &samples,
                &fixture_rig(),
                &FitConfig::default(),
                ProfileMeta::default(),
            )
            .unwrap()
        });

        let eye_fitted: Vec<_> = records
            .iter()
            .filter(|r| r.message == "eye fitted")
            .collect();
        assert_eq!(eye_fitted.len(), 1);
        let logged_loo = match eye_fitted[0].fields.get("loo_target_mean_deg") {
            Some(eye_log::Value::F64(v)) => *v,
            other => panic!("expected loo_target_mean_deg F64, got {other:?}"),
        };
        assert_eq!(outcome.reports[0].loo_target_mean_deg, logged_loo);
    }

    #[test]
    fn test_logs_eye_fitted_at_info() {
        let samples = generate_session(&SessionConfig::default());
        let (outcome, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            DotSessionFit::fit_with(
                &samples,
                &fixture_rig(),
                &FitConfig::default(),
                ProfileMeta::default(),
            )
            .unwrap()
        });

        let eye_fitted: Vec<_> = records
            .iter()
            .filter(|r| r.message == "eye fitted")
            .collect();
        assert_eq!(eye_fitted.len(), 1);
        let rec = eye_fitted[0];
        assert_eq!(rec.level, eye_log::Level::Info);
        assert_eq!(
            rec.fields.get("source"),
            Some(&eye_log::Value::Str("rgb-only".to_string()))
        );
        assert_eq!(
            rec.fields.get("eye"),
            Some(&eye_log::Value::Str("right".to_string()))
        );
        assert_eq!(
            rec.fields.get("model"),
            Some(&eye_log::Value::Str("Affine".to_string()))
        );
        let targets_used = match rec.fields.get("targets_used") {
            Some(eye_log::Value::U64(v)) => *v,
            other => panic!("expected targets_used U64, got {other:?}"),
        };
        let targets_rejected = match rec.fields.get("targets_rejected") {
            Some(eye_log::Value::U64(v)) => *v,
            other => panic!("expected targets_rejected U64, got {other:?}"),
        };
        assert!(targets_used >= 6, "{targets_used}");
        assert_eq!(targets_used + targets_rejected, 9);
        let rms_before = match rec.fields.get("rms_before_deg") {
            Some(eye_log::Value::F64(v)) => *v,
            other => panic!("expected rms_before_deg F64, got {other:?}"),
        };
        let rms_after = match rec.fields.get("rms_after_deg") {
            Some(eye_log::Value::F64(v)) => *v,
            other => panic!("expected rms_after_deg F64, got {other:?}"),
        };
        assert!(rms_after < rms_before, "{rms_after} vs {rms_before}");

        let profile_fitted: Vec<_> = records
            .iter()
            .filter(|r| r.message == "profile fitted")
            .collect();
        assert_eq!(profile_fitted.len(), 1);
        let prec = profile_fitted[0];
        assert_eq!(prec.level, eye_log::Level::Info);
        assert_eq!(prec.fields.get("sources"), Some(&eye_log::Value::U64(1)));
        assert_eq!(
            prec.fields.get("corrections"),
            Some(&eye_log::Value::U64(1))
        );
        assert_eq!(prec.fields.get("targets"), Some(&eye_log::Value::U64(9)));
        assert_eq!(prec.fields.get("samples"), Some(&eye_log::Value::U64(216)));
        assert!(matches!(
            prec.fields.get("name"),
            Some(eye_log::Value::Str(_))
        ));

        let loo_folds: Vec<_> = records.iter().filter(|r| r.message == "loo fold").collect();
        assert_eq!(loo_folds.len() as u64, targets_used);
        for fold in &loo_folds {
            assert_eq!(fold.level, eye_log::Level::Debug);
            let held_out = match fold.fields.get("held_out") {
                Some(eye_log::Value::U64(v)) => *v,
                other => panic!("expected held_out U64, got {other:?}"),
            };
            assert!(held_out <= 8, "{held_out}");
            match fold.fields.get("residual_deg") {
                Some(eye_log::Value::F64(v)) => assert!(v.is_finite(), "{v}"),
                other => panic!("expected residual_deg F64, got {other:?}"),
            }
        }

        let residual_and_rejected = records
            .iter()
            .filter(|r| r.message == "target residual" || r.message == "target rejected")
            .count();
        assert_eq!(residual_and_rejected, 9);

        let _ = outcome;
    }

    #[test]
    fn test_logs_target_rejected_at_debug() {
        let mut samples = generate_session(&SessionConfig::default());
        let per_target = SessionConfig::default().samples_per_target;
        for i in 0..per_target {
            let idx = 4 * per_target + i;
            add_yaw_bias_deg(&mut samples[idx], 8.0);
        }
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            DotSessionFit::fit_with(
                &samples,
                &fixture_rig(),
                &FitConfig::default(),
                ProfileMeta::default(),
            )
            .unwrap()
        });
        let rejected: Vec<_> = records
            .iter()
            .filter(|r| r.message == "target rejected")
            .collect();
        assert!(!rejected.is_empty());
        for r in &rejected {
            assert_eq!(r.level, eye_log::Level::Debug);
        }
        let huber = rejected
            .iter()
            .find(|r| {
                r.fields.get(field::REASON)
                    == Some(&eye_log::Value::Str("huber_outlier".to_string()))
            })
            .expect("no huber_outlier 'target rejected' record");
        assert_eq!(huber.level, eye_log::Level::Debug);
        let rho = match huber.fields.get("rho") {
            Some(eye_log::Value::F64(v)) => *v,
            other => panic!("expected rho F64, got {other:?}"),
        };
        let median_rho = match huber.fields.get("median_rho") {
            Some(eye_log::Value::F64(v)) => *v,
            other => panic!("expected median_rho F64, got {other:?}"),
        };
        assert!(rho > median_rho * 3.0, "{rho} vs {median_rho}");

        let mut samples = generate_session(&SessionConfig::default());
        for target in 0..9 {
            for i in 0..3 {
                let idx = target * per_target + i;
                add_yaw_bias_deg(&mut samples[idx], 15.0);
            }
        }
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            DotSessionFit::fit_with(
                &samples,
                &fixture_rig(),
                &FitConfig::default(),
                ProfileMeta::default(),
            )
            .unwrap()
        });
        let rejected = records
            .iter()
            .filter(|r| r.message == "target rejected")
            .count();
        let residual = records
            .iter()
            .filter(|r| r.message == "target residual")
            .count();
        assert_eq!(rejected, 0);
        assert_eq!(residual, 9);
    }

    #[test]
    fn test_logs_eye_omitted_at_debug() {
        let mut samples = generate_session(&SessionConfig::default());
        let left_cfg = SessionConfig {
            grid: [3, 3],
            samples_per_target: 24,
            side: Some(Side::Left),
            base_xy: (125.0, 60.0),
            sway_mm: 0.0,
            bias: UNBIASED,
            seed: 97,
            noise_deg: 1.0,
            source: RaySource::RgbOnly,
        };
        let per_target = left_cfg.samples_per_target;
        let left_samples: Vec<FitSample> = generate_session(&left_cfg)
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i / per_target < 2)
            .map(|(_, s)| s)
            .collect();
        samples.extend(left_samples);

        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            DotSessionFit::fit_with(
                &samples,
                &fixture_rig(),
                &FitConfig::default(),
                ProfileMeta::default(),
            )
            .unwrap()
        });

        let rec = records
            .iter()
            .find(|r| r.message == "eye omitted")
            .expect("no 'eye omitted' record");
        assert_eq!(rec.level, eye_log::Level::Debug);
        assert_eq!(
            rec.fields.get("eye"),
            Some(&eye_log::Value::Str("left".to_string()))
        );
        assert_eq!(
            rec.fields.get(field::REASON),
            Some(&eye_log::Value::Str("too_few_targets".to_string()))
        );
        assert_eq!(rec.fields.get("usable"), Some(&eye_log::Value::U64(2)));
        assert_eq!(
            rec.fields.get("min_targets_offset"),
            Some(&eye_log::Value::U64(3))
        );
    }

    #[test]
    fn test_logs_irls_iteration_at_trace() {
        let samples = generate_session(&SessionConfig::default());
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            DotSessionFit::fit_with(
                &samples,
                &fixture_rig(),
                &FitConfig::default(),
                ProfileMeta::default(),
            )
            .unwrap()
        });

        let iterations: Vec<_> = records
            .iter()
            .filter(|r| r.message == "irls iteration")
            .collect();
        assert!(
            !iterations.is_empty() && iterations.len() <= 10,
            "{}",
            iterations.len()
        );

        let mut prev_iter: Option<u64> = None;
        for (k, rec) in iterations.iter().enumerate() {
            assert_eq!(rec.level, eye_log::Level::Trace);
            assert_eq!(
                rec.fields.get("eye"),
                Some(&eye_log::Value::Str("right".to_string()))
            );
            let iteration = match rec.fields.get("iteration") {
                Some(eye_log::Value::U64(v)) => *v,
                other => panic!("expected iteration U64, got {other:?}"),
            };
            if let Some(prev) = prev_iter {
                assert!(iteration > prev, "{iteration} should exceed {prev}");
            } else {
                assert_eq!(iteration, 0);
                assert_eq!(rec.fields.get("step"), Some(&eye_log::Value::F64(0.0)));
            }
            prev_iter = Some(iteration);
            let converged = match rec.fields.get("converged") {
                Some(eye_log::Value::Bool(v)) => *v,
                other => panic!("expected converged Bool, got {other:?}"),
            };
            if converged {
                assert_eq!(k, iterations.len() - 1, "converged before the last record");
            }
        }
    }

    fn pose_ray(
        side: Option<Side>,
        origin: Point3<f64>,
        head_rotation: Option<UnitQuaternion<f64>>,
    ) -> GazeRay {
        GazeRay {
            side,
            timestamp: eye_core::Timestamp::from_nanos(0),
            origin,
            direction: Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
            angular_cov: Matrix2::identity() * 1e-4,
            origin_cov: Matrix3::identity(),
            head_rotation,
        }
    }

    #[test]
    fn test_calibration_pose_mean_origin_and_rotation() {
        let rot = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 10.0_f64.to_radians());
        let origins = [
            Point3::new(145.0, 75.0, -500.0),
            Point3::new(155.0, 85.0, -500.0),
        ];
        let target_mm = Point2::new(150.0, 80.0);
        let mut samples = Vec::new();
        for &origin in &origins {
            samples.push(FitSample {
                ray: pose_ray(Some(Side::Right), origin, Some(rot)),
                target_mm,
                source: RaySource::RgbOnly,
            });
            samples.push(FitSample {
                ray: pose_ray(Some(Side::Right), origin, None),
                target_mm,
                source: RaySource::RgbOnly,
            });
        }

        let pose = calibration_pose(&samples.iter().collect::<Vec<_>>());
        assert_eq!(pose.samples_total, 4);
        assert_eq!(pose.samples_with_head_pose, 2);
        let origin_mm = pose.eye_origin_mm[&EyeKey::Right];
        assert_abs_diff_eq!(origin_mm[0], 150.0, epsilon = 1e-9);
        assert_abs_diff_eq!(origin_mm[1], 80.0, epsilon = 1e-9);
        assert_abs_diff_eq!(origin_mm[2], -500.0, epsilon = 1e-9);
        let coords = pose.head_rotation.expect("pose recorded");
        assert_abs_diff_eq!(coords[0], rot.i, epsilon = 1e-9);
        assert_abs_diff_eq!(coords[1], rot.j, epsilon = 1e-9);
        assert_abs_diff_eq!(coords[2], rot.k, epsilon = 1e-9);
        assert_abs_diff_eq!(coords[3], rot.w, epsilon = 1e-9);
    }

    #[test]
    fn test_calibration_pose_without_head_pose_records_none_rotation() {
        let target_mm = Point2::new(150.0, 80.0);
        let origin = Point3::new(150.0, 80.0, -500.0);
        let samples = [
            FitSample {
                ray: pose_ray(Some(Side::Right), origin, None),
                target_mm,
                source: RaySource::RgbOnly,
            },
            FitSample {
                ray: pose_ray(Some(Side::Right), origin, None),
                target_mm,
                source: RaySource::RgbOnly,
            },
        ];

        let pose = calibration_pose(&samples.iter().collect::<Vec<_>>());
        assert_eq!(pose.head_rotation, None);
        assert_eq!(pose.samples_with_head_pose, 0);
        assert_eq!(pose.samples_total, 2);
    }

    #[test]
    fn test_fit_records_calibration_pose_mean_origin_and_rotation() {
        let rot = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 10.0_f64.to_radians());
        let mut samples = generate_session(&SessionConfig::default());
        for (i, s) in samples.iter_mut().enumerate() {
            if i % 2 == 0 {
                s.ray.head_rotation = Some(rot);
            }
        }
        let posed = samples
            .iter()
            .filter(|s| s.ray.head_rotation.is_some())
            .count();
        let expected_pose = calibration_pose(&samples.iter().collect::<Vec<_>>());

        let meta = ProfileMeta {
            provenance: Provenance {
                session_id: Some("sess-1".to_string()),
                ..Provenance::default()
            },
            ..ProfileMeta::default()
        };
        let outcome = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            meta.clone(),
        )
        .unwrap();

        assert_eq!(
            outcome.profile.calibration_pose,
            Some(expected_pose.clone())
        );
        assert_eq!(expected_pose.samples_total as usize, samples.len());
        assert_eq!(expected_pose.samples_with_head_pose as usize, posed);
        let coords = expected_pose.head_rotation.expect("pose recorded");
        assert_abs_diff_eq!(coords[0], rot.i, epsilon = 1e-9);
        assert_abs_diff_eq!(coords[1], rot.j, epsilon = 1e-9);
        assert_abs_diff_eq!(coords[2], rot.k, epsilon = 1e-9);
        assert_abs_diff_eq!(coords[3], rot.w, epsilon = 1e-9);
        assert_eq!(outcome.profile.provenance, meta.provenance);
    }

    #[test]
    fn test_fit_without_head_pose_records_none_rotation() {
        let samples = generate_session(&SessionConfig::default());
        assert!(samples.iter().all(|s| s.ray.head_rotation.is_none()));

        let outcome = DotSessionFit::fit_with(
            &samples,
            &fixture_rig(),
            &FitConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap();

        let pose = outcome.profile.calibration_pose.expect("pose recorded");
        assert_eq!(pose.head_rotation, None);
        assert_eq!(pose.samples_with_head_pose, 0);
        assert_eq!(pose.samples_total as usize, samples.len());
    }
}
