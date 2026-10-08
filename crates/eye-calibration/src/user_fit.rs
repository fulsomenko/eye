use std::collections::{BTreeMap, HashMap};

use eye_core::{GazeRay, Rig};
use eye_geometry::angles::yaw_pitch_from_direction;
use nalgebra::{Cholesky, Matrix2, Point2, Point3, SMatrix, SVector, Unit, Vector2};

use crate::correction::{AngularCorrection, CorrectionModel, EyeKey, UserProfile, design};
use crate::error::CalibrationError;

#[derive(Debug, Clone)]
pub struct FitSample {
    pub ray: GazeRay,
    pub target_mm: Point2<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FitConfig {
    pub sample_outlier_mads: f64,
    pub min_samples_per_target: usize,
    pub fixation_jitter_deg: f64,
    pub huber_k: f64,
    pub target_outlier_factor: f64,
    pub slope_prior_sigma: f64,
    pub offset_prior_sigma_deg: f64,
    pub min_targets_affine: usize,
    pub min_targets_offset: usize,
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
            offset_prior_sigma_deg: 10.0,
            min_targets_affine: 6,
            min_targets_offset: 3,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ProfileMeta {
    pub name: String,
    pub created_unix_s: u64,
    pub estimator: String,
}

#[derive(Debug, Clone)]
pub struct FitOutcome {
    pub profile: UserProfile,
    pub reports: Vec<EyeFitReport>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EyeFitReport {
    pub key: EyeKey,
    pub model: Option<CorrectionModel>,
    pub targets_used: Vec<u32>,
    pub targets_rejected: Vec<u32>,
    pub rms_before_deg: f64,
    pub rms_after_deg: f64,
    pub loo_rms_deg: f64,
}

#[derive(Debug, Clone)]
struct TargetAgg {
    index: u32,
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

        let mut by_eye_target: HashMap<(EyeKey, u32), Vec<&FitSample>> = HashMap::new();
        for s in samples {
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
        let s_slope = cfg.slope_prior_sigma;
        let prior6 = SMatrix::<f64, 6, 6>::from_diagonal(&SVector::<f64, 6>::from([
            1.0 / (s_off * s_off),
            1.0 / (s_slope * s_slope),
            1.0 / (s_slope * s_slope),
            1.0 / (s_off * s_off),
            1.0 / (s_slope * s_slope),
            1.0 / (s_slope * s_slope),
        ]));

        let mut eyes = BTreeMap::new();
        let mut reports = Vec::new();
        let mut max_usable = 0usize;

        for eye in eyes_present {
            let mut sample_rejected: Vec<u32> = Vec::new();
            let mut usable: Vec<TargetAgg> = Vec::new();

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
                    let o = yaw_pitch_from_direction(&s.ray.direction);
                    let desired_dir = Unit::new_normalize(target_point - s.ray.origin);
                    let d = yaw_pitch_from_direction(&desired_dir);
                    os.push(o);
                    ds.push(d);
                    es.push(o - d);
                }

                let keep = reject_sample_outliers(&es, cfg);
                let n_keep = keep.iter().filter(|&&k| k).count();
                if n_keep < cfg.min_samples_per_target {
                    sample_rejected.push(idx);
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

                usable.push(TargetAgg {
                    index: idx,
                    observed: o_k,
                    desired: d_k,
                    cov_inv,
                });
            }

            max_usable = max_usable.max(usable.len());

            if usable.len() < cfg.min_targets_offset {
                reports.push(EyeFitReport {
                    key: eye,
                    model: None,
                    targets_used: Vec::new(),
                    targets_rejected: sample_rejected,
                    rms_before_deg: 0.0,
                    rms_after_deg: 0.0,
                    loo_rms_deg: 0.0,
                });
                continue;
            }

            let mut weights = vec![1.0; usable.len()];
            let mut theta_prev: Option<SVector<f64, 6>> = None;
            let mut rho = vec![0.0; usable.len()];
            for _ in 0..10 {
                let (theta, _cov, _chi2) = solve_weighted(&usable, &weights, &prior6)
                    .expect("ridge prior keeps the normal matrix positive definite");
                rho = usable
                    .iter()
                    .map(|t| {
                        let r = t.desired - (t.observed + design(&t.observed) * theta);
                        (r.dot(&(t.cov_inv * r))).sqrt()
                    })
                    .collect();
                let converged = theta_prev
                    .map(|p| (theta - p).norm() < 1e-9)
                    .unwrap_or(false);
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
            let mut huber_rejected = Vec::new();
            let mut remaining = Vec::new();
            for (t, &r) in usable.iter().zip(&rho) {
                if r > cfg.target_outlier_factor * median_rho && r > 3.0 {
                    huber_rejected.push(t.index);
                } else {
                    remaining.push(t.clone());
                }
            }

            let mut targets_rejected = sample_rejected.clone();
            targets_rejected.extend(huber_rejected);
            targets_rejected.sort_unstable();

            if remaining.len() < cfg.min_targets_offset {
                reports.push(EyeFitReport {
                    key: eye,
                    model: None,
                    targets_used: Vec::new(),
                    targets_rejected,
                    rms_before_deg: 0.0,
                    rms_after_deg: 0.0,
                    loo_rms_deg: 0.0,
                });
                continue;
            }

            let model = if remaining.len() >= cfg.min_targets_affine {
                CorrectionModel::Affine
            } else {
                CorrectionModel::OffsetOnly
            };

            let ones = vec![1.0; remaining.len()];
            let (theta_final, cov_final_raw, chi2_final, dof) = match model {
                CorrectionModel::Affine => {
                    let (theta, cov, chi2) = solve_weighted(&remaining, &ones, &prior6)
                        .expect("ridge prior keeps the normal matrix positive definite");
                    (theta, cov, chi2, 2.0 * remaining.len() as f64 - 6.0)
                }
                CorrectionModel::OffsetOnly => {
                    let (theta2, cov2, chi2) = fit_offset_only(&remaining, s_off);
                    let theta = SVector::<f64, 6>::from([theta2.x, 0.0, 0.0, theta2.y, 0.0, 0.0]);
                    let mut cov = SMatrix::<f64, 6, 6>::zeros();
                    cov[(0, 0)] = cov2[(0, 0)];
                    cov[(0, 3)] = cov2[(0, 1)];
                    cov[(3, 0)] = cov2[(1, 0)];
                    cov[(3, 3)] = cov2[(1, 1)];
                    (theta, cov, chi2, 2.0 * remaining.len() as f64 - 2.0)
                }
            };
            let scale = if dof > 0.0 {
                (chi2_final / dof).max(1.0)
            } else {
                1.0
            };
            let cov_final = cov_final_raw * scale;

            let mut targets_used: Vec<u32> = remaining.iter().map(|t| t.index).collect();
            targets_used.sort_unstable();

            let rms_before = rms_deg(remaining.iter().map(|t| (t.desired, t.observed)));
            let rms_after = rms_deg(
                remaining
                    .iter()
                    .map(|t| (t.desired, t.observed + design(&t.observed) * theta_final)),
            );

            let loo_rms = loo_rms_deg(&remaining, model, s_off, &prior6);

            eyes.insert(
                eye,
                AngularCorrection {
                    theta: theta_to_array(&theta_final),
                    cov: cov_to_array(&cov_final),
                    model,
                    targets_used: targets_used.len() as u32,
                    rms_after_rad: rms_after.to_radians(),
                },
            );

            reports.push(EyeFitReport {
                key: eye,
                model: Some(model),
                targets_used,
                targets_rejected,
                rms_before_deg: rms_before,
                rms_after_deg: rms_after,
                loo_rms_deg: loo_rms,
            });
        }

        if eyes.is_empty() {
            return Err(CalibrationError::InsufficientData {
                what: "targets",
                need: cfg.min_targets_offset,
                got: max_usable,
            });
        }

        Ok(FitOutcome {
            profile: UserProfile {
                version: 1,
                name: meta.name,
                created_unix_s: meta.created_unix_s,
                rig_fingerprint: crate::correction::rig_fingerprint(rig),
                estimator: meta.estimator,
                eyes,
            },
            reports,
        })
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

fn loo_rms_deg(
    targets: &[TargetAgg],
    model: CorrectionModel,
    s_off: f64,
    prior6: &SMatrix<f64, 6, 6>,
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
            CorrectionModel::Affine => {
                let ones = vec![1.0; rest.len()];
                let (theta, _, _) = solve_weighted(&rest, &ones, prior6)
                    .expect("ridge prior keeps the normal matrix positive definite");
                held_out.observed + design(&held_out.observed) * theta
            }
            CorrectionModel::OffsetOnly => {
                let (theta2, _, _) = fit_offset_only(&rest, s_off);
                held_out.observed + theta2
            }
        };
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
    use eye_core::{CameraId, CameraModel, OutputId, ScreenModel, Side};
    use eye_geometry::angles::direction_from_yaw_pitch;
    use eye_geometry::synth::SplitMix64;
    use nalgebra::Matrix3;

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

    struct SessionConfig {
        grid: [u32; 2],
        samples_per_target: usize,
        side: Option<Side>,
        base_xy: (f64, f64),
        sway_mm: f64,
        bias: Bias,
        seed: u64,
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
                    + 1.0_f64.to_radians() * rng.gaussian();
                let obs_pitch = cfg.bias.gain_p * true_angles.y
                    + cfg.bias.offset_p_deg.to_radians()
                    + 1.0_f64.to_radians() * rng.gaussian();
                let direction = direction_from_yaw_pitch(&Vector2::new(obs_yaw, obs_pitch));
                let ray = GazeRay {
                    side: cfg.side,
                    origin,
                    direction,
                    angular_cov: Matrix2::identity() * 1.0_f64.to_radians().powi(2),
                    origin_cov: Matrix3::identity(),
                };
                out.push(FitSample { ray, target_mm });
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
        let right = outcome.profile.eyes.get(&EyeKey::Right).unwrap();
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
    fn test_recovers_with_4x4() {
        let cfg = SessionConfig {
            grid: [4, 4],
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
        let right = outcome.profile.eyes.get(&EyeKey::Right).unwrap();
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
        let right = outcome.profile.eyes.get(&EyeKey::Right).unwrap();
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
        let right = outcome.profile.eyes.get(&EyeKey::Right).unwrap();
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
        let right = outcome.profile.eyes.get(&EyeKey::Right).unwrap();
        let th = right.theta;
        assert_abs_diff_eq!(deg(th[0]), -1.9048, epsilon = 0.3);
        assert_abs_diff_eq!(deg(th[3]), 1.0309, epsilon = 0.3);
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
        let right = outcome.profile.eyes.get(&EyeKey::Right).unwrap();
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
        assert!(outcome.profile.eyes.contains_key(&EyeKey::Right));
        assert!(!outcome.profile.eyes.contains_key(&EyeKey::Left));
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
        assert!(outcome.profile.eyes.contains_key(&EyeKey::Cyclopean));
    }

    #[test]
    fn test_fit_records_rig_fingerprint() {
        let samples = generate_session(&SessionConfig::default());
        let rig = fixture_rig();
        let profile = DotSessionFit::fit(&samples, &rig).unwrap();
        assert_eq!(profile.rig_fingerprint, rig_fingerprint(&rig));
    }
}
