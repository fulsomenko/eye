//! Fused gaze estimation: RGB head pose + IR pupil, combined by stereo triangulation, the IR
//! pupil placed on the RGB eyeball model, or inverse-covariance fusion of independent rays,
//! selected per eye by the smallest resulting covariance.

use std::collections::HashMap;

use eye_core::log::field;
use eye_core::observation::{SCHEME_IR_PUPIL_PAIR, SCHEME_MEDIAPIPE_478};
use eye_core::stage::{GazeEstimator, StageError};
use eye_core::{
    CameraModel, Ellipse2, EyeObservation, FaceObservation, GazeRay, Measured, Observations, Rig,
    Side, Timestamp,
};
use eye_geometry::angles::{direction_from_yaw_pitch, yaw_pitch_from_direction};
use eye_geometry::eyeball::{
    EyeCentre, EyeParams, Kappa, gaze_ray, gaze_ray_pccr, optical_axis, visual_axis,
};
use eye_geometry::triangulation::{Triangulated, View, triangulate};
use eye_geometry::uncertainty::{block_diag, propagate_fn};
use nalgebra::{Matrix3, Point3, Unit, UnitQuaternion, Vector2, Vector3, Vector6};
use serde::Deserialize;

use crate::EstimateError;
use crate::ir::glint_near;
use crate::ir_pupil::{IrPupilEstimator, IrPupilOptions};
use crate::landmark::{LandmarkEstimator, LandmarkEye, LandmarkFrame, LandmarkOptions};
use crate::log::{side_str, trace_ray};
use crate::options::parse_options;
use crate::pccr::{PccrEstimator, PccrOptions};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IrChainKind {
    Pupil,
    #[default]
    Pccr,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct FusedOptions {
    pub ir: IrChainKind,
    /// Previous and next lit IR frames at most this far apart, ms (R30: 136 ms).
    pub max_bracket_ms: f64,
    /// A single IR observation this close to the RGB one is used as is, ms.
    pub max_skew_ms: f64,
    pub stereo: bool,
    pub landmark: LandmarkOptions,
    pub ir_pupil: IrPupilOptions,
    /// Mahalanobis distance squared above which a candidate is rejected against the reference
    /// (chi-square with 2 degrees of freedom; 9.21 is the 0.99 quantile).
    pub gate_chi2: f64,
    /// EMA weight of the per-source bias and variance tracker.
    pub bias_alpha: f64,
    /// Frames a source must be observed against the reference before its bias is subtracted.
    pub bias_warmup: u32,
    /// An IR glint within this many pixels of the cross-chain pupil selects `gaze_ray_pccr` over the pupil-sphere ray.
    pub max_glint_offset_px: f64,
    /// Extrapolate the landmark pose to the IR timestamp at most this far, ms (R30: 68 ms nominal).
    pub max_pose_extrapolation_ms: f64,
}

impl Default for FusedOptions {
    fn default() -> Self {
        Self {
            ir: IrChainKind::default(),
            max_bracket_ms: 150.0,
            max_skew_ms: 20.0,
            stereo: true,
            landmark: LandmarkOptions::default(),
            ir_pupil: IrPupilOptions::default(),
            gate_chi2: 9.21,
            bias_alpha: 0.05,
            bias_warmup: 10,
            max_glint_offset_px: 3.0,
            max_pose_extrapolation_ms: 100.0,
        }
    }
}

#[derive(Debug)]
enum IrChain {
    Pupil(Box<IrPupilEstimator>),
    Pccr(PccrEstimator),
}

impl IrChain {
    fn estimate(&mut self, obs: &[Observations], rig: &Rig) -> Result<Vec<GazeRay>, EstimateError> {
        match self {
            IrChain::Pupil(e) => e.estimate_rays(obs, rig),
            IrChain::Pccr(e) => e.estimate_rays(obs, rig),
        }
    }
}

/// Which candidate produced an output ray (logged at debug level).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusedSource {
    Stereo,
    IrOnRgbEyeball,
    IrGlintOnRgbEyeball,
    InverseCovariance,
    RgbOnly,
    IrOnly,
}

impl FusedSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stereo => "stereo",
            Self::IrOnRgbEyeball => "ir-on-rgb-eyeball",
            Self::IrGlintOnRgbEyeball => "ir-glint-on-rgb-eyeball",
            Self::InverseCovariance => "inverse-covariance",
            Self::RgbOnly => "rgb-only",
            Self::IrOnly => "ir-only",
        }
    }

    /// `Relative`: the ray's origin is a declared anchor, not a measurement (ir-pupil). Such a
    /// source is never selected over an `Absolute` reference unless it passed the gate; in a
    /// batch with no reference (IR-only) it is selected unchecked, see
    /// `GateDecision::mahalanobis2 == None`. Documentation-only API: `estimate_detailed` keys on
    /// `source == RgbOnly`, not on this method.
    pub fn reference_kind(self) -> ReferenceKind {
        match self {
            Self::IrOnly => ReferenceKind::Relative,
            _ => ReferenceKind::Absolute,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceKind {
    Absolute,
    Relative,
}

/// The outcome of gating one candidate against the side's reference ray.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateDecision {
    pub side: Side,
    pub source: FusedSource,
    /// `None` when no reference existed for the side (IR-only batch).
    pub mahalanobis2: Option<f64>,
    pub accepted: bool,
    /// Bias subtracted before gating, (yaw, pitch) rad; zero before warm-up.
    pub bias_rad: Vector2<f64>,
}

#[derive(Debug, Clone, Copy, Default)]
struct BiasState {
    mean: Vector2<f64>,
    var: Vector2<f64>,
    n: u32,
}

impl BiasState {
    /// First observation seeds the state (`mean = residual`, `var = 0`); afterwards an EMA of
    /// the residual and of its squared deviation from the current mean.
    fn update(&mut self, residual: Vector2<f64>, alpha: f64) {
        if self.n == 0 {
            self.mean = residual;
            self.var = Vector2::zeros();
        } else {
            let d = residual - self.mean;
            self.mean += alpha * d;
            self.var += alpha * (d.component_mul(&d) - self.var);
        }
        self.n += 1;
    }
}

/// `d^2 = delta^T (Sigma_c + Sigma_r)^-1 delta` over (yaw, pitch); `None` if the sum is
/// singular.
pub fn mahalanobis2(candidate: &GazeRay, reference: &GazeRay) -> Option<f64> {
    let delta = yaw_pitch_from_direction(&candidate.direction)
        - yaw_pitch_from_direction(&reference.direction);
    let cov = candidate.angular_cov + reference.angular_cov;
    let inv = cov.try_inverse()?;
    Some((delta.transpose() * inv * delta).x)
}

/// The ray with `bias` subtracted in (yaw, pitch) and `var` added to the diagonal of
/// `angular_cov`.
fn debias(ray: &GazeRay, state: &BiasState) -> GazeRay {
    let angles = yaw_pitch_from_direction(&ray.direction) - state.mean;
    let mut angular_cov = ray.angular_cov;
    angular_cov[(0, 0)] += state.var.x;
    angular_cov[(1, 1)] += state.var.y;
    GazeRay {
        direction: direction_from_yaw_pitch(&angles),
        angular_cov,
        ..ray.clone()
    }
}

/// The pose extrapolated to the IR timestamp, and the raw IR observation measured there, that
/// `cross_chain` falls back to when stereo is disabled or fails.
struct CrossChainIrTime<'a> {
    eye: &'a LandmarkEye,
    frame: &'a LandmarkFrame,
    measured: &'a Observations,
}

/// `fuse_inverse_covariance` with the singular case logged.
fn fuse_or_log(side: Side, rgb: &GazeRay, ir: &GazeRay) -> Option<GazeRay> {
    let fused = fuse_inverse_covariance(rgb, ir);
    if fused.is_none() {
        tracing::debug!(
            { field::REASON } = "singular_cov",
            side = side_str(side),
            "fusion skipped"
        );
    }
    fused
}

#[derive(Debug)]
pub struct FusedEstimator {
    options: FusedOptions,
    params: EyeParams,
    landmark: LandmarkEstimator,
    ir: IrChain,
    /// Latest ir-pupil-pair observation from earlier calls (the previous bracket).
    prev_ir: Option<Observations>,
    /// Most recent landmark head pose and its frame time, lent to IR-only batches within
    /// `max_bracket_ms`.
    last_viewer: Option<(Timestamp, UnitQuaternion<f64>)>,
    /// Most recent landmark frame, extrapolated to the IR timestamp for the cross chain.
    prev_frame: Option<LandmarkFrame>,
    bias: HashMap<(FusedSource, Side), BiasState>,
    decisions: Vec<GateDecision>,
}

impl FusedEstimator {
    pub const NAME: &'static str = "fused";

    pub fn new(options: FusedOptions) -> Self {
        let ir = match options.ir {
            IrChainKind::Pupil => {
                IrChain::Pupil(Box::new(IrPupilEstimator::new(options.ir_pupil.clone())))
            }
            IrChainKind::Pccr => IrChain::Pccr(PccrEstimator::new(PccrOptions {
                ipd_mm: options.ir_pupil.ipd_mm,
                apply_kappa: options.ir_pupil.apply_kappa,
                ..PccrOptions::default()
            })),
        };
        Self {
            landmark: LandmarkEstimator::new(options.landmark.clone()),
            params: EyeParams::default(),
            ir,
            prev_ir: None,
            last_viewer: None,
            prev_frame: None,
            bias: HashMap::new(),
            decisions: Vec::new(),
            options,
        }
    }

    pub fn from_config(table: &toml::Table, _rig: &eye_core::Rig) -> Result<Self, StageError> {
        Ok(Self::new(parse_options(Self::NAME, table)?))
    }

    /// Gate decisions of the last `estimate_detailed` call, grouped per side in the order each
    /// side was first seen among the candidates; within a side, in candidate order, with the
    /// recomputed `InverseCovariance` entry (when present) last.
    pub fn decisions(&self) -> &[GateDecision] {
        &self.decisions
    }

    /// Every candidate per side, before selection (tests and the bench's debug output use it).
    pub(crate) fn candidates(
        &mut self,
        obs: &[Observations],
        rig: &Rig,
    ) -> Result<Vec<(Side, FusedSource, GazeRay)>, EstimateError> {
        let rgb_obs = obs
            .iter()
            .find(|o| matches!(&o.face, Some(f) if f.scheme == SCHEME_MEDIAPIPE_478));
        let mut ir_pairs: Vec<Observations> = obs
            .iter()
            .filter(|o| matches!(&o.face, Some(f) if f.scheme == SCHEME_IR_PUPIL_PAIR))
            .cloned()
            .collect();
        ir_pairs.sort_by_key(|o| o.timestamp);

        let out = match rgb_obs {
            Some(rgb) => self.dual_or_rgb_candidates(rgb, &ir_pairs, rig)?,
            None => self.ir_only_candidates(obs, rig)?,
        };

        if let Some(latest) = ir_pairs.last() {
            self.prev_ir = Some(latest.clone());
        }

        Ok(out)
    }

    fn ir_only_candidates(
        &mut self,
        obs: &[Observations],
        rig: &Rig,
    ) -> Result<Vec<(Side, FusedSource, GazeRay)>, EstimateError> {
        let rays = self.ir.estimate(obs, rig)?;
        let mut logged = false;
        let mut out = Vec::new();
        for ray in rays {
            let Some(side) = ray.side else { continue };
            let lend = self
                .last_viewer
                .filter(|(t, _)| {
                    ray.timestamp.nanos_since(*t).unsigned_abs() as f64 / 1e6
                        <= self.options.max_bracket_ms
                })
                .map(|(_, v)| v);
            if let (Some((t, _)), Some(_)) = (self.last_viewer, lend)
                && !logged
            {
                logged = true;
                let age_ms = ray.timestamp.nanos_since(t) as f64 / 1e6;
                tracing::debug!(
                    { field::REASON } = "pose_lent",
                    age_ms,
                    "ir-only candidate stamped"
                );
            }
            let head_rotation = ray.head_rotation.or(lend);
            out.push((
                side,
                FusedSource::IrOnly,
                GazeRay {
                    head_rotation,
                    ..ray
                },
            ));
        }
        Ok(out)
    }

    fn dual_or_rgb_candidates(
        &mut self,
        rgb: &Observations,
        ir_pairs: &[Observations],
        rig: &Rig,
    ) -> Result<Vec<(Side, FusedSource, GazeRay)>, EstimateError> {
        let aligned = self.aligned_ir(ir_pairs, rgb.timestamp);

        let frame = self.landmark.estimate_frame(rgb, rig)?;
        let lent = frame.as_ref().map(|f| f.viewer);
        if let Some(f) = &frame {
            self.last_viewer = Some((f.timestamp, f.viewer));
        }

        let newest_ir = ir_pairs.last();
        let frame_ir = match (&frame, newest_ir) {
            (Some(f), Some(newest)) => extrapolate_frame(
                self.prev_frame.as_ref(),
                f,
                newest.timestamp,
                self.options.max_pose_extrapolation_ms,
            ),
            _ => None,
        };
        if let Some(f) = &frame {
            self.prev_frame = Some(f.clone());
        }

        if let (IrChain::Pupil(e), Some(f)) = (&mut self.ir, &frame) {
            let right = f.eyes.iter().find(|eye| eye.side == Side::Right);
            let left = f.eyes.iter().find(|eye| eye.side == Side::Left);
            if let (Some(right), Some(left)) = (right, left) {
                e.seed_anchor([right.centre, left.centre], f.viewer);
            }
        }

        let i_rays = match &aligned {
            Some(aligned_obs) => self.ir.estimate(std::slice::from_ref(aligned_obs), rig)?,
            None => self.ir.estimate(ir_pairs, rig)?,
        };

        let mut out = Vec::new();
        for ray in &i_rays {
            if let Some(side) = ray.side {
                out.push((
                    side,
                    FusedSource::IrOnly,
                    GazeRay {
                        head_rotation: ray.head_rotation.or(lent),
                        ..ray.clone()
                    },
                ));
            }
        }

        if let Some(frame) = &frame {
            for eye in &frame.eyes {
                let side = eye.side;
                out.push((side, FusedSource::RgbOnly, eye.ray.clone()));

                let at_ir = match (&frame_ir, newest_ir) {
                    (Some(frame_ir), Some(newest_ir)) => {
                        frame_ir.eyes.iter().find(|e| e.side == side).map(|eye_ir| {
                            CrossChainIrTime {
                                eye: eye_ir,
                                frame: frame_ir,
                                measured: newest_ir,
                            }
                        })
                    }
                    _ => None,
                };
                let cross = if aligned.is_some() || at_ir.is_some() {
                    self.cross_chain(eye, frame, aligned.as_ref(), at_ir, rig)?
                } else {
                    None
                };
                if let Some((x_ray, source)) = cross {
                    trace_ray(source.as_str(), &x_ray);
                    out.push((side, source, x_ray));
                }

                if aligned.is_none() {
                    continue;
                }
                let Some(i_ray) = i_rays.iter().find(|r| r.side == Some(side)) else {
                    continue;
                };
                if let Some(fused) = fuse_or_log(side, &eye.ray, i_ray) {
                    trace_ray(FusedSource::InverseCovariance.as_str(), &fused);
                    out.push((side, FusedSource::InverseCovariance, fused));
                }
            }
        }

        Ok(out)
    }

    /// The IR pupil pair aligned to `t`, by bracket interpolation or, failing that, a single
    /// observation within `max_skew_ms`.
    fn aligned_ir(&self, ir_pairs: &[Observations], t: Timestamp) -> Option<Observations> {
        let mut candidates: Vec<&Observations> =
            self.prev_ir.iter().chain(ir_pairs.iter()).collect();
        candidates.sort_by_key(|o| o.timestamp);

        let prev = candidates
            .iter()
            .filter(|o| o.timestamp <= t)
            .max_by_key(|o| o.timestamp);
        let next = candidates
            .iter()
            .filter(|o| o.timestamp >= t)
            .min_by_key(|o| o.timestamp);
        let bracket_ms = match (prev, next) {
            (Some(prev), Some(next)) => {
                let bracket_ms = next.timestamp.nanos_since(prev.timestamp) as f64 / 1e6;
                if bracket_ms <= self.options.max_bracket_ms
                    && let Some(interpolated) = interpolate_ir(prev, next, t)
                {
                    let offset_ms = t.nanos_since(prev.timestamp) as f64 / 1e6;
                    tracing::debug!(
                        { field::REASON } = "bracket",
                        bracket_ms,
                        offset_ms,
                        "ir aligned"
                    );
                    return Some(interpolated);
                }
                Some(bracket_ms)
            }
            _ => None,
        };

        let nearest = ir_pairs
            .iter()
            .min_by_key(|o| o.timestamp.nanos_since(t).unsigned_abs());
        let nearest_skew_ms =
            nearest.map(|o| o.timestamp.nanos_since(t).unsigned_abs() as f64 / 1e6);
        match (nearest, nearest_skew_ms) {
            (Some(o), Some(skew_ms)) if skew_ms <= self.options.max_skew_ms => {
                tracing::debug!(
                    { field::REASON } = "nearest",
                    skew_ms,
                    bracket_ms,
                    "ir aligned"
                );
                Some(o.clone())
            }
            _ => {
                tracing::debug!(
                    { field::REASON } = "none",
                    bracket_ms,
                    nearest_skew_ms,
                    max_bracket_ms = self.options.max_bracket_ms,
                    max_skew_ms = self.options.max_skew_ms,
                    "ir not aligned"
                );
                None
            }
        }
    }

    fn eye_params(&self) -> EyeParams {
        if self.options.landmark.apply_kappa {
            self.params
        } else {
            EyeParams {
                kappa: Kappa {
                    alpha_rad: 0.0,
                    beta_rad: 0.0,
                },
                ..self.params
            }
        }
    }

    /// Stereo triangulation (at the RGB time, from `eye`/`frame`/`aligned`) when it succeeds,
    /// else the IR pupil placed on the RGB eyeball (at the IR time, from `at_ir`, the pose
    /// extrapolated to when the pupil was actually observed).
    fn cross_chain(
        &self,
        eye: &LandmarkEye,
        frame: &LandmarkFrame,
        aligned: Option<&Observations>,
        at_ir: Option<CrossChainIrTime<'_>>,
        rig: &Rig,
    ) -> Result<Option<(GazeRay, FusedSource)>, EstimateError> {
        let side = eye.side;
        let params = self.eye_params();

        if self.options.stereo {
            let stereo_pupil_px = aligned.and_then(|aligned| {
                aligned
                    .face
                    .as_ref()
                    .and_then(|f| f.eye(side))
                    .and_then(|e| e.pupil)
                    .map(|m| m.map(|ellipse| ellipse.center()))
            });
            if let (Some(pupil_px), Some(aligned)) = (stereo_pupil_px, aligned) {
                let ir_cam = rig
                    .camera(aligned.camera.as_str())
                    .ok_or_else(|| EstimateError::UnknownCamera(aligned.camera.to_string()))?;
                let rgb_cam = rig
                    .camera(frame.camera.as_str())
                    .ok_or_else(|| EstimateError::UnknownCamera(frame.camera.to_string()))?;
                let views = (
                    View {
                        camera: rgb_cam,
                        pixel: eye.iris_px,
                    },
                    View {
                        camera: ir_cam,
                        pixel: pupil_px,
                    },
                );
                match triangulate(&views.0, &views.1) {
                    Err(e) => tracing::debug!(
                        { field::REASON } = "triangulate_failed",
                        side = side_str(side),
                        error = %e,
                        "stereo failed"
                    ),
                    Ok(t) => match stereo_ray(
                        side,
                        &t,
                        eye,
                        &params,
                        &frame.viewer,
                        rgb_cam,
                        frame.timestamp,
                    ) {
                        Some(ray) => return Ok(Some((ray, FusedSource::Stereo))),
                        None => tracing::debug!(
                            { field::REASON } = "stereo_no_root",
                            side = side_str(side),
                            rms_px = t.rms_px,
                            parallax_rad = t.parallax_rad,
                            "stereo failed"
                        ),
                    },
                }
            }
        }

        let Some(CrossChainIrTime {
            eye: eye_ir,
            frame: frame_ir,
            measured: ir_measured,
        }) = at_ir
        else {
            tracing::debug!(
                { field::REASON } = "no_ir_time",
                side = side_str(side),
                "cross chain skipped"
            );
            return Ok(None);
        };

        let Some(pupil_px) = ir_measured
            .face
            .as_ref()
            .and_then(|f| f.eye(side))
            .and_then(|e| e.pupil)
            .map(|m| m.map(|ellipse| ellipse.center()))
        else {
            tracing::debug!(
                { field::REASON } = "no_ir_pupil",
                side = side_str(side),
                "cross chain skipped"
            );
            return Ok(None);
        };
        let ir_cam = rig
            .camera(ir_measured.camera.as_str())
            .ok_or_else(|| EstimateError::UnknownCamera(ir_measured.camera.to_string()))?;

        let glint = ir_measured
            .face
            .as_ref()
            .and_then(|f| f.eye(side))
            .and_then(|e| glint_near(e, side, pupil_px.value(), self.options.max_glint_offset_px));
        let (ray, source) = match glint {
            Some(glint_px) => (
                gaze_ray_pccr(
                    side,
                    &eye_ir.centre,
                    ir_cam,
                    &pupil_px,
                    &glint_px,
                    &params,
                    Some(&frame_ir.viewer),
                    frame_ir.timestamp,
                )?,
                FusedSource::IrGlintOnRgbEyeball,
            ),
            None => (
                gaze_ray(
                    side,
                    &eye_ir.centre,
                    ir_cam,
                    &pupil_px,
                    &params,
                    Some(&frame_ir.viewer),
                    frame_ir.timestamp,
                )?,
                FusedSource::IrOnRgbEyeball,
            ),
        };
        Ok(Some((ray, source)))
    }

    /// The selected candidate per side, at most one per side.
    pub fn estimate_detailed(
        &mut self,
        obs: &[Observations],
        rig: &Rig,
    ) -> Result<Vec<(FusedSource, GazeRay)>, EstimateError> {
        let candidates = self.candidates(obs, rig)?;
        let observations = obs.len() as u64;

        let mut counts: HashMap<Side, u64> = HashMap::new();
        let mut side_order: Vec<Side> = Vec::new();
        let mut by_side: HashMap<Side, Vec<(FusedSource, GazeRay)>> = HashMap::new();
        for (side, source, ray) in candidates {
            *counts.entry(side).or_default() += 1;
            if !by_side.contains_key(&side) {
                side_order.push(side);
            }
            by_side.entry(side).or_default().push((source, ray));
        }

        self.decisions.clear();
        let mut best: HashMap<Side, (FusedSource, GazeRay, f64)> = HashMap::new();
        for side in side_order {
            let raw = by_side.remove(&side).unwrap_or_default();
            let eligible = self.gate_side(side, raw);
            for (source, ray) in eligible {
                let det = ray.angular_cov.determinant();
                let replace = match best.get(&side) {
                    None => true,
                    Some((_, _, d)) => det.total_cmp(d).is_lt(),
                };
                if replace {
                    best.insert(side, (source, ray, det));
                }
            }
        }
        if best.is_empty() {
            tracing::debug!(
                { field::REASON } = "no_candidates",
                observations,
                "no fused ray"
            );
        }
        for decision in &self.decisions {
            tracing::debug!(
                side = side_str(decision.side),
                source = decision.source.as_str(),
                mahalanobis2 = decision.mahalanobis2,
                accepted = decision.accepted,
                bias_yaw_rad = decision.bias_rad.x,
                bias_pitch_rad = decision.bias_rad.y,
                "fused candidate gated"
            );
        }
        let gated_out: HashMap<Side, u64> = {
            let mut m: HashMap<Side, u64> = HashMap::new();
            for decision in &self.decisions {
                if !decision.accepted {
                    *m.entry(decision.side).or_default() += 1;
                }
            }
            m
        };
        let mut out: Vec<(FusedSource, GazeRay)> = best
            .into_iter()
            .map(|(side, (source, ray, det))| {
                tracing::debug!(
                    side = side_str(side),
                    source = source.as_str(),
                    angular_cov_det = det,
                    candidates = counts[&side],
                    gated_out = gated_out.get(&side).copied().unwrap_or(0),
                    "fused ray selected"
                );
                (source, ray)
            })
            .collect();
        out.sort_by_key(|(_, r)| match r.side {
            Some(Side::Right) => 0,
            Some(Side::Left) => 1,
            None => 2,
        });
        Ok(out)
    }

    /// The candidates of one side eligible for selection: the reference (if any) plus every
    /// candidate that passed the gate, with `InverseCovariance` recomputed from the debiased
    /// `IrOnly` ray when that one is accepted. Appends to `self.decisions`.
    fn gate_side(
        &mut self,
        side: Side,
        raw: Vec<(FusedSource, GazeRay)>,
    ) -> Vec<(FusedSource, GazeRay)> {
        let reference = raw
            .iter()
            .find(|(source, _)| *source == FusedSource::RgbOnly)
            .map(|(_, ray)| ray.clone());

        let mut eligible: Vec<(FusedSource, GazeRay)> = Vec::new();
        if let Some(reference) = &reference {
            eligible.push((FusedSource::RgbOnly, reference.clone()));
        }

        let mut accepted_ir: Option<GazeRay> = None;
        let mut ir_bias_mean = Vector2::zeros();
        for (source, ray) in &raw {
            if *source == FusedSource::RgbOnly || *source == FusedSource::InverseCovariance {
                continue;
            }
            let Some(reference) = &reference else {
                self.decisions.push(GateDecision {
                    side,
                    source: *source,
                    mahalanobis2: None,
                    accepted: true,
                    bias_rad: Vector2::zeros(),
                });
                eligible.push((*source, ray.clone()));
                continue;
            };

            let residual = yaw_pitch_from_direction(&ray.direction)
                - yaw_pitch_from_direction(&reference.direction);
            let state = self.bias.entry((*source, side)).or_default();
            state.update(residual, self.options.bias_alpha);
            let warmed_up = state.n > self.options.bias_warmup;
            let bias_rad = if warmed_up {
                state.mean
            } else {
                Vector2::zeros()
            };
            let candidate = if warmed_up {
                debias(ray, state)
            } else {
                ray.clone()
            };
            let d2 = mahalanobis2(&candidate, reference);
            let accepted = d2.is_some_and(|d| d <= self.options.gate_chi2);

            self.decisions.push(GateDecision {
                side,
                source: *source,
                mahalanobis2: d2,
                accepted,
                bias_rad,
            });
            if accepted {
                if *source == FusedSource::IrOnly {
                    accepted_ir = Some(candidate.clone());
                    ir_bias_mean = bias_rad;
                }
                eligible.push((*source, candidate));
            }
        }

        if let (Some(reference), Some(ir)) = (&reference, &accepted_ir)
            && let Some(ic) = fuse_inverse_covariance(reference, ir)
        {
            let d2 = mahalanobis2(&ic, reference);
            self.decisions.push(GateDecision {
                side,
                source: FusedSource::InverseCovariance,
                mahalanobis2: d2,
                accepted: true,
                bias_rad: ir_bias_mean,
            });
            eligible.push((FusedSource::InverseCovariance, ic));
        }

        eligible
    }
}

fn stereo_ray(
    side: Side,
    t: &Triangulated,
    eye: &LandmarkEye,
    params: &EyeParams,
    viewer: &UnitQuaternion<f64>,
    rgb_cam: &CameraModel,
    at: Timestamp,
) -> Option<GazeRay> {
    let r_p = params.rotation_to_pupil_mm;
    let o_rgb = Point3::from(rgb_cam.screen_from_camera.translation.vector);
    let solve = |z: &Vector6<f64>| -> Option<(Point3<f64>, Unit<Vector3<f64>>)> {
        let (p, e) = (Point3::new(z[0], z[1], z[2]), Point3::new(z[3], z[4], z[5]));
        let w = Unit::try_new(e - o_rgb, 1e-12)?;
        let q = p - o_rgb;
        let qw = q.dot(&w);
        let disc = r_p * r_p - (q - w.into_inner() * qw).norm_squared();
        if disc < 0.0 {
            return None;
        }
        let e2 = o_rgb + w.into_inner() * (qw + disc.sqrt());
        let g = optical_axis(&e2, &p)?;
        Some((e2, visual_axis(&g, &params.kappa, side, viewer)))
    };
    let z = Vector6::new(
        t.point.x,
        t.point.y,
        t.point.z,
        eye.centre.position.x,
        eye.centre.position.y,
        eye.centre.position.z,
    );
    let cov = block_diag::<3, 3, 6>(&t.cov, &eye.centre.cov);
    let (angles, angular_cov) = propagate_fn::<2, 6>(
        |z| solve(z).map(|(_, d)| yaw_pitch_from_direction(&d)),
        &z,
        &cov,
    )?;
    let (origin, origin_cov) = propagate_fn::<3, 6>(|z| solve(z).map(|(o, _)| o.coords), &z, &cov)?;
    Some(GazeRay {
        side: Some(side),
        timestamp: at,
        origin: Point3::from(origin),
        direction: direction_from_yaw_pitch(&angles),
        angular_cov,
        origin_cov,
        head_rotation: Some(*viewer),
    })
}

impl GazeEstimator for FusedEstimator {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn estimate(&mut self, obs: &[Observations], rig: &Rig) -> Result<Vec<GazeRay>, StageError> {
        Ok(self
            .estimate_detailed(obs, rig)?
            .into_iter()
            .map(|(_, ray)| ray)
            .collect())
    }
}

/// Inverse-covariance fusion of two INDEPENDENT rays of the same eye; `None` if a covariance is
/// singular.
pub fn fuse_inverse_covariance(a: &GazeRay, b: &GazeRay) -> Option<GazeRay> {
    let (ia, ib) = (a.angular_cov.try_inverse()?, b.angular_cov.try_inverse()?);
    let angular_cov = (ia + ib).try_inverse()?;
    let theta = angular_cov
        * (ia * yaw_pitch_from_direction(&a.direction)
            + ib * yaw_pitch_from_direction(&b.direction));
    let eps = Matrix3::identity() * 1e-9;
    let (oa, ob) = (
        (a.origin_cov + eps).try_inverse()?,
        (b.origin_cov + eps).try_inverse()?,
    );
    let origin_cov = (oa + ob).try_inverse()?;
    let origin = origin_cov * (oa * a.origin.coords + ob * b.origin.coords);
    Some(GazeRay {
        side: a.side,
        timestamp: a.timestamp,
        origin: Point3::from(origin),
        direction: direction_from_yaw_pitch(&theta),
        angular_cov,
        origin_cov,
        head_rotation: a.head_rotation,
    })
}

/// The IR pupil observation linearly interpolated (or extrapolated) to `t`; `None` unless both
/// are `ir-pupil-pair` observations of the same camera with both pupils and `next` is later than
/// `prev`.
pub fn interpolate_ir(
    prev: &Observations,
    next: &Observations,
    t: Timestamp,
) -> Option<Observations> {
    if next.camera != prev.camera {
        return None;
    }
    let prev_face = prev
        .face
        .as_ref()
        .filter(|f| f.scheme == SCHEME_IR_PUPIL_PAIR)?;
    let next_face = next
        .face
        .as_ref()
        .filter(|f| f.scheme == SCHEME_IR_PUPIL_PAIR)?;
    let dt = next.timestamp.nanos_since(prev.timestamp);
    if dt <= 0 {
        return None;
    }
    let w = t.nanos_since(prev.timestamp) as f64 / dt as f64;

    let mut eyes = Vec::with_capacity(2);
    for side in [Side::Right, Side::Left] {
        let a = prev_face.eye(side)?.pupil?;
        let b = next_face.eye(side)?.pupil?;
        let (ac, bc) = (a.value().center(), b.value().center());
        let centre = ac + (bc - ac) * w;
        let sigma = a.sigma().max(b.sigma());
        let pupil = Ellipse2::new(
            centre,
            a.value().semi_major(),
            a.value().semi_minor(),
            a.value().angle(),
        )
        .ok()?;

        let mut eye = EyeObservation::new(side);
        eye.pupil = Some(Measured::new(pupil, sigma).ok()?);

        if let (Some(ag), Some(bg)) = (
            prev_face.eye(side).and_then(|e| e.glints.first()),
            next_face.eye(side).and_then(|e| e.glints.first()),
        ) {
            let (agv, bgv) = (*ag.value(), *bg.value());
            let gc = agv + (bgv - agv) * w;
            let gs = ag.sigma().max(bg.sigma());
            if let Ok(glint) = Measured::new(gc, gs) {
                eye.glints = vec![glint];
            }
        }
        eyes.push(eye);
    }

    let landmarks = eyes
        .iter()
        .map(|e| e.pupil.expect("pupil set above").value().center())
        .collect();

    Some(Observations {
        camera: prev.camera.clone(),
        timestamp: t,
        face: Some(FaceObservation {
            scheme: SCHEME_IR_PUPIL_PAIR,
            landmarks,
            eyes,
        }),
    })
}

/// `cur` extrapolated to `t` by constant velocity (per-eye centre, viewer angular velocity)
/// derived from `prev` to `cur`. Holds the pose (keeps `cur`'s centres, inflates each eye's
/// covariance by `(dt * 0.1 mm/ms)^2` per axis, leaves the viewer unchanged) when `prev` is
/// `None`, when `cur.timestamp <= prev.timestamp`, when `dt = t - cur.timestamp` exceeds
/// `max_ms`, or, per eye, when `prev` lacks that side. `None` when `cur` has no eyes or `t` is
/// before `cur.timestamp`.
pub fn extrapolate_frame(
    prev: Option<&LandmarkFrame>,
    cur: &LandmarkFrame,
    t: Timestamp,
    max_ms: f64,
) -> Option<LandmarkFrame> {
    if cur.eyes.is_empty() || t < cur.timestamp {
        return None;
    }
    let dt_ms = t.nanos_since(cur.timestamp) as f64 / 1e6;

    let velocity_source = prev.filter(|p| cur.timestamp > p.timestamp);
    let dt_rgb_ms = velocity_source.map(|p| cur.timestamp.nanos_since(p.timestamp) as f64 / 1e6);
    let held = dt_rgb_ms.is_none_or(|_| dt_ms > max_ms);

    let (viewer, rate_deg_s) = match (velocity_source, dt_rgb_ms) {
        (Some(p), Some(dt_rgb_ms)) if !held => {
            let w = (cur.viewer * p.viewer.inverse()).scaled_axis() / dt_rgb_ms;
            let extra = UnitQuaternion::from_scaled_axis(w * dt_ms);
            (extra * cur.viewer, w.norm().to_degrees() * 1000.0)
        }
        _ => (cur.viewer, 0.0),
    };

    let held_inflation = Matrix3::identity() * (dt_ms * 0.1).powi(2);
    let mut speed_mm_s = 0.0;
    let eyes = cur
        .eyes
        .iter()
        .map(|eye| {
            let prev_eye = velocity_source
                .filter(|_| !held)
                .and_then(|p| p.eyes.iter().find(|e| e.side == eye.side));
            match prev_eye {
                Some(prev_eye) => {
                    let dt_rgb_ms = dt_rgb_ms.expect("prev_eye implies a velocity source");
                    let v = (eye.centre.position - prev_eye.centre.position) / dt_rgb_ms;
                    speed_mm_s = v.norm() * 1000.0;
                    let position = eye.centre.position + v * dt_ms;
                    let ratio = dt_ms / dt_rgb_ms;
                    let cov =
                        eye.centre.cov + (ratio * ratio) * (eye.centre.cov + prev_eye.centre.cov);
                    LandmarkEye {
                        centre: EyeCentre { position, cov },
                        ..eye.clone()
                    }
                }
                None => LandmarkEye {
                    centre: EyeCentre {
                        position: eye.centre.position,
                        cov: eye.centre.cov + held_inflation,
                    },
                    ..eye.clone()
                },
            }
        })
        .collect();

    tracing::debug!(dt_ms, speed_mm_s, rate_deg_s, held, "pose extrapolated");

    Some(LandmarkFrame {
        timestamp: t,
        eyes,
        viewer,
        ..cur.clone()
    })
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use approx::assert_abs_diff_eq;
    use eye_core::{CameraId, Measured};
    use eye_geometry::eyeball::EyeParams;
    use eye_log::testing::capture_logs;
    use eye_log::{Level, Value};
    use nalgebra::{Isometry3, Matrix2, Matrix6, Point2, Translation3, Vector2, Vector3};
    use proptest::prelude::*;

    use super::*;
    use crate::ir_pupil::Trigger;
    use crate::landmark::LandmarkOptions;
    use crate::testutil::{
        EYE_CENTRES, synthetic_eye_centres, synthetic_ir_observation_at,
        synthetic_pccr_observation, synthetic_pccr_observation_posed, synthetic_rgb_observation,
        test_rig,
    };
    use eye_geometry::pnp::Pose;

    fn frontal_screen_from_head() -> Isometry3<f64> {
        Isometry3::from_parts(
            Translation3::new(155.0, 40.0, -500.0),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI),
        )
    }

    fn fused_options_no_kappa() -> FusedOptions {
        FusedOptions {
            ir: IrChainKind::Pupil,
            landmark: LandmarkOptions {
                apply_kappa: false,
                ..Default::default()
            },
            ir_pupil: IrPupilOptions {
                apply_kappa: false,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn angle_deg(a: &Unit<Vector3<f64>>, b: &Unit<Vector3<f64>>) -> f64 {
        a.dot(b).clamp(-1.0, 1.0).acos().to_degrees()
    }

    fn landmark_frame(
        timestamp_ms: f64,
        right_x_mm: f64,
        left_x_mm: f64,
        cov: Matrix3<f64>,
        yaw_deg: f64,
    ) -> LandmarkFrame {
        let iris_px = Measured::new(Point2::new(0.0, 0.0), 1.0).expect("valid sigma");
        let eye = |side: Side, x_mm: f64| LandmarkEye {
            side,
            centre: EyeCentre {
                position: Point3::new(x_mm, 40.0, -500.0),
                cov,
            },
            iris_px,
            ray: yaw_pitch_ray(0.0, 0.0, Matrix2::identity()),
        };
        LandmarkFrame {
            camera: CameraId::new("rgb"),
            timestamp: Timestamp::from_nanos((timestamp_ms * 1e6) as u64),
            pose: Pose {
                camera_from_object: Isometry3::identity(),
                cov: Matrix6::zeros(),
                rms_px: 0.0,
            },
            viewer: UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw_deg.to_radians()),
            eyes: vec![eye(Side::Right, right_x_mm), eye(Side::Left, left_x_mm)],
        }
    }

    fn yaw_pitch_ray(yaw_deg: f64, pitch_deg: f64, cov: Matrix2<f64>) -> GazeRay {
        let direction =
            direction_from_yaw_pitch(&Vector2::new(yaw_deg.to_radians(), pitch_deg.to_radians()));
        GazeRay {
            side: Some(Side::Right),
            timestamp: Timestamp::from_nanos(0),
            origin: Point3::new(155.0, 40.0, -500.0),
            direction,
            angular_cov: cov,
            origin_cov: Matrix3::zeros(),
            head_rotation: None,
        }
    }

    #[test]
    fn test_fuse_equal_covariances_averages_angles() {
        let cov = Matrix2::identity() * 1e-4;
        let a = yaw_pitch_ray(1.0, 0.0, cov);
        let b = yaw_pitch_ray(-1.0, 0.0, cov);

        let fused = fuse_inverse_covariance(&a, &b).expect("covariances invertible");

        let angles = yaw_pitch_from_direction(&fused.direction);
        assert_abs_diff_eq!(angles.x, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(
            fused.angular_cov,
            Matrix2::identity() * 5e-5,
            epsilon = 1e-15
        );
    }

    #[test]
    fn test_fuse_weights_toward_lower_variance() {
        let sigma_a = 1f64.to_radians();
        let sigma_b = 3f64.to_radians();
        let cov_a = Matrix2::identity() * (sigma_a * sigma_a);
        let cov_b = Matrix2::identity() * (sigma_b * sigma_b);
        let (yaw_a, yaw_b) = (2.0, 7.0);
        let a = yaw_pitch_ray(yaw_a, 0.0, cov_a);
        let b = yaw_pitch_ray(yaw_b, 0.0, cov_b);

        let fused = fuse_inverse_covariance(&a, &b).expect("covariances invertible");

        let angles = yaw_pitch_from_direction(&fused.direction);
        let expected = (9.0 * yaw_a.to_radians() + 1.0 * yaw_b.to_radians()) / 10.0;
        assert_abs_diff_eq!(angles.x, expected, epsilon = 1e-9);
    }

    proptest! {
        #[test]
        fn prop_fused_covariance_not_larger_than_inputs(
            la in prop::array::uniform4(-1.0f64..1.0),
            lb in prop::array::uniform4(-1.0f64..1.0),
            yaw_a in -0.2f64..0.2,
            yaw_b in -0.2f64..0.2,
        ) {
            let spd = |l: [f64; 4]| -> Matrix2<f64> {
                let lm = Matrix2::new(l[0], 0.0, l[1], l[2]);
                lm * lm.transpose() + Matrix2::identity() * 1e-6
            };
            let cov_a = spd(la);
            let cov_b = spd(lb);
            let a = yaw_pitch_ray(yaw_a.to_degrees(), 0.0, cov_a);
            let b = yaw_pitch_ray(yaw_b.to_degrees(), 0.0, cov_b);

            if let Some(fused) = fuse_inverse_covariance(&a, &b) {
                let det_fused = fused.angular_cov.determinant();
                let det_a = cov_a.determinant();
                let det_b = cov_b.determinant();
                prop_assert!(det_fused <= det_a.min(det_b) * (1.0 + 1e-9));
            }
        }
    }

    fn ir_pupil_observation(
        timestamp_ms: f64,
        right: (f64, f64),
        right_sigma: f64,
        left: (f64, f64),
        left_sigma: f64,
        right_glint: Option<(f64, f64)>,
        left_glint: Option<(f64, f64)>,
    ) -> Observations {
        let mut right_eye = EyeObservation::new(Side::Right);
        right_eye.pupil = Some(
            Measured::new(
                Ellipse2::circle(Point2::new(right.0, right.1), 3.0).expect("valid ellipse"),
                right_sigma,
            )
            .expect("valid sigma"),
        );
        if let Some((x, y)) = right_glint {
            right_eye.glints =
                vec![Measured::new(Point2::new(x, y), right_sigma).expect("valid sigma")];
        }
        let mut left_eye = EyeObservation::new(Side::Left);
        left_eye.pupil = Some(
            Measured::new(
                Ellipse2::circle(Point2::new(left.0, left.1), 3.0).expect("valid ellipse"),
                left_sigma,
            )
            .expect("valid sigma"),
        );
        if let Some((x, y)) = left_glint {
            left_eye.glints =
                vec![Measured::new(Point2::new(x, y), left_sigma).expect("valid sigma")];
        }
        Observations {
            camera: CameraId::new("ir"),
            timestamp: Timestamp::from_nanos((timestamp_ms * 1e6) as u64),
            face: Some(FaceObservation {
                scheme: SCHEME_IR_PUPIL_PAIR,
                landmarks: vec![Point2::new(right.0, right.1), Point2::new(left.0, left.1)],
                eyes: vec![right_eye, left_eye],
            }),
        }
    }

    #[test]
    fn test_interpolate_ir_to_rgb_timestamp() {
        let prev = ir_pupil_observation(
            0.0,
            (300.0, 180.0),
            0.2,
            (360.0, 181.0),
            0.2,
            Some((301.0, 180.0)),
            None,
        );
        let next = ir_pupil_observation(
            136.0,
            (302.0, 181.0),
            0.3,
            (362.0, 182.0),
            0.3,
            Some((303.0, 181.0)),
            None,
        );

        let interpolated = interpolate_ir(&prev, &next, Timestamp::from_nanos(68_000_000))
            .expect("both observations are ir-pupil-pair with both pupils");

        let face = interpolated.face.as_ref().expect("face is present");
        let right = face.eye(Side::Right).expect("right eye present");
        let left = face.eye(Side::Left).expect("left eye present");
        let right_pupil = right.pupil.expect("right pupil present");
        let left_pupil = left.pupil.expect("left pupil present");

        assert_abs_diff_eq!(
            right_pupil.value().center(),
            Point2::new(301.0, 180.5),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            left_pupil.value().center(),
            Point2::new(361.0, 181.5),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(right_pupil.sigma(), 0.3, epsilon = 1e-12);
        assert_abs_diff_eq!(left_pupil.sigma(), 0.3, epsilon = 1e-12);
        assert_eq!(right.glints.len(), 1);
        assert_abs_diff_eq!(
            *right.glints[0].value(),
            Point2::new(302.0, 180.5),
            epsilon = 1e-12
        );
        assert!(left.glints.is_empty());
        assert_eq!(interpolated.timestamp, Timestamp::from_nanos(68_000_000));
    }

    #[test]
    fn test_stereo_corrects_pnp_depth_error() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let scale = 1.06;
        let target = Point2::new(310.0, 170.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, scale, &params);

        let mut rgb =
            synthetic_rgb_observation(&rig, &screen_from_head, scale, target, 0.0, 0.0, 1);
        for eye in rgb.face.as_mut().expect("face present").eyes.iter_mut() {
            eye.iris = eye
                .iris
                .map(|m| Measured::new(m.into_value(), 1.0).expect("valid sigma"));
        }
        let mut ir = synthetic_ir_observation_at(&rig, centres, target, 0.0, 1);
        for eye in ir.face.as_mut().expect("face present").eyes.iter_mut() {
            eye.pupil = eye
                .pupil
                .map(|m| Measured::new(m.into_value(), 0.2).expect("valid sigma"));
        }
        ir.timestamp = rgb.timestamp;

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        let candidates = estimator
            .candidates(&[rgb, ir], &rig)
            .expect("estimate succeeds");

        let truth = |side: Side| -> Unit<Vector3<f64>> {
            let centre = match side {
                Side::Right => centres[0],
                Side::Left => centres[1],
            };
            Unit::new_normalize(Point3::new(target.x, target.y, 0.0) - centre)
        };

        let stereo = candidates
            .iter()
            .find(|(side, source, _)| *side == Side::Right && *source == FusedSource::Stereo)
            .expect("stereo candidate present")
            .2
            .clone();
        let rgb_only = candidates
            .iter()
            .find(|(side, source, _)| *side == Side::Right && *source == FusedSource::RgbOnly)
            .expect("rgb-only candidate present")
            .2
            .clone();

        let stereo_error = angle_deg(&stereo.direction, &truth(Side::Right));
        let rgb_error = angle_deg(&rgb_only.direction, &truth(Side::Right));
        assert!(stereo_error < 0.3, "stereo error {stereo_error} deg");
        assert!(rgb_error > 1.0, "rgb-only error {rgb_error} deg");
    }

    #[test]
    fn test_stereo_fires_when_ir_precedes_rgb_within_skew() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let scale = 1.06;
        let target = Point2::new(310.0, 170.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, scale, &params);

        let mut rgb =
            synthetic_rgb_observation(&rig, &screen_from_head, scale, target, 0.0, 0.0, 1);
        rgb.timestamp = Timestamp::from_nanos(10_000_000);
        for eye in rgb.face.as_mut().expect("face present").eyes.iter_mut() {
            eye.iris = eye
                .iris
                .map(|m| Measured::new(m.into_value(), 1.0).expect("valid sigma"));
        }
        let mut ir = synthetic_ir_observation_at(&rig, centres, target, 0.0, 1);
        for eye in ir.face.as_mut().expect("face present").eyes.iter_mut() {
            eye.pupil = eye
                .pupil
                .map(|m| Measured::new(m.into_value(), 0.2).expect("valid sigma"));
        }
        ir.timestamp = Timestamp::from_nanos(0);

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        let candidates = estimator
            .candidates(&[rgb, ir], &rig)
            .expect("estimate succeeds");

        let truth = Unit::new_normalize(Point3::new(target.x, target.y, 0.0) - centres[0]);
        let stereo = candidates
            .iter()
            .find(|(side, source, _)| *side == Side::Right && *source == FusedSource::Stereo)
            .expect(
                "stereo candidate present even though the IR observation is 10ms older \
                     than the RGB frame",
            )
            .2
            .clone();
        let stereo_error = angle_deg(&stereo.direction, &truth);
        assert!(stereo_error < 0.3, "stereo error {stereo_error} deg");
    }

    #[test]
    fn test_ir_on_rgb_eyeball_used_without_stereo() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 1);
        let mut ir = synthetic_ir_observation_at(&rig, centres, target, 0.0, 1);
        ir.timestamp = rgb.timestamp;

        let mut options = fused_options_no_kappa();
        options.stereo = false;
        let mut estimator = FusedEstimator::new(options);
        let candidates = estimator
            .candidates(&[rgb, ir], &rig)
            .expect("estimate succeeds");

        assert!(
            !candidates
                .iter()
                .any(|(_, source, _)| *source == FusedSource::Stereo)
        );

        for side in [Side::Right, Side::Left] {
            let centre = match side {
                Side::Right => centres[0],
                Side::Left => centres[1],
            };
            let truth = Unit::new_normalize(Point3::new(target.x, target.y, 0.0) - centre);
            let (_, _, found) = candidates
                .iter()
                .find(|(s, source, _)| *s == side && *source == FusedSource::IrOnRgbEyeball)
                .expect("ir-on-rgb-eyeball candidate present");
            let error = angle_deg(&found.direction, &truth);
            assert!(error < 0.3, "side {side:?}: error {error} deg");
        }
    }

    #[test]
    fn test_cross_chain_uses_glint_ray_when_glint_present() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 1);
        let mut ir =
            synthetic_pccr_observation(&rig, target, centres[0] - EYE_CENTRES[0], 0.0, 0.0, 1);
        ir.timestamp = rgb.timestamp;

        let mut options = fused_options_no_kappa();
        options.stereo = false;
        let mut estimator = FusedEstimator::new(options);
        let candidates = estimator
            .candidates(&[rgb, ir], &rig)
            .expect("estimate succeeds");

        for side in [Side::Right, Side::Left] {
            let centre = match side {
                Side::Right => centres[0],
                Side::Left => centres[1],
            };
            let truth = Unit::new_normalize(Point3::new(target.x, target.y, 0.0) - centre);
            let (_, _, found) = candidates
                .iter()
                .find(|(s, source, _)| *s == side && *source == FusedSource::IrGlintOnRgbEyeball)
                .expect("ir-glint-on-rgb-eyeball candidate present");
            let error = angle_deg(&found.direction, &truth);
            assert!(error < 0.05, "side {side:?}: error {error} deg");
        }
    }

    #[test]
    fn test_glint_candidate_is_gated_against_reference() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 1);
        let mut ir =
            synthetic_pccr_observation(&rig, target, centres[0] - EYE_CENTRES[0], 0.0, 0.0, 1);
        ir.timestamp = rgb.timestamp;

        let mut options = fused_options_no_kappa();
        options.stereo = false;
        let mut estimator = FusedEstimator::new(options);
        estimator
            .estimate_detailed(&[rgb, ir], &rig)
            .expect("estimate succeeds");

        for side in [Side::Right, Side::Left] {
            let decision = estimator
                .decisions()
                .iter()
                .find(|d| d.side == side && d.source == FusedSource::IrGlintOnRgbEyeball)
                .expect("ir-glint-on-rgb-eyeball decision present");
            assert!(decision.accepted, "side {side:?}: decision {decision:?}");
            assert!(
                decision.mahalanobis2.is_some(),
                "side {side:?}: decision {decision:?}"
            );
        }
    }

    #[test]
    fn test_cross_chain_glint_ray_is_second_order_in_centre_error() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 1);
        let ir_with_glint =
            synthetic_pccr_observation(&rig, target, centres[0] - EYE_CENTRES[0], 0.0, 0.0, 1);
        let ir_without_glint = synthetic_ir_observation_at(&rig, centres, target, 0.0, 1);

        let mut options = fused_options_no_kappa();
        options.stereo = false;
        let estimator = FusedEstimator::new(options);

        let mut reference = LandmarkEstimator::new(LandmarkOptions {
            apply_kappa: false,
            ..Default::default()
        });
        let frame = reference
            .estimate_frame(&rgb, &rig)
            .expect("estimate succeeds")
            .expect("frame present");
        let true_eye = frame
            .eyes
            .iter()
            .find(|e| e.side == Side::Right)
            .expect("right eye present")
            .clone();
        let mut shifted_eye = true_eye.clone();
        shifted_eye.centre.position.x += 3.0;

        let truth = Unit::new_normalize(Point3::new(target.x, target.y, 0.0) - centres[0]);

        let (glint_ray, glint_source) = estimator
            .cross_chain(
                &shifted_eye,
                &frame,
                Some(&ir_with_glint),
                Some(CrossChainIrTime {
                    eye: &shifted_eye,
                    frame: &frame,
                    measured: &ir_with_glint,
                }),
                &rig,
            )
            .expect("cross chain succeeds")
            .expect("glint candidate present");
        assert_eq!(glint_source, FusedSource::IrGlintOnRgbEyeball);
        let glint_error = angle_deg(&glint_ray.direction, &truth);

        let (pupil_ray, pupil_source) = estimator
            .cross_chain(
                &shifted_eye,
                &frame,
                Some(&ir_without_glint),
                Some(CrossChainIrTime {
                    eye: &shifted_eye,
                    frame: &frame,
                    measured: &ir_without_glint,
                }),
                &rig,
            )
            .expect("cross chain succeeds")
            .expect("pupil-sphere candidate present");
        assert_eq!(pupil_source, FusedSource::IrOnRgbEyeball);
        let pupil_error = angle_deg(&pupil_ray.direction, &truth);

        assert!(
            glint_error < 0.2,
            "glint candidate error {glint_error} deg under a 3 mm centre error"
        );
        assert!(
            pupil_error > 10.0,
            "pupil-sphere candidate error {pupil_error} deg should be first order in the centre \
             error"
        );
    }

    #[test]
    fn test_selection_picks_min_determinant_among_accepted() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 1.0, 3);
        let mut ir = synthetic_ir_observation_at(&rig, centres, target, 0.2, 3);
        ir.timestamp = rgb.timestamp;

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        let candidates = estimator
            .candidates(&[rgb.clone(), ir.clone()], &rig)
            .expect("estimate succeeds");

        for side in [Side::Right, Side::Left] {
            for source in [
                FusedSource::Stereo,
                FusedSource::InverseCovariance,
                FusedSource::RgbOnly,
                FusedSource::IrOnly,
            ] {
                assert!(
                    candidates
                        .iter()
                        .any(|(s, src, _)| *s == side && *src == source),
                    "side {side:?} missing candidate {source:?}"
                );
            }
        }

        let mut estimator2 = FusedEstimator::new(fused_options_no_kappa());
        let selected = estimator2
            .estimate_detailed(&[rgb, ir], &rig)
            .expect("estimate succeeds");

        for decision in estimator2.decisions() {
            assert!(
                decision.accepted,
                "side {:?} source {:?} unexpectedly rejected",
                decision.side, decision.source
            );
        }

        let mut accepted: std::collections::HashSet<(Side, FusedSource)> = [
            (Side::Right, FusedSource::RgbOnly),
            (Side::Left, FusedSource::RgbOnly),
        ]
        .into_iter()
        .collect();
        for decision in estimator2.decisions() {
            if decision.accepted {
                accepted.insert((decision.side, decision.source));
            }
        }

        for (source, selected_ray) in &selected {
            let side = selected_ray.side.expect("per-eye ray");
            let best = candidates
                .iter()
                .filter(|(s, src, _)| *s == side && accepted.contains(&(side, *src)))
                .min_by(|(_, _, a), (_, _, b)| {
                    a.angular_cov
                        .determinant()
                        .total_cmp(&b.angular_cov.determinant())
                })
                .expect("at least one accepted candidate");
            assert_eq!(*source, best.1);
            assert_abs_diff_eq!(
                selected_ray.angular_cov.determinant(),
                best.2.angular_cov.determinant(),
                epsilon = 1e-18
            );
        }
    }

    #[test]
    fn test_rgb_only_frame_passes_landmark_ray_through() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 1);

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        let selected = estimator
            .estimate_detailed(std::slice::from_ref(&rgb), &rig)
            .expect("estimate succeeds");

        let mut reference = LandmarkEstimator::new(LandmarkOptions {
            apply_kappa: false,
            ..Default::default()
        });
        let frame = reference
            .estimate_frame(&rgb, &rig)
            .expect("estimate succeeds")
            .expect("frame recovered");

        assert_eq!(selected.len(), frame.eyes.len());
        for (source, ray) in &selected {
            assert_eq!(*source, FusedSource::RgbOnly);
            let expected = frame
                .eyes
                .iter()
                .find(|e| Some(e.side) == ray.side)
                .expect("matching eye");
            assert_eq!(*ray, expected.ray);
        }
    }

    #[test]
    fn test_ir_only_frame_passes_ir_chain_ray_through() {
        let rig = test_rig();
        let target = Point2::new(155.0, 85.0);
        let ir = synthetic_ir_observation_at(&rig, EYE_CENTRES, target, 0.0, 1);

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        let selected = estimator
            .estimate_detailed(std::slice::from_ref(&ir), &rig)
            .expect("estimate succeeds");

        let mut reference = IrPupilEstimator::new(IrPupilOptions {
            apply_kappa: false,
            ..Default::default()
        });
        let rays = reference
            .estimate_rays(&[ir], &rig)
            .expect("estimate succeeds");

        assert_eq!(selected.len(), rays.len());
        for (source, ray) in &selected {
            assert_eq!(*source, FusedSource::IrOnly);
            let expected = rays
                .iter()
                .find(|r| r.side == ray.side)
                .expect("matching ray");
            assert_eq!(ray, expected);
        }
    }

    #[test]
    fn test_dual_mode_uses_previous_call_as_bracket() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let with_pupil_sigma = |mut obs: Observations| -> Observations {
            for eye in obs.face.as_mut().expect("face present").eyes.iter_mut() {
                eye.pupil = eye
                    .pupil
                    .map(|m| Measured::new(m.into_value(), 0.2).expect("valid sigma"));
            }
            obs
        };

        let mut ir0 = with_pupil_sigma(synthetic_ir_observation_at(&rig, centres, target, 0.0, 1));
        ir0.timestamp = Timestamp::from_nanos(0);
        let mut ir136 =
            with_pupil_sigma(synthetic_ir_observation_at(&rig, centres, target, 0.0, 2));
        ir136.timestamp = Timestamp::from_nanos(136_000_000);
        let mut rgb68 =
            synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 1);
        for eye in rgb68.face.as_mut().expect("face present").eyes.iter_mut() {
            eye.iris = eye
                .iris
                .map(|m| Measured::new(m.into_value(), 1.0).expect("valid sigma"));
        }
        rgb68.timestamp = Timestamp::from_nanos(68_000_000);

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        estimator.candidates(&[ir0], &rig).expect("call 1 succeeds");
        let candidates2 = estimator
            .candidates(&[ir136.clone(), rgb68.clone()], &rig)
            .expect("call 2 succeeds");

        for side in [Side::Right, Side::Left] {
            assert!(
                candidates2
                    .iter()
                    .any(|(s, src, _)| *s == side && *src == FusedSource::Stereo),
                "side {side:?} missing Stereo candidate"
            );
        }
        assert_eq!(
            estimator.prev_ir.as_ref().map(|o| o.timestamp),
            Some(Timestamp::from_nanos(136_000_000))
        );

        let mut fresh = FusedEstimator::new(fused_options_no_kappa());
        let candidates_fresh = fresh
            .candidates(&[ir136, rgb68], &rig)
            .expect("estimate succeeds");
        assert!(
            !candidates_fresh
                .iter()
                .any(|(_, src, _)| *src == FusedSource::Stereo)
        );
    }

    #[test]
    fn test_cross_chain_rays_carry_rgb_time() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let with_pupil_sigma = |mut obs: Observations| -> Observations {
            for eye in obs.face.as_mut().expect("face present").eyes.iter_mut() {
                eye.pupil = eye
                    .pupil
                    .map(|m| Measured::new(m.into_value(), 0.2).expect("valid sigma"));
            }
            obs
        };

        let mut ir0 = with_pupil_sigma(synthetic_ir_observation_at(&rig, centres, target, 0.0, 1));
        ir0.timestamp = Timestamp::from_nanos(0);
        let mut ir136 =
            with_pupil_sigma(synthetic_ir_observation_at(&rig, centres, target, 0.0, 2));
        ir136.timestamp = Timestamp::from_nanos(136_000_000);
        let mut rgb68 =
            synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 1);
        for eye in rgb68.face.as_mut().expect("face present").eyes.iter_mut() {
            eye.iris = eye
                .iris
                .map(|m| Measured::new(m.into_value(), 1.0).expect("valid sigma"));
        }
        rgb68.timestamp = Timestamp::from_nanos(68_000_000);

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        estimator.candidates(&[ir0], &rig).expect("call 1 succeeds");
        let candidates2 = estimator
            .candidates(&[ir136, rgb68], &rig)
            .expect("call 2 succeeds");

        for (_, source, ray) in &candidates2 {
            match source {
                FusedSource::Stereo
                | FusedSource::InverseCovariance
                | FusedSource::RgbOnly
                | FusedSource::IrOnly
                | FusedSource::IrGlintOnRgbEyeball => {
                    assert_eq!(ray.timestamp, Timestamp::from_nanos(68_000_000));
                    assert!(ray.head_rotation.is_some());
                }
                FusedSource::IrOnRgbEyeball => {}
            }
        }
    }

    #[test]
    fn test_ir_only_candidates_borrow_rgb_head_pose_in_dual_mode() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 1.0, 3);
        let mut ir = synthetic_ir_observation_at(&rig, centres, target, 0.2, 3);
        ir.timestamp = rgb.timestamp;

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        let candidates = estimator
            .candidates(&[rgb.clone(), ir], &rig)
            .expect("estimate succeeds");

        let mut reference = LandmarkEstimator::new(LandmarkOptions {
            apply_kappa: false,
            ..Default::default()
        });
        let frame = reference
            .estimate_frame(&rgb, &rig)
            .expect("estimate succeeds")
            .expect("frame recovered");

        let ir_only: Vec<_> = candidates
            .iter()
            .filter(|(_, source, _)| *source == FusedSource::IrOnly)
            .collect();
        assert!(!ir_only.is_empty(), "expected IrOnly candidates");
        for (side, _, ray) in ir_only {
            assert_eq!(
                ray.head_rotation,
                Some(frame.viewer),
                "side {side:?} missing lent head pose"
            );
        }
    }

    #[test]
    fn test_fused_seeds_ir_pupil_from_landmark_centres() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 3);
        let mut ir = synthetic_ir_observation_at(&rig, centres, target, 0.0, 3);
        ir.timestamp = rgb.timestamp;

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        let candidates = estimator
            .candidates(&[rgb.clone(), ir.clone()], &rig)
            .expect("estimate succeeds");

        let mut reference = LandmarkEstimator::new(LandmarkOptions {
            apply_kappa: false,
            ..Default::default()
        });
        let frame = reference
            .estimate_frame(&rgb, &rig)
            .expect("estimate succeeds")
            .expect("frame recovered");

        for side in [Side::Right, Side::Left] {
            let ir_only = candidates
                .iter()
                .find(|(s, src, _)| *s == side && *src == FusedSource::IrOnly)
                .expect("ir-only candidate present")
                .2
                .clone();
            let rgb_eye = frame
                .eyes
                .iter()
                .find(|e| e.side == side)
                .expect("rgb eye present");
            assert_abs_diff_eq!(ir_only.origin, rgb_eye.centre.position, epsilon = 1e-6);
            assert_abs_diff_eq!(ir_only.origin_cov, rgb_eye.centre.cov, epsilon = 1e-9);
            assert_eq!(ir_only.head_rotation, Some(frame.viewer));
        }

        match &estimator.ir {
            IrChain::Pupil(e) => assert_eq!(e.last_reanchor, Some(Trigger::Seeded)),
            IrChain::Pccr(_) => panic!("expected the pupil ir chain"),
        }

        estimator
            .estimate_detailed(&[rgb, ir], &rig)
            .expect("estimate succeeds");
        let ir_decisions: Vec<_> = estimator
            .decisions()
            .iter()
            .filter(|d| d.source == FusedSource::IrOnly)
            .collect();
        assert_eq!(ir_decisions.len(), 2);
        for d in ir_decisions {
            assert!(d.accepted, "{d:?}");
        }
    }

    #[test]
    fn test_ir_only_mode_lends_last_rgb_pose_within_bracket() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let mut ir0 = synthetic_ir_observation_at(&rig, centres, target, 0.0, 1);
        ir0.timestamp = Timestamp::from_nanos(0);
        let mut rgb0 = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 1);
        rgb0.timestamp = Timestamp::from_nanos(0);

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        estimator
            .candidates(&[rgb0.clone(), ir0], &rig)
            .expect("call 1 succeeds");

        let mut ir100 = synthetic_ir_observation_at(&rig, centres, target, 0.0, 2);
        ir100.timestamp = Timestamp::from_nanos(100_000_000);
        let candidates2 = estimator
            .candidates(&[ir100], &rig)
            .expect("call 2 succeeds");

        let mut reference = LandmarkEstimator::new(LandmarkOptions {
            apply_kappa: false,
            ..Default::default()
        });
        let frame = reference
            .estimate_frame(&rgb0, &rig)
            .expect("estimate succeeds")
            .expect("frame recovered");

        assert!(!candidates2.is_empty(), "expected IrOnly candidates");
        for (side, source, ray) in &candidates2 {
            assert_eq!(*source, FusedSource::IrOnly);
            assert_eq!(
                ray.head_rotation,
                Some(frame.viewer),
                "side {side:?} missing lent head pose"
            );
        }
    }

    #[test]
    fn test_ir_only_mode_has_no_pose_beyond_bracket() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let mut ir0 = synthetic_ir_observation_at(&rig, centres, target, 0.0, 1);
        ir0.timestamp = Timestamp::from_nanos(0);
        let mut rgb0 = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 1);
        rgb0.timestamp = Timestamp::from_nanos(0);

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        estimator
            .candidates(&[rgb0, ir0], &rig)
            .expect("call 1 succeeds");

        let mut ir400 = synthetic_ir_observation_at(&rig, centres, target, 0.0, 2);
        ir400.timestamp = Timestamp::from_nanos(400_000_000);
        let candidates2 = estimator
            .candidates(&[ir400], &rig)
            .expect("call 2 succeeds");

        assert!(!candidates2.is_empty(), "expected IrOnly candidates");
        for (side, source, ray) in &candidates2 {
            assert_eq!(*source, FusedSource::IrOnly);
            assert_eq!(
                ray.head_rotation, None,
                "side {side:?} unexpected lent pose"
            );
        }
    }

    #[test]
    fn test_bracket_wider_than_limit_disables_stereo_and_fusion_but_holds_fallback() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let mut ir_prev = synthetic_ir_observation_at(&rig, centres, target, 0.0, 1);
        ir_prev.timestamp = Timestamp::from_nanos(1_000_000_000 - 68_000_000);
        let mut ir_next = synthetic_ir_observation_at(&rig, centres, target, 0.0, 2);
        ir_next.timestamp = Timestamp::from_nanos(1_000_000_000 + 200_000_000);
        let ir_next_timestamp = ir_next.timestamp;
        let mut rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 1);
        rgb.timestamp = Timestamp::from_nanos(1_000_000_000);

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        estimator
            .candidates(&[ir_prev], &rig)
            .expect("call 1 succeeds");
        let candidates2 = estimator
            .candidates(&[ir_next, rgb], &rig)
            .expect("call 2 succeeds");

        assert!(!candidates2.iter().any(|(_, src, _)| matches!(
            src,
            FusedSource::Stereo | FusedSource::InverseCovariance
        )));
        assert!(
            candidates2
                .iter()
                .any(|(_, src, _)| *src == FusedSource::IrOnly)
        );
        let fallback = candidates2
            .iter()
            .find(|(_, src, _)| *src == FusedSource::IrOnRgbEyeball)
            .expect("the pose-extrapolation fallback still fires on a held pose");
        assert_eq!(fallback.2.timestamp, ir_next_timestamp);
    }

    #[test]
    fn test_ir_only_candidates_emitted_when_landmark_frame_fails() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let with_pupil_sigma = |mut obs: Observations| -> Observations {
            for eye in obs.face.as_mut().expect("face present").eyes.iter_mut() {
                eye.pupil = eye
                    .pupil
                    .map(|m| Measured::new(m.into_value(), 0.2).expect("valid sigma"));
            }
            obs
        };

        let mut ir0 = with_pupil_sigma(synthetic_ir_observation_at(&rig, centres, target, 0.0, 1));
        ir0.timestamp = Timestamp::from_nanos(0);
        let mut ir136 =
            with_pupil_sigma(synthetic_ir_observation_at(&rig, centres, target, 0.0, 2));
        ir136.timestamp = Timestamp::from_nanos(136_000_000);
        let mut rgb68 =
            synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 1);
        rgb68.timestamp = Timestamp::from_nanos(68_000_000);
        // Wrong landmark count fails PnP (landmark.rs head_pose), so estimate_frame returns None.
        rgb68.face.as_mut().expect("face present").landmarks.pop();

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        estimator.candidates(&[ir0], &rig).expect("call 1 succeeds");
        let candidates2 = estimator
            .candidates(&[ir136, rgb68], &rig)
            .expect("call 2 succeeds");

        assert!(
            !candidates2
                .iter()
                .any(|(_, src, _)| matches!(src, FusedSource::RgbOnly))
        );
        for side in [Side::Right, Side::Left] {
            assert!(
                candidates2
                    .iter()
                    .any(|(s, src, _)| *s == side && *src == FusedSource::IrOnly),
                "side {side:?} missing IrOnly candidate"
            );
        }
    }

    #[test]
    fn test_ir_only_present_when_one_rgb_eye_missing() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let mut rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 1);
        for eye in rgb.face.as_mut().expect("face present").eyes.iter_mut() {
            eye.iris = eye
                .iris
                .map(|m| Measured::new(m.into_value(), 1.0).expect("valid sigma"));
        }
        // Drop the right eye's iris: landmark.rs's eye() omits that side from frame.eyes.
        rgb.face
            .as_mut()
            .expect("face present")
            .eyes
            .iter_mut()
            .find(|e| e.side == Side::Right)
            .expect("right eye present")
            .iris = None;

        let mut ir = synthetic_ir_observation_at(&rig, centres, target, 0.0, 1);
        for eye in ir.face.as_mut().expect("face present").eyes.iter_mut() {
            eye.pupil = eye
                .pupil
                .map(|m| Measured::new(m.into_value(), 0.2).expect("valid sigma"));
        }
        ir.timestamp = rgb.timestamp;

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        let candidates = estimator
            .candidates(&[rgb, ir], &rig)
            .expect("estimate succeeds");

        assert!(
            !candidates
                .iter()
                .any(|(s, src, _)| *s == Side::Right && *src == FusedSource::RgbOnly)
        );
        assert!(
            candidates
                .iter()
                .any(|(s, src, _)| *s == Side::Right && *src == FusedSource::IrOnly),
            "right eye missing IrOnly candidate"
        );
        assert!(
            candidates
                .iter()
                .any(|(s, src, _)| *s == Side::Left && *src == FusedSource::RgbOnly)
        );
    }

    #[test]
    fn test_single_ir_beyond_skew_disables_cross_chain() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let mut ir = synthetic_ir_observation_at(&rig, centres, target, 0.0, 1);
        for eye in ir.face.as_mut().expect("face present").eyes.iter_mut() {
            eye.pupil = eye
                .pupil
                .map(|m| Measured::new(m.into_value(), 0.2).expect("valid sigma"));
        }
        ir.timestamp = Timestamp::from_nanos(30_000_000);
        let mut rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 1);
        for eye in rgb.face.as_mut().expect("face present").eyes.iter_mut() {
            eye.iris = eye
                .iris
                .map(|m| Measured::new(m.into_value(), 1.0).expect("valid sigma"));
        }

        let mut tight = FusedEstimator::new(FusedOptions {
            max_skew_ms: 20.0,
            ..fused_options_no_kappa()
        });
        let candidates_tight = tight
            .candidates(&[ir.clone(), rgb.clone()], &rig)
            .expect("estimate succeeds");
        assert!(
            !candidates_tight
                .iter()
                .any(|(_, src, _)| matches!(src, FusedSource::Stereo))
        );

        let mut wide = FusedEstimator::new(FusedOptions {
            max_skew_ms: 40.0,
            ..fused_options_no_kappa()
        });
        let candidates_wide = wide
            .candidates(&[ir, rgb], &rig)
            .expect("estimate succeeds");
        assert!(
            candidates_wide
                .iter()
                .any(|(_, src, _)| matches!(src, FusedSource::Stereo))
        );
    }

    #[test]
    fn test_stereo_cov_matches_monte_carlo() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);
        let (iris_sigma, pupil_sigma) = (1.0, 0.2);

        let mut rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 0);
        for eye in rgb.face.as_mut().expect("face present").eyes.iter_mut() {
            eye.iris = eye
                .iris
                .map(|m| Measured::new(m.into_value(), iris_sigma).expect("valid sigma"));
        }
        let mut ir = synthetic_ir_observation_at(&rig, centres, target, 0.0, 0);
        for eye in ir.face.as_mut().expect("face present").eyes.iter_mut() {
            eye.pupil = eye
                .pupil
                .map(|m| Measured::new(m.into_value(), pupil_sigma).expect("valid sigma"));
        }
        ir.timestamp = rgb.timestamp;

        let mut reference = FusedEstimator::new(fused_options_no_kappa());
        let predicted = reference
            .candidates(&[rgb, ir], &rig)
            .expect("estimate succeeds")
            .into_iter()
            .find(|(side, source, _)| *side == Side::Right && *source == FusedSource::Stereo)
            .expect("stereo candidate present")
            .2
            .angular_cov;

        let landmark_sigma = fused_options_no_kappa().landmark.landmark_sigma_px;
        let n = 2000;
        let mut samples = Vec::with_capacity(n);
        for seed in 0..n as u64 {
            let rgb_noisy = synthetic_rgb_observation(
                &rig,
                &screen_from_head,
                1.0,
                target,
                landmark_sigma,
                iris_sigma,
                seed,
            );
            let mut ir_noisy =
                synthetic_ir_observation_at(&rig, centres, target, pupil_sigma, seed);
            ir_noisy.timestamp = rgb_noisy.timestamp;

            let mut estimator = FusedEstimator::new(fused_options_no_kappa());
            let stereo = estimator
                .candidates(&[rgb_noisy, ir_noisy], &rig)
                .expect("estimate succeeds")
                .into_iter()
                .find(|(side, source, _)| *side == Side::Right && *source == FusedSource::Stereo)
                .expect("stereo candidate present")
                .2;
            samples.push(yaw_pitch_from_direction(&stereo.direction));
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
    fn test_outputs_are_per_eye() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 1);

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        let selected = estimator
            .estimate_detailed(&[rgb], &rig)
            .expect("estimate succeeds");

        let mut seen = Vec::new();
        for (_, ray) in &selected {
            assert!(ray.side.is_some());
            assert!(!seen.contains(&ray.side));
            seen.push(ray.side);
        }
    }

    #[test]
    fn test_from_config_nested_tables() {
        let rig = test_rig();
        let mut table = toml::Table::new();
        table.insert("ir".into(), "pupil".into());
        let mut landmark = toml::Table::new();
        landmark.insert("max_iris_miss_mm".into(), 3.0.into());
        table.insert("landmark".into(), landmark.into());

        let estimator = FusedEstimator::from_config(&table, &rig).expect("config parses");
        assert_abs_diff_eq!(
            estimator.options.landmark.max_iris_miss_mm,
            3.0,
            epsilon = 1e-12
        );

        let mut bad_top = table.clone();
        bad_top.insert("bogus".into(), 1.into());
        assert!(matches!(
            FusedEstimator::from_config(&bad_top, &rig),
            Err(StageError::Config(_))
        ));

        let mut bad_nested_landmark = toml::Table::new();
        bad_nested_landmark.insert("bogus".into(), 1.into());
        let mut bad_nested = toml::Table::new();
        bad_nested.insert("landmark".into(), bad_nested_landmark.into());
        assert!(matches!(
            FusedEstimator::from_config(&bad_nested, &rig),
            Err(StageError::Config(_))
        ));
    }

    #[test]
    fn test_ir_chain_default_matches_bench_decision() {
        assert_eq!(
            IrChainKind::default(),
            IrChainKind::Pccr,
            "ir-pccr loto 6.53 deg (168 samples) beats ir loto 11.39 deg (230 samples) \
             with enough samples (168 >= 115), so the default IR chain must be Pccr"
        );
    }

    #[test]
    fn test_logs_ir_aligned_at_debug() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let with_pupil_sigma = |mut obs: Observations| -> Observations {
            for eye in obs.face.as_mut().expect("face present").eyes.iter_mut() {
                eye.pupil = eye
                    .pupil
                    .map(|m| Measured::new(m.into_value(), 0.2).expect("valid sigma"));
            }
            obs
        };

        let mut ir0 = with_pupil_sigma(synthetic_ir_observation_at(&rig, centres, target, 0.0, 1));
        ir0.timestamp = Timestamp::from_nanos(0);
        let mut ir136 =
            with_pupil_sigma(synthetic_ir_observation_at(&rig, centres, target, 0.0, 2));
        ir136.timestamp = Timestamp::from_nanos(136_000_000);
        let rgb_timestamp = Timestamp::from_nanos(68_000_000);

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        estimator.candidates(&[ir0], &rig).expect("call 1 succeeds");

        let (_, logs) = capture_logs(tracing::Level::DEBUG, || {
            estimator.aligned_ir(std::slice::from_ref(&ir136), rgb_timestamp)
        });
        let rec = logs
            .iter()
            .find(|r| r.message == "ir aligned")
            .expect("ir aligned record present");
        assert_eq!(rec.level, Level::Debug);
        assert_eq!(rec.fields[field::REASON], Value::Str("bracket".into()));
        assert_eq!(rec.fields["bracket_ms"], Value::F64(136.0));
        assert_eq!(rec.fields["offset_ms"], Value::F64(68.0));

        let mut ir_single =
            with_pupil_sigma(synthetic_ir_observation_at(&rig, centres, target, 0.0, 1));
        ir_single.timestamp = Timestamp::from_nanos(30_000_000);

        let tight = FusedEstimator::new(FusedOptions {
            max_skew_ms: 20.0,
            ..fused_options_no_kappa()
        });
        let (_, logs_tight) = capture_logs(tracing::Level::DEBUG, || {
            tight.aligned_ir(std::slice::from_ref(&ir_single), Timestamp::from_nanos(0))
        });
        let rec_tight = logs_tight
            .iter()
            .find(|r| r.message == "ir not aligned")
            .expect("ir not aligned record present");
        assert_eq!(rec_tight.level, Level::Debug);
        assert_eq!(rec_tight.fields[field::REASON], Value::Str("none".into()));
        assert_eq!(rec_tight.fields["nearest_skew_ms"], Value::F64(30.0));
        assert_eq!(rec_tight.fields["max_skew_ms"], Value::F64(20.0));
        assert!(!rec_tight.fields.contains_key("bracket_ms"));

        let wide = FusedEstimator::new(FusedOptions {
            max_skew_ms: 40.0,
            ..fused_options_no_kappa()
        });
        let (_, logs_wide) = capture_logs(tracing::Level::DEBUG, || {
            wide.aligned_ir(std::slice::from_ref(&ir_single), Timestamp::from_nanos(0))
        });
        let rec_wide = logs_wide
            .iter()
            .find(|r| r.message == "ir aligned")
            .expect("ir aligned record present");
        assert_eq!(rec_wide.level, Level::Debug);
        assert_eq!(rec_wide.fields[field::REASON], Value::Str("nearest".into()));
        assert_eq!(rec_wide.fields["skew_ms"], Value::F64(30.0));
    }

    #[test]
    fn test_logs_fused_ray_selected_at_debug() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 1.0, 3);
        let mut ir = synthetic_ir_observation_at(&rig, centres, target, 0.2, 3);
        ir.timestamp = rgb.timestamp;

        let mut reference = FusedEstimator::new(fused_options_no_kappa());
        let candidates = reference
            .candidates(&[rgb.clone(), ir.clone()], &rig)
            .expect("estimate succeeds");

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        let (_, logs) = capture_logs(tracing::Level::DEBUG, || {
            estimator
                .estimate_detailed(&[rgb, ir], &rig)
                .expect("estimate succeeds")
        });

        let recs: Vec<_> = logs
            .iter()
            .filter(|r| r.message == "fused ray selected")
            .collect();
        assert_eq!(recs.len(), 2);
        for rec in recs {
            assert_eq!(rec.level, Level::Debug);
            let side = match &rec.fields["side"] {
                Value::Str(s) if s == "right" => Side::Right,
                Value::Str(s) if s == "left" => Side::Left,
                other => panic!("unexpected side {other:?}"),
            };
            assert_eq!(rec.fields["candidates"], Value::U64(4));
            let best = candidates
                .iter()
                .filter(|(s, _, _)| *s == side)
                .min_by(|(_, _, a), (_, _, b)| {
                    a.angular_cov
                        .determinant()
                        .total_cmp(&b.angular_cov.determinant())
                })
                .expect("at least one candidate");
            assert_eq!(rec.fields["source"], Value::Str(best.1.as_str().into()));
            match rec.fields["angular_cov_det"] {
                Value::F64(d) => {
                    assert_abs_diff_eq!(d, best.2.angular_cov.determinant(), epsilon = 1e-18)
                }
                ref other => panic!("expected F64, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_logs_fused_candidates_at_trace() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 1.0, 3);
        let mut ir = synthetic_ir_observation_at(&rig, centres, target, 0.2, 3);
        ir.timestamp = rgb.timestamp;

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        let (_, logs) = capture_logs(tracing::Level::TRACE, || {
            estimator
                .candidates(&[rgb, ir], &rig)
                .expect("estimate succeeds")
        });

        let gaze_rays: Vec<_> = logs.iter().filter(|r| r.message == "gaze ray").collect();
        for rec in &gaze_rays {
            assert_eq!(rec.level, Level::Trace);
        }
        let count_source = |s: &str| {
            gaze_rays
                .iter()
                .filter(|r| r.fields["source"] == Value::Str(s.into()))
                .count()
        };
        assert_eq!(count_source("stereo"), 2);
        assert_eq!(count_source("inverse-covariance"), 2);
        assert_eq!(count_source("ir-pupil"), 2);
        assert_eq!(count_source("landmark"), 2);
        assert_eq!(count_source("rgb-only"), 0);
        assert_eq!(count_source("ir-only"), 0);

        let aligned = logs
            .iter()
            .find(|r| r.message == "ir aligned")
            .expect("ir aligned record present");
        assert_eq!(aligned.level, Level::Debug);
        assert_eq!(aligned.fields[field::REASON], Value::Str("nearest".into()));
        assert_eq!(aligned.fields["skew_ms"], Value::F64(0.0));
        assert_eq!(aligned.fields["bracket_ms"], Value::F64(0.0));
    }

    #[test]
    fn test_logs_no_candidates_at_debug() {
        let rig = test_rig();
        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        let (_, logs) = capture_logs(tracing::Level::DEBUG, || {
            estimator
                .estimate_detailed(&[], &rig)
                .expect("estimate succeeds")
        });
        let rec = logs
            .iter()
            .find(|r| r.message == "no fused ray")
            .expect("no fused ray record present");
        assert_eq!(rec.level, Level::Debug);
        assert_eq!(
            rec.fields[field::REASON],
            Value::Str("no_candidates".into())
        );
        assert_eq!(rec.fields["observations"], Value::U64(0));
        let keys: Vec<&String> = rec.fields.keys().collect();
        assert_eq!(keys, vec!["observations", field::REASON]);
    }

    #[test]
    fn test_logs_fusion_skipped_at_debug() {
        let a = yaw_pitch_ray(0.0, 0.0, Matrix2::identity());
        let b = yaw_pitch_ray(1.0, 0.0, Matrix2::zeros());

        let (fused, logs) =
            capture_logs(tracing::Level::DEBUG, || fuse_or_log(Side::Right, &a, &b));

        assert!(fused.is_none());
        let rec = logs
            .iter()
            .find(|r| r.message == "fusion skipped")
            .expect("fusion skipped record present");
        assert_eq!(rec.level, Level::Debug);
        assert_eq!(rec.fields[field::REASON], Value::Str("singular_cov".into()));
        assert_eq!(rec.fields["side"], Value::Str("right".into()));
    }

    #[test]
    fn test_logs_stereo_failed_at_debug() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 1.0, 3);
        let mut ir = synthetic_ir_observation_at(&rig, centres, target, 0.0, 3);
        ir.timestamp = rgb.timestamp;

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        let (candidates, logs) = capture_logs(tracing::Level::TRACE, || {
            estimator
                .candidates(&[rgb, ir], &rig)
                .expect("estimate succeeds")
        });

        let stereo_failed: Vec<_> = logs
            .iter()
            .filter(|r| r.message == "stereo failed")
            .collect();
        assert_eq!(stereo_failed.len(), 2);
        for rec in &stereo_failed {
            assert_eq!(rec.level, Level::Debug);
            assert_eq!(
                rec.fields[field::REASON],
                Value::Str("triangulate_failed".into())
            );
            match &rec.fields["error"] {
                Value::Str(s) => assert!(s.contains("zero sigma"), "error message: {s}"),
                other => panic!("expected Str, got {other:?}"),
            }
        }
        assert!(
            !candidates
                .iter()
                .any(|(_, src, _)| *src == FusedSource::Stereo)
        );

        let gaze_rays: Vec<_> = logs.iter().filter(|r| r.message == "gaze ray").collect();
        for rec in &gaze_rays {
            assert_eq!(rec.level, Level::Trace);
        }
        let ir_on_rgb = gaze_rays
            .iter()
            .filter(|r| r.fields["source"] == Value::Str("ir-on-rgb-eyeball".into()))
            .count();
        assert_eq!(ir_on_rgb, 2);
    }

    #[test]
    fn test_mahalanobis2_matches_closed_form() {
        let cov_a = Matrix2::new(1e-4, 0.0, 0.0, 4e-4);
        let cov_b = Matrix2::new(3e-4, 0.0, 0.0, 1e-4);
        let reference = yaw_pitch_ray(0.0, 0.0, cov_b);
        let candidate = yaw_pitch_ray(0.02f64.to_degrees(), 0.01f64.to_degrees(), cov_a);

        let d2 = mahalanobis2(&candidate, &reference).expect("covariance sum is invertible");

        assert_abs_diff_eq!(d2, 1.2, epsilon = 1e-9);
    }

    #[test]
    fn test_bias_state_seeds_then_tracks() {
        let mut state = BiasState::default();
        let five_deg = Vector2::new(5f64.to_radians(), 5f64.to_radians());
        state.update(five_deg, 0.05);
        assert_abs_diff_eq!(state.mean, five_deg, epsilon = 1e-12);
        assert_eq!(state.n, 1);

        let seven_deg = Vector2::new(7f64.to_radians(), 7f64.to_radians());
        state.update(seven_deg, 0.05);
        let expected_mean = five_deg + 0.05 * (seven_deg - five_deg);
        let d = seven_deg - five_deg;
        let expected_var = 0.05 * d.component_mul(&d);
        assert_abs_diff_eq!(state.mean, expected_mean, epsilon = 1e-12);
        assert_abs_diff_eq!(state.var, expected_var, epsilon = 1e-12);
        assert_eq!(state.n, 2);
    }

    #[test]
    fn test_gate_rejects_wandering_ir_candidate_and_keeps_it_out_of_selection() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let rgb0 = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 1.0, 3);
        let mut ir0 = synthetic_ir_observation_at(&rig, centres, target, 0.05, 3);
        ir0.timestamp = rgb0.timestamp;

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        estimator
            .estimate_detailed(&[rgb0, ir0], &rig)
            .expect("calibrating call succeeds");

        let rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 1.0, 4);
        let ir_target = Point2::new(target.x + 300.0, target.y);
        let mut ir = synthetic_ir_observation_at(&rig, centres, ir_target, 0.05, 4);
        ir.timestamp = Timestamp::from_nanos(1_000_000);
        let rgb = Observations {
            timestamp: ir.timestamp,
            ..rgb
        };
        let selected = estimator
            .estimate_detailed(&[rgb.clone(), ir], &rig)
            .expect("estimate succeeds");

        for (source, _) in &selected {
            assert!(
                !matches!(source, FusedSource::IrOnly | FusedSource::InverseCovariance),
                "wandering ir candidate contaminated selection via {source:?}"
            );
        }

        let ir_decision = estimator
            .decisions()
            .iter()
            .find(|d| d.source == FusedSource::IrOnly)
            .expect("IrOnly decision present");
        assert!(!ir_decision.accepted);
        assert!(
            ir_decision.mahalanobis2.expect("mahalanobis2 computed") > 9.21,
            "{:?}",
            ir_decision.mahalanobis2
        );

        let mut reference = LandmarkEstimator::new(LandmarkOptions {
            apply_kappa: false,
            ..Default::default()
        });
        let frame = reference
            .estimate_frame(&rgb, &rig)
            .expect("estimate succeeds")
            .expect("frame present");
        for (source, selected_ray) in &selected {
            let side = selected_ray.side.expect("per-eye ray");
            let rgb_ref = frame
                .eyes
                .iter()
                .find(|e| e.side == side)
                .expect("rgb reference for side")
                .ray
                .clone();
            let d2 = mahalanobis2(selected_ray, &rgb_ref).expect("mahalanobis2 computed");
            assert!(
                d2 <= 9.21,
                "side {side:?} source {source:?}: selection pulled away from the RgbOnly \
                 reference by the wandering ir candidate (d2 = {d2})"
            );
        }
    }

    #[test]
    fn test_seeded_ir_covariance_passes_gate_and_dominates_selection() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let rgb0 = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 1.0, 3);
        let mut ir0 = synthetic_ir_observation_at(&rig, centres, target, 0.05, 3);
        ir0.timestamp = rgb0.timestamp;

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        estimator
            .estimate_detailed(&[rgb0, ir0], &rig)
            .expect("calibrating call succeeds");

        let rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 1.0, 4);
        let ir_target = Point2::new(target.x + 90.0, target.y);
        let mut ir = synthetic_ir_observation_at(&rig, centres, ir_target, 0.05, 4);
        ir.timestamp = Timestamp::from_nanos(1_000_000);
        let rgb = Observations {
            timestamp: ir.timestamp,
            ..rgb
        };
        let selected = estimator
            .estimate_detailed(&[rgb.clone(), ir], &rig)
            .expect("estimate succeeds");

        let ir_decision = estimator
            .decisions()
            .iter()
            .find(|d| d.source == FusedSource::IrOnly)
            .expect("IrOnly decision present");
        assert!(ir_decision.accepted, "{ir_decision:?}");
        let ic_decision = estimator
            .decisions()
            .iter()
            .find(|d| d.source == FusedSource::InverseCovariance)
            .expect("InverseCovariance decision present");
        assert!(ic_decision.accepted, "{ic_decision:?}");

        let mut reference = LandmarkEstimator::new(LandmarkOptions {
            apply_kappa: false,
            ..Default::default()
        });
        let frame = reference
            .estimate_frame(&rgb, &rig)
            .expect("estimate succeeds")
            .expect("frame present");
        for (source, selected_ray) in &selected {
            assert_eq!(
                *source,
                FusedSource::InverseCovariance,
                "the seeded IrOnly candidate's landmark covariance is tighter than the declared \
                 ambiguity it replaces, so it now outweighs the RgbOnly reference in the \
                 inverse-covariance fusion instead of losing selection on merit"
            );
            let side = selected_ray.side.expect("per-eye ray");
            let rgb_ref = frame
                .eyes
                .iter()
                .find(|e| e.side == side)
                .expect("rgb reference for side")
                .ray
                .clone();
            let error_deg = angle_deg(&selected_ray.direction, &rgb_ref.direction);
            let (lo, hi) = match side {
                Side::Right => (7.0, 9.0),
                Side::Left => (4.0, 6.0),
            };
            assert!(
                (lo..=hi).contains(&error_deg),
                "side {side:?} source {source:?}: expected the seeded ir candidate to pull the \
                 fused selection away from the RgbOnly reference by roughly the measured error \
                 (error = {error_deg} deg, expected in [{lo}, {hi}])"
            );
        }
    }

    #[test]
    fn test_gate_accepts_consistent_ir_candidate() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 1.0, 3);
        let mut ir = synthetic_ir_observation_at(&rig, centres, target, 0.2, 3);
        ir.timestamp = rgb.timestamp;

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        let selected = estimator
            .estimate_detailed(&[rgb, ir], &rig)
            .expect("estimate succeeds");

        for decision in estimator.decisions() {
            assert!(decision.accepted, "{decision:?} not accepted");
        }
        for (source, _) in &selected {
            assert!(
                matches!(
                    source,
                    FusedSource::Stereo
                        | FusedSource::IrOnRgbEyeball
                        | FusedSource::InverseCovariance
                ),
                "unexpected source {source:?}"
            );
        }
    }

    #[test]
    fn test_bias_warmup_gates_then_debiases() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);
        let ir_target = Point2::new(target.x + 500.0 * 35f64.to_radians().tan(), target.y);

        let calibrate = |estimator: &mut FusedEstimator| {
            let rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 1.0, 0);
            let mut ir = synthetic_ir_observation_at(&rig, centres, target, 0.2, 1000);
            ir.timestamp = rgb.timestamp;
            estimator
                .candidates(&[rgb, ir], &rig)
                .expect("calibrating call succeeds");
        };

        let mut probe = FusedEstimator::new(fused_options_no_kappa());
        calibrate(&mut probe);
        let probe_rgb = Observations {
            timestamp: Timestamp::from_nanos(1_000_000),
            ..synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 1.0, 0)
        };
        let mut probe_ir = synthetic_ir_observation_at(&rig, centres, ir_target, 0.2, 1000);
        probe_ir.timestamp = probe_rgb.timestamp;
        let probe_candidates = probe
            .candidates(&[probe_rgb.clone(), probe_ir.clone()], &rig)
            .expect("estimate succeeds");
        let probe_reference = probe_candidates
            .iter()
            .find(|(s, src, _)| *s == Side::Right && *src == FusedSource::RgbOnly)
            .expect("reference present")
            .2
            .clone();
        let probe_ir_ray = probe_candidates
            .iter()
            .find(|(s, src, _)| *s == Side::Right && *src == FusedSource::IrOnly)
            .expect("ir candidate present")
            .2
            .clone();
        let expected_offset = yaw_pitch_from_direction(&probe_ir_ray.direction)
            - yaw_pitch_from_direction(&probe_reference.direction);

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        calibrate(&mut estimator);
        let mut right_decisions = Vec::new();
        for i in 0..40u64 {
            let mut rgb =
                synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 1.0, i);
            let mut ir = synthetic_ir_observation_at(&rig, centres, ir_target, 0.2, i + 1000);
            rgb.timestamp = Timestamp::from_nanos((i + 1) * 1_000_000);
            ir.timestamp = rgb.timestamp;
            estimator
                .estimate_detailed(&[rgb, ir], &rig)
                .expect("estimate succeeds");
            let decision = *estimator
                .decisions()
                .iter()
                .find(|d| d.source == FusedSource::IrOnly && d.side == Side::Right)
                .expect("IrOnly decision present");
            right_decisions.push(decision);
        }

        for (i, d) in right_decisions.iter().take(10).enumerate() {
            assert_eq!(d.bias_rad, Vector2::zeros(), "decision {}", i + 1);
            assert!(!d.accepted, "decision {} unexpectedly accepted", i + 1);
        }
        for (i, d) in right_decisions.iter().skip(10).enumerate() {
            assert!(
                (d.bias_rad.x - expected_offset.x).abs() <= 0.15 * expected_offset.x.abs(),
                "decision {}: bias {} expected {}",
                i + 11,
                d.bias_rad.x,
                expected_offset.x
            );
            assert!(d.accepted, "decision {} not accepted", i + 11);
        }
    }

    #[test]
    fn test_selected_ray_tracks_reference_after_debias() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);
        let ir_target = Point2::new(target.x + 500.0 * 12f64.to_radians().tan(), target.y);

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        let calibrate_rgb =
            synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.05, 0);
        let mut calibrate_ir = synthetic_ir_observation_at(&rig, centres, target, 0.02, 1000);
        calibrate_ir.timestamp = calibrate_rgb.timestamp;
        estimator
            .candidates(&[calibrate_rgb, calibrate_ir], &rig)
            .expect("calibrating call succeeds");

        let mut selected = Vec::new();
        let mut last_rgb = None;
        let mut last_ir = None;
        for i in 0..40u64 {
            let rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.05, i);
            let mut ir = synthetic_ir_observation_at(&rig, centres, ir_target, 0.02, i + 1000);
            ir.timestamp = Timestamp::from_nanos((i + 1) * 1_000_000);
            let rgb = Observations {
                timestamp: ir.timestamp,
                ..rgb
            };
            selected = estimator
                .estimate_detailed(&[rgb.clone(), ir.clone()], &rig)
                .expect("estimate succeeds");
            last_rgb = Some(rgb);
            last_ir = Some(ir);
        }

        let ic_decision = estimator
            .decisions()
            .iter()
            .find(|d| d.side == Side::Right && d.source == FusedSource::InverseCovariance)
            .expect("InverseCovariance decision present");
        assert!(ic_decision.accepted);

        let mut probe = FusedEstimator::new(fused_options_no_kappa());
        let probe_candidates = probe
            .candidates(
                &[last_rgb.expect("ran once"), last_ir.expect("ran once")],
                &rig,
            )
            .expect("estimate succeeds");
        let reference_direction = probe_candidates
            .iter()
            .find(|(s, src, _)| *s == Side::Right && *src == FusedSource::RgbOnly)
            .expect("reference present")
            .2
            .direction;

        let (_, right_ray) = selected
            .iter()
            .find(|(_, ray)| ray.side == Some(Side::Right))
            .expect("right ray present");
        let error = angle_deg(&right_ray.direction, &reference_direction);
        assert!(error < 0.5, "error {error} deg");
    }

    #[test]
    fn test_ir_only_batch_has_no_reference_and_keeps_min_det() {
        let rig = test_rig();
        let target = Point2::new(155.0, 85.0);
        let ir = synthetic_ir_observation_at(&rig, EYE_CENTRES, target, 0.0, 1);

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        let selected = estimator
            .estimate_detailed(std::slice::from_ref(&ir), &rig)
            .expect("estimate succeeds");

        assert!(!estimator.decisions().is_empty());
        assert!(
            estimator
                .decisions()
                .iter()
                .all(|d| d.mahalanobis2.is_none())
        );

        let mut reference = IrPupilEstimator::new(IrPupilOptions {
            apply_kappa: false,
            ..Default::default()
        });
        let rays = reference
            .estimate_rays(&[ir], &rig)
            .expect("estimate succeeds");

        assert_eq!(selected.len(), rays.len());
        for (source, ray) in &selected {
            assert_eq!(*source, FusedSource::IrOnly);
            let expected = rays
                .iter()
                .find(|r| r.side == ray.side)
                .expect("matching ray");
            assert_eq!(ray, expected);
        }
    }

    #[test]
    fn test_from_config_parses_gate_options() {
        let rig = test_rig();
        let mut table = toml::Table::new();
        table.insert("gate_chi2".into(), 5.99.into());
        table.insert("bias_alpha".into(), 0.1.into());
        table.insert("bias_warmup".into(), 3.into());

        let estimator = FusedEstimator::from_config(&table, &rig).expect("config parses");
        assert_abs_diff_eq!(estimator.options.gate_chi2, 5.99, epsilon = 1e-12);
        assert_abs_diff_eq!(estimator.options.bias_alpha, 0.1, epsilon = 1e-12);
        assert_eq!(estimator.options.bias_warmup, 3);
    }

    #[test]
    fn test_logs_candidate_gated_at_debug() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let rgb = synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 1.0, 3);
        let mut ir = synthetic_ir_observation_at(&rig, centres, target, 0.2, 3);
        ir.timestamp = rgb.timestamp;

        let mut estimator = FusedEstimator::new(fused_options_no_kappa());
        let (_, logs) = capture_logs(tracing::Level::DEBUG, || {
            estimator
                .estimate_detailed(&[rgb, ir], &rig)
                .expect("estimate succeeds")
        });

        let recs: Vec<_> = logs
            .iter()
            .filter(|r| r.message == "fused candidate gated")
            .collect();
        assert!(!recs.is_empty());
        for rec in &recs {
            assert_eq!(rec.level, Level::Debug);
            assert!(matches!(rec.fields["accepted"], Value::Bool(_)));
            assert!(matches!(rec.fields["mahalanobis2"], Value::F64(_)));
        }
    }

    #[test]
    fn test_extrapolate_frame_constant_velocity() {
        let cov = Matrix3::identity() * 1e-6;
        let prev = landmark_frame(0.0, 150.0, 90.0, cov, 0.0);
        let cur = landmark_frame(66.0, 153.3, 93.3, cov, 0.0);

        let extrapolated =
            extrapolate_frame(Some(&prev), &cur, Timestamp::from_nanos(131_000_000), 100.0)
                .expect("prev present and within the limit");

        assert_eq!(extrapolated.timestamp, Timestamp::from_nanos(131_000_000));
        let right = extrapolated
            .eyes
            .iter()
            .find(|e| e.side == Side::Right)
            .expect("right eye present");
        assert_abs_diff_eq!(right.centre.position.x, 156.55, epsilon = 1e-9);
        assert!(
            right.centre.cov[(0, 0)] > cov[(0, 0)],
            "extrapolated covariance {} should exceed cur's {}",
            right.centre.cov[(0, 0)],
            cov[(0, 0)]
        );
    }

    #[test]
    fn test_extrapolate_frame_rotation_uses_scaled_axis() {
        let cov = Matrix3::identity() * 1e-6;
        let prev = landmark_frame(0.0, 150.0, 90.0, cov, 0.0);
        let cur = landmark_frame(66.0, 150.0, 90.0, cov, 2.0);

        let extrapolated =
            extrapolate_frame(Some(&prev), &cur, Timestamp::from_nanos(99_000_000), 100.0)
                .expect("prev present and within the limit");

        let yaw = extrapolated.viewer.euler_angles().1.to_degrees();
        assert_abs_diff_eq!(yaw, 3.0, epsilon = 0.01);
    }

    #[test]
    fn test_extrapolation_beyond_limit_holds_pose() {
        let cov = Matrix3::identity() * 1e-6;
        let prev = landmark_frame(0.0, 150.0, 90.0, cov, 0.0);
        let cur = landmark_frame(66.0, 153.3, 93.3, cov, 2.0);

        let extrapolated =
            extrapolate_frame(Some(&prev), &cur, Timestamp::from_nanos(216_000_000), 100.0)
                .expect("held pose is still Some");

        assert_eq!(extrapolated.timestamp, Timestamp::from_nanos(216_000_000));
        for (extrapolated_eye, cur_eye) in extrapolated.eyes.iter().zip(&cur.eyes) {
            assert_abs_diff_eq!(
                extrapolated_eye.centre.position,
                cur_eye.centre.position,
                epsilon = 1e-12
            );
            assert_abs_diff_eq!(
                extrapolated_eye.centre.cov,
                cur_eye.centre.cov + Matrix3::identity() * 225.0,
                epsilon = 1e-9
            );
        }
    }

    #[test]
    fn test_logs_pose_extrapolated_at_debug() {
        let cov = Matrix3::identity() * 1e-6;
        let prev = landmark_frame(0.0, 150.0, 90.0, cov, 0.0);
        let cur = landmark_frame(66.0, 153.3, 93.3, cov, 0.0);

        let (_, logs) = capture_logs(tracing::Level::DEBUG, || {
            extrapolate_frame(Some(&prev), &cur, Timestamp::from_nanos(131_000_000), 100.0)
        });

        let rec = logs
            .iter()
            .find(|r| r.message == "pose extrapolated")
            .expect("pose extrapolated record present");
        assert_eq!(rec.level, Level::Debug);
        match rec.fields["dt_ms"] {
            Value::F64(dt_ms) => assert_abs_diff_eq!(dt_ms, 65.0, epsilon = 1e-6),
            ref other => panic!("expected F64, got {other:?}"),
        }
        assert!(matches!(rec.fields["speed_mm_s"], Value::F64(_)));
        assert!(matches!(rec.fields["rate_deg_s"], Value::F64(_)));
        assert_eq!(rec.fields["held"], Value::Bool(false));
    }

    fn moving_head_pose(t_ms: f64) -> Isometry3<f64> {
        let t_sec = t_ms / 1000.0;
        Isometry3::from_parts(
            Translation3::new(155.0 + 50.0 * t_sec, 40.0, -500.0),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI + 30f64.to_radians() * t_sec),
        )
    }

    #[test]
    fn test_cross_chain_at_ir_time_beats_interpolation_during_motion() {
        let rig = test_rig();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();

        let rgb_prev = {
            let mut obs =
                synthetic_rgb_observation(&rig, &moving_head_pose(0.0), 1.0, target, 0.0, 0.0, 1);
            obs.timestamp = Timestamp::from_nanos(0);
            obs
        };
        let rgb_cur = {
            let mut obs =
                synthetic_rgb_observation(&rig, &moving_head_pose(66.0), 1.0, target, 0.0, 0.0, 1);
            for eye in obs.face.as_mut().expect("face present").eyes.iter_mut() {
                eye.iris = eye
                    .iris
                    .map(|m| Measured::new(m.into_value(), 1.0).expect("valid sigma"));
            }
            obs.timestamp = Timestamp::from_nanos(66_000_000);
            obs
        };
        let true_centres_at_ir = synthetic_eye_centres(&moving_head_pose(131.0), 1.0, &params);
        let ir_at_t_ir = {
            let mut obs = synthetic_ir_observation_at(&rig, true_centres_at_ir, target, 0.0, 2);
            for eye in obs.face.as_mut().expect("face present").eyes.iter_mut() {
                eye.pupil = eye
                    .pupil
                    .map(|m| Measured::new(m.into_value(), 0.2).expect("valid sigma"));
            }
            obs.timestamp = Timestamp::from_nanos(131_000_000);
            obs
        };

        let truth_right =
            Unit::new_normalize(Point3::new(target.x, target.y, 0.0) - true_centres_at_ir[0]);

        let run = |max_pose_extrapolation_ms: f64| -> GazeRay {
            let mut options = fused_options_no_kappa();
            options.stereo = false;
            options.max_pose_extrapolation_ms = max_pose_extrapolation_ms;
            let mut estimator = FusedEstimator::new(options);
            estimator
                .candidates(std::slice::from_ref(&rgb_prev), &rig)
                .expect("call 1 succeeds");
            let candidates = estimator
                .candidates(&[rgb_cur.clone(), ir_at_t_ir.clone()], &rig)
                .expect("call 2 succeeds");
            candidates
                .into_iter()
                .find(|(side, source, _)| {
                    *side == Side::Right && *source == FusedSource::IrOnRgbEyeball
                })
                .expect("ir-on-rgb-eyeball candidate present")
                .2
        };

        let extrapolated = run(100.0);
        let extrapolated_error = angle_deg(&extrapolated.direction, &truth_right);
        assert!(
            extrapolated_error < 0.3,
            "extrapolated error {extrapolated_error} deg"
        );

        let held = run(0.0);
        let held_error = angle_deg(&held.direction, &truth_right);
        assert!(held_error > 10.0, "held error {held_error} deg");
    }

    #[test]
    fn test_cross_chain_at_ir_time_with_glint_beats_interpolation_during_motion() {
        let rig = test_rig();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();

        let rgb_prev = {
            let mut obs =
                synthetic_rgb_observation(&rig, &moving_head_pose(0.0), 1.0, target, 0.0, 0.0, 1);
            obs.timestamp = Timestamp::from_nanos(0);
            obs
        };
        let rgb_cur = {
            let mut obs =
                synthetic_rgb_observation(&rig, &moving_head_pose(66.0), 1.0, target, 0.0, 0.0, 1);
            for eye in obs.face.as_mut().expect("face present").eyes.iter_mut() {
                eye.iris = eye
                    .iris
                    .map(|m| Measured::new(m.into_value(), 1.0).expect("valid sigma"));
            }
            obs.timestamp = Timestamp::from_nanos(66_000_000);
            obs
        };
        let ir_at_t_ir = {
            let mut obs = synthetic_pccr_observation_posed(
                &rig,
                target,
                &moving_head_pose(131.0),
                0.0,
                0.0,
                2,
            );
            obs.timestamp = Timestamp::from_nanos(131_000_000);
            obs
        };
        let true_centres_at_ir = synthetic_eye_centres(&moving_head_pose(131.0), 1.0, &params);
        let truth_right =
            Unit::new_normalize(Point3::new(target.x, target.y, 0.0) - true_centres_at_ir[0]);

        let mut options = fused_options_no_kappa();
        options.stereo = false;
        let mut estimator = FusedEstimator::new(options);
        estimator
            .candidates(&[rgb_prev], &rig)
            .expect("call 1 succeeds");
        let candidates = estimator
            .candidates(&[rgb_cur, ir_at_t_ir], &rig)
            .expect("call 2 succeeds");
        let glint_ray = candidates
            .into_iter()
            .find(|(side, source, _)| {
                *side == Side::Right && *source == FusedSource::IrGlintOnRgbEyeball
            })
            .expect("ir-glint-on-rgb-eyeball candidate present")
            .2;

        let error = angle_deg(&glint_ray.direction, &truth_right);
        assert!(error < 0.1, "glint candidate error {error} deg");
    }

    #[test]
    fn test_cross_chain_ray_timestamp_is_ir_time() {
        let rig = test_rig();
        let screen_from_head = frontal_screen_from_head();
        let target = Point2::new(100.0, 50.0);
        let params = EyeParams::default();
        let centres = synthetic_eye_centres(&screen_from_head, 1.0, &params);

        let with_pupil_sigma = |mut obs: Observations| -> Observations {
            for eye in obs.face.as_mut().expect("face present").eyes.iter_mut() {
                eye.pupil = eye
                    .pupil
                    .map(|m| Measured::new(m.into_value(), 0.2).expect("valid sigma"));
            }
            obs
        };

        let mut ir0 = with_pupil_sigma(synthetic_ir_observation_at(&rig, centres, target, 0.0, 1));
        ir0.timestamp = Timestamp::from_nanos(0);
        let mut ir136 =
            with_pupil_sigma(synthetic_ir_observation_at(&rig, centres, target, 0.0, 2));
        ir136.timestamp = Timestamp::from_nanos(136_000_000);
        let mut rgb68 =
            synthetic_rgb_observation(&rig, &screen_from_head, 1.0, target, 0.0, 0.0, 1);
        for eye in rgb68.face.as_mut().expect("face present").eyes.iter_mut() {
            eye.iris = eye
                .iris
                .map(|m| Measured::new(m.into_value(), 1.0).expect("valid sigma"));
        }
        rgb68.timestamp = Timestamp::from_nanos(68_000_000);

        let mut no_stereo = FusedEstimator::new(FusedOptions {
            stereo: false,
            ..fused_options_no_kappa()
        });
        no_stereo
            .candidates(&[ir0.clone()], &rig)
            .expect("call 1 succeeds");
        let candidates_no_stereo = no_stereo
            .candidates(&[ir136.clone(), rgb68.clone()], &rig)
            .expect("call 2 succeeds");
        let fallback = candidates_no_stereo
            .iter()
            .find(|(side, source, _)| {
                *side == Side::Right && *source == FusedSource::IrOnRgbEyeball
            })
            .expect("ir-on-rgb-eyeball candidate present");
        assert_eq!(fallback.2.timestamp, Timestamp::from_nanos(136_000_000));

        let mut with_stereo = FusedEstimator::new(fused_options_no_kappa());
        with_stereo
            .candidates(&[ir0], &rig)
            .expect("call 1 succeeds");
        let candidates_with_stereo = with_stereo
            .candidates(&[ir136, rgb68], &rig)
            .expect("call 2 succeeds");
        let stereo = candidates_with_stereo
            .iter()
            .find(|(side, source, _)| *side == Side::Right && *source == FusedSource::Stereo)
            .expect("stereo candidate present");
        assert_eq!(stereo.2.timestamp, Timestamp::from_nanos(68_000_000));
    }
}
