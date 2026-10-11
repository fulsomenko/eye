pub mod blob;
pub mod glint;
pub mod pupil;
#[cfg(test)]
mod testutil;

use std::collections::HashMap;

use eye_core::image::GrayView;
use eye_core::log::field;
use eye_core::observation::LandmarkScheme;
use eye_core::stage::{Detector, StageError};
use eye_core::{
    CameraId, Ellipse2, EyeObservation, FaceObservation, Frame, FrameSet, Illumination, Measured,
    Observations, PixelFormat, Side, Timestamp,
};
use nalgebra::Point2;

use crate::DetectError;
use crate::image::saturating_diff;
use crate::options::parse_options;

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct IrClassicOptions {
    pub max_pair_gap_ms: f64,
    pub background_size: usize,
    /// Seed threshold: `median(th) + threshold_k_mad * 1.4826 * MAD(th)`, floored by
    /// `min_threshold`.
    pub threshold_k_mad: f64,
    /// Grow threshold as a fraction of the seed threshold (hysteresis).
    pub threshold_hysteresis: f64,
    pub min_threshold: u8,
    pub pupil_area_px: [u32; 2],
    pub min_aspect: f64,
    pub min_pupil_contrast: f64,
    pub max_iris_ratio: f64,
    /// Floor on the radius used to place the iris/outer annuli; below it the "iris" annulus
    /// falls inside the pupil's own bright halo.
    pub min_ring_radius_px: f64,
    pub pair_separation_px: [f64; 2],
    pub max_pair_tilt_deg: f64,
    pub edge_rays: usize,
    pub max_candidates: usize,
    pub glints: bool,
    /// Search window half-width, in units of the pupil radius `sqrt(semi_major * semi_minor)`.
    pub glint_search_radius: f64,
    /// Lit levels the glint peak must exceed the pupil plateau by (E2: observed 15 to 60).
    pub glint_min_excess: f64,
    /// 1-sigma glint position noise in px.
    pub glint_sigma_px: f64,
    /// Lower bound on the 1-sigma pupil-centre noise from the ellipse fit, px.
    pub pupil_sigma_floor_px: f64,
    /// Candidates farther than this from the previous accepted pupil (scaled by the lit-frame
    /// gap over 133 ms) are gated out before contrast ranking.
    pub track_gate_px: f64,
    /// Accept a candidate only if its area is within this ratio of the previous accepted
    /// blob's area (both ways).
    pub track_area_ratio: f64,
    /// Forget the previous pair after this gap, ms.
    pub track_max_gap_ms: f64,
}

impl Default for IrClassicOptions {
    fn default() -> Self {
        Self {
            max_pair_gap_ms: 100.0,
            background_size: 15,
            threshold_k_mad: 12.0,
            threshold_hysteresis: 1.0,
            min_threshold: 30,
            pupil_area_px: [4, 120],
            min_aspect: 0.3,
            min_pupil_contrast: 1.3,
            max_iris_ratio: 0.8,
            min_ring_radius_px: 2.2,
            pair_separation_px: [35.0, 95.0],
            max_pair_tilt_deg: 30.0,
            edge_rays: 16,
            max_candidates: 64,
            glints: true,
            glint_search_radius: 2.0,
            glint_min_excess: 15.0,
            glint_sigma_px: 0.19,
            pupil_sigma_floor_px: 0.38,
            track_gate_px: 5.0,
            track_area_ratio: 2.0,
            track_max_gap_ms: 400.0,
        }
    }
}

type PupilAndGlint = (Measured<Ellipse2>, Option<Measured<Point2<f64>>>);

#[derive(Debug, Clone, Copy)]
pub(crate) struct PairPrior {
    pub at: Timestamp,
    pub right: (Point2<f64>, u32),
    pub left: (Point2<f64>, u32),
}

#[derive(Debug)]
pub struct IrClassicDetector {
    options: IrClassicOptions,
    last_dark: HashMap<CameraId, Frame>,
    last_pair: HashMap<CameraId, PairPrior>,
}

impl IrClassicDetector {
    pub const NAME: &'static str = "ir-classic";

    pub fn new(options: IrClassicOptions) -> Self {
        Self {
            options,
            last_dark: HashMap::new(),
            last_pair: HashMap::new(),
        }
    }

    pub fn from_config(table: &toml::Table, _rig: &eye_core::Rig) -> Result<Self, StageError> {
        Ok(Self::new(parse_options(Self::NAME, table)?))
    }

    pub fn accepts_frame(format: PixelFormat, illumination: Illumination) -> bool {
        format == PixelFormat::Gray8
            && matches!(illumination, Illumination::IrLit | Illumination::IrDark)
    }

    /// Pure core, unit-testable without frames. Returns `[right, left]`. Equivalent to
    /// `detect_pair_with_prior` with no prior.
    pub fn detect_pair(
        &self,
        lit: GrayView<'_>,
        dark: GrayView<'_>,
    ) -> Result<Option<[EyeObservation; 2]>, DetectError> {
        self.detect_pair_with_prior(lit, dark, None, Timestamp::from_nanos(0))
            .map(|r| r.map(|(eyes, _prior)| eyes))
    }

    /// `detect_pair`, gated against the previous accepted pair when one is given and still
    /// within `track_max_gap_ms` of `at`. Returns the accepted eyes plus the prior the caller
    /// should pass to the next frame.
    pub(crate) fn detect_pair_with_prior(
        &self,
        lit: GrayView<'_>,
        dark: GrayView<'_>,
        prior: Option<&PairPrior>,
        at: Timestamp,
    ) -> Result<Option<([EyeObservation; 2], PairPrior)>, DetectError> {
        let diff = saturating_diff(lit, dark)?;
        let diff = diff.view();
        let (cands, counts) = blob::candidates(diff, &self.options);

        let gap_ms = prior
            .map(|p| at.nanos_since(p.at).max(0) as f64 / 1e6)
            .unwrap_or(0.0);
        let active_prior = prior.filter(|_| gap_ms <= self.options.track_max_gap_ms);

        let Some((i, j, _switched)) =
            pupil::select_pair_with_prior(&cands, active_prior, gap_ms, &self.options)
        else {
            let reason = match cands.len() {
                0 => "no_candidates",
                1 => "single_candidate",
                _ => "no_pair",
            };
            tracing::debug!(
                { field::REASON } = reason,
                candidates = cands.len() as u64,
                rejected_area = counts.area,
                rejected_aspect = counts.aspect,
                rejected_contrast = counts.contrast,
                rejected_iris_ratio = counts.iris_ratio,
                components = counts.components_total,
                "pair rejected"
            );
            return Ok(None);
        };
        let (pupil_i, glint_i) = self.pupil_and_glint(lit, diff, &cands[i])?;
        let (pupil_j, glint_j) = self.pupil_and_glint(lit, diff, &cands[j])?;

        let (right, left) = if pupil_i.value().center().x <= pupil_j.value().center().x {
            ((pupil_i, glint_i), (pupil_j, glint_j))
        } else {
            ((pupil_j, glint_j), (pupil_i, glint_i))
        };
        let next_prior = if cands[i].centroid.x <= cands[j].centroid.x {
            PairPrior {
                at,
                right: (cands[i].centroid, cands[i].area),
                left: (cands[j].centroid, cands[j].area),
            }
        } else {
            PairPrior {
                at,
                right: (cands[j].centroid, cands[j].area),
                left: (cands[i].centroid, cands[i].area),
            }
        };

        let mut right_eye = EyeObservation::new(Side::Right);
        right_eye.pupil = Some(right.0);
        right_eye.glints = right.1.into_iter().collect();
        let mut left_eye = EyeObservation::new(Side::Left);
        left_eye.pupil = Some(left.0);
        left_eye.glints = left.1.into_iter().collect();

        for eye in [&right_eye, &left_eye] {
            let pupil = eye.pupil.expect("pupil set above");
            let ellipse = pupil.value();
            tracing::trace!(
                side = eye.side.as_str(),
                x = ellipse.center().x,
                y = ellipse.center().y,
                semi_major = ellipse.semi_major(),
                semi_minor = ellipse.semi_minor(),
                angle_rad = ellipse.angle(),
                sigma = pupil.sigma(),
                glint = !eye.glints.is_empty(),
                "pupil"
            );
        }

        Ok(Some(([right_eye, left_eye], next_prior)))
    }

    fn pupil_and_glint(
        &self,
        lit: GrayView<'_>,
        diff: GrayView<'_>,
        c: &blob::Candidate,
    ) -> Result<PupilAndGlint, DetectError> {
        let pupil = pupil::pupil_from_candidate(diff, c, &self.options)?;
        if !self.options.glints {
            return Ok((pupil, None));
        }

        let search = glint::GlintSearch {
            lit,
            pupil: pupil.value(),
            options: &self.options,
        };
        let Some(found) = glint::find_glint(&search)? else {
            return Ok((pupil, None));
        };

        let plateau_diff = glint::plateau_median(diff, pupil.value());
        let masked = glint::mask_glint(diff, found.value(), plateau_diff)?;
        let pupil = pupil::pupil_from_candidate(masked.view(), c, &self.options)?;
        Ok((pupil, Some(found)))
    }

    fn paired_dark(&self, lit: &Frame) -> Option<&Frame> {
        let h = lit.header();
        let Some(dark) = self.last_dark.get(&h.camera) else {
            tracing::debug!({ field::REASON } = "no_dark", "lit frame unpaired");
            return None;
        };
        let gap = h.timestamp.nanos_since(dark.header().timestamp);
        if gap > 0 && gap as f64 <= self.options.max_pair_gap_ms * 1e6 {
            Some(dark)
        } else {
            tracing::debug!(
                { field::REASON } = "dark_gap",
                gap_ns = gap,
                max_pair_gap_ms = self.options.max_pair_gap_ms,
                "lit frame unpaired"
            );
            None
        }
    }

    fn detect_frames(&mut self, frames: &FrameSet) -> Result<Vec<Observations>, DetectError> {
        let mut out = Vec::new();
        for frame in frames.frames() {
            let h = frame.header();
            if !Self::accepts_frame(h.format, h.illumination) {
                tracing::trace!(
                    { field::REASON } = "not_accepted",
                    format = ?h.format,
                    { field::ILLUMINATION } = h.illumination.as_str(),
                    "frame not accepted"
                );
                continue;
            }
            if h.illumination == Illumination::IrDark {
                self.last_dark.insert(h.camera.clone(), frame.clone());
                tracing::debug!("dark frame stored");
                continue;
            }
            let face = match self.paired_dark(frame) {
                Some(dark) => {
                    let start = std::time::Instant::now();
                    let prior = self.last_pair.get(&h.camera).copied();
                    let result = self.detect_pair_with_prior(
                        GrayView::from_frame(frame)?,
                        GrayView::from_frame(dark)?,
                        prior.as_ref(),
                        h.timestamp,
                    )?;
                    tracing::trace!(
                        { field::ELAPSED_US } = start.elapsed().as_micros() as u64,
                        face = result.is_some(),
                        "ir-classic frame"
                    );
                    match result {
                        Some((eyes, next_prior)) => {
                            self.last_pair.insert(h.camera.clone(), next_prior);
                            Some(face_from_pupils(eyes))
                        }
                        None => None,
                    }
                }
                None => None,
            };
            out.push(Observations {
                camera: h.camera.clone(),
                timestamp: h.timestamp,
                face,
            });
        }
        Ok(out)
    }
}

impl Detector for IrClassicDetector {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn accepts(&self, format: PixelFormat, illumination: Illumination) -> bool {
        Self::accepts_frame(format, illumination)
    }

    fn detect(&mut self, frames: &FrameSet) -> Result<Vec<Observations>, StageError> {
        Ok(self.detect_frames(frames)?)
    }
}

fn face_from_pupils(eyes: [EyeObservation; 2]) -> FaceObservation {
    let landmarks = eyes
        .iter()
        .filter_map(|e| e.pupil.map(|p| p.value().center()))
        .collect();
    FaceObservation {
        scheme: LandmarkScheme::IR_PUPIL_PAIR,
        landmarks,
        eyes: eyes.into(),
    }
}

#[cfg(test)]
mod tests {
    use eye_core::{CameraId, FrameHeader, Rig, Timestamp};
    use nalgebra::Point2;

    use super::*;
    use crate::ir::testutil::{SyntheticEye, SyntheticIr};

    fn ir_frame(
        illumination: Illumination,
        ts_ns: u64,
        seq: u64,
        image: &eye_core::image::GrayImage,
    ) -> Frame {
        Frame::new(
            FrameHeader {
                camera: CameraId::from("ir"),
                seq,
                timestamp: Timestamp::from_nanos(ts_ns),
                width: image.width(),
                height: image.height(),
                format: PixelFormat::Gray8,
                illumination,
            },
            image.data().to_vec().into(),
        )
        .unwrap()
    }

    fn rgb_frame(ts_ns: u64) -> Frame {
        Frame::new(
            FrameHeader {
                camera: CameraId::from("rgb"),
                seq: 0,
                timestamp: Timestamp::from_nanos(ts_ns),
                width: 1280,
                height: 720,
                format: PixelFormat::Rgb8,
                illumination: Illumination::Ambient,
            },
            vec![0u8; 1280 * 720 * 3].into(),
        )
        .unwrap()
    }

    fn assert_pupils_within(
        eyes: &[EyeObservation; 2],
        right_expected: Point2<f64>,
        left_expected: Point2<f64>,
        tol: f64,
    ) {
        let right = eyes[0].pupil.unwrap().value().center();
        let left = eyes[1].pupil.unwrap().value().center();
        assert_eq!(eyes[0].side, Side::Right);
        assert_eq!(eyes[1].side, Side::Left);
        assert!(
            (right - right_expected).norm() <= tol,
            "right pupil {:?} not within {tol} of {:?}",
            right,
            right_expected
        );
        assert!(
            (left - left_expected).norm() <= tol,
            "left pupil {:?} not within {tol} of {:?}",
            left,
            left_expected
        );
    }

    #[test]
    fn test_noise_free_pair_recovers_centres_within_0_05px() {
        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let scene = SyntheticIr::default_scene();
        let (lit, dark) = scene.render();
        let eyes = detector
            .detect_pair(lit.view(), dark.view())
            .unwrap()
            .expect("face detected");
        assert_pupils_within(
            &eyes,
            Point2::new(290.3, 180.7),
            Point2::new(350.6, 181.2),
            0.05,
        );
        for eye in &eyes {
            let sigma = eye.pupil.unwrap().sigma();
            assert!(sigma > 0.0 && sigma <= 0.38, "sigma {sigma} out of range");
        }
    }

    #[test]
    fn test_pupil_sigma_respects_floor() {
        let detector = IrClassicDetector::new(IrClassicOptions {
            pupil_sigma_floor_px: 0.3,
            ..IrClassicOptions::default()
        });
        let scene = SyntheticIr::default_scene();
        let (lit, dark) = scene.render();
        let eyes = detector
            .detect_pair(lit.view(), dark.view())
            .unwrap()
            .expect("face detected");
        for eye in &eyes {
            assert_eq!(eye.pupil.unwrap().sigma(), 0.3);
        }
    }

    #[test]
    fn test_pupil_sigma_below_default_floor_without_option() {
        let detector = IrClassicDetector::new(IrClassicOptions {
            pupil_sigma_floor_px: 0.05,
            ..IrClassicOptions::default()
        });
        let scene = SyntheticIr::default_scene();
        let (lit, dark) = scene.render();
        let eyes = detector
            .detect_pair(lit.view(), dark.view())
            .unwrap()
            .expect("face detected");
        for eye in &eyes {
            let sigma = eye.pupil.unwrap().sigma();
            assert!(
                sigma < 0.3,
                "sigma {sigma} did not come from the fit residual"
            );
        }
    }

    #[test]
    fn test_ir_classic_options_default_floor_is_0_38() {
        assert_eq!(IrClassicOptions::default().pupil_sigma_floor_px, 0.38);
    }

    #[test]
    fn test_ir_classic_options_default_glint_sigma_is_0_19() {
        assert_eq!(IrClassicOptions::default().glint_sigma_px, 0.19);
    }

    #[test]
    fn test_noisy_pair_recovers_centres_within_0_2px() {
        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let mut scene = SyntheticIr::default_scene();
        scene.noise_sigma = 4.0;
        scene.seed = 7;
        let (lit, dark) = scene.render();
        let eyes = detector
            .detect_pair(lit.view(), dark.view())
            .unwrap()
            .expect("face detected");
        assert_pupils_within(
            &eyes,
            Point2::new(290.3, 180.7),
            Point2::new(350.6, 181.2),
            0.2,
        );
    }

    #[test]
    fn test_dim_pupil_below_skin_level_is_detected() {
        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let mut scene = SyntheticIr::default_scene();
        for eye in &mut scene.eyes {
            eye.pupil_level = 70;
        }
        let (lit, dark) = scene.render();
        let eyes = detector
            .detect_pair(lit.view(), dark.view())
            .unwrap()
            .expect("face detected");
        assert_pupils_within(
            &eyes,
            Point2::new(290.3, 180.7),
            Point2::new(350.6, 181.2),
            0.05,
        );
    }

    #[test]
    fn test_dim_noisy_pupil_within_0_2px() {
        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let mut scene = SyntheticIr::default_scene();
        for eye in &mut scene.eyes {
            eye.pupil_level = 70;
        }
        scene.noise_sigma = 4.0;
        scene.seed = 7;
        let (lit, dark) = scene.render();
        let eyes = detector
            .detect_pair(lit.view(), dark.view())
            .unwrap()
            .expect("face detected");
        assert_pupils_within(
            &eyes,
            Point2::new(290.3, 180.7),
            Point2::new(350.6, 181.2),
            0.2,
        );
    }

    #[test]
    fn test_pupil_without_local_excess_is_not_detected() {
        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let mut scene = SyntheticIr::default_scene();
        for eye in &mut scene.eyes {
            eye.pupil_level = 55;
        }
        let (lit, dark) = scene.render();
        assert_eq!(detector.detect_pair(lit.view(), dark.view()).unwrap(), None);
    }

    #[test]
    fn test_sides_assigned_by_image_x() {
        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let scene = SyntheticIr::default_scene();
        let (lit, dark) = scene.render();
        let eyes = detector
            .detect_pair(lit.view(), dark.view())
            .unwrap()
            .expect("face detected");
        assert_eq!(eyes[0].side, Side::Right);
        assert_eq!(eyes[1].side, Side::Left);
        let right_center = eyes[0].pupil.unwrap().value().center();
        let left_center = eyes[1].pupil.unwrap().value().center();
        assert!(right_center.x < left_center.x);
        for eye in &eyes {
            assert!(eye.corners.is_none());
            assert!(eye.iris.is_none());
            assert!(eye.glints.is_empty());
        }

        let face = face_from_pupils(eyes.clone());
        assert_eq!(face.landmarks.len(), 2);
        assert_eq!(face.landmarks[0], right_center);
        assert_eq!(face.landmarks[1], left_center);
    }

    #[test]
    fn test_masking_glint_removes_pupil_bias() {
        let truth = Point2::new(290.3, 180.7);
        let glint_point = truth + nalgebra::Vector2::new(1.3, 0.0);

        let mut unmasked_scene = SyntheticIr::default_scene();
        unmasked_scene.eyes[0].glint = Some((glint_point, 1.0, 255.0));
        let (lit, dark) = unmasked_scene.render();

        let options_unmasked = IrClassicOptions {
            glints: false,
            ..IrClassicOptions::default()
        };
        let without_mask = IrClassicDetector::new(options_unmasked)
            .detect_pair(lit.view(), dark.view())
            .unwrap()
            .expect("face detected");
        let unmasked_center = without_mask[0].pupil.unwrap().value().center();

        let with_mask = IrClassicDetector::new(IrClassicOptions::default())
            .detect_pair(lit.view(), dark.view())
            .unwrap()
            .expect("face detected");
        let masked_center = with_mask[0].pupil.unwrap().value().center();

        assert!((masked_center - truth).norm() <= 0.05);
        assert!((masked_center - truth).norm() < (unmasked_center - truth).norm());
    }

    #[test]
    fn test_detect_pair_populates_glints_for_both_eyes() {
        let mut det: Box<dyn Detector> =
            Box::new(IrClassicDetector::from_config(&toml::Table::new(), &nominal_rig()).unwrap());

        let mut scene = SyntheticIr::default_scene();
        let offset = nalgebra::Vector2::new(0.7, -0.4);
        let right_truth = scene.eyes[0].pupil_center;
        let left_truth = scene.eyes[1].pupil_center;
        scene.eyes[0].glint = Some((right_truth + offset, 0.6, 255.0));
        scene.eyes[1].glint = Some((left_truth + offset, 0.6, 255.0));
        let (lit, dark) = scene.render();

        let dark_frames = FrameSet::single(ir_frame(Illumination::IrDark, 0, 0, &dark));
        det.detect(&dark_frames).unwrap();
        let lit_frames = FrameSet::single(ir_frame(Illumination::IrLit, 68_000_000, 1, &lit));
        let out = det.detect(&lit_frames).unwrap();
        let face = out[0].face.as_ref().expect("face detected");

        let right = &face.eyes[0];
        assert_eq!(right.side, Side::Right);
        assert_eq!(right.glints.len(), 1);
        assert!((*right.glints[0].value() - (right_truth + offset)).norm() <= 0.1);
        assert!((right.pupil.unwrap().value().center() - right_truth).norm() <= 0.05);

        let left = &face.eyes[1];
        assert_eq!(left.side, Side::Left);
        assert_eq!(left.glints.len(), 1);
        assert!((*left.glints[0].value() - (left_truth + offset)).norm() <= 0.1);
        assert!((left.pupil.unwrap().value().center() - left_truth).norm() <= 0.05);
    }

    #[test]
    fn test_glints_disabled_leaves_vec_empty() {
        let mut scene = SyntheticIr::default_scene();
        let glint_point = scene.eyes[0].pupil_center + nalgebra::Vector2::new(0.7, -0.4);
        scene.eyes[0].glint = Some((glint_point, 0.6, 255.0));
        let (lit, dark) = scene.render();

        let options = IrClassicOptions {
            glints: false,
            ..IrClassicOptions::default()
        };
        let eyes = IrClassicDetector::new(options)
            .detect_pair(lit.view(), dark.view())
            .unwrap()
            .expect("face detected");
        for eye in &eyes {
            assert!(eye.glints.is_empty());
        }
    }

    #[test]
    fn test_from_config_rejects_unknown_option_still() {
        let mut table = toml::Table::new();
        table.insert("not_a_real_option".into(), 1.into());
        let err = IrClassicDetector::from_config(&table, &nominal_rig()).unwrap_err();
        assert!(matches!(err, StageError::Config(_)));

        let mut table = toml::Table::new();
        table.insert("glint_min_excess".into(), 20.0.into());
        let detector = IrClassicDetector::from_config(&table, &nominal_rig()).unwrap();
        assert_eq!(detector.options.glint_min_excess, 20.0);
    }

    #[test]
    #[ignore = "needs EYE_RECORDING"]
    fn test_recording_glint_rate_and_jitter() {
        let dir = std::env::var("EYE_RECORDING").expect("EYE_RECORDING must be set");
        let index_path = std::path::Path::new(&dir).join("index.jsonl");
        let index = std::fs::read_to_string(&index_path).expect("index.jsonl readable");

        let targets_path = std::path::Path::new(&dir).join("targets.jsonl");
        let targets_raw = std::fs::read_to_string(&targets_path).expect("targets.jsonl readable");
        let records: Vec<eye_core::session::TargetRecord> = targets_raw
            .lines()
            .map(|line| serde_json::from_str(line).expect("targets.jsonl line parses"))
            .collect();
        let protocol = eye_core::session::ProtocolConfig::from_target_records(
            &records,
            &eye_core::session::ProtocolConfig::default(),
        )
        .unwrap_or_default();
        let settle_ns = protocol.settle_ms * 1_000_000;
        // Target windows: [shown_ns + settle, hidden_ns), so the eye has stopped travelling.
        let windows: Vec<(u64, u64)> = records
            .iter()
            .filter_map(|r| r.hidden_ns.map(|hidden| (r.shown_ns + settle_ns, hidden)))
            .collect();

        let mut detector = IrClassicDetector::new(IrClassicOptions::default());
        let mut lit_count = 0u64;
        // Indexed by side: [Right, Left].
        let mut glint_counts = [0u64; 2];
        let mut vectors: [Vec<(f64, f64)>; 2] = [Vec::new(), Vec::new()];
        let mut centres: [Vec<Point2<f64>>; 2] = [Vec::new(), Vec::new()];
        let mut prev_centre: [Option<Point2<f64>>; 2] = [None, None];
        let mut jump_counts = [0u64; 2];
        let mut window_centres: [Vec<Vec<Point2<f64>>>; 2] = [
            vec![Vec::new(); windows.len()],
            vec![Vec::new(); windows.len()],
        ];

        for line in index.lines() {
            let illumination = match json_string_field(line, "illumination") {
                Some("ir_lit") => Illumination::IrLit,
                Some("ir_dark") => Illumination::IrDark,
                _ => continue,
            };
            let seq = json_number_field(line, "seq").unwrap_or(0);
            let ts_ns = json_number_field(line, "timestamp_ns").unwrap_or(0);
            let path = std::path::Path::new(&dir).join(format!("frames/ir/{seq:08}.pgm"));
            let pixels = read_pgm(&path);
            let frame = ir_frame(illumination, ts_ns, seq, &pixels);

            let frames = FrameSet::single(frame.clone());
            let out = detector.detect_frames(&frames).unwrap();

            if illumination == Illumination::IrDark {
                continue;
            }

            lit_count += 1;
            if let Some(face) = out.first().and_then(|obs| obs.face.as_ref()) {
                for eye in &face.eyes {
                    let side = match eye.side {
                        Side::Right => 0,
                        Side::Left => 1,
                    };
                    if let Some(pupil) = eye.pupil {
                        let centre = pupil.value().center();
                        if let Some(prev) = prev_centre[side]
                            && (centre - prev).norm() > 3.0
                        {
                            jump_counts[side] += 1;
                        }
                        prev_centre[side] = Some(centre);
                        centres[side].push(centre);
                        if let Some(w) = windows
                            .iter()
                            .position(|&(start, end)| ts_ns >= start && ts_ns < end)
                        {
                            window_centres[side][w].push(centre);
                        }
                    }
                    if let (Some(pupil), Some(glint)) = (eye.pupil, eye.glints.first().copied()) {
                        glint_counts[side] += 1;
                        let v = *glint.value() - pupil.value().center();
                        vectors[side].push((v.x, v.y));
                    }
                }
            }
        }

        for (side, name) in [(0, "right"), (1, "left")] {
            let rate = if lit_count > 0 {
                100.0 * glint_counts[side] as f64 / lit_count as f64
            } else {
                0.0
            };
            let n = vectors[side].len().max(1) as f64;
            let mean_x = vectors[side].iter().map(|v| v.0).sum::<f64>() / n;
            let mean_y = vectors[side].iter().map(|v| v.1).sum::<f64>() / n;
            let std_x = (vectors[side]
                .iter()
                .map(|v| (v.0 - mean_x).powi(2))
                .sum::<f64>()
                / n)
                .sqrt();
            let std_y = (vectors[side]
                .iter()
                .map(|v| (v.1 - mean_y).powi(2))
                .sum::<f64>()
                / n)
                .sqrt();
            println!(
                "{name} eye glint detection rate: {}/{lit_count} ({rate:.1}%)",
                glint_counts[side]
            );
            println!("{name} eye pupil-glint vector std: x={std_x:.3} y={std_y:.3}");

            let mut window_stds: Vec<f64> = window_centres[side]
                .iter()
                .filter(|pts| pts.len() >= 2)
                .map(|pts| {
                    let n = pts.len() as f64;
                    let mean_x = pts.iter().map(|p| p.x).sum::<f64>() / n;
                    let mean_y = pts.iter().map(|p| p.y).sum::<f64>() / n;
                    (pts.iter()
                        .map(|p| (p.x - mean_x).powi(2) + (p.y - mean_y).powi(2))
                        .sum::<f64>()
                        / n)
                        .sqrt()
                })
                .collect();
            window_stds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let within_window_std = match window_stds.len() {
                0 => 0.0,
                len if len.is_multiple_of(2) => {
                    (window_stds[len / 2 - 1] + window_stds[len / 2]) / 2.0
                }
                len => window_stds[len / 2],
            };
            println!(
                "{name} eye within-window centre std (median over {} windows): {within_window_std:.3} px, jumps > 3px: {}/{}",
                window_stds.len(),
                jump_counts[side],
                centres[side].len()
            );
        }
    }

    #[test]
    fn test_ambient_ir_in_both_frames_is_ignored() {
        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let baseline = SyntheticIr::default_scene();
        let (base_lit, base_dark) = baseline.render();
        let base_eyes = detector
            .detect_pair(base_lit.view(), base_dark.view())
            .unwrap()
            .expect("face detected");

        let mut scene = SyntheticIr::default_scene();
        scene.ambient = Some((
            crate::image::Roi {
                x: 220,
                y: 130,
                width: 200,
                height: 100,
            },
            30,
        ));
        let (lit, dark) = scene.render();
        let eyes = detector
            .detect_pair(lit.view(), dark.view())
            .unwrap()
            .expect("face detected");

        assert_pupils_within(
            &eyes,
            base_eyes[0].pupil.unwrap().value().center(),
            base_eyes[1].pupil.unwrap().value().center(),
            1e-12,
        );
    }

    #[test]
    fn test_specular_highlight_on_skin_is_rejected() {
        let mut scene = SyntheticIr::default_scene();
        scene.specular = vec![
            (Point2::new(370.0, 240.0), 1.5, 255),
            (Point2::new(411.0, 181.0), 1.5, 255),
        ];
        let (lit, dark) = scene.render();
        let diff = saturating_diff(lit.view(), dark.view()).unwrap();
        let (cands, _counts) = blob::candidates(diff.view(), &IrClassicOptions::default());
        assert_eq!(cands.len(), 2);
        for c in &cands {
            assert!((3.0..=4.2).contains(&c.contrast));
            let ratio = c.iris_mean / c.outer_mean;
            assert!((0.5..=0.7).contains(&ratio));
        }

        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let eyes = detector
            .detect_pair(lit.view(), dark.view())
            .unwrap()
            .expect("face detected");
        assert_pupils_within(
            &eyes,
            Point2::new(290.3, 180.7),
            Point2::new(350.6, 181.2),
            0.05,
        );
    }

    #[test]
    fn test_dim_third_pupil_in_valid_pair_geometry_loses_on_contrast_sum() {
        let mut scene = SyntheticIr::default_scene();
        let mut third = SyntheticEye::at(Point2::new(411.0, 181.0));
        third.pupil_level = 120;
        scene.eyes.push(third);
        let (lit, dark) = scene.render();

        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let eyes = detector
            .detect_pair(lit.view(), dark.view())
            .unwrap()
            .expect("face detected");
        assert_pupils_within(
            &eyes,
            Point2::new(290.3, 180.7),
            Point2::new(350.6, 181.2),
            0.05,
        );
    }

    #[test]
    fn test_small_pupil_above_area_floor_is_detected() {
        let mut scene = SyntheticIr::default_scene();
        for eye in &mut scene.eyes {
            eye.pupil_radius = 1.1;
        }
        let (lit, dark) = scene.render();
        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let eyes = detector
            .detect_pair(lit.view(), dark.view())
            .unwrap()
            .expect("face detected");
        assert_pupils_within(
            &eyes,
            Point2::new(290.3, 180.7),
            Point2::new(350.6, 181.2),
            1.0,
        );
    }

    #[test]
    fn test_half_occluded_pupil_is_detected() {
        let mut scene = SyntheticIr::default_scene();
        let left_center = scene.eyes[1].pupil_center;
        let r = scene.eyes[1].pupil_radius;
        let cutline = left_center.y - r + 0.65 * (2.0 * r);
        let occluder_radius = 100.0;
        scene.specular = vec![(
            Point2::new(left_center.x, cutline - occluder_radius),
            occluder_radius,
            scene.skin_level,
        )];
        let (lit, dark) = scene.render();
        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let eyes = detector
            .detect_pair(lit.view(), dark.view())
            .unwrap()
            .expect("face detected");

        let right = eyes[0].pupil.unwrap().value().center();
        assert!(
            (right - Point2::new(290.3, 180.7)).norm() <= 0.05,
            "right pupil {right:?} moved"
        );

        let left = eyes[1].pupil.unwrap().value().center();
        assert!(
            (left.x - left_center.x).abs() <= 0.5,
            "left pupil x {} not within 0.5 of {}",
            left.x,
            left_center.x
        );
        let uncovered_centroid_y = (cutline + (left_center.y + r)) / 2.0;
        assert!(
            (left.y - uncovered_centroid_y).abs() <= 1.5,
            "left pupil y {} not within 1.5 of uncovered-segment centroid {uncovered_centroid_y}",
            left.y
        );
    }

    #[test]
    fn test_single_pupil_yields_no_face() {
        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let mut scene = SyntheticIr::default_scene();
        scene.eyes.truncate(1);
        let (lit, dark) = scene.render();
        assert_eq!(detector.detect_pair(lit.view(), dark.view()).unwrap(), None);
    }

    #[test]
    fn test_pair_separation_out_of_range_yields_no_face() {
        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let mut scene = SyntheticIr::default_scene();
        scene.eyes = vec![
            SyntheticEye::at(Point2::new(260.3, 180.7)),
            SyntheticEye::at(Point2::new(380.3, 181.2)),
        ];
        let (lit, dark) = scene.render();
        assert_eq!(detector.detect_pair(lit.view(), dark.view()).unwrap(), None);
    }

    #[test]
    fn test_tilted_pair_beyond_limit_yields_no_face() {
        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let mut scene = SyntheticIr::default_scene();
        scene.eyes = vec![
            SyntheticEye::at(Point2::new(290.3, 180.7)),
            SyntheticEye::at(Point2::new(336.3, 219.3)),
        ];
        let (lit, dark) = scene.render();
        assert_eq!(detector.detect_pair(lit.view(), dark.view()).unwrap(), None);
    }

    #[test]
    fn test_prior_rejects_brighter_distractor_near_previous_pupil() {
        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let baseline = SyntheticIr::default_scene();
        let (base_lit, base_dark) = baseline.render();
        let (_, prior) = detector
            .detect_pair_with_prior(
                base_lit.view(),
                base_dark.view(),
                None,
                Timestamp::from_nanos(0),
            )
            .unwrap()
            .expect("baseline face detected");

        let mut scene = SyntheticIr::default_scene();
        let mut distractor = SyntheticEye::at(Point2::new(300.0, 184.0));
        distractor.pupil_level = 240;
        scene.eyes.push(distractor);
        let (lit, dark) = scene.render();

        let eyes = detector
            .detect_pair(lit.view(), dark.view())
            .unwrap()
            .expect("face detected");
        let right = eyes[0].pupil.unwrap().value().center();
        assert!(
            (right - Point2::new(290.3, 180.7)).norm() > 1.0,
            "expected the distractor to win without a prior, got {right:?}"
        );

        let at = Timestamp::from_nanos(68_000_000);
        let (eyes, _next) = detector
            .detect_pair_with_prior(lit.view(), dark.view(), Some(&prior), at)
            .unwrap()
            .expect("face detected");
        assert_pupils_within(
            &eyes,
            Point2::new(290.3, 180.7),
            Point2::new(350.6, 181.2),
            0.05,
        );
    }

    #[test]
    fn test_prior_is_dropped_after_max_gap() {
        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let scene = SyntheticIr::default_scene();
        let (lit, dark) = scene.render();
        let (_, prior) = detector
            .detect_pair_with_prior(lit.view(), dark.view(), None, Timestamp::from_nanos(0))
            .unwrap()
            .expect("face detected");

        // 500ms exceeds the default track_max_gap_ms of 400ms.
        let at = Timestamp::from_nanos(500_000_000);
        use eye_log::Value;
        use eye_log::testing::capture_logs;
        let (result, logs) = capture_logs(tracing::Level::DEBUG, || {
            detector.detect_pair_with_prior(lit.view(), dark.view(), Some(&prior), at)
        });
        let (eyes, _next) = result.unwrap().expect("face detected");
        assert_pupils_within(
            &eyes,
            Point2::new(290.3, 180.7),
            Point2::new(350.6, 181.2),
            0.05,
        );
        let rec = logs
            .iter()
            .find(|r| r.message == "pair accepted")
            .expect("event emitted");
        assert_eq!(rec.fields["gated"], Value::Bool(false));
        assert_eq!(rec.fields["switched"], Value::Bool(false));
        assert!(!logs.iter().any(|r| r.message == "pair switched"));
    }

    #[test]
    fn test_gate_scales_with_frame_gap() {
        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let scene = SyntheticIr::default_scene();
        let (lit, dark) = scene.render();
        let (_, prior) = detector
            .detect_pair_with_prior(lit.view(), dark.view(), None, Timestamp::from_nanos(0))
            .unwrap()
            .expect("face detected");

        let mut moved = SyntheticIr::default_scene();
        moved.eyes[0].pupil_center.x += 8.0;
        let (lit2, dark2) = moved.render();

        // 266ms = 2 * 133ms, scaling the 5px base gate to 10px.
        let at = Timestamp::from_nanos(266_000_000);
        use eye_log::Value;
        use eye_log::testing::capture_logs;
        let (result, logs) = capture_logs(tracing::Level::DEBUG, || {
            detector.detect_pair_with_prior(lit2.view(), dark2.view(), Some(&prior), at)
        });
        let (eyes, _next) = result.unwrap().expect("face detected");
        assert_pupils_within(
            &eyes,
            Point2::new(298.3, 180.7),
            Point2::new(350.6, 181.2),
            0.5,
        );
        let rec = logs
            .iter()
            .find(|r| r.message == "pair accepted")
            .expect("event emitted");
        assert_eq!(rec.fields["gated"], Value::Bool(true));
        assert_eq!(rec.fields["switched"], Value::Bool(false));
    }

    #[test]
    fn test_switch_is_logged_when_no_candidate_passes_the_gate() {
        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let scene = SyntheticIr::default_scene();
        let (lit, dark) = scene.render();
        let (_, baseline_prior) = detector
            .detect_pair_with_prior(lit.view(), dark.view(), None, Timestamp::from_nanos(0))
            .unwrap()
            .expect("face detected");

        let stale_prior = PairPrior {
            at: Timestamp::from_nanos(0),
            right: (
                baseline_prior.right.0 + nalgebra::Vector2::new(40.0, 0.0),
                baseline_prior.right.1,
            ),
            left: (
                baseline_prior.left.0 + nalgebra::Vector2::new(40.0, 0.0),
                baseline_prior.left.1,
            ),
        };

        let at = Timestamp::from_nanos(68_000_000);
        use eye_log::Value;
        use eye_log::testing::capture_logs;
        let (result, logs) = capture_logs(tracing::Level::DEBUG, || {
            detector.detect_pair_with_prior(lit.view(), dark.view(), Some(&stale_prior), at)
        });
        let (eyes, _next) = result.unwrap().expect("face detected");
        assert_pupils_within(
            &eyes,
            Point2::new(290.3, 180.7),
            Point2::new(350.6, 181.2),
            0.05,
        );

        let switched_rec = logs
            .iter()
            .find(|r| r.message == "pair switched")
            .expect("event emitted");
        match switched_rec.fields["jump_px"] {
            Value::F64(j) => assert!((39.0..41.0).contains(&j), "jump_px {j} out of range"),
            ref other => panic!("expected F64 jump_px, got {other:?}"),
        }

        let accepted_rec = logs
            .iter()
            .find(|r| r.message == "pair accepted")
            .expect("event emitted");
        assert_eq!(accepted_rec.fields["switched"], Value::Bool(true));
        assert_eq!(accepted_rec.fields["gated"], Value::Bool(false));
    }

    #[test]
    fn test_dark_frame_yields_no_observations() {
        let mut detector = IrClassicDetector::new(IrClassicOptions::default());
        let scene = SyntheticIr::default_scene();
        let (_, dark) = scene.render();
        let frames = FrameSet::single(ir_frame(Illumination::IrDark, 0, 0, &dark));
        let out = detector.detect_frames(&frames).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_lit_without_prior_dark_yields_none_face() {
        let mut detector = IrClassicDetector::new(IrClassicOptions::default());
        let scene = SyntheticIr::default_scene();
        let (lit, _) = scene.render();
        let frames = FrameSet::single(ir_frame(Illumination::IrLit, 0, 0, &lit));
        let out = detector.detect_frames(&frames).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].face.is_none());
    }

    #[test]
    fn test_lit_after_dark_yields_face_with_lit_timestamp() {
        let mut det: Box<dyn Detector> =
            Box::new(IrClassicDetector::from_config(&toml::Table::new(), &nominal_rig()).unwrap());
        assert_eq!(det.name(), "ir-classic");
        let scene = SyntheticIr::default_scene();
        let (lit, dark) = scene.render();

        let dark_frames = FrameSet::single(ir_frame(Illumination::IrDark, 1_000_000_000, 0, &dark));
        let dark_out = det.detect(&dark_frames).unwrap();
        assert!(dark_out.is_empty());

        let lit_frames = FrameSet::single(ir_frame(Illumination::IrLit, 1_068_000_000, 1, &lit));
        let out = det.detect(&lit_frames).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].timestamp, Timestamp::from_nanos(1_068_000_000));
        let face = out[0].face.as_ref().expect("face detected");
        assert_eq!(face.scheme, LandmarkScheme::IR_PUPIL_PAIR);
        assert_eq!(face.eyes.len(), 2);
        assert_eq!(face.eyes[0].side, Side::Right);
        assert_eq!(face.eyes[1].side, Side::Left);
        let right = face.eyes[0].pupil.unwrap().value().center();
        let left = face.eyes[1].pupil.unwrap().value().center();
        assert!((right - Point2::new(290.3, 180.7)).norm() <= 0.05);
        assert!((left - Point2::new(350.6, 181.2)).norm() <= 0.05);
    }

    #[test]
    fn test_pair_gap_exceeded_yields_none_face() {
        let mut detector = IrClassicDetector::new(IrClassicOptions::default());
        let scene = SyntheticIr::default_scene();
        let (lit, dark) = scene.render();

        let dark_frames = FrameSet::single(ir_frame(Illumination::IrDark, 900_000_000, 0, &dark));
        detector.detect_frames(&dark_frames).unwrap();

        let lit_frames = FrameSet::single(ir_frame(Illumination::IrLit, 1_068_000_000, 1, &lit));
        let out = detector.detect_frames(&lit_frames).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].face.is_none());
    }

    #[test]
    fn test_warm_up_darks_then_lit_pairs_with_latest_dark() {
        let mut detector = IrClassicDetector::new(IrClassicOptions::default());
        let scene = SyntheticIr::default_scene();
        let (lit, clean_dark) = scene.render();

        let blocked_roi = crate::image::Roi {
            x: 240,
            y: 130,
            width: 200,
            height: 100,
        };
        let mut blocked_dark_data = clean_dark.data().to_vec();
        for y in blocked_roi.y..blocked_roi.y + blocked_roi.height {
            for x in blocked_roi.x..blocked_roi.x + blocked_roi.width {
                blocked_dark_data[(y * clean_dark.width() + x) as usize] = 255;
            }
        }
        let blocked_dark = eye_core::image::GrayImage::new(
            clean_dark.width(),
            clean_dark.height(),
            blocked_dark_data,
        )
        .unwrap();

        let darks = FrameSet::single(ir_frame(Illumination::IrDark, 0, 0, &blocked_dark));
        detector.detect_frames(&darks).unwrap();
        let darks = FrameSet::single(ir_frame(Illumination::IrDark, 68_000_000, 1, &blocked_dark));
        detector.detect_frames(&darks).unwrap();
        let darks = FrameSet::single(ir_frame(Illumination::IrDark, 136_000_000, 2, &clean_dark));
        detector.detect_frames(&darks).unwrap();

        let lits = FrameSet::single(ir_frame(Illumination::IrLit, 204_000_000, 3, &lit));
        let out = detector.detect_frames(&lits).unwrap();
        assert_eq!(out.len(), 1);
        let face = out[0].face.as_ref().expect("face detected");
        let right = face.eyes[0].pupil.unwrap().value().center();
        let left = face.eyes[1].pupil.unwrap().value().center();
        assert!((right - Point2::new(290.3, 180.7)).norm() <= 0.05);
        assert!((left - Point2::new(350.6, 181.2)).norm() <= 0.05);
    }

    #[test]
    fn test_dual_mode_set_detects_ir_and_ignores_rgb() {
        let mut detector = IrClassicDetector::new(IrClassicOptions::default());
        let scene = SyntheticIr::default_scene();
        let (lit, dark) = scene.render();

        let darks = FrameSet::single(ir_frame(Illumination::IrDark, 0, 0, &dark));
        detector.detect_frames(&darks).unwrap();

        let dual = FrameSet::new(vec![
            ir_frame(Illumination::IrLit, 68_000_000, 1, &lit),
            rgb_frame(0),
        ])
        .unwrap();
        let out = detector.detect_frames(&dual).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].camera, CameraId::from("ir"));
        assert!(out[0].face.is_some());
    }

    #[test]
    fn test_accepts_only_gray_ir_lit_or_dark() {
        for &format in &[PixelFormat::Gray8, PixelFormat::Rgb8, PixelFormat::Mjpeg] {
            for &illumination in &[
                Illumination::Ambient,
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::Unknown,
            ] {
                let expected = format == PixelFormat::Gray8
                    && matches!(illumination, Illumination::IrLit | Illumination::IrDark);
                assert_eq!(
                    IrClassicDetector::accepts_frame(format, illumination),
                    expected,
                    "format {format:?} illumination {illumination:?}"
                );
            }
        }
    }

    fn nominal_rig() -> Rig {
        let camera = eye_core::CameraModel {
            id: CameraId::from("ir"),
            width: 640,
            height: 360,
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 180.0,
            distortion: [0.0; 5],
            screen_from_camera: nalgebra::Isometry3::identity(),
        };
        let screen = eye_core::ScreenModel {
            output: eye_core::OutputId::from("eDP-1"),
            size_mm: nalgebra::Vector2::new(310.0, 170.0),
            size_px: (3840, 2160),
            scale: 2.0,
        };
        Rig::new(vec![camera], screen).unwrap()
    }

    #[test]
    fn test_from_config_empty_table_uses_defaults() {
        let table = toml::Table::new();
        let detector = IrClassicDetector::from_config(&table, &nominal_rig()).unwrap();
        assert_eq!(detector.options.max_pair_gap_ms, 100.0);
    }

    #[test]
    fn test_from_config_rejects_unknown_option() {
        let mut table = toml::Table::new();
        table.insert("not_a_real_option".into(), 1.into());
        let err = IrClassicDetector::from_config(&table, &nominal_rig()).unwrap_err();
        assert!(matches!(err, StageError::Config(_)));
    }

    #[test]
    fn test_from_config_rejects_threshold_percentile() {
        let mut table = toml::Table::new();
        table.insert("threshold_percentile".into(), 99.9.into());
        let err = IrClassicDetector::from_config(&table, &nominal_rig()).unwrap_err();
        assert!(matches!(err, StageError::Config(_)));
    }

    #[test]
    #[ignore = "needs EYE_RECORDING"]
    fn test_recording_detection_rate() {
        let dir = std::env::var("EYE_RECORDING").expect("EYE_RECORDING must be set");
        let index_path = std::path::Path::new(&dir).join("index.jsonl");
        let index = std::fs::read_to_string(&index_path).expect("index.jsonl readable");

        let mut detector = IrClassicDetector::new(IrClassicOptions::default());
        let mut last_dark: Option<Frame> = None;
        let mut lit_count = 0u64;
        let mut detected = 0u64;
        let mut sigmas = Vec::new();

        for line in index.lines() {
            let illumination = match json_string_field(line, "illumination") {
                Some("ir_lit") => Illumination::IrLit,
                Some("ir_dark") => Illumination::IrDark,
                _ => continue,
            };
            let seq = json_number_field(line, "seq").unwrap_or(0);
            let ts_ns = json_number_field(line, "timestamp").unwrap_or(0);
            let path = std::path::Path::new(&dir).join(format!("frames/ir/{seq:08}.pgm"));
            let pixels = read_pgm(&path);
            let frame = ir_frame(illumination, ts_ns, seq, &pixels);

            let frames = FrameSet::single(frame.clone());
            let out = detector.detect_frames(&frames).unwrap();

            if illumination == Illumination::IrDark {
                last_dark = Some(frame);
                continue;
            }

            lit_count += 1;
            let face = out.first().and_then(|obs| obs.face.as_ref());
            if let Some(face) = face {
                detected += 1;
                for eye in &face.eyes {
                    if let Some(pupil) = eye.pupil {
                        sigmas.push(pupil.sigma());
                    }
                }
            } else if let Some(dark) = &last_dark {
                let diff = saturating_diff(
                    GrayView::from_frame(&frame).unwrap(),
                    GrayView::from_frame(dark).unwrap(),
                )
                .unwrap();
                let (cands, _counts) = blob::candidates(diff.view(), &detector.options);
                println!(
                    "seq {seq} dropout: candidates = {:?}",
                    cands
                        .iter()
                        .map(|c| (c.area, c.contrast, c.iris_mean / c.outer_mean))
                        .collect::<Vec<_>>()
                );
            }
        }

        let rate = if lit_count > 0 {
            100.0 * detected as f64 / lit_count as f64
        } else {
            0.0
        };
        println!("detection rate: {detected}/{lit_count} ({rate:.1}%)");
        println!("sigma distribution: {sigmas:?}");
    }

    fn json_string_field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
        let needle = format!("\"{key}\":\"");
        let start = line.find(&needle)? + needle.len();
        let end = start + line[start..].find('"')?;
        Some(&line[start..end])
    }

    fn json_number_field(line: &str, key: &str) -> Option<u64> {
        let needle = format!("\"{key}\":");
        let start = line.find(&needle)? + needle.len();
        let end = start
            + line[start..]
                .find(|c: char| !(c.is_ascii_digit()))
                .unwrap_or(line.len() - start);
        line[start..end].parse().ok()
    }

    fn read_pgm(path: &std::path::Path) -> eye_core::image::GrayImage {
        let bytes = std::fs::read(path).expect("pgm readable");
        let mut pos = 0;
        let mut fields = Vec::new();
        while fields.len() < 4 {
            while bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            if bytes[pos] == b'#' {
                while bytes[pos] != b'\n' {
                    pos += 1;
                }
                continue;
            }
            let start = pos;
            while !bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            fields.push(String::from_utf8_lossy(&bytes[start..pos]).to_string());
        }
        pos += 1;
        assert_eq!(fields[0], "P5");
        let width: u32 = fields[1].parse().unwrap();
        let height: u32 = fields[2].parse().unwrap();
        let data = bytes[pos..pos + (width * height) as usize].to_vec();
        eye_core::image::GrayImage::new(width, height, data).unwrap()
    }

    #[test]
    fn test_logs_pair_rejected_at_debug() {
        use eye_log::testing::capture_logs;
        use eye_log::{Level as LogLevel, Value};

        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let mut scene = SyntheticIr::default_scene();
        scene.eyes.truncate(1);
        let (lit, dark) = scene.render();
        let (_, logs) = capture_logs(tracing::Level::TRACE, || {
            detector.detect_pair(lit.view(), dark.view())
        });
        let rec = logs
            .iter()
            .find(|r| r.message == "pair rejected" && r.target == "eye_detect::ir")
            .expect("event emitted");
        assert_eq!(rec.level, LogLevel::Debug);
        assert_eq!(
            rec.fields[field::REASON],
            Value::Str("single_candidate".into())
        );
        assert_eq!(rec.fields["candidates"], Value::U64(1));
        let mut keys: Vec<&str> = rec.fields.keys().map(String::as_str).collect();
        keys.sort_unstable();
        let mut expected = vec![
            field::REASON,
            "candidates",
            "rejected_area",
            "rejected_aspect",
            "rejected_contrast",
            "rejected_iris_ratio",
            "components",
        ];
        expected.sort_unstable();
        assert_eq!(keys, expected);

        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let mut scene = SyntheticIr::default_scene();
        for eye in &mut scene.eyes {
            eye.pupil_level = 55;
        }
        let (lit, dark) = scene.render();
        let (_, logs) = capture_logs(tracing::Level::TRACE, || {
            detector.detect_pair(lit.view(), dark.view())
        });
        let rec = logs
            .iter()
            .find(|r| r.message == "pair rejected" && r.target == "eye_detect::ir")
            .expect("event emitted");
        assert_eq!(
            rec.fields[field::REASON],
            Value::Str("no_candidates".into())
        );
        assert_eq!(rec.fields["candidates"], Value::U64(0));

        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let mut scene = SyntheticIr::default_scene();
        scene.eyes = vec![
            SyntheticEye::at(Point2::new(260.3, 180.7)),
            SyntheticEye::at(Point2::new(380.3, 181.2)),
        ];
        let (lit, dark) = scene.render();
        let (_, logs) = capture_logs(tracing::Level::TRACE, || {
            detector.detect_pair(lit.view(), dark.view())
        });
        let rec = logs
            .iter()
            .find(|r| r.message == "pair rejected" && r.target == "eye_detect::ir")
            .expect("event emitted");
        assert_eq!(rec.fields[field::REASON], Value::Str("no_pair".into()));
        assert_eq!(rec.fields["candidates"], Value::U64(2));

        let pupil_rec = logs
            .iter()
            .find(|r| r.message == "pair rejected" && r.target == "eye_detect::ir::pupil")
            .expect("pupil event emitted");
        assert_eq!(pupil_rec.level, LogLevel::Trace);
        assert_eq!(
            pupil_rec.fields[field::REASON],
            Value::Str("separation".into())
        );
        match pupil_rec.fields["dist_px"] {
            Value::F64(d) => assert!((119.0..121.0).contains(&d), "dist_px {d} out of range"),
            ref other => panic!("expected F64 dist_px, got {other:?}"),
        }
    }

    #[test]
    fn test_logs_pair_rejected_carries_gate_counts() {
        use eye_log::testing::capture_logs;
        use eye_log::{Level as LogLevel, Value};

        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let mut scene = SyntheticIr::default_scene();
        scene.eyes.truncate(1);
        let (lit, dark) = scene.render();
        let (_, logs) = capture_logs(tracing::Level::TRACE, || {
            detector.detect_pair(lit.view(), dark.view())
        });
        let rec = logs
            .iter()
            .find(|r| r.message == "pair rejected" && r.target == "eye_detect::ir")
            .expect("event emitted");
        assert_eq!(rec.level, LogLevel::Debug);
        for key in [
            "rejected_area",
            "rejected_aspect",
            "rejected_contrast",
            "rejected_iris_ratio",
            "components",
        ] {
            assert!(
                matches!(rec.fields[key], Value::U64(_)),
                "expected U64 {key}, got {:?}",
                rec.fields[key]
            );
        }
    }

    #[test]
    fn test_logs_pupil_at_trace() {
        use eye_log::testing::capture_logs;
        use eye_log::{Level as LogLevel, Value};

        let detector = IrClassicDetector::new(IrClassicOptions::default());
        let scene = SyntheticIr::default_scene();
        let (lit, dark) = scene.render();
        let (_, logs) = capture_logs(tracing::Level::TRACE, || {
            detector.detect_pair(lit.view(), dark.view())
        });

        let pupils: Vec<_> = logs.iter().filter(|r| r.message == "pupil").collect();
        assert_eq!(pupils.len(), 2);
        assert_eq!(pupils[0].level, LogLevel::Trace);
        assert_eq!(pupils[0].fields["side"], Value::Str("right".into()));
        assert_eq!(pupils[1].fields["side"], Value::Str("left".into()));

        for (rec, want_x) in pupils.iter().zip([290.3, 350.6]) {
            match (&rec.fields["x"], &rec.fields["sigma"], &rec.fields["glint"]) {
                (Value::F64(x), Value::F64(sigma), Value::Bool(glint)) => {
                    assert!(
                        (x - want_x).abs() <= 0.05,
                        "x {x} not within 0.05 of {want_x}"
                    );
                    assert!(*sigma > 0.0 && *sigma <= 0.38, "sigma {sigma} out of range");
                    assert!(!glint);
                }
                other => panic!("unexpected field types: {other:?}"),
            }
        }
    }

    #[test]
    fn test_logs_glint_at_trace() {
        use eye_log::testing::capture_logs;
        use eye_log::{Level as LogLevel, Value};

        let mut det: Box<dyn Detector> =
            Box::new(IrClassicDetector::from_config(&toml::Table::new(), &nominal_rig()).unwrap());
        let mut scene = SyntheticIr::default_scene();
        let offset = nalgebra::Vector2::new(0.7, -0.4);
        let right_truth = scene.eyes[0].pupil_center;
        let left_truth = scene.eyes[1].pupil_center;
        scene.eyes[0].glint = Some((right_truth + offset, 0.6, 255.0));
        scene.eyes[1].glint = Some((left_truth + offset, 0.6, 255.0));
        let (lit, dark) = scene.render();

        let dark_frames = FrameSet::single(ir_frame(Illumination::IrDark, 0, 0, &dark));
        det.detect(&dark_frames).unwrap();
        let lit_frames = FrameSet::single(ir_frame(Illumination::IrLit, 68_000_000, 1, &lit));
        let (_, logs) = capture_logs(tracing::Level::TRACE, || det.detect(&lit_frames).unwrap());

        let glints: Vec<_> = logs.iter().filter(|r| r.message == "glint").collect();
        assert_eq!(glints.len(), 2);
        for rec in &glints {
            assert_eq!(rec.level, LogLevel::Trace);
            assert_eq!(rec.fields["saturated"], Value::Bool(false));
            assert_eq!(rec.fields["sigma"], Value::F64(0.19));
        }

        let pupils: Vec<_> = logs.iter().filter(|r| r.message == "pupil").collect();
        assert_eq!(pupils.len(), 2);
        for rec in &pupils {
            assert_eq!(rec.fields["glint"], Value::Bool(true));
        }
    }

    #[test]
    fn test_logs_fit_fallback_at_debug() {
        use eye_log::testing::capture_logs;
        use eye_log::{Level as LogLevel, Value};

        let options = IrClassicOptions {
            edge_rays: 4,
            glints: false,
            ..IrClassicOptions::default()
        };
        let detector = IrClassicDetector::new(options);
        let scene = SyntheticIr::default_scene();
        let (lit, dark) = scene.render();
        let (eyes, logs) = capture_logs(tracing::Level::TRACE, || {
            detector.detect_pair(lit.view(), dark.view())
        });
        let eyes = eyes.unwrap().expect("face detected");

        let fallbacks: Vec<_> = logs
            .iter()
            .filter(|r| r.message == "ellipse fit fell back to circle")
            .collect();
        assert_eq!(fallbacks.len(), 2);
        for rec in &fallbacks {
            assert_eq!(rec.level, LogLevel::Debug);
            assert_eq!(
                rec.fields[field::REASON],
                Value::Str("too_few_edges".into())
            );
            match rec.fields["edge_points"] {
                Value::U64(n) => assert!(n <= 4, "edge_points {n}"),
                ref other => panic!("expected U64 edge_points, got {other:?}"),
            }
        }
        for eye in &eyes {
            assert_eq!(eye.pupil.unwrap().sigma(), 0.5);
        }
    }

    #[test]
    fn test_logs_lit_frame_unpaired_at_debug() {
        use eye_log::testing::capture_logs;
        use eye_log::{Level as LogLevel, Value};

        let mut detector = IrClassicDetector::new(IrClassicOptions::default());
        let scene = SyntheticIr::default_scene();
        let (lit, _dark) = scene.render();
        let frames = FrameSet::single(ir_frame(Illumination::IrLit, 0, 0, &lit));
        let (_, logs) = capture_logs(tracing::Level::TRACE, || {
            detector.detect_frames(&frames).unwrap()
        });
        let rec = logs
            .iter()
            .find(|r| r.message == "lit frame unpaired")
            .expect("event emitted");
        assert_eq!(rec.level, LogLevel::Debug);
        assert_eq!(rec.fields[field::REASON], Value::Str("no_dark".into()));

        let mut detector = IrClassicDetector::new(IrClassicOptions::default());
        let scene = SyntheticIr::default_scene();
        let (lit, dark) = scene.render();
        let dark_frames = FrameSet::single(ir_frame(Illumination::IrDark, 900_000_000, 0, &dark));
        detector.detect_frames(&dark_frames).unwrap();
        let lit_frames = FrameSet::single(ir_frame(Illumination::IrLit, 1_068_000_000, 1, &lit));
        let (_, logs) = capture_logs(tracing::Level::TRACE, || {
            detector.detect_frames(&lit_frames).unwrap()
        });
        let rec = logs
            .iter()
            .find(|r| r.message == "lit frame unpaired")
            .expect("event emitted");
        assert_eq!(rec.fields[field::REASON], Value::Str("dark_gap".into()));
        assert_eq!(rec.fields["gap_ns"], Value::I64(168_000_000));
        assert_eq!(rec.fields["max_pair_gap_ms"], Value::F64(100.0));
    }

    #[test]
    fn test_logs_dark_frame_stored_at_debug() {
        use eye_log::Level as LogLevel;
        use eye_log::testing::capture_logs;

        let mut detector = IrClassicDetector::new(IrClassicOptions::default());
        let scene = SyntheticIr::default_scene();
        let (_, dark) = scene.render();
        let frames = FrameSet::single(ir_frame(Illumination::IrDark, 0, 0, &dark));
        let (_, logs) = capture_logs(tracing::Level::TRACE, || {
            detector.detect_frames(&frames).unwrap()
        });

        let rec = logs
            .iter()
            .find(|r| r.message == "dark frame stored")
            .expect("event emitted");
        assert_eq!(rec.level, LogLevel::Debug);
        assert!(!logs.iter().any(|r| r.message == "pair rejected"));
    }

    #[test]
    fn test_logs_frame_elapsed_at_trace() {
        use eye_log::testing::capture_logs;
        use eye_log::{Level as LogLevel, Value};

        let mut detector = IrClassicDetector::new(IrClassicOptions::default());
        let scene = SyntheticIr::default_scene();
        let (lit, dark) = scene.render();
        let dark_frames = FrameSet::single(ir_frame(Illumination::IrDark, 0, 0, &dark));
        detector.detect_frames(&dark_frames).unwrap();
        let lit_frames = FrameSet::single(ir_frame(Illumination::IrLit, 68_000_000, 1, &lit));
        let (_, logs) = capture_logs(tracing::Level::TRACE, || {
            detector.detect_frames(&lit_frames).unwrap()
        });

        let rec = logs
            .iter()
            .find(|r| r.message == "ir-classic frame")
            .expect("event emitted");
        assert_eq!(rec.level, LogLevel::Trace);
        assert!(matches!(rec.fields[field::ELAPSED_US], Value::U64(_)));
        assert_eq!(rec.fields["face"], Value::Bool(true));
    }

    #[test]
    fn test_logs_frame_not_accepted_at_trace() {
        use eye_log::testing::capture_logs;
        use eye_log::{Level as LogLevel, Value};

        let mut detector = IrClassicDetector::new(IrClassicOptions::default());
        let scene = SyntheticIr::default_scene();
        let (_lit, dark) = scene.render();
        let dual = FrameSet::new(vec![
            ir_frame(Illumination::IrDark, 0, 0, &dark),
            rgb_frame(0),
        ])
        .unwrap();
        let (_, logs) = capture_logs(tracing::Level::TRACE, || {
            detector.detect_frames(&dual).unwrap()
        });

        let rec = logs
            .iter()
            .find(|r| r.message == "frame not accepted")
            .expect("event emitted");
        assert_eq!(rec.level, LogLevel::Trace);
        assert_eq!(rec.fields["format"], Value::Str("Rgb8".into()));
        match rec.fields[field::ILLUMINATION] {
            Value::Str(ref s) => assert!(!s.is_empty()),
            ref other => panic!("expected Str illumination, got {other:?}"),
        }
    }
}
