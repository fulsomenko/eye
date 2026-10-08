pub mod blob;
pub mod glint;
pub mod pupil;
#[cfg(test)]
mod testutil;

use std::collections::HashMap;

use eye_core::image::GrayView;
use eye_core::observation::SCHEME_IR_PUPIL_PAIR;
use eye_core::stage::{Detector, StageError};
use eye_core::{
    CameraId, EyeObservation, FaceObservation, Frame, FrameSet, Illumination, Observations,
    PixelFormat, Side,
};

use crate::DetectError;
use crate::image::saturating_diff;
use crate::options::parse_options;

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct IrClassicOptions {
    pub max_pair_gap_ms: f64,
    pub background_size: usize,
    pub threshold_percentile: f64,
    pub min_threshold: u8,
    pub pupil_area_px: [u32; 2],
    pub min_aspect: f64,
    pub min_pupil_contrast: f64,
    pub max_iris_ratio: f64,
    pub pair_separation_px: [f64; 2],
    pub max_pair_tilt_deg: f64,
    pub edge_rays: usize,
    pub max_candidates: usize,
}

impl Default for IrClassicOptions {
    fn default() -> Self {
        Self {
            max_pair_gap_ms: 100.0,
            background_size: 15,
            threshold_percentile: 99.9,
            min_threshold: 30,
            pupil_area_px: [7, 120],
            min_aspect: 0.5,
            min_pupil_contrast: 1.3,
            max_iris_ratio: 0.8,
            pair_separation_px: [35.0, 95.0],
            max_pair_tilt_deg: 30.0,
            edge_rays: 16,
            max_candidates: 64,
        }
    }
}

#[derive(Debug)]
pub struct IrClassicDetector {
    options: IrClassicOptions,
    last_dark: HashMap<CameraId, Frame>,
}

impl IrClassicDetector {
    pub const NAME: &'static str = "ir-classic";

    pub fn new(options: IrClassicOptions) -> Self {
        Self {
            options,
            last_dark: HashMap::new(),
        }
    }

    pub fn from_config(table: &toml::Table, _rig: &eye_core::Rig) -> Result<Self, StageError> {
        Ok(Self::new(parse_options(Self::NAME, table)?))
    }

    pub fn accepts_frame(format: PixelFormat, illumination: Illumination) -> bool {
        format == PixelFormat::Gray8
            && matches!(illumination, Illumination::IrLit | Illumination::IrDark)
    }

    /// Pure core, unit-testable without frames. Returns `[right, left]`.
    pub fn detect_pair(
        &self,
        lit: GrayView<'_>,
        dark: GrayView<'_>,
    ) -> Result<Option<[EyeObservation; 2]>, DetectError> {
        let diff = saturating_diff(lit, dark)?;
        let diff = diff.view();
        let cands = blob::candidates(diff, &self.options);
        let Some((i, j)) = pupil::select_pair(&cands, &self.options) else {
            return Ok(None);
        };
        let pupil_i = pupil::pupil_from_candidate(diff, &cands[i], &self.options)?;
        let pupil_j = pupil::pupil_from_candidate(diff, &cands[j], &self.options)?;

        let (right, left) = if pupil_i.value().center().x <= pupil_j.value().center().x {
            (pupil_i, pupil_j)
        } else {
            (pupil_j, pupil_i)
        };

        let mut right_eye = EyeObservation::new(Side::Right);
        right_eye.pupil = Some(right);
        let mut left_eye = EyeObservation::new(Side::Left);
        left_eye.pupil = Some(left);
        Ok(Some([right_eye, left_eye]))
    }

    fn paired_dark(&self, lit: &Frame) -> Option<&Frame> {
        let h = lit.header();
        let dark = self.last_dark.get(&h.camera)?;
        let gap = h.timestamp.nanos_since(dark.header().timestamp);
        (gap > 0 && gap as f64 <= self.options.max_pair_gap_ms * 1e6).then_some(dark)
    }

    fn detect_frames(&mut self, frames: &FrameSet) -> Result<Vec<Observations>, DetectError> {
        let mut out = Vec::new();
        for frame in frames.frames() {
            let h = frame.header();
            if !Self::accepts_frame(h.format, h.illumination) {
                continue;
            }
            if h.illumination == Illumination::IrDark {
                self.last_dark.insert(h.camera.clone(), frame.clone());
                continue;
            }
            let face = match self.paired_dark(frame) {
                Some(dark) => self
                    .detect_pair(GrayView::from_frame(frame)?, GrayView::from_frame(dark)?)?
                    .map(face_from_pupils),
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
        scheme: SCHEME_IR_PUPIL_PAIR,
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
            assert!(sigma > 0.0 && sigma <= 0.3, "sigma {sigma} out of range");
        }
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
        let cands = blob::candidates(diff.view(), &IrClassicOptions::default());
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
        assert_eq!(face.scheme, SCHEME_IR_PUPIL_PAIR);
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
                let cands = blob::candidates(diff.view(), &detector.options);
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
}
