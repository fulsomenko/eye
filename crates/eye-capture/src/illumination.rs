use std::time::Duration;

use eye_core::{CameraInfo, Frame, Illumination, PixelFormat, Timestamp};

use crate::{
    CaptureError,
    source::FrameSource,
    uvc_meta::{IlluminationMeta, MetaRecord},
};

#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TaggerConfig {
    /// |mean(t) - mean(t-1)| at or above this is an alternation vote.
    pub enter_contrast: f64,
    /// |mean(t) - mean(t-1)| below this is a flat (non-alternating) step.
    pub exit_contrast: f64,
    pub lock_votes: u32,
    pub relock_votes: u32,
    pub flat_frames: u32,
    pub sample_step: usize,
    /// Consecutive Gray8 frames without a matching FrameIllumination flag before metadata is abandoned.
    pub meta_max_misses: u32,
}

impl Default for TaggerConfig {
    fn default() -> Self {
        Self {
            enter_contrast: 10.0,
            exit_contrast: 5.0,
            lock_votes: 3,
            relock_votes: 3,
            flat_frames: 15,
            sample_step: 4,
            meta_max_misses: 8,
        }
    }
}

impl TaggerConfig {
    /// `sample_step >= 1`, `0 < exit_contrast <= enter_contrast` (finite), `lock_votes`, `relock_votes`, `meta_max_misses >= 1`.
    pub fn validate(&self) -> Result<(), String> {
        if self.sample_step < 1 {
            return Err("sample_step must be >= 1".into());
        }
        if !(self.exit_contrast.is_finite() && self.enter_contrast.is_finite()) {
            return Err("enter_contrast and exit_contrast must be finite".into());
        }
        if !(self.exit_contrast > 0.0 && self.exit_contrast <= self.enter_contrast) {
            return Err("exit_contrast must be > 0 and <= enter_contrast".into());
        }
        if self.lock_votes < 1 {
            return Err("lock_votes must be >= 1".into());
        }
        if self.relock_votes < 1 {
            return Err("relock_votes must be >= 1".into());
        }
        if self.meta_max_misses < 1 {
            return Err("meta_max_misses must be >= 1".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaggerState {
    Searching { candidate: Option<u8>, votes: u32 },
    Locked { lit_parity: u8, contradictions: u32 },
    NotAlternating,
}

#[derive(Debug)]
pub struct IlluminationTagger {
    cfg: TaggerConfig,
    state: TaggerState,
    prev: Option<(u64, f64)>,
    flat_run: u32,
}

enum Evidence {
    Vote(u8),
    Flat,
    None,
}

impl IlluminationTagger {
    pub fn new(cfg: TaggerConfig) -> Self {
        Self {
            cfg,
            state: TaggerState::Searching {
                candidate: None,
                votes: 0,
            },
            prev: None,
            flat_run: 0,
        }
    }

    pub fn state(&self) -> TaggerState {
        self.state
    }

    /// Brightness core: tag frame `seq` given its mean brightness.
    pub fn tag(&mut self, seq: u64, mean: f64) -> Illumination {
        let evidence = match self.prev {
            Some((prev_seq, prev_mean)) if prev_seq + 1 == seq => {
                let d = mean - prev_mean;
                if d.abs() >= self.cfg.enter_contrast {
                    Evidence::Vote(if d > 0.0 {
                        (seq % 2) as u8
                    } else {
                        ((seq + 1) % 2) as u8
                    })
                } else if d.abs() < self.cfg.exit_contrast {
                    Evidence::Flat
                } else {
                    Evidence::None
                }
            }
            _ => Evidence::None,
        };
        self.prev = Some((seq, mean));
        match evidence {
            Evidence::Flat => self.flat_run += 1,
            Evidence::Vote(_) => self.flat_run = 0,
            Evidence::None => {}
        }
        self.state = self.next_state(evidence);
        match self.state {
            TaggerState::Locked { lit_parity, .. } if (seq % 2) as u8 == lit_parity => {
                Illumination::IrLit
            }
            TaggerState::Locked { .. } => Illumination::IrDark,
            TaggerState::NotAlternating => Illumination::Ambient,
            TaggerState::Searching { .. } => Illumination::Unknown,
        }
    }

    /// Metadata says frame `seq` is lit / dark: lock the parity to it and return its tag.
    pub fn observe(&mut self, seq: u64, mean: f64, lit: bool) -> Illumination {
        let parity = (seq % 2) as u8;
        let lit_parity = if lit { parity } else { 1 - parity };
        self.prev = Some((seq, mean));
        self.flat_run = 0;
        self.state = TaggerState::Locked {
            lit_parity,
            contradictions: 0,
        };
        if lit {
            Illumination::IrLit
        } else {
            Illumination::IrDark
        }
    }

    /// Brightness-tags Gray8 frames in place; other formats are left untouched.
    pub fn tag_frame(&mut self, frame: &mut Frame) {
        let h = frame.header();
        if h.format != PixelFormat::Gray8 {
            return;
        }
        let (seq, mean) = (
            h.seq,
            mean_brightness(frame.data(), h.width, h.height, self.cfg.sample_step),
        );
        let illumination = self.tag(seq, mean);
        frame.set_illumination(illumination);
    }

    fn next_state(&self, evidence: Evidence) -> TaggerState {
        use TaggerState::*;
        let flat_out = self.flat_run >= self.cfg.flat_frames;
        match (self.state, evidence) {
            (Searching { candidate, votes }, Evidence::Vote(p)) => {
                let votes = if candidate == Some(p) { votes + 1 } else { 1 };
                if votes >= self.cfg.lock_votes {
                    Locked {
                        lit_parity: p,
                        contradictions: 0,
                    }
                } else {
                    Searching {
                        candidate: Some(p),
                        votes,
                    }
                }
            }
            (Locked { lit_parity, .. }, Evidence::Vote(p)) if p == lit_parity => Locked {
                lit_parity,
                contradictions: 0,
            },
            (
                Locked {
                    lit_parity,
                    contradictions,
                },
                Evidence::Vote(p),
            ) => {
                if contradictions + 1 >= self.cfg.relock_votes {
                    Locked {
                        lit_parity: p,
                        contradictions: 0,
                    }
                } else {
                    Locked {
                        lit_parity,
                        contradictions: contradictions + 1,
                    }
                }
            }
            (NotAlternating, Evidence::Vote(p)) if self.cfg.lock_votes <= 1 => Locked {
                lit_parity: p,
                contradictions: 0,
            },
            (NotAlternating, Evidence::Vote(p)) => Searching {
                candidate: Some(p),
                votes: 1,
            },
            (Searching { .. } | Locked { .. }, Evidence::Flat) if flat_out => NotAlternating,
            (state, _) => state,
        }
    }
}

pub fn mean_brightness(data: &[u8], width: u32, height: u32, step: usize) -> f64 {
    let (mut sum, mut n) = (0u64, 0u64);
    for row in data
        .chunks(width as usize)
        .take(height as usize)
        .step_by(step)
    {
        for v in row.iter().step_by(step) {
            sum += u64::from(*v);
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum as f64 / n as f64 }
}

/// A metadata buffer matches a frame when their timestamps differ by at most this.
pub const META_MATCH_TOLERANCE: Duration = Duration::from_millis(5);

#[derive(Debug)]
struct MetaMatcher {
    meta: Box<dyn IlluminationMeta>,
    pending: Option<MetaRecord>,
    misses: u32,
}

impl MetaMatcher {
    fn lookup(&mut self, t: Timestamp) -> Result<Option<bool>, CaptureError> {
        let tol = META_MATCH_TOLERANCE.as_nanos() as i64;
        loop {
            let record = match self.pending.take() {
                Some(r) => r,
                None => self.meta.next_record()?,
            };
            let d = record.timestamp.nanos_since(t);
            if d < -tol {
                continue;
            }
            if d > tol {
                self.pending = Some(record);
                return Ok(None);
            }
            return Ok(record.lit);
        }
    }
}

#[derive(Debug)]
pub struct TaggedSource<S: FrameSource> {
    inner: S,
    tagger: IlluminationTagger,
    meta: Option<MetaMatcher>,
}

impl<S: FrameSource> TaggedSource<S> {
    /// Brightness tagging only. `CaptureError::Config` when `cfg.validate()` fails.
    pub fn new(inner: S, cfg: TaggerConfig) -> Result<Self, CaptureError> {
        cfg.validate().map_err(|reason| CaptureError::Config {
            camera: inner.camera().id.to_string(),
            reason,
        })?;
        Ok(Self {
            inner,
            tagger: IlluminationTagger::new(cfg),
            meta: None,
        })
    }

    /// FrameIllumination metadata first, brightness as fallback.
    pub fn with_metadata(
        inner: S,
        meta: Box<dyn IlluminationMeta>,
        cfg: TaggerConfig,
    ) -> Result<Self, CaptureError> {
        let mut source = Self::new(inner, cfg)?;
        source.meta = Some(MetaMatcher {
            meta,
            pending: None,
            misses: 0,
        });
        Ok(source)
    }

    pub fn tagger(&self) -> &IlluminationTagger {
        &self.tagger
    }

    /// False once metadata was abandoned (or never given).
    pub fn metadata_active(&self) -> bool {
        self.meta.is_some()
    }

    fn meta_flag(&mut self, t: Timestamp) -> Option<bool> {
        let m = self.meta.as_mut()?;
        match m.lookup(t) {
            Ok(Some(lit)) => {
                m.misses = 0;
                Some(lit)
            }
            Ok(None) => {
                m.misses += 1;
                if m.misses >= self.tagger.cfg.meta_max_misses {
                    tracing::warn!(
                        camera = %self.inner.camera().id,
                        misses = m.misses,
                        "no FrameIllumination metadata; tagging by brightness"
                    );
                    self.meta = None;
                }
                None
            }
            Err(err) => {
                tracing::warn!(
                    camera = %self.inner.camera().id,
                    %err,
                    "metadata stream failed; tagging by brightness"
                );
                self.meta = None;
                None
            }
        }
    }
}

impl<S: FrameSource> FrameSource for TaggedSource<S> {
    fn camera(&self) -> &CameraInfo {
        self.inner.camera()
    }

    fn next_frame(&mut self) -> Result<Frame, CaptureError> {
        let mut frame = self.inner.next_frame()?;
        let h = frame.header();
        if h.format != PixelFormat::Gray8 {
            return Ok(frame);
        }
        let (seq, t) = (h.seq, h.timestamp);
        let mean = mean_brightness(frame.data(), h.width, h.height, self.tagger.cfg.sample_step);
        let tag = match self.meta_flag(t) {
            Some(lit) => self.tagger.observe(seq, mean, lit),
            None => self.tagger.tag(seq, mean),
        };
        frame.set_illumination(tag);
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use approx::assert_abs_diff_eq;
    use eye_core::{PixelFormat, Timestamp};
    use proptest::prelude::*;

    use super::*;
    use crate::testing::{VecSource, camera_info, gray_frame, mjpeg_frame};

    const T: u64 = 68_000_000;

    fn run(tagger: &mut IlluminationTagger, first_seq: u64, means: &[f64]) -> Vec<Illumination> {
        means
            .iter()
            .enumerate()
            .map(|(i, &m)| tagger.tag(first_seq + i as u64, m))
            .collect()
    }

    fn rec(k: u64, lit: Option<bool>) -> Option<MetaRecord> {
        Some(MetaRecord {
            timestamp: Timestamp::from_nanos(k * T + 2_000),
            lit,
        })
    }

    fn ir_frames(means: &[u8]) -> Vec<Frame> {
        means
            .iter()
            .enumerate()
            .map(|(k, &m)| gray_frame("ir", k as u64 + 1, (k as u64 + 1) * T, 64, 36, m))
            .collect()
    }

    #[derive(Debug)]
    struct VecMeta(VecDeque<Option<MetaRecord>>);

    impl IlluminationMeta for VecMeta {
        fn next_record(&mut self) -> Result<MetaRecord, CaptureError> {
            match self.0.pop_front().flatten() {
                Some(r) => Ok(r),
                None => Err(CaptureError::Timeout {
                    camera: "ir".into(),
                    timeout: Duration::from_millis(500),
                }),
            }
        }
    }

    // --- brightness ---

    #[test]
    fn test_mean_brightness_uniform_image_equals_value() {
        let data = vec![45u8; 640 * 360];
        assert_abs_diff_eq!(mean_brightness(&data, 640, 360, 4), 45.0);
    }

    #[test]
    fn test_mean_brightness_subsampled_close_to_full_mean() {
        let mut data = vec![0u8; 640 * 360];
        for y in 0..360 {
            for x in 0..640 {
                data[y * 640 + x] = (x % 256) as u8;
            }
        }
        let sampled = mean_brightness(&data, 640, 360, 4);
        let full = data.iter().map(|&v| f64::from(v)).sum::<f64>() / data.len() as f64;
        assert!((sampled - full).abs() < 2.0);
        assert!((sampled - full).abs() > 1e-9);
    }

    #[test]
    fn test_mean_ignores_bytes_past_height() {
        let mut data = vec![45u8; 640 * 360];
        data.extend(vec![255u8; 640]);
        assert_abs_diff_eq!(mean_brightness(&data, 640, 360, 4), 45.0);
    }

    #[test]
    fn test_inv_mode_02_sequence_locks_on_fourth_frame() {
        let mut tagger = IlluminationTagger::new(TaggerConfig::default());
        let means = [0.0, 49.0, 0.0, 48.0, 0.0, 48.0, 0.0, 47.0, 0.0, 47.0];
        let tags = run(&mut tagger, 100, &means);
        assert_eq!(
            tags,
            vec![
                Illumination::Unknown,
                Illumination::Unknown,
                Illumination::Unknown,
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
            ]
        );
    }

    #[test]
    fn test_warmup_sequence_locks_on_sixth_frame() {
        let mut tagger = IlluminationTagger::new(TaggerConfig::default());
        let means = [0.0, 0.0, 0.0, 52.9, 0.0, 76.1, 0.0, 76.0, 0.0];
        let tags = run(&mut tagger, 1, &means);
        assert_eq!(
            tags,
            vec![
                Illumination::Unknown,
                Illumination::Unknown,
                Illumination::Unknown,
                Illumination::Unknown,
                Illumination::Unknown,
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrDark,
            ]
        );
    }

    #[test]
    fn test_inv_mode_03_sequence_starting_lit_locks_even_parity() {
        let mut tagger = IlluminationTagger::new(TaggerConfig::default());
        let means = [46.0, 0.0, 46.0, 0.0, 46.0, 0.0];
        let tags = run(&mut tagger, 0, &means);
        assert_eq!(
            tags[3..],
            vec![
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrDark
            ]
        );
        assert!(matches!(
            tagger.state(),
            TaggerState::Locked { lit_parity: 0, .. }
        ));
    }

    #[test]
    fn test_emitter_off_sequence_becomes_ambient() {
        let mut tagger = IlluminationTagger::new(TaggerConfig::default());
        let cycle = [2.49, 3.27, 2.80, 2.76];
        let means: Vec<f64> = (0..20).map(|i| cycle[i % 4]).collect();
        let tags = run(&mut tagger, 0, &means);
        for t in &tags[0..15] {
            assert_eq!(*t, Illumination::Unknown);
        }
        for t in &tags[15..20] {
            assert_eq!(*t, Illumination::Ambient);
        }
    }

    #[test]
    fn test_dropped_frame_gap_keeps_phase() {
        let mut tagger = IlluminationTagger::new(TaggerConfig::default());
        let means = [0.0, 49.0, 0.0, 48.0, 0.0, 48.0];
        let _ = run(&mut tagger, 100, &means);
        assert_eq!(tagger.tag(107, 47.0), Illumination::IrLit);
        assert_eq!(tagger.tag(108, 0.0), Illumination::IrDark);
        assert!(matches!(
            tagger.state(),
            TaggerState::Locked {
                lit_parity: 1,
                contradictions: 0
            }
        ));
    }

    #[test]
    fn test_transient_bright_dark_frame_does_not_flip_phase() {
        let mut tagger = IlluminationTagger::new(TaggerConfig::default());
        let means = [0.0, 49.0, 0.0, 48.0, 0.0, 48.0, 60.0];
        let tags = run(&mut tagger, 100, &means);
        assert_eq!(
            tags[3..],
            vec![
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrDark,
            ]
        );
        assert!(matches!(
            tagger.state(),
            TaggerState::Locked {
                lit_parity: 1,
                contradictions: 1
            }
        ));
        assert_eq!(tagger.tag(107, 48.0), Illumination::IrLit);
        assert!(matches!(
            tagger.state(),
            TaggerState::Locked {
                lit_parity: 1,
                contradictions: 2
            }
        ));
        assert_eq!(tagger.tag(108, 0.0), Illumination::IrDark);
        assert!(matches!(
            tagger.state(),
            TaggerState::Locked {
                lit_parity: 1,
                contradictions: 0
            }
        ));
        assert_eq!(tagger.tag(109, 48.0), Illumination::IrLit);
    }

    #[test]
    fn test_phase_shift_without_seq_gap_relocks() {
        let mut tagger = IlluminationTagger::new(TaggerConfig::default());
        let mut means = vec![0.0, 49.0, 0.0, 48.0, 0.0, 48.0, 0.0, 48.0];
        means.extend([48.0, 0.0, 48.0, 0.0, 48.0, 0.0, 48.0]);
        let tags = run(&mut tagger, 100, &means);
        assert_eq!(
            tags[8..],
            vec![
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
            ]
        );
        assert!(matches!(
            tagger.state(),
            TaggerState::Locked {
                lit_parity: 0,
                contradictions: 0
            }
        ));
    }

    #[test]
    fn test_emitter_resumes_after_off_relocks() {
        let mut tagger = IlluminationTagger::new(TaggerConfig::default());
        let mut means: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 2.5 } else { 3.0 })
            .collect();
        means.extend([0.0, 46.0, 0.0, 46.0, 0.0, 46.0]);
        let tags = run(&mut tagger, 0, &means);
        assert_eq!(tags[20], Illumination::Ambient);
        assert_eq!(tags[21], Illumination::Unknown);
        assert_eq!(tags[22], Illumination::Unknown);
        assert_eq!(
            tags[23..26],
            vec![
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit
            ]
        );
    }

    #[test]
    fn test_ambient_ir_offset_still_alternates() {
        let mut tagger = IlluminationTagger::new(TaggerConfig::default());
        let means: Vec<f64> = (0..10)
            .map(|i| if i % 2 == 0 { 30.0 } else { 75.0 })
            .collect();
        let tags = run(&mut tagger, 0, &means);
        assert_eq!(
            tags[3..],
            vec![
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
            ]
        );
    }

    #[test]
    fn test_hysteresis_band_gives_no_evidence() {
        let mut tagger = IlluminationTagger::new(TaggerConfig::default());
        let means: Vec<f64> = (0..30)
            .map(|i| if i % 2 == 0 { 10.0 } else { 17.0 })
            .collect();
        let tags = run(&mut tagger, 0, &means);
        for t in &tags {
            assert_eq!(*t, Illumination::Unknown);
        }
        assert!(matches!(
            tagger.state(),
            TaggerState::Searching {
                candidate: None,
                votes: 0
            }
        ));
    }

    #[test]
    fn test_tagged_source_tags_gray_and_leaves_mjpeg() {
        let info = camera_info("ir", PixelFormat::Gray8, 64, 36);
        let means: [u8; 8] = [0, 46, 0, 46, 0, 46, 0, 46];
        let frames: Vec<Frame> = means
            .iter()
            .enumerate()
            .map(|(k, &m)| gray_frame("ir", k as u64, k as u64 * T, 64, 36, m))
            .collect();
        let src = VecSource::new(info, frames);
        let mut tagged = TaggedSource::new(src, TaggerConfig::default()).unwrap();
        let mut tags = Vec::new();
        for _ in 0..8 {
            tags.push(tagged.next_frame().unwrap().header().illumination);
        }
        assert_eq!(
            tags[3..],
            vec![
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
            ]
        );

        let info = camera_info("rgb", PixelFormat::Mjpeg, 1280, 720);
        let frames: Vec<Frame> = (0..3)
            .map(|k| mjpeg_frame("rgb", k, k * T, &[0xFF, 0xD8, 0xFF, 0xD9]))
            .collect();
        let src = VecSource::new(info, frames);
        let mut tagged = TaggedSource::new(src, TaggerConfig::default()).unwrap();
        for _ in 0..3 {
            assert_eq!(
                tagged.next_frame().unwrap().header().illumination,
                Illumination::Ambient
            );
        }
    }

    #[test]
    fn test_zero_sample_step_is_rejected() {
        let info = camera_info("ir", PixelFormat::Gray8, 64, 36);
        let src = VecSource::new(info, vec![]);
        let cfg = TaggerConfig {
            sample_step: 0,
            ..Default::default()
        };
        let err = TaggedSource::new(src, cfg).unwrap_err();
        assert!(matches!(err, CaptureError::Config { ref camera, .. } if camera == "ir"));
    }

    #[test]
    fn test_config_rejects_unknown_keys() {
        assert!(toml::from_str::<TaggerConfig>("enter_contrast = 12.0\nbogus = 1").is_err());
        let cfg: TaggerConfig = toml::from_str("enter_contrast = 12.0").unwrap();
        assert_eq!(cfg.enter_contrast, 12.0);
        assert_eq!(cfg.exit_contrast, TaggerConfig::default().exit_contrast);
    }

    proptest! {
        #[test]
        fn prop_alternating_stream_tagged_after_warmup(
            dark in 0.0f64..6.0,
            lit in 20.0f64..120.0,
            noise in prop::collection::vec(-2.0f64..2.0, 60),
            start_seq in 0u64..1000,
            lit_first: bool,
        ) {
            let mut tagger = IlluminationTagger::new(TaggerConfig::default());
            let means: Vec<f64> = noise
                .iter()
                .enumerate()
                .map(|(i, n)| {
                    let is_lit = (i % 2 == 0) == lit_first;
                    (if is_lit { lit } else { dark } + n).max(0.0)
                })
                .collect();
            let tags = run(&mut tagger, start_seq, &means);
            for (i, tag) in tags.iter().enumerate().skip(3) {
                let seq = start_seq + i as u64;
                let is_lit = (i % 2 == 0) == lit_first;
                let expected = if is_lit { Illumination::IrLit } else { Illumination::IrDark };
                prop_assert_eq!(*tag, expected, "seq {} index {}", seq, i);
            }
        }
    }

    // --- metadata ---

    #[test]
    fn test_metadata_tags_from_first_matched_frame() {
        let means: [u8; 8] = [0, 0, 0, 53, 0, 76, 0, 76];
        let frames = ir_frames(&means);
        let records: VecDeque<Option<MetaRecord>> = (2..=8)
            .map(|k| rec(k, Some(k >= 4 && k % 2 == 0)))
            .collect();
        let src = VecSource::new(camera_info("ir", PixelFormat::Gray8, 64, 36), frames);
        let mut tagged =
            TaggedSource::with_metadata(src, Box::new(VecMeta(records)), TaggerConfig::default())
                .unwrap();
        let mut tags = Vec::new();
        for _ in 0..8 {
            tags.push(tagged.next_frame().unwrap().header().illumination);
        }
        assert_eq!(
            tags,
            vec![
                Illumination::Unknown,
                Illumination::IrDark,
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
            ]
        );
        assert!(tagged.metadata_active());
    }

    #[test]
    fn test_metadata_overrides_brightness() {
        let means = [40u8; 6];
        let frames = ir_frames(&means);
        let records: VecDeque<Option<MetaRecord>> =
            (1..=6).map(|k| rec(k, Some(k % 2 == 0))).collect();
        let src = VecSource::new(camera_info("ir", PixelFormat::Gray8, 64, 36), frames);
        let mut tagged =
            TaggedSource::with_metadata(src, Box::new(VecMeta(records)), TaggerConfig::default())
                .unwrap();
        let mut tags = Vec::new();
        for _ in 0..6 {
            tags.push(tagged.next_frame().unwrap().header().illumination);
        }
        assert_eq!(
            tags,
            vec![
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
            ]
        );
    }

    #[test]
    fn test_missing_items_fall_back_after_max_misses() {
        let means: Vec<u8> = (0..12).map(|i| if i % 2 == 0 { 0 } else { 46 }).collect();
        let frames = ir_frames(&means);
        let records: VecDeque<Option<MetaRecord>> = (1..=12).map(|k| rec(k, None)).collect();
        let src = VecSource::new(camera_info("ir", PixelFormat::Gray8, 64, 36), frames);
        let mut tagged =
            TaggedSource::with_metadata(src, Box::new(VecMeta(records)), TaggerConfig::default())
                .unwrap();
        let mut tags = Vec::new();
        for _ in 0..12 {
            tags.push(tagged.next_frame().unwrap().header().illumination);
        }
        assert!(!tagged.metadata_active());
        assert_eq!(
            tags[3..],
            vec![
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
            ]
        );
    }

    #[test]
    fn test_metadata_error_falls_back_and_keeps_phase() {
        let means: [u8; 10] = [0, 0, 0, 53, 0, 76, 0, 76, 0, 76];
        let frames = ir_frames(&means);
        let mut records: VecDeque<Option<MetaRecord>> = (1..=6)
            .map(|k| rec(k, Some(k >= 4 && k % 2 == 0)))
            .collect();
        records.push_back(None);
        let src = VecSource::new(camera_info("ir", PixelFormat::Gray8, 64, 36), frames);
        let mut tagged =
            TaggedSource::with_metadata(src, Box::new(VecMeta(records)), TaggerConfig::default())
                .unwrap();
        let mut tags = Vec::new();
        for _ in 0..10 {
            tags.push(tagged.next_frame().unwrap().header().illumination);
        }
        assert_eq!(
            tags,
            vec![
                Illumination::IrDark,
                Illumination::IrDark,
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrDark,
                Illumination::IrLit,
            ]
        );
        assert!(!tagged.metadata_active());
    }

    #[test]
    fn test_metadata_of_dropped_frame_is_skipped() {
        let seqs = [1u64, 2, 4, 5];
        let frames: Vec<Frame> = seqs
            .iter()
            .map(|&k| gray_frame("ir", k, k * T, 64, 36, 40))
            .collect();
        let records: VecDeque<Option<MetaRecord>> =
            (1..=5).map(|k| rec(k, Some(matches!(k, 2 | 4)))).collect();
        let src = VecSource::new(camera_info("ir", PixelFormat::Gray8, 64, 36), frames);
        let mut tagged =
            TaggedSource::with_metadata(src, Box::new(VecMeta(records)), TaggerConfig::default())
                .unwrap();
        let mut tags = Vec::new();
        for _ in 0..4 {
            tags.push(tagged.next_frame().unwrap().header().illumination);
        }
        assert_eq!(
            tags,
            vec![
                Illumination::IrDark,
                Illumination::IrLit,
                Illumination::IrLit,
                Illumination::IrDark,
            ]
        );
    }

    // --- hardware (needs the IR camera, emitter on) ---

    #[test]
    #[ignore = "needs hardware"]
    fn test_live_metadata_alternates_with_emitter_on() {
        use std::path::PathBuf;

        use crate::{UvcMetaStream, V4l2Config, V4l2Source};

        let meta = UvcMetaStream::open(&PathBuf::from("/dev/video3"), Duration::from_millis(500))
            .expect("open /dev/video3");
        let video = V4l2Source::open(V4l2Config::new(
            "ir".into(),
            "/dev/video2".into(),
            PixelFormat::Gray8,
            640,
            360,
        ))
        .expect("open /dev/video2");
        let mut tagged =
            TaggedSource::with_metadata(video, Box::new(meta), TaggerConfig::default()).unwrap();

        let mut means = Vec::new();
        let mut tags = Vec::new();
        for _ in 0..40 {
            let frame = tagged.next_frame().expect("live frame");
            let h = frame.header();
            means.push(mean_brightness(frame.data(), h.width, h.height, 4));
            tags.push(h.illumination);
        }

        assert!(tagged.metadata_active());
        for i in 1..tags.len() {
            assert_ne!(
                tags[i],
                tags[i - 1],
                "frame {i} did not alternate (emitter off?)"
            );
        }
        for i in 1..tags.len() - 1 {
            if tags[i] == Illumination::IrLit {
                assert!(
                    means[i] - means[i - 1] >= 20.0 && means[i] - means[i + 1] >= 20.0,
                    "frame {i} lit mean {} not well above its dark neighbours ({}, {})",
                    means[i],
                    means[i - 1],
                    means[i + 1]
                );
            }
        }
    }

    #[test]
    #[ignore = "needs hardware"]
    fn test_live_brightness_alternates_with_emitter_on() {
        use crate::{V4l2Config, V4l2Source};

        let video = V4l2Source::open(V4l2Config::new(
            "ir".into(),
            "/dev/video2".into(),
            PixelFormat::Gray8,
            640,
            360,
        ))
        .expect("open /dev/video2");
        let mut tagged = TaggedSource::new(video, TaggerConfig::default()).unwrap();

        let mut means = Vec::new();
        let mut tags = Vec::new();
        for _ in 0..40 {
            let frame = tagged.next_frame().expect("live frame");
            let h = frame.header();
            means.push(mean_brightness(frame.data(), h.width, h.height, 4));
            tags.push(h.illumination);
        }

        for i in 6..tags.len() {
            assert_ne!(
                tags[i],
                tags[i - 1],
                "frame {i} did not alternate (emitter off?)"
            );
        }
        for i in 6..tags.len() - 1 {
            if tags[i] == Illumination::IrLit {
                assert!(
                    means[i] - means[i - 1] >= 20.0 && means[i] - means[i + 1] >= 20.0,
                    "frame {i} lit mean {} not well above its dark neighbours ({}, {})",
                    means[i],
                    means[i - 1],
                    means[i + 1]
                );
            }
        }
    }
}
