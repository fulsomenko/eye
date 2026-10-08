use std::time::Duration;

use eye_core::Illumination;
use eye_platform::emitter::{MODE_OFF, MODE_ON_DEFAULT};

use crate::{
    case::{
        Measurement, Needs, ParamError, TestCase, TestCtx, TestError, TestOutput, parse_params,
    },
    hw::{FrameMeta, Tagging, grab, tagged_ir},
    mode::EmitterSetting,
    stats,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Alternation {
    pub pairs: usize,
    pub alternating: f64,
    pub ambient: f64,
    pub lit_mean: f64,
    pub dark_mean: f64,
    pub mean: f64,
}

pub fn alternation(frames: &[FrameMeta]) -> Alternation {
    let pairs: Vec<_> = frames
        .windows(2)
        .filter(|w| w[1].seq == w[0].seq + 1)
        .collect();
    let alt = pairs
        .iter()
        .filter(|w| {
            matches!(
                (&w[0].illumination, &w[1].illumination),
                (Illumination::IrLit, Illumination::IrDark)
                    | (Illumination::IrDark, Illumination::IrLit)
            )
        })
        .count();
    let of = |want: fn(&Illumination) -> bool| -> Vec<f64> {
        frames
            .iter()
            .filter(|f| want(&f.illumination))
            .map(|f| f.mean)
            .collect()
    };
    let all: Vec<f64> = frames.iter().map(|f| f.mean).collect();
    Alternation {
        pairs: pairs.len(),
        alternating: if pairs.is_empty() {
            f64::NAN
        } else {
            alt as f64 / pairs.len() as f64
        },
        ambient: if frames.is_empty() {
            f64::NAN
        } else {
            frames
                .iter()
                .filter(|f| matches!(f.illumination, Illumination::Ambient))
                .count() as f64
                / frames.len() as f64
        },
        lit_mean: stats::mean(&of(|i| matches!(i, Illumination::IrLit))),
        dark_mean: stats::mean(&of(|i| matches!(i, Illumination::IrDark))),
        mean: stats::mean(&all),
    }
}

/// Grabs warmup + frames from the tagged IR stream; summarises the frames after warmup and reports
/// whether metadata tagging was still active at the end.
pub(crate) fn tagged_phase(
    ctx: &TestCtx,
    warmup: usize,
    frames: usize,
    tagging: Tagging,
) -> Result<(Alternation, bool), TestError> {
    let mut tagged = tagged_ir(ctx, tagging)?;
    let g = grab(ctx, &mut tagged, warmup + frames)?;
    Ok((alternation(&g.frames[warmup..]), tagged.metadata_active()))
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Expect {
    #[default]
    Auto,
    On,
    Off,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IrAlternationParams {
    pub frames: usize,
    pub warmup: usize,
    pub expect: Expect,
    pub tagging: Tagging,
    pub min_alternation: f64,
    pub min_contrast: f64,
    pub min_ambient: f64,
    pub max_off_mean: Option<f64>,
}

impl Default for IrAlternationParams {
    fn default() -> Self {
        Self {
            frames: 60,
            warmup: 16,
            expect: Expect::default(),
            tagging: Tagging::default(),
            min_alternation: 0.95,
            min_contrast: 20.0,
            min_ambient: 0.95,
            max_off_mean: None,
        }
    }
}

#[derive(Debug)]
struct IrAlternation {
    params: IrAlternationParams,
}

impl TestCase for IrAlternation {
    fn name(&self) -> &'static str {
        "ir_alternation"
    }

    fn needs(&self) -> Needs {
        Needs {
            ir: true,
            ..Needs::default()
        }
    }

    fn default_timeout(&self) -> Duration {
        let p = &self.params;
        Duration::from_secs_f64(10.0 + (p.warmup + p.frames) as f64 / 15.0)
    }

    fn run(&self, ctx: &TestCtx) -> Result<TestOutput, TestError> {
        let p = &self.params;
        let mut out = TestOutput::default();

        let (expect_on, note) = match p.expect {
            Expect::On => (true, "expected emitter on".to_owned()),
            Expect::Off => (false, "expected emitter off".to_owned()),
            Expect::Auto => match ctx.mode().emitter {
                EmitterSetting::On => (true, "expected emitter on".to_owned()),
                EmitterSetting::Off => (false, "expected emitter off".to_owned()),
                EmitterSetting::Keep => {
                    let mode = ctx.session().emitter()?.read_mode()?;
                    let on = mode >= 0x02;
                    (
                        on,
                        format!(
                            "expected emitter {} (read {mode:#04x})",
                            if on { "on" } else { "off" }
                        ),
                    )
                }
            },
        };
        out.note(note);

        if expect_on {
            ctx.instruct(
                "keep a face or a hand 30-50 cm in front of the camera for the IR contrast check",
            );
        }

        let tagging = if expect_on {
            p.tagging
        } else {
            Tagging::Brightness
        };
        let (a, meta_active) = tagged_phase(ctx, p.warmup, p.frames, tagging)?;
        out.push(Measurement::info("lit_mean", a.lit_mean, ""));
        out.push(Measurement::info("dark_mean", a.dark_mean, ""));
        out.push(Measurement::info("mean", a.mean, ""));
        out.push(Measurement::info(
            "metadata_active",
            if meta_active { 1.0 } else { 0.0 },
            "",
        ));

        if expect_on {
            out.push(Measurement::at_least(
                "alternation",
                a.alternating,
                "",
                p.min_alternation,
            ));
            out.push(Measurement::at_least(
                "contrast",
                a.lit_mean - a.dark_mean,
                "",
                p.min_contrast,
            ));
        } else {
            out.push(Measurement::at_least(
                "ambient",
                a.ambient,
                "",
                p.min_ambient,
            ));
            if let Some(max) = p.max_off_mean {
                out.push(Measurement::at_most("off_mean", a.mean, "", max));
            }
        }
        Ok(out)
    }
}

pub fn build_ir_alternation(params: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    let p: IrAlternationParams = parse_params(params)?;
    if p.frames == 0 {
        return Err(ParamError::Invalid("frames must be >= 1".into()));
    }
    for (name, v) in [
        ("min_alternation", p.min_alternation),
        ("min_contrast", p.min_contrast),
        ("min_ambient", p.min_ambient),
    ] {
        if !v.is_finite() || v < 0.0 {
            return Err(ParamError::Invalid(format!(
                "{name} must be finite and >= 0"
            )));
        }
    }
    if let Some(m) = p.max_off_mean
        && (!m.is_finite() || m < 0.0)
    {
        return Err(ParamError::Invalid(
            "max_off_mean must be finite and >= 0".into(),
        ));
    }
    Ok(Box::new(IrAlternation { params: p }))
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EmitterToggleParams {
    pub frames: usize,
    pub warmup: usize,
    pub min_alternation: f64,
    pub min_ambient: f64,
}

impl Default for EmitterToggleParams {
    fn default() -> Self {
        Self {
            frames: 30,
            warmup: 16,
            min_alternation: 0.9,
            min_ambient: 0.9,
        }
    }
}

#[derive(Debug)]
struct EmitterToggle {
    params: EmitterToggleParams,
}

impl TestCase for EmitterToggle {
    fn name(&self) -> &'static str {
        "emitter_toggle"
    }

    fn needs(&self) -> Needs {
        Needs {
            ir: true,
            emitter: true,
            ..Needs::default()
        }
    }

    fn default_timeout(&self) -> Duration {
        let p = &self.params;
        Duration::from_secs_f64(10.0 + 2.0 * (p.warmup + p.frames) as f64 / 15.0)
    }

    fn run(&self, ctx: &TestCtx) -> Result<TestOutput, TestError> {
        let p = &self.params;
        if ctx.mode().emitter == EmitterSetting::Keep {
            return Err(TestError::Other(
                "emitter_toggle needs a mode with emitter = on or off, so the runner restores the emitter if the test fails".into(),
            ));
        }
        ctx.instruct("keep a face or a hand 30-50 cm in front of the camera for the emitter check");
        let mut out = TestOutput::default();
        let mut em = ctx.session().emitter()?;
        let prior = em.read_mode()?;
        em.write_mode(MODE_ON_DEFAULT)?;
        let on = f64::from(MODE_ON_DEFAULT);
        out.push(Measurement::within(
            "on_readback",
            f64::from(em.read_mode()?),
            "",
            on,
            on,
        ));
        let (lit, _) = tagged_phase(ctx, p.warmup, p.frames, Tagging::Brightness)?;
        out.push(Measurement::at_least(
            "on_alternation",
            lit.alternating,
            "",
            p.min_alternation,
        ));
        em.write_mode(MODE_OFF)?;
        let off = f64::from(MODE_OFF);
        out.push(Measurement::within(
            "off_readback",
            f64::from(em.read_mode()?),
            "",
            off,
            off,
        ));
        let (dark, _) = tagged_phase(ctx, p.warmup, p.frames, Tagging::Brightness)?;
        out.push(Measurement::at_least(
            "off_ambient",
            dark.ambient,
            "",
            p.min_ambient,
        ));
        em.write_mode(prior)?;
        let before = f64::from(prior);
        out.push(Measurement::within(
            "restored_readback",
            f64::from(em.read_mode()?),
            "",
            before,
            before,
        ));
        Ok(out)
    }
}

pub fn build_emitter_toggle(params: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    let p: EmitterToggleParams = parse_params(params)?;
    if p.frames == 0 {
        return Err(ParamError::Invalid("frames must be >= 1".into()));
    }
    for (name, v) in [
        ("min_alternation", p.min_alternation),
        ("min_ambient", p.min_ambient),
    ] {
        if !v.is_finite() || v < 0.0 {
            return Err(ParamError::Invalid(format!(
                "{name} must be finite and >= 0"
            )));
        }
    }
    Ok(Box::new(EmitterToggle { params: p }))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, atomic::AtomicU8};

    use super::*;
    use crate::{
        mode::Role,
        testkit::{self, EmitterAwareSource, FakeEmitter, FakeMeta},
    };

    fn fm(seq: u64, illum: Illumination, mean: f64) -> FrameMeta {
        FrameMeta {
            seq,
            t: eye_core::Timestamp::from_nanos(seq * 66_666_666),
            received: eye_core::Timestamp::from_nanos(seq * 66_666_666),
            width: 64,
            height: 36,
            format: eye_core::PixelFormat::Gray8,
            illumination: illum,
            mean,
            payload_ok: true,
        }
    }

    #[test]
    fn test_alternation_summary_counts_pairs() {
        let tags = [
            Illumination::IrLit,
            Illumination::IrDark,
            Illumination::IrLit,
            Illumination::IrLit,
            Illumination::IrDark,
        ];
        let frames: Vec<_> = tags
            .iter()
            .enumerate()
            .map(|(i, &t)| fm(i as u64, t, 0.0))
            .collect();
        let a = alternation(&frames);
        assert_eq!(a.pairs, 4);
        assert_eq!(a.alternating, 0.75);

        let mut with_gap = frames.clone();
        with_gap[4].seq = 10;
        let a = alternation(&with_gap);
        assert_eq!(a.pairs, 3);
    }

    fn ir_frames(n: usize, warmup: usize, value: impl Fn(u64) -> u8) -> Vec<eye_core::Frame> {
        testkit::synth_gray("ir", warmup + n, 0, 0, 66_666_666, value)
    }

    #[test]
    fn test_ir_alternation_emitter_on_passes() {
        let frames = ir_frames(60, 16, |seq| if seq % 2 == 1 { 46 } else { 0 });
        let session =
            Arc::new(testkit::FakeSession::empty().with_source(Role::Ir, testkit::boxed(frames)));
        let ctx = testkit::ctx(session, testkit::mode_ir(EmitterSetting::On));
        let case = build_ir_alternation(&toml::Table::new()).unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(out.passed());
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "alternation")
                .unwrap()
                .value,
            1.0
        );
        approx::assert_relative_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "contrast")
                .unwrap()
                .value,
            46.0,
            epsilon = 1e-9
        );
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "metadata_active")
                .unwrap()
                .value,
            0.0
        );
    }

    #[test]
    fn test_ir_alternation_flat_stream_with_emitter_on_fails() {
        let frames = ir_frames(60, 16, |_| 3);
        let session =
            Arc::new(testkit::FakeSession::empty().with_source(Role::Ir, testkit::boxed(frames)));
        let ctx = testkit::ctx(session, testkit::mode_ir(EmitterSetting::On));
        let case = build_ir_alternation(&toml::Table::new()).unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(!out.passed());
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "alternation")
                .unwrap()
                .value,
            0.0
        );
    }

    #[test]
    fn test_ir_alternation_metadata_tags_but_contrast_checks_light() {
        let frames = ir_frames(60, 16, |_| 46);
        let records: std::collections::VecDeque<_> = (0..76u64)
            .map(|seq| eye_capture::MetaRecord {
                timestamp: eye_core::Timestamp::from_nanos(seq * 66_666_666),
                lit: Some(seq % 2 == 1),
            })
            .collect();
        let session = Arc::new(
            testkit::FakeSession::empty()
                .with_source(Role::Ir, testkit::boxed(frames))
                .with_meta(Box::new(FakeMeta { records })),
        );
        let ctx = testkit::ctx(session, testkit::mode_ir(EmitterSetting::On));
        let case = build_ir_alternation(&toml::Table::new()).unwrap();
        let out = case.run(&ctx).unwrap();
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "metadata_active")
                .unwrap()
                .value,
            1.0
        );
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "alternation")
                .unwrap()
                .value,
            1.0
        );
        approx::assert_relative_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "contrast")
                .unwrap()
                .value,
            0.0,
            epsilon = 1e-9
        );
        assert!(!out.passed());
    }

    #[test]
    fn test_ir_alternation_emitter_off_passes() {
        let cycle = [2.0, 3.0, 3.0, 2.0];
        let frames = ir_frames(60, 16, move |seq| cycle[(seq % 4) as usize] as u8);
        let session =
            Arc::new(testkit::FakeSession::empty().with_source(Role::Ir, testkit::boxed(frames)));
        let ctx = testkit::ctx(session, testkit::mode_ir(EmitterSetting::Off));
        let case = build_ir_alternation(&toml::Table::new()).unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(out.passed());
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "ambient")
                .unwrap()
                .value,
            1.0
        );
        approx::assert_relative_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "mean")
                .unwrap()
                .value,
            2.5,
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_ir_alternation_off_never_uses_metadata() {
        let frames = ir_frames(60, 16, |_| 3);
        let records: std::collections::VecDeque<_> = (0..76u64)
            .map(|seq| eye_capture::MetaRecord {
                timestamp: eye_core::Timestamp::from_nanos(seq * 66_666_666),
                lit: Some(seq % 2 == 1),
            })
            .collect();
        let session = Arc::new(
            testkit::FakeSession::empty()
                .with_source(Role::Ir, testkit::boxed(frames))
                .with_meta(Box::new(FakeMeta { records })),
        );
        let session_dyn: Arc<dyn crate::mode::ModeSession> = Arc::clone(&session) as _;
        let ctx = testkit::ctx(session_dyn, testkit::mode_ir(EmitterSetting::Off));
        let case = build_ir_alternation(&toml::Table::new()).unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(out.passed());
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "ambient")
                .unwrap()
                .value,
            1.0
        );
        assert_eq!(session.meta.lock().unwrap().len(), 1);
    }

    #[test]
    fn test_ir_alternation_alternating_stream_with_emitter_off_fails() {
        let frames = ir_frames(60, 16, |seq| if seq % 2 == 1 { 46 } else { 0 });
        let session =
            Arc::new(testkit::FakeSession::empty().with_source(Role::Ir, testkit::boxed(frames)));
        let ctx = testkit::ctx(session, testkit::mode_ir(EmitterSetting::Off));
        let case = build_ir_alternation(&toml::Table::new()).unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(!out.passed());
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "ambient")
                .unwrap()
                .value,
            0.0
        );
    }

    #[test]
    fn test_ir_alternation_max_off_mean_is_optional() {
        let frames = ir_frames(60, 16, |_| 30);
        let session =
            Arc::new(testkit::FakeSession::empty().with_source(Role::Ir, testkit::boxed(frames)));
        let ctx = testkit::ctx(session, testkit::mode_ir(EmitterSetting::Off));
        let case = build_ir_alternation(&toml::Table::new()).unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(out.passed());

        let frames = ir_frames(60, 16, |_| 30);
        let session =
            Arc::new(testkit::FakeSession::empty().with_source(Role::Ir, testkit::boxed(frames)));
        let ctx = testkit::ctx(session, testkit::mode_ir(EmitterSetting::Off));
        let mut table = toml::Table::new();
        table.insert("max_off_mean".into(), toml::Value::Float(15.0));
        let case = build_ir_alternation(&table).unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(!out.passed());
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "off_mean")
                .unwrap()
                .value,
            30.0
        );
    }

    #[test]
    fn test_ir_alternation_keep_reads_the_emitter() {
        let frames = ir_frames(60, 16, |seq| if seq % 2 == 1 { 46 } else { 0 });
        let emitter = FakeEmitter {
            mode: Arc::new(AtomicU8::new(3)),
            ignore_writes: false,
            writes: Arc::new(std::sync::Mutex::new(Vec::new())),
        };
        let session = Arc::new(
            testkit::FakeSession {
                emitter: Some(emitter),
                ..testkit::FakeSession::empty()
            }
            .with_source(Role::Ir, testkit::boxed(frames)),
        );
        let ctx = testkit::ctx(session, testkit::mode_ir(EmitterSetting::Keep));
        let case = build_ir_alternation(&toml::Table::new()).unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(out.passed());
        assert!(
            out.notes
                .iter()
                .any(|n| n.contains("expected emitter on (read 0x03)"))
        );
    }

    fn emitter_source(mode: Arc<AtomicU8>, frames: usize) -> Box<dyn eye_capture::FrameSource> {
        Box::new(EmitterAwareSource {
            mode,
            seq: 0,
            remaining: frames,
        })
    }

    #[test]
    fn test_emitter_toggle_round_trip_passes_and_restores() {
        let mode = Arc::new(AtomicU8::new(1));
        let writes = Arc::new(std::sync::Mutex::new(Vec::new()));
        let emitter = FakeEmitter {
            mode: Arc::clone(&mode),
            ignore_writes: false,
            writes: Arc::clone(&writes),
        };
        let session = Arc::new(
            testkit::FakeSession {
                emitter: Some(emitter),
                ..testkit::FakeSession::empty()
            }
            .with_source(Role::Ir, emitter_source(Arc::clone(&mode), 46))
            .with_source(Role::Ir, emitter_source(Arc::clone(&mode), 46)),
        );
        let ctx = testkit::ctx(session, testkit::mode_ir(EmitterSetting::Off));
        let case = build_emitter_toggle(&toml::Table::new()).unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(out.passed());
        assert_eq!(
            *writes.lock().unwrap(),
            vec![MODE_ON_DEFAULT, MODE_OFF, MODE_OFF]
        );
        assert_eq!(mode.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn test_emitter_toggle_ignored_write_fails() {
        let mode = Arc::new(AtomicU8::new(1));
        let emitter = FakeEmitter {
            mode: Arc::clone(&mode),
            ignore_writes: true,
            writes: Arc::new(std::sync::Mutex::new(Vec::new())),
        };
        let session = Arc::new(
            testkit::FakeSession {
                emitter: Some(emitter),
                ..testkit::FakeSession::empty()
            }
            .with_source(Role::Ir, emitter_source(Arc::clone(&mode), 46))
            .with_source(Role::Ir, emitter_source(Arc::clone(&mode), 46)),
        );
        let ctx = testkit::ctx(session, testkit::mode_ir(EmitterSetting::Off));
        let case = build_emitter_toggle(&toml::Table::new()).unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(!out.passed());
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "on_readback")
                .unwrap()
                .value,
            1.0
        );
    }

    #[test]
    fn test_emitter_toggle_refuses_keep_mode() {
        let session = Arc::new(testkit::FakeSession::empty());
        let ctx = testkit::ctx(session, testkit::mode_ir(EmitterSetting::Keep));
        let case = build_emitter_toggle(&toml::Table::new()).unwrap();
        let err = case.run(&ctx).unwrap_err();
        match err {
            TestError::Other(msg) => assert!(msg.contains("emitter = on or off")),
            other => panic!("expected TestError::Other, got {other:?}"),
        }
    }
}
