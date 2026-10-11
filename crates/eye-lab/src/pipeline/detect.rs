use std::{
    fmt,
    time::{Duration, Instant},
};

use eye_capture::{CaptureError, FrameSource};
use eye_core::{FrameSet, Illumination, Observations, Side};
use nalgebra::Point2;

use crate::{
    case::{
        Measurement, Needs, ParamError, TestCase, TestCtx, TestError, TestOutput, parse_params,
    },
    hw::{Tagging, tagged_ir},
    mode::Role,
    pipeline::{
        DetectorFactory, ScreenParams, lab_config, lab_rig, registry_detectors, validate_timing,
    },
    stats,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Cam {
    Rgb,
    #[default]
    Ir,
}

impl Cam {
    pub fn role(self) -> Role {
        match self {
            Cam::Rgb => Role::Rgb,
            Cam::Ir => Role::Ir,
        }
    }
}

/// Pupil first for IR, iris first for RGB; the other as fallback.
pub fn eye_centres(observations: &[Observations], role: Role) -> Vec<(Side, Point2<f64>)> {
    observations
        .iter()
        .filter_map(|o| o.face.as_ref())
        .flat_map(|f| f.eyes.iter())
        .filter_map(|e| {
            let ellipse = match role {
                Role::Ir => e.pupil.as_ref().or(e.iris.as_ref()),
                Role::Rgb => e.iris.as_ref().or(e.pupil.as_ref()),
            }?;
            Some((e.side, ellipse.value().center()))
        })
        .collect()
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DetectionStats {
    pub accepted: usize,
    /// Subset of `accepted` excluding `IrDark` frames.
    pub evaluable: usize,
    pub detected: usize,
    pub errors: usize,
    pub detect_ms: Vec<f64>,
    pub centres: Vec<(Side, Point2<f64>)>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Jitter {
    pub n: usize,
    pub rms_s2s: f64,
    pub std: f64,
}

/// Holmqvist RMS sample-to-sample: sqrt(sum |p[i+1] - p[i]|^2 / (n - 1)), NaN for n < 2;
/// std = sqrt(var_x + var_y) (population), NaN for n = 0.
pub fn jitter(points: &[Point2<f64>]) -> Jitter {
    let n = points.len();
    let rms_s2s = if n < 2 {
        f64::NAN
    } else {
        let sum_sq: f64 = points
            .windows(2)
            .map(|w| (w[1] - w[0]).norm_squared())
            .sum();
        (sum_sq / (n - 1) as f64).sqrt()
    };
    let std = if n == 0 {
        f64::NAN
    } else {
        let xs: Vec<f64> = points.iter().map(|p| p.x).collect();
        let ys: Vec<f64> = points.iter().map(|p| p.y).collect();
        let var = |v: &[f64]| stats::std_dev(v).powi(2);
        (var(&xs) + var(&ys)).sqrt()
    };
    Jitter { n, rms_s2s, std }
}

/// Opens the role's stream (IR through `hw::tagged_ir(ctx, Tagging::Auto)`, like the pipeline), builds the
/// detector for the role's `[detect]` section, waits `lead_s` after `instruct`, then feeds every accepted frame
/// as `FrameSet::single(frame)` until `seconds` elapsed or the source returns EndOfStream (fakes).
fn run_detector(
    ctx: &TestCtx,
    cam: Cam,
    seconds: f64,
    lead_s: f64,
    screen: &ScreenParams,
    build: &DetectorFactory,
    instruction: &str,
) -> Result<DetectionStats, TestError> {
    let role = cam.role();
    let config = lab_config(ctx)?;
    let mut source: Box<dyn FrameSource> = match role {
        Role::Ir => Box::new(tagged_ir(ctx, Tagging::Auto)?),
        Role::Rgb => ctx.session().open(role)?,
    };
    let target = ctx
        .mode()
        .target(role)
        .ok_or(crate::mode::ModeError::NoCamera(role))?;
    let rig = lab_rig(screen, &[(role, target.format.width, target.format.height)])?;
    let id = crate::modes::opener::camera_id(role);
    let section = config.detect.get(id.as_str()).ok_or_else(|| {
        TestError::Other(format!(
            "eye config has no [detect] entry for camera {:?}",
            id.as_str()
        ))
    })?;
    let mut detector = build(section, &rig).map_err(TestError::Other)?;
    ctx.instruct(instruction);
    ctx.sleep(Duration::from_secs_f64(lead_s))?;
    let mut stats = DetectionStats::default();
    let end = Instant::now() + Duration::from_secs_f64(seconds);
    while Instant::now() < end {
        ctx.check()?;
        let frame = match source.next_frame() {
            Ok(frame) => frame,
            Err(CaptureError::EndOfStream) => break,
            Err(e) => return Err(e.into()),
        };
        let h = frame.header();
        if !detector.accepts(h.format, h.illumination) {
            continue;
        }
        stats.accepted += 1;
        if h.illumination != Illumination::IrDark {
            stats.evaluable += 1;
        }
        let started = Instant::now();
        let result = detector.detect(&FrameSet::single(frame));
        stats.detect_ms.push(started.elapsed().as_secs_f64() * 1e3);
        match result {
            Ok(observations) => {
                let centres = eye_centres(&observations, role);
                if !centres.is_empty() {
                    stats.detected += 1;
                }
                stats.centres.extend(centres);
            }
            Err(_) => stats.errors += 1,
        }
    }
    Ok(stats)
}

fn needs_for(cam: Cam) -> Needs {
    match cam {
        Cam::Rgb => Needs {
            rgb: true,
            subject: true,
            ..Needs::default()
        },
        Cam::Ir => Needs {
            ir: true,
            subject: true,
            ..Needs::default()
        },
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DetectionParams {
    pub camera: Cam,
    pub seconds: f64,
    pub lead_s: f64,
    pub min_rate: f64,
    pub min_frames: usize,
    pub screen: ScreenParams,
}

impl Default for DetectionParams {
    fn default() -> Self {
        Self {
            camera: Cam::default(),
            seconds: 10.0,
            lead_s: 3.0,
            min_rate: 0.9,
            min_frames: 30,
            screen: ScreenParams::default(),
        }
    }
}

pub struct DetectionRate {
    p: DetectionParams,
    build: DetectorFactory,
}

impl fmt::Debug for DetectionRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DetectionRate")
            .field("p", &self.p)
            .finish_non_exhaustive()
    }
}

impl TestCase for DetectionRate {
    fn name(&self) -> &'static str {
        "detection_rate"
    }

    fn needs(&self) -> Needs {
        needs_for(self.p.camera)
    }

    fn default_timeout(&self) -> Duration {
        Duration::from_secs_f64(self.p.seconds + self.p.lead_s + 15.0)
    }

    fn run(&self, ctx: &TestCtx) -> Result<TestOutput, TestError> {
        let p = &self.p;
        let instruction = format!(
            "sit about 50 cm from the screen, face it, keep both eyes open for {} s, starting in {} s",
            p.seconds, p.lead_s
        );
        let stats = run_detector(
            ctx,
            p.camera,
            p.seconds,
            p.lead_s,
            &p.screen,
            &self.build,
            &instruction,
        )?;
        let mut out = TestOutput::default();
        let rate = stats.detected as f64 / stats.evaluable as f64;
        out.push(Measurement::at_least(
            "detection_rate",
            rate,
            "",
            p.min_rate,
        ));
        out.push(Measurement::at_least(
            "accepted_frames",
            stats.accepted as f64,
            "",
            p.min_frames as f64,
        ));
        out.push(Measurement::info(
            "evaluable_frames",
            stats.evaluable as f64,
            "",
        ));
        out.push(Measurement::info("detect_errors", stats.errors as f64, ""));
        out.push(Measurement::info(
            "detect_ms_p95",
            stats::pct(&stats.detect_ms, 95.0),
            "",
        ));
        Ok(out)
    }
}

pub fn build_detection_rate(params: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    let p: DetectionParams = parse_params(params)?;
    validate_timing(p.seconds, p.lead_s)?;
    Ok(Box::new(DetectionRate {
        p,
        build: registry_detectors(),
    }))
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct JitterParams {
    pub camera: Cam,
    pub seconds: f64,
    pub lead_s: f64,
    pub max_rms_s2s_px: Option<f64>,
    pub min_samples: usize,
    pub screen: ScreenParams,
}

impl Default for JitterParams {
    fn default() -> Self {
        Self {
            camera: Cam::default(),
            seconds: 8.0,
            lead_s: 3.0,
            max_rms_s2s_px: None,
            min_samples: 30,
            screen: ScreenParams::default(),
        }
    }
}

pub struct PupilJitter {
    p: JitterParams,
    build: DetectorFactory,
}

impl fmt::Debug for PupilJitter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PupilJitter")
            .field("p", &self.p)
            .finish_non_exhaustive()
    }
}

impl TestCase for PupilJitter {
    fn name(&self) -> &'static str {
        "pupil_jitter"
    }

    fn needs(&self) -> Needs {
        needs_for(self.p.camera)
    }

    fn default_timeout(&self) -> Duration {
        Duration::from_secs_f64(self.p.seconds + self.p.lead_s + 15.0)
    }

    fn run(&self, ctx: &TestCtx) -> Result<TestOutput, TestError> {
        let p = &self.p;
        let limit = p.max_rms_s2s_px.unwrap_or(match p.camera {
            Cam::Ir => 1.0,
            Cam::Rgb => 2.0,
        });
        let instruction = format!(
            "look at the camera lens and keep head and eyes still for {} s, starting in {} s",
            p.seconds, p.lead_s
        );
        let stats = run_detector(
            ctx,
            p.camera,
            p.seconds,
            p.lead_s,
            &p.screen,
            &self.build,
            &instruction,
        )?;
        let mut out = TestOutput::default();
        let mut evaluated = 0usize;
        for (side, name) in [(Side::Left, "left"), (Side::Right, "right")] {
            let points: Vec<Point2<f64>> = stats
                .centres
                .iter()
                .filter(|(s, _)| *s == side)
                .map(|(_, p)| *p)
                .collect();
            out.push(Measurement::info(
                format!("{name}.samples"),
                points.len() as f64,
                "",
            ));
            if points.len() >= p.min_samples {
                let j = jitter(&points);
                out.push(Measurement::at_most(
                    format!("{name}.rms_s2s_px"),
                    j.rms_s2s,
                    "",
                    limit,
                ));
                out.push(Measurement::info(format!("{name}.std_px"), j.std, ""));
                evaluated += 1;
            }
        }
        out.push(Measurement::at_least(
            "eyes_evaluated",
            evaluated as f64,
            "",
            1.0,
        ));
        Ok(out)
    }
}

pub fn build_pupil_jitter(params: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    let p: JitterParams = parse_params(params)?;
    validate_timing(p.seconds, p.lead_s)?;
    if let Some(m) = p.max_rms_s2s_px
        && (!m.is_finite() || m < 0.0)
    {
        return Err(ParamError::Invalid(
            "max_rms_s2s_px must be finite and >= 0".into(),
        ));
    }
    Ok(Box::new(PupilJitter {
        p,
        build: registry_detectors(),
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use approx::assert_relative_eq;
    use eye_core::{
        EyeObservation, FaceObservation, Illumination, Measured, observation::LandmarkScheme,
    };

    use super::*;
    use crate::{
        mode::EmitterSetting,
        testkit::{self, FakeSession},
    };

    #[test]
    fn test_eye_centres_prefer_pupil_for_ir_and_iris_for_rgb() {
        let mut eye = EyeObservation::new(Side::Left);
        eye.pupil = Some(
            Measured::new(
                eye_core::Ellipse2::circle(Point2::new(10.0, 10.0), 3.0).unwrap(),
                0.5,
            )
            .unwrap(),
        );
        eye.iris = Some(
            Measured::new(
                eye_core::Ellipse2::circle(Point2::new(20.0, 20.0), 5.0).unwrap(),
                0.5,
            )
            .unwrap(),
        );
        let face = FaceObservation {
            scheme: LandmarkScheme::IR_PUPIL_PAIR,
            landmarks: vec![],
            eyes: vec![eye],
        };
        let obs = vec![Observations {
            camera: eye_core::CameraId::from("ir"),
            timestamp: eye_core::Timestamp::from_nanos(0),
            face: Some(face),
        }];
        let ir = eye_centres(&obs, Role::Ir);
        assert_eq!(ir, vec![(Side::Left, Point2::new(10.0, 10.0))]);
        let rgb = eye_centres(&obs, Role::Rgb);
        assert_eq!(rgb, vec![(Side::Left, Point2::new(20.0, 20.0))]);

        let empty = vec![Observations::empty(
            eye_core::CameraId::from("ir"),
            eye_core::Timestamp::from_nanos(0),
        )];
        assert_eq!(eye_centres(&empty, Role::Ir), vec![]);
    }

    #[test]
    fn test_jitter_of_constant_points_is_zero() {
        let points: Vec<Point2<f64>> = (0..10).map(|_| Point2::new(100.0, 50.0)).collect();
        let j = jitter(&points);
        assert_eq!(j.rms_s2s, 0.0);
        assert_eq!(j.std, 0.0);
    }

    #[test]
    fn test_jitter_of_alternating_offset() {
        let points: Vec<Point2<f64>> = (0..20)
            .map(|i| {
                if i % 2 == 0 {
                    Point2::new(100.0, 50.0)
                } else {
                    Point2::new(100.4, 50.0)
                }
            })
            .collect();
        let j = jitter(&points);
        assert_relative_eq!(j.rms_s2s, 0.4, epsilon = 1e-9);
        assert_relative_eq!(j.std, 0.2, epsilon = 1e-9);
    }

    #[test]
    fn test_jitter_single_point_has_nan_rms() {
        let j = jitter(&[Point2::new(0.0, 0.0)]);
        assert!(j.rms_s2s.is_nan());
        assert_eq!(j.n, 1);
    }

    fn ir_frames_constant(n: usize, value: u8) -> Vec<eye_core::Frame> {
        testkit::synth_gray("ir", n, 0, 0, 66_666_666, move |_| value)
    }

    fn ctx_ir() -> (TestCtx, Arc<FakeSession>) {
        let session = Arc::new(FakeSession::empty());
        let session_dyn: Arc<dyn crate::mode::ModeSession> = Arc::clone(&session) as _;
        let ctx = testkit::ctx(session_dyn, testkit::mode_ir(EmitterSetting::On));
        (ctx, session)
    }

    #[test]
    fn test_detection_rate_counts_only_accepted_frames() {
        let frames = testkit::synth_gray("ir", 103, 0, 0, 66_666_666, |seq| {
            if seq % 2 == 1 { 46 } else { 0 }
        });
        let (ctx, session) = ctx_ir();
        session
            .sources
            .lock()
            .unwrap()
            .entry(Role::Ir)
            .or_default()
            .push_back(testkit::boxed(frames));
        let build: DetectorFactory =
            testkit::fake_detectors(vec![Illumination::IrLit], None, |_| 0.0);
        let case = DetectionRate {
            p: DetectionParams {
                seconds: 60.0,
                lead_s: 0.0,
                min_frames: 10,
                ..DetectionParams::default()
            },
            build,
        };
        let out = case.run(&ctx).unwrap();
        let accepted = out
            .measurements
            .iter()
            .find(|m| m.name == "accepted_frames")
            .unwrap();
        assert_eq!(accepted.value, 50.0);
        let rate = out
            .measurements
            .iter()
            .find(|m| m.name == "detection_rate")
            .unwrap();
        assert_eq!(rate.value, 1.0);
        assert!(out.passed());
    }

    #[test]
    fn test_detection_rate_ignores_dark_frames_in_denominator() {
        let frames = testkit::synth_gray("ir", 100, 0, 0, 66_666_666, |seq| {
            if seq % 2 == 1 { 46 } else { 0 }
        });
        let records: std::collections::VecDeque<_> = (0..100u64)
            .map(|seq| eye_capture::MetaRecord {
                timestamp: eye_core::Timestamp::from_nanos(seq * 66_666_666),
                lit: Some(seq % 2 == 1),
            })
            .collect();
        let session = Arc::new(
            testkit::FakeSession::empty()
                .with_source(Role::Ir, testkit::boxed(frames))
                .with_meta(Box::new(testkit::FakeMeta { records })),
        );
        let session_dyn: Arc<dyn crate::mode::ModeSession> = Arc::clone(&session) as _;
        let ctx = testkit::ctx(session_dyn, testkit::mode_ir(EmitterSetting::On));
        let build: DetectorFactory = testkit::fake_detectors(vec![], None, |_| 0.0);
        let case = DetectionRate {
            p: DetectionParams {
                seconds: 60.0,
                lead_s: 0.0,
                min_frames: 10,
                ..DetectionParams::default()
            },
            build,
        };
        let out = case.run(&ctx).unwrap();
        let accepted = out
            .measurements
            .iter()
            .find(|m| m.name == "accepted_frames")
            .unwrap();
        assert_eq!(accepted.value, 100.0);
        let evaluable = out
            .measurements
            .iter()
            .find(|m| m.name == "evaluable_frames")
            .unwrap();
        assert_eq!(evaluable.value, 50.0);
        let rate = out
            .measurements
            .iter()
            .find(|m| m.name == "detection_rate")
            .unwrap();
        assert_relative_eq!(rate.value, 1.0, epsilon = 1e-9);
        assert!(out.passed());
    }

    #[test]
    fn test_detection_rate_below_threshold_fails() {
        let frames = testkit::synth_gray("ir", 103, 0, 0, 66_666_666, |seq| {
            if seq % 2 == 1 { 46 } else { 0 }
        });
        let (ctx, session) = ctx_ir();
        session
            .sources
            .lock()
            .unwrap()
            .entry(Role::Ir)
            .or_default()
            .push_back(testkit::boxed(frames));
        let build: DetectorFactory = testkit::fake_detectors(vec![], Some(4), |_| 0.0);
        let case = DetectionRate {
            p: DetectionParams {
                seconds: 60.0,
                lead_s: 0.0,
                ..DetectionParams::default()
            },
            build,
        };
        let out = case.run(&ctx).unwrap();
        let accepted = out
            .measurements
            .iter()
            .find(|m| m.name == "accepted_frames")
            .unwrap();
        assert_eq!(accepted.value, 103.0);
        let rate = out
            .measurements
            .iter()
            .find(|m| m.name == "detection_rate")
            .unwrap();
        assert!(
            rate.value < 0.9,
            "expected a below-threshold rate, got {}",
            rate.value
        );
        assert!(!out.passed());
    }

    #[test]
    fn test_detection_errors_are_counted_not_fatal() {
        let frames = ir_frames_constant(40, 46);
        let (ctx, session) = ctx_ir();
        session
            .sources
            .lock()
            .unwrap()
            .entry(Role::Ir)
            .or_default()
            .push_back(testkit::boxed(frames));
        let build: DetectorFactory = testkit::fake_detectors(vec![], Some(5), |_| 0.0);
        let case = DetectionRate {
            p: DetectionParams {
                seconds: 60.0,
                lead_s: 0.0,
                ..DetectionParams::default()
            },
            build,
        };
        let out = case.run(&ctx).unwrap();
        let errors = out
            .measurements
            .iter()
            .find(|m| m.name == "detect_errors")
            .unwrap();
        assert_eq!(errors.value, 8.0);
    }

    #[test]
    fn test_pupil_jitter_passes_for_still_eye_and_fails_for_shaky_eye() {
        let frames = ir_frames_constant(103, 46);
        let (ctx, session) = ctx_ir();
        session
            .sources
            .lock()
            .unwrap()
            .entry(Role::Ir)
            .or_default()
            .push_back(testkit::boxed(frames.clone()));
        let build: DetectorFactory = testkit::fake_detectors(vec![], None, |_| 0.0);
        let case = PupilJitter {
            p: JitterParams {
                seconds: 60.0,
                lead_s: 0.0,
                ..JitterParams::default()
            },
            build,
        };
        let out = case.run(&ctx).unwrap();
        let left = out
            .measurements
            .iter()
            .find(|m| m.name == "left.rms_s2s_px")
            .unwrap();
        assert_eq!(left.value, 0.0);
        assert!(out.passed());
        let evaluated = out
            .measurements
            .iter()
            .find(|m| m.name == "eyes_evaluated")
            .unwrap();
        assert_eq!(evaluated.value, 1.0);
        let right_samples = out
            .measurements
            .iter()
            .find(|m| m.name == "right.samples")
            .unwrap();
        assert_eq!(right_samples.value, 0.0);
        assert!(
            !out.measurements
                .iter()
                .any(|m| m.name == "right.rms_s2s_px")
        );

        session
            .sources
            .lock()
            .unwrap()
            .entry(Role::Ir)
            .or_default()
            .push_back(testkit::boxed(frames));
        let build: DetectorFactory =
            testkit::fake_detectors(vec![], None, |s| if s % 2 == 0 { 0.0 } else { 2.0 });
        let case = PupilJitter {
            p: JitterParams {
                seconds: 60.0,
                lead_s: 0.0,
                ..JitterParams::default()
            },
            build,
        };
        let out = case.run(&ctx).unwrap();
        let left = out
            .measurements
            .iter()
            .find(|m| m.name == "left.rms_s2s_px")
            .unwrap();
        assert_relative_eq!(left.value, 2.0, epsilon = 1e-9);
        assert!(!out.passed());
    }

    #[test]
    fn test_missing_detect_section_is_error() {
        let toml = "[[camera]]\nid = \"ir\"\ndevice = \"/dev/video2\"\nformat = \"gray\"\nsize = [64, 36]\n[[camera]]\nid = \"rgb\"\ndevice = \"/dev/video0\"\nformat = \"mjpeg\"\nsize = [64, 36]\n[detect]\nrgb = \"mediapipe-ort\"\n[estimate]\nkind = \"ir-pupil\"\n";
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("eye.toml");
        std::fs::write(&path, toml).unwrap();
        let session = Arc::new(FakeSession::empty());
        let session_dyn: Arc<dyn crate::mode::ModeSession> = Arc::clone(&session) as _;
        let ctx = crate::case::TestCtx::new(
            session_dyn,
            testkit::mode_dual(EmitterSetting::On),
            Arc::new(std::sync::atomic::AtomicBool::new(false)),
            Arc::new(std::sync::atomic::AtomicBool::new(false)),
            Duration::from_secs(30),
            Arc::new(crate::case::RunOptions {
                eye_config: Some(path),
                ..crate::case::RunOptions::default()
            }),
        );
        let build: DetectorFactory = testkit::fake_detectors(vec![], None, |_| 0.0);
        let case = DetectionRate {
            p: DetectionParams::default(),
            build,
        };
        let err = case.run(&ctx).unwrap_err();
        match err {
            TestError::Other(msg) => assert!(msg.contains("no [detect] entry"), "{msg}"),
            other => panic!("expected TestError::Other, got {other:?}"),
        }
    }
}
