use std::time::Duration;

use crate::{
    case::{
        Measurement, Needs, ParamError, TestCase, TestCtx, TestError, TestOutput, parse_params,
    },
    hw::{Grab, RoleSel, grab},
    mode::Role,
    modes::opener::pixel_format,
    stats,
};

/// Opens, grabs `n` frames from, and drops each role `sel` selects in `ctx.mode()`, RGB first,
/// calling `f` with the grab before moving to the next role.
fn per_role(
    ctx: &TestCtx,
    sel: RoleSel,
    n: usize,
    mut f: impl FnMut(Role, &Grab, &mut TestOutput),
) -> Result<TestOutput, TestError> {
    let mut out = TestOutput::default();
    for role in sel.roles(ctx.mode()) {
        let mut source = ctx.session().open(role)?;
        let g = grab(ctx, source.as_mut(), n)?;
        drop(source);
        f(role, &g, &mut out);
    }
    Ok(out)
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpensParams {
    pub role: RoleSel,
    pub frames: usize,
    pub max_first_frame_ms: f64,
}

impl Default for OpensParams {
    fn default() -> Self {
        Self {
            role: RoleSel::default(),
            frames: 5,
            max_first_frame_ms: 2000.0,
        }
    }
}

#[derive(Debug)]
struct Opens {
    params: OpensParams,
}

impl TestCase for Opens {
    fn name(&self) -> &'static str {
        "opens"
    }

    fn needs(&self) -> Needs {
        self.params.role.needs()
    }

    fn default_timeout(&self) -> Duration {
        Duration::from_secs(10)
    }

    fn run(&self, ctx: &TestCtx) -> Result<TestOutput, TestError> {
        let p = &self.params;
        per_role(ctx, p.role, p.frames, |role, g, out| {
            let r = crate::hw::prefix(role);
            out.push(Measurement::at_least(
                format!("{r}.frames"),
                g.frames.len() as f64,
                "",
                p.frames as f64,
            ));
            out.push(Measurement::at_most(
                format!("{r}.first_frame_ms"),
                g.first_frame.as_secs_f64() * 1e3,
                "",
                p.max_first_frame_ms,
            ));
        })
    }
}

pub fn build_opens(params: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    let p: OpensParams = parse_params(params)?;
    if p.frames == 0 {
        return Err(ParamError::Invalid("frames must be >= 1".into()));
    }
    if !p.max_first_frame_ms.is_finite() || p.max_first_frame_ms < 0.0 {
        return Err(ParamError::Invalid(
            "max_first_frame_ms must be finite and >= 0".into(),
        ));
    }
    Ok(Box::new(Opens { params: p }))
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MeasuredFpsParams {
    pub role: RoleSel,
    pub frames: usize,
    pub warmup: usize,
    pub expected_fps: Option<f64>,
    pub fps_tolerance: f64,
    pub max_gap_factor: f64,
}

impl Default for MeasuredFpsParams {
    fn default() -> Self {
        Self {
            role: RoleSel::default(),
            frames: 90,
            warmup: 5,
            expected_fps: None,
            fps_tolerance: 1.0,
            max_gap_factor: 1.5,
        }
    }
}

#[derive(Debug)]
struct MeasuredFps {
    params: MeasuredFpsParams,
}

impl TestCase for MeasuredFps {
    fn name(&self) -> &'static str {
        "measured_fps"
    }

    fn needs(&self) -> Needs {
        self.params.role.needs()
    }

    fn default_timeout(&self) -> Duration {
        let p = &self.params;
        Duration::from_secs_f64(10.0 + 2.0 * (p.warmup + p.frames) as f64 / 15.0)
    }

    fn run(&self, ctx: &TestCtx) -> Result<TestOutput, TestError> {
        let p = &self.params;
        per_role(ctx, p.role, p.warmup + p.frames, |role, g, out| {
            let r = crate::hw::prefix(role);
            let target = ctx.mode().target(role).expect("role came from mode");
            let e = p.expected_fps.unwrap_or(f64::from(target.format.fps));
            let post: Vec<_> = g.frames[p.warmup..].iter().map(|f| f.t).collect();
            let measured = stats::fps(&post);
            out.push(Measurement::within(
                format!("{r}.fps"),
                measured,
                "",
                e - p.fps_tolerance,
                e + p.fps_tolerance,
            ));
            let intervals = stats::intervals_ms(&post);
            let max_interval = intervals
                .iter()
                .copied()
                .fold(f64::NAN, |acc, v| if acc.is_nan() { v } else { acc.max(v) });
            out.push(Measurement::at_most(
                format!("{r}.max_interval_ms"),
                max_interval,
                "",
                p.max_gap_factor * 1000.0 / e,
            ));
            out.push(Measurement::info(
                format!("{r}.interval_std_ms"),
                stats::std_dev(&intervals),
                "",
            ));
            let seqs: Vec<u64> = g.frames[p.warmup..].iter().map(|f| f.seq).collect();
            out.push(Measurement::info(
                format!("{r}.seq_gaps"),
                stats::seq_gaps(&seqs) as f64,
                "",
            ));
        })
    }
}

pub fn build_measured_fps(params: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    let p: MeasuredFpsParams = parse_params(params)?;
    if p.frames == 0 {
        return Err(ParamError::Invalid("frames must be >= 1".into()));
    }
    if let Some(e) = p.expected_fps
        && (!e.is_finite() || e <= 0.0)
    {
        return Err(ParamError::Invalid(
            "expected_fps must be finite and > 0".into(),
        ));
    }
    if !p.fps_tolerance.is_finite() || p.fps_tolerance < 0.0 {
        return Err(ParamError::Invalid(
            "fps_tolerance must be finite and >= 0".into(),
        ));
    }
    if !p.max_gap_factor.is_finite() || p.max_gap_factor < 0.0 {
        return Err(ParamError::Invalid(
            "max_gap_factor must be finite and >= 0".into(),
        ));
    }
    Ok(Box::new(MeasuredFps { params: p }))
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MonotonicTimestampsParams {
    pub role: RoleSel,
    pub frames: usize,
    pub max_age_ms: f64,
}

impl Default for MonotonicTimestampsParams {
    fn default() -> Self {
        Self {
            role: RoleSel::default(),
            frames: 60,
            max_age_ms: 200.0,
        }
    }
}

#[derive(Debug)]
struct MonotonicTimestamps {
    params: MonotonicTimestampsParams,
}

impl TestCase for MonotonicTimestamps {
    fn name(&self) -> &'static str {
        "monotonic_timestamps"
    }

    fn needs(&self) -> Needs {
        self.params.role.needs()
    }

    fn default_timeout(&self) -> Duration {
        let p = &self.params;
        Duration::from_secs_f64(10.0 + 2.0 * p.frames as f64 / 15.0)
    }

    fn run(&self, ctx: &TestCtx) -> Result<TestOutput, TestError> {
        let p = &self.params;
        per_role(ctx, p.role, p.frames, |role, g, out| {
            let r = crate::hw::prefix(role);
            let non_increasing = g.frames.windows(2).filter(|w| w[1].t <= w[0].t).count();
            out.push(Measurement::at_most(
                format!("{r}.non_increasing"),
                non_increasing as f64,
                "",
                0.0,
            ));
            let ages: Vec<f64> = g
                .frames
                .iter()
                .map(|f| stats::ms(f.received) - stats::ms(f.t))
                .collect();
            let max_age = ages
                .iter()
                .copied()
                .fold(f64::NAN, |acc, v| if acc.is_nan() { v } else { acc.max(v) });
            let min_age = ages
                .iter()
                .copied()
                .fold(f64::NAN, |acc, v| if acc.is_nan() { v } else { acc.min(v) });
            out.push(Measurement::at_most(
                format!("{r}.max_age_ms"),
                max_age,
                "",
                p.max_age_ms,
            ));
            out.push(Measurement::at_least(
                format!("{r}.min_age_ms"),
                min_age,
                "",
                0.0,
            ));
        })
    }
}

pub fn build_monotonic_timestamps(params: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    let p: MonotonicTimestampsParams = parse_params(params)?;
    if p.frames == 0 {
        return Err(ParamError::Invalid("frames must be >= 1".into()));
    }
    if !p.max_age_ms.is_finite() || p.max_age_ms < 0.0 {
        return Err(ParamError::Invalid(
            "max_age_ms must be finite and >= 0".into(),
        ));
    }
    Ok(Box::new(MonotonicTimestamps { params: p }))
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FrameFormatParams {
    pub role: RoleSel,
    pub frames: usize,
}

impl Default for FrameFormatParams {
    fn default() -> Self {
        Self {
            role: RoleSel::default(),
            frames: 10,
        }
    }
}

#[derive(Debug)]
struct FrameFormat {
    params: FrameFormatParams,
}

impl TestCase for FrameFormat {
    fn name(&self) -> &'static str {
        "frame_format"
    }

    fn needs(&self) -> Needs {
        self.params.role.needs()
    }

    fn default_timeout(&self) -> Duration {
        let p = &self.params;
        Duration::from_secs_f64(10.0 + 2.0 * p.frames as f64 / 15.0)
    }

    fn run(&self, ctx: &TestCtx) -> Result<TestOutput, TestError> {
        let p = &self.params;
        per_role(ctx, p.role, p.frames, |role, g, out| {
            let r = crate::hw::prefix(role);
            let target = ctx.mode().target(role).expect("role came from mode");
            let expected_format = pixel_format(&target.format.fourcc);
            let mismatched = g
                .frames
                .iter()
                .filter(|f| {
                    f.width != target.format.width
                        || f.height != target.format.height
                        || Some(f.format) != expected_format
                })
                .count();
            let invalid_payload = g.frames.iter().filter(|f| !f.payload_ok).count();
            out.push(Measurement::at_most(
                format!("{r}.mismatched"),
                mismatched as f64,
                "",
                0.0,
            ));
            out.push(Measurement::at_most(
                format!("{r}.invalid_payload"),
                invalid_payload as f64,
                "",
                0.0,
            ));
        })
    }
}

pub fn build_frame_format(params: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    let p: FrameFormatParams = parse_params(params)?;
    if p.frames == 0 {
        return Err(ParamError::Invalid("frames must be >= 1".into()));
    }
    Ok(Box::new(FrameFormat { params: p }))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        hw::RoleSel,
        testkit::{self, LiveStampedSource},
    };

    fn run_case(case: &dyn TestCase, ctx: &TestCtx) -> TestOutput {
        case.run(ctx).expect("case runs")
    }

    #[test]
    fn test_opens_reports_frames_and_first_frame_latency() {
        let frames = testkit::synth_mjpeg("rgb", 5, 0, 0, 33_333_333);
        let session =
            Arc::new(testkit::FakeSession::empty().with_source(Role::Rgb, testkit::boxed(frames)));
        let ctx = testkit::ctx(session, testkit::mode_rgb());
        let case = build_opens(&toml::Table::new()).unwrap();
        let out = run_case(&*case, &ctx);
        assert!(out.passed());
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "rgb.frames")
                .unwrap()
                .value,
            5.0
        );
        assert!(
            out.measurements
                .iter()
                .find(|m| m.name == "rgb.first_frame_ms")
                .unwrap()
                .value
                < 100.0
        );
    }

    #[test]
    fn test_opens_all_covers_both_streams_in_dual_mode() {
        let rgb = testkit::synth_mjpeg("rgb", 5, 0, 0, 33_333_333);
        let ir = testkit::synth_gray("ir", 5, 0, 0, 66_666_666, |_| 0);
        let session = Arc::new(
            testkit::FakeSession::empty()
                .with_source(Role::Rgb, testkit::boxed(rgb))
                .with_source(Role::Ir, testkit::boxed(ir)),
        );
        let ctx = testkit::ctx(
            session,
            testkit::mode_dual(crate::mode::EmitterSetting::Keep),
        );
        let case = build_opens(&toml::Table::new()).unwrap();
        let out = run_case(&*case, &ctx);
        let names: Vec<&str> = out.measurements.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "rgb.frames",
                "rgb.first_frame_ms",
                "ir.frames",
                "ir.first_frame_ms"
            ]
        );
    }

    #[test]
    fn test_opens_errors_when_stream_ends_early() {
        let frames = testkit::synth_mjpeg("rgb", 2, 0, 0, 33_333_333);
        let session =
            Arc::new(testkit::FakeSession::empty().with_source(Role::Rgb, testkit::boxed(frames)));
        let ctx = testkit::ctx(session, testkit::mode_rgb());
        let case = build_opens(&toml::Table::new()).unwrap();
        let err = case.run(&ctx).unwrap_err();
        assert!(matches!(
            err,
            TestError::Capture(eye_capture::CaptureError::EndOfStream)
        ));
    }

    #[test]
    fn test_measured_fps_30hz_passes() {
        let frames = testkit::synth_mjpeg("rgb", 95, 0, 0, 33_333_333);
        let session =
            Arc::new(testkit::FakeSession::empty().with_source(Role::Rgb, testkit::boxed(frames)));
        let ctx = testkit::ctx(session, testkit::mode_rgb());
        let case = build_measured_fps(&toml::Table::new()).unwrap();
        let out = run_case(&*case, &ctx);
        assert!(out.passed());
        let fps = out
            .measurements
            .iter()
            .find(|m| m.name == "rgb.fps")
            .unwrap()
            .value;
        approx::assert_relative_eq!(fps, 30.0, epsilon = 1e-6);
        let max_interval = out
            .measurements
            .iter()
            .find(|m| m.name == "rgb.max_interval_ms")
            .unwrap()
            .value;
        approx::assert_relative_eq!(max_interval, 33.333, epsilon = 1e-3);
    }

    #[test]
    fn test_measured_fps_15hz_fails_against_advertised_30() {
        let frames = testkit::synth_mjpeg("rgb", 95, 0, 0, 66_666_666);
        let session =
            Arc::new(testkit::FakeSession::empty().with_source(Role::Rgb, testkit::boxed(frames)));
        let ctx = testkit::ctx(session, testkit::mode_rgb());
        let case = build_measured_fps(&toml::Table::new()).unwrap();
        let out = run_case(&*case, &ctx);
        assert!(!out.passed());
        let fps = out
            .measurements
            .iter()
            .find(|m| m.name == "rgb.fps")
            .unwrap()
            .value;
        approx::assert_relative_eq!(fps, 15.0, epsilon = 1e-6);
    }

    #[test]
    fn test_measured_fps_expected_fps_override_passes_ir_15hz() {
        let frames = testkit::synth_gray("ir", 95, 0, 0, 66_666_666, |_| 0);
        let session =
            Arc::new(testkit::FakeSession::empty().with_source(Role::Ir, testkit::boxed(frames)));
        let ctx = testkit::ctx(session, testkit::mode_ir(crate::mode::EmitterSetting::Keep));
        let mut table = toml::Table::new();
        table.insert("expected_fps".into(), toml::Value::Float(15.0));
        let case = build_measured_fps(&table).unwrap();
        let out = run_case(&*case, &ctx);
        assert!(out.passed());
        let fps = out
            .measurements
            .iter()
            .find(|m| m.name == "ir.fps")
            .unwrap()
            .value;
        approx::assert_relative_eq!(fps, 15.0, epsilon = 1e-6);
        let max_interval = out
            .measurements
            .iter()
            .find(|m| m.name == "ir.max_interval_ms")
            .unwrap();
        approx::assert_relative_eq!(max_interval.value, 66.667, epsilon = 1e-3);
        assert!(max_interval.value <= 100.0);
    }

    #[test]
    fn test_single_stall_fails_max_interval() {
        let mut t_ns = 0u64;
        let mut times = Vec::with_capacity(95);
        for i in 0..95u64 {
            times.push(t_ns);
            let step = if i == 50 { 100_000_000 } else { 33_333_333 };
            t_ns += step;
        }
        let frames = testkit::mjpeg_at("rgb", &times, 64, 36);
        let session =
            Arc::new(testkit::FakeSession::empty().with_source(Role::Rgb, testkit::boxed(frames)));
        let ctx = testkit::ctx(session, testkit::mode_rgb());
        let case = build_measured_fps(&toml::Table::new()).unwrap();
        let out = run_case(&*case, &ctx);
        assert!(!out.passed());
        let max_interval = out
            .measurements
            .iter()
            .find(|m| m.name == "rgb.max_interval_ms")
            .unwrap()
            .value;
        approx::assert_relative_eq!(max_interval, 100.0, epsilon = 1e-3);
        let fps = out
            .measurements
            .iter()
            .find(|m| m.name == "rgb.fps")
            .unwrap();
        assert!(fps.pass);
        assert!(fps.value >= 29.0 && fps.value <= 31.0);
    }

    #[test]
    fn test_live_stamped_timestamps_pass() {
        let frames = testkit::synth_gray("ir", 60, 0, 0, 66_666_666, |_| 0);
        let source = LiveStampedSource {
            inner: testkit::FakeSource {
                info: testkit::camera_info("ir", eye_core::PixelFormat::Gray8, 64, 36),
                frames: frames.into(),
            },
            age: std::time::Duration::from_millis(5),
        };
        let session =
            Arc::new(testkit::FakeSession::empty().with_source(Role::Ir, Box::new(source)));
        let ctx = testkit::ctx(session, testkit::mode_ir(crate::mode::EmitterSetting::Keep));
        let case = build_monotonic_timestamps(&toml::Table::new()).unwrap();
        let out = run_case(&*case, &ctx);
        assert!(out.passed());
        let min_age = out
            .measurements
            .iter()
            .find(|m| m.name == "ir.min_age_ms")
            .unwrap()
            .value;
        let max_age = out
            .measurements
            .iter()
            .find(|m| m.name == "ir.max_age_ms")
            .unwrap()
            .value;
        assert!(min_age >= 0.0);
        assert!(max_age < 200.0);
    }

    #[test]
    fn test_wallclock_timestamps_fail_negative_age() {
        let base = 1_791_400_000u64 * 1_000_000_000;
        let frames = testkit::synth_gray("ir", 60, 0, base, 66_666_666, |_| 0);
        let session =
            Arc::new(testkit::FakeSession::empty().with_source(Role::Ir, testkit::boxed(frames)));
        let ctx = testkit::ctx(session, testkit::mode_ir(crate::mode::EmitterSetting::Keep));
        let case = build_monotonic_timestamps(&toml::Table::new()).unwrap();
        let out = run_case(&*case, &ctx);
        assert!(!out.passed());
        let min_age = out
            .measurements
            .iter()
            .find(|m| m.name == "ir.min_age_ms")
            .unwrap()
            .value;
        assert!(min_age < 0.0);
    }

    #[test]
    fn test_repeated_timestamp_counts_non_increasing() {
        let mut frames = testkit::synth_gray("ir", 60, 0, 0, 66_666_666, |_| 0);
        let (mut h3, d3) = frames[3].clone().into_parts();
        let (h4, _) = frames[4].clone().into_parts();
        h3.timestamp = h4.timestamp;
        frames[3] = eye_core::Frame::new(h3, d3).unwrap();
        let session =
            Arc::new(testkit::FakeSession::empty().with_source(Role::Ir, testkit::boxed(frames)));
        let ctx = testkit::ctx(session, testkit::mode_ir(crate::mode::EmitterSetting::Keep));
        let case = build_monotonic_timestamps(&toml::Table::new()).unwrap();
        let out = run_case(&*case, &ctx);
        assert!(!out.passed());
        let non_increasing = out
            .measurements
            .iter()
            .find(|m| m.name == "ir.non_increasing")
            .unwrap()
            .value;
        assert_eq!(non_increasing, 1.0);
    }

    #[test]
    fn test_frame_format_matches_mode_target() {
        let rgb = testkit::synth_mjpeg("rgb", 10, 0, 0, 33_333_333);
        let ir = testkit::synth_gray("ir", 10, 0, 0, 66_666_666, |_| 0);
        let session = Arc::new(
            testkit::FakeSession::empty()
                .with_source(Role::Rgb, testkit::boxed(rgb))
                .with_source(Role::Ir, testkit::boxed(ir)),
        );
        let ctx = testkit::ctx(
            session,
            testkit::mode_dual(crate::mode::EmitterSetting::Keep),
        );
        let case = build_frame_format(&toml::Table::new()).unwrap();
        let out = run_case(&*case, &ctx);
        assert!(out.passed());
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "rgb.mismatched")
                .unwrap()
                .value,
            0.0
        );
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "ir.invalid_payload")
                .unwrap()
                .value,
            0.0
        );
    }

    #[test]
    fn test_frame_format_flags_wrong_size() {
        let frames: Vec<eye_core::Frame> = (0..10u64)
            .map(|seq| {
                eye_core::Frame::new(
                    eye_core::FrameHeader {
                        camera: eye_core::CameraId::from("ir"),
                        seq,
                        timestamp: eye_core::Timestamp::from_nanos(seq * 66_666_666),
                        width: 32,
                        height: 18,
                        format: eye_core::PixelFormat::Gray8,
                        illumination: eye_core::Illumination::Unknown,
                    },
                    Arc::from(vec![0u8; 32 * 18]),
                )
                .unwrap()
            })
            .collect();
        let session =
            Arc::new(testkit::FakeSession::empty().with_source(Role::Ir, testkit::boxed(frames)));
        let ctx = testkit::ctx(session, testkit::mode_ir(crate::mode::EmitterSetting::Keep));
        let case = build_frame_format(&toml::Table::new()).unwrap();
        let out = run_case(&*case, &ctx);
        assert!(!out.passed());
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "ir.mismatched")
                .unwrap()
                .value,
            10.0
        );
    }

    #[test]
    fn test_roles_filter_follows_mode() {
        assert_eq!(
            RoleSel::Ir.roles(&testkit::mode_dual(crate::mode::EmitterSetting::Keep)),
            vec![Role::Ir]
        );
        assert_eq!(RoleSel::All.roles(&testkit::mode_rgb()), vec![Role::Rgb]);
    }
}
