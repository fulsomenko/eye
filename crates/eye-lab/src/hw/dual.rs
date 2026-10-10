use std::{thread::JoinHandle, time::Duration};

use crossbeam_channel::{Receiver, RecvTimeoutError};
use eye_capture::{CaptureError, FrameSource};
use eye_core::{Illumination, Timestamp};

use crate::{
    case::{
        Measurement, Needs, ParamError, TestCase, TestCtx, TestError, TestOutput, parse_params,
    },
    hw::{Tagging, tagged_ir},
    mode::Role,
    stats,
};

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DualParams {
    pub frames: usize,
    pub warmup: usize,
    pub expected_median_ms: Option<f64>,
    pub median_tolerance_ms: f64,
    pub max_abs_median_ms: f64,
    pub max_spread_ms: f64,
    pub max_drift_ms_per_min: Option<f64>,
    pub min_pair_rate: f64,
    pub min_dark_fraction: Option<f64>,
    pub expected_rgb_fps: Option<f64>,
    pub expected_ir_fps: Option<f64>,
    pub fps_tolerance: f64,
}

impl Default for DualParams {
    fn default() -> Self {
        Self {
            frames: 60,
            warmup: 10,
            expected_median_ms: None,
            median_tolerance_ms: 3.0,
            max_abs_median_ms: 8.0,
            max_spread_ms: 10.0,
            max_drift_ms_per_min: None,
            min_pair_rate: 0.95,
            min_dark_fraction: None,
            expected_rgb_fps: None,
            expected_ir_fps: None,
            fps_tolerance: 1.0,
        }
    }
}

pub type Stamp = (Timestamp, Illumination);
type Reading = Result<Stamp, CaptureError>;

/// Reads (timestamp, illumination) on its own thread until the receiver is dropped or the source fails.
fn spawn_reader(
    name: &str,
    mut source: Box<dyn FrameSource>,
) -> Result<(JoinHandle<()>, Receiver<Reading>), TestError> {
    let (tx, rx) = crossbeam_channel::bounded::<Reading>(64);
    let handle = eye_core::log::spawn_in_current_span(format!("eye-lab-{name}"), move || {
        loop {
            let item = source
                .next_frame()
                .map(|f| (f.header().timestamp, f.header().illumination));
            let stop = item.is_err();
            if tx.send(item).is_err() || stop {
                return;
            }
        }
    })
    .map_err(|e| TestError::Other(format!("spawning the {name} reader: {e}")))?;
    Ok((handle, rx))
}

/// Waits up to `wait` for one reading, then drains the channel. `Ok(true)` once the source reported EndOfStream.
fn take(
    rx: &Receiver<Reading>,
    out: &mut Vec<Stamp>,
    name: &str,
    wait: Duration,
) -> Result<bool, TestError> {
    let first = match rx.recv_timeout(wait) {
        Ok(item) => Some(item),
        Err(RecvTimeoutError::Timeout) => None,
        Err(RecvTimeoutError::Disconnected) => {
            return Err(TestError::Other(format!("{name} reader ended")));
        }
    };
    for item in first.into_iter().chain(rx.try_iter()) {
        match item {
            Ok(stamp) => out.push(stamp),
            Err(CaptureError::EndOfStream) => return Ok(true),
            Err(e) => return Err(e.into()),
        }
    }
    Ok(false)
}

#[derive(Debug, Clone, PartialEq)]
pub struct DualStats {
    /// RGB frames inside the IR time span (each has an IR frame on both sides).
    pub pairs: usize,
    /// Per paired RGB frame: t(nearest IR) - t(RGB), ms. R30 hardware: +3.0.
    pub offsets_ms: Vec<f64>,
    /// Per paired RGB frame: seconds since the first paired RGB frame.
    pub rgb_s: Vec<f64>,
    /// Fraction of paired RGB frames whose nearest IR frame is tagged IrDark.
    pub dark_fraction: f64,
    /// Half the median IR frame interval, ms.
    pub half_period_ms: f64,
    pub rgb_fps: f64,
    pub ir_fps: f64,
}

pub fn dual_stats(rgb: &[Timestamp], ir: &[Stamp]) -> DualStats {
    let ir_t: Vec<Timestamp> = ir.iter().map(|s| s.0).collect();
    let inside: Vec<Timestamp> = match (ir_t.first(), ir_t.last()) {
        (Some(&first), Some(&last)) => rgb
            .iter()
            .copied()
            .filter(|t| (first..=last).contains(t))
            .collect(),
        _ => Vec::new(),
    };
    let nearest = stats::nearest_indices(&inside, &ir_t);
    let offsets_ms: Vec<f64> = inside
        .iter()
        .zip(&nearest)
        .map(|(&r, &j)| stats::ms(ir_t[j]) - stats::ms(r))
        .collect();
    let dark = nearest
        .iter()
        .filter(|&&j| ir[j].1 == Illumination::IrDark)
        .count();
    let rgb_s = inside
        .iter()
        .map(|&t| (stats::ms(t) - stats::ms(inside[0])) / 1e3)
        .collect();
    DualStats {
        pairs: inside.len(),
        offsets_ms,
        rgb_s,
        dark_fraction: if inside.is_empty() {
            f64::NAN
        } else {
            dark as f64 / inside.len() as f64
        },
        half_period_ms: stats::pct(&stats::intervals_ms(&ir_t), 50.0) / 2.0,
        rgb_fps: stats::fps(rgb),
        ir_fps: stats::fps(&ir_t),
    }
}

/// Pure: the measurements of one run.
pub fn dual_output(s: &DualStats, p: &DualParams) -> TestOutput {
    let mut out = TestOutput::default();
    out.push(Measurement::info("pairs", s.pairs as f64, ""));

    let median = stats::pct(&s.offsets_ms, 50.0);
    match p.expected_median_ms {
        Some(e) => out.push(Measurement::within(
            "offset_median_ms",
            median,
            "",
            e - p.median_tolerance_ms,
            e + p.median_tolerance_ms,
        )),
        None => {
            out.push(Measurement::info("offset_median_ms", median, ""));
            out.push(Measurement::at_most(
                "offset_abs_median_ms",
                median.abs(),
                "",
                p.max_abs_median_ms,
            ));
        }
    }

    let p5 = stats::pct(&s.offsets_ms, 5.0);
    let p95 = stats::pct(&s.offsets_ms, 95.0);
    out.push(Measurement::at_most(
        "offset_spread_ms",
        p95 - p5,
        "",
        p.max_spread_ms,
    ));
    out.push(Measurement::info("offset_p5_ms", p5, ""));
    out.push(Measurement::info("offset_p95_ms", p95, ""));
    out.push(Measurement::info(
        "offset_std_ms",
        stats::std_dev(&s.offsets_ms),
        "",
    ));

    let drift = stats::slope_per_min(&s.rgb_s, &s.offsets_ms);
    match p.max_drift_ms_per_min {
        Some(max) => out.push(Measurement::within(
            "drift_ms_per_min",
            drift,
            "",
            -max,
            max,
        )),
        None => out.push(Measurement::info("drift_ms_per_min", drift, "")),
    }

    let pair_rate = if s.offsets_ms.is_empty() {
        f64::NAN
    } else {
        s.offsets_ms
            .iter()
            .filter(|d| d.abs() < s.half_period_ms)
            .count() as f64
            / s.offsets_ms.len() as f64
    };
    out.push(Measurement::at_least(
        "pair_rate",
        pair_rate,
        "",
        p.min_pair_rate,
    ));

    match p.min_dark_fraction {
        Some(min) => out.push(Measurement::at_least(
            "dark_fraction",
            s.dark_fraction,
            "",
            min,
        )),
        None => out.push(Measurement::info("dark_fraction", s.dark_fraction, "")),
    }

    match p.expected_rgb_fps {
        Some(e) => out.push(Measurement::within(
            "rgb.fps",
            s.rgb_fps,
            "",
            e - p.fps_tolerance,
            e + p.fps_tolerance,
        )),
        None => out.push(Measurement::info("rgb.fps", s.rgb_fps, "")),
    }
    match p.expected_ir_fps {
        Some(e) => out.push(Measurement::within(
            "ir.fps",
            s.ir_fps,
            "",
            e - p.fps_tolerance,
            e + p.fps_tolerance,
        )),
        None => out.push(Measurement::info("ir.fps", s.ir_fps, "")),
    }

    out
}

#[derive(Debug)]
struct DualSync {
    p: DualParams,
}

impl TestCase for DualSync {
    fn name(&self) -> &'static str {
        "dual_sync"
    }

    fn needs(&self) -> Needs {
        Needs {
            rgb: true,
            ir: true,
            ..Needs::default()
        }
    }

    fn default_timeout(&self) -> Duration {
        let p = &self.p;
        Duration::from_secs_f64(15.0 + (p.warmup + p.frames) as f64 / 7.5)
    }

    fn run(&self, ctx: &TestCtx) -> Result<TestOutput, TestError> {
        let p = &self.p;
        let need = p.warmup + p.frames;
        let ir_source: Box<dyn FrameSource> = Box::new(tagged_ir(ctx, Tagging::Auto)?);
        let (ir_handle, ir_rx) = spawn_reader("ir", ir_source)?;
        let mut ir: Vec<Stamp> = Vec::new();
        let mut ir_ended = false;
        while ir.is_empty() {
            ctx.check()?;
            ir_ended = take(&ir_rx, &mut ir, "ir", Duration::from_millis(10))?;
            if ir_ended && ir.is_empty() {
                return Err(TestError::Other(
                    "ir stream ended before its first frame".into(),
                ));
            }
        }
        let (rgb_handle, rgb_rx) = spawn_reader("rgb", ctx.session().open(Role::Rgb)?)?;
        let mut rgb: Vec<Stamp> = Vec::with_capacity(need);
        while rgb.len() < need {
            ctx.check()?;
            let rgb_ended = take(&rgb_rx, &mut rgb, "rgb", Duration::from_millis(10))?;
            if rgb_ended && rgb.len() < need {
                return Err(CaptureError::EndOfStream.into());
            }
            if !ir_ended {
                ir_ended = take(&ir_rx, &mut ir, "ir", Duration::ZERO)?;
            }
        }
        rgb.truncate(need);
        let last_rgb = rgb[need - 1].0;
        while !ir_ended && ir.last().is_some_and(|s| s.0 <= last_rgb) {
            ctx.check()?;
            ir_ended = take(&ir_rx, &mut ir, "ir", Duration::from_millis(10))?;
        }
        drop((rgb_rx, ir_rx));
        let _ = (rgb_handle.join(), ir_handle.join());
        let rgb_t: Vec<Timestamp> = rgb[p.warmup..].iter().map(|s| s.0).collect();
        Ok(dual_output(&dual_stats(&rgb_t, &ir), p))
    }
}

pub fn build_dual_sync(params: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    let p: DualParams = parse_params(params)?;
    if p.frames == 0 {
        return Err(ParamError::Invalid("frames must be >= 1".into()));
    }
    for (name, v) in [
        ("median_tolerance_ms", p.median_tolerance_ms),
        ("max_abs_median_ms", p.max_abs_median_ms),
        ("max_spread_ms", p.max_spread_ms),
        ("min_pair_rate", p.min_pair_rate),
        ("fps_tolerance", p.fps_tolerance),
    ] {
        if !v.is_finite() || v < 0.0 {
            return Err(ParamError::Invalid(format!(
                "{name} must be finite and >= 0"
            )));
        }
    }
    for (name, v) in [
        ("expected_rgb_fps", p.expected_rgb_fps),
        ("expected_ir_fps", p.expected_ir_fps),
    ] {
        if let Some(e) = v
            && (!e.is_finite() || e <= 0.0)
        {
            return Err(ParamError::Invalid(format!(
                "{name} must be finite and > 0"
            )));
        }
    }
    Ok(Box::new(DualSync { p }))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{mode::EmitterSetting, testkit};

    fn ir_fixture(n: usize) -> Vec<Stamp> {
        (0..n as u64)
            .map(|k| {
                (
                    Timestamp::from_nanos(100_000_000 + k * 66_000_000),
                    if k % 2 == 0 {
                        Illumination::IrDark
                    } else {
                        Illumination::IrLit
                    },
                )
            })
            .collect()
    }

    fn rgb_fixture(n: usize, first_k: u64, lead_ms: f64, drift_ns: i64) -> Vec<Timestamp> {
        (0..n as u64)
            .map(|j| {
                let ir_k = first_k + 2 * j;
                let ir_t = 100_000_000i64 + ir_k as i64 * 66_000_000;
                let t = ir_t - (lead_ms * 1e6) as i64 + drift_ns * j as i64;
                Timestamp::from_nanos(t.max(0) as u64)
            })
            .collect()
    }

    #[test]
    fn test_dual_stats_r30_pattern() {
        let ir = ir_fixture(84);
        let rgb = rgb_fixture(40, 2, 3.0, 0);
        let s = dual_stats(&rgb, &ir);
        assert_eq!(s.pairs, 40);
        approx::assert_abs_diff_eq!(stats::pct(&s.offsets_ms, 50.0), 3.0, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(
            stats::pct(&s.offsets_ms, 95.0) - stats::pct(&s.offsets_ms, 5.0),
            0.0,
            epsilon = 1e-6
        );
        assert_eq!(s.dark_fraction, 1.0);
        approx::assert_abs_diff_eq!(s.half_period_ms, 33.0, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(s.rgb_fps, 1000.0 / 132.0, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(s.ir_fps, 1000.0 / 66.0, epsilon = 1e-6);
    }

    #[test]
    fn test_dual_stats_rgb_on_lit_frames_has_zero_dark_fraction() {
        let ir = ir_fixture(84);
        let rgb = rgb_fixture(40, 3, 3.0, 0);
        let s = dual_stats(&rgb, &ir);
        assert_eq!(s.dark_fraction, 0.0);
    }

    #[test]
    fn test_dual_stats_excludes_rgb_outside_ir_span() {
        let ir = ir_fixture(84);
        let rgb = rgb_fixture(40, 0, 3.0, 0);
        let s = dual_stats(&rgb, &ir);
        assert_eq!(s.pairs, 39);
    }

    #[test]
    fn test_dual_stats_drift() {
        let ir = ir_fixture(84);
        let rgb = rgb_fixture(40, 2, 3.0, 1000);
        let s = dual_stats(&rgb, &ir);
        let drift = stats::slope_per_min(&s.rgb_s, &s.offsets_ms);
        approx::assert_abs_diff_eq!(drift, -0.06 / 0.132_001, epsilon = 1e-6);
    }

    fn output_for(first_k: u64, lead_ms: f64, drift_ns: i64, p: DualParams) -> TestOutput {
        let ir = ir_fixture(84);
        let rgb = rgb_fixture(40, first_k, lead_ms, drift_ns);
        let s = dual_stats(&rgb, &ir);
        dual_output(&s, &p)
    }

    #[test]
    fn test_dual_output_expected_median_is_the_regression_check() {
        let p = DualParams {
            expected_median_ms: Some(3.0),
            ..DualParams::default()
        };
        let out = output_for(2, 3.0, 0, p.clone());
        assert!(out.passed());
        assert!(
            !out.measurements
                .iter()
                .any(|m| m.name == "offset_abs_median_ms")
        );

        let out = output_for(2, 10.0, 0, p);
        let m = out
            .measurements
            .iter()
            .find(|m| m.name == "offset_median_ms")
            .unwrap();
        assert!(!m.pass);
    }

    #[test]
    fn test_dual_output_abs_median_rule_without_expected() {
        let out = output_for(2, 12.0, 0, DualParams::default());
        let abs_median = out
            .measurements
            .iter()
            .find(|m| m.name == "offset_abs_median_ms")
            .unwrap();
        assert_eq!(abs_median.value, 12.0);
        assert!(!abs_median.pass);
        let pair_rate = out
            .measurements
            .iter()
            .find(|m| m.name == "pair_rate")
            .unwrap();
        assert_eq!(pair_rate.value, 1.0);
    }

    #[test]
    fn test_dual_output_drift_fails_only_when_limited() {
        let out = output_for(
            2,
            3.0,
            1000,
            DualParams {
                max_drift_ms_per_min: Some(0.4),
                ..DualParams::default()
            },
        );
        let drift = out
            .measurements
            .iter()
            .find(|m| m.name == "drift_ms_per_min")
            .unwrap();
        assert!(!drift.pass);

        let out = output_for(2, 3.0, 1000, DualParams::default());
        let drift = out
            .measurements
            .iter()
            .find(|m| m.name == "drift_ms_per_min")
            .unwrap();
        assert!(drift.pass);
    }

    #[test]
    fn test_dual_output_dark_and_fps_expectations() {
        let p = DualParams {
            min_dark_fraction: Some(0.95),
            expected_rgb_fps: Some(7.5),
            expected_ir_fps: Some(15.0),
            ..DualParams::default()
        };
        let out = output_for(2, 3.0, 0, p.clone());
        assert!(out.passed());

        let out = output_for(3, 3.0, 0, p.clone());
        assert!(
            !out.measurements
                .iter()
                .find(|m| m.name == "dark_fraction")
                .unwrap()
                .pass
        );

        let p2 = DualParams {
            expected_rgb_fps: Some(30.0),
            ..p
        };
        let out = output_for(2, 3.0, 0, p2);
        assert!(
            !out.measurements
                .iter()
                .find(|m| m.name == "rgb.fps")
                .unwrap()
                .pass
        );
    }

    #[test]
    fn test_dual_sync_run_through_readers() {
        let ir_frames = testkit::synth_gray("ir", 170, 0, 100_000_000, 66_000_000, |seq| {
            if seq % 2 == 1 { 46 } else { 0 }
        });
        let rgb_t_ns: Vec<u64> = (0..70u64)
            .map(|j| 100_000_000 + (2 + 2 * j) * 66_000_000 - 3_000_000)
            .collect();
        let rgb_frames = testkit::mjpeg_at("rgb", &rgb_t_ns, 64, 36);
        let session = Arc::new(
            testkit::FakeSession::empty()
                .with_source(Role::Ir, testkit::boxed(ir_frames))
                .with_source(Role::Rgb, testkit::boxed(rgb_frames)),
        );
        let ctx = testkit::ctx(session, testkit::mode_dual(EmitterSetting::On));
        let mut table = toml::Table::new();
        table.insert("warmup".into(), toml::Value::Integer(10));
        table.insert("frames".into(), toml::Value::Integer(60));
        table.insert("expected_median_ms".into(), toml::Value::Float(3.0));
        table.insert("min_dark_fraction".into(), toml::Value::Float(0.95));
        let case = build_dual_sync(&table).unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(out.passed());
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "pairs")
                .unwrap()
                .value,
            60.0
        );
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "dark_fraction")
                .unwrap()
                .value,
            1.0
        );
    }

    #[test]
    fn test_dual_sync_rgb_stream_end_is_capture_error() {
        let ir_frames = testkit::synth_gray("ir", 170, 0, 100_000_000, 66_000_000, |seq| {
            if seq % 2 == 1 { 46 } else { 0 }
        });
        let rgb_frames = testkit::synth_mjpeg("rgb", 20, 0, 100_000_000, 132_000_000);
        let session = Arc::new(
            testkit::FakeSession::empty()
                .with_source(Role::Ir, testkit::boxed(ir_frames))
                .with_source(Role::Rgb, testkit::boxed(rgb_frames)),
        );
        let ctx = testkit::ctx(session, testkit::mode_dual(EmitterSetting::On));
        let case = build_dual_sync(&toml::Table::new()).unwrap();
        let err = case.run(&ctx).unwrap_err();
        assert!(matches!(err, TestError::Capture(CaptureError::EndOfStream)));
    }
}
