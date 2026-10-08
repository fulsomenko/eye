use std::time::{Duration, Instant};

use eye::tracker::{Tracker, TrackerError, TrackerStats};

use crate::{
    case::{
        Measurement, Needs, ParamError, TestCase, TestCtx, TestError, TestOutput, parse_params,
    },
    mode::Role,
    modes::opener::camera_id,
    pipeline::{ScreenParams, lab_config, lab_rig, validate_timing},
};

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LatencyParams {
    pub seconds: f64,
    pub lead_s: f64,
    pub max_p95_ms: f64,
    pub min_points_hz: f64,
    pub screen: ScreenParams,
}

impl Default for LatencyParams {
    fn default() -> Self {
        Self {
            seconds: 10.0,
            lead_s: 3.0,
            max_p95_ms: 50.0,
            min_points_hz: 6.0,
            screen: ScreenParams::default(),
        }
    }
}

#[derive(Debug)]
pub struct TrackerLatency {
    p: LatencyParams,
}

impl TestCase for TrackerLatency {
    fn name(&self) -> &'static str {
        "tracker_latency"
    }

    fn needs(&self) -> Needs {
        Needs {
            subject: true,
            any_stream: true,
            ..Needs::default()
        }
    }

    fn default_timeout(&self) -> Duration {
        Duration::from_secs_f64(self.p.seconds + self.p.lead_s + 20.0)
    }

    fn run(&self, ctx: &TestCtx) -> Result<TestOutput, TestError> {
        let p = &self.p;
        let config = lab_config(ctx)?;
        let streams: Vec<(Role, u32, u32)> = [Role::Rgb, Role::Ir]
            .into_iter()
            .filter(|&role| config.camera(camera_id(role).as_str()).is_some())
            .filter_map(|role| {
                ctx.mode()
                    .target(role)
                    .map(|t| (role, t.format.width, t.format.height))
            })
            .collect();
        let rig = lab_rig(&p.screen, &streams)?;
        ctx.instruct(&format!(
            "sit about 50 cm from the screen and look around it for {} s, starting in {} s",
            p.seconds, p.lead_s
        ));
        ctx.sleep(Duration::from_secs_f64(p.lead_s))?;
        let mut tracker = Tracker::from_config(&config, rig, None).map_err(tracker_error)?;
        let start = Instant::now();
        let mut points = 0usize;
        while start.elapsed() < Duration::from_secs_f64(p.seconds) {
            ctx.check()?;
            match tracker.next_timeout(Duration::from_millis(200)) {
                Ok(Some(_)) => points += 1,
                Ok(None) => {}
                Err(e) => return Err(tracker_error(e)),
            }
        }
        let stats = tracker.stats();
        let elapsed = start.elapsed();
        tracker.shutdown().map_err(tracker_error)?;
        Ok(latency_output(&stats, points, elapsed, p))
    }
}

fn tracker_error(e: TrackerError) -> TestError {
    match e {
        TrackerError::Capture { source, .. } => TestError::Capture(source),
        other => TestError::Other(other.to_string()),
    }
}

/// Pure, unit-tested.
pub fn latency_output(
    stats: &TrackerStats,
    points: usize,
    elapsed: Duration,
    p: &LatencyParams,
) -> TestOutput {
    let ms = |d: Duration| {
        if stats.capture_to_emit.count == 0 {
            f64::NAN
        } else {
            d.as_secs_f64() * 1e3
        }
    };
    let mut out = TestOutput::default();
    out.push(Measurement::at_most(
        "capture_to_emit_p95_ms",
        ms(stats.capture_to_emit.p95),
        "",
        p.max_p95_ms,
    ));
    out.push(Measurement::info(
        "capture_to_emit_p50_ms",
        ms(stats.capture_to_emit.p50),
        "",
    ));
    out.push(Measurement::info(
        "capture_to_emit_max_ms",
        ms(stats.capture_to_emit.max),
        "",
    ));
    let elapsed_s = elapsed.as_secs_f64();
    let points_hz = if elapsed_s > 0.0 {
        points as f64 / elapsed_s
    } else {
        f64::NAN
    };
    out.push(Measurement::at_least(
        "points_hz",
        points_hz,
        "",
        p.min_points_hz,
    ));
    out.push(Measurement::info("framesets", stats.framesets as f64, ""));
    out.push(Measurement::info(
        "frames_dropped",
        stats.frames_dropped.values().sum::<u64>() as f64,
        "",
    ));
    out.push(Measurement::info(
        "stage_errors",
        stats.stage_errors as f64,
        "",
    ));
    out
}

pub fn build_tracker_latency(params: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    let p: LatencyParams = parse_params(params)?;
    validate_timing(p.seconds, p.lead_s)?;
    for (name, v) in [
        ("max_p95_ms", p.max_p95_ms),
        ("min_points_hz", p.min_points_hz),
    ] {
        if !v.is_finite() || v < 0.0 {
            return Err(ParamError::Invalid(format!(
                "{name} must be finite and >= 0"
            )));
        }
    }
    Ok(Box::new(TrackerLatency { p }))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use eye_capture::CaptureError;
    use eye_core::CameraId;

    use super::*;

    #[test]
    fn test_latency_output_thresholds() {
        let stats = TrackerStats {
            capture_to_emit: eye::tracker::LatencySummary {
                count: 100,
                p50: Duration::from_millis(20),
                p95: Duration::from_millis(60),
                max: Duration::from_millis(80),
            },
            ..TrackerStats::default()
        };
        let out = latency_output(
            &stats,
            150,
            Duration::from_secs(10),
            &LatencyParams::default(),
        );
        let p95 = out
            .measurements
            .iter()
            .find(|m| m.name == "capture_to_emit_p95_ms")
            .unwrap();
        assert_eq!(p95.value, 60.0);
        assert!(!p95.pass);
        let hz = out
            .measurements
            .iter()
            .find(|m| m.name == "points_hz")
            .unwrap();
        assert_eq!(hz.value, 15.0);
        assert!(hz.pass);
    }

    #[test]
    fn test_latency_output_without_points_fails() {
        let stats = TrackerStats::default();
        let out = latency_output(
            &stats,
            0,
            Duration::from_secs(10),
            &LatencyParams::default(),
        );
        let p95 = out
            .measurements
            .iter()
            .find(|m| m.name == "capture_to_emit_p95_ms")
            .unwrap();
        assert!(p95.value.is_nan());
        assert!(!out.passed());
    }

    #[test]
    fn test_tracker_error_mapping() {
        let err = tracker_error(TrackerError::Capture {
            camera: "ir".into(),
            source: CaptureError::EndOfStream,
        });
        assert!(matches!(err, TestError::Capture(_)));

        let err = tracker_error(TrackerError::Stopped);
        assert!(matches!(err, TestError::Other(_)));
    }

    #[test]
    fn test_frames_dropped_sums_the_map() {
        let mut dropped = BTreeMap::new();
        dropped.insert(CameraId::from("rgb"), 3u64);
        dropped.insert(CameraId::from("ir"), 5u64);
        let stats = TrackerStats {
            frames_dropped: dropped,
            ..TrackerStats::default()
        };
        let out = latency_output(
            &stats,
            0,
            Duration::from_secs(10),
            &LatencyParams::default(),
        );
        let dropped = out
            .measurements
            .iter()
            .find(|m| m.name == "frames_dropped")
            .unwrap();
        assert_eq!(dropped.value, 8.0);
    }
}
