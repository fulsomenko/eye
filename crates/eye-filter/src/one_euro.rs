use eye_core::log::field;
use eye_core::stage::{GazeFilter, StageError};
use eye_core::{GazePoint, Rig, ScreenModel, Timestamp};
use nalgebra::{Matrix2, Vector2};

use crate::FilterError;
use crate::point::{cov_fields, dt_seconds, with_position};

#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OneEuroConfig {
    pub min_cutoff: f64,
    pub beta: f64,
    pub d_cutoff: f64,
    pub reset_after_s: f64,
}

impl Default for OneEuroConfig {
    fn default() -> Self {
        Self {
            min_cutoff: 0.3,
            beta: 0.002,
            d_cutoff: 1.0,
            reset_after_s: 0.5,
        }
    }
}

#[derive(Debug)]
pub struct OneEuroFilter {
    cfg: OneEuroConfig,
    screen: ScreenModel,
    state: Option<State>,
}

#[derive(Debug, Clone, Copy)]
struct State {
    t: Timestamp,
    x: Vector2<f64>,
    dx: Vector2<f64>,
    cov_lp: Matrix2<f64>,
}

fn alpha(cutoff: f64, dt: f64) -> f64 {
    let tau = 1.0 / (std::f64::consts::TAU * cutoff);
    1.0 / (1.0 + tau / dt)
}

impl OneEuroFilter {
    pub fn new(cfg: OneEuroConfig, screen: ScreenModel) -> Result<Self, FilterError> {
        let positive = |v: f64, name: &'static str| {
            if v.is_finite() && v > 0.0 {
                Ok(())
            } else {
                Err(FilterError::Param {
                    name,
                    reason: "must be > 0",
                })
            }
        };
        positive(cfg.min_cutoff, "min_cutoff")?;
        positive(cfg.d_cutoff, "d_cutoff")?;
        positive(cfg.reset_after_s, "reset_after_s")?;
        if !(cfg.beta.is_finite() && cfg.beta >= 0.0) {
            return Err(FilterError::Param {
                name: "beta",
                reason: "must be >= 0",
            });
        }
        Ok(Self {
            cfg,
            screen,
            state: None,
        })
    }

    pub fn from_config(table: &toml::Table, rig: &Rig) -> Result<Self, StageError> {
        let cfg: OneEuroConfig = table.clone().try_into().map_err(FilterError::from)?;
        Ok(Self::new(cfg, rig.screen().clone())?)
    }
}

impl GazeFilter for OneEuroFilter {
    fn apply(&mut self, point: GazePoint) -> GazePoint {
        let x = point.mm.coords;
        let Some(prev) = self.state else {
            self.state = Some(State {
                t: point.timestamp,
                x,
                dx: Vector2::zeros(),
                cov_lp: point.cov_mm,
            });
            tracing::debug!(
                { field::REASON } = "first_sample",
                { field::TS_NS } = point.timestamp.as_nanos(),
                "filter initialised"
            );
            return point;
        };
        let Some(dt) = dt_seconds(prev.t, point.timestamp) else {
            tracing::warn!(
                { field::REASON } = "non_increasing_timestamp",
                { field::TS_NS } = point.timestamp.as_nanos(),
                prev_ts_ns = prev.t.as_nanos(),
                "non-increasing gaze timestamp, passing through"
            );
            return point;
        };
        if dt > self.cfg.reset_after_s {
            tracing::debug!(
                { field::REASON } = "gap",
                dt_s = dt,
                reset_after_s = self.cfg.reset_after_s,
                "filter reset"
            );
            self.state = None;
            return self.apply(point);
        }
        let a_d = alpha(self.cfg.d_cutoff, dt);
        let dx = a_d * (x - prev.x) / dt + (1.0 - a_d) * prev.dx;
        let speed_mm_s = dx.norm();
        let cutoff_hz = self.cfg.min_cutoff + self.cfg.beta * speed_mm_s;
        let a = alpha(cutoff_hz, dt);
        let xf = a * x + (1.0 - a) * prev.x;
        // The covariance uses the rest cutoff, not the speed-adapted one: the reported
        // uncertainty would otherwise swell with every saccade or jitter and shrink when still.
        let a_rest = alpha(self.cfg.min_cutoff, dt);
        let cov_lp = prev.cov_lp + (point.cov_mm - prev.cov_lp) * a_rest;
        let cov = cov_lp * (a_rest / (2.0 - a_rest));
        self.state = Some(State {
            t: point.timestamp,
            x: xf,
            dx,
            cov_lp,
        });
        let (in_cov_xx, in_cov_xy, in_cov_yy) = cov_fields(&point.cov_mm);
        let (out_cov_xx, out_cov_xy, out_cov_yy) = cov_fields(&cov);
        tracing::trace!(
            dt_s = dt,
            speed_mm_s,
            cutoff_hz,
            alpha = a,
            alpha_d = a_d,
            in_x_mm = x.x,
            in_y_mm = x.y,
            out_x_mm = xf.x,
            out_y_mm = xf.y,
            in_cov_xx,
            in_cov_xy,
            in_cov_yy,
            out_cov_xx,
            out_cov_xy,
            out_cov_yy,
            "filter step"
        );
        with_position(point, xf.into(), cov, &self.screen)
    }

    fn reset(&mut self) {
        self.state = None;
        tracing::debug!({ field::REASON } = "external", "filter reset");
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use eye_core::{CameraId, OutputId};
    use eye_geometry::screen::{mm_to_px_logical, mm_to_px_physical};
    use eye_geometry::synth::SplitMix64;
    use nalgebra::{Isometry3, Matrix2, Point2};

    use super::*;

    const P30: u64 = 33_333_333;

    fn edp1() -> ScreenModel {
        ScreenModel {
            output: OutputId::new("eDP-1"),
            size_mm: Vector2::new(310.0, 170.0),
            size_px: (3840, 2160),
            scale: 2.0,
        }
    }

    fn rig() -> Rig {
        Rig::new(
            vec![eye_core::CameraModel {
                id: CameraId::new("ir"),
                width: 640,
                height: 360,
                fx: 457.0,
                fy: 457.0,
                cx: 320.0,
                cy: 180.0,
                distortion: [0.0; 5],
                screen_from_camera: Isometry3::identity(),
            }],
            edp1(),
        )
        .unwrap()
    }

    fn pt(t_ns: u64, x_mm: f64, y_mm: f64) -> GazePoint {
        let screen = edp1();
        let mm = Point2::new(x_mm, y_mm);
        let cov_mm = Matrix2::identity() * 100.0;
        let px_physical = mm_to_px_physical(&screen, &mm);
        GazePoint {
            timestamp: Timestamp::from_nanos(t_ns),
            output: OutputId::new("eDP-1"),
            mm,
            px_physical,
            px_logical: mm_to_px_logical(&screen, &mm),
            cov_mm,
            confidence: eye_geometry::screen::confidence_from_cov(&cov_mm),
        }
    }

    fn alpha03() -> f64 {
        1.0 / (1.0 + (1.0 / (TAU * 0.3)) / (P30 as f64 * 1e-9))
    }

    fn pt_cov(t_ns: u64, x_mm: f64, y_mm: f64, cov_val: f64) -> GazePoint {
        let screen = edp1();
        let mm = Point2::new(x_mm, y_mm);
        let cov_mm = Matrix2::identity() * cov_val;
        let px_physical = mm_to_px_physical(&screen, &mm);
        GazePoint {
            timestamp: Timestamp::from_nanos(t_ns),
            output: OutputId::new("eDP-1"),
            mm,
            px_physical,
            px_logical: mm_to_px_logical(&screen, &mm),
            cov_mm,
            confidence: eye_geometry::screen::confidence_from_cov(&cov_mm),
        }
    }

    fn filter(cfg: OneEuroConfig) -> OneEuroFilter {
        OneEuroFilter::new(cfg, edp1()).unwrap()
    }

    #[test]
    fn test_one_euro_default_is_heavy() {
        let cfg = OneEuroConfig::default();
        approx::assert_abs_diff_eq!(cfg.min_cutoff, 0.3, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(cfg.beta, 0.002, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(cfg.d_cutoff, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_first_sample_passes_through() {
        let mut f = filter(OneEuroConfig::default());
        let p = pt(0, 10.0, 20.0);
        assert_eq!(f.apply(p.clone()), p);
    }

    #[test]
    fn test_constant_input_stays_constant() {
        let mut f = filter(OneEuroConfig::default());
        for k in 0..100u64 {
            let out = f.apply(pt(k * P30, 50.0, 60.0));
            approx::assert_abs_diff_eq!(out.mm.x, 50.0, epsilon = 1e-12);
            approx::assert_abs_diff_eq!(out.mm.y, 60.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_step_response_matches_alpha() {
        let cfg = OneEuroConfig {
            min_cutoff: 1.0,
            beta: 0.0,
            ..OneEuroConfig::default()
        };
        let mut f = filter(cfg);
        f.apply(pt(0, 0.0, 0.0));
        let out = f.apply(pt(P30, 100.0, 0.0));
        let expected = 100.0 / (1.0 + (1.0 / (TAU * 1.0)) / (P30 as f64 * 1e-9));
        approx::assert_abs_diff_eq!(out.mm.x, expected, epsilon = 1e-9);
    }

    #[test]
    fn test_beta_reduces_lag_during_fast_motion() {
        let ramp = |beta: f64| {
            let cfg = OneEuroConfig {
                beta,
                ..OneEuroConfig::default()
            };
            let mut f = filter(cfg);
            let mut last = None;
            for k in 0..=10u64 {
                let truth = 2600.0 * k as f64 / 30.0;
                last = Some((truth, f.apply(pt(k * P30, truth, 0.0)).mm.x));
            }
            let (truth, out) = last.unwrap();
            (truth - out).abs()
        };
        let lag_with_beta = ramp(0.007);
        let lag_without_beta = ramp(0.0);
        assert!(lag_with_beta < lag_without_beta / 2.0);
    }

    #[test]
    fn test_stationary_noise_is_reduced_by_ema_factor() {
        let cfg = OneEuroConfig {
            beta: 0.0,
            ..OneEuroConfig::default()
        };
        let mut f = filter(cfg);
        let mut rng = SplitMix64::new(1);
        let mut outputs = Vec::with_capacity(5000);
        let mut last_cov = 0.0;
        for k in 0..5000u64 {
            let x = 10.0 * rng.gaussian();
            let out = f.apply(pt(k * P30, x, 0.0));
            last_cov = out.cov_mm[(0, 0)];
            outputs.push(out.mm.x);
        }
        let tail = &outputs[101..];
        let mean = tail.iter().sum::<f64>() / tail.len() as f64;
        let var = tail.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / tail.len() as f64;
        let std = var.sqrt();
        let expected_std = 10.0 * (alpha03() / (2.0 - alpha03())).sqrt();
        assert!(
            (std - expected_std).abs() < 0.1 * expected_std,
            "std = {std}, expected ~{expected_std}"
        );
        let expected_cov = 100.0 * alpha03() / (2.0 - alpha03());
        approx::assert_abs_diff_eq!(last_cov, expected_cov, epsilon = 1e-9);
    }

    #[test]
    fn test_px_fields_follow_filtered_mm() {
        let mut f = filter(OneEuroConfig::default());
        let mut rng = SplitMix64::new(4);
        let screen = edp1();
        for k in 0..200u64 {
            let x = 155.0 + 5.0 * rng.gaussian();
            let y = 85.0 + 5.0 * rng.gaussian();
            let out = f.apply(pt(k * P30, x, y));
            approx::assert_abs_diff_eq!(
                out.px_physical,
                mm_to_px_physical(&screen, &out.mm),
                epsilon = 1e-9
            );
            approx::assert_abs_diff_eq!(
                out.px_logical,
                Point2::new(out.px_physical.x / 2.0, out.px_physical.y / 2.0),
                epsilon = 1e-9
            );
        }
    }

    #[test]
    fn test_gap_longer_than_reset_restarts() {
        let mut f = filter(OneEuroConfig::default());
        f.apply(pt(0, 0.0, 0.0));
        let input = pt(1_000_000_000, 50.0, 0.0);
        let out = f.apply(input.clone());
        assert_eq!(out, input);
    }

    #[test]
    fn test_out_of_order_timestamp_passes_through_and_keeps_state() {
        let mut f = filter(OneEuroConfig::default());
        f.apply(pt(3 * P30, 0.0, 0.0));
        let out_of_order = pt(2 * P30, 50.0, 0.0);
        let out = f.apply(out_of_order.clone());
        assert_eq!(out, out_of_order);
        let last = f.apply(pt(4 * P30, 10.0, 0.0));

        let mut fresh = filter(OneEuroConfig::default());
        fresh.apply(pt(3 * P30, 0.0, 0.0));
        let expected = fresh.apply(pt(4 * P30, 10.0, 0.0));
        approx::assert_abs_diff_eq!(last.mm, expected.mm, epsilon = 1e-12);
    }

    #[test]
    fn test_reset_makes_next_sample_pass_through() {
        let mut f = filter(OneEuroConfig::default());
        f.apply(pt(0, 0.0, 0.0));
        f.apply(pt(P30, 10.0, 0.0));
        f.reset();
        let input = pt(2 * P30, 20.0, 0.0);
        let out = f.apply(input.clone());
        assert_eq!(out, input);
    }

    #[test]
    fn test_from_config_rejects_unknown_key() {
        let table: toml::Table = toml::from_str("min_cutoff = 1.0\nbogus = 2").unwrap();
        assert!(matches!(
            OneEuroFilter::from_config(&table, &rig()),
            Err(StageError::Config(_))
        ));

        let table: toml::Table = toml::from_str("kind = \"one-euro\"").unwrap();
        assert!(matches!(
            OneEuroFilter::from_config(&table, &rig()),
            Err(StageError::Config(_))
        ));
    }

    #[test]
    fn test_new_rejects_non_positive_cutoff() {
        let cfg = OneEuroConfig {
            min_cutoff: 0.0,
            ..OneEuroConfig::default()
        };
        assert!(matches!(
            OneEuroFilter::new(cfg, edp1()),
            Err(FilterError::Param {
                name: "min_cutoff",
                ..
            })
        ));

        let table: toml::Table = toml::from_str("min_cutoff = 0.0").unwrap();
        assert!(matches!(
            OneEuroFilter::from_config(&table, &rig()),
            Err(StageError::Config(_))
        ));
    }

    #[test]
    fn test_filter_is_object_safe_and_send() {
        let _boxed: Box<dyn GazeFilter> =
            Box::new(OneEuroFilter::from_config(&toml::Table::new(), &rig()).unwrap());

        fn assert_send<T: Send>() {}
        assert_send::<OneEuroFilter>();
    }

    #[test]
    fn test_boxed_filter_from_config_outputs_valid_points() {
        let mut boxed: Box<dyn GazeFilter> =
            Box::new(OneEuroFilter::from_config(&toml::Table::new(), &rig()).unwrap());
        let mut rng = SplitMix64::new(2);
        let screen = edp1();
        for k in 0..60u64 {
            let x = 155.0 + 5.0 * rng.gaussian();
            let y = 85.0 + 5.0 * rng.gaussian();
            let out = boxed.apply(pt(k * P30, x, y));
            assert!(out.validate().is_ok());
            approx::assert_abs_diff_eq!(
                out.px_logical,
                mm_to_px_logical(&screen, &out.mm),
                epsilon = 1e-9
            );
        }
    }

    #[test]
    fn test_one_euro_covariance_follows_lowpass() {
        let cfg = OneEuroConfig {
            beta: 0.0,
            ..OneEuroConfig::default()
        };
        let mut f = filter(cfg);
        f.apply(pt_cov(0, 0.0, 0.0, 100.0));
        let out = f.apply(pt_cov(P30, 0.0, 0.0, 10000.0));
        let a = alpha03();
        let cov_lp = 100.0 + (10000.0 - 100.0) * a;
        let expected = cov_lp * (a / (2.0 - a));
        let naive = 10000.0 * (a / (2.0 - a));
        assert!(
            (expected - naive).abs() > 1.0,
            "expected and naive should differ meaningfully: expected={expected}, naive={naive}"
        );
        approx::assert_abs_diff_eq!(out.cov_mm[(0, 0)], expected, epsilon = 1e-9);
    }

    #[test]
    fn test_covariance_does_not_change_with_speed() {
        let mut still = filter(OneEuroConfig::default());
        let mut moving = filter(OneEuroConfig::default());
        let mut out_still = None;
        let mut out_moving = None;
        for k in 0..10u64 {
            out_still = Some(still.apply(pt_cov(k * P30, 0.0, 0.0, 400.0)));
            out_moving = Some(moving.apply(pt_cov(k * P30, 40.0 * k as f64, 0.0, 400.0)));
        }
        let (still, moving) = (out_still.unwrap(), out_moving.unwrap());
        approx::assert_abs_diff_eq!(moving.cov_mm[(0, 0)], still.cov_mm[(0, 0)], epsilon = 1e-9);
        assert!(moving.px_logical.x != still.px_logical.x, "positions must differ");
    }

    #[test]
    fn test_one_euro_covariance_constant_input_unchanged() {
        let cfg = OneEuroConfig {
            beta: 0.0,
            ..OneEuroConfig::default()
        };
        let mut f = filter(cfg);
        f.apply(pt_cov(0, 0.0, 0.0, 100.0));
        let out = f.apply(pt_cov(P30, 0.0, 0.0, 100.0));
        let a = alpha03();
        let expected = 100.0 * (a / (2.0 - a));
        approx::assert_abs_diff_eq!(out.cov_mm[(0, 0)], expected, epsilon = 1e-9);
    }

    fn find<'a>(records: &'a [eye_log::Record], message: &str) -> Vec<&'a eye_log::Record> {
        records
            .iter()
            .filter(|r| r.target == "eye_filter::one_euro" && r.message == message)
            .collect()
    }

    fn f64_field(record: &eye_log::Record, name: &str) -> f64 {
        match record.fields.get(name) {
            Some(eye_log::Value::F64(v)) => *v,
            other => panic!("field {name}: {other:?}"),
        }
    }

    #[test]
    fn test_logs_filter_initialised_at_debug() {
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut f = filter(OneEuroConfig::default());
            f.apply(pt(0, 10.0, 20.0));
        });
        let recs = find(&records, "filter initialised");
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].level, eye_log::Level::Debug);
        assert_eq!(
            recs[0].fields.get("reason"),
            Some(&eye_log::Value::Str("first_sample".to_string()))
        );
        assert_eq!(recs[0].fields.get("ts_ns"), Some(&eye_log::Value::U64(0)));
    }

    #[test]
    fn test_logs_filter_step_at_trace() {
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut f = filter(OneEuroConfig::default());
            f.apply(pt(0, 100.0, 0.0));
            f.apply(pt(P30, 100.0, 0.0));
        });
        let recs = find(&records, "filter step");
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].level, eye_log::Level::Trace);
        approx::assert_abs_diff_eq!(f64_field(recs[0], "cutoff_hz"), 0.3, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(f64_field(recs[0], "in_x_mm"), 100.0, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(f64_field(recs[0], "out_x_mm"), 100.0, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(f64_field(recs[0], "in_cov_xx"), 100.0, epsilon = 1e-9);
        assert!(f64_field(recs[0], "out_cov_xx") < 100.0);
        for v in recs[0].fields.values() {
            assert!(matches!(v, eye_log::Value::F64(_)));
        }
    }

    #[test]
    fn test_logs_filter_reset_at_debug_on_gap() {
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut f = filter(OneEuroConfig::default());
            f.apply(pt(0, 0.0, 0.0));
            f.apply(pt(1_000_000_000, 50.0, 0.0));
        });
        let resets = find(&records, "filter reset");
        assert_eq!(resets.len(), 1);
        assert_eq!(resets[0].level, eye_log::Level::Debug);
        assert_eq!(
            resets[0].fields.get("reason"),
            Some(&eye_log::Value::Str("gap".to_string()))
        );
        assert!((f64_field(resets[0], "dt_s") - 1.0).abs() < 1e-9);
        approx::assert_abs_diff_eq!(f64_field(resets[0], "reset_after_s"), 0.5, epsilon = 1e-9);
        let inits = find(&records, "filter initialised");
        assert_eq!(
            inits.len(),
            2,
            "first sample plus the re-apply after the gap"
        );
    }

    #[test]
    fn test_logs_external_reset_at_debug() {
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut f = filter(OneEuroConfig::default());
            f.reset();
        });
        let recs = find(&records, "filter reset");
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].level, eye_log::Level::Debug);
        assert_eq!(
            recs[0].fields.get("reason"),
            Some(&eye_log::Value::Str("external".to_string()))
        );
    }

    #[test]
    fn test_logs_non_increasing_timestamp_at_warn() {
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut f = filter(OneEuroConfig::default());
            f.apply(pt(P30, 0.0, 0.0));
            f.apply(pt(P30, 0.0, 0.0));
        });
        let recs: Vec<_> = records
            .iter()
            .filter(|r| r.target == "eye_filter::one_euro" && r.level == eye_log::Level::Warn)
            .collect();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].fields.get("ts_ns"), Some(&eye_log::Value::U64(P30)));
        assert_eq!(
            recs[0].fields.get("prev_ts_ns"),
            Some(&eye_log::Value::U64(P30))
        );
        for v in recs[0].fields.values() {
            if let eye_log::Value::Str(s) = v {
                assert!(!s.contains("Timestamp("));
            }
        }
    }

    #[test]
    fn test_logs_reason_field_uses_vocabulary() {
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut f = filter(OneEuroConfig::default());
            f.apply(pt(0, 0.0, 0.0));
            f.apply(pt(1_000_000_000, 50.0, 0.0));
            f.reset();
        });
        let own: Vec<_> = records
            .iter()
            .filter(|r| r.target == "eye_filter::one_euro")
            .collect();
        assert!(!own.is_empty());
        for r in &own {
            if matches!(r.level, eye_log::Level::Debug | eye_log::Level::Warn) {
                assert!(
                    matches!(
                        r.fields.get(eye_core::log::field::REASON),
                        Some(eye_log::Value::Str(_))
                    ),
                    "{r:?}"
                );
            }
            if r.message == "filter initialised" || r.level == eye_log::Level::Warn {
                assert!(
                    matches!(
                        r.fields.get(eye_core::log::field::TS_NS),
                        Some(eye_log::Value::U64(_))
                    ),
                    "{r:?}"
                );
            }
            assert!(!r.fields.contains_key("timestamp"));
        }
    }

    #[test]
    fn test_one_euro_covariance_resets_with_position() {
        let cfg = OneEuroConfig {
            beta: 0.0,
            ..OneEuroConfig::default()
        };
        let mut f = filter(cfg);
        f.apply(pt_cov(0, 0.0, 0.0, 100.0));
        f.apply(pt_cov(P30, 0.0, 0.0, 10000.0));
        f.reset();
        f.apply(pt_cov(2 * P30, 0.0, 0.0, 500.0));
        let out = f.apply(pt_cov(3 * P30, 0.0, 0.0, 500.0));
        let a = alpha03();
        let expected = 500.0 * (a / (2.0 - a));
        approx::assert_abs_diff_eq!(out.cov_mm[(0, 0)], expected, epsilon = 1e-9);
    }
}
