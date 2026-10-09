use eye_core::log::field;
use eye_core::stage::{GazeFilter, StageError};
use eye_core::{GazePoint, Rig, ScreenModel, Timestamp};
use nalgebra::{Cholesky, Matrix2, Matrix2x4, Matrix4, Point2, Vector2, Vector4};

use crate::FilterError;
use crate::point::{cov_fields, dt_seconds, with_position};

#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KalmanConfig {
    pub accel_psd: f64,
    pub measurement_floor_mm: f64,
    pub initial_speed_sigma: f64,
    pub gate_chi2: f64,
    pub reset_after_outliers: u32,
    pub reset_after_s: f64,
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            accel_psd: 2.0e4,
            measurement_floor_mm: 1.0,
            initial_speed_sigma: 300.0,
            gate_chi2: 13.82,
            reset_after_outliers: 2,
            reset_after_s: 0.5,
        }
    }
}

#[derive(Debug)]
pub struct KalmanFilter {
    cfg: KalmanConfig,
    screen: ScreenModel,
    state: Option<KfState>,
}

#[derive(Debug, Clone, Copy)]
struct KfState {
    t: Timestamp,
    s: Vector4<f64>,
    p: Matrix4<f64>,
    outliers: u32,
}

impl KalmanFilter {
    /// `FilterError::Param` unless `accel_psd > 0`, `gate_chi2 > 0`, `initial_speed_sigma > 0`,
    /// `reset_after_s > 0` (all finite), `measurement_floor_mm >= 0` (finite), `reset_after_outliers >= 1`.
    pub fn new(cfg: KalmanConfig, screen: ScreenModel) -> Result<Self, FilterError> {
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
        positive(cfg.accel_psd, "accel_psd")?;
        positive(cfg.gate_chi2, "gate_chi2")?;
        positive(cfg.initial_speed_sigma, "initial_speed_sigma")?;
        positive(cfg.reset_after_s, "reset_after_s")?;
        if !(cfg.measurement_floor_mm.is_finite() && cfg.measurement_floor_mm >= 0.0) {
            return Err(FilterError::Param {
                name: "measurement_floor_mm",
                reason: "must be >= 0",
            });
        }
        if cfg.reset_after_outliers < 1 {
            return Err(FilterError::Param {
                name: "reset_after_outliers",
                reason: "must be >= 1",
            });
        }
        Ok(Self {
            cfg,
            screen,
            state: None,
        })
    }

    pub fn from_config(table: &toml::Table, rig: &Rig) -> Result<Self, StageError> {
        let cfg: KalmanConfig = table.clone().try_into().map_err(FilterError::from)?;
        Ok(Self::new(cfg, rig.screen().clone())?)
    }

    fn init(&mut self, point: GazePoint, z: Vector2<f64>, r: Matrix2<f64>) -> GazePoint {
        let mut p = Matrix4::zeros();
        p.fixed_view_mut::<2, 2>(0, 0).copy_from(&r);
        let v = self.cfg.initial_speed_sigma.powi(2);
        p[(2, 2)] = v;
        p[(3, 3)] = v;
        self.state = Some(KfState {
            t: point.timestamp,
            s: Vector4::new(z.x, z.y, 0.0, 0.0),
            p,
            outliers: 0,
        });
        with_position(point, z.into(), r, &self.screen)
    }

    fn output(&self, point: GazePoint, s: &Vector4<f64>, p: &Matrix4<f64>) -> GazePoint {
        with_position(
            point,
            Point2::new(s[0], s[1]),
            p.fixed_view::<2, 2>(0, 0).into_owned(),
            &self.screen,
        )
    }
}

impl GazeFilter for KalmanFilter {
    fn apply(&mut self, point: GazePoint) -> GazePoint {
        let z = point.mm.coords;
        let r = point.cov_mm + Matrix2::identity() * self.cfg.measurement_floor_mm.powi(2);
        let Some(st) = self.state else {
            tracing::debug!(
                { field::REASON } = "first_sample",
                { field::TS_NS } = point.timestamp.as_nanos(),
                "filter initialised"
            );
            return self.init(point, z, r);
        };
        let Some(dt) = dt_seconds(st.t, point.timestamp) else {
            tracing::warn!(
                { field::REASON } = "non_increasing_timestamp",
                { field::TS_NS } = point.timestamp.as_nanos(),
                prev_ts_ns = st.t.as_nanos(),
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
            return self.init(point, z, r);
        }
        let (s_pred, p_pred) = predict(&st.s, &st.p, dt, self.cfg.accel_psd);
        let y = z - s_pred.fixed_rows::<2>(0);
        let s_cov = p_pred.fixed_view::<2, 2>(0, 0) + r;
        let Some(chol) = Cholesky::new(s_cov) else {
            let (s_xx, s_xy, s_yy) = cov_fields(&s_cov);
            tracing::warn!(
                { field::REASON } = "cholesky_failed",
                s_xx,
                s_xy,
                s_yy,
                "innovation covariance not positive definite, re-initialising"
            );
            return self.init(point, z, r);
        };
        let d2 = y.dot(&chol.solve(&y));
        if d2 > self.cfg.gate_chi2 {
            let outliers = st.outliers + 1;
            if outliers >= self.cfg.reset_after_outliers {
                tracing::warn!(
                    { field::REASON } = "outliers",
                    outliers = u64::from(outliers),
                    d2,
                    { field::TS_NS } = point.timestamp.as_nanos(),
                    "filter reset after consecutive outliers"
                );
                return self.init(point, z, r);
            }
            tracing::debug!(
                { field::REASON } = "outlier",
                d2,
                gate_chi2 = self.cfg.gate_chi2,
                outliers = u64::from(outliers),
                reset_after_outliers = u64::from(self.cfg.reset_after_outliers),
                out_x_mm = s_pred[0],
                out_y_mm = s_pred[1],
                "measurement gated"
            );
            self.state = Some(KfState {
                t: point.timestamp,
                s: s_pred,
                p: p_pred,
                outliers,
            });
            return self.output(point, &s_pred, &p_pred);
        }
        let ph_t = p_pred.fixed_columns::<2>(0).into_owned();
        let k = ph_t * chol.inverse();
        let s_new = s_pred + k * y;
        let i_kh = Matrix4::identity() - k * Matrix2x4::identity();
        let p_new = i_kh * p_pred * i_kh.transpose() + k * r * k.transpose();
        let (in_cov_xx, in_cov_xy, in_cov_yy) = cov_fields(&point.cov_mm);
        let (out_cov_xx, out_cov_xy, out_cov_yy) =
            cov_fields(&p_new.fixed_view::<2, 2>(0, 0).into_owned());
        tracing::trace!(
            dt_s = dt,
            d2,
            in_x_mm = z.x,
            in_y_mm = z.y,
            in_cov_xx,
            in_cov_xy,
            in_cov_yy,
            out_x_mm = s_new[0],
            out_y_mm = s_new[1],
            out_cov_xx,
            out_cov_xy,
            out_cov_yy,
            vx_mm_s = s_new[2],
            vy_mm_s = s_new[3],
            "filter step"
        );
        self.state = Some(KfState {
            t: point.timestamp,
            s: s_new,
            p: p_new,
            outliers: 0,
        });
        self.output(point, &s_new, &p_new)
    }

    fn reset(&mut self) {
        self.state = None;
        tracing::debug!({ field::REASON } = "external", "filter reset");
    }
}

fn predict(s: &Vector4<f64>, p: &Matrix4<f64>, dt: f64, q: f64) -> (Vector4<f64>, Matrix4<f64>) {
    let mut f = Matrix4::identity();
    f[(0, 2)] = dt;
    f[(1, 3)] = dt;
    let (a, b, c) = (dt.powi(3) / 3.0, dt.powi(2) / 2.0, dt);
    #[rustfmt::skip]
    let qm = Matrix4::new(
        a, 0.0, b, 0.0,
        0.0, a, 0.0, b,
        b, 0.0, c, 0.0,
        0.0, b, 0.0, c,
    ) * q;
    (f * s, f * p * f.transpose() + qm)
}

#[cfg(test)]
mod tests {
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

    fn pt_var(t_ns: u64, x_mm: f64, y_mm: f64, var: f64) -> GazePoint {
        let screen = edp1();
        let mm = Point2::new(x_mm, y_mm);
        let cov_mm = Matrix2::identity() * var;
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

    fn pt(t_ns: u64, x_mm: f64, y_mm: f64) -> GazePoint {
        pt_var(t_ns, x_mm, y_mm, 100.0)
    }

    fn filter(cfg: KalmanConfig) -> KalmanFilter {
        KalmanFilter::new(cfg, edp1()).unwrap()
    }

    struct Sim {
        points: Vec<GazePoint>,
        truth: Vec<Vector2<f64>>,
    }

    fn simulate(steps: usize) -> Sim {
        let cfg = KalmanConfig::default();
        let dt = P30 as f64 * 1e-9;
        let q = predict(&Vector4::zeros(), &Matrix4::zeros(), dt, cfg.accel_psd).1;
        let l = Cholesky::new(q).unwrap().l();
        let mut rng = SplitMix64::new(42);
        let mut s = Vector4::new(150.0, 80.0, 0.0, 0.0);
        let mut f = Matrix4::identity();
        f[(0, 2)] = dt;
        f[(1, 3)] = dt;

        let mut points = Vec::with_capacity(steps);
        let mut truth = Vec::with_capacity(steps);
        for k in 0..steps {
            let noise = Vector4::new(
                rng.gaussian(),
                rng.gaussian(),
                rng.gaussian(),
                rng.gaussian(),
            );
            s = f * s + l * noise;
            truth.push(Vector2::new(s[0], s[1]));
            let zx = s[0] + 15.0 * rng.gaussian();
            let zy = s[1] + 15.0 * rng.gaussian();
            points.push(pt_var(k as u64 * P30, zx, zy, 225.0));
        }
        Sim { points, truth }
    }

    #[test]
    fn test_first_sample_passes_through_with_floor_added_to_cov() {
        let mut f = filter(KalmanConfig::default());
        let out = f.apply(pt_var(0, 100.0, 100.0, 25.0));
        approx::assert_abs_diff_eq!(out.mm.x, 100.0, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(out.mm.y, 100.0, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(out.cov_mm, Matrix2::identity() * 26.0, epsilon = 1e-12);
    }

    #[test]
    fn test_stationary_target_covariance_shrinks() {
        let mut f = filter(KalmanConfig::default());
        let mut first = None;
        let mut last = 0.0;
        for k in 0..60u64 {
            let out = f.apply(pt_var(k * P30, 100.0, 100.0, 100.0));
            if first.is_none() {
                first = Some(out.cov_mm[(0, 0)]);
            }
            last = out.cov_mm[(0, 0)];
        }
        approx::assert_abs_diff_eq!(first.unwrap(), 101.0, epsilon = 1e-9);
        assert!(last < 40.0, "last = {last}");
        assert!(last < 101.0);
    }

    #[test]
    fn test_rms_error_reduced_on_model_data() {
        let mut f = filter(KalmanConfig::default());
        let sim = simulate(2000);
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, p) in sim.points.iter().enumerate() {
            let out = f.apply(p.clone());
            if i > 50 {
                let err = Vector2::new(out.mm.x, out.mm.y) - sim.truth[i];
                let raw = Vector2::new(p.mm.x, p.mm.y) - sim.truth[i];
                num += err.norm_squared();
                den += raw.norm_squared();
            }
        }
        let ratio = (num / den).sqrt();
        assert!(ratio < 0.6, "ratio = {ratio}");
    }

    #[test]
    fn test_nees_is_consistent() {
        let mut f = filter(KalmanConfig::default());
        let sim = simulate(2000);
        let mut total = 0.0;
        let mut count = 0;
        for (i, p) in sim.points.iter().enumerate() {
            let out = f.apply(p.clone());
            if i > 50 {
                let err = Vector2::new(out.mm.x, out.mm.y) - sim.truth[i];
                let cov = out.cov_mm;
                let chol = Cholesky::new(cov).unwrap();
                total += err.dot(&chol.solve(&err));
                count += 1;
            }
        }
        let mean = total / count as f64;
        assert!((mean - 2.0).abs() < 0.15 * 2.0, "mean nees = {mean}");
    }

    #[test]
    fn test_single_spike_is_gated() {
        let mut f = filter(KalmanConfig::default());
        for k in 0..30u64 {
            f.apply(pt_var(k * P30, 100.0, 100.0, 25.0));
        }
        let out = f.apply(pt_var(30 * P30, 300.0, 100.0, 25.0));
        let d = ((out.mm.x - 100.0).powi(2) + (out.mm.y - 100.0).powi(2)).sqrt();
        assert!(d < 5.0, "d = {d}");

        let out2 = f.apply(pt_var(31 * P30, 100.0, 100.0, 25.0));
        let d2 = ((out2.mm.x - 100.0).powi(2) + (out2.mm.y - 100.0).powi(2)).sqrt();
        assert!(d2 < 1.0, "d2 = {d2}");

        let out3 = f.apply(pt_var(32 * P30, 300.0, 100.0, 25.0));
        let d3 = ((out3.mm.x - 100.0).powi(2) + (out3.mm.y - 100.0).powi(2)).sqrt();
        assert!(d3 < 5.0, "d3 = {d3}");
    }

    #[test]
    fn test_saccade_resets_after_consecutive_outliers() {
        let mut f = filter(KalmanConfig::default());
        for k in 0..30u64 {
            f.apply(pt_var(k * P30, 100.0, 100.0, 25.0));
        }
        let out1 = f.apply(pt_var(30 * P30, 250.0, 100.0, 25.0));
        let d1 = ((out1.mm.x - 100.0).powi(2) + (out1.mm.y - 100.0).powi(2)).sqrt();
        assert!(d1 < 5.0, "d1 = {d1}");

        let out2 = f.apply(pt_var(31 * P30, 250.0, 100.0, 25.0));
        approx::assert_abs_diff_eq!(out2.mm.x, 250.0, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(out2.mm.y, 100.0, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(out2.cov_mm, Matrix2::identity() * 26.0, epsilon = 1e-9);
    }

    fn find<'a>(records: &'a [eye_log::Record], message: &str) -> Vec<&'a eye_log::Record> {
        records
            .iter()
            .filter(|r| r.target == "eye_filter::kalman" && r.message == message)
            .collect()
    }

    fn f64_field(record: &eye_log::Record, name: &str) -> f64 {
        match record.fields.get(name) {
            Some(eye_log::Value::F64(v)) => *v,
            other => panic!("field {name}: {other:?}"),
        }
    }

    fn u64_field(record: &eye_log::Record, name: &str) -> u64 {
        match record.fields.get(name) {
            Some(eye_log::Value::U64(v)) => *v,
            other => panic!("field {name}: {other:?}"),
        }
    }

    #[test]
    fn test_logs_filter_initialised_at_debug() {
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut f = filter(KalmanConfig::default());
            f.apply(pt_var(0, 100.0, 100.0, 25.0));
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
            let mut f = filter(KalmanConfig::default());
            f.apply(pt_var(0, 100.0, 100.0, 25.0));
            f.apply(pt_var(P30, 100.0, 100.0, 25.0));
        });
        let recs = find(&records, "filter step");
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].level, eye_log::Level::Trace);
        assert!(f64_field(recs[0], "d2").is_finite());
        assert!(f64_field(recs[0], "out_cov_xx") < f64_field(recs[0], "in_cov_xx"));
        assert!(recs[0].fields.contains_key("vx_mm_s"));
        assert!(recs[0].fields.contains_key("vy_mm_s"));
    }

    #[test]
    fn test_logs_filter_reset_at_debug_on_gap() {
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut f = filter(KalmanConfig::default());
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
        let inits = find(&records, "filter initialised");
        assert_eq!(
            inits.len(),
            1,
            "the gap branch calls init() directly, which is event-free"
        );
    }

    #[test]
    fn test_logs_external_reset_at_debug() {
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut f = filter(KalmanConfig::default());
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
            let mut f = filter(KalmanConfig::default());
            f.apply(pt(P30, 0.0, 0.0));
            f.apply(pt(P30, 0.0, 0.0));
        });
        let recs: Vec<_> = records
            .iter()
            .filter(|r| r.target == "eye_filter::kalman" && r.level == eye_log::Level::Warn)
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
    fn test_logs_measurement_gated_at_debug() {
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut f = filter(KalmanConfig::default());
            for k in 0..30u64 {
                f.apply(pt_var(k * P30, 100.0, 100.0, 25.0));
            }
            f.apply(pt_var(30 * P30, 300.0, 100.0, 25.0));
            f.apply(pt_var(31 * P30, 100.0, 100.0, 25.0));
            f.apply(pt_var(32 * P30, 300.0, 100.0, 25.0));
        });
        let gated = find(&records, "measurement gated");
        assert_eq!(gated.len(), 2);
        for rec in &gated {
            assert_eq!(u64_field(rec, "outliers"), 1);
            assert!(f64_field(rec, "d2") > 13.82);
            approx::assert_abs_diff_eq!(f64_field(rec, "gate_chi2"), 13.82, epsilon = 1e-9);
        }
        let warns: Vec<_> = records
            .iter()
            .filter(|r| r.level == eye_log::Level::Warn)
            .collect();
        assert!(warns.is_empty(), "{warns:?}");
    }

    #[test]
    fn test_logs_outlier_reset_at_warn() {
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut f = filter(KalmanConfig::default());
            for k in 0..30u64 {
                f.apply(pt_var(k * P30, 100.0, 100.0, 25.0));
            }
            f.apply(pt_var(30 * P30, 250.0, 100.0, 25.0));
            f.apply(pt_var(31 * P30, 250.0, 100.0, 25.0));
        });
        let warns: Vec<_> = records
            .iter()
            .filter(|r| {
                r.target == "eye_filter::kalman"
                    && r.message == "filter reset after consecutive outliers"
            })
            .collect();
        assert_eq!(warns.len(), 1);
        assert_eq!(warns[0].level, eye_log::Level::Warn);
        assert_eq!(u64_field(warns[0], "outliers"), 2);
        let gated = find(&records, "measurement gated");
        assert_eq!(gated.len(), 1);
    }

    /// `s_cov = p_pred[0:2,0:2] + r`, and `predict` always adds
    /// `Q_pos = dt^3/3 * accel_psd * I` (about 0.25 for `dt = 1/30`,
    /// `accel_psd = 2e4`) to `p_pred`, so `s_cov` stays positive definite
    /// even with `measurement_floor_mm = 0.0` and a zero-variance point.
    /// This asserts that derivation: no `cholesky_failed` record is ever
    /// produced on a converged, finite run.
    #[test]
    fn test_logs_cholesky_failed_at_warn() {
        let cfg = KalmanConfig {
            measurement_floor_mm: 0.0,
            ..KalmanConfig::default()
        };
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut f = filter(cfg);
            for k in 0..60u64 {
                f.apply(pt_var(k * P30, 100.0, 100.0, 0.0));
            }
        });
        let warns: Vec<_> = records
            .iter()
            .filter(|r| r.target == "eye_filter::kalman" && r.level == eye_log::Level::Warn)
            .collect();
        assert!(
            warns.is_empty(),
            "cholesky_failed should be unreachable with finite inputs: {warns:?}"
        );
    }

    #[test]
    fn test_zero_measurement_cov_does_not_panic() {
        let mut f = filter(KalmanConfig::default());
        let mut rng = SplitMix64::new(3);
        for k in 0..100u64 {
            let x = 100.0 + 2.0 * rng.gaussian();
            let y = 100.0 + 2.0 * rng.gaussian();
            let out = f.apply(pt_var(k * P30, x, y, 0.0));
            assert!(out.validate().is_ok());
        }
    }

    #[test]
    fn test_px_fields_follow_filtered_mm() {
        let mut f = filter(KalmanConfig::default());
        let sim = simulate(200);
        let screen = edp1();
        for p in sim.points {
            let out = f.apply(p);
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
    fn test_non_increasing_timestamp_passes_through() {
        let mut f = filter(KalmanConfig::default());
        f.apply(pt(P30, 50.0, 50.0));
        let dup = pt(P30, 999.0, 999.0);
        let out = f.apply(dup.clone());
        assert_eq!(out, dup);

        let next = f.apply(pt(2 * P30, 60.0, 60.0));

        let mut fresh = filter(KalmanConfig::default());
        fresh.apply(pt(P30, 50.0, 50.0));
        let expected = fresh.apply(pt(2 * P30, 60.0, 60.0));

        approx::assert_abs_diff_eq!(next.mm, expected.mm, epsilon = 1e-12);
    }

    #[test]
    fn test_from_config_parses_defaults_and_rejects_unknown_key() {
        let cfg: KalmanConfig = toml::Table::new().try_into().unwrap();
        assert_eq!(cfg, KalmanConfig::default());

        assert!(KalmanFilter::from_config(&toml::Table::new(), &rig()).is_ok());

        let table: toml::Table = toml::from_str("q = 1").unwrap();
        assert!(matches!(
            KalmanFilter::from_config(&table, &rig()),
            Err(StageError::Config(_))
        ));
    }

    #[test]
    fn test_new_rejects_invalid_params() {
        let cfg = KalmanConfig {
            accel_psd: 0.0,
            ..KalmanConfig::default()
        };
        assert!(matches!(
            KalmanFilter::new(cfg, edp1()),
            Err(FilterError::Param {
                name: "accel_psd",
                ..
            })
        ));

        let cfg = KalmanConfig {
            gate_chi2: -1.0,
            ..KalmanConfig::default()
        };
        assert!(matches!(
            KalmanFilter::new(cfg, edp1()),
            Err(FilterError::Param {
                name: "gate_chi2",
                ..
            })
        ));

        let cfg = KalmanConfig {
            reset_after_outliers: 0,
            ..KalmanConfig::default()
        };
        assert!(matches!(
            KalmanFilter::new(cfg, edp1()),
            Err(FilterError::Param {
                name: "reset_after_outliers",
                ..
            })
        ));
    }

    #[test]
    fn test_boxed_filter_from_config_outputs_valid_points() {
        let mut boxed: Box<dyn GazeFilter> =
            Box::new(KalmanFilter::from_config(&toml::Table::new(), &rig()).unwrap());
        let sim = simulate(100);
        for p in sim.points {
            let out = boxed.apply(p);
            assert!(out.validate().is_ok());
        }
    }
}
