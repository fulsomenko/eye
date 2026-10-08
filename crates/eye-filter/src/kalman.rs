use eye_core::stage::{GazeFilter, StageError};
use eye_core::{GazePoint, Rig, ScreenModel, Timestamp};
use nalgebra::{Cholesky, Matrix2, Matrix2x4, Matrix4, Point2, Vector2, Vector4};

use crate::FilterError;
use crate::point::{dt_seconds, with_position};

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
            return self.init(point, z, r);
        };
        let Some(dt) = dt_seconds(st.t, point.timestamp) else {
            tracing::warn!(?point.timestamp, "non-increasing gaze timestamp, passing through");
            return point;
        };
        if dt > self.cfg.reset_after_s {
            return self.init(point, z, r);
        }
        let (s_pred, p_pred) = predict(&st.s, &st.p, dt, self.cfg.accel_psd);
        let y = z - s_pred.fixed_rows::<2>(0);
        let s_cov = p_pred.fixed_view::<2, 2>(0, 0) + r;
        let Some(chol) = Cholesky::new(s_cov) else {
            return self.init(point, z, r);
        };
        let d2 = y.dot(&chol.solve(&y));
        if d2 > self.cfg.gate_chi2 {
            let outliers = st.outliers + 1;
            if outliers >= self.cfg.reset_after_outliers {
                return self.init(point, z, r);
            }
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
