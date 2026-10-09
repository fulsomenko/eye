//! I-VT (Velocity-Threshold Identification) fixation/saccade classification
//! (Salvucci & Goldberg, ETRA 2000), with the Tobii I-VT merge/discard
//! post-processing (Olsen, Tobii white paper, 2012).
//!
//! This labels points; it never moves them, so it is not a `GazeFilter`. It
//! must run on already-filtered points: raw webcam noise (σ ≈ 2° per sample)
//! differentiates to velocities that straddle the threshold.

use std::collections::VecDeque;

use eye_core::log::field;
use eye_core::{GazePoint, Timestamp};
use nalgebra::{Point2, Vector2};

use crate::FilterError;
use crate::point::dt_seconds;

/// Label assigned to a single gaze sample by [`IvtClassifier::classify`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EyeMovement {
    Fixation,
    Saccade,
    /// Not enough history yet to compute a velocity (stream start, or just
    /// after a reset / timestamp discontinuity).
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IvtConfig {
    pub velocity_threshold_deg_s: f64,
    pub window_ms: f64,
    pub viewing_distance_mm: f64,
    pub min_fixation_ms: f64,
    pub merge_max_gap_ms: f64,
    pub merge_max_angle_deg: f64,
}

impl Default for IvtConfig {
    fn default() -> Self {
        Self {
            // Olsen 2012 default.
            velocity_threshold_deg_s: 30.0,
            // Olsen uses 20 ms; at 15-30 fps that holds one sample. 100 ms
            // holds 2-3 samples and suppresses filtered-noise velocity to
            // roughly 5 deg/s.
            window_ms: 100.0,
            viewing_distance_mm: 500.0,
            // Olsen uses 60 ms; 100 ms is the usual lower bound for a
            // physiological fixation, and webcam fixations scatter more
            // than a research tracker's.
            min_fixation_ms: 100.0,
            // Olsen's 75 ms assumes >= 60 Hz. One bad sample produces a gap
            // of two sample periods: ~133 ms at 15 Hz, measured 66.7 ms at
            // 30 Hz; 150 ms covers both. At the IR estimator's ~7.5 Hz
            // lit-frame rate the gap is ~267 ms, so that stream needs
            // `merge_max_gap_ms = 300` in config.
            merge_max_gap_ms: 150.0,
            // Olsen uses 0.5 deg; webcam fixations scatter more.
            merge_max_angle_deg: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IvtClassifier {
    cfg: IvtConfig,
    history: VecDeque<(Timestamp, Point2<f64>)>,
}

impl IvtClassifier {
    pub fn new(cfg: IvtConfig) -> Result<Self, FilterError> {
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
        positive(cfg.velocity_threshold_deg_s, "velocity_threshold_deg_s")?;
        positive(cfg.window_ms, "window_ms")?;
        positive(cfg.viewing_distance_mm, "viewing_distance_mm")?;
        positive(cfg.min_fixation_ms, "min_fixation_ms")?;
        positive(cfg.merge_max_gap_ms, "merge_max_gap_ms")?;
        positive(cfg.merge_max_angle_deg, "merge_max_angle_deg")?;
        Ok(Self {
            cfg,
            history: VecDeque::new(),
        })
    }

    pub fn classify(&mut self, point: &GazePoint) -> EyeMovement {
        let t = point.timestamp;
        if let Some(&(last, _)) = self.history.back()
            && dt_seconds(last, t).is_none()
        {
            tracing::warn!(
                { field::REASON } = "non_increasing_timestamp",
                { field::TS_NS } = t.as_nanos(),
                prev_ts_ns = last.as_nanos(),
                "non-increasing gaze timestamp, history cleared"
            );
            self.history.clear();
            return EyeMovement::Unknown;
        }
        let window = self.cfg.window_ms / 1000.0;
        while let Some(&(t0, _)) = self.history.front() {
            match dt_seconds(t0, t) {
                Some(dt) if dt > window => {
                    self.history.pop_front();
                }
                _ => break,
            }
        }
        self.history.push_back((t, point.mm));
        let (t0, p0) = self.history[0];
        let Some(dt) = dt_seconds(t0, t) else {
            tracing::trace!(
                label = "unknown",
                { field::REASON } = "insufficient_history",
                history_s = 0.0,
                "sample labelled"
            );
            return EyeMovement::Unknown;
        };
        if dt < window / 2.0 {
            tracing::trace!(
                label = "unknown",
                { field::REASON } = "insufficient_history",
                history_s = dt,
                "sample labelled"
            );
            return EyeMovement::Unknown;
        }
        let deg_s = ((point.mm - p0).norm() / self.cfg.viewing_distance_mm).to_degrees() / dt;
        let movement = if deg_s > self.cfg.velocity_threshold_deg_s {
            EyeMovement::Saccade
        } else {
            EyeMovement::Fixation
        };
        tracing::trace!(
            label = if movement == EyeMovement::Saccade {
                "saccade"
            } else {
                "fixation"
            },
            velocity_deg_s = deg_s,
            threshold_deg_s = self.cfg.velocity_threshold_deg_s,
            history_s = dt,
            "sample labelled"
        );
        movement
    }

    pub fn reset(&mut self) {
        self.history.clear();
        tracing::debug!({ field::REASON } = "external", "classifier reset");
    }
}

/// A fixation found by [`fixations`]: a group of consecutive `Fixation`
/// labels, possibly merged with near neighbours.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fixation {
    pub start: Timestamp,
    pub end: Timestamp,
    pub centroid_mm: Point2<f64>,
    pub samples: usize,
}

/// Offline: label every point with a fresh classifier, group consecutive
/// fixation samples, merge near neighbours, drop short fixations.
///
/// `Err(FilterError::Param { name: "points", reason: "not sorted by timestamp" })`
/// if any timestamp is not strictly greater than its predecessor.
pub fn fixations(points: &[GazePoint], cfg: &IvtConfig) -> Result<Vec<Fixation>, FilterError> {
    if points.windows(2).any(|w| w[1].timestamp <= w[0].timestamp) {
        return Err(FilterError::Param {
            name: "points",
            reason: "not sorted by timestamp",
        });
    }
    let mut classifier = IvtClassifier::new(*cfg)?;
    let mut candidates: Vec<Fixation> = Vec::new();
    let mut run: Option<(Timestamp, Timestamp, Vector2<f64>, usize)> = None;
    for p in points {
        if classifier.classify(p) == EyeMovement::Fixation {
            run = Some(match run {
                None => (p.timestamp, p.timestamp, p.mm.coords, 1),
                Some((s, _, sum, n)) => (s, p.timestamp, sum + p.mm.coords, n + 1),
            });
        } else if let Some((s, e, sum, n)) = run.take() {
            candidates.push(Fixation {
                start: s,
                end: e,
                centroid_mm: Point2::from(sum / n as f64),
                samples: n,
            });
        }
    }
    if let Some((s, e, sum, n)) = run.take() {
        candidates.push(Fixation {
            start: s,
            end: e,
            centroid_mm: Point2::from(sum / n as f64),
            samples: n,
        });
    }
    let num_candidates = candidates.len();

    let mut merged: Vec<Fixation> = Vec::new();
    for f in candidates {
        if let Some(last) = merged.last_mut() {
            let gap_ms = dt_seconds(last.end, f.start).unwrap_or(0.0) * 1000.0;
            let angle_deg =
                ((f.centroid_mm - last.centroid_mm).norm() / cfg.viewing_distance_mm).to_degrees();
            if gap_ms <= cfg.merge_max_gap_ms && angle_deg <= cfg.merge_max_angle_deg {
                let n = last.samples + f.samples;
                last.centroid_mm = Point2::from(
                    (last.centroid_mm.coords * last.samples as f64
                        + f.centroid_mm.coords * f.samples as f64)
                        / n as f64,
                );
                last.samples = n;
                last.end = f.end;
                tracing::debug!(
                    { field::REASON } = "near",
                    gap_ms,
                    angle_deg,
                    samples = last.samples as u64,
                    "fixations merged"
                );
                continue;
            }
        }
        merged.push(f);
    }

    let mut result = Vec::with_capacity(merged.len());
    for f in merged {
        let duration_ms = dt_seconds(f.start, f.end).unwrap_or(0.0) * 1000.0;
        if duration_ms >= cfg.min_fixation_ms {
            result.push(f);
        } else {
            tracing::debug!(
                { field::REASON } = "too_short",
                duration_ms,
                min_fixation_ms = cfg.min_fixation_ms,
                "fixation discarded"
            );
        }
    }

    tracing::info!(
        points = points.len() as u64,
        candidates = num_candidates as u64,
        fixations = result.len() as u64,
        "fixations labelled"
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use eye_core::OutputId;
    use eye_geometry::screen::{confidence_from_cov, mm_to_px_logical, mm_to_px_physical};
    use eye_geometry::synth::SplitMix64;
    use nalgebra::Matrix2;

    use super::*;

    const P30: u64 = 33_333_333;

    fn edp1() -> eye_core::ScreenModel {
        eye_core::ScreenModel {
            output: OutputId::new("eDP-1"),
            size_mm: Vector2::new(310.0, 170.0),
            size_px: (3840, 2160),
            scale: 2.0,
        }
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
            confidence: confidence_from_cov(&cov_mm),
        }
    }

    fn pt(t_ns: u64, x_mm: f64, y_mm: f64) -> GazePoint {
        pt_var(t_ns, x_mm, y_mm, 1.0)
    }

    /// `(t0, t1, p0, p1)`: a path segment spanning `[t0, t1)` seconds from
    /// `p0` to `p1`.
    type Segment = (f64, f64, (f64, f64), (f64, f64));

    /// Samples `k = 0..n` at `t_k = k * P30` ns along a piecewise-linear
    /// path given by `segments`, plus independent gaussian noise with std
    /// `sigma` mm on x then y.
    fn path(segments: &[Segment], n: u64, seed: u64, sigma: f64) -> Vec<GazePoint> {
        let mut rng = SplitMix64::new(seed);
        (0..n)
            .map(|k| {
                let t_ns = k * P30;
                let t = t_ns as f64 * 1e-9;
                let (x, y) = match segments.iter().find(|(t0, t1, ..)| *t0 <= t && t < *t1) {
                    Some((t0, t1, p0, p1)) => {
                        let frac = (t - t0) / (t1 - t0);
                        (p0.0 + (p1.0 - p0.0) * frac, p0.1 + (p1.1 - p0.1) * frac)
                    }
                    None => segments[0].2,
                };
                let x = x + sigma * rng.gaussian();
                let y = y + sigma * rng.gaussian();
                pt_var(t_ns, x, y, 1.0)
            })
            .collect()
    }

    fn classifier(cfg: IvtConfig) -> IvtClassifier {
        IvtClassifier::new(cfg).unwrap()
    }

    #[test]
    fn test_first_samples_are_unknown() {
        let mut c = classifier(IvtConfig::default());
        assert_eq!(c.classify(&pt(0, 100.0, 100.0)), EyeMovement::Unknown);
        assert_eq!(
            c.classify(&pt(P30, 100.0, 100.0)),
            EyeMovement::Unknown,
            "k=1, 33.3 ms of history is < window/2 = 50 ms"
        );
        assert_ne!(
            c.classify(&pt(2 * P30, 100.0, 100.0)),
            EyeMovement::Unknown,
            "k=2, 66.7 ms of history is >= window/2 = 50 ms"
        );
    }

    #[test]
    fn test_stationary_noise_is_fixation() {
        let mut c = classifier(IvtConfig::default());
        let mut rng = SplitMix64::new(5);
        let mut saw_non_unknown = false;
        for k in 0..90u64 {
            let x = 155.0 + 3.0 * rng.gaussian();
            let y = 85.0 + 3.0 * rng.gaussian();
            let label = c.classify(&pt(k * P30, x, y));
            if label != EyeMovement::Unknown {
                saw_non_unknown = true;
                assert_eq!(label, EyeMovement::Fixation, "k={k}");
            }
        }
        assert!(saw_non_unknown);
    }

    #[test]
    fn test_fast_motion_is_saccade() {
        let mut c = classifier(IvtConfig::default());
        for k in 0..5u64 {
            c.classify(&pt(k * P30, 100.0, 100.0));
        }
        let label = c.classify(&pt(5 * P30, 200.0, 100.0));
        assert_eq!(label, EyeMovement::Saccade);
    }

    #[test]
    fn test_scanpath_yields_three_fixations() {
        let a = (100.0, 100.0);
        let b = (200.0, 100.0);
        let c = (200.0, 0.0);
        let segments = [
            (0.0, 0.3, a, a),
            (0.3, 0.34, a, b),
            (0.34, 0.64, b, b),
            (0.64, 0.68, b, c),
            (0.68, 1.0, c, c),
        ];
        let points = path(&segments, 30, 4, 1.0);
        let result = fixations(&points, &IvtConfig::default()).unwrap();
        assert_eq!(result.len(), 3, "{result:?}");

        let expected_dur_ms = [233.3, 200.0, 166.7];
        for ((f, expected), expected_dur) in result.iter().zip([a, b, c]).zip(expected_dur_ms) {
            let dur_ms = dt_seconds(f.start, f.end).unwrap() * 1000.0;
            assert!((150.0..=250.0).contains(&dur_ms), "dur_ms = {dur_ms}");
            approx::assert_abs_diff_eq!(dur_ms, expected_dur, epsilon = 0.2);
            let dist = (f.centroid_mm - Point2::new(expected.0, expected.1)).norm();
            assert!(dist < 2.0, "centroid {:?} vs {:?}", f.centroid_mm, expected);
        }
    }

    #[test]
    fn test_short_fixation_is_discarded() {
        let a = (100.0, 100.0);
        let b = (200.0, 100.0);
        let c = (300.0, 100.0);
        let segments = [
            (0.0, 0.3, a, a),
            (0.3, 0.34, a, b),
            (0.34, 0.406, b, b),
            (0.406, 0.446, b, c),
            (0.446, 0.8, c, c),
        ];
        let points = path(&segments, 24, 6, 1.0);
        let result = fixations(&points, &IvtConfig::default()).unwrap();
        assert_eq!(result.len(), 2, "{result:?}");
        for f in &result {
            let dist = (f.centroid_mm - Point2::new(b.0, b.1)).norm();
            assert!(
                dist >= 5.0,
                "unexpected fixation near B: {:?}",
                f.centroid_mm
            );
        }
    }

    #[test]
    fn test_nearby_fixations_are_merged() {
        let mut rng = SplitMix64::new(8);
        let points: Vec<GazePoint> = (0..24u64)
            .map(|k| {
                let base_x = if k < 12 { 100.0 } else { 100.0 + 4.36 };
                let x = if k == 12 { base_x + 200.0 } else { base_x };
                let x = x + 0.3 * rng.gaussian();
                pt_var(k * P30, x, 100.0, 1.0)
            })
            .collect();
        let result = fixations(&points, &IvtConfig::default()).unwrap();
        assert_eq!(result.len(), 1, "{result:?}");
        approx::assert_abs_diff_eq!(result[0].centroid_mm.x, 102.06, epsilon = 0.5);
    }

    #[test]
    fn test_far_fixations_are_not_merged() {
        let mut rng = SplitMix64::new(8);
        let points: Vec<GazePoint> = (0..24u64)
            .map(|k| {
                let base_x = if k < 12 { 100.0 } else { 100.0 + 26.2 };
                let x = if k == 12 { base_x + 200.0 } else { base_x };
                let x = x + 0.3 * rng.gaussian();
                pt_var(k * P30, x, 100.0, 1.0)
            })
            .collect();
        let result = fixations(&points, &IvtConfig::default()).unwrap();
        assert_eq!(result.len(), 2, "{result:?}");
    }

    #[test]
    fn test_fixations_rejects_unsorted_points() {
        let a = (100.0, 100.0);
        let b = (200.0, 100.0);
        let c = (200.0, 0.0);
        let segments = [
            (0.0, 0.3, a, a),
            (0.3, 0.34, a, b),
            (0.34, 0.64, b, b),
            (0.64, 0.68, b, c),
            (0.68, 1.0, c, c),
        ];
        let mut points = path(&segments, 30, 4, 1.0);
        points.swap(3, 4);
        let result = fixations(&points, &IvtConfig::default());
        assert!(matches!(
            result,
            Err(FilterError::Param { name: "points", .. })
        ));
    }

    #[test]
    fn test_non_increasing_timestamp_returns_unknown_and_clears() {
        let mut c = classifier(IvtConfig::default());
        for k in 0..5u64 {
            c.classify(&pt(k * P30, 100.0, 100.0));
        }
        let label = c.classify(&pt(4 * P30, 999.0, 999.0));
        assert_eq!(label, EyeMovement::Unknown);
        let next = c.classify(&pt(5 * P30, 100.0, 100.0));
        assert_eq!(
            next,
            EyeMovement::Unknown,
            "history was cleared, so this is like the first sample again"
        );
    }

    #[test]
    fn test_config_rejects_unknown_key() {
        let table: toml::Table = toml::from_str("bogus = 1").unwrap();
        assert!(table.try_into::<IvtConfig>().is_err());

        let empty = toml::Table::new();
        let cfg: IvtConfig = empty.try_into().unwrap();
        assert_eq!(cfg, IvtConfig::default());
    }

    #[test]
    fn test_new_rejects_non_positive_threshold() {
        let cfg = IvtConfig {
            velocity_threshold_deg_s: 0.0,
            ..IvtConfig::default()
        };
        assert!(matches!(
            IvtClassifier::new(cfg),
            Err(FilterError::Param {
                name: "velocity_threshold_deg_s",
                ..
            })
        ));
    }

    fn find<'a>(records: &'a [eye_log::Record], message: &str) -> Vec<&'a eye_log::Record> {
        records
            .iter()
            .filter(|r| r.target == "eye_filter::fixation" && r.message == message)
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
    fn test_logs_sample_labelled_at_trace() {
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut c = classifier(IvtConfig::default());
            c.classify(&pt(0, 100.0, 100.0));
            c.classify(&pt(P30, 100.0, 100.0));
            c.classify(&pt(2 * P30, 100.0, 100.0));
        });
        let recs = find(&records, "sample labelled");
        assert_eq!(recs.len(), 3);
        for r in &recs {
            assert_eq!(r.level, eye_log::Level::Trace);
        }
        assert_eq!(
            recs[0].fields.get("label"),
            Some(&eye_log::Value::Str("unknown".to_string()))
        );
        assert_eq!(
            recs[0].fields.get("reason"),
            Some(&eye_log::Value::Str("insufficient_history".to_string()))
        );
        approx::assert_abs_diff_eq!(f64_field(recs[0], "history_s"), 0.0, epsilon = 1e-9);

        assert_eq!(
            recs[1].fields.get("label"),
            Some(&eye_log::Value::Str("unknown".to_string()))
        );
        assert_eq!(
            recs[1].fields.get("reason"),
            Some(&eye_log::Value::Str("insufficient_history".to_string()))
        );
        assert!((f64_field(recs[1], "history_s") - P30 as f64 * 1e-9).abs() < 1e-9);

        assert_eq!(
            recs[2].fields.get("label"),
            Some(&eye_log::Value::Str("fixation".to_string()))
        );
        assert!(recs[2].fields.contains_key("velocity_deg_s"));
        approx::assert_abs_diff_eq!(f64_field(recs[2], "threshold_deg_s"), 30.0, epsilon = 1e-9);
    }

    #[test]
    fn test_logs_non_increasing_timestamp_at_warn() {
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut c = classifier(IvtConfig::default());
            for k in 0..5u64 {
                c.classify(&pt(k * P30, 100.0, 100.0));
            }
            c.classify(&pt(4 * P30, 999.0, 999.0));
        });
        let recs: Vec<_> = records
            .iter()
            .filter(|r| r.target == "eye_filter::fixation" && r.level == eye_log::Level::Warn)
            .collect();
        assert_eq!(recs.len(), 1);
        assert_eq!(
            recs[0].fields.get("prev_ts_ns"),
            Some(&eye_log::Value::U64(4 * P30))
        );
    }

    #[test]
    fn test_logs_external_reset_at_debug() {
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut c = classifier(IvtConfig::default());
            c.classify(&pt(0, 100.0, 100.0));
            c.reset();
        });
        let recs = find(&records, "classifier reset");
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].level, eye_log::Level::Debug);
        assert_eq!(
            recs[0].fields.get("reason"),
            Some(&eye_log::Value::Str("external".to_string()))
        );
    }

    #[test]
    fn test_logs_fixations_labelled_at_info() {
        let a = (100.0, 100.0);
        let b = (200.0, 100.0);
        let c = (200.0, 0.0);
        let segments = [
            (0.0, 0.3, a, a),
            (0.3, 0.34, a, b),
            (0.34, 0.64, b, b),
            (0.64, 0.68, b, c),
            (0.68, 1.0, c, c),
        ];
        let points = path(&segments, 30, 4, 1.0);
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::INFO, || {
            fixations(&points, &IvtConfig::default()).unwrap();
        });
        let own: Vec<_> = records
            .iter()
            .filter(|r| r.target == "eye_filter::fixation")
            .collect();
        for r in &own {
            assert_eq!(r.level, eye_log::Level::Info, "{r:?}");
        }
        let recs = find(&records, "fixations labelled");
        assert_eq!(recs.len(), 1);
        assert_eq!(u64_field(recs[0], "points"), 30);
        assert_eq!(u64_field(recs[0], "candidates"), 3);
        assert_eq!(u64_field(recs[0], "fixations"), 3);
    }

    /// 32 points: `x = 100.0` for `k < 12`, a 200 mm spike at `k == 12`, `140.0`
    /// for `13..=18` (a 40 mm step from the 100.0 baseline), a second spike at
    /// `k == 19`, then `100.0`. The middle
    /// candidate run (`k = 16..=18`, 66.7 ms) is far enough from its
    /// neighbours (4.58 deg at 500 mm) that it is never merged, so
    /// `min_fixation_ms = 100.0` discards it on its own.
    fn discard_fixture() -> Vec<GazePoint> {
        (0..32u64)
            .map(|k| {
                let x = match k {
                    0..=11 => 100.0,
                    12 => 300.0,
                    13..=18 => 140.0,
                    19 => 300.0,
                    _ => 100.0,
                };
                pt_var(k * P30, x, 100.0, 1.0)
            })
            .collect()
    }

    #[test]
    fn test_logs_fixation_discarded_at_debug() {
        let points = discard_fixture();
        let (result, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            fixations(&points, &IvtConfig::default()).unwrap()
        });
        assert_eq!(result.len(), 2, "{result:?}");
        let recs = find(&records, "fixation discarded");
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].level, eye_log::Level::Debug);
        assert_eq!(
            recs[0].fields.get("reason"),
            Some(&eye_log::Value::Str("too_short".to_string()))
        );
        assert!((f64_field(recs[0], "duration_ms") - 2.0 * P30 as f64 * 1e-6).abs() < 1e-6);
        approx::assert_abs_diff_eq!(f64_field(recs[0], "min_fixation_ms"), 100.0, epsilon = 1e-9);
        assert!(find(&records, "fixations merged").is_empty());
        let info = find(&records, "fixations labelled");
        assert_eq!(info.len(), 1);
        assert_eq!(u64_field(info[0], "candidates"), 3);
        assert_eq!(u64_field(info[0], "fixations"), 2);
    }

    /// 24 points, no noise: `x = 100.0` for `k < 12`, `104.36` after, with a
    /// 200 mm spike at `k == 12`. This yields three candidate fixations
    /// (`k=2..=11`, `13..=14`, `16..=23`) close enough (<= 1 deg, <= 150 ms
    /// gap) to merge pairwise into one.
    fn merge_fixture() -> Vec<GazePoint> {
        (0..24u64)
            .map(|k| {
                let base_x = if k < 12 { 100.0 } else { 100.0 + 4.36 };
                let x = if k == 12 { base_x + 200.0 } else { base_x };
                pt_var(k * P30, x, 100.0, 1.0)
            })
            .collect()
    }

    #[test]
    fn test_logs_fixations_merged_at_debug() {
        let points = merge_fixture();
        let (result, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            fixations(&points, &IvtConfig::default()).unwrap()
        });
        assert_eq!(result.len(), 1, "{result:?}");
        let recs = find(&records, "fixations merged");
        assert_eq!(recs.len(), 2);
        for r in &recs {
            assert_eq!(r.level, eye_log::Level::Debug);
            assert!((f64_field(r, "gap_ms") - 2.0 * P30 as f64 * 1e-6).abs() < 1e-6);
            assert!(f64_field(r, "angle_deg") <= 1.0);
        }
        assert_eq!(u64_field(recs[0], "samples"), 12);
        assert_eq!(u64_field(recs[1], "samples"), 20);
        assert_eq!(u64_field(recs[1], "samples"), result[0].samples as u64);
        let info = find(&records, "fixations labelled");
        assert_eq!(info.len(), 1);
        assert_eq!(u64_field(info[0], "candidates"), 3);
        assert_eq!(u64_field(info[0], "fixations"), 1);
    }

    #[test]
    fn test_logs_reason_field_uses_vocabulary() {
        let points = discard_fixture();
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            fixations(&points, &IvtConfig::default()).unwrap();
        });
        let own: Vec<_> = records
            .iter()
            .filter(|r| r.target == "eye_filter::fixation")
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
            assert!(!r.fields.contains_key("timestamp"));
        }
    }
}
