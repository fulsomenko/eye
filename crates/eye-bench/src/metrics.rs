//! Pure scoring functions over recorded gaze/target samples. No I/O.

use std::time::Duration;

use eye_core::{Timestamp, grid::Grid};
use nalgebra::{Point2, Point3, Vector2, Vector3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
pub struct EvalPoint {
    pub timestamp: Timestamp,
    /// Eye position (mean of the ray origins of the FrameSet), screen frame, mm.
    pub eye_mm: Point3<f64>,
    pub gaze_mm: Point2<f64>,
    pub gaze_px_logical: Point2<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvalWindow {
    pub target_index: usize,
    pub start: Timestamp,
    pub end: Timestamp,
    pub target_mm: Point2<f64>,
    pub target_px_logical: Point2<f64>,
    /// Logical size of the output the target was shown on, px.
    pub screen_logical: Vector2<f64>,
    /// Points with `start <= timestamp < end`, in timestamp order.
    pub points: Vec<EvalPoint>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EvalInput {
    pub windows: Vec<EvalWindow>,
    /// Processing time of every FrameSet of the session (not only those inside windows).
    pub processing: Vec<Duration>,
}

mod millis {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        u64::try_from(d.as_millis())
            .unwrap_or(u64::MAX)
            .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        Ok(Duration::from_millis(u64::deserialize(d)?))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct MetricParams {
    /// `[cols, rows]`, same order as `[output].grid`.
    pub grids: Vec<[u32; 2]>,
    pub boundary_margin_px: f64,
    /// Integer milliseconds in TOML/JSON (`dropout_bin = 200`).
    #[serde(with = "millis")]
    pub dropout_bin: Duration,
}

impl Default for MetricParams {
    fn default() -> Self {
        Self {
            grids: vec![[3, 3], [4, 4]],
            boundary_margin_px: 20.0,
            dropout_bin: Duration::from_millis(200),
        }
    }
}

impl MetricParams {
    /// Err for a zero grid dimension, `dropout_bin == 0`, or a negative/non-finite margin.
    pub fn validate(&self) -> Result<Vec<Grid>, String> {
        if self.dropout_bin.is_zero() {
            return Err("dropout_bin must be non-zero".to_string());
        }
        if !self.boundary_margin_px.is_finite() || self.boundary_margin_px < 0.0 {
            return Err("boundary_margin_px must be finite and non-negative".to_string());
        }
        self.grids
            .iter()
            .map(|&[c, r]| Grid::new(c, r).ok_or_else(|| format!("invalid grid {c}x{r}")))
            .collect()
    }
}

impl EvalInput {
    /// Pooled input for an aggregate row: windows and processing times concatenated in argument order.
    pub fn concat<'a>(inputs: impl IntoIterator<Item = &'a EvalInput>) -> EvalInput {
        let mut windows = Vec::new();
        let mut processing = Vec::new();
        for input in inputs {
            windows.extend(input.windows.iter().cloned());
            processing.extend(input.processing.iter().copied());
        }
        EvalInput {
            windows,
            processing,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Summary {
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RegionHit {
    pub cols: u32,
    pub rows: u32,
    /// Windows evaluated: target clear of every interior cell boundary and at least one point.
    pub windows: usize,
    /// Windows whose target lies within `boundary_margin_px` of an interior boundary (with or without points).
    pub excluded: usize,
    pub hits: usize,
    pub hit_rate: Option<f64>,
    pub sample_hit_rate: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SessionMetrics {
    pub windows: usize,
    pub samples: usize,
    pub angular_error_deg: Option<Summary>,
    pub px_error_logical: Option<Summary>,
    pub accuracy_deg: Option<f64>,
    pub precision_rms_s2s_deg: Option<f64>,
    pub precision_pooled_rms_s2s_deg: Option<f64>,
    /// One entry per valid grid of `MetricParams::grids`, same order.
    pub regions: Vec<RegionHit>,
    pub processing_ms: Option<Summary>,
    pub dropout_rate: Option<f64>,
    pub output_rate_hz: Option<f64>,
}

fn lift(p: &Point2<f64>) -> Point3<f64> {
    Point3::new(p.x, p.y, 0.0)
}

pub fn vector_angle_deg(u: &Vector3<f64>, v: &Vector3<f64>) -> f64 {
    u.cross(v).norm().atan2(u.dot(v)).to_degrees()
}

pub fn angle_deg(eye: &Point3<f64>, a: &Point2<f64>, b: &Point2<f64>) -> f64 {
    vector_angle_deg(&(lift(a) - eye), &(lift(b) - eye))
}

/// Hyndman and Fan type 7: `h = (n - 1) p/100`, linear interpolation between `sorted[floor(h)]` and `sorted[ceil(h)]`.
pub fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    let n = sorted.len();
    if n == 0 {
        return None;
    }
    let h = (n as f64 - 1.0) * (p / 100.0).clamp(0.0, 1.0);
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    Some(sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo]))
}

pub(crate) fn median(values: &mut [f64]) -> Option<f64> {
    values.sort_by(f64::total_cmp);
    percentile(values, 50.0)
}

pub fn near_boundary(
    p: &Point2<f64>,
    screen_logical: &Vector2<f64>,
    grid: Grid,
    margin_px: f64,
) -> bool {
    let close = |v: f64, size: f64, n: u32| {
        (1..n).any(|k| (v - size * f64::from(k) / f64::from(n)).abs() < margin_px)
    };
    close(p.x, screen_logical.x, grid.cols()) || close(p.y, screen_logical.y, grid.rows())
}

fn summary(mut values: Vec<f64>) -> Option<Summary> {
    if values.is_empty() {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.sort_by(f64::total_cmp);
    Some(Summary {
        mean,
        p50: percentile(&values, 50.0)?,
        p95: percentile(&values, 95.0)?,
    })
}

fn accuracy(windows: &[EvalWindow]) -> Option<f64> {
    let offsets: Vec<f64> = windows
        .iter()
        .filter(|w| !w.points.is_empty())
        .map(|w| {
            let n = w.points.len() as f64;
            let gaze_sum = w
                .points
                .iter()
                .fold(Vector2::new(0.0, 0.0), |acc, p| acc + p.gaze_mm.coords);
            let eye_sum = w
                .points
                .iter()
                .fold(Vector3::new(0.0, 0.0, 0.0), |acc, p| acc + p.eye_mm.coords);
            let p_bar = Point2::from(gaze_sum / n);
            let e_bar = Point3::from(eye_sum / n);
            angle_deg(&e_bar, &p_bar, &w.target_mm)
        })
        .collect();
    (!offsets.is_empty()).then(|| offsets.iter().sum::<f64>() / offsets.len() as f64)
}

/// Per-window RMS of the sample-to-sample angle over pairs whose interval lies within
/// `[0.5, 1.5]` x the window's median interval; `None` for a window with fewer than two points.
fn window_rms_s2s(w: &EvalWindow) -> Option<f64> {
    if w.points.len() < 2 {
        return None;
    }
    let dt = |pair: &[EvalPoint]| (pair[1].timestamp.0 - pair[0].timestamp.0).as_secs_f64();
    let mut dts: Vec<f64> = w.points.windows(2).map(dt).collect();
    let mid = median(&mut dts)?;
    let (lo, hi) = (0.5 * mid, 1.5 * mid);
    let (sum_sq, n) = w.points.windows(2).fold((0.0, 0usize), |(s, n), pair| {
        if dt(pair) < lo || dt(pair) > hi {
            return (s, n);
        }
        let theta = vector_angle_deg(
            &(lift(&pair[0].gaze_mm) - pair[0].eye_mm),
            &(lift(&pair[1].gaze_mm) - pair[1].eye_mm),
        );
        (s + theta * theta, n + 1)
    });
    (n > 0).then(|| (sum_sq / n as f64).sqrt())
}

fn precision(windows: &[EvalWindow]) -> Option<f64> {
    let mut per_window: Vec<f64> = windows.iter().filter_map(window_rms_s2s).collect();
    median(&mut per_window)
}

fn precision_pooled(windows: &[EvalWindow]) -> Option<f64> {
    let (sum_sq, n) =
        windows
            .iter()
            .flat_map(|w| w.points.windows(2))
            .fold((0.0, 0usize), |(s, n), pair| {
                let theta = vector_angle_deg(
                    &(lift(&pair[0].gaze_mm) - pair[0].eye_mm),
                    &(lift(&pair[1].gaze_mm) - pair[1].eye_mm),
                );
                (s + theta * theta, n + 1)
            });
    (n > 0).then(|| (sum_sq / n as f64).sqrt())
}

fn region(windows: &[EvalWindow], grid: Grid, margin: f64) -> RegionHit {
    let (mut evaluated, mut excluded, mut hits, mut samples, mut sample_hits) = (0, 0, 0, 0, 0);
    for w in windows {
        if near_boundary(&w.target_px_logical, &w.screen_logical, grid, margin) {
            excluded += 1;
            continue;
        }
        if w.points.is_empty() {
            continue;
        }
        let size = (w.screen_logical.x, w.screen_logical.y);
        let target = grid.cell_of(w.target_px_logical, size);
        let mut xs: Vec<f64> = w.points.iter().map(|p| p.gaze_px_logical.x).collect();
        let mut ys: Vec<f64> = w.points.iter().map(|p| p.gaze_px_logical.y).collect();
        let med = median(&mut xs)
            .zip(median(&mut ys))
            .map(|(x, y)| Point2::new(x, y));
        evaluated += 1;
        if target.is_some() && med.and_then(|m| grid.cell_of(m, size)) == target {
            hits += 1;
        }
        for p in &w.points {
            samples += 1;
            if target.is_some() && grid.cell_of(p.gaze_px_logical, size) == target {
                sample_hits += 1;
            }
        }
    }
    RegionHit {
        cols: grid.cols(),
        rows: grid.rows(),
        windows: evaluated,
        excluded,
        hits,
        hit_rate: (evaluated > 0).then(|| hits as f64 / evaluated as f64),
        sample_hit_rate: (samples > 0).then(|| sample_hits as f64 / samples as f64),
    }
}

fn dropout(windows: &[EvalWindow], bin: Duration) -> Option<f64> {
    let bin_ns = bin.as_nanos();
    if bin_ns == 0 {
        return None;
    }
    let (mut total, mut empty) = (0u128, 0u128);
    for w in windows {
        let n = w.end.0.saturating_sub(w.start.0).as_nanos() / bin_ns;
        let mut filled = vec![false; usize::try_from(n).unwrap_or(0)];
        for p in &w.points {
            let i = p.timestamp.0.saturating_sub(w.start.0).as_nanos() / bin_ns;
            if let Some(slot) = usize::try_from(i).ok().and_then(|i| filled.get_mut(i)) {
                *slot = true;
            }
        }
        total += n;
        empty += filled.iter().filter(|f| !**f).count() as u128;
    }
    (total > 0).then(|| empty as f64 / total as f64)
}

pub fn compute(input: &EvalInput, params: &MetricParams) -> SessionMetrics {
    let windows = &input.windows;
    let samples = windows.iter().map(|w| w.points.len()).sum();

    let angular_errors: Vec<f64> = windows
        .iter()
        .flat_map(|w| {
            w.points
                .iter()
                .map(move |p| angle_deg(&p.eye_mm, &p.gaze_mm, &w.target_mm))
        })
        .collect();
    let px_errors: Vec<f64> = windows
        .iter()
        .flat_map(|w| {
            w.points
                .iter()
                .map(move |p| (p.gaze_px_logical - w.target_px_logical).norm())
        })
        .collect();
    let processing_ms: Vec<f64> = input
        .processing
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .collect();

    let total_window_secs: f64 = windows
        .iter()
        .map(|w| w.end.0.saturating_sub(w.start.0).as_secs_f64())
        .sum();
    let output_rate_hz = (total_window_secs > 0.0).then(|| samples as f64 / total_window_secs);

    let regions = params
        .grids
        .iter()
        .filter_map(|&[c, r]| Grid::new(c, r))
        .map(|g| region(windows, g, params.boundary_margin_px))
        .collect();

    SessionMetrics {
        windows: windows.len(),
        samples,
        angular_error_deg: summary(angular_errors),
        px_error_logical: summary(px_errors),
        accuracy_deg: accuracy(windows),
        precision_rms_s2s_deg: precision(windows),
        precision_pooled_rms_s2s_deg: precision_pooled(windows),
        regions,
        processing_ms: summary(processing_ms),
        dropout_rate: dropout(windows, params.dropout_bin),
        output_rate_hz,
    }
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use proptest::prelude::*;

    use super::*;

    const SCREEN_PX: (f64, f64) = (1920.0, 1080.0);
    fn eye() -> Point3<f64> {
        Point3::new(155.0, 85.0, -500.0)
    }
    fn d1() -> f64 {
        500.0 * (1.0f64).to_radians().tan()
    }

    fn pt(t_ms: u64, mm: Point2<f64>) -> EvalPoint {
        EvalPoint {
            timestamp: Timestamp(Duration::from_millis(t_ms)),
            eye_mm: eye(),
            gaze_mm: mm,
            gaze_px_logical: Point2::new(mm.x * 1920.0 / 310.0, mm.y * 1080.0 / 170.0),
        }
    }

    fn ptpx(t_ms: u64, px: Point2<f64>) -> EvalPoint {
        let mm = Point2::new(px.x * 310.0 / 1920.0, px.y * 170.0 / 1080.0);
        EvalPoint {
            timestamp: Timestamp(Duration::from_millis(t_ms)),
            eye_mm: eye(),
            gaze_mm: mm,
            gaze_px_logical: px,
        }
    }

    fn win(
        start_ms: u64,
        end_ms: u64,
        target_px: Point2<f64>,
        points: Vec<EvalPoint>,
    ) -> EvalWindow {
        let target_mm = Point2::new(target_px.x * 310.0 / 1920.0, target_px.y * 170.0 / 1080.0);
        EvalWindow {
            target_index: 0,
            start: Timestamp(Duration::from_millis(start_ms)),
            end: Timestamp(Duration::from_millis(end_ms)),
            target_mm,
            target_px_logical: target_px,
            screen_logical: Vector2::new(SCREEN_PX.0, SCREEN_PX.1),
            points,
        }
    }

    #[test]
    fn test_angle_one_degree_offset_is_one_degree() {
        let a = Point2::new(155.0 + d1(), 85.0);
        let b = Point2::new(155.0, 85.0);
        assert_relative_eq!(angle_deg(&eye(), &a, &b), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_angle_off_axis_is_45_degrees() {
        let eye = Point3::new(0.0, 0.0, -500.0);
        let a = Point2::new(500.0, 0.0);
        let b = Point2::new(0.0, 0.0);
        assert_relative_eq!(angle_deg(&eye, &a, &b), 45.0, epsilon = 1e-12);
    }

    #[test]
    fn test_angle_identical_points_is_zero() {
        let t = Point2::new(155.0, 85.0);
        assert_eq!(angle_deg(&eye(), &t, &t), 0.0);
    }

    #[test]
    fn test_px_error_is_euclidean_logical() {
        let input = EvalInput {
            windows: vec![win(
                0,
                800,
                Point2::new(960.0, 540.0),
                vec![ptpx(0, Point2::new(1000.0, 570.0))],
            )],
            processing: vec![],
        };
        let metrics = compute(&input, &MetricParams::default());
        assert_relative_eq!(metrics.px_error_logical.unwrap().mean, 50.0, epsilon = 1e-9);
    }

    #[test]
    fn test_scattered_samples_have_error_but_zero_accuracy_offset() {
        let points = (0..10)
            .map(|i| {
                let x = if i % 2 == 0 {
                    155.0 + d1()
                } else {
                    155.0 - d1()
                };
                pt(i as u64 * 80, Point2::new(x, 85.0))
            })
            .collect();
        let mut window = win(0, 800, Point2::new(960.0, 540.0), points);
        window.target_mm = Point2::new(155.0, 85.0);
        let input = EvalInput {
            windows: vec![window],
            processing: vec![],
        };
        let metrics = compute(&input, &MetricParams::default());
        assert_relative_eq!(
            metrics.angular_error_deg.as_ref().unwrap().mean,
            1.0,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(metrics.accuracy_deg.unwrap(), 0.0, epsilon = 1e-9);
        assert_relative_eq!(metrics.precision_rms_s2s_deg.unwrap(), 2.0, epsilon = 1e-9);
    }

    #[test]
    fn test_constant_offset_has_accuracy_and_zero_precision() {
        let points = (0..10)
            .map(|i| pt(i as u64 * 80, Point2::new(155.0 + d1(), 85.0)))
            .collect();
        let mut window = win(0, 800, Point2::new(960.0, 540.0), points);
        window.target_mm = Point2::new(155.0, 85.0);
        let input = EvalInput {
            windows: vec![window],
            processing: vec![],
        };
        let metrics = compute(&input, &MetricParams::default());
        assert_relative_eq!(metrics.accuracy_deg.unwrap(), 1.0, epsilon = 1e-9);
        assert_eq!(metrics.precision_rms_s2s_deg, Some(0.0));
        assert_relative_eq!(
            metrics.angular_error_deg.as_ref().unwrap().p95,
            1.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_precision_uses_per_sample_eye() {
        let gaze = Point2::new(155.0, 85.0);
        let p1 = EvalPoint {
            timestamp: Timestamp(Duration::from_millis(0)),
            eye_mm: Point3::new(150.0, 85.0, -500.0),
            gaze_mm: gaze,
            gaze_px_logical: Point2::new(gaze.x * 1920.0 / 310.0, gaze.y * 1080.0 / 170.0),
        };
        let p2 = EvalPoint {
            timestamp: Timestamp(Duration::from_millis(80)),
            eye_mm: Point3::new(160.0, 85.0, -500.0),
            gaze_mm: gaze,
            gaze_px_logical: Point2::new(gaze.x * 1920.0 / 310.0, gaze.y * 1080.0 / 170.0),
        };
        let input = EvalInput {
            windows: vec![win(0, 800, Point2::new(960.0, 540.0), vec![p1, p2])],
            processing: vec![],
        };
        let metrics = compute(&input, &MetricParams::default());
        assert_relative_eq!(
            metrics.precision_rms_s2s_deg.unwrap(),
            1.145_877_395_366_971_9,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_precision_single_point_window_is_none() {
        let input = EvalInput {
            windows: vec![win(
                0,
                800,
                Point2::new(960.0, 540.0),
                vec![pt(0, Point2::new(155.0, 85.0))],
            )],
            processing: vec![],
        };
        let metrics = compute(&input, &MetricParams::default());
        assert_eq!(metrics.precision_rms_s2s_deg, None);
        assert!(metrics.accuracy_deg.is_some());
    }

    fn dn(deg: f64) -> f64 {
        500.0 * deg.to_radians().tan()
    }

    #[test]
    fn test_precision_median_of_windows_ignores_one_jump() {
        fn alt_points(base_ms: u64) -> Vec<EvalPoint> {
            (0..10)
                .map(|i| {
                    let x = if i % 2 == 0 {
                        155.0 + d1()
                    } else {
                        155.0 - d1()
                    };
                    pt(base_ms + i as u64 * 80, Point2::new(x, 85.0))
                })
                .collect()
        }
        fn alt_window(base_ms: u64) -> EvalWindow {
            win(
                base_ms,
                base_ms + 800,
                Point2::new(960.0, 540.0),
                alt_points(base_ms),
            )
        }
        fn jump_window(base_ms: u64) -> EvalWindow {
            let mut points = alt_points(base_ms);
            points[5] = pt(base_ms + 5 * 80, Point2::new(155.0 + dn(10.0), 85.0));
            win(base_ms, base_ms + 800, Point2::new(960.0, 540.0), points)
        }
        let windows = vec![
            alt_window(0),
            alt_window(10_000),
            alt_window(20_000),
            jump_window(30_000),
        ];
        let input = EvalInput {
            windows,
            processing: vec![],
        };
        let metrics = compute(&input, &MetricParams::default());
        assert_relative_eq!(metrics.precision_rms_s2s_deg.unwrap(), 2.0, epsilon = 1e-9);
        assert!(metrics.precision_pooled_rms_s2s_deg.unwrap() > 2.0);
    }

    #[test]
    fn test_precision_skips_pairs_with_irregular_interval() {
        let dir_a = Point2::new(155.0, 85.0);
        let dir_b = Point2::new(155.0 + dn(5.0), 85.0);
        let mut points: Vec<EvalPoint> = (0..5u64).map(|i| pt(i * 33, dir_a)).collect();
        points.push(pt(134, dir_b));
        points.extend((0..5u64).map(|i| pt(165 + i * 33, dir_b)));
        let input = EvalInput {
            windows: vec![win(0, 400, Point2::new(960.0, 540.0), points)],
            processing: vec![],
        };
        let metrics = compute(&input, &MetricParams::default());
        assert_abs_diff_eq!(metrics.precision_rms_s2s_deg.unwrap(), 0.0, epsilon = 1e-12);
        assert!(metrics.precision_pooled_rms_s2s_deg.unwrap() > 1.0);
    }

    #[test]
    fn test_median_even_count_is_mean_of_middle() {
        assert_eq!(median(&mut [100.0, 200.0, 300.0, 1000.0]), Some(250.0));
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&mut []), None);
    }

    #[test]
    fn test_region_median_even_count_through_compute() {
        let points = [600.0, 620.0, 1300.0, 2000.0]
            .into_iter()
            .enumerate()
            .map(|(i, x)| ptpx(i as u64 * 10, Point2::new(x, 135.0)))
            .collect::<Vec<_>>();
        let input = EvalInput {
            windows: vec![win(0, 800, Point2::new(720.0, 135.0), points)],
            processing: vec![],
        };
        let metrics = compute(&input, &MetricParams::default());
        let r3 = metrics
            .regions
            .iter()
            .find(|r| r.cols == 3 && r.rows == 3)
            .unwrap();
        assert_eq!(r3.hits, 1);
        assert_eq!(r3.windows, 1);
        let r4 = metrics
            .regions
            .iter()
            .find(|r| r.cols == 4 && r.rows == 4)
            .unwrap();
        assert_eq!(r4.hits, 0);
        assert_eq!(r4.windows, 1);
    }

    #[test]
    fn test_percentile_type7_known_values() {
        let values: Vec<f64> = (1..=20).map(f64::from).collect();
        assert_eq!(percentile(&values, 50.0), Some(10.5));
        assert_eq!(percentile(&values, 95.0), Some(19.05));
        assert_eq!(percentile(&values, 100.0), Some(20.0));
        assert_eq!(percentile(&values, 0.0), Some(1.0));
        assert_eq!(percentile(&[], 50.0), None);
        assert_relative_eq!(
            percentile(&[1.0, 2.0, 3.0, 4.0], 95.0).unwrap(),
            3.85,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_offscreen_median_is_a_miss() {
        let points = (0..3)
            .map(|i| ptpx(i as u64 * 10, Point2::new(-5.0, 135.0)))
            .collect();
        let input = EvalInput {
            windows: vec![win(0, 800, Point2::new(240.0, 135.0), points)],
            processing: vec![],
        };
        let metrics = compute(&input, &MetricParams::default());
        let r3 = metrics
            .regions
            .iter()
            .find(|r| r.cols == 3 && r.rows == 3)
            .unwrap();
        assert_eq!(r3.hits, 0);
        assert_eq!(r3.windows, 1);
    }

    #[test]
    fn test_metric_params_rejects_zero_grid() {
        let params = MetricParams {
            grids: vec![[0, 3]],
            ..MetricParams::default()
        };
        assert!(params.validate().is_err());
        let params = MetricParams {
            dropout_bin: Duration::ZERO,
            ..MetricParams::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_region_hit_rate_3x3_vs_4x4() {
        let targets = [240.0, 720.0, 1200.0, 1680.0];
        let medians = [240.0, 720.0, 1300.0, 1680.0];
        let windows = targets
            .iter()
            .zip(medians.iter())
            .enumerate()
            .map(|(i, (&t, &m))| {
                win(
                    i as u64 * 1000,
                    i as u64 * 1000 + 800,
                    Point2::new(t, 135.0),
                    vec![ptpx(i as u64 * 1000, Point2::new(m, 135.0))],
                )
            })
            .collect();
        let input = EvalInput {
            windows,
            processing: vec![],
        };
        let metrics = compute(&input, &MetricParams::default());
        let r3 = metrics
            .regions
            .iter()
            .find(|r| r.cols == 3 && r.rows == 3)
            .unwrap();
        assert_eq!(r3.hits, 3);
        assert_eq!(r3.windows, 4);
        assert_eq!(r3.hit_rate, Some(0.75));
        let r4 = metrics
            .regions
            .iter()
            .find(|r| r.cols == 4 && r.rows == 4)
            .unwrap();
        assert_eq!(r4.hits, 4);
    }

    #[test]
    fn test_target_on_cell_boundary_is_excluded() {
        let input = EvalInput {
            windows: vec![win(
                0,
                800,
                Point2::new(960.0, 540.0),
                vec![ptpx(0, Point2::new(960.0, 540.0))],
            )],
            processing: vec![],
        };
        let metrics = compute(&input, &MetricParams::default());
        let r4 = metrics
            .regions
            .iter()
            .find(|r| r.cols == 4 && r.rows == 4)
            .unwrap();
        assert_eq!(r4.excluded, 1);
        assert_eq!(r4.windows, 0);
        let r3 = metrics
            .regions
            .iter()
            .find(|r| r.cols == 3 && r.rows == 3)
            .unwrap();
        assert_eq!(r3.excluded, 0);
        assert_eq!(r3.windows, 1);
    }

    #[test]
    fn test_region_uses_median_not_mean() {
        let mut points: Vec<EvalPoint> = (0..4)
            .map(|i| ptpx(i as u64 * 10, Point2::new(240.0, 135.0)))
            .collect();
        points.push(ptpx(40, Point2::new(5000.0, 5000.0)));
        let input = EvalInput {
            windows: vec![win(0, 800, Point2::new(240.0, 135.0), points)],
            processing: vec![],
        };
        let metrics = compute(&input, &MetricParams::default());
        let r3 = metrics
            .regions
            .iter()
            .find(|r| r.cols == 3 && r.rows == 3)
            .unwrap();
        assert_eq!(r3.hits, 1);
    }

    #[test]
    fn test_dropout_counts_empty_full_bins() {
        let params = MetricParams {
            dropout_bin: Duration::from_millis(100),
            ..MetricParams::default()
        };
        let points = (0..7)
            .map(|i| ptpx(50 + i * 100, Point2::new(960.0, 540.0)))
            .collect();
        let input = EvalInput {
            windows: vec![win(0, 1050, Point2::new(960.0, 540.0), points)],
            processing: vec![],
        };
        let metrics = compute(&input, &params);
        assert_relative_eq!(metrics.dropout_rate.unwrap(), 0.3, epsilon = 1e-12);
    }

    #[test]
    fn test_dropout_zero_for_7_5hz_output_with_default_bin() {
        let points = (0..8)
            .map(|j| EvalPoint {
                timestamp: Timestamp(Duration::from_nanos(j * 133_333_333)),
                eye_mm: eye(),
                gaze_mm: Point2::new(155.0, 85.0),
                gaze_px_logical: Point2::new(960.0, 540.0),
            })
            .collect();
        let input = EvalInput {
            windows: vec![win(0, 1000, Point2::new(960.0, 540.0), points)],
            processing: vec![],
        };
        let metrics = compute(&input, &MetricParams::default());
        assert_eq!(metrics.dropout_rate, Some(0.0));
        assert_relative_eq!(metrics.output_rate_hz.unwrap(), 8.0, epsilon = 1e-9);
    }

    #[test]
    fn test_processing_summary_in_ms() {
        let processing: Vec<Duration> = (1..=20).map(Duration::from_millis).collect();
        let input = EvalInput {
            windows: vec![],
            processing,
        };
        let metrics = compute(&input, &MetricParams::default());
        assert_relative_eq!(
            metrics.processing_ms.as_ref().unwrap().p50,
            10.5,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            metrics.processing_ms.as_ref().unwrap().p95,
            19.05,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_empty_input_yields_none_not_nan() {
        let metrics = compute(&EvalInput::default(), &MetricParams::default());
        assert_eq!(metrics.angular_error_deg, None);
        assert_eq!(metrics.px_error_logical, None);
        assert_eq!(metrics.accuracy_deg, None);
        assert_eq!(metrics.precision_rms_s2s_deg, None);
        assert_eq!(metrics.precision_pooled_rms_s2s_deg, None);
        assert_eq!(metrics.processing_ms, None);
        assert_eq!(metrics.dropout_rate, None);
        assert_eq!(metrics.output_rate_hz, None);
        assert_eq!(metrics.regions.len(), 2);
        for r in &metrics.regions {
            assert_eq!(r.hit_rate, None);
        }
        let json = serde_json::to_string(&metrics).unwrap();
        assert!(!json.contains("NaN"));
    }

    #[test]
    fn test_concat_pools_windows_in_order() {
        let a = EvalInput {
            windows: vec![win(0, 800, Point2::new(960.0, 540.0), vec![])],
            processing: vec![Duration::from_millis(1)],
        };
        let b = EvalInput {
            windows: vec![win(800, 1600, Point2::new(240.0, 135.0), vec![])],
            processing: vec![Duration::from_millis(2)],
        };
        let c = EvalInput::concat([&a, &b]);
        assert_eq!(c.windows.len(), 2);
        assert_eq!(
            c.windows[0].target_px_logical,
            a.windows[0].target_px_logical
        );
        assert_eq!(
            c.windows[1].target_px_logical,
            b.windows[0].target_px_logical
        );
        assert_eq!(
            c.processing,
            vec![Duration::from_millis(1), Duration::from_millis(2)]
        );
    }

    #[test]
    fn test_metric_params_deserialize_from_toml() {
        let toml_str = "grids = [[3, 3], [4, 4]]\nboundary_margin_px = 20.0\ndropout_bin = 200\n";
        let parsed: MetricParams = toml::from_str(toml_str).unwrap();
        assert_eq!(parsed, MetricParams::default());
        let bad = "foo = 1\n";
        assert!(toml::from_str::<MetricParams>(bad).is_err());
    }

    proptest! {
        #[test]
        fn test_angle_is_symmetric_and_nonnegative(
            ez in -900.0f64..-200.0,
            ax in 0.0f64..310.0, ay in 0.0f64..170.0,
            bx in 0.0f64..310.0, by in 0.0f64..170.0,
        ) {
            let eye = Point3::new(155.0, 85.0, ez);
            let a = Point2::new(ax, ay);
            let b = Point2::new(bx, by);
            let ab = angle_deg(&eye, &a, &b);
            let ba = angle_deg(&eye, &b, &a);
            prop_assert!((ab - ba).abs() < 1e-9);
            prop_assert!(ab >= 0.0);
        }

        #[test]
        fn test_percentile_monotonic(mut values in prop::collection::vec(-1000.0f64..1000.0, 1..50)) {
            values.sort_by(f64::total_cmp);
            let p50 = percentile(&values, 50.0).unwrap();
            let p95 = percentile(&values, 95.0).unwrap();
            let max = *values.last().unwrap();
            prop_assert!(p50 <= p95);
            prop_assert!(p95 <= max);
        }
    }
}
