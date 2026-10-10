//! Pin-point mode: a dot plus uncertainty ellipse for the latest gaze point.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use eye_core::GazePoint;
use eye_core::log::field;
use nalgebra::{Point2, Vector2};

use crate::canvas::{Canvas, Rgba};
use crate::ellipse::{K95, confidence_ellipse, cov_mm_to_logical_px};
use crate::fade::{FADE_START, fade};
use crate::scene::{PresentedAt, Scene, Schedule};
use crate::stats::PresentStats;

/// The scene draws the trajectory `max_lag` behind the newest sample, as a cubic Hermite
/// between the two newest samples with backward-difference tangents. This adds no filtering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interpolation {
    /// `Duration::ZERO` draws samples as they arrive, with no interpolation.
    pub max_lag: Duration,
}

const MIN_LAG: Duration = Duration::from_millis(16);

impl Interpolation {
    pub fn from_millis(ms: u64) -> Self {
        Self {
            max_lag: Duration::from_millis(ms),
        }
    }

    fn disabled(&self) -> bool {
        self.max_lag.is_zero()
    }
}

impl Default for Interpolation {
    fn default() -> Self {
        Self::from_millis(80)
    }
}

/// Thresholds for hiding the point entirely rather than fading it. `margin_px` also sets
/// the hysteresis band: a point must cross back inside the canvas shrunk by `margin_px`
/// to be shown again after leaving it grown by the same amount.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HideRules {
    pub margin_px: f64,
    pub min_confidence: f64,
}

impl Default for HideRules {
    fn default() -> Self {
        Self {
            margin_px: 24.0,
            min_confidence: 0.02,
        }
    }
}

/// How the point communicates certainty and how it reacts to small moves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointStyle {
    /// Colour the dot by confidence (green, white, red) instead of the scene colour.
    pub certainty_colors: bool,
    /// Draw the 95 % uncertainty ellipse around the dot.
    pub show_ellipse: bool,
    /// Radius, in standard deviations of the sample's own covariance (Mahalanobis), of the
    /// zone around the dot. Inside it the dot glides onto the samples with time constant
    /// `settle`; outside it the dot is dragged so it stays on the zone's edge. `0.0`
    /// follows every sample directly.
    pub hold_sigmas: f64,
    pub settle: Duration,
}

impl Default for PointStyle {
    fn default() -> Self {
        Self {
            certainty_colors: true,
            show_ellipse: false,
            hold_sigmas: 1.0,
            settle: Duration::from_millis(300),
        }
    }
}

/// Confidence at or above which the dot is green, and below which it is red.
pub const CERTAIN_CONFIDENCE: f64 = 0.3;
pub const UNCERTAIN_CONFIDENCE: f64 = 0.1;

fn certainty_rgb(confidence: f64) -> [u8; 3] {
    if confidence >= CERTAIN_CONFIDENCE {
        [60, 220, 90]
    } else if confidence < UNCERTAIN_CONFIDENCE {
        [235, 60, 60]
    } else {
        [240, 240, 240]
    }
}

fn rgba_of(color: [u8; 3], alpha: f64) -> Rgba {
    Rgba {
        r: color[0],
        g: color[1],
        b: color[2],
        a: alpha.round().clamp(0.0, 255.0) as u8,
    }
}

/// `was_hidden` carries the hysteresis: inside the band between the grown and shrunk
/// canvas, the previous state wins so a point hovering near the edge does not flicker.
fn offscreen_hidden(
    center: Point2<f64>,
    size: (f64, f64),
    margin_px: f64,
    was_hidden: bool,
) -> bool {
    let (w, h) = size;
    let outside_grown = center.x < -margin_px
        || center.x >= w + margin_px
        || center.y < -margin_px
        || center.y >= h + margin_px;
    let inside_shrunk = center.x >= margin_px
        && center.x < w - margin_px
        && center.y >= margin_px
        && center.y < h - margin_px;
    if outside_grown {
        true
    } else if inside_shrunk {
        false
    } else {
        was_hidden
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct EasedEllipse {
    axes: (f64, f64),
    angle: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Eased {
    center: Point2<f64>,
    ellipse: Option<EasedEllipse>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Knot {
    at: Instant,
    pose: Eased,
    /// Backward-difference tangent, in logical px per second.
    tangent: Vector2<f64>,
}

#[derive(Debug)]
pub struct PointScene {
    px_per_mm: Vector2<f64>,
    color: [u8; 3],
    interp: Interpolation,
    latest: Option<(GazePoint, Instant)>,
    /// At most two: the segment `step` interpolates across. Cleared by `render` on
    /// fade-out and on off-screen hide, so the next sample teleports.
    knots: VecDeque<Knot>,
    lag: Duration,
    drawn: Option<Eased>,
    hide: HideRules,
    style: PointStyle,
    hidden_offscreen: bool,
    present: PresentStats,
    /// `timestamp.as_nanos()` of the latest sample whose first on-target commit has not
    /// been handed out as a presentation mark yet. A newer sample replaces an unmarked
    /// older one: the dot never reached it, nothing to measure. `render` takes this (sets
    /// it to `None`) the moment the drawn pose catches up to the newest knot, so a sample
    /// is marked at most once even if `render` runs again before `on_presented` arrives.
    pending_mark: Option<u64>,
    /// The mark for the current commit, or `None` if this commit carries no mark. Reset
    /// to `None` at the top of every `render` and set at most once per sample, from
    /// `pending_mark`, when the drawn pose catches up to the newest knot.
    mark_ready: Option<u64>,
    /// `now - lag >= k1.at`, reevaluated on every sample so a knot whose lag is already
    /// zero starts `true`. `step` edge-triggers on this to return `true` exactly once on
    /// the frame it flips, forcing a `render` even when nothing else is moving.
    reached_rendered: bool,
}

impl PointScene {
    pub fn new(px_per_mm: Vector2<f64>, color: [u8; 3]) -> Self {
        Self::with_interpolation(px_per_mm, color, Interpolation::default())
    }

    pub fn with_interpolation(
        px_per_mm: Vector2<f64>,
        color: [u8; 3],
        interp: Interpolation,
    ) -> Self {
        Self {
            px_per_mm,
            color,
            interp,
            latest: None,
            knots: VecDeque::new(),
            lag: Duration::ZERO,
            drawn: None,
            hide: HideRules::default(),
            style: PointStyle::default(),
            hidden_offscreen: false,
            present: PresentStats::default(),
            pending_mark: None,
            mark_ready: None,
            reached_rendered: false,
        }
    }

    pub fn with_hide_rules(mut self, hide: HideRules) -> Self {
        self.hide = hide;
        self
    }

    pub fn with_style(mut self, style: PointStyle) -> Self {
        self.style = style;
        self
    }

    pub fn with_present_stats(mut self, stats: PresentStats) -> Self {
        self.present = stats;
        self
    }

    fn rgba(&self, alpha: f64) -> Rgba {
        rgba_of(self.color, alpha)
    }

    /// Where the dot's target goes for a new sample, given the current target and the time
    /// since the previous sample; `None` means jump to the sample.
    fn zoned_center(&self, msg: &GazePoint, dt: Duration) -> Option<Point2<f64>> {
        let current = self.knots.back()?.pose.center;
        if self.style.hold_sigmas <= 0.0 {
            return None;
        }
        let cov_px = cov_mm_to_logical_px(&msg.cov_mm, &self.px_per_mm);
        let inv = cov_px.try_inverse()?;
        let d = msg.px_logical - current;
        let m = (d.transpose() * inv * d)[(0, 0)].sqrt();
        if m > self.style.hold_sigmas {
            return Some(msg.px_logical - d * (self.style.hold_sigmas / m));
        }
        let settle = self.style.settle.as_secs_f64();
        let k = if settle > 0.0 {
            1.0 - (-dt.as_secs_f64() / settle).exp()
        } else {
            1.0
        };
        Some(current + d * k)
    }

    /// `now - lag >= k1.at`: the drawn pose has caught up to the newest knot.
    fn reached_latest(&self, now: Instant) -> bool {
        self.knots.back().is_some_and(|k1| {
            now.checked_sub(self.lag)
                .is_some_and(|target_time| target_time >= k1.at)
        })
    }

    fn target_for(&self, p: &GazePoint) -> Eased {
        let cov_px = cov_mm_to_logical_px(&p.cov_mm, &self.px_per_mm);
        let ellipse =
            confidence_ellipse(p.px_logical, &cov_px, K95, f64::MAX).map(|e| EasedEllipse {
                axes: e.semi_axes,
                angle: e.angle,
            });
        Eased {
            center: p.px_logical,
            ellipse,
        }
    }
}

/// Cubic Hermite on `[k0.at, k1.at]` at `tau` (the caller clamps `tau` into that interval).
/// Ellipse axes and angle interpolate linearly, not via Hermite.
fn hermite(k0: &Knot, k1: &Knot, tau: Instant) -> Eased {
    let span = k1.at.saturating_duration_since(k0.at).as_secs_f64();
    let h = if span > 0.0 {
        tau.saturating_duration_since(k0.at).as_secs_f64() / span
    } else {
        1.0
    };
    let h00 = 2.0 * h.powi(3) - 3.0 * h.powi(2) + 1.0;
    let h10 = h.powi(3) - 2.0 * h.powi(2) + h;
    let h01 = -2.0 * h.powi(3) + 3.0 * h.powi(2);
    let h11 = h.powi(3) - h.powi(2);
    let center = k0.pose.center.coords * h00
        + k0.tangent * span * h10
        + k1.pose.center.coords * h01
        + k1.tangent * span * h11;
    let ellipse = match (k0.pose.ellipse, k1.pose.ellipse) {
        (Some(a), Some(b)) => Some(EasedEllipse {
            axes: (
                a.axes.0 + h * (b.axes.0 - a.axes.0),
                a.axes.1 + h * (b.axes.1 - a.axes.1),
            ),
            angle: lerp_angle(a.angle, b.angle, h),
        }),
        _ => k1.pose.ellipse,
    };
    Eased {
        center: Point2::from(center),
        ellipse,
    }
}

/// An ellipse looks the same after a half turn, so the shortest path wraps on `PI`
/// rather than `TAU`.
fn lerp_angle(a: f64, b: f64, alpha: f64) -> f64 {
    use std::f64::consts::{FRAC_PI_2, PI};
    let diff = (b - a + FRAC_PI_2).rem_euclid(PI) - FRAC_PI_2;
    a + alpha * diff
}

impl Scene for PointScene {
    type Msg = GazePoint;

    fn on_msg(&mut self, msg: GazePoint, now: Instant) {
        let mut pose = self.target_for(&msg);
        let dt = self
            .latest
            .as_ref()
            .map_or(Duration::ZERO, |(_, received)| {
                now.saturating_duration_since(*received)
            });
        if let Some(center) = self.zoned_center(&msg, dt) {
            pose.center = center;
        }
        let first = self.knots.is_empty();
        let tangent = match self.knots.back() {
            Some(prev) if now > prev.at => {
                (pose.center - prev.pose.center) / now.duration_since(prev.at).as_secs_f64()
            }
            _ => Vector2::zeros(),
        };
        self.knots.push_back(Knot {
            at: now,
            pose,
            tangent,
        });
        if self.interp.disabled() {
            while self.knots.len() > 1 {
                self.knots.pop_front();
            }
            self.lag = Duration::ZERO;
            self.drawn = Some(pose);
        } else {
            while self.knots.len() > 2 {
                self.knots.pop_front();
            }
            if first {
                self.lag = Duration::ZERO;
                self.drawn = Some(pose);
            } else {
                let max_lag = self.interp.max_lag;
                self.lag = dt.min(max_lag).max(MIN_LAG.min(max_lag));
            }
        }
        self.pending_mark = Some(msg.timestamp.as_nanos());
        self.latest = Some((msg, now));
        self.reached_rendered = self.reached_latest(now);
    }

    fn step(&mut self, now: Instant) -> bool {
        let moving = if self.knots.len() < 2 {
            if let Some(k) = self.knots.back() {
                self.drawn = Some(k.pose);
            }
            false
        } else {
            let k0 = &self.knots[0];
            let k1 = &self.knots[1];
            match now.checked_sub(self.lag) {
                Some(target_time) if target_time <= k1.at => {
                    let tau = target_time.clamp(k0.at, k1.at);
                    self.drawn = Some(hermite(k0, k1, tau));
                    true
                }
                _ => {
                    self.drawn = Some(k1.pose);
                    false
                }
            }
        };
        let reached = self.reached_latest(now);
        let just_reached = reached && !self.reached_rendered;
        self.reached_rendered = reached;
        moving || just_reached
    }

    fn render(&mut self, canvas: &mut Canvas<'_>, now: Instant) -> Schedule {
        self.mark_ready = None;
        let Some((p, received)) = &self.latest else {
            return Schedule::Idle;
        };
        let age = now.saturating_duration_since(*received);
        let age_us = age.as_micros() as u64;
        let f = fade(age);
        if f == 0.0 {
            tracing::debug!(
                age_us,
                { field::REASON } = "faded",
                "stale gaze point hidden"
            );
            self.latest = None;
            self.knots.clear();
            self.drawn = None;
            return Schedule::Idle;
        }
        let (w, h) = canvas.logical_size();
        let center = p.px_logical;
        let confidence = p.confidence;
        self.hidden_offscreen = offscreen_hidden(
            center,
            (f64::from(w), f64::from(h)),
            self.hide.margin_px,
            self.hidden_offscreen,
        );
        if self.hidden_offscreen {
            self.knots.clear();
            self.drawn = None;
            return Schedule::Idle;
        }
        if confidence < self.hide.min_confidence {
            return Schedule::Idle;
        }
        let Some(drawn) = self.drawn else {
            return Schedule::Idle;
        };
        if self.reached_latest(now) {
            self.mark_ready = self.pending_mark.take();
        }
        let max_axis = f64::from(w).hypot(f64::from(h));
        if let Some(ellipse) = drawn.ellipse.filter(|_| self.style.show_ellipse) {
            let axes = (
                ellipse.axes.0.clamp(0.5, max_axis),
                ellipse.axes.1.clamp(0.5, max_axis),
            );
            canvas.fill_ellipse(drawn.center, axes, ellipse.angle, self.rgba(40.0 * f));
            canvas.stroke_ellipse(drawn.center, axes, ellipse.angle, 2.0, self.rgba(160.0 * f));
        }
        let conf = p.confidence.clamp(0.0, 1.0);
        let dot = if self.style.certainty_colors {
            rgba_of(certainty_rgb(conf), 230.0 * f)
        } else {
            self.rgba(230.0 * f * (0.4 + 0.6 * conf))
        };
        canvas.fill_circle(drawn.center, 6.0, dot);
        tracing::trace!(
            x = drawn.center.x,
            y = drawn.center.y,
            semi_major = drawn.ellipse.map_or(0.0, |e| e.axes.0),
            semi_minor = drawn.ellipse.map_or(0.0, |e| e.axes.1),
            angle_rad = drawn.ellipse.map_or(0.0, |e| e.angle),
            confidence = p.confidence,
            fade = f,
            age_us,
            { field::TS_NS } = p.timestamp.as_nanos(),
            "gaze point drawn"
        );
        if age < FADE_START {
            Schedule::At(*received + FADE_START)
        } else {
            Schedule::NextFrame
        }
    }

    fn presentation_mark(&self) -> Option<u64> {
        self.mark_ready
    }

    fn on_presented(&mut self, mark: u64, at: PresentedAt, _now: Instant) -> Schedule {
        let presented = match at {
            PresentedAt::Presentation(t) | PresentedAt::Commit(t) => t,
        };
        if let Some(d) = presented.0.checked_sub(Duration::from_nanos(mark)) {
            self.present.record(d);
            tracing::debug!(
                capture_to_present_us = d.as_micros() as u64,
                clock = at.clock_name(),
                { field::TS_NS } = mark,
                "gaze point presented"
            );
        }
        Schedule::Idle
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use eye_core::OutputId;
    use eye_log::Value;
    use eye_log::testing::capture_logs;
    use nalgebra::Matrix2;

    use super::*;
    use crate::canvas::bgra;
    use crate::ellipse::K95;

    fn point_at(now: Instant, cov_mm: Matrix2<f64>, confidence: f64) -> (GazePoint, Instant) {
        (
            GazePoint {
                timestamp: eye_core::Timestamp::from_nanos(0),
                output: OutputId::from("eDP-1"),
                mm: Point2::new(0.0, 0.0),
                px_physical: Point2::new(0.0, 0.0),
                px_logical: Point2::new(100.5, 100.5),
                cov_mm,
                confidence,
            },
            now,
        )
    }

    const LEGACY: PointStyle = PointStyle {
        certainty_colors: false,
        show_ellipse: true,
        hold_sigmas: 0.0,
        settle: Duration::ZERO,
    };

    fn point_px(now: Instant, x: f64, y: f64, cov: f64, confidence: f64) -> (GazePoint, Instant) {
        let (mut p, t) = point_at(now, Matrix2::identity() * cov, confidence);
        p.px_logical = Point2::new(x, y);
        (p, t)
    }

    fn zoned_scene() -> PointScene {
        PointScene::with_interpolation(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Interpolation::from_millis(0),
        )
    }

    #[test]
    fn test_small_move_within_zone_does_not_jump() {
        let now = Instant::now();
        let mut scene = zoned_scene();
        let (a, t) = point_px(now, 100.0, 100.0, 100.0, 0.5);
        scene.on_msg(a, t);
        let (b, t) = point_px(now + Duration::from_millis(33), 108.0, 100.0, 100.0, 0.5);
        scene.on_msg(b, t);
        scene.step(t);
        let x = scene.drawn.unwrap().center.x;
        assert!(x > 100.0 && x < 101.0, "x {x}");
    }

    #[test]
    fn test_dot_settles_onto_fixation_within_zone() {
        let now = Instant::now();
        let mut scene = zoned_scene();
        let (a, t) = point_px(now, 100.0, 100.0, 100.0, 0.5);
        scene.on_msg(a, t);
        for k in 1..=30u64 {
            let (b, t) = point_px(
                now + Duration::from_millis(33 * k),
                108.0,
                100.0,
                100.0,
                0.5,
            );
            scene.on_msg(b, t);
        }
        scene.step(now + Duration::from_millis(990));
        let x = scene.drawn.unwrap().center.x;
        assert_abs_diff_eq!(x, 108.0, epsilon = 0.5);
    }

    #[test]
    fn test_large_move_drags_dot_to_zone_edge() {
        let now = Instant::now();
        let mut scene = zoned_scene();
        let (a, t) = point_px(now, 100.0, 100.0, 100.0, 0.5);
        scene.on_msg(a, t);
        let (b, t) = point_px(now + Duration::from_millis(33), 130.0, 100.0, 100.0, 0.5);
        scene.on_msg(b, t);
        scene.step(t);
        assert_abs_diff_eq!(scene.drawn.unwrap().center.x, 120.0, epsilon = 1e-9);
    }

    #[test]
    fn test_dot_colour_follows_confidence() {
        for (confidence, rgb) in [
            (0.5, [60u8, 220, 90]),
            (0.2, [240, 240, 240]),
            (0.05, [235, 60, 60]),
        ] {
            let mut buf = vec![0u8; 200 * 200 * 4];
            let mut canvas = new_canvas(&mut buf);
            let now = Instant::now();
            let mut scene =
                PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]).with_hide_rules(HideRules {
                    margin_px: 24.0,
                    min_confidence: 0.0,
                });
            let (p, received) = point_at(now, Matrix2::new(100.0, 0.0, 0.0, 25.0), confidence);
            scene.on_msg(p, received);
            scene.render(&mut canvas, received);
            let px = bgra(&buf, 200, 100, 100);
            let alpha = f64::from(px[3]) / 255.0;
            for (channel, want) in [(px[2], rgb[0]), (px[1], rgb[1]), (px[0], rgb[2])] {
                assert_abs_diff_eq!(f64::from(channel), f64::from(want) * alpha, epsilon = 3.0);
            }
            assert_alpha(&buf, 122, 100, 0.0, 0.5);
        }
    }

    #[test]
    fn test_calibrated_residual_floor_keeps_point_visible() {
        let mut buf = vec![0u8; 200 * 200 * 4];
        let mut canvas = new_canvas(&mut buf);
        let now = Instant::now();
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]);
        let (p, received) = point_at(now, Matrix2::identity() * 55.5f64.powi(2), 0.062);
        scene.on_msg(p, received);
        scene.render(&mut canvas, received);
        let px = bgra(&buf, 200, 100, 100);
        let alpha = f64::from(px[3]) / 255.0;
        assert!(alpha > 0.0, "point should be visible, alpha {alpha}");
        for (channel, want) in [(px[2], 235u8), (px[1], 60), (px[0], 60)] {
            assert_abs_diff_eq!(f64::from(channel), f64::from(want) * alpha, epsilon = 3.0);
        }
        assert_eq!(scene.presentation_mark(), Some(0));
    }

    #[test]
    fn test_degenerate_ray_is_hidden() {
        let mut buf = vec![0u8; 200 * 200 * 4];
        let mut canvas = new_canvas(&mut buf);
        let now = Instant::now();
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]);
        let (p, received) = point_at(now, Matrix2::identity() * 105.0f64.powi(2), 0.005);
        scene.on_msg(p, received);
        scene.render(&mut canvas, received);
        assert_alpha(&buf, 100, 100, 0.0, 0.5);
    }

    fn new_canvas(buf: &mut [u8]) -> Canvas<'_> {
        Canvas::new(buf, (200, 200), 1).expect("size")
    }

    fn assert_alpha(buf: &[u8], x: u32, y: u32, expected: f64, tolerance: f64) {
        let px = bgra(buf, 200, x, y);
        assert_abs_diff_eq!(f64::from(px[3]), expected, epsilon = tolerance);
    }

    #[test]
    fn test_point_scene_draws_dot_and_ellipse() {
        let mut buf = vec![0u8; 200 * 200 * 4];
        let mut canvas = new_canvas(&mut buf);
        let now = Instant::now();
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]).with_style(LEGACY);
        let (p, received) = point_at(now, Matrix2::new(100.0, 0.0, 0.0, 25.0), 1.0);
        scene.on_msg(p, received);
        scene.render(&mut canvas, received);

        assert_alpha(&buf, 100, 100, 234.0, 3.0);
        let px = bgra(&buf, 200, 100, 100);
        assert_abs_diff_eq!(f64::from(px[2]), 234.0, epsilon = 3.0);
        assert_alpha(&buf, 122, 100, 40.0, 3.0);
        assert_alpha(&buf, 128, 100, 0.0, 0.5);
        assert_alpha(&buf, 100, 110, 40.0, 3.0);
        assert_alpha(&buf, 100, 116, 0.0, 0.5);
    }

    #[test]
    fn test_point_scene_without_point_draws_nothing_and_idles() {
        let mut buf = vec![0u8; 200 * 200 * 4];
        let mut canvas = new_canvas(&mut buf);
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]);
        let schedule = scene.render(&mut canvas, Instant::now());
        assert!(buf.iter().all(|&b| b == 0));
        assert_eq!(schedule, Schedule::Idle);
    }

    #[test]
    fn test_point_scene_schedules_fade_start() {
        let mut buf = vec![0u8; 200 * 200 * 4];
        let mut canvas = new_canvas(&mut buf);
        let now = Instant::now();
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]);
        let (p, received) = point_at(now, Matrix2::zeros(), 1.0);
        scene.on_msg(p, received);
        let schedule = scene.render(&mut canvas, received);
        assert_eq!(
            schedule,
            Schedule::At(received + std::time::Duration::from_millis(300))
        );
    }

    #[test]
    fn test_point_scene_fades_then_hides() {
        let mut buf = vec![0u8; 200 * 200 * 4];
        let now = Instant::now();
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]).with_style(LEGACY);
        let (p, received) = point_at(now, Matrix2::new(100.0, 0.0, 0.0, 25.0), 1.0);
        scene.on_msg(p, received);

        let t1 = received + std::time::Duration::from_millis(550);
        let schedule1 = {
            let mut canvas = new_canvas(&mut buf);
            scene.render(&mut canvas, t1)
        };
        assert_alpha(&buf, 100, 100, 126.0, 3.0);
        assert_eq!(schedule1, Schedule::NextFrame);

        buf.iter_mut().for_each(|b| *b = 0);
        let t2 = received + std::time::Duration::from_millis(900);
        let schedule2 = {
            let mut canvas = new_canvas(&mut buf);
            scene.render(&mut canvas, t2)
        };
        assert!(buf.iter().all(|&b| b == 0));
        assert_eq!(schedule2, Schedule::Idle);
    }

    #[test]
    fn test_point_scene_low_confidence_is_fainter() {
        let mut buf = vec![0u8; 200 * 200 * 4];
        let mut canvas = new_canvas(&mut buf);
        let now = Instant::now();
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]).with_style(LEGACY);
        let (p, received) = point_at(now, Matrix2::new(100.0, 0.0, 0.0, 25.0), 0.25);
        scene.on_msg(p, received);
        scene.render(&mut canvas, received);
        assert_alpha(&buf, 100, 100, 147.0, 3.0);
    }

    fn scene_no_zone(max_lag_ms: u64) -> PointScene {
        PointScene::with_interpolation(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Interpolation::from_millis(max_lag_ms),
        )
        .with_style(PointStyle {
            hold_sigmas: 0.0,
            ..PointStyle::default()
        })
    }

    fn feed(scene: &mut PointScene, now: Instant, samples: &[(u64, f64)]) {
        for &(ms, x) in samples {
            let (p, t) = point_px(now + Duration::from_millis(ms), x, 0.0, 0.0, 1.0);
            scene.on_msg(p, t);
        }
    }

    #[test]
    fn test_interpolation_passes_through_samples() {
        let now = Instant::now();
        let mut scene = scene_no_zone(80);
        feed(&mut scene, now, &[(0, 0.0), (33, 10.0)]);
        scene.step(now + Duration::from_millis(66));
        assert_abs_diff_eq!(scene.drawn.unwrap().center.x, 10.0, epsilon = 1e-9);

        feed(&mut scene, now, &[(66, 20.0)]);
        scene.step(now + Duration::from_millis(99));
        assert_abs_diff_eq!(scene.drawn.unwrap().center.x, 20.0, epsilon = 1e-9);
    }

    #[test]
    fn test_interpolation_velocity_is_continuous_across_samples() {
        let now = Instant::now();
        let mut scene = scene_no_zone(80);
        feed(&mut scene, now, &[(0, 0.0), (33, 10.0), (66, 30.0)]);

        let (k0, k1) = (scene.knots[0], scene.knots[1]);
        let before1 = hermite(&k0, &k1, now + Duration::from_millis(65));
        let before2 = hermite(&k0, &k1, now + Duration::from_millis(66));
        let v_before = before2.center.x - before1.center.x;

        feed(&mut scene, now, &[(99, 60.0)]);
        let (k0, k1) = (scene.knots[0], scene.knots[1]);
        let after1 = hermite(&k0, &k1, now + Duration::from_millis(66));
        let after2 = hermite(&k0, &k1, now + Duration::from_millis(67));
        let v_after = after2.center.x - after1.center.x;

        let rel_diff = (v_after - v_before).abs() / v_before.abs();
        assert!(
            rel_diff < 0.05,
            "v_before {v_before} v_after {v_after} rel_diff {rel_diff}"
        );
    }

    #[test]
    fn test_interpolation_holds_at_latest_sample_when_late() {
        let now = Instant::now();
        let mut scene = scene_no_zone(80);
        feed(&mut scene, now, &[(0, 0.0), (33, 10.0)]);

        let redraw = scene.step(now + Duration::from_millis(33 + 100));
        assert!(
            redraw,
            "the dot reaches the latest sample on this call for the first time and must \
             render once to catch it, rather than waiting for a later tick"
        );
        assert_abs_diff_eq!(scene.drawn.unwrap().center.x, 10.0, epsilon = 1e-9);

        let redraw_again = scene.step(now + Duration::from_millis(33 + 100 + 16));
        assert!(
            !redraw_again,
            "already caught; nothing left to render while idle"
        );
    }

    #[test]
    fn test_lag_is_clamped_to_max_lag() {
        let now = Instant::now();
        let mut scene = scene_no_zone(80);
        feed(&mut scene, now, &[(0, 0.0), (133, 10.0)]);

        let just_before = scene.step(now + Duration::from_millis(133 + 79));
        assert!(just_before);
        assert!(scene.drawn.unwrap().center.x < 10.0);

        scene.step(now + Duration::from_millis(133 + 80));
        assert_abs_diff_eq!(scene.drawn.unwrap().center.x, 10.0, epsilon = 1e-9);
    }

    #[test]
    fn test_interpolation_zero_draws_latest_immediately() {
        let now = Instant::now();
        let mut scene = scene_no_zone(0);
        feed(&mut scene, now, &[(0, 0.0), (33, 10.0)]);

        let redraw = scene.step(now + Duration::from_millis(33));
        assert!(!redraw);
        assert_abs_diff_eq!(scene.drawn.unwrap().center.x, 10.0, epsilon = 1e-9);
    }

    #[test]
    fn test_max_lag_below_16ms_does_not_panic() {
        let now = Instant::now();
        let mut scene = scene_no_zone(10);
        feed(&mut scene, now, &[(0, 0.0), (33, 10.0)]);

        scene.step(now + Duration::from_millis(33 + 10));
        assert_abs_diff_eq!(scene.drawn.unwrap().center.x, 10.0, epsilon = 1e-9);
    }

    #[test]
    fn test_first_sample_is_drawn_immediately() {
        let now = Instant::now();
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]);
        let (p, received) = point_at(now, Matrix2::zeros(), 1.0);
        scene.on_msg(p, received);
        assert_eq!(scene.drawn.unwrap().center, Point2::new(100.5, 100.5));
    }

    #[test]
    fn test_interpolation_ellipse_angle_takes_shortest_half_turn() {
        use std::f64::consts::FRAC_PI_2;

        let now = Instant::now();
        let a_angle = FRAC_PI_2 - 0.05;
        let b_angle = -FRAC_PI_2 + 0.05;
        let k0 = Knot {
            at: now,
            pose: Eased {
                center: Point2::new(0.0, 0.0),
                ellipse: Some(EasedEllipse {
                    axes: (30.0, 10.0),
                    angle: a_angle,
                }),
            },
            tangent: Vector2::zeros(),
        };
        let k1 = Knot {
            at: now + Duration::from_millis(100),
            pose: Eased {
                center: Point2::new(0.0, 0.0),
                ellipse: Some(EasedEllipse {
                    axes: (30.0, 10.0),
                    angle: b_angle,
                }),
            },
            tangent: Vector2::zeros(),
        };

        let eased = hermite(&k0, &k1, now + Duration::from_millis(50));
        let new_angle = eased.ellipse.unwrap().angle;
        assert_abs_diff_eq!(new_angle - a_angle, 0.05, epsilon = 0.005);
    }

    fn point_at_xy(
        now: Instant,
        px_logical: Point2<f64>,
        cov_mm: Matrix2<f64>,
        confidence: f64,
    ) -> (GazePoint, Instant) {
        (
            GazePoint {
                timestamp: eye_core::Timestamp::from_nanos(0),
                output: OutputId::from("eDP-1"),
                mm: Point2::new(0.0, 0.0),
                px_physical: px_logical,
                px_logical,
                cov_mm,
                confidence,
            },
            now,
        )
    }

    #[test]
    fn test_offscreen_point_draws_nothing() {
        let mut buf = vec![0u8; 200 * 200 * 4];
        let mut canvas = new_canvas(&mut buf);
        let now = Instant::now();
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]);
        let (p, received) = point_at_xy(
            now,
            Point2::new(-100.0, 100.0),
            Matrix2::new(2000.0, 0.0, 0.0, 2000.0),
            1.0,
        );
        scene.on_msg(p, received);
        scene.render(&mut canvas, received);
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_low_confidence_point_draws_nothing() {
        let mut buf = vec![0u8; 200 * 200 * 4];
        let mut canvas = new_canvas(&mut buf);
        let now = Instant::now();
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]);
        let (p, received) = point_at_xy(
            now,
            Point2::new(100.0, 100.0),
            Matrix2::new(100.0, 0.0, 0.0, 25.0),
            0.01,
        );
        scene.on_msg(p, received);
        scene.render(&mut canvas, received);
        assert!(buf.iter().all(|&b| b == 0));

        buf.iter_mut().for_each(|b| *b = 0);
        let mut canvas = new_canvas(&mut buf);
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]);
        let (p, received) = point_at_xy(
            now,
            Point2::new(100.0, 100.0),
            Matrix2::new(100.0, 0.0, 0.0, 25.0),
            0.3,
        );
        scene.on_msg(p, received);
        scene.render(&mut canvas, received);
        assert!(!buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_edge_hysteresis_requires_reentry_by_margin() {
        let mut buf = vec![0u8; 200 * 200 * 4];
        let now = Instant::now();
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]);

        let (p, received) = point_at_xy(now, Point2::new(-100.0, 100.0), Matrix2::zeros(), 1.0);
        scene.on_msg(p, received);
        {
            let mut canvas = new_canvas(&mut buf);
            scene.render(&mut canvas, received);
        }
        assert!(buf.iter().all(|&b| b == 0), "far off-screen hides");

        for x in [-1.0, 10.0] {
            let (p, received) = point_at_xy(now, Point2::new(x, 100.0), Matrix2::zeros(), 1.0);
            scene.on_msg(p, received);
            buf.iter_mut().for_each(|b| *b = 0);
            let mut canvas = new_canvas(&mut buf);
            scene.render(&mut canvas, received);
            assert!(
                buf.iter().all(|&b| b == 0),
                "x = {x} is inside the margin dead zone, stays hidden"
            );
        }

        let (p, received) = point_at_xy(now, Point2::new(30.0, 100.0), Matrix2::zeros(), 1.0);
        scene.on_msg(p, received);
        buf.iter_mut().for_each(|b| *b = 0);
        let mut canvas = new_canvas(&mut buf);
        scene.render(&mut canvas, received);
        assert!(
            !buf.iter().all(|&b| b == 0),
            "x = 30 is past the margin, shown again"
        );
    }

    #[test]
    fn test_onscreen_point_with_clipped_ellipse_still_draws() {
        let mut buf = vec![0u8; 200 * 200 * 4];
        let mut canvas = new_canvas(&mut buf);
        let now = Instant::now();
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]).with_style(LEGACY);
        let (p, received) = point_at_xy(
            now,
            Point2::new(5.0, 100.0),
            Matrix2::new(100.0, 0.0, 0.0, 25.0),
            1.0,
        );
        scene.on_msg(p, received);
        scene.render(&mut canvas, received);
        assert_alpha(&buf, 5, 100, 234.0, 3.0);
        assert_alpha(&buf, 27, 100, 40.0, 3.0);
        assert_alpha(&buf, 33, 100, 0.0, 0.5);
    }

    #[test]
    fn test_logs_gaze_point_drawn_at_trace() {
        let now = Instant::now();
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]);
        let (p, received) = point_at(now, Matrix2::new(100.0, 0.0, 0.0, 25.0), 1.0);

        let mut buf = vec![0u8; 200 * 200 * 4];
        let (_, records) = capture_logs(tracing::Level::TRACE, || {
            scene.on_msg(p, received);
            let mut canvas = new_canvas(&mut buf);
            scene.render(&mut canvas, received);
        });

        let rec = records
            .iter()
            .find(|r| r.message == "gaze point drawn")
            .expect("gaze point drawn record");
        assert_eq!(rec.level, eye_log::Level::Trace);
        assert_eq!(rec.fields["semi_major"], Value::F64(10.0 * K95));
        assert_eq!(rec.fields["semi_minor"], Value::F64(5.0 * K95));
        assert_eq!(rec.fields["confidence"], Value::F64(1.0));
        assert_eq!(rec.fields["fade"], Value::F64(1.0));
        assert_eq!(rec.fields["ts_ns"], Value::U64(0));
        assert!(!rec.fields.contains_key("pixels"));
        assert!(!rec.fields.contains_key("buf"));
    }

    #[test]
    fn test_logs_stale_gaze_point_hidden_at_debug() {
        let mut buf = vec![0u8; 200 * 200 * 4];
        let now = Instant::now();
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]);
        let (p, received) = point_at(now, Matrix2::new(100.0, 0.0, 0.0, 25.0), 1.0);
        scene.on_msg(p, received);

        let later = received + Duration::from_millis(900);
        let (_, records) = capture_logs(tracing::Level::TRACE, || {
            let mut canvas = new_canvas(&mut buf);
            scene.render(&mut canvas, later);
        });

        let rec = records
            .iter()
            .find(|r| r.message == "stale gaze point hidden")
            .expect("stale gaze point hidden record");
        assert_eq!(rec.level, eye_log::Level::Debug);
        assert_eq!(rec.fields["reason"], Value::Str("faded".to_string()));
        match rec.fields["age_us"] {
            Value::U64(v) => assert!(v >= 900_000, "age_us = {v}"),
            ref other => panic!("expected U64, got {other:?}"),
        }
    }

    #[test]
    fn test_point_scene_marks_first_on_target_commit() {
        let (w, h) = (200u32, 100u32);
        let mut buf = vec![0u8; (w * h * 4) as usize];
        let now = Instant::now();
        let mut scene = PointScene::with_interpolation(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Interpolation::from_millis(80),
        );
        // The scene teleports to the first sample it ever sees (nothing to ease from),
        // so settle a baseline point before sending the sample under test.
        let mut baseline = point_at_xy(now, Point2::new(0.0, 0.0), Matrix2::zeros(), 1.0).0;
        baseline.timestamp = eye_core::Timestamp::from_nanos(500_000);
        scene.on_msg(baseline, now);
        {
            let mut canvas = Canvas::new(&mut buf, (w, h), 1).expect("size");
            scene.render(&mut canvas, now);
        }

        let mut p = point_at_xy(now, Point2::new(100.0, 0.0), Matrix2::zeros(), 1.0).0;
        p.timestamp = eye_core::Timestamp::from_nanos(1_000_000);
        scene.on_msg(p, now);

        {
            let mut canvas = Canvas::new(&mut buf, (w, h), 1).expect("size");
            scene.render(&mut canvas, now);
        }
        assert_eq!(
            scene.presentation_mark(),
            None,
            "still easing toward target, nothing settled yet"
        );

        // Mirror surface.rs's frame loop: only draw when `step` says this frame changed.
        let mut redraw = true;
        let mut t = now;
        while redraw {
            t += Duration::from_millis(16);
            redraw = scene.step(t);
            if redraw {
                let mut canvas = Canvas::new(&mut buf, (w, h), 1).expect("size");
                scene.render(&mut canvas, t);
            }
        }
        assert_eq!(scene.presentation_mark(), Some(1_000_000));
    }

    #[test]
    fn test_newest_sample_marks_on_crossing_frame_with_realistic_spacing() {
        let (w, h) = (200u32, 100u32);
        let mut buf = vec![0u8; (w * h * 4) as usize];
        let now = Instant::now();
        let mut scene = PointScene::with_interpolation(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Interpolation::from_millis(80),
        );
        let mut baseline = point_at_xy(now, Point2::new(0.0, 0.0), Matrix2::zeros(), 1.0).0;
        baseline.timestamp = eye_core::Timestamp::from_nanos(500_000);
        scene.on_msg(baseline, now);
        {
            let mut canvas = Canvas::new(&mut buf, (w, h), 1).expect("size");
            scene.render(&mut canvas, now);
        }

        let second_at = now + Duration::from_millis(33);
        let mut p = point_at_xy(second_at, Point2::new(100.0, 0.0), Matrix2::zeros(), 1.0).0;
        p.timestamp = eye_core::Timestamp::from_nanos(1_000_000);
        scene.on_msg(p, second_at);
        {
            let mut canvas = Canvas::new(&mut buf, (w, h), 1).expect("size");
            scene.render(&mut canvas, second_at);
        }
        assert_eq!(
            scene.presentation_mark(),
            None,
            "still easing toward target, nothing settled yet"
        );

        let mut redraw = true;
        let mut t = second_at;
        let mut first_tick = true;
        while redraw {
            t += Duration::from_millis(if first_tick { 7 } else { 16 });
            first_tick = false;
            redraw = scene.step(t);
            if redraw {
                let mut canvas = Canvas::new(&mut buf, (w, h), 1).expect("size");
                scene.render(&mut canvas, t);
            }
        }
        assert_eq!(scene.presentation_mark(), Some(1_000_000));
        assert_abs_diff_eq!(scene.drawn.unwrap().center.x, 100.0, epsilon = 1e-9);
    }

    #[test]
    fn test_first_sample_marks_with_default_interpolation() {
        let mut buf = vec![0u8; 4];
        let now = Instant::now();
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]);
        let mut p = point_at_xy(now, Point2::new(0.0, 0.0), Matrix2::zeros(), 1.0).0;
        p.timestamp = eye_core::Timestamp::from_nanos(3_000_000);
        scene.on_msg(p, now);

        let mut canvas = Canvas::new(&mut buf, (1, 1), 1).expect("size");
        scene.render(&mut canvas, now);
        assert_eq!(
            scene.presentation_mark(),
            Some(3_000_000),
            "the dot is drawn at the knot immediately, so the first sample needs no lag"
        );
    }

    #[test]
    fn test_point_scene_without_interpolation_marks_first_render() {
        let mut buf = vec![0u8; 4];
        let now = Instant::now();
        let mut scene = PointScene::with_interpolation(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Interpolation::from_millis(0),
        );
        let mut p = point_at_xy(now, Point2::new(0.0, 0.0), Matrix2::zeros(), 1.0).0;
        p.timestamp = eye_core::Timestamp::from_nanos(2_000_000);
        scene.on_msg(p, now);

        let mut canvas = Canvas::new(&mut buf, (1, 1), 1).expect("size");
        scene.render(&mut canvas, now);
        assert_eq!(scene.presentation_mark(), Some(2_000_000));
    }

    #[test]
    fn test_hidden_render_clears_stale_mark() {
        let mut buf = vec![0u8; 4];
        let now = Instant::now();
        let mut scene = PointScene::with_interpolation(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Interpolation::from_millis(0),
        );
        let mut a = point_at_xy(now, Point2::new(0.0, 0.0), Matrix2::zeros(), 1.0).0;
        a.timestamp = eye_core::Timestamp::from_nanos(1_000_000);
        scene.on_msg(a, now);
        {
            let mut canvas = Canvas::new(&mut buf, (1, 1), 1).expect("size");
            scene.render(&mut canvas, now);
        }
        assert_eq!(scene.presentation_mark(), Some(1_000_000));

        let mut b = point_at_xy(now, Point2::new(0.0, 0.0), Matrix2::zeros(), 0.0).0;
        b.timestamp = eye_core::Timestamp::from_nanos(2_000_000);
        scene.on_msg(b, now);
        {
            let mut canvas = Canvas::new(&mut buf, (1, 1), 1).expect("size");
            scene.render(&mut canvas, now);
        }
        assert_eq!(scene.presentation_mark(), None);
    }

    #[test]
    fn test_newer_sample_replaces_unmarked_pending() {
        let (w, h) = (200u32, 100u32);
        let mut buf = vec![0u8; (w * h * 4) as usize];
        let now = Instant::now();
        let mut scene = PointScene::with_interpolation(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Interpolation::from_millis(80),
        );
        let mut p0 = point_at_xy(now, Point2::new(0.0, 0.0), Matrix2::zeros(), 1.0).0;
        p0.timestamp = eye_core::Timestamp::from_nanos(1_000_000);
        scene.on_msg(p0, now);

        let mut p1 = point_at_xy(now, Point2::new(50.0, 0.0), Matrix2::zeros(), 1.0).0;
        p1.timestamp = eye_core::Timestamp::from_nanos(2_000_000);
        scene.on_msg(p1, now);

        let mut redraw = true;
        let mut t = now;
        while redraw {
            t += Duration::from_millis(16);
            redraw = scene.step(t);
            if redraw {
                let mut canvas = Canvas::new(&mut buf, (w, h), 1).expect("size");
                scene.render(&mut canvas, t);
            }
        }
        assert_eq!(scene.presentation_mark(), Some(2_000_000));
    }

    #[test]
    fn test_settled_rerender_before_presented_does_not_remark() {
        let mut buf = vec![0u8; 4];
        let now = Instant::now();
        let mut scene = PointScene::with_interpolation(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Interpolation::from_millis(0),
        );
        let mut p = point_at_xy(now, Point2::new(0.0, 0.0), Matrix2::zeros(), 1.0).0;
        p.timestamp = eye_core::Timestamp::from_nanos(1_000_000);
        scene.on_msg(p, now);

        let mut canvas = Canvas::new(&mut buf, (1, 1), 1).expect("size");
        scene.render(&mut canvas, now);
        assert_eq!(scene.presentation_mark(), Some(1_000_000));

        scene.render(&mut canvas, now);
        assert_eq!(
            scene.presentation_mark(),
            None,
            "the mark was already handed out; a re-render before on_presented must not hand it out again"
        );
    }

    #[test]
    fn test_point_scene_records_capture_to_present_from_presentation_time() {
        let mut buf = vec![0u8; 4];
        let now = Instant::now();
        let mut scene = PointScene::with_interpolation(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Interpolation::from_millis(0),
        );
        let mut p = point_at_xy(now, Point2::new(0.0, 0.0), Matrix2::zeros(), 1.0).0;
        p.timestamp = eye_core::Timestamp::from_nanos(1_000_000);
        scene.on_msg(p, now);

        let mut canvas = Canvas::new(&mut buf, (1, 1), 1).expect("size");
        scene.render(&mut canvas, now);
        let mark = scene.presentation_mark().expect("marked");

        scene.on_presented(
            mark,
            PresentedAt::Presentation(eye_core::Timestamp::from_nanos(41_000_000)),
            now,
        );

        let summary = scene.present.summary();
        assert_eq!(summary.count, 1);
        assert_eq!(summary.p50, Duration::from_millis(40));
    }
}
