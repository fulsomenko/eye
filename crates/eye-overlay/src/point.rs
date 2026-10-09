//! Pin-point mode: a dot plus uncertainty ellipse for the latest gaze point.

use std::time::{Duration, Instant};

use eye_core::GazePoint;
use eye_core::log::field;
use nalgebra::{Point2, Vector2};

use crate::canvas::{Canvas, Rgba};
use crate::ellipse::{K95, confidence_ellipse, cov_mm_to_logical_px};
use crate::fade::{FADE_START, fade};
use crate::scene::{PresentedAt, Scene, Schedule};
use crate::stats::PresentStats;

/// Display-rate easing toward the latest sample. `tau` is the time constant of the
/// exponential approach; `snap_px` ends the animation when the drawn point (and, for the
/// ellipse, both drawn semi-axes) are this close to the target. `reset_after` teleports
/// instead of sweeping when a new sample arrives this long after the previous one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Easing {
    pub tau: Duration,
    pub snap_px: f64,
    pub reset_after: Duration,
}

impl Easing {
    /// `tau` zero reproduces the pre-easing behaviour: one draw per sample, no sweep.
    pub fn from_millis(easing_ms: u64) -> Self {
        Self {
            tau: Duration::from_millis(easing_ms),
            ..Self::default()
        }
    }

    fn disabled(&self) -> bool {
        self.tau.is_zero()
    }
}

impl Default for Easing {
    fn default() -> Self {
        Self {
            tau: Duration::from_millis(80),
            snap_px: 0.5,
            reset_after: Duration::from_secs(1),
        }
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
            min_confidence: 0.2,
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
    /// A new sample within this many standard deviations (its own covariance, Mahalanobis)
    /// of the current target leaves the dot where it is; `0.0` follows every sample.
    pub hold_sigmas: f64,
}

impl Default for PointStyle {
    fn default() -> Self {
        Self {
            certainty_colors: true,
            show_ellipse: false,
            hold_sigmas: 1.5,
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

#[derive(Debug)]
pub struct PointScene {
    px_per_mm: Vector2<f64>,
    color: [u8; 3],
    easing: Easing,
    latest: Option<(GazePoint, Instant)>,
    target: Option<Eased>,
    drawn: Option<Eased>,
    last_step: Option<Instant>,
    hide: HideRules,
    style: PointStyle,
    hidden_offscreen: bool,
    present: PresentStats,
    /// `timestamp.as_nanos()` of the latest sample whose first on-target commit has not
    /// been handed out as a presentation mark yet. A newer sample replaces an unmarked
    /// older one: the dot never reached it, nothing to measure. `render` takes this (sets
    /// it to `None`) the moment `drawn` catches up to `target`, so a sample is marked at
    /// most once even if `render` runs again before `on_presented` arrives.
    pending_mark: Option<u64>,
    /// The mark for the current commit, or `None` if this commit carries no mark. Reset
    /// to `None` at the top of every `render` and set at most once per sample, from
    /// `pending_mark`, when `drawn` catches up to `target`.
    mark_ready: Option<u64>,
}

impl PointScene {
    pub fn new(px_per_mm: Vector2<f64>, color: [u8; 3]) -> Self {
        Self::with_easing(px_per_mm, color, Easing::default())
    }

    pub fn with_easing(px_per_mm: Vector2<f64>, color: [u8; 3], easing: Easing) -> Self {
        Self {
            px_per_mm,
            color,
            easing,
            latest: None,
            target: None,
            drawn: None,
            last_step: None,
            hide: HideRules::default(),
            style: PointStyle::default(),
            hidden_offscreen: false,
            present: PresentStats::default(),
            pending_mark: None,
            mark_ready: None,
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

    fn held(&self, msg: &GazePoint) -> bool {
        let Some(current) = self.target else {
            return false;
        };
        if self.style.hold_sigmas <= 0.0 {
            return false;
        }
        let cov_px = cov_mm_to_logical_px(&msg.cov_mm, &self.px_per_mm);
        let Some(inv) = cov_px.try_inverse() else {
            return false;
        };
        let d = msg.px_logical - current.center;
        (d.transpose() * inv * d)[(0, 0)].sqrt() < self.style.hold_sigmas
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

    /// `true` while `drawn` is still short of `target` by more than `snap_px`, for the
    /// center and (when both have an ellipse) for either semi-axis.
    fn ease_toward_target(&mut self, dt: Duration) -> bool {
        let (Some(target), Some(drawn)) = (self.target, self.drawn) else {
            return false;
        };
        let alpha = 1.0 - (-dt.as_secs_f64() / self.easing.tau.as_secs_f64()).exp();
        let lerp = |a: f64, b: f64| a + alpha * (b - a);

        let center = Point2::new(
            lerp(drawn.center.x, target.center.x),
            lerp(drawn.center.y, target.center.y),
        );
        let center_moving = (center - target.center).norm() >= self.easing.snap_px;

        let (ellipse, ellipse_moving) = match (drawn.ellipse, target.ellipse) {
            (Some(d), Some(t)) => {
                let axes = (lerp(d.axes.0, t.axes.0), lerp(d.axes.1, t.axes.1));
                let angle = lerp_angle(d.angle, t.angle, alpha);
                let moving = (axes.0 - t.axes.0).abs() >= self.easing.snap_px
                    || (axes.1 - t.axes.1).abs() >= self.easing.snap_px;
                (Some(EasedEllipse { axes, angle }), moving)
            }
            // The ellipse appeared, vanished, or there was none to begin with: no sweep.
            _ => (target.ellipse, false),
        };

        let moving = center_moving || ellipse_moving;
        self.drawn = Some(if moving {
            Eased { center, ellipse }
        } else {
            target
        });
        moving
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
        let mut target = self.target_for(&msg);
        let stale = self.latest.as_ref().is_some_and(|(_, received)| {
            now.saturating_duration_since(*received) > self.easing.reset_after
        });
        if !stale
            && self.held(&msg)
            && let Some(current) = self.target
        {
            target.center = current.center;
        }
        if self.drawn.is_none() || stale || self.easing.disabled() {
            self.drawn = Some(target);
        } else if self.drawn == self.target {
            // Settled: no in-flight animation whose dt this would shorten.
            self.last_step = Some(now);
        }
        self.target = Some(target);
        self.pending_mark = Some(msg.timestamp.as_nanos());
        self.latest = Some((msg, now));
    }

    fn step(&mut self, now: Instant) -> bool {
        if self.easing.disabled() {
            if let Some(target) = self.target {
                self.drawn = Some(target);
            }
            self.last_step = Some(now);
            return false;
        }
        let dt = self
            .last_step
            .map_or(Duration::ZERO, |t| now.saturating_duration_since(t));
        self.last_step = Some(now);
        let before = self.drawn;
        let moving = self.ease_toward_target(dt);
        // The settling step snaps `drawn` to `target` and reports `moving == false`;
        // the surface must still draw this frame to present the settled point.
        moving || self.drawn != before
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
            self.target = None;
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
            self.target = None;
            self.drawn = None;
            return Schedule::Idle;
        }
        if confidence < self.hide.min_confidence {
            return Schedule::Idle;
        }
        let Some(drawn) = self.drawn else {
            return Schedule::Idle;
        };
        if self.drawn == self.target {
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
    };

    fn point_px(now: Instant, x: f64, y: f64, cov: f64, confidence: f64) -> (GazePoint, Instant) {
        let (mut p, t) = point_at(now, Matrix2::identity() * cov, confidence);
        p.px_logical = Point2::new(x, y);
        (p, t)
    }

    #[test]
    fn test_small_move_within_hold_keeps_dot() {
        let now = Instant::now();
        let mut scene = PointScene::with_easing(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Easing::from_millis(0),
        );
        let (a, t) = point_px(now, 100.0, 100.0, 100.0, 0.5);
        scene.on_msg(a, t);
        let (b, t) = point_px(now, 112.0, 100.0, 100.0, 0.5);
        scene.on_msg(b, t);
        scene.step(t);
        assert_eq!(scene.drawn.unwrap().center, Point2::new(100.0, 100.0));
    }

    #[test]
    fn test_move_beyond_hold_follows() {
        let now = Instant::now();
        let mut scene = PointScene::with_easing(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Easing::from_millis(0),
        );
        let (a, t) = point_px(now, 100.0, 100.0, 100.0, 0.5);
        scene.on_msg(a, t);
        let (b, t) = point_px(now, 130.0, 100.0, 100.0, 0.5);
        scene.on_msg(b, t);
        scene.step(t);
        assert_eq!(scene.drawn.unwrap().center, Point2::new(130.0, 100.0));
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

    #[test]
    fn test_easing_moves_drawn_point_toward_target_exponentially() {
        let now = Instant::now();
        let mut scene = PointScene::with_easing(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Easing {
                tau: Duration::from_millis(80),
                snap_px: 0.5,
                reset_after: Duration::from_secs(1),
            },
        );
        scene.drawn = Some(Eased {
            center: Point2::new(0.0, 0.0),
            ellipse: None,
        });
        scene.target = Some(Eased {
            center: Point2::new(100.0, 0.0),
            ellipse: None,
        });
        scene.last_step = Some(now);

        let moving = scene.step(now + Duration::from_millis(80));
        assert!(moving);
        assert_abs_diff_eq!(scene.drawn.unwrap().center.x, 63.2, epsilon = 0.5);
    }

    #[test]
    fn test_easing_moves_ellipse_axes_with_the_same_alpha() {
        let now = Instant::now();
        let mut scene = PointScene::with_easing(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Easing {
                tau: Duration::from_millis(80),
                snap_px: 0.5,
                reset_after: Duration::from_secs(1),
            },
        );
        scene.drawn = Some(Eased {
            center: Point2::new(0.0, 0.0),
            ellipse: Some(EasedEllipse {
                axes: (10.0, 10.0),
                angle: 0.0,
            }),
        });
        scene.target = Some(Eased {
            center: Point2::new(0.0, 0.0),
            ellipse: Some(EasedEllipse {
                axes: (30.0, 30.0),
                angle: 0.0,
            }),
        });
        scene.last_step = Some(now);

        let moving = scene.step(now + Duration::from_millis(80));
        assert!(moving);
        assert_abs_diff_eq!(
            scene.drawn.unwrap().ellipse.unwrap().axes.0,
            22.6,
            epsilon = 0.5
        );
        assert_abs_diff_eq!(
            scene.drawn.unwrap().ellipse.unwrap().axes.1,
            22.6,
            epsilon = 0.5
        );
    }

    #[test]
    fn test_easing_stops_within_snap_distance() {
        let now = Instant::now();
        let mut scene = PointScene::with_easing(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Easing {
                tau: Duration::from_millis(80),
                snap_px: 0.5,
                reset_after: Duration::from_secs(1),
            },
        );
        scene.drawn = Some(Eased {
            center: Point2::new(99.8, 0.0),
            ellipse: None,
        });
        scene.target = Some(Eased {
            center: Point2::new(100.0, 0.0),
            ellipse: None,
        });
        scene.last_step = Some(now);

        let redraw = scene.step(now + Duration::from_millis(80));
        assert!(
            redraw,
            "the settling step snaps drawn and must still be drawn"
        );
        assert_eq!(scene.drawn.unwrap().center, scene.target.unwrap().center);

        let redraw_again = scene.step(now + Duration::from_millis(160));
        assert!(!redraw_again, "already settled, nothing changed to redraw");
    }

    #[test]
    fn test_first_sample_is_drawn_without_easing() {
        let now = Instant::now();
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]);
        let (p, received) = point_at(now, Matrix2::zeros(), 1.0);
        scene.on_msg(p, received);
        assert_eq!(scene.drawn.unwrap().center, scene.target.unwrap().center);
        assert_eq!(scene.drawn.unwrap().center, Point2::new(100.5, 100.5));
    }

    #[test]
    fn test_stale_target_resets_drawn_point() {
        let now = Instant::now();
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]);
        let (p0, received0) = point_at(now, Matrix2::zeros(), 1.0);
        scene.on_msg(p0, received0);

        let later = received0 + Duration::from_millis(1500);
        let mut p1 = point_at(later, Matrix2::zeros(), 1.0).0;
        p1.px_logical = Point2::new(500.0, 500.0);
        scene.on_msg(p1, later);

        assert_eq!(scene.drawn.unwrap().center, Point2::new(500.0, 500.0));
    }

    #[test]
    fn test_easing_ellipse_angle_takes_shortest_half_turn() {
        use std::f64::consts::FRAC_PI_2;

        let now = Instant::now();
        let mut scene = PointScene::with_easing(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Easing {
                tau: Duration::from_millis(80),
                snap_px: 0.5,
                reset_after: Duration::from_secs(1),
            },
        );
        let drawn_angle = FRAC_PI_2 - 0.05;
        let target_angle = -FRAC_PI_2 + 0.05;
        // The center also moves so `ease_toward_target` keeps the eased ellipse
        // (angle included) instead of snapping straight to `target`.
        scene.drawn = Some(Eased {
            center: Point2::new(0.0, 0.0),
            ellipse: Some(EasedEllipse {
                axes: (30.0, 10.0),
                angle: drawn_angle,
            }),
        });
        scene.target = Some(Eased {
            center: Point2::new(100.0, 0.0),
            ellipse: Some(EasedEllipse {
                axes: (30.0, 10.0),
                angle: target_angle,
            }),
        });
        scene.last_step = Some(now);

        scene.step(now + Duration::from_millis(80));

        let new_angle = scene.drawn.unwrap().ellipse.unwrap().angle;
        assert_abs_diff_eq!(new_angle - drawn_angle, 0.063, epsilon = 0.01);
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
            0.1,
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
    fn test_sample_after_settled_pause_eases_from_arrival() {
        let now = Instant::now();
        let mut scene = PointScene::with_easing(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Easing {
                tau: Duration::from_millis(80),
                snap_px: 0.5,
                reset_after: Duration::from_secs(1),
            },
        );
        scene.drawn = Some(Eased {
            center: Point2::new(0.0, 0.0),
            ellipse: None,
        });
        scene.target = Some(Eased {
            center: Point2::new(0.0, 0.0),
            ellipse: None,
        });
        let (p0, _) = point_at_xy(now, Point2::new(0.0, 0.0), Matrix2::zeros(), 1.0);
        scene.latest = Some((p0, now));
        scene.last_step = Some(now);

        let later = now + Duration::from_millis(500);
        let (p1, _) = point_at_xy(later, Point2::new(100.0, 0.0), Matrix2::zeros(), 1.0);
        scene.on_msg(p1, later);

        let moving = scene.step(later + Duration::from_millis(16));
        assert!(moving);
        assert_abs_diff_eq!(scene.drawn.unwrap().center.x, 18.1, epsilon = 1.0);
    }

    #[test]
    fn test_sample_mid_animation_preserves_frame_clock() {
        let now = Instant::now();
        let mut scene = PointScene::with_easing(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Easing {
                tau: Duration::from_millis(80),
                snap_px: 0.5,
                reset_after: Duration::from_secs(1),
            },
        );
        scene.drawn = Some(Eased {
            center: Point2::new(0.0, 0.0),
            ellipse: None,
        });
        scene.target = Some(Eased {
            center: Point2::new(100.0, 0.0),
            ellipse: None,
        });
        let (p0, _) = point_at_xy(now, Point2::new(100.0, 0.0), Matrix2::zeros(), 1.0);
        scene.latest = Some((p0, now));
        let last_step = now - Duration::from_millis(50);
        scene.last_step = Some(last_step);

        let (p1, _) = point_at_xy(now, Point2::new(150.0, 0.0), Matrix2::zeros(), 1.0);
        scene.on_msg(p1, now);

        assert_eq!(scene.last_step, Some(last_step));
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
        let mut scene = PointScene::with_easing(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Easing {
                tau: Duration::from_millis(80),
                snap_px: 0.5,
                reset_after: Duration::from_secs(1),
            },
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
    fn test_point_scene_without_easing_marks_first_render() {
        let mut buf = vec![0u8; 4];
        let now = Instant::now();
        let mut scene = PointScene::with_easing(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Easing::from_millis(0),
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
        let mut scene = PointScene::with_easing(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Easing::from_millis(0),
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
        let mut scene = PointScene::with_easing(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Easing {
                tau: Duration::from_millis(80),
                snap_px: 0.5,
                reset_after: Duration::from_secs(1),
            },
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
        let mut scene = PointScene::with_easing(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Easing::from_millis(0),
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
        let mut scene = PointScene::with_easing(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Easing::from_millis(0),
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

    #[test]
    fn test_easing_zero_disables_stepping() {
        let now = Instant::now();
        let mut scene = PointScene::with_easing(
            Vector2::new(1.0, 1.0),
            [255, 64, 64],
            Easing::from_millis(0),
        );
        let (p, received) = point_at(now, Matrix2::zeros(), 1.0);
        scene.on_msg(p, received);
        assert_eq!(scene.drawn.unwrap().center, Point2::new(100.5, 100.5));

        let moving = scene.step(received + Duration::from_millis(80));
        assert!(!moving);
        assert_eq!(scene.drawn.unwrap().center, Point2::new(100.5, 100.5));
    }
}
