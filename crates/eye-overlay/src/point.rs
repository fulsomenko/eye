//! Pin-point mode: a dot plus uncertainty ellipse for the latest gaze point.

use std::time::{Duration, Instant};

use eye_core::GazePoint;
use nalgebra::{Point2, Vector2};

use crate::canvas::{Canvas, Rgba};
use crate::ellipse::{K95, confidence_ellipse, cov_mm_to_logical_px};
use crate::fade::{FADE_START, fade};
use crate::scene::{Scene, Schedule};

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
    hidden_offscreen: bool,
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
            hidden_offscreen: false,
        }
    }

    pub fn with_hide_rules(mut self, hide: HideRules) -> Self {
        self.hide = hide;
        self
    }

    fn rgba(&self, alpha: f64) -> Rgba {
        let a = alpha.round().clamp(0.0, 255.0) as u8;
        Rgba {
            r: self.color[0],
            g: self.color[1],
            b: self.color[2],
            a,
        }
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
        let target = self.target_for(&msg);
        let stale = self.latest.as_ref().is_some_and(|(_, received)| {
            now.saturating_duration_since(*received) > self.easing.reset_after
        });
        if self.drawn.is_none() || stale || self.easing.disabled() {
            self.drawn = Some(target);
        } else if self.drawn == self.target {
            // Settled: no in-flight animation whose dt this would shorten.
            self.last_step = Some(now);
        }
        self.target = Some(target);
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
        self.ease_toward_target(dt)
    }

    fn render(&mut self, canvas: &mut Canvas<'_>, now: Instant) -> Schedule {
        let Some((p, received)) = &self.latest else {
            return Schedule::Idle;
        };
        let age = now.saturating_duration_since(*received);
        let f = fade(age);
        if f == 0.0 {
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
        let max_axis = f64::from(w).hypot(f64::from(h));
        if let Some(ellipse) = drawn.ellipse {
            let axes = (
                ellipse.axes.0.clamp(0.5, max_axis),
                ellipse.axes.1.clamp(0.5, max_axis),
            );
            canvas.fill_ellipse(drawn.center, axes, ellipse.angle, self.rgba(40.0 * f));
            canvas.stroke_ellipse(drawn.center, axes, ellipse.angle, 2.0, self.rgba(160.0 * f));
        }
        let conf = p.confidence.clamp(0.0, 1.0);
        canvas.fill_circle(drawn.center, 6.0, self.rgba(230.0 * f * (0.4 + 0.6 * conf)));
        if age < FADE_START {
            Schedule::At(*received + FADE_START)
        } else {
            Schedule::NextFrame
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use eye_core::OutputId;
    use nalgebra::Matrix2;

    use super::*;
    use crate::canvas::bgra;

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
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]);
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
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]);
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
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]);
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

        let moving = scene.step(now + Duration::from_millis(80));
        assert!(!moving);
        assert_eq!(scene.drawn.unwrap().center, scene.target.unwrap().center);
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
        let mut scene = PointScene::new(Vector2::new(1.0, 1.0), [255, 64, 64]);
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
