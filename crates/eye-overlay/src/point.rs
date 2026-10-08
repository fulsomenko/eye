//! Pin-point mode: a dot plus uncertainty ellipse for the latest gaze point.

use std::time::Instant;

use eye_core::GazePoint;
use nalgebra::Vector2;

use crate::canvas::{Canvas, Rgba};
use crate::ellipse::{K95, confidence_ellipse, cov_mm_to_logical_px};
use crate::fade::{FADE_START, fade};
use crate::scene::{Scene, Schedule};

#[derive(Debug)]
pub struct PointScene {
    px_per_mm: Vector2<f64>,
    color: [u8; 3],
    latest: Option<(GazePoint, Instant)>,
}

impl PointScene {
    pub fn new(px_per_mm: Vector2<f64>, color: [u8; 3]) -> Self {
        Self {
            px_per_mm,
            color,
            latest: None,
        }
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
}

impl Scene for PointScene {
    type Msg = GazePoint;

    fn on_msg(&mut self, msg: GazePoint, now: Instant) {
        self.latest = Some((msg, now));
    }

    fn render(&mut self, canvas: &mut Canvas<'_>, now: Instant) -> Schedule {
        let Some((p, received)) = &self.latest else {
            return Schedule::Idle;
        };
        let age = now.saturating_duration_since(*received);
        let f = fade(age);
        if f == 0.0 {
            self.latest = None;
            return Schedule::Idle;
        }
        let (w, h) = canvas.logical_size();
        let max_axis = f64::from(w).hypot(f64::from(h));
        let cov_px = cov_mm_to_logical_px(&p.cov_mm, &self.px_per_mm);
        if let Some(e) = confidence_ellipse(p.px_logical, &cov_px, K95, max_axis) {
            canvas.fill_ellipse(e.center, e.semi_axes, e.angle, self.rgba(40.0 * f));
            canvas.stroke_ellipse(e.center, e.semi_axes, e.angle, 2.0, self.rgba(160.0 * f));
        }
        let conf = p.confidence.clamp(0.0, 1.0);
        canvas.fill_circle(p.px_logical, 6.0, self.rgba(230.0 * f * (0.4 + 0.6 * conf)));
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
    use nalgebra::{Matrix2, Point2};

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
        let (p, received) = point_at(now, Matrix2::new(100.0, 0.0, 0.0, 25.0), 0.0);
        scene.on_msg(p, received);
        scene.render(&mut canvas, received);
        assert_alpha(&buf, 100, 100, 118.0, 3.0);
    }
}
