//! Region mode: highlights the grid cell holding the latest gaze point.

use std::time::Instant;

use eye_core::GazePoint;
use eye_core::grid::Grid;
use nalgebra::{Point2, Vector2};

use crate::canvas::{Canvas, LogicalRect, Rgba};
use crate::ellipse::{ConfidenceEllipse, K95, confidence_ellipse, cov_mm_to_logical_px};
use crate::fade::{FADE_START, fade};
use crate::scene::{Scene, Schedule};

/// Fraction of a 64-point area-uniform polar sample of `e` that falls inside `[min, max)`.
pub fn containment_fraction(e: &ConfidenceEllipse, min: Point2<f64>, max: Point2<f64>) -> f64 {
    let (s, c) = e.angle.sin_cos();
    let mut inside = 0u32;
    for k in 0..8 {
        let r = ((f64::from(k) + 0.5) / 8.0).sqrt();
        for j in 0..8 {
            let phi = std::f64::consts::TAU * (f64::from(j) + 0.5) / 8.0;
            let (u, v) = (r * e.semi_axes.0 * phi.cos(), r * e.semi_axes.1 * phi.sin());
            let (x, y) = (e.center.x + u * c - v * s, e.center.y + u * s + v * c);
            if x >= min.x && x < max.x && y >= min.y && y < max.y {
                inside += 1;
            }
        }
    }
    f64::from(inside) / 64.0
}

#[derive(Debug)]
pub struct RegionScene {
    grid: Grid,
    px_per_mm: Vector2<f64>,
    color: [u8; 3],
    latest: Option<(GazePoint, Instant)>,
}

impl RegionScene {
    pub fn new(grid: Grid, px_per_mm: Vector2<f64>, color: [u8; 3]) -> Self {
        Self {
            grid,
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

impl Scene for RegionScene {
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
        let schedule = if age < FADE_START {
            Schedule::At(*received + FADE_START)
        } else {
            Schedule::NextFrame
        };
        let (w, h) = canvas.logical_size();
        let size = (f64::from(w), f64::from(h));
        let Some(cell) = self.grid.cell_of(p.px_logical, size) else {
            return schedule;
        };
        let (min, max) = self.grid.cell_bounds(cell, size);
        let cov_px = cov_mm_to_logical_px(&p.cov_mm, &self.px_per_mm);
        let containment = confidence_ellipse(p.px_logical, &cov_px, K95, size.0.hypot(size.1))
            .map_or(1.0, |e| containment_fraction(&e, min, max));
        let rect = LogicalRect {
            x: min.x,
            y: min.y,
            w: max.x - min.x,
            h: max.y - min.y,
        };
        canvas.fill_rect(rect, self.rgba((40.0 + 120.0 * containment) * f));
        let inset = LogicalRect {
            x: rect.x + 1.5,
            y: rect.y + 1.5,
            w: rect.w - 3.0,
            h: rect.h - 3.0,
        };
        canvas.stroke_rect(inset, 3.0, self.rgba(200.0 * f));
        schedule
    }
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use eye_core::{OutputId, Timestamp};
    use nalgebra::Matrix2;

    use super::*;
    use crate::canvas::bgra;

    fn new_canvas(buf: &mut [u8]) -> Canvas<'_> {
        Canvas::new(buf, (200, 200), 1).expect("size")
    }

    fn assert_alpha(buf: &[u8], x: u32, y: u32, expected: f64, tolerance: f64) {
        let px = bgra(buf, 200, x, y);
        assert_abs_diff_eq!(f64::from(px[3]), expected, epsilon = tolerance);
    }

    fn point_at(px_logical: Point2<f64>, cov_mm: Matrix2<f64>, confidence: f64) -> GazePoint {
        GazePoint {
            timestamp: Timestamp::from_nanos(0),
            output: OutputId::from("eDP-1"),
            mm: Point2::new(0.0, 0.0),
            px_physical: px_logical,
            px_logical,
            cov_mm,
            confidence,
        }
    }

    #[test]
    fn test_containment_on_vertical_boundary_is_half() {
        let e = ConfidenceEllipse {
            center: Point2::new(100.0, 75.0),
            semi_axes: (10.0, 10.0),
            angle: 0.0,
        };
        let fraction =
            containment_fraction(&e, Point2::new(100.0, 50.0), Point2::new(150.0, 100.0));
        assert_relative_eq!(fraction, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_containment_fully_inside_is_one() {
        let e = ConfidenceEllipse {
            center: Point2::new(75.0, 75.0),
            semi_axes: (5.0, 5.0),
            angle: 0.0,
        };
        let fraction = containment_fraction(&e, Point2::new(50.0, 50.0), Point2::new(100.0, 100.0));
        assert_relative_eq!(fraction, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_containment_at_cell_corner_is_quarter() {
        let e = ConfidenceEllipse {
            center: Point2::new(100.0, 100.0),
            semi_axes: (10.0, 10.0),
            angle: 0.0,
        };
        let fraction =
            containment_fraction(&e, Point2::new(100.0, 100.0), Point2::new(150.0, 150.0));
        assert_relative_eq!(fraction, 0.25, epsilon = 1e-12);
    }

    #[test]
    fn test_region_scene_highlights_cell() {
        let mut buf = vec![0u8; 200 * 200 * 4];
        let mut canvas = new_canvas(&mut buf);
        let now = Instant::now();
        let grid = Grid::new(4, 4).unwrap();
        let mut scene = RegionScene::new(grid, Vector2::new(1.0, 1.0), [64, 160, 255]);
        let p = point_at(
            Point2::new(60.5, 60.5),
            Matrix2::new(0.01, 0.0, 0.0, 0.01),
            1.0,
        );
        scene.on_msg(p, now);
        scene.render(&mut canvas, now);

        assert_alpha(&buf, 75, 75, 160.0, 3.0);
        assert!(bgra(&buf, 200, 51, 75)[3] as f64 >= 200.0);
        assert_alpha(&buf, 25, 25, 0.0, 0.0);
        assert_alpha(&buf, 125, 75, 0.0, 0.0);
    }

    #[test]
    fn test_region_scene_straddling_ellipse_is_fainter() {
        let mut buf = vec![0u8; 200 * 200 * 4];
        let mut canvas = new_canvas(&mut buf);
        let now = Instant::now();
        let grid = Grid::new(4, 4).unwrap();
        let mut scene = RegionScene::new(grid, Vector2::new(1.0, 1.0), [64, 160, 255]);
        let p = point_at(
            Point2::new(100.0, 75.0),
            Matrix2::new(25.0, 0.0, 0.0, 25.0),
            1.0,
        );
        scene.on_msg(p, now);
        scene.render(&mut canvas, now);

        assert_alpha(&buf, 125, 75, 100.0, 3.0);
    }

    #[test]
    fn test_region_scene_offscreen_point_draws_nothing() {
        let mut buf = vec![0u8; 200 * 200 * 4];
        let mut canvas = new_canvas(&mut buf);
        let now = Instant::now();
        let grid = Grid::new(4, 4).unwrap();
        let mut scene = RegionScene::new(grid, Vector2::new(1.0, 1.0), [64, 160, 255]);
        let p = point_at(
            Point2::new(-5.0, 20.0),
            Matrix2::new(0.01, 0.0, 0.0, 0.01),
            1.0,
        );
        scene.on_msg(p, now);
        let schedule = scene.render(&mut canvas, now);

        assert!(buf.iter().all(|&b| b == 0));
        assert_eq!(schedule, Schedule::At(now + FADE_START));
    }

    #[test]
    fn test_region_scene_fades_like_point_mode() {
        let mut buf = vec![0u8; 200 * 200 * 4];
        let mut canvas = new_canvas(&mut buf);
        let now = Instant::now();
        let grid = Grid::new(4, 4).unwrap();
        let mut scene = RegionScene::new(grid, Vector2::new(1.0, 1.0), [64, 160, 255]);
        let p = point_at(
            Point2::new(60.5, 60.5),
            Matrix2::new(0.01, 0.0, 0.0, 0.01),
            1.0,
        );
        scene.on_msg(p, now);
        let later = now + std::time::Duration::from_millis(550);
        scene.render(&mut canvas, later);

        assert_alpha(&buf, 75, 75, 80.0, 3.0);
    }
}
