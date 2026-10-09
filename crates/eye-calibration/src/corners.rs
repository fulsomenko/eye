use eye_core::image::GrayView;
use eye_core::log::field;
use nalgebra::{Matrix2, Point2, Vector2};

use crate::error::CalibrationError;

#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoardSpec {
    pub inner_cols: u32,
    pub inner_rows: u32,
    pub square_mm: f64,
}

impl BoardSpec {
    pub fn validate(&self) -> Result<(), CalibrationError> {
        let (c, r) = (self.inner_cols, self.inner_rows);
        let ok = c >= 3
            && r >= 3
            && !(c + r).is_multiple_of(2)
            && self.square_mm > 0.0
            && self.square_mm.is_finite();
        if ok {
            Ok(())
        } else {
            Err(CalibrationError::Param {
                name: "board",
                reason: format!(
                    "inner_cols={c}, inner_rows={r}, square_mm={sq} \
                     (need >=3x>=3 with an odd corner-count sum, and a positive finite square size)",
                    sq = self.square_mm
                ),
            })
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CornerConfig {
    pub blur_sigma_px: f64,
    pub nms_radius_px: u32,
    pub rel_threshold: f64,
    pub ring_radius_px: f64,
    pub subpix_radius_px: u32,
}

impl Default for CornerConfig {
    fn default() -> Self {
        Self {
            blur_sigma_px: 1.5,
            nms_radius_px: 3,
            rel_threshold: 0.05,
            ring_radius_px: 4.0,
            subpix_radius_px: 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CornerCandidate {
    pub px: Point2<f64>,
    pub response: f64,
}

pub fn corner_candidates(img: &GrayView<'_>, cfg: &CornerConfig) -> Vec<CornerCandidate> {
    candidates_in(&Smoothed::new(img, cfg.blur_sigma_px), cfg)
}

pub(crate) fn candidates_in(l: &Smoothed, cfg: &CornerConfig) -> Vec<CornerCandidate> {
    let (w, h) = (l.w, l.h);
    let s = saddle_response(l);
    let max_s = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max_s.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        tracing::debug!(
            { field::REASON } = "flat_image",
            max_response = max_s,
            "no corner response"
        );
        return Vec::new();
    }
    let threshold = cfg.rel_threshold * max_s;
    let r = i64::from(cfg.nms_radius_px);
    let mut out = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let v = s[y * w + x];
            if !v.is_finite() || v <= threshold || !is_local_max(&s, w, h, x, y, r) {
                continue;
            }
            let p0 = Point2::new(x as f64 + 0.5, y as f64 + 0.5);
            if !has_saddle_polarity(l, &p0, cfg.ring_radius_px) {
                continue;
            }
            if let Some(q) = refine_corner(l, p0, i64::from(cfg.subpix_radius_px)) {
                out.push(CornerCandidate { px: q, response: v });
            }
        }
    }
    tracing::trace!(
        candidates = out.len() as u64,
        threshold,
        max_response = max_s,
        "corner candidates"
    );
    out
}

/// Pixel `(x, y)` survives NMS when every neighbour in the `(2r+1)^2` window either has a
/// strictly smaller response, or ties and sits earlier in `(y, x)` order.
fn is_local_max(s: &[f64], w: usize, h: usize, x: usize, y: usize, r: i64) -> bool {
    let v = s[y * w + x];
    for dy in -r..=r {
        for dx in -r..=r {
            if dx == 0 && dy == 0 {
                continue;
            }
            let (nx, ny) = (x as i64 + dx, y as i64 + dy);
            if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                continue;
            }
            let (nx, ny) = (nx as usize, ny as usize);
            let o = s[ny * w + nx];
            if !o.is_finite() {
                continue;
            }
            let neighbour_wins_tie = (ny, nx) < (y, x);
            if !(o < v || (o <= v && !neighbour_wins_tie)) {
                return false;
            }
        }
    }
    true
}

fn saddle_response(l: &Smoothed) -> Vec<f64> {
    let (w, h) = (l.w, l.h);
    let mut s = vec![f64::NEG_INFINITY; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let lxx = l.at(x + 1, y) - 2.0 * l.at(x, y) + l.at(x - 1, y);
            let lyy = l.at(x, y + 1) - 2.0 * l.at(x, y) + l.at(x, y - 1);
            let lxy = (l.at(x + 1, y + 1) - l.at(x + 1, y - 1) - l.at(x - 1, y + 1)
                + l.at(x - 1, y - 1))
                / 4.0;
            s[y * w + x] = lxy * lxy - lxx * lyy;
        }
    }
    s
}

#[derive(Debug)]
pub(crate) struct Smoothed {
    w: usize,
    h: usize,
    data: Vec<f32>,
}

/// Normalised separable Gaussian kernel of radius `ceil(3 sigma)`.
fn gaussian_kernel(sigma: f64, radius: i64) -> Vec<f64> {
    let mut kernel: Vec<f64> = (-radius..=radius)
        .map(|d| (-((d * d) as f64) / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f64 = kernel.iter().sum();
    for w in &mut kernel {
        *w /= sum;
    }
    kernel
}

impl Smoothed {
    pub(crate) fn new(img: &GrayView<'_>, sigma: f64) -> Self {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let radius = (3.0 * sigma).ceil() as i64;
        let kernel = gaussian_kernel(sigma, radius);

        let mut tmp = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0;
                for (k, &wt) in kernel.iter().enumerate() {
                    let sx = (x as i64 + k as i64 - radius).clamp(0, w as i64 - 1) as u32;
                    acc += wt * f64::from(img.get(sx, y as u32));
                }
                tmp[y * w + x] = acc as f32;
            }
        }

        let mut data = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0;
                for (k, &wt) in kernel.iter().enumerate() {
                    let sy = (y as i64 + k as i64 - radius).clamp(0, h as i64 - 1) as usize;
                    acc += wt * f64::from(tmp[sy * w + x]);
                }
                data[y * w + x] = acc as f32;
            }
        }

        Self { w, h, data }
    }

    fn at(&self, x: usize, y: usize) -> f64 {
        f64::from(self.data[y * self.w + x])
    }

    pub(crate) fn gradient(&self, ix: i64, iy: i64) -> Option<Vector2<f64>> {
        if ix < 1 || iy < 1 || ix >= self.w as i64 - 1 || iy >= self.h as i64 - 1 {
            return None;
        }
        let (ix, iy) = (ix as usize, iy as usize);
        let gx = (self.at(ix + 1, iy) - self.at(ix - 1, iy)) / 2.0;
        let gy = (self.at(ix, iy + 1) - self.at(ix, iy - 1)) / 2.0;
        Some(Vector2::new(gx, gy))
    }

    pub(crate) fn gradient_at(&self, p: &Point2<f64>) -> Option<Vector2<f64>> {
        let (fx, fy) = (p.x - 0.5, p.y - 0.5);
        let (x0, y0) = (fx.floor() as i64, fy.floor() as i64);
        let (ax, ay) = (fx - x0 as f64, fy - y0 as f64);
        let g00 = self.gradient(x0, y0)?;
        let g10 = self.gradient(x0 + 1, y0)?;
        let g01 = self.gradient(x0, y0 + 1)?;
        let g11 = self.gradient(x0 + 1, y0 + 1)?;
        Some((g00 * (1.0 - ax) + g10 * ax) * (1.0 - ay) + (g01 * (1.0 - ax) + g11 * ax) * ay)
    }

    pub(crate) fn sample(&self, p: &Point2<f64>) -> f64 {
        let fx = (p.x - 0.5).clamp(0.0, (self.w - 1) as f64);
        let fy = (p.y - 0.5).clamp(0.0, (self.h - 1) as f64);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (ax, ay) = (fx - x0 as f64, fy - y0 as f64);
        let top = self.at(x0, y0) * (1.0 - ax) + self.at(x1, y0) * ax;
        let bot = self.at(x0, y1) * (1.0 - ax) + self.at(x1, y1) * ax;
        top * (1.0 - ay) + bot * ay
    }
}

fn has_saddle_polarity(l: &Smoothed, p: &Point2<f64>, radius: f64) -> bool {
    let v: Vec<f64> = (0..16)
        .map(|k| {
            let a = f64::from(k) * std::f64::consts::TAU / 16.0;
            l.sample(&Point2::new(p.x + radius * a.cos(), p.y + radius * a.sin()))
        })
        .collect();
    let mean = v.iter().sum::<f64>() / 16.0;
    (0..16)
        .filter(|&k| ((v[k] - mean) > 0.0) != ((v[(k + 1) % 16] - mean) > 0.0))
        .count()
        == 4
}

fn refine_corner(l: &Smoothed, q0: Point2<f64>, radius: i64) -> Option<Point2<f64>> {
    let sigma_w = radius as f64 / 2.0;
    let mut q = q0;
    for _ in 0..20 {
        let (mut a, mut b) = (Matrix2::<f64>::zeros(), Vector2::<f64>::zeros());
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let p = Point2::new(q.x + dx as f64, q.y + dy as f64);
                let g = l.gradient_at(&p)?;
                let w = (-((dx * dx + dy * dy) as f64) / (2.0 * sigma_w * sigma_w)).exp();
                let ggt = g * g.transpose() * w;
                a += ggt;
                b += ggt * p.coords;
            }
        }
        let next = Point2::from(a.try_inverse()? * b);
        let step = (next - q).norm();
        q = next;
        if (q - q0).norm() > radius as f64 {
            return None;
        }
        if step < 0.01 {
            return Some(q);
        }
    }
    Some(q)
}

#[cfg(test)]
mod tests {
    use eye_core::image::GrayImage;
    use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector2};

    use super::*;
    use crate::testutil::{fixture_ir_intrinsics, render_board, x_junction};

    #[test]
    fn test_saddle_response_positive_at_x_junction_and_not_on_edge() {
        let img = x_junction(Point2::new(20.3, 19.7), 0.5, 41);
        let l = Smoothed::new(&img.view(), 1.5);
        let s = saddle_response(&l);
        let w = l.w;
        let (mut best_idx, mut best_val) = (0usize, f64::NEG_INFINITY);
        for (i, &v) in s.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        let (bx, by) = (best_idx % w, best_idx / w);
        let centre = Point2::new(bx as f64 + 0.5, by as f64 + 0.5);
        assert!(
            (centre - Point2::new(20.3, 19.7)).norm() < 0.75,
            "centre={centre:?}"
        );

        let mut data = vec![0u8; 41 * 41];
        for y in 0..41usize {
            for x in 0..41usize {
                data[y * 41 + x] = if x < 20 { 30 } else { 220 };
            }
        }
        let edge_img = GrayImage::new(41, 41, data).expect("w * h bytes");
        let el = Smoothed::new(&edge_img.view(), 1.5);
        let es = saddle_response(&el);
        let edge_max = es
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            edge_max < 0.01 * best_val,
            "edge_max={edge_max} junction_max={best_val}"
        );
    }

    #[test]
    fn test_polarity_rejects_t_junction() {
        let mut data = vec![0u8; 41 * 41];
        for y in 0..41usize {
            for x in 0..41usize {
                let v = if (x as f64 + 0.5) < 20.5 {
                    30
                } else if (y as f64 + 0.5) < 20.5 {
                    128
                } else {
                    220
                };
                data[y * 41 + x] = v;
            }
        }
        let img = GrayImage::new(41, 41, data).expect("w * h bytes");
        let l = Smoothed::new(&img.view(), 1.5);
        assert!(!has_saddle_polarity(&l, &Point2::new(20.5, 20.5), 4.0));

        let xj = x_junction(Point2::new(20.3, 19.7), 0.5, 41);
        let xl = Smoothed::new(&xj.view(), 1.5);
        assert!(has_saddle_polarity(&xl, &Point2::new(20.5, 19.5), 4.0));
    }

    #[test]
    fn test_subpixel_refines_to_true_corner() {
        let centre = Point2::new(20.3, 19.7);
        let img = x_junction(centre, 0.5, 41);
        let l = Smoothed::new(&img.view(), 1.5);
        let refined = refine_corner(&l, Point2::new(20.5, 19.5), 5).expect("should refine");
        assert!(
            (refined - centre).norm() < 0.05,
            "refined={refined:?} centre={centre:?}"
        );

        let candidates = corner_candidates(&img.view(), &CornerConfig::default());
        assert_eq!(candidates.len(), 1, "candidates={candidates:?}");
        assert!((candidates[0].px - centre).norm() < 0.05);
    }

    #[test]
    fn test_subpixel_sweep() {
        let centres = [
            Point2::new(20.3, 19.7),
            Point2::new(20.0, 20.0),
            Point2::new(20.5, 20.5),
            Point2::new(20.25, 20.6),
            Point2::new(19.9, 20.15),
        ];
        let angles = [0.0, 0.3, 0.5, 0.785, 1.2];
        for &centre in &centres {
            for &angle in &angles {
                let img = x_junction(centre, angle, 41);
                let l = Smoothed::new(&img.view(), 1.5);
                let start = Point2::new(centre.x.floor() + 0.5, centre.y.floor() + 0.5);
                let refined = refine_corner(&l, start, 5)
                    .unwrap_or_else(|| panic!("no refinement for centre={centre:?} angle={angle}"));
                let d = (refined - centre).norm();
                assert!(d < 0.15, "centre={centre:?} angle={angle} d={d}");
            }
        }
    }

    #[test]
    fn test_rendered_frontal_board_every_corner_has_one_candidate() {
        let intr = fixture_ir_intrinsics();
        let spec = BoardSpec {
            inner_cols: 9,
            inner_rows: 6,
            square_mm: 25.0,
        };
        let pose = Isometry3::from_parts(
            Translation3::new(-100.0, -62.5, 350.0),
            UnitQuaternion::identity(),
        );
        let (img, truth) = render_board(&intr, &pose, &spec, 3);
        assert_eq!(truth.len(), 54);

        let candidates = corner_candidates(&img.view(), &CornerConfig::default());

        let mut sq_err = 0.0;
        for t in &truth {
            let matches: Vec<&CornerCandidate> = candidates
                .iter()
                .filter(|c| (c.px - t).norm() < 0.3)
                .collect();
            assert_eq!(
                matches.len(),
                1,
                "expected exactly one candidate near {t:?}, got {}",
                matches.len()
            );
            let d = (matches[0].px - t).norm();
            sq_err += d * d;
        }
        let rms = (sq_err / truth.len() as f64).sqrt();
        assert!(rms < 0.08, "rms={rms}");

        for c in &candidates {
            let min_d = truth
                .iter()
                .map(|t| (c.px - t).norm())
                .fold(f64::INFINITY, f64::min);
            assert!(
                min_d < 3.0,
                "stray candidate at {:?}, nearest truth {min_d}",
                c.px
            );
        }
    }

    #[test]
    fn test_board_spec_validation() {
        assert!(
            BoardSpec {
                inner_cols: 9,
                inner_rows: 6,
                square_mm: 25.0,
            }
            .validate()
            .is_ok()
        );
        assert!(matches!(
            BoardSpec {
                inner_cols: 8,
                inner_rows: 6,
                square_mm: 25.0,
            }
            .validate(),
            Err(CalibrationError::Param { name: "board", .. })
        ));
        assert!(matches!(
            BoardSpec {
                inner_cols: 2,
                inner_rows: 5,
                square_mm: 25.0,
            }
            .validate(),
            Err(CalibrationError::Param { name: "board", .. })
        ));
        assert!(matches!(
            BoardSpec {
                inner_cols: 9,
                inner_rows: 6,
                square_mm: 0.0,
            }
            .validate(),
            Err(CalibrationError::Param { name: "board", .. })
        ));
    }

    #[test]
    fn test_render_board_truth_matches_square_colours() {
        let intr = fixture_ir_intrinsics();
        let spec = BoardSpec {
            inner_cols: 9,
            inner_rows: 6,
            square_mm: 25.0,
        };
        let pose = Isometry3::from_parts(
            Translation3::new(-100.0, -62.5, 350.0),
            UnitQuaternion::identity(),
        );
        let (img, truth) = render_board(&intr, &pose, &spec, 3);
        let view = img.view();
        let dark = truth[0] + Vector2::new(-6.0, -6.0);
        let light = truth[0] + Vector2::new(6.0, -6.0);
        assert!(view.sample(dark.x, dark.y) < 60.0, "dark sample too bright");
        assert!(
            view.sample(light.x, light.y) > 190.0,
            "light sample too dark"
        );
    }
}
