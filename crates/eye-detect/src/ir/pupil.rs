use eye_core::image::GrayView;
use eye_core::{Ellipse2, Measured};
use nalgebra::Point2;

use crate::DetectError;
use crate::ellipse::fit_ellipse;
use crate::ir::IrClassicOptions;
use crate::ir::blob::Candidate;

fn weighted_centroid(diff: GrayView<'_>, c: &Candidate) -> Point2<f64> {
    let (w, h) = (diff.width(), diff.height());
    let x0 = c.bbox.x.saturating_sub(2);
    let y0 = c.bbox.y.saturating_sub(2);
    let x1 = (c.bbox.x + c.bbox.width + 2).min(w);
    let y1 = (c.bbox.y + c.bbox.height + 2).min(h);

    let mut sum_w = 0.0;
    let mut sum_wx = 0.0;
    let mut sum_wy = 0.0;
    for y in y0..y1 {
        for x in x0..x1 {
            let weight = (f64::from(diff.get(x, y)) - c.iris_mean).max(0.0);
            sum_w += weight;
            sum_wx += weight * (f64::from(x) + 0.5);
            sum_wy += weight * (f64::from(y) + 0.5);
        }
    }
    if sum_w > 0.0 {
        Point2::new(sum_wx / sum_w, sum_wy / sum_w)
    } else {
        c.centroid
    }
}

/// Sub-pixel pupil centre, edge points and ellipse fit for one candidate blob (algorithm steps 6
/// to 8): a weighted centroid, radial edge crossings at the blob/iris midlevel, then a direct
/// ellipse fit, falling back to a circle at the centroid when the fit is implausible.
pub(crate) fn pupil_from_candidate(
    diff: GrayView<'_>,
    c: &Candidate,
    options: &IrClassicOptions,
) -> Result<Measured<Ellipse2>, DetectError> {
    let r = (f64::from(c.area) / std::f64::consts::PI).sqrt();
    let centroid = weighted_centroid(diff, c);

    let level = (diff.sample(centroid.x, centroid.y) + c.iris_mean) / 2.0;
    let max_t = 3.0 * r;
    let mut points = Vec::with_capacity(options.edge_rays);
    for k in 0..options.edge_rays {
        let theta = 2.0 * std::f64::consts::PI * k as f64 / options.edge_rays as f64;
        let (dx, dy) = (theta.cos(), theta.sin());
        let mut prev_sample = diff.sample(centroid.x, centroid.y);
        let mut prev_t = 0.0;
        let mut t = 0.25;
        while t <= max_t {
            let sample = diff.sample(centroid.x + dx * t, centroid.y + dy * t);
            if sample < level {
                let frac = if (prev_sample - sample).abs() > 0.0 {
                    (level - prev_sample) / (sample - prev_sample)
                } else {
                    0.0
                };
                let crossing = prev_t + frac * (t - prev_t);
                points.push(Point2::new(
                    centroid.x + dx * crossing,
                    centroid.y + dy * crossing,
                ));
                break;
            }
            prev_sample = sample;
            prev_t = t;
            t += 0.25;
        }
    }

    let accepted_fit = (points.len() >= 6)
        .then(|| fit_ellipse(&points))
        .flatten()
        .filter(|fit| {
            let center_dist = (fit.ellipse.center() - centroid).norm();
            center_dist <= 1.0
                && fit.ellipse.semi_major() >= 0.5 * r
                && fit.ellipse.semi_major() <= 2.0 * r
                && fit.ellipse.semi_minor() >= 0.5 * r
                && fit.ellipse.semi_minor() <= 2.0 * r
        });

    let (ellipse, sigma) = match accepted_fit {
        Some(fit) => {
            let sigma = (fit.rms_residual * (2.0 / points.len() as f64).sqrt()).max(0.05);
            (fit.ellipse, sigma)
        }
        None => (Ellipse2::circle(centroid, r)?, 0.5),
    };

    Ok(Measured::new(ellipse, sigma)?)
}

/// Picks the pair of candidates whose separation and tilt are plausible for the two eyes
/// (algorithm step 9), preferring the pair with the largest combined contrast.
pub(crate) fn select_pair(
    candidates: &[Candidate],
    options: &IrClassicOptions,
) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize, f64)> = None;
    for i in 0..candidates.len() {
        for j in (i + 1)..candidates.len() {
            let a = &candidates[i];
            let b = &candidates[j];
            let dx = b.centroid.x - a.centroid.x;
            let dy = b.centroid.y - a.centroid.y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < options.pair_separation_px[0] || dist > options.pair_separation_px[1] {
                continue;
            }
            let tilt = fold_to_right_angle(dy.atan2(dx).to_degrees());
            if tilt > options.max_pair_tilt_deg {
                continue;
            }
            let score = a.contrast + b.contrast;
            if best.is_none_or(|(_, _, best_score)| score > best_score) {
                best = Some((i, j, score));
            }
        }
    }
    best.map(|(i, j, _)| (i, j))
}

fn fold_to_right_angle(deg: f64) -> f64 {
    let d = deg.abs() % 180.0;
    if d > 90.0 { 180.0 - d } else { d }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fold_to_right_angle_folds_near_vertical_and_horizontal() {
        assert_eq!(fold_to_right_angle(0.0), 0.0);
        assert_eq!(fold_to_right_angle(170.0), 10.0);
        assert_eq!(fold_to_right_angle(-170.0), 10.0);
        assert_eq!(fold_to_right_angle(90.0), 90.0);
    }
}
