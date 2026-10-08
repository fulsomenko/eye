use eye_core::image::GrayView;
use nalgebra::Point2;

use crate::image::Roi;
use crate::ir::IrClassicOptions;

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct Candidate {
    pub area: u32,
    pub centroid: Point2<f64>,
    pub bbox: Roi,
    pub peak: u8,
    pub iris_mean: f64,
    pub outer_mean: f64,
    pub contrast: f64,
    pub aspect: f64,
}

fn window_reduce_rows(data: &[u8], width: u32, height: u32, half: i64, max: bool) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        let row = &data[y * w..(y + 1) * w];
        for x in 0..w {
            let lo = (x as i64 - half).max(0) as usize;
            let hi = ((x as i64 + half).min(w as i64 - 1)) as usize;
            let reduced = if max {
                row[lo..=hi].iter().copied().max()
            } else {
                row[lo..=hi].iter().copied().min()
            };
            out[y * w + x] = reduced.expect("window is non-empty");
        }
    }
    out
}

fn window_reduce_cols(data: &[u8], width: u32, height: u32, half: i64, max: bool) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let mut out = vec![0u8; w * h];
    for x in 0..w {
        for y in 0..h {
            let lo = (y as i64 - half).max(0) as usize;
            let hi = ((y as i64 + half).min(h as i64 - 1)) as usize;
            let reduced = (lo..=hi)
                .map(|yy| data[yy * w + x])
                .reduce(|a, b| if max { a.max(b) } else { a.min(b) });
            out[y * w + x] = reduced.expect("window is non-empty");
        }
    }
    out
}

fn opening(diff: GrayView<'_>, size: usize) -> Vec<u8> {
    let half = (size / 2) as i64;
    let (w, h) = (diff.width(), diff.height());
    let eroded_rows = window_reduce_rows(diff.data(), w, h, half, false);
    let eroded = window_reduce_cols(&eroded_rows, w, h, half, false);
    let dilated_rows = window_reduce_rows(&eroded, w, h, half, true);
    window_reduce_cols(&dilated_rows, w, h, half, true)
}

fn threshold_level(th: &[u8], percentile: f64, min_threshold: u8) -> u8 {
    let mut hist = [0u32; 256];
    for &v in th {
        hist[v as usize] += 1;
    }
    let n = th.len() as f64;
    let target = percentile / 100.0 * n;
    let mut cum = 0u64;
    let mut p = 255u8;
    for (level, &count) in hist.iter().enumerate() {
        cum += u64::from(count);
        if cum as f64 >= target {
            p = level as u8;
            break;
        }
    }
    min_threshold.max(p.saturating_add(1))
}

fn label_components(th: &[u8], w: u32, h: u32, t: u8) -> Vec<Vec<usize>> {
    let mut labels = vec![0u32; th.len()];
    let mut components: Vec<Vec<usize>> = Vec::new();
    let mut next_label = 1u32;
    let mut stack: Vec<(u32, u32)> = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            if th[idx] < t || labels[idx] != 0 {
                continue;
            }
            let label = next_label;
            next_label += 1;
            labels[idx] = label;
            let mut pixels = vec![idx];
            stack.push((x, y));
            while let Some((cx, cy)) = stack.pop() {
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = i64::from(cx) + dx;
                        let ny = i64::from(cy) + dy;
                        if nx < 0 || ny < 0 || nx >= i64::from(w) || ny >= i64::from(h) {
                            continue;
                        }
                        let (nx, ny) = (nx as u32, ny as u32);
                        let nidx = (ny * w + nx) as usize;
                        if labels[nidx] == 0 && th[nidx] >= t {
                            labels[nidx] = label;
                            pixels.push(nidx);
                            stack.push((nx, ny));
                        }
                    }
                }
            }
            components.push(pixels);
        }
    }
    components
}

fn covariance_aspect(sxx: f64, syy: f64, sxy: f64) -> f64 {
    let trace = sxx + syy;
    let det = sxx * syy - sxy * sxy;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let lambda_max = (trace + disc) / 2.0;
    let lambda_min = (trace - disc) / 2.0;
    if lambda_max <= 0.0 {
        return 0.0;
    }
    (lambda_min.max(0.0) / lambda_max).sqrt()
}

fn annulus_mean(diff: GrayView<'_>, centroid: Point2<f64>, r_min: f64, r_max: f64) -> f64 {
    let (w, h) = (diff.width(), diff.height());
    let margin = r_max.ceil() as i64 + 1;
    let cx = centroid.x.floor() as i64;
    let cy = centroid.y.floor() as i64;
    let x0 = (cx - margin).max(0);
    let x1 = (cx + margin).min(i64::from(w) - 1);
    let y0 = (cy - margin).max(0);
    let y1 = (cy + margin).min(i64::from(h) - 1);

    let mut sum = 0.0;
    let mut count = 0u32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
            let d = ((px - centroid.x).powi(2) + (py - centroid.y).powi(2)).sqrt();
            if d >= r_min && d <= r_max {
                sum += f64::from(diff.get(x as u32, y as u32));
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / f64::from(count)
    }
}

/// Threshold, connected components and the per-blob filters of the IR pupil algorithm (steps 2
/// to 5). Returns the surviving candidates, sorted by peak value before the area/aspect/contrast
/// filters are applied.
pub(crate) fn candidates(diff: GrayView<'_>, options: &IrClassicOptions) -> Vec<Candidate> {
    let (w, h) = (diff.width(), diff.height());
    let bg = opening(diff, options.background_size);
    let th: Vec<u8> = diff
        .data()
        .iter()
        .zip(&bg)
        .map(|(&d, &b)| d.saturating_sub(b))
        .collect();
    let t = threshold_level(&th, options.threshold_percentile, options.min_threshold);

    let mut components: Vec<(Vec<usize>, u8)> = label_components(&th, w, h, t)
        .into_iter()
        .map(|pixels| {
            let peak = pixels.iter().map(|&i| th[i]).max().unwrap_or(0);
            (pixels, peak)
        })
        .collect();
    components.sort_by_key(|c| std::cmp::Reverse(c.1));
    components.truncate(options.max_candidates);

    let mut out = Vec::new();
    for (pixels, peak) in components {
        let area = pixels.len() as u32;
        if area < options.pupil_area_px[0] || area > options.pupil_area_px[1] {
            continue;
        }

        let (mut min_x, mut max_x, mut min_y, mut max_y) = (w, 0u32, h, 0u32);
        for &idx in &pixels {
            let (x, y) = (idx as u32 % w, idx as u32 / w);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        let bbox = Roi {
            x: min_x,
            y: min_y,
            width: max_x - min_x + 1,
            height: max_y - min_y + 1,
        };

        let n = f64::from(area);
        let (mut sx, mut sy) = (0.0, 0.0);
        for &idx in &pixels {
            let (x, y) = (idx as u32 % w, idx as u32 / w);
            sx += f64::from(x) + 0.5;
            sy += f64::from(y) + 0.5;
        }
        let centroid = Point2::new(sx / n, sy / n);

        let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
        for &idx in &pixels {
            let (x, y) = (idx as u32 % w, idx as u32 / w);
            let dx = f64::from(x) + 0.5 - centroid.x;
            let dy = f64::from(y) + 0.5 - centroid.y;
            sxx += dx * dx;
            syy += dy * dy;
            sxy += dx * dy;
        }
        let aspect = covariance_aspect(sxx / n, syy / n, sxy / n);
        if aspect < options.min_aspect {
            continue;
        }

        let r = (n / std::f64::consts::PI).sqrt();
        let iris_mean = annulus_mean(diff, centroid, 1.25 * r, 2.0 * r).max(1.0);
        let outer_mean = annulus_mean(diff, centroid, 2.5 * r, 3.5 * r).max(1.0);

        let blob_mean = pixels
            .iter()
            .map(|&idx| f64::from(diff.data()[idx]))
            .sum::<f64>()
            / n;
        let contrast = blob_mean / iris_mean;
        if contrast < options.min_pupil_contrast {
            continue;
        }
        let iris_ratio = iris_mean / outer_mean;
        if iris_ratio > options.max_iris_ratio {
            continue;
        }

        out.push(Candidate {
            area,
            centroid,
            bbox,
            peak,
            iris_mean,
            outer_mean,
            contrast,
            aspect,
        });
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::testutil::SyntheticIr;

    #[test]
    fn test_candidates_on_default_scene_finds_both_pupils() {
        let scene = SyntheticIr::default_scene();
        let (lit, dark) = scene.render();
        let diff = crate::image::saturating_diff(lit.view(), dark.view()).unwrap();
        let cands = candidates(diff.view(), &IrClassicOptions::default());
        assert_eq!(cands.len(), 2);
    }
}
