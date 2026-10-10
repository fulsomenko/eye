use eye_core::image::GrayView;
use eye_core::log::field;
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

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GateCounts {
    pub area: u64,
    pub aspect: u64,
    pub contrast: u64,
    pub iris_ratio: u64,
    pub truncated: bool,
    pub components_total: u64,
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

fn level_at_count(hist: &[u32; 256], target: u32) -> u8 {
    let mut cum = 0u32;
    for (level, &count) in hist.iter().enumerate() {
        cum += count;
        if cum > target {
            return level as u8;
        }
    }
    255
}

fn threshold_levels(th: &[u8], k_mad: f64, hysteresis: f64, min_threshold: u8) -> (u8, u8) {
    let mut hist = [0u32; 256];
    for &v in th {
        hist[v as usize] += 1;
    }
    let n = th.len() as u32;
    let median = level_at_count(&hist, n / 2);

    let mut dev_hist = [0u32; 256];
    for (level, &count) in hist.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let dev = (level as i32 - i32::from(median)).unsigned_abs() as usize;
        dev_hist[dev.min(255)] += count;
    }
    let mad = level_at_count(&dev_hist, n / 2);

    let raw = f64::from(median) + k_mad * 1.4826 * f64::from(mad);
    let t_hi = min_threshold.max(raw.round().clamp(0.0, 255.0) as u8);
    let t_lo = (f64::from(t_hi) * hysteresis).round().clamp(0.0, 255.0) as u8;
    (t_hi, t_lo)
}

fn label_components(th: &[u8], w: u32, h: u32, t_hi: u8, t_lo: u8) -> Vec<Vec<usize>> {
    let mut labels = vec![0u32; th.len()];
    let mut components: Vec<Vec<usize>> = Vec::new();
    let mut next_label = 1u32;
    let mut stack: Vec<(u32, u32)> = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            if th[idx] < t_hi || labels[idx] != 0 {
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
                        if labels[nidx] == 0 && th[nidx] >= t_lo {
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
/// to 5). Components already smaller than `pupil_area_px[0]` are excluded from the
/// `max_candidates` budget before truncation, since they can never pass the area gate; they are
/// still run through that gate below so rejection counts and traces stay accurate.
pub(crate) fn candidates(
    diff: GrayView<'_>,
    options: &IrClassicOptions,
) -> (Vec<Candidate>, GateCounts) {
    let (w, h) = (diff.width(), diff.height());
    let bg = opening(diff, options.background_size);
    let th: Vec<u8> = diff
        .data()
        .iter()
        .zip(&bg)
        .map(|(&d, &b)| d.saturating_sub(b))
        .collect();
    let (t_hi, t_lo) = threshold_levels(
        &th,
        options.threshold_k_mad,
        options.threshold_hysteresis,
        options.min_threshold,
    );

    let components: Vec<(Vec<usize>, u8)> = label_components(&th, w, h, t_hi, t_lo)
        .into_iter()
        .map(|pixels| {
            let peak = pixels.iter().map(|&i| th[i]).max().unwrap_or(0);
            (pixels, peak)
        })
        .collect();
    let components_total = components.len() as u64;

    let (mut plausible, mut sub_floor): (Vec<_>, Vec<_>) = components
        .into_iter()
        .partition(|(pixels, _)| pixels.len() as u32 >= options.pupil_area_px[0]);
    plausible.sort_by_key(|c| std::cmp::Reverse(c.1));
    let truncated = plausible.len() > options.max_candidates;
    plausible.truncate(options.max_candidates);
    let mut components = plausible;
    components.append(&mut sub_floor);

    let mut out = Vec::new();
    let (mut rejected_area, mut rejected_aspect, mut rejected_contrast, mut rejected_iris_ratio) =
        (0u64, 0u64, 0u64, 0u64);
    for (pixels, peak) in components {
        let area = pixels.len() as u32;

        let n = f64::from(area);
        let (mut sx, mut sy) = (0.0, 0.0);
        for &idx in &pixels {
            let (x, y) = (idx as u32 % w, idx as u32 / w);
            sx += f64::from(x) + 0.5;
            sy += f64::from(y) + 0.5;
        }
        let centroid = Point2::new(sx / n, sy / n);

        if area < options.pupil_area_px[0] || area > options.pupil_area_px[1] {
            rejected_area += 1;
            tracing::trace!(
                { field::REASON } = "area",
                x = centroid.x,
                y = centroid.y,
                value = f64::from(area),
                min = f64::from(options.pupil_area_px[0]),
                max = f64::from(options.pupil_area_px[1]),
                "component rejected"
            );
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
            rejected_aspect += 1;
            tracing::trace!(
                { field::REASON } = "aspect",
                x = centroid.x,
                y = centroid.y,
                value = aspect,
                min = options.min_aspect,
                "component rejected"
            );
            continue;
        }

        let r = (n / std::f64::consts::PI).sqrt();
        let r_ring = r.max(options.min_ring_radius_px);
        let iris_mean = annulus_mean(diff, centroid, 1.25 * r_ring, 2.0 * r_ring).max(1.0);
        let outer_mean = annulus_mean(diff, centroid, 2.5 * r_ring, 3.5 * r_ring).max(1.0);

        let blob_mean = pixels
            .iter()
            .map(|&idx| f64::from(diff.data()[idx]))
            .sum::<f64>()
            / n;
        let contrast = blob_mean / iris_mean;
        if contrast < options.min_pupil_contrast {
            rejected_contrast += 1;
            tracing::trace!(
                { field::REASON } = "contrast",
                x = centroid.x,
                y = centroid.y,
                value = contrast,
                min = options.min_pupil_contrast,
                "component rejected"
            );
            continue;
        }
        let iris_ratio = iris_mean / outer_mean;
        if iris_ratio > options.max_iris_ratio {
            rejected_iris_ratio += 1;
            tracing::trace!(
                { field::REASON } = "iris_ratio",
                x = centroid.x,
                y = centroid.y,
                value = iris_ratio,
                max = options.max_iris_ratio,
                "component rejected"
            );
            continue;
        }

        tracing::trace!(
            area = u64::from(area),
            x = centroid.x,
            y = centroid.y,
            peak = u64::from(peak),
            contrast,
            aspect,
            iris_ratio,
            "candidate"
        );

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

    let counts = GateCounts {
        area: rejected_area,
        aspect: rejected_aspect,
        contrast: rejected_contrast,
        iris_ratio: rejected_iris_ratio,
        truncated,
        components_total,
    };

    tracing::debug!(
        threshold = u64::from(t_hi),
        threshold_lo = u64::from(t_lo),
        components = counts.components_total,
        truncated = counts.truncated,
        rejected_area = counts.area,
        rejected_aspect = counts.aspect,
        rejected_contrast = counts.contrast,
        rejected_iris_ratio = counts.iris_ratio,
        accepted = out.len() as u64,
        "blob candidates"
    );

    (out, counts)
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
        let (cands, _counts) = candidates(diff.view(), &IrClassicOptions::default());
        assert_eq!(cands.len(), 2);
    }

    #[test]
    fn test_noise_specks_do_not_consume_candidate_budget() {
        let scene = SyntheticIr::default_scene();
        let (lit, dark) = scene.render();
        let diff = crate::image::saturating_diff(lit.view(), dark.view()).unwrap();
        let (w, h) = (diff.width(), diff.height());
        let mut data = diff.data().to_vec();

        // 100 single-pixel specks placed well outside the face ellipse (x in [220, 420]
        // around the default scene's face), at full brightness so they out-rank the real
        // pupils once sorted by peak.
        for i in 0..100u32 {
            let x = 10 + (i % 10) * 6;
            let y = 10 + (i / 10) * 6;
            data[(y * w + x) as usize] = 255;
        }

        let diff = eye_core::image::GrayImage::new(w, h, data).unwrap();
        let options = IrClassicOptions {
            max_candidates: 64,
            ..IrClassicOptions::default()
        };
        let (cands, counts) = candidates(diff.view(), &options);
        assert_eq!(
            cands.len(),
            2,
            "expected both real pupils to survive the noise-speck budget, truncated={}, components_total={}, rejected_area={}",
            counts.truncated,
            counts.components_total,
            counts.area
        );
    }

    #[test]
    fn test_logs_blob_candidates_at_debug() {
        use eye_log::testing::capture_logs;
        use eye_log::{Level as LogLevel, Value};

        let scene = SyntheticIr::default_scene();
        let (lit, dark) = scene.render();
        let diff = crate::image::saturating_diff(lit.view(), dark.view()).unwrap();
        let ((cands, _counts), logs) = capture_logs(tracing::Level::TRACE, || {
            candidates(diff.view(), &IrClassicOptions::default())
        });
        assert_eq!(cands.len(), 2);

        let summaries: Vec<_> = logs
            .iter()
            .filter(|r| r.message == "blob candidates")
            .collect();
        assert_eq!(summaries.len(), 1);
        let rec = summaries[0];
        assert_eq!(rec.level, LogLevel::Debug);
        assert_eq!(rec.fields["accepted"], Value::U64(2));
        assert_eq!(rec.fields["truncated"], Value::Bool(false));
        match rec.fields["threshold"] {
            Value::U64(t) => assert!(t >= 30, "threshold {t}"),
            ref other => panic!("expected U64 threshold, got {other:?}"),
        }
        match rec.fields["threshold_lo"] {
            Value::U64(t_lo) => assert!(t_lo >= 15, "threshold_lo {t_lo}"),
            ref other => panic!("expected U64 threshold_lo, got {other:?}"),
        }
        for key in [
            "rejected_area",
            "rejected_aspect",
            "rejected_contrast",
            "rejected_iris_ratio",
        ] {
            assert!(
                matches!(rec.fields[key], Value::U64(_)),
                "expected U64 {key}, got {:?}",
                rec.fields[key]
            );
        }

        let candidate_recs: Vec<_> = logs.iter().filter(|r| r.message == "candidate").collect();
        assert_eq!(candidate_recs.len(), 2);
        for rec in &candidate_recs {
            assert_eq!(rec.level, LogLevel::Trace);
            match (rec.fields["area"].clone(), rec.fields["contrast"].clone()) {
                (Value::U64(area), Value::F64(contrast)) => {
                    assert!((7..=120).contains(&area), "area {area} out of range");
                    assert!(contrast > 1.3, "contrast {contrast} not > 1.3");
                }
                other => panic!("unexpected field types: {other:?}"),
            }
        }
    }

    #[test]
    fn test_small_bright_pupil_passes_iris_ratio() {
        let mut scene = SyntheticIr::default_scene();
        for eye in &mut scene.eyes {
            eye.pupil_radius = 1.2;
        }
        let (lit, dark) = scene.render();
        let diff = crate::image::saturating_diff(lit.view(), dark.view()).unwrap();
        let (cands, counts) = candidates(diff.view(), &IrClassicOptions::default());
        assert_eq!(
            cands.len(),
            2,
            "expected both small pupils to pass, rejected_iris_ratio={}, rejected_area={}",
            counts.iris_ratio,
            counts.area
        );
    }

    #[test]
    fn test_skin_specular_still_rejected_by_iris_ratio() {
        let mut scene = SyntheticIr::default_scene();
        scene.specular = vec![
            (Point2::new(370.0, 240.0), 1.5, 255),
            (Point2::new(411.0, 181.0), 1.5, 255),
        ];
        let (lit, dark) = scene.render();
        let diff = crate::image::saturating_diff(lit.view(), dark.view()).unwrap();
        let (cands, counts) = candidates(diff.view(), &IrClassicOptions::default());
        assert_eq!(cands.len(), 2, "only the two real pupils should survive");
        assert_eq!(
            counts.iris_ratio, 2,
            "both specular highlights must still be rejected by iris_ratio"
        );
    }

    #[test]
    fn test_logs_component_rejected_at_trace_with_gate_and_value() {
        use eye_log::testing::capture_logs;
        use eye_log::{Level as LogLevel, Value};

        let (w, h) = (100u32, 100u32);
        let mut data = vec![0u8; (w * h) as usize];
        for x in 50..53 {
            data[(50 * w + x) as usize] = 50;
        }
        let diff = eye_core::image::GrayImage::new(w, h, data).unwrap();

        let ((cands, counts), logs) = capture_logs(tracing::Level::TRACE, || {
            candidates(diff.view(), &IrClassicOptions::default())
        });
        assert_eq!(cands.len(), 0);
        assert_eq!(counts.area, 1);

        let recs: Vec<_> = logs
            .iter()
            .filter(|r| r.message == "component rejected")
            .collect();
        assert_eq!(recs.len(), 1);
        let rec = recs[0];
        assert_eq!(rec.level, LogLevel::Trace);
        assert_eq!(rec.fields[field::REASON], Value::Str("area".into()));
        assert_eq!(rec.fields["value"], Value::F64(3.0));
        assert_eq!(rec.fields["min"], Value::F64(4.0));
    }

    #[test]
    fn test_threshold_levels_on_flat_image_is_min_threshold() {
        let th = vec![0u8; 230_400];
        assert_eq!(threshold_levels(&th, 6.0, 0.5, 30), (30, 15));
    }

    #[test]
    fn test_threshold_levels_follow_noise_floor() {
        use crate::ir::testutil::Xorshift64Star;

        let make = |sigma: f64| -> Vec<u8> {
            let mut rng = Xorshift64Star::new(7);
            (0..230_400)
                .map(|_| {
                    (20.0 + sigma * rng.next_gaussian())
                        .round()
                        .clamp(0.0, 255.0) as u8
                })
                .collect()
        };

        let th4 = make(4.0);
        let (t_hi4, t_lo4) = threshold_levels(&th4, 6.0, 0.5, 30);
        assert!((43..=51).contains(&t_hi4), "t_hi4 {t_hi4}");
        assert_eq!(t_lo4, (f64::from(t_hi4) * 0.5).round() as u8);

        let th8 = make(8.0);
        let (t_hi8, t_lo8) = threshold_levels(&th8, 6.0, 0.5, 30);
        assert!((60..=70).contains(&t_hi8), "t_hi8 {t_hi8}");
        assert_eq!(t_lo8, (f64::from(t_hi8) * 0.5).round() as u8);

        assert!(
            t_hi8 >= t_hi4 + 10,
            "t_hi8 {t_hi8} should exceed t_hi4 {t_hi4} by at least 10"
        );
    }

    #[test]
    fn test_motion_edge_does_not_raise_threshold_above_pupils() {
        let mut scene = SyntheticIr::default_scene();
        for eye in &mut scene.eyes {
            eye.pupil_level = 70;
        }
        let line_level = scene.skin_level + 60;
        scene.specular = (0..400)
            .map(|i| (Point2::new(120.0 + f64::from(i), 20.0), 0.5, line_level))
            .collect();
        let (lit, dark) = scene.render();
        let diff = crate::image::saturating_diff(lit.view(), dark.view()).unwrap();
        let (cands, counts) = candidates(diff.view(), &IrClassicOptions::default());
        assert_eq!(
            cands.len(),
            2,
            "expected both dim pupils to survive the motion-edge line, rejected_area={}, rejected_aspect={}, rejected_contrast={}, rejected_iris_ratio={}",
            counts.area,
            counts.aspect,
            counts.contrast,
            counts.iris_ratio
        );
    }

    #[test]
    fn test_hysteresis_keeps_full_pupil_area() {
        let mut scene = SyntheticIr::default_scene();
        scene.eyes.truncate(1);
        scene.eyes[0].pupil_level = 66;
        scene.noise_sigma = 0.0;
        let (lit, dark) = scene.render();
        let diff = crate::image::saturating_diff(lit.view(), dark.view()).unwrap();

        let options = IrClassicOptions {
            threshold_hysteresis: 0.5,
            ..IrClassicOptions::default()
        };
        let (cands, _counts) = candidates(diff.view(), &options);
        assert_eq!(cands.len(), 1);
        let hysteresis_area = cands[0].area as usize;

        let (w, h) = (diff.width(), diff.height());
        let bg = opening(diff.view(), options.background_size);
        let th: Vec<u8> = diff
            .data()
            .iter()
            .zip(&bg)
            .map(|(&d, &b)| d.saturating_sub(b))
            .collect();
        let (t_hi, t_lo) = threshold_levels(
            &th,
            options.threshold_k_mad,
            options.threshold_hysteresis,
            options.min_threshold,
        );

        let area_at = |t: u8| -> usize {
            label_components(&th, w, h, t, t)
                .iter()
                .map(Vec::len)
                .max()
                .unwrap_or(0)
        };
        let area_lo = area_at(t_lo);
        let area_hi = area_at(t_hi);

        assert_eq!(hysteresis_area, area_lo);
        assert!(
            area_lo > area_hi,
            "area_lo {area_lo} should exceed area_hi {area_hi}"
        );
    }
}
