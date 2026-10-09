use std::collections::HashSet;

use eye_core::image::{GrayImage, GrayView};
use eye_core::log::field;
use eye_core::{Ellipse2, Measured};
use nalgebra::Point2;

use crate::DetectError;
use crate::image::Roi;
use crate::ir::IrClassicOptions;

#[derive(Debug, Clone, Copy)]
pub struct GlintSearch<'a> {
    pub lit: GrayView<'a>,
    pub pupil: &'a Ellipse2,
    pub options: &'a IrClassicOptions,
}

pub fn find_glint(search: &GlintSearch<'_>) -> Result<Option<Measured<Point2<f64>>>, DetectError> {
    let GlintSearch {
        lit,
        pupil,
        options,
    } = *search;
    let r_pupil = (pupil.semi_major() * pupil.semi_minor()).sqrt();
    let half = (options.glint_search_radius * r_pupil).ceil() as u32 + 1;
    let roi = Roi::around(&pupil.center(), half, lit.width(), lit.height());
    if roi.width == 0 || roi.height == 0 {
        tracing::debug!(
            { field::REASON } = "roi_empty",
            x = pupil.center().x,
            y = pupil.center().y,
            half = u64::from(half),
            "glint search skipped"
        );
        return Ok(None);
    }

    let plateau = plateau_median(lit, pupil);

    let mut best = (roi.x, roi.y, lit.get(roi.x, roi.y));
    for y in roi.y..roi.y + roi.height {
        for x in roi.x..roi.x + roi.width {
            let v = lit.get(x, y);
            if v > best.2 {
                best = (x, y, v);
            }
        }
    }
    let (i, j, peak) = best;
    if f64::from(peak) < f64::from(plateau) + options.glint_min_excess {
        tracing::debug!(
            { field::REASON } = "glint_too_dim",
            peak = u64::from(peak),
            plateau = u64::from(plateau),
            min_excess = options.glint_min_excess,
            "glint rejected"
        );
        return Ok(None);
    }

    let point = if peak == 255 {
        saturated_centroid(lit, roi, (i, j), plateau)
    } else {
        subpixel_peak(lit, i, j, plateau)
    };

    tracing::trace!(
        x = point.x,
        y = point.y,
        peak = u64::from(peak),
        plateau = u64::from(plateau),
        saturated = peak == 255,
        sigma = options.glint_sigma_px,
        "glint"
    );

    Ok(Some(Measured::new(point, options.glint_sigma_px)?))
}

/// A copy of `diff` in which pixels whose centres lie within 1.5 px of `glint` are set to
/// `plateau`.
pub fn mask_glint(
    diff: GrayView<'_>,
    glint: &Point2<f64>,
    plateau: u8,
) -> Result<GrayImage, DetectError> {
    const RADIUS: f64 = 1.5;
    let (w, h) = (diff.width(), diff.height());
    let mut data = diff.data().to_vec();
    // Pixel centres within 1.5 px of a sub-pixel glint g all satisfy
    // floor(g) - 2 <= x <= floor(g) + 2, so a half-width of 2 contains the whole disc.
    let roi = Roi::around(glint, 2, w, h);
    for y in roi.y..roi.y + roi.height {
        for x in roi.x..roi.x + roi.width {
            let (dx, dy) = (f64::from(x) + 0.5 - glint.x, f64::from(y) + 0.5 - glint.y);
            if dx * dx + dy * dy <= RADIUS * RADIUS {
                data[(y * w + x) as usize] = plateau;
            }
        }
    }
    Ok(GrayImage::new(w, h, data)?)
}

fn inside_ellipse(e: &Ellipse2, x: f64, y: f64) -> bool {
    let c = e.center();
    let (dx, dy) = (x - c.x, y - c.y);
    let (ca, sa) = (e.angle().cos(), e.angle().sin());
    let u = dx * ca + dy * sa;
    let v = -dx * sa + dy * ca;
    (u / e.semi_major()).powi(2) + (v / e.semi_minor()).powi(2) <= 1.0
}

pub(crate) fn plateau_median(view: GrayView<'_>, ellipse: &Ellipse2) -> u8 {
    let margin = ellipse.semi_major().ceil() as i64 + 1;
    let (w, h) = (view.width(), view.height());
    let cx = ellipse.center().x.floor() as i64;
    let cy = ellipse.center().y.floor() as i64;
    let x0 = (cx - margin).max(0);
    let x1 = (cx + margin).min(i64::from(w) - 1);
    let y0 = (cy - margin).max(0);
    let y1 = (cy + margin).min(i64::from(h) - 1);

    let mut values = Vec::new();
    for y in y0..=y1 {
        for x in x0..=x1 {
            let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
            if inside_ellipse(ellipse, px, py) {
                values.push(view.get(x as u32, y as u32));
            }
        }
    }
    let unsaturated: Vec<u8> = values.iter().copied().filter(|&v| v < 255).collect();
    let mut values = if unsaturated.is_empty() {
        values
    } else {
        unsaturated
    };
    values.sort_unstable();
    values.get(values.len() / 2).copied().unwrap_or(0)
}

fn log_excess(lit: GrayView<'_>, x: i64, y: i64, plateau: u8) -> f64 {
    let x = x.clamp(0, i64::from(lit.width()) - 1) as u32;
    let y = y.clamp(0, i64::from(lit.height()) - 1) as u32;
    (f64::from(lit.get(x, y)) - f64::from(plateau))
        .max(0.5)
        .ln()
}

fn parabola_offset(l_minus: f64, l_zero: f64, l_plus: f64) -> f64 {
    let denom = l_minus - 2.0 * l_zero + l_plus;
    if denom == 0.0 {
        0.0
    } else {
        (0.5 * (l_minus - l_plus) / denom).clamp(-0.5, 0.5)
    }
}

fn subpixel_peak(lit: GrayView<'_>, i: u32, j: u32, plateau: u8) -> Point2<f64> {
    let (ii, ji) = (i64::from(i), i64::from(j));
    let l = |x: i64, y: i64| log_excess(lit, x, y, plateau);
    let l0 = l(ii, ji);
    let dx = parabola_offset(l(ii - 1, ji), l0, l(ii + 1, ji));
    let dy = parabola_offset(l(ii, ji - 1), l0, l(ii, ji + 1));
    Point2::new(f64::from(i) + 0.5 + dx, f64::from(j) + 0.5 + dy)
}

/// Centroid of `(lit - plateau)` weights over the 4-connected region of saturated (255) pixels
/// containing `seed` (confined to `roi`), dilated by one pixel (8-neighbourhood, confined to the
/// image bounds).
fn saturated_centroid(lit: GrayView<'_>, roi: Roi, seed: (u32, u32), plateau: u8) -> Point2<f64> {
    let mut region: HashSet<(u32, u32)> = HashSet::new();
    let mut stack = vec![seed];
    region.insert(seed);
    while let Some((x, y)) = stack.pop() {
        for (dx, dy) in [(-1i64, 0), (1, 0), (0, -1), (0, 1)] {
            let nx = i64::from(x) + dx;
            let ny = i64::from(y) + dy;
            if nx < i64::from(roi.x) || nx >= i64::from(roi.x + roi.width) {
                continue;
            }
            if ny < i64::from(roi.y) || ny >= i64::from(roi.y + roi.height) {
                continue;
            }
            let (nx, ny) = (nx as u32, ny as u32);
            if lit.get(nx, ny) == 255 && region.insert((nx, ny)) {
                stack.push((nx, ny));
            }
        }
    }

    let mut dilated = region.clone();
    for &(x, y) in &region {
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                let nx = i64::from(x) + dx;
                let ny = i64::from(y) + dy;
                if nx < 0 || ny < 0 || nx >= i64::from(lit.width()) || ny >= i64::from(lit.height())
                {
                    continue;
                }
                dilated.insert((nx as u32, ny as u32));
            }
        }
    }

    let mut sum_w = 0.0;
    let mut sum_wx = 0.0;
    let mut sum_wy = 0.0;
    for &(x, y) in &dilated {
        let w = (f64::from(lit.get(x, y)) - f64::from(plateau)).max(0.0);
        sum_w += w;
        sum_wx += w * (f64::from(x) + 0.5);
        sum_wy += w * (f64::from(y) + 0.5);
    }
    if sum_w > 0.0 {
        Point2::new(sum_wx / sum_w, sum_wy / sum_w)
    } else {
        Point2::new(f64::from(seed.0) + 0.5, f64::from(seed.1) + 0.5)
    }
}

#[cfg(test)]
mod tests {
    use eye_core::Ellipse2;
    use nalgebra::Vector2;
    use proptest::prelude::*;

    use super::*;
    use crate::ir::testutil::SyntheticIr;

    fn pupil_ellipse(center: Point2<f64>, radius: f64) -> Ellipse2 {
        Ellipse2::circle(center, radius).unwrap()
    }

    #[test]
    fn test_glint_offset_from_pupil_is_recovered_within_0_1px() {
        let mut scene = SyntheticIr::default_scene();
        let center = scene.eyes[0].pupil_center;
        let truth = center + Vector2::new(0.7, -0.4);
        scene.eyes[0].glint = Some((truth, 0.6, 255.0));
        let (lit, _dark) = scene.render();

        let options = IrClassicOptions::default();
        let pupil = pupil_ellipse(center, 3.0);
        let search = GlintSearch {
            lit: lit.view(),
            pupil: &pupil,
            options: &options,
        };
        let glint = find_glint(&search).unwrap().expect("glint found");
        assert!((glint.value() - truth).norm() <= 0.1);
    }

    #[test]
    fn test_saturated_glint_uses_region_centroid() {
        let mut scene = SyntheticIr::default_scene();
        let center = scene.eyes[0].pupil_center;
        let truth = center + Vector2::new(0.3, 0.2);
        scene.eyes[0].glint = Some((truth, 1.0, 600.0));
        let (lit, _dark) = scene.render();

        let options = IrClassicOptions::default();
        let pupil = pupil_ellipse(center, 3.0);
        let search = GlintSearch {
            lit: lit.view(),
            pupil: &pupil,
            options: &options,
        };
        let glint = find_glint(&search).unwrap().expect("glint found");
        assert!((glint.value() - truth).norm() <= 0.2);
    }

    #[test]
    fn test_no_glint_rendered_returns_none() {
        let scene = SyntheticIr::default_scene();
        let center = scene.eyes[0].pupil_center;
        let (lit, _dark) = scene.render();

        let options = IrClassicOptions::default();
        let pupil = pupil_ellipse(center, 3.0);
        let search = GlintSearch {
            lit: lit.view(),
            pupil: &pupil,
            options: &options,
        };
        assert_eq!(find_glint(&search).unwrap(), None);
    }

    #[test]
    fn test_glint_outside_search_radius_is_ignored() {
        let mut scene = SyntheticIr::default_scene();
        let center = scene.eyes[0].pupil_center;
        let truth = center + Vector2::new(10.0, 0.0);
        scene.eyes[0].glint = Some((truth, 0.6, 255.0));
        let (lit, _dark) = scene.render();

        let options = IrClassicOptions::default();
        let pupil = pupil_ellipse(center, 3.0);
        let search = GlintSearch {
            lit: lit.view(),
            pupil: &pupil,
            options: &options,
        };
        assert_eq!(find_glint(&search).unwrap(), None);
    }

    #[test]
    fn test_glint_sigma_comes_from_options() {
        let mut scene = SyntheticIr::default_scene();
        let center = scene.eyes[0].pupil_center;
        let truth = center + Vector2::new(0.5, 0.5);
        scene.eyes[0].glint = Some((truth, 0.6, 255.0));
        let (lit, _dark) = scene.render();

        let options = IrClassicOptions {
            glint_sigma_px: 0.7,
            ..IrClassicOptions::default()
        };
        let pupil = pupil_ellipse(center, 3.0);
        let search = GlintSearch {
            lit: lit.view(),
            pupil: &pupil,
            options: &options,
        };
        let glint = find_glint(&search).unwrap().expect("glint found");
        assert_eq!(glint.sigma(), 0.7);
    }

    #[test]
    fn test_logs_glint_too_dim_at_debug() {
        use eye_log::testing::capture_logs;
        use eye_log::{Level as LogLevel, Value};

        let scene = SyntheticIr::default_scene();
        let centers = [scene.eyes[0].pupil_center, scene.eyes[1].pupil_center];
        let (lit, _dark) = scene.render();

        let options = IrClassicOptions::default();
        let (_, logs) = capture_logs(tracing::Level::TRACE, || {
            for center in centers {
                let pupil = pupil_ellipse(center, 3.0);
                let search = GlintSearch {
                    lit: lit.view(),
                    pupil: &pupil,
                    options: &options,
                };
                assert_eq!(find_glint(&search).unwrap(), None);
            }
        });

        let rejected: Vec<_> = logs
            .iter()
            .filter(|r| r.message == "glint rejected")
            .collect();
        assert_eq!(rejected.len(), 2);
        for rec in &rejected {
            assert_eq!(rec.level, LogLevel::Debug);
            assert_eq!(
                rec.fields[field::REASON],
                Value::Str("glint_too_dim".into())
            );
            match (rec.fields["peak"].clone(), rec.fields["plateau"].clone()) {
                (Value::U64(peak), Value::U64(plateau)) => {
                    assert!(peak < plateau + 15, "peak {peak} plateau {plateau}");
                }
                other => panic!("unexpected field types: {other:?}"),
            }
        }
    }

    #[test]
    fn test_mask_glint_only_changes_pixels_within_radius() {
        let diff = GrayImage::new(16, 16, vec![0u8; 16 * 16]).unwrap();
        // 7.65, not 7.6: at 7.6 the pixel (9, 8) sits exactly on the disc boundary
        // (1.2^2 + 0.9^2 = 2.25) and the old sqrt(..) <= 1.5, the new dx*dx + dy*dy <= 2.25,
        // and this test's hypot(..) <= 1.5 agree there only by floating-point luck; at 7.65
        // no pixel centre is within 1e-6 of the boundary.
        let glint = Point2::new(8.3, 7.65);
        let plateau = 200u8;
        let masked = mask_glint(diff.view(), &glint, plateau).unwrap();
        for y in 0..16u32 {
            for x in 0..16u32 {
                let (cx, cy) = (f64::from(x) + 0.5, f64::from(y) + 0.5);
                let inside = (cx - glint.x).hypot(cy - glint.y) <= 1.5;
                let got = masked.view().get(x, y);
                if inside {
                    assert_eq!(got, plateau, "expected ({x},{y}) inside disc to be masked");
                } else {
                    assert_eq!(got, 0, "expected ({x},{y}) outside disc to be unmasked");
                }
            }
        }
    }

    #[test]
    fn test_mask_glint_handles_glint_at_image_edge() {
        let diff = GrayImage::new(16, 16, vec![0u8; 16 * 16]).unwrap();
        let plateau = 200u8;
        for glint in [Point2::new(0.4, 0.4), Point2::new(15.6, 15.6)] {
            let masked = mask_glint(diff.view(), &glint, plateau).unwrap();
            for y in 0..16u32 {
                for x in 0..16u32 {
                    let (cx, cy) = (f64::from(x) + 0.5, f64::from(y) + 0.5);
                    let inside = (cx - glint.x).hypot(cy - glint.y) <= 1.5;
                    let got = masked.view().get(x, y);
                    if inside {
                        assert_eq!(got, plateau, "expected ({x},{y}) inside disc to be masked");
                    } else {
                        assert_eq!(got, 0, "expected ({x},{y}) outside disc to be unmasked");
                    }
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        #[test]
        fn prop_glint_position_recovered(
            dx in -0.8f64..=0.8,
            dy in -0.8f64..=0.8,
            sigma in 0.5f64..=1.0,
        ) {
            let mut scene = SyntheticIr::default_scene();
            let center = scene.eyes[0].pupil_center;
            let truth = center + Vector2::new(dx, dy);
            scene.eyes[0].glint = Some((truth, sigma, 255.0));
            let (lit, _dark) = scene.render();

            let options = IrClassicOptions::default();
            let pupil = pupil_ellipse(center, 3.0);
            let search = GlintSearch {
                lit: lit.view(),
                pupil: &pupil,
                options: &options,
            };
            let glint = find_glint(&search).unwrap();
            prop_assume!(glint.is_some());
            let glint = glint.unwrap();
            prop_assert!((glint.value() - truth).norm() <= 0.1);
        }
    }
}
