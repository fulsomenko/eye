//! Assembles sub-pixel corner candidates (`corners.rs`) into a complete, consistently
//! indexed checkerboard grid.

use std::collections::{HashMap, VecDeque};
use std::f64::consts::FRAC_1_SQRT_2;

use eye_core::image::GrayView;
use nalgebra::{Point2, Point3, Vector2};

use crate::corners::{BoardSpec, CornerConfig, Smoothed, candidates_in};
use crate::error::CalibrationError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoardCorner {
    pub grid: (u32, u32),
    pub object_mm: Point3<f64>,
    pub image_px: Point2<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoardObservation {
    pub spec: BoardSpec,
    pub corners: Vec<BoardCorner>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectorConfig {
    pub corners: CornerConfig,
    pub match_tolerance: f64,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            corners: CornerConfig::default(),
            match_tolerance: 0.35,
        }
    }
}

/// `Ok(None)`: no complete board in view (normal during capture). `Err`: invalid spec.
pub fn detect_board(
    img: &GrayView<'_>,
    spec: &BoardSpec,
    cfg: &DetectorConfig,
) -> Result<Option<BoardObservation>, CalibrationError> {
    spec.validate()?;
    let (cols, rows) = (spec.inner_cols as usize, spec.inner_rows as usize);

    let l = Smoothed::new(img, cfg.corners.blur_sigma_px);
    let c = candidates_in(&l, &cfg.corners);
    if c.len() < cols * rows {
        return Ok(None);
    }
    let pts: Vec<Point2<f64>> = c.iter().map(|cand| cand.px).collect();

    let Some((seed, u0, v0)) = seed_basis(&pts) else {
        return Ok(None);
    };
    let Some(grid_map) = grow_grid(&pts, seed, u0, v0, cfg.match_tolerance) else {
        return Ok(None);
    };
    let Some(mut grid) = normalize_extents(&grid_map, cols, rows) else {
        return Ok(None);
    };

    if !is_right_handed(&pts[grid[0][0]], &pts[grid[1][0]], &pts[grid[0][1]]) {
        grid = relabel(&grid, cols, rows, false, true);
    }

    let (p00, p10, p01) = (pts[grid[0][0]], pts[grid[1][0]], pts[grid[0][1]]);
    let (w, h) = (f64::from(img.width()), f64::from(img.height()));
    let Some(flip) = half_turn_check(&l, &p00, &p10, &p01, w, h) else {
        return Ok(None);
    };
    if flip {
        grid = relabel(&grid, cols, rows, true, true);
    }

    let s = spec.square_mm;
    let mut corners = Vec::with_capacity(cols * rows);
    for j in 0..rows {
        for i in 0..cols {
            corners.push(BoardCorner {
                grid: (i as u32, j as u32),
                object_mm: Point3::new(i as f64 * s, j as f64 * s, 0.0),
                image_px: pts[grid[i][j]],
            });
        }
    }

    Ok(Some(BoardObservation {
        spec: *spec,
        corners,
    }))
}

fn nearest(pts: &[Point2<f64>], target: &Point2<f64>) -> Option<(usize, f64)> {
    pts.iter()
        .enumerate()
        .map(|(i, p)| (i, (p - target).norm()))
        .min_by(|a, b| a.1.partial_cmp(&b.1).expect("finite distance"))
}

fn seed_basis(pts: &[Point2<f64>]) -> Option<(usize, Vector2<f64>, Vector2<f64>)> {
    let n = pts.len();
    if n == 0 {
        return None;
    }
    let sum = pts
        .iter()
        .fold(Vector2::new(0.0, 0.0), |acc, p| acc + p.coords);
    let centroid = Point2::from(sum / n as f64);
    let (seed, _) = nearest(pts, &centroid)?;
    let p0 = pts[seed];

    let (u_idx, _) = (0..n)
        .filter(|&k| k != seed)
        .map(|k| (k, (pts[k] - p0).norm()))
        .min_by(|a, b| a.1.partial_cmp(&b.1).expect("finite distance"))?;
    let u = pts[u_idx] - p0;

    let mut best: Option<(usize, f64)> = None;
    for (k, &point) in pts.iter().enumerate() {
        if k == seed {
            continue;
        }
        let off = point - p0;
        let dist = off.norm();
        if dist <= 0.0 {
            continue;
        }
        let cos = u.dot(&off) / (u.norm() * dist);
        if cos.abs() <= FRAC_1_SQRT_2 && best.is_none_or(|(_, best_dist)| dist < best_dist) {
            best = Some((k, dist));
        }
    }
    let (v_idx, _) = best?;
    let v = pts[v_idx] - p0;
    Some((seed, u, v))
}

type QueueNode = ((i32, i32), usize, Vector2<f64>, Vector2<f64>);

/// BFS grid growth from `seed` with local basis `(u0, v0)`. `None` on ambiguity (a
/// coordinate or a candidate claimed by two different partners).
fn grow_grid(
    pts: &[Point2<f64>],
    seed: usize,
    u0: Vector2<f64>,
    v0: Vector2<f64>,
    match_tolerance: f64,
) -> Option<HashMap<(i32, i32), usize>> {
    let mut grid: HashMap<(i32, i32), usize> = HashMap::new();
    let mut used: HashMap<usize, (i32, i32)> = HashMap::new();
    grid.insert((0, 0), seed);
    used.insert(seed, (0, 0));

    let mut queue: VecDeque<QueueNode> = VecDeque::new();
    queue.push_back(((0, 0), seed, u0, v0));

    const STEPS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

    while let Some((coord, idx, u, v)) = queue.pop_front() {
        let p = pts[idx];
        for (di, dj) in STEPS {
            let predicted = p + f64::from(di) * u + f64::from(dj) * v;
            let tol = match_tolerance * u.norm().min(v.norm());
            let Some((m_idx, m_dist)) = nearest(pts, &predicted) else {
                continue;
            };
            if m_dist > tol {
                continue;
            }

            let new_coord = (coord.0 + di, coord.1 + dj);
            if let Some(&existing) = grid.get(&new_coord) {
                if existing != m_idx {
                    return None;
                }
                continue;
            }
            if used.contains_key(&m_idx) {
                return None;
            }

            let matched = pts[m_idx];
            let (mut nu, mut nv) = (u, v);
            if di != 0 {
                nu = (matched - p) / f64::from(di);
            } else {
                nv = (matched - p) / f64::from(dj);
            }
            grid.insert(new_coord, m_idx);
            used.insert(m_idx, new_coord);
            queue.push_back((new_coord, m_idx, nu, nv));
        }
    }
    Some(grid)
}

/// Shifts the grid so its minimum coordinate is `(0, 0)` and reshapes it into
/// `inner_cols x inner_rows` (swapping axes if the BFS grew the transposed way). `None`
/// when the grid has holes or the wrong extents.
fn normalize_extents(
    grid_map: &HashMap<(i32, i32), usize>,
    cols: usize,
    rows: usize,
) -> Option<Vec<Vec<usize>>> {
    let min_i = grid_map.keys().map(|k| k.0).min()?;
    let max_i = grid_map.keys().map(|k| k.0).max()?;
    let min_j = grid_map.keys().map(|k| k.1).min()?;
    let max_j = grid_map.keys().map(|k| k.1).max()?;
    let ni = (max_i - min_i + 1) as usize;
    let nj = (max_j - min_j + 1) as usize;
    if grid_map.len() != ni * nj {
        return None;
    }

    if ni == cols && nj == rows {
        Some(
            (0..cols)
                .map(|i| {
                    (0..rows)
                        .map(|j| grid_map[&(min_i + i as i32, min_j + j as i32)])
                        .collect()
                })
                .collect(),
        )
    } else if ni == rows && nj == cols {
        Some(
            (0..cols)
                .map(|i| {
                    (0..rows)
                        .map(|j| grid_map[&(min_i + j as i32, min_j + i as i32)])
                        .collect()
                })
                .collect(),
        )
    } else {
        None
    }
}

fn relabel(
    grid: &[Vec<usize>],
    cols: usize,
    rows: usize,
    flip_i: bool,
    flip_j: bool,
) -> Vec<Vec<usize>> {
    (0..cols)
        .map(|i| {
            let si = if flip_i { cols - 1 - i } else { i };
            (0..rows)
                .map(|j| {
                    let sj = if flip_j { rows - 1 - j } else { j };
                    grid[si][sj]
                })
                .collect()
        })
        .collect()
}

fn is_right_handed(p00: &Point2<f64>, p10: &Point2<f64>, p01: &Point2<f64>) -> bool {
    let (u, v) = (p10 - p00, p01 - p00);
    u.x * v.y - u.y * v.x > 0.0
}

fn disc_mean(l: &Smoothed, c: &Point2<f64>, r: f64) -> f64 {
    let mut sum = 0.0;
    let mut n = 0.0;
    for k in -2i32..=2 {
        for m in -2i32..=2 {
            if k * k + m * m <= 4 {
                sum += l.sample(&Point2::new(
                    c.x + r * f64::from(k) / 2.0,
                    c.y + r * f64::from(m) / 2.0,
                ));
                n += 1.0;
            }
        }
    }
    sum / n
}

/// `None` when either disc centre lies within `r` of the image border (the outer squares
/// are not fully visible). Otherwise `Some(true)` when the board is indexed rotated by
/// 180 deg (square (0,0) is brighter than square (1,0)).
fn half_turn_check(
    l: &Smoothed,
    p00: &Point2<f64>,
    p10: &Point2<f64>,
    p01: &Point2<f64>,
    w: f64,
    h: f64,
) -> Option<bool> {
    let (u, v) = (p10 - p00, p01 - p00);
    let r = 0.25 * u.norm().min(v.norm());
    let a_centre = p00 - 0.5 * (u + v);
    let b_centre = p00 + 0.5 * (u - v);
    for c in [a_centre, b_centre] {
        let dist_to_border = c.x.min(w - c.x).min(c.y).min(h - c.y);
        if dist_to_border < r {
            return None;
        }
    }
    Some(disc_mean(l, &a_centre, r) > disc_mean(l, &b_centre, r))
}

#[cfg(test)]
mod tests {
    use eye_core::image::GrayImage;
    use eye_geometry::synth::SplitMix64;
    use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};

    use super::*;
    use crate::testutil::{fixture_ir_intrinsics, fixture_rgb_intrinsics, render_board};

    fn board_spec() -> BoardSpec {
        BoardSpec {
            inner_cols: 9,
            inner_rows: 6,
            square_mm: 25.0,
        }
    }

    fn pose(rot: UnitQuaternion<f64>, d: f64) -> Isometry3<f64> {
        let t = rot * Vector3::new(-100.0, -62.5, 0.0) + Vector3::new(0.0, 0.0, d);
        Isometry3::from_parts(Translation3::from(t), rot)
    }

    fn rms(corners: &[BoardCorner], truth: &[Point2<f64>]) -> f64 {
        assert_eq!(corners.len(), truth.len());
        let sq_err: f64 = corners
            .iter()
            .zip(truth)
            .map(|(c, t)| (c.image_px - t).norm_squared())
            .sum();
        (sq_err / truth.len() as f64).sqrt()
    }

    #[test]
    fn test_frontal_board_ir_all_corners_found() {
        let intr = fixture_ir_intrinsics();
        let spec = board_spec();
        let (img, truth) = render_board(&intr, &pose(UnitQuaternion::identity(), 350.0), &spec, 3);

        let observation = detect_board(&img.view(), &spec, &DetectorConfig::default())
            .expect("valid spec")
            .expect("board found");

        assert_eq!(observation.corners.len(), 54);
        assert_eq!(observation.corners[0].grid, (0, 0));
        assert!(rms(&observation.corners, &truth) < 0.1, "rms too high");
    }

    #[test]
    fn test_tilted_board_with_distortion_found() {
        let intr = fixture_ir_intrinsics();
        let spec = board_spec();
        let qy = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 30f64.to_radians());
        let qx = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 40f64.to_radians());
        let (img, truth) = render_board(&intr, &pose(qy * qx, 400.0), &spec, 4);

        let observation = detect_board(&img.view(), &spec, &DetectorConfig::default())
            .expect("valid spec")
            .expect("board found");

        assert_eq!(observation.corners.len(), 54);
        assert!(rms(&observation.corners, &truth) < 0.15, "rms too high");
    }

    #[test]
    fn test_rgb_resolution_board_found() {
        let intr = fixture_rgb_intrinsics();
        let spec = board_spec();
        let (img, truth) = render_board(&intr, &pose(UnitQuaternion::identity(), 700.0), &spec, 6);

        let observation = detect_board(&img.view(), &spec, &DetectorConfig::default())
            .expect("valid spec")
            .expect("board found");

        assert_eq!(observation.corners.len(), 54);
        assert!(rms(&observation.corners, &truth) < 0.15, "rms too high");
    }

    #[test]
    fn test_board_rotated_180_keeps_physical_indexing() {
        let intr = fixture_ir_intrinsics();
        let spec = board_spec();
        let rot = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::PI);

        for seed in 1..=5u64 {
            let (img, truth) = render_board(&intr, &pose(rot, 350.0), &spec, seed);

            let observation = detect_board(&img.view(), &spec, &DetectorConfig::default())
                .unwrap_or_else(|e| panic!("seed={seed}: {e}"))
                .unwrap_or_else(|| panic!("seed={seed}: board not found"));

            assert_eq!(observation.corners.len(), 54, "seed={seed}");
            assert_eq!(observation.corners[0].grid, (0, 0), "seed={seed}");
            let d = (observation.corners[0].image_px - truth[0]).norm();
            assert!(d < 0.5, "seed={seed} d={d}");
        }
    }

    #[test]
    fn test_board_rotated_90_transposed_extents_handled() {
        let intr = fixture_ir_intrinsics();
        let spec = board_spec();
        let rot = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::FRAC_PI_2);
        let (img, truth) = render_board(&intr, &pose(rot, 450.0), &spec, 7);

        let observation = detect_board(&img.view(), &spec, &DetectorConfig::default())
            .expect("valid spec")
            .expect("board found");

        assert_eq!(observation.corners.len(), 54);
        assert!(rms(&observation.corners, &truth) < 0.15, "rms too high");
    }

    #[test]
    fn test_partial_board_returns_none() {
        let intr = fixture_ir_intrinsics();
        let spec = board_spec();
        // Shift the board right so inner column 8 falls off the image while column 7 stays in it.
        let t = Vector3::new(55.0, -62.5, 350.0);
        let camera_from_board =
            Isometry3::from_parts(Translation3::from(t), UnitQuaternion::identity());
        let (img, _truth) = render_board(&intr, &camera_from_board, &spec, 8);

        let observation =
            detect_board(&img.view(), &spec, &DetectorConfig::default()).expect("valid spec");

        assert!(observation.is_none());
    }

    #[test]
    fn test_noise_image_returns_none() {
        let (w, h) = (640u32, 360u32);
        let mut rng = SplitMix64::new(5);
        let data: Vec<u8> = (0..(w as usize * h as usize))
            .map(|_| (rng.uniform() * 256.0) as u8)
            .collect();
        let img = GrayImage::new(w, h, data).expect("w * h bytes");

        let observation = detect_board(&img.view(), &board_spec(), &DetectorConfig::default())
            .expect("valid spec");

        assert!(observation.is_none());
    }

    #[test]
    fn test_even_sum_spec_is_rejected() {
        let spec = BoardSpec {
            inner_cols: 8,
            inner_rows: 6,
            square_mm: 25.0,
        };
        let intr = fixture_ir_intrinsics();
        let img = GrayImage::new(
            intr.width,
            intr.height,
            vec![128u8; (intr.width * intr.height) as usize],
        )
        .expect("w * h bytes");

        let err = detect_board(&img.view(), &spec, &DetectorConfig::default())
            .expect_err("even sum rejected");
        assert!(matches!(err, CalibrationError::Param { name: "board", .. }));
    }
}
