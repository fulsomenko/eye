//! Test fixtures shared by the calibration corner-detection and grid-assembly tests.

use eye_core::image::GrayImage;
use eye_geometry::camera::{Distortion, Intrinsics};
use eye_geometry::synth::SplitMix64;
use nalgebra::{Isometry3, Matrix3, Point2, Point3};

use crate::corners::BoardSpec;

pub(crate) fn fixture_ir_intrinsics() -> Intrinsics {
    Intrinsics {
        width: 640,
        height: 360,
        fx: 457.0,
        fy: 457.0,
        cx: 320.0,
        cy: 180.0,
        distortion: Distortion {
            k1: 0.08,
            k2: -0.15,
            p1: 0.001,
            p2: -0.0005,
            k3: 0.05,
        },
    }
}

#[allow(dead_code)]
pub(crate) fn fixture_rgb_intrinsics() -> Intrinsics {
    Intrinsics {
        width: 1280,
        height: 720,
        fx: 914.0,
        fy: 914.0,
        cx: 640.0,
        cy: 360.0,
        distortion: Distortion {
            k1: 0.05,
            ..Distortion::default()
        },
    }
}

/// Colour at board-frame point `(x, y)` on the z=0 plane (mm). Square `(a, b)` is black
/// (30) iff `a + b` is even, so the square above-left of inner corner (0, 0) is black;
/// white (220) otherwise, including the one-square margin around the pattern; grey (128)
/// outside the paper or where the caller passes a point the ray never reaches.
fn board_colour(x: f64, y: f64, spec: &BoardSpec) -> f64 {
    let s = spec.square_mm;
    let a = (x / s).floor() as i64 + 1;
    let b = (y / s).floor() as i64 + 1;
    let cols = i64::from(spec.inner_cols);
    let rows = i64::from(spec.inner_rows);
    if (0..=cols).contains(&a) && (0..=rows).contains(&b) {
        if (a + b) % 2 == 0 { 30.0 } else { 220.0 }
    } else if (-1..=cols + 1).contains(&a) && (-1..=rows + 1).contains(&b) {
        220.0
    } else {
        128.0
    }
}

/// Bilinear interpolation of the `(4i, 4j)` normalized-coordinate lookup table at the
/// continuous pixel point `(px, py)`.
fn sample_table(table: &[(f64, f64)], gw: usize, gh: usize, px: f64, py: f64) -> (f64, f64) {
    let u = (px / 4.0).clamp(0.0, (gw - 1) as f64);
    let v = (py / 4.0).clamp(0.0, (gh - 1) as f64);
    let u0 = u.floor().min((gw - 2) as f64);
    let v0 = v.floor().min((gh - 2) as f64);
    let (ax, ay) = (u - u0, v - v0);
    let (i0, j0) = (u0 as usize, v0 as usize);
    let (i1, j1) = (i0 + 1, j0 + 1);
    let g = |i: usize, j: usize| table[j * gw + i];
    let (g00, g10, g01, g11) = (g(i0, j0), g(i1, j0), g(i0, j1), g(i1, j1));
    let top = (
        g00.0 * (1.0 - ax) + g10.0 * ax,
        g00.1 * (1.0 - ax) + g10.1 * ax,
    );
    let bot = (
        g01.0 * (1.0 - ax) + g11.0 * ax,
        g01.1 * (1.0 - ax) + g11.1 * ax,
    );
    (
        top.0 * (1.0 - ay) + bot.0 * ay,
        top.1 * (1.0 - ay) + bot.1 * ay,
    )
}

/// Rendered image and the true projections of the inner corners, row-major by (j, i).
///
/// Board frame: inner corner `(i, j)` at object `(i*s, j*s, 0)`. Each pixel is the mean of
/// an 8x8 supersampled grid plus Gaussian pixel noise. See `board_colour` for the paper
/// colour convention. Undistortion uses a lookup table sampled every 4 pixels, bilinearly
/// interpolated per sub-sample; the per-sub-sample ray/plane intersection and board-frame
/// transform run on plain `f64`, with no `nalgebra` calls inside the pixel loop.
pub(crate) fn render_board(
    intr: &Intrinsics,
    camera_from_board: &Isometry3<f64>,
    spec: &BoardSpec,
    noise_seed: u64,
) -> (GrayImage, Vec<Point2<f64>>) {
    let (w, h) = (intr.width, intr.height);
    let s = spec.square_mm;

    let gw = (w as usize).div_ceil(4) + 2;
    let gh = (h as usize).div_ceil(4) + 2;
    let mut table = vec![(0.0, 0.0); gw * gh];
    for j in 0..gh {
        for i in 0..gw {
            let px = Point2::new((4 * i) as f64, (4 * j) as f64);
            table[j * gw + i] = intr
                .pixel_to_normalized(&px)
                .map(|n| (n.x, n.y))
                .unwrap_or((0.0, 0.0));
        }
    }

    let rm: Matrix3<f64> = *camera_from_board.rotation.to_rotation_matrix().matrix();
    let t3 = camera_from_board.translation.vector;
    let r = [
        rm[(0, 0)],
        rm[(1, 0)],
        rm[(2, 0)],
        rm[(0, 1)],
        rm[(1, 1)],
        rm[(2, 1)],
        rm[(0, 2)],
        rm[(1, 2)],
        rm[(2, 2)],
    ];
    let t = [t3.x, t3.y, t3.z];
    let normal = [r[6], r[7], r[8]];
    let normal_dot_t = normal[0] * t[0] + normal[1] * t[1] + normal[2] * t[2];

    let mut rng = SplitMix64::new(noise_seed);
    let mut data = vec![0u8; w as usize * h as usize];
    for iy in 0..h {
        for ix in 0..w {
            let mut sum = 0.0f64;
            for sy in 0..8u32 {
                for sx in 0..8u32 {
                    let px = f64::from(ix) + (f64::from(sx) + 0.5) / 8.0;
                    let py = f64::from(iy) + (f64::from(sy) + 0.5) / 8.0;
                    let (nx, ny) = sample_table(&table, gw, gh, px, py);
                    let d = [nx, ny, 1.0];
                    let normal_dot_d = normal[0] * d[0] + normal[1] * d[1] + normal[2] * d[2];
                    let lambda = normal_dot_t / normal_dot_d;
                    sum += if lambda.is_finite() && lambda > 0.0 {
                        let rel = [
                            d[0] * lambda - t[0],
                            d[1] * lambda - t[1],
                            d[2] * lambda - t[2],
                        ];
                        let bx = r[0] * rel[0] + r[1] * rel[1] + r[2] * rel[2];
                        let by = r[3] * rel[0] + r[4] * rel[1] + r[5] * rel[2];
                        board_colour(bx, by, spec)
                    } else {
                        128.0
                    };
                }
            }
            let mean = sum / 64.0;
            let noisy = (mean + 2.0 * rng.gaussian()).round().clamp(0.0, 255.0);
            data[iy as usize * w as usize + ix as usize] = noisy as u8;
        }
    }

    let truth = (0..spec.inner_rows)
        .flat_map(|j| {
            (0..spec.inner_cols).map(move |i| Point3::new(f64::from(i) * s, f64::from(j) * s, 0.0))
        })
        .map(|p| {
            intr.project(&camera_from_board.transform_point(&p))
                .expect("fixture corners stay in front of the camera")
        })
        .collect();

    (GrayImage::new(w, h, data).expect("w * h bytes"), truth)
}

/// 41x41 ideal X-junction at `centre`, edges rotated by `angle_rad`, 4x4 supersampled, 30/220
/// grey. Quadrants alternate colour across the two edges through `centre`, so the image has a
/// saddle point exactly there.
pub(crate) fn x_junction(centre: Point2<f64>, angle_rad: f64, size: u32) -> GrayImage {
    let (cos_a, sin_a) = (angle_rad.cos(), angle_rad.sin());
    let mut data = vec![0u8; (size * size) as usize];
    for iy in 0..size {
        for ix in 0..size {
            let mut sum = 0.0f64;
            for sy in 0..4u32 {
                for sx in 0..4u32 {
                    let px = f64::from(ix) + (f64::from(sx) + 0.5) / 4.0;
                    let py = f64::from(iy) + (f64::from(sy) + 0.5) / 4.0;
                    let (dx, dy) = (px - centre.x, py - centre.y);
                    let u = dx * cos_a + dy * sin_a;
                    let v = dy * cos_a - dx * sin_a;
                    sum += if (u >= 0.0) == (v >= 0.0) {
                        220.0
                    } else {
                        30.0
                    };
                }
            }
            let mean = (sum / 16.0).round().clamp(0.0, 255.0);
            data[(iy * size + ix) as usize] = mean as u8;
        }
    }
    GrayImage::new(size, size, data).expect("w * h bytes")
}
