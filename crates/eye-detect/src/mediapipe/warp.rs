use nalgebra::Point2;

use crate::image::RgbImage;
use crate::mediapipe::roi::RotatedRect;

pub fn warp_roi_to_tensor(
    image: &RgbImage<'_>,
    roi: &RotatedRect,
    n: usize,
    range: (f32, f32),
    out: &mut [f32],
) {
    let (lo, hi) = range;
    for v in 0..n {
        for u in 0..n {
            let crop = Point2::new(u as f64 + 0.5, v as f64 + 0.5);
            let p = roi.crop_to_image(&crop, n);
            for c in 0..3 {
                let idx = (v * n + u) * 3 + c;
                out[idx] = (image.sample(p.x, p.y, c) / 255.0) as f32 * (hi - lo) + lo;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    fn checker_image(n: u32) -> RgbImage<'static> {
        let mut data = vec![0u8; (n * n) as usize * 3];
        for y in 0..n {
            for x in 0..n {
                let v = ((x + y) * 7 % 256) as u8;
                let idx = (y * n + x) as usize * 3;
                data[idx] = v;
                data[idx + 1] = v;
                data[idx + 2] = v;
            }
        }
        RgbImage::owned(n, n, data)
    }

    #[test]
    fn test_warp_identity_roi_reproduces_image() {
        let image = checker_image(256);
        let roi = RotatedRect {
            center: Point2::new(128.0, 128.0),
            size: 256.0,
            rotation: 0.0,
        };
        let mut out = vec![0.0f32; 256 * 256 * 3];
        warp_roi_to_tensor(&image, &roi, 256, (0.0, 1.0), &mut out);

        for v in 0..256usize {
            for u in 0..256usize {
                let expected = f64::from(image.get(u as u32, v as u32, 0)) / 255.0;
                let idx = (v * 256 + u) * 3;
                assert_abs_diff_eq!(out[idx] as f64, expected, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_warp_rotated_roi_puts_marker_at_expected_crop_px() {
        let mut data = vec![0u8; 1280 * 720 * 3];
        for y in 299..=301u32 {
            for x in 699..=701u32 {
                let idx = (y as usize * 1280 + x as usize) * 3;
                data[idx] = 255;
                data[idx + 1] = 255;
                data[idx + 2] = 255;
            }
        }
        let image = RgbImage::owned(1280, 720, data);
        let roi = RotatedRect {
            center: Point2::new(640.0, 360.0),
            size: 300.0,
            rotation: 0.3,
        };
        let mut out = vec![0.0f32; 256 * 256 * 3];
        warp_roi_to_tensor(&image, &roi, 256, (0.0, 1.0), &mut out);

        let mut best = (0usize, 0usize, -1.0f32);
        for v in 0..256usize {
            for u in 0..256usize {
                let idx = (v * 256 + u) * 3;
                if out[idx] > best.2 {
                    best = (u, v, out[idx]);
                }
            }
        }
        let (iu, iv, _) = best;
        assert!((iu as f64 - 161.82).abs() <= 1.0, "iu was {iu}");
        assert!((iv as f64 - 63.74).abs() <= 1.0, "iv was {iv}");
    }
}
