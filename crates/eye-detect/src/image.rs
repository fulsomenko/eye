use eye_core::{
    CoreError, Frame, PixelFormat,
    image::{GrayImage, GrayView},
};
use nalgebra::Point2;

use crate::{DetectError, mjpeg::decode_mjpeg_rgb};

#[derive(Debug, Clone, PartialEq)]
pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl RgbImage {
    pub fn from_frame(frame: &Frame) -> Result<Self, DetectError> {
        let header = frame.header();
        match header.format {
            PixelFormat::Rgb8 => Ok(RgbImage {
                width: header.width,
                height: header.height,
                data: frame.data().to_vec(),
            }),
            PixelFormat::Mjpeg => {
                let decoded = decode_mjpeg_rgb(frame.data())?;
                if (decoded.width, decoded.height) != (header.width, header.height) {
                    return Err(DetectError::Decode(format!(
                        "decoded {}x{} does not match header {}x{}",
                        decoded.width, decoded.height, header.width, header.height
                    )));
                }
                Ok(decoded)
            }
            PixelFormat::Gray8 => Err(DetectError::UnsupportedFormat("Gray8".into())),
        }
    }

    pub fn get(&self, x: u32, y: u32, channel: usize) -> u8 {
        let idx = (y as usize * self.width as usize + x as usize) * 3 + channel;
        self.data[idx]
    }

    pub fn sample(&self, x: f64, y: f64, channel: usize) -> f64 {
        let (w, h) = (self.width as usize, self.height as usize);
        let (u, v) = (x - 0.5, y - 0.5);
        let (u0, v0) = (u.floor(), v.floor());
        let (fx, fy) = (u - u0, v - v0);
        let clamp = |i: f64, n: usize| i.clamp(0.0, (n - 1) as f64) as usize;
        let (x0, x1) = (clamp(u0, w), clamp(u0 + 1.0, w));
        let (y0, y1) = (clamp(v0, h), clamp(v0 + 1.0, h));
        let px = |xi: usize, yi: usize| f64::from(self.get(xi as u32, yi as u32, channel));
        let top = px(x0, y0) * (1.0 - fx) + px(x1, y0) * fx;
        let bottom = px(x0, y1) * (1.0 - fx) + px(x1, y1) * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Roi {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Roi {
    pub fn around(center: &Point2<f64>, half: u32, width: u32, height: u32) -> Roi {
        let px = center.x.floor() as i64;
        let py = center.y.floor() as i64;
        let half = i64::from(half);
        let width_i = i64::from(width);
        let height_i = i64::from(height);

        let x_min = px.saturating_sub(half).clamp(0, width_i);
        let y_min = py.saturating_sub(half).clamp(0, height_i);
        let x_max = px.saturating_add(half).clamp(x_min - 1, width_i - 1);
        let y_max = py.saturating_add(half).clamp(y_min - 1, height_i - 1);

        Roi {
            x: x_min as u32,
            y: y_min as u32,
            width: (x_max - x_min + 1).max(0) as u32,
            height: (y_max - y_min + 1).max(0) as u32,
        }
    }
}

pub fn rgb_to_gray(rgb: &RgbImage) -> Result<GrayImage, DetectError> {
    let gray: Vec<u8> = rgb
        .data
        .as_chunks::<3>()
        .0
        .iter()
        .map(|&[r, g, b]| {
            ((77 * u32::from(r) + 150 * u32::from(g) + 29 * u32::from(b) + 128) >> 8) as u8
        })
        .collect();
    Ok(GrayImage::new(rgb.width, rgb.height, gray)?)
}

pub fn crop(view: GrayView<'_>, roi: Roi) -> Result<GrayImage, DetectError> {
    let (width, height) = (view.width(), view.height());
    let in_bounds = roi.x <= width
        && roi.y <= height
        && roi.width <= width - roi.x
        && roi.height <= height - roi.y;
    if !in_bounds {
        return Ok(GrayImage::new(0, 0, Vec::new())?);
    }
    let mut data = Vec::with_capacity(roi.width as usize * roi.height as usize);
    for y in roi.y..roi.y + roi.height {
        let row = view.row(y);
        let start = roi.x as usize;
        let end = start + roi.width as usize;
        data.extend_from_slice(&row[start..end]);
    }
    Ok(GrayImage::new(roi.width, roi.height, data)?)
}

pub fn saturating_diff(lit: GrayView<'_>, dark: GrayView<'_>) -> Result<GrayImage, DetectError> {
    if (lit.width(), lit.height()) != (dark.width(), dark.height()) {
        return Err(CoreError::ImageSize {
            width: lit.width(),
            height: lit.height(),
            expected: lit.data().len(),
            actual: dark.data().len(),
        }
        .into());
    }
    let data: Vec<u8> = lit
        .data()
        .iter()
        .zip(dark.data())
        .map(|(&l, &d)| l.saturating_sub(d))
        .collect();
    Ok(GrayImage::new(lit.width(), lit.height(), data)?)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use eye_core::{CameraId, FrameHeader, Illumination, Timestamp};
    use proptest::prelude::*;

    use super::*;

    fn frame(format: PixelFormat, width: u32, height: u32, data: Vec<u8>) -> Frame {
        Frame::new(
            FrameHeader {
                camera: CameraId::from("rgb"),
                seq: 0,
                timestamp: Timestamp::from_nanos(0),
                width,
                height,
                format,
                illumination: Illumination::Unknown,
            },
            data.into(),
        )
        .unwrap()
    }

    #[test]
    fn test_rgb_to_gray_uses_bt601_weights() {
        let rgb = RgbImage {
            width: 4,
            height: 1,
            data: vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255],
        };
        let gray = rgb_to_gray(&rgb).unwrap();
        assert_eq!(gray.data(), &[77, 149, 29, 255]);
    }

    #[test]
    fn test_saturating_diff_clamps_and_rejects_size_mismatch() {
        let lit_data = [5u8, 1];
        let dark_data = [1u8, 5];
        let lit = GrayView::new(2, 1, &lit_data).unwrap();
        let dark = GrayView::new(2, 1, &dark_data).unwrap();
        let diff = saturating_diff(lit, dark).unwrap();
        assert_eq!(diff.data(), &[4, 0]);

        let lit2 = GrayView::new(2, 1, &lit_data).unwrap();
        let dark2_data = [1u8, 5];
        let dark2 = GrayView::new(1, 2, &dark2_data).unwrap();
        let err = saturating_diff(lit2, dark2).unwrap_err();
        assert!(matches!(
            err,
            DetectError::Core(CoreError::ImageSize {
                width: 2,
                height: 1,
                ..
            })
        ));
    }

    #[test]
    fn test_rgb_sample_at_pixel_centre_returns_pixel_value() {
        let mut data = Vec::with_capacity(27);
        for i in 0..9u8 {
            data.extend_from_slice(&[i, i, i]);
        }
        let rgb = RgbImage {
            width: 3,
            height: 3,
            data,
        };
        for c in 0..3 {
            assert_abs_diff_eq!(
                rgb.sample(1.5, 1.5, c),
                f64::from(rgb.get(1, 1, c)),
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn test_rgb_sample_midway_between_centres_averages() {
        let rgb = RgbImage {
            width: 2,
            height: 1,
            data: vec![0, 0, 0, 100, 0, 0],
        };
        assert_abs_diff_eq!(rgb.sample(1.0, 0.5, 0), 50.0, epsilon = 1e-12);
        assert_abs_diff_eq!(rgb.sample(1.5, 0.5, 0), 100.0, epsilon = 1e-12);
    }

    #[test]
    fn test_roi_around_clamps_to_image() {
        assert_eq!(
            Roi::around(&Point2::new(2.0, 2.0), 5, 640, 360),
            Roi {
                x: 0,
                y: 0,
                width: 8,
                height: 8
            }
        );
        assert_eq!(
            Roi::around(&Point2::new(100.3, 50.7), 2, 640, 360),
            Roi {
                x: 98,
                y: 48,
                width: 5,
                height: 5
            }
        );
    }

    #[test]
    fn test_crop_copies_roi_rows() {
        let data: Vec<u8> = (0..12).collect();
        let view = GrayView::new(4, 3, &data).unwrap();
        let roi = Roi {
            x: 1,
            y: 1,
            width: 2,
            height: 2,
        };
        let cropped = crop(view, roi).unwrap();
        assert_eq!(cropped.data(), &[5, 6, 9, 10]);
    }

    #[test]
    fn test_crop_roi_outside_view_returns_error_without_panic() {
        let data: Vec<u8> = (0..12).collect();
        let view = GrayView::new(4, 3, &data).unwrap();
        let roi = Roi {
            x: 3,
            y: 0,
            width: 5,
            height: 1,
        };
        assert!(matches!(
            crop(view, roi),
            Err(DetectError::Core(CoreError::ZeroDimension { .. }))
        ));
    }

    #[test]
    fn test_roi_around_centre_beyond_right_edge_stays_in_bounds() {
        let roi = Roi::around(&Point2::new(1000.0, 10.0), 2, 640, 360);
        assert!(roi.x <= 640);
        assert!(roi.y <= 360);
        assert!(roi.x + roi.width <= 640);
        assert!(roi.y + roi.height <= 360);
    }

    #[test]
    fn test_rgb_from_gray_frame_is_unsupported_format() {
        let f = frame(PixelFormat::Gray8, 2, 1, vec![0, 0]);
        let err = RgbImage::from_frame(&f).unwrap_err();
        match err {
            DetectError::UnsupportedFormat(s) => assert_eq!(s, "Gray8"),
            other => panic!("expected UnsupportedFormat, got {other:?}"),
        }
    }

    #[test]
    fn test_rgb_from_mjpeg_frame_with_wrong_header_size_is_decode_error() {
        use image::{ExtendedColorType, codecs::jpeg::JpegEncoder};

        let mut jpeg = Vec::new();
        JpegEncoder::new_with_quality(&mut jpeg, 95)
            .encode(&vec![0u8; 32 * 32 * 3], 32, 32, ExtendedColorType::Rgb8)
            .unwrap();

        let f = frame(PixelFormat::Mjpeg, 64, 64, jpeg);
        let err = RgbImage::from_frame(&f).unwrap_err();
        match err {
            DetectError::Decode(msg) => {
                assert!(msg.contains("32x32"));
                assert!(msg.contains("64x64"));
            }
            other => panic!("expected Decode, got {other:?}"),
        }
    }

    proptest! {
        #[test]
        fn prop_rgb_sample_is_within_neighbour_range(
            data in prop::collection::vec(0u8..=255, 8 * 8 * 3),
            x in 0.0f64..8.0,
            y in 0.0f64..8.0,
        ) {
            let rgb = RgbImage { width: 8, height: 8, data: data.clone() };
            for c in 0..3 {
                let channel_values: Vec<u8> = data.iter().skip(c).step_by(3).copied().collect();
                let min = f64::from(*channel_values.iter().min().unwrap());
                let max = f64::from(*channel_values.iter().max().unwrap());
                let sampled = rgb.sample(x, y, c);
                prop_assert!(sampled >= min - 1e-9 && sampled <= max + 1e-9);
            }
        }
    }
}
