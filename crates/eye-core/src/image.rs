use crate::{CoreError, Frame, PixelFormat};

/// Borrowed row-major 8-bit gray image without row padding (`data.len() == width * height`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrayView<'a> {
    width: u32,
    height: u32,
    data: &'a [u8],
}

impl<'a> GrayView<'a> {
    pub fn new(width: u32, height: u32, data: &'a [u8]) -> Result<Self, CoreError> {
        if width == 0 || height == 0 {
            return Err(CoreError::ZeroDimension { width, height });
        }
        let expected = (width as usize).saturating_mul(height as usize);
        if data.len() != expected {
            return Err(CoreError::ImageSize {
                width,
                height,
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    pub fn from_frame(frame: &'a Frame) -> Result<Self, CoreError> {
        let header = frame.header();
        if header.format != PixelFormat::Gray8 {
            return Err(CoreError::NotGray(header.format));
        }
        Self::new(header.width, header.height, frame.data())
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Panics if `(x, y)` is outside the image.
    pub fn get(&self, x: u32, y: u32) -> u8 {
        assert!(
            x < self.width && y < self.height,
            "pixel ({x}, {y}) outside {}x{}",
            self.width,
            self.height
        );
        self.data[y as usize * self.width as usize + x as usize]
    }

    /// Panics if `y >= height`.
    pub fn row(&self, y: u32) -> &'a [u8] {
        let w = self.width as usize;
        let start = y as usize * w;
        &self.data[start..start + w]
    }

    /// Bilinear sample at continuous image coordinates (pixel centre at +0.5). Coordinates
    /// outside the image clamp to the border pixels.
    pub fn sample(&self, x: f64, y: f64) -> f64 {
        let (w, h) = (self.width as usize, self.height as usize);
        let (u, v) = (x - 0.5, y - 0.5);
        let (u0, v0) = (u.floor(), v.floor());
        let (fx, fy) = (u - u0, v - v0);
        let clamp = |i: f64, n: usize| i.clamp(0.0, (n - 1) as f64) as usize;
        let (x0, x1) = (clamp(u0, w), clamp(u0 + 1.0, w));
        let (y0, y1) = (clamp(v0, h), clamp(v0 + 1.0, h));
        let px = |xi: usize, yi: usize| f64::from(self.data[yi * w + xi]);
        let top = px(x0, y0) * (1.0 - fx) + px(x1, y0) * fx;
        let bottom = px(x0, y1) * (1.0 - fx) + px(x1, y1) * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

/// Owned row-major 8-bit gray image; same layout as [`GrayView`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrayImage {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

impl GrayImage {
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Result<Self, CoreError> {
        GrayView::new(width, height, &data)?;
        Ok(Self {
            width,
            height,
            data,
        })
    }

    pub fn view(&self) -> GrayView<'_> {
        GrayView {
            width: self.width,
            height: self.height,
            data: &self.data,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn into_data(self) -> Vec<u8> {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    use super::*;
    use crate::{CameraId, FrameHeader, Illumination, Timestamp};

    fn gray_frame(width: u32, height: u32, data: Vec<u8>) -> Frame {
        Frame::new(
            FrameHeader {
                camera: CameraId::from("ir"),
                seq: 0,
                timestamp: Timestamp::from_nanos(0),
                width,
                height,
                format: PixelFormat::Gray8,
                illumination: Illumination::Unknown,
            },
            data.into(),
        )
        .unwrap()
    }

    fn rgb_frame(width: u32, height: u32, data: Vec<u8>) -> Frame {
        Frame::new(
            FrameHeader {
                camera: CameraId::from("rgb"),
                seq: 0,
                timestamp: Timestamp::from_nanos(0),
                width,
                height,
                format: PixelFormat::Rgb8,
                illumination: Illumination::Unknown,
            },
            data.into(),
        )
        .unwrap()
    }

    #[test]
    fn test_gray_view_rejects_wrong_buffer_length() {
        let data = [0u8; 100];
        assert_eq!(
            GrayView::new(640, 360, &data),
            Err(CoreError::ImageSize {
                width: 640,
                height: 360,
                expected: 230_400,
                actual: 100,
            })
        );
    }

    #[test]
    fn test_gray_view_rejects_zero_dimension() {
        assert_eq!(
            GrayView::new(0, 4, &[]),
            Err(CoreError::ZeroDimension {
                width: 0,
                height: 4
            })
        );
    }

    #[test]
    fn test_gray_view_from_rgb_frame_is_not_gray() {
        let frame = rgb_frame(2, 1, vec![0u8; 6]);
        assert_eq!(
            GrayView::from_frame(&frame),
            Err(CoreError::NotGray(PixelFormat::Rgb8))
        );
    }

    #[test]
    fn test_gray_view_from_gray_frame_shares_pixels() {
        let frame = gray_frame(3, 2, vec![1, 2, 3, 4, 5, 6]);
        let view = GrayView::from_frame(&frame).unwrap();
        assert_eq!(view.width(), 3);
        assert_eq!(view.height(), 2);
        assert_eq!(view.row(1), [4, 5, 6]);
        assert_eq!(view.get(2, 0), 3);
        assert_eq!(view.data().as_ptr(), frame.data().as_ptr());
    }

    #[test]
    fn test_sample_at_pixel_centre_returns_pixel_value() {
        let data = vec![10, 20, 30, 40, 50, 60, 70, 80, 90];
        let image = GrayImage::new(3, 3, data).unwrap();
        let view = image.view();
        assert_abs_diff_eq!(view.sample(1.5, 1.5), 50.0);
        assert_abs_diff_eq!(view.sample(0.5, 2.5), 70.0);
    }

    #[test]
    fn test_sample_midway_between_centres_averages() {
        let data = vec![0, 100, 200, 0, 100, 200];
        let image = GrayImage::new(3, 2, data).unwrap();
        let view = image.view();
        assert_abs_diff_eq!(view.sample(1.0, 0.5), 50.0);
        assert_abs_diff_eq!(view.sample(2.0, 1.0), 150.0);
    }

    #[test]
    fn test_sample_outside_clamps_to_border() {
        let data = vec![0, 100, 200, 0, 100, 200];
        let image = GrayImage::new(3, 2, data).unwrap();
        let view = image.view();
        assert_abs_diff_eq!(view.sample(-5.0, 0.5), 0.0);
        assert_abs_diff_eq!(view.sample(10.0, 9.0), 200.0);
    }

    #[test]
    fn test_gray_image_new_rejects_short_buffer_and_views_back() {
        let err = GrayImage::new(2, 2, vec![0; 3]).unwrap_err();
        assert!(matches!(
            err,
            CoreError::ImageSize {
                expected: 4,
                actual: 3,
                ..
            }
        ));

        let image = GrayImage::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(image.view().get(1, 1), 4);
        assert_eq!(image.into_data(), vec![1, 2, 3, 4]);
    }

    proptest! {
        #[test]
        fn prop_sample_is_within_neighbour_range(
            data in prop::collection::vec(0u8..=255, 64),
            x in 0.0f64..8.0,
            y in 0.0f64..8.0,
        ) {
            let min = *data.iter().min().unwrap() as f64;
            let max = *data.iter().max().unwrap() as f64;
            let image = GrayImage::new(8, 8, data).unwrap();
            let sampled = image.view().sample(x, y);
            prop_assert!(sampled >= min - 1e-9 && sampled <= max + 1e-9);
        }
    }
}
