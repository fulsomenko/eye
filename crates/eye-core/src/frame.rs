use std::{fmt, sync::Arc, time::Duration};

use crate::{CameraId, CoreError, Illumination, PixelFormat, Timestamp};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameHeader {
    pub camera: CameraId,
    pub seq: u64,
    pub timestamp: Timestamp,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub illumination: Illumination,
}

#[derive(Clone, PartialEq, Eq)]
pub struct Frame {
    header: FrameHeader,
    data: Arc<[u8]>,
}

impl Frame {
    pub fn new(header: FrameHeader, data: Arc<[u8]>) -> Result<Self, CoreError> {
        let FrameHeader {
            width,
            height,
            format,
            ..
        } = header;
        if width == 0 || height == 0 {
            return Err(CoreError::ZeroDimension { width, height });
        }
        match format.bytes_per_pixel() {
            Some(bpp) => {
                let expected = (width as usize)
                    .checked_mul(height as usize)
                    .and_then(|pixels| pixels.checked_mul(bpp));
                if expected != Some(data.len()) {
                    return Err(CoreError::FrameSize {
                        format,
                        width,
                        height,
                        expected: expected.unwrap_or(usize::MAX),
                        actual: data.len(),
                    });
                }
            }
            None if data.is_empty() => return Err(CoreError::EmptyFrame { format }),
            None => {}
        }
        Ok(Self { header, data })
    }

    pub fn header(&self) -> &FrameHeader {
        &self.header
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn set_illumination(&mut self, illumination: Illumination) {
        self.header.illumination = illumination;
    }

    pub fn into_parts(self) -> (FrameHeader, Arc<[u8]>) {
        (self.header, self.data)
    }
}

impl fmt::Debug for Frame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Frame")
            .field("header", &self.header)
            .field("data_len", &self.data.len())
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameSet {
    frames: Vec<Frame>,
}

impl FrameSet {
    pub fn new(frames: Vec<Frame>) -> Result<Self, CoreError> {
        if frames.is_empty() {
            return Err(CoreError::EmptyFrameSet);
        }
        for (i, frame) in frames.iter().enumerate() {
            let camera = &frame.header.camera;
            if frames[..i]
                .iter()
                .any(|earlier| &earlier.header.camera == camera)
            {
                return Err(CoreError::DuplicateCamera(camera.clone()));
            }
        }
        Ok(Self { frames })
    }

    pub fn single(frame: Frame) -> Self {
        Self {
            frames: vec![frame],
        }
    }

    pub fn frames(&self) -> &[Frame] {
        &self.frames
    }

    pub fn get<Q: AsRef<str> + ?Sized>(&self, camera: &Q) -> Option<&Frame> {
        let camera = camera.as_ref();
        self.frames
            .iter()
            .find(|f| f.header.camera.as_str() == camera)
    }

    pub fn earliest(&self) -> Timestamp {
        self.timestamps()
            .fold(self.frames[0].header.timestamp, Ord::min)
    }

    pub fn spread(&self) -> Duration {
        let latest = self
            .timestamps()
            .fold(self.frames[0].header.timestamp, Ord::max);
        latest.abs_diff(self.earliest())
    }

    pub fn into_frames(self) -> Vec<Frame> {
        self.frames
    }

    fn timestamps(&self) -> impl Iterator<Item = Timestamp> + '_ {
        self.frames.iter().map(|f| f.header.timestamp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(
        camera: &str,
        ts_ns: u64,
        format: PixelFormat,
        width: u32,
        height: u32,
    ) -> FrameHeader {
        FrameHeader {
            camera: CameraId::from(camera),
            seq: 0,
            timestamp: Timestamp::from_nanos(ts_ns),
            width,
            height,
            format,
            illumination: Illumination::Unknown,
        }
    }

    #[test]
    fn test_frame_new_accepts_gray_640x360() {
        let data: Arc<[u8]> = vec![0u8; 230_400].into();
        let frame = Frame::new(header("ir", 0, PixelFormat::Gray8, 640, 360), data);
        assert!(frame.is_ok());
    }

    #[test]
    fn test_frame_new_rejects_short_gray_buffer() {
        let data: Arc<[u8]> = vec![0u8; 230_399].into();
        let err = Frame::new(header("ir", 0, PixelFormat::Gray8, 640, 360), data).unwrap_err();
        assert_eq!(
            err,
            CoreError::FrameSize {
                format: PixelFormat::Gray8,
                width: 640,
                height: 360,
                expected: 230_400,
                actual: 230_399,
            }
        );
    }

    #[test]
    fn test_frame_new_accepts_rgb_exact_size() {
        let data: Arc<[u8]> = vec![0u8; 24].into();
        let frame = Frame::new(header("rgb", 0, PixelFormat::Rgb8, 4, 2), data);
        assert!(frame.is_ok());
    }

    #[test]
    fn test_frame_new_rejects_zero_width() {
        let data: Arc<[u8]> = vec![0u8; 0].into();
        let err = Frame::new(header("ir", 0, PixelFormat::Gray8, 0, 360), data).unwrap_err();
        assert_eq!(
            err,
            CoreError::ZeroDimension {
                width: 0,
                height: 360
            }
        );
    }

    #[test]
    fn test_frame_new_rejects_empty_mjpeg() {
        let data: Arc<[u8]> = vec![].into();
        let err = Frame::new(header("rgb", 0, PixelFormat::Mjpeg, 1280, 720), data).unwrap_err();
        assert_eq!(
            err,
            CoreError::EmptyFrame {
                format: PixelFormat::Mjpeg
            }
        );
    }

    #[test]
    fn test_frame_new_accepts_any_nonempty_mjpeg() {
        let data: Arc<[u8]> = vec![0xFF, 0xD8, 0xFF, 0xD9].into();
        let frame = Frame::new(
            header("rgb", 0, PixelFormat::Mjpeg, 1280, 720),
            data.clone(),
        )
        .unwrap();
        assert_eq!(frame.data(), &*data);
    }

    #[test]
    fn test_frame_new_rejects_overflowing_rgb_size() {
        let data: Arc<[u8]> = vec![0u8; 3].into();
        let err = Frame::new(
            header("rgb", 0, PixelFormat::Rgb8, u32::MAX, u32::MAX),
            data,
        )
        .unwrap_err();
        assert_eq!(
            err,
            CoreError::FrameSize {
                format: PixelFormat::Rgb8,
                width: u32::MAX,
                height: u32::MAX,
                expected: usize::MAX,
                actual: 3,
            }
        );
    }

    #[test]
    fn test_frame_debug_omits_pixel_data() {
        let data: Arc<[u8]> = vec![0u8; 230_400].into();
        let frame = Frame::new(header("ir", 0, PixelFormat::Gray8, 640, 360), data).unwrap();
        let debug = format!("{:?}", frame);
        assert!(debug.contains("data_len: 230400"));
        assert!(debug.len() < 400);
    }

    #[test]
    fn test_set_illumination_keeps_data() {
        let data: Arc<[u8]> = vec![0u8; 230_400].into();
        let mut frame = Frame::new(header("ir", 0, PixelFormat::Gray8, 640, 360), data).unwrap();
        frame.set_illumination(Illumination::IrLit);
        assert_eq!(frame.header().illumination, Illumination::IrLit);
        assert_eq!(frame.data().len(), 230_400);
    }

    #[test]
    fn test_frame_set_rejects_empty() {
        let err = FrameSet::new(vec![]).unwrap_err();
        assert_eq!(err, CoreError::EmptyFrameSet);
    }

    #[test]
    fn test_frame_set_rejects_duplicate_camera() {
        let data: Arc<[u8]> = vec![0u8; 230_400].into();
        let a = Frame::new(header("ir", 0, PixelFormat::Gray8, 640, 360), data.clone()).unwrap();
        let b = Frame::new(header("ir", 1, PixelFormat::Gray8, 640, 360), data).unwrap();
        let err = FrameSet::new(vec![a, b]).unwrap_err();
        assert_eq!(err, CoreError::DuplicateCamera(CameraId::from("ir")));
    }

    #[test]
    fn test_frame_set_get_finds_by_camera_name() {
        let gray: Arc<[u8]> = vec![0u8; 230_400].into();
        let mjpeg: Arc<[u8]> = vec![0xFF, 0xD8, 0xFF, 0xD9].into();
        let ir = Frame::new(header("ir", 0, PixelFormat::Gray8, 640, 360), gray).unwrap();
        let rgb = Frame::new(header("rgb", 0, PixelFormat::Mjpeg, 1280, 720), mjpeg).unwrap();
        let set = FrameSet::new(vec![rgb, ir]).unwrap();
        assert_eq!(set.get("ir").unwrap().header().format, PixelFormat::Gray8);
        assert!(set.get("depth").is_none());
    }

    #[test]
    fn test_frame_set_get_accepts_camera_id() {
        let gray: Arc<[u8]> = vec![0u8; 230_400].into();
        let mjpeg: Arc<[u8]> = vec![0xFF, 0xD8, 0xFF, 0xD9].into();
        let ir = Frame::new(header("ir", 0, PixelFormat::Gray8, 640, 360), gray).unwrap();
        let rgb = Frame::new(header("rgb", 0, PixelFormat::Mjpeg, 1280, 720), mjpeg).unwrap();
        let set = FrameSet::new(vec![rgb, ir]).unwrap();
        let id = CameraId::from("rgb");
        assert_eq!(set.get(&id).unwrap().header().format, PixelFormat::Mjpeg);
    }

    #[test]
    fn test_frame_set_earliest_and_spread() {
        let gray: Arc<[u8]> = vec![0u8; 230_400].into();
        let mjpeg: Arc<[u8]> = vec![0xFF, 0xD8, 0xFF, 0xD9].into();
        let ir = Frame::new(
            header("ir", 1_004_000_000, PixelFormat::Gray8, 640, 360),
            gray,
        )
        .unwrap();
        let rgb = Frame::new(
            header("rgb", 1_000_000_000, PixelFormat::Mjpeg, 1280, 720),
            mjpeg,
        )
        .unwrap();
        let set = FrameSet::new(vec![ir, rgb]).unwrap();
        assert_eq!(set.earliest(), Timestamp::from_nanos(1_000_000_000));
        assert_eq!(set.spread(), Duration::from_millis(4));
    }
}
