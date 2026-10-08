use std::{collections::VecDeque, sync::Arc, time::Duration};

use eye_core::{CameraId, CameraInfo, Frame, FrameHeader, Illumination, PixelFormat, Timestamp};

use crate::{CaptureError, source::FrameSource};

#[derive(Debug)]
pub(crate) struct VecSource {
    info: CameraInfo,
    frames: VecDeque<Frame>,
}

impl VecSource {
    pub(crate) fn new(info: CameraInfo, frames: impl IntoIterator<Item = Frame>) -> Self {
        Self {
            info,
            frames: frames.into_iter().collect(),
        }
    }
}

impl FrameSource for VecSource {
    fn camera(&self) -> &CameraInfo {
        &self.info
    }

    fn next_frame(&mut self) -> Result<Frame, CaptureError> {
        self.frames.pop_front().ok_or(CaptureError::EndOfStream)
    }
}

/// Nominal frame_interval 33_333_333 ns.
pub(crate) fn camera_info(id: &str, format: PixelFormat, width: u32, height: u32) -> CameraInfo {
    CameraInfo {
        id: CameraId::from(id),
        format,
        width,
        height,
        frame_interval: Duration::from_nanos(33_333_333),
    }
}

/// Uniform Gray8 image of `value`, illumination Unknown.
pub(crate) fn gray_frame(
    camera: &str,
    seq: u64,
    t_ns: u64,
    width: u32,
    height: u32,
    value: u8,
) -> Frame {
    let header = FrameHeader {
        camera: CameraId::from(camera),
        seq,
        timestamp: Timestamp::from_nanos(t_ns),
        width,
        height,
        format: PixelFormat::Gray8,
        illumination: Illumination::Unknown,
    };
    Frame::new(header, Arc::from(vec![value; (width * height) as usize])).unwrap()
}

/// 1280x720 Mjpeg with the given bytes, illumination Ambient.
#[allow(dead_code)]
pub(crate) fn mjpeg_frame(camera: &str, seq: u64, t_ns: u64, bytes: &[u8]) -> Frame {
    let header = FrameHeader {
        camera: CameraId::from(camera),
        seq,
        timestamp: Timestamp::from_nanos(t_ns),
        width: 1280,
        height: 720,
        format: PixelFormat::Mjpeg,
        illumination: Illumination::Ambient,
    };
    Frame::new(header, Arc::from(bytes)).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_source_ends_with_end_of_stream() {
        let info = camera_info("ir", PixelFormat::Gray8, 640, 360);
        let frames = [
            gray_frame("ir", 0, 0, 640, 360, 1),
            gray_frame("ir", 1, 33_333_333, 640, 360, 2),
        ];
        let mut source = VecSource::new(info, frames);

        let first = source.next_frame().unwrap();
        assert_eq!(first.header().seq, 0);
        let second = source.next_frame().unwrap();
        assert_eq!(second.header().seq, 1);
        assert!(matches!(
            source.next_frame(),
            Err(CaptureError::EndOfStream)
        ));
    }
}
