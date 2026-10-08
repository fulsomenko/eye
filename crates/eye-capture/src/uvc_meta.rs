use std::{
    io,
    path::{Path, PathBuf},
    time::Duration,
};

use eye_core::Timestamp;
use v4l::{Device, buffer::Type, io::traits::CaptureStream};

use crate::CaptureError;

/// Microsoft UVC metadata item id of `MetadataId_FrameIllumination`.
pub const MS_FRAME_ILLUMINATION: u32 = 6;

/// One dequeued metadata buffer: its V4L2 buffer timestamp (equal to the video frame's) and the
/// FrameIllumination flag, `None` when the buffer carries no such item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetaRecord {
    pub timestamp: Timestamp,
    pub lit: Option<bool>,
}

pub trait IlluminationMeta: Send + std::fmt::Debug {
    fn next_record(&mut self) -> Result<MetaRecord, CaptureError>;
}

/// Walks the `uvc_meta_buf` entries of one UVCM buffer and returns the first FrameIllumination
/// flag (bit 0: 1 = lit). `None` for UVCH buffers, empty or malformed input.
pub fn frame_illumination(buf: &[u8]) -> Option<bool> {
    let mut i = 0;
    while i + 12 <= buf.len() {
        let length = usize::from(buf[i + 10]);
        let end = i + 10 + length;
        if length < 2 || end > buf.len() {
            return None;
        }
        let header = &buf[i + 10..end];
        let info = header[1];
        let mut j = 2 + if info & 0x04 != 0 { 4 } else { 0 } + if info & 0x08 != 0 { 6 } else { 0 };
        while j + 8 <= header.len() {
            let id = u32::from_le_bytes(header[j..j + 4].try_into().ok()?);
            let size = u32::from_le_bytes(header[j + 4..j + 8].try_into().ok()?) as usize;
            if size < 8 || j + size > header.len() {
                break;
            }
            if id == MS_FRAME_ILLUMINATION && size >= 12 {
                let flags = u32::from_le_bytes(header[j + 8..j + 12].try_into().ok()?);
                return Some(flags & 1 == 1);
            }
            j += size;
        }
        i = end;
    }
    None
}

/// The metadata node of a UVC function (`/dev/video3` for the IR camera), streamed with the safe v4l API.
pub struct UvcMetaStream {
    path: PathBuf,
    timeout: Duration,
    stream: v4l::io::mmap::Stream<'static>,
}

impl UvcMetaStream {
    /// 4 mmap buffers of type `MetaCapture`; `timeout` bounds every dequeue.
    pub fn open(path: &Path, timeout: Duration) -> Result<Self, CaptureError> {
        let open_err = |source| CaptureError::Open {
            path: path.to_owned(),
            source,
        };
        let dev = Device::with_path(path).map_err(open_err)?;
        let mut stream =
            v4l::io::mmap::Stream::with_buffers(&dev, Type::MetaCapture, 4).map_err(open_err)?;
        stream.set_timeout(timeout);
        Ok(Self {
            path: path.to_owned(),
            timeout,
            stream,
        })
    }
}

impl IlluminationMeta for UvcMetaStream {
    fn next_record(&mut self) -> Result<MetaRecord, CaptureError> {
        let camera = self.path.display().to_string();
        let (buf, meta) = match CaptureStream::next(&mut self.stream) {
            Ok(next) => next,
            Err(e) if e.kind() == io::ErrorKind::TimedOut => {
                return Err(CaptureError::Timeout {
                    camera,
                    timeout: self.timeout,
                });
            }
            Err(source) => return Err(CaptureError::Io { camera, source }),
        };
        let used = (meta.bytesused as usize).min(buf.len());
        Ok(MetaRecord {
            timestamp: Timestamp(Duration::from(meta.timestamp)),
            lit: frame_illumination(&buf[..used]),
        })
    }
}

impl std::fmt::Debug for UvcMetaStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UvcMetaStream")
            .field("path", &self.path)
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    const LIT_ENTRY: &str =
        "00cfec3a13c50100de071c8c096b35618ee73561eb0506000000100000000100000000000000";
    const DARK_ENTRY: &str =
        "b396fa3e13c5010022001c8d09ad4461412745612d0606000000100000000000000000000000";
    const STD_ENTRY: &str = "84d4ec3a13c50100de070c8c096b3561e1ee3561eb05";

    #[test]
    fn test_real_lit_buffer_is_lit() {
        let mut buf = hex(LIT_ENTRY);
        buf.extend(hex(STD_ENTRY));
        buf.extend(hex(STD_ENTRY));
        assert_eq!(frame_illumination(&buf), Some(true));
    }

    #[test]
    fn test_real_dark_buffer_is_dark() {
        let mut buf = hex(DARK_ENTRY);
        buf.extend(hex(STD_ENTRY));
        assert_eq!(frame_illumination(&buf), Some(false));
    }

    #[test]
    fn test_uvch_buffer_without_ms_items_is_none() {
        let mut buf = hex(STD_ENTRY);
        buf.extend(hex(STD_ENTRY));
        assert_eq!(frame_illumination(&buf), None);
        assert_eq!(frame_illumination(&[]), None);
    }

    #[test]
    fn test_truncated_entry_is_none() {
        let buf = hex(LIT_ENTRY);
        assert_eq!(frame_illumination(&buf[..30]), None);
    }

    #[test]
    fn test_item_in_second_entry_is_found() {
        let mut buf = hex(STD_ENTRY);
        buf.extend(hex(DARK_ENTRY));
        assert_eq!(frame_illumination(&buf), Some(false));
    }

    #[test]
    fn test_meta_stream_is_send() {
        fn f<T: Send>() {}
        f::<UvcMetaStream>();
    }
}
