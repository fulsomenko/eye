//! Camera frame sources, frame pairing, recording and replay.
#![forbid(unsafe_code)]

pub mod error;
pub mod format;
pub mod illumination;
pub mod pairing;
pub mod session;
pub mod source;
#[cfg(test)]
pub(crate) mod testing;
pub mod uvc_meta;
pub mod v4l2;

pub use error::CaptureError;
pub use format::StoredFormat;
pub use source::FrameSource;
pub use v4l2::{V4l2Config, V4l2Options, V4l2Source};
pub use {
    illumination::{IlluminationTagger, TaggedSource, TaggerConfig, TaggerState, mean_brightness},
    uvc_meta::{IlluminationMeta, MetaRecord, UvcMetaStream, frame_illumination},
};
