use std::{fmt, path::Path, time::Duration};

use eye::config::Config;
use eye_capture::{
    CaptureError, FrameSource, IlluminationMeta, UvcMetaStream, V4l2Config, V4l2Source,
};
use eye_core::{CameraId, PixelFormat};
use eye_platform::{
    CameraDevice, EmitterError, EmitterGuard, META_FORMAT_UVCM, MsxuIrEmitter, XuError,
    XuTransport, set_meta_format,
};

use crate::mode::{LabEmitter, Role, StreamTarget, Teardown};

/// Metadata dequeue timeout; equal to the video default (`V4l2Config::new`).
pub const META_TIMEOUT: Duration = Duration::from_millis(500);

/// How the host reaches hardware. Real: `V4l2Opener`. Tests: `testkit::FakeOpener`.
pub trait Opener: Send + Sync + fmt::Debug {
    fn open_source(
        &self,
        role: Role,
        target: &StreamTarget,
    ) -> Result<Box<dyn FrameSource>, CaptureError>;
    fn open_emitter(&self, camera: &CameraDevice) -> Result<Box<dyn LabEmitter>, EmitterError>;
    /// Applies `mode` through an `eye_platform::EmitterGuard`, returned as the step's teardown.
    fn guard_emitter(&self, camera: &CameraDevice, mode: u8) -> Result<Teardown, EmitterError>;
    /// Selects UVCM on a metadata node; returns the fourcc the driver now reports.
    fn prepare_meta(&self, node: &Path) -> Result<[u8; 4], XuError>;
    fn open_meta(&self, node: &Path) -> Result<Box<dyn IlluminationMeta>, CaptureError>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct V4l2Opener;

pub fn pixel_format(fourcc: &str) -> Option<PixelFormat> {
    match fourcc {
        "GREY" => Some(PixelFormat::Gray8),
        "MJPG" => Some(PixelFormat::Mjpeg),
        _ => None,
    }
}

/// Camera ids equal the eye config's ids so detector sections (`[detect] ir = ..`) match.
pub fn camera_id(role: Role) -> CameraId {
    CameraId::from(match role {
        Role::Rgb => Config::RGB_CAMERA_ID,
        Role::Ir => Config::IR_CAMERA_ID,
    })
}

impl Opener for V4l2Opener {
    fn open_source(
        &self,
        role: Role,
        target: &StreamTarget,
    ) -> Result<Box<dyn FrameSource>, CaptureError> {
        let format =
            pixel_format(&target.format.fourcc).ok_or_else(|| CaptureError::FormatRejected {
                path: target.node.clone(),
                requested: target.format.to_string(),
                actual: "a fourcc V4l2Source cannot capture".to_owned(),
            })?;
        let mut config = V4l2Config::new(
            camera_id(role),
            target.node.clone(),
            format,
            target.format.width,
            target.format.height,
        );
        config.fps = target.format.fps;
        Ok(Box::new(V4l2Source::open(config)?))
    }

    fn open_emitter(&self, camera: &CameraDevice) -> Result<Box<dyn LabEmitter>, EmitterError> {
        Ok(Box::new(MsxuIrEmitter::discover(camera)?))
    }

    fn guard_emitter(&self, camera: &CameraDevice, mode: u8) -> Result<Teardown, EmitterError> {
        Ok(Box::new(EmitterGuard::apply(camera, mode)?))
    }

    fn prepare_meta(&self, node: &Path) -> Result<[u8; 4], XuError> {
        set_meta_format(node, META_FORMAT_UVCM)
    }

    fn open_meta(&self, node: &Path) -> Result<Box<dyn IlluminationMeta>, CaptureError> {
        Ok(Box::new(UvcMetaStream::open(node, META_TIMEOUT)?))
    }
}

impl<T: XuTransport + Send> LabEmitter for MsxuIrEmitter<T> {
    fn read_mode(&self) -> Result<u8, EmitterError> {
        self.mode()
    }

    fn write_mode(&mut self, mode: u8) -> Result<(), EmitterError> {
        self.set_mode(mode)
    }
}
