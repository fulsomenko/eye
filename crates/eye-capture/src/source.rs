use eye_core::{CameraInfo, Frame};

use crate::CaptureError;

pub trait FrameSource: Send {
    fn camera(&self) -> &CameraInfo;
    fn next_frame(&mut self) -> Result<Frame, CaptureError>;
}

impl<S: FrameSource + ?Sized> FrameSource for Box<S> {
    fn camera(&self) -> &CameraInfo {
        (**self).camera()
    }

    fn next_frame(&mut self) -> Result<Frame, CaptureError> {
        (**self).next_frame()
    }
}
