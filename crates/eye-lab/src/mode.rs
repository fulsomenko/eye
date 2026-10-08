use std::{any::Any, collections::BTreeMap, fmt, io, path::PathBuf, sync::Arc};

use eye_capture::{CaptureError, FrameSource, IlluminationMeta};
use eye_platform::{CameraDevice, EmitterError};

use crate::sequence::{EmitterSel, ModeSpec, StreamFormat, StreamsSel};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Rgb,
    Ir,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitterSetting {
    Keep,
    On,
    Off,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamTarget {
    pub node: PathBuf,
    pub format: StreamFormat,
}

/// A concrete mode: wildcards resolved, nodes chosen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mode {
    pub emitter: EmitterSetting,
    pub rgb: Option<StreamTarget>,
    pub ir: Option<StreamTarget>,
}

impl Mode {
    pub fn none() -> Self {
        Self {
            emitter: EmitterSetting::Keep,
            rgb: None,
            ir: None,
        }
    }

    pub fn target(&self, role: Role) -> Option<&StreamTarget> {
        match role {
            Role::Rgb => self.rgb.as_ref(),
            Role::Ir => self.ir.as_ref(),
        }
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.rgb, &self.ir) {
            (None, None) => f.write_str("none")?,
            (Some(r), None) => write!(f, "rgb {}", r.format)?,
            (None, Some(i)) => write!(f, "ir {}", i.format)?,
            (Some(r), Some(i)) => write!(f, "dual {} + {}", r.format, i.format)?,
        }
        match self.emitter {
            EmitterSetting::Keep => Ok(()),
            EmitterSetting::On => f.write_str(", emitter on"),
            EmitterSetting::Off => f.write_str(", emitter off"),
        }
    }
}

/// Byte-level access to the IR emitter mode (byte 2 of the MSXU control: 0x01 off, 0x02/0x03 on).
pub trait LabEmitter: Send {
    fn read_mode(&self) -> Result<u8, EmitterError>;
    fn write_mode(&mut self, mode: u8) -> Result<(), EmitterError>;
}

/// What a running test may touch. Shared with the test thread.
pub trait ModeSession: Send + Sync + fmt::Debug {
    fn open(&self, role: Role) -> Result<Box<dyn FrameSource>, CaptureError>;
    /// The IR camera's per-frame FrameIllumination metadata stream; `None` when there is none.
    fn open_meta(&self) -> Result<Option<Box<dyn IlluminationMeta>>, CaptureError> {
        Ok(None)
    }
    fn emitter(&self) -> Result<Box<dyn LabEmitter>, ModeError>;
    fn device(&self, role: Role) -> Option<CameraDevice>;
}

/// Owned by the runner thread and dropped there after the step; its `Drop` restores hardware state.
pub type Teardown = Box<dyn Any + Send>;

#[derive(Debug)]
pub struct ActiveMode {
    pub session: Arc<dyn ModeSession>,
    pub teardown: Option<Teardown>,
}

pub trait ModeHost: fmt::Debug {
    fn expand(&self, spec: &ModeSpec) -> Result<Vec<Mode>, ModeError>;
    fn enter(&mut self, mode: &Mode) -> Result<ActiveMode, ModeError>;
    /// Extra `environment` entries for the report (kernel modules, camera ids).
    fn environment(&self) -> BTreeMap<String, String> {
        BTreeMap::new()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModeError {
    #[error("no {0:?} camera")]
    NoCamera(Role),
    #[error("emitter: {0}")]
    Emitter(#[from] EmitterError),
    #[error("{0}")]
    Unavailable(&'static str),
    #[error("no camera at {0}")]
    NoSuchCamera(PathBuf),
    #[error("{node}: {format} is not offered by the camera")]
    Unsupported { node: PathBuf, format: StreamFormat },
    #[error("{node}: {format} cannot be captured (V4l2Source supports GREY and MJPG)")]
    NotCapturable { node: PathBuf, format: StreamFormat },
    #[error("{0}: no capturable format")]
    NoCapturableFormat(PathBuf),
}

/// The host before EYE-97: only `streams = "none"`, `emitter = "keep"`.
#[derive(Debug, Default)]
pub struct NullHost;

impl ModeHost for NullHost {
    fn expand(&self, spec: &ModeSpec) -> Result<Vec<Mode>, ModeError> {
        if spec.streams == StreamsSel::None && spec.emitter == EmitterSel::Keep {
            Ok(vec![Mode::none()])
        } else {
            Err(ModeError::Unavailable(
                "this eye-lab build has no camera host",
            ))
        }
    }

    fn enter(&mut self, _mode: &Mode) -> Result<ActiveMode, ModeError> {
        Ok(ActiveMode {
            session: Arc::new(NoStreams),
            teardown: None,
        })
    }
}

#[derive(Debug)]
struct NoStreams;

impl ModeSession for NoStreams {
    fn open(&self, role: Role) -> Result<Box<dyn FrameSource>, CaptureError> {
        Err(CaptureError::Open {
            path: PathBuf::from("<none>"),
            source: io::Error::new(
                io::ErrorKind::NotFound,
                format!("mode has no {role:?} stream"),
            ),
        })
    }

    fn emitter(&self) -> Result<Box<dyn LabEmitter>, ModeError> {
        Err(ModeError::NoCamera(Role::Ir))
    }

    fn device(&self, _role: Role) -> Option<CameraDevice> {
        None
    }
}
