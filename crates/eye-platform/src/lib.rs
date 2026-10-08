//! Machine probing (displays, cameras, session) and IR emitter control.
#![deny(unsafe_code)]

pub mod camera;
pub mod display;
pub mod emitter;
pub mod error;
pub mod session;
pub mod uvc;

pub use display::wayland::{SelectedDisplayProbe, WaylandDisplayProbe, select_display_probe};
pub use display::{DisplayProbe, OutputInfo, Transform, hyprland::HyprlandDisplayProbe};
pub use error::ProbeError;
pub use session::{Compositor, SessionInfo, SessionType};

pub use camera::{
    CameraDevice, CameraKind, CameraProbe, FormatInfo, FrameSizeInfo, UsbIdentity, V4l2CameraProbe,
    usb_desc::{ExtensionUnit, Guid, extension_units},
};
pub use emitter::{
    EmitterControl, EmitterError, EmitterGuard, IrEmitter, MsxuIrEmitter, XuOpener,
    find_face_auth_control,
};
pub use uvc::{
    META_FORMAT_UVCM, UvcXuDevice, XuError, XuInfo, XuQuery, XuTransport, meta_format,
    set_meta_format,
};
