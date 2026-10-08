//! Fakes and fixtures shared by the `commands` unit tests.
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use eye_platform::emitter::{MSXU_FACE_AUTH_SELECTOR, MSXU_GUID};
use eye_platform::{
    CameraDevice, CameraKind, CameraProbe, DisplayProbe, EmitterError, ExtensionUnit, FormatInfo,
    FrameSizeInfo, IrEmitter, OutputInfo, ProbeError, Transform, UsbIdentity,
};

pub(crate) struct FakeDisplay(pub(crate) Option<Vec<OutputInfo>>);

impl DisplayProbe for FakeDisplay {
    fn outputs(&self) -> Result<Vec<OutputInfo>, ProbeError> {
        self.0.clone().ok_or(ProbeError::NotHyprland)
    }
}

pub(crate) struct FakeCameras(pub(crate) Option<Vec<CameraDevice>>);

impl CameraProbe for FakeCameras {
    fn cameras(&self) -> Result<Vec<CameraDevice>, ProbeError> {
        self.0.clone().ok_or_else(|| ProbeError::Io {
            path: PathBuf::from("/sys/class/video4linux"),
            source: io::Error::other("fake"),
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FakeEmitter {
    pub(crate) state: Arc<Mutex<bool>>,
    pub(crate) calls: Arc<Mutex<Vec<bool>>>,
    pub(crate) ignore_writes: bool,
}

impl FakeEmitter {
    pub(crate) fn new(initial: bool) -> Self {
        Self {
            state: Arc::new(Mutex::new(initial)),
            calls: Arc::new(Mutex::new(Vec::new())),
            ignore_writes: false,
        }
    }
}

impl IrEmitter for FakeEmitter {
    fn set_enabled(&mut self, on: bool) -> Result<(), EmitterError> {
        self.calls.lock().unwrap().push(on);
        if !self.ignore_writes {
            *self.state.lock().unwrap() = on;
        }
        Ok(())
    }

    fn is_enabled(&self) -> Result<bool, EmitterError> {
        Ok(*self.state.lock().unwrap())
    }
}

pub(crate) fn edp1() -> OutputInfo {
    OutputInfo {
        name: "eDP-1".to_string(),
        make: "AU Optronics".to_string(),
        model: "0x143B".to_string(),
        mode_px: (3840, 2160),
        refresh_hz: 60.025,
        physical_mm: Some((310, 170)),
        scale: 2.0,
        transform: Transform::Normal,
        logical_position: (0, 0),
        logical_size: (1920, 1080),
    }
}

pub(crate) fn latitude_cameras() -> Vec<CameraDevice> {
    vec![
        CameraDevice {
            node: PathBuf::from("/dev/video0"),
            card: "Integrated_Webcam_HD: Integrate".to_string(),
            driver: "uvcvideo".to_string(),
            bus: "usb-0000:00:14.0-6".to_string(),
            kind: CameraKind::Rgb,
            formats: vec![
                FormatInfo {
                    fourcc: "MJPG".to_string(),
                    sizes: vec![
                        FrameSizeInfo {
                            width: 1280,
                            height: 720,
                            fps: vec![30.0],
                        },
                        FrameSizeInfo {
                            width: 960,
                            height: 540,
                            fps: vec![30.0],
                        },
                        FrameSizeInfo {
                            width: 848,
                            height: 480,
                            fps: vec![30.0],
                        },
                        FrameSizeInfo {
                            width: 640,
                            height: 480,
                            fps: vec![30.0],
                        },
                        FrameSizeInfo {
                            width: 640,
                            height: 360,
                            fps: vec![30.0],
                        },
                    ],
                },
                FormatInfo {
                    fourcc: "YUYV".to_string(),
                    sizes: vec![FrameSizeInfo {
                        width: 640,
                        height: 480,
                        fps: vec![30.0],
                    }],
                },
            ],
            usb: Some(UsbIdentity {
                vendor_id: 0x0c45,
                product_id: 0x672c,
                interface: 0,
                sysfs_device: PathBuf::from("/sys/devices/fake-usb"),
            }),
            extension_units: vec![],
            metadata_node: Some(PathBuf::from("/dev/video1")),
        },
        CameraDevice {
            node: PathBuf::from("/dev/video2"),
            card: "Integrated_Webcam_HD: Integrate".to_string(),
            driver: "uvcvideo".to_string(),
            bus: "usb-0000:00:14.0-6".to_string(),
            kind: CameraKind::Ir,
            formats: vec![FormatInfo {
                fourcc: "GREY".to_string(),
                sizes: vec![FrameSizeInfo {
                    width: 640,
                    height: 360,
                    fps: vec![30.0],
                }],
            }],
            usb: Some(UsbIdentity {
                vendor_id: 0x0c45,
                product_id: 0x672c,
                interface: 2,
                sysfs_device: PathBuf::from("/sys/devices/fake-usb"),
            }),
            extension_units: vec![ExtensionUnit {
                interface: 2,
                unit_id: 4,
                guid: MSXU_GUID,
                num_controls: 2,
                selectors: vec![MSXU_FACE_AUTH_SELECTOR, 9],
            }],
            metadata_node: Some(PathBuf::from("/dev/video3")),
        },
    ]
}
