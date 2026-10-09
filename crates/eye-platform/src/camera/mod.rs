pub mod sysfs;
pub mod usb_desc;

use std::path::{Path, PathBuf};

use eye_core::log::field;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::{capability::Flags, format::description::Description as FormatDescription};

use crate::ProbeError;
use crate::camera::sysfs::SysfsNode;
use crate::camera::usb_desc::ExtensionUnit;

pub trait CameraProbe {
    fn cameras(&self) -> Result<Vec<CameraDevice>, ProbeError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraKind {
    Rgb,
    Ir,
    Other,
}

impl CameraKind {
    pub(crate) fn name(self) -> &'static str {
        match self {
            CameraKind::Rgb => "rgb",
            CameraKind::Ir => "ir",
            CameraKind::Other => "other",
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct FrameSizeInfo {
    pub width: u32,
    pub height: u32,
    pub fps: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct FormatInfo {
    pub fourcc: String,
    pub sizes: Vec<FrameSizeInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct UsbIdentity {
    pub vendor_id: u16,
    pub product_id: u16,
    pub interface: u8,
    pub sysfs_device: PathBuf,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CameraDevice {
    pub node: PathBuf,
    pub card: String,
    pub driver: String,
    pub bus: String,
    pub kind: CameraKind,
    pub formats: Vec<FormatInfo>,
    pub usb: Option<UsbIdentity>,
    pub extension_units: Vec<ExtensionUnit>,
    pub metadata_node: Option<PathBuf>,
}

impl CameraDevice {
    pub fn supports(&self, fourcc: &str, width: u32, height: u32) -> bool {
        self.formats
            .iter()
            .filter(|f| f.fourcc == fourcc)
            .flat_map(|f| f.sizes.iter())
            .any(|s| s.width == width && s.height == height)
    }
}

#[derive(Debug, Clone)]
pub struct V4l2CameraProbe {
    sysfs_class: PathBuf,
    dev_dir: PathBuf,
}

impl V4l2CameraProbe {
    pub fn new() -> Self {
        Self::with_roots(
            PathBuf::from("/sys/class/video4linux"),
            PathBuf::from("/dev"),
        )
    }

    pub fn with_roots(sysfs_class: PathBuf, dev_dir: PathBuf) -> Self {
        Self {
            sysfs_class,
            dev_dir,
        }
    }
}

impl Default for V4l2CameraProbe {
    fn default() -> Self {
        Self::new()
    }
}

impl CameraProbe for V4l2CameraProbe {
    fn cameras(&self) -> Result<Vec<CameraDevice>, ProbeError> {
        let nodes = sysfs::scan(&self.sysfs_class)?;
        let node_count = nodes.len();

        let mut opened: Vec<(SysfsNode, Device, v4l::capability::Capabilities)> = Vec::new();
        for node in nodes {
            let path = self.dev_dir.join(&node.name);
            let device = match Device::with_path(&path) {
                Ok(device) => device,
                Err(error) => {
                    tracing::warn!(node = %node.name, %error, "failed to open camera node");
                    continue;
                }
            };
            let caps = match device.query_caps() {
                Ok(caps) => caps,
                Err(error) => {
                    tracing::warn!(node = %node.name, %error, "failed to query capabilities");
                    continue;
                }
            };
            opened.push((node, device, caps));
        }

        let meta_nodes: Vec<&SysfsNode> = opened
            .iter()
            .filter(|(_, _, caps)| caps.capabilities.contains(Flags::META_CAPTURE))
            .map(|(node, _, _)| node)
            .collect();

        let mut cameras = Vec::new();
        for (node, device, caps) in &opened {
            if !caps.capabilities.contains(Flags::VIDEO_CAPTURE) {
                tracing::debug!(
                    node = %node.name,
                    meta_capture = caps.capabilities.contains(Flags::META_CAPTURE),
                    { field::REASON } = "no_video_capture",
                    "node skipped, not video capture"
                );
                continue;
            }

            let formats = enumerate_formats(device, &node.name);
            let sizes: usize = formats.iter().map(|f| f.sizes.len()).sum();
            let kind = classify(&formats);
            let metadata_node = metadata_node_for(node, &meta_nodes, &self.dev_dir);

            tracing::info!(
                node = %self.dev_dir.join(&node.name).display(),
                card = %caps.card,
                driver = %caps.driver,
                bus = %caps.bus,
                kind = kind.name(),
                formats = formats.len(),
                sizes,
                usb = node.usb.is_some(),
                vendor_id = node.usb.as_ref().map(|u| u.vendor_id),
                product_id = node.usb.as_ref().map(|u| u.product_id),
                extension_units = node.extension_units.len(),
                metadata_node = metadata_node.is_some(),
                "camera probed"
            );

            cameras.push(CameraDevice {
                node: self.dev_dir.join(&node.name),
                card: caps.card.clone(),
                driver: caps.driver.clone(),
                bus: caps.bus.clone(),
                kind,
                formats,
                usb: node.usb.clone(),
                extension_units: node.extension_units.clone(),
                metadata_node,
            });
        }

        tracing::debug!(
            nodes = node_count,
            opened = opened.len(),
            cameras = cameras.len(),
            "camera probe finished"
        );

        Ok(cameras)
    }
}

fn enumerate_formats(device: &Device, node_name: &str) -> Vec<FormatInfo> {
    let descriptions: Vec<FormatDescription> = match device.enum_formats() {
        Ok(descriptions) => descriptions,
        Err(error) => {
            tracing::warn!(node = node_name, %error, "failed to enumerate formats");
            return Vec::new();
        }
    };

    descriptions
        .into_iter()
        .filter_map(|desc| {
            let fourcc = match desc.fourcc.str() {
                Ok(s) => s.to_string(),
                Err(_) => return None,
            };
            let sizes = enumerate_sizes(device, desc.fourcc, node_name);
            tracing::trace!(
                node = node_name,
                fourcc = %fourcc,
                sizes = sizes.len(),
                "format enumerated"
            );
            Some(FormatInfo { fourcc, sizes })
        })
        .collect()
}

fn enumerate_sizes(
    device: &Device,
    fourcc: v4l::format::FourCC,
    node_name: &str,
) -> Vec<FrameSizeInfo> {
    let sizes = match device.enum_framesizes(fourcc) {
        Ok(sizes) => sizes,
        Err(error) => {
            tracing::warn!(node = node_name, %error, "failed to enumerate frame sizes");
            return Vec::new();
        }
    };

    let mut result = Vec::new();
    for size in sizes {
        for (width, height) in frame_size_bounds(size.size) {
            let fps = enumerate_fps(device, fourcc, width, height, node_name);
            result.push(FrameSizeInfo { width, height, fps });
        }
    }
    result
}

fn frame_size_bounds(size: v4l::framesize::FrameSizeEnum) -> Vec<(u32, u32)> {
    match size {
        v4l::framesize::FrameSizeEnum::Discrete(discrete) => {
            vec![(discrete.width, discrete.height)]
        }
        v4l::framesize::FrameSizeEnum::Stepwise(stepwise) => {
            let min = (stepwise.min_width, stepwise.min_height);
            let max = (stepwise.max_width, stepwise.max_height);
            if min == max {
                vec![min]
            } else {
                vec![min, max]
            }
        }
    }
}

fn enumerate_fps(
    device: &Device,
    fourcc: v4l::format::FourCC,
    width: u32,
    height: u32,
    node_name: &str,
) -> Vec<f64> {
    let intervals = match device.enum_frameintervals(fourcc, width, height) {
        Ok(intervals) => intervals,
        Err(error) => {
            tracing::warn!(node = node_name, %error, "failed to enumerate frame intervals");
            return Vec::new();
        }
    };

    let mut fps = Vec::new();
    for interval in intervals {
        match interval.interval {
            v4l::frameinterval::FrameIntervalEnum::Discrete(fraction) => {
                if fraction.numerator != 0 {
                    fps.push(fraction.denominator as f64 / fraction.numerator as f64);
                }
            }
            v4l::frameinterval::FrameIntervalEnum::Stepwise(stepwise) => {
                let min_fps = if stepwise.max.numerator != 0 {
                    stepwise.max.denominator as f64 / stepwise.max.numerator as f64
                } else {
                    0.0
                };
                let max_fps = if stepwise.min.numerator != 0 {
                    stepwise.min.denominator as f64 / stepwise.min.numerator as f64
                } else {
                    0.0
                };
                fps.push(max_fps);
                fps.push(min_fps);
            }
        }
    }
    fps
}

const LUMA: [&str; 4] = ["GREY", "Y10 ", "Y12 ", "Y16 "];
const COLOUR: [&str; 5] = ["MJPG", "YUYV", "NV12", "RGB3", "BGR3"];

pub(crate) fn classify(formats: &[FormatInfo]) -> CameraKind {
    if !formats.is_empty() && formats.iter().all(|f| LUMA.contains(&f.fourcc.as_str())) {
        CameraKind::Ir
    } else if formats.iter().any(|f| COLOUR.contains(&f.fourcc.as_str())) {
        CameraKind::Rgb
    } else {
        CameraKind::Other
    }
}

pub(crate) fn metadata_node_for(
    node: &SysfsNode,
    metas: &[&SysfsNode],
    dev_dir: &Path,
) -> Option<PathBuf> {
    let usb = node.usb.as_ref()?;
    metas
        .iter()
        .find(|m| {
            m.usb
                .as_ref()
                .is_some_and(|u| u.sysfs_device == usb.sysfs_device && u.interface == usb.interface)
        })
        .map(|m| dev_dir.join(&m.name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_grey_only_is_ir() {
        let formats = vec![FormatInfo {
            fourcc: "GREY".to_string(),
            sizes: vec![],
        }];
        assert_eq!(classify(&formats), CameraKind::Ir);
    }

    #[test]
    fn test_classify_mjpg_yuyv_is_rgb() {
        let formats = vec![
            FormatInfo {
                fourcc: "MJPG".to_string(),
                sizes: vec![],
            },
            FormatInfo {
                fourcc: "YUYV".to_string(),
                sizes: vec![],
            },
        ];
        assert_eq!(classify(&formats), CameraKind::Rgb);
    }

    #[test]
    fn test_classify_empty_is_other() {
        assert_eq!(classify(&[]), CameraKind::Other);
    }

    #[test]
    fn test_classify_unknown_fourcc_is_other() {
        let formats = vec![FormatInfo {
            fourcc: "H264".to_string(),
            sizes: vec![],
        }];
        assert_eq!(classify(&formats), CameraKind::Other);
    }

    #[test]
    fn test_supports_checks_fourcc_and_size() {
        let device = CameraDevice {
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
            usb: None,
            extension_units: vec![],
            metadata_node: Some(PathBuf::from("/dev/video1")),
        };

        assert!(device.supports("MJPG", 1280, 720));
        assert!(!device.supports("MJPG", 1920, 1080));
        assert!(!device.supports("GREY", 640, 360));
    }

    #[test]
    fn test_stepwise_frame_size_yields_bounds_only() {
        let size = v4l::framesize::FrameSizeEnum::Stepwise(v4l::framesize::Stepwise {
            min_width: 2,
            max_width: 8192,
            step_width: 1,
            min_height: 1,
            max_height: 8192,
            step_height: 1,
        });
        assert_eq!(frame_size_bounds(size), vec![(2, 1), (8192, 8192)]);
    }

    #[test]
    fn test_stepwise_frame_size_zero_step_does_not_panic() {
        let size = v4l::framesize::FrameSizeEnum::Stepwise(v4l::framesize::Stepwise {
            min_width: 2,
            max_width: 8192,
            step_width: 0,
            min_height: 1,
            max_height: 8192,
            step_height: 0,
        });
        assert_eq!(frame_size_bounds(size), vec![(2, 1), (8192, 8192)]);
    }

    #[test]
    fn test_stepwise_frame_size_equal_bounds_yields_single_entry() {
        let size = v4l::framesize::FrameSizeEnum::Stepwise(v4l::framesize::Stepwise {
            min_width: 640,
            max_width: 640,
            step_width: 1,
            min_height: 480,
            max_height: 480,
            step_height: 1,
        });
        assert_eq!(frame_size_bounds(size), vec![(640, 480)]);
    }

    #[test]
    fn test_discrete_frame_size_yields_single_entry() {
        let size = v4l::framesize::FrameSizeEnum::Discrete(v4l::framesize::Discrete {
            width: 1280,
            height: 720,
        });
        assert_eq!(frame_size_bounds(size), vec![(1280, 720)]);
    }

    #[test]
    #[ignore = "needs hardware"]
    fn test_live_probe_finds_rgb_and_ir() {
        let cameras = V4l2CameraProbe::new().cameras().expect("probe succeeds");

        let nodes: Vec<&Path> = cameras.iter().map(|c| c.node.as_path()).collect();
        assert_eq!(
            nodes,
            vec![Path::new("/dev/video0"), Path::new("/dev/video2")]
        );

        let video0 = cameras
            .iter()
            .find(|c| c.node == Path::new("/dev/video0"))
            .expect("video0 present");
        assert_eq!(video0.kind, CameraKind::Rgb);
        assert!(video0.supports("MJPG", 1280, 720));
        let mjpg_720p = video0
            .formats
            .iter()
            .find(|f| f.fourcc == "MJPG")
            .expect("video0 has MJPG format")
            .sizes
            .iter()
            .find(|s| s.width == 1280 && s.height == 720)
            .expect("video0 MJPG 1280x720 size present");
        assert_eq!(mjpg_720p.fps, vec![30.0]);
        assert_eq!(video0.metadata_node, Some(PathBuf::from("/dev/video1")));

        let video2 = cameras
            .iter()
            .find(|c| c.node == Path::new("/dev/video2"))
            .expect("video2 present");
        assert_eq!(video2.kind, CameraKind::Ir);
        assert_eq!(
            video2.formats,
            vec![FormatInfo {
                fourcc: "GREY".to_string(),
                sizes: vec![FrameSizeInfo {
                    width: 640,
                    height: 360,
                    fps: vec![30.0]
                }]
            }]
        );
        let usb = video2.usb.as_ref().expect("video2 has usb identity");
        assert_eq!(usb.interface, 2);
        let msxu = video2
            .extension_units
            .iter()
            .find(|u| u.unit_id == 4)
            .expect("MSXU unit present");
        assert_eq!(msxu.selectors, vec![6, 9]);
        assert_eq!(video2.metadata_node, Some(PathBuf::from("/dev/video3")));

        assert!(
            cameras
                .iter()
                .all(|c| c.node != Path::new("/dev/video1") && c.node != Path::new("/dev/video3"))
        );
    }

    #[test]
    #[ignore = "needs hardware"]
    fn test_logs_camera_probed_at_info() {
        use eye_log::Value;

        let (cameras, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            V4l2CameraProbe::new().cameras().expect("probe succeeds")
        });
        assert_eq!(cameras.len(), 2);

        let probed: Vec<_> = records
            .iter()
            .filter(|r| r.message == "camera probed")
            .collect();
        assert_eq!(probed.len(), 2);
        let video0 = probed
            .iter()
            .find(|r| r.fields.get("node") == Some(&Value::Str("/dev/video0".to_string())))
            .expect("video0 probed");
        assert_eq!(video0.level, eye_log::Level::Info);
        assert_eq!(
            video0.fields.get("kind"),
            Some(&Value::Str("rgb".to_string()))
        );
        let video2 = probed
            .iter()
            .find(|r| r.fields.get("node") == Some(&Value::Str("/dev/video2".to_string())))
            .expect("video2 probed");
        assert_eq!(video2.level, eye_log::Level::Info);
        assert_eq!(
            video2.fields.get("kind"),
            Some(&Value::Str("ir".to_string()))
        );
        assert_eq!(video2.fields.get("extension_units"), Some(&Value::U64(2)));

        let skipped: Vec<_> = records
            .iter()
            .filter(|r| r.message == "node skipped, not video capture")
            .collect();
        assert_eq!(skipped.len(), 2);
        assert!(skipped.iter().all(|r| r.level == eye_log::Level::Debug
            && r.fields.get("meta_capture") == Some(&Value::Bool(true))));

        let finished = records
            .iter()
            .find(|r| r.message == "camera probe finished")
            .expect("camera probe finished logged");
        assert_eq!(finished.level, eye_log::Level::Debug);
        assert_eq!(finished.fields.get("cameras"), Some(&Value::U64(2)));
        assert_eq!(finished.fields.get("nodes"), Some(&Value::U64(4)));
    }
}
