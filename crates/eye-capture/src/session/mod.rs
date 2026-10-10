pub mod id;
pub mod pnm;
pub mod reader;
pub mod replay;
pub mod writer;

use std::path::{Path, PathBuf};

use eye_core::{CameraId, CameraInfo, Illumination, Timestamp};

use crate::format::StoredFormat;

pub use eye_core::session::{ProtocolConfig, TargetClock, TargetRecord};
pub use id::SessionId;
pub use reader::Recording;
pub use replay::{Pacing, PacingOption, ReplayOptions, ReplaySource};
pub use writer::SessionWriter;

pub const FORMAT_VERSION: u32 = 1;
pub const SESSION_FILE: &str = "session.toml";
pub const INDEX_FILE: &str = "index.jsonl";
pub const TARGETS_FILE: &str = "targets.jsonl";
pub const FRAMES_DIR: &str = "frames";

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmitterState {
    On,
    Off,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordedCamera {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    pub format: StoredFormat,
    pub width: u32,
    pub height: u32,
    pub frame_interval_ns: u64,
}

impl RecordedCamera {
    pub fn from_info(
        info: &CameraInfo,
        device: Option<&Path>,
    ) -> Result<Self, crate::CaptureError> {
        let format =
            StoredFormat::try_from(info.format).map_err(|f| crate::CaptureError::Config {
                camera: info.id.to_string(),
                reason: format!("pixel format {f:?} cannot be recorded"),
            })?;
        Ok(Self {
            id: info.id.to_string(),
            device: device.map(|d| d.display().to_string()),
            format,
            width: info.width,
            height: info.height,
            frame_interval_ns: u64::try_from(info.frame_interval.as_nanos()).unwrap_or(u64::MAX),
        })
    }

    pub fn to_info(&self) -> CameraInfo {
        CameraInfo {
            id: CameraId::from(self.id.as_str()),
            format: self.format.into(),
            width: self.width,
            height: self.height,
            frame_interval: std::time::Duration::from_nanos(self.frame_interval_ns),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionMeta {
    pub format_version: u32,
    pub session_id: String,
    pub created_unix_s: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_rev: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emitter: Option<EmitterState>,
    #[serde(rename = "camera")]
    pub cameras: Vec<RecordedCamera>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rig: Option<toml::Table>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probe: Option<toml::Table>,
    /// The dot protocol the targets were shown with; `None` for free recordings and for files written before this field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol: Option<ProtocolConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexRecord {
    pub seq: u64,
    pub camera: String,
    pub timestamp_ns: u64,
    pub illumination: Illumination,
}

pub fn frame_path(session_dir: &Path, camera: &str, seq: u64, format: StoredFormat) -> PathBuf {
    session_dir
        .join(FRAMES_DIR)
        .join(camera)
        .join(format!("{seq:08}.{}", format.extension()))
}

pub fn timestamp_ns(t: Timestamp) -> u64 {
    t.as_nanos()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use eye_core::PixelFormat;

    use super::*;

    fn meta() -> SessionMeta {
        SessionMeta {
            format_version: 1,
            session_id: "20261007T221500Z".to_string(),
            created_unix_s: 1_791_411_300,
            git_rev: Some("abc1234".to_string()),
            emitter: Some(EmitterState::On),
            cameras: vec![
                RecordedCamera {
                    id: "rgb".to_string(),
                    device: Some("/dev/video0".to_string()),
                    format: StoredFormat::Mjpeg,
                    width: 1280,
                    height: 720,
                    frame_interval_ns: 33_333_333,
                },
                RecordedCamera {
                    id: "ir".to_string(),
                    device: Some("/dev/video2".to_string()),
                    format: StoredFormat::Gray8,
                    width: 640,
                    height: 360,
                    frame_interval_ns: 33_333_333,
                },
            ],
            rig: None,
            probe: Some({
                let mut table = toml::Table::new();
                table.insert(
                    "session_type".to_string(),
                    toml::Value::String("wayland".to_string()),
                );
                table
            }),
            protocol: None,
        }
    }

    #[test]
    fn test_session_toml_roundtrips() {
        let text = toml::to_string(&meta()).unwrap();
        assert!(text.contains("format_version = 1"));
        assert!(text.contains("format = \"gray\""));
        assert_eq!(text.matches("[[camera]]").count(), 2);
        let parsed: SessionMeta = toml::from_str(&text).unwrap();
        assert_eq!(parsed, meta());
    }

    #[test]
    fn test_session_toml_rejects_unknown_field() {
        let text = format!("bogus = 1\n{}", toml::to_string(&meta()).unwrap());
        let parsed: Result<SessionMeta, _> = toml::from_str(&text);
        assert!(parsed.is_err());
    }

    #[test]
    fn test_session_meta_round_trips_protocol() {
        let with_protocol = SessionMeta {
            protocol: Some(ProtocolConfig {
                grid: [4, 4],
                lead_in_ms: 1000,
                dwell_ms: 2500,
                settle_ms: 1000,
                window_ms: 1200,
            }),
            ..meta()
        };
        let text = toml::to_string(&with_protocol).unwrap();
        assert!(text.contains("[protocol]"));
        assert!(text.contains("grid = [4, 4]"));
        let parsed: SessionMeta = toml::from_str(&text).unwrap();
        assert_eq!(parsed, with_protocol);
    }

    #[test]
    fn test_session_meta_without_protocol_parses_as_none() {
        let text = toml::to_string(&meta()).unwrap();
        assert!(!text.contains("[protocol]"));
        let parsed: SessionMeta = toml::from_str(&text).unwrap();
        assert_eq!(parsed.protocol, None);
    }

    #[test]
    fn test_index_record_json_is_exact() {
        let record = IndexRecord {
            seq: 42,
            camera: "ir".to_string(),
            timestamp_ns: 123_456_789,
            illumination: Illumination::IrLit,
        };
        assert_eq!(
            serde_json::to_string(&record).unwrap(),
            r#"{"seq":42,"camera":"ir","timestamp_ns":123456789,"illumination":"ir_lit"}"#
        );
    }

    #[test]
    fn test_recorded_camera_info_roundtrip() {
        let info = CameraInfo {
            id: CameraId::from("ir"),
            format: PixelFormat::Gray8,
            width: 640,
            height: 360,
            frame_interval: Duration::from_nanos(33_333_333),
        };
        let rc = RecordedCamera::from_info(&info, Some(Path::new("/dev/video2"))).unwrap();
        assert_eq!(rc.device, Some("/dev/video2".to_string()));
        assert_eq!(rc.to_info(), info);
        assert!(toml::to_string(&rc).unwrap().contains("format = \"gray\""));

        let rgb_info = CameraInfo {
            id: CameraId::from("rgb"),
            format: PixelFormat::Rgb8,
            width: 1280,
            height: 720,
            frame_interval: Duration::from_nanos(33_333_333),
        };
        assert!(matches!(
            RecordedCamera::from_info(&rgb_info, None),
            Err(crate::CaptureError::Config { .. })
        ));
    }

    #[test]
    fn test_rgb8_cannot_be_recorded() {
        assert_eq!(
            StoredFormat::try_from(PixelFormat::Rgb8),
            Err(PixelFormat::Rgb8)
        );
    }
}
