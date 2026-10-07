use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::id::string_id;

string_id!(CameraId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PixelFormat {
    #[serde(rename = "gray")]
    Gray8,
    #[serde(rename = "rgb")]
    Rgb8,
    #[serde(rename = "mjpeg")]
    Mjpeg,
}

impl PixelFormat {
    pub const fn bytes_per_pixel(self) -> Option<usize> {
        match self {
            Self::Gray8 => Some(1),
            Self::Rgb8 => Some(3),
            Self::Mjpeg => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Illumination {
    Ambient,
    IrLit,
    IrDark,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraInfo {
    pub id: CameraId,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub frame_interval: Duration,
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn test_pixel_format_uses_config_names() {
        assert_eq!(
            serde_json::to_string(&PixelFormat::Gray8).unwrap(),
            "\"gray\""
        );
        assert_eq!(
            serde_json::to_string(&PixelFormat::Rgb8).unwrap(),
            "\"rgb\""
        );
        assert_eq!(
            serde_json::to_string(&PixelFormat::Mjpeg).unwrap(),
            "\"mjpeg\""
        );
    }

    #[test]
    fn test_illumination_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&Illumination::IrLit).unwrap(),
            "\"ir_lit\""
        );
        let parsed: Illumination = serde_json::from_str("\"ir_dark\"").unwrap();
        assert_eq!(parsed, Illumination::IrDark);
    }

    #[test]
    fn test_camera_id_serializes_as_plain_string() {
        let id = CameraId::from("rgb");
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"rgb\"");
        let parsed: CameraId = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, id);
    }

    #[test]
    fn test_camera_id_looks_up_map_by_str() {
        let mut map: HashMap<CameraId, i32> = HashMap::new();
        map.insert(CameraId::from("ir"), 1);
        assert_eq!(map.get("ir"), Some(&1));
    }

    #[test]
    fn test_camera_id_as_ref_is_the_name() {
        let id = CameraId::from("ir");
        assert_eq!(id.as_ref(), "ir");
    }

    #[test]
    fn test_camera_info_describes_ir_stream() {
        let info = CameraInfo {
            id: CameraId::from("ir"),
            width: 640,
            height: 360,
            format: PixelFormat::Gray8,
            frame_interval: Duration::from_millis(68),
        };
        assert_eq!(info, info.clone());
        assert_eq!(info.frame_interval.as_nanos(), 68_000_000);
    }
}
