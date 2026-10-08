use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use eye_core::{CameraId, CameraInfo, Frame};

use super::{IndexRecord, RecordedCamera, reader::load_frame};
use crate::{CaptureError, source::FrameSource};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pacing {
    AsFastAsPossible,
    RealTime { speed: f64 },
}

#[derive(Debug)]
pub struct ReplaySource {
    dir: PathBuf,
    camera: RecordedCamera,
    info: CameraInfo,
    records: std::vec::IntoIter<IndexRecord>,
    pacing: Pacing,
    start: Option<(Instant, u64)>,
}

impl ReplaySource {
    pub(crate) fn new(
        dir: PathBuf,
        camera: RecordedCamera,
        info: CameraInfo,
        records: Vec<IndexRecord>,
        pacing: Pacing,
    ) -> Self {
        Self {
            dir,
            camera,
            info,
            records: records.into_iter(),
            pacing,
            start: None,
        }
    }

    pub fn from_config(id: CameraId, options: &toml::Table) -> Result<Self, CaptureError> {
        let opts: ReplayOptions =
            options
                .clone()
                .try_into()
                .map_err(|e: toml::de::Error| CaptureError::Config {
                    camera: id.to_string(),
                    reason: e.to_string(),
                })?;
        let camera = opts.camera.unwrap_or_else(|| id.to_string());
        let mut source = super::reader::Recording::open(&opts.recording)?
            .source(&camera, opts.pacing.into_pacing(opts.speed))?;
        source.info.id = id;
        Ok(source)
    }
}

impl FrameSource for ReplaySource {
    fn camera(&self) -> &CameraInfo {
        &self.info
    }

    fn next_frame(&mut self) -> Result<Frame, CaptureError> {
        let record = self.records.next().ok_or(CaptureError::EndOfStream)?;
        if let Pacing::RealTime { speed } = self.pacing {
            let (start, first) = *self
                .start
                .get_or_insert((Instant::now(), record.timestamp_ns));
            let due = start + Duration::from_nanos(record.timestamp_ns - first).div_f64(speed);
            if let Some(wait) = due.checked_duration_since(Instant::now()) {
                std::thread::sleep(wait);
            }
        }
        load_frame(&self.dir, &self.camera, &self.info.id, &record)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayOptions {
    pub recording: PathBuf,
    #[serde(default)]
    pub camera: Option<String>,
    #[serde(default)]
    pub pacing: PacingOption,
    #[serde(default = "default_speed")]
    pub speed: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PacingOption {
    #[default]
    Fast,
    Realtime,
}

impl PacingOption {
    pub fn into_pacing(self, speed: f64) -> Pacing {
        match self {
            Self::Fast => Pacing::AsFastAsPossible,
            Self::Realtime => Pacing::RealTime { speed },
        }
    }
}

fn default_speed() -> f64 {
    1.0
}

#[cfg(test)]
mod tests {
    use eye_core::PixelFormat;

    use super::*;
    use crate::session::{EmitterState, RecordedFormat, SessionMeta, SessionWriter};

    fn meta() -> SessionMeta {
        SessionMeta {
            format_version: 1,
            session_id: "20261007T221500Z".to_string(),
            created_unix_s: 1_791_411_300,
            git_rev: None,
            emitter: Some(EmitterState::On),
            cameras: vec![
                RecordedCamera {
                    id: "rgb".to_string(),
                    device: None,
                    format: RecordedFormat::Mjpeg,
                    width: 1280,
                    height: 720,
                    frame_interval_ns: 33_333_333,
                },
                RecordedCamera {
                    id: "ir".to_string(),
                    device: None,
                    format: RecordedFormat::Gray8,
                    width: 640,
                    height: 360,
                    frame_interval_ns: 33_333_333,
                },
            ],
            rig: None,
            probe: None,
        }
    }

    fn write_fixture(dir: &std::path::Path) -> PathBuf {
        let mut writer = SessionWriter::create(dir, &meta()).unwrap();
        writer
            .write_frame(&crate::testing::gray_frame("ir", 0, 0, 640, 360, 9))
            .unwrap();
        writer
            .write_frame(&crate::testing::mjpeg_frame(
                "rgb",
                0,
                0,
                &[0xFF, 0xD8, 0xFF, 0xD9],
            ))
            .unwrap();
        writer.finish().unwrap()
    }

    #[test]
    fn test_from_config_replays_named_camera() {
        let root = tempfile::tempdir().unwrap();
        let dir = write_fixture(root.path());

        let mut table = toml::Table::new();
        table.insert(
            "recording".to_string(),
            toml::Value::String(dir.display().to_string()),
        );
        let mut source = ReplaySource::from_config(CameraId::from("ir"), &table).unwrap();
        let frame = source.next_frame().unwrap();
        assert_eq!(frame.header().camera, CameraId::from("ir"));
        assert_eq!(frame.header().format, PixelFormat::Gray8);

        let mut table_rgb = toml::Table::new();
        table_rgb.insert(
            "recording".to_string(),
            toml::Value::String(dir.display().to_string()),
        );
        table_rgb.insert("camera".to_string(), toml::Value::String("rgb".to_string()));
        let mut source = ReplaySource::from_config(CameraId::from("x"), &table_rgb).unwrap();
        let frame = source.next_frame().unwrap();
        assert_eq!(frame.header().camera, CameraId::from("x"));
        assert_eq!(frame.header().format, PixelFormat::Mjpeg);

        let mut table_bad = toml::Table::new();
        table_bad.insert(
            "recording".to_string(),
            toml::Value::String(dir.display().to_string()),
        );
        table_bad.insert("bogus".to_string(), toml::Value::Integer(1));
        assert!(matches!(
            ReplaySource::from_config(CameraId::from("ir"), &table_bad),
            Err(CaptureError::Config { .. })
        ));
    }
}
