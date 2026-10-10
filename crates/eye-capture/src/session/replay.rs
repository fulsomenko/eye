use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use eye_core::{CameraId, CameraInfo, Frame, log::field};

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
        let Some(record) = self.records.next() else {
            tracing::debug!({ field::CAMERA } = self.info.id.as_str(), "replay ended");
            return Err(CaptureError::EndOfStream);
        };
        let mut wait_us = 0u64;
        if let Pacing::RealTime { speed } = self.pacing {
            let (start, first) = *self
                .start
                .get_or_insert((Instant::now(), record.timestamp_ns));
            let due = start + Duration::from_nanos(record.timestamp_ns - first).div_f64(speed);
            if let Some(wait) = due.checked_duration_since(Instant::now()) {
                wait_us = wait.as_micros() as u64;
                std::thread::sleep(wait);
            }
        }
        let frame = load_frame(&self.dir, &self.camera, &self.info.id, &record)?;
        let h = frame.header();
        let _frame_span = eye_core::log::frame_span(
            h.camera.as_str(),
            h.seq,
            h.timestamp.as_nanos(),
            h.illumination.as_str(),
            1,
        )
        .entered();
        tracing::trace!(
            bytes = frame.data().len(),
            source = "replay",
            wait_us,
            "frame produced"
        );
        Ok(frame)
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
    use crate::{
        format::StoredFormat,
        session::{EmitterState, SessionId, SessionMeta, SessionWriter},
    };

    fn meta() -> SessionMeta {
        SessionMeta {
            format_version: 1,
            session_id: SessionId::new("20261007T221500Z").unwrap(),
            created_unix_s: 1_791_411_300,
            git_rev: None,
            emitter: Some(EmitterState::On),
            cameras: vec![
                RecordedCamera {
                    id: "rgb".to_string(),
                    device: None,
                    format: StoredFormat::Mjpeg,
                    width: 1280,
                    height: 720,
                    frame_interval_ns: 33_333_333,
                },
                RecordedCamera {
                    id: "ir".to_string(),
                    device: None,
                    format: StoredFormat::Gray8,
                    width: 640,
                    height: 360,
                    frame_interval_ns: 33_333_333,
                },
            ],
            rig: None,
            probe: None,
            protocol: None,
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

    fn table_for(dir: &std::path::Path) -> toml::Table {
        let mut table = toml::Table::new();
        table.insert(
            "recording".to_string(),
            toml::Value::String(dir.display().to_string()),
        );
        table
    }

    #[test]
    fn test_logs_frame_produced_at_trace_inside_frame_span() {
        let root = tempfile::tempdir().unwrap();
        let dir = write_fixture(root.path());
        let table = table_for(&dir);

        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut source = ReplaySource::from_config(CameraId::from("ir"), &table).unwrap();
            source.next_frame().unwrap();
        });

        let produced: Vec<_> = records
            .iter()
            .filter(|r| r.message == "frame produced")
            .collect();
        assert_eq!(produced.len(), 1);
        let rec = produced[0];
        assert_eq!(rec.target, "eye_capture::session::replay");
        assert_eq!(rec.level, eye_log::Level::Trace);
        assert_eq!(
            rec.context[field::CAMERA],
            eye_log::Value::Str("ir".to_string())
        );
        assert_eq!(rec.context[field::SEQ], eye_log::Value::U64(0));
        assert_eq!(rec.context[field::TS_NS], eye_log::Value::U64(0));
        assert_eq!(
            rec.context[field::ILLUMINATION],
            eye_log::Value::Str("unknown".to_string())
        );
        assert_eq!(rec.context[field::SET_CAMERAS], eye_log::Value::U64(1));
        assert_eq!(
            rec.fields["source"],
            eye_log::Value::Str("replay".to_string())
        );
        assert!(matches!(rec.fields["bytes"], eye_log::Value::U64(_)));

        let keys: Vec<&str> = rec.context.keys().map(|k| k.as_str()).collect();
        assert_eq!(
            keys,
            vec![
                field::CAMERA,
                field::ILLUMINATION,
                field::SEQ,
                field::SET_CAMERAS,
                field::TS_NS,
            ]
        );
    }

    #[test]
    fn test_logs_replay_ended_at_debug() {
        let root = tempfile::tempdir().unwrap();
        let dir = write_fixture(root.path());
        let table = table_for(&dir);

        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut source = ReplaySource::from_config(CameraId::from("ir"), &table).unwrap();
            loop {
                match source.next_frame() {
                    Ok(_) => {}
                    Err(CaptureError::EndOfStream) => break,
                    Err(e) => panic!("{e}"),
                }
            }
        });

        let last = records.last().unwrap();
        assert_eq!(last.message, "replay ended");
        assert_eq!(last.level, eye_log::Level::Debug);
        assert_eq!(
            last.fields[field::CAMERA],
            eye_log::Value::Str("ir".to_string())
        );
    }
}
