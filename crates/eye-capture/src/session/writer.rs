use std::{
    borrow::Cow,
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
};

use eye_core::{Frame, log::field};

use super::{
    FRAMES_DIR, INDEX_FILE, IndexRecord, RecordedCamera, SESSION_FILE, SessionMeta, TARGETS_FILE,
    TargetRecord, frame_path, id::valid_component, timestamp_ns,
};
use crate::{CaptureError, format::StoredFormat};

#[derive(Debug)]
pub struct SessionWriter {
    dir: PathBuf,
    cameras: HashMap<String, RecordedCamera>,
    index: BufWriter<File>,
    targets: BufWriter<File>,
    frame_count: u64,
    target_count: u64,
}

fn io_err(path: &Path) -> impl Fn(io::Error) -> CaptureError + '_ {
    move |source| CaptureError::RecordingIo {
        path: path.to_path_buf(),
        source,
    }
}

impl SessionWriter {
    pub fn create(root: &Path, meta: &SessionMeta) -> Result<Self, CaptureError> {
        if !valid_component(&meta.session_id) {
            return Err(CaptureError::RecordingFormat {
                path: root.join(&meta.session_id),
                reason: format!("invalid session id {:?}", meta.session_id),
            });
        }
        for camera in &meta.cameras {
            if !valid_component(&camera.id) {
                return Err(CaptureError::RecordingFormat {
                    path: root.join(&meta.session_id),
                    reason: format!("invalid camera id {:?}", camera.id),
                });
            }
        }

        fs::create_dir_all(root).map_err(io_err(root))?;
        let dir = root.join(&meta.session_id);
        fs::create_dir(&dir).map_err(io_err(&dir))?;

        for camera in &meta.cameras {
            let camera_dir = dir.join(FRAMES_DIR).join(&camera.id);
            fs::create_dir_all(&camera_dir).map_err(io_err(&camera_dir))?;
        }

        let session_path = dir.join(SESSION_FILE);
        let toml_text = toml::to_string(meta).map_err(|e| CaptureError::RecordingFormat {
            path: dir.clone(),
            reason: e.to_string(),
        })?;
        fs::write(&session_path, toml_text).map_err(io_err(&session_path))?;

        let index_path = dir.join(INDEX_FILE);
        let index = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&index_path)
                .map_err(io_err(&index_path))?,
        );

        let targets_path = dir.join(TARGETS_FILE);
        let targets = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&targets_path)
                .map_err(io_err(&targets_path))?,
        );

        let cameras = meta
            .cameras
            .iter()
            .map(|c| (c.id.clone(), c.clone()))
            .collect();

        tracing::info!(
            { field::SESSION_ID } = meta.session_id.as_str(),
            dir = %dir.display(),
            cameras = meta.cameras.len(),
            "session created"
        );

        Ok(Self {
            dir,
            cameras,
            index,
            targets,
            frame_count: 0,
            target_count: 0,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn bad(&self, reason: String) -> CaptureError {
        CaptureError::RecordingFormat {
            path: self.dir.clone(),
            reason,
        }
    }

    pub fn write_frame(&mut self, frame: &Frame) -> Result<(), CaptureError> {
        let h = frame.header();
        let cam = self
            .cameras
            .get(h.camera.as_str())
            .ok_or_else(|| self.bad(format!("unknown camera {}", h.camera)))?
            .clone();
        let format = StoredFormat::try_from(h.format)
            .map_err(|f| self.bad(format!("pixel format {f:?} cannot be recorded")))?;
        if format != cam.format || h.width != cam.width || h.height != cam.height {
            return Err(self.bad(format!(
                "frame {} of {} does not match session.toml",
                h.seq, cam.id
            )));
        }
        let path = frame_path(&self.dir, &cam.id, h.seq, format);
        let bytes: Cow<[u8]> = match format {
            StoredFormat::Mjpeg => Cow::Borrowed(frame.data()),
            StoredFormat::Gray8 => {
                if frame.data().len() != (h.width * h.height) as usize {
                    return Err(self.bad(format!(
                        "gray frame {} has {} bytes",
                        h.seq,
                        frame.data().len()
                    )));
                }
                Cow::Owned(super::pnm::encode_pgm(h.width, h.height, frame.data()))
            }
        };
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(io_err(&path))?;
        file.write_all(&bytes).map_err(io_err(&path))?;

        let record = IndexRecord {
            seq: h.seq,
            camera: cam.id.clone(),
            timestamp_ns: timestamp_ns(h.timestamp),
            illumination: h.illumination,
        };
        serde_json::to_writer(&mut self.index, &record).map_err(|e| self.bad(e.to_string()))?;
        self.index
            .write_all(b"\n")
            .map_err(io_err(&self.dir.join(INDEX_FILE)))?;
        self.frame_count += 1;
        tracing::debug!(
            { field::CAMERA } = cam.id.as_str(),
            { field::SEQ } = h.seq,
            bytes = bytes.len(),
            "frame written"
        );
        Ok(())
    }

    pub fn write_target(&mut self, target: &TargetRecord) -> Result<(), CaptureError> {
        serde_json::to_writer(&mut self.targets, target).map_err(|e| self.bad(e.to_string()))?;
        self.targets
            .write_all(b"\n")
            .map_err(io_err(&self.dir.join(TARGETS_FILE)))?;
        self.target_count += 1;
        tracing::debug!({ field::SEQ } = target.seq, "target written");
        Ok(())
    }

    pub fn finish(mut self) -> Result<PathBuf, CaptureError> {
        self.index
            .flush()
            .map_err(io_err(&self.dir.join(INDEX_FILE)))?;
        self.index
            .get_ref()
            .sync_all()
            .map_err(io_err(&self.dir.join(INDEX_FILE)))?;
        self.targets
            .flush()
            .map_err(io_err(&self.dir.join(TARGETS_FILE)))?;
        self.targets
            .get_ref()
            .sync_all()
            .map_err(io_err(&self.dir.join(TARGETS_FILE)))?;
        tracing::info!(
            dir = %self.dir.display(),
            frames = self.frame_count,
            targets = self.target_count,
            "session written"
        );
        Ok(self.dir)
    }
}

#[cfg(test)]
mod tests {
    use eye_core::session::TargetClock;

    use super::*;
    use crate::{
        session::{EmitterState, SessionMeta},
        testing::{gray_frame, mjpeg_frame},
    };

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
            probe: None,
            protocol: None,
        }
    }

    fn target(
        seq: u64,
        shown_ns: u64,
        hidden_ns: Option<u64>,
        clock: TargetClock,
        px: [f64; 2],
        mm: [f64; 2],
    ) -> TargetRecord {
        TargetRecord {
            seq,
            shown_ns,
            hidden_ns,
            clock,
            output: eye_core::OutputId::from("eDP-1"),
            px_logical: px,
            mm,
        }
    }

    #[test]
    fn test_writer_creates_layout() {
        let root = tempfile::tempdir().unwrap();
        let writer = SessionWriter::create(root.path(), &meta()).unwrap();
        let dir = writer.finish().unwrap();

        assert!(dir.join(SESSION_FILE).is_file());
        assert!(dir.join(INDEX_FILE).is_file());
        assert!(dir.join(TARGETS_FILE).is_file());
        assert!(dir.join(FRAMES_DIR).join("rgb").is_dir());
        assert!(dir.join(FRAMES_DIR).join("ir").is_dir());
        assert_eq!(fs::read_to_string(dir.join(INDEX_FILE)).unwrap(), "");
        assert_eq!(fs::read_to_string(dir.join(TARGETS_FILE)).unwrap(), "");
    }

    #[test]
    fn test_writer_writes_frames_and_index_lines() {
        let root = tempfile::tempdir().unwrap();
        let mut writer = SessionWriter::create(root.path(), &meta()).unwrap();

        let ir_frame = gray_frame("ir", 7, 1_000, 640, 360, 45);
        let ir_data = ir_frame.data().to_vec();
        writer.write_frame(&ir_frame).unwrap();

        let rgb_bytes = b"\xff\xd8abc\xff\xd9";
        let rgb_frame = mjpeg_frame("rgb", 3, 2_000, rgb_bytes);
        writer.write_frame(&rgb_frame).unwrap();

        let dir = writer.finish().unwrap();

        let ir_path = dir.join(FRAMES_DIR).join("ir").join("00000007.pgm");
        assert_eq!(
            fs::read(&ir_path).unwrap(),
            super::super::pnm::encode_pgm(640, 360, &ir_data)
        );

        let rgb_path = dir.join(FRAMES_DIR).join("rgb").join("00000003.jpg");
        assert_eq!(fs::read(&rgb_path).unwrap(), rgb_bytes);

        let index_text = fs::read_to_string(dir.join(INDEX_FILE)).unwrap();
        let lines: Vec<&str> = index_text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(
            lines[0],
            r#"{"seq":7,"camera":"ir","timestamp_ns":1000,"illumination":"unknown"}"#
        );
        assert_eq!(
            lines[1],
            r#"{"seq":3,"camera":"rgb","timestamp_ns":2000,"illumination":"ambient"}"#
        );
    }

    #[test]
    fn test_writer_appends_targets() {
        let root = tempfile::tempdir().unwrap();
        let mut writer = SessionWriter::create(root.path(), &meta()).unwrap();

        writer
            .write_target(&target(
                0,
                1_000_000_000,
                Some(3_000_000_000),
                TargetClock::Presentation,
                [960.0, 540.0],
                [155.0, 85.0],
            ))
            .unwrap();
        writer
            .write_target(&target(
                1,
                3_000_000_000,
                None,
                TargetClock::Commit,
                [96.0, 54.0],
                [15.5, 8.5],
            ))
            .unwrap();

        let dir = writer.finish().unwrap();
        let text = fs::read_to_string(dir.join(TARGETS_FILE)).unwrap();
        assert_eq!(text.lines().count(), 2);
    }

    #[test]
    fn test_target_record_line_roundtrips() {
        let root = tempfile::tempdir().unwrap();
        let mut writer = SessionWriter::create(root.path(), &meta()).unwrap();

        let first = target(
            0,
            1_000_000_000,
            Some(3_000_000_000),
            TargetClock::Presentation,
            [960.0, 540.0],
            [155.0, 85.0],
        );
        let second = target(
            1,
            3_000_000_000,
            None,
            TargetClock::Commit,
            [96.0, 54.0],
            [15.5, 8.5],
        );
        writer.write_target(&first).unwrap();
        writer.write_target(&second).unwrap();
        let dir = writer.finish().unwrap();

        let text = fs::read_to_string(dir.join(TARGETS_FILE)).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);

        let parsed_first: TargetRecord = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(
            serde_json::to_string(&parsed_first).unwrap(),
            serde_json::to_string(&first).unwrap()
        );
        let parsed_second: TargetRecord = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(
            serde_json::to_string(&parsed_second).unwrap(),
            serde_json::to_string(&second).unwrap()
        );
    }

    #[test]
    fn test_duplicate_seq_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let mut writer = SessionWriter::create(root.path(), &meta()).unwrap();

        writer
            .write_frame(&gray_frame("ir", 1, 0, 640, 360, 1))
            .unwrap();
        let err = writer
            .write_frame(&gray_frame("ir", 1, 1, 640, 360, 2))
            .unwrap_err();
        assert!(matches!(
            &err,
            CaptureError::RecordingIo { source, .. } if source.kind() == io::ErrorKind::AlreadyExists
        ));

        let dir = writer.finish().unwrap();
        let index_text = fs::read_to_string(dir.join(INDEX_FILE)).unwrap();
        assert_eq!(index_text.lines().count(), 1);
    }

    #[test]
    fn test_unknown_camera_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let mut writer = SessionWriter::create(root.path(), &meta()).unwrap();

        let err = writer
            .write_frame(&gray_frame("depth", 0, 0, 640, 360, 1))
            .unwrap_err();
        assert!(matches!(err, CaptureError::RecordingFormat { .. }));

        let dir = writer.finish().unwrap();
        assert!(!dir.join(FRAMES_DIR).join("depth").exists());
        let index_text = fs::read_to_string(dir.join(INDEX_FILE)).unwrap();
        assert_eq!(index_text.lines().count(), 0);
    }

    #[test]
    fn test_mismatched_size_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let mut writer = SessionWriter::create(root.path(), &meta()).unwrap();

        let err = writer
            .write_frame(&gray_frame("ir", 0, 0, 320, 240, 1))
            .unwrap_err();
        assert!(matches!(err, CaptureError::RecordingFormat { .. }));
    }

    #[test]
    fn test_writer_rejects_path_tricks_in_ids() {
        let root = tempfile::tempdir().unwrap();

        let mut bad_session = meta();
        bad_session.session_id = "../escape".to_string();
        let err = SessionWriter::create(root.path(), &bad_session).unwrap_err();
        assert!(matches!(err, CaptureError::RecordingFormat { .. }));
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
        assert!(!root.path().join("../escape").exists());

        let mut bad_camera = meta();
        bad_camera.cameras[0].id = "a/b".to_string();
        let err = SessionWriter::create(root.path(), &bad_camera).unwrap_err();
        assert!(matches!(err, CaptureError::RecordingFormat { .. }));
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
    }

    #[test]
    fn test_existing_session_dir_is_not_overwritten() {
        let root = tempfile::tempdir().unwrap();
        let _writer = SessionWriter::create(root.path(), &meta()).unwrap();
        let err = SessionWriter::create(root.path(), &meta()).unwrap_err();
        assert!(matches!(err, CaptureError::RecordingIo { .. }));
    }

    #[test]
    fn test_logs_session_created_and_written_at_info() {
        let root = tempfile::tempdir().unwrap();
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut writer = SessionWriter::create(root.path(), &meta()).unwrap();
            writer
                .write_frame(&gray_frame("ir", 7, 1_000, 640, 360, 45))
                .unwrap();
            writer
                .write_frame(&mjpeg_frame("rgb", 3, 2_000, b"\xff\xd8abc\xff\xd9"))
                .unwrap();
            writer
                .write_target(&target(
                    0,
                    1_000_000_000,
                    Some(3_000_000_000),
                    TargetClock::Presentation,
                    [960.0, 540.0],
                    [155.0, 85.0],
                ))
                .unwrap();
            writer.finish().unwrap();
        });

        let created: Vec<_> = records
            .iter()
            .filter(|r| r.message == "session created")
            .collect();
        assert_eq!(created.len(), 1);
        assert_eq!(created[0].level, eye_log::Level::Info);
        assert_eq!(created[0].fields["cameras"], eye_log::Value::U64(2));
        assert_eq!(
            created[0].fields[field::SESSION_ID],
            eye_log::Value::Str("20261007T221500Z".to_string())
        );

        let written: Vec<_> = records
            .iter()
            .filter(|r| r.message == "session written")
            .collect();
        assert_eq!(written.len(), 1);
        assert_eq!(written[0].level, eye_log::Level::Info);
        assert_eq!(written[0].fields["frames"], eye_log::Value::U64(2));
        assert_eq!(written[0].fields["targets"], eye_log::Value::U64(1));

        let frame_written: Vec<_> = records
            .iter()
            .filter(|r| r.message == "frame written")
            .collect();
        assert_eq!(frame_written.len(), 2);
        for rec in &frame_written {
            assert!(matches!(rec.fields["bytes"], eye_log::Value::U64(_)));
        }
    }
}
