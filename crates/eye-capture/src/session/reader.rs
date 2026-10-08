use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use eye_core::{CameraId, Frame, FrameHeader, Timestamp};

use super::{
    FORMAT_VERSION, INDEX_FILE, IndexRecord, RecordedCamera, RecordedFormat, SESSION_FILE,
    SessionMeta, TARGETS_FILE, TargetRecord, frame_path,
    pnm::decode_pgm,
    replay::{Pacing, ReplaySource},
};
use crate::CaptureError;

#[derive(Debug)]
pub struct Recording {
    dir: PathBuf,
    meta: SessionMeta,
    index: Vec<IndexRecord>,
    targets: Vec<TargetRecord>,
}

pub(crate) fn parse_jsonl<T: serde::de::DeserializeOwned>(
    path: &Path,
    text: &str,
) -> Result<Vec<T>, CaptureError> {
    let lines: Vec<&str> = text.split('\n').collect();
    let mut out = Vec::with_capacity(lines.len());
    for (i, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str(line) {
            Ok(v) => out.push(v),
            Err(_) if i + 1 == lines.len() => {
                tracing::warn!(path = %path.display(), line = i + 1, "ignoring truncated final line");
            }
            Err(e) => {
                return Err(CaptureError::RecordingFormat {
                    path: path.to_owned(),
                    reason: format!("line {}: {e}", i + 1),
                });
            }
        }
    }
    Ok(out)
}

fn read_to_string(path: &Path) -> Result<String, CaptureError> {
    fs::read_to_string(path).map_err(|source| CaptureError::RecordingIo {
        path: path.to_owned(),
        source,
    })
}

pub(crate) fn load_frame(
    dir: &Path,
    camera: &RecordedCamera,
    id: &CameraId,
    record: &IndexRecord,
) -> Result<Frame, CaptureError> {
    let path = frame_path(dir, &camera.id, record.seq, camera.format);
    let bytes = fs::read(&path).map_err(|source| CaptureError::RecordingIo {
        path: path.clone(),
        source,
    })?;
    let data: Arc<[u8]> = match camera.format {
        RecordedFormat::Mjpeg => bytes.into(),
        RecordedFormat::Gray8 => {
            let (w, h, raster) =
                decode_pgm(&bytes).map_err(|reason| CaptureError::RecordingFormat {
                    path: path.clone(),
                    reason,
                })?;
            if (w, h) != (camera.width, camera.height) {
                return Err(CaptureError::RecordingFormat {
                    path,
                    reason: format!(
                        "PGM is {w}x{h}, session.toml says {}x{}",
                        camera.width, camera.height
                    ),
                });
            }
            Arc::from(raster)
        }
    };
    let header = FrameHeader {
        camera: id.clone(),
        seq: record.seq,
        timestamp: Timestamp::from_nanos(record.timestamp_ns),
        width: camera.width,
        height: camera.height,
        format: camera.format.into(),
        illumination: record.illumination.into(),
    };
    Frame::new(header, data).map_err(|e| CaptureError::RecordingFormat {
        path,
        reason: e.to_string(),
    })
}

impl Recording {
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self, CaptureError> {
        let dir = dir.into();

        let session_path = dir.join(SESSION_FILE);
        let meta: SessionMeta = toml::from_str(&read_to_string(&session_path)?).map_err(|e| {
            CaptureError::RecordingFormat {
                path: session_path.clone(),
                reason: e.to_string(),
            }
        })?;
        if meta.format_version != FORMAT_VERSION {
            return Err(CaptureError::RecordingFormat {
                path: session_path,
                reason: format!("unsupported format_version {}", meta.format_version),
            });
        }

        let index_path = dir.join(INDEX_FILE);
        let index: Vec<IndexRecord> = parse_jsonl(&index_path, &read_to_string(&index_path)?)?;

        let mut last_by_camera: HashMap<&str, (u64, u64)> = HashMap::new();
        for record in &index {
            if !meta.cameras.iter().any(|c| c.id == record.camera) {
                return Err(CaptureError::RecordingFormat {
                    path: index_path,
                    reason: format!("no camera {} in session.toml", record.camera),
                });
            }
            if let Some(&(last_seq, last_ts)) = last_by_camera.get(record.camera.as_str())
                && (record.seq <= last_seq || record.timestamp_ns < last_ts)
            {
                return Err(CaptureError::RecordingFormat {
                    path: index_path,
                    reason: format!(
                        "camera {}: index is not monotonic at seq {}",
                        record.camera, record.seq
                    ),
                });
            }
            last_by_camera.insert(&record.camera, (record.seq, record.timestamp_ns));
        }

        let targets_path = dir.join(TARGETS_FILE);
        let targets: Vec<TargetRecord> =
            parse_jsonl(&targets_path, &read_to_string(&targets_path)?)?;

        Ok(Self {
            dir,
            meta,
            index,
            targets,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn meta(&self) -> &SessionMeta {
        &self.meta
    }

    pub fn camera(&self, id: &str) -> Option<&RecordedCamera> {
        self.meta.cameras.iter().find(|c| c.id == id)
    }

    pub fn index(&self) -> &[IndexRecord] {
        &self.index
    }

    pub fn targets(&self) -> &[TargetRecord] {
        &self.targets
    }

    pub fn duration(&self) -> Option<Duration> {
        let min = self.index.iter().map(|r| r.timestamp_ns).min()?;
        let max = self.index.iter().map(|r| r.timestamp_ns).max()?;
        Some(Duration::from_nanos(max - min))
    }

    pub fn read_frame(&self, record: &IndexRecord) -> Result<Frame, CaptureError> {
        let camera = self
            .camera(&record.camera)
            .ok_or_else(|| CaptureError::RecordingFormat {
                path: self.dir.clone(),
                reason: format!("no camera {} in session.toml", record.camera),
            })?;
        load_frame(
            &self.dir,
            camera,
            &CameraId::from(camera.id.as_str()),
            record,
        )
    }

    pub fn source(&self, camera: &str, pacing: Pacing) -> Result<ReplaySource, CaptureError> {
        if let Pacing::RealTime { speed } = pacing
            && !(speed.is_finite() && speed > 0.0)
        {
            return Err(CaptureError::RecordingFormat {
                path: self.dir.clone(),
                reason: format!("invalid replay speed {speed}"),
            });
        }
        let recorded =
            self.camera(camera)
                .cloned()
                .ok_or_else(|| CaptureError::RecordingFormat {
                    path: self.dir.clone(),
                    reason: format!("no camera {camera} in session.toml"),
                })?;
        let info = recorded.to_info();
        let records: Vec<IndexRecord> = self
            .index
            .iter()
            .filter(|r| r.camera == camera)
            .cloned()
            .collect();
        Ok(ReplaySource::new(
            self.dir.clone(),
            recorded,
            info,
            records,
            pacing,
        ))
    }

    /// Ordered by `(timestamp_ns, camera order in session.toml)`.
    pub fn merged_index(&self) -> Vec<&IndexRecord> {
        let order = |camera: &str| {
            self.meta
                .cameras
                .iter()
                .position(|c| c.id == camera)
                .unwrap_or(usize::MAX)
        };
        let mut out: Vec<&IndexRecord> = self.index.iter().collect();
        out.sort_by_key(|r| (r.timestamp_ns, order(&r.camera)));
        out
    }
}

#[cfg(test)]
mod tests {
    use eye_core::{Illumination, OutputId, session::TargetClock};

    use super::*;
    use crate::{
        pairing::Pairer,
        session::{EmitterState, PacingOption, SessionWriter},
        source::FrameSource,
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
                    format: RecordedFormat::Mjpeg,
                    width: 1280,
                    height: 720,
                    frame_interval_ns: 33_333_333,
                },
                RecordedCamera {
                    id: "ir".to_string(),
                    device: Some("/dev/video2".to_string()),
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

    fn target(seq: u64, shown_ns: u64) -> TargetRecord {
        TargetRecord {
            seq,
            shown_ns,
            hidden_ns: None,
            clock: TargetClock::Presentation,
            output: OutputId::from("eDP-1"),
            px_logical: [960.0, 540.0],
            mm: [155.0, 85.0],
        }
    }

    fn write_session(dir: &Path, n: u64) -> (PathBuf, Vec<Frame>) {
        let mut writer = SessionWriter::create(dir, &meta()).unwrap();
        let mut written = Vec::new();
        let mut rgb_seq = 0u64;
        for s in 1..=n {
            let illumination = if s <= 3 || s % 2 == 1 {
                Illumination::IrDark
            } else {
                Illumination::IrLit
            };
            let mut ir = gray_frame("ir", s, s * 68_000_000, 640, 360, (s * 7 % 256) as u8);
            ir.set_illumination(illumination);
            writer.write_frame(&ir).unwrap();
            written.push(ir);

            if s >= 3 && s % 2 == 1 {
                let seq = rgb_seq;
                rgb_seq += 1;
                let mut bytes = vec![0xFF, 0xD8];
                bytes.extend_from_slice(&(seq as u32).to_le_bytes());
                bytes.extend_from_slice(&[0xFF, 0xD9]);
                let rgb = mjpeg_frame("rgb", seq, s * 68_000_000 + 3_000_000, &bytes);
                writer.write_frame(&rgb).unwrap();
                written.push(rgb);
            }
        }
        writer.write_target(&target(0, 1_000_000_000)).unwrap();
        writer.write_target(&target(1, 3_000_000_000)).unwrap();
        written.sort_by_key(|f| f.header().timestamp);
        let dir = writer.finish().unwrap();
        (dir, written)
    }

    #[test]
    fn test_replay_is_bit_exact() {
        let root = tempfile::tempdir().unwrap();
        let (dir, written) = write_session(root.path(), 10);
        let recording = Recording::open(&dir).unwrap();

        let mut ir_source = recording.source("ir", Pacing::AsFastAsPossible).unwrap();
        let mut rgb_source = recording.source("rgb", Pacing::AsFastAsPossible).unwrap();

        let mut ir_expected: Vec<&Frame> = written
            .iter()
            .filter(|f| f.header().camera.as_str() == "ir")
            .collect();
        ir_expected.sort_by_key(|f| f.header().seq);
        let mut rgb_expected: Vec<&Frame> = written
            .iter()
            .filter(|f| f.header().camera.as_str() == "rgb")
            .collect();
        rgb_expected.sort_by_key(|f| f.header().seq);

        assert_eq!(ir_expected.len(), 10);
        assert_eq!(rgb_expected.len(), 4);

        for expected in &ir_expected {
            let frame = ir_source.next_frame().unwrap();
            assert_eq!(&frame, *expected);
        }
        assert!(matches!(
            ir_source.next_frame(),
            Err(CaptureError::EndOfStream)
        ));

        for expected in &rgb_expected {
            let frame = rgb_source.next_frame().unwrap();
            assert_eq!(&frame, *expected);
        }
        assert!(matches!(
            rgb_source.next_frame(),
            Err(CaptureError::EndOfStream)
        ));
    }

    #[test]
    fn test_open_reads_targets() {
        let root = tempfile::tempdir().unwrap();
        let (dir, _) = write_session(root.path(), 3);
        let recording = Recording::open(&dir).unwrap();
        assert_eq!(recording.targets().len(), 2);
        assert_eq!(
            serde_json::to_string(&recording.targets()[0]).unwrap(),
            serde_json::to_string(&target(0, 1_000_000_000)).unwrap()
        );
        assert_eq!(
            serde_json::to_string(&recording.targets()[1]).unwrap(),
            serde_json::to_string(&target(1, 3_000_000_000)).unwrap()
        );
    }

    #[test]
    fn test_corrupt_pgm_size_is_recording_format() {
        let root = tempfile::tempdir().unwrap();
        let (dir, _) = write_session(root.path(), 10);
        let frame_path = dir.join("frames").join("ir").join("00000002.pgm");
        fs::write(
            &frame_path,
            super::super::pnm::encode_pgm(640, 359, &vec![0u8; 640 * 359]),
        )
        .unwrap();

        let recording = Recording::open(&dir).unwrap();
        let mut source = recording.source("ir", Pacing::AsFastAsPossible).unwrap();
        source.next_frame().unwrap();
        let err = source.next_frame().unwrap_err();
        assert!(matches!(&err, CaptureError::RecordingFormat { path, .. } if *path == frame_path));
    }

    #[test]
    fn test_truncated_final_index_line_is_ignored() {
        let root = tempfile::tempdir().unwrap();
        let (dir, _) = write_session(root.path(), 3);
        let original_count = Recording::open(&dir).unwrap().index().len();

        let index_path = dir.join(INDEX_FILE);
        let mut text = fs::read_to_string(&index_path).unwrap();
        text.push_str(r#"{"seq":99,"cam"#);
        fs::write(&index_path, text).unwrap();

        let recording = Recording::open(&dir).unwrap();
        assert_eq!(recording.index().len(), original_count);
    }

    #[test]
    fn test_malformed_middle_line_is_error_with_line_number() {
        let root = tempfile::tempdir().unwrap();
        let (dir, _) = write_session(root.path(), 3);

        let index_path = dir.join(INDEX_FILE);
        let text = fs::read_to_string(&index_path).unwrap();
        let mut lines: Vec<&str> = text.lines().collect();
        lines[2] = "garbage";
        fs::write(&index_path, lines.join("\n") + "\n").unwrap();

        let err = Recording::open(&dir).unwrap_err();
        assert!(matches!(
            &err,
            CaptureError::RecordingFormat { reason, .. } if reason.starts_with("line 3:")
        ));
    }

    #[test]
    fn test_unknown_camera_in_index_is_error() {
        let root = tempfile::tempdir().unwrap();
        let (dir, _) = write_session(root.path(), 1);

        let index_path = dir.join(INDEX_FILE);
        let mut text = fs::read_to_string(&index_path).unwrap();
        text.push_str(r#"{"seq":0,"camera":"depth","timestamp_ns":0,"illumination":"unknown"}"#);
        text.push('\n');
        fs::write(&index_path, text).unwrap();

        let err = Recording::open(&dir).unwrap_err();
        assert!(matches!(err, CaptureError::RecordingFormat { .. }));
    }

    #[test]
    fn test_non_monotonic_timestamps_are_rejected() {
        let root = tempfile::tempdir().unwrap();
        let (dir, _) = write_session(root.path(), 3);

        let index_path = dir.join(INDEX_FILE);
        let text = fs::read_to_string(&index_path).unwrap();
        let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
        lines.swap(0, 1);
        fs::write(&index_path, lines.join("\n") + "\n").unwrap();

        let err = Recording::open(&dir).unwrap_err();
        assert!(matches!(err, CaptureError::RecordingFormat { .. }));
    }

    #[test]
    fn test_unsupported_format_version_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let (dir, _) = write_session(root.path(), 1);

        let session_path = dir.join(SESSION_FILE);
        let text = fs::read_to_string(&session_path).unwrap();
        let text = text.replacen("format_version = 1", "format_version = 2", 1);
        fs::write(&session_path, text).unwrap();

        let err = Recording::open(&dir).unwrap_err();
        assert!(matches!(err, CaptureError::RecordingFormat { .. }));
    }

    #[test]
    fn test_missing_frame_file_errors_on_that_frame() {
        let root = tempfile::tempdir().unwrap();
        let (dir, _) = write_session(root.path(), 3);
        fs::remove_file(dir.join("frames").join("ir").join("00000002.pgm")).unwrap();

        let recording = Recording::open(&dir).unwrap();
        let mut source = recording.source("ir", Pacing::AsFastAsPossible).unwrap();
        source.next_frame().unwrap();
        let err = source.next_frame().unwrap_err();
        assert!(matches!(err, CaptureError::RecordingIo { .. }));
    }

    #[test]
    fn test_realtime_pacing_respects_timestamps() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path();
        let mut writer = SessionWriter::create(dir, &meta()).unwrap();
        for i in 0..4u64 {
            let frame = gray_frame("ir", i, i * 33_333_333, 640, 360, 0);
            writer.write_frame(&frame).unwrap();
        }
        let dir = writer.finish().unwrap();
        let recording = Recording::open(&dir).unwrap();
        let mut source = recording
            .source("ir", Pacing::RealTime { speed: 1.0 })
            .unwrap();

        let start = std::time::Instant::now();
        for _ in 0..4 {
            source.next_frame().unwrap();
        }
        assert!(start.elapsed() >= Duration::from_millis(99));
    }

    #[test]
    fn test_invalid_speed_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let (dir, _) = write_session(root.path(), 1);
        let recording = Recording::open(&dir).unwrap();

        for speed in [0.0, -1.0, f64::NAN] {
            assert!(matches!(
                recording.source("ir", Pacing::RealTime { speed }),
                Err(CaptureError::RecordingFormat { .. })
            ));
        }

        assert_eq!(
            PacingOption::Fast.into_pacing(f64::NAN),
            Pacing::AsFastAsPossible
        );
        assert!(
            recording
                .source("ir", PacingOption::Fast.into_pacing(f64::NAN))
                .is_ok()
        );
    }

    #[test]
    fn test_merged_index_orders_by_timestamp_then_camera() {
        let root = tempfile::tempdir().unwrap();
        let writer_dir = root.path();
        let mut meta_both = meta();
        meta_both.cameras = vec![
            RecordedCamera {
                id: "rgb".to_string(),
                device: None,
                format: RecordedFormat::Mjpeg,
                width: 1280,
                height: 720,
                frame_interval_ns: 68_000_000,
            },
            RecordedCamera {
                id: "ir".to_string(),
                device: None,
                format: RecordedFormat::Gray8,
                width: 640,
                height: 360,
                frame_interval_ns: 68_000_000,
            },
        ];
        let mut writer = SessionWriter::create(writer_dir, &meta_both).unwrap();
        let ir0 = gray_frame("ir", 0, 0, 640, 360, 1);
        writer.write_frame(&ir0).unwrap();
        let rgb0 = mjpeg_frame("rgb", 0, 0, &[0xFF, 0xD8, 0xFF, 0xD9]);
        writer.write_frame(&rgb0).unwrap();
        let ir1 = gray_frame("ir", 1, 68_000_000, 640, 360, 2);
        writer.write_frame(&ir1).unwrap();
        let rgb1 = mjpeg_frame("rgb", 1, 68_000_000, &[0xFF, 0xD8, 0xFF, 0xD9]);
        writer.write_frame(&rgb1).unwrap();
        let dir = writer.finish().unwrap();

        let recording = Recording::open(&dir).unwrap();
        let merged = recording.merged_index();
        let order: Vec<(&str, u64)> = merged.iter().map(|r| (r.camera.as_str(), r.seq)).collect();
        assert_eq!(order, vec![("rgb", 0), ("ir", 0), ("rgb", 1), ("ir", 1)]);

        let (dir2, _) = write_session(root.path().join("other").as_path(), 10);
        let recording2 = Recording::open(&dir2).unwrap();
        let merged2 = recording2.merged_index();
        for pair in merged2.windows(2) {
            assert!(pair[0].timestamp_ns <= pair[1].timestamp_ns);
        }
    }

    #[test]
    fn test_replayed_pairing_is_identical_across_runs() {
        let root = tempfile::tempdir().unwrap();
        let (dir, _) = write_session(root.path(), 30);
        let recording = Recording::open(&dir).unwrap();
        let ir_info = recording.camera("ir").unwrap().to_info();
        let rgb_info = recording.camera("rgb").unwrap().to_info();

        let mut all_runs: Vec<Vec<(String, u64)>> = Vec::new();
        for _ in 0..5 {
            let mut pairer = Pairer::new(&[ir_info.clone(), rgb_info.clone()]).unwrap();
            let mut sets = Vec::new();
            for record in recording.merged_index() {
                let frame = recording.read_frame(record).unwrap();
                sets.extend(pairer.push(frame));
            }
            if let Some(set) = pairer.flush() {
                sets.push(set);
            }
            let seqs: Vec<(String, u64)> = sets
                .iter()
                .flat_map(|s| {
                    s.frames()
                        .iter()
                        .map(|f| (f.header().camera.to_string(), f.header().seq))
                })
                .collect();
            all_runs.push(seqs);
            assert_eq!(sets.len(), 30);
        }

        for run in &all_runs[1..] {
            assert_eq!(run, &all_runs[0]);
        }

        let expected_pairs: Vec<(u64, u64)> = (0..14).map(|j| (2 * j + 4, j)).collect();
        let mut found_pairs = Vec::new();
        let mut ir_alone = 0;
        let mut rgb_alone = 0;
        let mut idx = 0;
        while idx < all_runs[0].len() {
            let (camera, seq) = &all_runs[0][idx];
            if camera == "ir" {
                if idx + 1 < all_runs[0].len() && all_runs[0][idx + 1].0 == "rgb" {
                    found_pairs.push((*seq, all_runs[0][idx + 1].1));
                    idx += 2;
                    continue;
                }
                ir_alone += 1;
            } else {
                rgb_alone += 1;
            }
            idx += 1;
        }
        assert_eq!(found_pairs, expected_pairs);
        assert_eq!(ir_alone, 16);
        assert_eq!(rgb_alone, 0);
    }

    #[test]
    #[ignore = "needs EYE_RECORDING"]
    fn test_real_recording_replays_cleanly() {
        let dir =
            std::env::var("EYE_RECORDING").expect("EYE_RECORDING must point to a session dir");
        let recording = Recording::open(&dir).unwrap();
        for camera in &recording.meta().cameras.clone() {
            let mut source = recording
                .source(&camera.id, Pacing::AsFastAsPossible)
                .unwrap();
            let expected = recording
                .index()
                .iter()
                .filter(|r| r.camera == camera.id)
                .count();
            let mut count = 0;
            let mut last_ts: Option<u64> = None;
            loop {
                match source.next_frame() {
                    Ok(frame) => {
                        let ts = frame.header().timestamp.as_nanos();
                        assert!(last_ts.is_none_or(|l| ts >= l));
                        last_ts = Some(ts);
                        count += 1;
                    }
                    Err(CaptureError::EndOfStream) => break,
                    Err(e) => panic!("{e}"),
                }
            }
            assert_eq!(count, expected);
            println!(
                "{}: {count} frames over {:?}",
                camera.id,
                recording.duration()
            );
        }
    }
}
