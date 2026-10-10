use std::{io, path::PathBuf, sync::Arc, time::Duration};

use eye_core::{
    CameraId, CameraInfo, Frame, FrameHeader, Illumination, PixelFormat, Timestamp, log::field,
};
use v4l::{
    Device, Format,
    buffer::Flags,
    io::traits::CaptureStream,
    video::{Capture, capture::Parameters},
};

use crate::{CaptureError, format::StoredFormat, source::FrameSource};

fn default_fps() -> u32 {
    30
}

fn default_buffers() -> u32 {
    4
}

fn default_timeout_ms() -> u64 {
    2000
}

#[derive(Debug, Clone)]
pub struct V4l2Config {
    pub id: CameraId,
    pub device: PathBuf,
    pub format: PixelFormat,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub buffers: u32,
    pub timeout: Duration,
}

impl V4l2Config {
    /// fps 30, buffers 4, timeout 2000 ms.
    pub fn new(
        id: CameraId,
        device: PathBuf,
        format: PixelFormat,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            id,
            device,
            format,
            width,
            height,
            fps: default_fps(),
            buffers: default_buffers(),
            timeout: Duration::from_millis(default_timeout_ms()),
        }
    }
}

pub struct V4l2Source {
    info: CameraInfo,
    path: PathBuf,
    stride: u32,
    timeout: Duration,
    stream: v4l::io::mmap::Stream<'static>,
    seq: SeqWidener,
    started: bool,
    timeout_state: TimeoutState,
}

#[derive(Debug, Default)]
struct TimeoutState {
    pending_dequeue: bool,
}

impl TimeoutState {
    fn on_timeout(&mut self, started: bool) {
        self.pending_dequeue = started;
    }

    fn needs_dequeue(&self) -> bool {
        self.pending_dequeue
    }

    fn on_dequeued(&mut self) {
        self.pending_dequeue = false;
    }
}

impl std::fmt::Debug for V4l2Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("V4l2Source")
            .field("info", &self.info)
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl V4l2Source {
    pub fn open(config: V4l2Config) -> Result<Self, CaptureError> {
        let V4l2Config {
            id,
            device,
            format,
            width,
            height,
            fps,
            buffers,
            timeout,
        } = config;

        let fourcc = StoredFormat::try_from(format)
            .map_err(|f| CaptureError::UnsupportedFormat {
                path: device.clone(),
                format: format!("{f:?}"),
            })?
            .fourcc();

        let dev = Device::with_path(&device).map_err(|source| CaptureError::Open {
            path: device.clone(),
            source,
        })?;

        let requested = Format::new(width, height, fourcc);
        let actual =
            Capture::set_format(&dev, &requested).map_err(|source| CaptureError::Open {
                path: device.clone(),
                source,
            })?;
        if actual.width != width || actual.height != height || actual.fourcc != fourcc {
            return Err(CaptureError::FormatRejected {
                path: device,
                requested: format!("{width}x{height} {fourcc}"),
                actual: format!("{}x{} {}", actual.width, actual.height, actual.fourcc),
            });
        }

        let params = Capture::set_params(&dev, &Parameters::with_fps(fps)).map_err(|source| {
            CaptureError::Open {
                path: device.clone(),
                source,
            }
        })?;
        let frame_interval = Duration::from_secs_f64(
            f64::from(params.interval.numerator) / f64::from(params.interval.denominator),
        );

        let mut stream: v4l::io::mmap::Stream<'static> =
            v4l::io::mmap::Stream::with_buffers(&dev, v4l::buffer::Type::VideoCapture, buffers)
                .map_err(|source| CaptureError::Open {
                    path: device.clone(),
                    source,
                })?;
        stream.set_timeout(timeout);

        let stride = if actual.stride == 0 {
            width
        } else {
            actual.stride
        };

        tracing::info!(
            { field::CAMERA } = id.as_str(),
            device = %device.display(),
            width,
            height,
            format = ?format,
            fps,
            frame_interval_us = frame_interval.as_micros() as u64,
            buffers,
            stride,
            "camera opened"
        );

        Ok(Self {
            info: CameraInfo {
                id,
                width,
                height,
                format,
                frame_interval,
            },
            path: device,
            stride,
            timeout,
            stream,
            seq: SeqWidener::default(),
            started: false,
            timeout_state: TimeoutState::default(),
        })
    }

    /// Typed camera constructor (R3: frame sources are not registered). `options` is the
    /// `[[camera]]` table minus `id`/`source`/`emitter`.
    pub fn from_config(id: CameraId, options: &toml::Table) -> Result<Self, CaptureError> {
        Self::open(V4l2Options::parse(&id, options)?.into_config(id))
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct V4l2Options {
    pub device: PathBuf,
    pub format: StoredFormat,
    pub size: [u32; 2],
    #[serde(default = "default_fps")]
    pub fps: u32,
    #[serde(default = "default_buffers")]
    pub buffers: u32,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

impl V4l2Options {
    pub fn parse(id: &CameraId, options: &toml::Table) -> Result<Self, CaptureError> {
        options
            .clone()
            .try_into()
            .map_err(|e: toml::de::Error| CaptureError::Config {
                camera: id.to_string(),
                reason: e.to_string(),
            })
    }

    pub fn into_config(self, id: CameraId) -> V4l2Config {
        V4l2Config {
            id,
            device: self.device,
            format: self.format.pixel(),
            width: self.size[0],
            height: self.size[1],
            fps: self.fps,
            buffers: self.buffers,
            timeout: Duration::from_millis(self.timeout_ms),
        }
    }
}

impl FrameSource for V4l2Source {
    fn camera(&self) -> &CameraInfo {
        &self.info
    }

    fn next_frame(&mut self) -> Result<Frame, CaptureError> {
        const ENODEV: i32 = 19;
        let camera = self.info.id.to_string();
        loop {
            if self.timeout_state.needs_dequeue() {
                match CaptureStream::dequeue(&mut self.stream) {
                    Ok(_) => {
                        self.timeout_state.on_dequeued();
                        tracing::debug!(
                            { field::CAMERA } = camera.as_str(),
                            "recovered pending dequeue after timeout, dropping frame"
                        );
                    }
                    Err(e) if e.kind() == io::ErrorKind::TimedOut => {
                        return Err(timeout_error(camera, self.timeout));
                    }
                    Err(e) if e.raw_os_error() == Some(ENODEV) => {
                        return Err(CaptureError::Disconnected { camera });
                    }
                    Err(source) => return Err(CaptureError::Io { camera, source }),
                }
                continue;
            }

            let (buf, meta) = match CaptureStream::next(&mut self.stream) {
                Ok(next) => {
                    self.started = true;
                    next
                }
                Err(e) if e.kind() == io::ErrorKind::TimedOut => {
                    self.started = true;
                    self.timeout_state.on_timeout(self.started);
                    return Err(timeout_error(camera, self.timeout));
                }
                Err(e) if e.raw_os_error() == Some(ENODEV) => {
                    return Err(CaptureError::Disconnected { camera });
                }
                Err(source) => return Err(CaptureError::Io { camera, source }),
            };
            check_clock(meta.flags).map_err(|flags| CaptureError::NotMonotonic {
                camera: camera.clone(),
                flags,
            })?;
            if meta.flags.contains(Flags::ERROR) {
                tracing::warn!(
                    { field::CAMERA } = camera.as_str(),
                    { field::SEQ } = meta.sequence,
                    "driver flagged a corrupt buffer"
                );
                continue;
            }
            let used = (meta.bytesused as usize).min(buf.len());
            let payload = match self.info.format {
                PixelFormat::Mjpeg => mjpeg_payload(buf, used),
                _ => gray_payload(buf, used, self.info.width, self.info.height, self.stride),
            };
            let Some(data) = payload else {
                tracing::warn!(
                    { field::CAMERA } = camera.as_str(),
                    { field::SEQ } = meta.sequence,
                    used,
                    "dropping short or invalid frame"
                );
                continue;
            };
            let header = FrameHeader {
                camera: self.info.id.clone(),
                seq: self.seq.widen(meta.sequence),
                timestamp: to_timestamp(meta.timestamp),
                width: self.info.width,
                height: self.info.height,
                format: self.info.format,
                illumination: match self.info.format {
                    PixelFormat::Mjpeg => Illumination::Ambient,
                    _ => Illumination::Unknown,
                },
            };
            let _frame_span = eye_core::log::frame_span(
                header.camera.as_str(),
                header.seq,
                header.timestamp.as_nanos(),
                header.illumination.as_str(),
                1,
            )
            .entered();
            tracing::trace!(bytes = data.len(), source = "v4l2", "frame produced");
            return Ok(Frame::new(header, data)?);
        }
    }
}

fn timeout_error(camera: String, timeout: Duration) -> CaptureError {
    tracing::warn!(
        { field::CAMERA } = camera.as_str(),
        timeout_ms = timeout.as_millis() as u64,
        "no frame within timeout"
    );
    CaptureError::Timeout { camera, timeout }
}

pub(crate) fn check_clock(flags: v4l::buffer::Flags) -> Result<(), u32> {
    if flags.bits() & Flags::TIMESTAMP_MASK.bits() == Flags::TIMESTAMP_MONOTONIC.bits() {
        Ok(())
    } else {
        Err(flags.bits())
    }
}

pub(crate) fn to_timestamp(ts: v4l::Timestamp) -> Timestamp {
    Timestamp(Duration::from(ts))
}

pub(crate) fn gray_payload(
    buf: &[u8],
    used: usize,
    width: u32,
    height: u32,
    stride: u32,
) -> Option<Arc<[u8]>> {
    let (w, h, s) = (width as usize, height as usize, stride as usize);
    if h == 0 || s < w || used < s * (h - 1) + w {
        return None;
    }
    if s == w {
        return Some(Arc::from(&buf[..w * h]));
    }
    let mut packed = Vec::with_capacity(w * h);
    for row in buf[..used].chunks(s).take(h) {
        packed.extend_from_slice(&row[..w]);
    }
    Some(packed.into())
}

pub(crate) fn mjpeg_payload(buf: &[u8], used: usize) -> Option<Arc<[u8]>> {
    let b = buf.get(..used)?;
    b.starts_with(&[0xFF, 0xD8]).then(|| Arc::from(b))
}

#[derive(Debug, Default)]
pub(crate) struct SeqWidener {
    last: Option<u32>,
    high: u64,
}

impl SeqWidener {
    pub fn widen(&mut self, seq: u32) -> u64 {
        if let Some(last) = self.last
            && seq < last
            && last - seq > u32::MAX / 2
        {
            self.high += 1 << 32;
        }
        self.last = Some(seq);
        self.high | u64::from(seq)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn test_v4l_timestamp_converts_to_duration() {
        assert_eq!(
            to_timestamp(v4l::Timestamp::new(5, 250_000)),
            Timestamp(Duration::from_micros(5_250_000))
        );
    }

    #[test]
    fn test_monotonic_timestamp_flags_are_accepted() {
        assert!(check_clock(Flags::from(0x0001_2001)).is_ok());
    }

    #[test]
    fn test_unknown_timestamp_type_is_rejected() {
        assert_eq!(check_clock(Flags::from(0x0000_0001)), Err(1));
    }

    #[test]
    fn test_copy_timestamp_type_is_rejected() {
        assert_eq!(check_clock(Flags::from(0x4000)), Err(0x4000));
    }

    #[test]
    fn test_sequence_widening_crosses_u32_wrap() {
        let mut widener = SeqWidener::default();
        assert_eq!(widener.widen(0xFFFF_FFFE), 4_294_967_294);
        assert_eq!(widener.widen(0xFFFF_FFFF), 4_294_967_295);
        assert_eq!(widener.widen(0), 4_294_967_296);
        assert_eq!(widener.widen(1), 4_294_967_297);
    }

    #[test]
    fn test_sequence_gap_is_preserved() {
        let mut widener = SeqWidener::default();
        assert_eq!(widener.widen(10), 10);
        assert_eq!(widener.widen(12), 12);
        assert_eq!(widener.widen(13), 13);
    }

    #[test]
    fn test_gray_payload_copies_exact_image_from_larger_buffer() {
        let len = 640 * 360 + 4096;
        let buf: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        let used = 640 * 360;
        let payload = gray_payload(&buf, used, 640, 360, 640).unwrap();
        assert_eq!(payload.len(), 230_400);
        assert_eq!(&*payload, &buf[..230_400]);
    }

    #[test]
    fn test_gray_payload_repacks_padded_stride() {
        let buf: [u8; 18] = [1, 1, 1, 1, 9, 9, 2, 2, 2, 2, 9, 9, 3, 3, 3, 3, 9, 9];
        let payload = gray_payload(&buf, 18, 4, 3, 6).unwrap();
        assert_eq!(&*payload, &[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);

        let payload = gray_payload(&buf, 16, 4, 3, 6);
        assert!(payload.is_some());
    }

    #[test]
    fn test_short_gray_frame_is_dropped() {
        let buf = vec![0u8; 640 * 360];
        let used = 640 * 359;
        assert!(gray_payload(&buf, used, 640, 360, 640).is_none());
    }

    #[test]
    fn test_mjpeg_payload_truncates_to_bytesused() {
        let buf: [u8; 13] = [
            0xFF, 0xD8, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0xFF, 0xD9, 0x00, 0x00, 0x00,
        ];
        let payload = mjpeg_payload(&buf, 10).unwrap();
        assert_eq!(&*payload, &buf[..10]);
    }

    #[test]
    fn test_mjpeg_without_soi_is_dropped() {
        let buf: [u8; 4] = [0x00, 0x00, 0xFF, 0xD9];
        assert!(mjpeg_payload(&buf, 4).is_none());
        assert!(mjpeg_payload(&buf, 0).is_none());
    }

    #[test]
    fn test_options_parse_root_example_camera() {
        let toml_str = r#"
            device = "/dev/video2"
            format = "gray"
            size = [640, 360]
        "#;
        let table: toml::Table = toml_str.parse().unwrap();
        let options: V4l2Options = table.try_into().unwrap();
        assert_eq!(options.fps, 30);
        assert_eq!(options.buffers, 4);
        assert_eq!(options.timeout_ms, 2000);

        let config = options.into_config(CameraId::from("ir"));
        assert_eq!(config.format, PixelFormat::Gray8);
        assert_eq!(config.width, 640);
        assert_eq!(config.height, 360);
        assert_eq!(config.timeout, Duration::from_millis(2000));
    }

    #[test]
    fn test_options_reject_unknown_key() {
        let toml_str = r#"
            device = "/dev/video2"
            format = "gray"
            size = [640, 360]
            exposure = 3
        "#;
        let table: toml::Table = toml_str.parse().unwrap();
        let result = V4l2Source::from_config(CameraId::from("ir"), &table);
        assert!(matches!(result, Err(CaptureError::Config { ref camera, .. }) if camera == "ir"));
    }

    #[test]
    fn test_v4l2_options_rejects_unknown_format() {
        let toml_str = r#"
            device = "/dev/video0"
            format = "rgb"
            size = [1280, 720]
        "#;
        let table: toml::Table = toml_str.parse().unwrap();
        let result: Result<V4l2Options, _> = table.try_into();
        assert!(result.is_err());
    }

    #[test]
    fn test_open_rejects_rgb8_before_touching_device() {
        let config = V4l2Config::new(
            CameraId::from("rgb"),
            PathBuf::from("/nonexistent"),
            PixelFormat::Rgb8,
            640,
            360,
        );
        let result = V4l2Source::open(config);
        assert!(matches!(
            result,
            Err(CaptureError::UnsupportedFormat { .. })
        ));
    }

    #[test]
    fn test_open_missing_device_is_open_error() {
        let config = V4l2Config::new(
            CameraId::from("ir"),
            PathBuf::from("/dev/video-does-not-exist"),
            PixelFormat::Gray8,
            640,
            360,
        );
        let result = V4l2Source::open(config);
        assert!(matches!(
            result,
            Err(CaptureError::Open { ref path, .. }) if path == Path::new("/dev/video-does-not-exist")
        ));
    }

    #[test]
    fn test_v4l2_source_is_send() {
        fn f<T: Send>() {}
        f::<V4l2Source>();
    }

    #[test]
    fn test_pending_dequeue_is_set_only_after_start() {
        let mut state = TimeoutState::default();
        assert!(!state.needs_dequeue());

        state.on_timeout(false);
        assert!(!state.needs_dequeue());

        state.on_timeout(true);
        assert!(state.needs_dequeue());

        state.on_dequeued();
        assert!(!state.needs_dequeue());
    }

    #[test]
    #[ignore = "needs hardware"]
    fn test_live_dual_first_frame_timeout_recovers() {
        let ir_config = V4l2Config {
            timeout: Duration::from_millis(50),
            ..V4l2Config::new(
                CameraId::from("ir"),
                PathBuf::from("/dev/video2"),
                PixelFormat::Gray8,
                640,
                360,
            )
        };
        let mut ir = V4l2Source::open(ir_config).unwrap();
        for attempt in 0.. {
            match ir.next_frame() {
                Ok(_) => break,
                Err(CaptureError::Timeout { .. }) if attempt < 40 => continue,
                Err(e) => panic!("ir first frame failed: {e}"),
            }
        }

        let rgb_config = V4l2Config {
            timeout: Duration::from_millis(50),
            ..V4l2Config::new(
                CameraId::from("rgb"),
                PathBuf::from("/dev/video0"),
                PixelFormat::Mjpeg,
                1280,
                720,
            )
        };
        let mut rgb = V4l2Source::open(rgb_config).unwrap();

        let mut timed_out = false;
        let mut got_frame = false;
        for _ in 0..30 {
            match rgb.next_frame() {
                Ok(_) => {
                    got_frame = true;
                    break;
                }
                Err(CaptureError::Timeout { .. }) => timed_out = true,
                Err(e) => panic!("unexpected error before first frame: {e}"),
            }
        }
        assert!(timed_out, "expected at least one timeout before recovery");
        assert!(got_frame, "expected a frame after the recovery dequeue");

        for _ in 0..30 {
            match rgb.next_frame() {
                Ok(_) | Err(CaptureError::Timeout { .. }) => {}
                Err(e) => panic!("unexpected error after recovery: {e}"),
            }
        }
    }

    #[test]
    #[ignore = "needs hardware"]
    fn test_ir_node_streams_gray_640x360() {
        let config = V4l2Config::new(
            CameraId::from("ir"),
            PathBuf::from("/dev/video2"),
            PixelFormat::Gray8,
            640,
            360,
        );
        let (interval, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut source = V4l2Source::open(config).unwrap();
            let mut timestamps = Vec::new();
            for _ in 0..30 {
                let frame = source.next_frame().unwrap();
                assert_eq!(frame.data().len(), 230_400);
                assert_eq!(frame.header().format, PixelFormat::Gray8);
                assert_eq!(frame.header().illumination, Illumination::Unknown);
                timestamps.push(frame.header().timestamp);
            }
            source.camera().frame_interval
        });
        assert!((interval.as_secs_f64() - 0.033_333).abs() < 0.001);

        let opened: Vec<_> = records
            .iter()
            .filter(|r| r.message == "camera opened")
            .collect();
        assert_eq!(opened.len(), 1);
        assert_eq!(opened[0].level, eye_log::Level::Info);

        let produced: Vec<_> = records
            .iter()
            .filter(|r| r.message == "frame produced")
            .collect();
        assert_eq!(produced.len(), 30);
        for rec in &produced {
            assert_eq!(rec.level, eye_log::Level::Trace);
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
    }

    #[test]
    #[ignore = "needs hardware"]
    fn test_rgb_node_streams_mjpeg_1280x720() {
        let config = V4l2Config::new(
            CameraId::from("rgb"),
            PathBuf::from("/dev/video0"),
            PixelFormat::Mjpeg,
            1280,
            720,
        );
        let mut source = V4l2Source::open(config).unwrap();
        let mut timestamps = Vec::new();
        for _ in 0..30 {
            let frame = source.next_frame().unwrap();
            assert!(frame.data().starts_with(&[0xFF, 0xD8]));
            timestamps.push(frame.header().timestamp);
        }
        let median = median_delta_ms(&timestamps);
        assert!((30.0..=36.0).contains(&median), "median delta {median} ms");
    }

    #[test]
    #[ignore = "needs hardware"]
    fn test_buffer_timestamps_are_monotonic() {
        let config = V4l2Config::new(
            CameraId::from("ir"),
            PathBuf::from("/dev/video2"),
            PixelFormat::Gray8,
            640,
            360,
        );
        let mut source = V4l2Source::open(config).unwrap();
        let mut timestamps = Vec::new();
        for _ in 0..30 {
            let frame = source.next_frame().unwrap();
            timestamps.push(frame.header().timestamp);
        }
        for w in timestamps.windows(2) {
            assert!(w[1] > w[0]);
        }
        let median = median_delta_ms(&timestamps);
        println!("median IR delta: {median} ms");
        assert!((60.0..=72.0).contains(&median), "median delta {median} ms");
    }

    #[test]
    #[ignore = "needs hardware"]
    fn test_sequence_is_consecutive_under_light_load() {
        let config = V4l2Config::new(
            CameraId::from("ir"),
            PathBuf::from("/dev/video2"),
            PixelFormat::Gray8,
            640,
            360,
        );
        let mut source = V4l2Source::open(config).unwrap();
        let mut seqs = Vec::new();
        for _ in 0..60 {
            let frame = source.next_frame().unwrap();
            seqs.push(frame.header().seq);
        }
        let consecutive = seqs.windows(2).filter(|w| w[1] - w[0] == 1).count();
        assert!(
            consecutive >= 58,
            "only {consecutive} of 59 deltas were consecutive"
        );
    }

    struct SetOnDrop<'a>(&'a std::sync::atomic::AtomicBool);

    impl Drop for SetOnDrop<'_> {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[test]
    #[ignore = "needs hardware"]
    fn test_rgb_and_ir_stream_simultaneously() {
        let (ir_started_tx, ir_started_rx) = std::sync::mpsc::channel::<()>();
        let rgb_done = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            let ir = scope.spawn(|| {
                let ir_started_tx = ir_started_tx;
                let config = V4l2Config::new(
                    CameraId::from("ir"),
                    PathBuf::from("/dev/video2"),
                    PixelFormat::Gray8,
                    640,
                    360,
                );
                let mut source = V4l2Source::open(config).unwrap();
                let mut timestamps = Vec::new();
                timestamps.push(source.next_frame().unwrap().header().timestamp);
                ir_started_tx.send(()).unwrap();
                for _ in 0..19 {
                    timestamps.push(source.next_frame().unwrap().header().timestamp);
                }
                // Keep dequeuing after our 20 frames: if IR stops streaming
                // before RGB's 20 frames arrive, RGB's rate is no longer
                // measured under contention.
                while !rgb_done.load(std::sync::atomic::Ordering::Relaxed) {
                    let _ = source.next_frame();
                }
                median_delta_ms(&timestamps)
            });
            if ir_started_rx.recv().is_err() {
                panic!("IR thread exited before streaming started");
            }
            let rgb = scope.spawn(|| {
                let _done = SetOnDrop(&rgb_done);
                let config = V4l2Config::new(
                    CameraId::from("rgb"),
                    PathBuf::from("/dev/video0"),
                    PixelFormat::Mjpeg,
                    1280,
                    720,
                );
                let mut source = V4l2Source::open(config).unwrap();
                let mut timestamps = Vec::new();
                for _ in 0..20 {
                    timestamps.push(source.next_frame().unwrap().header().timestamp);
                }
                median_delta_ms(&timestamps)
            });

            let rgb_median = rgb.join().unwrap();
            let ir_median = ir.join().unwrap();
            println!("rgb median: {rgb_median} ms, ir median: {ir_median} ms");
            assert!(
                (120.0..=145.0).contains(&rgb_median),
                "rgb median {rgb_median} ms"
            );
            assert!(
                (60.0..=72.0).contains(&ir_median),
                "ir median {ir_median} ms"
            );
        });
    }

    #[cfg(test)]
    fn median_delta_ms(timestamps: &[Timestamp]) -> f64 {
        let mut deltas: Vec<f64> = timestamps
            .windows(2)
            .map(|w| w[1].nanos_since(w[0]) as f64 / 1_000_000.0)
            .collect();
        deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
        deltas[deltas.len() / 2]
    }
}
