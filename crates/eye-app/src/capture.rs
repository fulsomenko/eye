//! Reads frames off the configured [`FrameSource`]s on dedicated threads and forwards them to
//! the recording thread over one shared channel.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;
use eye_capture::{CaptureError, FrameSource};

use crate::commands::record::RecordSink;

const CHANNEL_CAPACITY: usize = 64;

#[derive(Debug)]
pub enum CaptureMsg {
    Frame(eye_core::Frame),
    Failed { camera: String, error: String },
}

/// One named thread (`eye-rec-<id>`) per source. Drop sets the stop flag and joins every thread.
#[derive(Debug)]
pub struct Capture {
    rx: Receiver<CaptureMsg>,
    stop: Arc<AtomicBool>,
    dropped: Arc<AtomicU64>,
    threads: Vec<JoinHandle<()>>,
    cameras: Vec<String>,
}

fn capture_loop(
    mut source: Box<dyn FrameSource>,
    id: String,
    tx: crossbeam_channel::Sender<CaptureMsg>,
    stop: Arc<AtomicBool>,
    dropped: Arc<AtomicU64>,
) {
    let mut last_warn: Option<Instant> = None;
    while !stop.load(Ordering::Relaxed) {
        match source.next_frame() {
            Ok(frame) => {
                if tx.try_send(CaptureMsg::Frame(frame)).is_err() {
                    dropped.fetch_add(1, Ordering::Relaxed);
                    let now = Instant::now();
                    if last_warn.is_none_or(|t| now.duration_since(t) >= Duration::from_secs(1)) {
                        tracing::warn!(camera = %id, "capture channel full; dropping a frame");
                        last_warn = Some(now);
                    }
                }
            }
            Err(CaptureError::Timeout { .. }) => continue,
            Err(CaptureError::EndOfStream) => break,
            Err(error) => {
                let mut msg = Some(CaptureMsg::Failed {
                    camera: id.clone(),
                    error: error.to_string(),
                });
                while !stop.load(Ordering::Relaxed) {
                    match tx.send_timeout(
                        msg.take().expect("msg set before retry"),
                        Duration::from_millis(50),
                    ) {
                        Ok(()) => break,
                        Err(crossbeam_channel::SendTimeoutError::Timeout(m)) => msg = Some(m),
                        Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => break,
                    }
                }
                if let Some(CaptureMsg::Failed { camera, error }) = msg {
                    tracing::error!(camera = %camera, %error, "camera failed and the failure could not be delivered before stop");
                }
                break;
            }
        }
    }
}

impl Capture {
    pub fn spawn(sources: Vec<Box<dyn FrameSource>>) -> anyhow::Result<Capture> {
        let (tx, rx) = crossbeam_channel::bounded(CHANNEL_CAPACITY);
        let stop = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicU64::new(0));
        let mut threads = Vec::with_capacity(sources.len());
        let mut cameras = Vec::with_capacity(sources.len());
        for source in sources {
            let id = source.camera().id.to_string();
            cameras.push(id.clone());
            let tx = tx.clone();
            let stop = Arc::clone(&stop);
            let dropped = Arc::clone(&dropped);
            let parent = tracing::Span::current();
            let handle = std::thread::Builder::new()
                .name(format!("eye-rec-{id}"))
                .spawn(move || {
                    let _enter = parent.entered();
                    capture_loop(source, id, tx, stop, dropped)
                })?;
            threads.push(handle);
        }
        drop(tx);
        Ok(Capture {
            rx,
            stop,
            dropped,
            threads,
            cameras,
        })
    }

    pub fn frames(&self) -> &Receiver<CaptureMsg> {
        &self.rx
    }

    /// Writes frames until every camera delivered one; error naming the silent cameras after `timeout`.
    pub fn preroll(&self, timeout: Duration, sink: &mut dyn RecordSink) -> anyhow::Result<()> {
        let mut seen: HashSet<&str> = HashSet::new();
        let deadline = crossbeam_channel::after(timeout);
        loop {
            if seen.len() == self.cameras.len() {
                return Ok(());
            }
            crossbeam_channel::select! {
                recv(self.rx) -> msg => match msg {
                    Ok(CaptureMsg::Frame(frame)) => {
                        let camera = frame.header().camera.as_str();
                        if let Some(id) = self.cameras.iter().find(|c| c.as_str() == camera) {
                            seen.insert(id.as_str());
                        }
                        sink.frame(&frame)?;
                    }
                    Ok(CaptureMsg::Failed { camera, error }) => {
                        anyhow::bail!("camera {camera} failed: {error}")
                    }
                    Err(_) => anyhow::bail!("all capture threads stopped"),
                },
                recv(deadline) -> _ => {
                    let silent: Vec<&str> = self
                        .cameras
                        .iter()
                        .map(String::as_str)
                        .filter(|c| !seen.contains(c))
                        .collect();
                    anyhow::bail!(
                        "no frame from {} within {timeout:?}",
                        silent.join(", ")
                    );
                }
            }
        }
    }

    /// Writes frames for `duration` (wall clock).
    pub fn write_for(&self, duration: Duration, sink: &mut dyn RecordSink) -> anyhow::Result<()> {
        let deadline = crossbeam_channel::after(duration);
        loop {
            crossbeam_channel::select! {
                recv(self.rx) -> msg => match msg {
                    Ok(CaptureMsg::Frame(frame)) => sink.frame(&frame)?,
                    Ok(CaptureMsg::Failed { camera, error }) => {
                        anyhow::bail!("camera {camera} failed: {error}")
                    }
                    Err(_) => anyhow::bail!("all capture threads stopped"),
                },
                recv(deadline) -> _ => return Ok(()),
            }
        }
    }

    /// Stops and joins the threads (closing the streams), writes what is left in the channel, returns `dropped`.
    pub fn stop_and_drain(mut self, sink: &mut dyn RecordSink) -> anyhow::Result<u64> {
        self.stop_and_join();
        while let Ok(msg) = self.rx.try_recv() {
            if let CaptureMsg::Frame(frame) = msg {
                sink.frame(&frame)?;
            }
        }
        Ok(self.dropped.load(Ordering::Relaxed))
    }

    fn stop_and_join(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for handle in std::mem::take(&mut self.threads) {
            let _ = handle.join();
        }
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    use eye_core::{
        CameraId, CameraInfo, Frame, FrameHeader, Illumination, PixelFormat, Timestamp,
    };

    use super::*;

    fn frame(camera: &str, seq: u64) -> eye_core::Frame {
        Frame::new(
            FrameHeader {
                camera: CameraId::from(camera),
                seq,
                timestamp: Timestamp::from_nanos(seq),
                width: 2,
                height: 2,
                format: PixelFormat::Gray8,
                illumination: Illumination::Unknown,
            },
            Arc::from(vec![0u8; 4]),
        )
        .expect("valid frame")
    }

    struct FakeSource {
        camera: CameraInfo,
        dropped: Arc<AtomicBool>,
    }

    impl FrameSource for FakeSource {
        fn camera(&self) -> &CameraInfo {
            &self.camera
        }

        fn next_frame(&mut self) -> Result<eye_core::Frame, CaptureError> {
            std::thread::sleep(Duration::from_millis(5));
            Err(CaptureError::Timeout {
                camera: self.camera.id.to_string(),
                timeout: Duration::from_millis(5),
            })
        }
    }

    impl Drop for FakeSource {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    struct OverflowThenFailSource {
        camera: CameraInfo,
        seq: u64,
    }

    impl FrameSource for OverflowThenFailSource {
        fn camera(&self) -> &CameraInfo {
            &self.camera
        }

        fn next_frame(&mut self) -> Result<eye_core::Frame, CaptureError> {
            self.seq += 1;
            if self.seq as usize <= CHANNEL_CAPACITY + 5 {
                Ok(frame("ir", self.seq))
            } else {
                Err(CaptureError::Disconnected {
                    camera: self.camera.id.to_string(),
                })
            }
        }
    }

    #[test]
    fn test_capture_failure_delivered_when_channel_fills_before_draining() {
        let source: Box<dyn FrameSource> = Box::new(OverflowThenFailSource {
            camera: CameraInfo {
                id: CameraId::from("ir"),
                width: 2,
                height: 2,
                format: PixelFormat::Gray8,
                frame_interval: Duration::from_millis(1),
            },
            seq: 0,
        });
        let capture = Capture::spawn(vec![source]).unwrap();

        // Nobody drains `capture.frames()` yet, so the channel fills past CHANNEL_CAPACITY
        // and the source hits its error while the channel is still full.
        std::thread::sleep(Duration::from_millis(200));

        let mut failed = None;
        for _ in 0..(CHANNEL_CAPACITY + 50) {
            match capture.frames().recv_timeout(Duration::from_secs(5)) {
                Ok(CaptureMsg::Failed { camera, error }) => {
                    failed = Some((camera, error));
                    break;
                }
                Ok(CaptureMsg::Frame(_)) => continue,
                Err(_) => break,
            }
        }

        assert_eq!(
            failed.map(|(camera, _)| camera),
            Some("ir".to_string()),
            "expected a CaptureMsg::Failed to be delivered even though the channel was full \
             while the camera failed"
        );
    }

    #[test]
    fn test_drop_stops_and_joins_threads_without_stop_and_drain() {
        let dropped = Arc::new(AtomicBool::new(false));
        let source: Box<dyn FrameSource> = Box::new(FakeSource {
            camera: CameraInfo {
                id: CameraId::from("ir"),
                width: 640,
                height: 360,
                format: PixelFormat::Gray8,
                frame_interval: Duration::from_millis(5),
            },
            dropped: Arc::clone(&dropped),
        });
        let capture = Capture::spawn(vec![source]).unwrap();

        drop(capture);

        assert!(
            dropped.load(Ordering::SeqCst),
            "the capture thread must exit (dropping its FrameSource) when Capture is dropped \
             without calling stop_and_drain"
        );
    }
}
