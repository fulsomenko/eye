mod capture;
mod fanout;
mod sources;
mod stats;

use std::fmt;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use eye_capture::{CaptureError, FrameSource};
use eye_core::{CameraId, GazePoint, GazeSink, Rig, stage::GazeCorrection};

use crate::config::Config;
use crate::pipeline::{Pipeline, PipelineError};
use crate::registry::Registry;

use capture::{CaptureThread, spawn_capture};
use fanout::{Output, OutputSender, PipelineThread, Shared, StatsInner};

pub use sources::open_sources;
pub use stats::{LatencySummary, TrackerStats};

#[derive(Debug, thiserror::Error)]
pub enum TrackerError {
    #[error(transparent)]
    Pipeline(#[from] PipelineError),
    #[error("camera {camera}: {source}")]
    Capture {
        camera: String,
        #[source]
        source: CaptureError,
    },
    #[error("all sources reached end of stream")]
    EndOfStream,
    #[error("tracker stopped")]
    Stopped,
    #[error("{0} thread panicked")]
    Panicked(&'static str),
    #[error("spawning a thread: {0}")]
    Spawn(#[source] std::io::Error),
}

pub struct TrackerOptions {
    /// Per-camera capture channel (`config.capture.channel_capacity`); full evicts the oldest frame.
    pub capture_capacity: usize,
    /// `next()` output and each subscriber channel.
    pub output_capacity: usize,
    /// Called on the pipeline thread for every point; must not block.
    pub sinks: Vec<Box<dyn GazeSink>>,
}

impl fmt::Debug for TrackerOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrackerOptions")
            .field("capture_capacity", &self.capture_capacity)
            .field("output_capacity", &self.output_capacity)
            .field(
                "sinks",
                &self.sinks.iter().map(|s| s.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

pub struct Tracker {
    output: Option<Receiver<Output>>,
    failed: Receiver<()>,
    new_subscribers: Sender<Sender<GazePoint>>,
    output_capacity: usize,
    shared: Arc<Shared>,
    captures: Vec<JoinHandle<()>>,
    pipeline: Option<JoinHandle<Result<(), TrackerError>>>,
    finished: bool,
}

impl fmt::Debug for Tracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tracker")
            .field(
                "cameras",
                &self
                    .shared
                    .dropped
                    .iter()
                    .map(|(id, _)| id.clone())
                    .collect::<Vec<CameraId>>(),
            )
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

impl Tracker {
    /// `from_config_with(&Registry::with_defaults(), config, rig, correction, Vec::new())`.
    pub fn from_config(
        config: &Config,
        rig: Rig,
        correction: Option<Box<dyn GazeCorrection>>,
    ) -> Result<Tracker, TrackerError> {
        Self::from_config_with(
            &Registry::with_defaults(),
            config,
            rig,
            correction,
            Vec::new(),
        )
    }

    /// Builds the Pipeline from `CameraConfig::to_info` FIRST (stage errors surface before any
    /// device opens), then `open_sources`, then `start` with capture capacity
    /// `config.capture.channel_capacity`, output capacity 2.
    pub fn from_config_with(
        registry: &Registry,
        config: &Config,
        rig: Rig,
        correction: Option<Box<dyn GazeCorrection>>,
        sinks: Vec<Box<dyn GazeSink>>,
    ) -> Result<Tracker, TrackerError> {
        let cameras: Vec<_> = config.cameras.iter().map(|c| c.to_info()).collect();
        let pipeline = Pipeline::from_config(registry, config, rig, &cameras, correction)?;
        let sources = open_sources(config)?;
        let options = TrackerOptions {
            capture_capacity: config.capture.channel_capacity,
            output_capacity: 2,
            sinks,
        };
        Self::start(pipeline, sources, options)
    }

    /// One capture thread per source (`eye-cap-<id>`) + one pipeline thread (`eye-pipeline`).
    pub fn start(
        pipeline: Pipeline,
        sources: Vec<Box<dyn FrameSource>>,
        options: TrackerOptions,
    ) -> Result<Tracker, TrackerError> {
        let output_capacity = options.output_capacity.max(1);
        let stop = Arc::new(AtomicBool::new(false));
        let mut threads: Vec<CaptureThread> = Vec::with_capacity(sources.len());
        for source in sources {
            match spawn_capture(source, options.capture_capacity, Arc::clone(&stop)) {
                Ok(t) => threads.push(t),
                Err(e) => {
                    stop.store(true, Ordering::Release);
                    for t in threads {
                        let _ = t.handle.join();
                    }
                    return Err(e);
                }
            }
        }
        let (failed_tx, failed) = crossbeam_channel::bounded(1);
        let shared = Arc::new(Shared {
            stop,
            failed_tx,
            dropped: threads
                .iter()
                .map(|t| (t.camera.clone(), Arc::clone(&t.dropped)))
                .collect(),
            stats: Mutex::new(StatsInner::default()),
        });
        let (out_tx, output) = crossbeam_channel::bounded(output_capacity);
        let (new_subscribers, subscriber_rx) = crossbeam_channel::unbounded();
        let (mut captures, mut receivers) = (Vec::new(), Vec::new());
        for t in threads {
            receivers.push((t.camera, t.rx));
            captures.push(t.handle);
        }
        let worker = PipelineThread {
            pipeline,
            captures: receivers,
            shared: Arc::clone(&shared),
            out: OutputSender {
                tx: out_tx,
                evict: output.clone(),
            },
            sinks: options.sinks,
            new_subscribers: subscriber_rx,
            subscribers: Vec::new(),
        };
        let mut tracker = Tracker {
            output: Some(output),
            failed,
            new_subscribers,
            output_capacity,
            shared,
            captures,
            pipeline: None,
            finished: false,
        };
        let cameras = worker.captures.len();
        let sinks = worker.sinks.len();
        let parent = tracing::Span::current();
        let handle = std::thread::Builder::new()
            .name("eye-pipeline".into())
            .spawn(move || {
                let _parent = parent.entered();
                worker.run()
            })
            .map_err(TrackerError::Spawn)?;
        tracker.pipeline = Some(handle);
        tracing::info!(
            cameras,
            capture_capacity = options.capture_capacity,
            output_capacity,
            sinks,
            "tracker started"
        );
        Ok(tracker)
    }

    /// Blocks for the next point. A pipeline failure takes priority over queued points; after any
    /// `Err` (including `EndOfStream`) every later call returns `Err(Stopped)`.
    #[expect(
        clippy::should_implement_trait,
        reason = "root API name; an Iterator would hide the blocking Result"
    )]
    pub fn next(&mut self) -> Result<GazePoint, TrackerError> {
        if self.finished {
            return Err(TrackerError::Stopped);
        }
        let Some(output) = self.output.as_ref() else {
            return Err(TrackerError::Stopped);
        };
        let outcome = crossbeam_channel::select_biased! {
            recv(self.failed) -> _ => None,
            recv(output) -> msg => match msg {
                Ok(Output::Point(p)) => return Ok(p),
                Ok(Output::End) => Some(TrackerError::EndOfStream),
                Err(_) => None,
            },
        };
        Err(self.finish_with(outcome))
    }

    /// `Ok(None)` when no point arrives within `timeout`; otherwise like `next`.
    pub fn next_timeout(&mut self, timeout: Duration) -> Result<Option<GazePoint>, TrackerError> {
        if self.finished {
            return Err(TrackerError::Stopped);
        }
        let Some(output) = self.output.as_ref() else {
            return Err(TrackerError::Stopped);
        };
        let outcome = crossbeam_channel::select_biased! {
            recv(self.failed) -> _ => None,
            recv(output) -> msg => match msg {
                Ok(Output::Point(p)) => return Ok(Some(p)),
                Ok(Output::End) => Some(TrackerError::EndOfStream),
                Err(_) => None,
            },
            default(timeout) => return Ok(None),
        };
        Err(self.finish_with(outcome))
    }

    fn finish_with(&mut self, outcome: Option<TrackerError>) -> TrackerError {
        self.finished = true;
        outcome.unwrap_or_else(|| self.join_pipeline().err().unwrap_or(TrackerError::Stopped))
    }

    /// Own `bounded(output_capacity)` channel; full drops the NEW point for this subscriber.
    /// Disconnects when the pipeline thread ends (error, end of stream, shutdown).
    pub fn subscribe(&self) -> Receiver<GazePoint> {
        let (tx, rx) = crossbeam_channel::bounded(self.output_capacity);
        let _ = self.new_subscribers.send(tx);
        rx
    }

    pub fn stats(&self) -> TrackerStats {
        self.shared.snapshot()
    }

    fn join_pipeline(&mut self) -> Result<(), TrackerError> {
        match self.pipeline.take() {
            Some(handle) => handle
                .join()
                .map_err(|_| TrackerError::Panicked("pipeline"))?,
            None => Ok(()),
        }
    }

    fn stop_and_join(&mut self) -> Result<(), TrackerError> {
        self.shared.stop.store(true, Ordering::Release);
        self.output = None;
        let should_log = self.pipeline.is_some();
        let result = self.join_pipeline();
        let mut capture_panicked = false;
        for handle in self.captures.drain(..) {
            capture_panicked |= handle.join().is_err();
        }
        if should_log {
            let stats = self.shared.snapshot();
            tracing::info!(
                framesets = stats.framesets,
                points_emitted = stats.points_emitted,
                no_gaze = stats.no_gaze,
                stage_errors = stats.stage_errors,
                frames_dropped = stats.frames_dropped.values().sum::<u64>(),
                subscriber_drops = stats.subscriber_drops,
                "tracker stopped"
            );
        }
        match result {
            Ok(()) if capture_panicked => Err(TrackerError::Panicked("capture")),
            other => other,
        }
    }

    /// Stops and joins every thread; returns the pipeline's error unless `next` already returned
    /// it.
    pub fn shutdown(mut self) -> Result<(), TrackerError> {
        self.stop_and_join()
    }
}

impl Drop for Tracker {
    fn drop(&mut self) {
        let _ = self.stop_and_join();
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    use eye_capture::pairing::Pairer;
    use eye_core::stage::Detector;
    use eye_core::{Illumination, PixelFormat, Timestamp};

    use super::*;
    use crate::registry::PassThroughFilter;
    use crate::testkit::{
        self, FakeDetector, FakeEstimator, FakeSink, ScriptSource, script_source,
    };

    fn start(
        source: ScriptSource,
        capacity: usize,
        sinks: Vec<Box<dyn GazeSink>>,
        detector: Box<dyn Detector>,
    ) -> Tracker {
        let pipeline = Pipeline::new(
            testkit::rig(),
            Pairer::new(&[testkit::info("ir", PixelFormat::Gray8, 66)])
                .expect("single-camera pairer"),
            vec![(CameraId::from("ir"), detector)],
            Box::new(FakeEstimator),
            Box::new(PassThroughFilter),
            None,
            3,
        );
        Tracker::start(
            pipeline,
            vec![Box::new(source) as Box<dyn FrameSource>],
            TrackerOptions {
                capture_capacity: capacity,
                output_capacity: capacity,
                sinks,
            },
        )
        .expect("tracker starts")
    }

    #[test]
    fn test_tracker_emits_points_then_end_of_stream() {
        let frames = (0..20u64)
            .map(|seq| Ok(testkit::frame("ir", seq, seq, Illumination::IrLit)))
            .collect();
        let mut t = start(
            script_source(frames),
            64,
            Vec::new(),
            Box::new(FakeDetector::new("fake")),
        );
        for seq in 0..20u64 {
            let point = t.next().expect("point");
            assert_eq!(point.timestamp, Timestamp::from_nanos(seq * 1_000_000));
        }
        assert!(matches!(t.next(), Err(TrackerError::EndOfStream)));
        assert!(matches!(t.next(), Err(TrackerError::Stopped)));
    }

    #[test]
    fn test_tracker_capture_drop_oldest_by_sequence() {
        let frames = (0..500u64)
            .map(|seq| Ok(testkit::frame("ir", seq, seq, Illumination::IrLit)))
            .collect();
        let mut t = start(
            script_source(frames),
            2,
            Vec::new(),
            Box::new(FakeDetector::new("fake")),
        );
        let done = t.subscribe();
        while done.recv().is_ok() {}

        let point = t.next().expect("the newest frame survives");
        assert_eq!(point.timestamp, Timestamp::from_nanos(499_000_000));
        assert!(matches!(t.next(), Err(TrackerError::EndOfStream)));

        let stats = t.stats();
        let dropped = stats
            .frames_dropped
            .get(&CameraId::from("ir"))
            .copied()
            .unwrap_or(0);
        assert_eq!(stats.framesets + dropped, 500);
    }

    #[test]
    fn test_tracker_timeout_is_retried() {
        let mut frames: Vec<Result<eye_core::Frame, CaptureError>> = Vec::new();
        for seq in 0..10u64 {
            frames.push(Err(CaptureError::Timeout {
                camera: "ir".to_string(),
                timeout: Duration::from_millis(1),
            }));
            frames.push(Ok(testkit::frame("ir", seq, seq, Illumination::IrLit)));
        }
        let mut t = start(
            script_source(frames),
            32,
            Vec::new(),
            Box::new(FakeDetector::new("fake")),
        );
        for seq in 0..10u64 {
            let point = t.next().expect("point");
            assert_eq!(point.timestamp, Timestamp::from_nanos(seq * 1_000_000));
        }
        assert!(matches!(t.next(), Err(TrackerError::EndOfStream)));
    }

    #[test]
    fn test_tracker_capture_error_is_fatal_via_next() {
        let frames = vec![
            Ok(testkit::frame("ir", 0, 0, Illumination::IrLit)),
            Ok(testkit::frame("ir", 1, 1, Illumination::IrLit)),
            Ok(testkit::frame("ir", 2, 2, Illumination::IrLit)),
            Err(CaptureError::Disconnected {
                camera: "ir".to_string(),
            }),
        ];
        let mut t = start(
            script_source(frames),
            8,
            Vec::new(),
            Box::new(FakeDetector::new("fake")),
        );
        let mut points = 0;
        loop {
            match t.next() {
                Ok(_) => points += 1,
                Err(TrackerError::Capture { camera, source }) => {
                    assert_eq!(camera, "ir");
                    assert!(matches!(source, CaptureError::Disconnected { .. }));
                    break;
                }
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }
        assert!(points <= 3);
        assert!(matches!(t.next(), Err(TrackerError::Stopped)));
    }

    #[test]
    fn test_tracker_stage_errors_fatal_after_threshold() {
        let frames = (0..10u64)
            .map(|seq| Ok(testkit::frame("ir", seq, seq, Illumination::IrLit)))
            .collect();
        let mut t = start(
            script_source(frames),
            32,
            Vec::new(),
            Box::new(FakeDetector::new("fake").failing_always()),
        );
        let Err(TrackerError::Pipeline(PipelineError::Stage { count, .. })) = t.next() else {
            panic!("expected a fatal stage error")
        };
        assert_eq!(count, 4);
        assert_eq!(t.stats().stage_errors, 3);
    }

    #[test]
    fn test_tracker_sink_error_removes_sink_and_continues() {
        let frames = (0..5u64)
            .map(|seq| Ok(testkit::frame("ir", seq, seq, Illumination::IrLit)))
            .collect();
        let pushes = Arc::new(AtomicUsize::new(0));
        let sink = FakeSink {
            pushes: pushes.clone(),
            fail_on: 1,
        };
        let mut t = start(
            script_source(frames),
            32,
            vec![Box::new(sink)],
            Box::new(FakeDetector::new("fake")),
        );
        for _ in 0..5 {
            t.next().expect("point");
        }
        assert_eq!(pushes.load(Ordering::SeqCst), 1);
        assert_eq!(t.stats().sinks, 0);
    }

    #[test]
    fn test_tracker_subscribe_receives_points() {
        let frames = (0..20u64)
            .map(|seq| Ok(testkit::frame("ir", seq, seq, Illumination::IrLit)))
            .collect();
        let (gate_tx, gate_rx) = crossbeam_channel::bounded(1);
        let mut source = script_source(frames);
        source.gate = Some(gate_rx);
        let mut t = start(source, 32, Vec::new(), Box::new(FakeDetector::new("fake")));
        let rx = t.subscribe();
        gate_tx.send(()).expect("gate receiver is alive");
        for _ in 0..20 {
            t.next().expect("point");
        }
        assert_eq!(rx.try_iter().count(), 20);
    }

    #[test]
    fn test_tracker_dropped_subscriber_is_pruned() {
        let frames = (0..5u64)
            .map(|seq| Ok(testkit::frame("ir", seq, seq, Illumination::IrLit)))
            .collect();
        let (gate_tx, gate_rx) = crossbeam_channel::bounded(1);
        let mut source = script_source(frames);
        source.gate = Some(gate_rx);
        let mut t = start(source, 32, Vec::new(), Box::new(FakeDetector::new("fake")));
        let rx = t.subscribe();
        drop(rx);
        gate_tx.send(()).expect("gate receiver is alive");
        for _ in 0..5 {
            t.next().expect("point");
        }
        assert_eq!(t.stats().subscribers, 0);
    }

    #[test]
    fn test_tracker_shutdown_returns_pipeline_error_when_unobserved() {
        let frames = vec![
            Ok(testkit::frame("ir", 0, 0, Illumination::IrLit)),
            Err(CaptureError::Disconnected {
                camera: "ir".to_string(),
            }),
        ];
        let t = start(
            script_source(frames),
            8,
            Vec::new(),
            Box::new(FakeDetector::new("fake")),
        );
        let rx = t.subscribe();
        while rx.recv().is_ok() {}
        let Err(TrackerError::Capture { camera, .. }) = t.shutdown() else {
            panic!("expected a fatal Capture error")
        };
        assert_eq!(camera, "ir");
    }

    #[test]
    fn test_next_timeout_returns_none_when_idle() {
        let (gate_tx, gate_rx) = crossbeam_channel::bounded(1);
        let mut source = script_source(Vec::new());
        source.gate = Some(gate_rx);
        let mut t = start(source, 8, Vec::new(), Box::new(FakeDetector::new("fake")));
        assert!(matches!(
            t.next_timeout(Duration::from_millis(10)),
            Ok(None)
        ));
        drop(gate_tx);
    }

    #[test]
    fn test_tracker_drop_joins_capture_threads() {
        let dropped = Arc::new(AtomicBool::new(false));
        let source = ScriptSource {
            info: testkit::info("ir", PixelFormat::Gray8, 66),
            script: VecDeque::new(),
            endless: Some((0, Duration::from_millis(1))),
            gate: None,
            dropped: dropped.clone(),
        };
        let t = start(source, 2, Vec::new(), Box::new(FakeDetector::new("fake")));
        drop(t);
        assert!(dropped.load(Ordering::SeqCst));
    }

    #[test]
    fn test_logs_tracker_started_and_stopped_at_info_once() {
        let frames = (0..3u64)
            .map(|seq| Ok(testkit::frame("ir", seq, seq, Illumination::IrLit)))
            .collect();
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::INFO, || {
            let t = start(
                script_source(frames),
                8,
                Vec::new(),
                Box::new(FakeDetector::new("fake")),
            );
            t.shutdown()
        });
        let started: Vec<_> = records
            .iter()
            .filter(|r| r.message == "tracker started")
            .collect();
        assert_eq!(started.len(), 1);
        let stopped: Vec<_> = records
            .iter()
            .filter(|r| r.message == "tracker stopped")
            .collect();
        assert_eq!(stopped.len(), 1);
        assert!(matches!(
            stopped[0].fields.get("frames_dropped"),
            Some(eye_log::Value::U64(_))
        ));
    }

    #[test]
    #[ignore = "needs hardware"]
    fn test_live_tracker_reports_latency() {
        use crate::config::{Config, EnvOverrides};
        use crate::testkit::fake_registry;

        let toml_str = r#"
            estimate = "fake"
            [[camera]]
            id = "ir"
            device = "/dev/video2"
            format = "gray"
            size = [640, 360]
            [detect]
            ir = "fake"
        "#;
        let mut config = Config::from_toml_str(toml_str).expect("parses");
        config.apply_env(&EnvOverrides::from_process_env());
        config.validate().expect("valid");

        let mut tracker =
            Tracker::from_config_with(&fake_registry(), &config, testkit::rig(), None, Vec::new())
                .expect("the ir camera opens");

        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while std::time::Instant::now() < deadline {
            let _ = tracker.next_timeout(Duration::from_millis(100));
        }

        let stats = tracker.stats();
        assert!(stats.capture_to_emit.count > 0);
        assert!(stats.framesets > 0);
        println!(
            "capture_to_emit: p50={:?} p95={:?} max={:?}",
            stats.capture_to_emit.p50, stats.capture_to_emit.p95, stats.capture_to_emit.max
        );
        println!("frames_dropped: {:?}", stats.frames_dropped);
    }
}
