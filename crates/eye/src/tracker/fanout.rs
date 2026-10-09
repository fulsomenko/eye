use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use crossbeam_channel::{Receiver, Select, Sender, TrySendError};
use eye_core::log::{field, span};
use eye_core::{CameraId, Frame, FrameSet, GazePoint, GazeSink, Timestamp};

use crate::pipeline::{Pipeline, RayStep};
use crate::tracker::TrackerError;
use crate::tracker::capture::CaptureMsg;
use crate::tracker::stats::{LatencyWindow, TrackerStats};

const POLL: Duration = Duration::from_millis(100);

#[derive(Debug)]
pub(crate) enum Output {
    Point(GazePoint),
    End,
}

#[derive(Debug)]
pub(crate) struct Shared {
    pub(crate) stop: Arc<AtomicBool>,
    pub(crate) failed_tx: Sender<()>,
    pub(crate) dropped: Vec<(CameraId, Arc<AtomicU64>)>,
    pub(crate) stats: Mutex<StatsInner>,
}

#[derive(Debug, Default)]
pub(crate) struct StatsInner {
    pub(crate) counts: TrackerStats,
    pub(crate) capture_to_emit: LatencyWindow,
    pub(crate) processing: LatencyWindow,
}

impl Shared {
    pub(crate) fn with_stats<R>(&self, f: impl FnOnce(&mut StatsInner) -> R) -> R {
        let mut guard = self.stats.lock().unwrap_or_else(PoisonError::into_inner);
        f(&mut guard)
    }

    pub(crate) fn snapshot(&self) -> TrackerStats {
        let mut stats = self.with_stats(|s| {
            let mut counts = s.counts.clone();
            counts.capture_to_emit = s.capture_to_emit.summary();
            counts.processing = s.processing.summary();
            counts
        });
        for (camera, dropped) in &self.dropped {
            stats
                .frames_dropped
                .insert(camera.clone(), dropped.load(Ordering::Relaxed));
        }
        stats
    }
}

#[derive(Debug)]
pub(crate) struct OutputSender {
    pub(crate) tx: Sender<Output>,
    pub(crate) evict: Receiver<Output>,
}

impl OutputSender {
    pub(crate) fn send(&self, msg: Output) {
        let msg = match self.tx.try_send(msg) {
            Ok(()) | Err(TrySendError::Disconnected(_)) => return,
            Err(TrySendError::Full(msg)) => msg,
        };
        let _ = self.evict.try_recv();
        let _ = self.tx.try_send(msg);
    }
}

pub(crate) struct PipelineThread {
    pub(crate) pipeline: Pipeline,
    pub(crate) captures: Vec<(CameraId, Receiver<CaptureMsg>)>,
    pub(crate) shared: Arc<Shared>,
    pub(crate) out: OutputSender,
    pub(crate) sinks: Vec<Box<dyn GazeSink>>,
    pub(crate) new_subscribers: Receiver<Sender<GazePoint>>,
    pub(crate) subscribers: Vec<Sender<GazePoint>>,
}

impl PipelineThread {
    pub(crate) fn run(mut self) -> Result<(), TrackerError> {
        let result = self.run_loop();
        if let Err(e) = &result {
            tracing::error!(error = %e, "pipeline thread stopped on error");
            let _ = self.shared.failed_tx.try_send(());
        }
        self.shared.stop.store(true, Ordering::Release);
        result
    }

    fn run_loop(&mut self) -> Result<(), TrackerError> {
        while !self.shared.stop.load(Ordering::Acquire) {
            if self.captures.is_empty() {
                while let Some(set) = self.pipeline.flush() {
                    self.process(&set)?;
                }
                let framesets = self.shared.with_stats(|s| s.counts.framesets);
                tracing::info!(framesets, "all captures ended; pipeline stopping");
                self.out.send(Output::End);
                return Ok(());
            }
            let selected = {
                let mut select = Select::new();
                for (_, rx) in &self.captures {
                    select.recv(rx);
                }
                let selected = match select.select_timeout(POLL) {
                    Ok(op) => {
                        let index = op.index();
                        Some((index, op.recv(&self.captures[index].1)))
                    }
                    Err(_) => None,
                };
                #[allow(clippy::let_and_return)]
                selected
            };
            let Some((index, msg)) = selected else {
                if let Some(set) = self.pipeline.flush() {
                    self.process(&set)?;
                }
                continue;
            };
            match msg {
                Ok(CaptureMsg::Frame(frame)) => {
                    if let Some(set) = self.pipeline.pair(frame) {
                        self.process(&set)?;
                    }
                }
                Ok(CaptureMsg::Failed(source)) => {
                    return Err(TrackerError::Capture {
                        camera: self.captures[index].0.to_string(),
                        source,
                    });
                }
                Err(_) => {
                    let (camera, _) = self.captures.remove(index);
                    tracing::debug!(
                        { field::CAMERA } = camera.as_str(),
                        "capture channel closed"
                    );
                }
            }
        }
        Ok(())
    }

    fn process(&mut self, set: &FrameSet) -> Result<(), TrackerError> {
        let primary = set
            .frames()
            .iter()
            .map(Frame::header)
            .max_by_key(|h| h.timestamp)
            .expect("a FrameSet is non-empty");
        let _frame = eye_core::log::frame_span(
            primary.camera.as_str(),
            primary.seq,
            primary.timestamp.as_nanos(),
            primary.illumination.as_str(),
            set.frames().len() as u64,
        )
        .entered();
        let spread_us = u64::try_from(set.spread().as_micros()).unwrap_or(u64::MAX);
        match set.frames() {
            [_, secondary] => tracing::trace!(
                spread_us,
                secondary_camera = secondary.header().camera.as_str(),
                secondary_seq = secondary.header().seq,
                "frame set"
            ),
            _ => tracing::trace!(spread_us, "frame set"),
        }

        let paired = Timestamp::now();
        self.shared.with_stats(|s| s.counts.framesets += 1);
        let batch = match self.pipeline.rays(set)? {
            RayStep::Rays(batch) => batch,
            RayStep::NoGaze => {
                self.shared.with_stats(|s| s.counts.no_gaze += 1);
                return Ok(());
            }
            RayStep::Skipped(_) => {
                self.shared.with_stats(|s| s.counts.stage_errors += 1);
                return Ok(());
            }
        };
        let Some(point) = self.pipeline.finish(&batch) else {
            self.shared.with_stats(|s| s.counts.no_gaze += 1);
            return Ok(());
        };
        self.fan_out(&point, paired);
        self.out.send(Output::Point(point));
        Ok(())
    }

    fn fan_out(&mut self, point: &GazePoint, paired: Timestamp) {
        let _stage = tracing::debug_span!(
            span::STAGE,
            { field::STAGE_KIND } = "output",
            { field::STAGE_NAME } = "fanout",
        )
        .entered();

        for tx in self.new_subscribers.try_iter().collect::<Vec<_>>() {
            self.subscribers.push(tx);
            tracing::debug!(subscribers = self.subscribers.len(), "subscriber added");
        }

        let mut subscriber_drops = 0u64;
        let mut remaining = self.subscribers.len() as u64;
        self.subscribers
            .retain_mut(|tx| match tx.try_send(point.clone()) {
                Ok(()) => true,
                Err(TrySendError::Full(_)) => {
                    subscriber_drops += 1;
                    tracing::debug!("point dropped for slow subscriber");
                    true
                }
                Err(TrySendError::Disconnected(_)) => {
                    remaining -= 1;
                    tracing::debug!(subscribers = remaining, "subscriber disconnected");
                    false
                }
            });

        self.sinks.retain_mut(|sink| match sink.push(point) {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!(sink = sink.name(), error = %e, "sink failed, removing it");
                false
            }
        });

        let now = Timestamp::now();
        let sinks = self.sinks.len();
        let subscribers = self.subscribers.len();
        let capture_to_emit = now.0.saturating_sub(point.timestamp.0);
        let processing = now.0.saturating_sub(paired.0);
        self.shared.with_stats(|s| {
            s.counts.points_emitted += 1;
            s.counts.subscriber_drops += subscriber_drops;
            s.counts.sinks = sinks;
            s.counts.subscribers = subscribers;
            s.capture_to_emit.record(capture_to_emit);
            s.processing.record(processing);
        });
        tracing::trace!(
            mm_x = point.mm.x,
            mm_y = point.mm.y,
            px_logical_x = point.px_logical.x,
            px_logical_y = point.px_logical.y,
            confidence = point.confidence,
            capture_to_emit_us = u64::try_from(capture_to_emit.as_micros()).unwrap_or(u64::MAX),
            processing_us = u64::try_from(processing.as_micros()).unwrap_or(u64::MAX),
            sinks,
            subscribers,
            "gaze point emitted"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use eye_capture::CaptureError;
    use eye_capture::pairing::Pairer;
    use eye_core::{Illumination, PixelFormat};

    use super::*;
    use crate::registry::PassThroughFilter;
    use crate::testkit::{self, FakeDetector, FakeEstimator, FakeSink};

    fn thread(sinks: Vec<Box<dyn GazeSink>>) -> (PipelineThread, Receiver<Output>) {
        let pipeline = Pipeline::new(
            testkit::rig(),
            Pairer::new(&[testkit::info("ir", PixelFormat::Gray8, 66)])
                .expect("single-camera pairer"),
            vec![(CameraId::from("ir"), Box::new(FakeDetector::new("fake")))],
            Box::new(FakeEstimator),
            Box::new(PassThroughFilter),
            None,
            3,
        );
        let (tx, rx) = crossbeam_channel::bounded(8);
        let (failed_tx, _failed_rx) = crossbeam_channel::bounded(1);
        let shared = Arc::new(Shared {
            stop: Arc::new(AtomicBool::new(false)),
            failed_tx,
            dropped: vec![],
            stats: Mutex::default(),
        });
        let out = OutputSender {
            tx,
            evict: rx.clone(),
        };
        let (_new_subs_tx, new_subscribers) = crossbeam_channel::unbounded();
        let thread = PipelineThread {
            pipeline,
            captures: vec![],
            shared,
            out,
            sinks,
            new_subscribers,
            subscribers: vec![],
        };
        (thread, rx)
    }

    #[test]
    fn test_logs_frame_set_at_trace_with_secondary() {
        let (mut thread, _rx) = thread(vec![]);
        let set = FrameSet::new(vec![
            testkit::frame("ir", 1, 200, Illumination::IrLit),
            testkit::frame_fmt("rgb", 0, 100, PixelFormat::Mjpeg, Illumination::Ambient),
        ])
        .expect("distinct cameras");
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let _ = thread.process(&set);
        });
        let rec = records
            .iter()
            .find(|r| r.message == "frame set")
            .expect("trace record present");
        assert_eq!(rec.level, eye_log::Level::Trace);
        assert_eq!(
            rec.fields.get("secondary_camera"),
            Some(&eye_log::Value::Str("rgb".to_string()))
        );
        assert_eq!(
            rec.context.get(field::SET_CAMERAS),
            Some(&eye_log::Value::U64(2))
        );
        assert_eq!(
            rec.context.get(field::CAMERA),
            Some(&eye_log::Value::Str("ir".to_string()))
        );
        assert!(!rec.fields.contains_key("set.cameras"));
    }

    #[test]
    fn test_logs_gaze_point_emitted_at_trace_inside_output_stage() {
        let (mut thread, _rx) = thread(vec![]);
        let set = FrameSet::single(testkit::frame("ir", 0, 100, Illumination::IrLit));
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let _ = thread.process(&set);
        });
        let rec = records
            .iter()
            .find(|r| r.message == "gaze point emitted")
            .expect("trace record present");
        assert_eq!(rec.level, eye_log::Level::Trace);
        assert_eq!(
            rec.context.get(field::STAGE_KIND),
            Some(&eye_log::Value::Str("output".to_string()))
        );
        assert_eq!(
            rec.context.get(field::STAGE_NAME),
            Some(&eye_log::Value::Str("fanout".to_string()))
        );
        assert!(matches!(
            rec.fields.get("confidence"),
            Some(eye_log::Value::F64(_))
        ));
        assert!(matches!(
            rec.fields.get("capture_to_emit_us"),
            Some(eye_log::Value::U64(_))
        ));
    }

    #[test]
    fn test_logs_sink_failed_at_warn() {
        let pushes = Arc::new(AtomicUsize::new(0));
        let (mut thread, _rx) = thread(vec![Box::new(FakeSink { pushes, fail_on: 1 })]);
        let set = FrameSet::single(testkit::frame("ir", 0, 100, Illumination::IrLit));
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::WARN, || {
            let _ = thread.process(&set);
        });
        let rec = records
            .iter()
            .find(|r| r.message == "sink failed, removing it")
            .expect("warn record present");
        assert_eq!(rec.level, eye_log::Level::Warn);
        assert_eq!(
            rec.fields.get("sink"),
            Some(&eye_log::Value::Str("fake".to_string()))
        );
        assert!(records.iter().all(|r| r.level != eye_log::Level::Error));
    }

    #[test]
    fn test_logs_point_dropped_for_slow_subscriber_at_debug() {
        let (mut thread, _rx) = thread(vec![]);
        let (tx, _rx_sub) = crossbeam_channel::bounded(1);
        thread.subscribers.push(tx);
        let point = testkit::point(0.0, 0.0, nalgebra::Matrix2::identity());
        let paired = Timestamp::now();
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::DEBUG, || {
            thread.fan_out(&point, paired);
            thread.fan_out(&point, paired);
        });
        let dropped: Vec<_> = records
            .iter()
            .filter(|r| r.message == "point dropped for slow subscriber")
            .collect();
        assert_eq!(dropped.len(), 1);
        assert_eq!(dropped[0].level, eye_log::Level::Debug);
    }

    #[test]
    fn test_logs_pipeline_thread_stopped_at_error() {
        let (mut thread, _rx) = thread(vec![]);
        let (tx, rx) = crossbeam_channel::unbounded();
        tx.send(CaptureMsg::Failed(CaptureError::Disconnected {
            camera: "ir".to_string(),
        }))
        .expect("receiver is alive");
        thread.captures.push((CameraId::from("ir"), rx));
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::ERROR, || thread.run());
        let errors: Vec<_> = records
            .iter()
            .filter(|r| r.level == eye_log::Level::Error)
            .collect();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].message, "pipeline thread stopped on error");
    }

    #[test]
    fn test_logs_all_captures_ended_at_info() {
        let (thread, _rx) = thread(vec![]);
        let (result, records) =
            eye_log::testing::capture_logs(tracing::Level::INFO, || thread.run());
        assert!(result.is_ok());
        let rec = records
            .iter()
            .find(|r| r.message == "all captures ended; pipeline stopping")
            .expect("info record present");
        assert_eq!(rec.level, eye_log::Level::Info);
        assert_eq!(rec.fields.get("framesets"), Some(&eye_log::Value::U64(0)));
    }
}
