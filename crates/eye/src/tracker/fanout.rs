use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use crossbeam_channel::{Receiver, Select, Sender, TrySendError};
use eye_core::{CameraId, FrameSet, GazePoint, GazeSink, Timestamp};

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
        if result.is_err() {
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
                    self.captures.remove(index);
                }
            }
        }
        Ok(())
    }

    fn process(&mut self, set: &FrameSet) -> Result<(), TrackerError> {
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
        self.subscribers.extend(self.new_subscribers.try_iter());

        let mut subscriber_drops = 0u64;
        self.subscribers
            .retain_mut(|tx| match tx.try_send(point.clone()) {
                Ok(()) => true,
                Err(TrySendError::Full(_)) => {
                    subscriber_drops += 1;
                    true
                }
                Err(TrySendError::Disconnected(_)) => false,
            });

        self.sinks.retain_mut(|sink| match sink.push(point) {
            Ok(()) => true,
            Err(e) => {
                tracing::error!(sink = sink.name(), error = %e, "sink failed, removing it");
                false
            }
        });

        let now = Timestamp::now();
        let sinks = self.sinks.len();
        let subscribers = self.subscribers.len();
        self.shared.with_stats(|s| {
            s.counts.points_emitted += 1;
            s.counts.subscriber_drops += subscriber_drops;
            s.counts.sinks = sinks;
            s.counts.subscribers = subscribers;
            s.capture_to_emit
                .record(now.0.saturating_sub(point.timestamp.0));
            s.processing.record(now.0.saturating_sub(paired.0));
        });
    }
}
