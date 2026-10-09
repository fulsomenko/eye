use std::collections::{BTreeMap, VecDeque};
use std::time::Duration;

use eye_core::CameraId;

const WINDOW: usize = 1024;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LatencySummary {
    pub count: usize,
    pub p50: Duration,
    pub p95: Duration,
    pub max: Duration,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrackerStats {
    pub framesets: u64,
    pub points_emitted: u64,
    pub no_gaze: u64,
    pub stage_errors: u64,
    /// Frames evicted by the capture side's drop-oldest channel, per camera.
    pub frames_dropped: BTreeMap<CameraId, u64>,
    /// Oldest points evicted from full subscriber channels (newest wins).
    pub subscriber_drops: u64,
    pub sinks: usize,
    pub subscribers: usize,
    /// `Timestamp::now() - point.timestamp` at emit: capture of the OLDEST frame whose ray
    /// contributed, to emit (live only).
    pub capture_to_emit: LatencySummary,
    /// `Timestamp::now()` at emit minus the moment `Pipeline::pair` completed the set.
    pub processing: LatencySummary,
    /// Frames delivered to the pipeline thread per camera (after capture-side eviction).
    pub frames_received: BTreeMap<CameraId, u64>,
    /// Per camera, how many received frames carried each `Illumination` (`Illumination::as_str` keys).
    pub illumination: BTreeMap<CameraId, BTreeMap<&'static str, u64>>,
    /// Points each sink reported as discarded (`GazeSink::dropped`), by sink name.
    pub sink_drops: BTreeMap<&'static str, u64>,
}

#[derive(Debug, Default)]
pub(crate) struct LatencyWindow {
    samples: VecDeque<Duration>,
    count: usize,
}

impl LatencyWindow {
    pub(crate) fn record(&mut self, d: Duration) {
        if self.samples.len() == WINDOW {
            self.samples.pop_front();
        }
        self.samples.push_back(d);
        self.count += 1;
    }

    pub(crate) fn summary(&self) -> LatencySummary {
        let mut sorted: Vec<Duration> = self.samples.iter().copied().collect();
        sorted.sort_unstable();
        let Some(&max) = sorted.last() else {
            return LatencySummary::default();
        };
        LatencySummary {
            count: self.count,
            p50: nearest_rank(&sorted, 500),
            p95: nearest_rank(&sorted, 950),
            max,
        }
    }
}

fn nearest_rank(sorted: &[Duration], per_mille: usize) -> Duration {
    let rank = (per_mille * sorted.len()).div_ceil(1000).max(1);
    sorted[rank - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_window_nearest_rank() {
        let mut window = LatencyWindow::default();
        for ms in (1..=100).rev() {
            window.record(Duration::from_millis(ms));
        }
        assert_eq!(
            window.summary(),
            LatencySummary {
                count: 100,
                p50: Duration::from_millis(50),
                p95: Duration::from_millis(95),
                max: Duration::from_millis(100),
            }
        );
    }

    #[test]
    fn test_latency_window_keeps_last_1024() {
        let mut window = LatencyWindow::default();
        for ms in 0..2000u64 {
            window.record(Duration::from_millis(ms));
        }
        let summary = window.summary();
        assert_eq!(summary.count, 2000);
        assert_eq!(summary.max, Duration::from_millis(1999));
        assert_eq!(summary.p50, Duration::from_millis(1487));
    }
}
