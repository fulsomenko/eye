use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

use eye_core::CameraId;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LatencySummary {
    pub count: usize,
    pub window: usize,
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

#[derive(Debug)]
pub(crate) struct LatencyWindow {
    samples: VecDeque<(Instant, Duration)>,
    horizon: Duration,
    count: usize,
}

impl LatencyWindow {
    pub(crate) fn new(horizon: Duration) -> Self {
        Self {
            samples: VecDeque::new(),
            horizon,
            count: 0,
        }
    }

    pub(crate) fn record(&mut self, now: Instant, d: Duration) {
        self.samples.push_back((now, d));
        self.count += 1;
        self.prune(now);
    }

    fn prune(&mut self, now: Instant) {
        while let Some(&(t, _)) = self.samples.front() {
            if now.saturating_duration_since(t) > self.horizon {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    pub(crate) fn summary(&self, now: Instant) -> LatencySummary {
        let mut sorted: Vec<Duration> = self
            .samples
            .iter()
            .filter(|&&(t, _)| now.saturating_duration_since(t) <= self.horizon)
            .map(|&(_, d)| d)
            .collect();
        sorted.sort_unstable();
        let Some(&max) = sorted.last() else {
            return LatencySummary {
                count: self.count,
                ..Default::default()
            };
        };
        LatencySummary {
            count: self.count,
            window: sorted.len(),
            p50: percentile(&sorted, 50.0),
            p95: percentile(&sorted, 95.0),
            max,
        }
    }
}

impl Default for LatencyWindow {
    fn default() -> Self {
        Self::new(Duration::from_secs(5))
    }
}

/// Hyndman and Fan type 7 on sorted durations, interpolating in nanoseconds.
fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = (n - 1) as f64 * (p / 100.0) + 1.0;
    let floor = h.floor() as usize;
    let frac = h - floor as f64;
    let lower = sorted[(floor - 1).min(n - 1)].as_nanos() as f64;
    let upper = sorted[floor.min(n - 1)].as_nanos() as f64;
    let nanos = lower + frac * (upper - lower);
    Duration::from_nanos(nanos.round() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_percentile_type7() {
        let sorted: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        assert_eq!(percentile(&sorted, 50.0), Duration::from_micros(50_500));
        assert_eq!(percentile(&sorted, 95.0), Duration::from_micros(95_050));

        let sorted: Vec<Duration> = (1..=4).map(Duration::from_millis).collect();
        assert_eq!(percentile(&sorted, 95.0), Duration::from_micros(3_850));
    }

    #[test]
    fn test_latency_window_drops_samples_older_than_horizon() {
        let t0 = Instant::now();
        let mut window = LatencyWindow::default();
        for ms in 1..=1000u64 {
            window.record(t0, Duration::from_millis(ms % 10 + 1));
        }
        let t1 = t0 + Duration::from_secs(6);
        for _ in 0..50 {
            window.record(t1, Duration::from_millis(200));
        }
        let summary = window.summary(t1);
        assert_eq!(summary.window, 50);
        assert_eq!(summary.p95, Duration::from_millis(200));
    }

    #[test]
    fn test_latency_window_empty_is_default() {
        assert_eq!(
            LatencyWindow::default().summary(Instant::now()),
            LatencySummary::default()
        );

        let t0 = Instant::now();
        let mut window = LatencyWindow::default();
        window.record(t0, Duration::from_millis(10));

        let summary = window.summary(t0 + Duration::from_secs(4));
        assert_eq!(summary.window, 1);

        let summary = window.summary(t0 + Duration::from_secs(6));
        assert_eq!(summary.window, 0);
    }
}
