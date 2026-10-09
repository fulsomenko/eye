//! Ring of recent `capture_to_present` durations, shared between the overlay thread and
//! whatever reads it (the app's `--stats` output).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const WINDOW: usize = 256;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PresentSummary {
    pub count: u64,
    pub p50: Duration,
    pub p95: Duration,
    pub max: Duration,
}

#[derive(Debug, Default)]
struct Ring {
    samples: VecDeque<Duration>,
    count: u64,
}

impl Ring {
    fn record(&mut self, d: Duration) {
        if self.samples.len() == WINDOW {
            self.samples.pop_front();
        }
        self.samples.push_back(d);
        self.count += 1;
    }

    fn summary(&self) -> PresentSummary {
        let mut sorted: Vec<Duration> = self.samples.iter().copied().collect();
        sorted.sort_unstable();
        let Some(&max) = sorted.last() else {
            return PresentSummary::default();
        };
        PresentSummary {
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

#[derive(Debug, Clone, Default)]
pub struct PresentStats(Arc<Mutex<Ring>>);

impl PresentStats {
    pub fn record(&self, d: Duration) {
        self.0.lock().expect("ring mutex poisoned").record(d);
    }

    pub fn summary(&self) -> PresentSummary {
        self.0.lock().expect("ring mutex poisoned").summary()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_present_stats_summary_nearest_rank() {
        let stats = PresentStats::default();
        for ms in 1..=100u64 {
            stats.record(Duration::from_millis(ms));
        }
        assert_eq!(
            stats.summary(),
            PresentSummary {
                count: 100,
                p50: Duration::from_millis(50),
                p95: Duration::from_millis(95),
                max: Duration::from_millis(100),
            }
        );
    }
}
