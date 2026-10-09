use std::time::Duration;

use eye_core::grid::Grid;
use eye_core::session::TargetRecord;
use eye_core::{ScreenModel, Timestamp};
use eye_geometry::screen::mm_to_px_logical;
use nalgebra::Point2;

use crate::error::CalibrationError;

#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProtocolConfig {
    pub grid: [u32; 2],
    pub lead_in_ms: u64,
    pub dwell_ms: u64,
    pub settle_ms: u64,
    pub window_ms: u64,
}

impl Default for ProtocolConfig {
    fn default() -> Self {
        Self {
            grid: [3, 3],
            lead_in_ms: 1000,
            dwell_ms: 1500,
            settle_ms: 600,
            window_ms: 800,
        }
    }
}

pub use eye_core::session::TargetTiming;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Target {
    pub index: u32,
    pub cell: (u32, u32),
    pub norm: Point2<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScheduledTarget {
    pub target: Target,
    pub onset: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FixationWindow {
    pub index: u32,
    pub cell: (u32, u32),
    pub onset: Timestamp,
    pub start: Timestamp,
    pub end: Timestamp,
    pub target_mm: Point2<f64>,
    pub target_px_logical: Point2<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetProtocol {
    cfg: ProtocolConfig,
    grid: Grid,
}

fn targets_err(reason: String) -> CalibrationError {
    CalibrationError::Param {
        name: "targets",
        reason,
    }
}

fn param_err(name: &'static str, reason: impl Into<String>) -> CalibrationError {
    CalibrationError::Param {
        name,
        reason: reason.into(),
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl TargetProtocol {
    pub fn new(cfg: ProtocolConfig) -> Result<Self, CalibrationError> {
        for &axis in &cfg.grid {
            if !(2..=8).contains(&axis) {
                return Err(param_err("grid", format!("grid axis {axis} outside 2..=8")));
            }
        }
        if cfg.dwell_ms == 0 || cfg.settle_ms == 0 || cfg.window_ms == 0 {
            return Err(param_err(
                "timing",
                "dwell_ms, settle_ms and window_ms must all be non-zero",
            ));
        }
        if cfg.settle_ms + cfg.window_ms > cfg.dwell_ms {
            return Err(param_err(
                "timing",
                format!(
                    "settle_ms ({}) + window_ms ({}) exceeds dwell_ms ({})",
                    cfg.settle_ms, cfg.window_ms, cfg.dwell_ms
                ),
            ));
        }
        let grid = Grid::new(cfg.grid[0], cfg.grid[1])
            .ok_or_else(|| param_err("grid", "grid axes must be non-zero"))?;
        Ok(Self { cfg, grid })
    }

    pub fn config(&self) -> &ProtocolConfig {
        &self.cfg
    }

    pub fn grid(&self) -> (u32, u32) {
        (self.grid.cols(), self.grid.rows())
    }

    pub fn sequence(&self, seed: u64) -> Vec<Target> {
        let (cols, rows) = self.grid();
        let mut cells: Vec<(u32, u32)> = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (c, r)))
            .collect();
        let mut s = seed;
        for i in (1..cells.len()).rev() {
            let j = (splitmix64(&mut s) % (i as u64 + 1)) as usize;
            cells.swap(i, j);
        }
        let targets: Vec<Target> = cells
            .into_iter()
            .enumerate()
            .map(|(k, (c, r))| Target {
                index: k as u32,
                cell: (c, r),
                norm: Point2::new(
                    (f64::from(c) + 0.5) / f64::from(cols),
                    (f64::from(r) + 0.5) / f64::from(rows),
                ),
            })
            .collect();
        tracing::debug!(
            seed,
            cols = u64::from(cols),
            rows = u64::from(rows),
            targets = targets.len() as u64,
            "target sequence generated"
        );
        targets
    }

    pub fn schedule(&self, seed: u64) -> Vec<ScheduledTarget> {
        let dwell = Duration::from_millis(self.cfg.dwell_ms);
        let lead_in = self.lead_in();
        self.sequence(seed)
            .into_iter()
            .enumerate()
            .map(|(k, target)| ScheduledTarget {
                target,
                onset: lead_in + dwell * (k as u32),
            })
            .collect()
    }

    pub fn timing(&self) -> TargetTiming {
        TargetTiming {
            settle: Duration::from_millis(self.cfg.settle_ms),
            window: Duration::from_millis(self.cfg.window_ms),
            dwell: Duration::from_millis(self.cfg.dwell_ms),
        }
    }

    pub fn display_sequence(
        &self,
        seed: u64,
        screen: &ScreenModel,
    ) -> Vec<(Point2<f64>, TargetTiming)> {
        let timing = self.timing();
        self.sequence(seed)
            .into_iter()
            .map(|t| (self.target_px_logical(&t, screen), timing))
            .collect()
    }

    pub fn lead_in(&self) -> Duration {
        Duration::from_millis(self.cfg.lead_in_ms)
    }

    pub fn target_mm(&self, target: &Target, screen: &ScreenModel) -> Point2<f64> {
        Point2::new(
            target.norm.x * screen.size_mm.x,
            target.norm.y * screen.size_mm.y,
        )
    }

    pub fn target_px_logical(&self, target: &Target, screen: &ScreenModel) -> Point2<f64> {
        mm_to_px_logical(screen, &self.target_mm(target, screen))
    }

    pub fn cell_of_mm(&self, mm: &Point2<f64>, screen: &ScreenModel) -> Option<(u32, u32)> {
        self.grid
            .cell_of(*mm, (screen.size_mm.x, screen.size_mm.y))
            .map(|c| (c.col, c.row))
    }

    pub fn fixation_windows(
        &self,
        records: &[TargetRecord],
        screen: &ScreenModel,
    ) -> Result<Vec<FixationWindow>, CalibrationError> {
        let settle = Duration::from_millis(self.cfg.settle_ms);
        let window = Duration::from_millis(self.cfg.window_ms);
        let min_gap = (settle + window).as_nanos() as u64;
        let mut out = Vec::with_capacity(records.len());
        for (k, r) in records.iter().enumerate() {
            if let Some(prev) = k.checked_sub(1).map(|p| &records[p]) {
                if r.shown_ns <= prev.shown_ns {
                    return Err(targets_err(format!(
                        "record {} is not after record {}",
                        r.seq, prev.seq
                    )));
                }
                if r.shown_ns - prev.shown_ns < min_gap {
                    return Err(targets_err(format!(
                        "records {} and {} are closer than settle + window",
                        prev.seq, r.seq
                    )));
                }
            }
            let mm = Point2::from(r.mm);
            let cell = self
                .cell_of_mm(&mm, screen)
                .ok_or_else(|| targets_err(format!("record {} lies outside the screen", r.seq)))?;
            let onset = Timestamp::from_nanos(r.shown_ns);
            let start = Timestamp(onset.0 + settle);
            let end = Timestamp(start.0 + window);
            tracing::trace!(
                index = k as u64,
                cell_col = u64::from(cell.0),
                cell_row = u64::from(cell.1),
                onset_ns = onset.as_nanos(),
                start_ns = start.as_nanos(),
                end_ns = end.as_nanos(),
                target_x_mm = mm.x,
                target_y_mm = mm.y,
                "fixation window"
            );
            out.push(FixationWindow {
                index: k as u32,
                cell,
                onset,
                start,
                end,
                target_mm: mm,
                target_px_logical: Point2::from(r.px_logical),
            });
        }
        tracing::info!(
            windows = out.len() as u64,
            settle_ms = self.cfg.settle_ms,
            window_ms = self.cfg.window_ms,
            "fixation windows built"
        );
        Ok(out)
    }
}

/// Sub-slice of `items` (which must be sorted by timestamp) inside `[window.start, window.end)`.
pub fn select_in_window<'a, T>(
    items: &'a [T],
    window: &FixationWindow,
    ts: impl Fn(&T) -> Timestamp,
) -> &'a [T] {
    let lo = items.partition_point(|x| ts(x) < window.start);
    let hi = items.partition_point(|x| ts(x) < window.end);
    &items[lo..hi]
}

#[cfg(test)]
mod tests {
    use eye_core::OutputId;
    use eye_core::session::TargetClock;
    use nalgebra::Vector2;

    use super::*;

    fn screen() -> ScreenModel {
        ScreenModel {
            output: OutputId::from("eDP-1"),
            size_mm: Vector2::new(310.0, 174.375),
            size_px: (3840, 2160),
            scale: 2.0,
        }
    }

    fn rec(seq: u64, seconds: f64, mm: [f64; 2]) -> TargetRecord {
        TargetRecord {
            seq,
            shown_ns: (seconds * 1e9).round() as u64,
            hidden_ns: None,
            clock: TargetClock::Presentation,
            output: OutputId::from("eDP-1"),
            px_logical: [0.0, 0.0],
            mm,
        }
    }

    #[test]
    fn test_default_config_values() {
        let d = ProtocolConfig::default();
        assert_eq!(
            d,
            ProtocolConfig {
                grid: [3, 3],
                lead_in_ms: 1000,
                dwell_ms: 1500,
                settle_ms: 600,
                window_ms: 800,
            }
        );
        let parsed: ProtocolConfig = toml::from_str("").unwrap();
        assert_eq!(parsed, d);
    }

    #[test]
    fn test_3x3_sequence_is_permutation_of_cell_centres() {
        let proto = TargetProtocol::new(ProtocolConfig {
            grid: [3, 3],
            ..ProtocolConfig::default()
        })
        .unwrap();
        let seq = proto.sequence(42);
        assert_eq!(seq.len(), 9);
        let mut cells: Vec<(u32, u32)> = seq.iter().map(|t| t.cell).collect();
        cells.sort();
        let mut expected_sorted: Vec<(u32, u32)> =
            (0..3).flat_map(|r| (0..3).map(move |c| (c, r))).collect();
        expected_sorted.sort();
        assert_eq!(cells, expected_sorted);
        for t in &seq {
            let ok = [1.0 / 6.0, 0.5, 5.0 / 6.0]
                .iter()
                .any(|e| (t.norm.x - e).abs() < 1e-12)
                && [1.0 / 6.0, 0.5, 5.0 / 6.0]
                    .iter()
                    .any(|e| (t.norm.y - e).abs() < 1e-12);
            assert!(ok);
        }
    }

    #[test]
    fn test_4x4_cell_centres() {
        let proto = TargetProtocol::new(ProtocolConfig {
            grid: [4, 4],
            ..ProtocolConfig::default()
        })
        .unwrap();
        let seq = proto.sequence(1);
        let expected_coords = [0.125, 0.375, 0.625, 0.875];
        for t in &seq {
            assert!(expected_coords.iter().any(|e| (t.norm.x - e).abs() < 1e-12));
            assert!(expected_coords.iter().any(|e| (t.norm.y - e).abs() < 1e-12));
        }
    }

    #[test]
    fn test_sequence_is_deterministic_per_seed() {
        let proto = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let a = proto.sequence(7);
        let b = proto.sequence(7);
        assert_eq!(a, b);
        let c = proto.sequence(1);
        let d = proto.sequence(2);
        assert_ne!(c, d);
    }

    #[test]
    fn test_logs_target_sequence_at_debug() {
        let proto = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let (_, records) =
            eye_log::testing::capture_logs(tracing::Level::DEBUG, || proto.sequence(7));
        let rec = records
            .iter()
            .find(|r| r.message == "target sequence generated")
            .expect("no 'target sequence generated' record");
        assert_eq!(rec.level, eye_log::Level::Debug);
        assert_eq!(rec.fields.get("seed"), Some(&eye_log::Value::U64(7)));
        assert_eq!(rec.fields.get("cols"), Some(&eye_log::Value::U64(3)));
        assert_eq!(rec.fields.get("rows"), Some(&eye_log::Value::U64(3)));
        assert_eq!(rec.fields.get("targets"), Some(&eye_log::Value::U64(9)));
    }

    #[test]
    fn test_display_sequence_matches_sequence() {
        let proto = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let screen = screen();
        let seq = proto.sequence(5);
        let disp = proto.display_sequence(5, &screen);
        assert_eq!(disp.len(), seq.len());
        for (t, (px, timing)) in seq.iter().zip(disp.iter()) {
            assert_eq!(*px, proto.target_px_logical(t, &screen));
            assert_eq!(*timing, proto.timing());
        }
    }

    #[test]
    fn test_display_sequence_carries_timing() {
        let proto = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let screen = screen();
        let timing = proto.timing();
        for (_, t) in proto.display_sequence(7, &screen) {
            assert_eq!(t, timing);
        }
    }

    #[test]
    fn test_timing_sums_within_dwell() {
        let proto = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let timing = proto.timing();
        assert!(timing.settle + timing.window <= timing.dwell);
    }

    #[test]
    fn test_schedule_onsets() {
        let proto = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let schedule = proto.schedule(3);
        assert_eq!(schedule.len(), 9);
        for (k, s) in schedule.iter().enumerate() {
            let expected = Duration::from_millis(1000 + 1500 * (k as u64));
            assert_eq!(s.onset, expected);
        }
        assert_eq!(
            schedule.last().unwrap().onset,
            Duration::from_millis(13_000)
        );
        assert_eq!(
            proto.lead_in() + Duration::from_millis(1500) * 9,
            Duration::from_millis(14_500)
        );
    }

    #[test]
    fn test_centre_target_mm_on_edp1() {
        let proto = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let screen = screen();
        let target = Target {
            index: 0,
            cell: (1, 1),
            norm: Point2::new(0.5, 0.5),
        };
        let mm = proto.target_mm(&target, &screen);
        assert!((mm.x - 155.0).abs() < 1e-9);
        assert!((mm.y - 87.1875).abs() < 1e-9);
    }

    #[test]
    fn test_cell_of_mm_round_trips_target_positions() {
        let proto = TargetProtocol::new(ProtocolConfig {
            grid: [4, 4],
            ..ProtocolConfig::default()
        })
        .unwrap();
        let screen = screen();
        for t in proto.sequence(9) {
            let mm = proto.target_mm(&t, &screen);
            assert_eq!(proto.cell_of_mm(&mm, &screen), Some(t.cell));
        }
        assert_eq!(proto.cell_of_mm(&Point2::new(-1.0, 3.0), &screen), None);
    }

    #[test]
    fn test_fixation_windows_from_records() {
        let proto = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let screen = screen();
        let records = vec![rec(0, 10.0, [155.0, 87.1875]), rec(1, 11.5, [51.0, 29.0])];
        let windows = proto.fixation_windows(&records, &screen).unwrap();
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].start, Timestamp::from_nanos(10_600_000_000));
        assert_eq!(windows[0].end, Timestamp::from_nanos(11_400_000_000));
        assert_eq!(windows[0].cell, (1, 1));
        assert_eq!(windows[0].target_mm, Point2::new(155.0, 87.1875));
        assert_eq!(windows[1].start, Timestamp::from_nanos(12_100_000_000));
        assert_eq!(windows[1].end, Timestamp::from_nanos(12_900_000_000));
        assert_eq!(windows[1].cell, (0, 0));
        assert_eq!(windows[1].target_mm, Point2::new(51.0, 29.0));
    }

    #[test]
    fn test_logs_fixation_windows_at_info() {
        let proto = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let screen = screen();
        let records = vec![rec(0, 10.0, [155.0, 87.1875]), rec(1, 11.5, [51.0, 29.0])];

        let (_, logs) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            proto.fixation_windows(&records, &screen).unwrap()
        });

        let summary = logs
            .iter()
            .find(|r| r.message == "fixation windows built")
            .expect("no 'fixation windows built' record");
        assert_eq!(summary.level, eye_log::Level::Info);
        assert_eq!(summary.fields.get("windows"), Some(&eye_log::Value::U64(2)));
        assert_eq!(
            summary.fields.get("settle_ms"),
            Some(&eye_log::Value::U64(600))
        );
        assert_eq!(
            summary.fields.get("window_ms"),
            Some(&eye_log::Value::U64(800))
        );

        let windows: Vec<_> = logs
            .iter()
            .filter(|r| r.message == "fixation window")
            .collect();
        assert_eq!(windows.len(), 2);
        for w in &windows {
            assert_eq!(w.level, eye_log::Level::Trace);
        }
        assert_eq!(
            windows[0].fields.get("start_ns"),
            Some(&eye_log::Value::U64(10_600_000_000))
        );
        assert_eq!(
            windows[0].fields.get("cell_col"),
            Some(&eye_log::Value::U64(1))
        );
        assert_eq!(
            windows[0].fields.get("cell_row"),
            Some(&eye_log::Value::U64(1))
        );
        assert_eq!(
            windows[1].fields.get("start_ns"),
            Some(&eye_log::Value::U64(12_100_000_000))
        );
        assert_eq!(
            windows[1].fields.get("cell_col"),
            Some(&eye_log::Value::U64(0))
        );
        assert_eq!(
            windows[1].fields.get("cell_row"),
            Some(&eye_log::Value::U64(0))
        );
    }

    #[test]
    fn test_commit_clock_records_accepted() {
        let proto = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let screen = screen();
        let mut records = vec![rec(0, 10.0, [155.0, 87.1875]), rec(1, 11.5, [51.0, 29.0])];
        for r in &mut records {
            r.clock = TargetClock::Commit;
        }
        let windows = proto.fixation_windows(&records, &screen).unwrap();

        let mut presentation = vec![rec(0, 10.0, [155.0, 87.1875]), rec(1, 11.5, [51.0, 29.0])];
        for r in &mut presentation {
            r.clock = TargetClock::Presentation;
        }
        let expected = proto.fixation_windows(&presentation, &screen).unwrap();
        assert_eq!(windows, expected);
    }

    #[test]
    fn test_records_too_close_errors() {
        let proto = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let screen = screen();
        let records = vec![rec(0, 10.0, [155.0, 87.1875]), rec(1, 11.0, [51.0, 29.0])];
        let err = proto.fixation_windows(&records, &screen).unwrap_err();
        assert!(matches!(
            err,
            CalibrationError::Param {
                name: "targets",
                ..
            }
        ));
    }

    #[test]
    fn test_non_increasing_records_error() {
        let proto = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let screen = screen();
        let records = vec![rec(0, 10.0, [155.0, 87.1875]), rec(1, 10.0, [51.0, 29.0])];
        let err = proto.fixation_windows(&records, &screen).unwrap_err();
        assert!(matches!(
            err,
            CalibrationError::Param {
                name: "targets",
                ..
            }
        ));
    }

    #[test]
    fn test_record_outside_screen_errors() {
        let proto = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let screen = screen();
        let records = vec![rec(0, 10.0, [400.0, 10.0])];
        let err = proto.fixation_windows(&records, &screen).unwrap_err();
        assert!(matches!(
            err,
            CalibrationError::Param {
                name: "targets",
                ..
            }
        ));
    }

    #[test]
    fn test_select_in_window_boundaries() {
        let window = FixationWindow {
            index: 0,
            cell: (0, 0),
            onset: Timestamp::from_nanos(0),
            start: Timestamp::from_nanos(10_600_000_000),
            end: Timestamp::from_nanos(11_400_000_000),
            target_mm: Point2::new(0.0, 0.0),
            target_px_logical: Point2::new(0.0, 0.0),
        };
        let items: Vec<Timestamp> = [10.59_f64, 10.60, 11.39, 11.40]
            .iter()
            .map(|s| Timestamp::from_nanos((*s * 1e9).round() as u64))
            .collect();
        let selected = select_in_window(&items, &window, |t| *t);
        assert_eq!(
            selected,
            &[
                Timestamp::from_nanos(10_600_000_000),
                Timestamp::from_nanos(11_390_000_000)
            ]
        );
    }

    #[test]
    fn test_config_validation() {
        let err = TargetProtocol::new(ProtocolConfig {
            settle_ms: 900,
            window_ms: 800,
            dwell_ms: 1500,
            ..ProtocolConfig::default()
        })
        .unwrap_err();
        assert!(matches!(
            err,
            CalibrationError::Param { name: "timing", .. }
        ));

        let err = TargetProtocol::new(ProtocolConfig {
            dwell_ms: 0,
            ..ProtocolConfig::default()
        })
        .unwrap_err();
        assert!(matches!(
            err,
            CalibrationError::Param { name: "timing", .. }
        ));

        let err = TargetProtocol::new(ProtocolConfig {
            grid: [1, 3],
            ..ProtocolConfig::default()
        })
        .unwrap_err();
        assert!(matches!(err, CalibrationError::Param { name: "grid", .. }));

        let err = TargetProtocol::new(ProtocolConfig {
            grid: [9, 3],
            ..ProtocolConfig::default()
        })
        .unwrap_err();
        assert!(matches!(err, CalibrationError::Param { name: "grid", .. }));

        let parsed: Result<ProtocolConfig, _> = toml::from_str("unknown_field = 1");
        assert!(parsed.is_err());
    }
}
