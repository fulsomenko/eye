use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::OutputId;

/// Which clock reading `shown_ns`/`hidden_ns` come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetClock {
    /// `wp_presentation` feedback: when the frame reached the screen.
    Presentation,
    /// `CLOCK_MONOTONIC` read right after `wl_surface.commit`, used when the compositor has no
    /// presentation feedback on `CLOCK_MONOTONIC`.
    Commit,
}

/// One line of a recording's `targets.jsonl`. Times are `CLOCK_MONOTONIC` nanoseconds, the clock
/// of [`crate::Timestamp`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetRecord {
    pub seq: u64,
    pub shown_ns: u64,
    pub hidden_ns: Option<u64>,
    pub clock: TargetClock,
    pub output: OutputId,
    pub px_logical: [f64; 2],
    pub mm: [f64; 2],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Timestamp;

    fn record() -> TargetRecord {
        TargetRecord {
            seq: 1,
            shown_ns: 5,
            hidden_ns: None,
            clock: TargetClock::Presentation,
            output: OutputId::from("eDP-1"),
            px_logical: [960.0, 540.0],
            mm: [155.0, 85.0],
        }
    }

    #[test]
    fn test_target_record_jsonl_shape() {
        let json = serde_json::to_string(&record()).unwrap();
        assert_eq!(
            json,
            r#"{"seq":1,"shown_ns":5,"hidden_ns":null,"clock":"presentation","output":"eDP-1","px_logical":[960.0,540.0],"mm":[155.0,85.0]}"#
        );
        let parsed: TargetRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, record());
    }

    #[test]
    fn test_target_record_hidden_and_commit_roundtrip() {
        let mut r = record();
        r.hidden_ns = Some(2_000_000_005);
        r.clock = TargetClock::Commit;
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains(r#""hidden_ns":2000000005"#));
        assert!(json.contains(r#""clock":"commit""#));
        let parsed: TargetRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, r);
    }

    #[test]
    fn test_target_record_rejects_negative_ns() {
        let json = r#"{"seq":1,"shown_ns":-5,"hidden_ns":null,"clock":"presentation","output":"eDP-1","px_logical":[960.0,540.0],"mm":[155.0,85.0]}"#;
        let parsed: Result<TargetRecord, _> = serde_json::from_str(json);
        assert!(parsed.is_err());
    }

    #[test]
    fn test_shown_ns_converts_from_timestamp_exactly() {
        let ts = Timestamp::from_nanos(123_456_789_012);
        let shown_ns = ts.as_nanos();
        assert_eq!(Timestamp::from_nanos(shown_ns), ts);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetTiming {
    /// Eye still travelling; samples ignored.
    pub settle: Duration,
    /// Samples are taken.
    pub window: Duration,
    pub dwell: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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

impl ProtocolConfig {
    /// D-A4: dwell = median(hidden_ns - shown_ns) over records with `hidden_ns`, snapped to the nearest
    /// 100 ms (`((median_ms / 100.0).round() * 100.0) as u64`, so the ~23 ms of frame-pacing jitter on
    /// every legacy recording derives exactly 1500), settle = round(0.4 dwell), window = round(8/15 dwell)
    /// (the ratios of `Self::default()`), grid from the count of distinct `(mm[0].to_bits(), mm[1].to_bits())`
    /// positions: 9 -> [3, 3], 16 -> [4, 4]. `lead_in_ms` from `fallback`.
    /// `None` when fewer than two records carry `hidden_ns` or the position count is neither 9 nor 16.
    pub fn from_target_records(
        records: &[TargetRecord],
        fallback: &ProtocolConfig,
    ) -> Option<ProtocolConfig> {
        let mut diffs: Vec<u64> = records
            .iter()
            .filter_map(|r| r.hidden_ns.map(|hidden| hidden.saturating_sub(r.shown_ns)))
            .collect();
        if diffs.len() < 2 {
            return None;
        }
        diffs.sort_unstable();
        let mid = diffs.len() / 2;
        let median_ns = if diffs.len().is_multiple_of(2) {
            (diffs[mid - 1] + diffs[mid]) as f64 / 2.0
        } else {
            diffs[mid] as f64
        };
        let median_ms = median_ns / 1_000_000.0;
        let dwell_ms = ((median_ms / 100.0).round() * 100.0) as u64;
        let settle_ms = (dwell_ms as f64 * 0.4).round() as u64;
        let window_ms = (dwell_ms as f64 * 8.0 / 15.0).round() as u64;

        let mut positions: Vec<(u64, u64)> = records
            .iter()
            .map(|r| (r.mm[0].to_bits(), r.mm[1].to_bits()))
            .collect();
        positions.sort_unstable();
        positions.dedup();
        let grid = match positions.len() {
            9 => [3, 3],
            16 => [4, 4],
            _ => return None,
        };

        Some(ProtocolConfig {
            grid,
            lead_in_ms: fallback.lead_in_ms,
            dwell_ms,
            settle_ms,
            window_ms,
        })
    }
}

#[cfg(test)]
mod protocol_config_from_target_records_tests {
    use super::*;

    fn record_at(seq: u64, shown_ns: u64, hidden_ns: Option<u64>, mm: [f64; 2]) -> TargetRecord {
        TargetRecord {
            seq,
            shown_ns,
            hidden_ns,
            clock: TargetClock::Commit,
            output: OutputId::from("eDP-1"),
            px_logical: [0.0, 0.0],
            mm,
        }
    }

    fn records(n: u64, dwell_ns: u64) -> Vec<TargetRecord> {
        (0..n)
            .map(|i| {
                let shown_ns = i * 2_000_000_000;
                record_at(i, shown_ns, Some(shown_ns + dwell_ns), [i as f64, 0.0])
            })
            .collect()
    }

    #[test]
    fn test_from_target_records_reproduces_default_for_1500_dwell() {
        let derived = ProtocolConfig::from_target_records(
            &records(16, 1_500_000_000),
            &ProtocolConfig::default(),
        )
        .expect("16 records with hidden_ns and 16 distinct positions derive a protocol");
        assert_eq!(
            derived,
            ProtocolConfig {
                grid: [4, 4],
                lead_in_ms: 1000,
                dwell_ms: 1500,
                settle_ms: 600,
                window_ms: 800,
            }
        );
    }

    #[test]
    fn test_from_target_records_rounds_jittered_dwell_to_default() {
        let derived = ProtocolConfig::from_target_records(
            &records(16, 1_523_000_000),
            &ProtocolConfig::default(),
        )
        .expect("jittered dwell still derives");
        assert_eq!(
            derived,
            ProtocolConfig {
                grid: [4, 4],
                lead_in_ms: 1000,
                dwell_ms: 1500,
                settle_ms: 600,
                window_ms: 800,
            }
        );
    }

    #[test]
    fn test_from_target_records_scales_for_2500_dwell() {
        let derived = ProtocolConfig::from_target_records(
            &records(9, 2_500_000_000),
            &ProtocolConfig::default(),
        )
        .expect("9 records with hidden_ns and 9 distinct positions derive a protocol");
        assert_eq!(
            derived,
            ProtocolConfig {
                grid: [3, 3],
                lead_in_ms: 1000,
                dwell_ms: 2500,
                settle_ms: 1000,
                window_ms: 1333,
            }
        );
    }

    #[test]
    fn test_from_target_records_needs_hidden_and_known_grid() {
        let fallback = ProtocolConfig::default();

        let without_hidden: Vec<TargetRecord> = (0..16)
            .map(|i| record_at(i, i * 2_000_000_000, None, [i as f64, 0.0]))
            .collect();
        assert_eq!(
            ProtocolConfig::from_target_records(&without_hidden, &fallback),
            None
        );

        let ten_positions: Vec<TargetRecord> = (0..10)
            .map(|i| {
                let shown_ns = i * 2_000_000_000;
                record_at(i, shown_ns, Some(shown_ns + 1_500_000_000), [i as f64, 0.0])
            })
            .collect();
        assert_eq!(
            ProtocolConfig::from_target_records(&ten_positions, &fallback),
            None
        );
    }
}

#[cfg(test)]
mod protocol_config_tests {
    use super::*;

    #[test]
    fn test_protocol_config_toml_round_trip() {
        let cfg = ProtocolConfig {
            grid: [4, 4],
            lead_in_ms: 1000,
            dwell_ms: 2500,
            settle_ms: 1000,
            window_ms: 1200,
        };
        let text = toml::to_string(&cfg).unwrap();
        assert!(text.contains("grid = [4, 4]"));
        assert!(text.contains("lead_in_ms = 1000"));
        assert!(text.contains("dwell_ms = 2500"));
        assert!(text.contains("settle_ms = 1000"));
        assert!(text.contains("window_ms = 1200"));
        let parsed: ProtocolConfig = toml::from_str(&text).unwrap();
        assert_eq!(parsed, cfg);
    }
}
