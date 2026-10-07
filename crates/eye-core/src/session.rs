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
