use std::time::Duration;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(from = "u64", into = "u64")]
pub struct Timestamp(pub Duration);

impl Timestamp {
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(Duration::from_nanos(nanos))
    }

    pub fn as_nanos(self) -> u64 {
        u64::try_from(self.0.as_nanos()).unwrap_or(u64::MAX)
    }

    pub fn nanos_since(self, other: Timestamp) -> i64 {
        let diff = i128::from(self.as_nanos()) - i128::from(other.as_nanos());
        i64::try_from(diff).unwrap_or(if diff < 0 { i64::MIN } else { i64::MAX })
    }

    pub fn abs_diff(self, other: Timestamp) -> Duration {
        if self >= other {
            self.0 - other.0
        } else {
            other.0 - self.0
        }
    }
}

impl From<u64> for Timestamp {
    fn from(nanos: u64) -> Self {
        Self::from_nanos(nanos)
    }
}

impl From<Timestamp> for u64 {
    fn from(timestamp: Timestamp) -> Self {
        timestamp.as_nanos()
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_nanos_since_earlier_is_negative() {
        let a = Timestamp::from_nanos(1_000);
        let b = Timestamp::from_nanos(3_500);
        assert_eq!(a.nanos_since(b), -2_500);
        assert_eq!(b.nanos_since(a), 2_500);
    }

    #[test]
    fn test_abs_diff_is_symmetric() {
        let a = Timestamp::from_nanos(1_000_000_000);
        let b = Timestamp::from_nanos(1_004_000_000);
        assert_eq!(a.abs_diff(b), Duration::from_millis(4));
        assert_eq!(b.abs_diff(a), Duration::from_millis(4));
    }

    #[test]
    fn test_nanos_since_saturates_at_i64_range() {
        let max = Timestamp::from_nanos(u64::MAX);
        let zero = Timestamp::from_nanos(0);
        assert_eq!(max.nanos_since(zero), i64::MAX);
        assert_eq!(zero.nanos_since(max), i64::MIN);
    }

    #[test]
    fn test_timestamp_serializes_as_integer_nanos() {
        let ts = Timestamp::from_nanos(1_234_567_890);
        let json = serde_json::to_string(&ts).unwrap();
        assert_eq!(json, "1234567890");
        let parsed: Timestamp = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, ts);
    }

    proptest! {
        #[test]
        fn test_nanos_roundtrip_preserves_value(n: u64) {
            prop_assert_eq!(Timestamp::from_nanos(n).as_nanos(), n);
        }
    }
}
