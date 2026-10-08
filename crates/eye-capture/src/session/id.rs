use std::time::{SystemTime, UNIX_EPOCH};

use crate::CaptureError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionId(String);

pub(crate) fn valid_component(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with('.')
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-')
}

impl SessionId {
    pub fn new(id: impl Into<String>) -> Result<Self, CaptureError> {
        let id = id.into();
        if valid_component(&id) {
            Ok(Self(id))
        } else {
            Err(CaptureError::Config {
                camera: "session".to_string(),
                reason: format!("invalid session id {id:?}"),
            })
        }
    }

    pub fn from_unix(secs: u64) -> Self {
        let days = (secs / 86_400) as i64;
        let rem = secs % 86_400;
        let (year, month, day) = civil_from_days(days);
        let hour = rem / 3_600;
        let minute = (rem % 3_600) / 60;
        let second = rem % 60;
        Self(format!(
            "{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z"
        ))
    }

    pub fn now() -> Self {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self::from_unix(secs)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

pub(crate) fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_id_from_unix() {
        assert_eq!(SessionId::from_unix(0).as_str(), "19700101T000000Z");
        assert_eq!(
            SessionId::from_unix(1_709_164_800).as_str(),
            "20240229T000000Z"
        );
        assert_eq!(
            SessionId::from_unix(1_791_411_300).as_str(),
            "20261007T221500Z"
        );
    }

    #[test]
    fn test_session_id_rejects_path_tricks() {
        assert!(SessionId::new("").is_err());
        assert!(SessionId::new("../x").is_err());
        assert!(SessionId::new("a/b").is_err());
        assert!(SessionId::new(".hidden").is_err());
        assert!(SessionId::new("dots-session_1.v2").is_ok());
    }
}
