use std::time::{SystemTime, UNIX_EPOCH};

pub fn utc_stamp(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    let s = secs.rem_euclid(86_400);
    format!(
        "{y:04}{m:02}{d:02}T{:02}{:02}{:02}Z",
        s / 3600,
        s % 3600 / 60,
        s % 60
    )
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_utc_stamp_epoch() {
        assert_eq!(utc_stamp(UNIX_EPOCH), "19700101T000000Z");
    }

    #[test]
    fn test_utc_stamp_known_instant() {
        let t = UNIX_EPOCH + Duration::from_secs(1_791_409_623);
        assert_eq!(utc_stamp(t), "20261007T214703Z");
    }

    #[test]
    fn test_utc_stamp_leap_day() {
        let t = UNIX_EPOCH + Duration::from_secs(1_709_251_199);
        assert_eq!(utc_stamp(t), "20240229T235959Z");
    }
}
