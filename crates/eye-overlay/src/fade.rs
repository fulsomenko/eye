//! Staleness fade: how faded a gaze point should look after it stopped updating.

use std::time::Duration;

pub const FADE_START: Duration = Duration::from_millis(300);
pub const FADE_END: Duration = Duration::from_millis(800);

pub fn fade(age: Duration) -> f64 {
    if age <= FADE_START {
        return 1.0;
    }
    if age >= FADE_END {
        return 0.0;
    }
    (FADE_END - age).as_secs_f64() / (FADE_END - FADE_START).as_secs_f64()
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_fade_profile() {
        assert_relative_eq!(fade(Duration::from_millis(0)), 1.0);
        assert_relative_eq!(fade(Duration::from_millis(300)), 1.0);
        assert_relative_eq!(fade(Duration::from_millis(550)), 0.5);
        assert_relative_eq!(fade(Duration::from_millis(800)), 0.0);
        assert_relative_eq!(fade(Duration::from_millis(2000)), 0.0);
    }
}
