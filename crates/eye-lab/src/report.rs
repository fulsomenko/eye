use crate::case::Measurement;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Pass,
    Fail,
    Skipped,
    Error,
}

impl Verdict {
    pub fn upper(self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Skipped => "SKIPPED",
            Verdict::Error => "ERROR",
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct StepResult {
    pub index: usize,
    pub label: String,
    pub test: String,
    pub origin: String,
    pub mode: String,
    pub verdict: Verdict,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub measurements: Vec<Measurement>,
    pub notes: Vec<String>,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct Tally {
    pub pass: usize,
    pub fail: usize,
    pub skipped: usize,
    pub error: usize,
}

pub fn tally(steps: &[StepResult]) -> Tally {
    let mut t = Tally::default();
    for step in steps {
        match step.verdict {
            Verdict::Pass => t.pass += 1,
            Verdict::Fail => t.fail += 1,
            Verdict::Skipped => t.skipped += 1,
            Verdict::Error => t.error += 1,
        }
    }
    t
}

pub fn exit_code(tally: &Tally, aborted: bool) -> u8 {
    if aborted || tally.error > 0 {
        2
    } else if tally.fail > 0 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exit_code_precedence() {
        assert_eq!(
            exit_code(
                &Tally {
                    error: 1,
                    fail: 1,
                    ..Default::default()
                },
                false
            ),
            2
        );
        assert_eq!(
            exit_code(
                &Tally {
                    fail: 1,
                    pass: 3,
                    ..Default::default()
                },
                false
            ),
            1
        );
        assert_eq!(
            exit_code(
                &Tally {
                    pass: 1,
                    skipped: 5,
                    ..Default::default()
                },
                false
            ),
            0
        );
        assert_eq!(exit_code(&Tally::default(), true), 2);
    }

    #[test]
    fn test_tally_counts_each_verdict() {
        let steps = [
            Verdict::Pass,
            Verdict::Fail,
            Verdict::Skipped,
            Verdict::Error,
            Verdict::Pass,
        ]
        .into_iter()
        .enumerate()
        .map(|(i, verdict)| StepResult {
            index: i + 1,
            label: "l".to_owned(),
            test: "t".to_owned(),
            origin: "o".to_owned(),
            mode: "none".to_owned(),
            verdict,
            reason: None,
            measurements: Vec::new(),
            notes: Vec::new(),
            duration_ms: 0,
        })
        .collect::<Vec<_>>();
        assert_eq!(
            tally(&steps),
            Tally {
                pass: 2,
                fail: 1,
                skipped: 1,
                error: 1
            }
        );
    }
}
