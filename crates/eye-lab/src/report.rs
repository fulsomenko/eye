use std::{
    collections::BTreeMap,
    fs, io,
    io::Write,
    path::{Path, PathBuf},
};

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

/// The sentence printed at the end of a run, in the markdown report and on stdout.
pub fn result_sentence(tally: &Tally, exit_code: u8) -> String {
    format!(
        "Result: {} pass, {} fail, {} skipped, {} error. Exit code {}.",
        tally.pass, tally.fail, tally.skipped, tally.error, exit_code
    )
}

pub const REPORT_SCHEMA: u32 = 1;

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Report {
    pub schema: u32,
    pub tool: String,
    pub sequence: String,
    pub sources: Vec<String>,
    pub started_unix_s: u64,
    pub duration_ms: u64,
    pub subject: bool,
    pub environment: BTreeMap<String, String>,
    pub steps: Vec<StepResult>,
    pub tally: Tally,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aborted: Option<String>,
    pub exit_code: u8,
}

/// hostname and kernel from /proc/sys/kernel/{hostname,osrelease}, trimmed; missing files are left out.
pub fn base_environment() -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    if let Ok(hostname) = fs::read_to_string("/proc/sys/kernel/hostname") {
        env.insert("hostname".to_owned(), hostname.trim().to_owned());
    }
    if let Ok(kernel) = fs::read_to_string("/proc/sys/kernel/osrelease") {
        env.insert("kernel".to_owned(), kernel.trim().to_owned());
    }
    env
}

fn escape_pipe(s: &str) -> String {
    s.replace('|', "\\|")
}

fn write_detail(out: &mut dyn Write, step: &StepResult) -> io::Result<()> {
    writeln!(
        out,
        "### {}. {} [{}]: {}",
        step.index,
        step.label,
        step.mode,
        step.verdict.upper()
    )?;
    writeln!(out)?;
    if let Some(reason) = &step.reason {
        writeln!(out, "- reason: {reason}")?;
    }
    for m in &step.measurements {
        writeln!(out, "- {m}")?;
    }
    for note in &step.notes {
        writeln!(out, "- note: {note}")?;
    }
    Ok(())
}

pub fn write_markdown(report: &Report, out: &mut dyn Write) -> io::Result<()> {
    writeln!(out, "# eye-lab report: {}", report.sequence)?;
    writeln!(out)?;
    writeln!(out, "- tool: {}", report.tool)?;
    writeln!(out, "- started: {} (unix s)", report.started_unix_s)?;
    writeln!(
        out,
        "- duration: {:.1} s",
        report.duration_ms as f64 / 1000.0
    )?;
    writeln!(
        out,
        "- subject: {}",
        if report.subject { "yes" } else { "no" }
    )?;
    for (k, v) in &report.environment {
        writeln!(out, "- {k}: {v}")?;
    }
    writeln!(out)?;
    writeln!(out, "| # | step | mode | verdict | measurements |")?;
    writeln!(out, "|---|---|---|---|---|")?;
    for step in &report.steps {
        let verdict_cell = match &step.reason {
            Some(reason) => format!("{}: {}", step.verdict.upper(), escape_pipe(reason)),
            None => step.verdict.upper().to_owned(),
        };
        let measurements_cell = step
            .measurements
            .iter()
            .map(|m| escape_pipe(&m.to_string()))
            .collect::<Vec<_>>()
            .join("; ");
        writeln!(
            out,
            "| {} | {} | {} | {} | {} |",
            step.index,
            escape_pipe(&step.label),
            escape_pipe(&step.mode),
            verdict_cell,
            measurements_cell
        )?;
    }
    writeln!(out)?;
    writeln!(out, "{}", result_sentence(&report.tally, report.exit_code))?;

    let details: Vec<&StepResult> = report
        .steps
        .iter()
        .filter(|s| matches!(s.verdict, Verdict::Fail | Verdict::Error))
        .collect();
    if !details.is_empty() {
        writeln!(out)?;
        writeln!(out, "## Details")?;
        for step in &details {
            writeln!(out)?;
            write_detail(out, step)?;
        }
    }
    Ok(())
}

/// Creates `dir` (and parents), writes `report.json` (pretty) and `report.md`; returns both paths.
pub fn write_files(report: &Report, dir: &Path) -> io::Result<(PathBuf, PathBuf)> {
    fs::create_dir_all(dir)?;
    let json_path = dir.join("report.json");
    let md_path = dir.join("report.md");

    let mut json_file = fs::File::create(&json_path)?;
    serde_json::to_writer_pretty(&mut json_file, report).map_err(io::Error::other)?;
    json_file.write_all(b"\n")?;

    let mut md_file = fs::File::create(&md_path)?;
    write_markdown(report, &mut md_file)?;

    Ok((json_path, md_path))
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

    fn golden_steps() -> Vec<StepResult> {
        vec![
            StepResult {
                index: 1,
                label: "selftest-check".to_owned(),
                test: "selftest-check".to_owned(),
                origin: "builtin:selftest.toml".to_owned(),
                mode: "none".to_owned(),
                verdict: Verdict::Pass,
                reason: None,
                measurements: vec![Measurement::at_least("value", 42.0, "", 40.0)],
                notes: Vec::new(),
                duration_ms: 500,
            },
            StepResult {
                index: 2,
                label: "selftest-check".to_owned(),
                test: "selftest-check".to_owned(),
                origin: "builtin:selftest.toml".to_owned(),
                mode: "none".to_owned(),
                verdict: Verdict::Fail,
                reason: None,
                measurements: vec![Measurement::at_least("value", 10.0, "", 40.0)],
                notes: Vec::new(),
                duration_ms: 500,
            },
            StepResult {
                index: 3,
                label: "selftest-skip".to_owned(),
                test: "selftest-skip".to_owned(),
                origin: "builtin:selftest.toml".to_owned(),
                mode: "none".to_owned(),
                verdict: Verdict::Skipped,
                reason: Some("demo".to_owned()),
                measurements: Vec::new(),
                notes: Vec::new(),
                duration_ms: 500,
            },
        ]
    }

    fn golden_report() -> Report {
        let steps = golden_steps();
        let tally = tally(&steps);
        let exit_code = exit_code(&tally, false);
        Report {
            schema: REPORT_SCHEMA,
            tool: "eye-lab 0.1.0".to_owned(),
            sequence: "demo".to_owned(),
            sources: vec!["builtin:selftest.toml".to_owned()],
            started_unix_s: 1_791_400_000,
            duration_ms: 1500,
            subject: true,
            environment: BTreeMap::from([("kernel".to_owned(), "7.2.8".to_owned())]),
            steps,
            tally,
            aborted: None,
            exit_code,
        }
    }

    #[test]
    fn test_markdown_report_golden() {
        let report = golden_report();
        let mut buf = Vec::new();
        write_markdown(&report, &mut buf).unwrap();
        let text = String::from_utf8(buf).unwrap();
        let expected = "\
# eye-lab report: demo

- tool: eye-lab 0.1.0
- started: 1791400000 (unix s)
- duration: 1.5 s
- subject: yes
- kernel: 7.2.8

| # | step | mode | verdict | measurements |
|---|---|---|---|---|
| 1 | selftest-check | none | PASS | value = 42.000 (>= 40) |
| 2 | selftest-check | none | FAIL | value = 10.000 (>= 40) FAIL |
| 3 | selftest-skip | none | SKIPPED: demo |  |

Result: 1 pass, 1 fail, 1 skipped, 0 error. Exit code 1.

## Details

### 2. selftest-check [none]: FAIL

- value = 10.000 (>= 40) FAIL
";
        assert_eq!(text, expected);
    }

    #[test]
    fn test_markdown_escapes_pipes() {
        let mut report = golden_report();
        report.steps[0].label = "a|b".to_owned();
        let mut buf = Vec::new();
        write_markdown(&report, &mut buf).unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert!(text.contains("a\\|b"), "{text}");
        assert!(!text.contains("a|b |"), "{text}");
    }

    #[test]
    fn test_json_report_shape() {
        let mut report = golden_report();
        report.steps[0]
            .measurements
            .push(Measurement::info("extra", f64::NAN, ""));
        let value = serde_json::to_value(&report).unwrap();
        assert_eq!(value["schema"], 1);
        assert_eq!(value["steps"][0]["verdict"], "pass");
        assert_eq!(
            value["steps"][0]["measurements"][0]["limit"]["op"],
            "at_least"
        );
        assert_eq!(value["steps"][2]["reason"], "demo");
        assert!(value["steps"][0].get("reason").is_none());
        assert!(value["steps"][0]["measurements"][1]["value"].is_null());
    }

    #[test]
    fn test_write_files_creates_nested_dir() {
        let report = golden_report();
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("a/b");
        let (json_path, md_path) = write_files(&report, &dir).unwrap();
        assert!(json_path.exists());
        assert!(md_path.exists());
        let md = fs::read_to_string(&md_path).unwrap();
        assert!(md.starts_with("# eye-lab report: demo"));
    }
}
