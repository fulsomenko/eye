use std::{
    fmt,
    io::Write,
    panic::{self, AssertUnwindSafe},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use crossbeam_channel::{Receiver, RecvTimeoutError};

use crate::{
    case::{
        Measurement, Needs, RunOptions, TestCase, TestCtx, TestError, TestOutput, TestRegistry,
    },
    mode::{Mode, ModeHost},
    report::{StepResult, Verdict},
    sequence::{LoadError, ModeSpec, ResolvedStep, StreamsSel},
};

#[derive(Debug, Clone)]
pub struct PlannedStep {
    pub origin: String,
    pub label: String,
    pub test: String,
    pub mode: ModeSpec,
    pub case: Arc<dyn TestCase>,
    pub timeout: Duration,
}

fn spec_admits(streams: StreamsSel, needs: Needs) -> Result<(), &'static str> {
    let has_rgb = matches!(
        streams,
        StreamsSel::Rgb | StreamsSel::Dual | StreamsSel::All
    );
    let has_ir = matches!(streams, StreamsSel::Ir | StreamsSel::Dual | StreamsSel::All);
    if needs.rgb && !has_rgb {
        return Err("an RGB stream");
    }
    if (needs.ir || needs.emitter) && !has_ir {
        return Err("an IR stream");
    }
    if needs.any_stream && streams == StreamsSel::None {
        return Err("a stream");
    }
    Ok(())
}

fn mode_admits(mode: &Mode, needs: Needs) -> bool {
    (!needs.rgb || mode.rgb.is_some())
        && (!(needs.ir || needs.emitter) || mode.ir.is_some())
        && (!needs.any_stream || mode.rgb.is_some() || mode.ir.is_some())
}

/// Builds every case up front: unknown tests, bad params and mode/needs mismatches fail
/// before any hardware is touched.
pub fn plan(
    steps: Vec<ResolvedStep>,
    registry: &TestRegistry,
) -> Result<Vec<PlannedStep>, LoadError> {
    steps
        .into_iter()
        .enumerate()
        .map(|(i, s)| {
            let step_num = i + 1;
            let info = registry
                .get(&s.test)
                .ok_or_else(|| LoadError::UnknownTest {
                    step: step_num,
                    test: s.test.clone(),
                    available: registry.names(),
                })?;
            let case = (info.factory)(&s.params).map_err(|source| LoadError::Params {
                step: step_num,
                test: s.test.clone(),
                source,
            })?;
            spec_admits(s.mode.streams, case.needs()).map_err(|need| LoadError::ModeMismatch {
                step: step_num,
                test: s.test.clone(),
                need,
                streams: s.mode.streams,
            })?;
            let timeout = s.timeout.unwrap_or_else(|| case.default_timeout());
            Ok(PlannedStep {
                origin: s.origin,
                label: s.label,
                test: s.test,
                mode: s.mode,
                case: Arc::from(case),
                timeout,
            })
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Abort {
    Interrupted,
    Hung { step: String },
}

impl fmt::Display for Abort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Abort::Interrupted => f.write_str("interrupted"),
            Abort::Hung { step } => write!(f, "hung test {step}"),
        }
    }
}

#[derive(Debug)]
pub struct RunOutcome {
    pub results: Vec<StepResult>,
    pub aborted: Option<Abort>,
}

pub fn run(
    steps: &[PlannedStep],
    host: &mut dyn ModeHost,
    options: &RunOptions,
    cancel: &Arc<AtomicBool>,
    progress: &mut dyn Write,
) -> RunOutcome {
    let options = Arc::new(options.clone());
    let mut results: Vec<StepResult> = Vec::new();
    let mut aborted: Option<Abort> = None;

    for step in steps {
        if let Some(abort) = &aborted {
            let index = results.len() + 1;
            results.push(StepResult {
                index,
                label: step.label.clone(),
                test: step.test.clone(),
                origin: step.origin.clone(),
                mode: "-".to_owned(),
                verdict: Verdict::Skipped,
                reason: Some(format!("not run: {abort}")),
                measurements: Vec::new(),
                notes: Vec::new(),
                duration_ms: 0,
            });
            continue;
        }

        if cancel.load(Ordering::SeqCst) {
            aborted = Some(Abort::Interrupted);
            let index = results.len() + 1;
            results.push(StepResult {
                index,
                label: step.label.clone(),
                test: step.test.clone(),
                origin: step.origin.clone(),
                mode: "-".to_owned(),
                verdict: Verdict::Skipped,
                reason: Some(format!("not run: {}", Abort::Interrupted)),
                measurements: Vec::new(),
                notes: Vec::new(),
                duration_ms: 0,
            });
            continue;
        }

        let modes = match host.expand(&step.mode) {
            Ok(modes) => modes,
            Err(e) => {
                let index = results.len() + 1;
                results.push(StepResult {
                    index,
                    label: step.label.clone(),
                    test: step.test.clone(),
                    origin: step.origin.clone(),
                    mode: "-".to_owned(),
                    verdict: Verdict::Error,
                    reason: Some(format!("mode: {e}")),
                    measurements: Vec::new(),
                    notes: Vec::new(),
                    duration_ms: 0,
                });
                continue;
            }
        };

        let needs = step.case.needs();
        let admissible: Vec<Mode> = modes
            .into_iter()
            .filter(|m| mode_admits(m, needs))
            .collect();
        if admissible.is_empty() {
            let index = results.len() + 1;
            results.push(StepResult {
                index,
                label: step.label.clone(),
                test: step.test.clone(),
                origin: step.origin.clone(),
                mode: "-".to_owned(),
                verdict: Verdict::Skipped,
                reason: Some(format!(
                    "no mode in this step satisfies the test's needs ({needs})"
                )),
                measurements: Vec::new(),
                notes: Vec::new(),
                duration_ms: 0,
            });
            continue;
        }

        for mode in admissible {
            let index = results.len() + 1;
            let _ = writeln!(progress, "> [{index}] {} [{mode}]", step.label);
            let start = Instant::now();
            let out = run_one(step, &mode, host, &options, cancel);
            let duration_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

            if out.hung {
                aborted = Some(Abort::Hung {
                    step: step.label.clone(),
                });
            }
            if cancel.load(Ordering::SeqCst) {
                aborted = Some(Abort::Interrupted);
            }

            let result = StepResult {
                index,
                label: step.label.clone(),
                test: step.test.clone(),
                origin: step.origin.clone(),
                mode: mode.to_string(),
                verdict: out.verdict,
                reason: out.reason,
                measurements: out.measurements,
                notes: out.notes,
                duration_ms,
            };
            let _ = writeln!(progress, "{}", format_result_line(&result));
            results.push(result);

            if aborted.is_some() {
                break;
            }
        }
    }

    RunOutcome { results, aborted }
}

fn format_result_line(r: &StepResult) -> String {
    let duration_s = r.duration_ms as f64 / 1e3;
    let mut line = format!(
        "[{}] {} [{}] {}",
        r.index,
        r.label,
        r.mode,
        r.verdict.upper()
    );
    if let Some(reason) = &r.reason {
        line.push_str(&format!(": {reason}"));
    }
    line.push_str(&format!(" ({duration_s:.1} s)"));
    line
}

type Returned = thread::Result<Result<TestOutput, TestError>>;

#[derive(Debug)]
enum Finished {
    Returned(Returned),
    ReturnedAfterTimeout(Returned),
    Hung,
    Abandoned,
    Vanished,
}

#[derive(Debug, Clone)]
struct Outcome {
    verdict: Verdict,
    reason: Option<String>,
    measurements: Vec<Measurement>,
    notes: Vec<String>,
    hung: bool,
}

impl Outcome {
    fn skipped(reason: impl Into<String>) -> Self {
        Self {
            verdict: Verdict::Skipped,
            reason: Some(reason.into()),
            measurements: Vec::new(),
            notes: Vec::new(),
            hung: false,
        }
    }

    fn error(reason: impl Into<String>) -> Self {
        Self {
            verdict: Verdict::Error,
            reason: Some(reason.into()),
            measurements: Vec::new(),
            notes: Vec::new(),
            hung: false,
        }
    }

    fn fail(reason: impl Into<String>) -> Self {
        Self {
            verdict: Verdict::Fail,
            reason: Some(reason.into()),
            measurements: Vec::new(),
            notes: Vec::new(),
            hung: false,
        }
    }
}

fn run_one(
    step: &PlannedStep,
    mode: &Mode,
    host: &mut dyn ModeHost,
    options: &Arc<RunOptions>,
    cancel: &Arc<AtomicBool>,
) -> Outcome {
    if step.case.needs().subject && !options.subject {
        return Outcome::skipped("needs subject (run without --no-subject)");
    }
    let active = match host.enter(mode) {
        Ok(active) => active,
        Err(e) => return Outcome::error(format!("mode setup failed: {e}")),
    };
    let step_cancel = Arc::new(AtomicBool::new(false));
    let ctx = TestCtx::new(
        Arc::clone(&active.session),
        mode.clone(),
        Arc::clone(cancel),
        Arc::clone(&step_cancel),
        step.timeout,
        Arc::clone(options),
    );
    let case = Arc::clone(&step.case);
    let (tx, rx) = crossbeam_channel::bounded::<Returned>(1);
    let spawned =
        eye_core::log::spawn_in_current_span(format!("eye-lab-{}", step.test), move || {
            let returned = panic::catch_unwind(AssertUnwindSafe(|| case.run(&ctx)));
            let _ = tx.send(returned);
        });
    let finished = match spawned {
        Ok(_) => wait(&rx, step.timeout, options.grace, &step_cancel, cancel),
        Err(e) => {
            drop(active.teardown);
            return Outcome::error(format!("spawning test thread: {e}"));
        }
    };
    drop(active.teardown);
    outcome(finished, step.timeout, options.grace)
}

/// Waits in 50 ms slices so a signal is noticed even while a test ignores `ctx.check()`.
fn wait(
    rx: &Receiver<Returned>,
    timeout: Duration,
    grace: Duration,
    step_cancel: &AtomicBool,
    cancel: &AtomicBool,
) -> Finished {
    let deadline = Instant::now() + timeout;
    let timed_out = loop {
        let slice = deadline
            .saturating_duration_since(Instant::now())
            .min(Duration::from_millis(50));
        match rx.recv_timeout(slice) {
            Ok(r) => return Finished::Returned(r),
            Err(RecvTimeoutError::Disconnected) => return Finished::Vanished,
            Err(RecvTimeoutError::Timeout) => {
                if cancel.load(Ordering::SeqCst) {
                    break false;
                }
                if Instant::now() >= deadline {
                    step_cancel.store(true, Ordering::SeqCst);
                    break true;
                }
            }
        }
    };
    match rx.recv_timeout(grace) {
        Ok(r) if timed_out => Finished::ReturnedAfterTimeout(r),
        Ok(r) => Finished::Returned(r),
        Err(RecvTimeoutError::Timeout) if timed_out => Finished::Hung,
        Err(RecvTimeoutError::Timeout) => Finished::Abandoned,
        Err(RecvTimeoutError::Disconnected) => Finished::Vanished,
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_owned()
    }
}

fn classify(r: Returned) -> Outcome {
    match r {
        Err(payload) => Outcome::error(format!("panicked: {}", panic_message(&*payload))),
        Ok(Ok(out)) => {
            if let Some(reason) = out.skipped {
                Outcome {
                    verdict: Verdict::Skipped,
                    reason: Some(reason),
                    measurements: out.measurements,
                    notes: out.notes,
                    hung: false,
                }
            } else if out.passed() {
                Outcome {
                    verdict: Verdict::Pass,
                    reason: None,
                    measurements: out.measurements,
                    notes: out.notes,
                    hung: false,
                }
            } else {
                Outcome {
                    verdict: Verdict::Fail,
                    reason: None,
                    measurements: out.measurements,
                    notes: out.notes,
                    hung: false,
                }
            }
        }
        Ok(Err(TestError::Cancelled)) => Outcome::error("interrupted"),
        Ok(Err(e @ (TestError::TimedOut(_) | TestError::Capture(_) | TestError::Emitter(_)))) => {
            Outcome::fail(e.to_string())
        }
        Ok(Err(e)) => Outcome::error(e.to_string()),
    }
}

fn outcome(finished: Finished, timeout: Duration, grace: Duration) -> Outcome {
    match finished {
        Finished::Returned(r) => classify(r),
        Finished::ReturnedAfterTimeout(r) => {
            let mut o = classify(r);
            if o.verdict != Verdict::Error {
                o.verdict = Verdict::Fail;
            }
            o.reason = Some(format!("timed out after {timeout:?}"));
            o
        }
        Finished::Hung => {
            let mut o = Outcome::error(format!(
                "hung: still running {grace:?} after its {timeout:?} timeout; run aborted"
            ));
            o.hung = true;
            o
        }
        Finished::Abandoned => Outcome::error(format!(
            "interrupted; test still running {grace:?} after the signal, abandoned"
        )),
        Finished::Vanished => Outcome::error("test thread exited without a result"),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, atomic::AtomicBool};

    use super::*;
    use crate::{
        case::{ParamError, TestRegistry},
        testkit::{FakeHost, FakeSession, NeedsCase, planned, planned_with_timeout},
    };

    fn build_needs_ir(_: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
        Ok(Box::new(NeedsCase(Needs {
            ir: true,
            ..Needs::default()
        })))
    }

    #[test]
    fn test_unknown_test_lists_available() {
        let registry = TestRegistry::builtin();
        let steps = vec![ResolvedStep {
            origin: "x".into(),
            label: "l".into(),
            test: "nope".into(),
            mode: ModeSpec::default(),
            params: toml::Table::new(),
            timeout: None,
        }];
        let err = plan(steps, &registry).unwrap_err();
        match err {
            LoadError::UnknownTest {
                step, available, ..
            } => {
                assert_eq!(step, 1);
                assert!(available.contains("selftest-check"));
            }
            other => panic!("expected UnknownTest, got {other:?}"),
        }
    }

    #[test]
    fn test_bad_params_name_the_step() {
        let registry = TestRegistry::builtin();
        let mut params = toml::Table::new();
        params.insert("valeu".into(), toml::Value::Float(1.0));
        let steps = vec![ResolvedStep {
            origin: "x".into(),
            label: "l".into(),
            test: "selftest-check".into(),
            mode: ModeSpec::default(),
            params,
            timeout: None,
        }];
        let err = plan(steps, &registry).unwrap_err();
        assert!(matches!(err, LoadError::Params { step: 1, .. }));
    }

    #[test]
    fn test_selftest_sleep_rejects_out_of_range_seconds() {
        let registry = TestRegistry::builtin();
        for seconds in [-1.0, 1e30] {
            let mut params = toml::Table::new();
            params.insert("seconds".into(), toml::Value::Float(seconds));
            let steps = vec![ResolvedStep {
                origin: "x".into(),
                label: "l".into(),
                test: "selftest-sleep".into(),
                mode: ModeSpec::default(),
                params,
                timeout: None,
            }];
            let err = plan(steps, &registry).unwrap_err();
            assert!(
                matches!(err, LoadError::Params { step: 1, .. }),
                "{seconds}"
            );
        }
    }

    #[test]
    fn test_mode_mismatch_is_rejected_before_running() {
        let mut registry = TestRegistry::empty();
        registry.register("needs-ir", "testkit needs ir", build_needs_ir);
        let steps = vec![ResolvedStep {
            origin: "x".into(),
            label: "l".into(),
            test: "needs-ir".into(),
            mode: ModeSpec {
                streams: StreamsSel::Rgb,
                ..ModeSpec::default()
            },
            params: toml::Table::new(),
            timeout: None,
        }];
        let err = plan(steps, &registry).unwrap_err();
        match err {
            LoadError::ModeMismatch { need, .. } => assert_eq!(need, "an IR stream"),
            other => panic!("expected ModeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_builtin_registry_builds_every_case_with_default_params() {
        let registry = TestRegistry::builtin();
        for info in registry.infos() {
            let case = (info.factory)(&toml::Table::new()).unwrap();
            assert_eq!(case.name(), info.name);
        }
    }

    fn fake_run_options() -> RunOptions {
        RunOptions {
            grace: Duration::from_millis(200),
            ..Default::default()
        }
    }

    #[test]
    fn test_pass_fail_skip_verdicts() {
        let mut host = FakeHost::new(Arc::new(FakeSession::empty()));
        let options = fake_run_options();
        let cancel = Arc::new(AtomicBool::new(false));
        let steps = vec![
            planned("selftest-check", "value = 42.0, min = 40.0", ""),
            planned("selftest-check", "value = 10.0, min = 40.0", ""),
            planned("selftest-skip", "", ""),
        ];
        let mut buf = Vec::new();
        let out = run(&steps, &mut host, &options, &cancel, &mut buf);
        let verdicts: Vec<_> = out.results.iter().map(|r| r.verdict).collect();
        assert_eq!(verdicts, [Verdict::Pass, Verdict::Fail, Verdict::Skipped]);
        assert_eq!(out.results[1].measurements[0].value, 10.0);
    }

    #[test]
    fn test_no_subject_skips_without_entering_mode() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut host = FakeHost {
            log: Arc::clone(&log),
            fail_enter: false,
            session: Arc::new(FakeSession::empty()),
        };
        let options = RunOptions {
            subject: false,
            ..fake_run_options()
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let steps = vec![planned("selftest-subject", "", "")];
        let mut buf = Vec::new();
        let out = run(&steps, &mut host, &options, &cancel, &mut buf);
        assert_eq!(out.results[0].verdict, Verdict::Skipped);
        assert_eq!(
            out.results[0].reason.as_deref(),
            Some("needs subject (run without --no-subject)")
        );
        assert!(log.lock().unwrap().is_empty());
    }

    #[test]
    fn test_panic_becomes_error_and_run_continues() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut host = FakeHost {
            log: Arc::clone(&log),
            fail_enter: false,
            session: Arc::new(FakeSession::empty()),
        };
        let options = fake_run_options();
        let cancel = Arc::new(AtomicBool::new(false));
        let steps = vec![
            planned("selftest-panic", "", ""),
            planned("selftest-check", "", ""),
        ];
        let mut buf = Vec::new();
        let out = run(&steps, &mut host, &options, &cancel, &mut buf);
        let verdicts: Vec<_> = out.results.iter().map(|r| r.verdict).collect();
        assert_eq!(verdicts, [Verdict::Error, Verdict::Pass]);
        assert!(
            out.results[0]
                .reason
                .as_deref()
                .unwrap()
                .starts_with("panicked: selftest panic")
        );
        let log = log.lock().unwrap();
        assert_eq!(log.iter().filter(|l| l.starts_with("enter")).count(), 2);
        assert_eq!(log.iter().filter(|l| l.starts_with("teardown")).count(), 2);
    }

    #[test]
    fn test_cooperative_timeout_is_fail_and_returns_promptly() {
        let mut host = FakeHost::new(Arc::new(FakeSession::empty()));
        let options = fake_run_options();
        let cancel = Arc::new(AtomicBool::new(false));
        let steps = vec![planned_with_timeout(
            "selftest-sleep",
            "seconds = 5.0",
            "",
            Duration::from_millis(100),
        )];
        let mut buf = Vec::new();
        let start = Instant::now();
        let out = run(&steps, &mut host, &options, &cancel, &mut buf);
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(out.results[0].verdict, Verdict::Fail);
        assert_eq!(
            out.results[0].reason.as_deref(),
            Some("timed out after 100ms")
        );
    }

    #[test]
    fn test_hung_test_aborts_run_and_tears_down_on_runner_thread() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut host = FakeHost {
            log: Arc::clone(&log),
            fail_enter: false,
            session: Arc::new(FakeSession::empty()),
        };
        let options = fake_run_options();
        let cancel = Arc::new(AtomicBool::new(false));
        let steps = vec![
            planned_with_timeout(
                "selftest-sleep",
                "seconds = 3.0, ignore_cancel = true",
                "",
                Duration::from_millis(100),
            ),
            planned("selftest-check", "", ""),
        ];
        let mut buf = Vec::new();
        let start = Instant::now();
        let out = run(&steps, &mut host, &options, &cancel, &mut buf);
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(
            out.aborted,
            Some(Abort::Hung {
                step: "selftest-sleep".to_owned()
            })
        );
        assert_eq!(out.results.len(), 2);
        assert_eq!(out.results[0].verdict, Verdict::Error);
        assert!(
            out.results[0]
                .reason
                .as_deref()
                .unwrap()
                .starts_with("hung:")
        );
        assert_eq!(out.results[1].verdict, Verdict::Skipped);
        assert_eq!(
            out.results[1].reason.as_deref(),
            Some("not run: hung test selftest-sleep")
        );
        let log = log.lock().unwrap();
        assert_eq!(log.last().map(String::as_str), Some("teardown none"));
        assert_eq!(log.iter().filter(|l| l.starts_with("enter")).count(), 1);
    }

    #[test]
    fn test_global_cancel_skips_everything() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut host = FakeHost {
            log: Arc::clone(&log),
            fail_enter: false,
            session: Arc::new(FakeSession::empty()),
        };
        let options = fake_run_options();
        let cancel = Arc::new(AtomicBool::new(true));
        let steps = vec![
            planned("selftest-check", "", ""),
            planned("selftest-check", "", ""),
        ];
        let mut buf = Vec::new();
        let out = run(&steps, &mut host, &options, &cancel, &mut buf);
        for r in &out.results {
            assert_eq!(r.verdict, Verdict::Skipped);
            assert_eq!(r.reason.as_deref(), Some("not run: interrupted"));
        }
        assert_eq!(out.aborted, Some(Abort::Interrupted));
        assert!(log.lock().unwrap().is_empty());
    }

    #[test]
    fn test_cancel_during_step_is_error_interrupted() {
        let mut host = FakeHost::new(Arc::new(FakeSession::empty()));
        let options = fake_run_options();
        let cancel = Arc::new(AtomicBool::new(false));
        let steps = vec![
            planned("selftest-sleep", "seconds = 5.0", ""),
            planned("selftest-check", "", ""),
        ];
        let cancel_clone = Arc::clone(&cancel);
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(100));
            cancel_clone.store(true, Ordering::SeqCst);
        });
        let mut buf = Vec::new();
        let start = Instant::now();
        let out = run(&steps, &mut host, &options, &cancel, &mut buf);
        handle.join().unwrap();
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(out.results[0].verdict, Verdict::Error);
        assert_eq!(out.results[0].reason.as_deref(), Some("interrupted"));
        assert_eq!(out.results[1].verdict, Verdict::Skipped);
        assert_eq!(
            out.results[1].reason.as_deref(),
            Some("not run: interrupted")
        );
        assert_eq!(out.aborted, Some(Abort::Interrupted));
    }

    #[test]
    fn test_cancel_during_uncooperative_step_returns_within_grace() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut host = FakeHost {
            log: Arc::clone(&log),
            fail_enter: false,
            session: Arc::new(FakeSession::empty()),
        };
        let options = fake_run_options();
        let cancel = Arc::new(AtomicBool::new(false));
        let steps = vec![
            planned_with_timeout(
                "selftest-sleep",
                "seconds = 3.0, ignore_cancel = true",
                "",
                Duration::from_secs(10),
            ),
            planned("selftest-check", "", ""),
        ];
        let cancel_clone = Arc::clone(&cancel);
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(100));
            cancel_clone.store(true, Ordering::SeqCst);
        });
        let mut buf = Vec::new();
        let start = Instant::now();
        let out = run(&steps, &mut host, &options, &cancel, &mut buf);
        handle.join().unwrap();
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(out.aborted, Some(Abort::Interrupted));
        assert_eq!(out.results[0].verdict, Verdict::Error);
        assert!(
            out.results[0]
                .reason
                .as_deref()
                .unwrap()
                .starts_with("interrupted")
        );
        assert_eq!(
            out.results[1].reason.as_deref(),
            Some("not run: interrupted")
        );
        assert_eq!(
            log.lock().unwrap().last().map(String::as_str),
            Some("teardown none")
        );
    }

    #[test]
    fn test_progress_writes_start_line_before_running_step() {
        let mut host = FakeHost::new(Arc::new(FakeSession::empty()));
        let options = fake_run_options();
        let cancel = Arc::new(AtomicBool::new(false));
        let steps = vec![planned("selftest-check", "", "")];
        let mut buf: Vec<u8> = Vec::new();
        run(&steps, &mut host, &options, &cancel, &mut buf);
        assert_eq!(
            String::from_utf8(buf).unwrap(),
            "> [1] selftest-check [none]\n[1] selftest-check [none] PASS (0.0 s)\n"
        );
    }

    #[test]
    fn test_mode_setup_failure_is_error() {
        let mut host = FakeHost {
            log: Arc::new(Mutex::new(Vec::new())),
            fail_enter: true,
            session: Arc::new(FakeSession::empty()),
        };
        let options = fake_run_options();
        let cancel = Arc::new(AtomicBool::new(false));
        let steps = vec![planned("selftest-check", "", "")];
        let mut buf = Vec::new();
        let out = run(&steps, &mut host, &options, &cancel, &mut buf);
        assert_eq!(out.results[0].verdict, Verdict::Error);
        assert!(
            out.results[0]
                .reason
                .as_deref()
                .unwrap()
                .starts_with("mode setup failed:")
        );
    }

    #[test]
    fn test_teardown_follows_each_step_in_order() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut host = FakeHost {
            log: Arc::clone(&log),
            fail_enter: false,
            session: Arc::new(FakeSession::empty()),
        };
        let options = fake_run_options();
        let cancel = Arc::new(AtomicBool::new(false));
        let steps = vec![
            planned("selftest-check", "", ""),
            planned("selftest-check", "", ""),
        ];
        let mut buf = Vec::new();
        run(&steps, &mut host, &options, &cancel, &mut buf);
        let log = log.lock().unwrap();
        assert_eq!(
            log.as_slice(),
            ["enter none", "teardown none", "enter none", "teardown none"]
        );
    }

    #[test]
    fn test_wildcard_mode_runs_only_admissible_modes() {
        let mut host = FakeHost::new(Arc::new(FakeSession::empty()));
        let options = fake_run_options();
        let cancel = Arc::new(AtomicBool::new(false));
        let steps = vec![planned("needs-ir", "", "streams = \"*\", emitter = \"*\"")];
        let mut buf = Vec::new();
        let out = run(&steps, &mut host, &options, &cancel, &mut buf);
        assert_eq!(out.results.len(), 4);
        let modes: Vec<_> = out.results.iter().map(|r| r.mode.clone()).collect();
        assert_eq!(
            modes,
            [
                "ir GREY 64x36@30, emitter off",
                "ir GREY 64x36@30, emitter on",
                "dual MJPG 64x36@30 + GREY 64x36@30, emitter off",
                "dual MJPG 64x36@30 + GREY 64x36@30, emitter on",
            ]
        );
    }

    #[test]
    fn test_runner_sets_emitter_during_step_and_restores_after() {
        use crate::testkit::{FakeOpener, SharedFakeXu, fake_live_host};

        let opener = Arc::new(FakeOpener::new(SharedFakeXu::with_mode(1)));
        let mut host = fake_live_host(Arc::clone(&opener));
        let options = fake_run_options();
        let cancel = Arc::new(AtomicBool::new(false));
        let steps = vec![planned(
            "read-emitter",
            "",
            "streams = \"ir\", emitter = \"on\"",
        )];
        let mut buf = Vec::new();
        let out = run(&steps, &mut host, &options, &cancel, &mut buf);
        assert_eq!(out.results[0].verdict, Verdict::Pass);
        assert_eq!(
            out.results[0].measurements[0].value,
            f64::from(eye_platform::emitter::MODE_ON_DEFAULT)
        );
        assert_eq!(opener.xu.mode(), 0x01);
    }

    #[test]
    fn test_runner_restores_emitter_after_panicking_step() {
        use crate::testkit::{FakeOpener, SharedFakeXu, fake_live_host};

        let opener = Arc::new(FakeOpener::new(SharedFakeXu::with_mode(1)));
        let mut host = fake_live_host(Arc::clone(&opener));
        let options = fake_run_options();
        let cancel = Arc::new(AtomicBool::new(false));
        let steps = vec![planned("selftest-panic", "", "emitter = \"on\"")];
        let mut buf = Vec::new();
        let out = run(&steps, &mut host, &options, &cancel, &mut buf);
        assert_eq!(out.results[0].verdict, Verdict::Error);
        assert_eq!(opener.xu.mode(), 0x01);
    }

    #[test]
    fn test_runner_restores_emitter_after_hung_step() {
        use crate::testkit::{FakeOpener, SharedFakeXu, fake_live_host, planned_with_timeout};

        let opener = Arc::new(FakeOpener::new(SharedFakeXu::with_mode(1)));
        let mut host = fake_live_host(Arc::clone(&opener));
        let options = fake_run_options();
        let cancel = Arc::new(AtomicBool::new(false));
        let steps = vec![planned_with_timeout(
            "selftest-sleep",
            "seconds = 3.0, ignore_cancel = true",
            "streams = \"ir\", emitter = \"on\"",
            Duration::from_millis(100),
        )];
        let mut buf = Vec::new();
        let start = Instant::now();
        let out = run(&steps, &mut host, &options, &cancel, &mut buf);
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(
            out.aborted,
            Some(Abort::Hung {
                step: "selftest-sleep".to_owned()
            })
        );
        assert_eq!(opener.xu.mode(), 0x01);
    }

    #[test]
    fn test_runner_emitter_each_runs_off_then_on() {
        use crate::testkit::{FakeOpener, SharedFakeXu, fake_live_host};

        let opener = Arc::new(FakeOpener::new(SharedFakeXu::with_mode(1)));
        let mut host = fake_live_host(Arc::clone(&opener));
        let options = fake_run_options();
        let cancel = Arc::new(AtomicBool::new(false));
        let steps = vec![planned(
            "read-emitter",
            "",
            "streams = \"ir\", emitter = \"*\"",
        )];
        let mut buf = Vec::new();
        let out = run(&steps, &mut host, &options, &cancel, &mut buf);
        assert_eq!(out.results.len(), 2);
        let values: Vec<f64> = out
            .results
            .iter()
            .map(|r| r.measurements[0].value)
            .collect();
        assert_eq!(
            values,
            [1.0, f64::from(eye_platform::emitter::MODE_ON_DEFAULT)]
        );
        assert_eq!(opener.xu.mode(), 0x01);
    }

    #[test]
    fn test_null_host_rejects_streams() {
        use crate::mode::{ModeError, NullHost};

        let ir_spec = ModeSpec {
            streams: StreamsSel::Ir,
            ..ModeSpec::default()
        };
        assert!(matches!(
            NullHost.expand(&ir_spec),
            Err(ModeError::Unavailable(_))
        ));
        assert_eq!(
            NullHost.expand(&ModeSpec::default()).unwrap(),
            vec![Mode::none()]
        );

        let mut host = NullHost;
        let options = fake_run_options();
        let cancel = Arc::new(AtomicBool::new(false));
        let steps = vec![planned("needs-ir", "", "streams = \"ir\"")];
        let mut buf = Vec::new();
        let out = run(&steps, &mut host, &options, &cancel, &mut buf);
        assert_eq!(out.results[0].verdict, Verdict::Error);
        assert!(
            out.results[0]
                .reason
                .as_deref()
                .unwrap()
                .starts_with("mode:")
        );
    }
}
