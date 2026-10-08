use std::{
    collections::BTreeMap,
    fmt,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use serde::Serialize;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Needs {
    pub rgb: bool,
    pub ir: bool,
    /// At least one stream of either role (cases that test every stream in the mode).
    pub any_stream: bool,
    /// The IR camera's emitter; implies an IR stream in the mode.
    pub emitter: bool,
    pub subject: bool,
}

/// The set flags joined with '+' in the order rgb, ir, stream (any_stream), emitter, subject; "-" when none.
impl fmt::Display for Needs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if self.rgb {
            parts.push("rgb");
        }
        if self.ir {
            parts.push("ir");
        }
        if self.any_stream {
            parts.push("stream");
        }
        if self.emitter {
            parts.push("emitter");
        }
        if self.subject {
            parts.push("subject");
        }
        if parts.is_empty() {
            f.write_str("-")
        } else {
            f.write_str(&parts.join("+"))
        }
    }
}

pub trait TestCase: Send + Sync + fmt::Debug {
    fn name(&self) -> &'static str;
    fn needs(&self) -> Needs;
    fn default_timeout(&self) -> Duration;
    fn run(&self, ctx: &TestCtx) -> Result<TestOutput, TestError>;
}

pub type Factory = fn(&toml::Table) -> Result<Box<dyn TestCase>, ParamError>;

#[derive(Debug, thiserror::Error)]
pub enum ParamError {
    #[error(transparent)]
    Toml(#[from] toml::de::Error),
    #[error("{0}")]
    Invalid(String),
}

pub fn parse_params<P: serde::de::DeserializeOwned>(params: &toml::Table) -> Result<P, ParamError> {
    Ok(params.clone().try_into()?)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Limit {
    AtLeast { min: f64 },
    AtMost { max: f64 },
    Within { min: f64, max: f64 },
}

impl Limit {
    /// Inclusive bounds; a non-finite value never passes.
    pub fn admits(self, v: f64) -> bool {
        v.is_finite()
            && match self {
                Limit::AtLeast { min } => v >= min,
                Limit::AtMost { max } => v <= max,
                Limit::Within { min, max } => v >= min && v <= max,
            }
    }
}

impl fmt::Display for Limit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Limit::AtLeast { min } => write!(f, ">= {min}"),
            Limit::AtMost { max } => write!(f, "<= {max}"),
            Limit::Within { min, max } => write!(f, "{min}..={max}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Measurement {
    pub name: String,
    pub value: f64,
    pub unit: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<Limit>,
    pub pass: bool,
}

impl Measurement {
    pub fn info(name: impl Into<String>, value: f64, unit: &'static str) -> Self {
        Self {
            name: name.into(),
            value,
            unit,
            limit: None,
            pass: true,
        }
    }

    pub fn checked(name: impl Into<String>, value: f64, unit: &'static str, limit: Limit) -> Self {
        Self {
            name: name.into(),
            value,
            unit,
            limit: Some(limit),
            pass: limit.admits(value),
        }
    }

    pub fn at_least(name: impl Into<String>, value: f64, unit: &'static str, min: f64) -> Self {
        Self::checked(name, value, unit, Limit::AtLeast { min })
    }

    pub fn at_most(name: impl Into<String>, value: f64, unit: &'static str, max: f64) -> Self {
        Self::checked(name, value, unit, Limit::AtMost { max })
    }

    pub fn within(
        name: impl Into<String>,
        value: f64,
        unit: &'static str,
        min: f64,
        max: f64,
    ) -> Self {
        Self::checked(name, value, unit, Limit::Within { min, max })
    }
}

impl fmt::Display for Measurement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = {:.3}", self.name, self.value)?;
        if !self.unit.is_empty() {
            write!(f, " {}", self.unit)?;
        }
        if let Some(limit) = self.limit {
            write!(f, " ({limit})")?;
        }
        if !self.pass {
            f.write_str(" FAIL")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TestOutput {
    pub measurements: Vec<Measurement>,
    pub notes: Vec<String>,
    pub skipped: Option<String>,
}

impl TestOutput {
    pub fn skip(reason: impl Into<String>) -> Self {
        Self {
            skipped: Some(reason.into()),
            ..Self::default()
        }
    }

    pub fn push(&mut self, m: Measurement) {
        self.measurements.push(m);
    }

    pub fn note(&mut self, s: impl Into<String>) {
        self.notes.push(s.into());
    }

    pub fn passed(&self) -> bool {
        self.measurements.iter().all(|m| m.pass)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TestError {
    #[error("interrupted")]
    Cancelled,
    #[error("timed out after {0:?}")]
    TimedOut(Duration),
    #[error(transparent)]
    Capture(#[from] eye_capture::CaptureError),
    #[error(transparent)]
    Emitter(#[from] eye_platform::EmitterError),
    #[error(transparent)]
    Mode(#[from] crate::mode::ModeError),
    #[error("{0}")]
    Other(String),
}

#[derive(Debug, Clone)]
pub struct RunOptions {
    pub subject: bool,
    /// How long a timed-out or interrupted test may take to stop before the runner gives up on it.
    pub grace: Duration,
    /// eye.toml for the pipeline cases (EYE-100); `None` = their embedded config.
    pub eye_config: Option<PathBuf>,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            subject: true,
            grace: Duration::from_secs(5),
            eye_config: None,
        }
    }
}

#[derive(Debug)]
pub struct TestCtx {
    session: Arc<dyn crate::mode::ModeSession>,
    mode: crate::mode::Mode,
    cancel: Arc<AtomicBool>,
    step_cancel: Arc<AtomicBool>,
    timeout: Duration,
    options: Arc<RunOptions>,
}

impl TestCtx {
    pub fn new(
        session: Arc<dyn crate::mode::ModeSession>,
        mode: crate::mode::Mode,
        cancel: Arc<AtomicBool>,
        step_cancel: Arc<AtomicBool>,
        timeout: Duration,
        options: Arc<RunOptions>,
    ) -> Self {
        Self {
            session,
            mode,
            cancel,
            step_cancel,
            timeout,
            options,
        }
    }

    pub fn session(&self) -> &dyn crate::mode::ModeSession {
        &*self.session
    }

    pub fn mode(&self) -> &crate::mode::Mode {
        &self.mode
    }

    pub fn options(&self) -> &RunOptions {
        &self.options
    }

    /// Call between frames. Global cancel (signal) wins over the step deadline.
    pub fn check(&self) -> Result<(), TestError> {
        if self.cancel.load(Ordering::SeqCst) {
            return Err(TestError::Cancelled);
        }
        if self.step_cancel.load(Ordering::SeqCst) {
            return Err(TestError::TimedOut(self.timeout));
        }
        Ok(())
    }

    /// Sleeps in slices of at most 50 ms, calling `check` before each slice.
    pub fn sleep(&self, total: Duration) -> Result<(), TestError> {
        let end = Instant::now() + total;
        loop {
            self.check()?;
            let now = Instant::now();
            if now >= end {
                return Ok(());
            }
            thread::sleep((end - now).min(Duration::from_millis(50)));
        }
    }

    /// One line on stderr for the person in front of the camera.
    pub fn instruct(&self, message: &str) {
        eprintln!("eye-lab: SUBJECT: {message}");
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TestInfo {
    pub name: &'static str,
    pub summary: &'static str,
    pub factory: Factory,
}

#[derive(Debug, Default)]
pub struct TestRegistry {
    tests: BTreeMap<&'static str, TestInfo>,
}

impl TestRegistry {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Every compiled-in case. Later lab cards each append one `register` call here.
    pub fn builtin() -> Self {
        let mut r = Self::empty();
        crate::selftest::register(&mut r);
        r
    }

    /// Panics on a duplicate name (programming error, caught by `test_builtin_registry_builds_every_case_with_default_params`).
    pub fn register(&mut self, name: &'static str, summary: &'static str, factory: Factory) {
        let previous = self.tests.insert(
            name,
            TestInfo {
                name,
                summary,
                factory,
            },
        );
        assert!(previous.is_none(), "duplicate test name {name}");
    }

    pub fn get(&self, name: &str) -> Option<&TestInfo> {
        self.tests.get(name)
    }

    pub fn infos(&self) -> impl Iterator<Item = &TestInfo> {
        self.tests.values()
    }

    pub fn names(&self) -> String {
        self.tests.keys().copied().collect::<Vec<_>>().join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_limits_are_inclusive_and_reject_nan() {
        let at_least = Limit::AtLeast { min: 40.0 };
        assert!(at_least.admits(40.0));
        assert!(!at_least.admits(39.999));
        assert!(!at_least.admits(f64::NAN));

        let within = Limit::Within { min: 1.0, max: 1.0 };
        assert!(within.admits(1.0));

        let at_most = Limit::AtMost { max: 0.0 };
        assert!(at_most.admits(0.0));
        assert!(!at_most.admits(1e-9));
    }

    #[test]
    fn test_ctx_check_prefers_global_cancel() {
        let ctx = crate::testkit::ctx(
            Arc::new(crate::testkit::FakeSession::empty()),
            crate::mode::Mode::none(),
        );
        ctx.cancel.store(true, Ordering::SeqCst);
        ctx.step_cancel.store(true, Ordering::SeqCst);
        assert!(matches!(ctx.check(), Err(TestError::Cancelled)));

        let ctx = crate::testkit::ctx(
            Arc::new(crate::testkit::FakeSession::empty()),
            crate::mode::Mode::none(),
        );
        ctx.step_cancel.store(true, Ordering::SeqCst);
        match ctx.check() {
            Err(TestError::TimedOut(d)) => assert_eq!(d, Duration::from_secs(30)),
            other => panic!("expected TimedOut, got {other:?}"),
        }
    }
}
