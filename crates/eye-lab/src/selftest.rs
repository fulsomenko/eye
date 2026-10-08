use std::time::Duration;

use serde::Deserialize;

use crate::{
    case::{Measurement, ParamError, TestCase, TestCtx, TestError, TestOutput, parse_params},
    case::{Needs, TestRegistry},
    sequence::MAX_TIMEOUT_S,
};

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct CheckParams {
    value: f64,
    min: f64,
}

impl Default for CheckParams {
    fn default() -> Self {
        Self {
            value: 42.0,
            min: 40.0,
        }
    }
}

#[derive(Debug)]
struct Check {
    params: CheckParams,
}

impl TestCase for Check {
    fn name(&self) -> &'static str {
        "selftest-check"
    }

    fn needs(&self) -> Needs {
        Needs::default()
    }

    fn default_timeout(&self) -> Duration {
        Duration::from_secs(10)
    }

    fn run(&self, _ctx: &TestCtx) -> Result<TestOutput, TestError> {
        let mut out = TestOutput::default();
        out.push(Measurement::at_least(
            "value",
            self.params.value,
            "",
            self.params.min,
        ));
        Ok(out)
    }
}

fn build_check(params: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    Ok(Box::new(Check {
        params: parse_params(params)?,
    }))
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct SkipParams {
    reason: String,
}

impl Default for SkipParams {
    fn default() -> Self {
        Self {
            reason: "selftest".to_owned(),
        }
    }
}

#[derive(Debug)]
struct Skip {
    params: SkipParams,
}

impl TestCase for Skip {
    fn name(&self) -> &'static str {
        "selftest-skip"
    }

    fn needs(&self) -> Needs {
        Needs::default()
    }

    fn default_timeout(&self) -> Duration {
        Duration::from_secs(10)
    }

    fn run(&self, _ctx: &TestCtx) -> Result<TestOutput, TestError> {
        Ok(TestOutput::skip(self.params.reason.clone()))
    }
}

fn build_skip(params: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    Ok(Box::new(Skip {
        params: parse_params(params)?,
    }))
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct SleepParams {
    seconds: f64,
    ignore_cancel: bool,
}

impl Default for SleepParams {
    fn default() -> Self {
        Self {
            seconds: 1.0,
            ignore_cancel: false,
        }
    }
}

#[derive(Debug)]
struct Sleep {
    params: SleepParams,
}

impl TestCase for Sleep {
    fn name(&self) -> &'static str {
        "selftest-sleep"
    }

    fn needs(&self) -> Needs {
        Needs::default()
    }

    fn default_timeout(&self) -> Duration {
        Duration::from_secs_f64(self.params.seconds) + Duration::from_secs(5)
    }

    fn run(&self, ctx: &TestCtx) -> Result<TestOutput, TestError> {
        if self.params.ignore_cancel {
            std::thread::sleep(Duration::from_secs_f64(self.params.seconds));
        } else {
            ctx.sleep(Duration::from_secs_f64(self.params.seconds))?;
        }
        Ok(TestOutput::default())
    }
}

fn build_sleep(params: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    let params: SleepParams = parse_params(params)?;
    if !params.seconds.is_finite() || params.seconds < 0.0 || params.seconds > MAX_TIMEOUT_S {
        return Err(ParamError::Invalid(
            "seconds must be within 0..=86400".into(),
        ));
    }
    Ok(Box::new(Sleep { params }))
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PanicParams {
    message: String,
}

impl Default for PanicParams {
    fn default() -> Self {
        Self {
            message: "selftest panic".to_owned(),
        }
    }
}

#[derive(Debug)]
struct Panic {
    params: PanicParams,
}

impl TestCase for Panic {
    fn name(&self) -> &'static str {
        "selftest-panic"
    }

    fn needs(&self) -> Needs {
        Needs::default()
    }

    fn default_timeout(&self) -> Duration {
        Duration::from_secs(10)
    }

    fn run(&self, _ctx: &TestCtx) -> Result<TestOutput, TestError> {
        panic!("{}", self.params.message);
    }
}

fn build_panic(params: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    Ok(Box::new(Panic {
        params: parse_params(params)?,
    }))
}

#[derive(Debug)]
struct Subject;

impl TestCase for Subject {
    fn name(&self) -> &'static str {
        "selftest-subject"
    }

    fn needs(&self) -> Needs {
        Needs {
            subject: true,
            ..Needs::default()
        }
    }

    fn default_timeout(&self) -> Duration {
        Duration::from_secs(10)
    }

    fn run(&self, _ctx: &TestCtx) -> Result<TestOutput, TestError> {
        Ok(TestOutput::default())
    }
}

fn build_subject(_params: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    Ok(Box::new(Subject))
}

pub fn register(r: &mut TestRegistry) {
    r.register(
        "selftest-check",
        "harness self-check: value >= min",
        build_check,
    );
    r.register(
        "selftest-skip",
        "harness self-check: always skipped",
        build_skip,
    );
    r.register(
        "selftest-sleep",
        "harness self-check: sleeps (optionally ignoring cancel)",
        build_sleep,
    );
    r.register("selftest-panic", "harness self-check: panics", build_panic);
    r.register(
        "selftest-subject",
        "harness self-check: needs a subject",
        build_subject,
    );
}
