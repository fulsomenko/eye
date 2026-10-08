use std::time::Duration;

use eye_platform::emitter::MODE_ON_DEFAULT;

use crate::{
    case::{
        Measurement, Needs, ParamError, TestCase, TestCtx, TestError, TestOutput, parse_params,
    },
    mode::EmitterSetting,
    regress::require_emitter,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Expect {
    #[default]
    On,
    Off,
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EmitterModeParams {
    pub expect: Expect,
}

#[derive(Debug)]
struct EmitterMode {
    params: EmitterModeParams,
}

impl TestCase for EmitterMode {
    fn name(&self) -> &'static str {
        "emitter_mode"
    }

    fn needs(&self) -> Needs {
        Needs {
            emitter: true,
            ..Needs::default()
        }
    }

    fn default_timeout(&self) -> Duration {
        Duration::from_secs(5)
    }

    fn run(&self, ctx: &TestCtx) -> Result<TestOutput, TestError> {
        require_emitter(ctx, EmitterSetting::Keep, "emitter_mode", "keep")?;
        let mut out = TestOutput::default();
        let byte = ctx.session().emitter()?.read_mode()?;
        out.note(format!("emitter byte {byte:#04x}"));
        let value = f64::from(byte);
        let on = f64::from(MODE_ON_DEFAULT);
        out.push(match self.params.expect {
            Expect::On => Measurement::at_least("mode", value, "", on),
            Expect::Off => Measurement::within("mode", value, "", 1.0, 1.0),
        });
        Ok(out)
    }
}

pub fn build(params: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    let params: EmitterModeParams = parse_params(params)?;
    Ok(Box::new(EmitterMode { params }))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, atomic::AtomicU8};

    use super::*;
    use crate::testkit::{self, FakeEmitter};

    fn ctx_with_byte(byte: u8, mode: EmitterSetting) -> (TestCtx, Arc<Mutex<Vec<u8>>>) {
        let writes = Arc::new(Mutex::new(Vec::new()));
        let emitter = FakeEmitter {
            mode: Arc::new(AtomicU8::new(byte)),
            ignore_writes: false,
            writes: Arc::clone(&writes),
        };
        let session = Arc::new(testkit::FakeSession {
            emitter: Some(emitter),
            ..testkit::FakeSession::empty()
        });
        (testkit::ctx(session, testkit::mode_ir(mode)), writes)
    }

    #[test]
    fn test_emitter_mode_on_passes_for_02_and_03() {
        for byte in [0x02u8, 0x03u8] {
            let (ctx, writes) = ctx_with_byte(byte, EmitterSetting::Keep);
            let case = build(&toml::Table::new()).unwrap();
            let out = case.run(&ctx).unwrap();
            assert!(out.passed(), "byte {byte:#04x}");
            assert!(writes.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn test_emitter_mode_on_fails_for_01() {
        let (ctx, _writes) = ctx_with_byte(0x01, EmitterSetting::Keep);
        let case = build(&toml::Table::new()).unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(!out.passed());
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "mode")
                .unwrap()
                .value,
            1.0
        );
    }

    #[test]
    fn test_emitter_mode_off_expectation() {
        let (ctx, _writes) = ctx_with_byte(0x01, EmitterSetting::Keep);
        let mut table = toml::Table::new();
        table.insert("expect".into(), toml::Value::String("off".into()));
        let case = build(&table).unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(out.passed());

        let (ctx, _writes) = ctx_with_byte(0x03, EmitterSetting::Keep);
        let case = build(&table).unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(!out.passed());
    }

    #[test]
    fn test_emitter_mode_requires_keep() {
        let (ctx, _writes) = ctx_with_byte(0x02, EmitterSetting::On);
        let case = build(&toml::Table::new()).unwrap();
        let err = case.run(&ctx).unwrap_err();
        match err {
            TestError::Other(msg) => assert!(msg.contains("emitter = \"keep\"")),
            other => panic!("expected TestError::Other, got {other:?}"),
        }
    }
}
