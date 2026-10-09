use std::time::{Duration, Instant};

use eye_platform::emitter::MODE_ON_DEFAULT;

use crate::{
    case::{
        Measurement, Needs, ParamError, TestCase, TestCtx, TestError, TestOutput, parse_params,
    },
    hw::{Tagging, grab, ir::tagged_phase},
    mode::{EmitterSetting, Role},
    regress::{Expect, require_emitter},
};

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PersistenceParams {
    pub reopens: usize,
    pub frames: usize,
    pub warmup: usize,
    pub rgb_cycle: bool,
    pub rgb_frames: usize,
    pub expect_reopen: Expect,
    pub autosuspend_wait_s: f64,
    pub expect_autosuspend: Expect,
    pub min_alternation: f64,
}

impl Default for PersistenceParams {
    fn default() -> Self {
        Self {
            reopens: 10,
            frames: 30,
            warmup: 8,
            rgb_cycle: false,
            rgb_frames: 5,
            expect_reopen: Expect::Persist,
            autosuspend_wait_s: 0.0,
            expect_autosuspend: Expect::Persist,
            min_alternation: 0.9,
        }
    }
}

/// One trial persisted iff the emitter byte still reads MODE_ON_DEFAULT AND the stream visibly alternates.
fn trial(ctx: &TestCtx, p: &PersistenceParams) -> Result<(bool, u8, f64), TestError> {
    let (a, _) = tagged_phase(ctx, p.warmup, p.frames, Tagging::Brightness)?;
    let byte = ctx.session().emitter()?.read_mode()?;
    Ok((
        byte == MODE_ON_DEFAULT && a.alternating >= p.min_alternation,
        byte,
        a.alternating,
    ))
}

/// `None` for `Report`: the count is reported without a limit.
fn expected_count(expect: Expect, trials: usize) -> Option<f64> {
    match expect {
        Expect::Persist => Some(trials as f64),
        Expect::Reset => Some(0.0),
        Expect::Report => None,
    }
}

fn push_persisted(out: &mut TestOutput, name: &str, count: f64, expect: Expect, trials: usize) {
    match expected_count(expect, trials) {
        Some(e) => out.push(Measurement::within(name, count, "", e, e)),
        None => {
            out.push(Measurement::info(name, count, ""));
            out.note("E1 row not pinned yet: reporting only (EYE-102)");
        }
    }
}

#[derive(Debug)]
struct EmitterPersistence {
    params: PersistenceParams,
}

impl TestCase for EmitterPersistence {
    fn name(&self) -> &'static str {
        "emitter_persistence"
    }

    fn needs(&self) -> Needs {
        Needs {
            ir: true,
            emitter: true,
            ..Needs::default()
        }
    }

    fn default_timeout(&self) -> Duration {
        let p = &self.params;
        Duration::from_secs_f64(
            30.0 + p.reopens as f64 * (2.0 + (p.warmup + p.frames + p.rgb_frames) as f64 / 15.0)
                + p.autosuspend_wait_s,
        )
    }

    fn run(&self, ctx: &TestCtx) -> Result<TestOutput, TestError> {
        require_emitter(ctx, EmitterSetting::On, "emitter_persistence", "on")?;
        let p = &self.params;
        let mut out = TestOutput::default();
        ctx.instruct("keep a face or a hand 30-50 cm in front of the camera for the IR checks");

        let mut persisted = 0usize;
        let mut rgb_skip_noted = false;
        for k in 0..p.reopens {
            if p.rgb_cycle {
                if ctx.mode().rgb.is_some() {
                    let mut rgb = ctx.session().open(Role::Rgb)?;
                    grab(ctx, rgb.as_mut(), p.rgb_frames)?;
                    drop(rgb);
                } else if !rgb_skip_noted {
                    out.note("rgb_cycle skipped: mode has no RGB stream");
                    rgb_skip_noted = true;
                }
            }
            let (ok, byte, alt) = trial(ctx, p)?;
            if ok {
                persisted += 1;
            } else {
                out.note(format!(
                    "reopen {k}: byte {byte:#04x}, alternation {alt:.2}"
                ));
            }
        }
        push_persisted(
            &mut out,
            "persisted_reopens",
            persisted as f64,
            p.expect_reopen,
            p.reopens,
        );
        out.push(Measurement::info("reopens", p.reopens as f64, ""));

        if p.autosuspend_wait_s > 0.0 {
            let dev = ctx.session().device(Role::Ir).ok_or_else(|| {
                TestError::Other("ir: no USB identity for the autosuspend check".into())
            })?;
            let status_path = dev
                .usb
                .map(|usb| usb.sysfs_device.join("power/runtime_status"))
                .ok_or_else(|| {
                    TestError::Other(format!(
                        "{}: no USB identity for the autosuspend check",
                        dev.node.display()
                    ))
                })?;

            let deadline = Instant::now() + Duration::from_secs_f64(p.autosuspend_wait_s);
            let mut reached = false;
            loop {
                let status = std::fs::read_to_string(&status_path).unwrap_or_default();
                if status.trim() == "suspended" {
                    reached = true;
                    break;
                }
                if Instant::now() >= deadline {
                    break;
                }
                ctx.sleep(Duration::from_secs(1))?;
            }
            out.push(Measurement::at_least(
                "autosuspend_reached",
                if reached { 1.0 } else { 0.0 },
                "",
                1.0,
            ));
            if !reached {
                out.note(
                    "device never reached runtime_status=suspended: check power/control = auto and other camera users (fuser -v /dev/video*)",
                );
                return Ok(out);
            }

            let (ok, byte, alt) = trial(ctx, p)?;
            push_persisted(
                &mut out,
                "persisted_after_autosuspend",
                if ok { 1.0 } else { 0.0 },
                p.expect_autosuspend,
                1,
            );
            out.note(format!(
                "autosuspend: byte {byte:#04x}, alternation {alt:.2}"
            ));
        }
        Ok(out)
    }
}

pub fn build(params: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    let p: PersistenceParams = parse_params(params)?;
    if p.reopens == 0 {
        return Err(ParamError::Invalid("reopens must be >= 1".into()));
    }
    if p.frames == 0 {
        return Err(ParamError::Invalid("frames must be >= 1".into()));
    }
    if !p.autosuspend_wait_s.is_finite() || p.autosuspend_wait_s < 0.0 {
        return Err(ParamError::Invalid(
            "autosuspend_wait_s must be finite and >= 0".into(),
        ));
    }
    Ok(Box::new(EmitterPersistence { params: p }))
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::{Arc, Mutex, atomic::AtomicU8},
        thread,
    };

    use super::*;
    use crate::testkit::{self, EmitterAwareSource, FakeEmitter, ResetOnStreamOn};

    fn table(entries: &[(&str, toml::Value)]) -> toml::Table {
        let mut t = toml::Table::new();
        for (k, v) in entries {
            t.insert((*k).to_owned(), v.clone());
        }
        t
    }

    fn ir_sources(
        n: usize,
        mode: Arc<AtomicU8>,
        frames_each: usize,
    ) -> Vec<Box<dyn eye_capture::FrameSource>> {
        (0..n)
            .map(|_| -> Box<dyn eye_capture::FrameSource> {
                Box::new(EmitterAwareSource {
                    mode: Arc::clone(&mode),
                    seq: 0,
                    remaining: frames_each,
                })
            })
            .collect()
    }

    fn session_with_ir(
        mode: Arc<AtomicU8>,
        sources: Vec<Box<dyn eye_capture::FrameSource>>,
    ) -> Arc<testkit::FakeSession> {
        let emitter = FakeEmitter {
            mode: Arc::clone(&mode),
            ignore_writes: false,
            writes: Arc::new(Mutex::new(Vec::new())),
        };
        let mut session = testkit::FakeSession {
            emitter: Some(emitter),
            ..testkit::FakeSession::empty()
        };
        for s in sources {
            session = session.with_source(Role::Ir, s);
        }
        Arc::new(session)
    }

    #[test]
    fn test_persistence_passes_when_byte_survives_reopens() {
        let mode = Arc::new(AtomicU8::new(MODE_ON_DEFAULT));
        let session = session_with_ir(Arc::clone(&mode), ir_sources(3, mode, 38));
        let ctx = testkit::ctx(session, testkit::mode_ir(EmitterSetting::On));
        let case = build(&table(&[("reopens", toml::Value::Integer(3))])).unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(out.passed());
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "persisted_reopens")
                .unwrap()
                .value,
            3.0
        );
    }

    #[test]
    fn test_persistence_detects_reset_on_reopen() {
        let mode = Arc::new(AtomicU8::new(MODE_ON_DEFAULT));
        let sources: Vec<Box<dyn eye_capture::FrameSource>> = (0..3)
            .map(|_| -> Box<dyn eye_capture::FrameSource> {
                Box::new(ResetOnStreamOn {
                    inner: EmitterAwareSource {
                        mode: Arc::clone(&mode),
                        seq: 0,
                        remaining: 38,
                    },
                    mode: Arc::clone(&mode),
                    started: false,
                })
            })
            .collect();
        let session = session_with_ir(Arc::clone(&mode), sources);
        let session_dyn: Arc<dyn crate::mode::ModeSession> = Arc::clone(&session) as _;
        let ctx = testkit::ctx(session_dyn, testkit::mode_ir(EmitterSetting::On));
        let case = build(&table(&[("reopens", toml::Value::Integer(3))])).unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(!out.passed());
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "persisted_reopens")
                .unwrap()
                .value,
            0.0
        );
        assert!(out.notes.iter().any(|n| n.contains("byte 0x01")));
        assert!(
            session
                .emitter
                .as_ref()
                .unwrap()
                .writes
                .lock()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn test_persistence_expect_reset_passes_on_resetting_device() {
        let mode = Arc::new(AtomicU8::new(MODE_ON_DEFAULT));
        let sources: Vec<Box<dyn eye_capture::FrameSource>> = (0..3)
            .map(|_| -> Box<dyn eye_capture::FrameSource> {
                Box::new(ResetOnStreamOn {
                    inner: EmitterAwareSource {
                        mode: Arc::clone(&mode),
                        seq: 0,
                        remaining: 38,
                    },
                    mode: Arc::clone(&mode),
                    started: false,
                })
            })
            .collect();
        let session = session_with_ir(Arc::clone(&mode), sources);
        let ctx = testkit::ctx(session, testkit::mode_ir(EmitterSetting::On));
        let case = build(&table(&[
            ("reopens", toml::Value::Integer(3)),
            ("expect_reopen", toml::Value::String("reset".into())),
        ]))
        .unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(out.passed());
    }

    #[test]
    fn test_persistence_report_only_never_fails() {
        let mode = Arc::new(AtomicU8::new(MODE_ON_DEFAULT));
        let sources: Vec<Box<dyn eye_capture::FrameSource>> = (0..3)
            .map(|_| -> Box<dyn eye_capture::FrameSource> {
                Box::new(ResetOnStreamOn {
                    inner: EmitterAwareSource {
                        mode: Arc::clone(&mode),
                        seq: 0,
                        remaining: 38,
                    },
                    mode: Arc::clone(&mode),
                    started: false,
                })
            })
            .collect();
        let session = session_with_ir(Arc::clone(&mode), sources);
        let ctx = testkit::ctx(session, testkit::mode_ir(EmitterSetting::On));
        let case = build(&table(&[
            ("reopens", toml::Value::Integer(3)),
            ("expect_reopen", toml::Value::String("report".into())),
        ]))
        .unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(out.passed());
        let m = out
            .measurements
            .iter()
            .find(|m| m.name == "persisted_reopens")
            .unwrap();
        assert!(m.limit.is_none());
        assert_eq!(m.value, 0.0);
        assert!(out.notes.iter().any(|n| n.contains("not pinned")));
    }

    #[test]
    fn test_persistence_requires_emitter_on_mode() {
        let session = Arc::new(testkit::FakeSession::empty());
        let ctx = testkit::ctx(session, testkit::mode_ir(EmitterSetting::Keep));
        let case = build(&toml::Table::new()).unwrap();
        let err = case.run(&ctx).unwrap_err();
        match err {
            TestError::Other(msg) => assert!(msg.contains("emitter = \"on\"")),
            other => panic!("expected TestError::Other, got {other:?}"),
        }
    }

    #[test]
    fn test_persistence_rgb_cycle_opens_rgb_between_reopens() {
        let mode = Arc::new(AtomicU8::new(MODE_ON_DEFAULT));
        let emitter = FakeEmitter {
            mode: Arc::clone(&mode),
            ignore_writes: false,
            writes: Arc::new(Mutex::new(Vec::new())),
        };
        let mut session = testkit::FakeSession {
            emitter: Some(emitter),
            ..testkit::FakeSession::empty()
        };
        for _ in 0..3 {
            session = session.with_source(
                Role::Ir,
                Box::new(EmitterAwareSource {
                    mode: Arc::clone(&mode),
                    seq: 0,
                    remaining: 38,
                }),
            );
        }
        for _ in 0..3 {
            session = session.with_source(
                Role::Rgb,
                testkit::boxed(testkit::synth_mjpeg("rgb", 5, 0, 0, 33_000_000)),
            );
        }
        let session = Arc::new(session);
        let session_dyn: Arc<dyn crate::mode::ModeSession> = Arc::clone(&session) as _;
        let ctx = testkit::ctx(session_dyn, testkit::mode_dual(EmitterSetting::On));
        let case = build(&table(&[
            ("reopens", toml::Value::Integer(3)),
            ("rgb_cycle", toml::Value::Boolean(true)),
        ]))
        .unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(out.passed());
        assert!(
            session
                .sources
                .lock()
                .unwrap()
                .get(&Role::Rgb)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn test_persistence_rgb_cycle_without_rgb_notes_skip() {
        let mode = Arc::new(AtomicU8::new(MODE_ON_DEFAULT));
        let session = session_with_ir(Arc::clone(&mode), ir_sources(3, mode, 38));
        let ctx = testkit::ctx(session, testkit::mode_ir(EmitterSetting::On));
        let case = build(&table(&[
            ("reopens", toml::Value::Integer(3)),
            ("rgb_cycle", toml::Value::Boolean(true)),
        ]))
        .unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(out.passed());
        let count = out
            .notes
            .iter()
            .filter(|n| n.contains("rgb_cycle skipped: mode has no RGB stream"))
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_autosuspend_waits_for_suspended_status() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("idVendor"), "0c45\n").unwrap();
        fs::create_dir_all(dir.path().join("power")).unwrap();
        let status_path = dir.path().join("power/runtime_status");
        fs::write(&status_path, "active\n").unwrap();
        let handle = {
            let status_path = status_path.clone();
            let parent = tracing::Span::current();
            thread::spawn(move || {
                let _enter = parent.entered();
                thread::sleep(Duration::from_millis(300));
                fs::write(&status_path, "suspended\n").unwrap();
            })
        };

        let mode = Arc::new(AtomicU8::new(MODE_ON_DEFAULT));
        let session = testkit::session_with_usb_dir(
            dir.path(),
            FakeEmitter {
                mode: Arc::clone(&mode),
                ignore_writes: false,
                writes: Arc::new(Mutex::new(Vec::new())),
            },
            ir_sources(2, mode, 38),
        );
        let ctx = testkit::ctx(Arc::new(session), testkit::mode_ir(EmitterSetting::On));
        let case = build(&table(&[
            ("reopens", toml::Value::Integer(1)),
            ("autosuspend_wait_s", toml::Value::Float(3.0)),
        ]))
        .unwrap();
        let out = case.run(&ctx).unwrap();
        handle.join().unwrap();
        assert!(out.passed());
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "autosuspend_reached")
                .unwrap()
                .value,
            1.0
        );
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "persisted_after_autosuspend")
                .unwrap()
                .value,
            1.0
        );
    }

    #[test]
    fn test_autosuspend_never_reached_fails_with_hint() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("idVendor"), "0c45\n").unwrap();
        fs::create_dir_all(dir.path().join("power")).unwrap();
        fs::write(dir.path().join("power/runtime_status"), "active\n").unwrap();

        let mode = Arc::new(AtomicU8::new(MODE_ON_DEFAULT));
        let session = testkit::session_with_usb_dir(
            dir.path(),
            FakeEmitter {
                mode: Arc::clone(&mode),
                ignore_writes: false,
                writes: Arc::new(Mutex::new(Vec::new())),
            },
            ir_sources(1, mode, 38),
        );
        let ctx = testkit::ctx(Arc::new(session), testkit::mode_ir(EmitterSetting::On));
        let case = build(&table(&[
            ("reopens", toml::Value::Integer(1)),
            ("autosuspend_wait_s", toml::Value::Float(1.5)),
        ]))
        .unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(!out.passed());
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "autosuspend_reached")
                .unwrap()
                .value,
            0.0
        );
        assert!(out.notes.iter().any(|n| n.contains("fuser -v /dev/video*")));
        assert!(
            !out.measurements
                .iter()
                .any(|m| m.name == "persisted_after_autosuspend")
        );
    }
}
