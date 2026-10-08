use crate::sequence::{LoadError, ResolvedStep, Sequence};

#[derive(Debug, Clone, Copy)]
pub struct SuiteDef {
    pub name: &'static str,
    pub summary: &'static str,
    pub parts: &'static [(&'static str, &'static str)],
}

const SELFTEST: &str = include_str!("suites/selftest.toml");
const SMOKE: &str = include_str!("suites/smoke.toml");
const HARDWARE: &str = include_str!("suites/hardware.toml");
const E1_REOPEN: &str = include_str!("suites/e1-reopen.toml");
const E1_AUTOSUSPEND: &str = include_str!("suites/e1-autosuspend.toml");
const E1_AFTER_EVENT: &str = include_str!("suites/e1-after-event.toml");
const E1_AFTER_RESET: &str = include_str!("suites/e1-after-reset.toml");
const E5: &str = include_str!("suites/e5.toml");
const E6: &str = include_str!("suites/e6.toml");

pub const BUILTIN: &[SuiteDef] = &[
    SuiteDef {
        name: "selftest",
        summary: "harness self-check, no hardware",
        parts: &[("selftest.toml", SELFTEST)],
    },
    SuiteDef {
        name: "smoke",
        summary: "both cameras open, frames well-formed; no emitter writes",
        parts: &[("smoke.toml", SMOKE)],
    },
    SuiteDef {
        name: "hardware",
        summary: "every camera mode, IR alternation, emitter round trip, dual timing (~2 min)",
        parts: &[("hardware.toml", HARDWARE)],
    },
    SuiteDef {
        name: "e1",
        summary: "E1 emitter persistence: reopen, RGB cycling, USB autosuspend (~7 min)",
        parts: &[
            ("e1-reopen.toml", E1_REOPEN),
            ("e1-autosuspend.toml", E1_AUTOSUSPEND),
        ],
    },
    SuiteDef {
        name: "e1-after-event",
        summary: "E1 human procedure: emitter still on after a persisting event",
        parts: &[("e1-after-event.toml", E1_AFTER_EVENT)],
    },
    SuiteDef {
        name: "e1-after-reset",
        summary: "E1 human procedure: emitter off after a resetting event",
        parts: &[("e1-after-reset.toml", E1_AFTER_RESET)],
    },
    SuiteDef {
        name: "e5",
        summary: "E5 RGB/IR timing against the measured R30 behaviour (~65 s)",
        parts: &[("e5.toml", E5)],
    },
    SuiteDef {
        name: "e6",
        summary: "E6 lit/dark tagging agreement, parity and margins (~85 s)",
        parts: &[("e6.toml", E6)],
    },
    SuiteDef {
        name: "regression",
        summary: "E1 reopen + E5 + E6 (~4 min)",
        parts: &[
            ("e1-reopen.toml", E1_REOPEN),
            ("e5.toml", E5),
            ("e6.toml", E6),
        ],
    },
    SuiteDef {
        name: "full",
        summary: "hardware + regression",
        parts: &[
            ("hardware.toml", HARDWARE),
            ("e1-reopen.toml", E1_REOPEN),
            ("e5.toml", E5),
            ("e6.toml", E6),
        ],
    },
];

pub fn find(name: &str) -> Option<&'static SuiteDef> {
    BUILTIN.iter().find(|s| s.name == name)
}

pub fn names() -> String {
    let mut names: Vec<&'static str> = BUILTIN.iter().map(|s| s.name).collect();
    names.sort_unstable();
    names.join(", ")
}

impl SuiteDef {
    pub fn load(&self) -> Result<Vec<ResolvedStep>, LoadError> {
        let mut steps = Vec::new();
        for (file, toml) in self.parts {
            let origin = format!("builtin:{file}");
            let seq = Sequence::from_toml_str(toml, &origin)?;
            steps.extend(seq.resolve(&origin)?);
        }
        Ok(steps)
    }

    pub fn sources(&self) -> Vec<String> {
        self.parts
            .iter()
            .map(|(file, _)| format!("builtin:{file}"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{case::TestRegistry, runner, sequence::Sequence};

    #[test]
    fn test_selftest_suite_loads_with_builtin_origins() {
        let def = find("selftest").expect("selftest suite registered");
        let steps = def.load().unwrap();
        assert_eq!(steps.len(), 3);
        for step in &steps {
            assert_eq!(step.origin, "builtin:selftest.toml");
        }
        assert_eq!(def.sources(), vec!["builtin:selftest.toml".to_owned()]);
    }

    #[test]
    fn test_every_builtin_suite_loads_and_plans() {
        let registry = TestRegistry::builtin();
        for def in BUILTIN {
            let steps = def.load().unwrap_or_else(|e| panic!("{}: {e}", def.name));
            runner::plan(steps, &registry).unwrap_or_else(|e| panic!("{}: {e}", def.name));
        }
    }

    #[test]
    fn test_suite_names_are_unique_and_full_is_hardware_plus_regression() {
        let mut names: Vec<&str> = BUILTIN.iter().map(|s| s.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), BUILTIN.len());

        let hardware = find("hardware").unwrap();
        let regression = find("regression").unwrap();
        let full = find("full").unwrap();
        let expected: Vec<String> = hardware
            .sources()
            .into_iter()
            .chain(regression.sources())
            .collect();
        assert_eq!(full.sources(), expected);

        for def in [regression, full] {
            assert!(
                !def.parts
                    .iter()
                    .any(|(file, _)| *file == "e1-autosuspend.toml"),
                "{} must not include e1-autosuspend.toml",
                def.name
            );
        }
    }

    #[test]
    fn test_suite_numbers_match_spike_rules() {
        fn step_params(toml_str: &str) -> Vec<toml::Table> {
            let seq = Sequence::from_toml_str(toml_str, "test").unwrap();
            seq.steps.into_iter().map(|s| s.params).collect()
        }

        let e5 = step_params(E5);
        let want: toml::Table = toml::from_str(
            r#"
            frames = 450
            expected_median_ms = 3.0
            median_tolerance_ms = 3.0
            max_spread_ms = 10.0
            max_drift_ms_per_min = 1.0
            min_pair_rate = 0.95
            min_dark_fraction = 0.95
            expected_rgb_fps = 7.5
            expected_ir_fps = 15.0
            fps_tolerance = 1.0
            "#,
        )
        .unwrap();
        assert_eq!(e5[0], want);

        let e6 = step_params(E6);
        let want: toml::Table = toml::from_str(
            r#"
            frames = 1200
            lit_threshold = 20.0
            min_agreement = 0.999
            max_dark_p95 = 15.0
            min_lit_p5 = 25.0
            require_metadata = true
            "#,
        )
        .unwrap();
        assert_eq!(e6[0], want);

        let e1_reopen = step_params(E1_REOPEN);
        assert_eq!(
            e1_reopen[0].get("expect_reopen").unwrap().as_str(),
            Some("persist")
        );
        assert_eq!(
            e1_reopen[1].get("expect_reopen").unwrap().as_str(),
            Some("report")
        );

        let e1_autosuspend = step_params(E1_AUTOSUSPEND);
        assert_eq!(
            e1_autosuspend[0]
                .get("expect_autosuspend")
                .unwrap()
                .as_str(),
            Some("report")
        );
    }

    #[test]
    fn test_hardware_suite_expands_to_34_runs_on_dev_fixture() {
        use std::sync::{Arc, atomic::AtomicBool};

        use crate::{
            case::RunOptions,
            testkit::{FakeOpener, SharedFakeXu, fake_live_host},
        };

        let def = find("hardware").unwrap();
        let steps = def.load().unwrap();
        let registry = TestRegistry::builtin();
        let planned = runner::plan(steps, &registry).unwrap();

        let opener = Arc::new(FakeOpener::new(SharedFakeXu::with_mode(1)));
        let mut host = fake_live_host(opener);
        let options = RunOptions {
            subject: false,
            ..RunOptions::default()
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let mut sink = Vec::new();
        let outcome = runner::run(&planned, &mut host, &options, &cancel, &mut sink);

        let no_mode_skips = outcome
            .results
            .iter()
            .filter(|r| {
                r.reason
                    .as_deref()
                    .is_some_and(|s| s.contains("no mode in this step satisfies"))
            })
            .count();
        assert_eq!(outcome.results.len() - no_mode_skips, 34);
    }
}
