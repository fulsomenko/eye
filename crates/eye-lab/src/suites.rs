use crate::sequence::{LoadError, ResolvedStep, Sequence};

#[derive(Debug, Clone, Copy)]
pub struct SuiteDef {
    pub name: &'static str,
    pub summary: &'static str,
    pub parts: &'static [(&'static str, &'static str)],
}

const SELFTEST: &str = include_str!("suites/selftest.toml");

pub const BUILTIN: &[SuiteDef] = &[SuiteDef {
    name: "selftest",
    summary: "harness self-check, no hardware",
    parts: &[("selftest.toml", SELFTEST)],
}];

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
    use crate::{case::TestRegistry, runner};

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
}
