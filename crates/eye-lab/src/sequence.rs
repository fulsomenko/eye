use std::{
    fmt, fs,
    path::{Path, PathBuf},
    str::FromStr,
    time::Duration,
};

use serde::Deserialize;

/// Largest accepted `timeout_s` (one day); keeps `Instant + timeout` and `Duration::from_secs_f64` from panicking.
pub const MAX_TIMEOUT_S: f64 = 86_400.0;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sequence {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub defaults: StepDefaults,
    #[serde(rename = "step", default)]
    pub steps: Vec<StepSpec>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepDefaults {
    pub mode: Option<ModeSpec>,
    pub timeout_s: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepSpec {
    pub test: String,
    pub label: Option<String>,
    pub mode: Option<ModeSpec>,
    #[serde(default)]
    pub params: toml::Table,
    pub timeout_s: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModeSpec {
    #[serde(default)]
    pub streams: StreamsSel,
    #[serde(default)]
    pub emitter: EmitterSel,
    #[serde(default)]
    pub rgb: FormatSel,
    #[serde(default)]
    pub ir: FormatSel,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StreamsSel {
    #[default]
    None,
    Rgb,
    Ir,
    Dual,
    #[serde(rename = "*")]
    All,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmitterSel {
    #[default]
    Keep,
    On,
    Off,
    #[serde(rename = "*")]
    Each,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(try_from = "String")]
pub enum FormatSel {
    #[default]
    Default,
    All,
    Exact(StreamFormat),
}

impl TryFrom<String> for FormatSel {
    type Error = FormatParseError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        match s.as_str() {
            "default" => Ok(Self::Default),
            "*" => Ok(Self::All),
            other => other.parse().map(Self::Exact),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StreamFormat {
    pub fourcc: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid stream format {input:?}: expected e.g. \"MJPG 1280x720@30\"")]
pub struct FormatParseError {
    pub input: String,
}

impl FromStr for StreamFormat {
    type Err = FormatParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let err = || FormatParseError {
            input: s.to_owned(),
        };
        let (fourcc, rest) = s.split_once(' ').ok_or_else(err)?;
        let (size, fps) = rest.split_once('@').ok_or_else(err)?;
        let (w, h) = size.split_once('x').ok_or_else(err)?;
        if fourcc.len() != 4 || !fourcc.bytes().all(|b| b.is_ascii_graphic()) {
            return Err(err());
        }
        Ok(Self {
            fourcc: fourcc.to_owned(),
            width: w.parse().map_err(|_| err())?,
            height: h.parse().map_err(|_| err())?,
            fps: fps.parse().map_err(|_| err())?,
        })
    }
}

impl fmt::Display for StreamFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {}x{}@{}",
            self.fourcc, self.width, self.height, self.fps
        )
    }
}

/// A step with defaults applied; `origin` names the file it came from.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedStep {
    pub origin: String,
    pub label: String,
    pub test: String,
    pub mode: ModeSpec,
    pub params: toml::Table,
    pub timeout: Option<Duration>,
}

impl Sequence {
    pub fn from_toml_str(s: &str, origin: &str) -> Result<Self, LoadError> {
        toml::from_str(s).map_err(|source| LoadError::Parse {
            origin: origin.to_owned(),
            source,
        })
    }

    pub fn load(path: &Path) -> Result<Self, LoadError> {
        let origin = path.display().to_string();
        let contents = fs::read_to_string(path).map_err(|source| LoadError::Io {
            path: path.to_owned(),
            source,
        })?;
        Self::from_toml_str(&contents, &origin)
    }

    /// Applies `[defaults]`; `timeout_s` must be finite, > 0 and <= MAX_TIMEOUT_S; rejects an empty step list.
    pub fn resolve(&self, origin: &str) -> Result<Vec<ResolvedStep>, LoadError> {
        if self.steps.is_empty() {
            return Err(LoadError::Empty {
                origin: origin.to_owned(),
            });
        }
        self.steps
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let mode = s
                    .mode
                    .clone()
                    .or_else(|| self.defaults.mode.clone())
                    .unwrap_or_default();
                let timeout = match s.timeout_s.or(self.defaults.timeout_s) {
                    None => None,
                    Some(v) if v.is_finite() && v > 0.0 && v <= MAX_TIMEOUT_S => {
                        Some(Duration::from_secs_f64(v))
                    }
                    Some(value) => {
                        return Err(LoadError::Timeout {
                            origin: origin.to_owned(),
                            step: i + 1,
                            test: s.test.clone(),
                            value,
                        });
                    }
                };
                Ok(ResolvedStep {
                    origin: origin.to_owned(),
                    label: s.label.clone().unwrap_or_else(|| s.test.clone()),
                    test: s.test.clone(),
                    mode,
                    params: s.params.clone(),
                    timeout,
                })
            })
            .collect()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parsing {origin}: {source}")]
    Parse {
        origin: String,
        #[source]
        source: toml::de::Error,
    },
    #[error("{origin}: no [[step]] entries")]
    Empty { origin: String },
    #[error(
        "{origin} step {step} ({test}): timeout_s must be a positive number of seconds up to 86400, got {value}"
    )]
    Timeout {
        origin: String,
        step: usize,
        test: String,
        value: f64,
    },
    #[error("step {step}: unknown test {test:?}; available: {available}")]
    UnknownTest {
        step: usize,
        test: String,
        available: String,
    },
    #[error("step {step} ({test}): {source}")]
    Params {
        step: usize,
        test: String,
        #[source]
        source: crate::case::ParamError,
    },
    #[error("step {step} ({test}): needs {need}, but its mode has streams = {streams:?}")]
    ModeMismatch {
        step: usize,
        test: String,
        need: &'static str,
        streams: StreamsSel,
    },
    #[error("unknown suite {name:?}; available: {available}")]
    UnknownSuite { name: String, available: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_sequence_applies_defaults() {
        let toml = "name = \"x\"\n[defaults]\ntimeout_s = 3.0\nmode = { streams = \"ir\" }\n[[step]]\ntest = \"selftest-check\"\n";
        let seq = Sequence::from_toml_str(toml, "x.toml").unwrap();
        let steps = seq.resolve("x.toml").unwrap();
        assert_eq!(steps.len(), 1);
        let step = &steps[0];
        assert_eq!(step.label, "selftest-check");
        assert_eq!(step.mode.streams, StreamsSel::Ir);
        assert_eq!(step.timeout, Some(Duration::from_secs_f64(3.0)));
        assert_eq!(step.origin, "x.toml");
    }

    #[test]
    fn test_step_mode_overrides_default_mode() {
        let toml = "name = \"x\"\n[defaults]\nmode = { streams = \"ir\" }\n[[step]]\ntest = \"selftest-check\"\nmode = { streams = \"rgb\" }\n";
        let seq = Sequence::from_toml_str(toml, "x.toml").unwrap();
        let steps = seq.resolve("x.toml").unwrap();
        assert_eq!(steps[0].mode.streams, StreamsSel::Rgb);
        assert_eq!(steps[0].mode.emitter, EmitterSel::Keep);
    }

    #[test]
    fn test_unknown_key_in_step_is_rejected() {
        let toml = "name = \"x\"\n[[step]]\ntset = \"x\"\n";
        let err = Sequence::from_toml_str(toml, "x.toml").unwrap_err();
        assert!(matches!(err, LoadError::Parse { .. }));
    }

    #[test]
    fn test_stream_format_round_trips() {
        let format: StreamFormat = "MJPG 1280x720@30".parse().unwrap();
        assert_eq!(
            format,
            StreamFormat {
                fourcc: "MJPG".to_owned(),
                width: 1280,
                height: 720,
                fps: 30,
            }
        );
        assert_eq!(format.to_string(), "MJPG 1280x720@30");
    }

    #[test]
    fn test_bad_stream_formats_are_rejected() {
        for input in [
            "MJPG 1280x720",
            "MJPEG 1280x720@30",
            "MJPG 1280*720@30",
            "MJPG 1280x720@x",
        ] {
            assert!(input.parse::<StreamFormat>().is_err(), "{input}");
        }
    }

    #[test]
    fn test_wildcards_parse() {
        let toml = "name = \"x\"\n[[step]]\ntest = \"t\"\nmode = { streams = \"*\", emitter = \"*\", rgb = \"*\", ir = \"default\" }\n";
        let seq = Sequence::from_toml_str(toml, "x.toml").unwrap();
        let mode = seq.steps[0].mode.as_ref().unwrap();
        assert_eq!(mode.streams, StreamsSel::All);
        assert_eq!(mode.emitter, EmitterSel::Each);
        assert_eq!(mode.rgb, FormatSel::All);
        assert_eq!(mode.ir, FormatSel::Default);
    }

    #[test]
    fn test_out_of_range_timeout_is_rejected() {
        for value in [0.0, -1.0, 86_400.5, 1e30] {
            let toml = format!("name = \"x\"\n[[step]]\ntest = \"t\"\ntimeout_s = {value:e}\n");
            let seq = Sequence::from_toml_str(&toml, "x.toml").unwrap();
            let err = seq.resolve("x.toml").unwrap_err();
            assert!(
                matches!(err, LoadError::Timeout { step: 1, .. }),
                "{value}: {err:?}"
            );
        }
        let seq = Sequence::from_toml_str(
            "name = \"x\"\n[[step]]\ntest = \"t\"\ntimeout_s = 86400.0\n",
            "x.toml",
        )
        .unwrap();
        let steps = seq.resolve("x.toml").unwrap();
        assert_eq!(steps[0].timeout, Some(Duration::from_secs_f64(86_400.0)));
    }

    #[test]
    fn test_empty_sequence_is_error() {
        let seq = Sequence::from_toml_str("name = \"x\"\n", "x.toml").unwrap();
        let err = seq.resolve("x.toml").unwrap_err();
        assert!(matches!(err, LoadError::Empty { .. }));
    }
}
