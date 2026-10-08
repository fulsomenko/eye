use serde::Deserialize;

/// `name = "kind"` shorthand, or a table with a string `kind` and the stage's options.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(try_from = "toml::Value")]
pub struct StageSection {
    pub kind: String,
    pub options: toml::Table,
}

impl StageSection {
    pub fn none() -> Self {
        Self {
            kind: "none".to_string(),
            options: toml::Table::new(),
        }
    }
}

impl TryFrom<toml::Value> for StageSection {
    type Error = String;

    fn try_from(v: toml::Value) -> Result<Self, String> {
        match v {
            toml::Value::String(kind) if !kind.is_empty() => Ok(Self {
                kind,
                options: toml::Table::new(),
            }),
            toml::Value::Table(mut options) => match options.remove("kind") {
                Some(toml::Value::String(kind)) if !kind.is_empty() => Ok(Self { kind, options }),
                Some(toml::Value::String(_)) => Err("`kind` must not be empty".into()),
                Some(other) => Err(format!(
                    "`kind` must be a string, found {}",
                    other.type_str()
                )),
                None => Err("missing `kind`".into()),
            },
            other => Err(format!(
                "expected an implementation name or a table, found {}",
                other.type_str()
            )),
        }
    }
}
