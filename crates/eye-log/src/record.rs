use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    Bool(bool),
    U64(u64),
    I64(i64),
    F64(f64),
    Str(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl From<tracing::Level> for Level {
    fn from(level: tracing::Level) -> Self {
        match level {
            tracing::Level::ERROR => Level::Error,
            tracing::Level::WARN => Level::Warn,
            tracing::Level::INFO => Level::Info,
            tracing::Level::DEBUG => Level::Debug,
            tracing::Level::TRACE => Level::Trace,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub schema: u32,
    pub ts_unix_ns: u64,
    pub level: Level,
    pub target: String,
    pub message: String,
    pub context: BTreeMap<String, Value>,
    pub fields: BTreeMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_serializes_lowercase() {
        assert_eq!(serde_json::to_string(&Level::Warn).unwrap(), "\"warn\"");
    }

    #[test]
    fn test_value_round_trips_each_variant() {
        for v in [
            Value::Bool(true),
            Value::U64(7),
            Value::I64(-3),
            Value::F64(1.5),
            Value::Str("x".to_string()),
        ] {
            let json = serde_json::to_string(&v).unwrap();
            let back: Value = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }
}
