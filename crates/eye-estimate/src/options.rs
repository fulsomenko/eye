use crate::EstimateError;

pub fn parse_options<T: serde::de::DeserializeOwned>(
    estimator: &'static str,
    table: &toml::Table,
) -> Result<T, EstimateError> {
    table
        .clone()
        .try_into()
        .map_err(|source| EstimateError::Config { estimator, source })
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct TestOptions {
        #[allow(dead_code)]
        threshold: u32,
    }

    #[test]
    fn test_parse_options_rejects_unknown_field() {
        let mut table = toml::Table::new();
        table.insert("threshold".into(), 1.into());
        table.insert("extra".into(), 2.into());

        let err = parse_options::<TestOptions>("test", &table).unwrap_err();
        assert!(matches!(
            err,
            EstimateError::Config {
                estimator: "test",
                ..
            }
        ));
    }
}
