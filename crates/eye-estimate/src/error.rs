use eye_core::stage::StageError;

#[derive(Debug, thiserror::Error)]
pub enum EstimateError {
    #[error("invalid options for estimator {estimator}: {source}")]
    Config {
        estimator: &'static str,
        #[source]
        source: toml::de::Error,
    },
    #[error("observation from camera {0:?} which is not in the rig")]
    UnknownCamera(String),
    #[error(transparent)]
    Geometry(#[from] eye_geometry::GeometryError),
}

impl From<EstimateError> for StageError {
    fn from(e: EstimateError) -> Self {
        match e {
            EstimateError::Config { .. } => Self::Config(e.to_string()),
            _ => Self::Failed(e.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_error_maps_to_stage_error_kinds() {
        let config = EstimateError::Config {
            estimator: "ir-pupil",
            source: toml::from_str::<toml::Table>("x =").expect_err("`x =` is not valid TOML"),
        };
        assert!(
            matches!(StageError::from(config), StageError::Config(m) if m.starts_with("invalid options for estimator ir-pupil"))
        );
        assert_eq!(
            StageError::from(EstimateError::UnknownCamera("depth".into())),
            StageError::Failed("observation from camera \"depth\" which is not in the rig".into())
        );
        assert_eq!(
            StageError::from(EstimateError::Geometry(
                eye_geometry::GeometryError::BehindCamera
            )),
            StageError::Failed("point is behind the camera".into())
        );
    }
}
