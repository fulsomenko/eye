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
        match &e {
            EstimateError::Config { .. } => Self::Config(Box::new(e)),
            _ => Self::Failed(Box::new(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use eye_core::stage::StageErrorKind;

    use super::*;

    #[test]
    fn test_estimate_error_maps_to_stage_error_kinds() {
        let config = EstimateError::Config {
            estimator: "ir-pupil",
            source: toml::from_str::<toml::Table>("x =").expect_err("`x =` is not valid TOML"),
        };
        let e = StageError::from(config);
        assert_eq!(e.kind(), StageErrorKind::Config);
        assert!(
            e.to_string()
                .starts_with("invalid configuration: invalid options for estimator ir-pupil")
        );
        assert!(
            e.source()
                .unwrap()
                .downcast_ref::<EstimateError>()
                .is_some()
        );

        let e = StageError::from(EstimateError::UnknownCamera("depth".into()));
        assert_eq!(e.kind(), StageErrorKind::Failed);
        assert_eq!(
            e.to_string(),
            "stage failed: observation from camera \"depth\" which is not in the rig"
        );
        assert!(
            e.source()
                .unwrap()
                .downcast_ref::<EstimateError>()
                .is_some()
        );

        let e = StageError::from(EstimateError::Geometry(
            eye_geometry::GeometryError::BehindCamera,
        ));
        assert_eq!(e.kind(), StageErrorKind::Failed);
        assert_eq!(e.to_string(), "stage failed: point is behind the camera");
        assert!(
            e.source()
                .unwrap()
                .downcast_ref::<EstimateError>()
                .is_some()
        );
    }
}
