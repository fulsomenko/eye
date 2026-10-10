mod builtin;
mod defaults;

use std::{
    collections::{BTreeMap, btree_map::Entry},
    fmt,
};

use eye_core::{
    Rig,
    log::field,
    stage::{Detector, GazeEstimator, GazeFilter, StageError},
};

pub use builtin::PassThroughFilter;

use crate::{
    config::StageSection,
    error::{ConfigError, StageKind},
};

type Ctor<T> = Box<dyn Fn(&toml::Table, &Rig) -> Result<T, StageError> + Send + Sync>;

/// (kind, name, Cargo feature of the `eye` crate, compiled in).
const FEATURE_GATED: &[(StageKind, &str, &str, bool)] = &[
    (
        StageKind::Detector,
        "ir-classic",
        "ir-classic",
        cfg!(feature = "ir-classic"),
    ),
    (
        StageKind::Detector,
        "mediapipe-ort",
        "mediapipe-ort",
        cfg!(feature = "mediapipe-ort"),
    ),
    (
        StageKind::Detector,
        "mediapipe-tract",
        "mediapipe-tract",
        cfg!(feature = "mediapipe-tract"),
    ),
];

#[derive(Default)]
pub struct Registry {
    detectors: BTreeMap<&'static str, Ctor<Box<dyn Detector>>>,
    estimators: BTreeMap<&'static str, Ctor<Box<dyn GazeEstimator>>>,
    filters: BTreeMap<&'static str, Ctor<Box<dyn GazeFilter>>>,
}

impl fmt::Debug for Registry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Registry")
            .field("detectors", &self.detectors.keys().collect::<Vec<_>>())
            .field("estimators", &self.estimators.keys().collect::<Vec<_>>())
            .field("filters", &self.filters.keys().collect::<Vec<_>>())
            .finish()
    }
}

fn build<T>(
    map: &BTreeMap<&'static str, Ctor<T>>,
    kind: StageKind,
    section: &StageSection,
    rig: &Rig,
) -> Result<T, ConfigError> {
    let Some(ctor) = map.get(section.kind.as_str()) else {
        if let Some(&(_, _, feature, _)) = FEATURE_GATED
            .iter()
            .find(|(k, n, _, compiled)| *k == kind && *n == section.kind && !compiled)
        {
            return Err(ConfigError::NotCompiled {
                kind,
                name: section.kind.clone(),
                feature,
            });
        }
        return Err(ConfigError::UnknownStage {
            kind,
            name: section.kind.clone(),
            available: map.keys().copied().collect(),
        });
    };
    match ctor(&section.options, rig) {
        Ok(value) => {
            tracing::debug!(
                { field::STAGE_KIND } = kind.span_kind(),
                { field::STAGE_NAME } = section.kind.as_str(),
                "stage constructed"
            );
            Ok(value)
        }
        Err(source) => Err(ConfigError::Stage {
            kind,
            name: section.kind.clone(),
            source,
        }),
    }
}

fn insert<C>(
    map: &mut BTreeMap<&'static str, C>,
    kind: StageKind,
    name: &'static str,
    ctor: C,
) -> Result<(), ConfigError> {
    match map.entry(name) {
        Entry::Occupied(_) => Err(ConfigError::DuplicateRegistration { kind, name }),
        Entry::Vacant(v) => {
            v.insert(ctor);
            Ok(())
        }
    }
}

impl Registry {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Built-ins plus every implementation compiled in by Cargo features.
    pub fn with_defaults() -> Self {
        let mut r = Self::empty();
        defaults::register_defaults(&mut r).expect("built-in stage names are unique");
        r
    }

    pub fn register_detector<F>(&mut self, name: &'static str, ctor: F) -> Result<(), ConfigError>
    where
        F: Fn(&toml::Table, &Rig) -> Result<Box<dyn Detector>, StageError> + Send + Sync + 'static,
    {
        insert(
            &mut self.detectors,
            StageKind::Detector,
            name,
            Box::new(ctor),
        )
    }

    pub fn register_estimator<F>(&mut self, name: &'static str, ctor: F) -> Result<(), ConfigError>
    where
        F: Fn(&toml::Table, &Rig) -> Result<Box<dyn GazeEstimator>, StageError>
            + Send
            + Sync
            + 'static,
    {
        insert(
            &mut self.estimators,
            StageKind::Estimator,
            name,
            Box::new(ctor),
        )
    }

    pub fn register_filter<F>(&mut self, name: &'static str, ctor: F) -> Result<(), ConfigError>
    where
        F: Fn(&toml::Table, &Rig) -> Result<Box<dyn GazeFilter>, StageError>
            + Send
            + Sync
            + 'static,
    {
        insert(&mut self.filters, StageKind::Filter, name, Box::new(ctor))
    }

    pub fn detector(
        &self,
        section: &StageSection,
        rig: &Rig,
    ) -> Result<Box<dyn Detector>, ConfigError> {
        build(&self.detectors, StageKind::Detector, section, rig)
    }

    pub fn estimator(
        &self,
        section: &StageSection,
        rig: &Rig,
    ) -> Result<Box<dyn GazeEstimator>, ConfigError> {
        build(&self.estimators, StageKind::Estimator, section, rig)
    }

    pub fn filter(
        &self,
        section: &StageSection,
        rig: &Rig,
    ) -> Result<Box<dyn GazeFilter>, ConfigError> {
        build(&self.filters, StageKind::Filter, section, rig)
    }

    /// Sorted.
    pub fn names(&self, kind: StageKind) -> Vec<&'static str> {
        match kind {
            StageKind::Detector => self.detectors.keys().copied().collect(),
            StageKind::Estimator => self.estimators.keys().copied().collect(),
            StageKind::Filter => self.filters.keys().copied().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use eye_core::{CameraId, CameraModel, GazePoint, OutputId, ScreenModel, Timestamp};
    use serde::Deserialize;

    use super::*;

    fn nominal_rig() -> Rig {
        let camera = CameraModel {
            id: CameraId::from("ir"),
            width: 640,
            height: 360,
            fx: 430.0,
            fy: 430.0,
            cx: 320.0,
            cy: 180.0,
            distortion: [0.0; 5],
            screen_from_camera: nalgebra::Isometry3::identity(),
        };
        let screen = ScreenModel {
            output: OutputId::from("eDP-1"),
            size_mm: nalgebra::Vector2::new(310.0, 170.0),
            size_px: (3840, 2160),
            scale: 2.0,
        };
        Rig::new(vec![camera], screen).unwrap()
    }

    fn section(kind: &str, options: toml::Table) -> StageSection {
        StageSection {
            kind: kind.to_string(),
            options,
        }
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct FakeOptions {
        #[allow(dead_code)]
        gain: f64,
    }

    struct FakeFilter;

    impl FakeFilter {
        fn from_table(table: &toml::Table) -> Result<Self, StageError> {
            let FakeOptions { .. } = table
                .clone()
                .try_into()
                .map_err(|e: toml::de::Error| StageError::Config(e.to_string()))?;
            Ok(Self)
        }
    }

    impl GazeFilter for FakeFilter {
        fn name(&self) -> &'static str {
            "fake"
        }

        fn apply(&mut self, point: GazePoint) -> GazePoint {
            point
        }

        fn reset(&mut self) {}
    }

    #[test]
    fn test_options_reach_constructor_without_kind_and_with_rig() {
        let mut registry = Registry::empty();
        registry
            .register_filter("f", |options, rig| {
                assert!(!options.contains_key("kind"));
                assert_eq!(options["gain"].as_float(), Some(1.5));
                assert_eq!(rig.screen().output.as_str(), "eDP-1");
                Ok(Box::new(PassThroughFilter) as Box<dyn GazeFilter>)
            })
            .unwrap();

        let mut options = toml::Table::new();
        options.insert("gain".to_string(), toml::Value::Float(1.5));
        let result = registry.filter(&section("f", options), &nominal_rig());
        assert!(result.is_ok());
    }

    #[test]
    fn test_with_defaults_registers_none_filter() {
        let registry = Registry::with_defaults();
        assert!(registry.names(StageKind::Filter).contains(&"none"));
    }

    #[test]
    fn test_with_defaults_has_no_duplicates() {
        let _ = Registry::with_defaults();
    }

    #[test]
    fn test_unknown_detector_lists_available_sorted() {
        let mut registry = Registry::empty();
        registry
            .register_detector("b-det", |_, _| unreachable!())
            .unwrap();
        registry
            .register_detector("a-det", |_, _| unreachable!())
            .unwrap();

        let Err(e) = registry.detector(&section("zzz", toml::Table::new()), &nominal_rig()) else {
            panic!("must fail")
        };
        match &e {
            ConfigError::UnknownStage {
                kind, available, ..
            } => {
                assert_eq!(*kind, StageKind::Detector);
                assert_eq!(available, &vec!["a-det", "b-det"]);
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert!(e.to_string().contains("a-det, b-det"));
    }

    #[test]
    #[cfg(not(feature = "mediapipe-tract"))]
    fn test_feature_gated_name_not_compiled_hints_feature() {
        let registry = Registry::empty();
        let Err(e) = registry.detector(
            &section("mediapipe-tract", toml::Table::new()),
            &nominal_rig(),
        ) else {
            panic!("must fail")
        };
        match &e {
            ConfigError::NotCompiled { feature, .. } => assert_eq!(*feature, "mediapipe-tract"),
            other => panic!("unexpected error: {other:?}"),
        }
        assert!(e.to_string().contains("--features mediapipe-tract"));
    }

    #[test]
    #[cfg(feature = "ir-classic")]
    fn test_compiled_but_unregistered_feature_name_is_unknown() {
        let registry = Registry::empty();
        let Err(e) = registry.detector(&section("ir-classic", toml::Table::new()), &nominal_rig())
        else {
            panic!("must fail")
        };
        assert!(matches!(e, ConfigError::UnknownStage { .. }));
    }

    #[test]
    fn test_duplicate_registration_errors() {
        let mut registry = Registry::empty();
        registry
            .register_filter("x", |_, _| {
                Ok(Box::new(PassThroughFilter) as Box<dyn GazeFilter>)
            })
            .unwrap();
        let err = registry
            .register_filter("x", |_, _| {
                Ok(Box::new(PassThroughFilter) as Box<dyn GazeFilter>)
            })
            .unwrap_err();
        match err {
            ConfigError::DuplicateRegistration { kind, name } => {
                assert_eq!(kind, StageKind::Filter);
                assert_eq!(name, "x");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_constructor_error_wrapped_with_kind_and_name() {
        let mut registry = Registry::empty();
        registry
            .register_estimator("e", |_, _| Err(StageError::Failed("boom".into())))
            .unwrap();

        let Err(e) = registry.estimator(&section("e", toml::Table::new()), &nominal_rig()) else {
            panic!("must fail")
        };
        let display = e.to_string();
        assert!(display.starts_with("building estimator \"e\": "));
        assert!(display.contains("boom"));
    }

    #[test]
    fn test_options_unknown_field_surfaces_through_registry() {
        let mut registry = Registry::empty();
        registry
            .register_filter("fake", |o, _| {
                Ok(Box::new(FakeFilter::from_table(o)?) as Box<dyn GazeFilter>)
            })
            .unwrap();

        let mut options = toml::Table::new();
        options.insert("gain".to_string(), toml::Value::Float(1.0));
        options.insert("gian".to_string(), toml::Value::Float(2.0));

        let Err(e) = registry.filter(&section("fake", options), &nominal_rig()) else {
            panic!("must fail")
        };
        assert!(e.to_string().contains("gian"));
    }

    #[test]
    fn test_none_filter_passes_point_through_and_rejects_options() {
        let registry = Registry::with_defaults();
        let rig = nominal_rig();

        let mut filter = registry
            .filter(&section("none", toml::Table::new()), &rig)
            .unwrap();
        let point = GazePoint {
            timestamp: Timestamp::from_nanos(0),
            output: OutputId::from("eDP-1"),
            mm: nalgebra::Point2::new(1.0, 2.0),
            px_physical: nalgebra::Point2::new(3.0, 4.0),
            px_logical: nalgebra::Point2::new(5.0, 6.0),
            cov_mm: nalgebra::Matrix2::zeros(),
            confidence: 1.0,
        };
        let out = filter.apply(point.clone());
        assert_eq!(out, point);

        let mut options = toml::Table::new();
        options.insert("x".to_string(), toml::Value::Integer(1));
        let Err(e) = registry.filter(&section("none", options), &rig) else {
            panic!("must fail")
        };
        assert!(e.to_string().contains("x"));
    }

    #[test]
    fn test_filter_names_equal_registry_keys() {
        let registry = Registry::with_defaults();
        let mut expected = vec![
            eye_filter::kalman::KalmanFilter::NAME,
            PassThroughFilter::NAME,
            eye_filter::one_euro::OneEuroFilter::NAME,
        ];
        expected.sort_unstable();
        assert_eq!(registry.names(StageKind::Filter), expected);

        for name in expected {
            let filter = registry
                .filter(&section(name, toml::Table::new()), &nominal_rig())
                .unwrap();
            assert_eq!(filter.name(), name);
        }
    }

    #[test]
    fn test_registry_is_send_sync_and_debug() {
        const fn is_send_sync<T: Send + Sync>() {}
        is_send_sync::<Registry>();

        let registry = Registry::with_defaults();
        assert!(format!("{registry:?}").contains("none"));
    }
}
