use super::{Registry, builtin::PassThroughFilter};
use crate::error::ConfigError;

pub(super) fn register_defaults(r: &mut Registry) -> Result<(), ConfigError> {
    r.register_filter("none", |o, rig| {
        Ok(Box::new(PassThroughFilter::from_config(o, rig)?))
    })?;

    #[cfg(feature = "ir-classic")]
    r.register_detector("ir-classic", |o, rig| {
        Ok(Box::new(eye_detect::ir::IrClassicDetector::from_config(
            o, rig,
        )?))
    })?;
    #[cfg(feature = "mediapipe-ort")]
    r.register_detector("mediapipe-ort", |o, rig| {
        Ok(Box::new(eye_detect::mediapipe::ort::from_config(o, rig)?))
    })?;
    #[cfg(feature = "mediapipe-tract")]
    r.register_detector("mediapipe-tract", |o, rig| {
        Ok(Box::new(eye_detect::mediapipe::tract::from_config(o, rig)?))
    })?;

    r.register_estimator("ir-pupil", |o, rig| {
        Ok(Box::new(
            eye_estimate::ir_pupil::IrPupilEstimator::from_config(o, rig)?,
        ))
    })?;
    r.register_estimator("landmark", |o, rig| {
        Ok(Box::new(
            eye_estimate::landmark::LandmarkEstimator::from_config(o, rig)?,
        ))
    })?;
    r.register_estimator("fused", |o, rig| {
        Ok(Box::new(eye_estimate::fused::FusedEstimator::from_config(
            o, rig,
        )?))
    })?;

    r.register_filter("one-euro", |o, rig| {
        Ok(Box::new(eye_filter::one_euro::OneEuroFilter::from_config(
            o, rig,
        )?))
    })?;
    r.register_filter("kalman", |o, rig| {
        Ok(Box::new(eye_filter::kalman::KalmanFilter::from_config(
            o, rig,
        )?))
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use eye_core::{CameraId, CameraModel, OutputId, Rig, ScreenModel, Timestamp};

    use super::*;
    use crate::{config::StageSection, error::StageKind};

    fn nominal_rig() -> Rig {
        let rgb = CameraModel {
            id: CameraId::from("rgb"),
            width: 1280,
            height: 720,
            fx: 860.0,
            fy: 860.0,
            cx: 640.0,
            cy: 360.0,
            distortion: [0.0; 5],
            screen_from_camera: nalgebra::Isometry3::identity(),
        };
        let ir = CameraModel {
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
        Rig::new(vec![rgb, ir], screen).unwrap()
    }

    fn section(kind: &str, options: toml::Table) -> StageSection {
        StageSection {
            kind: kind.to_string(),
            options,
        }
    }

    fn gaze_point(t_ms: u64, mm_x: f64, mm_y: f64) -> eye_core::GazePoint {
        eye_core::GazePoint {
            timestamp: Timestamp::from_nanos(t_ms * 1_000_000),
            output: OutputId::from("eDP-1"),
            mm: nalgebra::Point2::new(mm_x, mm_y),
            px_physical: nalgebra::Point2::new(0.0, 0.0),
            px_logical: nalgebra::Point2::new(0.0, 0.0),
            cov_mm: nalgebra::Matrix2::identity(),
            confidence: 1.0,
        }
    }

    #[test]
    fn test_one_euro_receives_rig_screen() {
        let registry = Registry::with_defaults();
        let rig = nominal_rig();
        let mut options = toml::Table::new();
        options.insert("min_cutoff".to_string(), toml::Value::Float(1.0));
        options.insert("beta".to_string(), toml::Value::Float(0.007));

        let mut filter = registry
            .filter(&section("one-euro", options), &rig)
            .unwrap();

        filter.apply(gaze_point(0, 155.0, 85.0));
        let out = filter.apply(gaze_point(33, 155.0, 85.0));

        assert_relative_eq!(out.px_logical.x, 960.0, epsilon = 1e-6);
        assert_relative_eq!(out.px_logical.y, 540.0, epsilon = 1e-6);
        assert_relative_eq!(out.px_physical.x, 1920.0, epsilon = 1e-6);
        assert_relative_eq!(out.px_physical.y, 1080.0, epsilon = 1e-6);
    }

    #[test]
    fn test_defaults_list_all_estimators_and_filters() {
        let registry = Registry::with_defaults();
        assert_eq!(
            registry.names(StageKind::Estimator),
            vec!["fused", "ir-pupil", "landmark"]
        );
        assert_eq!(
            registry.names(StageKind::Filter),
            vec!["kalman", "none", "one-euro"]
        );
    }

    #[test]
    #[cfg(feature = "ir-classic")]
    fn test_ir_classic_builds_from_empty_options() {
        let registry = Registry::with_defaults();
        let rig = nominal_rig();
        let detector = registry
            .detector(&section("ir-classic", toml::Table::new()), &rig)
            .unwrap();
        assert_eq!(detector.name(), "ir-classic");
    }

    #[test]
    #[cfg(not(feature = "mediapipe-tract"))]
    fn test_mediapipe_tract_reports_not_compiled() {
        let registry = Registry::with_defaults();
        let rig = nominal_rig();
        let Err(e) = registry.detector(&section("mediapipe-tract", toml::Table::new()), &rig)
        else {
            panic!("must fail")
        };
        match e {
            ConfigError::NotCompiled { feature, .. } => assert_eq!(feature, "mediapipe-tract"),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "mediapipe-ort")]
    fn test_mediapipe_ort_registered() {
        let registry = Registry::with_defaults();
        assert!(
            registry
                .names(StageKind::Detector)
                .contains(&"mediapipe-ort")
        );
    }

    #[test]
    fn test_estimator_unknown_option_names_stage() {
        let registry = Registry::with_defaults();
        let rig = nominal_rig();
        let mut options = toml::Table::new();
        options.insert("bogus".to_string(), toml::Value::Integer(1));

        let Err(e) = registry.estimator(&section("ir-pupil", options), &rig) else {
            panic!("must fail")
        };
        match &e {
            ConfigError::Stage { kind, name, .. } => {
                assert_eq!(*kind, StageKind::Estimator);
                assert_eq!(name, "ir-pupil");
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert!(e.to_string().contains("bogus"));
    }

    #[test]
    fn test_root_example_config_resolves_every_stage() {
        use crate::config::Config;

        let registry = Registry::with_defaults();
        let rig = nominal_rig();
        let config = Config::builtin_default();

        assert!(registry.estimator(&config.estimate, &rig).is_ok());
        assert!(registry.filter(&config.filter, &rig).is_ok());

        let ir_section = &config.detect["ir"];
        assert_eq!(ir_section.kind, "ir-classic");
        #[cfg(feature = "ir-classic")]
        assert!(registry.detector(ir_section, &rig).is_ok());
        #[cfg(not(feature = "ir-classic"))]
        {
            let Err(e) = registry.detector(ir_section, &rig) else {
                panic!("must fail")
            };
            assert!(matches!(e, ConfigError::NotCompiled { .. }));
        }

        let rgb_section = &config.detect["rgb"];
        assert_eq!(rgb_section.kind, "mediapipe-ort");
        #[cfg(feature = "mediapipe-ort")]
        assert!(
            registry
                .names(StageKind::Detector)
                .contains(&rgb_section.kind.as_str())
        );
        #[cfg(not(feature = "mediapipe-ort"))]
        {
            let Err(e) = registry.detector(rgb_section, &rig) else {
                panic!("must fail")
            };
            assert!(matches!(e, ConfigError::NotCompiled { .. }));
        }
    }
}
