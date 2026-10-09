pub mod detect;
pub mod latency;

use std::sync::Arc;

use eye::config::{CameraFormat, Config, EnvOverrides, StageSection};
use eye_calibration::nominal::{CameraStream, NominalRigConfig, nominal_rig};
use eye_core::{OutputId, Rig, ScreenModel, stage::Detector};
use nalgebra::Vector2;

use crate::{
    case::{TestCtx, TestError, TestRegistry},
    mode::Role,
    modes::opener::camera_id,
};

pub const EYE_IR_TOML: &str = include_str!("eye-ir.toml");

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ScreenParams {
    pub output: String,
    pub size_mm: [f64; 2],
    pub size_px: [u32; 2],
    pub scale: f64,
}

impl Default for ScreenParams {
    fn default() -> Self {
        Self {
            output: "eDP-1".into(),
            size_mm: [310.0, 170.0],
            size_px: [3840, 2160],
            scale: 2.0,
        }
    }
}

/// `--eye-config` (no env applied) or the embedded IR config; then every config camera is pointed
/// at the mode's stream for its role (device, format, size, fps), so the tracker captures exactly
/// what the step's mode says.
pub fn lab_config(ctx: &TestCtx) -> Result<Config, TestError> {
    let mut config = match &ctx.options().eye_config {
        Some(path) => {
            Config::load_with(Some(path.as_path()), &EnvOverrides::default())
                .map_err(other("eye config"))?
                .config
        }
        None => Config::from_toml_str(EYE_IR_TOML).map_err(other("embedded eye config"))?,
    };
    for camera in &mut config.cameras {
        let role = match camera.id.as_str() {
            Config::RGB_CAMERA_ID => Role::Rgb,
            Config::IR_CAMERA_ID => Role::Ir,
            id => {
                return Err(TestError::Other(format!(
                    "eye config camera {id:?} is neither \"rgb\" nor \"ir\""
                )));
            }
        };
        let target = ctx.mode().target(role).ok_or_else(|| {
            TestError::Other(format!(
                "eye config camera {:?} has no stream in mode {}",
                camera.id.as_str(),
                ctx.mode()
            ))
        })?;
        camera.device = target.node.clone();
        camera.format = match target.format.fourcc.as_str() {
            "GREY" => CameraFormat::Gray,
            "MJPG" => CameraFormat::Mjpeg,
            other => {
                return Err(TestError::Other(format!(
                    "mode format {other} has no eye config equivalent"
                )));
            }
        };
        camera.size = [target.format.width, target.format.height];
        camera.fps = target.format.fps;
    }
    config.validate().map_err(other("eye config"))?;
    Ok(config)
}

fn other<E: std::fmt::Display>(what: &'static str) -> impl Fn(E) -> TestError {
    move |e| TestError::Other(format!("{what}: {e}"))
}

pub(crate) fn validate_timing(seconds: f64, lead_s: f64) -> Result<(), crate::case::ParamError> {
    use crate::{case::ParamError, sequence::MAX_TIMEOUT_S};
    for (name, v) in [("seconds", seconds), ("lead_s", lead_s)] {
        if !v.is_finite() || v < 0.0 || v > MAX_TIMEOUT_S {
            return Err(ParamError::Invalid(format!(
                "{name} must be finite, >= 0 and <= {MAX_TIMEOUT_S}"
            )));
        }
    }
    Ok(())
}

/// Nominal rig over the mode's streams and a fixed screen (headless: no display probe).
pub fn lab_rig(screen: &ScreenParams, streams: &[(Role, u32, u32)]) -> Result<Rig, TestError> {
    let model = ScreenModel {
        output: OutputId::from(screen.output.as_str()),
        size_mm: Vector2::new(screen.size_mm[0], screen.size_mm[1]),
        size_px: (screen.size_px[0], screen.size_px[1]),
        scale: screen.scale,
    };
    let streams: Vec<CameraStream> = streams
        .iter()
        .map(|&(role, width, height)| CameraStream {
            id: camera_id(role),
            width,
            height,
        })
        .collect();
    nominal_rig(model, &streams, &NominalRigConfig::default()).map_err(other("nominal rig"))
}

pub type DetectorFactory =
    Arc<dyn Fn(&StageSection, &Rig) -> Result<Box<dyn Detector>, String> + Send + Sync>;

pub fn registry_detectors() -> DetectorFactory {
    Arc::new(|section, rig| {
        eye::registry::Registry::with_defaults()
            .detector(section, rig)
            .map_err(|e| e.to_string())
    })
}

pub fn register(r: &mut TestRegistry) {
    r.register(
        "detection_rate",
        "fraction of accepted frames with an eye found (subject)",
        detect::build_detection_rate,
    );
    r.register(
        "pupil_jitter",
        "pupil/iris centre RMS sample-to-sample while fixating (subject)",
        detect::build_pupil_jitter,
    );
    r.register(
        "tracker_latency",
        "Tracker frame-to-GazePoint latency and output rate (subject)",
        latency::build_tracker_latency,
    );
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use approx::assert_relative_eq;

    use super::*;
    use crate::{
        mode::EmitterSetting,
        sequence::StreamFormat,
        testkit::{self, FakeSession},
    };

    #[test]
    fn test_embedded_ir_config_parses_and_validates() {
        let config = Config::from_toml_str(EYE_IR_TOML).unwrap();
        config.validate().unwrap();
        assert_eq!(config.detect.get("ir").unwrap().kind, "ir-classic");
        assert_eq!(config.estimate.kind, "ir-pupil");
    }

    fn ctx_with_mode(mode: crate::mode::Mode) -> TestCtx {
        testkit::ctx(Arc::new(FakeSession::empty()), mode)
    }

    fn ctx_with_eye_config(mode: crate::mode::Mode, path: PathBuf) -> TestCtx {
        TestCtx::new(
            Arc::new(FakeSession::empty()),
            mode,
            Arc::new(std::sync::atomic::AtomicBool::new(false)),
            Arc::new(std::sync::atomic::AtomicBool::new(false)),
            std::time::Duration::from_secs(30),
            Arc::new(crate::case::RunOptions {
                eye_config: Some(path),
                ..crate::case::RunOptions::default()
            }),
        )
    }

    fn mode_with_ir_target(node: &str, format: &str) -> crate::mode::Mode {
        crate::mode::Mode {
            emitter: EmitterSetting::Keep,
            rgb: None,
            ir: Some(crate::mode::StreamTarget {
                node: PathBuf::from(node),
                format: format.parse::<StreamFormat>().unwrap(),
            }),
        }
    }

    #[test]
    fn test_lab_config_points_cameras_at_the_mode_stream() {
        let ctx = ctx_with_mode(mode_with_ir_target("/dev/video9", "GREY 320x180@15"));
        let config = lab_config(&ctx).unwrap();
        let camera = config.camera("ir").unwrap();
        assert_eq!(camera.device, PathBuf::from("/dev/video9"));
        assert_eq!(camera.format, CameraFormat::Gray);
        assert_eq!(camera.size, [320, 180]);
        assert_eq!(camera.fps, 15);
    }

    #[test]
    fn test_lab_config_rejects_camera_missing_from_mode() {
        let toml = "[[camera]]\nid = \"rgb\"\ndevice = \"/dev/video0\"\nformat = \"mjpeg\"\nsize = [1280, 720]\n[detect]\nrgb = \"mediapipe-ort\"\n[estimate]\nkind = \"ir-pupil\"\n";
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("eye.toml");
        std::fs::write(&path, toml).unwrap();
        let ctx = ctx_with_eye_config(mode_with_ir_target("/dev/video9", "GREY 320x180@15"), path);
        let err = lab_config(&ctx).unwrap_err();
        match err {
            TestError::Other(msg) => assert!(msg.contains("has no stream in mode"), "{msg}"),
            other => panic!("expected TestError::Other, got {other:?}"),
        }
    }

    #[test]
    fn test_lab_config_ignores_process_env_for_explicit_file() {
        let toml = "[[camera]]\nid = \"ir\"\ndevice = \"/dev/video2\"\nformat = \"gray\"\nsize = [640, 360]\n[detect]\nir = \"ir-classic\"\n[estimate]\nkind = \"ir-pupil\"\n";
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("eye.toml");
        std::fs::write(&path, toml).unwrap();
        let ctx = ctx_with_eye_config(mode_with_ir_target("/dev/video9", "GREY 320x180@15"), path);
        let config = lab_config(&ctx).unwrap();
        assert_eq!(
            config.camera("ir").unwrap().device,
            PathBuf::from("/dev/video9")
        );
    }

    #[test]
    fn test_lab_rig_uses_screen_params() {
        let rig = lab_rig(&ScreenParams::default(), &[(Role::Ir, 640, 360)]).unwrap();
        assert_eq!(rig.screen().size_px, (3840, 2160));
        assert_eq!(rig.screen().scale, 2.0);
        assert_relative_eq!(rig.screen().size_mm.x, 308.14848630853817, epsilon = 1e-9);
        assert_relative_eq!(rig.screen().size_mm.y, 173.33352354855273, epsilon = 1e-9);
        assert_eq!(rig.camera("ir").map(|c| c.width), Some(640));
    }

    #[test]
    fn test_pipeline_cases_skip_under_no_subject() {
        use std::sync::atomic::AtomicBool;

        use crate::{
            case::{RunOptions, TestRegistry},
            report::Verdict,
            runner, suites,
            testkit::FakeHost,
        };

        let def = suites::find("pipeline").expect("pipeline suite registered");
        let steps = def.load().unwrap();
        let registry = TestRegistry::builtin();
        let planned = runner::plan(steps, &registry).unwrap();

        let mut host = FakeHost::new(Arc::new(FakeSession::empty()));
        let options = RunOptions {
            subject: false,
            ..RunOptions::default()
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let mut buf = Vec::new();
        let out = runner::run(&planned, &mut host, &options, &cancel, &mut buf);

        let verdicts: Vec<_> = out.results.iter().map(|r| r.verdict).collect();
        assert_eq!(
            verdicts,
            [Verdict::Skipped, Verdict::Skipped, Verdict::Skipped]
        );
        for result in &out.results {
            assert_eq!(
                result.reason.as_deref(),
                Some("needs subject (run without --no-subject)")
            );
        }
        assert!(host.log.lock().unwrap().is_empty());
    }

    #[test]
    #[ignore = "needs hardware and a subject"]
    fn test_live_pipeline_suite_with_subject() {
        use std::sync::atomic::AtomicBool;

        use crate::{
            case::{RunOptions, TestRegistry},
            modes::{CameraSelection, LiveHost},
            report::Verdict,
            runner, suites,
        };

        let def = suites::find("pipeline").expect("pipeline suite registered");
        let steps = def.load().unwrap();
        let registry = TestRegistry::builtin();
        let planned = runner::plan(steps, &registry).unwrap();

        let mut host = LiveHost::probe(&CameraSelection::default()).unwrap();
        let options = RunOptions::default();
        let cancel = Arc::new(AtomicBool::new(false));
        let mut buf = Vec::new();
        let out = runner::run(&planned, &mut host, &options, &cancel, &mut buf);

        for result in &out.results {
            assert_eq!(
                result.verdict,
                Verdict::Pass,
                "{}: {:?}",
                result.test,
                result.reason
            );
        }
    }
}
