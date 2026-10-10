mod env;
mod stage;

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Duration,
};

use eye_core::log::field;
use eye_core::{CameraId, CameraInfo, PixelFormat};
use serde::{Deserialize, Deserializer, de::Error as _};

pub use env::EnvOverrides;
pub use stage::StageSection;

use crate::error::ConfigError;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(rename = "camera")]
    pub cameras: Vec<CameraConfig>,
    /// Key = camera id, value = the detector for that camera.
    #[serde(default)]
    pub detect: BTreeMap<String, StageSection>,
    pub estimate: StageSection,
    #[serde(default = "StageSection::none")]
    pub filter: StageSection,
    #[serde(default)]
    pub capture: CaptureConfig,
    #[serde(default)]
    pub tracker: TrackerConfig,
    #[serde(default)]
    pub output: OutputConfig,
    /// Opaque `[rig]` section, interpreted by eye-calibration's nominal rig.
    #[serde(default)]
    pub rig: Option<toml::Table>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraConfig {
    #[serde(deserialize_with = "de_camera_id")]
    pub id: CameraId,
    pub device: PathBuf,
    pub format: CameraFormat,
    /// `[width, height]`
    pub size: [u32; 2],
    #[serde(default = "CameraConfig::default_fps")]
    pub fps: u32,
    /// IR emitter driver (e.g. "msxu"); acted on by eye-app / eye-lab, never by `eye`.
    #[serde(default)]
    pub emitter: Option<String>,
}

impl CameraConfig {
    fn default_fps() -> u32 {
        30
    }

    /// The stream this camera is configured to deliver, before the device is opened.
    pub fn to_info(&self) -> CameraInfo {
        CameraInfo {
            id: self.id.clone(),
            format: self.format.pixel_format(),
            width: self.size[0],
            height: self.size[1],
            frame_interval: Duration::from_nanos(1_000_000_000 / u64::from(self.fps.max(1))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraFormat {
    Mjpeg,
    Gray,
}

impl CameraFormat {
    pub fn pixel_format(self) -> PixelFormat {
        match self {
            Self::Mjpeg => PixelFormat::Mjpeg,
            Self::Gray => PixelFormat::Gray8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CaptureConfig {
    pub channel_capacity: usize,
    /// Added to the RGB (secondary) camera's timestamps before pairing.
    pub rgb_offset_ns: i64,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            channel_capacity: 2,
            rgb_offset_ns: 3_000_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrackerConfig {
    pub max_consecutive_stage_errors: u32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            max_consecutive_stage_errors: 30,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputMode {
    #[default]
    Point,
    Region,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OutputKind {
    #[default]
    LayerShell,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputConfig {
    /// Overlay backend; only "layer-shell" exists.
    pub kind: OutputKind,
    /// Output name; `None` = the rig's screen output.
    pub target: Option<String>,
    pub mode: OutputMode,
    /// `[cols, rows]`, used by region mode.
    pub grid: [u32; 2],
    /// Point-mode maximum render lag behind the newest sample; 0 draws samples as they
    /// arrive. Ignored in region mode.
    pub easing_ms: u64,
    /// Point-mode hide margin, in logical px either side of the canvas; see
    /// `eye_overlay::point::HideRules`. Ignored in region mode.
    pub hide_margin_px: f64,
    /// Point-mode confidence threshold below which nothing is drawn. Ignored in region mode.
    pub hide_below_confidence: f64,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            kind: OutputKind::LayerShell,
            target: None,
            mode: OutputMode::Point,
            grid: [4, 4],
            easing_ms: 80,
            hide_margin_px: 24.0,
            hide_below_confidence: 0.2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigSource {
    File(PathBuf),
    BuiltinDefault,
}

#[derive(Debug, Clone)]
pub struct LoadedConfig {
    pub config: Config,
    pub source: ConfigSource,
    pub applied_env: Vec<&'static str>,
}

impl Config {
    pub const RGB_CAMERA_ID: &'static str = "rgb";
    pub const IR_CAMERA_ID: &'static str = "ir";

    /// Process env applied, then validated. What eye-app calls.
    pub fn load(explicit: Option<&Path>) -> Result<Config, ConfigError> {
        Self::load_with(explicit, &EnvOverrides::from_process_env()).map(|loaded| loaded.config)
    }

    /// The env is a value: the bench passes `&EnvOverrides::default()` so `.envrc.local` never changes a result.
    pub fn load_with(
        explicit: Option<&Path>,
        env: &EnvOverrides,
    ) -> Result<LoadedConfig, ConfigError> {
        let (mut config, source) = match explicit {
            Some(path) => (Self::read(path)?, ConfigSource::File(path.to_owned())),
            None => {
                use etcetera::BaseStrategy as _;
                let config_dir = etcetera::choose_base_strategy()
                    .ok()
                    .map(|strategy| strategy.config_dir());
                let cwd = std::env::current_dir().map_err(|source| ConfigError::Io {
                    path: PathBuf::from("."),
                    source,
                })?;
                match default_config_candidates(config_dir.as_deref(), &cwd)
                    .into_iter()
                    .find(|p| p.is_file())
                {
                    Some(path) => (Self::read(&path)?, ConfigSource::File(path)),
                    None => {
                        tracing::info!("no eye.toml found, using the built-in default");
                        (Self::builtin_default(), ConfigSource::BuiltinDefault)
                    }
                }
            }
        };
        let applied_env = config.apply_env(env);
        config.validate()?;
        let source_str = match &source {
            ConfigSource::File(path) => path.display().to_string(),
            ConfigSource::BuiltinDefault => "builtin".to_string(),
        };
        tracing::info!(
            source = %source_str,
            cameras = config.cameras.len(),
            detectors = config.detect.len(),
            estimate = config.estimate.kind.as_str(),
            filter = config.filter.kind.as_str(),
            applied_env = applied_env.len(),
            "config loaded"
        );
        Ok(LoadedConfig {
            config,
            source,
            applied_env,
        })
    }

    fn read(path: &Path) -> Result<Config, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.to_owned(),
            source,
        })?;
        toml::from_str(&text).map_err(|source| ConfigError::Parse {
            origin: path.display().to_string(),
            source,
        })
    }

    pub fn from_toml_str(s: &str) -> Result<Self, ConfigError> {
        toml::from_str(s).map_err(|source| ConfigError::Parse {
            origin: "inline config".to_string(),
            source,
        })
    }

    pub fn builtin_default() -> Self {
        toml::from_str(include_str!("default.toml")).expect("the built-in default config parses")
    }

    pub fn apply_env(&mut self, env: &EnvOverrides) -> Vec<&'static str> {
        let mut applied = Vec::new();
        for (var, id, value) in [
            ("EYE_CAMERA", Self::RGB_CAMERA_ID, &env.camera),
            ("EYE_IR_CAMERA", Self::IR_CAMERA_ID, &env.ir_camera),
        ] {
            let Some(device) = value else { continue };
            match self.cameras.iter_mut().find(|c| c.id.as_str() == id) {
                Some(cam) => {
                    cam.device = PathBuf::from(device);
                    applied.push(var);
                    tracing::debug!(
                        var,
                        { field::CAMERA } = id,
                        device = device.as_str(),
                        "env override applied"
                    );
                }
                None => {
                    tracing::warn!(
                        var,
                        { field::CAMERA } = id,
                        "ignored: no camera with this id"
                    );
                }
            }
        }
        if let Some(target) = &env.output {
            self.output.target = Some(target.clone());
            applied.push("EYE_OUTPUT");
        }
        applied
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.cameras.is_empty() {
            return Err(ConfigError::NoCameras);
        }

        let mut seen = std::collections::HashSet::new();
        for camera in &self.cameras {
            if !seen.insert(camera.id.as_str()) {
                return Err(ConfigError::DuplicateCamera(camera.id.as_str().to_string()));
            }
        }

        for camera in &self.cameras {
            if camera.size[0] == 0 || camera.size[1] == 0 {
                return Err(ConfigError::InvalidCamera {
                    camera: camera.id.as_str().to_string(),
                    reason: "size must be non-zero",
                });
            }
            if camera.fps == 0 {
                return Err(ConfigError::InvalidCamera {
                    camera: camera.id.as_str().to_string(),
                    reason: "fps must be non-zero",
                });
            }
        }

        if self.detect.is_empty() {
            return Err(ConfigError::NoDetectors);
        }

        for camera_id in self.detect.keys() {
            if !self.cameras.iter().any(|c| c.id.as_str() == camera_id) {
                let mut available: Vec<String> = self
                    .cameras
                    .iter()
                    .map(|c| c.id.as_str().to_string())
                    .collect();
                available.sort();
                return Err(ConfigError::UnknownDetectCamera {
                    camera: camera_id.clone(),
                    available,
                });
            }
        }

        if self.capture.channel_capacity < 1 {
            return Err(ConfigError::ZeroChannelCapacity);
        }

        if self.output.grid[0] == 0 || self.output.grid[1] == 0 {
            return Err(ConfigError::InvalidOutput("grid must be at least 1x1"));
        }
        if self.output.hide_margin_px < 0.0 {
            return Err(ConfigError::InvalidOutput(
                "hide_margin_px must be non-negative",
            ));
        }
        if !(0.0..=1.0).contains(&self.output.hide_below_confidence) {
            return Err(ConfigError::InvalidOutput(
                "hide_below_confidence must be between 0 and 1",
            ));
        }

        Ok(())
    }

    pub fn camera(&self, id: &str) -> Option<&CameraConfig> {
        self.cameras.iter().find(|c| c.id.as_str() == id)
    }
}

/// Lookup order: `<config_dir>/eye/eye.toml`, then `<cwd>/eye.toml`.
pub fn default_config_candidates(config_dir: Option<&Path>, cwd: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(config_dir) = config_dir {
        candidates.push(config_dir.join("eye").join("eye.toml"));
    }
    candidates.push(cwd.join("eye.toml"));
    candidates
}

fn de_camera_id<'de, D: Deserializer<'de>>(d: D) -> Result<CameraId, D::Error> {
    let s = String::deserialize(d)?;
    if s.is_empty() {
        return Err(D::Error::custom("camera id must not be empty"));
    }
    Ok(CameraId::from(s.as_str()))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use proptest::prelude::*;

    use super::*;

    fn unique_temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "eye-config-test-{}-{}-{}",
            std::process::id(),
            name,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time moves forward")
                .as_nanos()
        ))
    }

    fn write_temp_toml(name: &str, contents: &str) -> PathBuf {
        let path = unique_temp_path(name);
        std::fs::write(&path, contents).expect("write temp config");
        path
    }

    #[test]
    fn test_root_example_parses_expected_structure() {
        let config = Config::builtin_default();
        assert_eq!(config.cameras.len(), 2);

        let rgb = config.camera("rgb").expect("rgb camera exists");
        assert_eq!(rgb.device, PathBuf::from("/dev/video0"));
        assert_eq!(rgb.format, CameraFormat::Mjpeg);
        assert_eq!(rgb.size, [1280, 720]);
        assert_eq!(rgb.fps, 30);
        assert_eq!(rgb.emitter, None);

        let ir = config.camera("ir").expect("ir camera exists");
        assert_eq!(ir.device, PathBuf::from("/dev/video2"));
        assert_eq!(ir.format, CameraFormat::Gray);
        assert_eq!(ir.size, [640, 360]);
        assert_eq!(ir.emitter, Some("msxu".to_string()));

        assert_eq!(config.detect["ir"].kind, "ir-classic");
        assert_eq!(config.detect["rgb"].kind, "mediapipe-ort");
        assert_eq!(config.estimate.kind, "fused");

        assert_eq!(config.filter.kind, "one-euro");
        assert!(config.filter.options.is_empty());

        assert_eq!(
            config.output,
            OutputConfig {
                kind: OutputKind::LayerShell,
                target: Some("eDP-1".to_string()),
                mode: OutputMode::Point,
                grid: [4, 4],
                easing_ms: 80,
                hide_margin_px: 24.0,
                hide_below_confidence: 0.2,
            }
        );
        assert_eq!(config.rig, None);
        assert_eq!(config.capture.rgb_offset_ns, 3_000_000);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_builtin_default_sections_equal_struct_defaults() {
        let c = Config::builtin_default();
        assert_eq!(c.capture, CaptureConfig::default());
        assert_eq!(c.tracker, TrackerConfig::default());
        assert_eq!(
            OutputConfig {
                target: None,
                ..c.output.clone()
            },
            OutputConfig::default()
        );
    }

    #[test]
    fn test_builtin_default_filter_resolves_to_one_euro_default() {
        let c = Config::builtin_default();
        assert_eq!(c.filter.kind, "one-euro");
        assert!(c.filter.options.is_empty());
        let resolved: eye_filter::one_euro::OneEuroConfig = c
            .filter
            .options
            .clone()
            .try_into()
            .expect("empty table resolves");
        assert_eq!(resolved, eye_filter::one_euro::OneEuroConfig::default());
    }

    #[test]
    fn test_rig_section_is_kept_opaque() {
        let base = minimal_valid_toml();
        let with_rig = format!("{base}\n[rig]\nfov_deg = 80.0\n");
        let config = Config::from_toml_str(&with_rig).expect("parses with rig");
        let mut expected = toml::Table::new();
        expected.insert("fov_deg".into(), toml::Value::Float(80.0));
        assert_eq!(config.rig, Some(expected));

        let without_rig = Config::from_toml_str(&base).expect("parses without rig");
        assert_eq!(without_rig.rig, None);
    }

    #[test]
    fn test_capture_and_tracker_keys_are_snake_case() {
        let base = minimal_valid_toml();
        let toml_str = format!(
            "{base}\n[capture]\nrgb_offset_ns = -1500000\nchannel_capacity = 3\n[tracker]\nmax_consecutive_stage_errors = 5\n"
        );
        let config = Config::from_toml_str(&toml_str).expect("parses snake_case keys");
        assert_eq!(
            config.capture,
            CaptureConfig {
                channel_capacity: 3,
                rgb_offset_ns: -1_500_000,
            }
        );
        assert_eq!(config.tracker.max_consecutive_stage_errors, 5);

        let kebab = format!("{base}\n[capture]\nrgb-offset-ns = 1\n");
        let err = Config::from_toml_str(&kebab).expect_err("kebab-case key is rejected");
        assert!(matches!(err, ConfigError::Parse { .. }));
        assert!(err.to_string().contains("rgb-offset-ns"));
    }

    #[test]
    fn test_stage_shorthand_string_yields_empty_options() {
        let base = minimal_valid_toml();
        let config = Config::from_toml_str(&base).expect("parses");
        assert_eq!(config.estimate.kind, "fused");
        assert!(config.estimate.options.is_empty());
    }

    #[test]
    fn test_stage_table_without_kind_errors() {
        let toml_str = format!("{}\n[filter]\nbeta = 1.0\n", minimal_camera_and_estimate());
        let err = Config::from_toml_str(&toml_str).expect_err("missing kind errors");
        let message = err.to_string();
        assert!(message.contains("filter"));
        assert!(message.contains("missing `kind`"));
    }

    #[test]
    fn test_stage_kind_non_string_errors_with_type() {
        let toml_str = format!("{}\n[filter]\nkind = 3\n", minimal_camera_and_estimate());
        let err = Config::from_toml_str(&toml_str).expect_err("non-string kind errors");
        assert!(
            err.to_string()
                .contains("`kind` must be a string, found integer")
        );
    }

    #[test]
    fn test_missing_estimate_errors() {
        let toml_str = minimal_camera_only();
        let err = Config::from_toml_str(&toml_str).expect_err("missing estimate errors");
        assert!(err.to_string().contains("estimate"));
    }

    #[test]
    fn test_unknown_key_errors() {
        let toml_str = format!("{}\n[filtr]\nkind = \"x\"\n", minimal_camera_and_estimate());
        let err = Config::from_toml_str(&toml_str).expect_err("unknown top-level key errors");
        assert!(err.to_string().contains("filtr"));

        let toml_str2 = "estimate = \"fused\"\n[[camera]]\nid = \"rgb\"\ndevise = \"/dev/video0\"\nformat = \"mjpeg\"\nsize = [1280, 720]\n[detect]\nrgb = \"x\"\n";
        let err2 = Config::from_toml_str(toml_str2).expect_err("unknown camera key errors");
        assert!(err2.to_string().contains("devise"));
    }

    #[test]
    fn test_parse_error_reports_line_and_column() {
        let toml_str = "estimate = \"fused\"\n[[camera]]\nid = = \"rgb\"\n";
        let err = Config::from_toml_str(toml_str).expect_err("syntax error");
        assert!(err.to_string().contains("line 3"));
    }

    #[test]
    fn test_defaults_for_optional_sections() {
        let toml_str = minimal_camera_and_estimate();
        let config = Config::from_toml_str(&toml_str).expect("parses with defaults");
        assert_eq!(config.filter.kind, "none");
        assert_eq!(config.capture.channel_capacity, 2);
        assert_eq!(config.capture.rgb_offset_ns, 3_000_000);
        assert_eq!(config.tracker.max_consecutive_stage_errors, 30);
        assert_eq!(
            config.output,
            OutputConfig {
                kind: OutputKind::LayerShell,
                target: None,
                mode: OutputMode::Point,
                grid: [4, 4],
                easing_ms: 80,
                hide_margin_px: 24.0,
                hide_below_confidence: 0.2,
            }
        );
    }

    #[test]
    fn test_camera_format_maps_to_pixel_format() {
        assert_eq!(CameraFormat::Gray.pixel_format(), PixelFormat::Gray8);
        assert_eq!(CameraFormat::Mjpeg.pixel_format(), PixelFormat::Mjpeg);
    }

    #[test]
    fn test_camera_to_info_is_stream_description() {
        let config = Config::builtin_default();
        let ir = config.camera("ir").expect("ir camera exists");
        assert_eq!(
            ir.to_info(),
            CameraInfo {
                id: CameraId::from("ir"),
                format: PixelFormat::Gray8,
                width: 640,
                height: 360,
                frame_interval: Duration::from_nanos(33_333_333),
            }
        );

        let mut slow_ir = ir.clone();
        slow_ir.fps = 15;
        assert_eq!(
            slow_ir.to_info().frame_interval,
            Duration::from_nanos(66_666_666)
        );
    }

    #[test]
    fn test_duplicate_camera_id_errors() {
        let toml_str = "estimate = \"fused\"\n[[camera]]\nid = \"rgb\"\ndevice = \"/dev/video0\"\nformat = \"mjpeg\"\nsize = [1280, 720]\n[[camera]]\nid = \"rgb\"\ndevice = \"/dev/video1\"\nformat = \"mjpeg\"\nsize = [1280, 720]\n[detect]\nrgb = \"x\"\n";
        let config = Config::from_toml_str(toml_str).expect("parses");
        let err = config.validate().expect_err("duplicate id errors");
        assert_eq!(
            err.to_string(),
            ConfigError::DuplicateCamera("rgb".into()).to_string()
        );
    }

    #[test]
    fn test_empty_camera_id_errors() {
        let toml_str = "estimate = \"fused\"\n[[camera]]\nid = \"\"\ndevice = \"/dev/video0\"\nformat = \"mjpeg\"\nsize = [1280, 720]\n";
        let err = Config::from_toml_str(toml_str).expect_err("empty id errors");
        assert!(err.to_string().contains("camera id must not be empty"));
    }

    #[test]
    fn test_zero_size_camera_errors() {
        let toml_str = "estimate = \"fused\"\n[[camera]]\nid = \"rgb\"\ndevice = \"/dev/video0\"\nformat = \"mjpeg\"\nsize = [0, 720]\n[detect]\nrgb = \"x\"\n";
        let config = Config::from_toml_str(toml_str).expect("parses");
        let err = config.validate().expect_err("zero size errors");
        match err {
            ConfigError::InvalidCamera { camera, reason } => {
                assert_eq!(camera, "rgb");
                assert_eq!(reason, "size must be non-zero");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_detect_unknown_camera_errors_listing_ids() {
        let toml_str = format!("{}\n[detect]\ndepth = \"x\"\n", two_camera_and_estimate());
        let config = Config::from_toml_str(&toml_str).expect("parses");
        let err = config.validate().expect_err("unknown detect camera errors");
        match err {
            ConfigError::UnknownDetectCamera { camera, available } => {
                assert_eq!(camera, "depth");
                assert_eq!(available, vec!["ir".to_string(), "rgb".to_string()]);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_empty_detect_errors() {
        let toml_str = minimal_camera_and_estimate();
        let config = Config::from_toml_str(&toml_str).expect("parses");
        let err = config.validate().expect_err("empty detect errors");
        assert!(matches!(err, ConfigError::NoDetectors));
    }

    #[test]
    fn test_zero_channel_capacity_errors() {
        let toml_str = format!(
            "{}\n[capture]\nchannel_capacity = 0\n",
            minimal_camera_and_estimate_with_detect()
        );
        let config = Config::from_toml_str(&toml_str).expect("parses");
        let err = config.validate().expect_err("zero capacity errors");
        assert!(matches!(err, ConfigError::ZeroChannelCapacity));
    }

    #[test]
    fn test_output_kind_rejects_unknown_backend() {
        let err = toml::from_str::<OutputConfig>("kind = \"x11\"").unwrap_err();
        assert!(err.to_string().contains("unknown variant"), "{err}");
        assert_eq!(
            toml::from_str::<OutputConfig>("kind = \"layer-shell\"")
                .unwrap()
                .kind,
            OutputKind::LayerShell
        );

        let toml_str = format!(
            "{}\n[output]\nkind = \"x11\"\n",
            minimal_camera_and_estimate_with_detect()
        );
        let err = Config::from_toml_str(&toml_str).expect_err("unknown backend errors");
        assert!(err.to_string().contains("unknown variant"), "{err}");
    }

    #[test]
    fn test_output_grid_must_be_nonzero() {
        let toml_str = format!(
            "{}\n[output]\ngrid = [0, 3]\n",
            minimal_camera_and_estimate_with_detect()
        );
        let config = Config::from_toml_str(&toml_str).expect("parses");
        let err = config.validate().expect_err("bad grid errors");
        assert!(matches!(
            err,
            ConfigError::InvalidOutput("grid must be at least 1x1")
        ));
    }

    #[test]
    fn test_output_hide_margin_must_be_non_negative() {
        let toml_str = format!(
            "{}\n[output]\nhide_margin_px = -1.0\n",
            minimal_camera_and_estimate_with_detect()
        );
        let config = Config::from_toml_str(&toml_str).expect("parses");
        let err = config.validate().expect_err("negative margin errors");
        assert!(matches!(
            err,
            ConfigError::InvalidOutput("hide_margin_px must be non-negative")
        ));
    }

    #[test]
    fn test_output_hide_below_confidence_must_be_in_unit_range() {
        let toml_str = format!(
            "{}\n[output]\nhide_below_confidence = 1.5\n",
            minimal_camera_and_estimate_with_detect()
        );
        let config = Config::from_toml_str(&toml_str).expect("parses");
        let err = config
            .validate()
            .expect_err("out of range confidence errors");
        assert!(matches!(
            err,
            ConfigError::InvalidOutput("hide_below_confidence must be between 0 and 1")
        ));
    }

    #[test]
    fn test_env_camera_overrides_rgb_device() {
        let mut config = Config::builtin_default();
        let env = EnvOverrides {
            camera: Some("/dev/video4".to_string()),
            ir_camera: None,
            output: None,
        };
        let applied = config.apply_env(&env);
        assert_eq!(applied, vec!["EYE_CAMERA"]);
        assert_eq!(
            config.camera("rgb").expect("rgb exists").device,
            PathBuf::from("/dev/video4")
        );
        assert_eq!(
            config.camera("ir").expect("ir exists").device,
            PathBuf::from("/dev/video2")
        );
    }

    #[test]
    fn test_env_ir_camera_overrides_ir_device() {
        let mut config = Config::builtin_default();
        let env = EnvOverrides {
            camera: None,
            ir_camera: Some("/dev/video9".to_string()),
            output: None,
        };
        let applied = config.apply_env(&env);
        assert_eq!(applied, vec!["EYE_IR_CAMERA"]);
        assert_eq!(
            config.camera("ir").expect("ir exists").device,
            PathBuf::from("/dev/video9")
        );
    }

    #[test]
    fn test_env_override_for_missing_camera_is_ignored() {
        let toml_str = "estimate = \"fused\"\n[[camera]]\nid = \"ir\"\ndevice = \"/dev/video2\"\nformat = \"gray\"\nsize = [640, 360]\n[detect]\nir = \"ir-classic\"\n";
        let mut config = Config::from_toml_str(toml_str).expect("parses");
        let before = config.clone();
        let env = EnvOverrides {
            camera: Some("/dev/video4".to_string()),
            ir_camera: None,
            output: None,
        };
        let applied = config.apply_env(&env);
        assert_eq!(applied, Vec::<&'static str>::new());
        assert_eq!(config, before);
    }

    #[test]
    fn test_env_output_sets_target() {
        let mut config = Config::builtin_default();
        let env = EnvOverrides {
            camera: None,
            ir_camera: None,
            output: Some("DP-1".to_string()),
        };
        let applied = config.apply_env(&env);
        assert_eq!(applied, vec!["EYE_OUTPUT"]);
        assert_eq!(config.output.target, Some("DP-1".to_string()));
    }

    #[test]
    fn test_empty_env_value_is_unset() {
        let env = EnvOverrides::from_lookup(|_| Some(String::new()));
        assert_eq!(env, EnvOverrides::default());
    }

    #[test]
    fn test_default_candidates_xdg_then_cwd() {
        let candidates = default_config_candidates(Some(Path::new("/x/.config")), Path::new("/w"));
        assert_eq!(
            candidates,
            vec![
                PathBuf::from("/x/.config/eye/eye.toml"),
                PathBuf::from("/w/eye.toml"),
            ]
        );

        let candidates_no_xdg = default_config_candidates(None, Path::new("/w"));
        assert_eq!(candidates_no_xdg, vec![PathBuf::from("/w/eye.toml")]);
    }

    #[test]
    fn test_load_explicit_missing_path_errors_io() {
        let missing = unique_temp_path("missing");
        let err = Config::load_with(Some(&missing), &EnvOverrides::default())
            .expect_err("missing path errors");
        assert!(matches!(err, ConfigError::Io { .. }));
    }

    #[test]
    fn test_load_explicit_file_applies_env_then_validates() {
        let path = write_temp_toml("explicit", &two_camera_and_estimate_with_detect());
        let env = EnvOverrides {
            camera: None,
            ir_camera: Some("/dev/video9".to_string()),
            output: None,
        };
        let loaded = Config::load_with(Some(&path), &env).expect("loads and validates");
        assert_eq!(loaded.source, ConfigSource::File(path.clone()));
        assert_eq!(loaded.applied_env, vec!["EYE_IR_CAMERA"]);
        assert_eq!(
            loaded.config.camera("ir").expect("ir exists").device,
            PathBuf::from("/dev/video9")
        );
        std::fs::remove_file(&path).expect("remove temp file");
    }

    #[test]
    fn test_logs_config_loaded_at_info_and_env_override_at_debug() {
        let (_, override_records) = eye_log::testing::capture_logs(tracing::Level::DEBUG, || {
            let mut config =
                Config::from_toml_str(&two_camera_and_estimate_with_detect()).expect("parses");
            config.apply_env(&EnvOverrides {
                camera: None,
                ir_camera: Some("/dev/video9".to_string()),
                output: None,
            });
        });
        let debug_rec = override_records
            .iter()
            .find(|r| r.message == "env override applied")
            .expect("debug record present");
        assert_eq!(debug_rec.level, eye_log::Level::Debug);
        assert_eq!(
            debug_rec.fields.get("var"),
            Some(&eye_log::Value::Str("EYE_IR_CAMERA".to_string()))
        );

        let path = write_temp_toml("log-config-loaded", &two_camera_and_estimate_with_detect());
        let (_, loaded_records) = eye_log::testing::capture_logs(tracing::Level::DEBUG, || {
            Config::load_with(Some(&path), &EnvOverrides::default()).expect("loads")
        });
        std::fs::remove_file(&path).expect("remove temp file");
        let info_rec = loaded_records
            .iter()
            .find(|r| r.message == "config loaded")
            .expect("info record present");
        assert_eq!(info_rec.level, eye_log::Level::Info);
        assert_eq!(
            info_rec.fields.get("cameras"),
            Some(&eye_log::Value::U64(2))
        );
        assert_eq!(
            info_rec.fields.get("applied_env"),
            Some(&eye_log::Value::U64(0))
        );
    }

    proptest! {
        #[test]
        fn test_stage_section_split_preserves_other_keys(
            keys in proptest::collection::hash_set("[a-z_]{1,8}".prop_filter("not kind", |k| k != "kind"), 0..8),
        ) {
            let mut table = toml::Table::new();
            for (i, key) in keys.iter().enumerate() {
                table.insert(key.clone(), toml::Value::Integer(i as i64));
            }
            let mut input = table.clone();
            input.insert("kind".into(), toml::Value::String("k".into()));

            let section = StageSection::try_from(toml::Value::Table(input)).expect("parses");
            prop_assert_eq!(section.kind, "k");
            prop_assert_eq!(section.options, table);
        }
    }

    fn minimal_camera_only() -> String {
        "[[camera]]\nid = \"rgb\"\ndevice = \"/dev/video0\"\nformat = \"mjpeg\"\nsize = [1280, 720]\n"
            .to_string()
    }

    fn minimal_camera_and_estimate() -> String {
        "estimate = \"fused\"\n[[camera]]\nid = \"rgb\"\ndevice = \"/dev/video0\"\nformat = \"mjpeg\"\nsize = [1280, 720]\n".to_string()
    }

    fn minimal_camera_and_estimate_with_detect() -> String {
        format!(
            "{}\n[detect]\nrgb = \"mediapipe-ort\"\n",
            minimal_camera_and_estimate()
        )
    }

    fn two_camera_and_estimate() -> String {
        "estimate = \"fused\"\n[[camera]]\nid = \"rgb\"\ndevice = \"/dev/video0\"\nformat = \"mjpeg\"\nsize = [1280, 720]\n[[camera]]\nid = \"ir\"\ndevice = \"/dev/video2\"\nformat = \"gray\"\nsize = [640, 360]\n".to_string()
    }

    fn two_camera_and_estimate_with_detect() -> String {
        format!(
            "{}\n[detect]\nrgb = \"mediapipe-ort\"\nir = \"ir-classic\"\n",
            two_camera_and_estimate()
        )
    }

    fn minimal_valid_toml() -> String {
        minimal_camera_and_estimate_with_detect()
    }
}
