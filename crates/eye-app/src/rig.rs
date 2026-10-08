//! Picks the display output, resolves the active `Rig` (stored or nominal), and lists
//! IR-emitter-guarded cameras, all before any camera or compositor connection is opened.

use std::path::PathBuf;

use anyhow::Context as _;
use eye::config::Config;
use eye_calibration::nominal::{CameraStream, NominalRigConfig, nominal_rig};
use eye_calibration::store::ProfileStore;
use eye_core::Rig;
use eye_platform::OutputInfo;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RigSource {
    Stored,
    Nominal,
}

/// `[output].target` (or `EYE_OUTPUT`) if set; else the only probed output.
pub fn target_output<'a>(
    config: &Config,
    outputs: &'a [OutputInfo],
) -> anyhow::Result<&'a OutputInfo> {
    let names = || {
        outputs
            .iter()
            .map(|o| o.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    match (&config.output.target, outputs) {
        (Some(target), _) => outputs.iter().find(|o| &o.name == target).ok_or_else(|| {
            anyhow::anyhow!("output \"{target}\" not found; probed outputs: {}", names())
        }),
        (None, [only]) => Ok(only),
        (None, []) => anyhow::bail!("no display outputs probed; is this a Wayland session?"),
        (None, _) => anyhow::bail!(
            "several outputs ({}); set [output] target in eye.toml or EYE_OUTPUT",
            names()
        ),
    }
}

/// One `CameraStream` per `[[camera]]`, in config order.
pub fn camera_streams(config: &Config) -> Vec<CameraStream> {
    config
        .cameras
        .iter()
        .map(|c| CameraStream {
            id: c.id.clone(),
            width: c.size[0],
            height: c.size[1],
        })
        .collect()
}

/// `config.rig` (the `[rig]` table, absent = empty) deserialized into `NominalRigConfig`.
pub fn nominal_rig_config(config: &Config) -> anyhow::Result<NominalRigConfig> {
    NominalRigConfig::from_table(config.rig.as_ref()).context("[rig] section")
}

/// The stored rig for `output` if the store has one, else `nominal_rig(output.screen_model()?, ..)`.
pub fn resolve_rig(
    config: &Config,
    output: &OutputInfo,
    store: &ProfileStore,
) -> anyhow::Result<(Rig, RigSource)> {
    let (rig, source) = match store.load_rig(&output.id())? {
        Some(rig) => (rig, RigSource::Stored),
        None => (
            nominal_rig(
                output.screen_model()?,
                &camera_streams(config),
                &nominal_rig_config(config)?,
            )?,
            RigSource::Nominal,
        ),
    };
    tracing::info!(source = ?source, output = %output.name, "rig");
    Ok((rig, source))
}

/// `(id, device)` of every `[[camera]]` with `emitter = "msxu"`, in config order; input to
/// `commands::emitter::emitter_guards`.
pub fn msxu_cameras(config: &Config) -> Vec<(String, PathBuf)> {
    config
        .cameras
        .iter()
        .filter(|c| c.emitter.as_deref() == Some("msxu"))
        .map(|c| (c.id.to_string(), c.device.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use eye_core::CameraId;

    use super::*;
    use crate::testing::edp1;

    fn hdmi() -> OutputInfo {
        let mut o = edp1();
        o.name = "HDMI-A-1".to_string();
        o
    }

    fn config_with_toml(extra: &str) -> Config {
        let base = "estimate = \"fused\"\n[[camera]]\nid = \"rgb\"\ndevice = \"/dev/video0\"\nformat = \"mjpeg\"\nsize = [1280, 720]\n[detect]\nrgb = \"mediapipe-ort\"\n";
        Config::from_toml_str(&format!("{base}\n{extra}")).expect("parses")
    }

    #[test]
    fn test_target_output_prefers_config_then_single_output() {
        let outputs = vec![edp1(), hdmi()];

        let config = config_with_toml("[output]\ntarget = \"HDMI-A-1\"\n");
        let picked = target_output(&config, &outputs).expect("target resolves");
        assert_eq!(picked.name, "HDMI-A-1");

        let config_no_target = config_with_toml("");
        let single = [edp1()];
        let picked = target_output(&config_no_target, &single).expect("single output picked");
        assert_eq!(picked.name, "eDP-1");

        let err = target_output(&config_no_target, &outputs).expect_err("several outputs error");
        assert_eq!(
            err.to_string(),
            "several outputs (eDP-1, HDMI-A-1); set [output] target in eye.toml or EYE_OUTPUT"
        );
    }

    #[test]
    fn test_target_output_unknown_target_errors() {
        let config = config_with_toml("[output]\ntarget = \"HDMI-A-1\"\n");
        let single = [edp1()];
        let err = target_output(&config, &single).expect_err("unknown target errors");
        assert_eq!(
            err.to_string(),
            "output \"HDMI-A-1\" not found; probed outputs: eDP-1"
        );
    }

    #[test]
    fn test_target_output_without_outputs_errors() {
        let config = config_with_toml("");
        let err = target_output(&config, &[]).expect_err("no outputs errors");
        assert!(err.to_string().contains("no display outputs probed"));
    }

    #[test]
    fn test_camera_streams_from_config() {
        let config = Config::builtin_default();
        let streams = camera_streams(&config);
        assert_eq!(
            streams,
            vec![
                CameraStream {
                    id: CameraId::from("rgb"),
                    width: 1280,
                    height: 720,
                },
                CameraStream {
                    id: CameraId::from("ir"),
                    width: 640,
                    height: 360,
                },
            ]
        );
    }

    #[test]
    fn test_nominal_rig_config_rejects_unknown_key() {
        let config = config_with_toml("[rig]\nbogus = 1\n");
        let err = nominal_rig_config(&config).expect_err("unknown key errors");
        let chain: String = format!("{err:#}");
        assert!(chain.contains("bogus"), "chain was: {chain}");

        let config_without_rig = config_with_toml("");
        assert!(nominal_rig_config(&config_without_rig).is_ok());
    }

    #[test]
    fn test_resolve_rig_prefers_stored() {
        let config = Config::builtin_default();
        let output = edp1();
        let dir = tempfile::tempdir().expect("tempdir");
        let store = ProfileStore::at(dir.path());

        let (rig, source) = resolve_rig(&config, &output, &store).expect("resolves nominal");
        assert_eq!(source, RigSource::Nominal);
        assert_eq!(rig.screen().output.as_str(), "eDP-1");
        assert_eq!(rig.screen().size_px, (3840, 2160));

        store.save_rig(&rig).expect("save rig");

        let (rig2, source2) = resolve_rig(&config, &output, &store).expect("resolves stored");
        assert_eq!(source2, RigSource::Stored);
        assert_eq!(rig2.screen(), rig.screen());
    }

    #[test]
    fn test_msxu_cameras_lists_emitter_cameras() {
        let config = Config::builtin_default();
        assert_eq!(
            msxu_cameras(&config),
            vec![("ir".to_string(), PathBuf::from("/dev/video2"))]
        );

        let config_no_emitter = config_with_toml("");
        assert_eq!(msxu_cameras(&config_no_emitter), Vec::new());
    }
}
