use std::path::{Path, PathBuf};

use anyhow::Context;
use eye_platform::{
    CameraDevice, CameraKind, EmitterError, IrEmitter, MsxuIrEmitter, find_face_auth_control,
};

use crate::commands::probe::Probes;
use crate::ctx::Ctx;

#[derive(Debug, clap::Args)]
pub struct Args {
    #[arg(value_enum)]
    pub action: Action,
    /// IR video node. Default: $EYE_IR_CAMERA, else the first IR camera with an MSXU face-auth control.
    #[arg(long, value_name = "PATH")]
    pub device: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Action {
    On,
    Off,
    Status,
}

pub fn select_camera<'a>(
    flag: Option<&Path>,
    env: Option<&str>,
    cameras: &'a [CameraDevice],
) -> anyhow::Result<&'a CameraDevice> {
    if let Some(path) = flag.or_else(|| env.map(Path::new)) {
        return cameras
            .iter()
            .find(|c| c.node == path)
            .ok_or_else(|| anyhow::anyhow!("no V4L2 camera at {}", path.display()));
    }
    cameras
        .iter()
        .find(|c| c.kind == CameraKind::Ir && find_face_auth_control(&c.extension_units).is_some())
        .ok_or_else(|| {
            anyhow::anyhow!("no IR camera with a face-authentication control found; pass --device")
        })
}

pub fn apply(emitter: &mut dyn IrEmitter, action: Action) -> anyhow::Result<bool> {
    match action {
        Action::Status => Ok(emitter.is_enabled()?),
        Action::On | Action::Off => {
            let want = action == Action::On;
            emitter.set_enabled(want)?;
            let got = emitter.is_enabled()?;
            anyhow::ensure!(
                got == want,
                "emitter did not switch {}",
                if want { "on" } else { "off" }
            );
            Ok(got)
        }
    }
}

pub fn emitter_guards<G>(
    cameras: &[(String, PathBuf)],
    probed: &[CameraDevice],
    enable: impl Fn(&CameraDevice) -> Result<G, EmitterError>,
) -> anyhow::Result<Vec<G>> {
    cameras
        .iter()
        .map(|(id, node)| {
            let device = probed.iter().find(|c| &c.node == node).ok_or_else(|| {
                anyhow::anyhow!("no V4L2 camera at {} (camera \"{id}\")", node.display())
            })?;
            enable(device).with_context(|| format!("enabling the IR emitter of camera \"{id}\""))
        })
        .collect()
}

pub fn run(ctx: &Ctx, args: Args) -> anyhow::Result<()> {
    ctx.reject_output("emitter")?;
    let env = std::env::var("EYE_IR_CAMERA").ok();
    let probes = Probes::system();
    let cameras = probes.cameras.cameras().context("probing V4L2 cameras")?;
    let camera = select_camera(args.device.as_deref(), env.as_deref(), &cameras)?;
    let mut emitter = MsxuIrEmitter::discover(camera).context("opening the IR emitter")?;
    let state = apply(&mut emitter, args.action)?;
    println!("{}", if state { "on" } else { "off" });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{FakeEmitter, latitude_cameras};

    #[test]
    fn test_select_camera_order_flag_env_probe() {
        let cameras = latitude_cameras();

        let c = select_camera(
            Some(Path::new("/dev/video0")),
            Some("/dev/video2"),
            &cameras,
        )
        .unwrap();
        assert_eq!(c.node, PathBuf::from("/dev/video0"));

        let c = select_camera(None, Some("/dev/video0"), &cameras).unwrap();
        assert_eq!(c.node, PathBuf::from("/dev/video0"));

        let c = select_camera(None, None, &cameras).unwrap();
        assert_eq!(c.node, PathBuf::from("/dev/video2"));
    }

    #[test]
    fn test_select_camera_unknown_node_errors() {
        let cameras = latitude_cameras();
        let err = select_camera(Some(Path::new("/dev/video9")), None, &cameras).unwrap_err();
        assert_eq!(err.to_string(), "no V4L2 camera at /dev/video9");
    }

    #[test]
    fn test_select_camera_without_ir_errors() {
        let mut cameras = latitude_cameras();
        cameras.truncate(1);
        let err = select_camera(None, None, &cameras).unwrap_err();
        assert!(err.to_string().contains("--device"));
    }

    #[test]
    fn test_apply_on_reads_back_state() {
        let mut emitter = FakeEmitter::new(false);
        let calls = emitter.calls.clone();

        let result = apply(&mut emitter, Action::On).unwrap();

        assert!(result);
        assert_eq!(*calls.lock().unwrap(), vec![true]);
    }

    #[test]
    fn test_apply_on_fails_when_state_does_not_change() {
        let mut emitter = FakeEmitter::new(false);
        emitter.ignore_writes = true;

        let err = apply(&mut emitter, Action::On).unwrap_err();

        assert_eq!(err.to_string(), "emitter did not switch on");
    }

    #[test]
    fn test_emitter_guards_match_probed_nodes() {
        let probed = latitude_cameras();

        let cameras = vec![("ir".to_string(), PathBuf::from("/dev/video2"))];
        let result = emitter_guards(&cameras, &probed, |c: &CameraDevice| {
            Ok::<_, EmitterError>(c.node.clone())
        })
        .unwrap();
        assert_eq!(result, vec![PathBuf::from("/dev/video2")]);

        let cameras = vec![("ir".to_string(), PathBuf::from("/dev/video9"))];
        let err = emitter_guards(&cameras, &probed, |c: &CameraDevice| {
            Ok::<_, EmitterError>(c.node.clone())
        })
        .unwrap_err();
        assert!(err.to_string().contains("no V4L2 camera at /dev/video9"));

        let empty: Vec<(String, PathBuf)> = vec![];
        let result = emitter_guards(&empty, &probed, |c: &CameraDevice| {
            Ok::<_, EmitterError>(c.node.clone())
        })
        .unwrap();
        assert!(result.is_empty());

        let cameras = vec![("ir".to_string(), PathBuf::from("/dev/video2"))];
        let err = emitter_guards(
            &cameras,
            &probed,
            |c: &CameraDevice| -> Result<PathBuf, EmitterError> {
                Err(EmitterError::NoControl {
                    node: c.node.clone(),
                })
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("camera \"ir\""));
    }
}
