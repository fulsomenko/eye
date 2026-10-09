use eye_capture::{FrameSource, TaggedSource, TaggerConfig, V4l2Config, V4l2Source};
use eye_core::log::field;

use crate::config::{CameraFormat, Config};
use crate::tracker::TrackerError;

/// One V4L2 source per `[[camera]]`, in config order; Gray cameras are wrapped in `TaggedSource`.
pub fn open_sources(config: &Config) -> Result<Vec<Box<dyn FrameSource>>, TrackerError> {
    config
        .cameras
        .iter()
        .map(|c| {
            let mut v4l2 = V4l2Config::new(
                c.id.clone(),
                c.device.clone(),
                c.format.pixel_format(),
                c.size[0],
                c.size[1],
            );
            v4l2.fps = c.fps;
            let source = V4l2Source::open(v4l2).map_err(|source| TrackerError::Capture {
                camera: c.id.to_string(),
                source,
            })?;
            Ok(match c.format {
                CameraFormat::Gray => {
                    let tagged =
                        TaggedSource::new(source, TaggerConfig::default()).map_err(|source| {
                            TrackerError::Capture {
                                camera: c.id.to_string(),
                                source,
                            }
                        })?;
                    tracing::debug!(
                        { field::CAMERA } = c.id.as_str(),
                        "brightness tagger attached"
                    );
                    Box::new(tagged) as Box<dyn FrameSource>
                }
                CameraFormat::Mjpeg => Box::new(source),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::config::Config;
    use crate::error::ConfigError;
    use crate::pipeline::PipelineError;
    use crate::testkit::{self, TWO_CAMERA_TOML, fake_registry};
    use crate::tracker::{Tracker, TrackerError};

    use super::*;

    #[test]
    fn test_open_sources_missing_device_is_capture_error() {
        let config = Config::from_toml_str(TWO_CAMERA_TOML).expect("parses");
        let Err(TrackerError::Capture { camera, .. }) = open_sources(&config) else {
            panic!("expected a Capture error")
        };
        assert_eq!(camera, "rgb");
    }

    #[test]
    fn test_from_config_unknown_stage_fails_before_opening_cameras() {
        let toml_str = TWO_CAMERA_TOML.replace("estimate = \"fake\"", "estimate = \"nope\"");
        let config = Config::from_toml_str(&toml_str).expect("parses");
        let rig = testkit::rig();
        let Err(TrackerError::Pipeline(PipelineError::Config(ConfigError::UnknownStage {
            ..
        }))) = Tracker::from_config_with(&fake_registry(), &config, rig, None, Vec::new())
        else {
            panic!("expected a Config(UnknownStage) error")
        };
    }
}
