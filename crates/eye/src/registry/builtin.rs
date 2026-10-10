use eye_core::{
    GazePoint, Rig,
    stage::{GazeFilter, StageError},
};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NoOptions {}

#[derive(Debug, Default)]
pub struct PassThroughFilter;

impl PassThroughFilter {
    pub const NAME: &'static str = "none";

    pub fn from_config(table: &toml::Table, _rig: &Rig) -> Result<Self, StageError> {
        let NoOptions {} = table
            .clone()
            .try_into()
            .map_err(|e: toml::de::Error| StageError::Config(Box::new(e)))?;
        Ok(Self)
    }
}

impl GazeFilter for PassThroughFilter {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn apply(&mut self, point: GazePoint) -> GazePoint {
        point
    }

    fn reset(&mut self) {}
}
