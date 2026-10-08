pub mod emitter_mode;
pub mod persistence;
pub mod tagging;

use crate::{
    case::{TestCtx, TestError, TestRegistry},
    mode::EmitterSetting,
};

/// Expected outcome of an E1 row. `Report` measures without a limit: the row is not measured yet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Expect {
    #[default]
    Persist,
    Reset,
    Report,
}

pub fn require_emitter(
    ctx: &TestCtx,
    want: EmitterSetting,
    case: &str,
    setting: &str,
) -> Result<(), TestError> {
    if ctx.mode().emitter == want {
        Ok(())
    } else {
        Err(TestError::Other(format!(
            "{case} needs emitter = \"{setting}\""
        )))
    }
}

pub fn register(r: &mut TestRegistry) {
    r.register(
        "emitter_persistence",
        "E1: emitter byte survives reopen / RGB cycling / autosuspend without re-applying",
        persistence::build,
    );
    r.register(
        "ir_tagging_agreement",
        "E6: lit/dark tags agree with per-frame brightness; parity; margins",
        tagging::build,
    );
    r.register(
        "emitter_mode",
        "E1 human procedure: emitter byte left by an earlier event",
        emitter_mode::build,
    );
}
