use super::{Registry, builtin::PassThroughFilter};
use crate::error::ConfigError;

pub(super) fn register_defaults(r: &mut Registry) -> Result<(), ConfigError> {
    r.register_filter("none", |o, rig| {
        Ok(Box::new(PassThroughFilter::from_config(o, rig)?))
    })?;
    Ok(())
}
