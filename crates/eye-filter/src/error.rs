#[derive(Debug, thiserror::Error)]
pub enum FilterError {
    #[error("invalid filter config: {0}")]
    Config(#[from] toml::de::Error),
    #[error("invalid parameter `{name}`: {reason}")]
    Param {
        name: &'static str,
        reason: &'static str,
    },
}
