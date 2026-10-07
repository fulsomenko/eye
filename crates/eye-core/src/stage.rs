/// Error shared by every pipeline stage trait. Implementation crates convert
/// their own error enums into it at the trait boundary.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StageError {
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("stage failed: {0}")]
    Failed(String),
    #[error("stage closed")]
    Closed,
}
