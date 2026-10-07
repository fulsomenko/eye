#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GeometryError {
    #[error("need at least {need} observations, got {got}")]
    TooFewObservations { need: usize, got: usize },
    #[error("point is behind the camera")]
    BehindCamera,
    #[error("degenerate configuration: {0}")]
    Degenerate(&'static str),
    #[error("solver did not converge: {0}")]
    NotConverged(String),
    #[error("fitted parameters are not observable (J^T J is singular)")]
    Unobservable,
}
