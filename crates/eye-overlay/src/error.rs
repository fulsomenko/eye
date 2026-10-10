use smithay_client_toolkit::reexports::client::{
    ConnectError,
    globals::{BindError, GlobalError as WlGlobalError},
};

#[derive(Debug, thiserror::Error)]
pub enum OverlayError {
    #[error("connecting to the Wayland display: {0}")]
    Connect(#[from] ConnectError),
    #[error("reading Wayland globals: {0}")]
    Globals(#[from] WlGlobalError),
    #[error("compositor lacks {interface}: {source}")]
    MissingGlobal {
        interface: &'static str,
        #[source]
        source: BindError,
    },
    #[error("creating the input region: {0}")]
    Region(#[from] smithay_client_toolkit::error::GlobalError),
    #[error("output {wanted:?} not found; available: {available:?}")]
    OutputNotFound {
        wanted: String,
        available: Vec<String>,
    },
    #[error("output {0:?} disappeared")]
    OutputGone(String),
    #[error("compositor closed the layer surface")]
    SurfaceClosed,
    #[error("shm: {0}")]
    Shm(String),
    #[error("wayland dispatch: {0}")]
    Dispatch(String),
    #[error("event loop: {0}")]
    EventLoop(String),
    #[error("overlay thread exited")]
    Closed,
    #[error("overlay thread panicked")]
    Panicked,
    #[error("invalid grid {cols}x{rows}: both must be at least 1")]
    InvalidGrid { cols: u32, rows: u32 },
    #[error("spawning the overlay thread: {0}")]
    Spawn(#[source] std::io::Error),
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    #[test]
    fn test_overlay_error_spawn_displays_io_message() {
        let err = OverlayError::Spawn(std::io::Error::other("no threads"));
        assert_eq!(err.to_string(), "spawning the overlay thread: no threads");
        assert!(err.source().is_some());
    }
}
