use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum ProbeError {
    #[error("not a Wayland session (WAYLAND_DISPLAY unset); X11 and tty are out of scope")]
    NotWayland,
    #[error("not running under Hyprland (HYPRLAND_INSTANCE_SIGNATURE unset)")]
    NotHyprland,
    #[error("XDG_RUNTIME_DIR is unset")]
    NoRuntimeDir,
    #[error("I/O on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Hyprland IPC replied with non-JSON: {reply:?}")]
    Hyprland { reply: String },
    #[error("Hyprland IPC JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("output {output} reports no physical size")]
    MissingPhysicalSize { output: String },
    #[error("output {output} has invalid transform {value}")]
    InvalidTransform { output: String, value: u32 },
    #[error("output {name} not found; available: {available:?}")]
    OutputNotFound {
        name: String,
        available: Vec<String>,
    },
    #[error("Wayland: {reason}")]
    Wayland { reason: String },
    #[error("malformed USB descriptor at byte {offset}: {reason}")]
    Descriptor { offset: usize, reason: &'static str },
    #[error("V4L2 {op} on {node}: {source}")]
    V4l2 {
        node: PathBuf,
        op: &'static str,
        #[source]
        source: std::io::Error,
    },
}
