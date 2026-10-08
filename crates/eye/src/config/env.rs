#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EnvOverrides {
    pub camera: Option<String>,
    pub ir_camera: Option<String>,
    pub output: Option<String>,
}

impl EnvOverrides {
    pub fn from_process_env() -> Self {
        Self::from_lookup(|k| std::env::var(k).ok())
    }

    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Self {
        let get = |k: &str| lookup(k).filter(|v| !v.is_empty());
        Self {
            camera: get("EYE_CAMERA"),
            ir_camera: get("EYE_IR_CAMERA"),
            output: get("EYE_OUTPUT"),
        }
    }
}
