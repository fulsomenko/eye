use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ctx {
    pub config_path: Option<PathBuf>,
    pub output: Option<PathBuf>,
}

impl Ctx {
    pub fn new(cli_config: Option<PathBuf>, output: Option<PathBuf>) -> anyhow::Result<Self> {
        if let Some(path) = &cli_config {
            anyhow::ensure!(path.is_file(), "config file not found: {}", path.display());
        }
        Ok(Self {
            config_path: cli_config,
            output,
        })
    }

    pub fn reject_output(&self, command: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.output.is_none(),
            "eye {command} does not write a file; remove --output"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ctx_existing_explicit_config_is_kept() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let ctx = Ctx::new(Some(file.path().to_path_buf()), None).unwrap();
        assert_eq!(ctx.config_path, Some(file.path().to_path_buf()));

        let ctx = Ctx::new(None, None).unwrap();
        assert_eq!(ctx.config_path, None);
    }

    #[test]
    fn test_ctx_explicit_missing_config_is_error() {
        let err = Ctx::new(Some(PathBuf::from("/nonexistent/eye.toml")), None).unwrap_err();
        assert!(err.to_string().contains("config file not found"));
    }

    #[test]
    fn test_reject_output_errors_when_set() {
        let ctx = Ctx {
            config_path: None,
            output: Some(PathBuf::from("x")),
        };
        let err = ctx.reject_output("run").unwrap_err();
        assert!(err.to_string().contains("eye run"));

        let ctx = Ctx {
            config_path: None,
            output: None,
        };
        assert!(ctx.reject_output("run").is_ok());
    }
}
