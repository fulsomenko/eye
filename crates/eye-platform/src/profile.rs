//! Built-in hardware profiles, matched by DMI vendor/product, that supply the published
//! diagonal FOV per camera role and known quirks (IR frame rate, dual-stream RGB rate).
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use eye_core::log::field;

use crate::error::ProbeError;

const DEFAULT_DMI_DIR: &str = "/sys/class/dmi/id";
const BUILTIN_PROFILES_TOML: &str = include_str!("../profiles.toml");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DmiInfo {
    pub sys_vendor: String,
    pub product_name: String,
}

impl DmiInfo {
    /// `$EYE_DMI_DIR` if set, else `/sys/class/dmi/id`.
    pub fn read() -> Result<Self, ProbeError> {
        let dir = std::env::var_os("EYE_DMI_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_DMI_DIR));
        Self::read_from(&dir)
    }

    pub fn read_from(dir: &Path) -> Result<Self, ProbeError> {
        let sys_vendor = read_trimmed(&dir.join("sys_vendor"))?;
        let product_name = read_trimmed(&dir.join("product_name"))?;
        tracing::debug!(
            dir = %dir.display(),
            sys_vendor = %sys_vendor,
            product_name = %product_name,
            "dmi read"
        );
        Ok(Self {
            sys_vendor,
            product_name,
        })
    }
}

fn read_trimmed(path: &Path) -> Result<String, ProbeError> {
    fs::read_to_string(path)
        .map(|s| s.trim().to_string())
        .map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraRole {
    Rgb,
    Ir,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraProfile {
    pub role: CameraRole,
    /// Published diagonal FOV candidates in degrees: one value, or one per possible module.
    pub diag_fov_deg: Vec<f64>,
    #[serde(default)]
    pub nominal_fps: Option<f64>,
    #[serde(default)]
    pub dual_stream_rgb_fps: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareProfile {
    pub id: String,
    pub sys_vendor: String,
    /// Case-insensitive, whitespace-separated word subsequence of DMI product_name (other
    /// words, e.g. "16 inch", may appear between the matched words).
    pub product_match: String,
    pub verified: bool,
    pub source: String,
    #[serde(rename = "cameras")]
    pub cameras: Vec<CameraProfile>,
}

#[derive(Debug, serde::Deserialize)]
struct ProfilesFile {
    #[serde(rename = "profile")]
    profiles: Vec<HardwareProfile>,
}

/// Parsed once from the built-in `profiles.toml`. Panics only on a malformed built-in table.
pub fn builtin_profiles() -> &'static [HardwareProfile] {
    static PROFILES: OnceLock<Vec<HardwareProfile>> = OnceLock::new();
    PROFILES
        .get_or_init(|| {
            let file: ProfilesFile =
                toml::from_str(BUILTIN_PROFILES_TOML).expect("built-in profiles.toml parses");
            file.profiles
        })
        .as_slice()
}

/// `sys_vendor` equal ignoring ASCII case after trim, AND `product_match`'s words appear, in
/// order, among `product_name`'s words (ignoring ASCII case). First match in table order wins.
pub fn match_profile<'a>(
    profiles: &'a [HardwareProfile],
    dmi: &DmiInfo,
) -> Option<&'a HardwareProfile> {
    let vendor = dmi.sys_vendor.trim();
    let product_lower = dmi.product_name.trim().to_ascii_lowercase();
    let product_words: Vec<&str> = product_lower.split_whitespace().collect();
    let found = profiles.iter().find(|p| {
        p.sys_vendor.trim().eq_ignore_ascii_case(vendor)
            && is_word_subsequence(&product_words, &p.product_match.to_ascii_lowercase())
    });
    match found {
        Some(p) => tracing::info!(
            id = %p.id,
            verified = p.verified,
            cameras = p.cameras.len(),
            sys_vendor = vendor,
            product_name = %dmi.product_name.trim(),
            "hardware profile matched"
        ),
        None => {
            let vendor_known = profiles
                .iter()
                .any(|p| p.sys_vendor.trim().eq_ignore_ascii_case(vendor));
            tracing::debug!(
                sys_vendor = vendor,
                product_name = %dmi.product_name.trim(),
                profiles = profiles.len(),
                { field::REASON } = if vendor_known { "product_mismatch" } else { "unknown_vendor" },
                "no hardware profile for this machine"
            );
        }
    }
    found
}

fn is_word_subsequence(haystack_words: &[&str], needle: &str) -> bool {
    let mut haystack = haystack_words.iter();
    needle
        .split_whitespace()
        .all(|word| haystack.any(|h| *h == word))
}

#[cfg(test)]
mod tests {
    use eye_core::log::field;

    use super::*;

    #[test]
    fn test_builtin_table_parses() {
        let profiles = builtin_profiles();
        let ids: Vec<&str> = profiles.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, vec!["dell-latitude-7420", "hp-zbook-fury-16-g10"]);
    }

    #[test]
    fn test_match_dell_by_dmi() {
        let dmi = DmiInfo {
            sys_vendor: "Dell Inc.".to_string(),
            product_name: "Latitude 7420".to_string(),
        };
        let matched = match_profile(builtin_profiles(), &dmi).expect("matches");
        assert_eq!(matched.id, "dell-latitude-7420");
    }

    #[test]
    fn test_match_is_case_insensitive_substring() {
        let dmi = DmiInfo {
            sys_vendor: "HP".to_string(),
            product_name: "HP ZBook Fury 16 inch G10 Mobile Workstation PC".to_string(),
        };
        let matched = match_profile(builtin_profiles(), &dmi).expect("matches");
        assert_eq!(matched.id, "hp-zbook-fury-16-g10");

        let mismatched_case = DmiInfo {
            sys_vendor: "hp".to_string(),
            product_name: "hp zbook fury 16 inch g10 mobile workstation pc".to_string(),
        };
        let matched_mismatched =
            match_profile(builtin_profiles(), &mismatched_case).expect("matches ignoring case");
        assert_eq!(matched_mismatched.id, "hp-zbook-fury-16-g10");

        let dell_lowercase = DmiInfo {
            sys_vendor: "dell inc.".to_string(),
            product_name: "latitude 7420".to_string(),
        };
        let matched_dell =
            match_profile(builtin_profiles(), &dell_lowercase).expect("matches Dell ignoring case");
        assert_eq!(matched_dell.id, "dell-latitude-7420");
    }

    #[test]
    fn test_unknown_machine_has_no_profile() {
        let dmi = DmiInfo {
            sys_vendor: "LENOVO".to_string(),
            product_name: "ThinkPad X1".to_string(),
        };
        assert_eq!(match_profile(builtin_profiles(), &dmi), None);
    }

    #[test]
    fn test_dmi_read_from_trims() {
        let dir = std::env::temp_dir().join(format!(
            "eye-dmi-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("sys_vendor"), "Dell Inc.\n").unwrap();
        fs::write(dir.join("product_name"), "Latitude 7420\n").unwrap();

        let dmi = DmiInfo::read_from(&dir).expect("reads");
        assert_eq!(dmi.sys_vendor, "Dell Inc.");
        assert_eq!(dmi.product_name, "Latitude 7420");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_logs_hardware_profile_matched_at_info() {
        use eye_log::Value;

        let dmi = DmiInfo {
            sys_vendor: "Dell Inc.".to_string(),
            product_name: "Latitude 7420".to_string(),
        };
        let (matched, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            match_profile(builtin_profiles(), &dmi)
        });
        assert!(matched.is_some());

        let record = records
            .iter()
            .find(|r| r.message == "hardware profile matched")
            .expect("hardware profile matched logged");
        assert_eq!(record.level, eye_log::Level::Info);
        assert_eq!(
            record.fields.get("id"),
            Some(&Value::Str("dell-latitude-7420".to_string()))
        );
        assert!(matches!(
            record.fields.get("verified"),
            Some(Value::Bool(_))
        ));
    }

    #[test]
    fn test_logs_no_profile_at_debug_with_reason() {
        use eye_log::Value;

        let unknown_vendor = DmiInfo {
            sys_vendor: "LENOVO".to_string(),
            product_name: "ThinkPad X1".to_string(),
        };
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            match_profile(builtin_profiles(), &unknown_vendor)
        });
        let record = records
            .iter()
            .find(|r| r.message == "no hardware profile for this machine")
            .expect("no profile logged");
        assert_eq!(record.level, eye_log::Level::Debug);
        assert_eq!(
            record.fields.get(field::REASON),
            Some(&Value::Str("unknown_vendor".to_string()))
        );

        let product_mismatch = DmiInfo {
            sys_vendor: "Dell Inc.".to_string(),
            product_name: "XPS 13".to_string(),
        };
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            match_profile(builtin_profiles(), &product_mismatch)
        });
        let record = records
            .iter()
            .find(|r| r.message == "no hardware profile for this machine")
            .expect("no profile logged");
        assert_eq!(record.level, eye_log::Level::Debug);
        assert_eq!(
            record.fields.get(field::REASON),
            Some(&Value::Str("product_mismatch".to_string()))
        );
    }

    #[test]
    fn test_logs_dmi_read_at_debug() {
        use eye_log::Value;

        let dir = std::env::temp_dir().join(format!(
            "eye-dmi-log-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("sys_vendor"), "Dell Inc.\n").unwrap();
        fs::write(dir.join("product_name"), "Latitude 7420\n").unwrap();

        let (result, records) =
            eye_log::testing::capture_logs(tracing::Level::TRACE, || DmiInfo::read_from(&dir));
        result.expect("reads");

        let record = records
            .iter()
            .find(|r| r.message == "dmi read")
            .expect("dmi read logged");
        assert_eq!(record.level, eye_log::Level::Debug);
        assert_eq!(
            record.fields.get("sys_vendor"),
            Some(&Value::Str("Dell Inc.".to_string()))
        );
        assert_eq!(
            record.fields.get("product_name"),
            Some(&Value::Str("Latitude 7420".to_string()))
        );
        assert_eq!(
            record.fields.get("dir"),
            Some(&Value::Str(dir.display().to_string()))
        );

        fs::remove_dir_all(&dir).unwrap();
    }
}
