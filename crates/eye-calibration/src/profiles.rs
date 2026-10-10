//! Per-user calibration profile storage: `profiles/<name>.toml` under a
//! [`ProfileStore`], plus explicit-path read/write for `eye calibrate --output`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::correction::{
    AngularCorrection, CalibrationPose, EyeKey, PROFILE_VERSION, Provenance, UserProfile,
    legacy_source,
};
use crate::error::{CalibrationError, StoreError};
use crate::store::{ProfileStore, read_table, validate_name, write_atomic};

impl ProfileStore {
    pub fn profile_path(&self, name: &str) -> Result<PathBuf, CalibrationError> {
        validate_name(name)?;
        Ok(self.root().join("profiles").join(format!("{name}.toml")))
    }

    pub fn load_profile(&self, name: &str) -> Result<Option<UserProfile>, CalibrationError> {
        let path = self.profile_path(name)?;
        let Some(table) = read_table(&path)? else {
            tracing::debug!(name, path = %path.display(), "profile not found");
            return Ok(None);
        };
        Ok(Some(profile_from_table(table, &path)?))
    }

    pub fn save_profile(
        &self,
        name: &str,
        profile: &UserProfile,
    ) -> Result<PathBuf, CalibrationError> {
        let path = self.profile_path(name)?;
        write_profile(&path, profile)?;
        Ok(path)
    }

    /// Stems of `profiles/*.toml`, sorted; a crashed write's `*.toml.tmp` is
    /// ignored; a missing directory gives `[]`.
    pub fn list_profiles(&self) -> Result<Vec<String>, CalibrationError> {
        let dir = self.root().join("profiles");
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(source) => {
                return Err(StoreError::Io { path: dir, source }.into());
            }
        };
        let mut names = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|source| StoreError::Io {
                path: dir.clone(),
                source,
            })?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("toml")
                && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
            {
                names.push(stem.to_owned());
            }
        }
        names.sort();
        tracing::debug!(dir = %dir.display(), count = names.len() as u64, "profiles listed");
        Ok(names)
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct UserProfileV1 {
    #[serde(rename = "version")]
    _version: u32,
    name: String,
    created_unix_s: u64,
    rig_fingerprint: String,
    estimator: String,
    eyes: BTreeMap<EyeKey, AngularCorrection>,
    #[serde(default)]
    calibration_pose: Option<CalibrationPose>,
    #[serde(default)]
    provenance: Provenance,
}

impl From<UserProfileV1> for UserProfile {
    fn from(v1: UserProfileV1) -> Self {
        let corrections = if v1.eyes.is_empty() {
            BTreeMap::new()
        } else {
            BTreeMap::from([(legacy_source(&v1.estimator), v1.eyes)])
        };
        UserProfile {
            version: PROFILE_VERSION,
            name: v1.name,
            created_unix_s: v1.created_unix_s,
            rig_fingerprint: v1.rig_fingerprint,
            estimator: v1.estimator,
            corrections,
            calibration_pose: v1.calibration_pose,
            provenance: v1.provenance,
        }
    }
}

fn profile_version(table: &toml::Table) -> Result<u32, StoreError> {
    match table.get("version").and_then(toml::Value::as_integer) {
        None | Some(1) => Ok(1),
        Some(2) => Ok(2),
        Some(found) => Err(StoreError::UnsupportedVersion {
            what: "profile",
            found: u32::try_from(found).unwrap_or(u32::MAX),
        }),
    }
}

fn profile_from_table(table: toml::Table, path: &Path) -> Result<UserProfile, CalibrationError> {
    let profile = match profile_version(&table)? {
        1 => {
            let v1: UserProfileV1 = table.try_into().map_err(|source| StoreError::Parse {
                path: path.to_owned(),
                source,
            })?;
            let source = legacy_source(&v1.estimator);
            let profile = UserProfile::from(v1);
            tracing::info!(
                path = %path.display(),
                from_version = 1u64,
                source = source.as_str(),
                "profile migrated"
            );
            profile
        }
        _ => table.try_into().map_err(|source| StoreError::Parse {
            path: path.to_owned(),
            source,
        })?,
    };
    tracing::info!(
        path = %path.display(),
        name = %profile.name,
        estimator = %profile.estimator,
        rig_fingerprint = %profile.rig_fingerprint,
        sources = profile.corrections.len() as u64,
        corrections = profile.correction_count() as u64,
        "profile loaded"
    );
    Ok(profile)
}

pub fn write_profile(path: &Path, profile: &UserProfile) -> Result<(), CalibrationError> {
    let contents = toml::to_string_pretty(profile).map_err(StoreError::from)?;
    write_atomic(path, &contents)?;
    tracing::info!(
        path = %path.display(),
        name = %profile.name,
        sources = profile.corrections.len() as u64,
        corrections = profile.correction_count() as u64,
        "profile written"
    );
    Ok(())
}

pub fn read_profile(path: &Path) -> Result<UserProfile, CalibrationError> {
    let table = read_table(path)?.ok_or_else(|| StoreError::Io {
        path: path.to_owned(),
        source: std::io::Error::new(std::io::ErrorKind::NotFound, "profile file not found"),
    })?;
    profile_from_table(table, path)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use eye_core::{CameraId, GazeRay, OutputId, RaySource, ScreenModel, Side};
    use eye_geometry::angles::direction_from_yaw_pitch;
    use eye_geometry::synth::SplitMix64;
    use nalgebra::{Matrix2, Matrix3, Point2, Point3, Unit, Vector2, Vector3};

    use super::*;
    use crate::correction::{AngularCorrection, CorrectionModel, EyeKey};
    use crate::nominal::{CameraStream, NominalRigConfig, nominal_rig};
    use crate::user_fit::{DotSessionFit, FitSample};
    use eye_core::stage::GazeCorrection;

    fn test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("eye-profiles-{}-{name}", std::process::id()))
    }

    fn fixture_profile() -> UserProfile {
        let mut eyes = BTreeMap::new();
        eyes.insert(
            EyeKey::Left,
            AngularCorrection {
                theta: [0.01, -0.05, 1e-17, -0.02, 0.0, 0.03],
                cov: {
                    let mut cov = [[0.0; 6]; 6];
                    for (i, row) in cov.iter_mut().enumerate() {
                        for (j, v) in row.iter_mut().enumerate() {
                            *v = (i * 6 + j) as f64 * 1e-6;
                        }
                    }
                    cov
                },
                quad: [0.0; 6],
                quad_cov: [[0.0; 6]; 6],
                quad_cross_cov: [[0.0; 6]; 6],
                model: CorrectionModel::Affine,
                targets_used: 9,
                rms_after_rad: 0.004,
            },
        );
        eyes.insert(
            EyeKey::Right,
            AngularCorrection {
                theta: [0.02, -0.04, 2e-17, -0.03, 0.01, 0.02],
                cov: {
                    let mut cov = [[0.0; 6]; 6];
                    for (i, row) in cov.iter_mut().enumerate() {
                        for (j, v) in row.iter_mut().enumerate() {
                            *v = (i * 6 + j) as f64 * 1e-6 + 1.0;
                        }
                    }
                    cov
                },
                quad: [0.0; 6],
                quad_cov: [[0.0; 6]; 6],
                quad_cross_cov: [[0.0; 6]; 6],
                model: CorrectionModel::OffsetOnly,
                targets_used: 9,
                rms_after_rad: 0.004,
            },
        );
        eyes.insert(
            EyeKey::Cyclopean,
            AngularCorrection {
                theta: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                cov: {
                    let mut cov = [[0.0; 6]; 6];
                    for (i, row) in cov.iter_mut().enumerate() {
                        for (j, v) in row.iter_mut().enumerate() {
                            *v = (i * 6 + j) as f64 * 1e-6 + 2.0;
                        }
                    }
                    cov
                },
                quad: [0.0; 6],
                quad_cov: [[0.0; 6]; 6],
                quad_cross_cov: [[0.0; 6]; 6],
                model: CorrectionModel::Affine,
                targets_used: 9,
                rms_after_rad: 0.004,
            },
        );
        UserProfile {
            version: PROFILE_VERSION,
            name: "default".into(),
            created_unix_s: 1_791_400_000,
            rig_fingerprint: "0123456789abcdef".into(),
            estimator: "ir-pupil".into(),
            corrections: BTreeMap::from([(legacy_source("ir-pupil"), eyes)]),
            calibration_pose: None,
            provenance: crate::correction::Provenance::default(),
        }
    }

    #[test]
    fn test_profile_round_trip() {
        let dir = test_dir("round-trip");
        let store = ProfileStore::at(&dir);
        let profile = fixture_profile();

        let path = store.save_profile("default", &profile).unwrap();
        assert_eq!(path, dir.join("profiles").join("default.toml"));
        let back = store.load_profile("default").unwrap();
        assert_eq!(back, Some(profile.clone()));

        let explicit = dir.join("p.toml");
        write_profile(&explicit, &profile).unwrap();
        let back = read_profile(&explicit).unwrap();
        assert_eq!(back, profile);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_missing_profile_returns_none() {
        let dir = test_dir("missing");
        std::fs::create_dir_all(&dir).unwrap();
        let store = ProfileStore::at(&dir);
        assert_eq!(store.load_profile("default").unwrap(), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_logs_profile_loaded_at_info() {
        let dir = test_dir("logs-loaded");
        let store = ProfileStore::at(&dir);
        let profile = fixture_profile();

        let (_, records) = eye_log::testing::capture_logs(tracing::Level::INFO, || {
            store.save_profile("p", &profile).unwrap();
            store.load_profile("p").unwrap();
        });

        let written = records
            .iter()
            .find(|r| r.message == "profile written")
            .expect("no 'profile written' record");
        assert_eq!(written.level, eye_log::Level::Info);
        assert_eq!(
            written.fields.get("name"),
            Some(&eye_log::Value::Str(profile.name.clone()))
        );
        assert_eq!(
            written.fields.get("corrections"),
            Some(&eye_log::Value::U64(profile.correction_count() as u64))
        );

        let loaded = records
            .iter()
            .find(|r| r.message == "profile loaded")
            .expect("no 'profile loaded' record");
        assert_eq!(loaded.level, eye_log::Level::Info);
        assert_eq!(
            loaded.fields.get("name"),
            Some(&eye_log::Value::Str(profile.name.clone()))
        );
        assert_eq!(
            loaded.fields.get("sources"),
            Some(&eye_log::Value::U64(profile.corrections.len() as u64))
        );
        assert_eq!(
            loaded.fields.get("corrections"),
            Some(&eye_log::Value::U64(profile.correction_count() as u64))
        );
        assert!(matches!(
            loaded.fields.get("estimator"),
            Some(eye_log::Value::Str(_))
        ));
        assert!(matches!(
            loaded.fields.get("rig_fingerprint"),
            Some(eye_log::Value::Str(_))
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_logs_profile_not_found_at_debug() {
        let dir = test_dir("logs-not-found");
        std::fs::create_dir_all(&dir).unwrap();
        let store = ProfileStore::at(&dir);

        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            store.load_profile("missing-name").unwrap()
        });

        assert!(!records.iter().any(|r| r.message == "profile loaded"));
        let rec = records
            .iter()
            .find(|r| r.message == "profile not found")
            .expect("no 'profile not found' record");
        assert_eq!(rec.level, eye_log::Level::Debug);
        assert_eq!(
            rec.fields.get("name"),
            Some(&eye_log::Value::Str("missing-name".to_string()))
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_logs_profiles_listed_at_debug() {
        let dir = test_dir("logs-listed");
        let store = ProfileStore::at(&dir);
        store.save_profile("p", &fixture_profile()).unwrap();

        let (_, records) = eye_log::testing::capture_logs(tracing::Level::DEBUG, || {
            store.list_profiles().unwrap()
        });

        let rec = records
            .iter()
            .find(|r| r.message == "profiles listed")
            .expect("no 'profiles listed' record");
        assert_eq!(rec.level, eye_log::Level::Debug);
        assert_eq!(rec.fields.get("count"), Some(&eye_log::Value::U64(1)));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_profile_without_new_blocks_loads_with_defaults() {
        let dir = test_dir("no-new-blocks");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p.toml");
        std::fs::write(
            &path,
            r#"version = 1
name = "max"
created_unix_s = 1700000000
rig_fingerprint = "deadbeefcafef00d"
estimator = "ir-pupil"
[eyes]
"#,
        )
        .unwrap();
        let profile = read_profile(&path).unwrap();
        assert_eq!(profile.calibration_pose, None);
        assert_eq!(profile.provenance, crate::correction::Provenance::default());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_unsupported_profile_version_rejected() {
        let dir = test_dir("unsupported-version");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p.toml");
        std::fs::write(&path, "version = 3\nfoo = 1\n").unwrap();
        let err = read_profile(&path).unwrap_err();
        assert!(matches!(
            err,
            CalibrationError::Store(StoreError::UnsupportedVersion {
                what: "profile",
                found: 3
            })
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_v1_fused_profile_loads_into_rgb_only() {
        let dir = test_dir("v1-fused");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p.toml");
        std::fs::write(
            &path,
            r#"version = 1
name = "max"
created_unix_s = 1700000000
rig_fingerprint = "deadbeefcafef00d"
estimator = "fused"
[eyes.left]
theta = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
cov = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
model = "affine"
targets_used = 9
rms_after_rad = 0.001
"#,
        )
        .unwrap();

        let (profile, records) =
            eye_log::testing::capture_logs(tracing::Level::INFO, || read_profile(&path).unwrap());

        assert_eq!(profile.version, PROFILE_VERSION);
        assert_eq!(profile.corrections.len(), 1);
        let entry = profile
            .correction(RaySource::RgbOnly, EyeKey::Left)
            .unwrap();
        assert_eq!(entry.theta, [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(
            profile
                .correction(RaySource::IrOnly, EyeKey::Left)
                .is_none()
        );

        let rec = records
            .iter()
            .find(|r| r.message == "profile migrated")
            .expect("no 'profile migrated' record");
        assert_eq!(rec.level, eye_log::Level::Info);
        assert_eq!(
            rec.fields.get("source"),
            Some(&eye_log::Value::Str("rgb-only".to_string()))
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_v1_pccr_profile_loads_into_ir_only() {
        let dir = test_dir("v1-pccr");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p.toml");
        std::fs::write(
            &path,
            r#"version = 1
name = "max"
created_unix_s = 1700000000
rig_fingerprint = "deadbeefcafef00d"
estimator = "pccr"
[eyes.right]
theta = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
cov = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
model = "affine"
targets_used = 9
rms_after_rad = 0.001
"#,
        )
        .unwrap();

        let profile = read_profile(&path).unwrap();
        assert_eq!(profile.corrections.len(), 1);
        assert!(
            profile
                .correction(RaySource::IrOnly, EyeKey::Right)
                .is_some()
        );
        assert!(
            profile
                .correction(RaySource::RgbOnly, EyeKey::Right)
                .is_none()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_v1_profile_saves_as_v2() {
        let dir = test_dir("v1-saves-as-v2");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p.toml");
        std::fs::write(
            &path,
            r#"version = 1
name = "max"
created_unix_s = 1700000000
rig_fingerprint = "deadbeefcafef00d"
estimator = "fused"
[eyes.left]
theta = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
cov = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
model = "affine"
targets_used = 9
rms_after_rad = 0.001
"#,
        )
        .unwrap();
        let migrated = read_profile(&path).unwrap();

        write_profile(&path, &migrated).unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("version = 2"));
        assert!(contents.contains("[corrections.rgb-only.left]"));

        let reloaded = read_profile(&path).unwrap();
        assert_eq!(reloaded, migrated);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_v2_profile_with_eyes_table_rejected() {
        let dir = test_dir("v2-eyes-table");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p.toml");
        std::fs::write(
            &path,
            r#"version = 2
name = "max"
created_unix_s = 1700000000
rig_fingerprint = "deadbeefcafef00d"
estimator = "fused"
[eyes]
[corrections]
"#,
        )
        .unwrap();
        let err = read_profile(&path).unwrap_err();
        assert!(
            matches!(err, CalibrationError::Store(StoreError::Parse { .. })),
            "expected Parse error, got {err:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_unknown_profile_key_rejected() {
        let dir = test_dir("unknown-key");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p.toml");
        std::fs::write(
            &path,
            r#"version = 1
name = "default"
created_unix_s = 1
rig_fingerprint = "a"
estimator = "ir-pupil"
foo = 1
[eyes]
"#,
        )
        .unwrap();
        let err = read_profile(&path).unwrap_err();
        assert!(
            matches!(err, CalibrationError::Store(StoreError::Parse { .. })),
            "expected Parse error, got {err:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_invalid_profile_names_rejected() {
        let dir = test_dir("invalid-names");
        let store = ProfileStore::at(&dir);
        for bad in ["../etc", "", ".hidden", &"a".repeat(65)] {
            let err = store.profile_path(bad).unwrap_err();
            assert!(
                matches!(err, CalibrationError::Store(StoreError::InvalidName(_))),
                "expected InvalidName for {bad:?}, got {err:?}"
            );
        }
    }

    #[test]
    fn test_save_profile_is_atomic_no_tmp_left_behind() {
        let dir = test_dir("atomic");
        let store = ProfileStore::at(&dir);
        store.save_profile("default", &fixture_profile()).unwrap();
        let entries: Vec<_> = std::fs::read_dir(dir.join("profiles"))
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(entries, vec![std::ffi::OsString::from("default.toml")]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_list_profiles_sorted_ignores_tmp() {
        let dir = test_dir("list");
        let store = ProfileStore::at(&dir);

        assert_eq!(store.list_profiles().unwrap(), Vec::<String>::new());

        let profile = fixture_profile();
        store.save_profile("b", &profile).unwrap();
        store.save_profile("a", &profile).unwrap();
        store.save_profile("default", &profile).unwrap();
        std::fs::write(dir.join("profiles").join("c.toml.tmp"), "stray").unwrap();

        assert_eq!(
            store.list_profiles().unwrap(),
            vec!["a".to_string(), "b".to_string(), "default".to_string()]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn fitted_rig() -> eye_core::Rig {
        let screen = ScreenModel {
            output: OutputId::from("eDP-1"),
            size_mm: Vector2::new(310.0, 174.375),
            size_px: (3840, 2160),
            scale: 2.0,
        };
        let streams = vec![CameraStream {
            id: CameraId::from("ir"),
            width: 640,
            height: 360,
        }];
        nominal_rig(screen, &streams, &NominalRigConfig::default()).unwrap()
    }

    fn fitted_samples() -> Vec<FitSample> {
        let targets = [
            Point2::new(50.0, 40.0),
            Point2::new(155.0, 40.0),
            Point2::new(260.0, 40.0),
            Point2::new(50.0, 140.0),
            Point2::new(260.0, 140.0),
        ];
        let origin = Point3::new(155.0, 85.0, -500.0);
        let mut rng = SplitMix64::new(42);
        let mut samples = Vec::new();
        for target_mm in targets {
            for _ in 0..6 {
                let true_dir =
                    Unit::new_normalize(Point3::new(target_mm.x, target_mm.y, 0.0) - origin);
                let a = eye_geometry::angles::yaw_pitch_from_direction(&true_dir);
                let noisy = Vector2::new(
                    a.x + 0.1_f64.to_radians() * rng.gaussian(),
                    a.y + 0.1_f64.to_radians() * rng.gaussian(),
                );
                let ray = GazeRay {
                    side: Some(Side::Right),
                    timestamp: eye_core::Timestamp::from_nanos(0),
                    origin,
                    direction: direction_from_yaw_pitch(&noisy),
                    angular_cov: Matrix2::identity() * 1.0_f64.to_radians().powi(2),
                    origin_cov: Matrix3::identity(),
                    head_rotation: None,
                };
                samples.push(FitSample {
                    ray,
                    target_mm,
                    source: RaySource::RgbOnly,
                });
            }
        }
        samples
    }

    #[test]
    fn test_fitted_profile_round_trips_and_corrects_identically() {
        let dir = test_dir("fitted-round-trip");
        let rig = fitted_rig();
        let samples = fitted_samples();
        let profile = DotSessionFit::fit(&samples, &rig).unwrap();

        let store = ProfileStore::at(&dir);
        store.save_profile("default", &profile).unwrap();
        let reloaded = ProfileStore::at(&dir)
            .load_profile("default")
            .unwrap()
            .unwrap();

        let probe = GazeRay {
            side: Some(Side::Right),
            timestamp: eye_core::Timestamp::from_nanos(0),
            origin: Point3::new(155.0, 85.0, -500.0),
            direction: Unit::new_normalize(Vector3::new(0.1, -0.05, 1.0)),
            angular_cov: Matrix2::identity() * 1e-4,
            origin_cov: Matrix3::zeros(),
            head_rotation: None,
        };
        assert_eq!(profile.correct(&probe), reloaded.correct(&probe));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
