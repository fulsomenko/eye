//! Per-user calibration profile storage: `profiles/<name>.toml` under a
//! [`ProfileStore`], plus explicit-path read/write for `eye calibrate --output`.

use std::path::{Path, PathBuf};

use crate::correction::UserProfile;
use crate::error::{CalibrationError, StoreError};
use crate::store::{ProfileStore, check_version, read_table, validate_name, write_atomic};

impl ProfileStore {
    pub fn profile_path(&self, name: &str) -> Result<PathBuf, CalibrationError> {
        validate_name(name)?;
        Ok(self.root().join("profiles").join(format!("{name}.toml")))
    }

    pub fn load_profile(&self, name: &str) -> Result<Option<UserProfile>, CalibrationError> {
        let path = self.profile_path(name)?;
        read_table(&path)?
            .map(|table| profile_from_table(table, &path))
            .transpose()
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
        Ok(names)
    }
}

fn profile_from_table(table: toml::Table, path: &Path) -> Result<UserProfile, CalibrationError> {
    check_version(&table, "profile")?;
    Ok(table.try_into().map_err(|source| StoreError::Parse {
        path: path.to_owned(),
        source,
    })?)
}

pub fn write_profile(path: &Path, profile: &UserProfile) -> Result<(), CalibrationError> {
    let contents = toml::to_string_pretty(profile).map_err(StoreError::from)?;
    write_atomic(path, &contents)?;
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

    use eye_core::{CameraId, GazeRay, OutputId, ScreenModel, Side};
    use eye_geometry::angles::direction_from_yaw_pitch;
    use eye_geometry::synth::SplitMix64;
    use nalgebra::{Matrix2, Matrix3, Point2, Point3, Unit, UnitQuaternion, Vector2, Vector3};

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
            version: 1,
            name: "default".into(),
            created_unix_s: 1_791_400_000,
            rig_fingerprint: "0123456789abcdef".into(),
            estimator: "ir-pupil".into(),
            eyes,
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
    fn test_unsupported_profile_version_rejected() {
        let dir = test_dir("unsupported-version");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p.toml");
        std::fs::write(&path, "version = 2\nfoo = 1\n").unwrap();
        let err = read_profile(&path).unwrap_err();
        assert!(matches!(
            err,
            CalibrationError::Store(StoreError::UnsupportedVersion {
                what: "profile",
                found: 2
            })
        ));
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
                    origin,
                    direction: direction_from_yaw_pitch(&noisy),
                    angular_cov: Matrix2::identity() * 1.0_f64.to_radians().powi(2),
                    origin_cov: Matrix3::identity(),
                    head_rotation: UnitQuaternion::identity(),
                };
                samples.push(FitSample { ray, target_mm });
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
            origin: Point3::new(155.0, 85.0, -500.0),
            direction: Unit::new_normalize(Vector3::new(0.1, -0.05, 1.0)),
            angular_cov: Matrix2::identity() * 1e-4,
            origin_cov: Matrix3::zeros(),
            head_rotation: UnitQuaternion::identity(),
        };
        assert_eq!(profile.correct(&probe), reloaded.correct(&probe));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
