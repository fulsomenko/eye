//! Rig file storage and the one `Rig` serialization shared by the store, the
//! recorder's `session.toml` snapshot, and the bench.

use std::fs::{self, File};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use etcetera::{AppStrategy, AppStrategyArgs, choose_app_strategy};
use eye_core::{CameraId, CameraModel, OutputId, Rig, ScreenModel};
use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector2, Vector3};

use crate::error::{CalibrationError, StoreError};

#[derive(Debug, Clone)]
pub struct ProfileStore {
    root: PathBuf,
}

impl ProfileStore {
    pub fn open_default() -> Result<Self, CalibrationError> {
        let strategy = choose_app_strategy(AppStrategyArgs {
            top_level_domain: "org".into(),
            author: "eye".into(),
            app_name: "eye".into(),
        })
        .map_err(|_| StoreError::NoHome)?;
        Ok(Self {
            root: strategy.data_dir(),
        })
    }

    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn rig_path(&self, output: &OutputId) -> Result<PathBuf, CalibrationError> {
        validate_name(output.as_str())?;
        Ok(self.root.join("rigs").join(format!("{output}.toml")))
    }

    pub fn load_rig(&self, output: &OutputId) -> Result<Option<Rig>, CalibrationError> {
        let path = self.rig_path(output)?;
        let Some(table) = read_table(&path)? else {
            tracing::debug!(
                output = output.as_str(),
                path = %path.display(),
                "rig not found"
            );
            return Ok(None);
        };
        Ok(Some(rig_from_table_at(&table, &path)?))
    }

    pub fn save_rig(&self, rig: &Rig) -> Result<PathBuf, CalibrationError> {
        let path = self.rig_path(&rig.screen().output)?;
        write_rig(&path, rig)?;
        Ok(path)
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RigFile {
    version: u32,
    screen: ScreenFile,
    camera: Vec<CameraFile>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ScreenFile {
    output: String,
    size_mm: [f64; 2],
    size_px: [u32; 2],
    scale: f64,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CameraFile {
    id: String,
    size: [u32; 2],
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    distortion: [f64; 5],
    rotation_vector_rad: [f64; 3],
    translation_mm: [f64; 3],
}

impl From<&Rig> for RigFile {
    fn from(rig: &Rig) -> Self {
        let screen = rig.screen();
        Self {
            version: 1,
            screen: ScreenFile {
                output: screen.output.as_str().to_owned(),
                size_mm: [screen.size_mm.x, screen.size_mm.y],
                size_px: [screen.size_px.0, screen.size_px.1],
                scale: screen.scale,
            },
            camera: rig.cameras().iter().map(CameraFile::from).collect(),
        }
    }
}

impl From<&CameraModel> for CameraFile {
    fn from(m: &CameraModel) -> Self {
        let r = m.screen_from_camera.rotation.scaled_axis();
        let t = m.screen_from_camera.translation.vector;
        Self {
            id: m.id.as_str().to_owned(),
            size: [m.width, m.height],
            fx: m.fx,
            fy: m.fy,
            cx: m.cx,
            cy: m.cy,
            distortion: m.distortion,
            rotation_vector_rad: [r.x, r.y, r.z],
            translation_mm: [t.x, t.y, t.z],
        }
    }
}

impl From<CameraFile> for CameraModel {
    fn from(c: CameraFile) -> Self {
        CameraModel {
            id: CameraId::from(c.id.as_str()),
            width: c.size[0],
            height: c.size[1],
            fx: c.fx,
            fy: c.fy,
            cx: c.cx,
            cy: c.cy,
            distortion: c.distortion,
            screen_from_camera: Isometry3::from_parts(
                Translation3::from(Vector3::from(c.translation_mm)),
                UnitQuaternion::from_scaled_axis(Vector3::from(c.rotation_vector_rad)),
            ),
        }
    }
}

pub fn rig_to_table(rig: &Rig) -> toml::Table {
    toml::Table::try_from(RigFile::from(rig))
        .expect("RigFile holds only strings, numbers and arrays")
}

pub fn rig_from_table(table: &toml::Table) -> Result<Rig, CalibrationError> {
    rig_from_table_at(table, Path::new("<rig table>"))
}

fn rig_from_table_at(table: &toml::Table, path: &Path) -> Result<Rig, CalibrationError> {
    check_version(table, "rig")?;
    let file: RigFile = table
        .clone()
        .try_into()
        .map_err(|source| StoreError::Parse {
            path: path.to_owned(),
            source,
        })?;
    let screen = ScreenModel {
        output: OutputId::from(file.screen.output.as_str()),
        size_mm: Vector2::from(file.screen.size_mm),
        size_px: (file.screen.size_px[0], file.screen.size_px[1]),
        scale: file.screen.scale,
    };
    let output = screen.output.as_str().to_owned();
    let cameras: Vec<CameraModel> = file.camera.into_iter().map(CameraModel::from).collect();
    let camera_count = cameras.len() as u64;
    let rig = Rig::new(cameras, screen)?;
    tracing::info!(
        path = %path.display(),
        output,
        cameras = camera_count,
        "rig loaded"
    );
    Ok(rig)
}

pub fn write_rig(path: &Path, rig: &Rig) -> Result<(), CalibrationError> {
    let contents = toml::to_string_pretty(&RigFile::from(rig)).map_err(StoreError::from)?;
    write_atomic(path, &contents)?;
    tracing::info!(
        path = %path.display(),
        output = rig.screen().output.as_str(),
        cameras = rig.cameras().len() as u64,
        "rig written"
    );
    Ok(())
}

pub fn read_rig(path: &Path) -> Result<Rig, CalibrationError> {
    let table = read_table(path)?.ok_or_else(|| StoreError::Io {
        path: path.to_owned(),
        source: std::io::Error::new(std::io::ErrorKind::NotFound, "rig file not found"),
    })?;
    rig_from_table_at(&table, path)
}

pub(crate) fn validate_name(name: &str) -> Result<(), StoreError> {
    let valid = !name.is_empty()
        && name.len() <= 64
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-');
    if valid {
        Ok(())
    } else {
        Err(StoreError::InvalidName(name.to_owned()))
    }
}

pub(crate) fn check_version(table: &toml::Table, what: &'static str) -> Result<(), StoreError> {
    match table.get("version").and_then(toml::Value::as_integer) {
        Some(1) | None => Ok(()),
        Some(found) => Err(StoreError::UnsupportedVersion {
            what,
            found: u32::try_from(found).unwrap_or(u32::MAX),
        }),
    }
}

pub(crate) fn write_atomic(path: &Path, contents: &str) -> Result<(), StoreError> {
    let io = |source| StoreError::Io {
        path: path.to_owned(),
        source,
    };
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(io)?;
    }
    let tmp = path.with_extension("toml.tmp");
    let mut f = File::create(&tmp).map_err(io)?;
    f.write_all(contents.as_bytes()).map_err(io)?;
    f.sync_all().map_err(io)?;
    fs::rename(&tmp, path).map_err(io)
}

pub(crate) fn read_table(path: &Path) -> Result<Option<toml::Table>, StoreError> {
    let contents = match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(StoreError::Io {
                path: path.to_owned(),
                source,
            });
        }
    };
    toml::from_str(&contents)
        .map(Some)
        .map_err(|source| StoreError::Parse {
            path: path.to_owned(),
            source,
        })
}

#[cfg(test)]
mod tests {
    use eye_core::{CameraId, OutputId};
    use nalgebra::{UnitQuaternion, Vector2, Vector3};

    use super::*;
    use crate::nominal::{CameraStream, NominalRigConfig, nominal_rig};

    fn test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("eye-store-{}-{name}", std::process::id()))
    }

    fn fixture_rig() -> Rig {
        let screen = ScreenModel {
            output: OutputId::from("eDP-1"),
            size_mm: Vector2::new(310.0, 170.0),
            size_px: (3840, 2160),
            scale: 2.0,
        };
        let streams = vec![
            CameraStream {
                id: CameraId::from("rgb"),
                width: 1280,
                height: 720,
            },
            CameraStream {
                id: CameraId::from("ir"),
                width: 640,
                height: 360,
            },
        ];
        let rig = nominal_rig(screen, &streams, &NominalRigConfig::default()).unwrap();
        let (mut cameras, screen) = rig.into_parts();
        let ir = cameras.iter_mut().find(|c| c.id.as_str() == "ir").unwrap();
        ir.fx = 457.123456789;
        ir.fy = 456.9;
        ir.cx = 320.4;
        ir.cy = 180.2;
        ir.distortion = [0.08, -0.15, 0.001, -0.0005, 0.0];
        ir.screen_from_camera = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(167.5, -7.0, 0.0),
            UnitQuaternion::from_scaled_axis(Vector3::new(0.01, 3.1, -0.02)),
        );
        Rig::new(cameras, screen).unwrap()
    }

    fn assert_rig_round_trips(rig: &Rig, back: &Rig, tol: f64) {
        assert_eq!(back.screen(), rig.screen());
        assert_eq!(rig.cameras().len(), back.cameras().len());
        for (orig, got) in rig.cameras().iter().zip(back.cameras().iter()) {
            assert_eq!(orig.id, got.id);
            assert_eq!(orig.fx.to_bits(), got.fx.to_bits());
            assert_eq!(orig.fy.to_bits(), got.fy.to_bits());
            assert_eq!(orig.cx.to_bits(), got.cx.to_bits());
            assert_eq!(orig.cy.to_bits(), got.cy.to_bits());
            for (a, b) in orig.distortion.iter().zip(got.distortion.iter()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
            let angle = orig
                .screen_from_camera
                .rotation
                .angle_to(&got.screen_from_camera.rotation);
            assert!(angle.abs() <= tol, "rotation differs by {angle} rad");
            let dt = orig.screen_from_camera.translation.vector
                - got.screen_from_camera.translation.vector;
            assert!(dt.norm() <= tol, "translation differs by {}", dt.norm());
        }
    }

    #[test]
    fn test_rig_table_round_trip() {
        let rig = fixture_rig();
        let table = rig_to_table(&rig);
        let back = rig_from_table(&table).unwrap();
        assert_rig_round_trips(&rig, &back, 1e-12);
    }

    #[test]
    fn test_rig_round_trip_is_exact() {
        let dir = test_dir("round-trip");
        let store = ProfileStore::at(&dir);
        let rig = fixture_rig();
        let path = store.save_rig(&rig).unwrap();
        assert_eq!(path, dir.join("rigs").join("eDP-1.toml"));
        let back = store.load_rig(&OutputId::from("eDP-1")).unwrap().unwrap();
        assert_rig_round_trips(&rig, &back, 1e-12);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_write_and_read_rig_explicit_path() {
        let dir = test_dir("explicit-path");
        let rig = fixture_rig();
        let path = dir.join("r.toml");
        write_rig(&path, &rig).unwrap();
        let back = read_rig(&path).unwrap();
        assert_rig_round_trips(&rig, &back, 1e-12);

        let missing = dir.join("missing.toml");
        let err = read_rig(&missing).unwrap_err();
        assert!(matches!(
            err,
            CalibrationError::Store(StoreError::Io { .. })
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_missing_rig_returns_none() {
        let dir = test_dir("missing-rig");
        std::fs::create_dir_all(&dir).unwrap();
        let store = ProfileStore::at(&dir);
        assert_eq!(store.load_rig(&OutputId::from("eDP-1")).unwrap(), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_logs_rig_loaded_at_info() {
        let dir = test_dir("logs-loaded");
        let store = ProfileStore::at(&dir);
        let rig = fixture_rig();

        let (_, records) = eye_log::testing::capture_logs(tracing::Level::INFO, || {
            store.save_rig(&rig).unwrap();
            store.load_rig(&OutputId::from("eDP-1")).unwrap();
        });

        let written = records
            .iter()
            .find(|r| r.message == "rig written")
            .expect("no 'rig written' record");
        assert_eq!(written.level, eye_log::Level::Info);
        assert_eq!(
            written.fields.get("cameras"),
            Some(&eye_log::Value::U64(rig.cameras().len() as u64))
        );
        match written.fields.get("path") {
            Some(eye_log::Value::Str(p)) => assert!(p.ends_with("rigs/eDP-1.toml"), "{p}"),
            other => panic!("expected path Str, got {other:?}"),
        }

        let loaded = records
            .iter()
            .find(|r| r.message == "rig loaded")
            .expect("no 'rig loaded' record");
        assert_eq!(loaded.level, eye_log::Level::Info);
        assert_eq!(
            loaded.fields.get("cameras"),
            Some(&eye_log::Value::U64(rig.cameras().len() as u64))
        );
        match loaded.fields.get("path") {
            Some(eye_log::Value::Str(p)) => assert!(p.ends_with("rigs/eDP-1.toml"), "{p}"),
            other => panic!("expected path Str, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_logs_rig_not_found_at_debug() {
        let dir = test_dir("logs-not-found");
        std::fs::create_dir_all(&dir).unwrap();
        let store = ProfileStore::at(&dir);

        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            store.load_rig(&OutputId::from("eDP-1")).unwrap();
        });

        assert!(!records.iter().any(|r| r.message == "rig loaded"));
        let rec = records
            .iter()
            .find(|r| r.message == "rig not found")
            .expect("no 'rig not found' record");
        assert_eq!(rec.level, eye_log::Level::Debug);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_rig_keyed_by_output() {
        let dir = test_dir("keyed-by-output");
        let store = ProfileStore::at(&dir);
        store.save_rig(&fixture_rig()).unwrap();
        assert_eq!(store.load_rig(&OutputId::from("HDMI-A-1")).unwrap(), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_unsupported_version_rejected() {
        let dir = test_dir("unsupported-version");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.toml");
        std::fs::write(&path, "version = 2\nfoo = 1\n").unwrap();
        let err = read_rig(&path).unwrap_err();
        assert!(matches!(
            err,
            CalibrationError::Store(StoreError::UnsupportedVersion {
                what: "rig",
                found: 2
            })
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_unknown_key_rejected() {
        let dir = test_dir("unknown-key");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.toml");
        std::fs::write(
            &path,
            r#"version = 1
[screen]
output = "eDP-1"
size_mm = [310.0, 174.4]
size_px = [3840, 2160]
scale = 2.0
foo = 1
[[camera]]
id = "ir"
size = [640, 360]
fx = 429.26
fy = 429.26
cx = 320.0
cy = 180.0
distortion = [0.0, 0.0, 0.0, 0.0, 0.0]
rotation_vector_rad = [0.0, 3.141592653589793, 0.0]
translation_mm = [155.0, -7.0, 0.0]
"#,
        )
        .unwrap();
        let err = read_rig(&path).unwrap_err();
        match &err {
            CalibrationError::Store(StoreError::Parse { source, .. }) => {
                let message = source.to_string();
                assert!(
                    message.contains("foo"),
                    "expected error to name unknown field `foo`, got {message:?}"
                );
            }
            other => panic!("expected StoreError::Parse, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_invalid_rig_file_rejected() {
        let dir = test_dir("invalid-rig");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.toml");
        std::fs::write(
            &path,
            r#"version = 1
[screen]
output = "eDP-1"
size_mm = [310.0, 174.4]
size_px = [3840, 2160]
scale = 2.0
[[camera]]
id = "ir"
size = [640, 360]
fx = 0.0
fy = 429.26
cx = 320.0
cy = 180.0
distortion = [0.0, 0.0, 0.0, 0.0, 0.0]
rotation_vector_rad = [0.0, 3.141592653589793, 0.0]
translation_mm = [155.0, -7.0, 0.0]
"#,
        )
        .unwrap();
        let err = read_rig(&path).unwrap_err();
        assert!(matches!(err, CalibrationError::Core(_)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_invalid_names_rejected() {
        let dir = test_dir("invalid-names");
        let store = ProfileStore::at(&dir);
        for bad in ["../etc", "", ".hidden", &"a".repeat(65)] {
            let err = store.rig_path(&OutputId::from(bad)).unwrap_err();
            assert!(
                matches!(err, CalibrationError::Store(StoreError::InvalidName(_))),
                "expected InvalidName for {bad:?}, got {err:?}"
            );
        }
        for good in ["eDP-1", "HDMI-A-1"] {
            store.rig_path(&OutputId::from(good)).unwrap();
        }
    }

    #[test]
    fn test_save_is_atomic_no_tmp_left_behind() {
        let dir = test_dir("atomic");
        let store = ProfileStore::at(&dir);
        store.save_rig(&fixture_rig()).unwrap();
        let entries: Vec<_> = std::fs::read_dir(dir.join("rigs"))
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(entries, vec![std::ffi::OsString::from("eDP-1.toml")]);
        let contents = std::fs::read_to_string(dir.join("rigs").join("eDP-1.toml")).unwrap();
        contents.parse::<toml::Table>().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_open_default_root_is_absolute_and_named_eye() {
        let store = ProfileStore::open_default().unwrap();
        assert!(store.root().is_absolute());
        assert_eq!(
            store.root().file_name().and_then(|n| n.to_str()),
            Some("eye")
        );
    }
}
