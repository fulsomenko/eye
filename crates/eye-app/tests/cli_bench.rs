use assert_cmd::cargo::cargo_bin_cmd;
use eye::config::{Config, EnvOverrides};
use eye_bench::matrix::BenchMatrix;
use eye_bench::testing::{FOUR_BY_FOUR_CENTRES, SyntheticSession, write_synthetic_session};
use predicates::prelude::*;

#[test]
fn test_bench_without_inputs_exits_1() {
    let tmp = tempfile::tempdir().expect("tempdir");
    cargo_bin_cmd!("eye")
        .current_dir(tmp.path())
        .env("XDG_CONFIG_HOME", tmp.path())
        .env_remove("EYE_CONFIG")
        .args(["bench"])
        .assert()
        .code(1)
        .stderr(predicate::str::contains(
            "give RECORDING... or --matrix FILE",
        ));
}

#[test]
fn test_bench_bad_calibration_value_exits_2() {
    let tmp = tempfile::tempdir().expect("tempdir");
    cargo_bin_cmd!("eye")
        .current_dir(tmp.path())
        .env("XDG_CONFIG_HOME", tmp.path())
        .env_remove("EYE_CONFIG")
        .args(["bench", "--calibration", "magic", "x"])
        .assert()
        .code(2);
}

#[test]
fn test_bench_synthetic_session_writes_report() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let spec = SyntheticSession {
        targets: FOUR_BY_FOUR_CENTRES.to_vec(),
        size: (640, 360),
        ..Default::default()
    };
    write_synthetic_session(tmp.path(), "s", &spec).expect("writes synthetic session");

    let config_path = tmp.path().join("eye.toml");
    std::fs::write(
        &config_path,
        "[[camera]]\n\
         id = \"ir\"\n\
         device = \"/dev/null\"\n\
         format = \"gray\"\n\
         size = [640, 360]\n\
         \n\
         [detect]\n\
         ir = \"ir-classic\"\n\
         \n\
         [estimate]\n\
         kind = \"ir-pupil\"\n",
    )
    .expect("writes config");

    let session_dir = tmp.path().join("s");
    let output_dir = tmp.path().join("r");

    let assert = cargo_bin_cmd!("eye")
        .current_dir(tmp.path())
        .env("XDG_CONFIG_HOME", tmp.path())
        .env_remove("EYE_CONFIG")
        .args([
            "bench",
            "--config",
            config_path.to_str().expect("utf8 path"),
            "--calibration",
            "none",
            "--output",
            output_dir.to_str().expect("utf8 path"),
            session_dir.to_str().expect("utf8 path"),
        ])
        .assert()
        .code(0);

    let stdout = String::from_utf8_lossy(&assert.get_output().stdout).into_owned();
    assert!(stdout.starts_with("# eye bench report"));

    let json_path = output_dir.join("report.json");
    let md_path = output_dir.join("report.md");
    assert!(json_path.exists());
    assert!(md_path.exists());

    let json = std::fs::read_to_string(&json_path).expect("reads report.json");
    let value: serde_json::Value = serde_json::from_str(&json).expect("valid json");
    let rows = value["rows"].as_array().expect("rows array");

    let session_rows: Vec<&serde_json::Value> =
        rows.iter().filter(|r| r["kind"] == "session").collect();
    assert_eq!(session_rows.len(), 1);
    assert_eq!(session_rows[0]["pipeline"], "eye");
    assert_eq!(session_rows[0]["session"], "s");
    assert_eq!(session_rows[0]["status"], "ok");
    assert_eq!(session_rows[0]["metrics"]["samples"], 0);

    let aggregate_rows: Vec<&serde_json::Value> =
        rows.iter().filter(|r| r["kind"] == "aggregate").collect();
    assert_eq!(aggregate_rows.len(), 1);
    assert_eq!(aggregate_rows[0]["pipeline"], "eye");
}

#[test]
fn test_baseline_pipeline_filters_equal_builtin_default() {
    let builtin = Config::builtin_default();
    for path in [
        "../../bench/baseline/rgb.toml",
        "../../bench/baseline/ir.toml",
        "../../bench/baseline/fused.toml",
    ] {
        let config = Config::load_with(Some(std::path::Path::new(path)), &EnvOverrides::default())
            .expect("loads baseline config")
            .config;
        assert_eq!(config.filter, builtin.filter, "{path}");
    }
}

#[test]
fn test_fused_pccr_config_parses() {
    let builtin = Config::builtin_default();
    let config = Config::load_with(
        Some(std::path::Path::new("../../bench/baseline/fused-pccr.toml")),
        &EnvOverrides::default(),
    )
    .expect("loads baseline config")
    .config;
    assert_eq!(config.filter, builtin.filter);

    let matrix =
        BenchMatrix::from_path(std::path::Path::new("../../bench/baseline/dual.bench.toml"))
            .expect("loads dual bench matrix");
    assert!(
        matrix.pipelines.iter().any(|p| p.name == "fused-pccr"),
        "dual.bench.toml must list a fused-pccr pipeline"
    );
}

#[test]
fn test_baseline_matrices_resolve_relative_paths() {
    let baseline_dir = std::path::Path::new("../../bench/baseline");
    let repo_root = baseline_dir
        .join("../..")
        .canonicalize()
        .expect("canonicalizes repo root");

    let mut matrix_paths: Vec<std::path::PathBuf> = std::fs::read_dir(baseline_dir)
        .expect("reads bench/baseline")
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.to_string_lossy().ends_with(".bench.toml"))
        .collect();
    matrix_paths.sort();
    assert!(
        !matrix_paths.is_empty(),
        "expected at least one bench.toml under bench/baseline"
    );

    for path in &matrix_paths {
        let text = std::fs::read_to_string(path).expect("reads matrix toml");
        let raw: toml::Value = toml::from_str(&text).expect("parses matrix toml");

        let raw_recordings = raw["recordings"].as_array().expect("recordings array");
        for entry in raw_recordings {
            let entry = entry.as_str().expect("recording entries are strings");
            assert!(
                !std::path::Path::new(entry).is_absolute(),
                "{path:?}: recording entry {entry:?} must be relative"
            );
        }
        if let Some(profile) = raw.get("evaluation").and_then(|e| e.get("profile")) {
            let profile = profile.as_str().expect("evaluation.profile is a string");
            assert!(
                !std::path::Path::new(profile).is_absolute(),
                "{path:?}: evaluation.profile {profile:?} must be relative"
            );
        }

        let matrix = BenchMatrix::from_path(path).expect("loads bench matrix");
        for recording in &matrix.recordings {
            let cwd = std::env::current_dir().expect("current dir");
            let absolute = normalize_lexically(&cwd.join(recording));
            assert!(
                absolute.starts_with(&repo_root),
                "{path:?}: resolved recording {recording:?} (normalized {absolute:?}) must be under the repo root {repo_root:?}"
            );
        }
        for pipeline in &matrix.pipelines {
            if let Some(config) = &pipeline.config {
                assert!(
                    config.exists(),
                    "{path:?}: pipeline {:?} config {config:?} must exist on disk",
                    pipeline.name
                );
            }
        }
    }
}

/// Collapses `.` and `..` components without touching the filesystem (the recordings a
/// committed matrix points at need not exist in every checkout).
fn normalize_lexically(path: &std::path::Path) -> std::path::PathBuf {
    use std::path::Component;
    let mut out = std::path::PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[test]
#[ignore = "needs EYE_RECORDING and the MediaPipe models (detect-model-fetch)"]
fn test_bench_real_recording() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let recording = std::env::var("EYE_RECORDING").expect("EYE_RECORDING set");
    let output_dir = tmp.path().join("r");

    let assert = cargo_bin_cmd!("eye")
        .current_dir(tmp.path())
        .env("XDG_CONFIG_HOME", tmp.path())
        .env_remove("EYE_CONFIG")
        .args([
            "bench",
            "--output",
            output_dir.to_str().expect("utf8 path"),
            &recording,
        ])
        .assert()
        .code(0);

    let stdout = String::from_utf8_lossy(&assert.get_output().stdout).into_owned();
    assert!(stdout.starts_with("# eye bench report"));

    let json = std::fs::read_to_string(output_dir.join("report.json")).expect("reads report.json");
    let value: serde_json::Value = serde_json::from_str(&json).expect("valid json");
    let rows = value["rows"].as_array().expect("rows array");
    assert!(rows.iter().any(|r| {
        r["kind"] == "session"
            && r["status"] == "ok"
            && r["metrics"]["samples"].as_u64().unwrap_or(0) > 0
    }));
}
