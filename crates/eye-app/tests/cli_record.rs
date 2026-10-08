use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use assert_cmd::cargo::cargo_bin_cmd;
use serde_json::Value;

fn session_lines(path: &Path, file: &str) -> Vec<Value> {
    fs::read_to_string(path.join(file))
        .unwrap_or_default()
        .lines()
        .map(|line| serde_json::from_str(line).expect("valid json line"))
        .collect()
}

fn emitter_status() -> String {
    String::from_utf8_lossy(
        &cargo_bin_cmd!("eye")
            .env_remove("EYE_CONFIG")
            .env_remove("EYE_IR_CAMERA")
            .args(["emitter", "status"])
            .output()
            .expect("eye emitter status runs")
            .stdout,
    )
    .trim()
    .to_string()
}

#[test]
#[ignore = "needs hardware"]
fn test_record_free_two_seconds() {
    let dir = tempfile::tempdir().expect("tempdir");
    let session_dir = dir.path().join("s");

    let before = emitter_status();

    cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .env_remove("EYE_IR_CAMERA")
        .args([
            "record",
            "--targets",
            "0",
            "--duration-s",
            "2",
            "--output",
            session_dir.to_str().expect("utf8 path"),
        ])
        .assert()
        .success();

    let mode = fs::metadata(&session_dir)
        .expect("session dir exists")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o700);

    let session_toml =
        fs::read_to_string(session_dir.join("session.toml")).expect("session.toml exists");
    let meta: toml::Value = toml::from_str(&session_toml).expect("session.toml is valid toml");
    assert!(meta.get("git_rev").is_some());
    assert!(meta.get("rig").is_some());
    assert_eq!(
        meta.get("emitter").and_then(toml::Value::as_str),
        Some("on")
    );

    let index = session_lines(&session_dir, "index.jsonl");
    let ir_frames: Vec<&Value> = index.iter().filter(|r| r["camera"] == "ir").collect();
    let rgb_frames: Vec<&Value> = index.iter().filter(|r| r["camera"] == "rgb").collect();
    assert!(
        ir_frames.len() >= 25,
        "expected at least 25 ir frames, got {}",
        ir_frames.len()
    );
    assert!(
        rgb_frames.len() >= 10,
        "expected at least 10 rgb frames, got {}",
        rgb_frames.len()
    );

    for record in ir_frames.iter().skip(3) {
        let illumination = record["illumination"].as_str().expect("illumination tag");
        assert!(
            illumination == "ir_lit" || illumination == "ir_dark",
            "unexpected illumination tag: {illumination}"
        );
    }
    let tail: Vec<&str> = ir_frames
        .iter()
        .skip(3)
        .map(|r| r["illumination"].as_str().expect("illumination tag"))
        .collect();
    if let Some(offset) = tail
        .iter()
        .zip(tail.iter().skip(1))
        .position(|(a, b)| a == b)
    {
        panic!(
            "ir illumination did not alternate at index {}: {:?} then {:?}",
            offset + 3,
            tail[offset],
            tail[offset + 1]
        );
    }

    let after = emitter_status();
    assert_eq!(before, after);

    let _ = fs::remove_dir_all(&session_dir);
}

#[test]
#[ignore = "needs wayland"]
fn test_record_with_targets() {
    let dir = tempfile::tempdir().expect("tempdir");
    let session_dir = dir.path().join("s");

    cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .env_remove("EYE_IR_CAMERA")
        .args([
            "record",
            "--targets",
            "9",
            "--output",
            session_dir.to_str().expect("utf8 path"),
        ])
        .assert()
        .success();

    let targets = session_lines(&session_dir, "targets.jsonl");
    assert_eq!(targets.len(), 9);
    for record in targets.iter().take(8) {
        assert!(
            !record["hidden_ns"].is_null(),
            "expected hidden_ns on non-final target: {record}"
        );
    }

    let index = session_lines(&session_dir, "index.jsonl");
    let min_ts = index
        .iter()
        .map(|r| r["timestamp_ns"].as_u64().expect("timestamp_ns is u64"))
        .min()
        .expect("at least one frame");
    let max_ts = index
        .iter()
        .map(|r| r["timestamp_ns"].as_u64().expect("timestamp_ns is u64"))
        .max()
        .expect("at least one frame");
    for record in &targets {
        let shown_ns = record["shown_ns"].as_u64().expect("shown_ns is u64");
        assert!(
            shown_ns >= min_ts && shown_ns <= max_ts,
            "target shown_ns {shown_ns} outside frame range [{min_ts}, {max_ts}]"
        );
    }

    let _ = fs::remove_dir_all(&session_dir);
}
