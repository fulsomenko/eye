use std::fs;
use std::io::{Read, Write};
use std::os::unix::net::UnixListener;
use std::path::Path;
use std::process::{Command, Stdio};

use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;

#[test]
fn test_help_lists_all_subcommands() {
    cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("probe"))
        .stdout(predicate::str::contains("emitter"))
        .stdout(predicate::str::contains("record"))
        .stdout(predicate::str::contains("calibrate"))
        .stdout(predicate::str::contains("run"))
        .stdout(predicate::str::contains("bench"));
}

#[test]
fn test_version_includes_git_rev() {
    cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::is_match(r"^eye \d+\.\d+\.\d+ \(.+\)\n$").unwrap());
}

#[test]
fn test_unknown_subcommand_exits_2() {
    cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .arg("frobnicate")
        .assert()
        .code(2);
}

#[test]
fn test_missing_subcommand_exits_2() {
    cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .assert()
        .code(2);
}

#[test]
fn test_missing_explicit_config_exits_1() {
    cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .args(["probe", "--config", "/nonexistent/eye.toml"])
        .assert()
        .code(1)
        .stderr(predicate::str::contains("error: config file not found"));
}

#[test]
fn test_eye_config_env_is_honoured() {
    cargo_bin_cmd!("eye")
        .env("EYE_CONFIG", "/nonexistent/eye.toml")
        .arg("probe")
        .assert()
        .code(1)
        .stderr(predicate::str::contains("error: config file not found"));
}

fn fake_hyprland_runtime_dir() -> (tempfile::TempDir, std::path::PathBuf) {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../eye-platform/tests/fixtures/hyprland-monitors-latitude7420.json");
    let contents = fs::read_to_string(fixture).expect("fixture is readable");
    let dir = tempfile::tempdir().expect("tempdir");
    let hypr_dir = dir.path().join("hypr").join("test-sig");
    fs::create_dir_all(&hypr_dir).expect("hypr dir");
    let socket_path = hypr_dir.join(".socket.sock");
    let listener = UnixListener::bind(&socket_path).expect("bind fake hyprland socket");
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf);
            let _ = stream.write_all(contents.as_bytes());
        }
    });
    let runtime_dir = dir.path().to_path_buf();
    (dir, runtime_dir)
}

fn fixture_cmd() -> (Command, tempfile::TempDir) {
    let (dir, runtime_dir) = fake_hyprland_runtime_dir();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_eye"));
    cmd.env_remove("EYE_CONFIG")
        .env_remove("EYE_IR_CAMERA")
        .env("XDG_RUNTIME_DIR", &runtime_dir)
        .env("HYPRLAND_INSTANCE_SIGNATURE", "test-sig")
        .env("WAYLAND_DISPLAY", "wayland-test")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    (cmd, dir)
}

#[test]
fn test_log_file_writes_jsonl_with_run_context() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let log_path = tmp.path().join("p.jsonl");
    let (mut cmd, _runtime) = fixture_cmd();
    cmd.arg("--log-file").arg(&log_path).arg("probe");
    let child = cmd.spawn().expect("spawning eye");
    let pid = child.id();
    let output = child.wait_with_output().expect("waiting for eye");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let contents = fs::read_to_string(&log_path).expect("log file was written");
    let first_line = contents.lines().next().expect("at least one log line");
    let value: serde_json::Value = serde_json::from_str(first_line).expect("line is valid JSON");
    assert_eq!(value["context"]["command"], "probe");
    let run_id = value["context"]["run.id"]
        .as_str()
        .expect("run.id is a string")
        .to_string();
    assert!(!run_id.is_empty());
    assert!(
        run_id.ends_with(&pid.to_string()),
        "run id {run_id:?} does not end in the child pid {pid}"
    );
    assert_eq!(value["schema"], 1);
}

#[test]
fn test_log_file_flag_after_subcommand_parses() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let log_path = tmp.path().join("q.jsonl");
    let (mut cmd, _runtime) = fixture_cmd();
    cmd.arg("probe").arg("--log-file").arg(&log_path);
    let status = cmd.status().expect("running eye");
    assert!(status.success());
    assert!(log_path.exists(), "log file was not created");
}

#[test]
fn test_bad_log_level_is_usage_error() {
    cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .args(["--log-level", "eye[", "probe"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("invalid filter directive"));
}

#[test]
fn test_log_format_json_terminal() {
    let (mut cmd, _runtime) = fixture_cmd();
    cmd.args(["--log-format", "json", "probe"]);
    let output = cmd.output().expect("running eye");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let mut saw_run_started = false;
    for line in stderr.lines().filter(|l| !l.trim().is_empty()) {
        let value: serde_json::Value =
            serde_json::from_str(line).unwrap_or_else(|e| panic!("line {line:?} not JSON: {e}"));
        if value["fields"]["message"] == "run started" && value["span"]["name"] == "run" {
            saw_run_started = true;
        }
    }
    assert!(saw_run_started, "stderr:\n{stderr}");
}
