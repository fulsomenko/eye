use std::fs;
use std::io::{Read, Write};
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::thread;

use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;

fn fixture_hyprland_json() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../eye-platform/tests/fixtures/hyprland-monitors-latitude7420.json");
    fs::read_to_string(path).expect("fixture is readable")
}

fn fake_hyprland_runtime_dir() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let hypr_dir = dir.path().join("hypr").join("test-sig");
    fs::create_dir_all(&hypr_dir).expect("hypr dir");
    let socket_path = hypr_dir.join(".socket.sock");
    let listener = UnixListener::bind(&socket_path).expect("bind fake hyprland socket");
    thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf);
            let _ = stream.write_all(fixture_hyprland_json().as_bytes());
        }
    });
    let runtime_dir = dir.path().to_path_buf();
    (dir, runtime_dir)
}

#[test]
fn test_probe_json_with_fake_hyprland_socket() {
    let (_dir, runtime_dir) = fake_hyprland_runtime_dir();

    let assert = cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .env_remove("EYE_IR_CAMERA")
        .env("XDG_RUNTIME_DIR", &runtime_dir)
        .env("HYPRLAND_INSTANCE_SIGNATURE", "test-sig")
        .env("WAYLAND_DISPLAY", "wayland-test")
        .args(["probe", "--json"])
        .assert()
        .success();

    let stdout = assert.get_output().stdout.clone();
    let value: serde_json::Value = serde_json::from_slice(&stdout).expect("stdout is valid JSON");
    assert_eq!(value["display_backend"], "hyprland");
    assert_eq!(value["outputs"][0]["name"], "eDP-1");
    assert_eq!(value["outputs"][0]["scale"], 2.0);
    assert!(value["cameras"].is_array());
}

fn fake_dmi_dir(sys_vendor: &str, product_name: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    fs::write(dir.path().join("sys_vendor"), format!("{sys_vendor}\n")).expect("write sys_vendor");
    fs::write(dir.path().join("product_name"), format!("{product_name}\n"))
        .expect("write product_name");
    dir
}

#[test]
fn test_probe_json_reports_profile() {
    let (_dir, runtime_dir) = fake_hyprland_runtime_dir();
    let dmi_dir = fake_dmi_dir("Dell Inc.", "Latitude 7420");

    let assert = cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .env_remove("EYE_IR_CAMERA")
        .env("XDG_RUNTIME_DIR", &runtime_dir)
        .env("HYPRLAND_INSTANCE_SIGNATURE", "test-sig")
        .env("WAYLAND_DISPLAY", "wayland-test")
        .env("EYE_DMI_DIR", dmi_dir.path())
        .args(["probe", "--json"])
        .assert()
        .success();

    let stdout = assert.get_output().stdout.clone();
    let value: serde_json::Value = serde_json::from_slice(&stdout).expect("stdout is valid JSON");
    assert_eq!(
        value["hardware_profile"],
        serde_json::json!({"id": "dell-latitude-7420", "verified": true})
    );
}

#[test]
fn test_probe_output_flag_writes_file() {
    let (dir, runtime_dir) = fake_hyprland_runtime_dir();
    let out_file = dir.path().join("probe.json");

    let assert = cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .env_remove("EYE_IR_CAMERA")
        .env("XDG_RUNTIME_DIR", &runtime_dir)
        .env("HYPRLAND_INSTANCE_SIGNATURE", "test-sig")
        .env("WAYLAND_DISPLAY", "wayland-test")
        .args(["--output", out_file.to_str().unwrap(), "probe", "--json"])
        .assert()
        .success();
    assert.stdout(predicate::str::is_empty());

    let content = fs::read_to_string(&out_file).unwrap();
    let _: serde_json::Value = serde_json::from_str(&content).expect("file is valid JSON");
}

#[test]
fn test_emitter_requires_action_exits_2() {
    cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .env_remove("EYE_IR_CAMERA")
        .arg("emitter")
        .assert()
        .code(2);
}

#[test]
fn test_emitter_unknown_action_exits_2() {
    cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .env_remove("EYE_IR_CAMERA")
        .args(["emitter", "blink"])
        .assert()
        .code(2);
}

#[test]
fn test_emitter_rejects_output_flag() {
    cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .env_remove("EYE_IR_CAMERA")
        .args(["--output", "x", "emitter", "status"])
        .assert()
        .code(1)
        .stderr(predicate::str::contains(
            "eye emitter does not write a file",
        ));
}

#[test]
#[ignore = "needs hardware"]
fn test_emitter_on_off_round_trip() {
    let run_status = || {
        cargo_bin_cmd!("eye")
            .env_remove("EYE_CONFIG")
            .env_remove("EYE_IR_CAMERA")
            .args(["emitter", "status"])
            .output()
            .unwrap()
    };
    let run = |action: &str| {
        cargo_bin_cmd!("eye")
            .env_remove("EYE_CONFIG")
            .env_remove("EYE_IR_CAMERA")
            .args(["emitter", action])
            .assert()
            .success();
    };

    let before = String::from_utf8_lossy(&run_status().stdout)
        .trim()
        .to_string();

    run("on");
    assert_eq!(String::from_utf8_lossy(&run_status().stdout).trim(), "on");

    run("off");
    assert_eq!(String::from_utf8_lossy(&run_status().stdout).trim(), "off");

    run(&before);
}
