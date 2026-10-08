use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;

#[test]
fn test_cli_calibrate_camera_help_lists_modes() {
    let assert = cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .args(["calibrate-camera", "--help"])
        .assert()
        .success();
    let output = assert.get_output();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("intrinsics"), "stdout was: {stdout}");
    assert!(stdout.contains("stereo"), "stdout was: {stdout}");
}

#[test]
fn test_cli_intrinsics_requires_camera() {
    cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .args(["calibrate-camera", "intrinsics"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("--camera"));
}

#[test]
fn test_cli_unknown_board_rejected() {
    cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .args([
            "calibrate-camera",
            "intrinsics",
            "--camera",
            "ir",
            "--board",
            "poster",
        ])
        .assert()
        .code(2);
}
