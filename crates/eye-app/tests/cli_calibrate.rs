use assert_cmd::cargo::cargo_bin_cmd;
use eye_bench::testing::{FOUR_BY_FOUR_CENTRES, SyntheticSession, write_synthetic_session};
use predicates::prelude::*;

#[test]
fn test_calibrate_from_missing_dir_exits_1() {
    cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .env_remove("EYE_CAMERA")
        .env_remove("EYE_IR_CAMERA")
        .env_remove("EYE_OUTPUT")
        .args(["calibrate", "--from", "/nonexistent"])
        .assert()
        .code(1)
        .stderr(predicate::str::contains("error:"));
}

#[test]
fn test_calibrate_from_blank_session_reports_no_samples() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let spec = SyntheticSession {
        targets: FOUR_BY_FOUR_CENTRES[..9].to_vec(),
        size: (640, 360),
        code_frames: false,
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
    let output_path = tmp.path().join("p.toml");

    cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .env_remove("EYE_CAMERA")
        .env_remove("EYE_IR_CAMERA")
        .env_remove("EYE_OUTPUT")
        .args([
            "calibrate",
            "--config",
            config_path.to_str().expect("utf8 path"),
            "--from",
            session_dir.to_str().expect("utf8 path"),
            "--output",
            output_path.to_str().expect("utf8 path"),
        ])
        .assert()
        .code(1)
        .stderr(predicate::str::contains("no gaze samples"));

    assert!(!output_path.exists());
}

#[test]
#[ignore = "needs hardware and wayland; subject present"]
fn test_calibrate_live() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let output_path = tmp.path().join("p.toml");

    cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .env_remove("EYE_CAMERA")
        .env_remove("EYE_IR_CAMERA")
        .env_remove("EYE_OUTPUT")
        .args([
            "calibrate",
            "--output",
            output_path.to_str().expect("utf8 path"),
        ])
        .assert()
        .success();

    eye_calibration::profiles::read_profile(&output_path).expect("profile parses");

    let runtime_dir = std::env::var_os("XDG_RUNTIME_DIR").map(std::path::PathBuf::from);
    if let Some(runtime_dir) = runtime_dir {
        let stray = std::fs::read_dir(&runtime_dir)
            .expect("read runtime dir")
            .filter_map(|e| e.ok())
            .any(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("eye-calibrate-")
            });
        assert!(!stray, "a temp recording directory survived the command");
    }
}
