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
