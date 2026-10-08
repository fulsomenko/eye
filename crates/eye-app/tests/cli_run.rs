use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;

#[cfg(unix)]
fn emitter_status() -> String {
    String::from_utf8_lossy(
        &cargo_bin_cmd!("eye")
            .env_remove("EYE_CONFIG")
            .args(["emitter", "status"])
            .output()
            .expect("eye emitter status runs")
            .stdout,
    )
    .trim()
    .to_string()
}

#[test]
fn test_run_rejects_output_flag() {
    cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .args(["--output", "x", "run"])
        .assert()
        .code(1)
        .stderr(predicate::str::contains("eye run"));
}

#[test]
fn test_run_profile_conflicts_with_no_profile() {
    cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .args(["run", "--profile", "a", "--no-profile"])
        .assert()
        .code(2);
}

#[test]
fn test_run_bad_grid_exits_2() {
    cargo_bin_cmd!("eye")
        .env_remove("EYE_CONFIG")
        .args(["run", "--grid", "0x3"])
        .assert()
        .code(2);
}

#[test]
#[ignore = "needs hardware and wayland"]
fn test_run_stops_cleanly_on_sigint() {
    use std::io::Read;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    use nix::sys::signal::{Signal, kill};
    use nix::unistd::Pid;

    let before = emitter_status();

    let mut child = Command::new(assert_cmd::cargo::cargo_bin!("eye"))
        .env_remove("EYE_CONFIG")
        .args(["run", "--no-profile", "--stats"])
        .stderr(Stdio::piped())
        .spawn()
        .expect("eye run spawns");

    std::thread::sleep(Duration::from_secs(6));

    kill(Pid::from_raw(child.id() as i32), Signal::SIGINT).expect("sigint sent");

    let deadline = Instant::now() + Duration::from_secs(3);
    let status = loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "did not exit within 3s of SIGINT"
        );
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(status.code(), Some(0));

    let mut stderr = String::new();
    child
        .stderr
        .take()
        .expect("stderr piped")
        .read_to_string(&mut stderr)
        .expect("read stderr");
    assert!(stderr.contains("points in"), "stderr: {stderr}");

    assert_eq!(emitter_status(), before);
}

#[test]
#[ignore = "needs hardware and wayland"]
fn test_run_second_sigint_forces_exit() {
    use std::process::{Command, Stdio};
    use std::time::Duration;

    use nix::sys::signal::{Signal, kill};
    use nix::unistd::Pid;

    let mut child = Command::new(assert_cmd::cargo::cargo_bin!("eye"))
        .env_remove("EYE_CONFIG")
        .args(["run", "--no-profile", "--stats"])
        .stderr(Stdio::piped())
        .spawn()
        .expect("eye run spawns");

    std::thread::sleep(Duration::from_secs(6));

    let pid = Pid::from_raw(child.id() as i32);
    kill(pid, Signal::SIGINT).expect("first sigint sent");
    std::thread::sleep(Duration::from_millis(50));
    kill(pid, Signal::SIGINT).expect("second sigint sent");

    let status = child.wait().expect("child exits");
    assert_eq!(status.code(), Some(130));
}
