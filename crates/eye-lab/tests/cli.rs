use std::{
    fs,
    io::{BufRead, BufReader},
    path::Path,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

use nix::{
    sys::signal::{Signal, kill},
    unistd::Pid,
};

fn bin() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_eye-lab"));
    cmd.env_remove("EYE_CAMERA").env_remove("EYE_IR_CAMERA");
    cmd
}

fn write_lab_toml(dir: &Path, contents: &str) -> std::path::PathBuf {
    let path = dir.join("lab.toml");
    fs::write(&path, contents).expect("writing lab.toml");
    path
}

/// Polls `child` up to `deadline`, killing and panicking if it has not exited by then.
fn wait_within(mut child: Child, deadline: Duration) -> std::process::ExitStatus {
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().expect("polling child status") {
            return status;
        }
        if start.elapsed() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("process did not exit within {deadline:?}");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn test_cli_selftest_suite_exits_0_and_writes_reports() {
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out");
    let status = bin()
        .args(["run", "--suite", "selftest", "--out"])
        .arg(&out)
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(0));

    let json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(out.join("report.json")).unwrap()).unwrap();
    assert_eq!(json["tally"]["pass"], 2);
    assert_eq!(json["tally"]["fail"], 0);
    assert_eq!(json["tally"]["skipped"], 1);
    assert_eq!(json["tally"]["error"], 0);
    assert_eq!(json["exit_code"], 0);

    let md = fs::read_to_string(out.join("report.md")).unwrap();
    assert!(md.starts_with("# eye-lab report: selftest"), "{md}");
}

#[test]
fn test_cli_failing_check_exits_1() {
    let tmp = tempfile::tempdir().unwrap();
    let file = write_lab_toml(
        tmp.path(),
        "name = \"x\"\n[[step]]\ntest = \"selftest-check\"\nparams = { value = 1.0 }\n",
    );
    let out = tmp.path().join("out");
    let status = bin()
        .args(["run", "--file"])
        .arg(&file)
        .arg("--out")
        .arg(&out)
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(1));

    let json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(out.join("report.json")).unwrap()).unwrap();
    assert_eq!(json["steps"][0]["verdict"], "fail");
}

#[test]
fn test_cli_unknown_test_exits_2_without_report() {
    let tmp = tempfile::tempdir().unwrap();
    let file = write_lab_toml(tmp.path(), "name = \"x\"\n[[step]]\ntest = \"nope\"\n");
    let out = tmp.path().join("out");
    let result = bin()
        .args(["run", "--file"])
        .arg(&file)
        .arg("--out")
        .arg(&out)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("unknown test \"nope\""), "{stderr}");
    assert!(!out.exists());
}

#[test]
fn test_cli_usage_error_exits_2() {
    let result = bin().arg("run").output().unwrap();
    assert_eq!(result.status.code(), Some(2));

    let tmp = tempfile::tempdir().unwrap();
    let file = write_lab_toml(
        tmp.path(),
        "name = \"x\"\n[[step]]\ntest = \"selftest-check\"\n",
    );
    let result = bin()
        .args(["run", "--suite", "selftest", "--file"])
        .arg(&file)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
}

#[test]
fn test_cli_unknown_suite_exits_2() {
    let result = bin().args(["run", "--suite", "nope"]).output().unwrap();
    assert_eq!(result.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("selftest"), "{stderr}");
}

#[test]
fn test_cli_hung_test_exits_2_promptly() {
    let tmp = tempfile::tempdir().unwrap();
    let file = write_lab_toml(
        tmp.path(),
        "name = \"x\"\n[[step]]\ntest = \"selftest-sleep\"\nparams = { seconds = 30.0, ignore_cancel = true }\ntimeout_s = 0.2\n",
    );
    let out = tmp.path().join("out");
    let child = bin()
        .args(["run", "--file"])
        .arg(&file)
        .arg("--out")
        .arg(&out)
        .arg("--grace-s")
        .arg("0.3")
        .spawn()
        .unwrap();
    let status = wait_within(child, Duration::from_secs(5));
    assert_eq!(status.code(), Some(2));

    let json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(out.join("report.json")).unwrap()).unwrap();
    let aborted = json["aborted"].as_str().unwrap();
    assert!(aborted.starts_with("hung test"), "{aborted}");
}

#[test]
fn test_cli_sigint_interrupts_and_exits_2() {
    let tmp = tempfile::tempdir().unwrap();
    let file = write_lab_toml(
        tmp.path(),
        "name = \"x\"\n[[step]]\ntest = \"selftest-sleep\"\nparams = { seconds = 30.0 }\n[[step]]\ntest = \"selftest-check\"\n",
    );
    let out = tmp.path().join("out");
    let mut child = bin()
        .args(["run", "--file"])
        .arg(&file)
        .arg("--out")
        .arg(&out)
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).unwrap();
        assert!(n > 0, "child exited before printing the start line");
        if line.starts_with("> [1] ") {
            break;
        }
    }

    kill(Pid::from_raw(child.id() as i32), Signal::SIGINT).unwrap();
    let status = wait_within(child, Duration::from_secs(5));
    assert_eq!(status.code(), Some(2));

    let json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(out.join("report.json")).unwrap()).unwrap();
    assert_eq!(json["steps"][0]["verdict"], "error");
    assert_eq!(json["steps"][0]["reason"], "interrupted");
    assert_eq!(json["steps"][1]["reason"], "not run: interrupted");
    assert_eq!(json["aborted"], "interrupted");
}

#[test]
fn test_cli_second_sigint_exits_130() {
    let tmp = tempfile::tempdir().unwrap();
    let file = write_lab_toml(
        tmp.path(),
        "name = \"x\"\n[[step]]\ntest = \"selftest-sleep\"\nparams = { seconds = 30.0, ignore_cancel = true }\n",
    );
    let out = tmp.path().join("out");
    let mut child = bin()
        .args(["run", "--file"])
        .arg(&file)
        .arg("--out")
        .arg(&out)
        .arg("--grace-s")
        .arg("10")
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).unwrap();
        assert!(n > 0, "child exited before printing the start line");
        if line.starts_with("> [1] ") {
            break;
        }
    }

    kill(Pid::from_raw(child.id() as i32), Signal::SIGINT).unwrap();
    std::thread::sleep(Duration::from_millis(100));
    kill(Pid::from_raw(child.id() as i32), Signal::SIGINT).unwrap();

    let status = wait_within(child, Duration::from_secs(5));
    assert_eq!(status.code(), Some(130));
    assert!(!out.exists());
}

#[test]
fn test_cli_no_subject_skips_subject_tests() {
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out");
    let status = bin()
        .args(["run", "--suite", "selftest", "--no-subject", "--out"])
        .arg(&out)
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(0));

    let json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(out.join("report.json")).unwrap()).unwrap();
    assert_eq!(json["tally"]["skipped"], 2);
}

#[test]
fn test_cli_list_tests_and_suites() {
    let result = bin().arg("list-tests").output().unwrap();
    assert_eq!(result.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.lines().any(|l| l.starts_with("selftest-check")),
        "{stdout}"
    );

    let result = bin().arg("suites").output().unwrap();
    assert_eq!(result.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.lines().any(|l| l.starts_with("selftest")),
        "{stdout}"
    );
}

#[test]
fn test_cli_help_shows_subcommand_and_arg_descriptions() {
    let result = bin().arg("--help").output().unwrap();
    assert_eq!(result.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(stdout.contains("Run a sequence of tests"), "{stdout}");
    assert!(
        stdout.contains("List the registered test cases"),
        "{stdout}"
    );
    assert!(stdout.contains("List the built-in suites"), "{stdout}");

    let result = bin().args(["run", "--help"]).output().unwrap();
    assert_eq!(result.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.contains("Skip every test that needs a person in front of the camera"),
        "{stdout}"
    );
    assert!(
        stdout.contains("Report directory (default: lab-reports/<unix seconds>-<sequence>)"),
        "{stdout}"
    );
    assert!(
        stdout.contains(
            "Seconds a timed-out or interrupted test may take to stop before it is abandoned"
        ),
        "{stdout}"
    );
    assert!(
        stdout.contains("A built-in suite (see `eye-lab suites`)"),
        "{stdout}"
    );
    assert!(stdout.contains("A lab.toml file"), "{stdout}");
}

#[test]
fn test_cli_modes_json_is_an_array() {
    let result = bin().args(["modes", "--json"]).output().unwrap();
    assert_eq!(result.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&result.stdout);
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert!(value.is_array(), "{stdout}");
}

#[test]
fn test_cli_unknown_camera_node_exits_2() {
    let result = bin()
        .args(["run", "--suite", "selftest", "--rgb", "/dev/eye-lab-none"])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("/dev/eye-lab-none"), "{stderr}");
}

#[test]
fn test_cli_grace_s_zero_error_names_valid_range() {
    let result = bin()
        .args(["run", "--suite", "selftest", "--grace-s", "0"])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("must be > 0 and <= 86400, got 0"),
        "{stderr}"
    );
    assert!(!stderr.contains("0..=86400"), "{stderr}");
}
