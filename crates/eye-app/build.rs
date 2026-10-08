use std::process::Command;

fn main() {
    println!("cargo::rerun-if-env-changed=EYE_GIT_REV");
    println!("cargo::rerun-if-changed=../../.git/HEAD");
    println!("cargo::rerun-if-changed=../../.git/index");
    let rev = std::env::var("EYE_GIT_REV")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(git_rev)
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo::rustc-env=EYE_GIT_REV={rev}");
}

fn git_rev() -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let rev = String::from_utf8(out.stdout).ok()?.trim().to_owned();
    let dirty = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .is_ok_and(|o| !o.stdout.is_empty());
    Some(if dirty { format!("{rev}-dirty") } else { rev })
}
