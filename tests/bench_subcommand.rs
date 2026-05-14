//! End-to-end test for `ea bench`. Invokes the actual `ea` binary built by
//! `cargo test`, runs the tiny fixture, and inspects stdout/stderr/exit/baseline.

#![cfg(feature = "llvm")]

use std::path::{Path, PathBuf};
use std::process::Command;

fn ea_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ea"))
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("bench")
}

fn copy_fixture(to: &Path) {
    for name in ["tiny.ea", "tiny_harness.c", "tiny.bench.toml"] {
        std::fs::copy(fixture_dir().join(name), to.join(name)).unwrap();
    }
}

fn run_ea_bench(workdir: &Path, extra_args: &[&str]) -> std::process::Output {
    let mut cmd = Command::new(ea_binary());
    cmd.arg("bench").arg("tiny.bench.toml").args(extra_args);
    cmd.current_dir(workdir);
    cmd.output().unwrap()
}

#[test]
fn first_run_emits_no_baseline_message() {
    let td = tempfile::tempdir().unwrap();
    copy_fixture(td.path());
    let out = run_ea_bench(td.path(), &["--no-diff"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("\"kernel\": \"noop\""),
        "stdout was: {stdout}"
    );
    assert!(stdout.contains("\"median_ns\": 42"));
    assert!(stdout.contains("\"schema_version\": 1"));
}

#[test]
fn no_baseline_message_when_file_absent() {
    let td = tempfile::tempdir().unwrap();
    copy_fixture(td.path());
    let out = run_ea_bench(td.path(), &[]);
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("no baseline yet"), "stderr was: {stderr}");
}

#[test]
fn update_baseline_writes_file_and_subsequent_run_diffs() {
    let td = tempfile::tempdir().unwrap();
    copy_fixture(td.path());

    let out = run_ea_bench(td.path(), &["--update-baseline"]);
    assert!(out.status.success());
    assert!(td.path().join("tiny.baseline.json").exists());

    let out2 = run_ea_bench(td.path(), &[]);
    assert!(out2.status.success());
    let stderr = String::from_utf8_lossy(&out2.stderr);
    assert!(stderr.contains("noop"), "stderr: {stderr}");
    assert!(stderr.contains("(baseline 42 ns"), "stderr: {stderr}");
    assert!(stderr.contains("0 regressions"));
}

#[test]
fn regressed_baseline_warns() {
    let td = tempfile::tempdir().unwrap();
    copy_fixture(td.path());
    let baseline = r#"{
  "schema_version": 1,
  "name": "tiny",
  "eacompute_version": "1.13.0",
  "git_sha": null,
  "timestamp": "2026-05-13T00:00:00Z",
  "env": {
    "os": "linux",
    "arch": "ARCH_PLACEHOLDER",
    "host_cpu": "any",
    "target_cpu": "native",
    "target_features": "",
    "opt_level": 3,
    "pinned": false
  },
  "measurements": [
    {"kernel": "noop", "median_ns": 30}
  ]
}
"#
    .replace("ARCH_PLACEHOLDER", std::env::consts::ARCH);
    std::fs::write(td.path().join("tiny.baseline.json"), baseline).unwrap();

    let out = run_ea_bench(td.path(), &[]);
    assert!(out.status.success(), "should be warn-only");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("1 regressions"), "stderr: {stderr}");
}

#[test]
fn stale_baseline_skips_diff() {
    let td = tempfile::tempdir().unwrap();
    copy_fixture(td.path());
    let baseline = r#"{
  "schema_version": 1,
  "name": "tiny",
  "eacompute_version": "1.13.0",
  "git_sha": null,
  "timestamp": "2026-05-13T00:00:00Z",
  "env": {
    "os": "linux",
    "arch": "ppc64le",
    "host_cpu": "any",
    "target_cpu": "native",
    "target_features": "",
    "opt_level": 3,
    "pinned": false
  },
  "measurements": [{"kernel": "noop", "median_ns": 42}]
}
"#;
    std::fs::write(td.path().join("tiny.baseline.json"), baseline).unwrap();
    let out = run_ea_bench(td.path(), &[]);
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("baseline mismatch"), "stderr: {stderr}");
}

#[test]
fn arch_skip() {
    let td = tempfile::tempdir().unwrap();
    copy_fixture(td.path());
    // Pick the arch that is NOT the host arch so the skip fires.
    let other_arch = if std::env::consts::ARCH == "x86_64" {
        "aarch64"
    } else {
        "x86_64"
    };
    let manifest = format!(
        r#"name = "tiny"
kernel = "tiny.ea"
harness = "tiny_harness.c"
baseline = "tiny.baseline.json"
arch = ["{other_arch}"]
"#
    );
    std::fs::write(td.path().join("tiny.bench.toml"), manifest).unwrap();
    let out = run_ea_bench(td.path(), &[]);
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("skipped"), "stderr: {stderr}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.is_empty(),
        "stdout should be empty when skipped, got: {stdout}"
    );
}
