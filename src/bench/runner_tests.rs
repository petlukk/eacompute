use super::*;
use std::path::PathBuf;

fn fixture_manifest() -> Manifest {
    Manifest {
        path: PathBuf::from("/work/x.bench.toml"),
        name: "x".into(),
        kernel: PathBuf::from("/work/x.ea"),
        harness: PathBuf::from("/work/x.c"),
        baseline: PathBuf::from("/work/x.baseline.json"),
        arch: vec!["x86_64".into()],
        ea_flags: vec!["--fp16".into()],
        cc_flags: vec!["-O2".into(), "-lm".into()],
    }
}

#[test]
fn plan_ea_args_include_lib_target_and_passthrough() {
    let td = PathBuf::from("/tmp/bench-test");
    let passthrough = vec!["--avx512".to_string(), "--opt-level=3".to_string()];
    let opts = Options {
        ea_binary: Path::new("/usr/local/bin/ea"),
        cc_binary: "cc",
        tempdir: &td,
        ea_passthrough: &passthrough,
        use_taskset: true,
    };
    let p = build_plan(&fixture_manifest(), &opts);
    assert_eq!(p.ea_cmd[0], "/usr/local/bin/ea");
    assert!(p.ea_cmd.contains(&"--lib".to_string()));
    assert!(p.ea_cmd.contains(&"--avx512".to_string()));
    assert!(p.ea_cmd.contains(&"--fp16".to_string()));
    assert!(p.ea_cmd.contains(&"--opt-level=3".to_string()));
    assert_eq!(p.ea_cmd.last().unwrap(), "/work/x.ea");
}

#[test]
fn plan_cc_links_against_kernel_lib() {
    let td = PathBuf::from("/tmp/bench-test");
    let opts = Options {
        ea_binary: Path::new("ea"),
        cc_binary: "cc",
        tempdir: &td,
        ea_passthrough: &[],
        use_taskset: false,
    };
    let p = build_plan(&fixture_manifest(), &opts);
    assert_eq!(p.cc_cmd[0], "cc");
    assert!(p.cc_cmd.iter().any(|a| a.starts_with("-L/tmp")));
    assert!(p.cc_cmd.contains(&"-lx_bench".to_string()));
    assert!(p.cc_cmd.contains(&"-Wl,-rpath,/tmp/bench-test".to_string()));
    assert!(p.cc_cmd.contains(&"-lm".to_string()));
}

#[test]
fn plan_harness_pins_when_requested() {
    let td = PathBuf::from("/tmp/bench-test");
    let opts = Options {
        ea_binary: Path::new("ea"),
        cc_binary: "cc",
        tempdir: &td,
        ea_passthrough: &[],
        use_taskset: true,
    };
    let p = build_plan(&fixture_manifest(), &opts);
    assert_eq!(p.harness_cmd[0], "taskset");
    assert_eq!(p.harness_cmd[1], "-c");
    assert_eq!(p.harness_cmd[2], "0");
}

#[test]
fn plan_harness_skips_taskset_when_not_requested() {
    let td = PathBuf::from("/tmp/bench-test");
    let opts = Options {
        ea_binary: Path::new("ea"),
        cc_binary: "cc",
        tempdir: &td,
        ea_passthrough: &[],
        use_taskset: false,
    };
    let p = build_plan(&fixture_manifest(), &opts);
    assert!(!p.harness_cmd[0].contains("taskset"));
}
