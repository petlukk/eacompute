use crate::bench::error::BenchError;
use crate::bench::manifest::Manifest;
use crate::bench::report::Measurement;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Plan-only: what commands would `run` invoke for this manifest + options?
/// Used by tests; the real `run` builds these and executes them.
#[derive(Debug)]
pub struct RunPlan {
    pub ea_cmd: Vec<String>,
    pub cc_cmd: Vec<String>,
    pub harness_cmd: Vec<String>,
    pub lib_path: PathBuf,
    pub exe_path: PathBuf,
}

pub struct Options<'a> {
    pub ea_binary: &'a Path,
    pub cc_binary: &'a str,
    pub tempdir: &'a Path,
    pub ea_passthrough: &'a [String],
    pub use_taskset: bool,
}

pub fn build_plan(m: &Manifest, o: &Options<'_>) -> RunPlan {
    let lib_name = format!("{}_bench", m.name);
    let lib_path = o.tempdir.join(format!("lib{lib_name}.so"));
    let exe_path = o.tempdir.join(format!("{}_bench", m.name));

    // The `ea` CLI expects: ea <file.ea> [flags...]
    let mut ea = vec![o.ea_binary.display().to_string()];
    ea.push(m.kernel.display().to_string());
    ea.push("--lib".to_string());
    ea.push("-o".to_string());
    ea.push(lib_path.display().to_string());
    ea.extend(o.ea_passthrough.iter().cloned());
    ea.extend(m.ea_flags.iter().cloned());

    let mut cc = vec![o.cc_binary.to_string()];
    cc.push(m.harness.display().to_string());
    cc.push(format!("-L{}", o.tempdir.display()));
    cc.push(format!("-l{lib_name}"));
    cc.push(format!("-Wl,-rpath,{}", o.tempdir.display()));
    cc.push("-o".to_string());
    cc.push(exe_path.display().to_string());
    cc.extend(m.cc_flags.iter().cloned());

    let harness = if o.use_taskset {
        vec![
            "taskset".to_string(),
            "-c".to_string(),
            "0".to_string(),
            exe_path.display().to_string(),
        ]
    } else {
        vec![exe_path.display().to_string()]
    };

    RunPlan {
        ea_cmd: ea,
        cc_cmd: cc,
        harness_cmd: harness,
        lib_path,
        exe_path,
    }
}

pub fn taskset_available() -> bool {
    Command::new("taskset")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

pub fn run(m: &Manifest, o: &Options<'_>) -> std::result::Result<Vec<Measurement>, BenchError> {
    use crate::bench::report::parse_measurement_line;
    use std::io::Write;

    let plan = build_plan(m, o);

    // 1) Build kernel.
    let ea_out = Command::new(&plan.ea_cmd[0])
        .args(&plan.ea_cmd[1..])
        .output()
        .map_err(|e| BenchError::Io(e, PathBuf::from(&plan.ea_cmd[0])))?;
    let _ = std::io::stderr().write_all(&ea_out.stderr);
    if !ea_out.status.success() {
        return Err(BenchError::EaBuild {
            stderr: String::from_utf8_lossy(&ea_out.stderr).into_owned(),
        });
    }

    // 2) Build harness.
    let cc_out = Command::new(&plan.cc_cmd[0])
        .args(&plan.cc_cmd[1..])
        .output()
        .map_err(|e| BenchError::Io(e, PathBuf::from(&plan.cc_cmd[0])))?;
    if !cc_out.status.success() {
        return Err(BenchError::CcBuild {
            stderr: String::from_utf8_lossy(&cc_out.stderr).into_owned(),
        });
    }

    // 3) Run harness.
    let harness_out = Command::new(&plan.harness_cmd[0])
        .args(&plan.harness_cmd[1..])
        .output()
        .map_err(|e| BenchError::Io(e, PathBuf::from(&plan.harness_cmd[0])))?;

    let harness_stderr = String::from_utf8_lossy(&harness_out.stderr);
    for line in harness_stderr.lines() {
        eprintln!("[harness] {line}");
    }

    if !harness_out.status.success() {
        let code = harness_out.status.code().unwrap_or(-1);
        let tail: Vec<&str> = harness_stderr.lines().rev().take(20).collect();
        let tail: Vec<&str> = tail.into_iter().rev().collect();
        return Err(BenchError::HarnessRun {
            exit_code: code,
            stderr_tail: tail.join("\n"),
        });
    }

    // 4) Parse harness stdout (JSONL).
    let stdout = String::from_utf8_lossy(&harness_out.stdout);
    let mut measurements = Vec::new();
    for (idx, line) in stdout.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        measurements.push(parse_measurement_line(line, idx + 1)?);
    }
    Ok(measurements)
}

#[cfg(test)]
#[path = "runner_tests.rs"]
mod tests;
