pub mod error;
pub mod manifest;
pub mod report;
pub mod runner;

pub use error::BenchError;

#[cfg(feature = "llvm")]
pub fn run(args: &[String]) -> Result<i32, BenchError> {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    // ---- Parse args ----
    let mut manifest_path: Option<PathBuf> = None;
    let mut update_baseline = false;
    let mut no_diff = false;
    let mut out_path: Option<PathBuf> = None;
    let mut ea_passthrough: Vec<String> = Vec::new();
    let mut opt_level: u8 = 3;
    let mut target_cpu: Option<String> = None;
    let mut extra_features = String::new();

    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--update-baseline" => update_baseline = true,
            "--no-diff" => no_diff = true,
            "--out" => {
                i += 1;
                let p = args
                    .get(i)
                    .ok_or_else(|| BenchError::Unsupported("--out requires a path".into()))?;
                out_path = Some(PathBuf::from(p));
            }
            s if s.starts_with("--target=") => {
                let v = &s["--target=".len()..];
                target_cpu = (v != "native").then(|| v.to_string());
                ea_passthrough.push(s.to_string());
            }
            s if s.starts_with("--opt-level=") => {
                let v = &s["--opt-level=".len()..];
                opt_level = v
                    .parse::<u8>()
                    .map_err(|e| BenchError::Unsupported(format!("--opt-level: {e}")))?;
                ea_passthrough.push(s.to_string());
            }
            "--avx512" => {
                extra_features = "+avx512f,+avx512vl,+avx512bw".into();
                ea_passthrough.push(a.clone());
            }
            "--fp16" => {
                append_feat(&mut extra_features, "+fullfp16");
                ea_passthrough.push(a.clone());
            }
            "--i8mm" => {
                append_feat(&mut extra_features, "+i8mm");
                ea_passthrough.push(a.clone());
            }
            "--dotprod" => {
                append_feat(&mut extra_features, "+dotprod");
                ea_passthrough.push(a.clone());
            }
            other if !other.starts_with('-') && manifest_path.is_none() => {
                manifest_path = Some(PathBuf::from(other));
            }
            other => return Err(BenchError::Unsupported(format!("unknown option '{other}'"))),
        }
        i += 1;
    }

    let manifest_path =
        manifest_path.ok_or_else(|| BenchError::Unsupported("expected <manifest.toml>".into()))?;

    // ---- Load manifest ----
    let text = std::fs::read_to_string(&manifest_path)
        .map_err(|e| BenchError::Io(e, manifest_path.clone()))?;
    let m = manifest::parse(&text, &manifest_path)?;

    // ---- Arch skip ----
    let host_arch = std::env::consts::ARCH.to_string();
    if !m.arch.iter().any(|a| a == &host_arch) {
        eprintln!("skipped: {} (not applicable on {})", m.name, host_arch);
        return Ok(0);
    }

    // ---- Set up tempdir ----
    let td = tempfile::tempdir().map_err(|e| BenchError::Io(e, PathBuf::from(".")))?;

    // ---- Locate our own `ea` binary ----
    let ea_self = std::env::current_exe().map_err(|e| BenchError::Io(e, PathBuf::from(".")))?;
    let cc_binary = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let use_taskset = cfg!(target_os = "linux") && runner::taskset_available();

    let opts = runner::Options {
        ea_binary: &ea_self,
        cc_binary: &cc_binary,
        tempdir: td.path(),
        ea_passthrough: &ea_passthrough,
        use_taskset,
    };

    // ---- Run ----
    let measurements = runner::run(&m, &opts)?;

    // ---- Assemble result ----
    let host_cpu = host_cpu_name();
    let timestamp = report::iso8601_utc(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    );

    let result = report::Result_ {
        schema_version: 1,
        name: m.name.clone(),
        eacompute_version: env!("CARGO_PKG_VERSION").to_string(),
        git_sha: git_short_sha(),
        timestamp,
        env: report::Env {
            os: std::env::consts::OS.into(),
            arch: host_arch,
            host_cpu,
            target_cpu: target_cpu.unwrap_or_else(|| "native".into()),
            target_features: extra_features,
            opt_level,
            pinned: use_taskset,
        },
        measurements,
    };

    // ---- Emit JSON ----
    let json = report::write_result_json(&result);
    match &out_path {
        Some(p) => std::fs::write(p, &json).map_err(|e| BenchError::Io(e, p.clone()))?,
        None => print!("{json}"),
    }

    // ---- Diff ----
    if !no_diff {
        if m.baseline.exists() {
            match report::read_baseline(&m.baseline) {
                Ok(baseline) => {
                    let d = report::diff(&result, &baseline);
                    if d.stale {
                        eprintln!(
                            "baseline mismatch: {} — skipping diff (use --update-baseline to refresh)",
                            d.stale_reason.unwrap()
                        );
                    } else {
                        eprint!("{}", report::format_diff(&d, &result));
                    }
                }
                Err(e) => eprintln!(
                    "warning: cannot read baseline {}: {e}",
                    m.baseline.display()
                ),
            }
        } else {
            eprintln!(
                "no baseline yet — run with --update-baseline to create one ({})",
                m.baseline.display()
            );
        }
    }

    // ---- Update baseline if requested ----
    if update_baseline {
        std::fs::write(&m.baseline, &json).map_err(|e| BenchError::Io(e, m.baseline.clone()))?;
        eprintln!("wrote baseline {}", m.baseline.display());
    }

    Ok(0)
}

#[cfg(feature = "llvm")]
fn append_feat(features: &mut String, feat: &str) {
    if features.is_empty() {
        *features = feat.to_string();
    } else {
        *features = format!("{features},{feat}");
    }
}

#[cfg(feature = "llvm")]
fn host_cpu_name() -> String {
    use inkwell::targets::{InitializationConfig, Target, TargetMachine};
    let _ = Target::initialize_native(&InitializationConfig::default());
    TargetMachine::get_host_cpu_name()
        .to_str()
        .unwrap_or("unknown")
        .to_string()
}

#[cfg(feature = "llvm")]
fn git_short_sha() -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}
