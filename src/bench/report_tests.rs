use super::*;

fn sample_env() -> Env {
    Env {
        os: "linux".into(),
        arch: "x86_64".into(),
        host_cpu: "znver4".into(),
        target_cpu: "native".into(),
        target_features: "+avx512f".into(),
        opt_level: 3,
        pinned: true,
    }
}

fn sample_result() -> Result_ {
    Result_ {
        schema_version: 1,
        name: "tiny".into(),
        eacompute_version: "1.13.0".into(),
        git_sha: Some("f2ca320".into()),
        timestamp: "2026-05-14T10:23:00Z".into(),
        env: sample_env(),
        measurements: vec![Measurement {
            kernel: "noop".into(),
            median_ns: 42,
            p10_ns: Some(41),
            p90_ns: Some(43),
            n_inner: Some(200),
            n_runs: Some(10),
            extra: vec![],
        }],
    }
}

#[test]
fn parse_minimal_measurement() {
    let line = r#"{"kernel":"foo","median_ns":1234}"#;
    let m = parse_measurement_line(line, 1).unwrap();
    assert_eq!(m.kernel, "foo");
    assert_eq!(m.median_ns, 1234);
    assert_eq!(m.p10_ns, None);
}

#[test]
fn parse_full_measurement() {
    let line = r#"{"kernel":"foo","median_ns":1234,"p10_ns":1200,"p90_ns":1260,"n_inner":200,"n_runs":10}"#;
    let m = parse_measurement_line(line, 1).unwrap();
    assert_eq!(m.p10_ns, Some(1200));
    assert_eq!(m.p90_ns, Some(1260));
    assert_eq!(m.n_inner, Some(200));
    assert_eq!(m.n_runs, Some(10));
}

#[test]
fn parse_measurement_preserves_unknown_keys() {
    let line = r#"{"kernel":"foo","median_ns":1234,"speedup":"2.93x"}"#;
    let m = parse_measurement_line(line, 1).unwrap();
    assert_eq!(m.extra.len(), 1);
    assert_eq!(m.extra[0].0, "speedup");
    assert_eq!(m.extra[0].1, "\"2.93x\"");
}

#[test]
fn parse_missing_required_key_errors() {
    let line = r#"{"kernel":"foo"}"#;
    let err = parse_measurement_line(line, 7).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("median_ns"), "got: {msg}");
    assert!(msg.contains("line 7"), "got: {msg}");
}

#[test]
fn parse_malformed_json_errors() {
    let line = r#"{"kernel": broken"#;
    assert!(parse_measurement_line(line, 1).is_err());
}

#[test]
fn write_json_roundtrips_through_baseline_reader() {
    let r = sample_result();
    let text = write_result_json(&r);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("b.json");
    std::fs::write(&path, &text).unwrap();
    let r2 = read_baseline(&path).unwrap();
    assert_eq!(r, r2);
}

#[test]
fn write_json_starts_with_brace_ends_with_newline() {
    let text = write_result_json(&sample_result());
    assert!(text.starts_with('{'));
    assert!(text.ends_with("}\n"));
}

#[test]
fn iso8601_known_unix_time() {
    // 2026-05-14T10:23:00Z = 1778754180
    assert_eq!(iso8601_utc(1778754180), "2026-05-14T10:23:00Z");
    // Unix epoch
    assert_eq!(iso8601_utc(0), "1970-01-01T00:00:00Z");
    // Leap-day check: 2024-02-29T00:00:00Z = 1709164800
    assert_eq!(iso8601_utc(1709164800), "2024-02-29T00:00:00Z");
}

#[test]
fn write_json_roundtrips_with_git_sha_none() {
    let mut r = sample_result();
    r.git_sha = None;
    let text = write_result_json(&r);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("b.json");
    std::fs::write(&path, &text).unwrap();
    let r2 = read_baseline(&path).unwrap();
    assert_eq!(r, r2);
}

#[test]
fn write_json_roundtrips_control_char_in_string() {
    let mut r = sample_result();
    // Inject a U+0001 control char into a string field; writer emits ,
    // reader must decode it back. Covers Fix 2.
    r.env.target_features = "feat\u{0001}flag".into();
    let text = write_result_json(&r);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("b.json");
    std::fs::write(&path, &text).unwrap();
    let r2 = read_baseline(&path).unwrap();
    assert_eq!(r, r2);
}

#[test]
fn parse_measurement_handles_utf8_in_value() {
    // Non-ASCII multi-byte UTF-8 in the `kernel` field. Covers Fix 1.
    let line = r#"{"kernel":"naïve_kernel","median_ns":42}"#;
    let m = parse_measurement_line(line, 1).unwrap();
    assert_eq!(m.kernel, "naïve_kernel");
}

fn measurement(name: &str, ns: u64) -> Measurement {
    Measurement {
        kernel: name.into(),
        median_ns: ns,
        p10_ns: None,
        p90_ns: None,
        n_inner: None,
        n_runs: None,
        extra: vec![],
    }
}

fn result_with(env: Env, ms: Vec<Measurement>) -> Result_ {
    Result_ {
        schema_version: 1,
        name: "x".into(),
        eacompute_version: "1.13.0".into(),
        git_sha: None,
        timestamp: "2026-05-14T00:00:00Z".into(),
        env,
        measurements: ms,
    }
}

#[test]
fn diff_no_regression_below_threshold() {
    let base = result_with(sample_env(), vec![measurement("k", 1000)]);
    let curr = result_with(sample_env(), vec![measurement("k", 1099)]); // +9.9%
    let d = diff(&curr, &base);
    assert_eq!(d.regressions, 0);
    assert_eq!(d.kernels.len(), 1);
    assert!(!d.stale);
}

#[test]
fn diff_regression_above_threshold() {
    let base = result_with(sample_env(), vec![measurement("k", 1000)]);
    let curr = result_with(sample_env(), vec![measurement("k", 1101)]); // +10.1%
    let d = diff(&curr, &base);
    assert_eq!(d.regressions, 1);
}

#[test]
fn diff_threshold_boundary_inclusive_on_pass() {
    // Exactly +10.00% should NOT be flagged (strict > 10%)
    let base = result_with(sample_env(), vec![measurement("k", 1000)]);
    let curr = result_with(sample_env(), vec![measurement("k", 1100)]);
    let d = diff(&curr, &base);
    assert_eq!(d.regressions, 0);
}

#[test]
fn diff_speedup_does_not_regress() {
    let base = result_with(sample_env(), vec![measurement("k", 1000)]);
    let curr = result_with(sample_env(), vec![measurement("k", 500)]);
    let d = diff(&curr, &base);
    assert_eq!(d.regressions, 0);
}

#[test]
fn diff_missing_kernel_in_current_flagged() {
    let base = result_with(sample_env(), vec![measurement("a", 1), measurement("b", 2)]);
    let curr = result_with(sample_env(), vec![measurement("a", 1)]);
    let d = diff(&curr, &base);
    let b_row = d.kernels.iter().find(|k| k.kernel == "b").unwrap();
    assert!(matches!(b_row.status, KernelStatus::MissingFromCurrent));
}

#[test]
fn diff_new_kernel_flagged() {
    let base = result_with(sample_env(), vec![measurement("a", 1)]);
    let curr = result_with(sample_env(), vec![measurement("a", 1), measurement("c", 2)]);
    let d = diff(&curr, &base);
    let c_row = d.kernels.iter().find(|k| k.kernel == "c").unwrap();
    assert!(matches!(c_row.status, KernelStatus::NewKernel));
}

#[test]
fn diff_stale_on_arch_mismatch() {
    let mut e = sample_env();
    e.arch = "aarch64".into();
    let base = result_with(e, vec![measurement("k", 1000)]);
    let curr = result_with(sample_env(), vec![measurement("k", 1000)]);
    let d = diff(&curr, &base);
    assert!(d.stale);
    assert!(d.stale_reason.as_ref().unwrap().contains("arch"));
}

#[test]
fn diff_stale_on_features_mismatch() {
    let mut e = sample_env();
    e.target_features = "+sse2".into();
    let base = result_with(e, vec![measurement("k", 1000)]);
    let curr = result_with(sample_env(), vec![measurement("k", 1000)]);
    let d = diff(&curr, &base);
    assert!(d.stale);
    assert!(d.stale_reason.as_ref().unwrap().contains("target_features"));
}

#[test]
fn diff_stale_on_opt_level_mismatch() {
    let mut e = sample_env();
    e.opt_level = 2;
    let base = result_with(e, vec![measurement("k", 1000)]);
    let curr = result_with(sample_env(), vec![measurement("k", 1000)]);
    let d = diff(&curr, &base);
    assert!(d.stale);
    assert!(d.stale_reason.as_ref().unwrap().contains("opt_level"));
}

#[test]
fn format_diff_human_summary() {
    let base = result_with(sample_env(), vec![measurement("k", 1000)]);
    let curr = result_with(sample_env(), vec![measurement("k", 1099)]);
    let d = diff(&curr, &base);
    let txt = format_diff(&d, &curr);
    assert!(txt.contains("k"));
    assert!(txt.contains("1099"));
    assert!(txt.contains("+9.9%") || txt.contains("+9.90%"));
    assert!(txt.contains("0 regressions"));
}
