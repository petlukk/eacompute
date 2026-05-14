use crate::bench::error::BenchError;
use std::path::Path;

pub const REGRESSION_THRESHOLD: f64 = 0.10;

#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    pub kernel: String,
    pub median_ns: u64,
    pub p10_ns: Option<u64>,
    pub p90_ns: Option<u64>,
    pub n_inner: Option<u64>,
    pub n_runs: Option<u64>,
    pub extra: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Env {
    pub os: String,
    pub arch: String,
    pub host_cpu: String,
    pub target_cpu: String,
    pub target_features: String,
    pub opt_level: u8,
    pub pinned: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Result_ {
    pub schema_version: u32,
    pub name: String,
    pub eacompute_version: String,
    pub git_sha: Option<String>,
    pub timestamp: String,
    pub env: Env,
    pub measurements: Vec<Measurement>,
}

// --- ISO 8601 (Howard Hinnant's date algorithm; epoch is unix) ---

pub fn iso8601_utc(unix_secs: u64) -> String {
    let total = unix_secs as i64;
    let days = total.div_euclid(86_400);
    let secs_today = total.rem_euclid(86_400) as u32;
    let hour = secs_today / 3600;
    let min = (secs_today % 3600) / 60;
    let sec = secs_today % 60;

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y_base = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y_base + 1 } else { y_base };

    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{min:02}:{sec:02}Z")
}

// --- JSON writer (string-builder, hand-rolled) ---

fn esc(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn write_result_json(r: &Result_) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str("  \"schema_version\": ");
    s.push_str(&r.schema_version.to_string());
    s.push_str(",\n  \"name\": ");
    esc(&r.name, &mut s);
    s.push_str(",\n  \"eacompute_version\": ");
    esc(&r.eacompute_version, &mut s);
    s.push_str(",\n  \"git_sha\": ");
    match &r.git_sha {
        Some(g) => esc(g, &mut s),
        None => s.push_str("null"),
    }
    s.push_str(",\n  \"timestamp\": ");
    esc(&r.timestamp, &mut s);
    s.push_str(",\n  \"env\": {\n");
    s.push_str("    \"os\": ");
    esc(&r.env.os, &mut s);
    s.push_str(",\n    \"arch\": ");
    esc(&r.env.arch, &mut s);
    s.push_str(",\n    \"host_cpu\": ");
    esc(&r.env.host_cpu, &mut s);
    s.push_str(",\n    \"target_cpu\": ");
    esc(&r.env.target_cpu, &mut s);
    s.push_str(",\n    \"target_features\": ");
    esc(&r.env.target_features, &mut s);
    s.push_str(",\n    \"opt_level\": ");
    s.push_str(&r.env.opt_level.to_string());
    s.push_str(",\n    \"pinned\": ");
    s.push_str(if r.env.pinned { "true" } else { "false" });
    s.push_str("\n  },\n");
    s.push_str("  \"measurements\": [");
    for (i, m) in r.measurements.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str("\n    {");
        s.push_str("\"kernel\": ");
        esc(&m.kernel, &mut s);
        s.push_str(", \"median_ns\": ");
        s.push_str(&m.median_ns.to_string());
        if let Some(v) = m.p10_ns {
            s.push_str(", \"p10_ns\": ");
            s.push_str(&v.to_string());
        }
        if let Some(v) = m.p90_ns {
            s.push_str(", \"p90_ns\": ");
            s.push_str(&v.to_string());
        }
        if let Some(v) = m.n_inner {
            s.push_str(", \"n_inner\": ");
            s.push_str(&v.to_string());
        }
        if let Some(v) = m.n_runs {
            s.push_str(", \"n_runs\": ");
            s.push_str(&v.to_string());
        }
        for (k, raw_val) in &m.extra {
            s.push_str(", ");
            esc(k, &mut s);
            s.push_str(": ");
            s.push_str(raw_val);
        }
        s.push('}');
    }
    s.push_str("\n  ]\n}\n");
    s
}

// --- JSON value parser (returns Vec<(key, raw_value_text)> for top-level object) ---

fn parse_object(
    text: &str,
    context: &str,
) -> std::result::Result<Vec<(String, String)>, BenchError> {
    let bytes = text.as_bytes();
    let mut i = 0;
    skip_ws(bytes, &mut i);
    if i >= bytes.len() || bytes[i] != b'{' {
        return Err(BenchError::JsonParse {
            context: context.into(),
            msg: "expected '{'".into(),
        });
    }
    i += 1;
    let mut out = Vec::new();
    skip_ws(bytes, &mut i);
    if i < bytes.len() && bytes[i] == b'}' {
        return Ok(out);
    }
    loop {
        skip_ws(bytes, &mut i);
        let key = parse_string_value(bytes, &mut i, context)?;
        skip_ws(bytes, &mut i);
        if i >= bytes.len() || bytes[i] != b':' {
            return Err(BenchError::JsonParse {
                context: context.into(),
                msg: format!("expected ':' after key '{key}'"),
            });
        }
        i += 1;
        skip_ws(bytes, &mut i);
        let value_start = i;
        skip_value(bytes, &mut i, context)?;
        let value_text = std::str::from_utf8(&bytes[value_start..i])
            .map_err(|_| BenchError::JsonParse {
                context: context.into(),
                msg: "non-utf8 value".into(),
            })?
            .trim()
            .to_string();
        out.push((key, value_text));
        skip_ws(bytes, &mut i);
        if i < bytes.len() && bytes[i] == b',' {
            i += 1;
            continue;
        }
        if i < bytes.len() && bytes[i] == b'}' {
            break;
        }
        return Err(BenchError::JsonParse {
            context: context.into(),
            msg: "expected ',' or '}'".into(),
        });
    }
    Ok(out)
}

fn skip_ws(bytes: &[u8], i: &mut usize) {
    while *i < bytes.len() && bytes[*i].is_ascii_whitespace() {
        *i += 1;
    }
}

fn parse_string_value(
    bytes: &[u8],
    i: &mut usize,
    context: &str,
) -> std::result::Result<String, BenchError> {
    if *i >= bytes.len() || bytes[*i] != b'"' {
        return Err(BenchError::JsonParse {
            context: context.into(),
            msg: "expected string".into(),
        });
    }
    *i += 1;
    let mut out = String::new();
    while *i < bytes.len() {
        match bytes[*i] {
            b'"' => {
                *i += 1;
                return Ok(out);
            }
            b'\\' => {
                *i += 1;
                if *i >= bytes.len() {
                    break;
                }
                match bytes[*i] {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    other => {
                        return Err(BenchError::JsonParse {
                            context: context.into(),
                            msg: format!("unsupported escape: \\{}", other as char),
                        });
                    }
                }
                *i += 1;
            }
            b => {
                out.push(b as char);
                *i += 1;
            }
        }
    }
    Err(BenchError::JsonParse {
        context: context.into(),
        msg: "unterminated string".into(),
    })
}

fn skip_value(bytes: &[u8], i: &mut usize, context: &str) -> std::result::Result<(), BenchError> {
    skip_ws(bytes, i);
    if *i >= bytes.len() {
        return Err(BenchError::JsonParse {
            context: context.into(),
            msg: "unexpected end of input".into(),
        });
    }
    match bytes[*i] {
        b'"' => {
            parse_string_value(bytes, i, context)?;
            Ok(())
        }
        b'{' => skip_balanced(bytes, i, b'{', b'}', context),
        b'[' => skip_balanced(bytes, i, b'[', b']', context),
        b't' | b'f' | b'n' => {
            while *i < bytes.len() && bytes[*i].is_ascii_alphabetic() {
                *i += 1;
            }
            Ok(())
        }
        b'-' | b'0'..=b'9' => {
            while *i < bytes.len()
                && (bytes[*i].is_ascii_digit()
                    || matches!(bytes[*i], b'-' | b'+' | b'.' | b'e' | b'E'))
            {
                *i += 1;
            }
            Ok(())
        }
        other => Err(BenchError::JsonParse {
            context: context.into(),
            msg: format!("unexpected byte '{}'", other as char),
        }),
    }
}

fn skip_balanced(
    bytes: &[u8],
    i: &mut usize,
    open: u8,
    close: u8,
    context: &str,
) -> std::result::Result<(), BenchError> {
    let mut depth = 0i32;
    while *i < bytes.len() {
        match bytes[*i] {
            b'"' => {
                parse_string_value(bytes, i, context)?;
                continue;
            }
            b if b == open => {
                depth += 1;
            }
            b if b == close => {
                depth -= 1;
                *i += 1;
                if depth == 0 {
                    return Ok(());
                }
                continue;
            }
            _ => {}
        }
        *i += 1;
    }
    Err(BenchError::JsonParse {
        context: context.into(),
        msg: "unbalanced delimiters".into(),
    })
}

fn parse_u64_value(s: &str, context: &str) -> std::result::Result<u64, BenchError> {
    s.trim().parse::<u64>().map_err(|e| BenchError::JsonParse {
        context: context.into(),
        msg: format!("expected u64, got '{s}': {e}"),
    })
}

fn parse_string_literal(s: &str, context: &str) -> std::result::Result<String, BenchError> {
    let bytes = s.as_bytes();
    let mut i = 0;
    parse_string_value(bytes, &mut i, context)
}

// --- JSONL line parser (one measurement per line) ---

pub fn parse_measurement_line(
    line: &str,
    line_no: usize,
) -> std::result::Result<Measurement, BenchError> {
    let context = format!("line {line_no}");
    let pairs = parse_object(line, &context).map_err(|e| match e {
        BenchError::JsonParse { msg, .. } => BenchError::HarnessLine {
            line_no,
            raw: line.to_string(),
            msg,
        },
        other => other,
    })?;
    let mut kernel: Option<String> = None;
    let mut median_ns: Option<u64> = None;
    let mut p10_ns = None;
    let mut p90_ns = None;
    let mut n_inner = None;
    let mut n_runs = None;
    let mut extra = Vec::new();
    for (k, v) in pairs {
        match k.as_str() {
            "kernel" => kernel = Some(parse_string_literal(&v, &context)?),
            "median_ns" => median_ns = Some(parse_u64_value(&v, &context)?),
            "p10_ns" => p10_ns = Some(parse_u64_value(&v, &context)?),
            "p90_ns" => p90_ns = Some(parse_u64_value(&v, &context)?),
            "n_inner" => n_inner = Some(parse_u64_value(&v, &context)?),
            "n_runs" => n_runs = Some(parse_u64_value(&v, &context)?),
            _ => extra.push((k, v)),
        }
    }
    let kernel = kernel.ok_or_else(|| BenchError::HarnessLine {
        line_no,
        raw: line.to_string(),
        msg: "missing 'kernel'".into(),
    })?;
    let median_ns = median_ns.ok_or_else(|| BenchError::HarnessLine {
        line_no,
        raw: line.to_string(),
        msg: "missing 'median_ns'".into(),
    })?;
    Ok(Measurement {
        kernel,
        median_ns,
        p10_ns,
        p90_ns,
        n_inner,
        n_runs,
        extra,
    })
}

// --- Baseline reader ---

pub fn read_baseline(path: &Path) -> std::result::Result<Result_, BenchError> {
    let text = std::fs::read_to_string(path).map_err(|e| BenchError::Io(e, path.to_path_buf()))?;
    let context = format!("baseline {}", path.display());
    let pairs = parse_object(&text, &context)?;
    let mut r = Result_ {
        schema_version: 0,
        name: String::new(),
        eacompute_version: String::new(),
        git_sha: None,
        timestamp: String::new(),
        env: Env {
            os: String::new(),
            arch: String::new(),
            host_cpu: String::new(),
            target_cpu: String::new(),
            target_features: String::new(),
            opt_level: 0,
            pinned: false,
        },
        measurements: Vec::new(),
    };
    for (k, v) in pairs {
        match k.as_str() {
            "schema_version" => r.schema_version = parse_u64_value(&v, &context)? as u32,
            "name" => r.name = parse_string_literal(&v, &context)?,
            "eacompute_version" => r.eacompute_version = parse_string_literal(&v, &context)?,
            "git_sha" => {
                r.git_sha = if v.trim() == "null" {
                    None
                } else {
                    Some(parse_string_literal(&v, &context)?)
                };
            }
            "timestamp" => r.timestamp = parse_string_literal(&v, &context)?,
            "env" => r.env = parse_env(&v, &context)?,
            "measurements" => r.measurements = parse_measurements_array(&v, &context)?,
            _ => {}
        }
    }
    Ok(r)
}

fn parse_env(v: &str, context: &str) -> std::result::Result<Env, BenchError> {
    let pairs = parse_object(v, context)?;
    let mut env = Env {
        os: String::new(),
        arch: String::new(),
        host_cpu: String::new(),
        target_cpu: String::new(),
        target_features: String::new(),
        opt_level: 0,
        pinned: false,
    };
    for (k, v) in pairs {
        match k.as_str() {
            "os" => env.os = parse_string_literal(&v, context)?,
            "arch" => env.arch = parse_string_literal(&v, context)?,
            "host_cpu" => env.host_cpu = parse_string_literal(&v, context)?,
            "target_cpu" => env.target_cpu = parse_string_literal(&v, context)?,
            "target_features" => env.target_features = parse_string_literal(&v, context)?,
            "opt_level" => env.opt_level = parse_u64_value(&v, context)? as u8,
            "pinned" => env.pinned = v.trim() == "true",
            _ => {}
        }
    }
    Ok(env)
}

fn parse_measurements_array(
    v: &str,
    context: &str,
) -> std::result::Result<Vec<Measurement>, BenchError> {
    let bytes = v.as_bytes();
    let mut i = 0;
    skip_ws(bytes, &mut i);
    if i >= bytes.len() || bytes[i] != b'[' {
        return Err(BenchError::JsonParse {
            context: context.into(),
            msg: "measurements: expected '['".into(),
        });
    }
    i += 1;
    let mut out = Vec::new();
    let mut line_no = 0;
    loop {
        skip_ws(bytes, &mut i);
        if i < bytes.len() && bytes[i] == b']' {
            return Ok(out);
        }
        let start = i;
        skip_value(bytes, &mut i, context)?;
        let obj_text = std::str::from_utf8(&bytes[start..i]).unwrap();
        line_no += 1;
        out.push(parse_measurement_line(obj_text, line_no)?);
        skip_ws(bytes, &mut i);
        if i < bytes.len() && bytes[i] == b',' {
            i += 1;
            continue;
        }
        if i < bytes.len() && bytes[i] == b']' {
            return Ok(out);
        }
        return Err(BenchError::JsonParse {
            context: context.into(),
            msg: "measurements: expected ',' or ']'".into(),
        });
    }
}

#[cfg(test)]
mod tests {
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
}
