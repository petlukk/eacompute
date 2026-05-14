use crate::bench::error::BenchError;
use std::path::Path;

pub const REGRESSION_THRESHOLD: f64 = 0.10;

#[path = "report_writer.rs"]
mod writer;
pub use writer::{iso8601_utc, write_result_json};

#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    pub kernel: String,
    pub median_ns: u64,
    pub p10_ns: Option<u64>,
    pub p90_ns: Option<u64>,
    pub n_inner: Option<u64>,
    pub n_runs: Option<u64>,
    /// Pass-through unknown JSON keys from the harness line.
    ///
    /// The second element is **raw JSON value text** as it appeared in the
    /// input — e.g. `("speedup", "\"2.93x\"")` for a string value (note the
    /// embedded quotes), or `("ratio", "3.14")` for a number. Parsed-from-JSONL
    /// `Measurement`s carry this faithfully. If you construct one by hand and
    /// the extra value should be a JSON string, you must include the quotes
    /// yourself; for a JSON number, omit them.
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

fn decode_u_escape(
    bytes: &[u8],
    i: &mut usize,
    context: &str,
) -> std::result::Result<char, BenchError> {
    if *i + 4 > bytes.len() {
        return Err(BenchError::JsonParse {
            context: context.into(),
            msg: "\\u escape needs 4 hex digits".into(),
        });
    }
    let mut value: u32 = 0;
    for _ in 0..4 {
        let c = bytes[*i];
        let d = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            other => {
                return Err(BenchError::JsonParse {
                    context: context.into(),
                    msg: format!("invalid hex digit in \\u escape: '{}'", other as char),
                });
            }
        };
        value = value * 16 + d as u32;
        *i += 1;
    }
    if (0xD800..=0xDFFF).contains(&value) {
        return Err(BenchError::JsonParse {
            context: context.into(),
            msg: format!("surrogate code point U+{value:04X} not supported"),
        });
    }
    char::from_u32(value).ok_or_else(|| BenchError::JsonParse {
        context: context.into(),
        msg: format!("invalid code point U+{value:04X}"),
    })
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
    let mut run_start = *i;
    while *i < bytes.len() {
        match bytes[*i] {
            b'"' => {
                // SAFETY: bytes is the byte view of a &str, so any contiguous
                // range starting after a non-multibyte boundary (we only step
                // past whole UTF-8 sequences here) is itself valid UTF-8.
                out.push_str(std::str::from_utf8(&bytes[run_start..*i]).unwrap());
                *i += 1;
                return Ok(out);
            }
            b'\\' => {
                out.push_str(std::str::from_utf8(&bytes[run_start..*i]).unwrap());
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
                    b'u' => {
                        *i += 1;
                        let cp = decode_u_escape(bytes, i, context)?;
                        out.push(cp);
                        run_start = *i;
                        continue;
                    }
                    other => {
                        return Err(BenchError::JsonParse {
                            context: context.into(),
                            msg: format!("unsupported escape: \\{}", other as char),
                        });
                    }
                }
                *i += 1;
                run_start = *i;
            }
            _ => {
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
#[path = "report_tests.rs"]
mod tests;
