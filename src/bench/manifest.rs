use crate::bench::error::BenchError;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub path: PathBuf,
    pub name: String,
    pub kernel: PathBuf,
    pub harness: PathBuf,
    pub baseline: PathBuf,
    pub arch: Vec<String>,
    pub ea_flags: Vec<String>,
    pub cc_flags: Vec<String>,
}

pub fn parse(text: &str, manifest_path: &Path) -> Result<Manifest, BenchError> {
    let dir = manifest_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));

    let mut name: Option<String> = None;
    let mut kernel: Option<String> = None;
    let mut harness: Option<String> = None;
    let mut baseline: Option<String> = None;
    let mut arch: Option<Vec<String>> = None;
    let mut ea_flags: Vec<String> = Vec::new();
    let mut cc_flags: Option<Vec<String>> = None;

    for (idx, raw_line) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let eq = line.find('=').ok_or_else(|| BenchError::ManifestParse {
            path: manifest_path.to_path_buf(),
            line: line_no,
            msg: format!("expected '=' in '{line}'"),
        })?;
        let key = line[..eq].trim();
        let value = line[eq + 1..].trim();

        // Strip trailing inline comment ("# ...") outside of strings.
        let value = strip_inline_comment(value);

        match key {
            "name" => name = Some(parse_string(value, manifest_path, line_no)?),
            "kernel" => kernel = Some(parse_string(value, manifest_path, line_no)?),
            "harness" => harness = Some(parse_string(value, manifest_path, line_no)?),
            "baseline" => baseline = Some(parse_string(value, manifest_path, line_no)?),
            "arch" => arch = Some(parse_string_array(value, manifest_path, line_no)?),
            "ea_flags" => ea_flags = parse_string_array(value, manifest_path, line_no)?,
            "cc_flags" => cc_flags = Some(parse_string_array(value, manifest_path, line_no)?),
            other => {
                return Err(BenchError::ManifestParse {
                    path: manifest_path.to_path_buf(),
                    line: line_no,
                    msg: format!("unknown key '{other}'"),
                });
            }
        }
    }

    let name = name.ok_or_else(|| missing("name", manifest_path))?;
    let kernel = kernel.ok_or_else(|| missing("kernel", manifest_path))?;
    let harness = harness.ok_or_else(|| missing("harness", manifest_path))?;
    let baseline = baseline.ok_or_else(|| missing("baseline", manifest_path))?;
    let arch = arch.ok_or_else(|| missing("arch", manifest_path))?;

    if arch.is_empty() {
        return Err(BenchError::ManifestParse {
            path: manifest_path.to_path_buf(),
            line: 0,
            msg: "arch list must not be empty".to_string(),
        });
    }
    for a in &arch {
        match a.as_str() {
            "x86_64" | "aarch64" => {}
            other => {
                return Err(BenchError::ManifestParse {
                    path: manifest_path.to_path_buf(),
                    line: 0,
                    msg: format!("unknown arch '{other}' (expected 'x86_64' or 'aarch64')"),
                });
            }
        }
    }

    let cc_flags = cc_flags.unwrap_or_else(|| vec!["-O2".to_string()]);

    Ok(Manifest {
        path: manifest_path.to_path_buf(),
        name,
        kernel: resolve(&dir, &kernel),
        harness: resolve(&dir, &harness),
        baseline: resolve(&dir, &baseline),
        arch,
        ea_flags,
        cc_flags,
    })
}

fn resolve(dir: &Path, p: &str) -> PathBuf {
    let pb = PathBuf::from(p);
    if pb.is_absolute() { pb } else { dir.join(pb) }
}

fn missing(key: &str, path: &Path) -> BenchError {
    BenchError::ManifestParse {
        path: path.to_path_buf(),
        line: 0,
        msg: format!("missing required key '{key}'"),
    }
}

fn strip_inline_comment(v: &str) -> &str {
    let bytes = v.as_bytes();
    let mut in_str = false;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'"' => in_str = !in_str,
            b'#' if !in_str => return v[..i].trim(),
            _ => {}
        }
    }
    v.trim()
}

fn parse_string(v: &str, path: &Path, line: usize) -> Result<String, BenchError> {
    let bytes = v.as_bytes();
    if bytes.len() < 2 || bytes[0] != b'"' || bytes[bytes.len() - 1] != b'"' {
        return Err(BenchError::ManifestParse {
            path: path.to_path_buf(),
            line,
            msg: format!("expected quoted string, got: {v}"),
        });
    }
    Ok(v[1..v.len() - 1].to_string())
}

fn parse_string_array(v: &str, path: &Path, line: usize) -> Result<Vec<String>, BenchError> {
    let v = v.trim();
    if !v.starts_with('[') || !v.ends_with(']') {
        return Err(BenchError::ManifestParse {
            path: path.to_path_buf(),
            line,
            msg: format!("expected array, got: {v}"),
        });
    }
    let inner = &v[1..v.len() - 1];
    let inner = inner.trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for piece in inner.split(',') {
        let piece = piece.trim();
        out.push(parse_string(piece, path, line)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    #[test]
    fn parse_minimal_valid() {
        let text = r#"
            name = "tiny"
            kernel = "tiny.ea"
            harness = "tiny.c"
            baseline = "tiny.baseline.json"
            arch = ["x86_64"]
        "#;
        let m = parse(text, &p("/work/tiny.bench.toml")).unwrap();
        assert_eq!(m.name, "tiny");
        assert_eq!(m.kernel, p("/work/tiny.ea"));
        assert_eq!(m.harness, p("/work/tiny.c"));
        assert_eq!(m.baseline, p("/work/tiny.baseline.json"));
        assert_eq!(m.arch, vec!["x86_64".to_string()]);
        assert!(m.ea_flags.is_empty());
        assert_eq!(m.cc_flags, vec!["-O2".to_string()]);
    }

    #[test]
    fn parse_with_optional_flags() {
        let text = r#"
            name = "fp16_kv"
            kernel = "fp16_kv_bench.ea"
            harness = "fp16_kv_harness.c"
            baseline = "fp16_kv.baseline.json"
            arch = ["aarch64"]
            ea_flags = ["--fp16"]
            cc_flags = ["-O2", "-lm"]
        "#;
        let m = parse(text, &p("/work/fp16_kv.bench.toml")).unwrap();
        assert_eq!(m.ea_flags, vec!["--fp16".to_string()]);
        assert_eq!(m.cc_flags, vec!["-O2".to_string(), "-lm".to_string()]);
    }

    #[test]
    fn comments_and_blank_lines_ignored() {
        let text = r#"
            # comment
            name = "x"
            # another
            kernel = "x.ea"
            harness = "x.c"
            baseline = "x.json"
            arch = ["x86_64"]

            # trailing
        "#;
        assert!(parse(text, &p("/x.bench.toml")).is_ok());
    }

    #[test]
    fn missing_required_key_errors() {
        let text = r#"
            name = "x"
            kernel = "x.ea"
            harness = "x.c"
            # baseline missing
            arch = ["x86_64"]
        "#;
        let err = parse(text, &p("/x.bench.toml")).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("baseline"), "got: {msg}");
    }

    #[test]
    fn unknown_key_errors() {
        let text = r#"
            name = "x"
            kernel = "x.ea"
            harness = "x.c"
            baseline = "x.json"
            arch = ["x86_64"]
            wat = "huh"
        "#;
        let err = parse(text, &p("/x.bench.toml")).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown key 'wat'"), "got: {msg}");
    }

    #[test]
    fn unknown_arch_errors() {
        let text = r#"
            name = "x"
            kernel = "x.ea"
            harness = "x.c"
            baseline = "x.json"
            arch = ["sparc"]
        "#;
        let err = parse(text, &p("/x.bench.toml")).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown arch 'sparc'"), "got: {msg}");
    }

    #[test]
    fn paths_resolve_relative_to_manifest_dir() {
        let text = r#"
            name = "x"
            kernel = "sub/x.ea"
            harness = "sub/x.c"
            baseline = "sub/x.json"
            arch = ["x86_64"]
        "#;
        let m = parse(text, &p("/work/x.bench.toml")).unwrap();
        assert_eq!(m.kernel, p("/work/sub/x.ea"));
    }

    #[test]
    fn absolute_paths_pass_through() {
        let text = r#"
            name = "x"
            kernel = "/abs/x.ea"
            harness = "/abs/x.c"
            baseline = "/abs/x.json"
            arch = ["x86_64"]
        "#;
        let m = parse(text, &p("/work/x.bench.toml")).unwrap();
        assert_eq!(m.kernel, p("/abs/x.ea"));
    }
}
