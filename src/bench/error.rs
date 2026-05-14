use std::fmt;
use std::io;
use std::path::PathBuf;

#[derive(Debug)]
pub enum BenchError {
    Io(io::Error, PathBuf),
    ManifestParse {
        path: PathBuf,
        line: usize,
        msg: String,
    },
    JsonParse {
        context: String,
        msg: String,
    },
    HarnessLine {
        line_no: usize,
        raw: String,
        msg: String,
    },
    EaBuild {
        stderr: String,
    },
    CcBuild {
        stderr: String,
    },
    HarnessRun {
        exit_code: i32,
        stderr_tail: String,
    },
    Unsupported(String),
}

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BenchError::Io(e, p) => write!(f, "io error reading '{}': {e}", p.display()),
            BenchError::ManifestParse { path, line, msg } => {
                write!(
                    f,
                    "manifest parse error at {}:{line}: {msg}",
                    path.display()
                )
            }
            BenchError::JsonParse { context, msg } => {
                write!(f, "json parse error ({context}): {msg}")
            }
            BenchError::HarnessLine { line_no, raw, msg } => {
                write!(f, "malformed harness output line {line_no} ({msg}): {raw}")
            }
            BenchError::EaBuild { stderr } => write!(f, "ea --lib failed:\n{stderr}"),
            BenchError::CcBuild { stderr } => write!(f, "cc failed:\n{stderr}"),
            BenchError::HarnessRun {
                exit_code,
                stderr_tail,
            } => {
                write!(f, "harness exited with code {exit_code}\n{stderr_tail}")
            }
            BenchError::Unsupported(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for BenchError {}
