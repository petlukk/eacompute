#[cfg(feature = "llvm")]
mod tests {
    use std::process::Command;
    use tempfile::TempDir;

    #[test]
    fn test_output_mode_executable() {
        let dir = TempDir::new().expect("failed to create temp dir");
        let exe_name = if cfg!(target_os = "windows") {
            "test_exe.exe"
        } else {
            "test_exe"
        };
        let exe_path = dir.path().join(exe_name);

        ea_compiler::compile(
            r#"
            export func main() {
                println(42)
            }
            "#,
            &dir.path().join("unused.o"),
            ea_compiler::OutputMode::Executable(exe_path.to_str().unwrap().to_string()),
        )
        .expect("OutputMode::Executable compilation failed");

        assert!(exe_path.exists(), "executable was not created");

        let output = Command::new(&exe_path)
            .output()
            .expect("failed to run executable");
        let stdout = String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n");
        assert_eq!(stdout.trim(), "42");
    }
}
