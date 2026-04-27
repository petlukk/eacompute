#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use ea_compiler::{CompileOptions, OutputMode};
    use std::process::Command;
    use tempfile::TempDir;

    /// Smoke test: exp_poly_f32(splat(0.0)) should give all 1.0s.
    /// Exercises the full pipeline (range reduction with n=0, polynomial
    /// returns 1, ldexp adds 0 to exponent — round trip).
    #[test]
    fn test_exp_poly_f32x4_at_zero() {
        let ea = r#"
            export func k(out: *mut f32) {
                let z: f32x4 = splat(0.0)
                let r: f32x4 = exp_poly_f32(z)
                store(out, 0, r)
            }
        "#;
        let c = r#"
            #include <stdio.h>
            extern void k(float *out);
            int main(void) {
                float out[4] = {0};
                k(out);
                for (int i = 0; i < 4; ++i) printf("%g\n", out[i]);
                return 0;
            }
        "#;
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("k.o");
        let cpath = dir.path().join("h.c");
        let bin = dir.path().join("k_bin");
        let opts = CompileOptions {
            opt_level: 3,
            target_cpu: None,
            extra_features: String::new(),
            target_triple: None,
        };
        ea_compiler::compile_with_options(ea, &obj, OutputMode::ObjectFile, &opts)
            .expect("compile failed");
        std::fs::write(&cpath, c).expect("write c");
        let status = Command::new("cc")
            .args([
                cpath.to_str().unwrap(),
                obj.to_str().unwrap(),
                "-o",
                bin.to_str().unwrap(),
                "-lm",
            ])
            .status()
            .expect("link failed");
        assert!(status.success(), "linker failed");
        let out = Command::new(&bin).output().expect("run failed");
        let stdout = String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n");
        assert_eq!(stdout.trim(), "1\n1\n1\n1");
    }
}
