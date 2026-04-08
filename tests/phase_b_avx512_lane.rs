//! AVX-512 lane-movement intrinsic tests: concat, lo/hi extractors,
//! per-sublane 32-bit broadcasts. IR-based verification following the
//! pattern established in tests/phase_b_avx512_dotprod.rs.
//!
//! Tests added incrementally by Tasks 2-4 of the lane intrinsics plan.

#[cfg(all(feature = "llvm", target_arch = "x86_64"))]
mod tests {
    use ea_compiler::{CompileOptions, OutputMode, compile_with_options};
    use tempfile::TempDir;

    #[allow(dead_code)]
    fn avx512_opts() -> CompileOptions {
        CompileOptions {
            opt_level: 0,
            extra_features: "+avx512f,+avx512vl,+avx512bw".to_string(),
            ..CompileOptions::default()
        }
    }

    #[allow(dead_code)]
    fn compile_to_ir(source: &str, name: &str) -> String {
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join(format!("{name}.ll"));
        compile_with_options(source, &ir_path, OutputMode::LlvmIr, &avx512_opts())
            .unwrap_or_else(|e| panic!("compile failed: {e:?}"));
        std::fs::read_to_string(&ir_path).unwrap_or_default()
    }
}
