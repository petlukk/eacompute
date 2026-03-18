#[cfg(feature = "llvm")]
mod tests {
    use ea_compiler::{CompileOptions, inspect_source};

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_inspect_simd_kernel_has_vector_instructions() {
        let source = r#"
            export kernel vscale(data: *f32, out: *mut f32, factor: f32)
                over i in n step 4
            {
                let v: f32x4 = load(data, i)
                store(out, i, v .* splat(factor))
            }
        "#;
        let report = inspect_source(source, &CompileOptions::default()).unwrap();
        let func = report
            .functions
            .iter()
            .find(|f| f.name == "vscale")
            .expect("vscale not found in report");
        assert!(
            func.vector_instructions > 0,
            "SIMD kernel should have vector instructions, got 0"
        );
        assert!(func.exported, "vscale should be exported");
    }

    #[test]
    fn test_inspect_scalar_function_no_vector() {
        let source = r#"
            export func add(a: i32, b: i32) -> i32 {
                return a + b
            }
        "#;
        let report = inspect_source(source, &CompileOptions::default()).unwrap();
        let func = report
            .functions
            .iter()
            .find(|f| f.name == "add")
            .expect("add not found in report");
        assert_eq!(
            func.vector_instructions, 0,
            "scalar function should have 0 vector instructions"
        );
    }

    #[test]
    fn test_inspect_kernel_has_loops() {
        let source = r#"
            export kernel inc(data: *i32, out: *mut i32)
                over i in n step 1
            {
                out[i] = data[i] + 1
            }
        "#;
        let report = inspect_source(source, &CompileOptions::default()).unwrap();
        let func = report
            .functions
            .iter()
            .find(|f| f.name == "inc")
            .expect("inc not found in report");
        assert!(
            func.loops >= 1,
            "kernel with while-loop should report >= 1 loop, got {}",
            func.loops
        );
    }

    #[test]
    fn test_inspect_multiple_exports() {
        let source = r#"
            export func alpha(x: i32) -> i32 { return x + 1 }
            export func beta(x: i32) -> i32 { return x * 2 }
        "#;
        let report = inspect_source(source, &CompileOptions::default()).unwrap();
        let names: Vec<&str> = report.functions.iter().map(|f| f.name.as_str()).collect();
        assert!(names.contains(&"alpha"), "missing alpha in report");
        assert!(names.contains(&"beta"), "missing beta in report");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_inspect_memory_ops() {
        let source = r#"
            export kernel vscale(data: *f32, out: *mut f32, factor: f32)
                over i in n step 4
            {
                let v: f32x4 = load(data, i)
                store(out, i, v .* splat(factor))
            }
        "#;
        let report = inspect_source(source, &CompileOptions::default()).unwrap();
        let func = report
            .functions
            .iter()
            .find(|f| f.name == "vscale")
            .unwrap();
        assert!(
            func.loads > 0,
            "kernel with load() should count loads, got 0"
        );
        assert!(
            func.stores > 0,
            "kernel with store() should count stores, got 0"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_inspect_fma_count() {
        let source = r#"
            export kernel fma_kernel(a: *f32, b: *f32, c: *f32, out: *mut f32)
                over i in n step 4
            {
                let va: f32x4 = load(a, i)
                let vb: f32x4 = load(b, i)
                let vc: f32x4 = load(c, i)
                store(out, i, fma(va, vb, vc))
            }
        "#;
        let report = inspect_source(source, &CompileOptions::default()).unwrap();
        let func = report
            .functions
            .iter()
            .find(|f| f.name == "fma_kernel")
            .unwrap();
        assert!(
            func.fma_ops > 0,
            "kernel with fma() should detect FMA instructions, got 0"
        );
    }

    #[test]
    fn test_inspect_display_shows_memory_ops() {
        let source = r#"
            export func add(a: i32, b: i32) -> i32 { return a + b }
        "#;
        let report = inspect_source(source, &CompileOptions::default()).unwrap();
        let output = format!("{report}");
        assert!(
            output.contains("loads:"),
            "display should contain 'loads:', got:\n{output}"
        );
        assert!(
            output.contains("stores:"),
            "display should contain 'stores:', got:\n{output}"
        );
    }

    #[test]
    fn test_inspect_hint_no_simd() {
        let source = r#"
            export func scalar_sum(data: *i32, n: i32) -> i32 {
                let mut acc: i32 = 0
                let mut i: i32 = 0
                while i < n {
                    acc = acc + data[i]
                    i = i + 1
                }
                return acc
            }
        "#;
        let opts = CompileOptions {
            opt_level: 0,
            ..CompileOptions::default()
        };
        let report = inspect_source(source, &opts).unwrap();
        let func = report
            .functions
            .iter()
            .find(|f| f.name == "scalar_sum")
            .unwrap();
        assert!(
            func.hints.iter().any(|h| h.contains("vector")),
            "scalar loop should hint about SIMD, got hints: {:?}",
            func.hints
        );
    }

    #[test]
    fn test_inspect_display_shows_hints() {
        let source = r#"
            export func scalar_sum(data: *i32, n: i32) -> i32 {
                let mut acc: i32 = 0
                let mut i: i32 = 0
                while i < n {
                    acc = acc + data[i]
                    i = i + 1
                }
                return acc
            }
        "#;
        let opts = CompileOptions {
            opt_level: 0,
            ..CompileOptions::default()
        };
        let report = inspect_source(source, &opts).unwrap();
        let output = format!("{report}");
        assert!(
            output.contains("hint:"),
            "display should show hints section, got:\n{output}"
        );
    }

    #[test]
    fn test_inspect_display_format() {
        let source = r#"
            export func identity(x: i32) -> i32 { return x }
        "#;
        let report = inspect_source(source, &CompileOptions::default()).unwrap();
        let output = format!("{report}");
        assert!(
            output.contains("=== identity (exported) ==="),
            "display should contain function name with export tag, got:\n{output}"
        );
        assert!(
            output.contains("vector instructions:"),
            "display should contain 'vector instructions:', got:\n{output}"
        );
        assert!(
            output.contains("scalar instructions:"),
            "display should contain 'scalar instructions:', got:\n{output}"
        );
        assert!(
            output.contains("loops:"),
            "display should contain 'loops:', got:\n{output}"
        );
    }
}
