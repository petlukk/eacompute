use std::process::Command;
use tempfile::TempDir;

pub struct TestOutput {
    pub stdout: String,
    #[allow(dead_code)]
    pub stderr: String,
    #[allow(dead_code)]
    pub exit_code: i32,
}

pub fn compile_and_run(source: &str) -> TestOutput {
    let dir = TempDir::new().expect("failed to create temp dir");
    let obj_path = dir.path().join("test.o");
    let bin_path = dir.path().join("test_bin");

    ea_compiler::compile(source, &obj_path, ea_compiler::OutputMode::ObjectFile)
        .expect("compilation failed");

    let link_status = Command::new("cc")
        .args([
            obj_path.to_str().unwrap(),
            "-o",
            bin_path.to_str().unwrap(),
            "-lm",
        ])
        .status()
        .expect("failed to invoke linker");
    assert!(link_status.success(), "linking failed");

    let output = Command::new(&bin_path)
        .output()
        .expect("failed to execute binary");
    TestOutput {
        stdout: String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n"),
        stderr: String::from_utf8_lossy(&output.stderr).replace("\r\n", "\n"),
        exit_code: output.status.code().unwrap_or(-1),
    }
}

#[allow(dead_code)]
pub fn compile_and_link_with_c(ea_source: &str, c_source: &str) -> TestOutput {
    let dir = TempDir::new().expect("failed to create temp dir");
    let obj_path = dir.path().join("kernel.o");
    let c_path = dir.path().join("harness.c");
    let bin_path = dir.path().join("test_bin");

    ea_compiler::compile(ea_source, &obj_path, ea_compiler::OutputMode::ObjectFile)
        .expect("compilation failed");
    std::fs::write(&c_path, c_source).expect("failed to write C harness");

    let link_status = Command::new("cc")
        .args([
            c_path.to_str().unwrap(),
            obj_path.to_str().unwrap(),
            "-o",
            bin_path.to_str().unwrap(),
            "-lm",
        ])
        .status()
        .expect("failed to invoke linker");
    assert!(link_status.success(), "linking C harness failed");

    let output = Command::new(&bin_path)
        .output()
        .expect("failed to execute binary");
    TestOutput {
        stdout: String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n"),
        stderr: String::from_utf8_lossy(&output.stderr).replace("\r\n", "\n"),
        exit_code: output.status.code().unwrap_or(-1),
    }
}

#[allow(dead_code)]
pub fn assert_output(source: &str, expected: &str) {
    let result = compile_and_run(source);
    assert_eq!(result.stdout.trim(), expected);
}

#[allow(dead_code)]
pub fn assert_output_lines(source: &str, expected: &[&str]) {
    let result = compile_and_run(source);
    let lines: Vec<&str> = result.stdout.trim().lines().collect();
    assert_eq!(lines, expected);
}

#[allow(dead_code)]
pub fn assert_c_interop(ea_source: &str, c_source: &str, expected: &str) {
    let result = compile_and_link_with_c(ea_source, c_source);
    assert_eq!(result.stdout.trim(), expected);
}

#[allow(dead_code)]
pub fn assert_typecheck_error(src: &str, expected_substr: &str) {
    let dir = TempDir::new().unwrap();
    let obj = dir.path().join("t.o");
    let err = ea_compiler::compile(src, &obj, ea_compiler::OutputMode::ObjectFile)
        .expect_err("expected type error");
    let msg = format!("{err}");
    assert!(
        msg.contains(expected_substr),
        "expected error to contain {expected_substr:?}, got: {msg}"
    );
}

#[allow(dead_code)]
pub fn assert_shared_lib_interop(ea_source: &str, c_source: &str, expected: &str) {
    let dir = TempDir::new().expect("failed to create temp dir");
    let so_path = dir.path().join("libkernel.so");
    let c_path = dir.path().join("harness.c");
    let bin_path = dir.path().join("test_bin");

    ea_compiler::compile(
        ea_source,
        &dir.path().join("kernel.o"),
        ea_compiler::OutputMode::SharedLib(so_path.to_str().unwrap().to_string()),
    )
    .expect("shared lib compilation failed");

    assert!(so_path.exists(), "shared library was not created");

    std::fs::write(&c_path, c_source).expect("failed to write C harness");

    let link_status = Command::new("cc")
        .args([
            c_path.to_str().unwrap(),
            "-o",
            bin_path.to_str().unwrap(),
            so_path.to_str().unwrap(),
            "-lm",
            &format!("-Wl,-rpath,{}", dir.path().to_str().unwrap()),
        ])
        .status()
        .expect("failed to invoke linker");
    assert!(link_status.success(), "linking shared lib harness failed");

    let output = Command::new(&bin_path)
        .output()
        .expect("failed to execute binary");
    let stdout = String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n");
    assert_eq!(stdout.trim(), expected);
}

#[allow(dead_code)]
pub fn compile_to_ir(source: &str) -> String {
    ea_compiler::compile_to_ir(source).expect("compilation failed")
}

/// Run a transcendental approximation intrinsic over `inputs` lane-by-lane,
/// compare each output to the corresponding libm reference, and assert that
/// every absolute error is within `abs_tol`. Shared by the phase14
/// transcendental test files (`tanh_approx`, `log_approx`, `sin_cos_approx`).
///
/// - `vector_type`: `"f32x4"` or `"f32x8"` (sets `lanes` to 4 or 8).
/// - `intrinsic`: Eä intrinsic name, e.g. `"tanh_approx_f32"`.
/// - `libm_ref`: C reference function, e.g. `"tanhf"`.
/// - `padding`: value used to pad `inputs` up to a multiple of `lanes` so the
///   kernel's `while i + lanes <= n` loop covers the original input range.
///   Must satisfy `intrinsic(padding) ≈ libm_ref(padding)` within `abs_tol`
///   (use `0.0` for tanh / sin, `1.0` for log, etc.).
#[allow(dead_code)]
pub fn assert_transcendental_accuracy(
    inputs: &[f32],
    vector_type: &str,
    intrinsic: &str,
    libm_ref: &str,
    abs_tol: f32,
    padding: f32,
) {
    use ea_compiler::{CompileOptions, OutputMode};

    let lanes = if vector_type == "f32x4" { 4 } else { 8 };

    let ea = format!(
        r#"
        export func k(input: *f32, output: *mut f32, n: i32) {{
            let mut i: i32 = 0
            while i + {lanes} <= n {{
                let v: {vector_type} = load(input, i)
                let r: {vector_type} = {intrinsic}(v)
                store(output, i, r)
                i = i + {lanes}
            }}
        }}
        "#
    );

    let mut padded = inputs.to_vec();
    while !padded.len().is_multiple_of(lanes) {
        padded.push(padding);
    }
    let n = padded.len();
    let original_n = inputs.len();

    let inputs_str = padded
        .iter()
        .map(|f| format!("{f:.10e}f"))
        .collect::<Vec<_>>()
        .join(", ");

    let c = format!(
        r#"
        #include <stdio.h>
        #include <math.h>
        extern void k(const float *input, float *output, int n);
        int main(void) {{
            float in[{n}] = {{{inputs_str}}};
            float out[{n}] = {{0}};
            k(in, out, {n});
            for (int i = 0; i < {original_n}; ++i) {{
                float ref = {libm_ref}(in[i]);
                float got = out[i];
                float abs_err = fabsf(got - ref);
                if (abs_err > {abs_tol}f) {{
                    printf("FAIL i=%d in=%g got=%g ref=%g abs=%g\n", i, in[i], got, ref, abs_err);
                    return 1;
                }}
            }}
            printf("OK\n");
            return 0;
        }}
        "#
    );

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
    ea_compiler::compile_with_options(&ea, &obj, OutputMode::ObjectFile, &opts)
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
    assert_eq!(
        stdout.trim(),
        "OK",
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Asserts that at least one of `expected_mnemonics` appears in the disassembly
/// of the object file produced by compiling `ea_source`.
///
/// Why a list: LLVM 18 lowers both `llvm.x86.avx2.permps` and `llvm.x86.avx2.permd`
/// to `vpermps` because they share an opcode (`66 0F 38 16`) and execute identically
/// on 32-bit lanes. Either mnemonic is a valid live-intrinsic signal. The assertion
/// only fails when NEITHER appears — which would mean the intrinsic compiled to a
/// `call` to an external symbol (LLVM-7+ deprecation case from PR #8 / PR #9).
#[allow(dead_code)]
pub fn assert_intrinsic_in_disassembly(ea_source: &str, expected_mnemonics: &[&str]) {
    let dir = TempDir::new().unwrap();
    let obj = dir.path().join("t.o");
    ea_compiler::compile(ea_source, &obj, ea_compiler::OutputMode::ObjectFile)
        .expect("compile to object file");
    let output = Command::new("objdump")
        .args(["-d", "-Mintel"])
        .arg(&obj)
        .output()
        .expect("run objdump");
    let asm = String::from_utf8_lossy(&output.stdout);
    assert!(
        expected_mnemonics.iter().any(|m| asm.contains(m)),
        "expected one of {expected_mnemonics:?} in disassembly. \
         If you see 'call' to an external symbol instead, the LLVM \
         intrinsic name has been deprecated and codegen must switch \
         to a pattern-matched fallback. Disassembly:\n{asm}"
    );
}
