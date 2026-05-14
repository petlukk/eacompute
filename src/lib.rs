pub mod ast;
pub mod bind_cmake;
pub mod bind_common;
pub mod bind_cpp;
pub mod bind_python;
pub mod bind_pytorch;
pub mod bind_rust;
pub mod desugar;
pub mod error;
pub mod header;
pub mod lexer;
pub mod metadata;
pub mod parser;
pub mod typeck;

#[cfg(feature = "llvm")]
pub mod bench;
#[cfg(feature = "llvm")]
pub mod codegen;
#[cfg(feature = "llvm")]
pub mod inspect;
#[cfg(feature = "llvm")]
pub mod target;

use ast::Stmt;
use lexer::{Lexer, Token};
use parser::Parser;
use typeck::TypeChecker;

pub use typeck::{DeprecationInfo, DeprecationWarning};

pub fn tokenize(source: &str) -> error::Result<Vec<Token>> {
    Lexer::new(source).tokenize()
}

pub fn parse(tokens: Vec<Token>) -> error::Result<Vec<Stmt>> {
    Parser::new(tokens).parse_program()
}

pub fn desugar(stmts: Vec<Stmt>) -> error::Result<Vec<Stmt>> {
    desugar::desugar_kernels(stmts)
}

pub fn check_types(stmts: &[Stmt]) -> error::Result<()> {
    check_types_with_warnings(stmts).map(|_| ())
}

/// Type-check `stmts` and return any deprecation warnings collected during
/// the pass. Empty vec if no deprecated intrinsics were used.
pub fn check_types_with_warnings(stmts: &[Stmt]) -> error::Result<Vec<DeprecationWarning>> {
    let mut tc = TypeChecker::new();
    tc.check_program(stmts)?;
    Ok(tc.warnings())
}

#[cfg(feature = "llvm")]
pub enum OutputMode {
    ObjectFile,
    Executable(String),
    SharedLib(String),
    LlvmIr,
    Asm,
}

#[cfg(feature = "llvm")]
#[derive(Clone, Debug)]
pub struct CompileOptions {
    pub opt_level: u8,
    pub target_cpu: Option<String>,
    /// Extra target features, e.g. "+avx512f" for AVX-512, "+fullfp16" for ARM FP16 compute.
    pub extra_features: String,
    /// Cross-compilation target triple, e.g. "aarch64-unknown-linux-gnu".
    pub target_triple: Option<String>,
}

#[cfg(feature = "llvm")]
impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            opt_level: 3,
            target_cpu: None, // native
            extra_features: String::new(),
            target_triple: None,
        }
    }
}

#[cfg(feature = "llvm")]
impl CompileOptions {
    pub fn is_arm(&self) -> bool {
        let triple = self.target_triple.as_deref().unwrap_or(Self::host_triple());
        triple.starts_with("aarch64") || triple.starts_with("arm")
    }

    fn host_triple() -> &'static str {
        if cfg!(target_arch = "aarch64") {
            "aarch64-unknown-linux-gnu"
        } else {
            "x86_64-unknown-linux-gnu"
        }
    }
}

#[cfg(feature = "llvm")]
static INIT_LLVM: std::sync::Once = std::sync::Once::new();

#[cfg(feature = "llvm")]
fn init_llvm() {
    INIT_LLVM.call_once(|| {
        use inkwell::targets::{InitializationConfig, Target};
        Target::initialize_native(&InitializationConfig::default())
            .expect("Failed to initialize LLVM native target");
        Target::initialize_aarch64(&InitializationConfig::default());
    });
}

#[cfg(feature = "llvm")]
fn validate_fp16_compatibility(opts: &CompileOptions) -> error::Result<()> {
    if !opts.is_arm() && opts.extra_features.contains("fullfp16") {
        return Err(error::CompileError::codegen_error(
            "--fp16 is incompatible with non-ARM target",
        ));
    }
    Ok(())
}

#[cfg(feature = "llvm")]
pub fn compile(source: &str, output_path: &std::path::Path, mode: OutputMode) -> error::Result<()> {
    compile_with_options(source, output_path, mode, &CompileOptions::default())
}

pub fn compile_with_options(
    source: &str,
    output_path: &std::path::Path,
    mode: OutputMode,
    opts: &CompileOptions,
) -> error::Result<()> {
    validate_fp16_compatibility(opts)?;
    init_llvm();

    let tokens = tokenize(source)?;
    let stmts = parse(tokens)?;
    let stmts = desugar::filter_cfg(stmts, opts.is_arm());
    let stmts = desugar(stmts)?;
    for warning in check_types_with_warnings(&stmts)? {
        eprintln!("{warning}");
    }

    let context = inkwell::context::Context::create();
    let mut cg = codegen::CodeGenerator::new(&context, "ea_module", opts);
    cg.compile_program(&stmts)?;

    match mode {
        OutputMode::ObjectFile => {
            target::write_object_file(cg.module(), output_path, opts)?;
        }
        OutputMode::Executable(ref exe_name) => {
            let tmp_dir = std::env::temp_dir().join("ea_build");
            std::fs::create_dir_all(&tmp_dir).map_err(|e| {
                error::CompileError::codegen_error(format!("failed to create temp dir: {e}"))
            })?;
            let obj_path = tmp_dir.join("temp.o");
            target::write_object_file(cg.module(), &obj_path, opts)?;

            let status = std::process::Command::new("cc")
                .arg(&obj_path)
                .arg("-o")
                .arg(exe_name)
                .arg("-lm")
                .status()
                .map_err(|e| {
                    error::CompileError::codegen_error(format!("failed to invoke linker: {e}"))
                })?;

            let _ = std::fs::remove_dir_all(&tmp_dir);

            if !status.success() {
                return Err(error::CompileError::codegen_error("linker failed"));
            }
        }
        OutputMode::SharedLib(ref lib_name) => {
            target::write_object_file(cg.module(), output_path, opts)?;

            // Pick the linker by *target*, not host. A windows triple
            // means we need a PE DLL even when we're on Linux.
            let target_is_windows = opts
                .target_triple
                .as_deref()
                .map(|t| t.contains("windows"))
                .unwrap_or(cfg!(target_os = "windows"));
            let host_is_windows = cfg!(target_os = "windows");

            if target_is_windows && host_is_windows {
                // Native Windows: lld-link.exe (ships with LLVM 18).
                let out_flag = format!("/OUT:{}", lib_name);
                let output = std::process::Command::new("lld-link.exe")
                    .arg("/DLL")
                    .arg("/NOLOGO")
                    .arg("/NODEFAULTLIB")
                    .arg("/NOENTRY")
                    .arg(&out_flag)
                    .arg(output_path)
                    .output()
                    .map_err(|e| {
                        error::CompileError::codegen_error(format!(
                            "failed to invoke lld-link: {e}"
                        ))
                    })?;
                let _ = std::fs::remove_file(output_path);
                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let detail = if !stderr.is_empty() { stderr } else { stdout };
                    return Err(error::CompileError::codegen_error(format!(
                        "lld-link failed:\n{}",
                        detail.trim()
                    )));
                }
            } else if target_is_windows {
                // Cross from non-Windows host: mingw-w64. Override with
                // WINDOWS_CC env if a different cross-gcc is needed.
                let cc = std::env::var("WINDOWS_CC")
                    .unwrap_or_else(|_| "x86_64-w64-mingw32-gcc".to_string());
                let status = std::process::Command::new(&cc)
                    // mingw provides DllMainCRTStartup, msvcrt math
                    // (fmaf/expf), and libgcc compiler-rt (__extendhfsf2).
                    // -static-libgcc avoids a runtime libgcc_s_seh-1.dll dep.
                    .arg("-shared")
                    .arg("-static-libgcc")
                    .arg(output_path)
                    .arg("-o")
                    .arg(lib_name)
                    .status()
                    .map_err(|e| {
                        error::CompileError::codegen_error(format!("failed to invoke {cc}: {e}"))
                    })?;
                let _ = std::fs::remove_file(output_path);
                if !status.success() {
                    return Err(error::CompileError::codegen_error(
                        "shared library linking failed (mingw-w64)",
                    ));
                }
            } else {
                let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
                let status = std::process::Command::new(&cc)
                    .arg("-shared")
                    .arg(output_path)
                    .arg("-o")
                    .arg(lib_name)
                    .arg("-lm")
                    .status()
                    .map_err(|e| {
                        error::CompileError::codegen_error(format!("failed to invoke linker: {e}"))
                    })?;
                let _ = std::fs::remove_file(output_path);
                if !status.success() {
                    return Err(error::CompileError::codegen_error(
                        "shared library linking failed",
                    ));
                }
            }
        }
        OutputMode::LlvmIr => {
            if opts.opt_level > 0 {
                let machine = target::create_target_machine(opts)?;
                target::optimize_module(cg.module(), &machine, opts.opt_level)?;
            }
            let ir = cg.print_ir();
            std::fs::write(output_path, ir).map_err(|e| {
                error::CompileError::codegen_error(format!("failed to write IR: {e}"))
            })?;
        }
        OutputMode::Asm => {
            target::write_asm_file(cg.module(), output_path, opts)?;
        }
    }

    Ok(())
}

#[cfg(feature = "llvm")]
pub fn compile_to_ir(source: &str) -> error::Result<String> {
    compile_to_ir_with_options(source, CompileOptions::default())
}

#[cfg(feature = "llvm")]
pub fn compile_to_ir_with_options(source: &str, opts: CompileOptions) -> error::Result<String> {
    validate_fp16_compatibility(&opts)?;
    init_llvm();

    let tokens = tokenize(source)?;
    let stmts = parse(tokens)?;
    let stmts = desugar::filter_cfg(stmts, opts.is_arm());
    let stmts = desugar(stmts)?;
    for warning in check_types_with_warnings(&stmts)? {
        eprintln!("{warning}");
    }

    let context = inkwell::context::Context::create();
    let mut cg = codegen::CodeGenerator::new(&context, "ea_module", &opts);
    cg.compile_program(&stmts)?;

    Ok(cg.print_ir())
}

#[cfg(feature = "llvm")]
pub fn inspect_source(
    source: &str,
    opts: &CompileOptions,
) -> error::Result<inspect::InspectReport> {
    validate_fp16_compatibility(opts)?;
    init_llvm();

    let tokens = tokenize(source)?;
    let stmts = parse(tokens)?;
    let stmts = desugar::filter_cfg(stmts, opts.is_arm());
    let stmts = desugar(stmts)?;
    for warning in check_types_with_warnings(&stmts)? {
        eprintln!("{warning}");
    }

    let context = inkwell::context::Context::create();
    let mut cg = codegen::CodeGenerator::new(&context, "ea_module", opts);
    cg.compile_program(&stmts)?;

    let machine = target::create_target_machine(opts)?;

    if opts.opt_level > 0 {
        target::optimize_module(cg.module(), &machine, opts.opt_level)?;
    }

    inspect::analyze_module(cg.module(), &machine)
}
