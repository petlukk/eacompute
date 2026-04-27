# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Eä is a SIMD kernel compiler. Write compute kernels in Eä's explicit syntax, compile to native code (.o/.so/.dll), and generate language bindings (Python, Rust, C++, PyTorch, CMake). No runtime, no GC, no standard library. Just kernels. All exported functions use C ABI.

Written in Rust (~11,100 lines), targeting LLVM 18. The binary is `ea`, the library is `ea_compiler`.

## Hard Rules

These are inviolable. Every change must respect them.

1. **No file exceeds 500 lines.** Split before you hit the limit.
2. **Every feature proven by end-to-end test.** If it's not tested, it doesn't exist.
3. **No fake functions. No stubs.** No `// TODO`, `// HACK`, `// for now`, `// hardcoded`, `// placeholder`, `// temporary`. If it does not compile and pass tests, it is not code.
4. **No premature features.** Don't build what isn't needed yet.
5. **Delete, don't comment.** Dead code gets removed, not commented out.
6. **C interop is the product.** Everything serves the goal of producing callable object files.

## Design Philosophy

- **Explicit over implicit.** No auto-masking magic, no silent scalar fallbacks, no hidden performance cliffs. If the programmer writes SIMD-looking code, it must produce SIMD instructions or fail at compile time.
- **No fake functions.** If hardware doesn't support an operation (e.g., scatter without AVX-512), the compiler errors instead of silently emitting slow scalar code. The programmer sees the cost.
- **Concrete, not generic.** No generics, no traits, no polymorphism. Write separate kernels for each type. Kernel code is monomorphic.
- **One file, one compilation unit.** No modules, no imports. Compose at the C level.
- **All memory is caller-provided.** No allocator, no GC. Pointers come from the host language.
- **Clippy clean, fmt clean.** Run `cargo fmt && cargo clippy --all-targets --all-features -- -D warnings` before committing.

## Build Commands

```bash
# Prerequisites: LLVM 18 dev libraries
sudo apt install llvm-18-dev clang-18 libpolly-18-dev libzstd-dev

# Build
cargo build

# Lint (must pass before committing)
cargo fmt && cargo clippy --all-targets --all-features -- -D warnings

# Run all tests (~420 tests, most are end-to-end)
cargo test --tests --features=llvm

# Run a single test
cargo test test_name

# Run tests in a specific file
cargo test --test phase3

# Build release binary
cargo build --release

# Run the compiler
./target/debug/ea kernel.ea          # compile to .o
./target/debug/ea kernel.ea --lib    # compile to .so + .ea.json
./target/debug/ea bind kernel.ea --python  # generate bindings (needs .ea.json)
```

The `llvm` feature is on by default. All LLVM-dependent code is behind `#[cfg(feature = "llvm")]`. Without it, tokenizer, parser, desugar, type checker, and binding generators still compile.

On Windows, set `LLVM_SYS_181_PREFIX=C:\llvm` to point at the LLVM 18 installation.

## Compiler Pipeline

```
Source (.ea) → Lexer (logos) → Parser → Desugar → Type Check → Codegen (LLVM 18) → .o/.so
                                                                                  → .ea.json → ea bind → .py/.rs/.hpp/_torch.py/CMakeLists.txt
```

Key design: the `kernel` construct is **syntactic sugar** — `desugar.rs` transforms it into a plain function with a while-loop (+ optional tail loop) before type checking or codegen ever see it. All downstream passes only work with functions. `Stmt::Kernel` only exists in the AST before `desugar()` runs. The desugarer auto-injects the `n: i32` parameter (from the kernel's range bound) into the function signature.

## Source Layout (`src/`)

| Module | Role |
|---|---|
| `main.rs` | CLI entry point (compile/bind subcommands) |
| `lib.rs` | Library API: `tokenize()` → `parse()` → `desugar()` → `check_types()` → `compile()` |
| `lexer/` | Token definitions (`logos`-generated), `Token`, `Span`, `Position` |
| `parser/` | Recursive descent parser → `Vec<Stmt>`. Split: `statements.rs`, `expressions.rs`, `loops.rs` |
| `ast/` | AST types: `Stmt`, `Expr`, `TypeAnnotation`, `BinaryOp`, `TailStrategy` |
| `desugar.rs` | Kernel → function transformation |
| `typeck/` | Type checker. Split: `types.rs`, `check.rs`, `expr_check.rs`, `intrinsics.rs`, `intrinsics_conv.rs`, `intrinsics_simd.rs`, `intrinsics_memory.rs`, `const_eval.rs` |
| `codegen/` | LLVM IR generation. Split: `statements.rs`, `expressions.rs`, `comparisons.rs`, `builtins.rs`, `simd.rs`, `simd_arithmetic.rs`, `simd_math.rs`, `simd_memory.rs`, `simd_masked.rs`, `structs.rs`, `simd_conv.rs` (conversions: to_f32 / to_f64 / to_i32 / to_f16), `simd_fp16.rs` (native f16 codegen: splat, load, store, fma, reductions) |
| `inspect.rs` | `ea inspect` — post-optimization instruction mix, loops, vector width, register analysis |
| `target.rs` | LLVM target machine creation, optimization passes, object/assembly file writing |
| `error.rs` | Error types with source-context caret formatting |
| `metadata.rs` | JSON metadata generation (.ea.json) for binding generators |
| `header.rs` | C header (.h) generation |
| `bind_common.rs` | Shared binding types, hand-rolled JSON parser (no serde), length-collapsing heuristic |
| `bind_python.rs`, `bind_rust.rs`, `bind_cpp.rs`, `bind_pytorch.rs`, `bind_cmake.rs` | Language-specific binding generators |

## Test Structure (`tests/`)

Tests are end-to-end: compile Eä source → link with C harness via `cc` → execute binary → compare stdout. The shared test harness is in `tests/common/mod.rs`:

| Helper | What it does |
|---|---|
| `assert_output(source, expected)` | Compile + run Eä, assert stdout matches |
| `assert_output_lines(source, lines)` | Same but compares line-by-line against `&[&str]` |
| `assert_c_interop(ea, c, expected)` | Compile Eä + C harness, link, assert stdout |
| `assert_shared_lib_interop(ea, c, expected)` | Full `.so` path with `-Wl,-rpath` |
| `compile_to_ir(source)` | Returns LLVM IR string for IR-level assertions |
| `compile_and_run(source)` | Returns `TestOutput { stdout, stderr, exit_code }` |

Test files are named by development phase (`phase3.rs` through `phase14_arm.rs`) plus feature-specific files (`kernel_tests.rs`, `tail_tests.rs`, `const_tests.rs`, `output_tests.rs`, `bind.rs`, `bind_generators.rs`, `restrict.rs`, `static_assert_tests.rs`, `multi_kernel_tests.rs`, `inspect_tests.rs`).

**New test file boilerplate** (for LLVM-dependent tests):
```rust
#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    #[test]
    fn test_something() {
        assert_output("export func main() { println(42); }", "42");
    }
}
```

Tests that don't need LLVM (e.g., `bind.rs`, `static_assert_tests.rs`) omit the `#[cfg(feature = "llvm")]` wrappers. x86-64-specific tests (AVX2 intrinsics, `f32x8`) use `#[cfg(target_arch = "x86_64")]` on individual test functions.

**Error testing pattern** — type error tests call `check_types` directly, skipping `desugar()`:
```rust
let tokens = ea_compiler::tokenize(source).unwrap();
let stmts = ea_compiler::parse(tokens).unwrap();
ea_compiler::check_types(&stmts).unwrap_err()  // no desugar()
```

## CLI Reference

**Compile**: `ea <file.ea> [flags]`

| Flag | Effect |
|---|---|
| `-o <name>` | Link to executable via `cc` |
| `--lib` | Produce `.so`/`.dll` + `.ea.json` metadata |
| `--opt-level=N` | 0–3, default 3 |
| `--avx512` | Enables AVX-512 features; errors on ARM targets |
| `--fp16` | Enables ARM FP16 (FEAT_FP16) compute on f16x{4,8} types; errors on non-ARM targets |
| `--dotprod` | Enables ARM dot-product instructions (sdot/udot/vdot); ARM-only |
| `--i8mm` | Enables ARM Int8 Matrix Multiply instructions (smmla/ummla/usmmla); ARM-only |
| `--target=CPU` | LLVM CPU name (e.g., `skylake`, `native`) |
| `--target-triple=T` | Cross-compile (e.g., `aarch64-unknown-linux-gnu`) |
| `--emit-llvm` | Write `.ll` file, print IR to stdout |
| `--emit-asm` | Write `.s` file |
| `--header` | Write `.h` C header |
| `--emit-ast` / `--emit-tokens` | Debug output (no LLVM needed) |

**Bind**: `ea bind <file.ea> --python --rust --cpp --pytorch --cmake` (at least one required). Reads `<file.ea>.json` — compile with `--lib` first.

**Inspect**: `ea inspect <file.ea>` — post-optimization instruction analysis. Same target flags as compile.

Compiler status output goes to stderr so `--emit-llvm` stdout is uncontaminated.

## Key Conventions

- **Dot operators for SIMD**: element-wise vector ops use `.+`, `.-`, `.*`, `./`, `.&`, `.|`, `.^`, `.<<`, `.>>`, `.<`, `.>`, `.==`, etc.
- **No serde**: JSON parsing in `bind_common.rs` is hand-written. Keep it that way.
- **Length collapsing**: binding generators auto-detect length params (`n`/`len`/`length`/`count`/`size`/`num` after a pointer arg) and fill them from host-language array sizes.
- **Output annotations**: `out name: *mut T [cap: expr, count: path]` marks parameters that bindings auto-allocate and return.
- **C ABI everywhere**: every `export func` uses C calling convention. The entire FFI story depends on this.
- **Targets**: x86-64 (AVX2 default, AVX-512 via `--avx512`), AArch64/NEON (via `--target=aarch64`).
- **`println` is the only output primitive**: accepts integers, floats, bools, string literals. No format strings. Lowers to `printf`.

## Files Near the 500-Line Limit

These files need splitting before adding code to them:

| File | Lines |
|---|---|
| `src/codegen/statements.rs` | **503** (over limit — needs split) |
| `src/parser/expressions.rs` | 495 |
| `src/parser/statements.rs` | 463 |

## CI

Three-platform CI (`.github/workflows/ci.yml`): Linux x86_64, Linux ARM64 (native runner, not QEMU), Windows. Release workflow (`.github/workflows/release.yml`) builds binaries on `v*` tag push. CI test command: `cargo test --tests --features=llvm`.

## eabrain
Use `eabrain search <query>` to find Ea kernels across all projects.
Use `eabrain ref <name>` to look up Ea language reference.
Use `eabrain remember <note>` to save context between sessions.
