# CLI Reference

The `ea` binary provides three commands: compile (default), bind, and inspect.

## Compile

```bash
ea <file.ea> [flags]
```

Compile an Ea source file to a native object file (`.o`) by default.

### Flags

| Flag | Effect |
|------|--------|
| `-o <name>` | Link the object file into an executable via `cc` |
| `--lib` | Produce a shared library (`.so`/`.dll`) and metadata (`.ea.json`) |
| `--opt-level=N` | Optimization level 0--3 (default: 3) |
| `--avx512` | Enable AVX-512 vector types and intrinsics. Errors on ARM targets |
| `--dotprod` | Enable ARMv8.2-A dot product extension (`vdot_i32`). ARM targets only |
| `--target=CPU` | LLVM CPU name, e.g. `skylake`, `znver3`, `native` (default: `native`) |
| `--target-triple=T` | Cross-compile to a different architecture, e.g. `aarch64-unknown-linux-gnu` |
| `--emit-llvm` | Write LLVM IR to a `.ll` file and print it to stdout |
| `--emit-asm` | Write assembly to a `.s` file |
| `--header` | Generate a C header (`.h`) for the exported functions |
| `--emit-ast` | Print the parsed AST. Does not require LLVM |
| `--emit-tokens` | Print the token stream. Does not require LLVM |
| `--help` / `-h` | Print usage information |
| `--version` / `-V` | Print compiler version |

### Examples

```bash
# Compile to object file
ea kernel.ea

# Compile and link to executable
ea kernel.ea -o kernel

# Build shared library for Python/Rust/C++ consumption
ea kernel.ea --lib

# Cross-compile for ARM
ea kernel.ea --lib --target-triple=aarch64-unknown-linux-gnu

# Emit LLVM IR for debugging
ea kernel.ea --emit-llvm

# Compile with AVX-512 support
ea kernel.ea --lib --avx512

# Cross-compile for ARM with dot product extension
ea kernel.ea --lib --target-triple=aarch64-unknown-linux-gnu --dotprod

# Generate C header
ea kernel.ea --header
```

Compiler status output goes to stderr, so `--emit-llvm` stdout is clean for piping.

## Bind

```bash
ea bind <file.ea> --python [--rust] [--cpp] [--pytorch] [--cmake]
```

Generate language bindings from a compiled kernel. At least one language flag is required.

Requires the `.ea.json` metadata file produced by `ea <file.ea> --lib`. Run the `--lib` compile first.

### Language Flags

| Flag | Output | Description |
|------|--------|-------------|
| `--python` | `<name>.py` | Python wrapper using `ctypes` |
| `--rust` | `<name>.rs` | Rust FFI bindings |
| `--cpp` | `<name>.hpp` | C++ header with inline wrappers |
| `--pytorch` | `<name>_torch.py` | PyTorch custom op wrapper |
| `--cmake` | `CMakeLists.txt` | CMake build file for C++ integration |

### Example

```bash
# Full workflow: compile, then generate Python bindings
ea kernel.ea --lib
ea bind kernel.ea --python
```

## Inspect

```bash
ea inspect <file.ea> [target flags]
```

Post-optimization analysis of the compiled kernel. Shows instruction mix, loop structure, vector width usage, and register pressure. Accepts the same target flags as compile (`--target`, `--target-triple`, `--avx512`, `--dotprod`).

### Example

```bash
ea inspect kernel.ea
ea inspect kernel.ea --avx512
ea inspect kernel.ea --target-triple=aarch64-unknown-linux-gnu
```

## Print Target

```bash
ea --print-target
```

Print the resolved native CPU name for the current machine. Useful for understanding which CPU features the compiler will target by default.
