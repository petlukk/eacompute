# ea-compiler pip package & documentation

**Date**: 2026-03-17
**Status**: Approved

## Problem

Eä requires LLVM 18, Rust, and a cargo build to use. This limits adoption to experienced systems programmers. Python developers — the primary audience for eastat, eavec, and the binding generators — bounce at the install step.

## Solution

Two deliverables:

1. **`ea-compiler` pip package** — zero-install Eä for Python users
2. **User-friendly documentation** — layered docs from 60-second quickstart to full reference

## Package: ea-compiler

### Distribution model

Bundle the pre-built `ea` CLI binary inside a platform-specific wheel. The Python package wraps subprocess calls to the binary. No PyO3, no maturin, no LLVM on the user's machine.

Rationale: the `ea` compiler runs in <100ms for typical kernels. Subprocess overhead is negligible. The binary-building CI already exists in eacompute's release workflow.

### Repository

New repo: `ea-compiler-py` (alongside eastat, eavec, easobel). Keeps the eacompute repo pure Rust. The Python package has its own release cycle.

### Package name

- PyPI: `ea-compiler`
- Import: `import ea`
- Pattern follows `scikit-learn` → `import sklearn`, `Pillow` → `import PIL`

### Wheel contents

```
ea/
  __init__.py          # ea.load(), ea.compile(), version
  _compiler.py         # subprocess wrapper around ea binary
  _cache.py            # __eacache__/ management
  _bindings.py         # runtime ctypes binding generation from .ea.json
  bin/
    ea                 # pre-built binary (ea.exe on Windows)
  py.typed
```

### Dependencies

- `numpy>=1.21` (for array marshalling in bindings)
- Python >=3.9

### Core API

```python
import ea
import numpy as np

# Primary API: compile + cache + bind in one call
kernel = ea.load("scale.ea")
dst = kernel.scale(src, factor=2.0)

# With options
kernel = ea.load("scale.ea", target="native", opt_level=3, avx512=True)

# Lower-level
ea.compile("scale.ea")                    # compile to .so
ea.compile("scale.ea", emit_asm=True)     # inspect assembly
```

### ea.load() behavior

1. Determine CPU target name (default: `native`). Resolve to concrete CPU name via `ea --print-target` subcommand (new, must be added to eacompute — prints the LLVM CPU name that `native` resolves to, e.g., `znver4`). Fall back to a normalized `platform.machine()` if subcommand is unavailable (maps `x86_64`/`AMD64` → `x86-64`, `aarch64`/`ARM64` → `generic`). The fallback produces a coarser cache key but is functionally correct.
2. Build cache key: `{cpu}-{ea_version}` (e.g., `znver4-1.7.0`). The `ea` binary version is obtained from `ea --version`.
3. Check `__eacache__/{cache_key}/kernel.so` (or `.dll` on Windows) — if exists and newer than `kernel.ea`, use cached.
4. If stale or missing:
   a. Create cache directory `__eacache__/{cache_key}/`
   b. Copy `kernel.ea` into the cache directory (as a temp working copy, bare filename — not absolute path)
   c. Run `ea kernel.ea --lib` from within the cache directory (cwd = cache dir). This produces `kernel.so` and `kernel.ea.json` in the cache directory.
   d. Remove the temp `.ea` copy (in a `finally` block — cleaned up on both success and failure)
5. Read `.ea.json` metadata from cache directory. The `"library"` field is a bare filename (e.g., `kernel.so`) since compilation happens in the cache dir.
6. Build ctypes bindings dynamically, loading the `.so`/`.dll` from the cache directory by absolute path.
7. Return a module-like object with callable methods.

Note: compiling in the cache directory avoids issues with the `-o` flag not supporting output directories, and ensures `.ea.json`'s `"library"` field is a bare filename.

### Cache structure

```
__eacache__/
  znver4-1.7.0/             # CPU target + compiler version
    kernel.so               # compiled shared library (or .dll on Windows)
    kernel.ea.json          # metadata for binding generation
```

Cache invalidation: recompile when `.ea` file mtime > cached `.so` mtime, when CPU target changes, or when `ea` binary version changes (each gets a separate subdirectory).

Concurrent access: last writer wins. Two processes compiling the same kernel simultaneously may both write to the cache, but the result is deterministic (same source + same compiler = same output). No file locking needed.

The `__eacache__/` directory is created relative to the `.ea` source file's parent directory (same convention as `__pycache__/`). Users can delete it at any time. It regenerates automatically. `ea.clear_cache(path)` is provided as a convenience (clears cache for a specific `.ea` file, or all caches in a directory). Add `__eacache__/` to `.gitignore`.

`ea.compile()` does NOT use the cache — it always recompiles and writes output next to the source file (matching CLI behavior). Only `ea.load()` caches.

### Runtime binding generation

`_bindings.py` reads `.ea.json` at runtime and dynamically creates Python callables. No static `.py` files generated.

Each generated function:
1. Validates numpy array dtypes
2. Extracts ctypes pointers from numpy arrays
3. Auto-collapses length params (`n`/`len`/`length`/`count`/`size`/`num` after a pointer arg filled from array.size)
4. Auto-allocates output arrays (params annotated `out` with `cap`/`count`)
5. Calls the C function via ctypes.CDLL
6. Returns output arrays or scalars

Type mapping:

| Ea type | ctypes | Python arg |
|---------|--------|------------|
| `*f32` / `*restrict f32` | `POINTER(c_float)` | `np.ndarray` (float32) |
| `*mut f32` | `POINTER(c_float)` | `np.ndarray` or auto-allocated |
| `i32` | `c_int32` | `int` |
| `i64` | `c_int64` | `int` |
| `f32` | `c_float` | `float` |
| `f64` | `c_double` | `float` |
| `u8` | `c_uint8` | `int` |
| `u16` | `c_uint16` | `int` |
| `u32` | `c_uint32` | `int` |
| `u64` | `c_uint64` | `int` |
| `bool` | `c_bool` | `bool` |

Length collapsing and output allocation reuse the same rules as `bind_common.rs` in eacompute.

### Platform wheels

| Platform | Wheel tag | Binary in wheel |
|----------|-----------|-----------------|
| Linux x86_64 | `manylinux_2_17_x86_64` | `ea/bin/ea` (ELF) |
| Linux aarch64 | `manylinux_2_17_aarch64` | `ea/bin/ea` (ELF) |
| Windows x86_64 | `win_amd64` | `ea/bin/ea.exe` (PE) |

### CI/CD

**Publish workflow** (`.github/workflows/publish.yml`):
1. Trigger: `ea-compiler-v*` tag push or `workflow_dispatch`
2. Matrix: 3 platforms (linux x86_64, linux aarch64, windows x86_64)
3. Download pre-built `ea` binary from `petlukk/eacompute` GitHub Releases (specified by `EA_VERSION` env var)
4. Place binary in `ea/bin/`
5. `python -m build --wheel`
6. Retag wheel with platform-specific tag
7. Test in clean venv: `pip install dist/*.whl && python -c "import ea; print(ea.__version__)"`
8. Publish to PyPI via OIDC trusted publishing

This follows the exact pattern used by eastat and eavec.

### Version synchronization

The `ea-compiler` Python package version tracks the bundled `ea` binary version. Example: `ea-compiler 1.7.0` bundles `ea 1.7.0`. The `EA_VERSION` env var in the publish workflow pins this. `ea.__version__` returns the Python package version, `ea.compiler_version()` returns the bundled binary's version string.

If a user's `.ea` source uses features from a newer compiler version, the bundled `ea` binary will produce a compile error (the standard "unsupported feature" error from the compiler). The fix is `pip install --upgrade ea-compiler`.

### Error handling

- If `ea` binary not found in package: raise `RuntimeError` with install instructions
- If `.ea` file not found: raise `FileNotFoundError`
- If compilation fails: raise `ea.CompileError(message, stderr, exit_code)` — inherits from `RuntimeError`. `message` is a summary, `stderr` contains the full compiler output with caret-style error messages.
- If numpy array dtype doesn't match: raise `TypeError` with expected dtype

### Thread safety

The kernel object returned by `ea.load()` wraps a `ctypes.CDLL` and is safe for concurrent calls from multiple threads. The Eä kernels themselves are pure functions operating on caller-provided buffers with no global state.

### ea.compile() return value

`ea.compile(path)` returns a `pathlib.Path` to the produced `.so`/`.dll`. With `emit_asm=True` or `emit_llvm=True`, returns the path to the `.s` or `.ll` file respectively.

## Documentation

### Location and tooling

Lives in `eacompute/docs/`. Rendered with mdBook. Deployed to GitHub Pages via GitHub Actions on push to `docs/`.

### Structure

```
docs/
  book.toml
  src/
    SUMMARY.md

    # Layer 1: Hook (beginner, <5 minutes)
    getting-started/
      index.md            # "Make Python fast in 60 seconds"
      install.md          # pip install ea-compiler
      first-kernel.md     # copy-paste scale example, see it work

    # Layer 2: Guide (intermediate, learn the language)
    guide/
      why-ea.md           # what Ea is, when to use it, when not to
      language.md         # types, operators, control flow, mutability
      simd.md             # vectors, dot operators, loads, stores
      kernels.md          # kernel construct, tail strategies
      structs.md          # struct definitions and field access
      common-intrinsics.md # top ~10 intrinsics with examples (splat, load, store, reduce, fma, select)

    # Layer 3: Reference (experienced, look things up)
    reference/
      types.md            # complete type table with sizes
      intrinsics.md       # exhaustive: every intrinsic with signature, constraints, example
      cli.md              # ea compile/bind/inspect flags
      python-api.md       # ea.load() options, caching, target
      bindings.md         # output annotations, length collapsing rules
      arm.md              # NEON constraints, portable patterns

    # Layer 4: Cookbook (real-world patterns)
    cookbook/
      numpy-comparison.md # side-by-side NumPy vs Ea
      image-processing.md # sobel, convolution
      text-processing.md  # CSV scanning, tokenizer
      ml-preprocessing.md # normalize, dot product, cosine similarity
```

### Principles

- Every page has a runnable example that works with `pip install ea-compiler`
- Getting Started mentions zero Rust/LLVM/cargo — only `pip install`
- Guide teaches Ea's philosophy: explicit over implicit, no hidden costs
- Reference is exhaustive and searchable
- Cookbook links to real demo packages (eastat, eavec, easobel) as production examples
- "Building from source" is an appendix, not a prerequisite

### Deploy

GitHub Actions workflow in eacompute repo:
1. Trigger: push to `docs/**` on main
2. Install mdBook
3. `mdbook build docs/`
4. Deploy to GitHub Pages

## Prerequisites in eacompute

The following changes are needed in the eacompute compiler repo before the Python package can be built:

1. **`ea --print-target` subcommand** — prints the LLVM CPU name that `native` resolves to (e.g., `znver4`, `skylake`, `cortex-a76`). Needed for cache key generation. Small addition to `main.rs`.
2. **`ea --version` flag** — already exists (`src/main.rs:21-23`). Prints compiler version from `Cargo.toml`. Used for cache key generation.

Both are trivial additions (<20 lines each).

## Out of scope

- PyO3/maturin FFI bindings (future optimization)
- Autoresearch integration in the package
- Online playground / web editor
- Homebrew / AUR / other package managers
- macOS support (no current CI for it)

## User journey

```
pip install ea-compiler          # 10 seconds
                |
                v
  Read first-kernel.md           # 60 seconds, copy-paste example
                |
                v
  import ea                      # write kernel.ea, call ea.load()
  kernel = ea.load("my.ea")      # compiles, caches, returns callable
  result = kernel.my_func(data)  # numpy in, numpy out
                |
                v
  Explore guide, reference,      # go deeper as needed
  cookbook
```
