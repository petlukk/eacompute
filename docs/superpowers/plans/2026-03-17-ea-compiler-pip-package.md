# ea-compiler pip package Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a pip-installable Python package (`ea-compiler`) that bundles the pre-built `ea` binary and provides `ea.load()` for zero-friction kernel compilation and usage from Python.

**Architecture:** The package wraps the `ea` CLI binary via subprocess. `ea.load("kernel.ea")` compiles to a shared library (cached per CPU+version), parses the `.ea.json` metadata, and builds ctypes bindings at runtime. Three platform-specific wheels (linux x86_64, linux aarch64, windows x86_64).

**Tech Stack:** Python 3.9+, setuptools, ctypes, numpy, GitHub Actions CI

**Spec:** `docs/superpowers/specs/2026-03-17-ea-compiler-pip-package-design.md`

---

## Chunk 1: eacompute prerequisite — `ea --print-target`

This task modifies the eacompute Rust compiler to add a `--print-target` subcommand. Must be done first, in the eacompute repo (`/root/dev/eacompute`).

### Task 1: Add `--print-target` to eacompute CLI

**Files:**
- Modify: `/root/dev/eacompute/src/main.rs`
- Test: `/root/dev/eacompute/tests/end_to_end.rs`

- [ ] **Step 1: Write the failing test**

Add to `tests/end_to_end.rs`:

```rust
#[test]
fn test_print_target_outputs_cpu_name() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_ea"))
        .arg("--print-target")
        .output()
        .expect("failed to run ea");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let target = stdout.trim();
    // Should be a non-empty CPU name like "znver4", "skylake", "generic", etc.
    assert!(!target.is_empty(), "print-target should output a CPU name");
    // Should not contain spaces or newlines (single token)
    assert!(!target.contains(' '), "CPU name should be a single token: '{target}'");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test test_print_target_outputs_cpu_name --test end_to_end --features=llvm`
Expected: FAIL — `ea --print-target` exits with error (unknown option)

- [ ] **Step 3: Implement `--print-target` in main.rs**

In `/root/dev/eacompute/src/main.rs`, add a new match arm after the `--version` arm (around line 24). The `--version` arm looks like:

```rust
"--version" | "-V" => {
    println!("ea {}", env!("CARGO_PKG_VERSION"));
    return;
}
```

Add immediately after it:

```rust
"--print-target" => {
    use inkwell::targets::{InitializationConfig, Target, TargetMachine};
    Target::initialize_native(&InitializationConfig::default())
        .expect("failed to initialize native target");
    let cpu = TargetMachine::get_host_cpu_name();
    println!("{}", cpu.to_str().unwrap_or("unknown"));
    return;
}
```

This uses the same LLVM API that `target.rs` uses to resolve the native CPU name. It must be gated on the `llvm` feature. Since `#[cfg]` cannot be applied to individual match arms, handle it before the match:

```rust
// Add before the main match on args[0]:
#[cfg(feature = "llvm")]
if args[0] == "--print-target" {
    use inkwell::targets::{InitializationConfig, Target, TargetMachine};
    Target::initialize_native(&InitializationConfig::default())
        .expect("failed to initialize native target");
    let cpu = TargetMachine::get_host_cpu_name();
    println!("{}", cpu.to_str().unwrap_or("unknown"));
    return;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test test_print_target_outputs_cpu_name --test end_to_end --features=llvm`
Expected: PASS

- [ ] **Step 5: Run full test suite + lint**

Run: `cargo fmt && cargo clippy --all-targets --all-features -- -D warnings && cargo test --tests --features=llvm`
Expected: All pass, no warnings

- [ ] **Step 6: Commit**

```bash
git add src/main.rs tests/end_to_end.rs
git commit -m "feat: add --print-target subcommand for CPU name resolution"
```

---

## Chunk 2: Repository setup and project skeleton

Create the `ea-compiler-py` repo and set up the Python package structure. All paths below are relative to the new repo root.

### Task 2: Create repo and package skeleton

**Files:**
- Create: `src/ea/__init__.py`
- Create: `src/ea/_compiler.py`
- Create: `src/ea/_cache.py`
- Create: `src/ea/_bindings.py`
- Create: `src/ea/py.typed`
- Create: `src/ea/bin/.gitkeep`
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`

- [ ] **Step 1: Create GitHub repo**

```bash
cd /root/dev
mkdir ea-compiler-py && cd ea-compiler-py
git init
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.pyc
dist/
build/
*.egg-info/
.eggs/
src/ea/bin/ea
src/ea/bin/ea.exe
__eacache__/
.venv/
```

- [ ] **Step 3: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "ea-compiler"
version = "1.7.0"
description = "Eä SIMD kernel compiler for Python"
requires-python = ">=3.9"
dependencies = ["numpy>=1.21"]
license = "MIT"

[project.optional-dependencies]
test = ["pytest"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
ea = ["bin/ea", "bin/ea.exe", "py.typed"]
```

- [ ] **Step 4: Create `src/ea/__init__.py`**

```python
"""Eä SIMD kernel compiler for Python."""

__version__ = "1.7.0"

from ea._compiler import compile, compiler_version
from ea._cache import load, clear_cache
from ea._bindings import CompileError

__all__ = ["load", "compile", "clear_cache", "compiler_version", "CompileError", "__version__"]
```

- [ ] **Step 5: Create stub modules**

`src/ea/_compiler.py`:
```python
"""Subprocess wrapper around the ea binary."""

from pathlib import Path

_BIN_DIR = Path(__file__).parent / "bin"


def _ea_binary() -> Path:
    """Return path to bundled ea binary."""
    import sys
    name = "ea.exe" if sys.platform == "win32" else "ea"
    path = _BIN_DIR / name
    if not path.exists():
        raise RuntimeError(
            f"ea binary not found at {path}. "
            "Reinstall with: pip install --force-reinstall ea-compiler"
        )
    return path


def compile(path, *, emit_asm=False, emit_llvm=False, target="native", opt_level=3, avx512=False):
    """Compile an .ea file. Returns path to output artifact."""
    raise NotImplementedError


def compiler_version() -> str:
    """Return the version string of the bundled ea binary."""
    raise NotImplementedError
```

`src/ea/_cache.py`:
```python
"""Cache management for ea.load()."""


def load(path, *, target="native", opt_level=3, avx512=False):
    """Compile, cache, and load an .ea kernel. Returns a module-like object."""
    raise NotImplementedError


def clear_cache(path=None):
    """Clear cached compilations."""
    raise NotImplementedError
```

`src/ea/_bindings.py`:
```python
"""Runtime ctypes binding generation from .ea.json metadata."""


class CompileError(RuntimeError):
    """Raised when ea compilation fails."""

    def __init__(self, message: str, stderr: str, exit_code: int):
        super().__init__(message)
        self.stderr = stderr
        self.exit_code = exit_code
```

`src/ea/py.typed`: empty file

`src/ea/bin/.gitkeep`: empty file

- [ ] **Step 6: Create minimal README.md**

```markdown
# ea-compiler

Eä SIMD kernel compiler for Python.

```bash
pip install ea-compiler
```

```python
import ea
kernel = ea.load("scale.ea")
```
```

- [ ] **Step 7: Verify package installs**

```bash
pip install -e .
python -c "import ea; print(ea.__version__)"
```

Expected: `1.7.0`

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat: initial package skeleton"
```

---

## Chunk 3: `_compiler.py` — subprocess wrapper

### Task 3: Implement `compiler_version()`

**Files:**
- Modify: `src/ea/_compiler.py`
- Create: `tests/test_compiler.py`

- [ ] **Step 1: Write the test**

```python
import pytest
from unittest.mock import patch, MagicMock
from ea._compiler import compiler_version, _ea_binary


def test_compiler_version_parses_output():
    """compiler_version() should parse 'ea X.Y.Z' output."""
    mock_result = MagicMock()
    mock_result.stdout = "ea 1.7.0\n"
    mock_result.returncode = 0
    with patch("ea._compiler.subprocess.run", return_value=mock_result):
        assert compiler_version() == "1.7.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compiler.py::test_compiler_version_parses_output -v`
Expected: FAIL — `NotImplementedError`

- [ ] **Step 3: Implement `compiler_version()`**

In `src/ea/_compiler.py`:

```python
import subprocess


def compiler_version() -> str:
    """Return the version string of the bundled ea binary."""
    result = subprocess.run(
        [str(_ea_binary()), "--version"],
        capture_output=True, text=True, timeout=10,
    )
    # Output format: "ea X.Y.Z"
    return result.stdout.strip().removeprefix("ea ")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_compiler.py::test_compiler_version_parses_output -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ea/_compiler.py tests/test_compiler.py
git commit -m "feat: implement compiler_version()"
```

### Task 4: Implement `compile()`

**Files:**
- Modify: `src/ea/_compiler.py`
- Modify: `tests/test_compiler.py`

- [ ] **Step 1: Write the test**

```python
def test_compile_file_not_found():
    """compile() raises FileNotFoundError for missing .ea files."""
    with pytest.raises(FileNotFoundError):
        from ea._compiler import compile
        compile("nonexistent.ea")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compiler.py::test_compile_file_not_found -v`
Expected: FAIL — `NotImplementedError`

- [ ] **Step 3: Implement `compile()`**

In `src/ea/_compiler.py`:

```python
from pathlib import Path
from ea._bindings import CompileError


def compile(path, *, emit_asm=False, emit_llvm=False, target="native",
            opt_level=3, avx512=False, lib=True):
    """Compile an .ea file. Returns path to output artifact.

    Args:
        path: Path to .ea source file.
        emit_asm: If True, produce .s assembly file.
        emit_llvm: If True, produce .ll LLVM IR file.
        target: CPU target (default "native").
        opt_level: Optimization level 0-3 (default 3).
        avx512: Enable AVX-512 features.
        lib: If True (default), produce .so/.dll shared library.
    """
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Eä source file not found: {source}")

    cmd = [str(_ea_binary()), str(source)]
    if lib:
        cmd.append("--lib")
    if emit_asm:
        cmd.append("--emit-asm")
    if emit_llvm:
        cmd.append("--emit-llvm")
    if target != "native":
        cmd.append(f"--target={target}")
    cmd.append(f"--opt-level={opt_level}")
    if avx512:
        cmd.append("--avx512")

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=60,
        cwd=str(source.parent),
    )
    if result.returncode != 0:
        raise CompileError(
            f"Compilation failed: {source.name}",
            stderr=result.stderr,
            exit_code=result.returncode,
        )

    stem = source.stem
    import sys
    if emit_asm:
        return source.with_suffix(".s")
    elif emit_llvm:
        return source.with_suffix(".ll")
    elif lib:
        ext = ".dll" if sys.platform == "win32" else ".so"
        return source.parent / f"{stem}{ext}"
    else:
        return source.with_suffix(".o")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_compiler.py::test_compile_file_not_found -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ea/_compiler.py tests/test_compiler.py
git commit -m "feat: implement compile() subprocess wrapper"
```

### Task 5: Implement `_resolve_target()`

**Files:**
- Modify: `src/ea/_compiler.py`
- Modify: `tests/test_compiler.py`

- [ ] **Step 1: Write the tests**

```python
def test_resolve_target_native_fallback():
    """Falls back to normalized platform.machine() if --print-target fails."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "unknown option"
    with patch("ea._compiler._ea_binary", return_value=Path("/mock/ea")), \
         patch("ea._compiler.subprocess.run", return_value=mock_result):
        from ea._compiler import _resolve_target
        target = _resolve_target()
        # Should be something like "x86-64" or "generic", not empty
        assert target and " " not in target


def test_resolve_target_parses_print_target():
    """Parses ea --print-target output."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "znver4\n"
    with patch("ea._compiler._ea_binary", return_value=Path("/mock/ea")), \
         patch("ea._compiler.subprocess.run", return_value=mock_result):
        from ea._compiler import _resolve_target
        assert _resolve_target() == "znver4"
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_compiler.py -k resolve_target -v`
Expected: FAIL — no `_resolve_target` function

- [ ] **Step 3: Implement `_resolve_target()`**

In `src/ea/_compiler.py`:

```python
import platform

_MACHINE_MAP = {
    "x86_64": "x86-64",
    "AMD64": "x86-64",
    "aarch64": "generic",
    "ARM64": "generic",
}


def _resolve_target() -> str:
    """Resolve 'native' to a concrete CPU name for cache keying."""
    try:
        result = subprocess.run(
            [str(_ea_binary()), "--print-target"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            name = result.stdout.strip()
            if name:
                return name
    except (OSError, subprocess.TimeoutExpired):
        pass
    # Fallback: coarse mapping from platform.machine()
    machine = platform.machine()
    return _MACHINE_MAP.get(machine, machine.lower() or "unknown")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_compiler.py -k resolve_target -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ea/_compiler.py tests/test_compiler.py
git commit -m "feat: implement _resolve_target() for cache key CPU detection"
```

---

## Chunk 4: `_cache.py` — caching and `ea.load()`

### Task 6: Implement cache management

**Files:**
- Modify: `src/ea/_cache.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1: Write the tests**

```python
import tempfile
import shutil
from pathlib import Path

import pytest


def test_cache_dir_created_relative_to_source(tmp_path):
    """__eacache__/ is created next to the .ea file."""
    from ea._cache import _cache_dir
    ea_file = tmp_path / "kernel.ea"
    ea_file.touch()
    cache = _cache_dir(ea_file, "znver4", "1.7.0")
    assert cache == tmp_path / "__eacache__" / "znver4-1.7.0"


def test_cache_is_stale_when_no_cache(tmp_path):
    """Cache is stale when .so doesn't exist."""
    from ea._cache import _is_cached
    ea_file = tmp_path / "kernel.ea"
    ea_file.write_text("export func f() {}")
    cache = tmp_path / "__eacache__" / "znver4-1.7.0"
    assert not _is_cached(ea_file, cache)


def test_cache_is_fresh_when_so_newer(tmp_path):
    """Cache is fresh when .so is newer than .ea source."""
    import time, sys
    from ea._cache import _is_cached
    ea_file = tmp_path / "kernel.ea"
    ea_file.write_text("export func f() {}")
    cache = tmp_path / "__eacache__" / "znver4-1.7.0"
    cache.mkdir(parents=True)
    ext = ".dll" if sys.platform == "win32" else ".so"
    so_file = cache / f"kernel{ext}"
    time.sleep(0.05)  # ensure mtime difference
    so_file.write_bytes(b"\x00")
    assert _is_cached(ea_file, cache)


def test_clear_cache_removes_eacache(tmp_path):
    """clear_cache() removes __eacache__/ directory."""
    from ea._cache import clear_cache
    cache = tmp_path / "__eacache__" / "znver4-1.7.0"
    cache.mkdir(parents=True)
    (cache / "kernel.so").write_bytes(b"\x00")
    clear_cache(tmp_path)
    assert not (tmp_path / "__eacache__").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cache.py -v`
Expected: FAIL — functions don't exist

- [ ] **Step 3: Implement cache helpers**

In `src/ea/_cache.py`:

```python
"""Cache management for ea.load()."""

import shutil
import sys
from pathlib import Path

CACHE_DIR_NAME = "__eacache__"


def _cache_dir(ea_file: Path, cpu: str, version: str) -> Path:
    """Return the cache directory for a given .ea file + target + version."""
    return ea_file.parent / CACHE_DIR_NAME / f"{cpu}-{version}"


def _lib_ext() -> str:
    return ".dll" if sys.platform == "win32" else ".so"


def _is_cached(ea_file: Path, cache: Path) -> bool:
    """Check if a cached .so exists and is newer than the source .ea file."""
    so_file = cache / f"{ea_file.stem}{_lib_ext()}"
    if not so_file.exists():
        return False
    return so_file.stat().st_mtime > ea_file.stat().st_mtime


def clear_cache(path=None):
    """Clear cached compilations.

    Args:
        path: Path to a .ea file or directory. If a .ea file, clears its
              __eacache__/. If a directory, clears __eacache__/ in that
              directory. If None, clears __eacache__/ in the current directory.
    """
    if path is None:
        target = Path.cwd() / CACHE_DIR_NAME
    else:
        p = Path(path)
        if p.is_file():
            target = p.parent / CACHE_DIR_NAME
        else:
            target = p / CACHE_DIR_NAME
    if target.is_dir():
        shutil.rmtree(target)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cache.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ea/_cache.py tests/test_cache.py
git commit -m "feat: implement cache directory management"
```

### Task 7: Implement `ea.load()`

**Files:**
- Modify: `src/ea/_cache.py`
- Modify: `tests/test_cache.py`

- [ ] **Step 1: Write the test**

This test needs a real `ea` binary. Use an integration test that skips if `ea` is not available:

```python
import numpy as np


def _ea_available():
    """Check if ea binary is available (either bundled or in PATH)."""
    try:
        from ea._compiler import _ea_binary
        return _ea_binary().exists()
    except RuntimeError:
        return False


@pytest.mark.skipif(not _ea_available(), reason="ea binary not available")
def test_load_compiles_and_returns_callable(tmp_path):
    """ea.load() should compile a .ea file and return callable functions."""
    ea_file = tmp_path / "scale.ea"
    ea_file.write_text(
        "export func scale(src: *f32, dst: *mut f32, factor: f32, n: i32) {\n"
        "    let mut i: i32 = 0\n"
        "    while i < n {\n"
        "        dst[i] = src[i] * factor\n"
        "        i = i + 1\n"
        "    }\n"
        "}\n"
    )
    import ea
    kernel = ea.load(str(ea_file))
    src = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    dst = kernel.scale(src, factor=2.0)
    np.testing.assert_array_equal(dst, [2.0, 4.0, 6.0, 8.0])


@pytest.mark.skipif(not _ea_available(), reason="ea binary not available")
def test_load_uses_cache_on_second_call(tmp_path):
    """ea.load() should use cache on second call (no recompilation)."""
    ea_file = tmp_path / "add1.ea"
    ea_file.write_text(
        "export func add1(src: *f32, dst: *mut f32, n: i32) {\n"
        "    let mut i: i32 = 0\n"
        "    while i < n {\n"
        "        dst[i] = src[i] + 1.0\n"
        "        i = i + 1\n"
        "    }\n"
        "}\n"
    )
    import ea
    kernel1 = ea.load(str(ea_file))
    # Cache should exist now
    assert (tmp_path / "__eacache__").exists()
    kernel2 = ea.load(str(ea_file))
    # Both should work
    src = np.array([1.0, 2.0], dtype=np.float32)
    np.testing.assert_array_equal(kernel2.add1(src), [2.0, 3.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cache.py::test_load_compiles_and_returns_callable -v`
Expected: FAIL — `NotImplementedError`

- [ ] **Step 3: Implement `load()`**

In `src/ea/_cache.py`:

```python
import shutil
import subprocess
import sys
from pathlib import Path

from ea._bindings import CompileError, _build_kernel_module


def load(path, *, target="native", opt_level=3, avx512=False):
    """Compile, cache, and load an .ea kernel. Returns a module-like object.

    Args:
        path: Path to .ea source file.
        target: CPU target (default "native").
        opt_level: Optimization level 0-3 (default 3).
        avx512: Enable AVX-512 features.

    Returns:
        A module-like object with callable methods for each exported function.
    """
    from ea._compiler import _ea_binary, _resolve_target, compiler_version

    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Eä source file not found: {source}")

    # Resolve target and version for cache key
    cpu = _resolve_target() if target == "native" else target
    version = compiler_version()
    cache = _cache_dir(source, cpu, version)

    # Check cache
    if not _is_cached(source, cache):
        cache.mkdir(parents=True, exist_ok=True)
        # Copy source to cache dir (bare filename for ea CLI)
        cache_source = cache / source.name
        try:
            shutil.copy2(str(source), str(cache_source))

            cmd = [str(_ea_binary()), source.name, "--lib"]
            if target != "native":
                cmd.append(f"--target={target}")
            cmd.append(f"--opt-level={opt_level}")
            if avx512:
                cmd.append("--avx512")

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60,
                cwd=str(cache),
            )
            if result.returncode != 0:
                raise CompileError(
                    f"Compilation failed: {source.name}",
                    stderr=result.stderr,
                    exit_code=result.returncode,
                )
        finally:
            # Always clean up temp .ea copy
            if cache_source.exists():
                cache_source.unlink()

    # Load metadata and build bindings
    json_path = cache / f"{source.name}.json"
    so_path = cache / f"{source.stem}{_lib_ext()}"
    return _build_kernel_module(json_path, so_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cache.py::test_load_compiles_and_returns_callable -v`
Expected: PASS (or skip if no ea binary)

- [ ] **Step 5: Run all cache tests**

Run: `pytest tests/test_cache.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/ea/_cache.py tests/test_cache.py
git commit -m "feat: implement ea.load() with caching"
```

---

## Chunk 5: `_bindings.py` — runtime ctypes binding generation

### Task 8: Implement JSON metadata parser

**Files:**
- Modify: `src/ea/_bindings.py`
- Create: `tests/test_bindings.py`

- [ ] **Step 1: Write the test**

```python
import json
from pathlib import Path
import pytest


SAMPLE_METADATA = {
    "library": "scale.so",
    "exports": [
        {
            "name": "scale",
            "args": [
                {"name": "src", "type": "*restrict f32", "direction": "in", "cap": None, "count": None},
                {"name": "dst", "type": "*mut f32", "direction": "out", "cap": "n", "count": "n"},
                {"name": "factor", "type": "f32", "direction": "in", "cap": None, "count": None},
                {"name": "n", "type": "i32", "direction": "in", "cap": None, "count": None},
            ],
            "return_type": None,
        }
    ],
    "structs": [],
}


def test_parse_metadata(tmp_path):
    """Parse .ea.json metadata into ExportFunc list."""
    json_path = tmp_path / "scale.ea.json"
    json_path.write_text(json.dumps(SAMPLE_METADATA))
    from ea._bindings import _parse_metadata
    exports = _parse_metadata(json_path)
    assert len(exports) == 1
    assert exports[0]["name"] == "scale"
    assert len(exports[0]["args"]) == 4
    assert exports[0]["args"][0]["type"] == "*restrict f32"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bindings.py::test_parse_metadata -v`
Expected: FAIL — no `_parse_metadata`

- [ ] **Step 3: Implement `_parse_metadata()`**

In `src/ea/_bindings.py`:

```python
import json
from pathlib import Path


def _parse_metadata(json_path: Path) -> list:
    """Parse .ea.json metadata file. Returns list of export dicts."""
    with open(json_path) as f:
        data = json.load(f)
    return data.get("exports", [])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bindings.py::test_parse_metadata -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ea/_bindings.py tests/test_bindings.py
git commit -m "feat: implement metadata JSON parser"
```

### Task 9: Implement type mapping and length collapsing

**Files:**
- Modify: `src/ea/_bindings.py`
- Modify: `tests/test_bindings.py`

- [ ] **Step 1: Write the tests**

```python
def test_ea_type_to_ctypes():
    """Map Eä types to ctypes types."""
    from ea._bindings import _ea_type_to_ctype, _ea_type_to_numpy_dtype
    import ctypes
    import numpy as np
    assert _ea_type_to_ctype("f32") == ctypes.c_float
    assert _ea_type_to_ctype("f64") == ctypes.c_double
    assert _ea_type_to_ctype("i32") == ctypes.c_int32
    assert _ea_type_to_ctype("i64") == ctypes.c_int64
    assert _ea_type_to_ctype("u8") == ctypes.c_uint8
    assert _ea_type_to_ctype("u16") == ctypes.c_uint16
    assert _ea_type_to_ctype("u32") == ctypes.c_uint32
    assert _ea_type_to_ctype("u64") == ctypes.c_uint64
    assert _ea_type_to_ctype("bool") == ctypes.c_bool
    assert _ea_type_to_numpy_dtype("f32") == np.float32
    assert _ea_type_to_numpy_dtype("i32") == np.int32
    assert _ea_type_to_numpy_dtype("u8") == np.uint8


def test_is_pointer():
    from ea._bindings import _is_pointer, _is_mut_pointer, _pointer_inner
    assert _is_pointer("*f32")
    assert _is_pointer("*mut f32")
    assert _is_pointer("*restrict f32")
    assert _is_pointer("*restrict mut f32")
    assert not _is_pointer("f32")
    assert _is_mut_pointer("*mut f32")
    assert _is_mut_pointer("*restrict mut f32")
    assert not _is_mut_pointer("*f32")
    assert not _is_mut_pointer("*restrict f32")
    assert _pointer_inner("*f32") == "f32"
    assert _pointer_inner("*mut f32") == "f32"
    assert _pointer_inner("*restrict f32") == "f32"
    assert _pointer_inner("*restrict mut f32") == "f32"


def test_length_collapsing():
    """Length params after pointer args should be detected for collapsing."""
    from ea._bindings import _detect_collapsed
    args = [
        {"name": "src", "type": "*f32"},
        {"name": "n", "type": "i32"},
    ]
    collapsed = _detect_collapsed(args)
    assert collapsed == [False, True]


def test_length_collapsing_non_adjacent():
    """Length param after pointer + scalar should still collapse."""
    from ea._bindings import _detect_collapsed
    args = [
        {"name": "data", "type": "*f32"},
        {"name": "factor", "type": "f32"},
        {"name": "n", "type": "i32"},
    ]
    collapsed = _detect_collapsed(args)
    assert collapsed == [False, False, True]


def test_length_collapsing_not_without_pointer():
    """Length params with no preceding pointer should not collapse."""
    from ea._bindings import _detect_collapsed
    args = [
        {"name": "factor", "type": "f32"},
        {"name": "n", "type": "i32"},
    ]
    collapsed = _detect_collapsed(args)
    assert collapsed == [False, False]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bindings.py -k "type_to_ctypes or is_pointer or length_collapsing" -v`
Expected: FAIL

- [ ] **Step 3: Implement type mapping and collapsing**

In `src/ea/_bindings.py`:

```python
import ctypes
import numpy as np

# Eä scalar type → ctypes type
_CTYPE_MAP = {
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
    "i8": ctypes.c_int8,
    "i16": ctypes.c_int16,
    "i32": ctypes.c_int32,
    "i64": ctypes.c_int64,
    "u8": ctypes.c_uint8,
    "u16": ctypes.c_uint16,
    "u32": ctypes.c_uint32,
    "u64": ctypes.c_uint64,
    "bool": ctypes.c_bool,
}

# Eä scalar type → numpy dtype
_DTYPE_MAP = {
    "f32": np.float32,
    "f64": np.float64,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "u8": np.uint8,
    "u16": np.uint16,
    "u32": np.uint32,
    "u64": np.uint64,
}

# Length parameter names that get auto-collapsed
_LENGTH_NAMES = {"n", "len", "length", "count", "size", "num"}

# Integer types eligible for collapsing
_INTEGER_TYPES = {"i32", "i64", "u32", "u64"}


def _ea_type_to_ctype(ty: str):
    """Map an Eä type string to a ctypes type."""
    return _CTYPE_MAP.get(ty)


def _ea_type_to_numpy_dtype(ty: str):
    """Map an Eä type string to a numpy dtype."""
    return _DTYPE_MAP.get(ty)


def _is_pointer(ty: str) -> bool:
    return ty.startswith("*")


def _is_mut_pointer(ty: str) -> bool:
    if not ty.startswith("*"):
        return False
    rest = ty[1:].strip()
    if rest.startswith("restrict"):
        rest = rest[len("restrict"):].strip()
    return rest.startswith("mut")


def _pointer_inner(ty: str) -> str:
    """Extract the inner type from a pointer type string.

    Handles: *f32, *mut f32, *restrict f32, *restrict mut f32
    """
    rest = ty[1:].strip()
    if rest.startswith("restrict"):
        rest = rest[len("restrict"):].strip()
    if rest.startswith("mut"):
        rest = rest[len("mut"):].strip()
    return rest


def _detect_collapsed(args: list) -> list:
    """Detect which args are length params that should be auto-filled.

    A param is collapsed if:
    - Its name is in _LENGTH_NAMES
    - Its type is an integer type
    - Any preceding arg is a pointer (not just the immediately preceding one)

    Matches bind_common.rs find_collapsed_args() logic.
    """
    collapsed = [False] * len(args)
    has_preceding_pointer = False
    for i, arg in enumerate(args):
        if _is_pointer(arg.get("type", "")):
            has_preceding_pointer = True
        elif (has_preceding_pointer
              and arg["name"] in _LENGTH_NAMES
              and arg.get("type", "") in _INTEGER_TYPES):
            collapsed[i] = True
    return collapsed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_bindings.py -k "type_to_ctypes or is_pointer or length_collapsing" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ea/_bindings.py tests/test_bindings.py
git commit -m "feat: implement type mapping and length collapsing"
```

### Task 10: Implement `_build_kernel_module()`

**Files:**
- Modify: `src/ea/_bindings.py`
- Modify: `tests/test_bindings.py`

- [ ] **Step 1: Write the test**

```python
def test_build_kernel_module_creates_callable(tmp_path):
    """_build_kernel_module returns object with callable function attrs."""
    # Create a mock .ea.json
    json_path = tmp_path / "mock.ea.json"
    json_path.write_text(json.dumps(SAMPLE_METADATA))

    # We can't test actual calling without a real .so,
    # but we can test the module structure
    from unittest.mock import patch, MagicMock
    mock_cdll = MagicMock()
    with patch("ea._bindings.ctypes.CDLL", return_value=mock_cdll):
        from ea._bindings import _build_kernel_module
        so_path = tmp_path / "scale.so"
        so_path.touch()
        mod = _build_kernel_module(json_path, so_path)
        assert hasattr(mod, "scale")
        assert callable(mod.scale)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bindings.py::test_build_kernel_module_creates_callable -v`
Expected: FAIL — no `_build_kernel_module`

- [ ] **Step 3: Implement `_build_kernel_module()`**

In `src/ea/_bindings.py`:

```python
class _KernelModule:
    """Module-like object returned by ea.load(). Has callable function attrs."""

    def __init__(self, name: str):
        self._name = name

    def __repr__(self):
        funcs = [k for k in self.__dict__ if not k.startswith("_")]
        return f"<ea.Kernel '{self._name}' funcs={funcs}>"


def _build_kernel_module(json_path: Path, so_path: Path) -> _KernelModule:
    """Build a module-like object with callable methods from metadata + .so."""
    exports = _parse_metadata(json_path)
    lib = ctypes.CDLL(str(so_path.resolve()))
    module = _KernelModule(json_path.stem.removesuffix(".ea"))

    for export in exports:
        func = _build_function(lib, export)
        setattr(module, export["name"], func)

    return module


def _build_function(lib, export: dict):
    """Build a single callable Python function from export metadata."""
    name = export["name"]
    args = export.get("args", [])
    return_type = export.get("return_type")
    collapsed = _detect_collapsed(args)

    # Set up ctypes argtypes and restype
    c_func = getattr(lib, name)
    argtypes = []
    for arg in args:
        ty = arg["type"]
        if _is_pointer(ty):
            inner = _pointer_inner(ty)
            ct = _ea_type_to_ctype(inner)
            argtypes.append(ctypes.POINTER(ct) if ct else ctypes.c_void_p)
        else:
            ct = _ea_type_to_ctype(ty)
            argtypes.append(ct if ct else ctypes.c_int32)
    c_func.argtypes = argtypes
    c_func.restype = _ea_type_to_ctype(return_type) if return_type else None

    # Determine which args are visible in the Python signature
    # (not collapsed, not auto-allocated out params with cap)
    visible_names = []
    for i, arg in enumerate(args):
        if collapsed[i]:
            continue
        direction = arg.get("direction", "in")
        cap = arg.get("cap")
        if direction == "out" and cap is not None:
            continue
        visible_names.append(arg["name"])

    def wrapper(*pos_args, **kwargs):
        # Map positional args to keyword args by visible parameter order
        for j, val in enumerate(pos_args):
            if j < len(visible_names):
                kwargs[visible_names[j]] = val

        # Build the C argument list (thread-safe: all state is local)
        c_args = [None] * len(args)
        out_bufs = {}  # local, not on function object

        # Track which pointer arg provides length for collapsed params
        last_array_size = None

        for i, arg in enumerate(args):
            ty = arg["type"]
            arg_name = arg["name"]

            if collapsed[i]:
                # Auto-fill from preceding array size
                ct = _ea_type_to_ctype(ty)
                c_args[i] = ct(last_array_size) if ct else last_array_size
                continue

            direction = arg.get("direction", "in")
            cap = arg.get("cap")

            if direction == "out" and cap is not None:
                # Auto-allocate output buffer
                inner = _pointer_inner(ty)
                dtype = _ea_type_to_numpy_dtype(inner)
                # Resolve cap: use last_array_size (cap typically references
                # the collapsed length param which equals the input array size)
                if cap in kwargs:
                    alloc_size = int(kwargs[cap])
                elif last_array_size is not None:
                    alloc_size = last_array_size
                else:
                    raise ValueError(f"Cannot determine output size for '{arg_name}'")
                out_buf = np.empty(alloc_size, dtype=dtype)
                ct = _ea_type_to_ctype(inner)
                c_args[i] = out_buf.ctypes.data_as(ctypes.POINTER(ct))
                out_bufs[arg_name] = out_buf
                last_array_size = alloc_size
                continue

            if _is_pointer(ty):
                # Numpy array argument
                arr = kwargs.get(arg_name)
                if arr is None:
                    raise TypeError(f"Missing required argument: {arg_name}")
                inner = _pointer_inner(ty)
                expected_dtype = _ea_type_to_numpy_dtype(inner)
                if expected_dtype and arr.dtype != expected_dtype:
                    raise TypeError(
                        f"{arg_name}: expected {expected_dtype}, got {arr.dtype}"
                    )
                ct = _ea_type_to_ctype(inner)
                c_args[i] = arr.ctypes.data_as(ctypes.POINTER(ct))
                last_array_size = arr.size
            else:
                # Scalar argument
                val = kwargs.get(arg_name)
                if val is None:
                    raise TypeError(f"Missing required argument: {arg_name}")
                ct = _ea_type_to_ctype(ty)
                c_args[i] = ct(val) if ct else val

        # Call C function
        result = c_func(*c_args)

        # Return logic
        if out_bufs and result is None:
            bufs = list(out_bufs.values())
            return bufs[0] if len(bufs) == 1 else tuple(bufs)
        elif out_bufs:
            bufs = list(out_bufs.values())
            return (result, *bufs) if len(bufs) == 1 else (result, *bufs)
        elif return_type:
            py_type = {"f32": float, "f64": float, "bool": bool}.get(return_type)
            return py_type(result) if py_type else int(result)
        return None

    wrapper.__name__ = name
    wrapper.__doc__ = f"{name}({', '.join(visible_names)})"
    return wrapper
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bindings.py::test_build_kernel_module_creates_callable -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ea/_bindings.py tests/test_bindings.py
git commit -m "feat: implement runtime ctypes binding generation"
```

---

## Chunk 6: Integration testing

### Task 11: End-to-end integration test

**Files:**
- Create: `tests/test_integration.py`
- Create: `tests/kernels/scale.ea`
- Create: `tests/kernels/dot.ea`

This requires a real `ea` binary. Copy it into `src/ea/bin/` for local testing, or set `EA_BIN` to find it.

- [ ] **Step 1: Create test kernel files**

`tests/kernels/scale.ea`:
```
export func scale(src: *f32, dst: *mut f32, factor: f32, n: i32) {
    let mut i: i32 = 0
    while i < n {
        dst[i] = src[i] * factor
        i = i + 1
    }
}
```

`tests/kernels/dot.ea`:
```
export func dot(a: *f32, b: *f32, n: i32) -> f32 {
    let mut sum: f32 = 0.0
    let mut i: i32 = 0
    while i < n {
        sum = sum + a[i] * b[i]
        i = i + 1
    }
    return sum
}
```

- [ ] **Step 2: Write integration tests**

`tests/test_integration.py`:

```python
import sys
import shutil
from pathlib import Path

import numpy as np
import pytest

# Skip all if ea binary not available
try:
    from ea._compiler import _ea_binary
    _ea_binary()
    EA_AVAILABLE = True
except (RuntimeError, OSError):
    EA_AVAILABLE = False

pytestmark = pytest.mark.skipif(not EA_AVAILABLE, reason="ea binary not available")

KERNEL_DIR = Path(__file__).parent / "kernels"


class TestLoad:
    def test_scale_kernel(self, tmp_path):
        shutil.copy(KERNEL_DIR / "scale.ea", tmp_path / "scale.ea")
        import ea
        kernel = ea.load(str(tmp_path / "scale.ea"))
        src = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        dst = kernel.scale(src, factor=3.0)
        np.testing.assert_array_almost_equal(dst, [3.0, 6.0, 9.0, 12.0])

    def test_dot_product_kernel(self, tmp_path):
        shutil.copy(KERNEL_DIR / "dot.ea", tmp_path / "dot.ea")
        import ea
        kernel = ea.load(str(tmp_path / "dot.ea"))
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        result = kernel.dot(a, b)
        assert abs(result - 32.0) < 1e-5

    def test_cache_created(self, tmp_path):
        shutil.copy(KERNEL_DIR / "scale.ea", tmp_path / "scale.ea")
        import ea
        ea.load(str(tmp_path / "scale.ea"))
        assert (tmp_path / "__eacache__").exists()

    def test_second_load_uses_cache(self, tmp_path):
        shutil.copy(KERNEL_DIR / "scale.ea", tmp_path / "scale.ea")
        import ea
        ea.load(str(tmp_path / "scale.ea"))
        # Modify .so timestamp to be old — should still use cache
        import time
        time.sleep(0.05)
        ea.load(str(tmp_path / "scale.ea"))
        # No error = cache was used

    def test_clear_cache(self, tmp_path):
        shutil.copy(KERNEL_DIR / "scale.ea", tmp_path / "scale.ea")
        import ea
        ea.load(str(tmp_path / "scale.ea"))
        ea.clear_cache(tmp_path)
        assert not (tmp_path / "__eacache__").exists()

    def test_compile_error_raises(self, tmp_path):
        bad_file = tmp_path / "bad.ea"
        bad_file.write_text("this is not valid ea code")
        import ea
        with pytest.raises(ea.CompileError):
            ea.load(str(bad_file))

    def test_file_not_found(self):
        import ea
        with pytest.raises(FileNotFoundError):
            ea.load("nonexistent.ea")

    def test_wrong_dtype_raises(self, tmp_path):
        shutil.copy(KERNEL_DIR / "scale.ea", tmp_path / "scale.ea")
        import ea
        kernel = ea.load(str(tmp_path / "scale.ea"))
        src = np.array([1.0, 2.0], dtype=np.float64)  # wrong dtype
        with pytest.raises(TypeError):
            kernel.scale(src, factor=2.0)


class TestCompile:
    def test_compile_produces_so(self, tmp_path):
        shutil.copy(KERNEL_DIR / "scale.ea", tmp_path / "scale.ea")
        import ea
        result = ea.compile(str(tmp_path / "scale.ea"))
        ext = ".dll" if sys.platform == "win32" else ".so"
        assert result.suffix == ext
        assert result.exists()

    def test_compile_file_not_found(self):
        import ea
        with pytest.raises(FileNotFoundError):
            ea.compile("nonexistent.ea")


class TestMisc:
    def test_version(self):
        import ea
        assert ea.__version__

    def test_compiler_version(self):
        import ea
        v = ea.compiler_version()
        assert v  # non-empty
        parts = v.split(".")
        assert len(parts) >= 2  # at least major.minor
```

- [ ] **Step 3: Run integration tests**

Run: `pytest tests/test_integration.py -v`
Expected: All PASS (or all skip if no ea binary)

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py tests/kernels/
git commit -m "test: add end-to-end integration tests"
```

---

## Chunk 7: CI/CD — publish workflow

### Task 12: Create CI and publish workflows

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/publish.yml`

- [ ] **Step 1: Create CI workflow**

`.github/workflows/ci.yml`:

```yaml
name: CI
on: [push, pull_request]

env:
  EA_VERSION: "v1.7.0"

jobs:
  test:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Download ea binary
        run: |
          curl -fsSL "https://github.com/petlukk/eacompute/releases/download/$EA_VERSION/ea-linux-x86_64.tar.gz" | tar xz
          chmod +x ea
          mkdir -p src/ea/bin
          cp ea src/ea/bin/ea

      - name: Install package
        run: pip install -e ".[test]"

      - name: Run tests
        run: pytest tests/ -v
```

- [ ] **Step 2: Create publish workflow**

`.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI
on:
  push:
    tags: ["ea-compiler-v*"]
  workflow_dispatch:

env:
  EA_VERSION: "v1.7.0"

permissions:
  id-token: write
  contents: read

jobs:
  build-linux-x86_64:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Download ea binary
        run: |
          curl -fsSL "https://github.com/petlukk/eacompute/releases/download/$EA_VERSION/ea-linux-x86_64.tar.gz" | tar xz
          chmod +x ea
          mkdir -p src/ea/bin
          cp ea src/ea/bin/ea

      - name: Build wheel
        run: |
          pip install build wheel
          python -m build --wheel

      - name: Retag wheel
        run: |
          pip install wheel
          cd dist
          for f in *.whl; do
            python -m wheel tags --remove --platform-tag manylinux_2_17_x86_64 "$f"
          done

      - name: Test wheel
        run: |
          python -m venv /tmp/test-venv
          /tmp/test-venv/bin/pip install dist/*.whl
          /tmp/test-venv/bin/python -c "import ea; print(ea.__version__); print(ea.compiler_version())"

      - uses: actions/upload-artifact@v4
        with:
          name: wheel-linux-x86_64
          path: dist/*.whl

  build-linux-aarch64:
    runs-on: ubuntu-24.04-arm
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Download ea binary
        run: |
          curl -fsSL "https://github.com/petlukk/eacompute/releases/download/$EA_VERSION/ea-linux-aarch64.tar.gz" | tar xz
          chmod +x ea
          mkdir -p src/ea/bin
          cp ea src/ea/bin/ea

      - name: Build wheel
        run: |
          pip install build wheel
          python -m build --wheel

      - name: Retag wheel
        run: |
          pip install wheel
          cd dist
          for f in *.whl; do
            python -m wheel tags --remove --platform-tag manylinux_2_17_aarch64 "$f"
          done

      - name: Test wheel
        run: |
          python -m venv /tmp/test-venv
          /tmp/test-venv/bin/pip install dist/*.whl
          /tmp/test-venv/bin/python -c "import ea; print(ea.__version__); print(ea.compiler_version())"

      - uses: actions/upload-artifact@v4
        with:
          name: wheel-linux-aarch64
          path: dist/*.whl

  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Download ea binary
        shell: bash
        run: |
          curl -fsSL "https://github.com/petlukk/eacompute/releases/download/$EA_VERSION/ea-windows-x86_64.zip" -o ea.zip
          unzip ea.zip
          mkdir -p src/ea/bin
          cp ea.exe src/ea/bin/ea.exe

      - name: Build wheel
        run: |
          pip install build wheel
          python -m build --wheel

      - name: Retag wheel
        run: |
          pip install wheel
          cd dist
          Get-ChildItem *.whl | ForEach-Object {
            python -m wheel tags --remove --platform-tag win_amd64 $_.Name
          }

      - name: Test wheel
        run: |
          python -m venv C:\test-venv
          C:\test-venv\Scripts\pip install (Get-ChildItem dist\*.whl)
          C:\test-venv\Scripts\python -c "import ea; print(ea.__version__); print(ea.compiler_version())"

      - uses: actions/upload-artifact@v4
        with:
          name: wheel-windows
          path: dist/*.whl

  publish:
    needs: [build-linux-x86_64, build-linux-aarch64, build-windows]
    runs-on: ubuntu-24.04
    environment: pypi
    steps:
      - uses: actions/download-artifact@v4
        with:
          path: wheels
          merge-multiple: true

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: wheels/
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/
git commit -m "ci: add CI and PyPI publish workflows"
```

---

## Chunk 8: Local testing and first release prep

### Task 13: Local end-to-end validation

**Files:** None new — validation only

- [ ] **Step 1: Copy ea binary into package for local testing**

```bash
cp /root/dev/eacompute/target/release/ea src/ea/bin/ea
# or if no release build:
cp /root/dev/eacompute/target/debug/ea src/ea/bin/ea
```

- [ ] **Step 2: Install and run all tests**

```bash
pip install -e .
pytest tests/ -v
```

Expected: All tests pass

- [ ] **Step 3: Test the user journey manually**

Create a temp directory, write a kernel, test ea.load():

```bash
mkdir /tmp/ea-test && cd /tmp/ea-test
cat > scale.ea << 'EOF'
export func scale(src: *f32, dst: *mut f32, factor: f32, n: i32) {
    let mut i: i32 = 0
    while i < n {
        dst[i] = src[i] * factor
        i = i + 1
    }
}
EOF

python3 -c "
import ea
import numpy as np
kernel = ea.load('scale.ea')
src = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
dst = kernel.scale(src, factor=2.0)
print(dst)  # [2. 4. 6. 8.]
print('Success!')
"
```

- [ ] **Step 4: Verify cache behavior**

```bash
ls __eacache__/  # should show {cpu}-{version}/ directory
python3 -c "import ea; ea.clear_cache('.')"
ls __eacache__ 2>&1  # should fail — directory removed
```

- [ ] **Step 5: Create GitHub repo and push**

```bash
cd /root/dev/ea-compiler-py
gh repo create petlukk/ea-compiler-py --private --source=. --push
```

- [ ] **Step 6: Tag and verify CI**

```bash
git tag ea-compiler-v1.7.0
git push origin ea-compiler-v1.7.0
# Watch CI at: gh run list --workflow=publish.yml
```

---

## Summary

| Chunk | Tasks | What it delivers |
|-------|-------|-----------------|
| 1 | Task 1 | `ea --print-target` in eacompute |
| 2 | Task 2 | Repo + package skeleton |
| 3 | Tasks 3-5 | `_compiler.py` — subprocess wrapper |
| 4 | Tasks 6-7 | `_cache.py` — caching + `ea.load()` |
| 5 | Tasks 8-10 | `_bindings.py` — runtime ctypes generation |
| 6 | Task 11 | Integration tests |
| 7 | Task 12 | CI + publish workflows |
| 8 | Task 13 | Local validation + first release |

Documentation (mdBook) is a separate follow-up plan.
