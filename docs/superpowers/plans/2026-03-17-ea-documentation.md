# Eä Documentation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship user-friendly documentation for Eä using mdBook, deployed to GitHub Pages. Four layers: Getting Started (60-second hook) → Guide (learn the language) → Reference (exhaustive lookup) → Cookbook (real-world patterns).

**Architecture:** mdBook in `eacompute/docs/`, GitHub Actions deploys to Pages on push. Every page has runnable examples assuming `pip install ea-compiler`.

**Tech Stack:** mdBook, GitHub Actions, GitHub Pages

**Spec:** `docs/superpowers/specs/2026-03-17-ea-compiler-pip-package-design.md` (Documentation section)

---

## Chunk 1: mdBook setup and Getting Started

### Task 1: Initialize mdBook structure

**Files:**
- Create: `docs/book.toml`
- Create: `docs/src/SUMMARY.md`
- Create: `docs/src/getting-started/index.md`
- Create: `docs/src/getting-started/install.md`
- Create: `docs/src/getting-started/first-kernel.md`

- [ ] **Step 1: Install mdBook**

```bash
cargo install mdbook
```

- [ ] **Step 2: Create `docs/book.toml`**

```toml
[book]
authors = ["Peter Lukka"]
language = "en"
multilingual = false
src = "src"
title = "Eä — Compute Kernel Compiler"

[build]
build-dir = "book"

[output.html]
default-theme = "navy"
preferred-dark-theme = "navy"
git-repository-url = "https://github.com/petlukk/eacompute"
```

- [ ] **Step 3: Create `docs/src/SUMMARY.md`**

```markdown
# Summary

# Getting Started

- [Make Python Fast in 60 Seconds](getting-started/index.md)
- [Installation](getting-started/install.md)
- [Your First Kernel](getting-started/first-kernel.md)

# Guide

- [Why Eä?](guide/why-ea.md)
- [Language Basics](guide/language.md)
- [SIMD Vectors](guide/simd.md)
- [Kernels](guide/kernels.md)
- [Structs](guide/structs.md)
- [Common Intrinsics](guide/common-intrinsics.md)

# Reference

- [Type System](reference/types.md)
- [All Intrinsics](reference/intrinsics.md)
- [CLI Reference](reference/cli.md)
- [Python API](reference/python-api.md)
- [Binding Annotations](reference/bindings.md)
- [ARM / NEON](reference/arm.md)

# Cookbook

- [Eä vs NumPy](cookbook/numpy-comparison.md)
- [Image Processing](cookbook/image-processing.md)
- [Text Processing](cookbook/text-processing.md)
- [ML Preprocessing](cookbook/ml-preprocessing.md)
```

- [ ] **Step 4: Write `docs/src/getting-started/index.md`**

The landing page. Must hook a Python developer in 60 seconds. Show the end-to-end experience: install, write kernel, call from Python, see result.

```markdown
# Make Python Fast in 60 Seconds

Eä is a compute kernel compiler. You write small, focused compute functions — Eä compiles them to native code and gives you Python functions that take NumPy arrays.

No C. No Cython. No Numba JIT warmup. Just fast code.

## The pitch

```bash
pip install ea-compiler
```

Write a kernel (`scale.ea`):

\```
export func scale(src: *f32, dst: *mut f32, factor: f32, n: i32) {
    let mut i: i32 = 0
    while i < n {
        dst[i] = src[i] * factor
        i = i + 1
    }
}
\```

Call it from Python:

\```python
import ea
import numpy as np

kernel = ea.load("scale.ea")
src = np.random.randn(1_000_000).astype(np.float32)
dst = np.empty_like(src)
kernel.scale(src, dst, factor=2.0)
\```

That's it. `ea.load()` compiles your kernel to a native shared library, caches it, and gives you a callable Python function. No Rust, no LLVM, no build step — just `pip install` and go.

## What makes Eä different?

- **Explicit SIMD.** You control the vector width. No auto-vectorization guessing.
- **No runtime.** Compiles to plain `.so`/`.dll` files. Zero overhead beyond function call.
- **No magic.** If your code looks like it processes 8 floats at a time, it does. No silent scalar fallback.
- **Instant bindings.** Python, Rust, C++, PyTorch — generated from one source.

## Next steps

- [Install ea-compiler](install.md)
- [Write your first SIMD kernel](first-kernel.md)
```

- [ ] **Step 5: Write `docs/src/getting-started/install.md`**

```markdown
# Installation

## From pip (recommended)

\```bash
pip install ea-compiler
\```

This gives you the `ea` compiler and the Python `ea.load()` API. No other dependencies needed (besides NumPy).

Works on:
- Linux x86_64
- Linux aarch64 (ARM)
- Windows x86_64

## Verify installation

\```python
import ea
print(ea.__version__)           # e.g., "1.7.0"
print(ea.compiler_version())    # same, from the bundled binary
\```

## Building from source

For development or unsupported platforms, see the [eacompute README](https://github.com/petlukk/eacompute) for instructions on building the compiler from source. This requires Rust and LLVM 18.
```

- [ ] **Step 6: Write `docs/src/getting-started/first-kernel.md`**

```markdown
# Your First Kernel

Let's write a kernel that scales an array by a constant factor, then upgrade it to use SIMD.

## Step 1: Scalar version

Create `scale.ea`:

\```
export func scale(src: *f32, dst: *mut f32, factor: f32, n: i32) {
    let mut i: i32 = 0
    while i < n {
        dst[i] = src[i] * factor
        i = i + 1
    }
}
\```

Key things to notice:
- `export` makes the function callable from Python (C ABI)
- `*f32` is an immutable pointer to float32 — your input array
- `*mut f32` is a mutable pointer — your output array
- `n: i32` is the array length — the caller provides this
- All types are explicit. No inference, no ambiguity.

## Step 2: Call from Python

\```python
import ea
import numpy as np

kernel = ea.load("scale.ea")

src = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
dst = np.empty_like(src)
kernel.scale(src, dst, factor=2.0)
print(dst)  # [2. 4. 6. 8.]
\```

`ea.load()` compiles `scale.ea` the first time, then caches the result in `__eacache__/`. Subsequent calls load from cache instantly.

## Step 3: SIMD version

Now let's process 8 floats at a time:

\```
export func scale_simd(src: *f32, dst: *mut f32, factor: f32, n: i32) {
    let s: f32x8 = splat(factor)
    let mut i: i32 = 0
    while i < n {
        let v: f32x8 = load(src, i)
        store(dst, i, v .* s)
        i = i + 8
    }
}
\```

What changed:
- `f32x8` — a vector of 8 floats (256-bit, uses AVX2 on x86)
- `splat(factor)` — broadcasts the scalar to all 8 lanes
- `load(src, i)` — loads 8 consecutive floats starting at index `i`
- `v .* s` — element-wise multiply (the dot prefix `.` means "vector operation")
- `store(dst, i, ...)` — writes 8 floats back
- We increment by 8, not 1

> **Important:** This only works when `n` is a multiple of 8. For arbitrary lengths, see [Kernels](../guide/kernels.md) for tail-handling strategies.

## Step 4: Compare performance

\```python
import ea
import numpy as np
import time

kernel = ea.load("scale.ea")
src = np.random.randn(10_000_000).astype(np.float32)
dst = np.empty_like(src)

# Eä scalar
start = time.perf_counter()
for _ in range(100):
    kernel.scale(src, dst, factor=2.0)
ea_scalar = (time.perf_counter() - start) / 100

# Eä SIMD
start = time.perf_counter()
for _ in range(100):
    kernel.scale_simd(src, dst, factor=2.0)
ea_simd = (time.perf_counter() - start) / 100

# NumPy
start = time.perf_counter()
for _ in range(100):
    dst = src * 2.0
numpy_time = (time.perf_counter() - start) / 100

print(f"Eä scalar: {ea_scalar*1000:.2f} ms")
print(f"Eä SIMD:   {ea_simd*1000:.2f} ms")
print(f"NumPy:     {numpy_time*1000:.2f} ms")
\```

For this simple operation, all three will be similar (it's bandwidth-bound). Eä shines on compute-bound workloads — stencils, fused operations, custom reductions. See the [Cookbook](../cookbook/numpy-comparison.md) for real-world comparisons.
```

- [ ] **Step 7: Build and verify**

```bash
cd docs && mdbook build && mdbook serve
```

Open `http://localhost:3000` and verify the pages render correctly.

- [ ] **Step 8: Commit**

```bash
git add -f docs/book.toml docs/src/
git commit -m "docs: add mdBook setup and Getting Started pages"
```

---

## Chunk 2: Guide — learn the language

### Task 2: Write Guide pages

**Files:**
- Create: `docs/src/guide/why-ea.md`
- Create: `docs/src/guide/language.md`
- Create: `docs/src/guide/simd.md`
- Create: `docs/src/guide/kernels.md`
- Create: `docs/src/guide/structs.md`
- Create: `docs/src/guide/common-intrinsics.md`

- [ ] **Step 1: Write `guide/why-ea.md`**

Cover:
- What Eä is: a compute kernel compiler, not a general-purpose language
- The problem: Python is slow for tight loops, existing solutions (Cython, Numba, C extensions) are complex
- Eä's philosophy: explicit over implicit, no hidden performance cliffs
- When to use Eä: compute-bound workloads (stencils, FMA chains, reductions, custom algorithms)
- When NOT to use Eä: bandwidth-bound workloads (simple element-wise ops where NumPy already saturates memory bandwidth), general-purpose programming, string processing
- The compilation model: .ea → .so/.dll → callable from any language via C ABI

Keep it under 150 lines. No code examples longer than 10 lines.

- [ ] **Step 2: Write `guide/language.md`**

Cover with examples:
- Scalar types: `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `f32`, `f64`, `bool`
- Variables: `let x: i32 = 5`, `let mut x: i32 = 0` (all types explicit)
- Constants: `const PI: f32 = 3.14159`
- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Comparison: `<`, `>`, `<=`, `>=`, `==`, `!=`
- Logical: `and`, `or`, `not`
- Control flow: `if`/`else if`/`else`, `while`, `for i in 0..n step 1`, `foreach`
- Loop modifiers: `unroll(N) { while ... }` — compile-time unrolling
- Compile-time checks: `static_assert(CONST_EXPR, "message")` — fails compilation if false
- Functions: `func name(args) -> ReturnType { ... }`, `export func` for C ABI
- Pointers: `*T` (read-only), `*mut T` (writable), `*restrict T`, `*restrict mut T`
- Pointer indexing: `ptr[i]` for read, `ptr[i] = value` for write (on `*mut`)
- Type casts: `to_f32(x)`, `to_i32(x)`, `to_f64(x)`, `to_i64(x)`
- No: generics, traits, modules, imports, allocations, strings (except println)

Keep it under 250 lines. Lots of small examples.

- [ ] **Step 3: Write `guide/simd.md`**

Cover:
- Vector types: `f32x4`, `f32x8`, `i32x4`, `i32x8`, `u8x16`, `u8x32`, etc.
- Dot operators: `.+`, `.-`, `.*`, `./`, `.>`, `.<`, `.==`, `.>=`, `.<=`, `.!=`, `.&`, `.|`, `.^`
- `splat(scalar)` — broadcast to all lanes
- `load(ptr, index)` / `store(ptr, index, value)` — vector memory access
- Element access: `v[0]`, `v[1]`, etc.
- `select(mask, a, b)` — conditional per-lane select
- Vector widths and what hardware they need (128-bit = SSE/NEON, 256-bit = AVX2, 512-bit = AVX-512)
- The philosophy: if you write `f32x8`, you get 8-wide SIMD. No silent fallback.

Include a side-by-side scalar vs SIMD example.

- [ ] **Step 4: Write `guide/kernels.md`**

Cover:
- The `kernel` construct: syntactic sugar for vectorized loops with tail handling
- Syntax: `kernel name(params) over i in n step 8 tail strategy { tail_body } { main_body }`
- The desugared form: main while-loop + optional tail loop
- Tail strategies:
  - `scalar` — process remainder one element at a time
  - `mask` — use masked load/store for the remainder
  - `pad` — caller guarantees padding, no tail needed
  - No tail clause — skip remainder entirely
- The `n` parameter: auto-injected from the range bound
- Example: a complete kernel with `tail scalar`

- [ ] **Step 5: Write `guide/structs.md`**

Cover:
- Struct definition: `struct Point { x: f32, y: f32 }`
- Field access: `p.x`, `p.y`
- Struct as function parameter: `func process(p: *Point)`
- Field assignment: `p.x = value` (on mutable pointer)
- Passing structs by pointer from C/Python
- No methods, no impl blocks — structs are plain data

Keep short (~80 lines).

- [ ] **Step 6: Write `guide/common-intrinsics.md`**

Cover the top ~10 most-used intrinsics with a brief example each:
- `splat(scalar)` — broadcast
- `load(ptr, index)` / typed loads (`load_f32`, `load_u8`, etc.)
- `store(ptr, index, value)` / `stream_store(ptr, index, value)`
- `fma(a, b, c)` — fused multiply-add (scalar and vector)
- `reduce_add(vector)` / `reduce_max(vector)` / `reduce_min(vector)`
- `select(mask, a, b)`
- `sqrt(x)` / `rsqrt(x)`
- `min(a, b)` / `max(a, b)`
- `movemask(bool_vector)` — x86 only

Point to the reference page for the full list.

- [ ] **Step 7: Build and verify**

```bash
cd docs && mdbook build
```

- [ ] **Step 8: Commit**

```bash
git add -f docs/src/guide/
git commit -m "docs: add Guide section — language, SIMD, kernels, structs, intrinsics"
```

---

## Chunk 3: Reference — exhaustive lookup

### Task 3: Write Reference pages

**Files:**
- Create: `docs/src/reference/types.md`
- Create: `docs/src/reference/intrinsics.md`
- Create: `docs/src/reference/cli.md`
- Create: `docs/src/reference/python-api.md`
- Create: `docs/src/reference/bindings.md`
- Create: `docs/src/reference/arm.md`

- [ ] **Step 1: Write `reference/types.md`**

Complete type table:

| Type | Size | Description |
|------|------|-------------|
| `i8` | 1 byte | Signed 8-bit integer |
| `u8` | 1 byte | Unsigned 8-bit integer |
| ... (all scalar types) |
| `f32x4` | 16 bytes | 4-wide float32 vector (SSE/NEON) |
| `f32x8` | 32 bytes | 8-wide float32 vector (AVX2) |
| ... (all vector types) |

Include pointer types, vector types, struct types. Note which vector types require which hardware.

- [ ] **Step 2: Write `reference/intrinsics.md`**

Every intrinsic with:
- Signature (input types, return type)
- Brief description
- Hardware constraints (if any, e.g., "x86 only", "AVX-512 only")
- One-line example

Organized by category: Memory, Math, Reduction, Vector, Conversion, Debug.

Full list from the codebase:
- Memory: `load`, typed loads (`load_f32`, `load_f64`, `load_i32`, `load_i16`, `load_i8`, `load_u8`, `load_u16`, `load_u32`, `load_u64`, `load_f32x4`, `load_f32x8`, `load_f32x16`, `load_i32x4`, `load_i32x8`, `load_i16x8`, `load_i8x16`, `load_u8x16`, `load_u8x32`), `store`, `stream_store`, `load_masked`, `store_masked`, `gather`, `scatter`, `prefetch`
- Math: `sqrt`, `rsqrt`, `exp`, `fma`, `min`, `max` (note: `min`/`max` work on both scalar and vector types)
- Reduction: `reduce_add`, `reduce_max`, `reduce_min`, `reduce_add_fast`
- Vector: `splat`, `shuffle`, `select`, `movemask` (x86 only)
- Conversion: `to_f32`, `to_f64`, `to_i32`, `to_i64`, `widen_i8_f32x4`, `widen_u8_f32x4`, `widen_i8_f32x8`, `widen_u8_f32x8`, `widen_u8_i32x4`, `widen_u8_i32x8`, `narrow_f32x4_i8`, `maddubs_i16` (signature: `(u8x16, i8x16) -> i16x8`), `maddubs_i32` (signature: `(u8x16, i8x16) -> i32x4`)
- Debug: `println` (accepts integers, floats, bools, string literals, vectors)

- [ ] **Step 3: Write `reference/cli.md`**

Document all CLI commands and flags from `src/usage.rs`:
- `ea <file.ea> [flags]` — compile
- `ea bind <file.ea> --python [--rust] [--cpp] [--pytorch] [--cmake]` — bindings
- `ea inspect <file.ea>` — instruction analysis
- `ea --print-target` — CPU name
- All flags: `-o`, `--lib`, `--opt-level`, `--avx512`, `--target`, `--target-triple`, `--emit-llvm`, `--emit-asm`, `--header`, `--emit-ast`, `--emit-tokens`, `--help`/`-h`, `--version`/`-V`

- [ ] **Step 4: Write `reference/python-api.md`**

Document the `ea` Python module:
- `ea.load(path, *, target="native", opt_level=3, avx512=False)` — returns kernel module
- `ea.compile(path, *, ...)` — returns Path to .so
- `ea.clear_cache(path=None)`
- `ea.compiler_version()` → str
- `ea.__version__` → str
- `ea.CompileError` — attributes: `message`, `stderr`, `exit_code`
- Cache behavior: `__eacache__/{cpu}-{version}/`, mtime-based, auto-recompiles
- Kernel module: `kernel.func_name(array, scalar=val)` — positional and keyword args
- Length collapsing: `n`/`len`/`length`/`count`/`size`/`num` after pointer auto-filled
- Output allocation: `out` params with `[cap: ...]` auto-allocated and returned
- Thread safety: kernel objects are safe for concurrent use

- [ ] **Step 5: Write `reference/bindings.md`**

Document output annotations:
- Syntax: `out name: *mut T [cap: expr, count: path]`
- Length collapsing rules (which param names, which types)
- How bindings auto-allocate output arrays
- Examples in Python, Rust, C++
- The `.ea.json` metadata format

- [ ] **Step 6: Write `reference/arm.md`**

Document ARM/NEON constraints:
- 128-bit max vector width (f32x4, i32x4, u8x16)
- No 256-bit or 512-bit types on ARM
- No `movemask` on ARM
- No `gather`/`scatter` on ARM
- Cross-compilation: `--target-triple=aarch64-unknown-linux-gnu`
- Strategy: write `*_arm.ea` variants with 128-bit types

- [ ] **Step 7: Build and verify**

```bash
cd docs && mdbook build
```

- [ ] **Step 8: Commit**

```bash
git add -f docs/src/reference/
git commit -m "docs: add Reference section — types, intrinsics, CLI, Python API, bindings, ARM"
```

---

## Chunk 4: Cookbook and GitHub Pages deployment

### Task 4: Write Cookbook pages

**Files:**
- Create: `docs/src/cookbook/numpy-comparison.md`
- Create: `docs/src/cookbook/image-processing.md`
- Create: `docs/src/cookbook/text-processing.md`
- Create: `docs/src/cookbook/ml-preprocessing.md`

- [ ] **Step 1: Write `cookbook/numpy-comparison.md`**

Side-by-side comparisons:
- Scale array: NumPy one-liner vs Eä kernel (bandwidth-bound — similar performance, explain why)
- Dot product: NumPy `np.dot` vs Eä with `fma` and `reduce_add` (compute-bound — Eä can win)
- Fused operations: `(a * scale + bias).clip(0, 1)` — NumPy does 3 passes, Eä fuses into 1
- Rule of thumb: when does Eä win? (>1 arithmetic op per element loaded)

- [ ] **Step 2: Write `cookbook/image-processing.md`**

Patterns:
- Stencil (Sobel edge detection) — links to easobel package
- Convolution (3x3 kernel) — sliding window with vectors
- Pixel pipeline (u8 input → f32 processing → u8 output) — widen/narrow intrinsics

- [ ] **Step 3: Write `cookbook/text-processing.md`**

Patterns:
- SIMD character search (chunk-skip pattern using movemask)
- CSV column counting
- Links to eastat package

- [ ] **Step 4: Write `cookbook/ml-preprocessing.md`**

Patterns:
- Normalize array (subtract mean, divide by std)
- Dot product / cosine similarity — links to eavec package
- FMA chains for polynomial evaluation

- [ ] **Step 5: Commit**

```bash
git add -f docs/src/cookbook/
git commit -m "docs: add Cookbook section — NumPy comparison, image, text, ML patterns"
```

### Task 5: GitHub Pages deployment workflow

**Files:**
- Create: `.github/workflows/docs.yml`

- [ ] **Step 1: Create docs deploy workflow**

`.github/workflows/docs.yml`:

```yaml
name: Deploy Documentation
on:
  push:
    branches: [main]
    paths: ["docs/**"]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4

      - name: Install mdBook
        run: |
          curl -fsSL https://github.com/rust-lang/mdBook/releases/download/v0.4.40/mdbook-v0.4.40-x86_64-unknown-linux-gnu.tar.gz | tar xz
          chmod +x mdbook

      - name: Build docs
        run: ./mdbook build docs/

      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: docs/book

  deploy:
    needs: build
    runs-on: ubuntu-24.04
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Commit**

```bash
git add -f .github/workflows/docs.yml
git commit -m "ci: add GitHub Pages deployment for docs"
```

---

## Summary

| Chunk | Tasks | What it delivers |
|-------|-------|-----------------|
| 1 | Task 1 | mdBook setup + Getting Started (3 pages) |
| 2 | Task 2 | Guide section (6 pages) |
| 3 | Task 3 | Reference section (6 pages) |
| 4 | Tasks 4-5 | Cookbook (4 pages) + GitHub Pages deploy |

Total: 19 documentation pages + deployment workflow.
