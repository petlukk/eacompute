# Eä

Write compute kernels in explicit, portable syntax. Compile to shared libraries. Generate native bindings for Python, Rust, C++, PyTorch, and CMake.

No runtime. No garbage collector. No glue code.

Targets x86-64 (AVX2, AVX-512) and AArch64 (NEON).

## What the code looks like

```
export kernel vscale(data: *f32, out result: *mut f32 [cap: n], factor: f32)
    over i in n step 8
    tail scalar { result[i] = data[i] * factor }
{
    let v: f32x8 = load(data, i)
    store(result, i, v .* splat(factor))
}
```

Compile, bind, call:

```bash
ea kernel.ea --lib                        # -> kernel.so + kernel.ea.json
ea bind kernel.ea --python --rust --cpp   # -> kernel.py, kernel.rs, kernel.hpp
```

```python
import numpy as np, kernel
data = np.random.rand(1_000_000).astype(np.float32)
result = kernel.vscale(data, 2.0)  # output auto-allocated, length auto-filled, dtype checked
```

One kernel. Any host language. The binding handles allocation, length inference, and type checking.

## Measured results

Three workloads benchmarked against industry tools. Warm-cache medians, 20–50 timed runs, 5–10 warmup. Source, data, and scripts in each demo directory.

| Workload | Compared against | Speedup | Method |
|----------|-----------------|---------|--------|
| [Vector search](demo/eavec/) (dim=384) | FAISS IndexFlatIP | **4–8×** | Dual-acc FMA, f32x8, next-vector prefetch |
| [Sobel edge detection](demo/sobel/) (720p–4K) | OpenCV | **5–6×** (single-threaded) | Stencil f32x4, prefetch, L3 scaling analysis |
| [CSV analytics](demo/eastat/) (10–544 MB) | polars | **1.4–2.2×** | Structural scan, SIMD reduction, binary search |

All three use `ea bind` for Python integration — zero manual ctypes. Validated across multiple input sizes. Full methodology and additional demos (conv2d at 265×, tokenizer at 406× vs NumPy) in [`COMPUTE_PATTERNS.md`](COMPUTE_PATTERNS.md).

## `ea bind`

Reads the compiler's JSON metadata and generates idiomatic wrappers per target:

```bash
ea bind kernel.ea --python    # -> kernel.py         (NumPy + ctypes)
ea bind kernel.ea --rust      # -> kernel.rs         (FFI + safe wrappers)
ea bind kernel.ea --cpp       # -> kernel.hpp        (std::span + extern "C")
ea bind kernel.ea --pytorch   # -> kernel_torch.py   (autograd.Function)
ea bind kernel.ea --cmake     # -> CMakeLists.txt + EaCompiler.cmake
```

Pointer args become slices/arrays/tensors. Length params collapse. Types are checked at the boundary. Multiple targets in one invocation: `ea bind kernel.ea --python --rust --cpp`

## `ea inspect`

See what the compiler produced:

```bash
ea kernel.ea --emit-asm       # assembly output
ea kernel.ea --emit-llvm      # LLVM IR
ea kernel.ea --header         # C header
```

## Quick start

```bash
# Requirements: LLVM 18, Rust
sudo apt install llvm-18-dev clang-18 libpolly-18-dev libzstd-dev
cargo build --features=llvm

# Compile + bind + run
ea kernel.ea --lib
ea bind kernel.ea --python
python -c "import kernel; print(kernel.vscale([1.0, 2.0, 3.0], 10.0))"

# Run a demo
cd demo/eastat && python run.py

# Tests (420 passing)
cargo test --features=llvm
```

## SIMD types and operations

`f32x4`, `f32x8`, `f32x16`¹, `f64x2`, `f64x4`, `i32x4`, `i32x8`, `i32x16`¹, `i8x16`, `i8x32`, `u8x16`, `i16x8`, `i16x16`

`load`, `store`, `splat`, `fma`, `shuffle`, `select`, `load_masked`, `store_masked`, `gather`, `scatter`¹, `prefetch`

`reduce_add`, `reduce_max`, `reduce_min`, `min`, `max`

`maddubs_i16(u8x16, i8x16) -> i16x8` — SSSE3 pmaddubsw, 16 pairs/cycle
`maddubs_i32(u8x16, i8x16) -> i32x4` — pmaddubsw+pmaddwd, safe i32 accumulation

`widen_u8_f32x4`, `widen_i8_f32x4`, `widen_u8_f32x8`, `widen_i8_f32x8`, `widen_u8_f32x16`¹, `widen_i8_f32x16`¹, `widen_u8_i32x4`, `widen_u8_i32x8`, `widen_u8_i32x16`¹, `narrow_f32x4_i8`, `sqrt`, `rsqrt`, `exp`, `to_f32`, `to_i32`, `to_f64`, `to_i64`

Bitwise: `.&`, `.|`, `.^` on integer vectors. Restrict pointers: `*restrict T`, `*mut restrict T`.

¹ Requires `--avx512`

## Kernel constructs

```
export kernel name(...) over i in n step N tail <strategy> { ... }
```

Tail strategies: `tail scalar { ... }` (scalar fallback), `tail mask { ... }` (masked SIMD), `tail pad` (caller pads input). Output annotations (`out name: *mut T [cap: expr]`) drive auto-allocation in bindings.

Also: `for i in 0..n step 8 { ... }` counted loops, `foreach (i in 0..n) { ... }` element-wise loops (LLVM auto-vectorizes at O2+), `unroll(N)`, compile-time `const`, `static_assert`, `#[cfg(x86_64)]` / `#[cfg(aarch64)]` conditional compilation, C-compatible structs, multi-kernel files, pointer-to-pointer `**T` parameters.

## Kernel fusion

Fusion eliminates memory round-trips between pipeline stages:

```
3 kernels (unfused):  8.55 ms   — 0.9× (slightly slower, FFI + memory overhead)
1 kernel  (fused):    0.07 ms   — 111× faster than NumPy
```

> If data leaves registers, you probably ended a kernel too early.

Analysis of when fusion helps and when it hurts: [`COMPUTE_PATTERNS.md`](COMPUTE_PATTERNS.md).

## Design

Explicit over implicit. SIMD width, loop stepping, and memory access are programmer-controlled. No hidden allocations, no auto-vectorizer in the default path, no runtime. Ea is not a general-purpose language — no strings, collections, or modules. It accelerates host languages, it does not replace them.

## Architecture

```
.ea -> Lexer -> Parser -> Desugar -> Type Check -> Codegen (LLVM 18) -> .o / .so
                                                                      -> .ea.json -> ea bind
```

~10,900 lines of Rust. 420+ tests covering SIMD ops, C interop, structs, kernel constructs, tail strategies, binding generation, ARM targets. CI on x86-64, AArch64, Windows.

[`BENCHMARKS.md`](BENCHMARKS.md) — performance tables. [`CHANGELOG.md`](CHANGELOG.md) — version history. [`1.6.md`](1.6.md) — language specification.

## License

Apache 2.0
