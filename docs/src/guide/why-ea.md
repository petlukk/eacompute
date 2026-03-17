# Why Ea

## What Ea is

Ea is a compute kernel compiler. You write small, focused numerical routines in Ea's explicit syntax, compile them to native shared libraries (`.so` on Linux, `.dll` on Windows), and call them from Python, Rust, C++, or PyTorch via C ABI. Ea is not a general-purpose programming language. It has no standard library, no garbage collector, no runtime. It compiles kernels.

## The problem

Python is slow for tight numerical loops. When you need to run a custom algorithm over millions of elements -- a stencil, a reduction with custom logic, a chain of fused multiply-adds -- you hit a wall. The usual options are:

- **NumPy**: fast for operations it supports, but custom multi-step algorithms require chaining calls that allocate intermediates and re-scan memory.
- **Numba**: JIT-compiles Python, but debugging is difficult, compilation is unpredictable, and SIMD vectorization is implicit (you hope the compiler figures it out).
- **Cython**: requires learning a hybrid language, managing build systems, and still gives you limited control over vectorization.
- **Writing C extensions**: full control, but high friction. Manual Python/C bridging, header files, build scripts.

Ea targets the gap: you want native-speed kernels with explicit SIMD, but you do not want to maintain a C build system.

## The philosophy

Ea is built on one principle: **explicit over implicit**.

- If you write `f32x8`, you get 8-wide SIMD. The compiler will not silently fall back to scalar code.
- If hardware does not support an operation (e.g., scatter without AVX-512), the compiler errors. It does not emit slow scalar code behind your back.
- All types are explicit. No type inference. You see exactly what is happening.
- All memory is caller-provided. Ea kernels never allocate. Pointers come from the host language.

There are no hidden performance cliffs. The code you write is the code that runs.

## When to use Ea

Ea excels at **compute-bound** workloads where you do significant work per element loaded from memory:

- **Stencil operations**: convolutions, blurs, edge detection -- each output element reads from multiple inputs.
- **Fused multiply-add chains**: polynomial evaluation, IIR filters, dot products.
- **Custom reductions**: computing statistics, finding patterns, accumulating results with non-trivial logic.
- **Particle simulations**: N-body interactions, force calculations.
- **Image processing pipelines**: per-pixel math with multiple operations fused into one pass.

The common thread: you load data, do many arithmetic operations on it, then store results. The CPU spends most of its time computing, not waiting for memory.

## When NOT to use Ea

Ea is the wrong tool for:

- **Bandwidth-bound workloads**: if you are just adding two arrays element-wise, NumPy already saturates memory bandwidth. Ea cannot make memory faster.
- **General programming**: no strings, no file I/O, no networking, no data structures beyond structs.
- **Prototyping**: write your algorithm in Python first, profile it, then port the hot loop to Ea.
- **GPU workloads**: Ea targets CPUs (x86-64 with AVX2/AVX-512, AArch64 with NEON).

A good rule of thumb: if your inner loop does fewer than 4 arithmetic operations per element loaded, NumPy is probably fast enough.

## The compilation model

```
kernel.ea  -->  ea --lib  -->  kernel.so + kernel.ea.json
                                   |
                          ea bind --python
                                   |
                              kernel.py  (generated wrapper)
```

One `.ea` file is one compilation unit. There are no imports, no modules. If you need to compose kernels, you do it at the C level -- each kernel is an independent shared library with a C ABI entry point.

The generated `.ea.json` metadata file describes the function signatures. The `ea bind` command reads it to generate idiomatic wrappers for your target language.

## Quick taste

Here is a complete Ea kernel that scales an array of floats using 8-wide SIMD:

```
export kernel vscale(data: *f32, out: *mut f32, factor: f32)
    over i in n step 8
    tail scalar {
        out[i] = data[i] * factor
    }
{
    let vf: f32x8 = splat(factor)
    store(out, i, load(data, i) .* vf)
}
```

Compile and use from Python:

```bash
ea kernel.ea --lib
ea bind kernel.ea --python
```

```python
import numpy as np
from kernel import vscale

data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
result = vscale(data, factor=3.0)
# result: [3.0, 6.0, 9.0, 12.0, 15.0]
```

The main body processes 8 elements at a time with SIMD. The `tail scalar` block handles any remainder elements one at a time. The `n` parameter (the array length) is automatically injected into the function signature from the `over i in n` clause.
