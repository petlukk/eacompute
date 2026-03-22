# Eä — Compute Kernel Compiler

Write compute kernels. Compile to native code. Call from Python.

No C. No Cython. No Numba JIT warmup. Just fast code.

```bash
pip install ea-compiler
```

Write a kernel (`scale.ea`):

```
export func scale(src: *f32, dst: *mut f32, factor: f32, n: i32) {
    let mut i: i32 = 0
    while i < n {
        dst[i] = src[i] * factor
        i = i + 1
    }
}
```

Call it from Python:

```python
import ea
import numpy as np

kernel = ea.load("scale.ea")
src = np.random.randn(1_000_000).astype(np.float32)
dst = np.empty_like(src)
kernel.scale(src, dst, factor=2.0)
```

That's it. `ea.load()` compiles your kernel to a native shared library, caches it, and gives you a callable Python function.

## What makes Eä different?

- **Explicit SIMD.** You control the vector width. No auto-vectorization guessing.
- **No runtime.** Compiles to plain `.so`/`.dll` files. Zero overhead beyond function call.
- **No magic.** If your code looks like it processes 8 floats at a time, it does. No silent scalar fallback.
- **Instant bindings.** Python, Rust, C++, PyTorch — generated from one source.

## Get started

- [Install ea-compiler](getting-started/install.md)
- [Write your first SIMD kernel](getting-started/first-kernel.md)
- [Why Eä?](guide/why-ea.md)

## Explore

- [Documentation](getting-started/index.md) — language guide, SIMD vectors, kernels, structs
- [Examples](examples/cqt-audio-visualizer.md) — real-world benchmarks and walkthroughs
- [Blog](blog/numpy-6x.md) — technical deep dives
- [GitHub](https://github.com/petlukk/eacompute) — source code
