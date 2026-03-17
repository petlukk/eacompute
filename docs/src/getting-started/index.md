# Make Python Fast in 60 Seconds

Eä is a compute kernel compiler. You write small, focused compute functions — Eä compiles them to native code and gives you Python functions that take NumPy arrays.

No C. No Cython. No Numba JIT warmup. Just fast code.

## The pitch

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

That's it. `ea.load()` compiles your kernel to a native shared library, caches it, and gives you a callable Python function. No Rust, no LLVM, no build step — just `pip install` and go.

## What makes Eä different?

- **Explicit SIMD.** You control the vector width. No auto-vectorization guessing.
- **No runtime.** Compiles to plain `.so`/`.dll` files. Zero overhead beyond function call.
- **No magic.** If your code looks like it processes 8 floats at a time, it does. No silent scalar fallback.
- **Instant bindings.** Python, Rust, C++, PyTorch — generated from one source.

## Next steps

- [Install ea-compiler](install.md)
- [Write your first SIMD kernel](first-kernel.md)
