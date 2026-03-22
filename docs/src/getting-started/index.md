# Getting Started

Eä is a compute kernel compiler. You write small, focused compute functions, compile them to native code (`.so`/`.dll`), and call them from Python, Rust, C++, or PyTorch via C ABI.

No runtime, no GC, no standard library. Just kernels.

## Quick start

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

`ea.load()` compiles your kernel to a native shared library, caches it, and gives you a callable Python function. No Rust, no LLVM, no build step.

## Next steps

- [Installation](install.md) — platform-specific setup
- [Your First Kernel](first-kernel.md) — write a SIMD kernel step by step
- [Why Eä?](../guide/why-ea.md) — design philosophy and when to use Eä
