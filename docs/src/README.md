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

## Built with Eä

Eä is used across domains — vector search, image processing, audio,
encrypted search, LLM inference, agent runtimes — not because we
targeted each one, but because explicit compute turns out to be the
right foundation for all of them.

### Write your own kernels
```bash
pip install ea-compiler
```

`ea.load("kernel.ea")` compiles your kernel to native SIMD, caches it,
and returns a callable Python function. No C toolchain, no Cython, no
JIT warmup. Works on Linux x86_64, Linux aarch64, and Windows x86_64.

### Or use a pre-built package

If you don't need a custom kernel, the ecosystem has you covered:

| Package | Domain | Result |
|---------|--------|--------|
| [eavec](https://github.com/petlukk/eavec) | Vector similarity search | 4–8× faster than FAISS at dim=384 |
| [easobel](https://github.com/petlukk/easobel) | Edge detection | 5× faster than OpenCV single-threaded |
| [eastack](https://github.com/petlukk/eastack) | Frame stacking | 1.76× faster than NumPy at DRAM scale |
| [eastat](https://github.com/petlukk/eastat) | CSV statistics | Single-pass SIMD over columnar data |
| [eachacha](https://github.com/petlukk/eachacha) | Encrypted search | Search ChaCha20 ciphertext without decrypting |
| [ea-CQT](https://github.com/petlukk/ea-CQT-audio-visualizer) | Audio / spectrum | 1.9× faster than NumPy BLAS, 2.1× faster than FFT |

Pre-compiled `.so` included — `pip install <package>` and go.

### LLM inference infrastructure

| Package | What | Result |
|---------|------|--------|
| [eakv](https://github.com/petlukk/eakv) | Q4 KV cache · C + AVX-512 | 6–13× compression vs F16, 5–8× faster fused attention |
| [eabitnet](https://github.com/petlukk/eabitnet) | BitNet 1-bit inference kernels | Drop-in for BitNet's hand-written C intrinsics, runs on Pi 5 |

Both plug into llama.cpp and form the inference stack powering eaclaw's
local LLM path.

### Applications

[eaclaw](https://github.com/petlukk/eaclaw) — Cache-resident SIMD agent
framework. Full hot-path kernel set within L1 icache.

[eaclaw-eye](https://github.com/petlukk/eaclaw-eye) — WhatsApp-controlled
security camera on a Raspberry Pi 5. 8KB CNN model, entire inference
pipeline L1-resident. Motion detection in nanoseconds, classification
in microseconds. One binary, zero cloud. Total hardware cost: ~$95.

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
