# Installation

## From pip (recommended)

```bash
pip install ea-compiler
```

This gives you the `ea` compiler and the Python `ea.load()` API. No other dependencies needed (besides NumPy).

Works on:
- Linux x86_64
- Linux aarch64 (ARM)
- Windows x86_64

## Verify installation

```python
import ea
print(ea.__version__)           # e.g., "1.7.0"
print(ea.compiler_version())    # same, from the bundled binary
```

## Building from source

For development or unsupported platforms, see the [eacompute README](https://github.com/petlukk/eacompute) for instructions on building the compiler from source. This requires Rust and LLVM 18.
