# Python API

The `ea` Python package (`pip install ea-compiler`) provides a high-level interface for compiling and loading Ea kernels.

## Functions

### ea.load

```python
ea.load(path, *, target="native", opt_level=3, avx512=False) -> KernelModule
```

Compile an `.ea` file to a shared library and load it. Returns a `KernelModule` object with each exported function available as a method.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | required | Path to the `.ea` source file |
| `target` | `str` | `"native"` | LLVM CPU name (e.g. `"skylake"`, `"native"`) |
| `opt_level` | `int` | `3` | Optimization level 0--3 |
| `avx512` | `bool` | `False` | Enable AVX-512 types and intrinsics |

```python
import ea
import numpy as np

k = ea.load("kernel.ea")
a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.zeros(3, dtype=np.float32)
k.my_func(a, b, len(a))
```

### ea.compile

```python
ea.compile(path, *, emit_asm=False, emit_llvm=False, target="native", opt_level=3, avx512=False, lib=True) -> Path
```

Compile an `.ea` file without loading it. Returns the path to the output file. Useful when you need the `.so` or `.ea.json` for another tool.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | required | Path to the `.ea` source file |
| `emit_asm` | `bool` | `False` | Also write a `.s` assembly file |
| `emit_llvm` | `bool` | `False` | Also write a `.ll` LLVM IR file |
| `target` | `str` | `"native"` | LLVM CPU name |
| `opt_level` | `int` | `3` | Optimization level 0--3 |
| `avx512` | `bool` | `False` | Enable AVX-512 |
| `lib` | `bool` | `True` | Produce `.so` + `.ea.json` |

### ea.clear_cache

```python
ea.clear_cache(path=None)
```

Clear the compilation cache. If `path` is given, clear only the cache for that `.ea` file. If `None`, clear the entire cache directory.

### ea.compiler_version

```python
ea.compiler_version() -> str
```

Return the version string of the `ea` compiler binary.

### ea.\_\_version\_\_

```python
ea.__version__ -> str
```

The version of the `ea` Python package.

## Exceptions

### ea.CompileError

Raised when compilation fails. Inherits from `RuntimeError`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `stderr` | `str` | Full compiler error output |
| `exit_code` | `int` | Compiler process exit code |

```python
try:
    k = ea.load("broken.ea")
except ea.CompileError as e:
    print(e.stderr)
    print(e.exit_code)
```

## Caching

Compiled shared libraries are cached in a `__eacache__/` directory next to the `.ea` source file. The cache key includes the CPU name and compiler version:

```
__eacache__/{cpu}-{version}/kernel.so
```

The cache is invalidated by file modification time (mtime). If the source file is newer than the cached library, `ea.load()` recompiles automatically.

## Length Collapsing

When a function parameter named `n`, `len`, `length`, `count`, `size`, or `num` appears immediately after a pointer parameter and has an integer type, the Python binding automatically fills it from the array's length. You do not need to pass it explicitly.

```
// Ea source
export func scale(data: *mut f32, n: i32, factor: f32) { ... }
```

```python
# Python: n is auto-filled from len(data)
k.scale(data, factor=2.0)
```

## Output Allocation

Parameters annotated with `out` and a `[cap: ...]` clause are automatically allocated by the Python binding and returned as the function's result.

```
// Ea source
export func transform(input: *f32, n: i32, out result: *mut f32 [cap: n]) { ... }
```

```python
# Python: result is allocated and returned
output = k.transform(input_array)
```

If `[count: path]` is also specified, the returned array is trimmed to the actual output length.

## Thread Safety

Loaded kernel modules and their functions are safe for concurrent use from multiple threads. The compiled code is stateless -- all memory is caller-provided via arguments.
