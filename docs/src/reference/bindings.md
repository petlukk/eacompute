# Binding Annotations

Ea generates language bindings from compiled kernel metadata. This page documents the annotation syntax, the `.ea.json` metadata format, and the available binding generators.

## Output Annotations

Mark a parameter as caller-allocated output with the `out` keyword and a capacity clause:

```
export func filter(
    input: *f32,
    n: i32,
    out result: *mut f32 [cap: n, count: out_count],
    out out_count: *mut i32 [cap: 1],
) { ... }
```

### Syntax

```
out name: *mut T [cap: <expr>]
out name: *mut T [cap: <expr>, count: <path>]
```

| Clause | Required | Description |
|--------|----------|-------------|
| `cap` | Yes | Number of elements to allocate. Can reference other parameters (e.g. `n`, `width * height`) |
| `count` | No | Parameter or expression giving the actual output length. The binding trims the returned array to this length |

In generated bindings, `out` parameters are not part of the function signature. They are allocated internally and returned as the function's result.

## Length Collapsing

When a pointer parameter is followed immediately by an integer parameter whose name matches one of the recognized patterns, the binding generators automatically fill the integer from the array's length.

### Recognized Names

`n`, `len`, `length`, `count`, `size`, `num`

### Rules

1. The integer parameter must appear immediately after a pointer parameter
2. The integer parameter must have an integer type (`i32`, `i64`, `u32`, etc.)
3. The pointer and integer are collapsed into a single array argument in the binding

```
// These two parameters collapse into one array argument:
export func sum(data: *f32, n: i32) -> f32 { ... }
```

```python
# Python: just pass the array, n is filled automatically
result = k.sum(my_array)
```

## Metadata Format (.ea.json)

Compiling with `--lib` produces a `.ea.json` file alongside the shared library. This file describes the exported API and is consumed by `ea bind`.

```json
{
  "library": "libkernel.so",
  "exports": [
    {
      "name": "scale",
      "args": [
        {"name": "data", "type": "*mut f32"},
        {"name": "n", "type": "i32"},
        {"name": "factor", "type": "f32"}
      ],
      "return_type": null
    }
  ],
  "structs": [
    {
      "name": "Point",
      "fields": [
        {"name": "x", "type": "f32"},
        {"name": "y", "type": "f32"}
      ]
    }
  ]
}
```

Output-annotated parameters include additional fields:

```json
{
  "name": "result",
  "type": "*mut f32",
  "output": true,
  "cap": "n",
  "count": "out_count"
}
```

## Binding Generators

Generate bindings with `ea bind <file.ea>` and one or more language flags. The `.ea.json` file must exist (run `ea <file.ea> --lib` first).

| Flag | Output File | Description |
|------|-------------|-------------|
| `--python` | `<name>.py` | Python wrapper using `ctypes`. Arrays as NumPy ndarrays. |
| `--rust` | `<name>.rs` | Rust `extern "C"` declarations with safe wrapper functions |
| `--cpp` | `<name>.hpp` | C++ header-only bindings with `std::span` parameters |
| `--pytorch` | `<name>_torch.py` | PyTorch custom op with `torch.Tensor` parameters |
| `--cmake` | `CMakeLists.txt` | CMake project for linking the shared library in C++ |

All generated bindings use C ABI function pointers loaded from the shared library at runtime.
