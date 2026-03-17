# Kernels

The `kernel` construct is Ea's main abstraction for writing vectorized loops with automatic tail handling. It is syntactic sugar -- the compiler transforms it into a plain function with a while-loop before any further compilation.

## Basic syntax

```
export kernel name(params)
    over i in n step S
    tail strategy { tail_body }
{
    main_body
}
```

- **`name`**: the kernel's name, becomes the C ABI symbol
- **`params`**: function parameters (pointers, scalars)
- **`over i in n`**: the loop variable `i` iterates from 0 to `n`
- **`step S`**: how many elements the main body processes per iteration
- **`tail strategy`**: how to handle remainder elements when `n` is not a multiple of `S`
- **`main_body`**: the code that runs for each full chunk of `S` elements

The range variable (here `n`) is automatically injected as an `i32` parameter into the function signature. You do not declare it in the parameter list.

## How it desugars

A kernel like this:

```
export kernel scale(data: *f32, out: *mut f32, factor: f32)
    over i in n step 4
    tail scalar { out[i] = data[i] * factor }
{
    out[i] = data[i] * factor
    out[i + 1] = data[i + 1] * factor
    out[i + 2] = data[i + 2] * factor
    out[i + 3] = data[i + 3] * factor
}
```

becomes equivalent to:

```
export func scale(data: *f32, out: *mut f32, factor: f32, n: i32) {
    let mut i: i32 = 0
    while i + 4 <= n {
        out[i] = data[i] * factor
        out[i + 1] = data[i + 1] * factor
        out[i + 2] = data[i + 2] * factor
        out[i + 3] = data[i + 3] * factor
        i = i + 4
    }
    // tail: process remainder one at a time
    while i < n {
        out[i] = data[i] * factor
        i = i + 1
    }
}
```

The generated C signature is `void scale(float*, float*, float, int)` -- the `n` parameter appears last.

## Tail strategies

The tail handles remainder elements when `n` is not evenly divisible by the step.

### tail scalar

Process remainder elements one at a time. The tail body runs in a loop with step 1:

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

The main body uses 8-wide SIMD. The tail body is scalar code that handles 0 to 7 leftover elements. This is the most common tail strategy.

### tail mask

Use masked load/store for the remainder. The tail body runs once (not in a loop) and must handle all remaining elements using masked operations:

```
export kernel vscale(data: *f32, out: *mut f32, factor: f32)
    over i in len step 4
    tail mask {
        let rem: i32 = len - i
        let vf: f32x4 = splat(factor)
        let v: f32x4 = load_masked(data, i, rem)
        store_masked(out, i, v .* vf, rem)
    }
{
    let vf: f32x4 = splat(factor)
    store(out, i, load(data, i) .* vf)
}
```

The `rem` variable tells the masked operations how many elements are valid. This avoids the scalar loop entirely but requires masked intrinsics.

### tail pad

The caller guarantees the input length is a multiple of the step. No tail body is generated:

```
export kernel fill(out: *mut i32, val: i32)
    over i in n step 4
    tail pad
{
    out[i] = val
    out[i + 1] = val
    out[i + 2] = val
    out[i + 3] = val
}
```

This produces the most efficient code but shifts the responsibility to the caller. If `n` is not a multiple of the step, the kernel will skip the remaining elements.

### No tail clause

If you omit the tail clause entirely, the kernel only runs the main body. Remaining elements are not processed:

```
export kernel double_it(data: *i32, out: *mut i32)
    over i in n step 1
{
    out[i] = data[i] * 2
}
```

With `step 1`, there is no remainder, so omitting the tail is safe. With larger steps, you must ensure `n` is always a multiple of the step, or accept that trailing elements are skipped.

## Complete example

A SIMD dot product kernel that handles any input length:

```
export kernel dot(a: *f32, b: *f32, out: *mut f32)
    over i in n step 8
    tail scalar {
        out[0] = out[0] + a[i] * b[i]
    }
{
    let va: f32x8 = load(a, i)
    let vb: f32x8 = load(b, i)
    let products: f32x8 = va .* vb
    let sum: f32 = reduce_add(products)
    out[0] = out[0] + sum
}
```

The main body loads 8-wide vectors, multiplies them, reduces to a scalar sum, and accumulates into `out[0]`. The scalar tail handles any remaining 0-7 elements.
