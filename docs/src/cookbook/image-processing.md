# Image Processing

Image processing is one of Eä's strongest use cases. Stencil operations (convolution, edge detection, blur) read multiple neighbors per output pixel, giving high arithmetic intensity that keeps the CPU compute-bound rather than memory-bound.

## Sobel edge detection

The Sobel filter computes horizontal and vertical gradients using 3x3 stencils. Each output pixel reads 9 input values and performs 9 multiplications plus additions -- well above the threshold where Eä beats NumPy.

The pattern: for each pixel, load the 3x3 neighborhood, multiply by Sobel coefficients, and sum:

```
export func sobel_x(
    src: *f32, dst: *mut f32,
    width: i32, height: i32
) {
    let neg1: f32x8 = splat(-1.0)
    let pos1: f32x8 = splat(1.0)
    let neg2: f32x8 = splat(-2.0)
    let pos2: f32x8 = splat(2.0)
    let mut y: i32 = 1
    while y < height - 1 {
        let mut x: i32 = 0
        while x < width - 2 {
            let row_above: i32 = (y - 1) * width + x
            let row_center: i32 = y * width + x
            let row_below: i32 = (y + 1) * width + x

            let tl: f32x8 = load(src, row_above)
            let tr: f32x8 = load(src, row_above + 2)
            let ml: f32x8 = load(src, row_center)
            let mr: f32x8 = load(src, row_center + 2)
            let bl: f32x8 = load(src, row_below)
            let br: f32x8 = load(src, row_below + 2)

            let gx: f32x8 = (tr .* pos1) .+ (mr .* pos2) .+ (br .* pos1)
                .+ (tl .* neg1) .+ (ml .* neg2) .+ (bl .* neg1)
            store(dst, row_center + 1, gx)
            x = x + 8
        }
        y = y + 1
    }
}
```

For a production-ready Sobel implementation with both Gx/Gy gradients and magnitude computation, see [easobel](https://github.com/petlukk/easobel).

## Pixel pipeline: u8 to f32 and back

Images from disk arrive as `u8` (0-255). SIMD math works best on `f32`. The typical pattern:

1. Load u8 pixels
2. Widen to f32 (0.0 to 255.0)
3. Process in f32 (normalize, filter, blend)
4. Narrow back to u8

### Widening: u8 to f32

```
export func normalize_u8_to_f32(src: *u8, dst: *mut f32, n: i32) {
    let scale: f32x4 = splat(0.00392156862)
    let mut i: i32 = 0
    while i < n {
        let pixels: f32x4 = widen_u8_f32x4(src, i)
        let normalized: f32x4 = pixels .* scale
        store(dst, i, normalized)
        i = i + 4
    }
}
```

`widen_u8_f32x4(ptr, offset)` loads 4 bytes from `src + offset`, zero-extends each to 32 bits, and converts to float. The result is a `f32x4` with values in 0.0 to 255.0. Multiply by `1/255` to get the 0.0-1.0 range.

### Narrowing: f32 to u8

```
export func f32_to_u8(src: *f32, dst: *mut u8, n: i32) {
    let s255: f32x4 = splat(255.0)
    let zero: f32x4 = splat(0.0)
    let mut i: i32 = 0
    while i < n {
        let v: f32x4 = load(src, i)
        let clamped: f32x4 = min(max(v, zero), s255)
        narrow_f32x4_i8(dst, i, clamped)
        i = i + 4
    }
}
```

`narrow_f32x4_i8(ptr, offset, vec)` converts 4 floats to integers, saturates to 0-255, and stores 4 bytes. Always clamp before narrowing to avoid overflow.

### Saturating arithmetic on u8 pixels

For operations that stay in u8 (brightness adjustment, blending), use `sat_add` and `sat_sub` instead of widening to f32. These clamp to 0-255 in a single instruction on both ARM (NEON) and x86 (SSE2):

```
export func brighten(src: *u8, dst: *mut u8, boost: u8, n: i32) {
    let b: u8x16 = splat(boost)
    let mut i: i32 = 0
    while i < n {
        let pixels: u8x16 = load_u8x16(src, i)
        let bright: u8x16 = sat_add(pixels, b)
        store(dst, i, bright)
        i = i + 16
    }
}
```

No widening, no clamping, no f32 intermediates. One load, one saturating add, one store. This also works with `i8x16` (signed), `i16x8`, and `u16x8`.

### Putting it together

A full pixel pipeline (load u8, process in f32, store u8) processes 4 pixels per iteration on both x86 and ARM. Use `f32x4` for the pipeline to keep it portable -- `f32x8` works only on x86 with AVX2.

## Why image stencils are compute-bound

A 3x3 convolution on a single-channel image performs 9 multiplications and 8 additions per output pixel, but only produces 1 output value. That is 17 arithmetic operations per output float -- far above the ~2 ops/element threshold where Eä's operation fusion matters.

For multi-channel images (RGB, RGBA), the arithmetic intensity is even higher because you process 3-4 channels per pixel position.

Compare this to simple brightness adjustment (`pixel * 1.1`), which is 1 op per element -- bandwidth-bound, and NumPy handles it just as fast. See [Eä vs NumPy](numpy-comparison.md) for more on this distinction.

## Frame differencing with abs_diff (ARM)

On ARM, `abs_diff` computes per-pixel absolute difference in a single NEON instruction. Useful for motion detection and video anomaly:

```
export func frame_diff(a: *u8, b: *u8, dst: *mut u8, n: i32) {
    let mut i: i32 = 0
    while i < n {
        let fa: u8x16 = load_u8x16(a, i)
        let fb: u8x16 = load_u8x16(b, i)
        let diff: u8x16 = abs_diff(fa, fb)
        store(dst, i, diff)
        i = i + 16
    }
}
```

`abs_diff` supports `i8x16`, `u8x16`, `i16x8`, `u16x8`, `i32x4`, `u32x4`. ARM-only -- on x86, use `max(a .- b, b .- a)` explicitly.

## Tips

- **Border handling**: the examples above skip border pixels (starting at y=1, ending at height-1). For production kernels, handle borders separately with scalar code or clamped indexing.
- **Separable filters**: Gaussian blur and similar filters can be split into horizontal and vertical passes, reducing a 3x3 stencil from 9 to 6 operations. Each pass is still compute-bound.
- **ARM portability**: use `f32x4` and `i32x4` for kernels that need to run on both x86 and ARM. The 128-bit types work on both architectures.
