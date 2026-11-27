# Combining comparison masks

Since v1.21.0 a comparison of two `TxN` vectors gives a mask of type
`mask<T>xN`, and masks of the same source combine directly:

```
let c: u8x16 = load(p, i)
let ws: u64 = nibble_mask_u8x16((c .== splat(32)) .| (c .== splat(9)))  // ARM
let ws: u32 = movemask_u8x16((c .== splat(32)) .| (c .== splat(9)))     // x86
```

Each `.|` is one `vpor` (ARM `orr`). Before v1.21.0 this was written as
`select(c .== splat(32), ff, z) .| select(c .== splat(9), ff, z)`.

- **Only masks of the same source combine.** `(a_i32x8 .< b) .& (c_f32x8 .< d)`
  is a type error: the two masks live in different registers' layouts, and
  combining them would need a hidden pack or extend.
- **There is no mask NOT.** Flip the comparison (`.==` to `.!=`, `.<` to
  `.>=`), or, on integer lanes, use `andnot_*` (`a & !b` in one `vpandn` /
  `bic`).
- **Masks are written inline.** A `let` needs a type annotation, and mask
  types have no spelling.
