# Text Processing

Text processing benefits from SIMD when you can skip large regions of uninteresting bytes. The key pattern is **chunk-skip**: load 32 bytes at a time, check if any byte matches a target character, and skip the entire chunk if none match.

## The chunk-skip pattern

Most bytes in a text file are not the character you are looking for. A newline scanner over a 1 MB file might find 10,000 newlines among 1,000,000 bytes -- 99% of chunks contain no match and can be skipped with a single comparison and branch.

```
export func count_newlines(data: *u8, n: i32) -> i32 {
    let newline: u8x32 = splat_u8x32(10)
    let mut count: i32 = 0
    let mut i: i32 = 0
    while i < n - 31 {
        let chunk: u8x32 = load_u8x32(data, i)
        let matches: u8x32 = chunk .== newline
        let mask: i32 = movemask(matches)
        if mask != 0 {
            let mut j: i32 = 0
            while j < 32 {
                if load_u8(data, i + j) == 10 {
                    count = count + 1
                }
                j = j + 1
            }
        }
        i = i + 32
    }
    let mut k: i32 = i
    while k < n {
        if load_u8(data, k) == 10 {
            count = count + 1
        }
        k = k + 1
    }
    count
}
```

The structure:

1. **Load 32 bytes** as `u8x32`
2. **Compare** with `splat_u8x32(target)` using `.==` to get a match vector
3. **`movemask()`** collapses the vector comparison to a single `i32` bitmask
4. **If mask is 0**: no matches in this chunk, skip ahead 32 bytes (fast path)
5. **If mask is nonzero**: scan the 32 bytes individually (slow path, but rare)
6. **Scalar tail**: handle remaining bytes that do not fill a full 32-byte chunk

The fast path processes 32 bytes in about 3 instructions. On typical text, 90-99% of chunks take the fast path.

## Why not extract individual bit positions?

Ea does not have shift operators (`<<`, `>>`), so you cannot extract individual bit positions from the movemask result. Instead, when a chunk contains matches, fall back to byte-by-byte scanning within that 32-byte window. This is still fast because:

- Hot chunks are rare (most text is not your target character)
- The 32-byte scan is a tight loop that fits in L1 cache
- The overall speedup comes from skipping cold chunks, not from optimizing hot ones

## ARM portability

`movemask` is an x86-only intrinsic (it maps to `vpmovmskb`). On ARM/NEON, there is no equivalent single instruction. For portable kernels, write separate x86 and ARM versions:

- **x86**: use `u8x32` + `movemask` as shown above
- **ARM**: use `u8x16` with scalar fallback, or structure the algorithm to avoid needing a bitmask

See [ARM / NEON](../reference/arm.md) for details on architecture-specific intrinsics.

## CSV parsing

CSV parsing combines the chunk-skip pattern with state tracking. You need to find commas and newlines while respecting quoted fields -- a comma inside quotes is not a delimiter.

The approach:

1. Scan for quote characters (`"`) using chunk-skip to track quote state
2. Scan for delimiters (`,` and `\n`) using chunk-skip, filtering by quote state
3. Parse numeric fields from the delimited regions

This is compute-bound because each byte potentially involves comparison against multiple target characters plus state logic -- exactly the kind of branching work that NumPy cannot vectorize.

For a complete CSV statistics package built on this pattern, see [eastat](https://github.com/petlukk/eastat).

## Tips

- **Buffer alignment**: SIMD loads from unaligned addresses work but may be slower on older hardware. If you control the buffer, align to 32 bytes.
- **Tail handling**: always include a scalar loop for the last `n % 32` bytes. Forgetting the tail is the most common bug in SIMD text processing.
- **Multi-character search**: to find any of several characters (e.g., `<`, `>`, `&` for HTML), do multiple `.==` comparisons and combine with `.|` before the movemask.
