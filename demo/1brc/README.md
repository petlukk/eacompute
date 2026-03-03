# 1BRC: One Billion Row Challenge with Ea SIMD Kernels

The [1 Billion Row Challenge](https://1brc.dev/): parse a 13GB CSV with 1 billion rows of `StationName;Temperature\n`, compute min/mean/max per station, print sorted.

This demo showcases Ea's SIMD byte-processing capabilities — newline scanning via `u8x16 + movemask`, integer temperature parsing, and the limits of kernel-accelerated Python.

## Architecture

```
kernels/
  scan.ea           — SIMD newline scanner (x86, u8x16 + movemask)
  scan_arm.ea       — scalar newline scanner (ARM fallback)
  parse_temp.ea     — batch temperature parser (cross-platform, scalar)
build_kernels.sh    — compile with arch detection
generate.py         — generate test data (configurable row count)
solve.py            — main solver: read → scan → parse → aggregate → print
bench.py            — benchmark: phase breakdown + pure Python + polars
```

Pipeline: `file read → Ea scan (find newlines) → Ea parse (semicolons + temperatures) → Python dict aggregate → sort + print`

## Kernels

### scan.ea (x86) / scan_arm.ea (ARM)

Two exports: `count_lines` and `extract_lines`.

**count_lines**: SIMD newline counting via `u8x16` compare + `select` + `reduce_add`. Portable across x86 and ARM. Used as a fast pre-allocation pass.

**extract_lines** (x86): Finds exact newline byte positions using `movemask` + bit extraction:
1. Load 16 bytes as `u8x16`, compare with `splat(10)` (newline)
2. `select(cmp, splat(0xFF), splat(0))` → byte mask
3. `movemask(mask)` → 16-bit integer bitmask
4. Bit extraction loop: `while m != 0 { if m % 2 == 1 { emit position } m = m / 2 }`

**Why u8x16 not u8x32**: `movemask` on `u8x32` returns a 32-bit value in `i32`. If bit 31 is set, the value is negative, and `/ 2` (signed division) gives wrong results for bit extraction. `u8x16` produces a 16-bit mask (max 65535), always positive. Throughput is the same — we're memory-bandwidth bound.

**Why `% 2` and `/ 2`**: Ea has no scalar bitwise operators (`&`, `|`, `>>`, `<<`). Bit extraction uses `m % 2` for LSB test and `m / 2` for right-shift. This works correctly because the mask is always non-negative.

**extract_lines** (ARM): Scalar byte-by-byte scan. No `movemask` equivalent on NEON, and without scalar bitwise ops, there's no fast way to extract bit positions from a SIMD comparison result.

### parse_temp.ea (cross-platform)

Export: `batch_parse_temps`. For each line delimited by newline positions:
1. Scan backward from newline to find `;` (3-6 bytes, faster than forward scan)
2. Parse temperature bytes after `;` into `i32` tenths: `"12.3" → 123`, `"-5.7" → -57`
3. Output: semicolon offset (for Python name extraction) + temperature in tenths

No float arithmetic — pure integer. The 1BRC temperature format is always `[-]D[D].D` (one decimal digit), so `i32` tenths are exact.

## Benchmark Results (10M rows, 131 MB)

```
Phase breakdown (Ea, 1 worker):
  read           :    344 ms
  scan (Ea SIMD) :    394 ms   — find newline positions
  parse (Ea)     :    487 ms   — batch_parse_temps (i32 tenths)
  aggregate (Py) :   7737 ms   — Python dict accumulation ← bottleneck
  sort+print     :      1 ms
  total          :   8963 ms

Comparison:
  Pure Python    :  11230 ms   (line-by-line, float())
  Ea speedup     :    1.3x
  Polars         :   2630 ms

Bottleneck analysis:
  scan+parse     :   9.8%  (Ea kernels)
  aggregate      :  86.3%  (Python dict — the bottleneck)
```

**1.3x faster than pure Python.** The Ea kernels process 131 MB in ~881 ms (scan + parse, ~149 MB/s). The speedup comes from:
1. **Integer temperature parsing** — Ea parses all 10M temperatures to `i32` tenths in 487ms, vs Python's `float()` + `round()` + `int()` per row
2. **SIMD newline scanning** — u8x16 compare + movemask finds all 10M positions in 394ms
3. **Zero-copy pointer passing** — `ctypes.c_char_p` wraps the `bytes` buffer without copying

The bottleneck is the per-row Python aggregation loop (86%) — dict lookups, min/max/sum updates. On multi-core machines, `ProcessPoolExecutor` scales this near-linearly by splitting the file into chunks aligned to newline boundaries.

## Optimizations Applied

| # | Optimization | Where | Impact |
|---|---|---|---|
| 1 | SIMD newline scanning | scan.ea | ~4x vs scalar (but not the bottleneck) |
| 2 | Integer temps (i32 tenths) | parse_temp.ea | Avoids `float()`, exact arithmetic |
| 3 | Backward `;` scan | parse_temp.ea | 4-6 byte scan vs full line scan |
| 4 | Two-pass scan (count → extract) | scan.ea | Pre-allocate exact array sizes |
| 5 | Multi-process parallelism | solve.py | ~Nx on N cores for aggregate phase |
| 6 | `dict.get()` instead of `in` + `[]` | solve.py | One hash lookup per row instead of two |
| 7 | Prefetch in scan loop | scan.ea | Hides memory latency |
| 8 | Chunk boundary alignment | solve.py | Clean line boundaries, no partial lines |
| 9 | `memoryview` for name extraction | solve.py | Avoids intermediate bytearray copies |
| 10 | Buffered data generation | generate.py | 64KB write batches |

## Ea Language Constraints

This demo exercises several Ea constraints:

- **No scalar bitwise ops** — bit extraction from `movemask` uses `% 2` and `/ 2`
- **No ctz/popcnt** — bit scanning is a loop
- **No hash map** — aggregation must be in Python (but Python `dict` IS a C hash table)
- **movemask is x86-only** — ARM variant uses scalar scan for position extraction
- **All memory caller-provided** — numpy/ctypes allocate, Ea fills

## Usage

```bash
# Build kernels
EA_BIN=./target/debug/ea bash demo/1brc/build_kernels.sh

# Generate test data
python3 demo/1brc/generate.py 1000000    # 1M rows
python3 demo/1brc/generate.py 1000000000  # 1B rows (13 GB)

# Run solver
python3 demo/1brc/solve.py demo/1brc/data/measurements_1M.txt

# Run benchmark
python3 demo/1brc/bench.py demo/1brc/data/measurements_1M.txt
```
