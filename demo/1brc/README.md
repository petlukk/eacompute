# 1BRC: One Billion Row Challenge with Ea Kernels

The [1 Billion Row Challenge](https://1brc.dev/): parse a 13GB CSV with 1 billion rows of `StationName;Temperature\n`, compute min/mean/max per station, print sorted.

This demo showcases Ea's byte-processing capabilities — SIMD newline scanning, integer temperature parsing, and a native open-addressing hash table that eliminates the Python aggregation bottleneck.

## Architecture

```
kernels/
  scan.ea           — SIMD newline scanner (x86, u8x16 + movemask)
  scan_arm.ea       — scalar newline scanner (ARM fallback)
  parse_temp.ea     — standalone temperature parser (cross-platform)
  aggregate.ea      — fused parse + aggregate with hash table (cross-platform)
build_kernels.sh    — compile with arch detection
generate.py         — generate test data (configurable row count)
solve.py            — main solver: read → scan → parse+aggregate → print
bench.py            — benchmark: phase breakdown + pure Python + polars
```

Pipeline: `file read → Ea scan (find newlines) → Ea parse+aggregate (hash table) → sort + print`

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

### aggregate.ea (cross-platform)

Export: `parse_aggregate`. Fused parse + aggregate kernel with an open-addressing hash table. For each line delimited by newline positions:

1. **Find semicolon** — backward scan from newline (3-6 bytes)
2. **Hash station name** — polynomial hash: `h = h * 31 + byte`, slot via `((h % 1024) + 1024) % 1024`
3. **Parse temperature** — integer parsing to `i32` tenths, no floats
4. **Hash table insert/update** — 1024-slot open-addressing table with linear probing, byte-by-byte key comparison

The hash table uses 6 parallel arrays (all caller-allocated):
- `ht_keys` (1024 × 64 bytes) — station name storage
- `ht_key_len` (1024 × i32) — name length per slot (0 = empty)
- `ht_min/ht_max/ht_sum/ht_count` (1024 × i32 each) — aggregation stats

Total: ~84KB — fits L2 cache. With ~400 stations in 1024 slots (load factor 0.4), most lookups hit on first probe.

### parse_temp.ea (cross-platform)

Standalone temperature parser, kept for reference. The fused `aggregate.ea` kernel inlines this logic.

## Benchmark Results (10M rows, 131 MB)

```
Phase breakdown (Ea, 1 worker):
  read              :    370 ms
  scan (Ea SIMD)    :    244 ms   — find newline positions
  parse+agg (Ea HT) :    446 ms   — fused hash table kernel
  sort+print        :      0 ms
  total             :   1061 ms

Comparison:
  Pure Python    :   6384 ms   (line-by-line, float())
  Ea speedup     :    6.0x
  Polars         :   2949 ms
  Ea vs Polars   :    2.8x faster
```

**6.0x faster than pure Python, 2.8x faster than Polars.** The fused parse+aggregate kernel processes all 10M rows in 446ms — down from 7,737ms when aggregation was a Python dict loop (17x improvement on that phase alone).

### v1 → v2 comparison

| Metric | v1 (Python dict) | v2 (Ea hash table) | Change |
|--------|-------------------|---------------------|--------|
| Total (1 worker) | 8,963 ms | 1,061 ms | **8.5x faster** |
| vs Pure Python | 1.3x | 6.0x | |
| vs Polars | 3.4x slower | 2.8x faster | |
| Aggregate phase | 7,737 ms (86%) | 446 ms (42%) | **17x faster** |

## Optimizations Applied

| # | Optimization | Where | Impact |
|---|---|---|---|
| 1 | Fused parse+aggregate kernel | aggregate.ea | Eliminates 10M Python dict lookups |
| 2 | Open-addressing hash table | aggregate.ea | 84KB, L2-resident, 0.4 load factor |
| 3 | Polynomial hash (h*31+byte) | aggregate.ea | No bitwise ops needed, good distribution |
| 4 | SIMD newline scanning | scan.ea | u8x16 compare + movemask |
| 5 | Integer temps (i32 tenths) | aggregate.ea | No float arithmetic, exact |
| 6 | Backward `;` scan | aggregate.ea | 4-6 byte scan vs full line |
| 7 | Two-pass scan (count → extract) | scan.ea | Pre-allocate exact array sizes |
| 8 | Multi-process parallelism | solve.py | ~Nx on N cores, per-worker hash tables |
| 9 | Prefetch in scan loop | scan.ea | Hides memory latency |
| 10 | Chunk boundary alignment | solve.py | Clean line boundaries, no partial lines |

## Ea Language Constraints

This demo exercises several Ea constraints:

- **No scalar bitwise ops** — hash function uses `* 31 + byte` (polynomial), bit extraction uses `% 2` and `/ 2`
- **No ctz/popcnt** — bit scanning is a loop
- **No allocator** — hash table arrays are caller-provided (Python ctypes)
- **movemask is x86-only** — ARM variant uses scalar scan for position extraction
- **No break/continue** — hash table probing uses flag variables

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
