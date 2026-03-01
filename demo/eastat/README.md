# Eastat

CSV column statistics powered by [Eä](https://github.com/petlukk/eacompute) `ea bind`.

Zero manual ctypes. Every kernel call goes through auto-generated Python bindings from `ea bind --python`.

## Results

### Three-tool comparison

| File | eastat | polars | pandas | vs polars | vs pandas |
|------|--------|--------|--------|-----------|-----------|
| 10 MB (stress) | **42 ms** | 43 ms | 212 ms | 1.0x (tied) | 5.1x |
| 47 MB (1M rows) | **209 ms** | 453 ms | 670 ms | 2.2x | 3.2x |
| 544 MB (11M rows) | **5,284 ms** | 7,363 ms | 18,552 ms | 1.4x | 3.5x |

All three tools compute equivalent statistics: count, mean, std, min, 25%, 50%, 75%, max.

Eastat beats pandas 3–5x at every size, beats polars 1.4–2.2x, and handles adversarial CSVs without falling off a cliff.

### Where the speedup comes from

- **Streaming, not materializing**: eastat mmap's the file, scans structure, and computes stats per-column without building a DataFrame. Pandas and polars both materialize the full table.
- **SIMD reductions**: fused sum + min + max + sum-of-squares in a single pass with f32x8 dual accumulators and FMA. Percentiles via SIMD binary search (no sort).
- **f32 precision**: f32x8 processes 2x more values per vector lane than f64x4. See [Precision](#precision-f32-vs-f64) for the measured cost of this trade-off.

This is the same architectural advantage that makes DuckDB and polars faster than pandas on analytical queries — avoid materializing what you don't need. Eastat pushes it further by fusing the parse and compute phases.

### Scaling behavior

The pandas advantage *grows* with file size (3.2x → 3.5x) because DataFrame materialization overhead scales with row count. The polars advantage *narrows* (2.2x → 1.4x) because polars' Rust allocator and lazy evaluation handle large files more memory-efficiently.

Throughput drops from 224 MB/s at 47 MB to 103 MB/s at 544 MB. The 544 MB file is 34x the 16 MB L3 cache on the test machine (AMD EPYC 9354P, 3.8 GB RAM). On a machine with 16+ GB RAM and a larger L3, the throughput at scale would be higher — the streaming architecture is memory-bandwidth-bound, not compute-bound, and the constrained RAM on the test machine amplifies page-fault and TLB pressure.

For files under 128 MB, eastat uses a single-pass scan with generous buffer allocation. For larger files, it switches to a two-pass strategy — count-only scan first, then exact allocation and position extraction — saving ~400 MB of buffer allocation on the 544 MB file without hurting the common case.

### Phase breakdown (47 MB)

```
eastat breakdown:
  scan  (structural extraction):     97 ms   <- Ea kernel (quote-aware, ~480 MB/s)
  layout (row/delim index):          27 ms   <- Ea kernel
  stats  (parse + reduce + pct):     86 ms   <- Ea kernel
  total:                            209 ms

polars breakdown:
  read_csv:                         245 ms
  .describe():                      208 ms
  total:                            453 ms

pandas breakdown:
  read_csv (parse -> DataFrame):    500 ms
  .describe() full:                 170 ms
  total:                            670 ms
```

### Stress-test results (100k rows, adversarial CSV)

Tested on a file with: UTF-8 BOM, `\r\n` line endings, quoted fields containing commas and newlines, doubled quotes, empty fields, non-numeric values in numeric columns, whitespace-padded numbers, 500-character fields, negative numbers, scientific notation.

| Tool | Total | vs eastat |
|------|-------|-----------|
| **eastat** | **42 ms** | — |
| polars | 43 ms | tied |
| pandas | 212 ms | 5.1x slower |

The messier the data, the bigger the gap over pandas. Eastat's quote-aware scan handles edge cases at the same throughput; pandas' general-purpose C parser pays more overhead per edge case. Polars' Rust CSV parser is equally efficient on messy data — the tie is a strong result for both tools.

**What matched**: row count (100,000 exact), null counts (5/6 columns), column names, all numeric stats within tolerance.

**What diverged**: `score` column — pandas treats it as string entirely (due to ~4% `N/A`/`-` values). Eastat parses the 94% that are numeric and computes stats on that subset. Scientific notation (`1e3`) is not parsed by `batch_atof` and is counted as null.

## Precision (f32 vs f64)

Eastat computes all statistics in f32. Pandas and polars use f64. Measured on the same data:

| Stat | Max relative error | Source |
|------|--------------------|--------|
| mean | 1.4e-06 | f32 rounding |
| stddev | 1.4e-05 | sum-of-squares accumulation |
| min/max | <1e-08 | exact for representable values |
| percentiles | 5.3e-05 | method difference (binary search vs linear interpolation) |

The largest divergence (5e-05) comes from percentile method differences, not f32 accumulation drift. On the clean dataset (values 0–100k), mean and stddev agree to 6 significant figures. On the stress dataset (values up to ~1e9), the error stays below 1e-04.

Anyone who needs f64 precision on `describe()` already knows they need it. For analytics — spotting distributions, outliers, column profiles — f32 is sufficient.

Run `python bench.py --precision test_file.csv` to see exact divergence on your data.

## Known limitations

| Pattern | Status |
|---------|--------|
| Quoted fields with embedded commas | Handled (quote-aware scan) |
| Quoted fields with embedded newlines | Handled (quote-aware scan skips LFs inside quotes) |
| UTF-8 BOM prefix | Handled — 3-byte BOM stripped from header before parsing |
| Windows line endings (`\r\n`) | Handled — `\r` stripped from header; row boundaries use LF |
| Scientific notation (`1e3`) | **Not parsed** — `batch_atof` handles decimal notation only; counted as null |
| Non-numeric values in numeric columns | Handled — `batch_atof` skips unparseable fields, column falls back to string if >50% fail |
| Empty fields (nulls) | Handled — excluded from count and statistics |
| Whitespace-padded numerics | Handled — `batch_atof` strips leading whitespace |
| Very long fields (>256 chars) | Handled (tested with 500-char fields) |

## Code size

For the work eastat actually does — structural CSV scanning, numeric parsing, and column statistics including percentiles — the entire implementation is 1,042 lines compiling to 38 KB of shared objects. Pandas needs roughly 10x the code for the same computation (~6,000 lines Python + ~4,000 lines C/Cython). Polars' Python surface is comparable in size, but dispatches to a Rust core of tens of thousands of lines.

This is not a claim of feature equivalence. Pandas and polars are general-purpose DataFrame libraries. Eastat does one thing. The point is that `ea bind` makes writing that one thing remarkably compact — you write the kernels, the bindings are generated, and there's no framework in between.

| Component | Lines |
|-----------|-------|
| `csv_parse.ea` (7 kernel exports) | 474 |
| `csv_stats.ea` (2 kernel exports) | 164 |
| `eastat.py` (pipeline + output) | 404 |
| **Total** | **1,042** |

`grep -c ctypes eastat.py` → **0**

## Kernels

**csv_parse.ea** — 7 exports:
- `count_positions_quoted` — count-only structural scan (no position output, used for exact allocation on large files)
- `extract_positions_quoted` — full structural scan (delimiters + newlines with positions, quote-aware)
- `build_row_arrays` — row start/end from LF positions (replaces Python edge-case logic)
- `build_row_delim_index` — per-row delimiter mapping via O(n) merge-scan
- `compute_field_bounds` — field start/end for any column
- `batch_atof` — ASCII-to-float parser at C speed
- `field_length_stats` — string column min/max/total length

**csv_stats.ea** — 2 exports:
- `f32_column_stats` — fused sum + min + max + sum-of-squares (f32x8 dual-accumulator with FMA)
- `f32_percentiles` — binary-search p25/p50/p75 with SIMD counting (3 simultaneous searches per data pass)

## Usage

```bash
# Build (compile kernels + generate Python bindings)
./build.sh

# Generate test data
python generate_test.py --rows=1000000

# Generate adversarial test data
python generate_test.py --stress --rows=100000

# Run
python eastat.py test_1000000.csv
python eastat.py --json test_1000000.csv
python eastat.py -c 2,5 test_1000000.csv

# Benchmark vs pandas and polars
python bench.py                                    # three-tool comparison
python bench.py --precision test_1000000.csv       # f32 vs f64 precision analysis
python bench.py --sizes 47MB,500MB                 # scaling test
```

## Pipeline

```
1. mmap file → uint8 array
2. count_positions_quoted (large files) or single-pass extract
3. extract_positions_quoted → delimiter + LF positions
4. build_row_arrays + build_row_delim_index → row layout
5. per column: compute_field_bounds → batch_atof → f32_column_stats → f32_percentiles
6. print results
```

Every kernel call is one line:

```python
csv_parse.count_positions_quoted(text, delim, counts)
csv_parse.extract_positions_quoted(text, delim, buf1, buf2, counts)
csv_stats.f32_column_stats(values, out_sum, out_min, out_max, out_sumsq)
csv_stats.f32_percentiles(values, min_val, max_val, out_p25, out_p50, out_p75)
```
