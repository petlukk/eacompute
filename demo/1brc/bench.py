#!/usr/bin/env python3
"""1BRC benchmark: per-phase timing, multi-process, pure Python baseline."""

import ctypes
import os
import sys
import time

# Import solver components
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from solve import (
    _load_lib,
    _setup_parse,
    _setup_scan,
    format_results,
    solve,
)


def bench_phases(path: str) -> dict:
    """Single-worker phase breakdown."""
    scan_lib = _setup_scan(_load_lib("scan"))
    parse_lib = _setup_parse(_load_lib("parse_temp"))
    size = os.path.getsize(path)

    # Phase 1: Read file as bytes (zero-copy ctypes pointer via c_char_p)
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        chunk_bytes = f.read()
    chunk_len = len(chunk_bytes)
    buf_ptr = ctypes.cast(ctypes.c_char_p(chunk_bytes), ctypes.c_void_p)
    t_read = time.perf_counter() - t0

    # Phase 2: Scan (count + extract)
    t0 = time.perf_counter()
    nl_count = ctypes.c_int(0)
    scan_lib.count_lines(buf_ptr, chunk_len, ctypes.byref(nl_count))
    n_lines = nl_count.value

    IntArray = ctypes.c_int * n_lines
    nl_pos = IntArray()
    extract_count = ctypes.c_int(0)
    scan_lib.extract_lines(buf_ptr, chunk_len, nl_pos, ctypes.byref(extract_count))
    n_lines = extract_count.value
    t_scan = time.perf_counter() - t0

    # Phase 3: Parse
    t0 = time.perf_counter()
    semi_off = IntArray()
    temps = IntArray()
    parse_lib.batch_parse_temps(buf_ptr, nl_pos, n_lines, 0, semi_off, temps)
    t_parse = time.perf_counter() - t0

    # Phase 4: Aggregate (bytes slicing returns bytes — hashable, no extra copy)
    t0 = time.perf_counter()
    stations: dict[bytes, list] = {}
    _get = stations.get
    for i in range(n_lines):
        ls = 0 if i == 0 else nl_pos[i - 1] + 1
        name = chunk_bytes[ls : ls + semi_off[i]]
        t = temps[i]
        entry = _get(name)
        if entry is not None:
            if t < entry[0]:
                entry[0] = t
            if t > entry[1]:
                entry[1] = t
            entry[2] += t
            entry[3] += 1
        else:
            stations[name] = [t, t, t, 1]
    t_agg = time.perf_counter() - t0

    # Phase 5: Sort + format
    t0 = time.perf_counter()
    output = format_results(stations)
    t_sort = time.perf_counter() - t0

    t_total = t_read + t_scan + t_parse + t_agg + t_sort
    return {
        "read": t_read, "scan": t_scan, "parse": t_parse,
        "aggregate": t_agg, "sort": t_sort, "total": t_total,
        "n_lines": n_lines, "output": output, "results": stations,
    }


def bench_pure_python(path: str) -> tuple[float, dict]:
    """Pure Python baseline: line-by-line, split, float()."""
    t0 = time.perf_counter()
    stations: dict[str, list] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            semi = line.index(";")
            name = line[:semi]
            temp = int(round(float(line[semi + 1:]) * 10))
            entry = stations.get(name)
            if entry is not None:
                if temp < entry[0]:
                    entry[0] = temp
                if temp > entry[1]:
                    entry[1] = temp
                entry[2] += temp
                entry[3] += 1
            else:
                stations[name] = [temp, temp, temp, 1]
    elapsed = time.perf_counter() - t0
    return elapsed, stations


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <measurements.txt>")
        sys.exit(1)

    path = sys.argv[1]
    size = os.path.getsize(path)
    size_mb = size / (1024 * 1024)

    # Count lines for display
    phases = bench_phases(path)
    n_rows = phases["n_lines"]
    n_workers = os.cpu_count() or 1

    print(f"1BRC Benchmark: {path} ({n_rows:,} rows, {size_mb:.1f} MB)")
    print()

    # Phase breakdown
    p = phases
    t = p["total"]
    print(f"Phase breakdown (Ea, 1 worker):")
    print(f"  read           : {p['read']*1000:8.1f} ms")
    print(f"  scan (Ea SIMD) : {p['scan']*1000:8.1f} ms")
    print(f"  parse (Ea)     : {p['parse']*1000:8.1f} ms")
    print(f"  aggregate (Py) : {p['aggregate']*1000:8.1f} ms")
    print(f"  sort+print     : {p['sort']*1000:8.1f} ms")
    print(f"  total          : {t*1000:8.1f} ms")
    print()

    # Multi-process
    t_mp = 0.0
    if n_workers > 1:
        t0 = time.perf_counter()
        mp_results = solve(path, n_workers)
        t_mp = time.perf_counter() - t0
        print(f"Multi-process ({n_workers} workers):")
        print(f"  total          : {t_mp*1000:8.1f} ms")
        if t > 0:
            print(f"  speedup vs 1w  : {t/t_mp:8.1f}x")
        print()

    # Pure Python
    t_py, py_results = bench_pure_python(path)
    print(f"Comparison:")
    print(f"  Pure Python    : {t_py*1000:8.1f} ms")
    if t > 0:
        print(f"  Ea speedup     : {t_py/t:8.1f}x (vs 1 worker)")
    if n_workers > 1 and t_mp > 0:
        print(f"  Ea MP speedup  : {t_py/t_mp:8.1f}x (vs {n_workers} workers)")
    print()

    # Polars (optional)
    try:
        import polars as pl  # noqa: F401
        t0 = time.perf_counter()
        df = pl.read_csv(path, separator=";", has_header=False, new_columns=["station", "temp"])
        df.group_by("station").agg(
            pl.col("temp").min().alias("min"),
            pl.col("temp").mean().alias("mean"),
            pl.col("temp").max().alias("max"),
        ).sort("station")
        t_polars = time.perf_counter() - t0
        print(f"  Polars         : {t_polars*1000:8.1f} ms")
        print()
    except ImportError:
        pass

    # Bottleneck analysis
    t_kernels = p["scan"] + p["parse"]
    if t > 0:
        print(f"Bottleneck analysis:")
        print(f"  read           : {p['read']/t*100:5.1f}%")
        print(f"  scan+parse     : {t_kernels/t*100:5.1f}%  (Ea kernels)")
        print(f"  aggregate      : {p['aggregate']/t*100:5.1f}%  (Python dict)")
        print(f"  sort+print     : {p['sort']/t*100:5.1f}%")

    # Verify correctness: compare Ea vs pure Python
    ea_results = phases["results"]
    mismatches = 0
    for name_bytes, (mn, mx, sm, ct) in ea_results.items():
        name_str = name_bytes.decode()
        py_entry = py_results.get(name_str)
        if py_entry is None:
            mismatches += 1
            continue
        if mn != py_entry[0] or mx != py_entry[1] or ct != py_entry[3]:
            mismatches += 1
    if mismatches == 0:
        print(f"\nCorrectness: PASS (Ea matches pure Python on all {len(ea_results)} stations)")
    else:
        print(f"\nCorrectness: FAIL ({mismatches} mismatches)")


if __name__ == "__main__":
    main()
