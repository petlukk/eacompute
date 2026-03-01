#!/usr/bin/env python3
"""
Honest benchmark: Eastat (Ea kernels) vs pandas.

In-process timing with phase breakdowns showing where each tool spends time.
Both tools compute equivalent statistics: count, mean, std, min, max, 25%, 50%, 75%.

Usage:
    python bench.py [test_file.csv]
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def bench_eastat(filepath):
    """Benchmark eastat in-process with phase breakdown."""
    # Import locally to get clean timing
    from eastat import process

    # Warmup
    process(filepath)

    # Timed run
    _, _, _, _, timings = process(filepath)

    return timings


def bench_pandas(filepath):
    """Benchmark pandas with phase breakdown."""
    try:
        import pandas as pd
    except ImportError:
        print("  pandas not installed, skipping")
        return None

    # Warmup
    _ = pd.read_csv(filepath).describe()

    # Phase 1: read_csv
    t0 = time.perf_counter()
    df = pd.read_csv(filepath)
    t_read = time.perf_counter() - t0

    # Phase 2: .describe() full (count, mean, std, min, 25%, 50%, 75%, max)
    t0 = time.perf_counter()
    _ = df.describe()
    t_describe = time.perf_counter() - t0

    return {
        'read_csv': t_read,
        'describe': t_describe,
        'total': t_read + t_describe,
    }


def main():
    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
    else:
        candidates = [
            SCRIPT_DIR / "test_1000000.csv",
            SCRIPT_DIR / "test_10k.csv",
        ]
        filepath = None
        for c in candidates:
            if c.exists():
                filepath = c
                break
        if filepath is None:
            print("No test file found. Run generate_test.py first.")
            print("  python generate_test.py --rows=1000000")
            sys.exit(1)

    file_size = filepath.stat().st_size
    size_mb = file_size / (1024**2)
    print(f"File: {filepath.name} ({size_mb:.1f} MB)")
    print("=" * 64)

    # --- Kernel Analysis ---
    print("\n=== Kernel Analysis (ea inspect) ===")
    ea_root = SCRIPT_DIR / ".." / ".."
    for kernel in ["kernels/csv_parse.ea", "kernels/csv_stats.ea"]:
        kernel_path = SCRIPT_DIR / kernel
        if kernel_path.exists():
            inspect = subprocess.run(
                ["cargo", "run", "--features=llvm", "--release", "--",
                 "inspect", str(kernel_path)],
                capture_output=True, text=True, cwd=str(ea_root),
            )
            if inspect.returncode == 0:
                print(f"\n--- {kernel} ---")
                print(inspect.stdout)
    print("=" * 64)

    # --- Eastat ---
    ea = bench_eastat(filepath)

    print("\neastat breakdown:")
    print(f"  scan  (structural extraction):  {ea.get('scan', 0)*1000:7.1f} ms")
    print(f"  layout (row/delim index):       {ea.get('layout', 0)*1000:7.1f} ms")
    print(f"  stats  (parse + reduce):        {ea.get('stats', 0)*1000:7.1f} ms")
    ea_total = ea.get('total', 0)
    print(f"  total:                          {ea_total*1000:7.1f} ms")

    # --- pandas ---
    pd_timings = bench_pandas(filepath)

    if pd_timings:
        print(f"\npandas breakdown:")
        print(f"  read_csv (parse -> DataFrame): {pd_timings['read_csv']*1000:7.1f} ms")
        print(f"  .describe() full:              {pd_timings['describe']*1000:7.1f} ms")
        print(f"  total:                         {pd_timings['total']*1000:7.1f} ms")

    # --- Comparison ---
    print("\n" + "=" * 64)

    if pd_timings and ea_total > 0:
        ratio = pd_timings['total'] / ea_total

        print(f"\nEquivalent work (count/mean/std/min/25%/50%/75%/max):")
        print(f"  eastat vs pandas .describe():  {ratio:.1f}x {'faster' if ratio > 1 else 'slower'}")

        print(f"\nBoth tools compute: count, mean, std, min, 25%, 50%, 75%, max.")
        print(f"eastat uses f32 SIMD reductions; pandas uses f64.")


if __name__ == '__main__':
    main()
