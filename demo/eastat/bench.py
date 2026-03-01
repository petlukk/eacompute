#!/usr/bin/env python3
"""
Honest benchmark: Eastat (Ea kernels) vs pandas vs polars.

In-process timing with phase breakdowns showing where each tool spends time.
All tools compute equivalent statistics: count, mean, std, min, max, 25%, 50%, 75%.

Includes f32 vs f64 precision comparison and multi-size scaling support.

Usage:
    python bench.py [test_file.csv]
    python bench.py --sizes 47MB,500MB        # multi-size scaling test
    python bench.py --precision test_file.csv  # detailed precision comparison
"""

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Eastat benchmark
# ---------------------------------------------------------------------------

def bench_eastat(filepath):
    """Benchmark eastat in-process with phase breakdown."""
    from eastat import process

    # Warmup
    process(filepath)

    # Timed run
    results, headers, n_rows, col_count, timings = process(filepath)

    return timings, results


# ---------------------------------------------------------------------------
# Pandas benchmark
# ---------------------------------------------------------------------------

def bench_pandas(filepath):
    """Benchmark pandas with phase breakdown. Returns (timings, describe_df) or None."""
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
    desc = df.describe()
    t_describe = time.perf_counter() - t0

    timings = {
        'read_csv': t_read,
        'describe': t_describe,
        'total': t_read + t_describe,
    }
    return timings, desc


# ---------------------------------------------------------------------------
# Polars benchmark
# ---------------------------------------------------------------------------

def bench_polars(filepath):
    """Benchmark polars with phase breakdown. Returns (timings, describe_df) or None."""
    try:
        import polars as pl
    except ImportError:
        print("  polars not installed, skipping")
        return None

    # Warmup
    _ = pl.read_csv(filepath).describe()

    # Phase 1: read_csv
    t0 = time.perf_counter()
    df = pl.read_csv(filepath)
    t_read = time.perf_counter() - t0

    # Phase 2: .describe()
    t0 = time.perf_counter()
    desc = df.describe()
    t_describe = time.perf_counter() - t0

    timings = {
        'read_csv': t_read,
        'describe': t_describe,
        'total': t_read + t_describe,
    }
    return timings, desc


# ---------------------------------------------------------------------------
# Precision comparison (f32 vs f64)
# ---------------------------------------------------------------------------

def compare_precision(ea_results, pd_desc):
    """Compare eastat f32 results against pandas f64 results.

    Reports per-stat relative error and max divergence.
    """
    import pandas as pd

    print("\n=== Precision Comparison (eastat f32 vs pandas f64) ===\n")

    stat_map = {
        'mean': 'mean', 'stddev': 'std',
        'min': 'min', 'max': 'max',
        'p25': '25%', 'p50': '50%', 'p75': '75%',
    }

    max_rel_err = 0.0
    max_rel_col = ''
    max_rel_stat = ''

    pd_cols = list(pd_desc.columns)

    for r in ea_results:
        if r['type'] not in ('integer', 'float'):
            continue

        col_name = r['name']
        if col_name not in pd_cols:
            continue

        print(f"  Column: {col_name}")
        print(f"    {'Stat':<8} {'eastat (f32)':>16} {'pandas (f64)':>16} {'rel err':>12}")
        print(f"    {'─'*8} {'─'*16} {'─'*16} {'─'*12}")

        for ea_key, pd_key in stat_map.items():
            ea_val = r.get(ea_key)
            if ea_val is None:
                continue

            try:
                pd_val = float(pd_desc.loc[pd_key, col_name])
            except (KeyError, ValueError):
                continue

            if pd_val == 0:
                rel_err = abs(ea_val - pd_val)
            else:
                rel_err = abs(ea_val - pd_val) / abs(pd_val)

            marker = ''
            if rel_err > 1e-3:
                marker = ' <-- drift'
            elif rel_err > 1e-5:
                marker = ' *'

            if rel_err > max_rel_err:
                max_rel_err = rel_err
                max_rel_col = col_name
                max_rel_stat = ea_key

            print(f"    {ea_key:<8} {ea_val:>16.6g} {pd_val:>16.6g} {rel_err:>12.2e}{marker}")

        print()

    print(f"  Max relative error: {max_rel_err:.2e} ({max_rel_stat} on \"{max_rel_col}\")")
    if max_rel_err > 1e-3:
        print(f"  WARNING: >0.1% divergence detected. f32 precision may be insufficient")
        print(f"  for this data distribution (large values or high variance).")
    elif max_rel_err > 1e-5:
        print(f"  Note: minor f32 rounding visible but within typical tolerance.")
    else:
        print(f"  f32 and f64 results agree to ~6 significant figures.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_test_file():
    """Find a test CSV file."""
    candidates = [
        SCRIPT_DIR / "test_1000000.csv",
        SCRIPT_DIR / "test_10k.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def run_single_benchmark(filepath, show_precision=False, show_inspect=False):
    """Run benchmark on a single file."""
    file_size = filepath.stat().st_size
    size_mb = file_size / (1024**2)
    print(f"\nFile: {filepath.name} ({size_mb:.1f} MB)")
    print("=" * 72)

    # --- Kernel Analysis ---
    if show_inspect:
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
        print("=" * 72)

    # --- Eastat ---
    ea_timings, ea_results = bench_eastat(filepath)

    print("\neastat breakdown:")
    print(f"  scan  (structural extraction):  {ea_timings.get('scan', 0)*1000:7.1f} ms")
    print(f"  layout (row/delim index):       {ea_timings.get('layout', 0)*1000:7.1f} ms")
    print(f"  stats  (parse + reduce):        {ea_timings.get('stats', 0)*1000:7.1f} ms")
    ea_total = ea_timings.get('total', 0)
    print(f"  total:                          {ea_total*1000:7.1f} ms")

    throughput_mbs = size_mb / ea_total if ea_total > 0 else 0
    print(f"  throughput:                     {throughput_mbs:7.0f} MB/s")

    # --- Pandas ---
    pd_result = bench_pandas(filepath)
    pd_timings = None
    pd_desc = None

    if pd_result:
        pd_timings, pd_desc = pd_result
        print(f"\npandas breakdown:")
        print(f"  read_csv (parse -> DataFrame): {pd_timings['read_csv']*1000:7.1f} ms")
        print(f"  .describe() full:              {pd_timings['describe']*1000:7.1f} ms")
        print(f"  total:                         {pd_timings['total']*1000:7.1f} ms")

    # --- Polars ---
    pl_result = bench_polars(filepath)
    pl_timings = None

    if pl_result:
        pl_timings, _ = pl_result
        print(f"\npolars breakdown:")
        print(f"  read_csv:                      {pl_timings['read_csv']*1000:7.1f} ms")
        print(f"  .describe():                   {pl_timings['describe']*1000:7.1f} ms")
        print(f"  total:                         {pl_timings['total']*1000:7.1f} ms")

    # --- Comparison ---
    print("\n" + "─" * 72)

    if ea_total > 0:
        print(f"\nEquivalent work (count/mean/std/min/25%/50%/75%/max):")
        if pd_timings:
            ratio = pd_timings['total'] / ea_total
            print(f"  eastat vs pandas:  {ratio:.1f}x {'faster' if ratio > 1 else 'slower'}")
        if pl_timings:
            ratio = pl_timings['total'] / ea_total
            print(f"  eastat vs polars:  {ratio:.1f}x {'faster' if ratio > 1 else 'slower'}")
        if pd_timings and pl_timings:
            ratio = pd_timings['total'] / pl_timings['total']
            print(f"  polars vs pandas:  {ratio:.1f}x {'faster' if ratio > 1 else 'slower'}")

    print(f"\nNotes: eastat uses f32 SIMD reductions; pandas/polars use f64.")

    # --- Precision comparison ---
    if show_precision and pd_desc is not None:
        compare_precision(ea_results, pd_desc)

    return {
        'file': filepath.name,
        'size_mb': size_mb,
        'eastat_ms': ea_total * 1000,
        'pandas_ms': pd_timings['total'] * 1000 if pd_timings else None,
        'polars_ms': pl_timings['total'] * 1000 if pl_timings else None,
        'throughput_mbs': throughput_mbs,
    }


def run_scaling_test(sizes):
    """Generate and benchmark across multiple file sizes."""
    from generate_test import generate_csv, parse_size

    print("=== Scaling Test ===\n")
    results = []

    for size_str in sizes:
        target = parse_size(size_str)
        output_path = SCRIPT_DIR / f"test_{size_str.lower().strip()}.csv"

        if not output_path.exists():
            print(f"Generating {size_str} test file...")
            generate_csv(str(output_path), 0, target_size=target)

        r = run_single_benchmark(output_path, show_precision=False)
        results.append(r)

    # Summary table
    print("\n" + "=" * 72)
    print("=== Scaling Summary ===\n")
    print(f"  {'Size':>8} {'eastat':>10} {'pandas':>10} {'polars':>10} {'ea MB/s':>10} {'vs pd':>8} {'vs pl':>8}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")

    for r in results:
        ea = f"{r['eastat_ms']:.0f} ms"
        pd_s = f"{r['pandas_ms']:.0f} ms" if r['pandas_ms'] else "n/a"
        pl_s = f"{r['polars_ms']:.0f} ms" if r['polars_ms'] else "n/a"
        tp = f"{r['throughput_mbs']:.0f}"

        if r['pandas_ms'] and r['eastat_ms'] > 0:
            vs_pd = f"{r['pandas_ms'] / r['eastat_ms']:.1f}x"
        else:
            vs_pd = "n/a"

        if r['polars_ms'] and r['eastat_ms'] > 0:
            vs_pl = f"{r['polars_ms'] / r['eastat_ms']:.1f}x"
        else:
            vs_pl = "n/a"

        print(f"  {r['size_mb']:>7.0f}M {ea:>10} {pd_s:>10} {pl_s:>10} {tp:>10} {vs_pd:>8} {vs_pl:>8}")

    print(f"\n  Linear scaling: eastat throughput should stay ~constant across sizes.")
    print(f"  If pandas speedup grows with size, it confirms memory-overhead advantage.")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark eastat vs pandas vs polars',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('file', nargs='?', help='CSV file to benchmark')
    parser.add_argument('--precision', action='store_true',
                        help='Show detailed f32 vs f64 precision comparison')
    parser.add_argument('--inspect', action='store_true',
                        help='Show ea inspect kernel analysis')
    parser.add_argument('--sizes', type=str, default=None,
                        help='Comma-separated sizes for scaling test (e.g., 47MB,500MB,2GB)')

    args = parser.parse_args()

    if args.sizes:
        sizes = [s.strip() for s in args.sizes.split(',')]
        run_scaling_test(sizes)
        return

    if args.file:
        filepath = Path(args.file)
    else:
        filepath = find_test_file()
        if filepath is None:
            print("No test file found. Run generate_test.py first.")
            print("  python generate_test.py --rows=1000000")
            sys.exit(1)

    run_single_benchmark(filepath, show_precision=args.precision,
                         show_inspect=args.inspect)


if __name__ == '__main__':
    main()
