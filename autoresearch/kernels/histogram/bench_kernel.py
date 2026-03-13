#!/usr/bin/env python3
"""Compile and benchmark an Ea histogram kernel. Outputs one JSON line to stdout.

Runs across multiple dataset sizes (64K, 256K, 1M, 16M) to prevent
overfitting to a single cache/memory behavior. The reported time_us
is the median across all dataset sizes.
"""

import json
import os
import subprocess
import sys
import time
import ctypes
import numpy as np
from pathlib import Path

DATASET_SIZES = [64_000, 256_000, 1_000_000, 16_000_000]
NUM_RUNS = 100
WARMUP_RUNS = 10
SEED = 42

I32_PTR = ctypes.POINTER(ctypes.c_int32)
I32 = ctypes.c_int32
HIST_ARGTYPES = [I32_PTR, I32_PTR, I32]


def count_loc(path):
    """Count non-blank, non-comment lines."""
    count = 0
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("//"):
                count += 1
    return count


def output(correct, time_us=None, min_us=None, loc=None, error=None,
           breakdown=None):
    """Print JSON result and exit."""
    result = {
        "correct": correct,
        "time_us": time_us,
        "min_us": min_us,
        "loc": loc,
        "error": error,
    }
    if breakdown:
        result["breakdown"] = breakdown
    print(json.dumps(result))
    sys.exit(0)


def bench_at_size(ea_func, ref_func, size):
    """Benchmark at a specific array size. Returns (median_us, min_us) or error string."""
    np.random.seed(SEED)
    data = np.random.randint(0, 256, size=size).astype(np.int32)

    data_ptr = data.ctypes.data_as(I32_PTR)
    n = I32(size)

    # Correctness check
    ref_hist = np.zeros(256, dtype=np.int32)
    ref_hist_ptr = ref_hist.ctypes.data_as(I32_PTR)
    ref_func(data_ptr, ref_hist_ptr, n)

    ea_hist = np.zeros(256, dtype=np.int32)
    ea_hist_ptr = ea_hist.ctypes.data_as(I32_PTR)
    ea_func(data_ptr, ea_hist_ptr, n)

    expected = np.bincount(data, minlength=256).astype(np.int32)

    if not np.array_equal(ea_hist, expected):
        diff = np.abs(ea_hist.astype(np.int64) - expected.astype(np.int64))
        max_idx = np.argmax(diff)
        return f"correctness: ea hist[{max_idx}]={ea_hist[max_idx]} expected={expected[max_idx]} (size={size})"

    if not np.array_equal(ref_hist, expected):
        diff = np.abs(ref_hist.astype(np.int64) - expected.astype(np.int64))
        max_idx = np.argmax(diff)
        return f"correctness(ref): ref hist[{max_idx}]={ref_hist[max_idx]} expected={expected[max_idx]} (size={size})"

    # Benchmark
    for _ in range(WARMUP_RUNS):
        warmup_hist = np.zeros(256, dtype=np.int32)
        warmup_ptr = warmup_hist.ctypes.data_as(I32_PTR)
        ea_func(data_ptr, warmup_ptr, n)

    times = []
    for _ in range(NUM_RUNS):
        run_hist = np.zeros(256, dtype=np.int32)
        run_ptr = run_hist.ctypes.data_as(I32_PTR)
        start = time.perf_counter()
        ea_func(data_ptr, run_ptr, n)
        times.append(time.perf_counter() - start)

    times.sort()
    median_us = times[len(times) // 2] * 1e6
    min_us = times[0] * 1e6
    return (round(median_us, 1), round(min_us, 1))


def main():
    if len(sys.argv) < 3:
        print("Usage: bench_kernel.py <kernel.ea> <reference.so>", file=sys.stderr)
        sys.exit(1)

    kernel_path = Path(sys.argv[1]).resolve()
    ref_so_path = Path(sys.argv[2]).resolve()
    ea_binary = str(Path(os.environ.get("EA_BINARY", "./target/release/ea")).resolve())

    # --- Compile ---
    kernel_dir = kernel_path.parent
    so_name = kernel_path.stem + ".so"
    so_path = kernel_dir / so_name

    # Remove stale .so to avoid benchmarking old code on compile failure
    for stale in [so_path, Path(so_name), kernel_path.with_suffix(".so")]:
        if stale.exists():
            stale.unlink()

    result = subprocess.run(
        [ea_binary, str(kernel_path), "--lib", "--opt-level=3"],
        capture_output=True, text=True, cwd=str(kernel_dir),
    )
    if result.returncode != 0:
        error_msg = result.stderr.strip() or result.stdout.strip()
        output(False, error=f"compile: {error_msg}")

    if not so_path.exists():
        output(False, error="compile: .so not found after compilation")

    # --- Load libraries ---
    try:
        ea_lib = ctypes.CDLL(str(so_path.resolve()))
        ref_lib = ctypes.CDLL(str(ref_so_path.resolve()))
    except OSError as e:
        output(False, error=f"load: {e}")

    try:
        ea_func = ea_lib.histogram_i32
        ea_func.argtypes = HIST_ARGTYPES
        ea_func.restype = None

        ref_func = ref_lib.histogram_i32_c
        ref_func.argtypes = HIST_ARGTYPES
        ref_func.restype = None
    except AttributeError as e:
        output(False, error=f"symbol: {e}")

    # --- Benchmark across all dataset sizes ---
    breakdown = {}
    all_medians = []

    for size in DATASET_SIZES:
        label = f"{size // 1000}K" if size < 1_000_000 else f"{size // 1_000_000}M"
        result = bench_at_size(ea_func, ref_func, size)

        if isinstance(result, str):
            output(False, error=result)

        median_us, min_us = result
        breakdown[label] = {"median_us": median_us, "min_us": min_us}
        all_medians.append(median_us)
        print(f"  {label}: {median_us} us median, {min_us} us min", file=sys.stderr)

    # Aggregate: median of medians across all sizes
    all_medians.sort()
    aggregate_median = all_medians[len(all_medians) // 2]
    aggregate_min = min(m["min_us"] for m in breakdown.values())
    loc = count_loc(kernel_path)

    output(True, time_us=aggregate_median, min_us=aggregate_min, loc=loc,
           breakdown=breakdown)


if __name__ == "__main__":
    main()
