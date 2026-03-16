#!/usr/bin/env python3
"""Compile and benchmark Ea frame_stats kernel. Outputs JSON to stdout.

Benchmarks frame_stats (single-pass min/max/sum) across multiple array sizes.
"""

import json
import os
import subprocess
import sys
import time
import ctypes
import numpy as np
from pathlib import Path

# Real image sizes: 512², 1024², 1600² (HST), 2048², 4096²
DATASET_SIZES = [512 * 512, 1024 * 1024, 1600 * 1600, 2048 * 2048, 4096 * 4096]
BYTES_PER_ELEM = 4  # read data only, 3 scalar outputs negligible
NUM_RUNS = 50
WARMUP_RUNS = 10
SEED = 42

F32_PTR = ctypes.POINTER(ctypes.c_float)
I32 = ctypes.c_int32


def count_loc(path):
    count = 0
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("//"):
                count += 1
    return count


def output(correct, time_us=None, min_us=None, loc=None, error=None,
           breakdown=None):
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


def bench_at_size(ea_func, ref_func, n):
    rng = np.random.default_rng(SEED)
    data = rng.uniform(-100, 100, n).astype(np.float32)

    dp = data.ctypes.data_as(F32_PTR)

    ea_min = np.zeros(1, dtype=np.float32)
    ea_max = np.zeros(1, dtype=np.float32)
    ea_sum = np.zeros(1, dtype=np.float32)
    ref_min = np.zeros(1, dtype=np.float32)
    ref_max = np.zeros(1, dtype=np.float32)
    ref_sum = np.zeros(1, dtype=np.float32)

    ea_func(dp, I32(n),
            ea_min.ctypes.data_as(F32_PTR),
            ea_max.ctypes.data_as(F32_PTR),
            ea_sum.ctypes.data_as(F32_PTR))
    ref_func(dp, I32(n),
             ref_min.ctypes.data_as(F32_PTR),
             ref_max.ctypes.data_as(F32_PTR),
             ref_sum.ctypes.data_as(F32_PTR))

    if abs(ea_min[0] - ref_min[0]) > 1e-3:
        return f"min mismatch: ea={ea_min[0]} ref={ref_min[0]} (N={n})"
    if abs(ea_max[0] - ref_max[0]) > 1e-3:
        return f"max mismatch: ea={ea_max[0]} ref={ref_max[0]} (N={n})"
    # Absolute tolerance scaled by N — float32 accumulation error grows with sqrt(N)
    sum_atol = max(1.0, abs(ref_sum[0]) * 1e-3, n ** 0.5 * 1e-3)
    if abs(ea_sum[0] - ref_sum[0]) > sum_atol:
        return f"sum mismatch: ea={ea_sum[0]} ref={ref_sum[0]} diff={abs(ea_sum[0]-ref_sum[0]):.2e} (N={n})"

    for _ in range(WARMUP_RUNS):
        ea_func(dp, I32(n),
                ea_min.ctypes.data_as(F32_PTR),
                ea_max.ctypes.data_as(F32_PTR),
                ea_sum.ctypes.data_as(F32_PTR))

    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        ea_func(dp, I32(n),
                ea_min.ctypes.data_as(F32_PTR),
                ea_max.ctypes.data_as(F32_PTR),
                ea_sum.ctypes.data_as(F32_PTR))
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

    kernel_dir = kernel_path.parent
    so_name = kernel_path.stem + ".so"
    so_path = kernel_dir / so_name

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

    try:
        ea_lib = ctypes.CDLL(str(so_path.resolve()))
        ref_lib = ctypes.CDLL(str(ref_so_path.resolve()))
    except OSError as e:
        output(False, error=f"load: {e}")

    try:
        ea_func = ea_lib.frame_stats
        ea_func.argtypes = [F32_PTR, I32, F32_PTR, F32_PTR, F32_PTR]
        ea_func.restype = None

        ref_func = ref_lib.frame_stats_ref
        ref_func.argtypes = [F32_PTR, I32, F32_PTR, F32_PTR, F32_PTR]
        ref_func.restype = None
    except AttributeError as e:
        output(False, error=f"symbol: {e}")

    breakdown = {}
    all_medians = []

    for n in DATASET_SIZES:
        label = f"N={n}"
        result = bench_at_size(ea_func, ref_func, n)

        if isinstance(result, str):
            output(False, error=result)

        median_us, min_us = result
        total_bytes = n * BYTES_PER_ELEM
        gbs = total_bytes / (median_us / 1e6) / 1e9
        breakdown[label] = {"median_us": median_us, "min_us": min_us, "gbs": round(gbs, 1)}
        all_medians.append(median_us)
        print(f"  {label}: {median_us} us median, {min_us} us min  |  {gbs:.1f} GB/s", file=sys.stderr)

    # Primary metric: largest size (real-world, exceeds cache)
    largest_label = f"N={DATASET_SIZES[-1]}"
    aggregate_median = breakdown[largest_label]["median_us"]
    aggregate_min = breakdown[largest_label]["min_us"]
    loc = count_loc(kernel_path)

    output(True, time_us=aggregate_median, min_us=aggregate_min, loc=loc,
           breakdown=breakdown)


if __name__ == "__main__":
    main()
