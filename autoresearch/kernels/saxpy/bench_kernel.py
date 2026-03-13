#!/usr/bin/env python3
"""Compile and benchmark an Eä SAXPY kernel. Outputs one JSON line to stdout.

SAXPY: y[i] = a * x[i] + y[i]. In-place read-modify-write, bandwidth-bound.
Signature: saxpy_f32x8(a: f32, x: *f32, y: *mut f32, len: i32) -> void
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

FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
I32 = ctypes.c_int32
SAXPY_ARGTYPES = [ctypes.c_float, FLOAT_PTR, FLOAT_PTR, I32]


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
    x = np.random.uniform(-1, 1, size).astype(np.float32)
    y_orig = np.random.uniform(-1, 1, size).astype(np.float32)
    a = np.float32(2.5)

    xp = x.ctypes.data_as(FLOAT_PTR)

    # Correctness check: run reference
    y_ref = y_orig.copy()
    ref_func(ctypes.c_float(a), xp, y_ref.ctypes.data_as(FLOAT_PTR), I32(size))

    # Correctness check: run ea
    y_ea = y_orig.copy()
    ea_func(ctypes.c_float(a), xp, y_ea.ctypes.data_as(FLOAT_PTR), I32(size))

    if not np.allclose(y_ea, y_ref, rtol=1e-5):
        diff = np.abs(y_ea - y_ref)
        max_idx = np.argmax(diff)
        return f"correctness: max diff {diff[max_idx]:.6f} at index {max_idx} (size={size})"

    # Benchmark: fresh y copy each run to avoid accumulation
    y_bench = y_orig.copy()
    yp = y_bench.ctypes.data_as(FLOAT_PTR)

    for _ in range(WARMUP_RUNS):
        np.copyto(y_bench, y_orig)
        ea_func(ctypes.c_float(a), xp, yp, I32(size))

    times = []
    for _ in range(NUM_RUNS):
        np.copyto(y_bench, y_orig)
        start = time.perf_counter()
        ea_func(ctypes.c_float(a), xp, yp, I32(size))
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
        ea_func = ea_lib.saxpy_f32x8
        ea_func.argtypes = SAXPY_ARGTYPES
        ea_func.restype = None

        ref_func = ref_lib.saxpy_f32x8_c
        ref_func.argtypes = SAXPY_ARGTYPES
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
        print(f"  {label}: {median_us} µs median, {min_us} µs min", file=sys.stderr)

    all_medians.sort()
    aggregate_median = all_medians[len(all_medians) // 2]
    aggregate_min = min(m["min_us"] for m in breakdown.values())
    loc = count_loc(kernel_path)

    output(True, time_us=aggregate_median, min_us=aggregate_min, loc=loc,
           breakdown=breakdown)


if __name__ == "__main__":
    main()
