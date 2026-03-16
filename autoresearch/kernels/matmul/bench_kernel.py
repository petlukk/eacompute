#!/usr/bin/env python3
"""Compile and benchmark an Ea matrix-multiply kernel. Outputs one JSON line to stdout.

Runs across multiple matrix sizes (64x64, 128x128, 256x256) to prevent
overfitting to a single cache behavior. The reported time_us is the
median across all matrix sizes.
"""

import json
import os
import subprocess
import sys
import time
import ctypes
import numpy as np
from pathlib import Path

MATRIX_SIZES = [64, 128, 256]
# Memory traffic: read A, B, write C = 3 * N * N * 4 bytes
NUM_RUNS = 50
WARMUP_RUNS = 10
SEED = 42

FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
FLOAT_PTR_MUT = ctypes.POINTER(ctypes.c_float)
I32 = ctypes.c_int32
MATMUL_ARGTYPES = [FLOAT_PTR, FLOAT_PTR, FLOAT_PTR_MUT, I32]


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


def bench_at_size(ea_func, ref_func, n):
    """Benchmark at a specific matrix dimension. Returns (median_us, min_us) or error string."""
    np.random.seed(SEED)
    total = n * n
    a = np.random.uniform(-1, 1, total).astype(np.float32)
    b = np.random.uniform(-1, 1, total).astype(np.float32)
    ea_c = np.zeros(total, dtype=np.float32)
    ref_c = np.zeros(total, dtype=np.float32)

    ap = a.ctypes.data_as(FLOAT_PTR)
    bp = b.ctypes.data_as(FLOAT_PTR)
    ea_cp = ea_c.ctypes.data_as(FLOAT_PTR_MUT)
    ref_cp = ref_c.ctypes.data_as(FLOAT_PTR_MUT)
    ni = I32(n)

    # Correctness check against numpy
    ref_func(ap, bp, ref_cp, ni)
    ea_func(ap, bp, ea_cp, ni)

    a_mat = a.reshape(n, n)
    b_mat = b.reshape(n, n)
    np_c = (a_mat @ b_mat).ravel()

    if not np.allclose(ea_c, np_c, rtol=1e-3, atol=1e-5):
        diff = np.abs(ea_c - np_c)
        max_idx = np.argmax(diff)
        return f"correctness: max diff {diff[max_idx]:.6f} at index {max_idx} (n={n})"

    # Benchmark
    for _ in range(WARMUP_RUNS):
        ea_func(ap, bp, ea_cp, ni)

    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        ea_func(ap, bp, ea_cp, ni)
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
        ea_func = ea_lib.matmul_f32
        ea_func.argtypes = MATMUL_ARGTYPES
        ea_func.restype = None

        ref_func = ref_lib.matmul_f32_c
        ref_func.argtypes = MATMUL_ARGTYPES
        ref_func.restype = None
    except AttributeError as e:
        output(False, error=f"symbol: {e}")

    # --- Benchmark across all matrix sizes ---
    breakdown = {}
    all_medians = []

    for n in MATRIX_SIZES:
        label = f"{n}x{n}"
        result = bench_at_size(ea_func, ref_func, n)

        if isinstance(result, str):
            output(False, error=result)

        median_us, min_us = result
        total_bytes = 3 * n * n * 4
        gbs = total_bytes / (median_us / 1e6) / 1e9
        breakdown[label] = {"median_us": median_us, "min_us": min_us, "gbs": round(gbs, 1)}
        all_medians.append(median_us)
        print(f"  {label}: {median_us} us median, {min_us} us min  |  {gbs:.1f} GB/s", file=sys.stderr)

    # Aggregate: use 128x128 median as the primary metric
    # Primary metric: largest size (real-world, exceeds cache)
    largest_label = f"{MATRIX_SIZES[-1]}x{MATRIX_SIZES[-1]}"
    aggregate_median = breakdown[largest_label]["median_us"]
    aggregate_min = breakdown[largest_label]["min_us"]
    loc = count_loc(kernel_path)

    output(True, time_us=aggregate_median, min_us=aggregate_min, loc=loc,
           breakdown=breakdown)


if __name__ == "__main__":
    main()
