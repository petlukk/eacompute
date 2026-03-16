#!/usr/bin/env python3
"""Compile and benchmark Ea masked scatter-add kernel. Outputs JSON to stdout.

Benchmarks scatter_add across multiple input sizes.
"""

import json
import os
import subprocess
import sys
import time
import ctypes
import numpy as np
from pathlib import Path

INPUT_SIZES = [64000, 256000, 1000000, 4000000]
OUTPUT_SIZE = 10000
BYTES_PER_ELEM = 16  # read value 4B + read index 4B + read mask 4B + read/write output 4B
NUM_RUNS = 50
WARMUP_RUNS = 10
SEED = 42
THRESHOLD = 0.5

F32_PTR = ctypes.POINTER(ctypes.c_float)
I32_PTR = ctypes.POINTER(ctypes.c_int32)
I32 = ctypes.c_int32
F32 = ctypes.c_float


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


def make_test_data(n, rng):
    values = rng.uniform(-1.0, 1.0, n).astype(np.float32)
    indices = rng.integers(0, OUTPUT_SIZE, n).astype(np.int32)
    mask = rng.uniform(0.0, 1.0, n).astype(np.float32)
    return values, indices, mask


def bench_at_size(ea_func, ref_func, n):
    rng = np.random.default_rng(SEED)
    values, indices, mask = make_test_data(n, rng)

    vp = values.ctypes.data_as(F32_PTR)
    ip = indices.ctypes.data_as(I32_PTR)
    mp = mask.ctypes.data_as(F32_PTR)

    # Run reference
    ref_output = np.zeros(OUTPUT_SIZE, dtype=np.float32)
    ref_op = ref_output.ctypes.data_as(F32_PTR)
    ref_func(vp, ip, mp, ref_op, I32(n), F32(THRESHOLD))

    # Run Ea kernel for correctness
    ea_output = np.zeros(OUTPUT_SIZE, dtype=np.float32)
    ea_op = ea_output.ctypes.data_as(F32_PTR)
    ea_func(vp, ip, mp, ea_op, I32(n), F32(THRESHOLD))

    # Exact float match (same order of operations)
    if not np.array_equal(ea_output, ref_output):
        max_diff = np.max(np.abs(ea_output - ref_output))
        mismatches = np.sum(ea_output != ref_output)
        return f"correctness: {mismatches} mismatches, max_diff={max_diff} (N={n})"

    # Benchmark
    for _ in range(WARMUP_RUNS):
        ea_output[:] = 0.0
        ea_op = ea_output.ctypes.data_as(F32_PTR)
        ea_func(vp, ip, mp, ea_op, I32(n), F32(THRESHOLD))

    times = []
    for _ in range(NUM_RUNS):
        ea_output[:] = 0.0
        ea_op = ea_output.ctypes.data_as(F32_PTR)
        start = time.perf_counter()
        ea_func(vp, ip, mp, ea_op, I32(n), F32(THRESHOLD))
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
        ea_func = ea_lib.scatter_add
        ea_func.argtypes = [F32_PTR, I32_PTR, F32_PTR, F32_PTR, I32, F32]
        ea_func.restype = None

        ref_func = ref_lib.scatter_add_ref
        ref_func.argtypes = [F32_PTR, I32_PTR, F32_PTR, F32_PTR, I32, F32]
        ref_func.restype = None
    except AttributeError as e:
        output(False, error=f"symbol: {e}")

    breakdown = {}
    all_medians = []

    for n in INPUT_SIZES:
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

    # Primary metric: largest size
    largest_label = f"N={INPUT_SIZES[-1]}"
    aggregate_median = breakdown[largest_label]["median_us"]
    aggregate_min = breakdown[largest_label]["min_us"]
    loc = count_loc(kernel_path)

    output(True, time_us=aggregate_median, min_us=aggregate_min, loc=loc,
           breakdown=breakdown)


if __name__ == "__main__":
    main()
