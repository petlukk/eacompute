#!/usr/bin/env python3
"""Compile and benchmark Ea preprocess_fused kernel. Outputs JSON to stdout.

Benchmarks preprocess_fused (normalize + standardize + clip) across MNIST-scale sizes.
"""

import json
import os
import subprocess
import sys
import time
import ctypes
import numpy as np
from pathlib import Path

SIZES = [28 * 28 * 100, 28 * 28 * 1000, 28 * 28 * 10000, 28 * 28 * 60000]
NUM_RUNS = 50
WARMUP_RUNS = 10
SEED = 42

# MNIST normalization constants
SCALE = 1.0 / 255.0
MEAN = 0.1307
INV_STD = 1.0 / 0.3081

F32_PTR = ctypes.POINTER(ctypes.c_float)
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


def bench_at_size(ea_func, ref_func, n):
    rng = np.random.default_rng(SEED)
    # Simulate uint8 pixel values cast to f32 (range 0-255)
    input_data = rng.integers(0, 256, n).astype(np.float32)
    output_ea = np.zeros(n, dtype=np.float32)
    output_ref = np.zeros(n, dtype=np.float32)

    inp = input_data.ctypes.data_as(F32_PTR)
    out_ea = output_ea.ctypes.data_as(F32_PTR)
    out_ref = output_ref.ctypes.data_as(F32_PTR)

    ea_func(inp, out_ea, I32(n), F32(SCALE), F32(MEAN), F32(INV_STD))
    ref_func(inp, out_ref, I32(n), F32(SCALE), F32(MEAN), F32(INV_STD))

    max_diff = np.max(np.abs(output_ea - output_ref))
    if max_diff > 1e-5:
        return f"correctness: max_diff={max_diff} (N={n})"

    for _ in range(WARMUP_RUNS):
        ea_func(inp, out_ea, I32(n), F32(SCALE), F32(MEAN), F32(INV_STD))

    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        ea_func(inp, out_ea, I32(n), F32(SCALE), F32(MEAN), F32(INV_STD))
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
        ea_func = ea_lib.preprocess_fused
        ea_func.argtypes = [F32_PTR, F32_PTR, I32, F32, F32, F32]
        ea_func.restype = None

        ref_func = ref_lib.preprocess_fused_ref
        ref_func.argtypes = [F32_PTR, F32_PTR, I32, F32, F32, F32]
        ref_func.restype = None
    except AttributeError as e:
        output(False, error=f"symbol: {e}")

    breakdown = {}

    for n in SIZES:
        label = f"N={n}"
        result = bench_at_size(ea_func, ref_func, n)

        if isinstance(result, str):
            output(False, error=result)

        median_us, min_us = result
        breakdown[label] = {"median_us": median_us, "min_us": min_us}
        print(f"  {label}: {median_us} us median, {min_us} us min", file=sys.stderr)

    # Primary metric: largest size (full MNIST dataset scale)
    largest_label = f"N={SIZES[-1]}"
    aggregate_median = breakdown[largest_label]["median_us"]
    aggregate_min = breakdown[largest_label]["min_us"]
    loc = count_loc(kernel_path)

    output(True, time_us=aggregate_median, min_us=aggregate_min, loc=loc,
           breakdown=breakdown)


if __name__ == "__main__":
    main()
