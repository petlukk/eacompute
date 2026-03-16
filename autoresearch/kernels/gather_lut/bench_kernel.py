#!/usr/bin/env python3
"""Compile and benchmark Ea LUT-apply gather kernel. Outputs JSON to stdout.

Benchmarks lut_apply_gather across multiple array sizes.
Tests SIMD gather vs scalar indexing for lookup table application.
"""

import json
import os
import subprocess
import sys
import time
import ctypes
import numpy as np
from pathlib import Path

SIZES = [1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]
BYTES_PER_ELEM = 5  # 1 byte read (u8 data) + 4 bytes write (f32 out); LUT is tiny/cached
NUM_RUNS = 50
WARMUP_RUNS = 10
SEED = 42

U8_PTR = ctypes.POINTER(ctypes.c_uint8)
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
    data = rng.integers(0, 256, n, dtype=np.uint8)
    lut = rng.uniform(-10.0, 10.0, 256).astype(np.float32)
    out_ea = np.zeros(n, dtype=np.float32)
    out_ref = np.zeros(n, dtype=np.float32)

    dp = data.ctypes.data_as(U8_PTR)
    lp = lut.ctypes.data_as(F32_PTR)
    oep = out_ea.ctypes.data_as(F32_PTR)
    orp = out_ref.ctypes.data_as(F32_PTR)

    ea_func(dp, lp, oep, I32(n))
    ref_func(dp, lp, orp, I32(n))

    if not np.array_equal(out_ea, out_ref):
        mismatches = np.where(out_ea != out_ref)[0]
        first = mismatches[0]
        return (f"correctness: ea[{first}]={out_ea[first]} "
                f"ref[{first}]={out_ref[first]} (N={n})")

    for _ in range(WARMUP_RUNS):
        ea_func(dp, lp, oep, I32(n))

    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        ea_func(dp, lp, oep, I32(n))
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
        ea_func = ea_lib.lut_apply_gather
        ea_func.argtypes = [U8_PTR, F32_PTR, F32_PTR, I32]
        ea_func.restype = None

        ref_func = ref_lib.lut_apply_gather_ref
        ref_func.argtypes = [U8_PTR, F32_PTR, F32_PTR, I32]
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
        total_bytes = n * BYTES_PER_ELEM
        gbs = total_bytes / (median_us / 1e6) / 1e9
        breakdown[label] = {"median_us": median_us, "min_us": min_us, "gbs": round(gbs, 1)}
        print(f"  {label}: {median_us} us median, {min_us} us min  |  {gbs:.1f} GB/s", file=sys.stderr)

    largest_label = f"N={SIZES[-1]}"
    aggregate_median = breakdown[largest_label]["median_us"]
    aggregate_min = breakdown[largest_label]["min_us"]
    loc = count_loc(kernel_path)

    output(True, time_us=aggregate_median, min_us=aggregate_min, loc=loc,
           breakdown=breakdown)


if __name__ == "__main__":
    main()
