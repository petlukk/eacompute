#!/usr/bin/env python3
"""Compile and benchmark an Eä FMA kernel. Outputs one JSON line to stdout."""

import json
import os
import subprocess
import sys
import time
import ctypes
import numpy as np
from pathlib import Path

ARRAY_SIZE = 1_000_000
NUM_RUNS = 100
WARMUP_RUNS = 10
SEED = 42

FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
I32 = ctypes.c_int32
FMA_ARGTYPES = [FLOAT_PTR, FLOAT_PTR, FLOAT_PTR, FLOAT_PTR, I32]


def count_loc(path):
    """Count non-blank, non-comment lines."""
    count = 0
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("//"):
                count += 1
    return count


def output(correct, time_us=None, min_us=None, loc=None, error=None):
    """Print JSON result and exit."""
    print(json.dumps({
        "correct": correct,
        "time_us": time_us,
        "min_us": min_us,
        "loc": loc,
        "error": error,
    }))
    sys.exit(0)



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

    # Find the compiled .so (ea places it in cwd)
    if not so_path.exists():
        output(False, error="compile: .so not found after compilation")

    # --- Load libraries ---
    try:
        ea_lib = ctypes.CDLL(str(so_path.resolve()))
        ref_lib = ctypes.CDLL(str(ref_so_path.resolve()))
    except OSError as e:
        output(False, error=f"load: {e}")

    try:
        ea_func = ea_lib.fma_kernel_f32x8
        ea_func.argtypes = FMA_ARGTYPES
        ea_func.restype = None

        ref_func = ref_lib.fma_kernel_f32x8_c
        ref_func.argtypes = FMA_ARGTYPES
        ref_func.restype = None
    except AttributeError as e:
        output(False, error=f"symbol: {e}")

    # --- Prepare data ---
    np.random.seed(SEED)
    a = np.random.uniform(-1, 1, ARRAY_SIZE).astype(np.float32)
    b = np.random.uniform(-1, 1, ARRAY_SIZE).astype(np.float32)
    c = np.random.uniform(-1, 1, ARRAY_SIZE).astype(np.float32)
    ea_result = np.zeros(ARRAY_SIZE, dtype=np.float32)
    ref_result = np.zeros(ARRAY_SIZE, dtype=np.float32)

    ap = a.ctypes.data_as(FLOAT_PTR)
    bp = b.ctypes.data_as(FLOAT_PTR)
    cp = c.ctypes.data_as(FLOAT_PTR)
    ea_rp = ea_result.ctypes.data_as(FLOAT_PTR)
    ref_rp = ref_result.ctypes.data_as(FLOAT_PTR)
    n = I32(ARRAY_SIZE)

    # --- Correctness check ---
    ref_func(ap, bp, cp, ref_rp, n)
    ea_func(ap, bp, cp, ea_rp, n)

    if not np.allclose(ea_result, ref_result, rtol=1e-5):
        diff = np.abs(ea_result - ref_result)
        max_idx = np.argmax(diff)
        output(False, error=f"correctness: max diff {diff[max_idx]:.6f} at index {max_idx}")

    # --- Benchmark (no fork needed — correctness already verified) ---
    for _ in range(WARMUP_RUNS):
        ea_func(ap, bp, cp, ea_rp, n)

    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        ea_func(ap, bp, cp, ea_rp, n)
        times.append(time.perf_counter() - start)

    times.sort()
    median_us = times[len(times) // 2] * 1e6
    min_us = times[0] * 1e6
    loc = count_loc(kernel_path)

    output(True, time_us=round(median_us, 1), min_us=round(min_us, 1), loc=loc)


if __name__ == "__main__":
    main()
