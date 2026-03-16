#!/usr/bin/env python3
"""Compile and benchmark an Ea Sobel edge detection kernel. Outputs JSON to stdout.

Runs across multiple image sizes (512x512, 1024x1024, 2048x2048) to prevent
overfitting to a single cache behavior. The reported time_us is the
median across all sizes.
"""

import json
import os
import subprocess
import sys
import time
import ctypes
import numpy as np
from pathlib import Path

IMAGE_SIZES = [(512, 512), (1024, 1024), (2048, 2048)]
# Memory traffic: read f32 input + write f32 output per pixel; stencil reuse from cache
BYTES_PER_PIXEL = 8
NUM_RUNS = 50
WARMUP_RUNS = 10
SEED = 42

FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
FLOAT_PTR_MUT = ctypes.POINTER(ctypes.c_float)
I32 = ctypes.c_int32
SOBEL_ARGTYPES = [FLOAT_PTR, FLOAT_PTR_MUT, I32, I32]


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


def sobel_numpy(img, w, h):
    """Reference Sobel in pure NumPy for correctness check."""
    img2d = img.reshape(h, w)
    out = np.zeros_like(img2d)
    gx = (img2d[:-2, 2:] - img2d[:-2, :-2]
          + 2.0 * (img2d[1:-1, 2:] - img2d[1:-1, :-2])
          + img2d[2:, 2:] - img2d[2:, :-2])
    gy = (img2d[2:, :-2] - img2d[:-2, :-2]
          + 2.0 * (img2d[2:, 1:-1] - img2d[:-2, 1:-1])
          + img2d[2:, 2:] - img2d[:-2, 2:])
    out[1:-1, 1:-1] = np.abs(gx) + np.abs(gy)
    return out.ravel()


def bench_at_size(ea_func, ref_func, w, h):
    """Benchmark at a specific image size. Returns (median_us, min_us) or error string."""
    np.random.seed(SEED)
    total = w * h
    img = np.random.uniform(0, 1, total).astype(np.float32)
    ea_out = np.zeros(total, dtype=np.float32)
    ref_out = np.zeros(total, dtype=np.float32)

    ip = img.ctypes.data_as(FLOAT_PTR)
    ea_op = ea_out.ctypes.data_as(FLOAT_PTR_MUT)
    ref_op = ref_out.ctypes.data_as(FLOAT_PTR_MUT)
    wi = I32(w)
    hi = I32(h)

    # Correctness: compare Ea against numpy reference
    ref_func(ip, ref_op, wi, hi)
    ea_func(ip, ea_op, wi, hi)

    np_out = sobel_numpy(img, w, h)

    # Compare interior only (border is zero)
    ea_2d = ea_out.reshape(h, w)[1:-1, 1:-1].ravel()
    np_2d = np_out.reshape(h, w)[1:-1, 1:-1].ravel()

    if not np.allclose(ea_2d, np_2d, rtol=1e-4, atol=1e-5):
        diff = np.abs(ea_2d - np_2d)
        max_idx = np.argmax(diff)
        return f"correctness: max diff {diff[max_idx]:.6f} at index {max_idx} ({w}x{h})"

    # Benchmark
    for _ in range(WARMUP_RUNS):
        ea_func(ip, ea_op, wi, hi)

    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        ea_func(ip, ea_op, wi, hi)
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
        ea_func = ea_lib.sobel
        ea_func.argtypes = SOBEL_ARGTYPES
        ea_func.restype = None

        ref_func = ref_lib.sobel_ref
        ref_func.argtypes = SOBEL_ARGTYPES
        ref_func.restype = None
    except AttributeError as e:
        output(False, error=f"symbol: {e}")

    # --- Benchmark across all image sizes ---
    breakdown = {}
    all_medians = []

    for w, h in IMAGE_SIZES:
        label = f"{w}x{h}"
        result = bench_at_size(ea_func, ref_func, w, h)

        if isinstance(result, str):
            output(False, error=result)

        median_us, min_us = result
        total_bytes = w * h * BYTES_PER_PIXEL
        gbs = total_bytes / (median_us / 1e6) / 1e9
        breakdown[label] = {"median_us": median_us, "min_us": min_us, "gbs": round(gbs, 1)}
        all_medians.append(median_us)
        print(f"  {label}: {median_us} us median, {min_us} us min  |  {gbs:.1f} GB/s", file=sys.stderr)

    # Primary metric: largest size (real-world, exceeds cache)
    largest_label = f"{IMAGE_SIZES[-1][0]}x{IMAGE_SIZES[-1][1]}"
    aggregate_median = breakdown[largest_label]["median_us"]
    aggregate_min = breakdown[largest_label]["min_us"]
    loc = count_loc(kernel_path)

    output(True, time_us=aggregate_median, min_us=aggregate_min, loc=loc,
           breakdown=breakdown)


if __name__ == "__main__":
    main()
