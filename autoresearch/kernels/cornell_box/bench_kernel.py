#!/usr/bin/env python3
"""Compile and benchmark an Ea Cornell Box ray tracer. Outputs JSON to stdout.

Runs across multiple resolutions (128x128, 256x256, 512x512).
The reported time_us is the median across all sizes.
"""

import json
import os
import subprocess
import sys
import time
import ctypes
import numpy as np
from pathlib import Path

RESOLUTIONS = [(128, 128), (256, 256), (512, 512)]
# Memory traffic: write RGB output; geometry is tiny and cached
NUM_RUNS = 20
WARMUP_RUNS = 5

F32_PTR = ctypes.POINTER(ctypes.c_float)
I32 = ctypes.c_int32
RENDER_ARGTYPES = [F32_PTR, I32, I32]


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


def bench_at_size(ea_func, ref_func, w, h):
    total = w * h * 3
    ea_out = np.zeros(total, dtype=np.float32)
    ref_out = np.zeros(total, dtype=np.float32)

    ea_func(ea_out.ctypes.data_as(F32_PTR), I32(w), I32(h))
    ref_func(ref_out.ctypes.data_as(F32_PTR), I32(w), I32(h))

    # Ray tracers have edge pixels where compiler float rounding causes
    # different hit/miss decisions. Allow up to 0.5% of pixels to differ.
    num_pixels = w * h
    ea_rgb = ea_out.reshape(num_pixels, 3)
    ref_rgb = ref_out.reshape(num_pixels, 3)
    pixel_diff = np.max(np.abs(ea_rgb - ref_rgb), axis=1)
    bad_pixels = np.sum(pixel_diff > 1e-3)
    bad_ratio = bad_pixels / num_pixels
    if bad_ratio > 0.005:
        worst = np.argmax(pixel_diff)
        return (f"correctness: {bad_pixels}/{num_pixels} pixels differ ({bad_ratio:.2%}), "
                f"worst at pixel {worst} (ea={ea_rgb[worst]} ref={ref_rgb[worst]}, {w}x{h})")

    buf = np.zeros(total, dtype=np.float32)
    bp = buf.ctypes.data_as(F32_PTR)

    for _ in range(WARMUP_RUNS):
        ea_func(bp, I32(w), I32(h))

    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        ea_func(bp, I32(w), I32(h))
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
        ea_func = ea_lib.render
        ea_func.argtypes = RENDER_ARGTYPES
        ea_func.restype = None

        ref_func = ref_lib.render_ref
        ref_func.argtypes = RENDER_ARGTYPES
        ref_func.restype = None
    except AttributeError as e:
        output(False, error=f"symbol: {e}")

    breakdown = {}
    all_medians = []

    for w, h in RESOLUTIONS:
        label = f"{w}x{h}"
        result = bench_at_size(ea_func, ref_func, w, h)

        if isinstance(result, str):
            output(False, error=result)

        median_us, min_us = result
        total_bytes = w * h * 3 * 4
        gbs = total_bytes / (median_us / 1e6) / 1e9
        breakdown[label] = {"median_us": median_us, "min_us": min_us, "gbs": round(gbs, 1)}
        all_medians.append(median_us)
        print(f"  {label}: {median_us} us median, {min_us} us min  |  {gbs:.1f} GB/s", file=sys.stderr)

    # Primary metric: largest size (real-world, exceeds cache)
    largest_label = f"{RESOLUTIONS[-1][0]}x{RESOLUTIONS[-1][1]}"
    aggregate_median = breakdown[largest_label]["median_us"]
    aggregate_min = breakdown[largest_label]["min_us"]
    loc = count_loc(kernel_path)

    output(True, time_us=aggregate_median, min_us=aggregate_min, loc=loc,
           breakdown=breakdown)


if __name__ == "__main__":
    main()
