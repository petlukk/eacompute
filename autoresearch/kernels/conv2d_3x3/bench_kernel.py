#!/usr/bin/env python3
"""Compile and benchmark an Ea 3x3 int8 convolution kernel. Outputs JSON to stdout.

Benchmarks the i32 accumulator variant (conv2d_3x3_u8i8_safe) across
multiple spatial sizes with fixed C_in=64.
"""

import json
import os
import subprocess
import sys
import time
import ctypes
import numpy as np
from pathlib import Path

# (H, W, C_in) — C_in must be multiple of 32
CONV_SIZES = [(32, 32, 64), (64, 64, 64), (128, 128, 64)]
NUM_RUNS = 100
WARMUP_RUNS = 20
SEED = 42

U8_PTR = ctypes.POINTER(ctypes.c_uint8)
I8_PTR = ctypes.POINTER(ctypes.c_int8)
I32_PTR = ctypes.POINTER(ctypes.c_int32)
I32 = ctypes.c_int32
CONV_ARGTYPES = [U8_PTR, I8_PTR, I32_PTR, I32, I32, I32]


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


def conv2d_numpy(src, wt, H, W, C_in):
    """Reference conv in numpy for correctness."""
    stride = (W + 2) * C_in
    dst = np.zeros(H * W, dtype=np.int32)
    for row in range(H):
        for col in range(W):
            acc = np.int32(0)
            for dr in range(3):
                for dc in range(3):
                    src_off = (row + dr) * stride + (col + dc) * C_in
                    wt_off = (dr * 3 + dc) * C_in
                    s = src[src_off:src_off + C_in].astype(np.int32)
                    w = wt[wt_off:wt_off + C_in].astype(np.int32)
                    acc += np.sum(s * w)
            dst[row * W + col] = acc
    return dst


def bench_at_size(ea_func, ref_func, H, W, C_in):
    np.random.seed(SEED)
    padded_h = H + 2
    padded_w = W + 2
    src = np.random.randint(0, 256, padded_h * padded_w * C_in, dtype=np.uint8)
    wt = np.random.randint(-128, 128, 9 * C_in, dtype=np.int8)
    ea_dst = np.zeros(H * W, dtype=np.int32)
    ref_dst = np.zeros(H * W, dtype=np.int32)

    sp = src.ctypes.data_as(U8_PTR)
    wp = wt.ctypes.data_as(I8_PTR)
    ea_dp = ea_dst.ctypes.data_as(I32_PTR)
    ref_dp = ref_dst.ctypes.data_as(I32_PTR)

    ref_func(sp, wp, ref_dp, I32(H), I32(W), I32(C_in))
    ea_func(sp, wp, ea_dp, I32(H), I32(W), I32(C_in))

    if not np.array_equal(ea_dst, ref_dst):
        diff = np.abs(ea_dst.astype(np.int64) - ref_dst.astype(np.int64))
        max_idx = np.argmax(diff)
        return (f"correctness: ea={ea_dst[max_idx]} ref={ref_dst[max_idx]} "
                f"at index {max_idx} ({H}x{W}xC{C_in})")

    for _ in range(WARMUP_RUNS):
        ea_func(sp, wp, ea_dp, I32(H), I32(W), I32(C_in))

    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        ea_func(sp, wp, ea_dp, I32(H), I32(W), I32(C_in))
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
        ea_func = ea_lib.conv2d_3x3_u8i8_safe
        ea_func.argtypes = CONV_ARGTYPES
        ea_func.restype = None

        ref_func = ref_lib.conv2d_3x3_ref
        ref_func.argtypes = CONV_ARGTYPES
        ref_func.restype = None
    except AttributeError as e:
        output(False, error=f"symbol: {e}")

    breakdown = {}
    all_medians = []

    for H, W, C_in in CONV_SIZES:
        label = f"{H}x{W}xC{C_in}"
        result = bench_at_size(ea_func, ref_func, H, W, C_in)

        if isinstance(result, str):
            output(False, error=result)

        median_us, min_us = result
        breakdown[label] = {"median_us": median_us, "min_us": min_us}
        all_medians.append(median_us)
        print(f"  {label}: {median_us} us median, {min_us} us min", file=sys.stderr)

    all_medians.sort()
    aggregate_median = all_medians[len(all_medians) // 2]
    aggregate_min = min(m["min_us"] for m in breakdown.values())
    loc = count_loc(kernel_path)

    output(True, time_us=aggregate_median, min_us=aggregate_min, loc=loc,
           breakdown=breakdown)


if __name__ == "__main__":
    main()
