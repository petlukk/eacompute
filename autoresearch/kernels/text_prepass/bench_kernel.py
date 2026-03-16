#!/usr/bin/env python3
"""Compile and benchmark Ea text_prepass_fused kernel. Outputs JSON to stdout.

Benchmarks text_prepass_fused across multiple text sizes.
"""

import json
import os
import subprocess
import sys
import time
import ctypes
import numpy as np
from pathlib import Path

TEXT_SIZES = [1024, 10 * 1024, 100 * 1024, 738 * 1024]
BYTES_PER_ELEM = 4  # read 1 byte + write 3 bytes (flags, lower, boundaries)
NUM_RUNS = 50
WARMUP_RUNS = 10
SEED = 42

U8_PTR = ctypes.POINTER(ctypes.c_uint8)
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


def make_text_data(n, seed):
    """Generate realistic text-like byte data with mixed ASCII classes."""
    rng = np.random.default_rng(seed)
    # Mix of lowercase, uppercase, digits, spaces, punctuation, newlines
    choices = (
        list(range(97, 123))   # a-z
        + list(range(65, 91))  # A-Z
        + list(range(48, 58))  # 0-9
        + [32, 32, 32, 32]     # spaces (weighted)
        + [10, 13, 9]          # newlines, CR, tab
        + [33, 44, 46, 59, 63] # punctuation: ! , . ; ?
    )
    indices = rng.integers(0, len(choices), size=n)
    return np.array([choices[i] for i in indices], dtype=np.uint8)


def bench_at_size(ea_func, ref_func, n):
    text = make_text_data(n, SEED)

    ea_flags = np.zeros(n, dtype=np.uint8)
    ea_lower = np.zeros(n, dtype=np.uint8)
    ea_bound = np.zeros(n, dtype=np.uint8)

    ref_flags = np.zeros(n, dtype=np.uint8)
    ref_lower = np.zeros(n, dtype=np.uint8)
    ref_bound = np.zeros(n, dtype=np.uint8)

    tp = text.ctypes.data_as(U8_PTR)

    ea_func(tp,
            ea_flags.ctypes.data_as(U8_PTR),
            ea_lower.ctypes.data_as(U8_PTR),
            ea_bound.ctypes.data_as(U8_PTR),
            I32(n))

    ref_func(tp,
             ref_flags.ctypes.data_as(U8_PTR),
             ref_lower.ctypes.data_as(U8_PTR),
             ref_bound.ctypes.data_as(U8_PTR),
             I32(n))

    if not np.array_equal(ea_flags, ref_flags):
        mismatch = np.where(ea_flags != ref_flags)[0]
        idx = int(mismatch[0])
        return (f"flags mismatch at {idx}: ea={ea_flags[idx]} "
                f"ref={ref_flags[idx]} byte={text[idx]} (N={n})")

    if not np.array_equal(ea_lower, ref_lower):
        mismatch = np.where(ea_lower != ref_lower)[0]
        idx = int(mismatch[0])
        return (f"lower mismatch at {idx}: ea={ea_lower[idx]} "
                f"ref={ref_lower[idx]} byte={text[idx]} (N={n})")

    if not np.array_equal(ea_bound, ref_bound):
        mismatch = np.where(ea_bound != ref_bound)[0]
        idx = int(mismatch[0])
        return (f"boundary mismatch at {idx}: ea={ea_bound[idx]} "
                f"ref={ref_bound[idx]} byte={text[idx]} (N={n})")

    # Re-allocate for benchmark to avoid warm-cache bias on output arrays
    ea_flags = np.zeros(n, dtype=np.uint8)
    ea_lower = np.zeros(n, dtype=np.uint8)
    ea_bound = np.zeros(n, dtype=np.uint8)

    fp = ea_flags.ctypes.data_as(U8_PTR)
    lp = ea_lower.ctypes.data_as(U8_PTR)
    bp = ea_bound.ctypes.data_as(U8_PTR)

    for _ in range(WARMUP_RUNS):
        ea_func(tp, fp, lp, bp, I32(n))

    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        ea_func(tp, fp, lp, bp, I32(n))
        times.append(time.perf_counter() - start)

    times.sort()
    median_us = times[len(times) // 2] * 1e6
    min_us = times[0] * 1e6
    return (round(median_us, 1), round(min_us, 1))


def main():
    if len(sys.argv) < 3:
        print("Usage: bench_kernel.py <kernel.ea> <reference.so>",
              file=sys.stderr)
        sys.exit(1)

    kernel_path = Path(sys.argv[1]).resolve()
    ref_so_path = Path(sys.argv[2]).resolve()
    ea_binary = str(Path(
        os.environ.get("EA_BINARY", "./target/release/ea")).resolve())

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
        ea_func = ea_lib.text_prepass_fused
        ea_func.argtypes = [U8_PTR, U8_PTR, U8_PTR, U8_PTR, I32]
        ea_func.restype = None

        ref_func = ref_lib.text_prepass_fused_ref
        ref_func.argtypes = [U8_PTR, U8_PTR, U8_PTR, U8_PTR, I32]
        ref_func.restype = None
    except AttributeError as e:
        output(False, error=f"symbol: {e}")

    breakdown = {}
    all_medians = []

    for n in TEXT_SIZES:
        label = f"N={n}"
        result = bench_at_size(ea_func, ref_func, n)

        if isinstance(result, str):
            output(False, error=result)

        median_us, min_us = result
        total_bytes = n * BYTES_PER_ELEM
        gbs = total_bytes / (median_us / 1e6) / 1e9
        breakdown[label] = {"median_us": median_us, "min_us": min_us, "gbs": round(gbs, 1)}
        all_medians.append(median_us)
        print(f"  {label}: {median_us} us median, {min_us} us min  |  {gbs:.1f} GB/s",
              file=sys.stderr)

    # Primary metric: largest size (Pride and Prejudice scale)
    largest_label = f"N={TEXT_SIZES[-1]}"
    aggregate_median = breakdown[largest_label]["median_us"]
    aggregate_min = breakdown[largest_label]["min_us"]
    loc = count_loc(kernel_path)

    output(True, time_us=aggregate_median, min_us=aggregate_min, loc=loc,
           breakdown=breakdown)


if __name__ == "__main__":
    main()
