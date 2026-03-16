#!/usr/bin/env python3
"""Compile and benchmark an Eä batch dot product kernel. Outputs JSON to stdout.

Runs across multiple (dim, n_vecs) configs to prevent overfitting.
The reported time_us is the median across all configs.

Correctness uses rtol=1e-3 due to FP associativity differences.
"""

import json
import os
import subprocess
import sys
import time
import ctypes
import numpy as np
from pathlib import Path

CONFIGS = [
    (384, 1_000),
    (384, 10_000),
    (768, 1_000),
    (768, 10_000),
]
NUM_RUNS = 50
WARMUP_RUNS = 10
SEED = 42

FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
I32 = ctypes.c_int32
DOT_ARGTYPES = [FLOAT_PTR, FLOAT_PTR, I32, I32, FLOAT_PTR]


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


def bench_at_config(ea_func, ref_func, dim, n_vecs):
    np.random.seed(SEED)
    query = np.random.randn(dim).astype(np.float32)
    query /= np.linalg.norm(query)
    db = np.random.randn(n_vecs * dim).astype(np.float32)

    ea_out = np.zeros(n_vecs, dtype=np.float32)
    ref_out = np.zeros(n_vecs, dtype=np.float32)

    qp = query.ctypes.data_as(FLOAT_PTR)
    dp = db.ctypes.data_as(FLOAT_PTR)
    d = I32(dim)
    n = I32(n_vecs)

    ref_func(qp, dp, d, n, ref_out.ctypes.data_as(FLOAT_PTR))
    ea_func(qp, dp, d, n, ea_out.ctypes.data_as(FLOAT_PTR))

    if not np.allclose(ea_out, ref_out, rtol=1e-3, atol=1e-5):
        max_diff = np.max(np.abs(ea_out - ref_out))
        worst = np.argmax(np.abs(ea_out - ref_out))
        return (f"correctness: max_diff={max_diff:.6f} at vec {worst} "
                f"(ea={ea_out[worst]:.6f} ref={ref_out[worst]:.6f}, "
                f"dim={dim} n={n_vecs})")

    for _ in range(WARMUP_RUNS):
        ea_func(qp, dp, d, n, ea_out.ctypes.data_as(FLOAT_PTR))

    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        ea_func(qp, dp, d, n, ea_out.ctypes.data_as(FLOAT_PTR))
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
        ea_func = ea_lib.batch_dot
        ea_func.argtypes = DOT_ARGTYPES
        ea_func.restype = None

        ref_func = ref_lib.batch_dot_ref
        ref_func.argtypes = DOT_ARGTYPES
        ref_func.restype = None
    except AttributeError as e:
        output(False, error=f"symbol: {e}")

    breakdown = {}
    all_medians = []

    for dim, n_vecs in CONFIGS:
        label = f"d{dim}_n{n_vecs // 1000}K"
        result = bench_at_config(ea_func, ref_func, dim, n_vecs)

        if isinstance(result, str):
            output(False, error=result)

        median_us, min_us = result
        breakdown[label] = {"median_us": median_us, "min_us": min_us}
        all_medians.append(median_us)
        print(f"  {label}: {median_us} µs median, {min_us} µs min",
              file=sys.stderr)

    # Primary metric: largest size (real-world, exceeds cache)
    largest_label = f"d{CONFIGS[-1][0]}_n{CONFIGS[-1][1] // 1000}K"
    aggregate_median = breakdown[largest_label]["median_us"]
    aggregate_min = breakdown[largest_label]["min_us"]
    loc = count_loc(kernel_path)

    output(True, time_us=aggregate_median, min_us=aggregate_min, loc=loc,
           breakdown=breakdown)


if __name__ == "__main__":
    main()
