#!/usr/bin/env python3
"""Compile and benchmark an Ea particle life kernel. Outputs JSON to stdout.

Runs across multiple particle counts (500, 1000, 2000) to capture O(N²) scaling.
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

PARTICLE_COUNTS = [500, 1000, 2000]
NUM_RUNS = 10
WARMUP_RUNS = 3
SEED = 42

NUM_TYPES = 6
R_MAX = 80.0
FRICTION = 0.5
DT = 0.5
SIZE = 800.0

F32_PTR = ctypes.POINTER(ctypes.c_float)
I32_PTR = ctypes.POINTER(ctypes.c_int32)
I32 = ctypes.c_int32
F32 = ctypes.c_float

STEP_ARGTYPES = [
    F32_PTR, F32_PTR, F32_PTR, F32_PTR, I32_PTR, F32_PTR,
    I32, I32, F32, F32, F32, F32,
]


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


def make_state(n, rng):
    px = rng.uniform(0, SIZE, n).astype(np.float32)
    py = rng.uniform(0, SIZE, n).astype(np.float32)
    vx = np.zeros(n, dtype=np.float32)
    vy = np.zeros(n, dtype=np.float32)
    types = rng.integers(0, NUM_TYPES, n).astype(np.int32)
    matrix = rng.uniform(-1, 1, NUM_TYPES * NUM_TYPES).astype(np.float32)
    return px, py, vx, vy, types, matrix


def ptr(arr):
    if arr.dtype == np.float32:
        return arr.ctypes.data_as(F32_PTR)
    return arr.ctypes.data_as(I32_PTR)


def call_step(func, px, py, vx, vy, types, matrix, n):
    func(
        ptr(px), ptr(py), ptr(vx), ptr(vy), ptr(types), ptr(matrix),
        I32(n), I32(NUM_TYPES),
        F32(R_MAX), F32(DT), F32(FRICTION), F32(SIZE),
    )


def bench_at_size(ea_func, ref_func, n):
    rng = np.random.default_rng(SEED)

    # Run reference
    rpx, rpy, rvx, rvy, rtypes, rmatrix = make_state(n, rng)
    call_step(ref_func, rpx, rpy, rvx, rvy, rtypes, rmatrix, n)

    # Run Ea with same initial state
    rng2 = np.random.default_rng(SEED)
    epx, epy, evx, evy, etypes, ematrix = make_state(n, rng2)
    call_step(ea_func, epx, epy, evx, evy, etypes, ematrix, n)

    # Correctness: compare positions and velocities after one step
    if not np.allclose(epx, rpx, rtol=1e-4, atol=1e-4):
        diff = np.abs(epx - rpx)
        idx = np.argmax(diff)
        return f"correctness: px diff {diff[idx]:.6f} at index {idx} (N={n})"
    if not np.allclose(epy, rpy, rtol=1e-4, atol=1e-4):
        diff = np.abs(epy - rpy)
        idx = np.argmax(diff)
        return f"correctness: py diff {diff[idx]:.6f} at index {idx} (N={n})"

    # Benchmark: fresh state each run for determinism
    rng3 = np.random.default_rng(SEED + 100)
    bpx, bpy, bvx, bvy, btypes, bmatrix = make_state(n, rng3)

    for _ in range(WARMUP_RUNS):
        # Copy state so warmup doesn't mutate benchmark state
        wpx, wpy = bpx.copy(), bpy.copy()
        wvx, wvy = bvx.copy(), bvy.copy()
        call_step(ea_func, wpx, wpy, wvx, wvy, btypes, bmatrix, n)

    times = []
    for _ in range(NUM_RUNS):
        tpx, tpy = bpx.copy(), bpy.copy()
        tvx, tvy = bvx.copy(), bvy.copy()
        start = time.perf_counter()
        call_step(ea_func, tpx, tpy, tvx, tvy, btypes, bmatrix, n)
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
        ea_func = ea_lib.particle_life_step
        ea_func.argtypes = STEP_ARGTYPES
        ea_func.restype = None

        ref_func = ref_lib.particle_life_step_ref
        ref_func.argtypes = STEP_ARGTYPES
        ref_func.restype = None
    except AttributeError as e:
        output(False, error=f"symbol: {e}")

    breakdown = {}
    all_medians = []

    for n in PARTICLE_COUNTS:
        label = f"N={n}"
        result = bench_at_size(ea_func, ref_func, n)

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
