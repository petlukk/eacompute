#!/usr/bin/env python3
"""
Scale sweep: bandwidth analysis across image sizes.

Measures effective GB/s for each method to find:
  - How close we are to STREAM peak
  - Where batched overtakes single-frame
  - Where stream_store (NT) overtakes regular store

Each size runs in a subprocess (no cross-contamination).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import subprocess
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).parent

N_FRAMES = 16
WARMUP = 5
RUNS = 30

SIZES = [512, 1024, 2048, 4096]

WORKER = '''
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys, time, json
import numpy as np

sys.path.insert(0, "{demo_dir}")
import stack as _stack

size = {size}
n_frames = {n_frames}
n_pixels = size * size
WARMUP = {warmup}
RUNS = {runs}

rng = np.random.RandomState(42)
flat_frames = [rng.rand(n_pixels).astype(np.float32) for _ in range(n_frames)]
acc = np.zeros(n_pixels, dtype=np.float32)
out = np.zeros(n_pixels, dtype=np.float32)
factor = np.float32(1.0 / n_frames)

def bench(func):
    for _ in range(WARMUP):
        func()
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]

def numpy_stream():
    acc[:] = 0.0
    for flat in flat_frames:
        np.add(acc, flat, out=acc)
    np.multiply(acc, factor, out=out)

def ea_single():
    acc[:] = 0.0
    for flat in flat_frames:
        _stack.accumulate_f32x8(acc, flat)
    _stack.scale_f32x8(acc, out, float(factor))

def ea_batched():
    acc[:] = 0.0
    i = 0
    while i + 8 <= n_frames:
        _stack.accumulate_batch8_f32x8(
            acc,
            flat_frames[i], flat_frames[i+1], flat_frames[i+2], flat_frames[i+3],
            flat_frames[i+4], flat_frames[i+5], flat_frames[i+6], flat_frames[i+7],
        )
        i += 8
    while i + 4 <= n_frames:
        _stack.accumulate_batch4_f32x8(
            acc,
            flat_frames[i], flat_frames[i+1], flat_frames[i+2], flat_frames[i+3],
        )
        i += 4
    while i < n_frames:
        _stack.accumulate_f32x8(acc, flat_frames[i])
        i += 1
    _stack.scale_f32x8(acc, out, float(factor))

def ea_batched_nt():
    acc[:] = 0.0
    i = 0
    while i + 8 <= n_frames:
        _stack.accumulate_batch8_f32x8_nt(
            acc,
            flat_frames[i], flat_frames[i+1], flat_frames[i+2], flat_frames[i+3],
            flat_frames[i+4], flat_frames[i+5], flat_frames[i+6], flat_frames[i+7],
        )
        i += 8
    while i + 4 <= n_frames:
        _stack.accumulate_batch4_f32x8(
            acc,
            flat_frames[i], flat_frames[i+1], flat_frames[i+2], flat_frames[i+3],
        )
        i += 4
    while i < n_frames:
        _stack.accumulate_f32x8(acc, flat_frames[i])
        i += 1
    _stack.scale_f32x8(acc, out, float(factor))

t_np = bench(numpy_stream)
t_ea = bench(ea_single)
t_bat = bench(ea_batched)
t_nt = bench(ea_batched_nt)

# Correctness check
ea_batched()
r_bat = out.copy()
numpy_stream()
r_np = out.copy()
assert np.allclose(r_bat, r_np, atol=1e-5), "batched result mismatch"

print(json.dumps({{
    "t_np": t_np,
    "t_ea": t_ea,
    "t_bat": t_bat,
    "t_nt": t_nt,
}}))
'''


def main():
    import numpy as np

    # Memory traffic model (bytes)
    def single_traffic(n_pixels, n_frames):
        # per frame: read acc + read frame + write acc = 3 * 4B * n_pixels
        return n_frames * 3 * 4 * n_pixels

    def batched_traffic(n_pixels, n_frames):
        # 1 read acc + N frame reads + 1 write acc = (N+2) * 4B * n_pixels
        return (n_frames + 2) * 4 * n_pixels

    print(f"Scale Sweep: {N_FRAMES} frames, single-threaded, {RUNS} runs median")
    print(f"STREAM peak (this machine): ~21 GB/s (single-thread, Copy/Add)")
    print()
    print(f"{'Size':>10}  {'AccMB':>5}  "
          f"{'NP str':>7}  {'Ea 1fr':>7}  {'Ea bat':>7}  {'Ea NT':>7}  "
          f"{'bat/NP':>6}  {'NT/bat':>6}  "
          f"{'NP GB/s':>7}  {'bat GB/s':>8}")
    print("-" * 100)

    for size in SIZES:
        n_pixels = size * size
        acc_mb = n_pixels * 4 / 1024 / 1024
        label = f"{size}x{size}"

        code = WORKER.format(
            demo_dir=str(DEMO_DIR),
            size=size,
            n_frames=N_FRAMES,
            warmup=WARMUP,
            runs=RUNS,
        )

        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=600,
            )
        except subprocess.TimeoutExpired:
            print(f"{label:>10}  {acc_mb:>5.1f}  TIMEOUT")
            continue

        if result.returncode != 0:
            err = result.stderr.strip().split('\n')[-1][:60]
            print(f"{label:>10}  {acc_mb:>5.1f}  FAILED: {err}")
            continue

        d = json.loads(result.stdout.strip())
        t_np = d["t_np"]
        t_ea = d["t_ea"]
        t_bat = d["t_bat"]
        t_nt = d["t_nt"]

        # Effective bandwidth (using single-frame traffic model for NP,
        # batched traffic model for batched)
        gbs_np = single_traffic(n_pixels, N_FRAMES) / (t_np / 1000) / 1e9
        gbs_bat = batched_traffic(n_pixels, N_FRAMES) / (t_bat / 1000) / 1e9

        bat_vs_np = t_np / t_bat
        nt_vs_bat = t_bat / t_nt

        print(f"{label:>10}  {acc_mb:>5.1f}  "
              f"{t_np:>6.1f}ms  {t_ea:>6.1f}ms  {t_bat:>6.1f}ms  {t_nt:>6.1f}ms  "
              f"{bat_vs_np:>5.2f}x  {nt_vs_bat:>5.2f}x  "
              f"{gbs_np:>6.1f}    {gbs_bat:>6.1f}")

    print()
    print("Legend:")
    print("  NP str  = NumPy streaming (acc += frame loop)")
    print("  Ea 1fr  = Ea single-frame accumulate")
    print("  Ea bat  = Ea batched (8 frames/pass, regular store)")
    print("  Ea NT   = Ea batched (8 frames/pass, stream_store)")
    print("  bat/NP  = speedup of Ea batched vs NumPy streaming")
    print("  NT/bat  = speedup of NT over regular store (>1 = NT wins)")
    print("  GB/s    = effective bandwidth (traffic model / wall time)")
    print()
    print("When acc > L3 cache (~20MB), expect:")
    print("  - bat/NP to increase (fewer DRAM round-trips for acc)")
    print("  - NT/bat to increase (stream_store avoids write-allocate)")


if __name__ == "__main__":
    main()
