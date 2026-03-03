#!/usr/bin/env python3
"""
Plot scale sweep results: speedup and effective bandwidth across image sizes.

Runs bench_scale.py worker subprocess for each size, collects data, and
produces a two-panel plot saved to scale_sweep.png.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEMO_DIR = Path(__file__).parent

N_FRAMES = 16
WARMUP = 5
RUNS = 30

SIZES = [512, 1024, 2048, 4096]

# Same worker template as bench_scale.py
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

t_np = bench(numpy_stream)
t_ea = bench(ea_single)
t_bat = bench(ea_batched)

print(json.dumps({{
    "t_np": t_np,
    "t_ea": t_ea,
    "t_bat": t_bat,
}}))
'''


def single_traffic(n_pixels, n_frames):
    """Memory traffic for single-frame loop (bytes)."""
    return n_frames * 3 * 4 * n_pixels


def batched_traffic(n_pixels, n_frames):
    """Memory traffic for batched loop (bytes)."""
    return (n_frames + 2) * 4 * n_pixels


def collect_data():
    """Run benchmarks and return results dict per size."""
    results = []
    for size in SIZES:
        n_pixels = size * size
        label = f"{size}x{size}"
        print(f"  Benchmarking {label}...", end=" ", flush=True)

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
            print("TIMEOUT")
            continue

        if result.returncode != 0:
            err = result.stderr.strip().split('\n')[-1][:60]
            print(f"FAILED: {err}")
            continue

        d = json.loads(result.stdout.strip())
        speedup = d["t_np"] / d["t_bat"]
        gbs_np = single_traffic(n_pixels, N_FRAMES) / (d["t_np"] / 1000) / 1e9
        gbs_bat = batched_traffic(n_pixels, N_FRAMES) / (d["t_bat"] / 1000) / 1e9

        results.append({
            "size": size,
            "label": label,
            "acc_mb": n_pixels * 4 / 1024 / 1024,
            "t_np": d["t_np"],
            "t_ea": d["t_ea"],
            "t_bat": d["t_bat"],
            "speedup": speedup,
            "gbs_np": gbs_np,
            "gbs_bat": gbs_bat,
        })
        print(f"{speedup:.2f}x speedup")

    return results


def plot(results):
    """Create two-panel plot: speedup and effective bandwidth."""
    labels = [r["label"] for r in results]
    speedups = [r["speedup"] for r in results]
    t_np = [r["t_np"] for r in results]
    t_ea = [r["t_ea"] for r in results]
    t_bat = [r["t_bat"] for r in results]
    acc_mbs = [r["acc_mb"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left panel: wall time comparison ---
    x = np.arange(len(labels))
    width = 0.25

    bars_np = ax1.bar(x - width, t_np, width, label="NumPy streaming", color="#4878cf")
    bars_ea = ax1.bar(x, t_ea, width, label="Eä single-frame", color="#6acc65")
    bars_bat = ax1.bar(x + width, t_bat, width, label="Eä batched (8/pass)", color="#d65f5f")

    ax1.set_xlabel("Image size")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title(f"Frame stacking: {N_FRAMES} frames, single-threaded")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(loc="upper left")
    ax1.set_yscale("log")

    # Add speedup labels on batched bars
    for i, (bar, sp) in enumerate(zip(bars_bat, speedups)):
        ax1.annotate(f"{sp:.1f}x",
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold",
                     color="#d65f5f")

    # --- Right panel: speedup vs cache hierarchy ---
    ax2.plot(acc_mbs, speedups, "o-", color="#d65f5f", linewidth=2, markersize=8)
    ax2.set_xlabel("Accumulator size (MB)")
    ax2.set_ylabel("Speedup (Eä batched / NumPy)")
    ax2.set_title("Speedup vs cache hierarchy")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(acc_mbs)
    ax2.set_xticklabels([f"{m:.0f} MB" for m in acc_mbs])
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Break-even")

    # Annotate cache boundaries
    ax2.axvspan(0.5, 1.5, alpha=0.08, color="green", label="L2 (~1 MB)")
    ax2.axvspan(1.5, 20, alpha=0.08, color="blue", label="L3 (~20 MB)")
    ax2.axvspan(20, 128, alpha=0.08, color="red", label="DRAM")
    ax2.legend(loc="upper right", fontsize=8)

    for i, (mb, sp) in enumerate(zip(acc_mbs, speedups)):
        # Last point: offset left to avoid clipping
        offset = (-40, 8) if i == len(acc_mbs) - 1 else (8, 8)
        ax2.annotate(f"{sp:.2f}x",
                     xy=(mb, sp), xytext=offset,
                     textcoords="offset points", fontsize=9)

    plt.tight_layout()
    out_path = DEMO_DIR / "scale_sweep.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


def main():
    print(f"Scale sweep: {N_FRAMES} frames, {RUNS} runs median\n")
    results = collect_data()
    if results:
        plot(results)
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
