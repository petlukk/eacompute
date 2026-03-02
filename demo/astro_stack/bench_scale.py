#!/usr/bin/env python3
"""
Scale benchmark: how does Ea vs NumPy change with data size?

Each config runs in a subprocess to avoid cross-contamination from
memory pressure, GC pauses, or cache pollution between tests.
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

CONFIGS = [
    (1024, 1024, 3),
    (1024, 1024, 16),
    (1600, 1600, 3),
    (1600, 1600, 16),
    (2048, 2048, 10),
    (2048, 2048, 20),
    (4096, 4096, 5),
    (4096, 4096, 10),
]

WORKER = '''
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys, time, json
import numpy as np

sys.path.insert(0, "{demo_dir}")
import stack as _stack

w, h, n_frames = {w}, {h}, {n_frames}
n_pixels = w * h
WARMUP = 5
RUNS = 30

rng = np.random.RandomState(42)
flat_frames = [rng.rand(n_pixels).astype(np.float32) for _ in range(n_frames)]
acc = np.zeros(n_pixels, dtype=np.float32)
out = np.zeros(n_pixels, dtype=np.float32)
factor = np.float32(1.0 / n_frames)

# NumPy batch: only if total < 400 MB
data_mb = n_frames * n_pixels * 4 / 1024 / 1024
t_batch = None
if data_mb < 400:
    stacked_3d = np.array([f.reshape(h, w) for f in flat_frames])
    def numpy_batch():
        return np.mean(stacked_3d, axis=0)
    for _ in range(WARMUP): numpy_batch()
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        numpy_batch()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    t_batch = times[len(times) // 2]
    del stacked_3d

def numpy_streaming():
    acc[:] = 0.0
    for flat in flat_frames:
        np.add(acc, flat, out=acc)
    np.multiply(acc, factor, out=out)

def ea_streaming():
    acc[:] = 0.0
    for flat in flat_frames:
        _stack.accumulate_f32x8(acc, flat)
    _stack.scale_f32x8(acc, out, float(factor))

for _ in range(WARMUP):
    numpy_streaming()
    ea_streaming()

times_np = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    numpy_streaming()
    times_np.append((time.perf_counter() - t0) * 1000)

times_ea = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    ea_streaming()
    times_ea.append((time.perf_counter() - t0) * 1000)

times_np.sort()
times_ea.sort()

print(json.dumps({{
    "t_batch": t_batch,
    "t_stream": times_np[len(times_np) // 2],
    "t_ea": times_ea[len(times_ea) // 2],
    "data_mb": data_mb,
}}))
'''


def main():
    import numpy as np
    print(f"NumPy {np.__version__}, single-threaded, 30 runs/config, median")
    print(f"Each config in separate process (no cross-contamination)")
    print()
    print(f"{'Config':>16}  {'Data':>6}  "
          f"{'NP batch':>9}  {'NP stream':>10}  {'Ea stream':>10}  "
          f"{'Ea/NP(stream)':>14}")
    print("-" * 82)

    for w, h, n_frames in CONFIGS:
        label = f"{w}x{h}x{n_frames}"
        code = WORKER.format(demo_dir=str(DEMO_DIR), w=w, h=h, n_frames=n_frames)

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"{label:>16}  FAILED: {result.stderr.strip()[:80]}")
            continue

        data = json.loads(result.stdout.strip())
        t_batch = data["t_batch"]
        t_stream = data["t_stream"]
        t_ea = data["t_ea"]
        data_mb = data["data_mb"]

        ratio = t_ea / t_stream
        tag = "FASTER" if ratio < 1.0 else "slower"
        batch_str = f"{t_batch:.1f}ms" if t_batch else "skip(OOM)"

        print(f"{label:>16}  {data_mb:>5.0f}M  "
              f"{batch_str:>9}  {t_stream:>9.1f}ms  {t_ea:>9.1f}ms  "
              f"{ratio:>8.2f}x {tag:>6}")

    print()


if __name__ == "__main__":
    main()
