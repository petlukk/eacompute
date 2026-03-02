#!/usr/bin/env python3
"""
Benchmark Ea stacking kernel on AstroBurst HST test data.

Uses the 1600x1600 float32 FITS files from:
  https://github.com/samuelkriegerbonini-dev/AstroBurst/tree/main/exampleFits/sample-data

Three HST WFPC2 narrowband images of the Eagle Nebula:
  502nmos.fits  (O-III 502nm)
  656nmos.fits  (H-alpha 656nm)
  673nmos.fits  (S-II 673nm)

Methodology:
  - All buffers pre-allocated before timing.
  - Frames pre-flattened to contiguous f32 arrays for both implementations.
  - Three comparisons:
    A) NumPy batch mean on pre-built 3D array (idiomatic NumPy, best case)
    B) NumPy streaming accumulate (same algorithm as Ea, pure NumPy)
    C) Ea streaming accumulate (SIMD kernel via auto-generated bindings)
  - 50 runs, median reported. 5 warmup runs discarded.
  - Single-threaded: OMP/MKL/OPENBLAS threads set to 1.

Expected result: Ea and NumPy streaming perform the same. Frame accumulation
is bandwidth-bound (load+add+store), so explicit SIMD doesn't beat NumPy's
already-vectorized np.add ufunc. This benchmark confirms that.
"""

import os
# Force single-threaded NumPy before import
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
from pathlib import Path
import numpy as np

DEMO_DIR = Path(__file__).parent
HST_DIR = DEMO_DIR / "hst_data"

WARMUP = 5
RUNS = 50


def read_fits_f32(path):
    """Read a simple 2D FITS image as float32. No dependencies."""
    with open(str(path), 'rb') as f:
        header = {}
        while True:
            block = f.read(2880)
            if not block:
                raise ValueError(f"Unexpected end of FITS file: {path}")
            text = block.decode('ascii', errors='replace')
            for i in range(0, len(text), 80):
                card = text[i:i+80]
                if card.startswith('END'):
                    break
                if len(card) > 8 and card[8] == '=':
                    key = card[:8].strip()
                    val = card[10:30].strip().strip("'").strip()
                    header[key] = val
            if 'END' in text:
                break

        naxis1 = int(header['NAXIS1'])
        naxis2 = int(header['NAXIS2'])
        bitpix = int(header['BITPIX'])

        if bitpix == -32:
            dtype = np.float32
        elif bitpix == -64:
            dtype = np.float64
        elif bitpix == 16:
            dtype = np.int16
        elif bitpix == 32:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported BITPIX={bitpix}")

        be_dtype = np.dtype(dtype).newbyteorder('>')
        nbytes = naxis1 * naxis2 * abs(bitpix) // 8
        data = np.frombuffer(f.read(nbytes), dtype=be_dtype)
        return data.astype(np.float32).reshape(naxis2, naxis1)


def benchmark(func, warmup=WARMUP, runs=RUNS):
    """Benchmark a zero-arg callable. Returns (median_ms, std_ms)."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    median = times[len(times) // 2]
    return median, float(np.std(times))


def main():
    fits_files = sorted(HST_DIR.glob("*.fits"))
    if not fits_files:
        print(f"ERROR: No FITS files in {HST_DIR}")
        print("Download from: https://github.com/samuelkriegerbonini-dev/AstroBurst"
              "/tree/main/exampleFits/sample-data")
        sys.exit(1)

    print("AstroBurst HST Test Data Benchmark")
    print("=" * 60)
    print()

    # Load and normalize frames
    raw_frames = []
    for f in fits_files:
        img = read_fits_f32(f)
        img = np.nan_to_num(img, nan=0.0)
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = (img - lo) / (hi - lo)
        raw_frames.append(img)
        print(f"  {f.name}  {img.shape[1]}x{img.shape[0]}  "
              f"raw range [{lo:.4g}, {hi:.4g}]")

    h, w = raw_frames[0].shape
    n_pixels = h * w
    n_frames = len(raw_frames)
    print()
    print(f"  Frames : {n_frames}")
    print(f"  Image  : {w}x{h} ({n_pixels:,} pixels, "
          f"{n_pixels * 4 / 1024 / 1024:.1f} MB/frame)")
    print(f"  NumPy  : {np.__version__}")
    print(f"  Threads: OMP=1, MKL=1, OPENBLAS=1 (single-threaded)")
    print()

    # ---------------------------------------------------------------
    # Pre-allocate ALL buffers outside the benchmark loop
    # ---------------------------------------------------------------

    # For NumPy batch: pre-build 3D array (this is the best-case scenario
    # where frames are already in a single contiguous block)
    stacked_3d = np.array(raw_frames)  # shape: (N, H, W), contiguous

    # For streaming (both NumPy and Ea): pre-flatten frames
    flat_frames = [np.ascontiguousarray(f, dtype=np.float32).ravel()
                   for f in raw_frames]

    # Pre-allocate accumulator and output buffers
    acc = np.zeros(n_pixels, dtype=np.float32)
    out = np.zeros(n_pixels, dtype=np.float32)
    factor = np.float32(1.0 / n_frames)

    # Import Ea bindings
    sys.path.insert(0, str(DEMO_DIR))
    import stack as _stack

    # ---------------------------------------------------------------
    # Define benchmark functions (zero allocation inside timed code)
    # ---------------------------------------------------------------

    def numpy_batch():
        """np.mean on pre-built 3D array. Best-case NumPy."""
        return np.mean(stacked_3d, axis=0)

    def numpy_streaming():
        """Same streaming algorithm as Ea, pure NumPy."""
        acc[:] = 0.0
        for flat in flat_frames:
            np.add(acc, flat, out=acc)
        np.multiply(acc, factor, out=out)

    def ea_streaming():
        """Ea SIMD kernel, same streaming algorithm."""
        acc[:] = 0.0
        for flat in flat_frames:
            _stack.accumulate_f32x8(acc, flat)
        _stack.scale_f32x8(acc, out, float(factor))

    # ---------------------------------------------------------------
    # Correctness: all three must agree
    # ---------------------------------------------------------------
    print("=== Correctness ===")
    ref = numpy_batch()

    numpy_streaming()
    result_np_stream = out.reshape(h, w).copy()

    ea_streaming()
    result_ea = out.reshape(h, w).copy()

    diff_stream = np.abs(result_np_stream - ref)
    diff_ea = np.abs(result_ea - ref)
    diff_ea_np = np.abs(result_ea - result_np_stream)

    print(f"  NumPy stream vs batch : max={diff_stream.max():.2e}  "
          f"mean={diff_stream.mean():.2e}")
    print(f"  Ea vs NumPy batch     : max={diff_ea.max():.2e}  "
          f"mean={diff_ea.mean():.2e}")
    print(f"  Ea vs NumPy stream    : max={diff_ea_np.max():.2e}  "
          f"mean={diff_ea_np.mean():.2e}")

    all_match = diff_ea.max() < 1e-5 and diff_stream.max() < 1e-5
    print(f"  All match: {'YES' if all_match else 'NO'}")
    print()

    # ---------------------------------------------------------------
    # Frame stats
    # ---------------------------------------------------------------
    print("=== Frame Stats (Ea kernel) ===")
    flat_result = np.ascontiguousarray(result_ea, dtype=np.float32).ravel()
    out_min = np.zeros(1, dtype=np.float32)
    out_max = np.zeros(1, dtype=np.float32)
    out_sum = np.zeros(1, dtype=np.float32)
    _stack.frame_stats(flat_result, out_min, out_max, out_sum)
    print(f"  Min : {out_min[0]:.6f}")
    print(f"  Max : {out_max[0]:.6f}")
    print(f"  Mean: {out_sum[0] / n_pixels:.6f}")
    print()

    # ---------------------------------------------------------------
    # Benchmark
    # ---------------------------------------------------------------
    print(f"=== Performance ({RUNS} runs, median, single-threaded) ===")
    print(f"  {n_frames} frames x {w}x{h}")
    print()

    t_batch, s_batch = benchmark(numpy_batch)
    print(f"  NumPy batch  (np.mean)  : {t_batch:8.2f} ms  +/-{s_batch:.2f}")

    t_stream, s_stream = benchmark(numpy_streaming)
    print(f"  NumPy stream (same algo): {t_stream:8.2f} ms  +/-{s_stream:.2f}")

    t_ea, s_ea = benchmark(ea_streaming)
    print(f"  Ea stream    (SIMD)     : {t_ea:8.2f} ms  +/-{s_ea:.2f}")
    print()

    # Report speedups against both baselines
    print("  Speedup:")
    print(f"    Ea vs NumPy batch  : {t_batch / t_ea:.2f}x "
          f"{'faster' if t_ea < t_batch else 'slower'}")
    print(f"    Ea vs NumPy stream : {t_stream / t_ea:.2f}x "
          f"{'faster' if t_ea < t_stream else 'slower'} "
          f"(apples-to-apples, same algorithm)")
    print()

    # Memory comparison (only applies to streaming vs batch)
    print("=== Memory (streaming vs batch) ===")
    mb_per_frame = n_pixels * 4 / 1024 / 1024
    print(f"  Streaming (Ea or NumPy): {mb_per_frame:.1f} MB "
          f"(one accumulator + one output)")
    print(f"  NumPy batch            : {n_frames * mb_per_frame:.1f} MB "
          f"(all frames in 3D array + output)")
    print(f"  Note: both streaming variants use the same memory.")
    print()

    # Save stacked result
    out_path = DEMO_DIR / "hst_stacked.png"
    try:
        from PIL import Image
        clamped = np.clip(result_ea, 0, 1)
        Image.fromarray((clamped * 255).astype(np.uint8), mode="L").save(
            str(out_path))
        print(f"  Saved: {out_path.name}")
    except ImportError:
        print("  (install Pillow to save output image)")


if __name__ == "__main__":
    main()
