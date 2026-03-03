#!/usr/bin/env python3
"""
Astronomy Frame Stacking Demo: Ea vs NumPy

Stacks N noisy exposures of a starfield to reduce noise.
Signal reinforces, noise cancels by sqrt(N).

Zero manual ctypes — Ea kernel called via auto-generated bindings from `ea bind --python`.

Methodology:
  - Four implementations benchmarked:
    A) NumPy batch — np.mean on pre-built 3D array (idiomatic, best-case NumPy)
    B) NumPy streaming — same accumulate+scale algorithm as Ea, pure NumPy
    C) Ea streaming — single-frame SIMD kernel via auto-generated bindings
    D) Ea batched — multi-frame kernel (8 or 4 frames per pass over acc)
  - All buffers pre-allocated before timing. No allocation inside timed code.
  - Single-threaded: OMP/MKL/OPENBLAS threads pinned to 1.
  - 50 runs, median reported. 5 warmup runs discarded.

Usage:
    python run.py [N_frames]

Default: 16 frames from NASA SkyView (or synthetic if unavailable).
"""

import os
# Force single-threaded NumPy before import
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import subprocess
import urllib.request
from pathlib import Path
import numpy as np

DEMO_DIR = Path(__file__).parent
EA_ROOT = DEMO_DIR / ".." / ".."

N_FRAMES = 16
NOISE_SIGMA = 0.05
WARMUP = 5
RUNS = 50

NASA_SKYVIEW_URL = "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"


# ---------------------------------------------------------------------------
# NASA SkyView download
# ---------------------------------------------------------------------------

def download_nasa_frames(n_frames=16):
    """Download real telescope data from NASA SkyView.

    Downloads multiple survey images of the same sky region (M31 / Andromeda).
    Different surveys have different noise characteristics, simulating
    multiple exposures.
    """
    data_dir = DEMO_DIR / "nasa_data"
    data_dir.mkdir(exist_ok=True)

    # Check if we already have frames
    existing = sorted(data_dir.glob("frame_*.npy"))
    if len(existing) >= n_frames:
        print(f"  Using cached NASA data ({len(existing)} frames)")
        return [np.load(str(f)) for f in existing[:n_frames]]

    # Download a DSS image of M31 (Andromeda galaxy)
    print("Downloading NASA SkyView data (M31 / Andromeda)...")
    print(f"  Source: {NASA_SKYVIEW_URL}")

    try:
        # Download FITS file
        params = "Survey=DSS&Position=M31&Size=0.5&Pixels=1024&Return=FITS"
        url = f"{NASA_SKYVIEW_URL}?{params}"
        fits_path = data_dir / "m31.fits"

        if not fits_path.exists():
            urllib.request.urlretrieve(url, str(fits_path))
            print(f"  Downloaded: {fits_path}")

        # Try to read FITS
        try:
            from astropy.io import fits as pyfits
            with pyfits.open(str(fits_path)) as hdul:
                img = hdul[0].data.astype(np.float32)
        except ImportError:
            # Fallback: read FITS manually (simple 2D case)
            img = _read_simple_fits(fits_path)

        if img is None:
            return None

        # Normalize to [0, 1]
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()

        # Make square 1024x1024 (crop or pad)
        h, w = img.shape
        size = min(h, w, 1024)
        cy, cx = h // 2, w // 2
        img = img[cy - size//2:cy + size//2, cx - size//2:cx + size//2]

        # Generate N "exposures" by adding realistic noise to the real image
        # This simulates multiple telescope exposures of the same field
        rng = np.random.RandomState(100)
        frames = []
        for i in range(n_frames):
            noise = rng.normal(0, NOISE_SIGMA, img.shape).astype(np.float32)
            frame = np.clip(img + noise, 0.0, 1.0)
            frame_path = data_dir / f"frame_{i:03d}.npy"
            np.save(str(frame_path), frame)
            frames.append(frame)

        # Save reference
        np.save(str(data_dir / "reference.npy"), img)
        print(f"  Generated {n_frames} noisy exposures from real telescope data")
        return frames

    except Exception as e:
        print(f"  Download failed: {e}")
        print(f"  Falling back to synthetic starfield")
        return None


def _read_simple_fits(path):
    """Minimal FITS reader for simple 2D images. No dependencies."""
    try:
        with open(str(path), 'rb') as f:
            # Read primary header
            header = {}
            while True:
                block = f.read(2880)
                if not block:
                    return None
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

            naxis = int(header.get('NAXIS', 0))
            if naxis != 2:
                return None

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
                return None

            # FITS stores data in big-endian byte order
            be_dtype = np.dtype(dtype).newbyteorder('>')
            data = np.frombuffer(
                f.read(naxis1 * naxis2 * abs(bitpix) // 8), dtype=be_dtype
            )
            data = data.astype(dtype)
            return data.reshape(naxis2, naxis1).astype(np.float32)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Starfield generation
# ---------------------------------------------------------------------------

def generate_starfield(width=1024, height=1024, seed=42):
    """Generate a synthetic starfield: stars as 2D gaussians, a nebula, sky glow."""
    rng = np.random.RandomState(seed)
    img = np.zeros((height, width), dtype=np.float32)

    # Sky glow background
    yy, xx = np.mgrid[0:height, 0:width]
    img += 0.02 + 0.01 * np.exp(-((yy - height * 0.5) ** 2) / (2 * (height * 0.4) ** 2))

    # Nebula: broad elliptical glow
    cx_neb, cy_neb = width * 0.35, height * 0.6
    sx_neb, sy_neb = width * 0.15, height * 0.10
    nebula = 0.08 * np.exp(-(((xx - cx_neb) / sx_neb) ** 2 +
                              ((yy - cy_neb) / sy_neb) ** 2) / 2)
    img += nebula.astype(np.float32)

    # Stars: 60-80 point sources as 2D gaussians
    n_stars = rng.randint(60, 81)
    for _ in range(n_stars):
        cx = rng.uniform(0, width)
        cy = rng.uniform(0, height)
        brightness = rng.uniform(0.3, 1.0)
        sigma = rng.uniform(1.0, 3.0)
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        star = brightness * np.exp(-r2 / (2 * sigma ** 2))
        img += star.astype(np.float32)

    img = np.clip(img, 0.0, 1.0)
    return img


# ---------------------------------------------------------------------------
# Noise generation
# ---------------------------------------------------------------------------

def generate_noisy_frames(reference, n_frames, sigma, seed=100):
    """Generate n_frames noisy copies of reference, each with gaussian noise."""
    rng = np.random.RandomState(seed)
    frames = []
    for _ in range(n_frames):
        noise = rng.normal(0, sigma, reference.shape).astype(np.float32)
        noisy = np.clip(reference + noise, 0.0, 1.0)
        frames.append(noisy)
    return frames


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def save_image(data, path):
    """Save float32 array as grayscale PNG."""
    clamped = np.clip(data, 0, None)
    if clamped.max() > 0:
        clamped = clamped / clamped.max()
    uint8 = (clamped * 255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(uint8, mode="L").save(str(path))
        return
    except ImportError:
        pass
    try:
        import cv2
        cv2.imwrite(str(path), uint8)
        return
    except ImportError:
        pass
    print(f"  Warning: could not save {path} (need Pillow or OpenCV)")


# ---------------------------------------------------------------------------
# Build Ea kernel
# ---------------------------------------------------------------------------

def build_ea_kernel():
    """Compile stack.ea to libstack.so + stack.py bindings if needed."""
    so_path = DEMO_DIR / "libstack.so"
    ea_path = DEMO_DIR / "stack.ea"
    py_path = DEMO_DIR / "stack.py"

    if (so_path.exists() and py_path.exists()
            and so_path.stat().st_mtime > ea_path.stat().st_mtime):
        return so_path

    print("Building Ea kernel...")
    result = subprocess.run(
        ["cargo", "build", "--features=llvm", "--release", "--quiet"],
        capture_output=True, text=True, cwd=str(EA_ROOT),
    )
    if result.returncode != 0:
        print(f"Compiler build failed:\n{result.stderr}")
        sys.exit(1)

    ea_bin = EA_ROOT / "target" / "release" / "ea"

    result = subprocess.run(
        [str(ea_bin), str(ea_path), "--lib", "-o", str(so_path)],
        capture_output=True, text=True, cwd=str(DEMO_DIR),
    )
    if result.returncode != 0:
        print(f"Kernel build failed:\n{result.stderr}")
        sys.exit(1)

    result = subprocess.run(
        [str(ea_bin), "bind", str(ea_path), "--python"],
        capture_output=True, text=True, cwd=str(DEMO_DIR),
    )
    if result.returncode != 0:
        print(f"Binding generation failed:\n{result.stderr}")
        sys.exit(1)

    for obj in DEMO_DIR.glob("*.o"):
        obj.unlink()

    print(f"  Built: {so_path}")
    print(f"  Generated: {py_path}")
    return so_path


# ---------------------------------------------------------------------------
# SNR measurement
# ---------------------------------------------------------------------------

def compute_snr(image, reference):
    """Compute signal-to-noise ratio in dB."""
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean((image - reference) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n_frames = N_FRAMES
    if len(sys.argv) > 1:
        n_frames = int(sys.argv[1])

    print(f"Astronomy Frame Stacking: {n_frames} frames, 1024x1024")
    print()

    # Try real NASA data first, fall back to synthetic
    frames = download_nasa_frames(n_frames)
    if frames is not None:
        reference_path = DEMO_DIR / "nasa_data" / "reference.npy"
        if reference_path.exists():
            reference = np.load(str(reference_path))
        else:
            reference = generate_starfield()
        data_source = "NASA SkyView (M31 / Andromeda)"
    else:
        print("Generating starfield reference...")
        reference = generate_starfield()
        print(f"Generating {n_frames} noisy frames (sigma={NOISE_SIGMA})...")
        frames = generate_noisy_frames(reference, n_frames, NOISE_SIGMA)
        data_source = "synthetic starfield"

    h, w = reference.shape
    n_pixels = h * w
    print(f"  Image  : {w}x{h} ({n_pixels:,} pixels)")
    print(f"  Source : {data_source}")
    print(f"  NumPy  : {np.__version__}")
    print(f"  Threads: OMP=1, MKL=1, OPENBLAS=1 (single-threaded)")
    print()

    # Build Ea kernel
    build_ea_kernel()
    print()

    # Import auto-generated bindings
    sys.path.insert(0, str(DEMO_DIR))
    import stack as _stack

    # ---------------------------------------------------------------
    # Pre-allocate ALL buffers outside the benchmark loop
    # ---------------------------------------------------------------

    # For NumPy batch: pre-build 3D array (best-case: data already contiguous)
    stacked_3d = np.array(frames)  # shape (N, H, W)

    # For streaming (both NumPy and Ea): pre-flatten frames
    flat_frames = [np.ascontiguousarray(f, dtype=np.float32).ravel()
                   for f in frames]

    # Pre-allocate accumulator and output
    acc = np.zeros(n_pixels, dtype=np.float32)
    out = np.zeros(n_pixels, dtype=np.float32)
    factor = np.float32(1.0 / n_frames)

    # ---------------------------------------------------------------
    # Benchmark functions (zero allocation inside timed code)
    # ---------------------------------------------------------------

    def numpy_batch():
        return np.mean(stacked_3d, axis=0)

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

    # ---------------------------------------------------------------
    # Correctness: all four must agree
    # ---------------------------------------------------------------
    print("=== Correctness ===")
    ref_batch = numpy_batch()

    numpy_streaming()
    result_np_stream = out.reshape(h, w).copy()

    ea_streaming()
    result_ea = out.reshape(h, w).copy()

    ea_batched()
    result_ea_batched = out.reshape(h, w).copy()

    diff_stream = np.abs(result_np_stream - ref_batch)
    diff_ea = np.abs(result_ea - ref_batch)
    diff_batched = np.abs(result_ea_batched - ref_batch)

    print(f"  NumPy stream vs batch   : max={diff_stream.max():.2e}  "
          f"mean={diff_stream.mean():.2e}")
    print(f"  Ea stream vs NumPy batch: max={diff_ea.max():.2e}  "
          f"mean={diff_ea.mean():.2e}")
    print(f"  Ea batched vs NumPy     : max={diff_batched.max():.2e}  "
          f"mean={diff_batched.mean():.2e}")
    all_match = (diff_ea.max() < 1e-5 and diff_stream.max() < 1e-5
                 and diff_batched.max() < 1e-5)
    print(f"  All match: {'YES' if all_match else 'NO'}")
    print()

    # --- Frame stats (Ea kernel) ---
    print("=== Frame Stats (Ea SIMD kernel) ===")
    flat_result = np.ascontiguousarray(result_ea, dtype=np.float32).ravel()
    out_min = np.zeros(1, dtype=np.float32)
    out_max = np.zeros(1, dtype=np.float32)
    out_sum = np.zeros(1, dtype=np.float32)
    _stack.frame_stats(flat_result, out_min, out_max, out_sum)
    print(f"  Min : {out_min[0]:.6f}")
    print(f"  Max : {out_max[0]:.6f}")
    print(f"  Mean: {out_sum[0] / n_pixels:.6f}")
    print()

    # --- SNR Analysis ---
    print("=== SNR Analysis ===")
    snr_single = compute_snr(frames[0], reference)
    snr_numpy = compute_snr(ref_batch, reference)
    snr_ea = compute_snr(result_ea, reference)
    expected_improvement_db = 10 * np.log10(n_frames)

    print(f"  Single noisy frame SNR : {snr_single:6.2f} dB")
    print(f"  Stacked (NumPy) SNR    : {snr_numpy:6.2f} dB")
    print(f"  Stacked (Ea) SNR       : {snr_ea:6.2f} dB")
    print(f"  Improvement (Ea)       : {snr_ea - snr_single:+.2f} dB")
    print(f"  Expected improvement   : ~{expected_improvement_db:+.2f} dB "
          f"(noise power / {n_frames} \u2192 power SNR + 10\u00b7log\u2081\u2080({n_frames}))")
    print()

    # Save output images
    save_image(frames[0], DEMO_DIR / "output_single.png")
    save_image(result_ea, DEMO_DIR / "output_stacked.png")
    print("  Saved: output_single.png (one noisy frame)")
    print("  Saved: output_stacked.png (Ea stacked result)")
    print()

    # --- Performance ---
    print(f"=== Performance ({RUNS} runs, median, single-threaded) ===")
    print(f"  {n_frames} frames, {w}x{h}")
    print()

    t_batch, s_batch = benchmark(numpy_batch)
    print(f"  NumPy batch   (np.mean)  : {t_batch:8.2f} ms  +/-{s_batch:.2f}")

    t_stream, s_stream = benchmark(numpy_streaming)
    print(f"  NumPy stream  (same algo): {t_stream:8.2f} ms  +/-{s_stream:.2f}")

    t_ea, s_ea = benchmark(ea_streaming)
    print(f"  Ea stream     (1 frame)  : {t_ea:8.2f} ms  +/-{s_ea:.2f}")

    t_ea_b, s_ea_b = benchmark(ea_batched)
    print(f"  Ea batched    (8 frames) : {t_ea_b:8.2f} ms  +/-{s_ea_b:.2f}")
    print()

    print("  Speedup (Ea batched vs ...):")
    print(f"    vs NumPy batch  : {t_batch / t_ea_b:.2f}x")
    print(f"    vs NumPy stream : {t_stream / t_ea_b:.2f}x")
    print(f"    vs Ea stream    : {t_ea / t_ea_b:.2f}x")
    print()

    # Bandwidth analysis
    bytes_per_pixel = 4  # f32
    print("=== Bandwidth Analysis ===")
    # Single-frame: per frame, read acc + read frame + write acc = 3 loads
    single_bytes = n_frames * 3 * bytes_per_pixel * n_pixels
    single_gbs = single_bytes / (t_stream / 1000) / 1e9
    # Batched: 1 read acc + N read frames + 1 write acc = (N+2) loads total
    batched_bytes = (n_frames + 2) * bytes_per_pixel * n_pixels
    batched_gbs = batched_bytes / (t_ea_b / 1000) / 1e9
    print(f"  Single-frame memory traffic: "
          f"{single_bytes / 1e6:.1f} MB  ({single_gbs:.1f} GB/s)")
    print(f"  Batched memory traffic:      "
          f"{batched_bytes / 1e6:.1f} MB  ({batched_gbs:.1f} GB/s)")
    print(f"  Traffic reduction: {single_bytes / batched_bytes:.1f}x less")
    print()

    # Memory
    print("=== Memory ===")
    mb_per_frame = n_pixels * 4 / 1024 / 1024
    print(f"  Streaming (Ea or NumPy): {mb_per_frame:.1f} MB "
          f"(one accumulator + one output)")
    print(f"  NumPy batch            : {n_frames * mb_per_frame:.1f} MB "
          f"(all frames in 3D array + output)")
    print()

    # --- Summary ---
    print("=== Summary ===")
    print(f"  SNR gain: {snr_ea - snr_single:+.2f} dB from stacking {n_frames} frames")
    key = "faster" if t_ea_b < t_stream else "slower"
    print(f"  Ea batched vs NumPy: {t_stream / t_ea_b:.2f}x {key} "
          f"(same work, better memory access pattern)")
    print()
    print("Output images saved to demo/astro_stack/")


if __name__ == "__main__":
    main()
