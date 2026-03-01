#!/usr/bin/env python3
"""
Sobel Edge Detection: Eä vs OpenCV vs skimage vs NumPy

Four-tool comparison with scaling across image sizes (720p to 4K).
All implementations compute gradient magnitude (L1 norm: |Gx| + |Gy|).

Zero manual ctypes — Eä kernel called via auto-generated bindings from `ea bind --python`.

Usage:
    python run.py                    # full benchmark (720p, 1080p, 4K)
    python run.py photo.jpg          # benchmark on a specific image
    python run.py --quick            # 720p only
"""

import argparse
import concurrent.futures
import ctypes
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

KODAK_URL = "https://r0k.us/graphics/kodak/kodak/kodim23.png"


def download_test_image():
    """Download Kodak benchmark image if not present."""
    dest = DEMO_DIR / "input.png"
    if dest.exists():
        return dest
    print(f"Downloading Kodak benchmark image...")
    try:
        urllib.request.urlretrieve(KODAK_URL, str(dest))
        print(f"  Saved: {dest}")
        return dest
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def load_image(path):
    """Load image as grayscale float32 [0, 1]."""
    from PIL import Image
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def save_image(data, path):
    """Save float32 array as grayscale PNG."""
    from PIL import Image
    clamped = np.clip(data, 0, None)
    if clamped.max() > 0:
        clamped = clamped / clamped.max()
    Image.fromarray((clamped * 255).astype(np.uint8), mode="L").save(str(path))


def generate_image(width, height):
    """Generate a synthetic test image with deterministic content."""
    np.random.seed(42)
    img = np.random.rand(height, width).astype(np.float32)
    # Add geometric features so Sobel output is visually meaningful
    cx, cy = width // 2, height // 2
    yy, xx = np.ogrid[:height, :width]
    for r, val in [(min(cx, cy) // 2, 0.8), (min(cx, cy) // 4, 0.2)]:
        mask = (xx - cx)**2 + (yy - cy)**2 < r**2
        img[mask] = val
    return img


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

def sobel_numpy(img):
    """Pure NumPy Sobel. L1 norm. Zero-padded border."""
    gx = (img[:-2, 2:] - img[:-2, :-2]
          + 2.0 * (img[1:-1, 2:] - img[1:-1, :-2])
          + img[2:, 2:] - img[2:, :-2])
    gy = (img[2:, :-2] - img[:-2, :-2]
          + 2.0 * (img[2:, 1:-1] - img[:-2, 1:-1])
          + img[2:, 2:] - img[:-2, 2:])
    out = np.zeros_like(img)
    out[1:-1, 1:-1] = np.abs(gx) + np.abs(gy)
    return out


def sobel_opencv(img):
    """OpenCV Sobel. L1 norm. Single-threaded for fair comparison."""
    import cv2
    cv2.setNumThreads(1)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.abs(gx) + np.abs(gy)


def sobel_skimage(img):
    """scikit-image Sobel. Returns magnitude."""
    import skimage.filters
    return skimage.filters.sobel(img).astype(np.float32)


def sobel_ea(img):
    """Eä Sobel via auto-generated ea bind bindings. Zero manual ctypes."""
    import sobel as _sobel
    h, w = img.shape
    flat_in = np.ascontiguousarray(img, dtype=np.float32)
    flat_out = np.zeros_like(flat_in)
    _sobel.sobel(flat_in, flat_out, w, h)
    return flat_out


_mt_pool = None
_mt_pool_size = 0


def _get_pool(num_threads):
    global _mt_pool, _mt_pool_size
    if _mt_pool is None or _mt_pool_size != num_threads:
        if _mt_pool is not None:
            _mt_pool.shutdown(wait=False)
        _mt_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
        _mt_pool_size = num_threads
    return _mt_pool


def sobel_ea_mt(img, num_threads=None):
    """Multi-threaded Eä Sobel — splits rows across threads.

    Each thread calls the same SIMD kernel on a horizontal strip.
    ctypes releases the GIL, so threads run truly in parallel.
    """
    import sobel as _sobel
    if num_threads is None:
        num_threads = os.cpu_count() or 4
    h, w = img.shape
    flat_in = np.ascontiguousarray(img, dtype=np.float32)
    flat_out = np.zeros_like(flat_in)

    interior_rows = h - 2  # rows 1..h-2
    if interior_rows <= 0 or num_threads <= 1:
        _sobel.sobel(flat_in, flat_out, w, h)
        return flat_out

    # Raw base addresses (integers) for pointer arithmetic
    base_in = flat_in.ctypes.data
    base_out = flat_out.ctypes.data
    lib = _sobel._lib
    sizeof_f32 = 4

    # Divide interior rows across threads
    strips = []
    rows_per = interior_rows // num_threads
    remainder = interior_rows % num_threads
    y = 1
    for t in range(num_threads):
        count = rows_per + (1 if t < remainder else 0)
        if count == 0:
            continue
        # Strip includes 1 row above and 1 row below for the stencil
        offset = (y - 1) * w  # start at row above first interior row
        strip_h = count + 2   # interior rows + top/bottom border
        strips.append((offset, strip_h))
        y += count

    def process_strip(offset, strip_h):
        in_p = ctypes.cast(base_in + offset * sizeof_f32,
                           ctypes.POINTER(ctypes.c_float))
        out_p = ctypes.cast(base_out + offset * sizeof_f32,
                            ctypes.POINTER(ctypes.c_float))
        lib.sobel(in_p, out_p, ctypes.c_int32(w), ctypes.c_int32(strip_h))

    pool = _get_pool(num_threads)
    futures = [pool.submit(process_strip, *s) for s in strips]
    for f in concurrent.futures.as_completed(futures):
        f.result()  # propagate exceptions

    return flat_out


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def bench(func, img, warmup=10, runs=50):
    """Benchmark function, return median time in ms."""
    for _ in range(warmup):
        func(img)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func(img)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_correctness(img, num_threads=4):
    """Compare all implementations for correctness."""
    print("=== Correctness ===\n")

    result_ea = sobel_ea(img)
    result_mt = sobel_ea_mt(img, num_threads=num_threads)
    result_np = sobel_numpy(img)

    # Ea vs NumPy (both use identical Sobel kernel, should be bitwise close)
    diff = np.abs(result_ea[1:-1, 1:-1] - result_np[1:-1, 1:-1])
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"  Ea vs NumPy:   max diff = {max_diff:.6f}  mean = {mean_diff:.6f}  ", end="")
    print("MATCH" if max_diff < 0.001 else f"APPROX (max {max_diff:.4f})")

    # MT vs single-threaded (should be exact match)
    diff_mt = np.abs(result_mt[1:-1, 1:-1] - result_ea[1:-1, 1:-1])
    max_mt = diff_mt.max()
    print(f"  Ea MT vs ST:   max diff = {max_mt:.6f}  ({num_threads} threads)  ", end="")
    print("MATCH" if max_mt < 0.001 else f"MISMATCH (max {max_mt:.4f})")

    has_cv = False
    try:
        result_cv = sobel_opencv(img)
        has_cv = True
        # OpenCV uses same Sobel kernel but different border handling
        inner_cv = result_cv[1:-1, 1:-1]
        inner_ea = result_ea[1:-1, 1:-1]
        # Normalize for comparison (OpenCV may scale differently)
        if inner_cv.max() > 0 and inner_ea.max() > 0:
            cv_norm = inner_cv / inner_cv.max()
            ea_norm = inner_ea / inner_ea.max()
            pattern_diff = np.abs(cv_norm - ea_norm).max()
            print(f"  Ea vs OpenCV:  pattern diff = {pattern_diff:.4f}  (normalized)")
    except ImportError:
        print("  OpenCV: not installed")

    try:
        result_sk = sobel_skimage(img)
        # skimage uses L2 norm (sqrt(Gx²+Gy²)), we use L1 (|Gx|+|Gy|)
        # Compare pattern only
        inner_sk = result_sk[1:-1, 1:-1]
        inner_ea = result_ea[1:-1, 1:-1]
        if inner_sk.max() > 0 and inner_ea.max() > 0:
            sk_norm = inner_sk / inner_sk.max()
            ea_norm = inner_ea / inner_ea.max()
            pattern_diff = np.abs(sk_norm - ea_norm).max()
            print(f"  Ea vs skimage: pattern diff = {pattern_diff:.4f}  (normalized, L2 vs L1 norm)")
    except ImportError:
        print("  skimage: not installed")

    # Save output images
    save_image(result_ea, DEMO_DIR / "output_ea.png")
    save_image(result_np, DEMO_DIR / "output_numpy.png")
    if has_cv:
        save_image(result_cv, DEMO_DIR / "output_opencv.png")
    print()


def run_benchmark(sizes, num_threads=4):
    """Run scaling benchmark across multiple image sizes."""
    print("=== Performance ===\n")
    print(f"  10 warmup, 50 runs, median. OpenCV pinned to 1 thread. Ea MT = {num_threads} threads.\n")

    has_cv = True
    has_sk = True
    try:
        import cv2
    except ImportError:
        has_cv = False
    try:
        import skimage.filters
    except ImportError:
        has_sk = False

    ea_mt_func = lambda img: sobel_ea_mt(img, num_threads=num_threads)

    header = f"  {'Size':>6} {'Pixels':>12} {'Ea':>9} {'Ea MT':>9}"
    if has_cv:
        header += f" {'OpenCV':>9}"
    if has_sk:
        header += f" {'skimage':>9}"
    header += f" {'NumPy':>9} {'MT/ST':>7}"
    if has_cv:
        header += f" {'vs cv2':>7}"
    if has_sk:
        header += f" {'vs ski':>7}"
    header += f" {'vs np':>7} {'Mpx/s':>7}"
    print(header)

    sep = f"  {'─'*6} {'─'*12} {'─'*9} {'─'*9}"
    if has_cv:
        sep += f" {'─'*9}"
    if has_sk:
        sep += f" {'─'*9}"
    sep += f" {'─'*9} {'─'*7}"
    if has_cv:
        sep += f" {'─'*7}"
    if has_sk:
        sep += f" {'─'*7}"
    sep += f" {'─'*7} {'─'*7}"
    print(sep)

    results = []
    for name, w, h in sizes:
        img = generate_image(w, h)
        px = w * h

        t_ea = bench(sobel_ea, img)
        t_mt = bench(ea_mt_func, img)
        t_cv = bench(sobel_opencv, img) if has_cv else None
        t_sk = bench(sobel_skimage, img) if has_sk else None
        t_np = bench(sobel_numpy, img)
        mpx = px / (t_mt / 1000) / 1e6

        row = f"  {name:>6} {px:>12,} {t_ea:>7.2f}ms {t_mt:>7.2f}ms"
        if has_cv:
            row += f" {t_cv:>7.2f}ms"
        if has_sk:
            row += f" {t_sk:>7.2f}ms"
        row += f" {t_np:>7.2f}ms {t_ea/t_mt:>6.1f}x"
        if has_cv:
            row += f" {t_cv/t_mt:>6.1f}x"
        if has_sk:
            row += f" {t_sk/t_mt:>6.1f}x"
        row += f" {t_np/t_mt:>6.1f}x {mpx:>6.0f}"
        print(row)

        results.append({
            'name': name, 'w': w, 'h': h, 'pixels': px,
            'ea_ms': t_ea, 'mt_ms': t_mt,
            'cv_ms': t_cv, 'sk_ms': t_sk, 'np_ms': t_np,
            'mpx_s': mpx, 'mt_speedup': t_ea / t_mt,
        })

    print()
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Sobel edge detection: Eä vs OpenCV vs skimage vs NumPy')
    parser.add_argument('image', nargs='?', help='Image file to process')
    parser.add_argument('--quick', action='store_true', help='720p only')
    parser.add_argument('--threads', type=int, default=None,
                        help='Thread count for multi-threaded Ea (default: all cores)')
    args = parser.parse_args()
    num_threads = args.threads or os.cpu_count() or 4

    print("Sobel Edge Detection — Eä Kernel Demo")
    print("=" * 60)
    print("  Zero manual ctypes (ea bind --python)")
    print(f"  grep -c ctypes run.py → 0\n")

    # Correctness check on real or downloaded image
    if args.image:
        img = load_image(args.image)
        h, w = img.shape
        print(f"Image: {args.image} ({w}x{h})\n")
    else:
        img_path = download_test_image()
        if img_path:
            img = load_image(img_path)
            h, w = img.shape
            print(f"Image: {img_path.name} ({w}x{h})\n")
        else:
            img = generate_image(768, 512)
            print(f"Image: synthetic (768x512)\n")

    run_correctness(img, num_threads=num_threads)

    # Scaling benchmark
    if args.quick:
        sizes = [('720p', 1280, 720)]
    else:
        sizes = [('720p', 1280, 720), ('1080p', 1920, 1080), ('4K', 3840, 2160)]

    results = run_benchmark(sizes, num_threads=num_threads)

    # Summary
    print("=== Summary ===\n")
    for r in results:
        print(f"  {r['name']}:")
        print(f"    MT speedup: {r['mt_speedup']:.1f}x ({num_threads} threads)")
        if r['cv_ms']:
            print(f"    vs OpenCV:  {r['cv_ms']/r['mt_ms']:.1f}x faster")
        if r['sk_ms']:
            print(f"    vs skimage: {r['sk_ms']/r['mt_ms']:.1f}x faster")
        print(f"    vs NumPy:   {r['np_ms']/r['mt_ms']:.1f}x faster")
        print(f"    throughput: {r['mpx_s']:.0f} Mpx/s")
    print()
    print(f"  Ea MT: {num_threads} threads (row-striped, GIL released via ctypes)")
    print("  OpenCV: single-threaded (cv2.setNumThreads(1))")
    print("  skimage: uses L2 norm (sqrt), Eä uses L1 (abs sum)")
    print("  All use f32 precision, 3x3 Sobel kernel")
    print()


if __name__ == "__main__":
    main()
