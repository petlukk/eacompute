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

def run_correctness(img):
    """Compare all implementations for correctness."""
    print("=== Correctness ===\n")

    result_ea = sobel_ea(img)
    result_np = sobel_numpy(img)

    # Ea vs NumPy (both use identical Sobel kernel, should be bitwise close)
    diff = np.abs(result_ea[1:-1, 1:-1] - result_np[1:-1, 1:-1])
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"  Ea vs NumPy:   max diff = {max_diff:.6f}  mean = {mean_diff:.6f}  ", end="")
    print("MATCH" if max_diff < 0.001 else f"APPROX (max {max_diff:.4f})")

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


def run_benchmark(sizes):
    """Run scaling benchmark across multiple image sizes."""
    print("=== Performance ===\n")
    print(f"  10 warmup, 50 runs, median. OpenCV pinned to 1 thread.\n")

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

    header = f"  {'Size':>6} {'Pixels':>12} {'Ea':>9}"
    if has_cv:
        header += f" {'OpenCV':>9}"
    if has_sk:
        header += f" {'skimage':>9}"
    header += f" {'NumPy':>9}"
    if has_cv:
        header += f" {'vs cv2':>7}"
    if has_sk:
        header += f" {'vs ski':>7}"
    header += f" {'vs np':>7} {'Mpx/s':>7}"
    print(header)

    sep = f"  {'─'*6} {'─'*12} {'─'*9}"
    if has_cv:
        sep += f" {'─'*9}"
    if has_sk:
        sep += f" {'─'*9}"
    sep += f" {'─'*9}"
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
        t_cv = bench(sobel_opencv, img) if has_cv else None
        t_sk = bench(sobel_skimage, img) if has_sk else None
        t_np = bench(sobel_numpy, img)
        mpx = px / (t_ea / 1000) / 1e6

        row = f"  {name:>6} {px:>12,} {t_ea:>7.2f}ms"
        if has_cv:
            row += f" {t_cv:>7.2f}ms"
        if has_sk:
            row += f" {t_sk:>7.2f}ms"
        row += f" {t_np:>7.2f}ms"
        if has_cv:
            row += f" {t_cv/t_ea:>6.1f}x"
        if has_sk:
            row += f" {t_sk/t_ea:>6.1f}x"
        row += f" {t_np/t_ea:>6.1f}x {mpx:>6.0f}"
        print(row)

        results.append({
            'name': name, 'w': w, 'h': h, 'pixels': px,
            'ea_ms': t_ea, 'cv_ms': t_cv, 'sk_ms': t_sk, 'np_ms': t_np,
            'mpx_s': mpx,
        })

    print()
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Sobel edge detection: Eä vs OpenCV vs skimage vs NumPy')
    parser.add_argument('image', nargs='?', help='Image file to process')
    parser.add_argument('--quick', action='store_true', help='720p only')
    args = parser.parse_args()

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

    run_correctness(img)

    # Scaling benchmark
    if args.quick:
        sizes = [('720p', 1280, 720)]
    else:
        sizes = [('720p', 1280, 720), ('1080p', 1920, 1080), ('4K', 3840, 2160)]

    results = run_benchmark(sizes)

    # Summary
    print("=== Summary ===\n")
    for r in results:
        print(f"  {r['name']}:")
        if r['cv_ms']:
            print(f"    vs OpenCV:  {r['cv_ms']/r['ea_ms']:.1f}x faster")
        if r['sk_ms']:
            print(f"    vs skimage: {r['sk_ms']/r['ea_ms']:.1f}x faster")
        print(f"    vs NumPy:   {r['np_ms']/r['ea_ms']:.1f}x faster")
        print(f"    throughput: {r['mpx_s']:.0f} Mpx/s")
    print()
    print("  OpenCV: single-threaded (cv2.setNumThreads(1))")
    print("  skimage: uses L2 norm (sqrt), Eä uses L1 (abs sum)")
    print("  All use f32 precision, 3x3 Sobel kernel")
    print()


if __name__ == "__main__":
    main()
