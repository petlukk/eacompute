#!/usr/bin/env python3
"""
AVX-512 vs AVX2 comparison across all autoresearch kernels.

Compiles each kernel twice (AVX2 default vs --avx512), benchmarks both,
and reports which is faster. AVX-512 can cause frequency throttling on
some CPUs (especially Intel), making wider SIMD paradoxically slower.

AMD Zen 4 handles AVX-512 without downclocking (runs 512-bit as 2×256),
but register pressure and code bloat can still hurt.

Usage:
    python3 autoresearch/avx512_comparison.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
KERNELS_DIR = REPO / "autoresearch" / "kernels"
EA = str(REPO / "target" / "release" / "ea")

# Only test kernels where AVX-512 could plausibly help
# (float/int SIMD kernels, not scalar-heavy ones like particle_life)
KERNELS = [
    "fma",
    "reduction",
    "dot_product",
    "saxpy",
    "clamp",
    "frame_stats",
    "video_anomaly",
    "preprocess_fused",
    "threshold_u8",
    "sobel",
    "edge_detect_fused",
    "batch_dot",
    "batch_cosine",
]


def compile_kernel(kernel_path, output_so, avx512=False):
    """Compile a kernel with or without AVX-512."""
    cmd = [EA, str(kernel_path), "--lib", "--opt-level=3"]
    if avx512:
        cmd.append("--avx512")

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(kernel_path.parent),
    )
    return result.returncode == 0, result.stderr


def run_benchmark(bench_py, kernel_ea, ref_so):
    """Run a benchmark and return the JSON result."""
    env = os.environ.copy()
    env["EA_BINARY"] = EA

    result = subprocess.run(
        [sys.executable, str(bench_py), str(kernel_ea), str(ref_so)],
        capture_output=True, text=True, timeout=300, env=env,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return None


def main():
    print("AVX-512 vs AVX2 Comparison")
    print(f"CPU: AMD EPYC 9354P (Zen 4)")
    print(f"Scoring on: largest dataset size")
    print()
    print(f"{'Kernel':<22} {'AVX2 (µs)':>10} {'AVX512 (µs)':>12} {'Diff':>8} {'Winner':>8}")
    print("-" * 65)

    results = []

    for name in KERNELS:
        kdir = KERNELS_DIR / name
        bench = kdir / "bench_kernel.py"
        best = kdir / "best_kernel.ea"
        ref_so = kdir / "reference.so"

        if not bench.exists() or not best.exists() or not ref_so.exists():
            print(f"{name:<22} SKIP (missing files)")
            continue

        # Compile AVX2 (default)
        avx2_so = kdir / "best_kernel.so"
        ok_avx2, err_avx2 = compile_kernel(best, avx2_so, avx512=False)
        if not ok_avx2:
            print(f"{name:<22} SKIP (AVX2 compile failed)")
            continue

        # Benchmark AVX2
        r_avx2 = run_benchmark(bench, best, ref_so)
        if not r_avx2 or not r_avx2.get("correct"):
            print(f"{name:<22} SKIP (AVX2 benchmark failed)")
            continue
        t_avx2 = r_avx2["time_us"]

        # Now compile with AVX-512
        # We need to compile to a different .so, then benchmark
        # Copy kernel to a temp file to avoid overwriting
        avx512_ea = kdir / "kernel_avx512.ea"
        avx512_ea.write_text(best.read_text())

        ok_512, err_512 = compile_kernel(avx512_ea, kdir / "kernel_avx512.so", avx512=True)
        if not ok_512:
            # Some kernels may not support AVX-512 types
            avx512_ea.unlink(missing_ok=True)
            print(f"{name:<22} {t_avx2:>10.1f}     SKIP (512 compile fail)")
            continue

        # Benchmark AVX-512
        r_512 = run_benchmark(bench, avx512_ea, ref_so)
        avx512_ea.unlink(missing_ok=True)
        (kdir / "kernel_avx512.so").unlink(missing_ok=True)
        (kdir / "kernel_avx512.ea.json").unlink(missing_ok=True)

        if not r_512 or not r_512.get("correct"):
            print(f"{name:<22} {t_avx2:>10.1f}     SKIP (512 benchmark failed)")
            continue
        t_512 = r_512["time_us"]

        diff_pct = (t_avx2 - t_512) / t_avx2 * 100
        winner = "AVX-512" if diff_pct > 1.0 else "AVX2" if diff_pct < -1.0 else "tie"

        print(f"{name:<22} {t_avx2:>10.1f} {t_512:>12.1f} {diff_pct:>+7.1f}% {winner:>8}")
        results.append({
            "kernel": name,
            "avx2_us": t_avx2,
            "avx512_us": t_512,
            "diff_pct": round(diff_pct, 1),
            "winner": winner,
        })

    print()
    print("Legend:")
    print("  Diff > 0: AVX-512 faster")
    print("  Diff < 0: AVX-512 slower (frequency throttle or code bloat)")
    print("  |Diff| < 1%: tie (within noise)")
    print()

    avx512_wins = sum(1 for r in results if r["winner"] == "AVX-512")
    avx2_wins = sum(1 for r in results if r["winner"] == "AVX2")
    ties = sum(1 for r in results if r["winner"] == "tie")
    print(f"Score: AVX-512 wins {avx512_wins}, AVX2 wins {avx2_wins}, ties {ties}")


if __name__ == "__main__":
    main()
