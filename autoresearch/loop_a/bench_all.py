#!/usr/bin/env python3
"""Run all 5 Loop B benchmarks and return aggregate results as JSON.

Compiles each kernel with the current compiler, benchmarks against
pre-compiled C references, and reports per-kernel results.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

KERNELS = ["fma", "reduction", "dot_product", "saxpy", "clamp",
           "matmul", "prefix_sum", "histogram", "iir_ema",
           "particle_life", "gather_lut", "masked_scatter"]


def main():
    if len(sys.argv) < 2:
        print("Usage: bench_all.py <repo_root>", file=sys.stderr)
        sys.exit(1)

    repo_root = Path(sys.argv[1]).resolve()
    kernels_dir = repo_root / "autoresearch" / "kernels"
    ea_binary = str(repo_root / "target" / "release" / "ea")

    os.environ["EA_BINARY"] = ea_binary

    results = {}
    all_ok = True

    for kernel in KERNELS:
        kdir = kernels_dir / kernel
        bench_script = kdir / "bench_kernel.py"
        kernel_ea = kdir / "best_kernel.ea"
        ref_so = kdir / "reference.so"

        # Compile C reference if needed
        ref_c = kdir / "reference.c"
        if not ref_so.exists() and ref_c.exists():
            gcc_flags = ["-O3", "-march=native", "-shared", "-fPIC"]
            if kernel != "reduction":
                gcc_flags.append("-mfma")
            subprocess.run(
                ["gcc"] + gcc_flags + [str(ref_c), "-o", str(ref_so)],
                check=True,
            )

        if not bench_script.exists() or not kernel_ea.exists():
            results[kernel] = {"error": "missing files"}
            all_ok = False
            continue

        try:
            proc = subprocess.run(
                ["python3", str(bench_script), str(kernel_ea), str(ref_so)],
                capture_output=True, text=True, timeout=300,
            )
            if proc.returncode != 0:
                results[kernel] = {"error": proc.stderr.strip()[:200]}
                all_ok = False
                continue

            data = json.loads(proc.stdout.strip())
            results[kernel] = data
            if not data.get("correct"):
                all_ok = False

            status = "OK" if data.get("correct") else "FAIL"
            time_str = f"{data.get('time_us', '?')} µs"
            print(f"  {kernel}: {status} {time_str}", file=sys.stderr)

        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            results[kernel] = {"error": str(e)[:200]}
            all_ok = False

    output = {"all_ok": all_ok, "kernels": results}
    print(json.dumps(output))


if __name__ == "__main__":
    main()
