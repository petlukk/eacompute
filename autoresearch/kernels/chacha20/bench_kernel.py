#!/usr/bin/env python3
"""Compile and benchmark Ea ChaCha20 kernel. Outputs JSON to stdout.

Benchmarks chacha20_encrypt across multiple data sizes against the C reference.
Correctness: exact byte-for-byte match (no tolerance — this is a cipher).
"""

import json
import os
import subprocess
import sys
import time
import ctypes
import numpy as np
from pathlib import Path

DATA_SIZES = [64 * 1024, 256 * 1024, 1024 * 1024, 16 * 1024 * 1024]
BYTES_PER_ELEM = 2  # read plaintext + write ciphertext
NUM_RUNS = 100
WARMUP_RUNS = 10
SEED = 42

U8_PTR = ctypes.POINTER(ctypes.c_uint8)
I32_PTR = ctypes.POINTER(ctypes.c_int32)
I32 = ctypes.c_int32
U32 = ctypes.c_uint32


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


def make_key_nonce():
    """Fixed key (bytes 0-31) and nonce (12 zero bytes)."""
    key = np.arange(32, dtype=np.uint8)
    key_i32 = np.frombuffer(key.tobytes(), dtype=np.int32)
    nonce = np.zeros(3, dtype=np.int32)
    return key_i32, nonce


def bench_at_size(ea_func, ref_func, n, key_i32, nonce):
    rng = np.random.default_rng(SEED)
    plaintext = rng.integers(0, 256, n, dtype=np.uint8)
    ea_ct = np.zeros(n, dtype=np.uint8)
    ref_ct = np.zeros(n, dtype=np.uint8)

    # Scratch buffer for Ea kernel (64 bytes, dual-pointer)
    ks_buf = np.zeros(16, dtype=np.int32)

    pt_p = plaintext.ctypes.data_as(U8_PTR)
    ea_ct_p = ea_ct.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    ref_ct_p = ref_ct.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    key_p = key_i32.ctypes.data_as(I32_PTR)
    nonce_p = nonce.ctypes.data_as(I32_PTR)
    ks_i32_p = ks_buf.ctypes.data_as(I32_PTR)
    ks_u8_p = ctypes.cast(ks_i32_p, U8_PTR)

    # Run Ea kernel
    ea_func(key_p, nonce_p, I32(0), pt_p, ea_ct_p, I32(n), ks_i32_p, ks_u8_p)

    # Run C reference
    ref_func(key_p, nonce_p, U32(0), pt_p, ref_ct_p, I32(n))

    # Exact byte-for-byte comparison
    if not np.array_equal(ea_ct, ref_ct):
        mismatches = np.where(ea_ct != ref_ct)[0]
        first = int(mismatches[0])
        return (f"correctness: mismatch at byte {first}: "
                f"ea=0x{ea_ct[first]:02x} ref=0x{ref_ct[first]:02x} (N={n})")

    # Benchmark
    for _ in range(WARMUP_RUNS):
        ea_func(key_p, nonce_p, I32(0), pt_p, ea_ct_p, I32(n),
                ks_i32_p, ks_u8_p)

    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        ea_func(key_p, nonce_p, I32(0), pt_p, ea_ct_p, I32(n),
                ks_i32_p, ks_u8_p)
        times.append(time.perf_counter() - start)

    times.sort()
    median_us = times[len(times) // 2] * 1e6
    min_us = times[0] * 1e6
    return (round(median_us, 1), round(min_us, 1))


def main():
    if len(sys.argv) < 3:
        print("Usage: bench_kernel.py <kernel.ea> <reference.so>",
              file=sys.stderr)
        sys.exit(1)

    kernel_path = Path(sys.argv[1]).resolve()
    ref_so_path = Path(sys.argv[2]).resolve()
    ea_binary = str(Path(
        os.environ.get("EA_BINARY", "./target/release/ea")).resolve())

    kernel_dir = kernel_path.parent
    so_name = kernel_path.stem + ".so"
    so_path = kernel_dir / so_name

    # Clean stale .so
    for stale in [so_path, Path(so_name), kernel_path.with_suffix(".so")]:
        if stale.exists():
            stale.unlink()

    # Compile Ea kernel
    result = subprocess.run(
        [ea_binary, str(kernel_path), "--lib", "--opt-level=3"],
        capture_output=True, text=True, cwd=str(kernel_dir),
    )
    if result.returncode != 0:
        error_msg = result.stderr.strip() or result.stdout.strip()
        output(False, error=f"compile: {error_msg}")

    if not so_path.exists():
        output(False, error="compile: .so not found after compilation")

    # Load shared libraries
    try:
        ea_lib = ctypes.CDLL(str(so_path.resolve()))
        ref_lib = ctypes.CDLL(str(ref_so_path.resolve()))
    except OSError as e:
        output(False, error=f"load: {e}")

    # Set up function signatures
    try:
        ea_func = ea_lib.chacha20_encrypt
        ea_func.argtypes = [I32_PTR, I32_PTR, I32, U8_PTR,
                            ctypes.POINTER(ctypes.c_uint8), I32,
                            I32_PTR, U8_PTR]
        ea_func.restype = None

        ref_func = ref_lib.chacha20_encrypt_ref
        ref_func.argtypes = [I32_PTR, I32_PTR, U32, U8_PTR,
                             ctypes.POINTER(ctypes.c_uint8), I32]
        ref_func.restype = None
    except AttributeError as e:
        output(False, error=f"symbol: {e}")

    key_i32, nonce = make_key_nonce()

    breakdown = {}
    for n in DATA_SIZES:
        label = f"N={n}"
        result = bench_at_size(ea_func, ref_func, n, key_i32, nonce)

        if isinstance(result, str):
            output(False, error=result)

        median_us, min_us = result
        total_bytes = n * BYTES_PER_ELEM
        gbs = total_bytes / (median_us / 1e6) / 1e9
        breakdown[label] = {
            "median_us": median_us, "min_us": min_us, "gbs": round(gbs, 2)
        }
        print(f"  {label}: {median_us} us median, {min_us} us min  |  "
              f"{gbs:.2f} GB/s", file=sys.stderr)

    # Primary metric: largest size (16MB, exceeds cache)
    largest_label = f"N={DATA_SIZES[-1]}"
    aggregate_median = breakdown[largest_label]["median_us"]
    aggregate_min = breakdown[largest_label]["min_us"]
    loc = count_loc(kernel_path)

    output(True, time_us=aggregate_median, min_us=aggregate_min, loc=loc,
           breakdown=breakdown)


if __name__ == "__main__":
    main()
