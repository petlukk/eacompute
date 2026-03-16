#!/usr/bin/env python3
"""Compile and benchmark Ea parse_aggregate kernel. Outputs JSON to stdout.

Benchmarks parse_aggregate across multiple line counts.
Generates 1BRC-style data inline: "StationName;12.3\n" format.
"""

import json
import os
import subprocess
import sys
import time
import ctypes
import numpy as np
from pathlib import Path

LINE_COUNTS = [10_000, 100_000, 1_000_000, 10_000_000]
# Memory traffic: read text buffer + read newline positions (4B each) + hash table writes (negligible)
# Average line ~15 bytes -> total ~15*n_lines text + 4*n_lines newline positions
BYTES_PER_LINE = 19  # ~15 bytes text + 4 bytes newline index
NUM_RUNS = 20
WARMUP_RUNS = 5
SEED = 42

TABLE_SIZE = 1024
KEY_STRIDE = 64

U8_PTR = ctypes.POINTER(ctypes.c_uint8)
I32_PTR = ctypes.POINTER(ctypes.c_int32)
I32 = ctypes.c_int32

STATION_NAMES = [
    "Hamburg", "Berlin", "Munich", "Frankfurt", "Cologne",
    "Stuttgart", "Dusseldorf", "Leipzig", "Dortmund", "Essen",
    "Bremen", "Dresden", "Hanover", "Nuremberg", "Duisburg",
    "Bochum", "Wuppertal", "Bielefeld", "Bonn", "Munster",
    "Mannheim", "Augsburg", "Wiesbaden", "Aachen", "Kiel",
    "Halle", "Magdeburg", "Freiburg", "Krefeld", "Mainz",
    "Rostock", "Kassel", "Erfurt", "Potsdam", "Saarbrucken",
    "Oldenburg", "Regensburg", "Heidelberg", "Darmstadt", "Wurzburg",
    "Wolfsburg", "Ulm", "Jena", "Trier", "Cottbus",
    "Siegen", "Hildesheim", "Salzgitter", "Chemnitz", "Zwickau",
]


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


def generate_data(n_lines, rng):
    """Generate 1BRC-style text buffer and newline positions."""
    lines = []
    for _ in range(n_lines):
        station = STATION_NAMES[rng.integers(0, len(STATION_NAMES))]
        sign = "-" if rng.random() < 0.2 else ""
        int_part = rng.integers(0, 50)
        dec_part = rng.integers(0, 10)
        lines.append(f"{station};{sign}{int_part}.{dec_part}")

    text = "\n".join(lines) + "\n"
    text_bytes = text.encode("ascii")

    nl_positions = []
    for i, b in enumerate(text_bytes):
        if b == ord("\n"):
            nl_positions.append(i)

    text_arr = np.frombuffer(text_bytes, dtype=np.uint8).copy()
    nl_arr = np.array(nl_positions[:n_lines], dtype=np.int32)

    return text_arr, nl_arr


def alloc_hash_table():
    """Allocate and zero-initialize hash table arrays."""
    ht_keys = np.zeros(TABLE_SIZE * KEY_STRIDE, dtype=np.uint8)
    ht_key_len = np.zeros(TABLE_SIZE, dtype=np.int32)
    ht_min = np.zeros(TABLE_SIZE, dtype=np.int32)
    ht_max = np.zeros(TABLE_SIZE, dtype=np.int32)
    ht_sum = np.zeros(TABLE_SIZE, dtype=np.int32)
    ht_count = np.zeros(TABLE_SIZE, dtype=np.int32)
    out_n = np.zeros(1, dtype=np.int32)
    return ht_keys, ht_key_len, ht_min, ht_max, ht_sum, ht_count, out_n


def call_func(func, text_arr, nl_arr, n_lines, text_start):
    """Call parse_aggregate function and return hash table results."""
    ht_keys, ht_key_len, ht_min, ht_max, ht_sum, ht_count, out_n = alloc_hash_table()

    func(
        text_arr.ctypes.data_as(U8_PTR),
        nl_arr.ctypes.data_as(I32_PTR),
        I32(n_lines),
        I32(text_start),
        ht_keys.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ht_key_len.ctypes.data_as(I32_PTR),
        ht_min.ctypes.data_as(I32_PTR),
        ht_max.ctypes.data_as(I32_PTR),
        ht_sum.ctypes.data_as(I32_PTR),
        ht_count.ctypes.data_as(I32_PTR),
        out_n.ctypes.data_as(I32_PTR),
    )

    return ht_key_len.copy(), ht_min.copy(), ht_max.copy(), ht_sum.copy(), ht_count.copy(), out_n[0]


def check_correctness(ea_results, ref_results, n_lines):
    """Compare hash table outputs between Ea and reference."""
    ea_kl, ea_mn, ea_mx, ea_sm, ea_ct, ea_ns = ea_results
    ref_kl, ref_mn, ref_mx, ref_sm, ref_ct, ref_ns = ref_results

    if ea_ns != ref_ns:
        return f"n_stations mismatch: ea={ea_ns} ref={ref_ns} (N={n_lines})"

    for slot in range(TABLE_SIZE):
        if ref_kl[slot] == 0 and ea_kl[slot] == 0:
            continue
        if ref_kl[slot] != ea_kl[slot]:
            return f"key_len mismatch at slot {slot}: ea={ea_kl[slot]} ref={ref_kl[slot]}"
        if ref_ct[slot] != ea_ct[slot]:
            return f"count mismatch at slot {slot}: ea={ea_ct[slot]} ref={ref_ct[slot]}"
        if ref_mn[slot] != ea_mn[slot]:
            return f"min mismatch at slot {slot}: ea={ea_mn[slot]} ref={ref_mn[slot]}"
        if ref_mx[slot] != ea_mx[slot]:
            return f"max mismatch at slot {slot}: ea={ea_mx[slot]} ref={ref_mx[slot]}"
        if ref_sm[slot] != ea_sm[slot]:
            return f"sum mismatch at slot {slot}: ea={ea_sm[slot]} ref={ref_sm[slot]}"

    return None


def bench_at_size(ea_func, ref_func, n_lines):
    rng = np.random.default_rng(SEED)
    text_arr, nl_arr = generate_data(n_lines, rng)
    text_start = 0

    ea_results = call_func(ea_func, text_arr, nl_arr, n_lines, text_start)
    ref_results = call_func(ref_func, text_arr, nl_arr, n_lines, text_start)

    err = check_correctness(ea_results, ref_results, n_lines)
    if err:
        return err

    for _ in range(WARMUP_RUNS):
        ht_keys, ht_key_len, ht_min, ht_max, ht_sum, ht_count, out_n = alloc_hash_table()
        ea_func(
            text_arr.ctypes.data_as(U8_PTR),
            nl_arr.ctypes.data_as(I32_PTR),
            I32(n_lines),
            I32(text_start),
            ht_keys.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ht_key_len.ctypes.data_as(I32_PTR),
            ht_min.ctypes.data_as(I32_PTR),
            ht_max.ctypes.data_as(I32_PTR),
            ht_sum.ctypes.data_as(I32_PTR),
            ht_count.ctypes.data_as(I32_PTR),
            out_n.ctypes.data_as(I32_PTR),
        )

    times = []
    for _ in range(NUM_RUNS):
        ht_keys, ht_key_len, ht_min, ht_max, ht_sum, ht_count, out_n = alloc_hash_table()
        start = time.perf_counter()
        ea_func(
            text_arr.ctypes.data_as(U8_PTR),
            nl_arr.ctypes.data_as(I32_PTR),
            I32(n_lines),
            I32(text_start),
            ht_keys.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ht_key_len.ctypes.data_as(I32_PTR),
            ht_min.ctypes.data_as(I32_PTR),
            ht_max.ctypes.data_as(I32_PTR),
            ht_sum.ctypes.data_as(I32_PTR),
            ht_count.ctypes.data_as(I32_PTR),
            out_n.ctypes.data_as(I32_PTR),
        )
        times.append(time.perf_counter() - start)

    times.sort()
    median_us = times[len(times) // 2] * 1e6
    min_us = times[0] * 1e6
    return (round(median_us, 1), round(min_us, 1))


def main():
    if len(sys.argv) < 3:
        print("Usage: bench_kernel.py <kernel.ea> <reference.so>", file=sys.stderr)
        sys.exit(1)

    kernel_path = Path(sys.argv[1]).resolve()
    ref_so_path = Path(sys.argv[2]).resolve()
    ea_binary = str(Path(os.environ.get("EA_BINARY", "./target/release/ea")).resolve())

    kernel_dir = kernel_path.parent
    so_name = kernel_path.stem + ".so"
    so_path = kernel_dir / so_name

    for stale in [so_path, Path(so_name), kernel_path.with_suffix(".so")]:
        if stale.exists():
            stale.unlink()

    result = subprocess.run(
        [ea_binary, str(kernel_path), "--lib", "--opt-level=3"],
        capture_output=True, text=True, cwd=str(kernel_dir),
    )
    if result.returncode != 0:
        error_msg = result.stderr.strip() or result.stdout.strip()
        output(False, error=f"compile: {error_msg}")

    if not so_path.exists():
        output(False, error="compile: .so not found after compilation")

    try:
        ea_lib = ctypes.CDLL(str(so_path.resolve()))
        ref_lib = ctypes.CDLL(str(ref_so_path.resolve()))
    except OSError as e:
        output(False, error=f"load: {e}")

    try:
        ea_func = ea_lib.parse_aggregate
        ea_func.argtypes = [U8_PTR, I32_PTR, I32, I32,
                            ctypes.POINTER(ctypes.c_uint8), I32_PTR,
                            I32_PTR, I32_PTR, I32_PTR, I32_PTR, I32_PTR]
        ea_func.restype = None

        ref_func = ref_lib.parse_aggregate_ref
        ref_func.argtypes = [U8_PTR, I32_PTR, I32, I32,
                             ctypes.POINTER(ctypes.c_uint8), I32_PTR,
                             I32_PTR, I32_PTR, I32_PTR, I32_PTR, I32_PTR]
        ref_func.restype = None
    except AttributeError as e:
        output(False, error=f"symbol: {e}")

    breakdown = {}

    for n in LINE_COUNTS:
        label = f"N={n}"
        result = bench_at_size(ea_func, ref_func, n)

        if isinstance(result, str):
            output(False, error=result)

        median_us, min_us = result
        total_bytes = n * BYTES_PER_LINE
        gbs = total_bytes / (median_us / 1e6) / 1e9
        breakdown[label] = {"median_us": median_us, "min_us": min_us, "gbs": round(gbs, 1)}
        print(f"  {label}: {median_us} us median, {min_us} us min  |  {gbs:.1f} GB/s", file=sys.stderr)

    largest_label = f"N={LINE_COUNTS[-1]}"
    aggregate_median = breakdown[largest_label]["median_us"]
    aggregate_min = breakdown[largest_label]["min_us"]
    loc = count_loc(kernel_path)

    output(True, time_us=aggregate_median, min_us=aggregate_min, loc=loc,
           breakdown=breakdown)


if __name__ == "__main__":
    main()
