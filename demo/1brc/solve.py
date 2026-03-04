#!/usr/bin/env python3
"""1BRC solver: Ea SIMD scan -> Ea fused parse+aggregate kernel -> merge -> sort+print"""

import ctypes
import os
import sys
from concurrent.futures import ProcessPoolExecutor


def _load_lib(name: str) -> ctypes.CDLL:
    """Load a shared library from the same directory as this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, f"lib{name}.so")
    return ctypes.CDLL(path)


def _setup_scan(lib: ctypes.CDLL) -> tuple:
    """Configure ctypes signatures for scan kernel."""
    lib.count_lines.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.count_lines.restype = None
    lib.extract_lines.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    lib.extract_lines.restype = None
    return lib


def _setup_parse(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Configure ctypes signatures for parse_temp kernel."""
    lib.batch_parse_temps.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    lib.batch_parse_temps.restype = None
    return lib


def _setup_aggregate(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Configure ctypes signatures for aggregate kernel."""
    lib.parse_aggregate.argtypes = [
        ctypes.c_void_p,                    # text
        ctypes.POINTER(ctypes.c_int),       # nl_pos
        ctypes.c_int,                       # n
        ctypes.c_int,                       # text_start
        ctypes.POINTER(ctypes.c_ubyte),     # ht_keys
        ctypes.POINTER(ctypes.c_int),       # ht_key_len
        ctypes.POINTER(ctypes.c_int),       # ht_min
        ctypes.POINTER(ctypes.c_int),       # ht_max
        ctypes.POINTER(ctypes.c_int),       # ht_sum
        ctypes.POINTER(ctypes.c_int),       # ht_count
        ctypes.POINTER(ctypes.c_int),       # out_n_stations
    ]
    lib.parse_aggregate.restype = None
    return lib


# Globals set per-process (cannot pickle ctypes objects)
_scan_lib = None
_agg_lib = None


def _init_worker() -> None:
    """Initialize kernel libraries in each worker process."""
    global _scan_lib, _agg_lib
    _scan_lib = _setup_scan(_load_lib("scan"))
    _agg_lib = _setup_aggregate(_load_lib("aggregate"))


def solve_chunk(path: str, start: int, end: int) -> dict:
    """Process one chunk of the file. Returns {name_bytes: [min, max, sum, count]}."""
    global _scan_lib, _agg_lib
    if _scan_lib is None:
        _init_worker()

    chunk_len = end - start

    with open(path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(chunk_len)

    buf_ptr = ctypes.cast(ctypes.c_char_p(chunk_bytes), ctypes.c_void_p)

    # Count newlines
    nl_count = ctypes.c_int(0)
    _scan_lib.count_lines(buf_ptr, chunk_len, ctypes.byref(nl_count))
    n_lines = nl_count.value
    if n_lines == 0:
        return {}

    # Extract newline positions
    IntArray = ctypes.c_int * n_lines
    nl_pos = IntArray()
    extract_count = ctypes.c_int(0)
    _scan_lib.extract_lines(buf_ptr, chunk_len, nl_pos, ctypes.byref(extract_count))
    n_lines = extract_count.value

    # Allocate hash table arrays
    TABLE_SIZE = 1024
    KEY_STRIDE = 64
    KeyArray = ctypes.c_ubyte * (TABLE_SIZE * KEY_STRIDE)
    SlotArray = ctypes.c_int * TABLE_SIZE

    ht_keys = KeyArray()
    ht_key_len = SlotArray()
    ht_min = SlotArray(*([9999] * TABLE_SIZE))
    ht_max = SlotArray(*([-9999] * TABLE_SIZE))
    ht_sum = SlotArray()
    ht_count = SlotArray()
    out_n = ctypes.c_int(0)

    # Fused parse + aggregate
    _agg_lib.parse_aggregate(
        buf_ptr, nl_pos, n_lines, 0,
        ht_keys, ht_key_len, ht_min, ht_max, ht_sum, ht_count,
        ctypes.byref(out_n),
    )

    # Read back filled slots
    stations: dict[bytes, list] = {}
    for slot in range(TABLE_SIZE):
        klen = ht_key_len[slot]
        if klen > 0:
            base = slot * KEY_STRIDE
            name = bytes(ht_keys[base : base + klen])
            stations[name] = [ht_min[slot], ht_max[slot], ht_sum[slot], ht_count[slot]]

    return stations


def find_chunk_boundaries(path: str, n_chunks: int) -> list[tuple[int, int]]:
    """Split file into chunks aligned to newline boundaries."""
    size = os.path.getsize(path)
    if n_chunks <= 1:
        return [(0, size)]

    chunk_size = size // n_chunks
    boundaries = []
    with open(path, "rb") as f:
        start = 0
        for _ in range(n_chunks - 1):
            pos = start + chunk_size
            if pos >= size:
                break
            f.seek(pos)
            rest = f.read(256)
            nl_idx = rest.find(b"\n")
            if nl_idx < 0:
                break
            end = pos + nl_idx + 1
            boundaries.append((start, end))
            start = end
        boundaries.append((start, size))
    return boundaries


def merge_results(partials: list[dict]) -> dict:
    """Merge partial station dicts into one."""
    merged: dict[bytes, list] = {}
    for part in partials:
        for name, (mn, mx, sm, ct) in part.items():
            entry = merged.get(name)
            if entry is not None:
                if mn < entry[0]:
                    entry[0] = mn
                if mx > entry[1]:
                    entry[1] = mx
                entry[2] += sm
                entry[3] += ct
            else:
                merged[name] = [mn, mx, sm, ct]
    return merged


def solve(path: str, n_workers: int | None = None) -> dict:
    """Full 1BRC solve: scan -> fused parse+aggregate -> merge."""
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = min(n_workers, os.cpu_count() or 1)

    boundaries = find_chunk_boundaries(path, n_workers)

    if len(boundaries) == 1:
        _init_worker()
        return solve_chunk(path, boundaries[0][0], boundaries[0][1])

    with ProcessPoolExecutor(
        max_workers=len(boundaries), initializer=_init_worker
    ) as pool:
        futures = [
            pool.submit(solve_chunk, path, start, end)
            for start, end in boundaries
        ]
        partials = [f.result() for f in futures]

    return merge_results(partials)


def format_results(results: dict) -> str:
    """Format results as sorted station lines: Name=min/mean/max"""
    lines = []
    for name in sorted(results):
        mn, mx, sm, ct = results[name]
        mean = sm / ct
        lines.append(
            f"{name.decode()}={mn / 10:.1f}/{mean / 10:.1f}/{mx / 10:.1f}"
        )
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <measurements.txt> [n_workers]")
        sys.exit(1)

    path = sys.argv[1]
    n_workers = int(sys.argv[2]) if len(sys.argv) >= 3 else None

    results = solve(path, n_workers)
    print(format_results(results))


if __name__ == "__main__":
    main()
