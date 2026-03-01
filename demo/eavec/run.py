#!/usr/bin/env python3
"""
Vector Search Benchmark: Eä vs NumPy vs FAISS

Benchmarks cosine similarity, dot product, and L2 distance on realistic
embedding dimensions (384, 768, 1536) and database sizes (10K–1M vectors).

Eä kernels use dual-accumulator FMA with f32x8 SIMD — the same pattern
that powers production vector databases.

Usage:
    python run.py                    # full benchmark
    python run.py --quick            # dim=384, N=10K only
    python run.py --threads 8        # MT with 8 threads
    python run.py --no-faiss         # skip FAISS comparison
"""

import argparse
import concurrent.futures
import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_db(n_vecs, dim, seed=42):
    """Generate random float32 vectors, L2-normalized."""
    rng = np.random.Generator(np.random.PCG64(seed))
    db = rng.standard_normal((n_vecs, dim), dtype=np.float32)
    norms = np.linalg.norm(db, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    db /= norms
    return np.ascontiguousarray(db)


def generate_query(dim, seed=123):
    """Generate a single random normalized query vector."""
    rng = np.random.Generator(np.random.PCG64(seed))
    q = rng.standard_normal(dim, dtype=np.float32)
    q /= np.linalg.norm(q)
    return np.ascontiguousarray(q)


# ---------------------------------------------------------------------------
# Eä wrappers
# ---------------------------------------------------------------------------

def ea_batch_dot(query, db, n_vecs, dim):
    """Eä batch dot product via generated bindings."""
    import search as _search
    out = np.empty(n_vecs, dtype=np.float32)
    _search.batch_dot(query, db.ravel(), dim, n_vecs, out)
    return out


def ea_batch_cosine(query, db, n_vecs, dim):
    """Eä batch cosine similarity."""
    import search as _search
    out = np.empty(n_vecs, dtype=np.float32)
    _search.batch_cosine(query, db.ravel(), dim, n_vecs, out)
    return out


def ea_batch_l2(query, db, n_vecs, dim):
    """Eä batch L2 squared distance."""
    import search as _search
    out = np.empty(n_vecs, dtype=np.float32)
    _search.batch_l2(query, db.ravel(), dim, n_vecs, out)
    return out


def ea_batch_dot_mt(query, db, n_vecs, dim, num_threads):
    """Multi-threaded Eä batch dot — partition vectors across threads."""
    import search as _search
    lib = _search._lib
    out = np.empty(n_vecs, dtype=np.float32)
    flat_db = np.ascontiguousarray(db.ravel())

    q_ptr = query.ctypes.data
    db_ptr = flat_db.ctypes.data
    out_ptr = out.ctypes.data
    sizeof_f32 = 4

    strips = []
    per = n_vecs // num_threads
    rem = n_vecs % num_threads
    start = 0
    for t in range(num_threads):
        count = per + (1 if t < rem else 0)
        if count == 0:
            continue
        strips.append((start, count))
        start += count

    def process_strip(s_start, s_count):
        db_off = s_start * dim * sizeof_f32
        out_off = s_start * sizeof_f32
        lib.batch_dot(
            ctypes.cast(q_ptr, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(db_ptr + db_off, ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(dim),
            ctypes.c_int32(s_count),
            ctypes.cast(out_ptr + out_off, ctypes.POINTER(ctypes.c_float)),
        )

    pool = _get_pool(num_threads)
    futures = [pool.submit(process_strip, *s) for s in strips]
    for f in concurrent.futures.as_completed(futures):
        f.result()

    return out


# ---------------------------------------------------------------------------
# NumPy baselines
# ---------------------------------------------------------------------------

def np_batch_dot(query, db):
    """NumPy batch dot product: db @ query."""
    return db @ query


def np_batch_cosine(query, db):
    """NumPy batch cosine similarity."""
    dots = db @ query
    db_norms = np.linalg.norm(db, axis=1)
    q_norm = np.linalg.norm(query)
    denom = db_norms * q_norm
    denom[denom == 0] = 1.0
    return dots / denom


def np_batch_l2(query, db):
    """NumPy batch L2 squared distance."""
    diff = db - query[np.newaxis, :]
    return np.sum(diff * diff, axis=1)


# ---------------------------------------------------------------------------
# FAISS wrappers
# ---------------------------------------------------------------------------

def faiss_build_ip(db):
    """Build FAISS IndexFlatIP (inner product). Separate from search timing."""
    import faiss
    faiss.omp_set_num_threads(1)
    dim = db.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(db))
    return index


def faiss_build_l2(db):
    """Build FAISS IndexFlatL2. Separate from search timing."""
    import faiss
    faiss.omp_set_num_threads(1)
    dim = db.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.ascontiguousarray(db))
    return index


def faiss_search(index, query, k):
    """Search a pre-built FAISS index. This is what gets timed."""
    D, I = index.search(query.reshape(1, -1), k)
    return D[0], I[0]


# ---------------------------------------------------------------------------
# Thread pool
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def bench(func, warmup=5, runs=20):
    """Benchmark function, return median time in seconds."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def run_correctness(dim, n_vecs, num_threads, use_faiss=True):
    """Validate Eä results against NumPy and FAISS."""
    print("=== Correctness ===\n")
    print(f"  dim={dim}, n_vecs={n_vecs:,}\n")

    db = generate_db(n_vecs, dim)
    query = generate_query(dim)

    # Batch dot
    ea_dot = ea_batch_dot(query, db, n_vecs, dim)
    np_dot = np_batch_dot(query, db)
    diff_dot = np.max(np.abs(ea_dot - np_dot))
    print(f"  Ea vs NumPy dot:    max diff = {diff_dot:.2e}  ", end="")
    print("MATCH" if diff_dot < 1e-4 else f"FAIL (max {diff_dot:.4f})")

    # Batch cosine
    ea_cos = ea_batch_cosine(query, db, n_vecs, dim)
    np_cos = np_batch_cosine(query, db)
    diff_cos = np.max(np.abs(ea_cos - np_cos))
    print(f"  Ea vs NumPy cosine: max diff = {diff_cos:.2e}  ", end="")
    print("MATCH" if diff_cos < 1e-4 else f"FAIL (max {diff_cos:.4f})")

    # Batch L2
    ea_l2 = ea_batch_l2(query, db, n_vecs, dim)
    np_l2 = np_batch_l2(query, db)
    diff_l2 = np.max(np.abs(ea_l2 - np_l2))
    print(f"  Ea vs NumPy L2:     max diff = {diff_l2:.2e}  ", end="")
    print("MATCH" if diff_l2 < 1e-4 else f"FAIL (max {diff_l2:.4f})")

    # FAISS correctness
    if use_faiss:
        try:
            index_ip = faiss_build_ip(db)
            faiss_d, _ = faiss_search(index_ip, query, n_vecs)
            # FAISS returns scores sorted descending; compare unsorted values
            faiss_sorted = np.sort(faiss_d)
            ea_sorted = np.sort(ea_dot)
            diff_faiss = np.max(np.abs(faiss_sorted - ea_sorted))
            print(f"  Ea vs FAISS IP:     max diff = {diff_faiss:.2e}  ", end="")
            print("MATCH" if diff_faiss < 1e-4 else f"FAIL (max {diff_faiss:.4f})")

            index_l2 = faiss_build_l2(db)
            faiss_l2d, _ = faiss_search(index_l2, query, n_vecs)
            faiss_l2_sorted = np.sort(faiss_l2d)
            ea_l2_sorted = np.sort(ea_l2)
            diff_faiss_l2 = np.max(np.abs(faiss_l2_sorted - ea_l2_sorted))
            print(f"  Ea vs FAISS L2:     max diff = {diff_faiss_l2:.2e}  ", end="")
            print("MATCH" if diff_faiss_l2 < 1e-4 else f"FAIL (max {diff_faiss_l2:.4f})")
        except ImportError:
            print("  FAISS: not installed")

    # MT vs ST
    ea_dot_mt = ea_batch_dot_mt(query, db, n_vecs, dim, num_threads)
    diff_mt = np.max(np.abs(ea_dot_mt - ea_dot))
    print(f"  Ea MT vs ST:        max diff = {diff_mt:.2e}  ({num_threads} threads)  ", end="")
    print("MATCH" if diff_mt < 1e-6 else f"MISMATCH (max {diff_mt:.4f})")

    # Top-k overlap
    import search as _search
    k = min(10, n_vecs)
    work = ea_dot.copy()
    out_vals = np.empty(k, dtype=np.float32)
    out_idx = np.empty(k, dtype=np.int32)
    _search.top_k(ea_dot, work, k, out_vals, out_idx)
    np_topk = np.argpartition(-np_dot, k)[:k]
    overlap = len(set(out_idx.tolist()) & set(np_topk.tolist()))
    print(f"  top_k:              overlap = {overlap}/{k}  ", end="")
    print("MATCH" if overlap == k else "PARTIAL")

    print()


# ---------------------------------------------------------------------------
# Scaling benchmark
# ---------------------------------------------------------------------------

def run_benchmark(configs, num_threads, use_faiss):
    """Run scaling benchmark across dims and sizes."""
    print("=== Performance (batch_dot / inner product) ===\n")
    print(f"  5 warmup, 20 runs, median. Eä MT = {num_threads} threads.")
    if use_faiss:
        print(f"  FAISS: IndexFlatIP, omp_threads=1 (search only, index pre-built).")
    print()

    has_faiss = False
    if use_faiss:
        try:
            import faiss
            has_faiss = True
        except ImportError:
            print("  FAISS not installed, skipping. Install: pip install faiss-cpu\n")

    header = f"  {'dim':>5} {'N':>10} {'Ea ST':>10} {'Ea MT':>10} {'NumPy':>10}"
    if has_faiss:
        header += f" {'FAISS':>10}"
    header += f" {'MT/ST':>7} {'vs np':>7}"
    if has_faiss:
        header += f" {'vs fss':>7}"
    header += f" {'Mvec/s':>8}"
    print(header)

    sep = f"  {'─'*5} {'─'*10} {'─'*10} {'─'*10} {'─'*10}"
    if has_faiss:
        sep += f" {'─'*10}"
    sep += f" {'─'*7} {'─'*7}"
    if has_faiss:
        sep += f" {'─'*7}"
    sep += f" {'─'*8}"
    print(sep)

    results = []
    for dim, n_vecs in configs:
        db = generate_db(n_vecs, dim)
        query = generate_query(dim)

        t_st = bench(lambda: ea_batch_dot(query, db, n_vecs, dim))
        t_mt = bench(lambda: ea_batch_dot_mt(query, db, n_vecs, dim, num_threads))
        t_np = bench(lambda: np_batch_dot(query, db))

        t_faiss = None
        if has_faiss:
            index_ip = faiss_build_ip(db)
            t_faiss = bench(lambda: faiss_search(index_ip, query, n_vecs))

        mvec = n_vecs / t_mt / 1e6

        row = f"  {dim:>5} {n_vecs:>10,} {t_st*1000:>8.2f}ms {t_mt*1000:>8.2f}ms {t_np*1000:>8.2f}ms"
        if has_faiss:
            row += f" {t_faiss*1000:>8.2f}ms"
        row += f" {t_st/t_mt:>6.1f}x {t_np/t_mt:>6.1f}x"
        if has_faiss and t_faiss:
            row += f" {t_faiss/t_mt:>6.1f}x"
        row += f" {mvec:>7.1f}"
        print(row)

        results.append({
            'dim': dim, 'n_vecs': n_vecs,
            'st_ms': t_st * 1000, 'mt_ms': t_mt * 1000,
            'np_ms': t_np * 1000, 'faiss_ms': t_faiss * 1000 if t_faiss else None,
            'mt_speedup': t_st / t_mt, 'vs_np': t_np / t_mt,
            'vs_faiss': (t_faiss / t_mt) if t_faiss else None,
            'mvec_s': mvec,
        })

    print()
    return results


def run_metric_comparison(dim, n_vecs, num_threads, use_faiss=True):
    """Compare performance across all three distance metrics."""
    print("=== Metric Comparison ===\n")
    print(f"  dim={dim}, n_vecs={n_vecs:,}\n")

    db = generate_db(n_vecs, dim)
    query = generate_query(dim)

    has_faiss = False
    if use_faiss:
        try:
            import faiss
            has_faiss = True
        except ImportError:
            pass

    t_dot = bench(lambda: ea_batch_dot(query, db, n_vecs, dim))
    t_cos = bench(lambda: ea_batch_cosine(query, db, n_vecs, dim))
    t_l2 = bench(lambda: ea_batch_l2(query, db, n_vecs, dim))

    t_np_dot = bench(lambda: np_batch_dot(query, db))
    t_np_cos = bench(lambda: np_batch_cosine(query, db))
    t_np_l2 = bench(lambda: np_batch_l2(query, db))

    t_f_dot = None
    t_f_l2 = None
    if has_faiss:
        index_ip = faiss_build_ip(db)
        index_l2 = faiss_build_l2(db)
        t_f_dot = bench(lambda: faiss_search(index_ip, query, n_vecs))
        t_f_l2 = bench(lambda: faiss_search(index_l2, query, n_vecs))

    header = f"  {'Metric':<12} {'Ea ST':>10} {'NumPy':>10}"
    if has_faiss:
        header += f" {'FAISS':>10}"
    header += f" {'vs np':>7}"
    if has_faiss:
        header += f" {'vs fss':>7}"
    print(header)

    sep = f"  {'─'*12} {'─'*10} {'─'*10}"
    if has_faiss:
        sep += f" {'─'*10}"
    sep += f" {'─'*7}"
    if has_faiss:
        sep += f" {'─'*7}"
    print(sep)

    def metric_row(name, t_ea, t_np, t_f):
        row = f"  {name:.<12} {t_ea*1000:>8.2f}ms {t_np*1000:>8.2f}ms"
        if has_faiss and t_f is not None:
            row += f" {t_f*1000:>8.2f}ms"
        elif has_faiss:
            row += f" {'n/a':>10}"
        row += f" {t_np/t_ea:>6.1f}x"
        if has_faiss and t_f is not None:
            row += f" {t_f/t_ea:>6.1f}x"
        elif has_faiss:
            row += f" {'':>7}"
        return row

    print(metric_row('dot', t_dot, t_np_dot, t_f_dot))
    print(metric_row('cosine', t_cos, t_np_cos, None))
    print(metric_row('l2_sq', t_l2, t_np_l2, t_f_l2))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Vector search benchmark: Eä vs NumPy vs FAISS')
    parser.add_argument('--quick', action='store_true',
                        help='dim=384, N=10K only')
    parser.add_argument('--threads', type=int, default=None,
                        help='Thread count for MT (default: all cores)')
    parser.add_argument('--no-faiss', action='store_true',
                        help='Skip FAISS comparison')
    args = parser.parse_args()
    num_threads = args.threads or os.cpu_count() or 4

    print("Vector Search — Eä Kernel Demo")
    print("=" * 60)
    print(f"  SIMD: dual-accumulator FMA, f32x8 (AVX2)")
    print(f"  MT: {num_threads} threads (GIL released via ctypes)")
    print()

    use_faiss = not args.no_faiss

    # Correctness on small data
    run_correctness(384, 1000, num_threads, use_faiss=use_faiss)

    # Metric comparison
    if args.quick:
        run_metric_comparison(384, 10_000, num_threads, use_faiss=use_faiss)
    else:
        run_metric_comparison(768, 100_000, num_threads, use_faiss=use_faiss)

    # Scaling benchmark
    if args.quick:
        configs = [(384, 10_000)]
    else:
        configs = [
            (384, 10_000),
            (384, 100_000),
            (384, 500_000),
            (768, 10_000),
            (768, 100_000),
            (1536, 10_000),
            (1536, 100_000),
        ]

    results = run_benchmark(configs, num_threads, use_faiss=use_faiss)

    # Summary
    print("=== Summary ===\n")
    for r in results:
        line = (f"  dim={r['dim']:>4}  N={r['n_vecs']:>10,}  "
                f"Ea MT: {r['mt_ms']:.1f}ms  "
                f"{r['vs_np']:.1f}x vs NumPy")
        if r.get('vs_faiss') is not None:
            line += f"  {r['vs_faiss']:.1f}x vs FAISS"
        line += f"  {r['mvec_s']:.1f} Mvec/s"
        print(line)
    print()
    print(f"  FMA + dual accumulators + restrict pointers → no-alias SIMD loops")
    print(f"  Pre-normalized data: batch_dot = cosine similarity (standard FAISS trick)")
    if use_faiss:
        print(f"  FAISS: IndexFlatIP brute-force, 1 OMP thread (search only, index pre-built)")
    print()


if __name__ == "__main__":
    main()
