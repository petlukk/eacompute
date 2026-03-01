# eavec — SIMD Vector Search Kernels

Eä kernels for embedding similarity search: dot product, cosine similarity, L2 distance.
Dual-accumulator FMA with f32x8 (AVX2), benchmarked against NumPy and FAISS.

## Results

**Headline: 3–4x faster than FAISS at dim=384** — the most common embedding dimension
(all-MiniLM-L6, e5-small, many production models).

batch_dot / inner product, single-threaded:

| dim  | N     | Eä ST    | NumPy    | FAISS    | vs NumPy | vs FAISS |
|------|-------|----------|----------|----------|----------|----------|
| 384  | 10K   | 0.3 ms   | 0.3 ms   | 1.4 ms   | 0.8x     | 4.3x     |
| 384  | 100K  | 6.0 ms   | 24.5 ms  | 25.7 ms  | 4.0x     | 4.2x     |
| 384  | 500K  | 40.9 ms  | 83.4 ms  | 190.1 ms | 1.5x     | 3.4x     |
| 768  | 100K  | 15.0 ms  | 40.3 ms  | 45.5 ms  | 1.8x     | 2.1x     |
| 1536 | 100K  | 28.7 ms  | 48.0 ms  | 56.0 ms  | 1.3x     | 1.5x     |

**Honest note on dim=384, N=10K:** Eä is 0.8x NumPy. At that size, NumPy's BLAS
(OpenBLAS/MKL) is highly optimized for small matrix-vector products and Eä's kernel
entry cost is visible. The advantage appears at 100K+ vectors where the inner loop
dominates.

**Scaling story:** Eä's advantage is largest at dim=384 (compute-bound) and narrows
at dim=1536. At 1536 dimensions with 100K vectors you're moving 580MB of data — the
bottleneck shifts from compute to memory bandwidth, and 1.5x over FAISS is still a
win but the ceiling is lower.

**FAISS overhead:** FAISS' `IndexFlatIP.search()` carries dispatch cost — index
bookkeeping, thread pool management, result sorting — even in single-threaded mode.
Eä's kernel is the raw dot product with no framework around it. That dispatch cost
is a larger fraction at small N (4.3x at 10K) and shrinks at large N (3.4x at 500K)
as the actual compute dominates.

### Metric comparison (dim=768, N=100K)

| Metric | Eä ST    | NumPy     | FAISS    | vs NumPy | vs FAISS |
|--------|----------|-----------|----------|----------|----------|
| dot    | 12.2 ms  | 25.6 ms   | 36.7 ms  | 2.1x     | 3.0x     |
| cosine | 12.5 ms  | 322.0 ms  | —        | 25.8x    | —        |
| l2_sq  | 20.2 ms  | 570.8 ms  | 33.7 ms  | 28.3x    | 1.7x     |

The 25.8x cosine speedup vs NumPy is an architectural win, not a kernel efficiency
win. NumPy computes cosine as `dot(a,b) / (norm(a) * norm(b))` — three separate
passes over the data. Eä's `batch_cosine` fuses the dot product and both norms
into a single pass with 4 accumulators, reading each vector exactly once. FAISS
handles this by requiring pre-normalization + IndexFlatIP (the standard approach),
which is why there's no direct cosine comparison.

### Multi-threaded

MT results on this machine (WSL2, 1 vCPU) are mostly slower than ST due to thread
pool overhead with no actual parallelism. On bare metal with multiple cores, the
ctypes calls release the GIL and threads run truly in parallel — the Sobel demo
shows the scaling pattern.

## How it works

**Dual-accumulator FMA** — two independent f32x8 accumulator chains per inner loop.
Superscalar CPUs execute both FMA pipelines in parallel (ILP), doubling throughput
over a single accumulator.

**`*restrict` pointers** — tells LLVM that query and database pointers don't alias,
enabling vectorization without runtime alias checks.

**Single-pass cosine** — `cosine_f32` computes dot product, `||a||²`, and `||b||²`
simultaneously using 6 accumulators, reading each vector exactly once.

**Pre-normalization trick** — for pre-normalized vectors (unit length), `batch_dot`
= cosine similarity. This is the standard FAISS `IndexFlatIP` approach.

## Correctness

Eä validated against both NumPy and FAISS with max absolute error < 1e-4:

```
Ea vs NumPy dot:    max diff = 3.35e-08  MATCH
Ea vs NumPy cosine: max diff = 3.73e-08  MATCH
Ea vs NumPy L2:     max diff = 4.77e-07  MATCH
Ea vs FAISS IP:     max diff = 2.24e-08  MATCH
Ea vs FAISS L2:     max diff = 2.38e-07  MATCH
top_k:              overlap = 10/10      MATCH
```

## Kernels

### `similarity.ea` — single-pair primitives

| Function     | Returns | Description                                      |
|-------------|---------|--------------------------------------------------|
| `dot_f32`   | f32     | Dot product, dual-acc FMA                        |
| `norm_f32`  | f32     | L2 norm (sqrt of sum of squares)                 |
| `cosine_f32`| f32     | Cosine similarity, single-pass 6-accumulator     |
| `l2_sq_f32` | f32     | Squared L2 distance                              |

### `search.ea` — batch operations

| Function             | Description                                         |
|---------------------|-----------------------------------------------------|
| `batch_dot`         | Query vs N vectors, FMA inner loop                  |
| `batch_cosine`      | Query vs N vectors, precomputed query norm          |
| `batch_l2`          | Query vs N vectors, fused diff-squared accumulation |
| `normalize_vectors` | In-place L2 normalization                           |
| `threshold_filter`  | Stream compaction: indices where score > threshold  |
| `top_k`             | K passes of SIMD max-finding                       |

## Usage

```bash
# Build
bash build.sh

# Run benchmark
python3 run.py                    # full scaling benchmark (vs NumPy + FAISS)
python3 run.py --quick            # dim=384, N=10K only
python3 run.py --threads 8        # multi-threaded Eä
python3 run.py --no-faiss         # skip FAISS comparison
```

## Requirements

- Eä compiler (built from repo root)
- Python 3, NumPy
- Optional: FAISS (`pip install faiss-cpu`)
