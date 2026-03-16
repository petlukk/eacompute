# Eä Autoresearch

Automated kernel optimization loop inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch). An AI agent iteratively rewrites Eä SIMD kernels to find the fastest correct implementation.

## How It Works

A bash orchestrator runs a modify → compile → benchmark → keep/discard loop:

1. **Agent turn** — Claude Code CLI (`claude -p`) receives the current best kernel, performance history, and Eä language rules. It proposes a single optimization with a stated hypothesis and outputs a new kernel.
2. **Compile** — The Eä compiler builds the kernel to a shared library (`--lib --opt-level=3`).
3. **Benchmark** — The kernel runs 50 times across multiple dataset sizes (real-world image/text dimensions up to DRAM-exceeding scale) to prevent overfitting to cache behavior. Correctness is verified at each size against a hand-written C reference. **The primary metric is the time at the largest size** — this prevents optimizations that win in cache but regress at real-world scale (see "Cache Overspill" below).
4. **Evaluate** — The new kernel is accepted only if the largest-size improvement is at least 0.5% (or equal speed with fewer lines of code). This threshold prevents accepting noise or size-specific optimizations.
5. **Log & repeat** — Results are appended to `history.json`. The last 10 attempts are fed back to the agent so it can learn from what worked and what didn't.

The agent has no control over measurement. It can only produce kernel source code.

## Running

```bash
# Default: 20 iterations, 5-minute timeout per iteration
TIMEOUT=300 MAX_ITERATIONS=5 bash autoresearch/orchestrator_matmul.sh

# IMPORTANT: Run one orchestrator at a time.
# Parallel claude -p calls cause timeouts due to API rate limits.
```

Requires: Eä compiler (built automatically), GCC, Python 3 with numpy, Claude Code CLI.

### Available Benchmarks

**Float compute kernels:**
```bash
bash autoresearch/orchestrator.sh              # FMA (throughput-bound)
bash autoresearch/orchestrator_reduction.sh    # Horizontal reduction (latency-bound)
bash autoresearch/orchestrator_dot_product.sh  # Dot product (hybrid)
bash autoresearch/orchestrator_saxpy.sh        # SAXPY (bandwidth-bound)
bash autoresearch/orchestrator_clamp.sh        # Clamp (select/masking)
bash autoresearch/orchestrator_matmul.sh       # Matrix multiply (compute-bound)
bash autoresearch/orchestrator_prefix_sum.sh   # Prefix sum (sequential)
bash autoresearch/orchestrator_histogram.sh    # Histogram (random access)
bash autoresearch/orchestrator_frame_stats.sh  # Triple reduction min/max/sum (bandwidth-bound)
```

**Demo-derived kernels:**
```bash
bash autoresearch/orchestrator_sobel.sh                # Sobel edge detection (stencil)
bash autoresearch/orchestrator_conv2d_3x3.sh           # 3x3 int8 convolution (maddubs)
bash autoresearch/orchestrator_cornell_box.sh          # Ray tracer (compute-bound)
bash autoresearch/orchestrator_particle_life.sh        # N-body simulation (O(N²))
bash autoresearch/orchestrator_video_anomaly.sh        # Frame diff+threshold+count
bash autoresearch/orchestrator_batch_dot.sh            # Batch dot product (eavec)
bash autoresearch/orchestrator_batch_cosine.sh         # Batch cosine similarity (eavec)
bash autoresearch/orchestrator_preprocess_fused.sh     # MNIST normalize+standardize+clip
bash autoresearch/orchestrator_threshold_u8.sh         # Binary threshold (u8x32)
bash autoresearch/orchestrator_dot_u8i8.sh             # Int8 dot product (maddubs)
bash autoresearch/orchestrator_text_prepass.sh         # Fused classify+lowercase+boundary
bash autoresearch/orchestrator_edge_detect_fused.sh    # Fused blur+sobel+threshold
bash autoresearch/orchestrator_parse_aggregate.sh      # 1BRC fused parse+hash aggregate
```

### Iteration Budget

| Iterations | Phase | What to expect |
|------------|-------|----------------|
| 5–10 | Smoke test | Verifies the loop works. Hill-climbing finds the obvious wins (unroll, multi-accumulator). |
| 10–30 | Micro-optimization | Diminishing returns. Agent explores prefetch distances, stride tuning, instruction ordering. Most attempts rejected. |
| 100+ | Discovery | Agent exhausts textbook tricks and may try combinations or approaches a human wouldn't consider. Uncharted territory. |

All 22 benchmarks have completed initial 5-iteration runs with the largest-size metric.

## Results

All results on AMD EPYC 9354P. **Every kernel scored on the largest (real-world) dataset size** — no cache-fitting shortcuts. 5-iteration runs.

### Full Results Table

| Kernel | Largest size | Baseline (µs) | Best (µs) | **Gain** | Key optimization |
|--------|-------------|---------------|-----------|---------|-----------------|
| matmul | 256×256 | 751 | 330 | **56%** | k×8 unroll, fewer prefetches |
| conv2d_3x3 | 128×128×C64 | 515 | 276 | **47%** | 4x col unroll + prefetch + restrict |
| edge_detect_fused | 2048×2048 | 11,498 | 6,773 | **41%** | f32x4 → f32x8 width doubling |
| parse_aggregate | 10M lines | 340,154 | 199,785 | **41%** | Fixed-pattern temp parsing |
| clamp | 16M floats | 4,371 | 3,330 | **24%** | stream_store + prefetch + separated loads |
| dot_u8i8 | N=65536 | 2.89 | 2.21 | **24%** | 2 independent maddubs accumulators |
| threshold_u8 | 4096×4096 | 874 | 682 | **22%** | u8x16 → u8x32 + stream_store |
| batch_dot | 768×10K | 907 | 735 | **19%** | Tuned prefetch for 3KB stride |
| preprocess_fused | 47M pixels | 13,158 | 11,357 | **14%** | Precomputed constants + 4x unroll + stream_store |
| FMA | 16M floats | 227 | 205 | **10%** | 12x unroll + stream_store |
| video_anomaly | 4M floats | 864 | 795 | **8%** | max() abs pattern replaces select |
| text_prepass | 738KB text | 234 | 215 | **8%** | Split classify/boundary loops + stream_store |
| reduction | 16M floats | 1,608 | 1,559 | **3%** | 8 loads per iteration |
| cornell_box | 512×512 | 9,572 | 9,380 | **2%** | rsqrt replaces sqrt+div |
| sobel | 2048×2048 | 2,314 | 2,277 | **2%** | FMA fusion |
| dot_product | 16M floats | 3,517 | 3,482 | **1%** | Removed prefetch overhead |
| frame_stats | 4096×4096 | 2,103 | 2,103 | **0%** | Bandwidth-bound (triple reduction) |
| particle_life | N=2000 | 5,420 | 5,420 | **0%** | Already tight scalar code |
| batch_cosine | 768×10K | 878 | 878 | **0%** | Bandwidth-bound |
| SAXPY | 16M floats | — | — | **0%** | Bandwidth-bound |
| prefix_sum | 16M floats | — | — | **0%** | Sequential dependency |
| histogram | 16M floats | — | — | **0%** | Random access pattern |

### What Works (and What Doesn't)

**Strong wins (>20%) come from three sources:**
1. **SIMD width increases** — f32x4→f32x8 gave 41% on edge detection, u8x16→u8x32 gave 22% on threshold
2. **Algorithmic restructuring** — fixed-pattern parsing eliminated inner loops for 41% on 1BRC, 4x col unroll gave 47% on conv2d
3. **stream_store on write-only outputs** — 24% on clamp, 22% on threshold (but catastrophic on read-modify-write like SAXPY)

**What doesn't work at real-world scale:**
- **Multi-accumulator on bandwidth-bound kernels** — reduction showed "61% gain" at small sizes but only 3% at 16M (DRAM-bound). The old metric hid this.
- **Prefetch on sequential access** — hardware prefetch is already effective. Manual prefetch mostly adds overhead, except for strided access patterns (batch_dot 3KB stride).
- **Any optimization on pure bandwidth-bound kernels** — SAXPY, frame_stats, batch_cosine all hit ~30 GB/s wall. Nothing helps.

### Corrected Claims

Several earlier results were measured with median-of-sizes (now replaced by largest-size scoring):

| Kernel | Old claim | Honest result | What happened |
|--------|----------|---------------|---------------|
| Reduction | 61% | 3% | Latency hiding helps in L2/L3 but DRAM bandwidth is the bottleneck at 16M |
| Dot product | 35% | 1% | Same — multi-accumulator wins in cache, irrelevant at DRAM scale |
| Sobel | 68% | 2% | f32x4→f32x8 + 2-row unroll helped in cache; bandwidth-bound at 2048² |
| Cornell box | 41% | 2% | Most gains were from prior optimization; little room left at 512² |

## Cache Overspill

**Problem discovered 2026-03-16:** The original benchmark metric (median-of-sizes) hid severe regressions at real-world data sizes. A `frame_stats` kernel with 4 accumulators showed "25% gain" at 1M elements (fits L3) but was **31% slower** at 16M elements (64MB, DRAM-bound). The median picked the 1M result, hiding the DRAM regression.

**Fix:** All benchmarks now score on the **largest** dataset size, which exceeds cache and represents real-world usage. Optimizations that only win at cache-fitting sizes are rejected.

**Rule:** If data fits in cache during benchmarking but won't in production, the benchmark is lying. Always score on production-scale data.

| Size | Cache tier | Risk |
|------|-----------|------|
| < 256KB | L2 | High — everything is fast in L2 |
| 256KB – 4MB | L3 | Medium — latency tricks help but may not scale |
| > 10MB | DRAM | Low — this is the truth |

## AVX-512 vs AVX2

**AVX-512 is not free.** Wider SIMD can cause frequency throttling (Intel), register pressure, and code bloat. Even on AMD Zen 4 (which doesn't downclock), AVX-512 loses on nearly half the kernels.

Run the comparison:
```bash
python3 autoresearch/avx512_comparison.py
```

Results on AMD EPYC 9354P (Zen 4), scored on largest dataset size:

| Kernel | AVX2 (µs) | AVX-512 (µs) | Diff | Winner |
|--------|-----------|-------------|------|--------|
| dot_product | 4963 | 3878 | **+22%** | AVX-512 |
| preprocess_fused | 13023 | 11491 | **+12%** | AVX-512 |
| clamp | 4608 | 4370 | +5% | AVX-512 |
| batch_cosine | 996 | 947 | +5% | AVX-512 |
| frame_stats | 2411 | 2317 | +4% | AVX-512 |
| fma | 9163 | 8894 | +3% | AVX-512 |
| saxpy | 4921 | 4816 | +2% | AVX-512 |
| batch_dot | 767 | 803 | **-5%** | AVX2 |
| edge_detect_fused | 6670 | 7296 | **-9%** | AVX2 |
| sobel | 2333 | 2789 | **-20%** | AVX2 |
| reduction | 1979 | 2385 | **-21%** | AVX2 |
| video_anomaly | 770 | 943 | **-23%** | AVX2 |
| threshold_u8 | 675 | 850 | **-26%** | AVX2 |

**Score: AVX-512 wins 7, AVX2 wins 6.**

**Pattern:** AVX-512 helps on **broad linear sweeps** (dot_product, preprocess_fused, clamp) where doubling vector width directly doubles throughput. AVX-512 **hurts** on:
- **Stencil kernels** (sobel -20%, edge_detect -9%) — wider vectors process more columns but stencil neighbors don't align, causing extra shuffles
- **Already-optimized AVX2 kernels** (threshold_u8 -26%, video_anomaly -23%) — hand-tuned u8x32/f32x8 code with stream_store doesn't benefit from wider vectors
- **Multi-accumulator reductions** (reduction -21%) — 4 accumulators × 512-bit = 8 ZMM registers, leaving less room for other temporaries

**Takeaway:** Don't assume `--avx512` is faster. Benchmark both. The autoresearch agent should explore width as an optimization dimension — the benchmark decides.

## Loop A: Compiler Optimization

Loop A modifies the Eä compiler itself (Rust source) to improve codegen quality. All 22 Loop B benchmarks serve as a regression gate — compiler changes are only accepted if no kernel gets slower and at least one gets faster.

### Pipeline

1. **Agent turn** — Claude Code CLI receives a feature request, relevant compiler source files, a compiler guide (showing how to add intrinsics), and performance history. It outputs unified diffs.
2. **Auto-format** — `cargo fmt` normalizes the agent's code.
3. **Quality gate** — `cargo clippy` + all ~424 tests must pass.
4. **Release build** — The compiler is rebuilt with the agent's changes.
5. **Benchmark gate** — All 5 Loop B kernels are benchmarked. No kernel may regress >0.5%, and at least one must improve ≥0.5%.
6. **Accept/reject** — On rejection, `git checkout -- src/ tests/` reverts changes and the original binary is restored.

### Running

```bash
# Default: 20 iterations, 600s timeout
bash autoresearch/loop_a/orchestrator.sh

# Customize
MAX_ITERATIONS=5 TIMEOUT=300 bash autoresearch/loop_a/orchestrator.sh
```

### Smoke Test Results

First successful end-to-end run: the agent proposed vector min/max intrinsics (correct code, all 424 tests passed including 3 new ones). Rejected by benchmark gate because no existing kernel uses min/max yet — correct behavior.

### Configuration

The orchestrator has a hardcoded `FEATURE_REQUEST` and `SOURCE_FILES` array. To change the target:

1. Edit `FEATURE_REQUEST` in `orchestrator.sh`
2. Update `SOURCE_FILES` to point at relevant compiler source files
3. The agent receives these files as context along with `compiler_guide.md`

## Design

See `docs/superpowers/specs/2026-03-13-autoresearch-design.md` for the full spec.

- **Loop B**: Agent optimizes `.ea` kernel code (22 benchmarks across float, int8, fusion, stencil, hash table patterns)
- **Loop A**: Agent optimizes compiler internals, using Loop B benchmarks as regression gate (infrastructure complete)
- **Loop C** (future): Agent explores language design
