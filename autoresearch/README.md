# Eä Autoresearch

Automated kernel optimization loop inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch). An AI agent iteratively rewrites Eä SIMD kernels to find the fastest correct implementation.

## How It Works

A bash orchestrator runs a modify → compile → benchmark → keep/discard loop:

1. **Agent turn** — Claude Code CLI (`claude -p`) receives the current best kernel, performance history, and Eä language rules. It proposes a single optimization with a stated hypothesis and outputs a new kernel.
2. **Compile** — The Eä compiler builds the kernel to a shared library (`--lib --opt-level=3`).
3. **Benchmark** — The kernel runs 100 times across four dataset sizes (64K, 256K, 1M, 16M floats) to prevent overfitting to a single cache behavior. Correctness is verified at each size against a hand-written C AVX2 reference (rtol=1e-5). The primary metric is the median time across all sizes.
4. **Evaluate** — The new kernel is accepted only if the median improvement across all dataset sizes is at least 0.5% (or equal speed with fewer lines of code). This threshold prevents accepting noise or size-specific optimizations.
5. **Log & repeat** — Results are appended to `history.json`. The last 10 attempts are fed back to the agent so it can learn from what worked and what didn't.

The agent has no control over measurement. It can only produce kernel source code.

## Running

```bash
# Default: 20 iterations, 3-minute timeout per iteration
bash autoresearch/orchestrator.sh

# Customize
MAX_ITERATIONS=50 TIMEOUT=120 bash autoresearch/orchestrator.sh
```

Requires: Eä compiler (built automatically), GCC, Python 3 with numpy, Claude Code CLI.

### Available Benchmarks

```bash
bash autoresearch/orchestrator.sh              # FMA (throughput-bound)
bash autoresearch/orchestrator_reduction.sh    # Horizontal reduction (latency-bound)
bash autoresearch/orchestrator_dot_product.sh  # Dot product (hybrid)
bash autoresearch/orchestrator_saxpy.sh        # SAXPY (bandwidth-bound)
bash autoresearch/orchestrator_clamp.sh        # Clamp (select/masking)
```

### Iteration Budget

| Iterations | Phase | What to expect |
|------------|-------|----------------|
| 5–10 | Smoke test | Verifies the loop works. Hill-climbing finds the obvious wins (unroll, multi-accumulator). |
| 10–30 | Micro-optimization | Diminishing returns. Agent explores prefetch distances, stride tuning, instruction ordering. Most attempts rejected. |
| 100+ | Discovery | Agent exhausts textbook tricks and may try combinations or approaches a human wouldn't consider. Uncharted territory. |

We are currently running 5-iteration runs to validate infrastructure and establish baselines.

## Results

### FMA Kernel (throughput-bound)

Target: `result[i] = a[i] * b[i] + c[i]` — fused multiply-add on 1M floats.

5-iteration run on AMD EPYC 9354P:

| # | Hypothesis | Time (µs) | LOC | Result |
|---|-----------|-----------|-----|--------|
| — | Baseline (single f32x8 loop) | 236.2 | 67 | — |
| 1 | 4x loop unroll for ILP | 221.2 | 86 | **Accepted (+6.4%)** |
| 2 | stream_store to bypass cache | 256.1 | 86 | Rejected (slower) |
| 3 | 8x unroll | 224.8 | 102 | Rejected (register pressure) |
| 4 | Interleaved loads and FMAs | 236.7 | 86 | Rejected |
| 5 | 6x unroll (middle ground) | 223.2 | 78 | Rejected (<0.5% gain) |

The agent found a 6.4% improvement on its first attempt by unrolling the SIMD loop 4x, exposing more independent FMA operations for out-of-order execution. Subsequent attempts explored stream stores, higher unroll factors, and instruction reordering — all correct but none faster than the 4x unroll.

### Horizontal Reduction (latency-bound)

Target: `total = sum(data[0..len])` — single-accumulator f32x8 sum.

5-iteration run on AMD EPYC 9354P:

| # | Hypothesis | Time (µs) | LOC | Result |
|---|-----------|-----------|-----|--------|
| — | Baseline (single f32x8 accumulator) | 101.2 | 101 | — |
| 1 | 4 independent accumulators for ILP | 39.3 | 118 | **Accepted (+61.2%)** |
| 2 | 8 accumulators (64 elems/iter) | 50.4 | 134 | Rejected (register pressure) |
| 3 | Remove prefetch, simplify loop | 46.7 | 117 | Rejected (slower) |
| 4 | 8 vectors/iter with 4 accumulators | 53.3 | 125 | Rejected (slower) |
| 5 | Double prefetch distance | 39.9 | 119 | Rejected (<0.5% gain) |

Massive 61% gain by breaking the loop-carried dependency chain. Reduction is latency-bound — 4 independent accumulator chains let the OOO engine interleave `vaddps` instructions across their ~4-cycle latency. 8 accumulators regressed due to register spilling.

### Dot Product (hybrid)

Target: `result = sum(a[i] * b[i])` — FMA + reduction in one kernel.

5-iteration run on AMD EPYC 9354P:

| # | Hypothesis | Time (µs) | LOC | Result |
|---|-----------|-----------|-----|--------|
| — | Baseline (single f32x8 FMA accumulator) | 138.2 | 42 | — |
| 1 | 4 accumulators | — | — | Compile error (width mismatch) |
| 2 | 2 accumulators + prefetch | 90.5 | 56 | **Accepted (+34.5%)** |
| 3 | 4 accumulators (fixed syntax) | 109.0 | 64 | Rejected (register pressure) |
| 4 | 4 accumulators + fewer prefetches | — | — | Same compile error |
| 5 | Increase prefetch distance | 96.5 | 56 | Rejected (<0.5% gain) |

Hybrid kernel: 34.5% gain from 2 accumulators. With two input arrays (vs one for reduction), register pressure kicks in at 4 accumulators instead of 8. The improvement falls between pure throughput (FMA +6%) and pure latency (reduction +61%).

### SAXPY (bandwidth-bound)

Target: `y[i] = a * x[i] + y[i]` — scalar broadcast + FMA + in-place write.

5-iteration run on AMD EPYC 9354P:

| # | Hypothesis | Time (µs) | LOC | Result |
|---|-----------|-----------|-----|--------|
| — | Baseline (simple f32x8 loop) | 98.2 | 36 | — |
| 1 | 4x unroll + prefetch | 106.2 | 54 | Rejected (+8% slower) |
| 2 | 2x unroll + stream_store | 384.8 | 46 | Rejected (4x slower!) |
| 3 | Prefetch only | 125.2 | 38 | Rejected |
| 4 | 2x unroll, no prefetch | 114.3 | 46 | Rejected |
| 5 | stream_store only | 312.6 | 36 | Rejected (3x slower!) |

Zero improvement — the simple loop already saturates memory bandwidth. Every optimization made it worse. `stream_store` was catastrophic (3-4x regression) because SAXPY reads and writes `y` in the same iteration, so non-temporal stores invalidate the cache for the next load. A good optimizer must also know when to say "no optimization possible."

### Clamp (select/masking)

Target: `result[i] = clamp(data[i], lo, hi)` — conditional masking with `select`.

5-iteration run on AMD EPYC 9354P:

| # | Hypothesis | Time (µs) | LOC | Result |
|---|-----------|-----------|-----|--------|
| — | Baseline (select-based, single vector) | 91.4 | 67 | — |
| 1 | min/max intrinsics | — | — | Compile error: min/max unsupported on vectors |
| 2 | min/max + 4x unroll | — | — | Same compile error |
| 3 | 4x unroll + prefetch (select) | 83.1 | 83 | **Accepted (+9.1%)** |
| 4 | stream_store | 166.4 | 83 | Rejected (2x slower) |
| 5 | 8x unroll + deeper prefetch | 89.3 | 95 | Rejected |

9% gain from unrolling. The agent discovered that `min`/`max` intrinsics don't support vector types — a concrete language feature request (Loop C candidate). The `select`-based clamp generates 4 instructions per operation (2x `vcmpps` + 2x `vblendvps`) vs the C reference's 2-instruction `vmaxps`/`vminps`, leaving more compute headroom for ILP than FMA.

### Bottleneck Profile Summary

The agent implicitly learned an empirical model of the Zen 4 microarchitecture:

| Kernel | Bottleneck | Best strategy | Improvement |
|--------|-----------|---------------|-------------|
| FMA | Throughput | 8x unroll + stream_store | +6% |
| Dot product | Hybrid | 2 FMA accumulators | +35% |
| Reduction | Latency | 4 independent accumulators | +61% |
| SAXPY | Bandwidth | Do nothing | +0% |
| Clamp | Masking | 4x unroll | +9% |

Key insights: (1) latency hiding via multi-accumulator works, (2) register pressure limits accumulator count based on input arity, (3) hardware prefetch handles sequential access — manual prefetch adds noise, not speed, (4) bandwidth-bound kernels can't be optimized with compute tricks, (5) missing vector min/max forces suboptimal select codegen.

### Width as a Dimension

The agent is free to choose any vector width — including pure scalar. Width is treated as an optimization dimension, not a default:

| Width | When it wins |
|-------|-------------|
| Scalar | Dependency chains, small data, branch-heavy |
| f32x4 | Moderate parallelism, lower register pressure |
| f32x8 | High parallelism, compute-bound, AVX2 |
| f32x16 | Extreme parallelism, AVX-512 only |

The benchmark decides which width is best. Wider SIMD is not assumed to be faster.

## Loop A: Compiler Optimization

Loop A modifies the Eä compiler itself (Rust source) to improve codegen quality. All 5 Loop B benchmarks serve as a regression gate — compiler changes are only accepted if no kernel gets slower and at least one gets faster.

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

- **Loop B**: Agent optimizes `.ea` kernel code (5 benchmarks working)
- **Loop A**: Agent optimizes compiler internals, using Loop B benchmarks as regression gate (infrastructure complete)
- **Loop C** (future): Agent explores language design
