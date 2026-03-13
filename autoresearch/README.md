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

## First Results (FMA Kernel)

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

## Design

See `docs/superpowers/specs/2026-03-13-autoresearch-design.md` for the full spec. This is Loop B (kernel optimization) — the first of three planned loops:

- **Loop B** (this): Agent optimizes `.ea` kernel code
- **Loop A** (future): Agent optimizes compiler internals, using Loop B benchmarks as regression gate
- **Loop C** (future): Agent explores language design, using Loop B benchmarks as regression gate
