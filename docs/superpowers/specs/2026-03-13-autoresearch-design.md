# Eä Autoresearch: Iterative Kernel Optimization

## Overview

An autoresearch-style iteration loop for Eä kernel optimization, inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch). An AI agent iteratively improves `.ea` kernel files, measured against hand-written C SIMD baselines. The loop is fully automated: modify → compile → benchmark → keep/discard → repeat.

This spec covers **Loop B (kernel optimization)** — the first of three planned loops:
- **Loop B**: Agent optimizes `.ea` kernel code (this spec)
- **Loop A**: Agent optimizes compiler internals (future, uses Loop B benchmarks as regression gate)
- **Loop C**: Agent explores language design (future, uses Loop B benchmarks as regression gate)

## Architecture

```
orchestrator.sh (bash loop)
│
├─ claude -p "..."         # AI agent modifies kernel.ea (headless CLI)
├─ ea kernel.ea --lib --opt-level=3  # compile (no AI)
├─ python3 bench_kernel.py # measure + correctness check (no AI)
├─ evaluate: keep/discard  # compare against best
└─ log to history.json     # record attempt
```

The agent has no control over measurement. The shell orchestrator handles compilation, benchmarking, evaluation, and logging. The agent only produces kernel source code.

## Evaluation Model

### Hard Gates (must pass or attempt is discarded)
1. **Correctness**: kernel output must match C reference within rtol=1e-5
2. **No regression**: must beat or match current best time

### Target Function
The optimization target is `fma_kernel_f32x8` — the widest SIMD variant, the "Eä way". Other variants (f32x4, foreach, foreach+unroll) may remain in the kernel for comparison but are not measured by the loop.

### Metrics
- **Primary**: median wall time (µs) over 100 runs, 1M f32 elements
- **Secondary**: lines of code (fewer is better at equal performance)

Median is used instead of mean to reduce sensitivity to scheduler jitter and cache noise.

### Acceptance Criteria
A new kernel is accepted if:
- Correctness check passes, AND
- Time is at least 0.5% faster than current best, OR
- Time is within 0.5% and LOC is strictly lower

The 0.5% threshold prevents accepting changes based on measurement noise.

## Starting Benchmark: FMA

The first kernel target is fused multiply-add: `result[i] = a[i] * b[i] + c[i]`

**Measured cycle time:**
- Compilation: ~0.2s
- Benchmark (100 runs): ~2s
- Agent thinking: ~30-60s
- Total per iteration: ~1-2 minutes

**Current baseline (from bench.py on AMD EPYC 9354P):**
- Ea f32x8: 199.3 µs avg
- GCC f32x8 (hand-written AVX2 intrinsics): 201.2 µs avg
- Ea is already competitive — the agent will need creative approaches to improve further

## Directory Structure

```
autoresearch/
├── program.md              # agent rules and constraints
├── orchestrator.sh         # the iteration loop
├── bench_kernel.py         # compile + benchmark + correctness → JSON
├── build_prompt.py         # assembles prompt from program.md + kernel + history
├── parse_agent_output.py   # splits hypothesis + kernel from agent output
├── log_result.py           # appends entry to history.json
├── kernels/
│   └── fma/
│       ├── kernel.ea       # working copy (agent edits this)
│       ├── best_kernel.ea  # current champion (auto-managed)
│       └── reference.c     # hand-written AVX2 baseline (read-only)
├── history.json            # log of all attempts
└── results/                # future: visualization
```

## Agent Rules (program.md)

The agent receives these rules each iteration:

### Constraints
1. Only modify `kernel.ea` — nothing else
2. Only valid Eä syntax — no inventing intrinsics or features
3. Correctness is non-negotiable — output must match reference
4. One change per iteration — isolate variables, state hypothesis
5. All functions must keep the same `export func` signatures
6. No dead code, no comments longer than one line

### The Eä Way (optimization priority)
1. Use the widest SIMD type available (f32x8 > f32x4 > scalar on AVX2)
2. Write explicit vector loops with load/store — don't rely on auto-vectorization
3. Use explicit tail handling for remainder elements
4. Prefer intrinsics (fma, reduce_add) over manual scalar equivalents
5. foreach/unroll are fallbacks, not first choices

### Available Eä Features

**SIMD types:**

| 128-bit | 256-bit | 512-bit |
|---------|---------|---------|
| f32x4   | f32x8   | f32x16  |
| f64x2   | f64x4   |         |
| i32x4   | i32x8   | i32x16  |
| i16x8   | i16x16  |         |
| i8x16   | i8x32   |         |
| u8x16   | u8x32   |         |

**Intrinsics:**
- Memory: load, store, stream_store, gather, scatter
- Arithmetic: fma, sqrt, rsqrt, exp, min, max
- Reduction: reduce_add, reduce_max, reduce_min
- Construction: splat, select
- Conversion: widen_i8_f32x{4,8,16}, widen_u8_f32x{4,8,16}, widen_u8_i32x{4,8,16}
- Integer: maddubs_i16, maddubs_i32

**Loop constructs:** while, foreach, unroll(N)

**Other:** const, static_assert, *restrict, *restrict mut

### Output Format
```
HYPOTHESIS: <what you're trying and why>

```ea
<complete kernel.ea content>
```
```

The agent wraps kernel source in a fenced code block (` ```ea `). The parser extracts content between the first code fence pair, making it robust to preamble text or trailing commentary.

### Context Provided Each Iteration
- The full program.md rules
- Current best kernel.ea source
- Current best score (µs) and LOC
- Last 10 history entries (hypothesis, time, LOC, accepted/rejected)

## Orchestrator Loop

```
for each iteration (max 20, 3-minute timeout):
    1. PROMPT  — build_prompt.py assembles program.md + best kernel + history
    2. AGENT   — claude -p "$PROMPT" > agent_output.txt
    3. PARSE   — parse_agent_output.py extracts hypothesis + kernel.ea
    4. COMPILE — ea kernel.ea --lib (if fails → log + revert → next)
    5. BENCH   — bench_kernel.py outputs JSON {correct, time_us, min_us, loc}
    6. EVAL    — if correct AND (faster OR equal+fewer LOC): accept, else revert
    7. LOG     — log_result.py appends to history.json
```

### Configuration
- **Max iterations**: 20 (configurable)
- **Timeout**: 3 minutes per iteration
- **History window**: last 10 attempts passed to agent
- **Benchmark runs**: 100 per measurement
- **Array size**: 1,000,000 f32 elements
- **Ea binary**: `./target/release/ea` (orchestrator runs `cargo build --release` at setup)
- **Compile flags**: `--lib --opt-level=3` (fixed for all iterations)
- **Improvement threshold**: 0.5% (minimum improvement to accept)

## bench_kernel.py

### Interface
- **Input**: path to `kernel.ea` and path to pre-built `reference.so`
- **Compiles**: `ea kernel.ea --lib --opt-level=3` (release-quality codegen)
- **Loads**: the compiled `.so` and the pre-built `reference.so` via ctypes
- **Measures**: calls `fma_kernel_f32x8` specifically (the optimization target)
- **Correctness**: runs both Ea and C on same input, compares output (rtol=1e-5)
- **Timing**: 100 runs, reports median and min

### Output (JSON to stdout)
```json
{
  "correct": true,
  "time_us": 199.3,
  "min_us": 185.4,
  "loc": 42,
  "error": null
}
```

### Failure modes
- **Compile error**: `{"correct": false, "time_us": null, "min_us": null, "loc": null, "error": "compile: <message>"}`
- **Correctness failure**: `{"correct": false, "time_us": null, "min_us": null, "loc": null, "error": "correctness: max diff 0.5 at index 42"}`
- **Runtime crash**: `{"correct": false, "time_us": null, "min_us": null, "loc": null, "error": "crash: <signal>"}`

Always outputs exactly one JSON line to stdout. The orchestrator never needs to parse stderr.

## Helper Scripts

### build_prompt.py
Reads program.md, current best kernel, and history.json. Assembles a single prompt string with the current best score, LOC, and last 10 history entries injected into the template.

### parse_agent_output.py
Extracts the `HYPOTHESIS:` line and the content between the first code fence pair (` ```ea ` or ` ``` `). Writes the hypothesis to `hypothesis.txt` and the kernel source to `kernel.ea`. Only overwrites `kernel.ea` on successful parse — if extraction fails, exits non-zero and `kernel.ea` remains unchanged (best_kernel.ea is the revert source).

### log_result.py
Appends a JSON object to the history array: `{iteration, hypothesis, time_us, loc, correct, accepted, kernel_hash}`. The `kernel_hash` (SHA-256 of kernel source) enables detecting duplicate attempts without storing full source in the log.

## Future Expansion

### Loop A (Compiler Internals)
Agent modifies Eä Rust source → `cargo build` → run Loop B benchmark suite as regression gate → measure codegen quality. Same orchestrator pattern, different target files and longer cycle time.

### Loop C (Language Design)
Agent proposes new syntax/intrinsics → implements in compiler → writes kernel using them → measures both performance and expressiveness. Requires passing both Loop A and Loop B gates.

### UX vs Performance Conflicts
When a change improves UX (fewer LOC, clearer syntax) but affects performance:
- Performance has a hard floor — must not regress beyond current best
- UX improvements are only accepted if they pass the performance gate
- This matches Eä's philosophy: "no hidden performance cliffs"

### Additional Benchmarks
After FMA, add horizontal reduction, then more complex kernels (convolution, dot product). Each becomes an additional regression gate for Loops A and C.
