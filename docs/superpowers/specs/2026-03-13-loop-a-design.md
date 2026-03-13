# Loop A: Compiler Optimization via Autoresearch

## Overview

Loop A is the second autoresearch loop. Where Loop B optimizes kernel source code, Loop A optimizes the Eä compiler itself — modifying Rust code in `src/` to improve codegen quality, then validating against the full Loop B benchmark suite as a regression gate.

## Architecture

```
FEATURE_REQUESTS.md → build_prompt → agent (claude -p)
                                         ↓
                                    HYPOTHESIS + diff
                                         ↓
                              apply patch to src/
                                         ↓
                        cargo fmt --check   → fail = REJECT
                        cargo clippy        → fail = REJECT
                        cargo test          → fail = REJECT
                                         ↓
                      benchmark all 5 kernels (FMA, reduction,
                      dot product, SAXPY, clamp)
                                         ↓
                      no kernel slower + ≥1 faster? → ACCEPT
                                         ↓
                              log + repeat
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Mode | Hybrid: feature-request-driven first, then generell codegen exploration | Safe start with bounded tasks, open for discovery later |
| Regression gate | cargo fmt + clippy + tests + all 5 benchmarks | Matches CLAUDE.md hard rules, catches both correctness and performance regressions |
| Agent context | Guided: compiler_guide.md + relevant source files | Best knowledge/context ratio without overwhelming the agent |
| Granularity | One minimal diff per iteration | Matches Loop B philosophy, better diagnostics, each step verifiable |

## Iteration Loop

### 1. Build Prompt

The prompt assembler combines:

- **compiler_guide.md** — static guide describing compiler architecture, how to add intrinsics, patterns to follow
- **Current feature request** — from FEATURE_REQUESTS.md (or "improve codegen" in generell mode)
- **Relevant source files** — the 1-3 files the agent needs to modify (e.g., `intrinsics_simd.rs` + `simd_math.rs` for vector min/max)
- **History** — last 10 iteration results (hypothesis, accepted/rejected, errors)
- **Benchmark baselines** — current best times for all 5 kernels

### 2. Agent Turn

The agent receives the prompt via `claude -p` and outputs:

```
HYPOTHESIS: <what change and why>
FILE: <path relative to repo root>
```diff
<unified diff>
```
```

Multiple files may be changed in one iteration (e.g., typeck + codegen + test), each with its own FILE + diff block.

### 3. Apply Patch

The parser extracts FILE paths and diffs, applies them with `git apply`. On failure → REJECT as PARSE_ERROR, revert via `git checkout -- src/ tests/`.

### 4. Quality Gate (hard gates — all must pass)

```bash
cargo fmt --check                                    # formatting
cargo clippy --all-targets --all-features -- -D warnings  # lint
cargo test --tests --features=llvm                   # 421 tests
```

Any failure → REJECT immediately, revert changes, log error. No benchmarking on broken code.

### 5. Benchmark Gate

Run all 5 Loop B benchmarks against the modified compiler:

```bash
for kernel in fma reduction dot_product saxpy clamp; do
    # Compile kernel with NEW compiler, benchmark against C reference
    python3 autoresearch/kernels/$kernel/bench_kernel.py \
        autoresearch/kernels/$kernel/best_kernel.ea \
        autoresearch/kernels/$kernel/reference.so
done
```

**Acceptance criterion:**
```python
for kernel in all_kernels:
    if new_time > baseline_time * 1.005:  # 0.5% regression
        REJECT("regression in {kernel}")
if no_kernel_improved:
    REJECT("no improvement")
ACCEPT
```

A compiler change is accepted only if it makes at least one kernel faster without making any kernel slower.

### 6. Log & Revert/Keep

- **Accepted**: commit the change, update baselines, update FEATURE_REQUESTS.md if a request is resolved
- **Rejected**: `git checkout -- src/ tests/`, log hypothesis + error/times

## compiler_guide.md

A static document provided to the agent each iteration. Contents:

### Pipeline Overview
```
Source (.ea) → Lexer → Parser → Desugar → Type Check → Codegen (LLVM 18) → .o/.so
```

### How to Add a New Intrinsic

**Step 1: Type checking** — `src/typeck/intrinsics.rs` or `src/typeck/intrinsics_simd.rs`

Example (sqrt accepts scalars and vectors):
```rust
fn check_sqrt(&self, name: &str, args: &[Expr], locals: &HashMap<String, (Type, bool)>, span: &Span) -> Result<Type> {
    let arg_type = self.check_expr(&args[0], locals)?;
    match &arg_type {
        Type::F32 | Type::F64 | Type::FloatLiteral => Ok(arg_type),
        Type::Vector { elem, .. } if elem.is_float() => Ok(arg_type),
        _ => Err(...)
    }
}
```

**Step 2: Codegen dispatch** — `src/codegen/simd.rs`

Add match arm to route the intrinsic name to a compile function.

**Step 3: LLVM IR generation** — `src/codegen/simd_math.rs`

Example (sqrt lowers to LLVM intrinsic):
```rust
BasicValueEnum::VectorValue(vv) => {
    let vec_ty = vv.get_type();
    let intrinsic_name = self.llvm_vector_intrinsic_name("llvm.sqrt", vec_ty);
    let fn_type = vec_ty.fn_type(&[vec_ty.into()], false);
    let intrinsic = self.module.get_function(&intrinsic_name)
        .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_type, None));
    // ... build_call
}
```

**Step 4: Tests** — `tests/` directory

End-to-end pattern:
```rust
#[test]
fn test_new_intrinsic_f32x8() {
    assert_output(r#"
        func main() {
            let v: f32x8 = splat(16.0)
            let r: f32x8 = new_intrinsic(v)
            println(r[0])
        }
    "#, "expected_value");
}
```

### File Map for Common Changes

| Change type | Files to modify |
|------------|----------------|
| New intrinsic (vector) | `typeck/intrinsics.rs` or `typeck/intrinsics_simd.rs` + `codegen/simd.rs` + `codegen/simd_math.rs` + `tests/` |
| Extend existing intrinsic to vectors | `typeck/intrinsics_simd.rs` + `codegen/simd_math.rs` + `tests/` |
| Codegen optimization | `codegen/simd_arithmetic.rs` or `codegen/simd_math.rs` + `tests/` |
| New LLVM optimization pass | `target.rs` |

### Hard Rules

- No file exceeds 500 lines
- `cargo fmt && cargo clippy --all-targets --all-features -- -D warnings` must pass
- Every feature proven by end-to-end test
- No stubs, TODOs, or placeholders

### Files Near 500-Line Limit

| File | Lines | Action |
|------|-------|--------|
| `src/typeck/intrinsics_simd.rs` | 499 | Split before adding code |
| `src/codegen/expressions.rs` | 492 | Split before adding code |

## Agent Output Format

```
HYPOTHESIS: <one sentence describing the change and expected effect>

FILE: src/typeck/intrinsics_simd.rs
```diff
@@ -430,7 +430,9 @@
 existing context
-old line
+new line
 existing context
```

FILE: src/codegen/simd_math.rs
```diff
@@ -147,6 +147,20 @@
 existing context
+new code
 existing context
```
```

Rules:
1. One change per iteration. State hypothesis clearly.
2. Only valid Rust. Must compile, pass clippy, pass all tests.
3. Diffs must apply cleanly with `git apply`.
4. Include test additions when adding new functionality.
5. Do not modify test infrastructure (`tests/common/mod.rs`).

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| MAX_ITERATIONS | 20 | Maximum iterations per run |
| TIMEOUT | 600 | Seconds per agent turn (10 min) |
| THRESHOLD | 0.5 | Minimum improvement % to accept |

## Modes

### Feature-Request Mode (default)

Agent receives a specific feature request from `FEATURE_REQUESTS.md` with:
- The error message the agent encountered
- The impact description
- The suggested fix direction
- The relevant source files pre-loaded

### Generell Codegen Mode (future)

When FEATURE_REQUESTS.md is empty or all resolved, the agent switches to:
- Seeing all 5 benchmark results with breakdowns
- Freedom to modify any codegen file
- Goal: reduce instruction count, improve vectorization, optimize LLVM pass pipeline
- Broader `compiler_guide.md` section on codegen optimization strategies

## Directory Structure

```
autoresearch/
├── loop_a/
│   ├── orchestrator.sh          # Main loop (build → test → bench → evaluate)
│   ├── compiler_guide.md        # Static guide for the agent
│   ├── build_prompt.py          # Assembles prompt from guide + request + files + history
│   ├── parse_agent_output.py    # Extracts hypothesis + diffs, applies patches
│   ├── bench_all.py             # Runs all 5 Loop B benchmarks, returns aggregate JSON
│   ├── log_result.py            # Appends to history.json
│   ├── history.json             # Iteration history
│   └── baselines.json           # Current best benchmark times (updated on accept)
├── kernels/                     # Loop B kernels (unchanged, used as regression gate)
│   ├── fma/
│   ├── reduction/
│   ├── dot_product/
│   ├── saxpy/
│   └── clamp/
├── FEATURE_REQUESTS.md          # Input queue for feature-request mode
└── README.md
```

## First Target: Vector min/max

The first Loop A run will target the highest-priority feature request:

**Goal:** Extend `min`/`max` intrinsics to accept SIMD float types (f32x4, f32x8, f32x16, f64x2, f64x4).

**Expected changes:**
1. `src/typeck/intrinsics_simd.rs` — extend `check_min_max()` to accept vector float types
2. `src/codegen/simd_math.rs` — extend `compile_min_max()` to handle VectorValue, lower to `llvm.minnum`/`llvm.maxnum` vector variants
3. `tests/min_max_tests.rs` — add vector min/max tests

**Expected effect:** Clamp benchmark improves because the agent can use `min(max(v, lo), hi)` (2 instructions) instead of `select`-based clamp (4 instructions).

**Verification:** Clamp benchmark should improve ≥5%, no other kernel regresses.
