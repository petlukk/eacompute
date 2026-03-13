# Autoresearch: FMA Kernel Optimization Loop — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an automated iteration loop where an AI agent optimizes Eä FMA kernels, measured against hand-written C SIMD baselines.

**Architecture:** A bash orchestrator spawns Claude Code CLI (`claude -p`) per iteration. The agent outputs modified kernel source. The orchestrator compiles, benchmarks, and keeps/discards based on correctness + median time improvement. All measurement is outside agent control.

**Tech Stack:** Bash (orchestrator), Python 3 (benchmark/helpers), Claude Code CLI (`claude -p`), Eä compiler (`ea`), GCC (C reference), numpy + ctypes (benchmark harness)

**Spec:** `docs/superpowers/specs/2026-03-13-autoresearch-design.md`

---

## File Map

| File | Purpose | New/Existing |
|------|---------|--------------|
| `autoresearch/kernels/fma/kernel.ea` | Working copy of FMA kernel (agent edits) | New (copy from benchmarks/) |
| `autoresearch/kernels/fma/best_kernel.ea` | Current best kernel (auto-managed) | New (copy from benchmarks/) |
| `autoresearch/kernels/fma/reference.c` | Hand-written AVX2 C baseline (read-only) | New (copy from benchmarks/) |
| `autoresearch/bench_kernel.py` | Compile + correctness + benchmark → JSON | New |
| `autoresearch/parse_agent_output.py` | Extract hypothesis + kernel from agent output | New |
| `autoresearch/build_prompt.py` | Assemble prompt from program.md + kernel + history | New |
| `autoresearch/log_result.py` | Append result to history.json | New |
| `autoresearch/program.md` | Agent rules and constraints | New |
| `autoresearch/orchestrator.sh` | Main iteration loop | New |
| `autoresearch/.gitignore` | Ignore generated artifacts | New |

---

## Chunk 1: Kernel Files and Benchmark Script

### Task 1: Set Up Directory Structure and Seed Files

**Files:**
- Create: `autoresearch/kernels/fma/kernel.ea`
- Create: `autoresearch/kernels/fma/best_kernel.ea`
- Create: `autoresearch/kernels/fma/reference.c`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p autoresearch/kernels/fma autoresearch/results
```

- [ ] **Step 2: Copy seed kernel**

Copy `benchmarks/fma_kernel/kernel.ea` to `autoresearch/kernels/fma/kernel.ea` and `autoresearch/kernels/fma/best_kernel.ea` (identical at start).

```bash
cp benchmarks/fma_kernel/kernel.ea autoresearch/kernels/fma/kernel.ea
cp benchmarks/fma_kernel/kernel.ea autoresearch/kernels/fma/best_kernel.ea
```

- [ ] **Step 3: Copy C reference**

Copy `benchmarks/fma_kernel/reference.c` to `autoresearch/kernels/fma/reference.c`.

```bash
cp benchmarks/fma_kernel/reference.c autoresearch/kernels/fma/reference.c
```

- [ ] **Step 4: Create .gitignore for generated artifacts**

```bash
cat > autoresearch/.gitignore << 'EOF'
# Generated artifacts
*.so
*.ea.json
history.json
agent_output.txt
hypothesis.txt
results/
EOF
```

- [ ] **Step 5: Commit**

```bash
git add autoresearch/kernels/ autoresearch/.gitignore
git commit -m "autoresearch: seed FMA kernel, C reference, and .gitignore"
```

---

### Task 2: Write bench_kernel.py

**Files:**
- Create: `autoresearch/bench_kernel.py`

This is the core measurement script. It compiles the Eä kernel, checks correctness against the C reference, and outputs a single JSON line. It must handle all failure modes gracefully.

- [ ] **Step 1: Write bench_kernel.py**

```python
#!/usr/bin/env python3
"""Compile and benchmark an Eä FMA kernel. Outputs one JSON line to stdout."""

import json
import os
import subprocess
import sys
import time
import ctypes
import hashlib
import numpy as np
from pathlib import Path

ARRAY_SIZE = 1_000_000
NUM_RUNS = 100
WARMUP_RUNS = 10
SEED = 42

FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
I32 = ctypes.c_int32
FMA_ARGTYPES = [FLOAT_PTR, FLOAT_PTR, FLOAT_PTR, FLOAT_PTR, I32]


def count_loc(path):
    """Count non-blank, non-comment lines."""
    count = 0
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("//"):
                count += 1
    return count


def output(correct, time_us=None, min_us=None, loc=None, error=None):
    """Print JSON result and exit."""
    print(json.dumps({
        "correct": correct,
        "time_us": time_us,
        "min_us": min_us,
        "loc": loc,
        "error": error,
    }))
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        print("Usage: bench_kernel.py <kernel.ea> <reference.so>", file=sys.stderr)
        sys.exit(1)

    kernel_path = Path(sys.argv[1])
    ref_so_path = Path(sys.argv[2])
    ea_binary = os.environ.get("EA_BINARY", "./target/release/ea")

    # --- Compile ---
    kernel_dir = kernel_path.parent
    so_name = kernel_path.stem + ".so"
    so_path = kernel_dir / so_name

    # Remove stale .so to avoid benchmarking old code on compile failure
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

    # Find the compiled .so (ea places it in cwd)
    if not so_path.exists():
        output(False, error="compile: .so not found after compilation")

    # --- Load libraries ---
    try:
        ea_lib = ctypes.CDLL(str(so_path.resolve()))
        ref_lib = ctypes.CDLL(str(ref_so_path.resolve()))
    except OSError as e:
        output(False, error=f"load: {e}")

    try:
        ea_func = ea_lib.fma_kernel_f32x8
        ea_func.argtypes = FMA_ARGTYPES
        ea_func.restype = None

        ref_func = ref_lib.fma_kernel_f32x8_c
        ref_func.argtypes = FMA_ARGTYPES
        ref_func.restype = None
    except AttributeError as e:
        output(False, error=f"symbol: {e}")

    # --- Prepare data ---
    np.random.seed(SEED)
    a = np.random.uniform(-1, 1, ARRAY_SIZE).astype(np.float32)
    b = np.random.uniform(-1, 1, ARRAY_SIZE).astype(np.float32)
    c = np.random.uniform(-1, 1, ARRAY_SIZE).astype(np.float32)
    ea_result = np.zeros(ARRAY_SIZE, dtype=np.float32)
    ref_result = np.zeros(ARRAY_SIZE, dtype=np.float32)

    ap = a.ctypes.data_as(FLOAT_PTR)
    bp = b.ctypes.data_as(FLOAT_PTR)
    cp = c.ctypes.data_as(FLOAT_PTR)
    ea_rp = ea_result.ctypes.data_as(FLOAT_PTR)
    ref_rp = ref_result.ctypes.data_as(FLOAT_PTR)
    n = I32(ARRAY_SIZE)

    # --- Correctness check (crash-safe) ---
    import signal

    def run_kernel_safe(func, ap, bp, cp, rp, n):
        """Run a kernel function, catching segfaults via fork."""
        pid = os.fork()
        if pid == 0:
            # Child: run the kernel, exit 0 on success
            try:
                func(ap, bp, cp, rp, n)
                os._exit(0)
            except Exception:
                os._exit(1)
        else:
            _, status = os.waitpid(pid, 0)
            if os.WIFSIGNALED(status):
                sig = os.WTERMSIG(status)
                return f"crash: signal {sig} ({signal.Signals(sig).name})"
            if os.WEXITSTATUS(status) != 0:
                return "crash: unknown error"
            return None

    # Run reference (should never crash)
    ref_func(ap, bp, cp, ref_rp, n)

    # Run ea kernel (might crash if agent wrote bad code)
    crash_err = run_kernel_safe(ea_func, ap, bp, cp, ea_rp, n)
    if crash_err:
        output(False, error=crash_err)

    if not np.allclose(ea_result, ref_result, rtol=1e-5):
        diff = np.abs(ea_result - ref_result)
        max_idx = np.argmax(diff)
        output(False, error=f"correctness: max diff {diff[max_idx]:.6f} at index {max_idx}")

    # --- Benchmark (no fork needed — correctness already verified) ---
    for _ in range(WARMUP_RUNS):
        ea_func(ap, bp, cp, ea_rp, n)

    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        ea_func(ap, bp, cp, ea_rp, n)
        times.append(time.perf_counter() - start)

    times.sort()
    median_us = times[len(times) // 2] * 1e6
    min_us = times[0] * 1e6
    loc = count_loc(kernel_path)

    output(True, time_us=round(median_us, 1), min_us=round(min_us, 1), loc=loc)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test bench_kernel.py with existing kernel**

First compile the C reference, then run the benchmark:

```bash
cd autoresearch
gcc -O3 -march=native -ffast-math -shared -fPIC \
    kernels/fma/reference.c -o kernels/fma/reference.so
cd ..
cargo build --release
python3 autoresearch/bench_kernel.py \
    autoresearch/kernels/fma/kernel.ea \
    autoresearch/kernels/fma/reference.so
```

Expected: a single JSON line like `{"correct": true, "time_us": 199.3, "min_us": 185.4, "loc": 42, "error": null}`

- [ ] **Step 3: Test bench_kernel.py with broken kernel (compile error)**

Create a temporary broken kernel and verify the error JSON:

```bash
echo "this is not valid ea" > /tmp/broken.ea
python3 autoresearch/bench_kernel.py /tmp/broken.ea autoresearch/kernels/fma/reference.so
```

Expected: `{"correct": false, "time_us": null, "min_us": null, "loc": null, "error": "compile: ..."}`

- [ ] **Step 4: Commit**

```bash
git add autoresearch/bench_kernel.py
git commit -m "autoresearch: add bench_kernel.py — compile + benchmark + correctness → JSON"
```

---

## Chunk 2: Helper Scripts

### Task 3: Write parse_agent_output.py

**Files:**
- Create: `autoresearch/parse_agent_output.py`

Extracts the `HYPOTHESIS:` line and kernel source from agent output. Kernel must be inside a code fence (` ```ea ` or ` ``` `). Only writes files on successful parse.

- [ ] **Step 1: Write parse_agent_output.py**

```python
#!/usr/bin/env python3
"""Extract hypothesis and kernel source from agent output.

Looks for HYPOTHESIS: line and first code fence block.
Only writes output files on successful extraction.
"""

import re
import sys
from pathlib import Path


def extract(text):
    """Return (hypothesis, kernel_source) or raise ValueError."""
    # Extract hypothesis
    hyp_match = re.search(r"^HYPOTHESIS:\s*(.+)$", text, re.MULTILINE)
    if not hyp_match:
        raise ValueError("No HYPOTHESIS: line found")
    hypothesis = hyp_match.group(1).strip()

    # Extract code fence content — prefer ```ea fences, fall back to bare ```
    fence_match = re.search(r"```ea\s*\n(.*?)```", text, re.DOTALL)
    if not fence_match:
        fence_match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if not fence_match:
        raise ValueError("No code fence block found")
    kernel = fence_match.group(1).strip() + "\n"

    if len(kernel.strip()) < 10:
        raise ValueError(f"Kernel too short ({len(kernel.strip())} chars)")

    return hypothesis, kernel


def main():
    if len(sys.argv) < 4:
        print("Usage: parse_agent_output.py <input> <kernel.ea> <hypothesis.txt>",
              file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    kernel_path = Path(sys.argv[2])
    hyp_path = Path(sys.argv[3])

    text = input_path.read_text()

    try:
        hypothesis, kernel = extract(text)
    except ValueError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        sys.exit(1)

    kernel_path.write_text(kernel)
    hyp_path.write_text(hypothesis + "\n")
    print(f"Extracted: {len(kernel)} bytes, hypothesis: {hypothesis[:60]}...",
          file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test with valid agent output**

```bash
cat > /tmp/test_agent_output.txt << 'AGENT'
Here's my optimization attempt:

HYPOTHESIS: Try loop unrolling with 2x accumulators to exploit ILP

```ea
const AVX_WIDTH: i32 = 8

export func fma_kernel_f32x8(
    a: *restrict f32,
    b: *restrict f32,
    c: *restrict f32,
    result: *restrict mut f32,
    len: i32
) {
    let mut i: i32 = 0
    while i + AVX_WIDTH <= len {
        let va: f32x8 = load(a, i)
        let vb: f32x8 = load(b, i)
        let vc: f32x8 = load(c, i)
        let vresult: f32x8 = fma(va, vb, vc)
        store(result, i, vresult)
        i = i + AVX_WIDTH
    }
    while i < len {
        result[i] = a[i] * b[i] + c[i]
        i = i + 1
    }
}
```

This should improve ILP by...
AGENT
python3 autoresearch/parse_agent_output.py \
    /tmp/test_agent_output.txt /tmp/test_kernel.ea /tmp/test_hyp.txt
echo "Exit code: $?"
cat /tmp/test_hyp.txt
head -3 /tmp/test_kernel.ea
```

Expected: exit code 0, hypothesis extracted, kernel starts with `const AVX_WIDTH`.

- [ ] **Step 3: Test with invalid agent output (no code fence)**

```bash
echo "HYPOTHESIS: something\nno code here" > /tmp/bad_output.txt
python3 autoresearch/parse_agent_output.py /tmp/bad_output.txt /tmp/k.ea /tmp/h.txt
echo "Exit code: $?"
```

Expected: exit code 1, stderr says "No code fence block found".

- [ ] **Step 4: Commit**

```bash
git add autoresearch/parse_agent_output.py
git commit -m "autoresearch: add parse_agent_output.py — extract hypothesis + kernel from agent output"
```

---

### Task 4: Write build_prompt.py

**Files:**
- Create: `autoresearch/build_prompt.py`

Assembles the prompt from program.md + current best kernel + history. Outputs the complete prompt to stdout for the orchestrator to pipe to `claude -p`.

- [ ] **Step 1: Write build_prompt.py**

```python
#!/usr/bin/env python3
"""Assemble the agent prompt from program.md, kernel, and history."""

import json
import sys
from pathlib import Path


def format_history(entries):
    """Format last N history entries as a readable summary."""
    if not entries:
        return "No previous attempts."
    lines = []
    for e in entries:
        status = "ACCEPTED" if e.get("accepted") else "REJECTED"
        time_str = f"{e['time_us']} µs" if e.get("time_us") else "N/A"
        lines.append(
            f"  #{e['iteration']}: {status} | {time_str} | "
            f"LOC {e.get('loc', '?')} | {e.get('hypothesis', '?')}"
        )
    return "\n".join(lines)


def count_loc(path):
    """Count non-blank, non-comment lines."""
    count = 0
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("//"):
                count += 1
    return count


def main():
    if len(sys.argv) < 5:
        print("Usage: build_prompt.py <program.md> <kernel.ea> <history.json> <best_score>",
              file=sys.stderr)
        sys.exit(1)

    program_path = Path(sys.argv[1])
    kernel_path = Path(sys.argv[2])
    history_path = Path(sys.argv[3])
    best_score = sys.argv[4]

    program = program_path.read_text()
    kernel = kernel_path.read_text()
    loc = count_loc(kernel_path)

    history = json.loads(history_path.read_text()) if history_path.exists() else []
    last_10 = history[-10:]

    prompt = f"""{program}

## Current Best
Score: {best_score} µs (median over 100 runs)
LOC: {loc}

## Current kernel.ea
```ea
{kernel.rstrip()}
```

## History (last {len(last_10)} attempts)
{format_history(last_10)}
"""
    print(prompt)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test with seed data**

```bash
# Create a minimal program.md for testing
echo "# Test Program" > /tmp/test_program.md
echo "[]" > /tmp/test_history.json
python3 autoresearch/build_prompt.py \
    /tmp/test_program.md \
    autoresearch/kernels/fma/best_kernel.ea \
    /tmp/test_history.json \
    "199.3" | head -20
```

Expected: output starts with `# Test Program`, includes `Score: 199.3 µs`, includes kernel source in a code fence.

- [ ] **Step 3: Commit**

```bash
git add autoresearch/build_prompt.py
git commit -m "autoresearch: add build_prompt.py — assemble agent prompt from parts"
```

---

### Task 5: Write log_result.py

**Files:**
- Create: `autoresearch/log_result.py`

Appends a result entry to history.json. Includes kernel_hash for duplicate detection.

- [ ] **Step 1: Write log_result.py**

```python
#!/usr/bin/env python3
"""Append a result entry to history.json."""

import hashlib
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 8:
        print("Usage: log_result.py <history.json> <iteration> <hypothesis> "
              "<time_us> <loc> <correct> <accepted> [kernel.ea]",
              file=sys.stderr)
        sys.exit(1)

    history_path = Path(sys.argv[1])
    iteration = int(sys.argv[2])
    hypothesis = sys.argv[3]
    time_us = float(sys.argv[4]) if sys.argv[4] != "null" else None
    loc = int(sys.argv[5]) if sys.argv[5] != "null" else None
    correct = sys.argv[6] == "true"
    accepted = sys.argv[7] == "true"

    kernel_hash = None
    if len(sys.argv) > 8:
        kernel_path = Path(sys.argv[8])
        if kernel_path.exists():
            kernel_hash = hashlib.sha256(kernel_path.read_bytes()).hexdigest()[:12]

    history = json.loads(history_path.read_text()) if history_path.exists() else []

    history.append({
        "iteration": iteration,
        "hypothesis": hypothesis,
        "time_us": time_us,
        "loc": loc,
        "correct": correct,
        "accepted": accepted,
        "kernel_hash": kernel_hash,
    })

    history_path.write_text(json.dumps(history, indent=2) + "\n")
    print(f"Logged iteration {iteration}: "
          f"{'ACCEPTED' if accepted else 'REJECTED'} "
          f"({time_us} µs, LOC {loc})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test logging**

```bash
echo "[]" > /tmp/test_hist.json
python3 autoresearch/log_result.py \
    /tmp/test_hist.json 1 "test hypothesis" 199.3 42 true true \
    autoresearch/kernels/fma/kernel.ea
cat /tmp/test_hist.json
```

Expected: JSON array with one entry containing all fields plus a 12-char kernel_hash.

- [ ] **Step 3: Commit**

```bash
git add autoresearch/log_result.py
git commit -m "autoresearch: add log_result.py — append results to history.json"
```

---

## Chunk 3: Program.md and Orchestrator

### Task 6: Write program.md

**Files:**
- Create: `autoresearch/program.md`

The agent rules document. This is what the AI agent receives each iteration.

- [ ] **Step 1: Write program.md**

```markdown
# Eä Kernel Optimization — FMA

You are optimizing an Eä SIMD kernel for fused multiply-add: `result[i] = a[i] * b[i] + c[i]`.
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `fma_kernel_f32x8` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence.

## Rules

1. Only valid Eä syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match the C reference within rtol=1e-5.
3. One change per iteration. State your hypothesis clearly.
4. The `fma_kernel_f32x8` function signature must not change.
5. Other variants (f32x4, foreach, foreach+unroll) may be included but are not measured.
6. No dead code. No comments longer than one line.

## The Eä Way (optimization priority)

1. Use the widest SIMD type available (f32x8 > f32x4 > scalar on AVX2)
2. Write explicit vector loops with load/store — don't rely on auto-vectorization
3. Use explicit tail handling for remainder elements
4. Prefer intrinsics (fma, reduce_add) over manual scalar equivalents
5. foreach/unroll are fallbacks, not first choices

## Available Eä Features

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

**Dot operators for SIMD element-wise ops:** `.+`, `.-`, `.*`, `./`, `.==`, `.<`, `.>`, `.&`, `.|`, `.^`

## Output Format

Your output MUST contain:
1. A line starting with HYPOTHESIS: followed by your reasoning
2. The complete kernel.ea wrapped in a markdown code fence tagged as ea

Example structure:

    HYPOTHESIS: trying X because Y

    (open code fence with ea tag)
    const AVX_WIDTH: i32 = 8
    export func fma_kernel_f32x8(...) { ... }
    (close code fence)

Do NOT omit the hypothesis or the code fence.
```

- [ ] **Step 2: Commit**

```bash
git add autoresearch/program.md
git commit -m "autoresearch: add program.md — agent rules and constraints"
```

---

### Task 7: Write orchestrator.sh

**Files:**
- Create: `autoresearch/orchestrator.sh`

The main iteration loop. This is the entry point — run it and the whole system goes.

- [ ] **Step 1: Write orchestrator.sh**

```bash
#!/bin/bash
set -euo pipefail

# --- Configuration ---
MAX_ITERATIONS="${MAX_ITERATIONS:-20}"
TIMEOUT="${TIMEOUT:-180}"
THRESHOLD="0.5"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EA_BINARY="$REPO_ROOT/target/release/ea"

KERNEL="$SCRIPT_DIR/kernels/fma/kernel.ea"
BEST="$SCRIPT_DIR/kernels/fma/best_kernel.ea"
REF_C="$SCRIPT_DIR/kernels/fma/reference.c"
REF_SO="$SCRIPT_DIR/kernels/fma/reference.so"
HISTORY="$SCRIPT_DIR/history.json"
PROGRAM="$SCRIPT_DIR/program.md"

AGENT_OUTPUT="$SCRIPT_DIR/agent_output.txt"
HYPOTHESIS_FILE="$SCRIPT_DIR/hypothesis.txt"

export EA_BINARY

# --- Setup ---
echo "=== Eä Autoresearch: FMA Kernel Optimization ==="
echo "Max iterations: $MAX_ITERATIONS"
echo "Timeout per iteration: ${TIMEOUT}s"
echo "Improvement threshold: ${THRESHOLD}%"
echo ""

# Build release binary
echo "Building Eä compiler (release)..."
(cd "$REPO_ROOT" && cargo build --release --quiet)
echo "Ea binary: $EA_BINARY"

# Compile C reference once
echo "Compiling C reference..."
gcc -O3 -march=native -ffast-math -shared -fPIC "$REF_C" -o "$REF_SO"

# Initialize history
[ -f "$HISTORY" ] || echo "[]" > "$HISTORY"

# If best_kernel.ea already exists (from a prior run), use it as source of truth
# Otherwise initialize both from the seed kernel
if [ -f "$BEST" ]; then
    cp "$BEST" "$KERNEL"
else
    cp "$KERNEL" "$BEST"
fi

# Get baseline score
echo "Running baseline benchmark..."
BASELINE=$(python3 "$SCRIPT_DIR/bench_kernel.py" "$BEST" "$REF_SO")
BEST_SCORE=$(echo "$BASELINE" | python3 -c "import sys,json; print(json.load(sys.stdin)['time_us'])")
BEST_LOC=$(echo "$BASELINE" | python3 -c "import sys,json; print(json.load(sys.stdin)['loc'])")

echo "Baseline: ${BEST_SCORE} µs, ${BEST_LOC} LOC"
echo ""

# --- Main Loop ---
for i in $(seq 1 "$MAX_ITERATIONS"); do
    echo "=== Iteration $i / $MAX_ITERATIONS ==="

    # Build prompt
    PROMPT=$(python3 "$SCRIPT_DIR/build_prompt.py" \
        "$PROGRAM" "$BEST" "$HISTORY" "$BEST_SCORE")

    # Agent turn
    if ! timeout "$TIMEOUT" claude -p "$PROMPT" --output-format text \
        > "$AGENT_OUTPUT" 2>/dev/null; then
        echo "  TIMEOUT or agent error"
        python3 "$SCRIPT_DIR/log_result.py" "$HISTORY" "$i" \
            "TIMEOUT" "null" "null" "false" "false"
        continue
    fi

    # Parse agent output
    if ! python3 "$SCRIPT_DIR/parse_agent_output.py" \
        "$AGENT_OUTPUT" "$KERNEL" "$HYPOTHESIS_FILE"; then
        echo "  PARSE ERROR"
        python3 "$SCRIPT_DIR/log_result.py" "$HISTORY" "$i" \
            "PARSE_ERROR" "null" "null" "false" "false"
        continue
    fi
    HYPOTHESIS=$(cat "$HYPOTHESIS_FILE")
    echo "  Hypothesis: $HYPOTHESIS"

    # Benchmark
    RESULT=$(python3 "$SCRIPT_DIR/bench_kernel.py" "$KERNEL" "$REF_SO")
    CORRECT=$(echo "$RESULT" | python3 -c "import sys,json; print(str(json.load(sys.stdin)['correct']).lower())")
    TIME_US=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['time_us'] if d['time_us'] else 'null')")
    LOC=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['loc'] if d['loc'] else 'null')")
    ERROR=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error') or '')")

    if [ "$CORRECT" != "true" ]; then
        echo "  REJECTED (incorrect): $ERROR"
        cp "$BEST" "$KERNEL"
        python3 "$SCRIPT_DIR/log_result.py" "$HISTORY" "$i" \
            "$HYPOTHESIS" "$TIME_US" "$LOC" "false" "false" "$KERNEL"
        continue
    fi

    # Evaluate: is it better?
    ACCEPTED=$(python3 -c "
import sys
t, b, threshold = float('$TIME_US'), float('$BEST_SCORE'), float('$THRESHOLD')
l, bl = int('$LOC'), int('$BEST_LOC')
improvement = (b - t) / b * 100
if improvement >= threshold:
    print('true')
elif abs(improvement) < threshold and l < bl:
    print('true')
else:
    print('false')
")

    if [ "$ACCEPTED" = "true" ]; then
        IMPROVEMENT=$(python3 -c "print(f'{(float(\"$BEST_SCORE\") - float(\"$TIME_US\")) / float(\"$BEST_SCORE\") * 100:.2f}')")
        echo "  ACCEPTED: ${TIME_US} µs (${IMPROVEMENT}% improvement), LOC ${LOC}"
        cp "$KERNEL" "$BEST"
        BEST_SCORE="$TIME_US"
        BEST_LOC="$LOC"
    else
        echo "  REJECTED: ${TIME_US} µs (best: ${BEST_SCORE} µs), LOC ${LOC}"
        cp "$BEST" "$KERNEL"
    fi

    python3 "$SCRIPT_DIR/log_result.py" "$HISTORY" "$i" \
        "$HYPOTHESIS" "$TIME_US" "$LOC" "true" "$ACCEPTED" "$KERNEL"
done

echo ""
echo "=== Done ==="
echo "Best: ${BEST_SCORE} µs, ${BEST_LOC} LOC"
echo "History: $HISTORY"
echo "Best kernel: $BEST"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x autoresearch/orchestrator.sh
```

- [ ] **Step 3: Test orchestrator setup phase only (no agent call)**

Verify the setup phase works — builds ea, compiles reference, gets baseline:

```bash
cd /root/dev/eacompute
# Quick smoke test: just run the setup (Ctrl+C after baseline prints)
MAX_ITERATIONS=0 bash autoresearch/orchestrator.sh
```

Expected: prints "Building Eä compiler", "Compiling C reference", "Baseline: ~199 µs", then exits.

- [ ] **Step 4: Commit**

```bash
git add autoresearch/orchestrator.sh
git commit -m "autoresearch: add orchestrator.sh — main iteration loop"
```

---

### Task 8: End-to-End Dry Run

Test the full loop with a single iteration to verify all pieces work together.

- [ ] **Step 1: Run with MAX_ITERATIONS=1**

```bash
cd /root/dev/eacompute
MAX_ITERATIONS=1 bash autoresearch/orchestrator.sh
```

Expected:
- Setup completes (build, compile reference, baseline)
- Agent produces output (or times out — both are valid for this test)
- If agent succeeds: parse, benchmark, evaluate, log all run
- `history.json` has one entry
- Exit with summary

- [ ] **Step 2: Check history.json**

```bash
cat autoresearch/history.json | python3 -m json.tool
```

Expected: JSON array with one entry containing iteration, hypothesis, time_us, loc, correct, accepted, kernel_hash.

- [ ] **Step 3: Commit any fixes needed**

If the dry run exposed issues, fix them and commit:

```bash
git add autoresearch/
git commit -m "autoresearch: fix issues found during dry run"
```

---

### Task 9: Full Run (5 Iterations)

Once the dry run passes, do a real test with 5 iterations.

- [ ] **Step 1: Run with MAX_ITERATIONS=5**

```bash
cd /root/dev/eacompute
MAX_ITERATIONS=5 bash autoresearch/orchestrator.sh
```

Observe: does the agent make meaningful hypotheses? Does the accept/reject logic work? Does history accumulate correctly?

- [ ] **Step 2: Review results**

```bash
cat autoresearch/history.json | python3 -m json.tool
diff autoresearch/kernels/fma/kernel.ea autoresearch/kernels/fma/best_kernel.ea
```

- [ ] **Step 3: Commit final state**

```bash
git add autoresearch/
git commit -m "autoresearch: complete Loop B infrastructure, verified with 5-iteration run"
```
