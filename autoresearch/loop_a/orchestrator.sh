#!/bin/bash
set -euo pipefail

# --- Configuration ---
MAX_ITERATIONS="${MAX_ITERATIONS:-20}"
TIMEOUT="${TIMEOUT:-600}"
THRESHOLD="0.5"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOOP_A_DIR="$SCRIPT_DIR"
EA_BINARY="$REPO_ROOT/target/release/ea"

PROGRAM="$LOOP_A_DIR/program.md"
GUIDE="$LOOP_A_DIR/compiler_guide.md"
HISTORY="$LOOP_A_DIR/history.json"
BASELINES="$LOOP_A_DIR/baselines.json"

AGENT_OUTPUT="$LOOP_A_DIR/agent_output.txt"
HYPOTHESIS_FILE="$LOOP_A_DIR/hypothesis.txt"

# Feature request — override with FEATURE_REQUEST env var
FEATURE_REQUEST="${FEATURE_REQUEST:-Explore the codegen pipeline for optimization opportunities. Look at how LLVM IR is generated for common patterns (loops, FMA, load/store, select, reductions) and propose changes that produce better machine code. Focus on patterns used by the 8 benchmark kernels: FMA, reduction, dot product, SAXPY, clamp, matmul, prefix sum, histogram.}"

# Source files for agent context — override with SOURCE_FILES env var (space-separated)
if [ -z "${SOURCE_FILES+x}" ]; then
    SOURCE_FILES=(
        "src/codegen/expressions.rs"
        "src/codegen/simd.rs"
        "src/codegen/simd_arithmetic.rs"
    )
fi

export EA_BINARY
unset CLAUDECODE 2>/dev/null || true

# --- Setup ---
echo "=== Eä Autoresearch Loop A: Compiler Optimization ==="
echo "Max iterations: $MAX_ITERATIONS"
echo "Timeout per iteration: ${TIMEOUT}s"
echo "Feature request: ${FEATURE_REQUEST:0:80}..."
echo ""

# Build release binary
echo "Building Eä compiler (release)..."
(cd "$REPO_ROOT" && cargo build --release --quiet)

# Compile C references for all kernels
echo "Compiling C references..."
for kernel_dir in "$REPO_ROOT"/autoresearch/kernels/*/; do
    ref_c="$kernel_dir/reference.c"
    ref_so="$kernel_dir/reference.so"
    if [ -f "$ref_c" ] && [ ! -f "$ref_so" ]; then
        kernel_name=$(basename "$kernel_dir")
        gcc_flags="-O3 -march=native -shared -fPIC"
        if [ "$kernel_name" != "reduction" ]; then
            gcc_flags="$gcc_flags -mfma"
        fi
        gcc $gcc_flags "$ref_c" -o "$ref_so"
    fi
done

# Initialize history
[ -f "$HISTORY" ] || echo "[]" > "$HISTORY"

# Get baseline benchmarks
echo "Running baseline benchmarks..."
BASELINE_JSON=$(python3 "$LOOP_A_DIR/bench_all.py" "$REPO_ROOT")
BASELINE_OK=$(echo "$BASELINE_JSON" | python3 -c "import sys,json; print(str(json.load(sys.stdin)['all_ok']).lower())")

if [ "$BASELINE_OK" != "true" ]; then
    echo "ERROR: Baseline benchmarks failed"
    echo "$BASELINE_JSON" | python3 -m json.tool
    exit 1
fi

# Save baselines
echo "$BASELINE_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
baselines = {}
for k, v in data['kernels'].items():
    if v.get('time_us'):
        baselines[k] = v['time_us']
json.dump(baselines, sys.stdout, indent=2)
" > "$BASELINES"
echo "Baselines:"
cat "$BASELINES"
echo ""

# --- Main Loop ---
for i in $(seq 1 "$MAX_ITERATIONS"); do
    echo "=== Iteration $i / $MAX_ITERATIONS ==="

    # Build prompt with relevant source files
    SOURCE_ARGS=""
    for sf in "${SOURCE_FILES[@]}"; do
        SOURCE_ARGS="$SOURCE_ARGS $REPO_ROOT/$sf"
    done

    PROMPT=$(python3 "$LOOP_A_DIR/build_prompt.py" \
        "$PROGRAM" "$GUIDE" "$FEATURE_REQUEST" "$HISTORY" "$BASELINES" \
        $SOURCE_ARGS)

    # Agent turn
    if ! timeout "$TIMEOUT" claude -p "$PROMPT" --output-format text \
        --tools "" \
        > "$AGENT_OUTPUT" 2>/dev/null; then
        echo "  TIMEOUT or agent error"
        python3 "$LOOP_A_DIR/log_result.py" "$HISTORY" "$i" \
            "TIMEOUT" "false" "null"
        continue
    fi

    # Parse and apply diffs
    if ! python3 "$LOOP_A_DIR/parse_agent_output.py" \
        "$AGENT_OUTPUT" "$REPO_ROOT" "$HYPOTHESIS_FILE"; then
        echo "  PARSE/APPLY ERROR"
        git -C "$REPO_ROOT" checkout -- src/ tests/ 2>/dev/null || true
        python3 "$LOOP_A_DIR/log_result.py" "$HISTORY" "$i" \
            "PARSE_ERROR" "false" "null"
        continue
    fi
    HYPOTHESIS=$(cat "$HYPOTHESIS_FILE")
    echo "  Hypothesis: $HYPOTHESIS"

    # Auto-format before checking (LLM diffs rarely match rustfmt exactly)
    (cd "$REPO_ROOT" && cargo fmt 2>/dev/null)

    # Quality gate: clippy
    if ! (cd "$REPO_ROOT" && cargo clippy --all-targets --all-features -- -D warnings 2>/dev/null); then
        echo "  REJECTED: clippy failed"
        git -C "$REPO_ROOT" checkout -- src/ tests/
        python3 "$LOOP_A_DIR/log_result.py" "$HISTORY" "$i" \
            "$HYPOTHESIS" "false" "clippy failed"
        continue
    fi

    # Quality gate: tests
    if ! (cd "$REPO_ROOT" && cargo test --tests --features=llvm 2>/dev/null); then
        echo "  REJECTED: tests failed"
        git -C "$REPO_ROOT" checkout -- src/ tests/
        python3 "$LOOP_A_DIR/log_result.py" "$HISTORY" "$i" \
            "$HYPOTHESIS" "false" "tests failed"
        continue
    fi

    # Rebuild release binary with changes
    echo "  Building release binary..."
    if ! (cd "$REPO_ROOT" && cargo build --release --quiet 2>/dev/null); then
        echo "  REJECTED: release build failed"
        git -C "$REPO_ROOT" checkout -- src/ tests/
        python3 "$LOOP_A_DIR/log_result.py" "$HISTORY" "$i" \
            "$HYPOTHESIS" "false" "release build failed"
        continue
    fi

    # Benchmark gate
    echo "  Running benchmarks..."
    BENCH_JSON=""
    if ! BENCH_JSON=$(python3 "$LOOP_A_DIR/bench_all.py" "$REPO_ROOT" 2>/dev/null); then
        echo "  REJECTED: benchmark crashed"
        git -C "$REPO_ROOT" checkout -- src/ tests/
        (cd "$REPO_ROOT" && cargo build --release --quiet)
        python3 "$LOOP_A_DIR/log_result.py" "$HISTORY" "$i" \
            "$HYPOTHESIS" "false" "benchmark crashed"
        continue
    fi

    # Evaluate: no regression + at least one improvement
    ACCEPTED=$(python3 -c "
import json, sys
baselines = json.load(open('$BASELINES'))
bench = json.loads('''$BENCH_JSON''')
threshold = float('$THRESHOLD')

if not bench.get('all_ok'):
    print('false')
    sys.exit()

any_improved = False
for kernel, data in bench['kernels'].items():
    new_time = data.get('time_us')
    old_time = baselines.get(kernel)
    if new_time is None or old_time is None:
        continue
    regression = (new_time - old_time) / old_time * 100
    if regression > threshold:
        print('false')
        sys.exit()
    improvement = (old_time - new_time) / old_time * 100
    if improvement >= threshold:
        any_improved = True

print('true' if any_improved else 'false')
")

    if [ "$ACCEPTED" = "true" ]; then
        echo "  ACCEPTED!"
        # Show improvements
        python3 -c "
import json
baselines = json.load(open('$BASELINES'))
bench = json.loads('''$BENCH_JSON''')
for kernel, data in sorted(bench['kernels'].items()):
    new = data.get('time_us')
    old = baselines.get(kernel)
    if new and old:
        pct = (old - new) / old * 100
        marker = ' <<<' if abs(pct) > 0.5 else ''
        print(f'  {kernel}: {old} -> {new} µs ({pct:+.2f}%){marker}')
"
        # Update baselines
        echo "$BENCH_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
baselines = {}
for k, v in data['kernels'].items():
    if v.get('time_us'):
        baselines[k] = v['time_us']
json.dump(baselines, sys.stdout, indent=2)
" > "$BASELINES"
        python3 "$LOOP_A_DIR/log_result.py" "$HISTORY" "$i" \
            "$HYPOTHESIS" "true" "null" "$BENCH_JSON"
    else
        echo "  REJECTED: no improvement or regression detected"
        python3 -c "
import json
baselines = json.load(open('$BASELINES'))
bench = json.loads('''$BENCH_JSON''')
for kernel, data in sorted(bench['kernels'].items()):
    new = data.get('time_us')
    old = baselines.get(kernel)
    if new and old:
        pct = (old - new) / old * 100
        print(f'  {kernel}: {old} -> {new} µs ({pct:+.2f}%)')
"
        git -C "$REPO_ROOT" checkout -- src/ tests/
        # Rebuild with original code
        (cd "$REPO_ROOT" && cargo build --release --quiet)
        python3 "$LOOP_A_DIR/log_result.py" "$HISTORY" "$i" \
            "$HYPOTHESIS" "false" "no improvement" "$BENCH_JSON"
    fi
done

echo ""
echo "=== Done ==="
echo "History: $HISTORY"
echo "Baselines: $BASELINES"
