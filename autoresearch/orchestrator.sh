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

# Allow nested claude invocations (when running inside Claude Code)
unset CLAUDECODE 2>/dev/null || true

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
BENCH_JSON="$BASELINE"
BEST_SCORE=$(echo "$BASELINE" | python3 -c "import sys,json; print(json.load(sys.stdin)['time_us'])")
BEST_LOC=$(echo "$BASELINE" | python3 -c "import sys,json; print(json.load(sys.stdin)['loc'])")

echo "Baseline: ${BEST_SCORE} µs, ${BEST_LOC} LOC"
echo ""

# --- Main Loop ---
for i in $(seq 1 "$MAX_ITERATIONS"); do
    echo "=== Iteration $i / $MAX_ITERATIONS ==="

    # Build prompt
    PROMPT=$(python3 "$SCRIPT_DIR/build_prompt.py" \
        "$PROGRAM" "$BEST" "$HISTORY" "$BEST_SCORE" "$BENCH_JSON")

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

    # Benchmark (may crash if kernel segfaults — handle gracefully)
    RESULT=""
    if ! RESULT=$(python3 "$SCRIPT_DIR/bench_kernel.py" "$KERNEL" "$REF_SO" 2>/dev/null); then
        echo "  CRASHED during benchmark"
        cp "$BEST" "$KERNEL"
        python3 "$SCRIPT_DIR/log_result.py" "$HISTORY" "$i" \
            "$HYPOTHESIS" "null" "null" "false" "false" "$KERNEL"
        continue
    fi

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
        BENCH_JSON="$RESULT"
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
