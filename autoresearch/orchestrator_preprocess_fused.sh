#!/bin/bash
set -euo pipefail

MAX_ITERATIONS="${MAX_ITERATIONS:-5}"
TIMEOUT="${TIMEOUT:-180}"
THRESHOLD="0.5"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EA_BINARY="$REPO_ROOT/target/release/ea"

KERNEL_DIR="$SCRIPT_DIR/kernels/preprocess_fused"
KERNEL="$KERNEL_DIR/kernel.ea"
BEST="$KERNEL_DIR/best_kernel.ea"
REF_C="$KERNEL_DIR/reference.c"
REF_SO="$KERNEL_DIR/reference.so"
BENCH="$KERNEL_DIR/bench_kernel.py"
PROGRAM="$KERNEL_DIR/program.md"
HISTORY="$KERNEL_DIR/history.json"

AGENT_OUTPUT="$SCRIPT_DIR/agent_output.txt"
HYPOTHESIS_FILE="$SCRIPT_DIR/hypothesis.txt"

export EA_BINARY
unset CLAUDECODE 2>/dev/null || true

echo "=== Ea Autoresearch: Preprocess Fused Optimization ==="
echo "Max iterations: $MAX_ITERATIONS"
echo "Timeout per iteration: ${TIMEOUT}s"
echo "Improvement threshold: ${THRESHOLD}%"
echo ""

echo "Building Ea compiler (release)..."
(cd "$REPO_ROOT" && cargo build --release --quiet)
echo "Ea binary: $EA_BINARY"

echo "Compiling C reference..."
gcc -O3 -march=native -mfma -shared -fPIC "$REF_C" -o "$REF_SO" -lm

[ -f "$HISTORY" ] || echo "[]" > "$HISTORY"

if [ -f "$BEST" ]; then
    cp "$BEST" "$KERNEL"
else
    cp "$KERNEL" "$BEST"
fi

echo "Running baseline benchmark..."
BASELINE=$(python3 "$BENCH" "$BEST" "$REF_SO")
BENCH_JSON="$BASELINE"
BEST_SCORE=$(echo "$BASELINE" | python3 -c "import sys,json; print(json.load(sys.stdin)['time_us'])")
BEST_LOC=$(echo "$BASELINE" | python3 -c "import sys,json; print(json.load(sys.stdin)['loc'])")

echo "Baseline: ${BEST_SCORE} us, ${BEST_LOC} LOC"
echo ""

for i in $(seq 1 "$MAX_ITERATIONS"); do
    echo "=== Iteration $i / $MAX_ITERATIONS ==="

    PROMPT=$(python3 "$SCRIPT_DIR/build_prompt.py" \
        "$PROGRAM" "$BEST" "$HISTORY" "$BEST_SCORE" "$BENCH_JSON")

    if ! timeout "$TIMEOUT" claude -p "$PROMPT" --output-format text \
        > "$AGENT_OUTPUT" 2>/dev/null; then
        echo "  TIMEOUT or agent error"
        python3 "$SCRIPT_DIR/log_result.py" "$HISTORY" "$i" \
            "TIMEOUT" "null" "null" "false" "false"
        continue
    fi

    if ! python3 "$SCRIPT_DIR/parse_agent_output.py" \
        "$AGENT_OUTPUT" "$KERNEL" "$HYPOTHESIS_FILE"; then
        echo "  PARSE ERROR"
        python3 "$SCRIPT_DIR/log_result.py" "$HISTORY" "$i" \
            "PARSE_ERROR" "null" "null" "false" "false"
        continue
    fi
    HYPOTHESIS=$(cat "$HYPOTHESIS_FILE")
    echo "  Hypothesis: $HYPOTHESIS"

    RESULT=""
    if ! RESULT=$(python3 "$BENCH" "$KERNEL" "$REF_SO" 2>/dev/null); then
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
        echo "  ACCEPTED: ${TIME_US} us (${IMPROVEMENT}% improvement), LOC ${LOC}"
        cp "$KERNEL" "$BEST"
        BEST_SCORE="$TIME_US"
        BEST_LOC="$LOC"
        BENCH_JSON="$RESULT"
    else
        echo "  REJECTED: ${TIME_US} us (best: ${BEST_SCORE} us), LOC ${LOC}"
        cp "$BEST" "$KERNEL"
    fi

    python3 "$SCRIPT_DIR/log_result.py" "$HISTORY" "$i" \
        "$HYPOTHESIS" "$TIME_US" "$LOC" "true" "$ACCEPTED" "$KERNEL"
done

echo ""
echo "=== Done ==="
echo "Best: ${BEST_SCORE} us, ${BEST_LOC} LOC"
echo "History: $HISTORY"
echo "Best kernel: $BEST"
