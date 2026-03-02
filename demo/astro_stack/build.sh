#!/bin/bash
# Build the Eä stacking kernel: compile to .so + generate Python bindings via ea bind.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EA_ROOT="$SCRIPT_DIR/../.."

if [ ! -d "$EA_ROOT" ]; then
    echo "ERROR: Ea compiler not found at $EA_ROOT"
    exit 1
fi

echo "Building Ea compiler..."
(cd "$EA_ROOT" && cargo build --features=llvm --release --quiet)
EA="$EA_ROOT/target/release/ea"

echo "Compiling stack.ea -> libstack.so"
(cd "$SCRIPT_DIR" && "$EA" stack.ea --lib -o libstack.so)

echo "Generating stack.py (ea bind --python)"
(cd "$SCRIPT_DIR" && "$EA" bind stack.ea --python)

rm -f "$SCRIPT_DIR"/*.o

echo ""
echo "Done. Build artifacts:"
ls -lh "$SCRIPT_DIR"/libstack.so "$SCRIPT_DIR"/stack.py 2>/dev/null || true

echo ""
echo "Kernel analysis:"
"$EA" inspect "$SCRIPT_DIR/stack.ea"
