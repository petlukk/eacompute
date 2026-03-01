#!/bin/bash
# Build the Eä Sobel kernel: compile to .so + generate Python bindings via ea bind.
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

echo "Compiling sobel.ea -> libsobel.so"
(cd "$SCRIPT_DIR" && "$EA" sobel.ea --lib -o libsobel.so)

echo "Generating sobel.py (ea bind --python)"
(cd "$SCRIPT_DIR" && "$EA" bind sobel.ea --python)

rm -f "$SCRIPT_DIR"/*.o

echo ""
echo "Done. Build artifacts:"
ls -lh "$SCRIPT_DIR"/libsobel.so "$SCRIPT_DIR"/sobel.py 2>/dev/null || true
