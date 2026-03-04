#!/bin/bash
# Build 1BRC Ea kernels.
# Produces .so + .ea.json for Python ctypes loading.
# scan.ea uses #[cfg(x86_64)] / #[cfg(aarch64)] for platform-specific extract_lines.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KERNEL_DIR="$SCRIPT_DIR/kernels"

# Use EA_BIN if set, otherwise build from source
if [ -n "${EA_BIN:-}" ]; then
    EA="$(cd "$(dirname "$EA_BIN")" && pwd)/$(basename "$EA_BIN")"
    echo "Using EA_BIN=$EA"
else
    EA_ROOT="$SCRIPT_DIR/../.."
    echo "Building Ea compiler..."
    (cd "$EA_ROOT" && cargo build --features=llvm --release --quiet)
    EA="$EA_ROOT/target/release/ea"
fi

ARCH=$(uname -m)
echo "Architecture: $ARCH"

# Compile scan kernel (single file, #[cfg] selects platform-specific extract_lines)
echo "Compiling scan.ea -> libscan.so"
(cd "$SCRIPT_DIR" && "$EA" "$KERNEL_DIR/scan.ea" --lib -o libscan.so)

# Compile parse_temp kernel (cross-platform)
echo "Compiling parse_temp.ea -> libparse_temp.so"
(cd "$SCRIPT_DIR" && "$EA" "$KERNEL_DIR/parse_temp.ea" --lib -o libparse_temp.so)

# Compile aggregate kernel (cross-platform)
echo "Compiling aggregate.ea -> libaggregate.so"
(cd "$SCRIPT_DIR" && "$EA" "$KERNEL_DIR/aggregate.ea" --lib -o libaggregate.so)

# Clean up intermediate object files
rm -f "$SCRIPT_DIR"/*.o

echo ""
echo "Done. Build artifacts:"
ls -lh "$SCRIPT_DIR"/lib*.so 2>/dev/null || true
