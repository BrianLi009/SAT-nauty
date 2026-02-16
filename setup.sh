#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Build nauty
echo "=== Building nauty ==="
cd "$ROOT/nauty2_8_8"
make
echo "nauty built successfully"

# Build cadical-rcl
echo ""
echo "=== Building cadical-rcl ==="
cd "$ROOT/cadical-rcl"
./configure
make
echo "cadical-rcl built successfully"

echo ""
echo "All components built successfully!"
