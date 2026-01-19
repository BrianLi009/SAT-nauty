#!/bin/bash
# save as build-all.sh in SAT-nauty directory
set -e

# Build Glasgow first
echo "=== Building Glasgow ==="
cd glasgow
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBOOST_ROOT=/opt/homebrew/opt/boost
cmake --build build --target gss
echo "✅ Glasgow built successfully"

# Build cadical-rcl
echo -e "\n=== Building cadical-rcl ==="
cd ../cadical-rcl
[ -x configure ] || chmod +x configure
./configure
make
echo "✅ cadical-rcl built successfully"

echo -e "\nAll components built successfully!"