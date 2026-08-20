#!/usr/bin/env bash
# builds the knn_cpp extension in place so `import knn_cpp` works from the
# repo root without a full pip install
set -euo pipefail

cd "$(dirname "$0")/../app/cpp"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
find . -name "knn_cpp*.so" -exec cp {} ../../../ \;

echo "built knn_cpp — run 'python -c \"import knn_cpp\"' from the repo root to verify"
