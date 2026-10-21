#!/usr/bin/env bash
# moefast-35B — build helper
# Configures a CUDA build of llama.cpp and compiles llama-server (and llama-cli).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

JOBS="${JOBS:-$(nproc)}"
BUILD_DIR="${BUILD_DIR:-build}"

say() { printf '\033[1;36m[moefast]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[moefast] %s\033[0m\n' "$*" >&2; exit 1; }

# --- preflight -------------------------------------------------------------
command -v cmake >/dev/null || die "cmake not found (need >= 3.24)"
cmake_ver="$(cmake --version | head -1 | awk '{print $3}')"
say "cmake $cmake_ver"

# locate nvcc (PATH, /usr/local/cuda, /usr/local/cuda-*)
NVCC="$(command -v nvcc || true)"
if [ -z "$NVCC" ]; then
  for c in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
    [ -x "$c" ] && { NVCC="$c"; break; }
  done
fi
if [ -n "$NVCC" ]; then
  export CUDACXX="$NVCC"
  export PATH="$(dirname "$NVCC"):$PATH"
  say "nvcc: $($NVCC --version | tail -1)"
else
  say "ERROR: nvcc not found (PATH or /usr/local/cuda*/bin) — install the CUDA toolkit, or build CPU-only by hand"
  exit 1
fi

[ -f "$ROOT/CMakeLists.txt" ] || die "run this from inside the llama.cpp fork"
[ -d "$ROOT/moefast" ] || die "moefast/ directory missing — wrong repo?"

# --- configure + build -----------------------------------------------------
say "configuring ($BUILD_DIR)"
cmake -B "$BUILD_DIR" -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=OFF

say "building with $JOBS jobs (this takes a while on first run)"
cmake --build "$BUILD_DIR" --target llama-server llama-cli -j "$JOBS"

say "done."
say "  server: $ROOT/$BUILD_DIR/bin/llama-server"
say "  next:   ./moefast/run.sh /path/to/model.gguf"
