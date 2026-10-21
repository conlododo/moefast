#!/usr/bin/env bash
# moefast-35B — launch llama-server with the tuned settings for a 16 GB GPU.
#
#   ./moefast/run.sh /path/to/model.gguf [extra llama-server args...]
#
# Env:
#   MOEFAST_PORT=8080        port
#   MOEFAST_CTX=8192         context size per slot
#   MOEFAST_PARALLEL=1       slots (concurrent requests)
#   MOEFAST_MTP=0            >0 enables draft-mtp with that many drafts (e.g. 3)
#   MOEFAST_HEAT=path        heat table (default: moefast/heat/heat_qwen36.bin if present)
#   MOEFAST_CACHE_MIB=auto   MoE expert cache size: "auto", 0 to disable, or MiB
#   MOEFAST_HOST=127.0.0.1   bind address
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/build/bin/llama-server}"

MODEL="${1:-}"
[ -n "$MODEL" ] || { echo "usage: $0 /path/to/model.gguf [extra args]"; exit 1; }
shift || true

PORT="${MOEFAST_PORT:-8080}"
CTX="${MOEFAST_CTX:-8192}"
PARALLEL="${MOEFAST_PARALLEL:-1}"
MTP="${MOEFAST_MTP:-0}"
CACHE_MIB="${MOEFAST_CACHE_MIB:-auto}"
HOST="${MOEFAST_HOST:-127.0.0.1}"
HEAT="${MOEFAST_HEAT:-$ROOT/moefast/heat/heat_qwen36.bin}"

[ -x "$BIN" ] || { echo "llama-server not built: $BIN (run ./moefast/install.sh)"; exit 1; }
[ -f "$MODEL" ] || { echo "model not found: $MODEL"; exit 1; }

if [ -f "$HEAT" ]; then
  export LLAMA_EXPERT_HEAT="$HEAT"
  echo "[moefast] heat table: $HEAT"
else
  echo "[moefast] no heat table — the MoE cache will learn at runtime (slower first tokens)"
fi

ARGS=(
  -m "$MODEL"
  -ngl 99
  -ot "exps=CPU"          # keep the experts in host RAM; the router stays on the GPU
  -c "$CTX"
  -b 2048 -ub 2048        # large ubatch: prompt processing with CPU experts
  --parallel "$PARALLEL"
  --moe-cache-mib "$CACHE_MIB"   # auto = free VRAM after KV/buffers, minus headroom
  --fit off
  --port "$PORT"
  --host "$HOST"
)

if [ "$MTP" != "0" ] && [ "$MTP" != "" ]; then
  ARGS+=( --spec-type draft-mtp --spec-draft-n-max "$MTP" )
  echo "[moefast] MTP enabled (n-max=$MTP)"
fi

echo "[moefast] $BIN ${ARGS[*]} $*"
exec "$BIN" "${ARGS[@]}" "$@"
