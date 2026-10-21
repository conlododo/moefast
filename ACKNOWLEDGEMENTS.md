# Acknowledgements

This project builds on other people's work. Please keep these credits when you redistribute it.

## llama.cpp

- Project: **llama.cpp** — https://github.com/ggml-org/llama.cpp
- License: **MIT**
- We ship a fork with small patches; the whole inference engine, the CUDA kernels, the GGUF format,
  the server and the CLI are theirs.

### MoE expert cache (the core mechanism we use)

- **#29887** — *llama : add a GPU cache for MoE experts kept in host memory* (merged 2026-10-07)
  https://github.com/ggml-org/llama.cpp/pull/29887
- **#30112** — *llama: support MoE cache over multiple GPUs* (merged 2026-10-08)
  https://github.com/ggml-org/llama.cpp/pull/30112
- Implementation: `src/llama-moe-cache.cpp` in llama.cpp.

### Earlier community work on the same idea

- **#27861** — *llama: GPU-resident LRU cache for host-offloaded MoE expert weights*
  by **csantiago78** — https://github.com/ggml-org/llama.cpp/pull/27861
  (the unmerged prototype that started the whole expert-cache line of work; the discussion in that
  PR is where we learned about the router-on-CPU pitfall and the small-batch limitation)
- **#26563** — expert caching with a routing heat map, by **miltos22**
  https://github.com/ggml-org/llama.cpp/pull/26563
  (reference for the heat-table idea we implement)
- **ParmesanParty/llama.cpp** — dynamic hot-expert cache (`LLAMA_ARG_MOE_HOT_*`)
  https://github.com/ParmesanParty/llama.cpp
  (reference for parameter design)
- **neurall/llama.cpp** — VRAM-filling auto sizing, pay-back filtered eviction, CPU/GPU overlap
  https://github.com/neurall/llama.cpp
  (reference for auto sizing, which we implement in a simpler form)
- Discussion **#24528** — *RFC: MoE expert cache, VRAM caching of hot CPU-resident experts with
  hybrid hit/miss execution* https://github.com/ggml-org/llama.cpp/discussions/24528
- Issue **#20757** — two-tier GPU+RAM expert cache measurements
  https://github.com/ggml-org/llama.cpp/issues/20757

## Strata

- **Niko1221/Strata** — https://github.com/Niko1221/Strata
  Their `q36_engine.cu` hybrid mode (hot experts on GPU, cold experts on a CPU thread pool) is the
  behaviour we target, and their logs were the reference we measured against.

## ik_llama.cpp

- **ikawrakow/ik_llama.cpp** — https://github.com/ikawrakow/ik_llama.cpp
  MoE offload tuning notes (ubatch sizes, `-ot` widening) informed our defaults.

## Models

- **Qwen3.6-35B-A3B** by the Qwen team; GGUF quantization by **Unsloth**
  (`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, which also provides the MTP block we optionally use).

## Our own additions

- Heat-table collector (`llama-route-dump`) and aggregation/plan tools
- `--moe-cache-mib auto` (free-VRAM-based sizing)
- Heat pre-seeding (`LLAMA_EXPERT_HEAT`)
