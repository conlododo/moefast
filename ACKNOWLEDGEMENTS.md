# Acknowledgements

`moefast` is a launcher. It does not contain inference code, patches or forks.

## llama.cpp

- **llama.cpp** — https://github.com/ggml-org/llama.cpp (MIT)
- The **MoE expert cache** that makes large MoE models usable on small-VRAM GPUs is llama.cpp's own work:
  - **#29887** — *llama : add a GPU cache for MoE experts kept in host memory* (merged 2026-10-07)
  - **#30112** — *llama: support MoE cache over multiple GPUs* (merged 2026-10-08)
  - exposed as `--moe-cache-mib`; implementation in `src/llama-moe-cache.cpp`
- `gguf-py` (also llama.cpp) is used to read model metadata.

## Prior art on expert caching (design references, not code)

- **#27861** by *csantiago78* — the original community prototype of a GPU-resident expert LRU cache
- **#26563** by *miltos22* — expert caching with a routing heat map
- **neurall/llama.cpp**, **ParmesanParty/llama.cpp** — VRAM-filling auto sizing, pay-back filtering,
  CPU/GPU overlap
- **Niko1221/Strata** — a dedicated engine doing the same hot-expert / cold-expert split; used as a
  benchmark reference
- Discussion **#24528** — MoE expert cache RFC (the place where most of the pitfalls are documented)

## Our own (0.1, archived)

Version 0.1 was a patched llama.cpp build that added `--moe-cache-mib auto` and heat-table
pre-seeding on top of the upstream cache. Measurements showed the pre-seeding added little, and
auto-sizing is now done by this launcher, so 0.1 is archived and users should run **stock llama.cpp +
`moefast`**.

## Models

- **Qwen3.6-35B-A3B** / **Qwen3.8-Flash-Next** by the Qwen team; GGUF quantization by **Unsloth**.
- **gpt-oss-20b** by OpenAI.

## License

MIT — see [LICENSE](LICENSE).
