# heat_qwen36.bin

Routing heat table for **Qwen3.6-35B-A3B** (`qwen35moe`), format `MOEHEAT1`:

```
magic     8 B   "MOEHEAT1"
n_layer   4 B   little endian (40)
n_expert  4 B   little endian (256)
bytes     8 B × n_layer   per-layer bytes per expert (gate+up+down)
counts    4 B × n_layer × n_expert   routing counts from 128 prompts
```

Collected with `llama-route-dump` on the base model over 128 prompts (code / Chinese Q&A / agent /
long document, 32 each), 1.42M routing events. The distribution is a property of the base model, so
fine-tunes of the same base share it.

It is only used to pre-seed the MoE expert cache (`LLAMA_EXPERT_HEAT`). Without it, the cache still
works — it just starts cold and learns by LRU.
