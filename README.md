# moefast — run a 35B MoE model fast on a single 16 GB GPU

`moefast` is a small patch set for [llama.cpp](https://github.com/ggml-org/llama.cpp) that lets a
30B-class MoE model (**Qwen3.6-35B-A3B**, `qwen35moe`) run on one 16 GB consumer GPU at
**~110 tok/s decode**, with a 200K context, while still leaving VRAM for your desktop.

Tested on: RTX 5060 Ti 16 GB · Ryzen 9 9900X · 30 GB RAM · Ubuntu 24.04.
Model: Qwen3.6-35B-A3B UD-Q4_K_XL (753-tensor MTP GGUF), fine-tuned merge.

---

## Measured on a 16 GB card

Same model file, greedy, `max_tokens=192`, fresh prompt per run, 3 runs per size (medians).
*TTFT* = time to the first streamed token.

| Stack | TTFT ~50 tok | TTFT ~1K tok | TTFT ~4K tok | decode ~50 | decode ~1K | decode ~4K |
|---|---:|---:|---:|---:|---:|---:|
| **moefast** | 0.299 s | **0.615 s** | **1.427 s** | 103.1 t/s | **114.8 t/s** | **113.4 t/s** |
| vanilla llama.cpp (experts on CPU) | 0.603 s | 1.259 s | 2.186 s | 56.1 t/s | 60.7 t/s | 59.7 t/s |
| Strata (16 GB instance, `--mtp 2`) | **0.166 s** | 0.723 s | 1.448 s | **120.2 t/s** | 111.7 t/s | 89.0 t/s |

The short version: Strata wins tiny prompts; **moefast is fastest from ~1K tokens up** and stays flat
at 4K, while vanilla llama.cpp is last everywhere.

---

## Two ways to use it

### Option A — ready-to-clone branch (no patching)

```bash
# if the fork's default branch is set to feat/moe-cache-main:
git clone https://github.com/conlododo/llama.cpp
# otherwise:
git clone -b feat/moe-cache-main https://github.com/conlododo/llama.cpp

cd llama.cpp
./moefast/install.sh
./moefast/run.sh /path/to/model.gguf
```

Ready-made branch: **https://github.com/conlododo/llama.cpp/tree/feat/moe-cache-main**
(tag `moefast-v1`), upstream `1ba568c` + 8 commits.

The branch is upstream `1ba568c` + the same 8 small commits you see in `patches/`, and it already
contains the `moefast/` directory, so nothing else is needed.

### Option B — apply the patch to your own upstream checkout

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 1ba568c
git apply -3 /path/to/moefast/patches/moefast.patch
cp -r /path/to/moefast .
./moefast/install.sh && ./moefast/run.sh /path/to/model.gguf
```

Use this if you need to stay on your own fork/commit. `-3` makes git do a 3-way merge, which
usually absorbs small upstream drift.

---

## How it works

| Piece | What it does | Where it comes from |
|---|---|---|
| **MoE expert cache** | keeps the hottest host-resident experts in VRAM (LRU), cold experts are computed on the CPU — nothing is copied per token | llama.cpp upstream (`--moe-cache-mib`, PRs #29887 / #30112) |
| **Auto sizing** `--moe-cache-mib auto` | sizes the cache from the *measured free VRAM* after KV/compute buffers, with a configurable reserve | this patch (`src/llama-context.cpp`) |
| **Heat pre-seeding** `LLAMA_EXPERT_HEAT=heat/heat_qwen36.bin` | preloads the hottest experts per layer at startup, so the first token is already fast instead of waiting for the LRU to learn | this patch (`src/llama-moe-cache.cpp`) |
| **MTP** (optional) | `--spec-type draft-mtp --spec-draft-n-max 3` — +26% on short contexts | llama.cpp upstream |

The whole diff against upstream is **123 lines in 4 files** — see
[`patches/moefast.patch`](patches/moefast.patch).

This repo ships **only the patch** (no vendored llama.cpp source): the patch is pinned to upstream
commit `1ba568c` (2026-10-20 master). Upstream moves fast — the three files we touch each changed
once within the next day — so on newer revisions expect a small conflict and use `git apply -3`
(or rebase the patch; it is 4 files, ~120 lines).

---

## Quick start

```bash
# 1. upstream at the commit this patch was written against
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 1ba568c          # pinned: 2026-10-20 master

# 2. apply the patch
git apply /path/to/moefast.patch        # or: git apply moefast/patches/moefast.patch
# if upstream has moved on and this fails, try a 3-way merge:
#   git apply -3 moefast/patches/moefast.patch

# 3. drop this directory next to it (so `moefast/` sits inside the llama.cpp root)
cp -r /path/to/moefast .

# 4. build (CUDA)
./moefast/install.sh                    # -> build/bin/llama-server

# 5. run
./moefast/run.sh /path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
#    -> http://127.0.0.1:8080/   (OpenAI-compatible: /v1)
```

Requirements: CMake ≥ 3.24, CUDA toolkit, C++17 compiler. No prebuilt binaries on purpose — a CUDA
binary only runs on one arch/OS/CUDA combination.

---

## Usage

### `run.sh`

```bash
./moefast/run.sh /path/to/model.gguf [extra llama-server args...]
```

Defaults (tuned for 16 GB):

```
-ngl 99 -ot "exps=CPU" -c 8192 -b 2048 -ub 2048 \
--moe-cache-mib auto --fit off --parallel 1
LLAMA_EXPERT_HEAT=moefast/heat/heat_qwen36.bin
```

Environment knobs:

| Variable | Default | Meaning |
|---|---|---|
| `MOEFAST_PORT` | `8080` | HTTP port |
| `MOEFAST_CTX` | `8192` | context per slot |
| `MOEFAST_PARALLEL` | `1` | concurrent slots |
| `MOEFAST_MTP` | `0` | `>0` enables `draft-mtp` with N drafts (e.g. `3`) |
| `MOEFAST_CACHE_MIB` | `auto` | expert cache: `auto`, `0`, or MiB |
| `MOEFAST_HEAT` | `moefast/heat/heat_qwen36.bin` | heat table path |
| `MOEFAST_HOST` | `127.0.0.1` | bind address |

### VRAM policy (leave room for your desktop / browser / video)

Three ways, in order of precedence:

```bash
# a) exact cache size in MiB
LLAMA_MOE_CACHE_MIB=7168 llama-server ... --moe-cache-mib auto

# b) auto, but never eat the last N MiB
LLAMA_MOE_CACHE_RESERVE_MIB=2048 llama-server ... --moe-cache-mib auto

# c) plain upstream behaviour
llama-server ... --moe-cache-mib 7168
```

Measured on a 16 GB card (200K ctx, KV k8v4):

| Cache | VRAM used | Left for the system | decode (1K prompt) |
|---|---:|---:|---:|
| auto (full) | 15.6 GB | 0.7 GB | 98.5 tok/s |
| **7168 MiB** | **12.6 GB** | **3.7 GB** | 82.2 tok/s |

Everything not in the cache stays in host RAM (mmap page cache), so lowering the cache is all you
need to do — no extra flags.

### systemd user unit (recommended for daily use)

```ini
# ~/.config/systemd/user/lododo.service
[Unit]
Description=moefast llama-server
After=network.target

[Service]
Environment=LLAMA_EXPERT_HEAT=/path/to/llama.cpp/moefast/heat/heat_qwen36.bin
Environment=LLAMA_MOE_CACHE_MIB=7168
ExecStart=/path/to/llama.cpp/build/bin/llama-server \
  -m "/path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" --alias moefast \
  -ngl 99 -ot "exps=CPU" -c 204800 -ctk q8_0 -ctv q4_0 -fa on \
  -b 2048 -ub 2048 -n 8192 --parallel 1 --moe-cache-mib auto --fit off \
  --port 8080 --host 127.0.0.1
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload && systemctl --user enable --now lododo
```

---

## Gotchas (learned the hard way)

1. **Use `-ot "exps=CPU"`, not `-ncmoe` / `-cmoe`.** The latter also move the router
   (`ffn_gate_inp`) to the CPU, and then the expert cache silently disables itself.
2. **`-ub 2048`** is what makes prompt processing usable with CPU-resident experts
   (1,302 → 2,270 tok/s in our tests).
3. **MTP needs the GGUF that contains the MTP block** (`blk.<n>.nextn.*`, 753 tensors for the
   Unsloth MTP build). A 733-tensor file fails with
   `context type MTP requested but model doesn't contain MTP layers`.
4. **MTP is context-dependent**: +26% on short contexts, slightly negative past ~1K tokens.
5. **`-ctv q4_0` requires `-fa on`** (quantized V cache needs Flash Attention).
6. Long sessions eat the *gateway* process memory, not VRAM — if you drive this from an agent
   framework, keep session context bounded.

---

## Heat tables

`heat/heat_qwen36.bin` (40 KB, format `MOEHEAT1`) contains per-layer expert sizes plus routing counts
collected from 128 prompts on the Qwen3.6-35B-A3B base model. See
[`heat/README.md`](heat/README.md) for the binary layout.

Routing distribution is a property of the base model, so fine-tunes of the same base share it.
If you drop the file, everything still works — the cache just starts cold and learns by LRU.

---

## Repository layout

```
moefast/
├── README.md              this file
├── ACKNOWLEDGEMENTS.md    credits (llama.cpp, PR authors, prior art)
├── LICENSE                MIT
├── install.sh             configure + build llama-server
├── run.sh                 launch with tuned defaults
├── Dockerfile             build/run from a CUDA image
├── heat/
│   ├── heat_qwen36.bin    routing heat table (40 KB)
│   └── README.md          binary format
└── patches/
    └── moefast.patch      +123/−4 lines against upstream 1ba568c
```

## Credits

The MoE expert cache is llama.cpp's work (PRs **#29887** and **#30112**). Our patches add auto
sizing and heat pre-seeding on top. Earlier community prototypes — **#27861** by *csantiago78*,
**#26563** by *miltos22*, *ParmesanParty/llama.cpp*, *neurall/llama.cpp* — shaped the design, and
**Strata** was the reference we benchmarked against. Full list in
[ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md).

## License

MIT — see [LICENSE](LICENSE).
