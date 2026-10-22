#!/usr/bin/env python3
"""moefast — one-command launcher for big MoE models on small VRAM.

    ./moefast run model.gguf [--extra args...]

What it does:
  1. reads the GGUF metadata (arch, layers, experts, top_k, per-expert bytes)
  2. measures this machine (VRAM total/free, RAM)
  3. computes the parameters (moe cache size, ctx, batch sizes, offload)
  4. writes/validates a config next to the model  (<model>.moefast.json)
  5. starts llama-server with those parameters

The heavy lifting (expert cache, cold-expert CPU compute) is llama.cpp's own.
This tool only decides the numbers and checks that they make sense.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

GGUF_PY_CANDIDATES = [
    os.environ.get("MOEFAST_GGUF_PY", ""),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"),
    "/usr/local/lib/gguf-py",
]
for _p in GGUF_PY_CANDIDATES:
    if _p and os.path.isdir(os.path.join(_p, "gguf")) and _p not in sys.path:
        sys.path.insert(0, _p)
        break

try:
    from gguf import GGUFReader
except Exception as e:  # pragma: no cover
    print(f"moefast: gguf-py not found (set MOEFAST_GGUF_PY): {e}", file=sys.stderr)
    sys.exit(3)

# 不再按架构写白名单：完全由 GGUF 元数据驱动（有专家张量 → MoE，否则按 dense 处理）
MIB = 1024 * 1024


# ---------------------------------------------------------------- model probe
def probe_model(path: str) -> dict:
    r = GGUFReader(path)
    f = r.fields

    def kv(k, d=None):
        v = f.get(k)
        return v.contents() if v is not None else d

    arch = str(kv("general.architecture", "") or "")
    name = str(kv("general.name", "") or "")
    n_layer = int(kv(f"{arch}.block_count", 0) or 0)
    n_nextn = int(kv(f"{arch}.nextn_predict_layers", 0) or 0)
    n_expert = int(kv(f"{arch}.expert_count", 0) or 0)
    n_used = int(kv(f"{arch}.expert_used_count", 0) or 0)
    n_ctx_train = int(kv(f"{arch}.context_length", 0) or 0)

    exp_bytes = {}          # il -> per-expert bytes (weights only)
    exp_total = 0
    aux_total = 0
    ple_total = 0
    fused = False
    for t in r.tensors:
        nm = t.name
        if "nextn" in nm:
            continue
        if "ple" in nm or "ngram" in nm or "per_layer_token_embd" in nm:
            ple_total += int(t.n_bytes)
            aux_total += int(t.n_bytes)
            continue
        if "_exps" not in nm:
            if ".ffn_" not in nm and int(t.n_bytes) > 64 * MIB:
                aux_total += int(t.n_bytes)
            continue
        if "gate_up_exps" in nm:
            fused = True
        parts = nm.split(".")
        if len(parts) < 3 or not parts[0].startswith("blk"):
            continue
        il, ne = int(parts[1]), list(t.shape)
        if n_nextn and n_layer and il >= n_layer - n_nextn:
            continue
        if nm.endswith(".weight") and len(ne) >= 3:
            exp_bytes[il] = int(t.n_bytes) // int(ne[2])
            exp_total += int(t.n_bytes)

    per_layer = [exp_bytes.get(i, 0) for i in range(max(0, n_layer - n_nextn))]
    is_moe = bool(per_layer) and (n_expert > 0)
    warnings = []
    if n_expert > 0 and not per_layer:
        warnings.append("metadata says this is a MoE, but no `*_exps.weight` tensors were found "
                        "(legacy per-expert layout?) — running without the expert cache")
    if fused:
        warnings.append("fused `gate_up_exps` layout detected — the expert cache may not support it; "
                        "llama.cpp will tell you at load time")

    return {
        "file_bytes": os.path.getsize(path),
        "arch": arch, "name": name, "n_layer": len(per_layer),
        "n_layer_meta": n_layer, "n_expert": n_expert, "n_expert_used": n_used,
        "n_ctx_train": n_ctx_train, "per_layer_expert_bytes": per_layer,
        "expert_bytes_total": exp_total, "aux_bytes": aux_total, "ple_bytes": ple_total,
        "fused": fused, "is_moe": is_moe, "warnings": warnings,
    }


# ------------------------------------------------------------- machine probe
def probe_machine() -> dict:
    vram_total = vram_free = 0
    gpu = ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        if out:
            g, tot, free = [x.strip() for x in out[0].split(",")]
            gpu, vram_total, vram_free = g, int(tot), int(free)
    except Exception:
        pass
    ram_gib = 0
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    ram_gib = int(line.split()[1]) // (1024 * 1024)
                    break
    except Exception:
        pass
    return {"gpu": gpu, "vram_mib": vram_total, "vram_free_mib": vram_free, "ram_gib": ram_gib}


# ------------------------------------------------------------------- sizing
def compute_params(model: dict, machine: dict, ctx_req: int, parallel: int, reserve_mib: int) -> dict:
    per_layer = model["per_layer_expert_bytes"]
    file_mib = int(model.get("file_bytes", 0) / MIB)
    dense_mib = max(0, file_mib - int(model["expert_bytes_total"] / MIB))   # 非专家部分（含 attn/embd/ngram）
    n_layers = max(1, len(per_layer))
    top_k = max(1, model["n_expert_used"])
    bytes_per_slot = max(1, sum(per_layer) // n_layers)          # per layer, per expert

    # 用“当前空闲显存”而不是总显存：机器上可能已经有别的进程在跑
    if not model.get("is_moe"):
        return {"moe_cache_mib": 0, "min_cache_mib": 0, "slots_per_layer": 0,
                "ctx": min(ctx_req, model["n_ctx_train"] or ctx_req),
                "batch": 2048, "ubatch": 2048, "ngl": 99, "ot": [],
                "parallel": parallel, "mtp": 0}

    vram = machine.get("vram_free_mib") or machine["vram_mib"] or 8192
    reserve = max(reserve_mib, int(vram * 0.15), 700)
    kv_mib = estimate_kv_mib(model, ctx_req, parallel)
    usable = max(0, vram - reserve - kv_mib - dense_mib)
    # 缓存大于“专家总量”没有意义
    cache_mib = int(min(usable, model["expert_bytes_total"] / MIB))
    min_cache_mib = int((top_k * n_layers * bytes_per_slot) / MIB) + 64
    if cache_mib < min_cache_mib:
        cache_mib = 0                                           # let the caller warn

    slots = (cache_mib * MIB) // bytes_per_slot if cache_mib else 0
    return {
        "moe_cache_mib": cache_mib,
        "covers_all_experts": cache_mib >= int(model["expert_bytes_total"] / MIB) - 8,
        "dense_mib": dense_mib,
        "min_cache_mib": min_cache_mib,
        "slots_per_layer": slots // n_layers if slots else 0,
        "ctx": min(ctx_req, model["n_ctx_train"] or ctx_req),
        "batch": 2048, "ubatch": 2048,
        "ngl": 99, "ot": ["exps=CPU"], "parallel": parallel, "mtp": 0,
    }


def estimate_kv_mib(model: dict, ctx: int, parallel: int) -> int:
    # conservative: ~1.625 B/elem for k8v4 is not known here, assume f16 (4 B) with 20 KiB/cell/layer
    # (only used as headroom; llama.cpp reports the real value at load time)
    return int(ctx * parallel * 20 * 1024 / MIB)


# ------------------------------------------------------------------ config io
def config_path(model_path: str) -> str:
    return os.path.splitext(model_path)[0] + ".moefast.json"


def load_config(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"moefast: cannot parse {path}: {e}", file=sys.stderr)
        return None


def save_config(path: str, model: dict, machine: dict, params: dict) -> None:
    cfg = {
        "_comment": "auto-generated by `moefast run`; edit and it will be used next time",
        "model": {k: model[k] for k in ("arch", "name", "n_layer", "n_expert", "n_expert_used")},
        "machine": {"gpu": machine["gpu"], "vram_mib": machine["vram_mib"], "ram_gib": machine["ram_gib"]},
        "params": params,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")


def next_suggested_path(path: str) -> str:
    base, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{base}.suggested-{i}{ext}"):
        i += 1
    return f"{base}.suggested-{i}{ext}"


# -------------------------------------------------------------- validation
def validate(cfg: dict, model: dict, machine: dict, path: str, fix_hint: bool) -> tuple[bool, dict]:
    """returns (ok_to_run, params)"""
    cm, mm, pm = cfg.get("model", {}), cfg.get("machine", {}), cfg.get("params", {})

    # 1) model identity must match
    for k in ("arch", "n_layer", "n_expert", "n_expert_used"):
        if cm.get(k) != model[k]:
            print(f"moefast: ✖ config belongs to a different model "
                  f"({k}: config={cm.get(k)} current={model[k]})", file=sys.stderr)
            print(f"         config: {path}", file=sys.stderr)
            print("         fix: delete the config (or rename it) and run again", file=sys.stderr)
            return False, pm

    # 2) resources: if VRAM shrank, suggest new params
    if mm.get("vram_mib") and machine["vram_mib"] and mm["vram_mib"] != machine["vram_mib"]:
        suggest = compute_params(model, machine, pm.get("ctx", 8192), pm.get("parallel", 1),
                                 int(pm.get("reserve_mib", 0) or 0))
        newp = dict(pm)
        for k in ("moe_cache_mib", "min_cache_mib", "slots_per_layer"):
            newp[k] = suggest[k]
        sug = {"model": cm, "machine": {"gpu": machine["gpu"], "vram_mib": machine["vram_mib"],
                                        "ram_gib": machine["ram_gib"]}, "params": newp}
        sp = next_suggested_path(path)
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(sug, f, ensure_ascii=False, indent=2)
        print(f"moefast: ⚠ machine changed ({mm['vram_mib']} → {machine['vram_mib']} MiB VRAM)")
        print(f"         suggested config written: {sp}")
        print(f"         (rename it to {os.path.basename(path)} to use it)")

    # 3) cache sanity
    min_cache = pm.get("min_cache_mib") or 0
    cache = int(pm.get("moe_cache_mib", 0))
    if cache and min_cache and cache < min_cache:
        print(f"moefast: ✖ moe_cache_mib={cache} < minimum {min_cache} MiB "
              f"(cache would be bypassed / refused)", file=sys.stderr)
        return False, pm
    return True, pm


# --------------------------------------------------------------------- main
def cmd_run(args) -> int:
    model_path = args.model
    if not os.path.isfile(model_path):
        print(f"moefast: model not found: {model_path}", file=sys.stderr)
        return 2

    model = probe_model(model_path)
    for w in model.get("warnings", []):
        print(f"moefast: ⚠ {w}", file=sys.stderr)

    machine = probe_machine()
    path = args.config or config_path(model_path)
    cfg = load_config(path)

    if cfg is None:
        params = compute_params(model, machine, args.ctx, args.parallel, args.reserve)
        if params["moe_cache_mib"] < params["min_cache_mib"]:
            print(f"moefast: ⚠ this card is too small for the expert cache "
                  f"(need ≥{params['min_cache_mib']} MiB). Running with cache disabled.",
                  file=sys.stderr)
            params["moe_cache_mib"] = 0
        save_config(path, model, machine, params)
        print(f"moefast: config written → {path}")
    else:
        ok, params = validate(cfg, model, machine, path, args.fix)
        if not ok:
            return 2
        if not params.get("moe_cache_mib") and params.get("min_cache_mib"):
            print(f"moefast: ⚠ cache disabled in config (min useful = {params['min_cache_mib']} MiB)")

    # build the command
    bin_path = args.server or os.environ.get("MOEFAST_LLAMA_SERVER") or shutil.which("llama-server")
    if not bin_path or not os.path.exists(bin_path):
        print("moefast: llama-server not found (use --server /path/to/llama-server)", file=sys.stderr)
        return 3

    cmd = [bin_path, "-m", model_path,
           "-ngl", str(params["ngl"]),
           "-c", str(params["ctx"]),
           "-b", str(params["batch"]), "-ub", str(params["ubatch"]),
           "--parallel", str(params["parallel"]),
           "--port", str(args.port), "--host", args.host]
    for pat in params.get("ot") or []:
        cmd += ["-ot", pat]
    if params.get("moe_cache_mib", 0) > 0:
        cmd += ["--moe-cache-mib", str(params["moe_cache_mib"])]
    if params.get("mtp", 0) > 0:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(params["mtp"])]
    cmd += args.extra

    print(f"moefast: arch={model['arch']} layers={model['n_layer']} experts={model['n_expert']} "
          f"top-{model['n_expert_used']}")
    print(f"moefast: gpu={machine['gpu']} vram={machine['vram_mib']}MiB ram={machine['ram_gib']}GiB")
    if model.get("is_moe"):
        print(f"moefast: moe cache={params['moe_cache_mib']}MiB "
              f"(~{params.get('slots_per_layer', 0)} slots/layer, min {params.get('min_cache_mib', 0)}) "
              f"ctx={params['ctx']}")
    else:
        print(f"moefast: dense model (no experts) — expert cache not used, ctx={params['ctx']}")
    print("moefast: " + " ".join(cmd))

    if args.print_only:
        return 0
    os.execv(cmd[0], cmd)
    return 0


def cmd_show(args) -> int:
    model = probe_model(args.model)
    machine = probe_machine()
    params = compute_params(model, machine, args.ctx, args.parallel, args.reserve)
    print(json.dumps({"model": model, "machine": machine, "computed": params},
                     ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="moefast",
                                 description="one-command launcher for big MoE models on small VRAM")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="probe → compute → write config → start llama-server")
    r.add_argument("model")
    r.add_argument("--config", default=None, help="config path (default: <model>.moefast.json)")
    r.add_argument("--server", default=None, help="path to llama-server")
    r.add_argument("--port", default=8080)
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--ctx", type=int, default=8192)
    r.add_argument("--parallel", type=int, default=1)
    r.add_argument("--reserve", type=int, default=0, help="VRAM reserve in MiB")
    r.add_argument("--mtp", type=int, default=0)
    r.add_argument("--print-only", action="store_true", help="print the command, do not exec")
    r.add_argument("--fix", action="store_true", help="(reserved) rewrite config when resources change")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("show", help="print the probe + computed parameters (no config, no server)")
    s.add_argument("model")
    s.add_argument("--ctx", type=int, default=8192)
    s.add_argument("--parallel", type=int, default=1)
    s.add_argument("--reserve", type=int, default=0)
    s.set_defaults(func=cmd_show)

    args, unknown = ap.parse_known_args()
    if args.cmd == "run":
        args.extra = unknown
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
