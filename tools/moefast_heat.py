#!/usr/bin/env python3
"""moefast-heat — 热表工具（0.2）

子命令：
  check    只读 GGUF 元数据，判定支持性 / 计算最小缓存 / 推荐配置（秒级，不加载权重）
  auto     首次引导：跑内置 prompt 学习路由 → 生成 heat.bin（0.2 实现中）
  convert  把学习跑导出的 JSON 转成 heat.bin（0.2 实现中）

用法：
  python3 moefast_heat.py check --model m.gguf --vram 8192
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# 允许直接运行（把 llama.cpp 的 gguf-py 加进路径）
GGUF_PY = os.environ.get("MOEFAST_GGUF_PY", "/home/hurry/llama.cpp-src/gguf-py")
if os.path.isdir(GGUF_PY) and GGUF_PY not in sys.path:
    sys.path.insert(0, GGUF_PY)

try:
    from gguf import GGUFReader
except Exception as e:  # pragma: no cover
    print(f"需要 gguf-py（{GGUF_PY}）：{e}", file=sys.stderr)
    sys.exit(2)

SUPPORTED_ARCHS = {
    "qwen35moe", "qwen3moe", "openai-moe", "qwen4exp", "mixtral", "gemma4",
}

# 专家张量后缀 → (类型, 专家轴)
EXPS_KINDS = (
    (".weight", "weight", 2),
    (".bias",   "bias",   1),
    (".scale",  "scale",  0),
)


def _f32_per_elem(ggml_type: str) -> float:
    t = ggml_type.lower()
    if t in ("f32",):
        return 4.0
    if t in ("f16", "bf16"):
        return 2.0
    if t.startswith("q") or t.startswith("iq") or t.startswith("mxfp"):
        # 粗略：按常见 K-quant 的 BPW 估计（只用于打印，不参与校验）
        bpw = {
            "q4_0": 4.5, "q4_1": 5.0, "q5_0": 5.5, "q5_1": 6.0,
            "q8_0": 8.5, "q2_k": 2.6, "q3_k": 3.4, "q4_k": 4.5,
            "q5_k": 5.5, "q6_k": 6.6, "q8_k": 8.5,
            "iq2_xxs": 2.1, "iq2_xs": 2.3, "iq3_xxs": 2.8, "iq3_s": 3.3,
            "iq4_xs": 4.3, "mxfp4": 4.25,
        }
        return bpw.get(t, 4.5) / 8.0
    return 4.0


def inspect(model_path: str, vram_mib: int, ctx: int = 8192, args_kv: str = "k8v4") -> dict:
    r = GGUFReader(model_path)
    fields = {k: v for k, v in r.fields.items()}

    def kv(name: str, default=None):
        v = fields.get(name)
        return v.contents() if v is not None else default

    arch = str(kv("general.architecture", "") or "")
    name = str(kv("general.name", "") or "")
    n_layer = int(kv(f"{arch}.block_count", 0) or 0)
    n_expert = int(kv(f"{arch}.expert_count", 0) or 0)
    n_expert_used = int(kv(f"{arch}.expert_used_count", 0) or 0)
    n_ctx_train = int(kv(f"{arch}.context_length", 0) or 0)
    n_nextn = int(kv(f"{arch}.nextn_predict_layers", 0) or 0)
    n_head = int(kv(f"{arch}.attention.head_count", 0) or 0)
    n_head_kv_raw = kv(f"{arch}.attention.head_count_kv", 0)
    if isinstance(n_head_kv_raw, (list, tuple)):
        n_head_kv = max([int(x) for x in n_head_kv_raw] or [0])
    else:
        n_head_kv = int(n_head_kv_raw or 0)
    n_embd = int(kv(f"{arch}.embedding_length", 0) or 0)
    k_len = int(kv(f"{arch}.attention.key_length", 0) or (n_embd // n_head if n_head else 0))
    v_len = int(kv(f"{arch}.attention.value_length", 0) or (n_embd // n_head if n_head else 0))

    # 逐层收集 _exps 张量
    layers: dict[int, dict] = {}
    ple = 0
    unknown = []
    for t in r.tensors:
        nm = t.name
        if "nextn" in nm or ".mtp" in nm:
            continue
        if "ple" in nm or "per_layer_token_embd" in nm or "ngram" in nm:
            ple += 1
            continue
        if "ffn_" not in nm or "_exps" not in nm and "gate_up_exps" not in nm:
            continue
        parts = nm.split(".")
        if len(parts) < 3 or not parts[0].startswith("blk"):
            unknown.append(nm)
            continue
        il = int(parts[1])
        if n_nextn > 0 and n_layer > 0 and il >= n_layer - n_nextn:
            continue
        kind = None
        axis = None
        for suf, k, ax in EXPS_KINDS:
            if nm.endswith(suf):
                kind, axis = k, ax
                break
        if kind is None:
            unknown.append(nm)
            continue
        # 张量形状：t.shape 是 GGML ne 顺序
        ne = list(t.shape)
        if len(ne) <= axis:
            unknown.append(nm)
            continue
        ent = layers.setdefault(il, {"weight": 0, "bias": 0, "scale": 0,
                                     "n_expert": ne[axis], "types": set()})
        ent[kind] += int(t.n_bytes)
        ent["n_expert"] = int(ne[axis])
        ent["types"].add(str(t.tensor_type.name).lower())

    fused = any("gate_up_exps" in t.name for t in r.tensors)

    # 每层专家字节（取该层所有专家类张量的总和 / 专家数）
    per_layer_bytes = {}
    for il, e in layers.items():
        if e["n_expert"] > 0:
            per_layer_bytes[il] = (e["weight"] + e["bias"] + e["scale"]) // e["n_expert"]

    expert_bytes = sum(per_layer_bytes.values())
    layout = "fused" if fused else "stacked"
    supported = True
    reasons = []

    if arch not in SUPPORTED_ARCHS:
        supported = False
        reasons.append(f"arch `{arch}` 不在支持清单")
    if not layers:
        supported = False
        reasons.append("未找到任何 ffn_*_exps 张量（不是 MoE 或布局未知）")
    if fused:
        supported = False
        reasons.append("检测到 fused `gate_up_exps`（分段布局，0.2 不支持）")
    counts = {e["n_expert"] for e in layers.values()}
    if len(counts) > 1:
        supported = False
        reasons.append(f"同模型内专家数不一致：{sorted(counts)}")
    if n_expert and counts and n_expert not in counts:
        reasons.append(f"元数据 expert_count={n_expert} 与张量推导 {sorted(counts)} 不一致")

    n_moe_layers = max(1, len(layers))
    per_expert_layer = expert_bytes // n_moe_layers if expert_bytes else 0   # 每层每专家字节
    all_experts_bytes = per_expert_layer * (max(counts) if counts else 0) * n_moe_layers

    # KV 估算（k8v4: K 1.0625 B + V 0.5625 B / 元素；f16: 4 B）
    kv_bytes_per_elem = 1.0625 + 0.5625 if args_kv == "k8v4" else 4.0
    kv_bytes = int(ctx * n_moe_layers * n_head_kv * (k_len + v_len) * kv_bytes_per_elem)

    min_slots = n_expert_used or max(counts or {1})
    # 最坏情况：一个 token 在每层都要 top_k 个专家 → 需要 top_k × 层数 个槽
    min_cache_bytes = min_slots * expert_bytes + 200 * 1024 * 1024

    reserve = max(700 * 1024 ** 2, int(vram_mib * 1024 ** 2 * 0.15))
    usable = vram_mib * 1024 ** 2 - reserve - kv_bytes
    slots_total = max(0, int(usable // per_expert_layer)) if per_expert_layer else 0
    per_layer_slots = slots_total // n_moe_layers

    return {
        "file": os.path.basename(model_path),
        "name": name,
        "arch": arch,
        "supported": supported,
        "reasons": reasons,
        "n_layer_with_moe": len(layers),
        "n_layer_meta": n_layer,
        "n_expert": n_expert or (max(counts) if counts else 0),
        "n_expert_used": n_expert_used,
        "n_ctx_train": n_ctx_train,
        "layout": layout,
        "has_bias": any(e["bias"] for e in layers.values()),
        "has_scale": any(e["scale"] for e in layers.values()),
        "expert_bytes_per_layer": per_expert_layer,
        "all_experts_bytes": all_experts_bytes,
        "kv_bytes_est": kv_bytes,
        "ctx_assumed": ctx,
        "expert_types": sorted({t for e in layers.values() for t in e["types"]}),
        "ple_tensors": ple,
        "unknown_exps": unknown[:5],
        "min_cache_mib": (min_cache_bytes + 1024 ** 2 - 1) // 1024 ** 2,
        "recommend": {
            "vram_mib": vram_mib,
            "reserve_mib": reserve // 1024 ** 2,
            "cache_mib": max(0, usable // 1024 ** 2),
            "slots_per_layer": per_layer_slots,
            "slots_total": slots_total,
            "kv_mib": kv_bytes // 1024 ** 2,
            "cache_effective_mib": (slots_total * per_expert_layer) // 1024 ** 2,
        },
    }


def cmd_check(args) -> int:
    info = inspect(args.model, args.vram, args.ctx, args.kv)
    ok = "✅ 支持" if info["supported"] else "❌ 不支持"
    print(f"file      : {info['file']}")
    print(f"arch      : {info['arch'] or '(unknown)'}   {ok}")
    if not info["supported"]:
        for r in info["reasons"]:
            print(f"  - {r}")
    print(f"layers    : {info['n_layer_with_moe']} (meta {info['n_layer_meta']})")
    print(f"experts   : {info['n_expert']}  top-{info['n_expert_used']}")
    print(f"layout    : {info['layout']}   bias={'yes' if info['has_bias'] else 'no'}"
          f"  scale={'yes' if info['has_scale'] else 'no'}")
    print(f"ctx_train : {info['n_ctx_train']}")
    print(f"per-expert: {info['expert_bytes_per_layer'] / 1024**2:.2f} MiB/layer"
          f"  (全部专家 {info['all_experts_bytes'] / 1024**3:.2f} GiB)")
    print(f"types     : {', '.join(info['expert_types'])}")
    if info["ple_tensors"]:
        print(f"ple       : {info['ple_tensors']} 个非专家张量已排除")
    print(f"min cache : {info['min_cache_mib']} MiB  (slots/layer ≥ top-{info['n_expert_used'] or '?'}，否则缓存被绕过)")
    rec = info["recommend"]
    print(f"kv est    : {rec['kv_mib']} MiB (ctx {info['ctx_assumed']}, {args.kv})")
    print(f"recommend : cache {rec['cache_mib']} MiB (reserve {rec['reserve_mib']} MiB) →"
          f" ~{rec['slots_per_layer']} slots/layer ({rec['slots_total']} total)")
    if rec["slots_per_layer"] < (info["n_expert_used"] or 1):
        print(f"⚠️  该显存下 slots/layer < top_k（{rec['slots_per_layer']} < {info['n_expert_used']}）："
              f"缓存会被静默绕过，至少需要 {info['min_cache_mib']} MiB")
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0 if info["supported"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(prog="moefast-heat")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="只读元数据判定支持性")
    c.add_argument("--model", required=True)
    c.add_argument("--vram", type=int, default=8192)
    c.add_argument("--ctx", type=int, default=8192, help="计划使用的上下文长度（影响 KV 估算）")
    c.add_argument("--kv", default="k8v4", choices=["k8v4", "f16"], help="KV 量化档位")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_check)

    a = sub.add_parser("auto", help="首次引导（0.2 实现中）")
    a.add_argument("--model", required=True)
    a.add_argument("--vram", type=int, default=8192)
    a.add_argument("--out", default="heat.bin")
    a.add_argument("--prompts", default=None)
    a.set_defaults(func=lambda _a: (print("auto: 尚未实现（下一步）"), 2)[1])

    v = sub.add_parser("convert", help="JSON → heat.bin（0.2 实现中）")
    v.add_argument("--from", dest="src", required=True)
    v.add_argument("--to", dest="dst", default="heat.bin")
    v.set_defaults(func=lambda _a: (print("convert: 尚未实现（下一步）"), 2)[1])

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
