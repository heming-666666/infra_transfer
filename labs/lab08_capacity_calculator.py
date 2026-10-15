r"""Lab08：由 HF config.json + 你提供的硬件数字，估算显存、KV 容量与屋顶线上限。

用法（硬件数字请自行从官方规格表查得后传入；脚本不内置任何真实型号参数）：
  python lab08_capacity_calculator.py path/to/config.json --hbm-gb <每卡GB> \
      --hbm-tbps <每卡TB/s> --tflops <所选 dtype 的每卡稠密峰值TFLOPS> [--tp 4]
自测：python lab08_capacity_calculator.py --test
约定：GB=1e9 字节、TB/s=1e12 字节/秒、TFLOPS=1e12 FLOP/s；1 次乘加 = 2 FLOP。
估算忽略：bias、激活显存（用 --reserve-gb 预留）、通信、kernel 效率；MoE 按
Qwen-MoE 式字段（num_experts / num_experts_per_tok / moe_intermediate_size）近似，
MLA（如含 kv_lora_rank 的模型）的 KV 公式不同，本脚本不适用。
"""
from __future__ import annotations

import argparse
import json
import math
import sys


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg) if "num_hidden_layers" not in cfg else cfg


def model_numbers(c: dict) -> dict[str, float]:
    h, L, V = c["hidden_size"], c["num_hidden_layers"], c["vocab_size"]
    H = c["num_attention_heads"]
    KV = c.get("num_key_value_heads") or H
    D = c.get("head_dim") or h // H
    if "kv_lora_rank" in c:
        sys.exit("检测到 MLA（kv_lora_rank），本脚本的 KV 公式不适用")
    attn = h * H * D + 2 * h * KV * D + H * D * h            # q, k, v, o
    norms = 2 * h + (2 * D if str(c.get("model_type", "")).startswith("qwen3") else 0)
    E, k = c.get("num_experts", 0), c.get("num_experts_per_tok", 0)
    if E:                                                     # MoE：专家 + 路由
        expert = 3 * h * c["moe_intermediate_size"]
        mlp, mlp_act = E * expert + h * E, k * expert + h * E
    else:
        mlp = mlp_act = 3 * h * c["intermediate_size"]        # gate, up, down
    tied = c.get("tie_word_embeddings", False)
    total = V * h * (1 if tied else 2) + L * (attn + mlp + norms) + h
    return {"L": L, "H": H, "KV": KV, "D": D, "total": total,
            # 每 token 参与矩阵乘的参数（含 LM head，不含嵌入查表与 norm）
            "matmul_active": L * (attn + mlp_act) + V * h,
            "expert_frac": (E * expert) / mlp if E else 0.0, "E": E, "k": k}


def analyze(m: dict, hw: dict) -> dict[str, float]:
    tp, wb, kb = hw["tp"], hw["weight_bytes"], hw["kv_bytes"]
    kv_tok = 2 * m["L"] * m["KV"] * m["D"] * kb   # 每 token 的 KV 字节（全模型）
    kv_heads_gpu = max(1, m["KV"] // tp)                      # tp > KV 时 KV 头被复制
    kv_tok_gpu = 2 * m["L"] * kv_heads_gpu * m["D"] * kb
    w_gpu = m["total"] * wb / tp
    budget = hw["hbm_gb"] * 1e9 * hw["mem_frac"] - w_gpu - hw["reserve_gb"] * 1e9
    return {"params": m["total"], "weight_bytes": m["total"] * wb,
            "kv_per_token": kv_tok, "kv_per_token_per_gpu": kv_tok_gpu,
            "max_kv_tokens": max(0, math.floor(budget / kv_tok_gpu)),
            "ridge": hw["tflops"] / hw["hbm_tbps"]}           # FLOP/字节


def step(m: dict, hw: dict, a: dict, B: int, S: int, prefill: bool) -> dict[str, float]:
    """prefill: 1 条序列 S 个 token；decode: B 条序列各 1 个 token，上下文长 S。"""
    T = S if prefill else B
    attn = (2 * S * S if prefill else 4 * S * B) * m["L"] * m["H"] * m["D"]
    flops = 2 * m["matmul_active"] * T + attn
    wbytes = a["weight_bytes"]
    if m["E"] and not prefill:  # MoE 解码：只读被路由到的专家（均匀路由的上界）
        frac = min(1.0, B * m["k"] / m["E"])
        wbytes *= 1 - m["expert_frac"] * (1 - frac)
    kv = a["kv_per_token"] * (S if prefill else B * S)    # prefill 写 KV，decode 读 KV
    byts = wbytes + kv
    t = max(flops / (hw["tflops"] * 1e12 * hw["tp"]),           # 屋顶线：取两者较慢者
            byts / (hw["hbm_tbps"] * 1e12 * hw["tp"]))
    return {"flops": flops, "bytes": byts, "ai": flops / byts, "time": t,
            "tok_s": T / t,
            "bound": "compute" if flops / byts >= a["ridge"] else "memory"}


def report(m: dict, hw: dict, ctx: int, batches: list[int], prompt: int) -> None:
    a = analyze(m, hw)
    print(f"params={a['params']/1e9:.3f}B  weights={a['weight_bytes']/1e9:.2f} GB "
          f"(per GPU {a['weight_bytes']/hw['tp']/1e9:.2f} GB, tp={hw['tp']})")
    print(f"KV/token={a['kv_per_token']/1024:.1f} KiB (per GPU "
          f"{a['kv_per_token_per_gpu']/1024:.1f} KiB)  "
          f"max KV tokens={a['max_kv_tokens']:,}"
          f"  -> ~{a['max_kv_tokens'] // ctx} seqs @ {ctx} tok")
    print(f"ridge point = {a['ridge']:.1f} FLOP/byte")
    p = step(m, hw, a, 1, prompt, True)
    print(f"prefill {prompt} tok: AI={p['ai']:.1f} ({p['bound']}-bound) "
          f">= {p['time']*1e3:.4g} ms")
    print(f"decode @ ctx={ctx}（上界：忽略通信与 kernel 效率）")
    print(f"{'batch':>6}{'AI':>9}{'bound':>9}{'step ms>=':>12}{'tok/s<=':>12}"
          f"{'per-seq':>11}")
    for B in batches:
        d = step(m, hw, a, B, ctx, False)
        print(f"{B:6d}{d['ai']:9.1f}{d['bound']:>9}{d['time']*1e3:12.4g}"
              f"{d['tok_s']:12.4g}{d['tok_s']/B:11.4g}")


def run_tests() -> bool:
    # 手算：attn=8*8+2*8*4+8*8=192, mlp=3*8*16=384, norms=16 → 每层 592；
    # total = 10*8*2 + 2*592 + 8 = 1352；matmul = 2*(192+384) + 80 = 1232
    cfg = {"vocab_size": 10, "hidden_size": 8, "num_hidden_layers": 2,
           "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
           "intermediate_size": 16, "tie_word_embeddings": False, "model_type": "llama"}
    hw = {"hbm_gb": 1, "hbm_tbps": 2, "tflops": 100, "tp": 1, "mem_frac": 0.5,
          "reserve_gb": 0, "weight_bytes": 2, "kv_bytes": 2}
    m = model_numbers(cfg)
    a = analyze(m, hw)
    # decode B=2,S=10: FLOPs=2*1232*2 + 4*10*2*(2*2*4)=6208，字节=2704+2*10*32=3344
    # prefill S=100:   FLOPs=2*1232*100 + 2*100*100*(2*2*4)=566400，字节=2704+100*32
    d = step(m, hw, a, 2, 10, False)
    p = step(m, hw, a, 1, 100, True)
    want = [("params", m["total"], 1352), ("matmul params", m["matmul_active"], 1232),
            ("KV bytes/token = 2*2*1*4*2", a["kv_per_token"], 32),
            ("max KV tokens = floor((0.5e9-2704)/32)", a["max_kv_tokens"], 15624915),
            ("ridge = 100/2", a["ridge"], 50.0), ("decode FLOPs", d["flops"], 6208),
            ("decode bytes", d["bytes"], 3344),
            ("decode time = 3344/2e12", d["time"], 1.672e-9),
            ("prefill FLOPs", p["flops"], 566400), ("prefill bytes", p["bytes"], 5904)]
    ok = True
    for name, got, exp in want:
        good = math.isclose(got, exp, rel_tol=1e-9)
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} {name} | got={got:.10g} want={exp:.10g}")
    for name, good in [("decode B=2 is memory-bound", d["bound"] == "memory"),
                       ("prefill 100 tok is compute-bound (AI 95.9 > 50)",
                        p["bound"] == "compute"),
                       ("tp=2 with 1 KV head replicates KV (per-GPU KV not halved)",
                        analyze(m, {**hw, "tp": 2})["kv_per_token_per_gpu"] == 32)]:
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} {name}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="HF config.json 路径")
    ap.add_argument("--hbm-gb", type=float, help="每卡显存容量 (GB)")
    ap.add_argument("--hbm-tbps", type=float, help="每卡显存带宽 (TB/s)")
    ap.add_argument("--tflops", type=float, help="每卡所选 dtype 的稠密峰值 (TFLOPS)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mem-frac", type=float, default=0.9, help="可用显存比例")
    ap.add_argument("--reserve-gb", type=float, default=0.0, help="每卡预留给激活等")
    ap.add_argument("--weight-bytes", type=float, default=2, help="bf16=2, fp8=1")
    ap.add_argument("--kv-bytes", type=float, default=2)
    ap.add_argument("--ctx", type=int, default=4096, help="解码时的上下文长度")
    ap.add_argument("--prompt", type=int, default=2048)
    ap.add_argument("--batches", default="1,8,32,128,256")
    ap.add_argument("--device", default="cpu", help="纯 Python 计算，此参数不起作用")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    for f in (sys.stdout, sys.stderr):  # Windows 控制台默认 GBK 时强制 UTF-8
        if hasattr(f, "reconfigure"):
            f.reconfigure(encoding="utf-8")
    if args.test:
        ok = run_tests()
        sys.exit(0 if ok else 1)
    if not (args.config and args.hbm_gb and args.hbm_tbps and args.tflops):
        ap.error("需要 config 路径以及 --hbm-gb --hbm-tbps --tflops")
    hw = {"hbm_gb": args.hbm_gb, "hbm_tbps": args.hbm_tbps, "tflops": args.tflops,
          "tp": args.tp, "mem_frac": args.mem_frac, "reserve_gb": args.reserve_gb,
          "weight_bytes": args.weight_bytes, "kv_bytes": args.kv_bytes}
    report(model_numbers(load(args.config)), hw, args.ctx,
           [int(b) for b in args.batches.split(",")], args.prompt)


if __name__ == "__main__":
    main()
