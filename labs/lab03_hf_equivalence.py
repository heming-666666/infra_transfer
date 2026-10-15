"""Lab03：随机初始化小号 HF Llama / Qwen3，把权重拷进 lab02，验证 logits 一致。

本地（不下载任何东西）：python lab03_hf_equivalence.py --device cuda --test
H800 上（需要能访问 Hugging Face 或已有本地权重；本机未运行）：
    python lab03_hf_equivalence.py --device cuda --pretrained <hf_id 或本地目录>
    例：--pretrained Qwen/Qwen3-0.6B（以模型仓库里的实际 config 为准）
  只支持 lab02 能表达的子集：rope_type=default、无 attention/mlp bias、silu、
  全部是全注意力层（无滑窗）的 Llama / Qwen3。其他情况会直接报错退出。
--save-tiny DIR：把随机小 Qwen3 存成 HF 格式，可用 --pretrained DIR 离线演练该路径。
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from torch import Tensor

from lab02_tiny_llama import (RESULTS, Config, TinyLlama, check, greedy, pick_device,
                              utf8_console)

# HF 参数名（model.layers.{i}. 之后的部分）→ lab02 模块名
# 真实推理引擎的权重加载器做的也是这件事
RENAME = {"self_attn.q_proj": "wq", "self_attn.k_proj": "wk", "self_attn.v_proj": "wv",
          "self_attn.o_proj": "wo", "self_attn.q_norm": "q_norm",
          "self_attn.k_norm": "k_norm", "mlp.gate_proj": "w_gate",
          "mlp.up_proj": "w_up", "mlp.down_proj": "w_down",
          "input_layernorm": "attn_norm", "post_attention_layernorm": "mlp_norm"}


def to_lab02_config(hc, max_seq: int = 4096) -> Config:
    rp = getattr(hc, "rope_parameters", None) or {}
    bad = [rp.get("rope_type", "default") != "default", hc.hidden_act != "silu",
           getattr(hc, "attention_bias", False), getattr(hc, "mlp_bias", False),
           any(t != "full_attention" for t in getattr(hc, "layer_types", None) or []),
           hc.model_type not in ("llama", "qwen3")]
    if any(bad):
        sys.exit(f"不支持的配置（lab02 只实现默认 RoPE/无 bias/全注意力）：{bad}")
    return Config(vocab=hc.vocab_size, dim=hc.hidden_size,
                  n_layers=hc.num_hidden_layers,
                  n_heads=hc.num_attention_heads, n_kv_heads=hc.num_key_value_heads,
                  head_dim=hc.head_dim, ffn_dim=hc.intermediate_size,
                  rope_theta=float(rp.get("rope_theta", 10000.0)), eps=hc.rms_norm_eps,
                  qk_norm=hc.model_type == "qwen3",
                  max_seq=min(hc.max_position_embeddings, max_seq))


def convert(sd: dict[str, Tensor]) -> dict[str, Tensor]:
    emb = sd["model.embed_tokens.weight"]
    out = {"tok_emb.weight": emb, "norm.weight": sd["model.norm.weight"],
           "lm_head.weight": sd.get("lm_head.weight", emb)}  # 绑定词嵌入时共用
    for k, v in sd.items():
        if k.startswith("model.layers."):
            i, rest = k.split(".", 3)[2:]
            mod, param = rest.rsplit(".", 1)
            out[f"layers.{i}.{RENAME[mod]}.{param}"] = v
    return out


def load_into_lab02(hf, max_seq: int = 4096) -> TinyLlama:
    ours = TinyLlama(to_lab02_config(hf.config, max_seq))
    ours.load_state_dict(convert(hf.state_dict()), strict=True)  # 多/少一个都报错
    return ours.to(next(hf.parameters()).dtype)


def tiny_hf(arch: str):
    from transformers import (LlamaConfig, LlamaForCausalLM, Qwen3Config,
                              Qwen3ForCausalLM)
    kw = dict(vocab_size=320, hidden_size=128, intermediate_size=344,
              num_hidden_layers=3, num_attention_heads=8, num_key_value_heads=2,
              head_dim=32, max_position_embeddings=512, rms_norm_eps=1e-6,
              rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
              tie_word_embeddings=arch == "qwen3", attn_implementation="eager",
              bos_token_id=None, eos_token_id=None, pad_token_id=None)
    cls, mcls = ((Qwen3Config, Qwen3ForCausalLM) if arch == "qwen3"
                 else (LlamaConfig, LlamaForCausalLM))
    hf = mcls(cls(**kw)).eval()
    with torch.no_grad():  # 加大随机权重、随机化所有 norm 权重：映射错了一定能看出来
        for p in hf.parameters():
            if p.dim() == 1:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(0, 0.08)
    return hf


def compare(hf, ours: TinyLlama, ids: Tensor, n_new: int) -> tuple[float, bool]:
    with torch.no_grad():
        ref = hf(input_ids=ids).logits.float()
    diff = (ours(ids).float() - ref).abs().max().item()
    prompt = ids[:1, :16]
    g_hf = hf.generate(prompt, max_new_tokens=n_new, do_sample=False, use_cache=True)
    same = g_hf[:, prompt.shape[1]:].tolist() == greedy(ours, prompt, n_new, True)
    return diff, same


def run_tiny(dev: torch.device) -> None:
    for arch in ("llama", "qwen3"):
        torch.manual_seed(0)
        hf = tiny_hf(arch)
        ours = load_into_lab02(hf).to(dev).eval()
        hf.to(dev)
        ids = torch.randint(0, hf.config.vocab_size, (2, 48), device=dev)
        diff, same = compare(hf, ours, ids, 16)
        check(f"{arch}: lab02 logits == HF logits (fp32)", diff < 1e-4,
              f"max|diff|={diff:.3e}")
        check(f"{arch}: lab02 greedy(KV cache) == HF generate", same)


def run_pretrained(src: str, dev: torch.device, dtype: torch.dtype) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf = AutoModelForCausalLM.from_pretrained(src, dtype=dtype,
                                              attn_implementation="eager").eval()
    ours = load_into_lab02(hf).to(dev).eval()
    hf.to(dev)
    try:
        tok = AutoTokenizer.from_pretrained(src)
        ids = tok("大模型推理的两个阶段是", return_tensors="pt").input_ids.to(dev)
        assert ids.numel() > 0, "分词器为空"  # 无词表文件时 transformers 可能返回空分词器
    except Exception as e:  # 本地演练目录没有分词器：退回随机 token
        tok, ids = None, torch.randint(0, hf.config.vocab_size, (1, 16), device=dev)
        print(f"[info] 无可用分词器（{type(e).__name__}），使用随机 token")
    with torch.no_grad():
        ref = hf(input_ids=ids).logits.float()
    rel = (ours(ids).float() - ref).abs().max().item() / ref.abs().max().item()
    tol = 1e-3 if dtype == torch.float32 else 5e-2  # 经验阈值，未在真实权重上标定
    check(f"pretrained logits match (rel max diff, {dtype})", rel < tol, f"{rel:.3e}")
    out = greedy(ours, ids, 16, True)
    g_hf = hf.generate(ids, max_new_tokens=16, do_sample=False)[:, ids.shape[1]:]
    check("pretrained greedy == HF generate", out == g_hf.tolist())
    if tok is not None:
        print("lab02 生成：", tok.decode(out[0]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--test", action="store_true", help="随机小模型等价性测试")
    ap.add_argument("--pretrained", default=None, help="HF 模型 id 或本地目录（H800）")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--save-tiny", default=None, help="保存随机小 Qwen3 到该目录")
    args = ap.parse_args()
    utf8_console()
    if args.pretrained is None:          # 本地路径：强制离线，保证不会下载
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import transformers
    dev = pick_device(args.device)
    name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
    print(f"device={name} torch={torch.__version__} "
          f"transformers={transformers.__version__}")
    if args.save_tiny:
        torch.manual_seed(0)
        tiny_hf("qwen3").save_pretrained(args.save_tiny)
        print(f"saved tiny random Qwen3 to {args.save_tiny}")
    if args.pretrained:
        run_pretrained(args.pretrained, dev, getattr(torch, args.dtype))
    elif args.test or not args.save_tiny:
        run_tiny(dev)
    if RESULTS:
        print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed")
        sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
