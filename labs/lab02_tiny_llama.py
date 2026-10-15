"""Lab02：最小 Llama/Qwen3 风格解码器 + 显式 KV cache（随机权重）。

结构：Embedding → N × [RMSNorm → GQA 注意力(RoPE) → 残差 → RMSNorm → SwiGLU → 残差]
      → RMSNorm → LM head。qk_norm=True 时在 RoPE 前对每个头的 q/k 做 RMSNorm（Qwen3）。
prefill(tokens)：一次并行处理整段提示词，建立 KV cache（计算密集）。
decode_step(token, cache)：每步只算 1 个新 token，复用历史 K/V（访存密集）。
运行：python lab02_tiny_llama.py --device cpu --test
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class Config:
    vocab: int = 512
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2       # GQA：每 4 个 query 头共享 1 组 K/V
    head_dim: int = 32        # Qwen3 等模型里 head_dim 可以 ≠ dim / n_heads
    ffn_dim: int = 688
    rope_theta: float = 10000.0
    eps: float = 1e-6
    qk_norm: bool = False     # True = Qwen3 风格
    max_seq: int = 2048


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(dim)), eps

    def forward(self, x: Tensor) -> Tensor:
        h = x.float()  # 与 HF 一致：在 fp32 里求均方根
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(x.dtype)


def rope_table(cfg: Config) -> tuple[Tensor, Tensor]:
    """HF 约定（rotate_half）：前半维与后半维成对旋转。返回 cos/sin: [max_seq, D]。"""
    d = cfg.head_dim
    inv = 1.0 / cfg.rope_theta ** (torch.arange(0, d, 2).float() / d)
    ang = torch.arange(cfg.max_seq).float()[:, None] * inv[None, :]
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos(), ang.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: [B, T, H, D]；cos/sin: [T, D]。"""
    x1, x2 = x.chunk(2, dim=-1)
    rot = torch.cat([-x2, x1], dim=-1)
    cos, sin = cos[None, :, None, :].to(x.dtype), sin[None, :, None, :].to(x.dtype)
    return x * cos + rot * sin


class Block(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.c = c
        self.attn_norm, self.mlp_norm = RMSNorm(c.dim, c.eps), RMSNorm(c.dim, c.eps)
        self.wq = nn.Linear(c.dim, c.n_heads * c.head_dim, bias=False)
        self.wk = nn.Linear(c.dim, c.n_kv_heads * c.head_dim, bias=False)
        self.wv = nn.Linear(c.dim, c.n_kv_heads * c.head_dim, bias=False)
        self.wo = nn.Linear(c.n_heads * c.head_dim, c.dim, bias=False)
        norm = (lambda: RMSNorm(c.head_dim, c.eps)) if c.qk_norm else nn.Identity
        self.q_norm, self.k_norm = norm(), norm()
        self.w_gate = nn.Linear(c.dim, c.ffn_dim, bias=False)
        self.w_up = nn.Linear(c.dim, c.ffn_dim, bias=False)
        self.w_down = nn.Linear(c.ffn_dim, c.dim, bias=False)

    def attention(self, x: Tensor, cos: Tensor, sin: Tensor,
                  past: tuple[Tensor, Tensor] | None) -> tuple[Tensor, tuple]:
        c, (B, T, _) = self.c, x.shape
        q = self.q_norm(self.wq(x).view(B, T, c.n_heads, c.head_dim))
        k = self.k_norm(self.wk(x).view(B, T, c.n_kv_heads, c.head_dim))
        v = self.wv(x).view(B, T, c.n_kv_heads, c.head_dim)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if past is not None:  # 把新 token 的 K/V 接在历史后面
            k, v = torch.cat([past[0], k], 1), torch.cat([past[1], v], 1)
        S, rep = k.shape[1], c.n_heads // c.n_kv_heads
        kk, vv = k.repeat_interleave(rep, 2), v.repeat_interleave(rep, 2)
        s = torch.einsum("bthd,bshd->bhts", q, kk) * c.head_dim ** -0.5
        # 因果掩码：第 t 个新 token 的绝对位置是 S-T+t，只能看见不晚于它的 key
        qpos = torch.arange(S - T, S, device=x.device)[:, None]
        kpos = torch.arange(S, device=x.device)[None, :]
        p = s.masked_fill(kpos > qpos, float("-inf")).float().softmax(-1)
        o = torch.einsum("bhts,bshd->bthd", p.to(q.dtype), vv).reshape(B, T, -1)
        return self.wo(o), (k, v)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor,
                past: tuple[Tensor, Tensor] | None) -> tuple[Tensor, tuple]:
        h, kv = self.attention(self.attn_norm(x), cos, sin, past)
        x = x + h
        m = self.mlp_norm(x)
        return x + self.w_down(F.silu(self.w_gate(m)) * self.w_up(m)), kv


@dataclass
class KVCache:
    """每层一对 [B, T, n_kv_heads, head_dim]。真实引擎会预分配或分页（见 lab05）。"""
    layers: list[tuple[Tensor, Tensor]] = field(default_factory=list)

    def seq_len(self) -> int:
        return self.layers[0][0].shape[1] if self.layers else 0

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for kv in self.layers for t in kv)


class TinyLlama(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab, bias=False)
        cos, sin = rope_table(cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def forward(self, tokens: Tensor, cache: KVCache | None = None) -> Tensor:
        """tokens: [B, T] → logits [B, T, vocab]。cache=None 即无缓存整段前向。"""
        start = cache.seq_len() if cache is not None else 0
        cos = self.cos[start:start + tokens.shape[1]]
        sin = self.sin[start:start + tokens.shape[1]]
        x, new = self.tok_emb(tokens), []
        for i, blk in enumerate(self.layers):
            past = cache.layers[i] if cache is not None and cache.layers else None
            x, kv = blk(x, cos, sin, past)
            new.append(kv)
        if cache is not None:
            cache.layers = new
        return self.lm_head(self.norm(x))

    def prefill(self, tokens: Tensor) -> tuple[Tensor, KVCache]:
        cache = KVCache()
        return self.forward(tokens, cache), cache

    def decode_step(self, token: Tensor, cache: KVCache) -> Tensor:
        """token: [B, 1]；原地扩展 cache，返回 [B, 1, vocab]。"""
        return self.forward(token, cache)


def greedy(model: TinyLlama, prompt: Tensor, n_new: int, use_cache: bool) -> list:
    out = prompt
    if use_cache:
        logits, cache = model.prefill(prompt)
        for i in range(n_new):
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], 1)
            if i + 1 < n_new:
                logits = model.decode_step(nxt, cache)
    else:  # 无缓存：每步把整段重新算一遍，总计算量 O(T^2)
        for _ in range(n_new):
            nxt = model(out)[:, -1].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], 1)
    return out[:, prompt.shape[1]:].tolist()


def pick_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，回退到 CPU")
        return torch.device("cpu")
    return torch.device(name)


def utf8_console() -> None:
    """Windows 控制台默认 GBK 时，强制 UTF-8 输出中文。"""
    for f in (sys.stdout, sys.stderr):
        if hasattr(f, "reconfigure"):
            f.reconfigure(encoding="utf-8")


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


RESULTS: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" | {detail}" if detail else ""))


def run_tests(model: TinyLlama, dev: torch.device) -> None:
    c, B, P, T = model.cfg, 2, 24, 40
    toks = torch.randint(0, c.vocab, (B, T), device=dev)
    full = model(toks)                                  # 无缓存整段前向
    lp, cache = model.prefill(toks[:, :P])
    steps = [model.decode_step(toks[:, t:t + 1], cache) for t in range(P, T)]
    diff = (torch.cat([lp, *steps], 1) - full).abs().max().item()
    check("prefill+decode logits == full forward", diff < 1e-4, f"max|diff|={diff:.3e}")
    prompt = toks[:1, :8]
    a, b = greedy(model, prompt, 24, True), greedy(model, prompt, 24, False)
    check("greedy with cache == without cache", a == b, f"tokens={a[0][:8]}...")
    for dt in (torch.float32, torch.bfloat16):
        m = model if dt == torch.float32 else TinyLlama(c).to(dev, dt)
        _, kc = m.prefill(toks[:, :T])
        want = 2 * c.n_layers * c.n_kv_heads * c.head_dim * (4 if dt == torch.float32
                                                            else 2) * T * B
        check(f"KV bytes formula ({dt})", kc.nbytes() == want,
              f"measured={kc.nbytes()} = 2*{c.n_layers}*{c.n_kv_heads}*{c.head_dim}"
              f"*{kc.layers[0][0].element_size()}B*{T}tok*{B}batch")


def bench(model: TinyLlama, dev: torch.device, plen: int, steps: int) -> None:
    toks = torch.randint(0, model.cfg.vocab, (1, plen), device=dev)
    for _ in range(2):                                  # 预热
        _, cache = model.prefill(toks)
        model.decode_step(toks[:, :1], cache)
    sync(dev); t0 = time.perf_counter()
    logits, cache = model.prefill(toks)
    sync(dev); t_pre = time.perf_counter() - t0
    nxt = logits[:, -1:].argmax(-1)
    t0 = time.perf_counter()
    for _ in range(steps):
        nxt = model.decode_step(nxt, cache)[:, -1:].argmax(-1)
    sync(dev); t_dec = (time.perf_counter() - t0) / steps
    print(f"prefill {plen} tok: {t_pre*1e3:.2f} ms ({t_pre/plen*1e6:.1f} us/tok) | "
          f"decode: {t_dec*1e3:.2f} ms/tok | KV={cache.nbytes()/1024:.0f} KiB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--bench-prompt", type=int, default=256)
    ap.add_argument("--bench-steps", type=int, default=32)
    args = ap.parse_args()
    utf8_console()
    dev = pick_device(args.device)
    torch.manual_seed(0)
    model = TinyLlama(Config()).to(dev).eval()          # 先在 CPU 上按种子初始化
    n_par = sum(p.numel() for p in model.parameters())
    print(f"device={dev} torch={torch.__version__} params={n_par/1e6:.2f}M")
    if args.test:
        run_tests(model, dev)
    bench(model, dev, args.bench_prompt, args.bench_steps)
    if RESULTS:
        print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed")
        sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
