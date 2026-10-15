"""Lab04：logits 处理器（重复惩罚、温度、top-k、top-p、min-p）与采样器。

本实验的处理顺序：重复惩罚 → 温度 → top-k → top-p → min-p → 采样（参考 HF 约定）。
不同推理框架的顺序可能不同（例如 min-p 与 top-k/top-p 的先后），
顺序不同时保留集合也可能不同，以所用框架的源码为准。
运行：python lab04_sampling.py --device cpu --test
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import torch
from torch import Tensor

NEG_INF = float("-inf")


def repetition_penalty(logits: Tensor, prev: Tensor, penalty: float) -> Tensor:
    """HF 语义：出现过的 token，正 logit 除以 penalty，负 logit 乘以 penalty。
    logits: [B, V]；prev: [B, L] 历史 token id。"""
    if penalty == 1.0 or prev.numel() == 0:
        return logits
    got = logits.gather(-1, prev)
    return logits.scatter(-1, prev, torch.where(got > 0, got / penalty, got * penalty))


def top_k(logits: Tensor, k: int) -> Tensor:
    """保留最大的 k 个；与第 k 名并列的也保留。k<=0 表示不启用。"""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = logits.topk(k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, NEG_INF)


def top_p(logits: Tensor, p: float) -> Tensor:
    """按概率从大到小累加，保留累计刚达到 p 的最小前缀（至少 1 个）。"""
    if p >= 1.0:
        return logits
    srt, idx = logits.sort(dim=-1, descending=True)
    probs = srt.softmax(-1)
    rm = (probs.cumsum(-1) - probs) >= p      # 排在前面的概率和已够 p → 它多余
    rm[..., 0] = False
    return logits.masked_fill(torch.zeros_like(rm).scatter(-1, idx, rm), NEG_INF)


def min_p(logits: Tensor, mp: float) -> Tensor:
    """保留 prob >= mp × 最大概率 的 token。"""
    if mp <= 0:
        return logits
    probs = logits.softmax(-1)
    keep = probs >= mp * probs.max(-1, keepdim=True).values
    return logits.masked_fill(~keep, NEG_INF)


def sample(logits: Tensor, gen: torch.Generator | None = None) -> Tensor:
    """按 softmax(logits) 抽样：argmax(p_i / E_i)，E_i ~ Exp(1) 独立。
    E_i/p_i ~ Exp(速率 p_i)，最小者是 j 的概率恰为 p_j / Σp（"指数竞赛"）。
    这样无需 torch.multinomial，推理引擎中常见这种写法。"""
    probs = logits.softmax(-1)
    e = torch.empty_like(probs).exponential_(1.0, generator=gen)
    return (probs / e).argmax(-1)


@dataclass
class SamplingParams:
    temperature: float = 1.0      # 0 表示贪心
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0


def process(logits: Tensor, prev: Tensor, sp: SamplingParams) -> Tensor:
    logits = repetition_penalty(logits.float(), prev, sp.repetition_penalty)
    logits = logits / sp.temperature
    return min_p(top_p(top_k(logits, sp.top_k), sp.top_p), sp.min_p)


def next_token(logits: Tensor, prev: Tensor, sp: SamplingParams,
               gen: torch.Generator | None = None) -> Tensor:
    if sp.temperature <= 0:
        logits = repetition_penalty(logits.float(), prev, sp.repetition_penalty)
        return logits.argmax(-1)
    return sample(process(logits, prev, sp), gen)


RESULTS: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" | {detail}" if detail else ""))


def kept(x: Tensor) -> set[int]:
    return set(torch.nonzero(x[0] > NEG_INF).flatten().tolist())


def run_tests(dev: torch.device, n: int) -> None:
    # logits = log(p)：softmax 后正好是 p，方便手算。按概率排序：1>3>5>0>4>6>2
    p = torch.tensor([[0.10, 0.40, 0.02, 0.25, 0.05, 0.15, 0.03]], device=dev)
    lg, none = p.log(), torch.empty(1, 0, dtype=torch.long, device=dev)
    for name, got, want in [
        ("top_k=3", kept(top_k(lg, 3)), {1, 3, 5}),
        ("top_k=1 keeps ties", kept(top_k(torch.tensor([[1., 2, 2, 0]]), 1)), {1, 2}),
        ("top_p=0.3", kept(top_p(lg, 0.3)), {1}),
        ("top_p=0.6", kept(top_p(lg, 0.6)), {1, 3}),
        ("top_p=0.75", kept(top_p(lg, 0.75)), {1, 3, 5}),
        ("top_p=0.92", kept(top_p(lg, 0.92)), {1, 3, 5, 0, 4}),
        ("min_p=0.3 (thr 0.12)", kept(min_p(lg, 0.3)), {1, 3, 5}),
        ("min_p=0.2 (thr 0.08)", kept(min_p(lg, 0.2)), {1, 3, 5, 0}),
        ("min_p=0.06 (thr 0.024)", kept(min_p(lg, 0.06)), {1, 3, 5, 0, 4, 6}),
        ("top_k=4 then top_p=0.8", kept(process(lg, none, SamplingParams(1, 4, .8))),
         {1, 3, 5}),
    ]:
        check(f"kept set {name}", got == want, f"{sorted(got)}")
    t = (lg / 0.5).softmax(-1)
    check("temperature 0.5 == p^2 / sum(p^2)", torch.allclose(t, p**2 / (p**2).sum()))
    rp = repetition_penalty(torch.tensor([[2.0, -1.0, 0.5, 3.0]]),
                            torch.tensor([[0, 1, 1]]), 2.0)
    check("repetition penalty", rp.tolist() == [[1.0, -2.0, 0.5, 3.0]],
          f"{rp.tolist()}")
    greedy = next_token(lg, none, SamplingParams(0)).item()
    check("temperature 0 -> greedy", greedy == 1)
    # 经验分布：N 次独立采样的频率应接近“过滤并重新归一化”后的分布（手算闭式）
    cases = [(SamplingParams(1.0, 4, 0.8), {1: 0.5, 3: 0.3125, 5: 0.1875}),
             (SamplingParams(0.5, 0, 1.0, 0.2), {1: 0.16 / 0.2225, 3: 0.0625 / 0.2225})]
    for sp, want in cases:
        gen = torch.Generator(device=dev).manual_seed(0)
        toks = next_token(lg.expand(n, -1), none.expand(n, 0), sp, gen)
        freq = torch.bincount(toks, minlength=7).float() / n
        exp = torch.tensor([want.get(i, 0.0) for i in range(7)], device=dev)
        err = (freq - exp).abs().max().item()
        outside = int(freq[exp == 0].sum().item() * n)
        check(f"empirical dist {sp}", err < 5e-3 and outside == 0,
              f"N={n} max|freq-exp|={err:.5f} samples_outside_kept={outside}")
    g1, g2 = (torch.Generator(device=dev).manual_seed(7) for _ in range(2))
    a = next_token(lg.expand(64, -1), none.expand(64, 0), SamplingParams(), g1)
    b = next_token(lg.expand(64, -1), none.expand(64, 0), SamplingParams(), g2)
    check("same seed -> same samples", torch.equal(a, b))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--samples", type=int, default=200_000)
    args = ap.parse_args()
    for f in (sys.stdout, sys.stderr):  # Windows 控制台默认 GBK 时强制 UTF-8
        if hasattr(f, "reconfigure"):
            f.reconfigure(encoding="utf-8")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，回退到 CPU")
        args.device = "cpu"
    dev = torch.device(args.device)
    torch.manual_seed(0)
    lg = torch.randn(1, 10, device=dev) * 2
    print("probs:", [round(x, 3) for x in lg.softmax(-1)[0].tolist()])
    for sp in [SamplingParams(1.0, 3), SamplingParams(1.0, 0, 0.9),
               SamplingParams(0.7, 0, 1.0, 0.1)]:
        print(f"{sp} -> kept {sorted(kept(process(lg, lg[:, :0].long(), sp)))}")
    if args.test:
        run_tests(dev, args.samples)
        print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed")
        sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
