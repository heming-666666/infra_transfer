"""Lab07：单个 Transformer 块的张量并行（Megatron 式）+ all_reduce 带宽测量。

列并行：Q/K/V、gate/up 按输出维切分 → 每个 rank 持有一部分注意力头 / FFN 列，
        本地即可算完自己的头，无需通信。
行并行：O-proj、down-proj 按输入维切分 → 每个 rank 得到部分和，用 all_reduce 相加。
所以每个块前向只需要 2 次 all_reduce（注意力后、MLP 后），每次 T×dim 个元素。
Linux / H800（本机未运行）：
  torchrun --nproc_per_node=4 lab07_tensor_parallel.py --backend nccl --test
  CPU 演练：torchrun --nproc_per_node=2 lab07_tensor_parallel.py --backend gloo --test
Windows 本机：实测 torch 2.5.1 / 2.13.0 的 torchrun 即使设 USE_LIBUV=0 也会在
  rendezvous 阶段报“PyTorch 未编译 libuv”而失败，因此提供自带启动器（同样走 env://）：
  python lab07_tensor_parallel.py --spawn 2 --backend gloo --test
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor

W = dict[str, Tensor]


def make_weights(dim: int, H: int, KV: int, D: int, ffn: int, seed: int) -> W:
    g = torch.Generator().manual_seed(seed)

    def w(o: int, i: int) -> Tensor:
        return torch.randn(o, i, generator=g) * i ** -0.5
    return {"wq": w(H * D, dim), "wk": w(KV * D, dim), "wv": w(KV * D, dim),
            "wo": w(dim, H * D), "w_gate": w(ffn, dim), "w_up": w(ffn, dim),
            "w_down": w(dim, ffn)}


def shard(w: W, rank: int, tp: int) -> W:
    """nn.Linear 权重形状为 [out, in]：列并行切第 0 维，行并行切第 1 维。"""
    col = {k: w[k].chunk(tp, 0)[rank] for k in ("wq", "wk", "wv", "w_gate", "w_up")}
    row = {k: w[k].chunk(tp, 1)[rank] for k in ("wo", "w_down")}
    return {k: v.contiguous() for k, v in {**col, **row}.items()}


def rms(x: Tensor) -> Tensor:  # 省略可学习权重：norm 权重在各 rank 上是复制的
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def block(x: Tensor, w: W, H: int, KV: int, D: int, reduce: bool) -> Tensor:
    """x: [T, dim]。H/KV 为本 rank 持有的头数；reduce=False 即单进程参考实现。"""
    T = x.shape[0]
    h = rms(x)
    q = (h @ w["wq"].T).view(T, H, D).transpose(0, 1)
    k = (h @ w["wk"].T).view(T, KV, D).transpose(0, 1).repeat_interleave(H // KV, 0)
    v = (h @ w["wv"].T).view(T, KV, D).transpose(0, 1).repeat_interleave(H // KV, 0)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # 本地头，无通信
    o = a.transpose(0, 1).reshape(T, H * D) @ w["wo"].T            # 部分和
    if reduce:
        dist.all_reduce(o)                                         # 通信 #1
    x = x + o
    h = rms(x)
    m = (F.silu(h @ w["w_gate"].T) * (h @ w["w_up"].T)) @ w["w_down"].T
    if reduce:
        dist.all_reduce(m)                                         # 通信 #2
    return x + m


def bench_allreduce(dev: torch.device, max_mb: int, iters: int, rank: int, world: int):
    if rank == 0:
        print(f"{'bytes':>12}{'time(us)':>11}{'algbw GB/s':>12}{'busbw GB/s':>12}")
    size = 4096
    while size <= max_mb * 2**20:
        t = torch.zeros(size // 4, device=dev)
        for _ in range(3):
            dist.all_reduce(t)
        if dev.type == "cuda":
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            for _ in range(iters):
                dist.all_reduce(t)
            e.record()
            torch.cuda.synchronize()
            dt = s.elapsed_time(e) / 1e3 / iters
        else:
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(iters):
                dist.all_reduce(t)
            dt = (time.perf_counter() - t0) / iters
        dtt = torch.tensor([dt], device=dev)
        dist.all_reduce(dtt, op=dist.ReduceOp.MAX)                 # 取最慢的 rank
        alg = size / dtt.item() / 1e9
        if rank == 0:  # busbw = algbw × 2(n-1)/n（与 nccl-tests 的约定一致）
            print(f"{size:12d}{dtt.item()*1e6:11.1f}{alg:12.3f}"
                  f"{alg * 2 * (world - 1) / world:12.3f}")
        size *= 4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    ap.add_argument("--device", default="auto", help="auto: nccl→cuda，gloo→cpu")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--max-mb", type=int, default=16, help="all_reduce 最大消息 MiB")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--spawn", type=int, default=0,
                    help="不用 torchrun，自行启动 N 个进程")
    args = ap.parse_args()
    if args.spawn:                      # 自带启动器：设置 torchrun 本来会设置的环境变量
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29512")
        os.environ.setdefault("USE_LIBUV", "0")
        try:
            mp.spawn(spawn_entry, args=(args,), nprocs=args.spawn)
        except mp.ProcessExitedException as e:
            sys.exit(f"worker failed: {e}")
        return
    worker(args)


def spawn_entry(rank: int, args: argparse.Namespace) -> None:
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(args.spawn))
    worker(args)


def worker(args: argparse.Namespace) -> None:
    for f in (sys.stdout, sys.stderr):  # Windows 控制台默认 GBK 时强制 UTF-8
        if hasattr(f, "reconfigure"):
            f.reconfigure(encoding="utf-8")
    want = "cuda" if args.device == "auto" and args.backend == "nccl" else args.device
    if want.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，回退到 CPU")
        want = "cpu"
    if args.backend == "nccl" and want == "cpu":
        sys.exit("nccl 需要 CUDA；在 CPU 上请用 --backend gloo")
    local = int(os.environ.get("LOCAL_RANK", 0))
    dev = torch.device(f"cuda:{local}" if want.startswith("cuda") else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    dist.init_process_group(args.backend)
    rank, tp = dist.get_rank(), dist.get_world_size()
    H, KV, D = args.heads, args.kv_heads, args.head_dim
    assert H % tp == 0 and KV % tp == 0 and args.ffn % tp == 0, "头数/FFN 需被 TP 整除"
    # 每个 rank 用同一种子生成同一份完整权重，再各取自己的分片（真实系统只加载分片）
    full = make_weights(args.dim, H, KV, D, args.ffn, seed=0)
    mine = {k: v.to(dev) for k, v in shard(full, rank, tp).items()}
    x = torch.randn(args.seq, args.dim, generator=torch.Generator().manual_seed(1))
    ref = block(x.to(dev), {k: v.to(dev) for k, v in full.items()}, H, KV, D, False)
    out = block(x.to(dev), mine, H // tp, KV // tp, D, True)
    diff = (out - ref).abs().max().reshape(1)
    dist.all_reduce(diff, op=dist.ReduceOp.MAX)
    t = torch.full((4,), float(rank + 1), device=dev)
    dist.all_reduce(t)
    n_full = sum(v.numel() for v in full.values())
    n_mine = sum(v.numel() for v in mine.values())
    if rank == 0:
        print(f"backend={args.backend} world={tp} device={dev} "
              f"torch={torch.__version__}")
        print(f"weights/rank: {n_mine} of {n_full} params ({n_mine / n_full:.3f})")
        print(f"comm per block forward: 2 all_reduce x {args.seq}x{args.dim} fp32 "
              f"= {2 * args.seq * args.dim * 4} bytes")
    ok1 = diff.item() < 1e-4
    ok2 = bool(torch.all(t == tp * (tp + 1) / 2))
    if rank == 0 and args.test:
        print(f"{'PASS' if ok1 else 'FAIL'} TP block output == single-process reference"
              f" | max|diff|={diff.item():.3e}")
        print(f"{'PASS' if ok2 else 'FAIL'} all_reduce sum correct | {t[0].item()}")
    bench_allreduce(dev, args.max_mb, args.iters, rank, tp)
    dist.barrier()
    dist.destroy_process_group()
    if args.test and not (ok1 and ok2):
        sys.exit(1)


if __name__ == "__main__":
    main()
