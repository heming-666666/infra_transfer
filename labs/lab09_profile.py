r"""Lab09：用 torch.profiler 剖析 lab02 模型的 prefill 与 decode。

- 打印按自身耗时排序的算子表（top ops），导出 Chrome/Perfetto trace 到 labs/out/。
  trace 可在 https://ui.perfetto.dev 或 chrome://tracing 打开。
- 每个阶段同时打 record_function（torch.profiler 可见）和 NVTX 区间（nsys 可见）。
- GPU 上用 CUDA events 计时（kernel 是异步的，CPU 计时器不准）。
本机：python lab09_profile.py --device cuda --test
H800 + Nsight Systems（本机未安装 nsys，未运行）：
  nsys profile -t cuda,nvtx,osrt -o out/lab09_nsys \
      python lab09_profile.py --device cuda --no-torch-profiler --dim 2048 --layers 16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import torch
from torch.profiler import (ProfilerActivity, profile, record_function,
                            supported_activities)

from lab02_tiny_llama import (RESULTS, Config, TinyLlama, check, pick_device,
                              utf8_console)


@contextmanager
def region(name: str, dev: torch.device) -> Iterator[None]:
    """record_function + NVTX：一处标注，两种工具都能看到。"""
    if dev.type == "cuda":
        torch.cuda.nvtx.range_push(name)
    try:
        with record_function(name):
            yield
    finally:
        if dev.type == "cuda":
            torch.cuda.nvtx.range_pop()


def timed_ms(fn: Callable[[], object], dev: torch.device, n: int = 1) -> float:
    if dev.type == "cuda":
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / n
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) * 1e3 / n


def workload(model: TinyLlama, toks: torch.Tensor, steps: int,
             dev: torch.device) -> None:
    with region("prefill", dev):
        logits, cache = model.prefill(toks)
    nxt = logits[:, -1:].argmax(-1)
    for _ in range(steps):
        with region("decode_step", dev):
            nxt = model.decode_step(nxt, cache)[:, -1:].argmax(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--prompt", type=int, default=256)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--no-torch-profiler", action="store_true",
                    help="在 nsys 下运行时使用")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    utf8_console()
    dev = pick_device(args.device)
    torch.manual_seed(0)
    cfg = Config(vocab=args.vocab, dim=args.dim, n_layers=args.layers, n_heads=8,
                 n_kv_heads=2, head_dim=args.dim // 8, ffn_dim=args.dim * 8 // 3)
    model = TinyLlama(cfg).to(dev, getattr(torch, args.dtype)).eval()
    name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
    print(f"device={name} torch={torch.__version__} dtype={args.dtype} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    toks = torch.randint(0, cfg.vocab, (1, args.prompt), device=dev)
    for _ in range(2):                                        # 预热（cuBLAS 初始化等）
        workload(model, toks, 2, dev)
    t_pre = timed_ms(lambda: model.prefill(toks), dev, 3)
    _, cache = model.prefill(toks)
    one = toks[:, :1]
    t_dec = timed_ms(lambda: model.decode_step(one, cache), dev, args.steps)
    timer = "CUDA events" if dev.type == "cuda" else "perf_counter"
    print(f"prefill {args.prompt} tok: {t_pre:.2f} ms "
          f"({t_pre/args.prompt*1e3:.1f} us/tok) | "
          f"decode: {t_dec:.2f} ms/tok  [{timer}]")
    if args.no_torch_profiler:              # nsys 模式：只跑带 NVTX 的负载
        workload(model, toks, args.steps, dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return
    gpu_trace = dev.type == "cuda" and ProfilerActivity.CUDA in supported_activities()
    if dev.type == "cuda" and not gpu_trace:  # 例：Windows 上 kineto 没有可用的 CUPTI
        print("[warn] 本环境 torch.profiler 不支持 CUDA 活动（CUPTI 不可用），"
              "只记录 CPU 侧算子；GPU 时间以上面的 CUDA events 为准，或用 nsys")
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if gpu_trace else [])
    with profile(activities=acts, record_shapes=True) as prof:
        workload(model, toks, args.steps, dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    avg = prof.key_averages()
    key = "self_cpu_time_total"
    if gpu_trace:
        key = ("self_device_time_total" if hasattr(avg[0], "self_device_time_total")
               else "self_cuda_time_total")
    table = avg.table(sort_by=key, row_limit=args.top)
    print(table)
    out = Path(__file__).resolve().parent / "out" / f"lab09_trace_{dev.type}.json"
    out.parent.mkdir(exist_ok=True)
    prof.export_chrome_trace(str(out))
    print(f"trace -> {out} ({out.stat().st_size/1e6:.1f} MB)")
    if args.test:
        ev = json.loads(out.read_text(encoding="utf-8"))["traceEvents"]
        names = {e.get("name") for e in ev}
        check("trace has prefill/decode_step regions",
              {"prefill", "decode_step"} <= names)
        mm = [n for n in ("aten::mm", "aten::addmm", "aten::bmm", "aten::matmul",
                          "aten::linear") if n in table]
        check("top-ops table lists matmul ops", bool(mm), f"{mm}")
        if gpu_trace:
            n_k = sum(1 for e in ev if e.get("cat") == "kernel")
            check("trace has CUDA kernel events", n_k > 0, f"{n_k} kernels")
        elif dev.type == "cuda":
            print("SKIP trace has CUDA kernel events | CUPTI 不可用，无法在本机验证")
        check("prefill per-token cost < decode per-token cost",
              t_pre / args.prompt < t_dec, f"{t_pre/args.prompt:.3f} vs {t_dec:.3f} ms")
        print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed")
        sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
