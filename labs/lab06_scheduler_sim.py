"""Lab06：推理服务调度的离散时间模拟器。

比较三种策略：(a) 静态批；(b) 连续批 + 预填充优先（有新请求就先做一轮纯预填充）；
(c) 连续批 + 分块预填充（每步 token 预算内先排解码，再用剩余预算排预填充块）。
KV 容量以块计；显存不足时按“后来先抢占”的顺序抢占请求，被抢占者丢掉 KV，
之后把 prompt+已生成 token 重新预填充（preemption by recompute）。

注意：迭代耗时模型是“示意性”的参数化模型，不是任何硬件上的实测值：
    t_iter = c0 + c_pf×预填充token数 + c_dec×解码序列数 + c_attn×本步涉及的上下文token数
它只用来直观比较调度策略的趋势；想要真实数字，请在 H800 上用 vLLM/SGLang 实测。
运行：python lab06_scheduler_sim.py --test   （纯 Python；--device 不起作用）
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from collections import deque
from dataclasses import dataclass


@dataclass
class Params:  # 时间单位 ms；全部为示意值
    c0: float = 5.0          # 每次迭代的固定开销（kernel 启动、调度、采样…）
    c_pf: float = 0.06       # 每个预填充 token
    c_dec: float = 0.25      # 每个解码序列
    c_attn: float = 0.0005   # 每个被读取的上下文 token（KV 读取）
    block: int = 16          # 每个 KV 块的 token 数
    kv_blocks: int = 4096    # KV 容量（块）
    max_seqs: int = 64       # 同时运行的最大序列数
    static_batch: int = 32
    max_batched: int = 8192  # (b) 每轮预填充的 token 上限
    budget: int = 512        # (c) 每步 token 预算


@dataclass
class Req:
    rid: int
    arrival: float
    prompt: int
    out: int
    computed: int = 0        # 已有 KV 的 token 数
    gen: int = 0             # 已生成 token 数
    blocks: int = 0
    first: float = -1.0
    finish: float = -1.0

    @property
    def ntok(self) -> int:
        return self.prompt + self.gen


class Sim:
    def __init__(self, reqs: list[Req], p: Params) -> None:
        self.p, self.reqs, self.t = p, reqs, 0.0
        self.pending = deque(sorted(reqs, key=lambda r: r.arrival))
        self.waiting: deque[Req] = deque()
        self.running: list[Req] = []
        self.free, self.peak, self.preempts, self.max_tok = p.kv_blocks, 0, 0, 0
        self.preempted_now = False

    def admit(self) -> None:
        while self.pending and self.pending[0].arrival <= self.t:
            self.waiting.append(self.pending.popleft())

    def alloc(self, r: Req, n: int) -> bool:
        need = math.ceil((r.computed + n) / self.p.block) - r.blocks
        if need > self.free:
            return False
        self.free -= need
        r.blocks += need
        self.peak = max(self.peak, self.p.kv_blocks - self.free)
        assert 0 <= self.free <= self.p.kv_blocks, "KV 容量越界"
        return True

    def release(self, r: Req) -> None:
        self.free, r.blocks = self.free + r.blocks, 0

    def preempt(self, r: Req) -> None:
        self.running.remove(r)
        self.release(r)
        r.computed = 0                   # 丢掉 KV，之后重算 prompt+已生成部分
        self.waiting.appendleft(r)
        self.preempts += 1
        self.preempted_now = True

    def cost(self, n_pf: int, n_dec: int, n_ctx: int) -> float:
        p = self.p
        return p.c0 + p.c_pf * n_pf + p.c_dec * n_dec + p.c_attn * n_ctx

    def schedule_running(self, budget: float) -> list[tuple[Req, int]]:
        sched = []
        for r in list(self.running):
            n = int(min(r.ntok - r.computed, budget))
            if r not in self.running or n <= 0:
                continue
            while not self.alloc(r, n):  # 显存不足：抢占最后加入的请求
                victim = self.running[-1]
                self.preempt(victim)
                if victim is r:
                    break
            if r in self.running:
                sched.append((r, n))
                budget -= n
        return sched

    def admit_waiting(self, sched: list, budget: float, chunk: bool) -> None:
        while (self.waiting and budget > 0 and not self.preempted_now
               and len(self.running) < self.p.max_seqs):
            r = self.waiting[0]
            n = r.ntok - r.computed
            n = int(min(n, budget)) if chunk else n
            if n > budget or not self.alloc(r, n):
                break
            self.running.append(self.waiting.popleft())
            sched.append((r, n))
            budget -= n

    def execute(self, sched: list[tuple[Req, int]]) -> None:
        dec = [r for r, n in sched if r.gen > 0 and r.computed == r.ntok - 1]
        n_pf = sum(n for r, n in sched) - len(dec)
        self.max_tok = max(self.max_tok, n_pf + len(dec))
        self.t += self.cost(n_pf, len(dec), sum(r.computed + n for r, n in sched))
        for r, n in sched:
            r.computed += n
            if r.computed == r.ntok:      # 所有已知 token 都算完 → 采样出一个新 token
                r.gen += 1
                r.first = self.t if r.first < 0 else r.first
                if r.gen == r.out:
                    r.finish = self.t
                    self.running.remove(r)
                    self.release(r)


def run_static(sim: Sim) -> None:
    """静态批：凑一批，按最长 prompt 补齐预填充，解码到整批最长输出才接下一批。"""
    while sim.pending or sim.waiting:
        sim.admit()
        if not sim.waiting:
            sim.t = sim.pending[0].arrival
            continue
        batch: list[Req] = []
        while sim.waiting and len(batch) < sim.p.static_batch:
            r = sim.waiting[0]
            r.computed = r.prompt + r.out  # 无法抢占：按最大长度预留 KV
            if not sim.alloc(r, 0):
                r.computed = 0
                break
            batch.append(sim.waiting.popleft())
        assert batch, "单个请求超过 KV 容量"
        n, L = len(batch), max(r.prompt for r in batch)
        for step in range(max(r.out for r in batch)):  # 已完成的序列仍占批位(padding)
            if step == 0:                              # 预填充同时产出第 1 个 token
                sim.t += sim.cost(n * L, 0, n * L)
            else:
                sim.t += sim.cost(0, n, n * (L + step))
            for r in batch:
                if r.gen < r.out:
                    r.gen += 1
                    r.first = sim.t if r.first < 0 else r.first
                    r.finish = sim.t if r.gen == r.out else r.finish
        for r in batch:
            sim.release(r)


def simulate(work: list[tuple[float, int, int]], policy: str, p: Params) -> Sim:
    sim = Sim([Req(i, a, pl, o) for i, (a, pl, o) in enumerate(work)], p)
    if policy == "static":
        run_static(sim)
        return sim
    while sim.pending or sim.waiting or sim.running:
        sim.admit()
        if not sim.waiting and not sim.running:
            sim.t = sim.pending[0].arrival
            continue
        sim.preempted_now = False
        if policy == "prefill_first":
            sched: list = []
            sim.admit_waiting(sched, p.max_batched, chunk=False)
            sched = sched or sim.schedule_running(math.inf)
        else:
            sched = sim.schedule_running(p.budget)
            sim.admit_waiting(sched, p.budget - sum(n for _, n in sched), chunk=True)
        assert sched, "调度不出任何请求（单个请求超过 KV 容量？）"
        sim.execute(sched)
    return sim


def workload(n: int, rate: float, seed: int) -> list[tuple[float, int, int]]:
    rng, t, out = random.Random(seed), 0.0, []
    for _ in range(n):  # 泊松到达：指数分布的到达间隔（rate: 请求/秒）
        t += rng.expovariate(rate) * 1000
        out.append((t, rng.randint(64, 1024), rng.randint(16, 256)))
    return out


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def stats(sim: Sim) -> dict[str, float]:
    rs = sim.reqs
    span = max(r.finish for r in rs) - min(r.arrival for r in rs)
    ttft = [r.first - r.arrival for r in rs]
    tpot = [(r.finish - r.first) / (r.out - 1) for r in rs if r.out > 1]
    return {"tok/s": sum(r.out for r in rs) / span * 1e3, "ttft50": pct(ttft, .5),
            "ttft99": pct(ttft, .99), "tpot50": pct(tpot, .5), "tpot99": pct(tpot, .99)}


RESULTS: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" | {detail}" if detail else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--rate", type=float, default=6.0, help="请求/秒")
    ap.add_argument("--kv-blocks", type=int, default=Params.kv_blocks)
    ap.add_argument("--budget", type=int, default=Params.budget)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="纯 Python 实验，此参数不起作用")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    for f in (sys.stdout, sys.stderr):  # Windows 控制台默认 GBK 时强制 UTF-8
        if hasattr(f, "reconfigure"):
            f.reconfigure(encoding="utf-8")
    work = workload(args.n, args.rate, args.seed)
    p = Params(kv_blocks=args.kv_blocks, budget=args.budget)
    print(f"{args.n} reqs, Poisson {args.rate}/s, "
          f"KV={p.kv_blocks} blocks x {p.block} tok  (illustrative cost model, ms)")
    print(f"{'policy':14}{'tok/s':>8}{'TTFT50':>9}{'TTFT99':>9}{'TPOT50':>8}"
          f"{'TPOT99':>8}{'preempt':>8}{'peakKV':>7}")
    sims = {}
    for pol in ("static", "prefill_first", "chunked"):
        sims[pol] = sim = simulate(work, pol, p)
        s = stats(sim)
        print(f"{pol:14}{s['tok/s']:8.0f}{s['ttft50']:9.0f}{s['ttft99']:9.0f}"
              f"{s['tpot50']:8.1f}{s['tpot99']:8.1f}{sim.preempts:8d}{sim.peak:7d}")
    if not args.test:
        return
    tight = Params(kv_blocks=300, budget=args.budget)
    for pol in ("prefill_first", "chunked"):
        sims[pol + "@tight"] = simulate(work, pol, tight)
    for name, sim in sims.items():
        ok = all(r.gen == r.out and r.arrival <= r.first <= r.finish for r in sim.reqs)
        check(f"{name}: every request gets exactly its tokens", ok)
        check(f"{name}: KV capacity never exceeded", sim.peak <= sim.p.kv_blocks,
              f"peak={sim.peak}/{sim.p.kv_blocks} preemptions={sim.preempts}")
    mt = sims["chunked"].max_tok
    check("chunked: per-iteration tokens <= budget", mt <= p.budget, f"max={mt}")
    pre = sims["prefill_first@tight"].preempts + sims["chunked@tight"].preempts
    check("tight KV forces preemption (recompute path exercised)", pre > 0, f"{pre}")
    thr = {k: stats(sims[k])["tok/s"] for k in ("static", "prefill_first", "chunked")}
    check("continuous batching beats static in throughput",
          min(thr["prefill_first"], thr["chunked"]) > thr["static"],
          ", ".join(f"{k}={v:.0f}" for k, v in thr.items()))
    print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
