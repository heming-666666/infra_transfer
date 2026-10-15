"""Lab05：玩具版分页 KV cache（PagedAttention 的数据结构）+ 自动前缀缓存。

- 物理块池：固定大小的块、引用计数、空闲队列；序列用“块表”把逻辑块映射到物理块。
- slot = 物理块号 × block_size + 块内偏移；写 K/V 时按 slot 散写（slot mapping）。
- 参考 paged attention：按块表 gather K/V 再算注意力，必须与连续存储的结果一致。
  （真实 kernel 不会真的 gather 出连续副本，而是在 kernel 内按块表寻址。）
- 前缀缓存：只缓存“写满”的块，键 = hash(父块哈希, 本块 token)。ref=0 的缓存块
  留在空闲队列里（LRU 顺序），真正被重新分配时才驱逐。只共享满块，因此无需写时复制。
运行：python lab05_paged_kv.py --device cpu --test
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import OrderedDict

import torch
from torch import Tensor


def block_hash(parent: str, tokens: list[int]) -> str:
    """链式哈希：块内 token 相同但前文不同，K/V 也不同，所以要带上父哈希。"""
    return hashlib.sha256(f"{parent}|{tokens}".encode()).hexdigest()[:16]


class BlockPool:
    def __init__(self, n: int) -> None:
        self.n, self.ref = n, [0] * n
        # 空闲队列兼 LRU：队首 = 最久未用，优先被分配/驱逐
        self.free: OrderedDict[int, None] = OrderedDict.fromkeys(range(n))
        self.hash_of: dict[int, str] = {}
        self.cached: dict[str, int] = {}
        self.hits = self.evictions = 0

    def allocate(self) -> int:
        if not self.free:
            raise MemoryError("KV 块耗尽")
        blk, _ = self.free.popitem(last=False)
        if blk in self.hash_of:                      # 仍缓存着旧前缀：此刻驱逐
            del self.cached[self.hash_of.pop(blk)]
            self.evictions += 1
        self.ref[blk] = 1
        return blk

    def lookup(self, h: str) -> int | None:
        blk = self.cached.get(h)
        if blk is not None:
            if self.ref[blk] == 0:
                del self.free[blk]                   # 命中空闲缓存块：从空闲队列“复活”
            self.ref[blk] += 1
            self.hits += 1
        return blk

    def release(self, blk: int) -> None:
        self.ref[blk] -= 1
        assert self.ref[blk] >= 0
        if self.ref[blk] == 0:
            self.free[blk] = None                    # 放队尾：最后被驱逐

    def register(self, blk: int, h: str) -> None:
        if h not in self.cached:
            self.cached[h], self.hash_of[blk] = blk, h


class PagedKV:
    def __init__(self, n_blocks: int, bs: int, n_kv: int, d: int, dev: torch.device):
        self.bs, self.pool = bs, BlockPool(n_blocks)
        self.k = torch.zeros(n_blocks, bs, n_kv, d, device=dev)
        self.v = torch.zeros_like(self.k)
        self.tables: dict[int, list[int]] = {}       # seq -> 块表
        self.tokens: dict[int, list[int]] = {}       # seq -> 已写入 K/V 的 token
        self.hashes: dict[int, list[str]] = {}       # seq -> 满块哈希链

    def add(self, sid: int, prompt: list[int]) -> int:
        """登记请求并查前缀缓存，返回命中的 token 数（这些不必重算）。"""
        table: list[int] = []
        hashes: list[str] = []
        bs = self.bs
        for i in range((len(prompt) - 1) // bs):  # 至少留 1 个 token 现算出 logits
            h = block_hash(hashes[-1] if hashes else "", prompt[i * bs:(i + 1) * bs])
            blk = self.pool.lookup(h)
            if blk is None:
                break
            table.append(blk)
            hashes.append(h)
        hit = len(table) * bs
        self.tables[sid], self.hashes[sid] = table, hashes
        self.tokens[sid] = prompt[:hit]
        return hit

    def append(self, sid: int, toks: list[int], k: Tensor, v: Tensor) -> list[int]:
        """写入新 token 的 K/V（[n, n_kv, d]），按需分配块，返回 slot mapping。"""
        table, seq, slots = self.tables[sid], self.tokens[sid], []
        for t in toks:
            pos = len(seq)
            if pos // self.bs == len(table):
                table.append(self.pool.allocate())
            slots.append(table[pos // self.bs] * self.bs + pos % self.bs)
            seq.append(t)
        idx = torch.tensor(slots, device=self.k.device)
        self.k.view(-1, *self.k.shape[2:])[idx] = k.to(self.k.device)
        self.v.view(-1, *self.v.shape[2:])[idx] = v.to(self.v.device)
        hs, bs = self.hashes[sid], self.bs
        while len(hs) < len(seq) // bs:               # 新写满的块：登记到前缀缓存
            i = len(hs)
            hs.append(block_hash(hs[-1] if hs else "", seq[i * bs:(i + 1) * bs]))
            self.pool.register(table[i], hs[-1])
        return slots

    def free(self, sid: int) -> None:
        for blk in reversed(self.tables.pop(sid)):   # 逆序释放：尾部块先被驱逐
            self.pool.release(blk)
        del self.tokens[sid], self.hashes[sid]

    def gather(self, sid: int) -> tuple[Tensor, Tensor]:
        tb = torch.tensor(self.tables[sid], device=self.k.device)
        n = len(self.tokens[sid])
        return self.k[tb].flatten(0, 1)[:n], self.v[tb].flatten(0, 1)[:n]


def attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """单个 query token 的 GQA 注意力。q: [H, d]；k/v: [n, KV, d]。"""
    rep = q.shape[0] // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    w = torch.einsum("hd,nhd->hn", q, k) * q.shape[-1] ** -0.5
    return torch.einsum("hn,nhd->hd", w.softmax(-1), v)


def fake_kv(tokens: list[int], start: int, n_kv: int, d: int) -> tuple[Tensor, Tensor]:
    """假“模型”：位置 i 的 K/V 由 tokens[:i+1] 决定（真实模型同样依赖全部前文）。"""
    ks, vs, h = [], [], 0
    for i, t in enumerate(tokens):
        h = (h * 1_000_003 + t + 1) % (2**61 - 1)
        if i >= start:
            g = torch.Generator().manual_seed(h)
            ks.append(torch.randn(n_kv, d, generator=g))
            vs.append(torch.randn(n_kv, d, generator=g))
    return torch.stack(ks), torch.stack(vs)


RESULTS: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" | {detail}" if detail else ""))


def leak_free(kv: PagedKV) -> bool:
    """块账目：空闲队列 = ref 为 0 的块；ref 总和 = 所有块表长度之和。"""
    p = kv.pool
    return (set(p.free) == {b for b in range(p.n) if p.ref[b] == 0}
            and sum(p.ref) == sum(map(len, kv.tables.values()))
            and all(p.cached[h] == b for b, h in p.hash_of.items()))


def run(kv: PagedKV, sid: int, toks: list[int], H: int, d: int) -> tuple[int, float]:
    """处理一个请求：查缓存 → 只为未命中的部分写 K/V → 比较 paged 与连续注意力。"""
    n_kv = kv.k.shape[2]
    hit = kv.add(sid, toks)
    k, v = fake_kv(toks, hit, n_kv, d)
    kv.append(sid, toks[hit:], k, v)
    q = torch.randn(H, d, device=kv.k.device)
    kf, vf = fake_kv(toks, 0, n_kv, d)                 # 连续存储的参考答案
    ref = attention(q, kf.to(q), vf.to(q))
    return hit, (attention(q, *kv.gather(sid)) - ref).abs().max().item()


def run_tests(dev: torch.device) -> None:
    H, n_kv, d, bs = 8, 2, 16, 4
    kv, ok = PagedKV(64, bs, n_kv, d, dev), True
    seqs = {s: [100 * s + i for i in range(n)] for s, n in [(1, 37), (2, 50), (3, 16)]}
    for s in seqs:
        kv.add(s, [])
    for off in range(0, 50, 7):                        # 交错写入：物理块不连续
        for s, tk in seqs.items():
            if off < len(tk):
                kv.append(s, tk[off:off + 7], *fake_kv(tk[:off + 7], off, n_kv, d))
    for s, tk in seqs.items():
        k, v = fake_kv(tk, 0, n_kv, d)
        q = torch.randn(H, d, device=dev)
        ref = attention(q, k.to(q), v.to(q))
        ok &= (attention(q, *kv.gather(s)) - ref).abs().max().item() < 1e-5
    check("paged attention == contiguous attention", ok,
          f"seq1 block_table={kv.tables[1][:6]}...")
    for s in seqs:
        kv.free(s)
    check("no leak after interleaved alloc/free",
          leak_free(kv) and sum(kv.pool.ref) == 0)
    # 前缀缓存：系统提示 10 个 token（bs=4 → 2 个满块），每个请求追加不同用户 token
    sysp = list(range(1000, 1010))
    kv, hits, ok = PagedKV(12, bs, n_kv, d, dev), [], True
    for sid, user in enumerate([[1, 2, 3], [4, 5], [6, 7, 8, 9]]):
        hit, diff = run(kv, sid, sysp + user, H, d)
        hits.append(hit)
        ok &= diff < 1e-5 and leak_free(kv)
    shared = kv.tables[0][:2]
    check("requests sharing system prompt hit 2 blocks", hits == [0, 8, 8],
          f"hit tokens={hits} pool.hits={kv.pool.hits}")
    check("shared blocks ref-counted", [kv.pool.ref[b] for b in shared] == [3, 3])
    check("shared-block attention == contiguous + no leak", ok)
    for sid in range(3):
        kv.free(sid)
    hit, diff = run(kv, 3, sysp + [42], H, d)          # 所有请求结束后缓存仍可命中
    check("cache survives after requests finish", hit == 8 and diff < 1e-5,
          f"hit={hit}")
    kv.free(3)
    run(kv, 4, list(range(5000, 5045)), H, d)          # 45 token 占满 12 块 → 驱逐
    kv.free(4)
    hit, _ = run(kv, 5, sysp + [1], H, d)
    check("LRU eviction under pressure", kv.pool.evictions > 0 and hit == 0,
          f"evictions={kv.pool.evictions} hit_after={hit}")
    kv.free(5)
    p = kv.pool
    check("block accounting never leaks", leak_free(kv) and len(p.free) == p.n,
          f"free={len(p.free)}/{p.n}")
    # 精确 LRU：先后缓存前缀 X、Y，再占用 8 块；应只驱逐最久未用的 X 尾块
    kv = PagedKV(12, bs, n_kv, d, dev)
    X, Y = list(range(2000, 2009)), list(range(3000, 3009))
    for sid, t in [(10, X), (11, Y)]:
        run(kv, sid, t, H, d)
        kv.free(sid)
    hx = [block_hash("", X[:4])]
    hx.append(block_hash(hx[0], X[4:8]))
    hy = [block_hash("", Y[:4])]
    hy.append(block_hash(hy[0], Y[4:8]))
    kv.add(12, [])
    kv.append(12, list(range(32)), *fake_kv(list(range(32)), 0, n_kv, d))
    alive = [h in kv.pool.cached for h in hx + hy]
    check("LRU evicts least-recently-freed block first",
          alive == [True, False, True, True],
          f"X blocks alive={alive[:2]} Y blocks alive={alive[2:]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    for f in (sys.stdout, sys.stderr):  # Windows 控制台默认 GBK 时强制 UTF-8
        if hasattr(f, "reconfigure"):
            f.reconfigure(encoding="utf-8")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，回退到 CPU")
        args.device = "cpu"
    dev = torch.device(args.device)
    torch.manual_seed(0)
    kv = PagedKV(16, 4, 2, 8, dev)
    for sid, n in [(0, 6), (1, 9)]:
        kv.add(sid, [])
        slots = kv.append(sid, list(range(n)), *fake_kv(list(range(n)), 0, 2, 8))
        print(f"seq{sid}: {n} tokens  block_table={kv.tables[sid]}  slots={slots}")
    print(f"free blocks={len(kv.pool.free)}  cached full blocks={len(kv.pool.cached)}")
    if args.test:
        run_tests(dev)
        print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed")
        sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
