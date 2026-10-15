"""Lab01：玩具版字节级 BPE（byte-level BPE）与流式反分词（detokenize）。

要点
1. 初始词表 = 256 个字节，任何字符串都能编码，不存在 OOV。
2. 训练：反复统计相邻 token 对的频次，把最频繁的一对合并成新 token。
3. 一个汉字占 3 个 UTF-8 字节；语料里没见过的汉字会被拆成多个字节 token。
4. 流式输出时，一个 token 可能只含半个字符；直接 decode 会得到 U+FFFD(�)。
   流式反分词器把不完整的 UTF-8 尾巴暂存起来，等后续 token 补齐再输出。
运行：python lab01_bpe_toy.py --test   （纯 Python；--device 仅为统一接口，不起作用）
"""
from __future__ import annotations

import argparse
import codecs
import random
import re
import sys
from collections import Counter

CORPUS = """
大模型推理分为预填充和解码两个阶段。预填充阶段一次处理整段提示词，计算密集；
解码阶段每一步只生成一个词元，需要反复读取权重和键值缓存，访存密集。
The prefill phase processes the whole prompt in parallel and is compute bound.
The decode phase generates one token per step and is memory bandwidth bound.
键值缓存保存每一层的键和值，避免重复计算历史词元。
The KV cache stores the keys and values of every layer for every previous token.
分页注意力把键值缓存切成固定大小的块，用块表记录逻辑块到物理块的映射。
PagedAttention splits the KV cache into fixed size blocks and uses a block table.
连续批处理在每一步迭代都可以加入新请求，也可以移除已经完成的请求。
Continuous batching adds new requests and removes finished requests at every step.
张量并行把一个线性层的权重切到多张显卡上，再用全规约把部分结果加起来。
Tensor parallelism shards each linear layer across GPUs and then uses all reduce.
分词器把文本切成词元，流式输出时必须小心处理不完整的字节序列。
The tokenizer splits text into tokens; streaming output must handle partial bytes.
吞吐量、首词元延迟和每词元延迟是推理服务最重要的三个指标。
Throughput, time to first token and time per output token are the key metrics.
采样时可以使用温度、top-k、top-p 和 min-p 来控制输出的多样性。
Sampling uses temperature, top-k, top-p and min-p to control the diversity.
前缀缓存让共享系统提示词的请求复用已经计算好的键值块，减少预填充的计算。
Prefix caching lets requests that share a system prompt reuse cached KV blocks.
推理服务的调度器决定每一步处理哪些请求，显存不够时会抢占请求并重新计算。
The scheduler decides which requests run in each step and may preempt requests.
"""

# 预分词：英文词、数字、连续汉字、其他符号、空白各自成块；合并不跨块。
PAT = re.compile(r" ?[A-Za-z]+| ?\d+| ?[一-鿿]+"
                 r"| ?[^\sA-Za-z\d一-鿿]+|\s+")
Merges = dict[tuple[int, int], int]
REPL = chr(0xFFFD)  # U+FFFD 替换字符：解码遇到非法/不完整字节时产生


def merge(ids: tuple[int, ...], pair: tuple[int, int], new: int) -> tuple[int, ...]:
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


def train_bpe(text: str, n_merges: int) -> Merges:
    """返回 {(左, 右): 新 id}；新 id 越小越先学到，编码时优先合并。"""
    words = Counter(tuple(w.encode("utf-8")) for w in PAT.findall(text))
    merges: Merges = {}
    for new in range(256, 256 + n_merges):
        pairs: Counter[tuple[int, int]] = Counter()
        for w, c in words.items():
            for p in zip(w, w[1:]):
                pairs[p] += c
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))   # 并列时取较大的对：结果确定
        if pairs[best] < 2:                              # 只出现一次的对不值得合并
            break
        merges[best] = new
        words = Counter({merge(w, best, new): c for w, c in words.items()})
    return merges


def build_vocab(merges: Merges) -> dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), t in merges.items():
        vocab[t] = vocab[a] + vocab[b]
    return vocab


def encode(text: str, merges: Merges) -> list[int]:
    out: list[int] = []
    for w in PAT.findall(text):
        ids = tuple(w.encode("utf-8"))
        while len(ids) >= 2:                     # 每次合并“最早学到”的那一对
            cand = [p for p in zip(ids, ids[1:]) if p in merges]
            if not cand:
                break
            best = min(cand, key=lambda p: merges[p])
            ids = merge(ids, best, merges[best])
        out.extend(ids)
    return out


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def utf8_len(lead: int) -> int:
    """由首字节判断字符应有几个字节；续字节(10xxxxxx)返回 0。"""
    if lead < 0x80:
        return 1
    if lead < 0xC0:
        return 0
    return 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4


class StreamDetokenizer:
    """逐 token 输入，只输出完整的 UTF-8 字符，不完整的尾部字节先暂存。"""

    def __init__(self, vocab: dict[int, bytes]) -> None:
        self.vocab, self.buf = vocab, b""

    def push(self, tid: int) -> str:
        self.buf += self.vocab[tid]
        cut = len(self.buf)
        for back in range(1, min(4, len(self.buf)) + 1):  # 从尾部找最后一个首字节
            need = utf8_len(self.buf[-back])
            if need:
                if need > back:          # 该字符还没收全：扣住这 back 个字节
                    cut -= back
                break
        out, self.buf = self.buf[:cut], self.buf[cut:]
        return out.decode("utf-8", errors="replace")

    def flush(self) -> str:
        out, self.buf = self.buf, b""
        return out.decode("utf-8", errors="replace")


def stream(ids: list[int], vocab: dict[int, bytes]) -> list[str]:
    det = StreamDetokenizer(vocab)
    return [det.push(i) for i in ids] + [det.flush()]


RESULTS: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" | {detail}" if detail else ""))


def run_tests(merges: Merges, vocab: dict[int, bytes], sample: str) -> None:
    rng = random.Random(0)
    pool = list("abcXYZ 019,.!é ж") + list(CORPUS[:120]) + ["鲲", "鹏", "🚀", "😀"]
    rand = ["".join(rng.choice(pool) for _ in range(rng.randint(1, 40)))
            for _ in range(300)]
    texts = [CORPUS, sample] + rand
    check("merges learned >= 200", len(merges) >= 200, f"{len(merges)} merges")
    check("round-trip decode(encode(s)) == s",
          all(decode(encode(t, merges), vocab) == t for t in texts),
          f"{len(texts)} texts")
    pieces = [stream(encode(t, merges), vocab) for t in texts]
    check("streaming output == full decode", all("".join(p) == t
                                                for p, t in zip(pieces, texts)))
    check("streaming emits no U+FFFD", all(REPL not in s for p in pieces for s in p))
    raw = list(sample.encode("utf-8"))           # 最坏情况：每个字节一个 token
    check("byte-by-byte streaming ok", "".join(stream(raw, vocab)) == sample)
    inc = codecs.getincrementaldecoder("utf-8")()  # 标准库的同类实现，作交叉验证
    ref = [inc.decode(vocab[i]) for i in encode(sample, merges)]
    check("matches codecs incremental decoder",
          "".join(ref) == "".join(stream(encode(sample, merges), vocab)))
    naive = [vocab[i].decode("utf-8", errors="replace") for i in encode(sample, merges)]
    check("naive per-token decode DOES produce U+FFFD", any(REPL in s for s in naive))
    check("unseen CJK char -> multiple tokens", len(encode("鲲", merges)) >= 2,
          f"encode('鲲')={encode('鲲', merges)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merges", type=int, default=300)
    ap.add_argument("--device", default="cpu", help="纯 Python 实验，此参数不起作用")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    for f in (sys.stdout, sys.stderr):  # Windows 控制台默认 GBK 时强制 UTF-8
        if hasattr(f, "reconfigure"):
            f.reconfigure(encoding="utf-8")
    merges = train_bpe(CORPUS, args.merges)
    vocab = build_vocab(merges)
    print(f"learned {len(merges)} merges, vocab size {len(vocab)}")
    longest = sorted(vocab.values(), key=len)[-5:]
    print("longest tokens:", [b.decode("utf-8", errors="replace") for b in longest])
    sample = "推理引擎用 KV cache 加速解码，鲲鹏🚀"
    ids = encode(sample, merges)
    nbytes = len(sample.encode("utf-8"))
    print(f"sample: {sample!r} -> {nbytes} bytes, {len(ids)} tokens")
    for ch in "推鲲🚀":
        t = encode(ch, merges)
        print(f"  {ch!r}: utf-8={ch.encode('utf-8').hex(' ')} -> tokens {t}")
    naive = [vocab[i].decode("utf-8", errors="replace") for i in ids]
    print("naive per-token :", "|".join(naive))
    print("streaming pieces:", "|".join(stream(ids, vocab)))
    if args.test:
        run_tests(merges, vocab, sample)
        print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed")
        sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
