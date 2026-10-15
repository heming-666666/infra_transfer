"""Lab10：玩具多模态数据通路（随机权重）。

图像：缩放/归一化 → 切 p×p patch → ViT 式 patch 嵌入 → 相邻 2×2 token 合并 → MLP 投影
      → 按占位符位置写入文本嵌入序列（视觉 token 数 = (H/p/2)×(W/p/2)）。
音频：torch.stft + 手写 mel 滤波器组（HTK mel 公式）→ log-Mel
      → 再用 stride-s 卷积下采样成音频 token。
所有尺寸（图像大小、p、维度、n_mels、归一化均值/方差等）都是“示意性”默认值，
可用命令行修改，不代表任何具体真实模型的配置。
运行：python lab10_multimodal_toy.py --device cpu --test
"""
from __future__ import annotations

import argparse
import math
import sys

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def preprocess(img: Tensor, h: int, w: int, mean: float, std: float) -> Tensor:
    """img: [3, H0, W0]，取值 [0,1] → 缩放到 [3, h, w] 并归一化。"""
    x = F.interpolate(img[None], size=(h, w), mode="bicubic", align_corners=False)[0]
    return (x.clamp(0, 1) - mean) / std


def patchify(x: Tensor, p: int) -> Tensor:
    """[C, H, W] → [(H/p)·(W/p), C·p·p]，按行优先排列 patch。"""
    C, H, W = x.shape
    x = x.reshape(C, H // p, p, W // p, p).permute(1, 3, 0, 2, 4)
    return x.reshape((H // p) * (W // p), C * p * p)


def merge2x2(t: Tensor, gh: int, gw: int) -> Tensor:
    """[gh·gw, d] → [(gh/2)·(gw/2), 4d]：空间上相邻的 2×2 个 token 拼接成 1 个。"""
    d = t.shape[-1]
    t = t.reshape(gh // 2, 2, gw // 2, 2, d).permute(0, 2, 1, 3, 4)
    return t.reshape((gh // 2) * (gw // 2), 4 * d)


def mel_filterbank(n_mels: int, n_fft: int, sr: int) -> Tensor:
    """三角滤波器组 [n_mels, n_fft//2+1]（HTK 公式；其他实现如 Slaney 式细节不同）。"""
    def hz2mel(f: Tensor) -> Tensor:
        return 2595 * torch.log10(1 + f / 700)
    mels = torch.linspace(0, float(hz2mel(torch.tensor(sr / 2))), n_mels + 2)
    hz = 700 * (10 ** (mels / 2595) - 1)
    f = torch.linspace(0, sr / 2, n_fft // 2 + 1)
    lo, c, hi = hz[:-2, None], hz[1:-1, None], hz[2:, None]
    return torch.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)).clamp(min=0)


def log_mel(wav: Tensor, n_fft: int, hop: int, fb: Tensor,
            center: bool = True) -> Tensor:
    win = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(wav, n_fft, hop, window=win, center=center, return_complex=True)
    return torch.log10((fb.to(wav.device) @ spec.abs() ** 2).clamp(min=1e-10))


RESULTS: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" | {detail}" if detail else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--image-hw", type=int, nargs=2, default=[224, 320])
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--vit-dim", type=int, default=192)
    ap.add_argument("--llm-dim", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=1000)
    ap.add_argument("--norm", type=float, nargs=2, default=[0.5, 0.5], help="mean std")
    ap.add_argument("--n-mels", type=int, default=80)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--audio-stride", type=int, default=2)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    for f in (sys.stdout, sys.stderr):  # Windows 控制台默认 GBK 时强制 UTF-8
        if hasattr(f, "reconfigure"):
            f.reconfigure(encoding="utf-8")
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，回退到 CPU")
        a.device = "cpu"
    dev = torch.device(a.device)
    # cuDNN 卷积默认允许 TF32（fp32 matmul 默认不允许），两者在 GPU 上会差 ~1e-4；
    # 为了严格比较“卷积 == 切 patch 后的线性层”，这里关闭 TF32
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    (H, W), p = a.image_hw, a.patch
    assert H % (2 * p) == 0 and W % (2 * p) == 0, "H、W 需是 2p 的倍数（2×2 合并）"
    gh, gw = H // p, W // p
    # ---- 图像通路 ----
    img = torch.rand(3, 300, 451, device=dev)                  # 任意原始尺寸的合成图
    x = preprocess(img, H, W, *a.norm)
    patches = patchify(x, p)                                   # [N, 3p²]
    conv = nn.Conv2d(3, a.vit_dim, p, stride=p).to(dev)        # ViT 的 patch 嵌入
    with torch.no_grad():
        emb_conv = conv(x[None]).flatten(2)[0].T               # [N, vit_dim]
        emb_lin = patches @ conv.weight.reshape(a.vit_dim, -1).T + conv.bias
        merged = merge2x2(emb_lin, gh, gw)                     # [N/4, 4·vit_dim]
        proj = nn.Sequential(nn.Linear(4 * a.vit_dim, a.llm_dim), nn.GELU(),
                             nn.Linear(a.llm_dim, a.llm_dim)).to(dev)
        vis = proj(merged)                                     # [M, llm_dim]
        M, IMG = vis.shape[0], a.vocab - 1             # 最后一个 id 当图像占位符
        ids = torch.cat([torch.randint(0, IMG, (5,)), torch.full((M,), IMG),
                         torch.randint(0, IMG, (7,))]).to(dev)
        tok_emb = nn.Embedding(a.vocab, a.llm_dim).to(dev)
        mask = ids == IMG
        seq = tok_emb(ids).masked_scatter(mask[:, None], vis)  # 按顺序填入占位符位置
    print(f"image {tuple(img.shape)} -> {tuple(x.shape)} -> "
          f"patches {tuple(patches.shape)} -> merged {tuple(merged.shape)} -> "
          f"visual tokens {tuple(vis.shape)}")
    print(f"text+image sequence: {len(ids)} tokens ({M} image placeholders)")
    # ---- 音频通路 ----
    sr, n_fft, hop = 16000, 400, 160
    n = int(a.seconds * sr)
    t = torch.arange(n, device=dev) / sr
    wav = 0.5 * torch.sin(2 * math.pi * 440 * t) + 0.01 * torch.randn(n, device=dev)
    fb = mel_filterbank(a.n_mels, n_fft, sr)
    mel = log_mel(wav, n_fft, hop, fb)
    mel_nc = log_mel(wav, n_fft, hop, fb, center=False)
    down = nn.Conv1d(a.n_mels, a.llm_dim, 3, stride=a.audio_stride, padding=1).to(dev)
    with torch.no_grad():
        audio_tok = down(mel[None])[0].T                       # [帧数/stride, llm_dim]
    empty = int((fb.sum(1) == 0).sum())
    print(f"audio {a.seconds}s @16kHz = {n} samples -> log-mel {tuple(mel.shape)} "
          f"(center=False: {mel_nc.shape[1]} frames) -> "
          f"audio tokens {tuple(audio_tok.shape)}"
          f"; empty mel filters={empty}")
    if not a.test:
        return
    N = gh * gw
    check("patch count = (H/p)(W/p)", patches.shape == (N, 3 * p * p), f"{N} patches")
    d = (emb_conv - emb_lin).abs().max().item()
    check("conv patch-embed == linear on patchified pixels", d < 1e-4,
          f"max|diff|={d:.2e}")
    check("2x2 merge -> N/4 tokens", merged.shape == (N // 4, 4 * a.vit_dim), f"M={M}")
    g = emb_lin.reshape(gh, gw, -1)                            # 网格 [gh, gw, d]
    nb = [torch.cat([g[2 * i + u, 2 * j + v] for u in (0, 1) for v in (0, 1)])
          for i in range(gh // 2) for j in range(gw // 2)]
    check("merged token k = concat of its 2x2 spatial neighbours",
          torch.equal(merged, torch.stack(nb)))
    check("placeholders == visual tokens, seq len = text + M",
          int(mask.sum()) == M and len(ids) == 12 + M)
    check("visual embeddings scattered in order", torch.equal(seq[mask], vis))
    check("text embeddings untouched", torch.equal(seq[~mask], tok_emb(ids[~mask])))
    want_c, want_nc = 1 + n // hop, 1 + (n - n_fft) // hop
    check("STFT frames (center=True) = 1 + N//hop", mel.shape == (a.n_mels, want_c),
          f"{want_c}")
    check("STFT frames (center=False) = 1 + (N-n_fft)//hop", mel_nc.shape[1] == want_nc,
          f"{want_nc}")
    want_tok = (want_c + 2 - 3) // a.audio_stride + 1
    check("audio tokens = floor((F+2-3)/s)+1", audio_tok.shape[0] == want_tok,
          f"{want_tok}")
    win = torch.hann_window(n_fft, device=dev)
    power = torch.stft(wav, n_fft, hop, window=win, return_complex=True).abs().pow(2)
    peak = int(power.mean(1).argmax())
    check("440 Hz sine peaks at bin round(440/(sr/n_fft)) = 11", peak == 11,
          f"bin={peak}")
    print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
