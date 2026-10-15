# 模型推理 Infra 七天学习：学习资源、阅读顺序、模型与数据、H800 环境与能力阶梯（研究笔记）

> 访问日期：除特别注明外，全部链接均于 2026-10-15 访问。GitHub 元数据来自 `api.github.com/repos/...`，模型参数量来自 `huggingface.co/api/models/...` 的 `safetensors` 字段。bf16 体积按“参数量 × 2 字节”换算，属于推算，不是实测的磁盘占用。
> 标注规则：「未核实」表示没有拿到一手来源的事实，「证据不足」表示抓取失败或来源相互矛盾。
> 线索文件 `D:\Code\research_notes\AI Infra 论文分布与 VLA 推理选题\learning_path_deadlines_rankings.md`（Q1–Q3，2026-10-12）中的论文 arXiv ID 已由该文件作者用 Semantic Scholar 核对过。本笔记对其中的仓库和课程重新核实，论文 ID 沿用该文件，并加注“线索文件”。

---

## Q1 代码阅读仓库：存在性、许可证、活跃度与推荐顺序

### Takeaway
所有目标仓库都存在。nano-vllm（约 1.2k 行）和 mini-sglang（约 5k 行）是读懂 vLLM/SGLang 内部机制最短的路径，两者都是 MIT 许可，而且都能在 H800 上直接跑 Qwen3-0.6B。gpt-fast 适合看单请求 decode 的极致优化，即 torch.compile 与静态 KV cache。nanoGPT 已在 README 中引导用户改用 nanochat，llm.c 与 minbpe 只作为补充。tiny-llm 面向 Apple Silicon/MLX，在 H800 上不能直接用，只适合参考思路。

### Cited Findings
| 仓库 | 规模/定位 | 许可证 | 创建 → 最后 push | Stars（约） | 来源 |
|---|---|---|---|---|---|
| GeeeekExplorer/nano-vllm | “~1,200 lines of Python”。实现了 prefix caching、TP、torch.compile、CUDA graph。README 中，RTX 4070 Laptop + Qwen3-0.6B 的吞吐为 1434.13 tok/s，vLLM 为 1361.84 tok/s；安装命令 `pip install git+https://github.com/GeeeekExplorer/nano-vllm.git` | MIT | 2025-06-09 → 2026-04-26 | 15.8k | [API](https://api.github.com/repos/GeeeekExplorer/nano-vllm)，[README](https://github.com/GeeeekExplorer/nano-vllm) |
| sgl-project/mini-sglang | “approximately 5,000 lines of Python”。实现了 Radix Cache、Chunked Prefill、Overlap Scheduling、TP，并接入 FlashAttention/FlashInfer。仅支持 Linux。启动命令 `python -m minisgl --model "Qwen/Qwen3-0.6B"`，另有 `--shell` 交互模式。README 的基准测试在 H200 上完成 | MIT | 2025-09-01 → 2026-05-17 | 5.3k | [API](https://api.github.com/repos/sgl-project/mini-sglang)，[README](https://github.com/sgl-project/mini-sglang) |
| meta-pytorch/gpt-fast（原 pytorch-labs/gpt-fast） | “Simple and efficient pytorch-native transformer text generation in <1000 LOC” | BSD-3-Clause | 2023-10-17 → 2025-08-22 | 6.3k | [API](https://api.github.com/repos/pytorch-labs/gpt-fast) |
| karpathy/nanoGPT | README 顶部写有 “Update Nov 2025 … nanochat … It is very likely you meant to use/find nanochat instead.” | MIT | 2022-12-28 → 2025-11-12 | 63.8k | [API](https://api.github.com/repos/karpathy/nanoGPT)，[README](https://github.com/karpathy/nanoGPT) |
| karpathy/nanochat | “The best ChatGPT that $100 can buy.”，侧重训练全流程 | MIT | 2025-10-13 → 2026-09-07 | 58.6k | [API](https://api.github.com/repos/karpathy/nanochat) |
| karpathy/minbpe | BPE tokenizer 的最小实现 | MIT | 2024-02-16 → 2024-07-01 | 10.8k | [API](https://api.github.com/repos/karpathy/minbpe) |
| karpathy/llm.c | “LLM training in simple, raw C/CUDA” | MIT | 2024-04-08 → 2025-06-26 | 31.2k | [API](https://api.github.com/repos/karpathy/llm.c) |
| skyzh/tiny-llm | “learn LLM inference system on Apple Silicon for systems engineers: build a tiny vLLM + Qwen” | Apache-2.0 | 2025-04-19 → 2026-09-27 | 4.8k | [API](https://api.github.com/repos/skyzh/tiny-llm) |
| gpu-mode/lectures | GPU MODE 讲座材料，主页指向 YouTube @GPUMODE | Apache-2.0 | → 2026-09-09 | 6.9k | [API](https://api.github.com/repos/gpu-mode/lectures)，[YouTube](https://www.youtube.com/@GPUMODE) |
| zhaochenyang20/Awesome-ML-SYS-Tutorial（中文） | 作者自述为 “My learning notes for ML SYS”。线索文件称其中有大量 SGLang/RLHF 系统的中文源码解读 | Apache-2.0 | → 2026-10-13 | 7.5k | [API](https://api.github.com/repos/zhaochenyang20/Awesome-ML-SYS-Tutorial) |

官方设计文档：
- vLLM “Architecture Overview” 页包含 Entrypoints、V1 Process Architecture、LLM Engine、Worker、Model Runner、Model、Class Hierarchy 等小节 — [vLLM docs](https://docs.vllm.ai/en/latest/design/arch_overview.html)
- vLLM 指标设计页列出了 `/metrics` 暴露的 Prometheus 指标 — [vLLM metrics design](https://docs.vllm.ai/en/latest/design/metrics.html)
- HF Transformers “Optimizing inference” 页（`llm_optims`）覆盖 static KV cache + torch.compile（文中称“up to a 4x speed up”）、speculative decoding（注明 “doesn't support batched inputs”）、prompt lookup decoding、FlashAttention-2/SDPA、量化和 Continuous Batching。该页面目前只在 main 版文档中，v5.19.0 中没有 — [HF docs main](https://huggingface.co/docs/transformers/main/en/llm_optims)
- BentoML “LLM Inference Handbook”：原地址 `bentoml.com/llm/` 现在 301 跳转到 [handbook.modular.com](https://handbook.modular.com/)。章节包括 Foundations（含 Key metrics）、Planning your deployment、Model preparation、Model interaction、Inference optimization、Kernel optimization、Infra and operations；内容涉及 TTFT、TPS、goodput、continuous batching、prefix caching、PD disaggregation、KV cache offloading 和 KV Cache Memory Calculator。原因是 Modular 于 2026-02 收购了 BentoML — [Modular 博客](https://www.modular.com/blog/bentoml-joins-modular)，[BentoML 博客](https://www.bentoml.com/blog/bentoml-is-joining-modular)

### Inferences
- 推荐的初学者阅读顺序（结合用户已复现 OxyGen、理解 prefill/KV cache 这一背景）如下：
  1. HF `llm_optims` 页，加上 Handbook 的 Foundations/Key metrics，用约半天统一术语（TTFT/TPOT/ITL/goodput）。
  2. gpt-fast：理解单 batch decode 为什么受内存带宽限制，以及 static KV cache 加 CUDA graph/compile 的作用。
  3. **nano-vllm**（主线，1–1.5 天）：依次读 scheduler → block manager（prefix caching）→ model runner（CUDA graph）→ TP。
  4. **mini-sglang**（约 1 天）：读 radix cache、chunked prefill、overlap scheduling，并与 nano-vllm 对照。
  5. 回到 vLLM V1 的 arch_overview 和 SGLang 官方文档，把读到的概念映射到生产代码。
  6. 选读：GPU MODE 讲座、Awesome-ML-SYS-Tutorial 的中文 SGLang 解读。
  7. minbpe、llm.c、nanoGPT/nanochat 偏训练和 tokenizer，七天推理计划中只需各看 0.5 小时，用于补齐背景。
- 上述仓库中，与 OxyGen 的 shared-prefill KV cache 最直接相关的是 mini-sglang 的 Radix Cache 和 nano-vllm 的 prefix caching。

### Gaps
- nano-vllm/mini-sglang 当前 main 分支所需的 torch/flash-attn 精确版本没有逐一核对（README 摘要中未给出），需在安装当天查看 `pyproject.toml`。「未核实」
- tiny-llm 在 Linux/CUDA 上能否运行：其描述明确写的是 Apple Silicon，CUDA 支持「未核实」。
- SGLang 官方 “design docs” 没有找到单独的总览页，本次只核实了安装页。
- 线索文件中的 AISystem（chenzomi12）和 LeetCUDA（xlite-dev）本次没有重新核实，标「未核实」。

---

## Q2 课程与讲座

### Takeaway
CS336（Spring 2026）、CMU 11-868（已有 2026 Fall 页面）、MIT 6.5940（课程页显示 Fall 2026）、GPU MODE 都可以访问。七天计划不可能学完任何一门课，只适合挑选讲次：CS336 的 GPU/Kernels/Parallelism/Inference 四讲，GPU MODE 中 FlashAttention/Triton 相关的讲座。

### Cited Findings
- Stanford CS336 “Language Modeling from Scratch”：当前开课为 Spring 2026（原 `stanford-cs336.github.io` 301 到 `cs336.stanford.edu`）。相关讲次有 L5 “GPUs, TPUs”、L6 “Kernels, Triton”、L7/L8 “Parallelism”、L10 “Inference”；作业为 Basics、Systems、Scaling、Data、Alignment and Reasoning RL。全部录像在 [YouTube 播放列表](https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) — [cs336.stanford.edu](http://cs336.stanford.edu/)
- CMU 11-868 LLM Systems：已列出 2024 Spring、2025 Spring、2026 Spring、2026 Fall 四期。2026 Fall 由 Lei Li 主讲，大纲含 “architecting high-throughput serving engines”、CUDA/Triton kernel、量化等，并提到 vLLM、SGLang、FlashAttention。页面上没有看到逐讲日程，也没有说明录像是否公开 — [llmsystem.github.io](https://llmsystem.github.io/)，[2026 Fall](https://llmsystem.github.io/llmsystem2026fall/)
- MIT 6.5940 “TinyML and Efficient AI Computing”：`efficientml.ai` 301 到 [hanlab.mit.edu/course](https://hanlab.mit.edu/course)，页面显示 Fall 2026。HAN Lab 的 YouTube 频道是 https://www.youtube.com/c/MITHANLab。本次抓取没有得到具体讲次列表。
- GPU MODE：讲座材料仓库是 Apache-2.0，2026-09-09 仍在更新 — [GitHub](https://github.com/gpu-mode/lectures)
- 中文资源：Awesome-ML-SYS-Tutorial（Apache-2.0，2026-10-13 仍在更新）— [GitHub](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial)

### Inferences
- 七天内的课程用量建议：CS336 的 L5、L6、L10 必看，L7/L8 用于理解 TP 通信；GPU MODE 挑 1–2 讲（Flash Attention、Triton 入门）；CMU 11-868 和 MIT 6.5940 只翻看 slides，用来查漏补缺。
- 上述课程以英文为主，中文资源适合配合源码阅读。

### Gaps
- MIT 6.5940 Fall 2026 的讲次与录像链接「证据不足」（页面摘要不含讲次）。
- CMU 11-868 是否公开录像「证据不足」。
- GPU MODE 的具体讲次编号和标题本次没有逐条核实。
- 其他中文资源（知乎专栏、B 站课程等）没有找到能核实的一手来源，不推荐写入正文，或须标「未核实」。

---

## Q3 经典论文阅读清单（标出前 8 篇必读）

### Takeaway
初学者 7 天内应精读 8 篇，按顺序为：PagedAttention → Orca → FlashAttention → Sarathi-Serve → DistServe → SGLang → Speculative Decoding (Leviathan) → Mooncake。其余作为“扩展/按需”。

### Cited Findings
**★ 前 8 篇（推荐先读，按顺序）**
1. ★ Orca：iteration-level scheduling，即 continuous batching 的起源，OSDI 2022 — [USENIX](https://www.usenix.org/conference/osdi22/presentation/yu)（线索文件）
2. ★ vLLM/PagedAttention，SOSP 2023 — [arXiv 2309.06180](https://arxiv.org/abs/2309.06180)（线索文件）
3. ★ FlashAttention，NeurIPS 2022 — [arXiv 2205.14135](https://arxiv.org/abs/2205.14135)；FA2，ICLR 2024 — [2307.08691](https://arxiv.org/abs/2307.08691)；FA3（Hopper/H800 相关），NeurIPS 2024 — [2407.08608](https://arxiv.org/abs/2407.08608)（线索文件）
4. ★ Sarathi-Serve：chunked prefill，OSDI 2024 — [arXiv 2403.02310](https://arxiv.org/abs/2403.02310)（前身 SARATHI [2308.16369](https://arxiv.org/abs/2308.16369)）（线索文件）
5. ★ DistServe：PD 分离，以 goodput 为目标，OSDI 2024 — [arXiv 2401.09670](https://arxiv.org/abs/2401.09670)（线索文件）
6. ★ SGLang：RadixAttention，NeurIPS 2024 — [arXiv 2312.07104](https://arxiv.org/abs/2312.07104)（线索文件）
7. ★ Speculative decoding，Leviathan 等，ICML 2023 — [arXiv 2211.17192](https://arxiv.org/abs/2211.17192)（线索文件）
8. ★ Mooncake：KVCache-centric 分离式架构，FAST 2025 — [USENIX](https://www.usenix.org/conference/fast25/presentation/qin)，[arXiv 2407.00079](https://arxiv.org/abs/2407.00079)（线索文件）

**扩展（按主题）**
- Splitwise，ISCA 2024 — [arXiv 2311.18677](https://arxiv.org/abs/2311.18677)（线索文件）
- EAGLE，ICML 2024 — [2401.15077](https://arxiv.org/abs/2401.15077)；EAGLE-2，EMNLP 2024 — [2406.16858](https://arxiv.org/abs/2406.16858)；EAGLE-3，NeurIPS 2025 — [2503.01840](https://arxiv.org/abs/2503.01840)；Medusa — [2401.10774](https://arxiv.org/abs/2401.10774)；Chen 等 speculative sampling — [2302.01318](https://arxiv.org/abs/2302.01318)（线索文件）
- FlashInfer：MLSys 2025 Outstanding Paper。用 block-sparse 格式统一 KV cache 布局，提供 JIT 模板和兼容 CUDAGraph 的负载均衡调度，已集成进 SGLang、vLLM、MLC-Engine — [arXiv 2501.01005](https://arxiv.org/abs/2501.01005)，[MLSys 2025 Awards](https://mlsys.org/virtual/2025/awards_detail)
- Llumnix：多实例间的请求实时迁移与动态调度，OSDI 2024，pp.173–191。论文称尾延迟改善一个数量级，在相近尾延迟下节省最多 36% 成本 — [arXiv 2406.03243](https://arxiv.org/abs/2406.03243)，[USENIX](https://www.usenix.org/conference/osdi24/presentation/sun-biao)，代码 [AlibabaPAI/llumnix](https://github.com/AlibabaPAI/llumnix)
- 2025–2026 重要工作（来源是检索结果中的引用条目，非逐篇一手核实）：NanoFlow（OSDI 2025，[arXiv 2408.12757](https://arxiv.org/abs/2408.12757)）；MegaScale-Infer（disaggregated expert parallelism，[arXiv 2504.02263](https://arxiv.org/abs/2504.02263)，venue「未核实」）；IC-Cache（SOSP 2025）；Pensieve（EuroSys 2025）；“KVCache cache in the wild”（ATC 2025）；Mooncake 期刊版（ACM TOS 2025）— 以上 venue 信息仅来自 WebSearch 摘要中其他论文参考文献的条目，未打开原文或会议页，一律标「未核实（二手）」；正文引用前需逐篇核对
- 量化：GPTQ [2210.17323](https://arxiv.org/abs/2210.17323)，AWQ（MLSys 2024）[2306.00978](https://arxiv.org/abs/2306.00978)，SmoothQuant（ICML 2023）[2211.10438](https://arxiv.org/abs/2211.10438)（线索文件）
- 综述：“Towards Efficient Generative LLM Serving” ACM CSUR — [2312.15234](https://arxiv.org/abs/2312.15234)；“A Survey on Efficient Inference for LLMs” — [2404.14294](https://arxiv.org/abs/2404.14294)（线索文件）
- 基准方法学：MLPerf Inference（ISCA 2020）定义了 single-stream/server 场景与时延约束 — [1911.02549](https://arxiv.org/abs/1911.02549)（线索文件）

### Inferences
- 阅读顺序把 Orca 放在 PagedAttention 之前：先理解“为什么要按 iteration 调度”，再理解“为什么 KV 要分页”。
- 每篇论文都对应一个实验：Orca/PagedAttention 对应 nano-vllm 的 scheduler/block manager；Sarathi 对应 vLLM `--enable-chunked-prefill` 的开关对比（参数名「未核实」，以当版 `vllm serve --help` 为准）；SGLang 对应 radix cache 命中率；DistServe/Mooncake 对应 4 卡上 1P+1D 或 2P+2D 的实验；Leviathan/EAGLE 对应 speculative decoding 前后的 ITL 对比。
- FA3 专门针对 Hopper（H100/H800 都是 sm90），因此在 H800 上有可读性和可实验性。

### Gaps
- 多数 2025–2026 新论文的 venue 只来自引用条目，未逐篇访问官方页面。
- 本次没有检索 2026 年的 OSDI/SOSP/MLSys 推理系统新论文清单，「证据不足」。

---

## Q4 H800 服务器环境事实（截至 2026-10-15）

### Takeaway
目前 vLLM 最新是 0.31.0，固定 torch==2.13.0，默认 wheel 编译于 CUDA 12.9，另提供 cu128、cu130、cu131 变体。SGLang 最新是 0.5.21，**只支持 CUDA 13**，cu129 镜像的最后一版是 v0.5.19-cu129。CUDA 13.x 要求驱动 ≥580（R580 分支）。因此，若要在同一台 H800 上同时跑两者，**驱动应 ≥580**。最稳妥的做法是用官方 Docker 镜像并钉死 tag，各引擎使用独立的 venv/容器。

### Cited Findings
**vLLM**
- PyPI 最新为 0.31.0，`requires_python <3.15,>=3.10`，依赖 `torch==2.13.0`、`torchvision==0.28.0`、`torchaudio==2.11.0` — [PyPI JSON](https://pypi.org/pypi/vllm/json)
- 安装文档：Linux，Python 3.11–3.14（与 PyPI 的 ≥3.10 不一致，以文档为准）；compute capability ≥7.5；“vLLM's binaries are compiled with CUDA 12.9 and public PyTorch release versions by default”，另有 CUDA 12.8/13.0/13.1 变体 — [vLLM GPU install](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)
- 安装命令（文档原文）：
  - `uv pip install vllm --torch-backend=auto`，之后运行 `vllm download-kernels`
  - `pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129`
  - 指定 CUDA 13：`uv pip install vllm --extra-index-url https://download.pytorch.org/whl/cu130`
  — 同上
- Docker：`vllm/vllm-openai:latest`。官方 run 示例为 `docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HF_TOKEN=$HF_TOKEN" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model Qwen/Qwen3-0.6B` — [vLLM Docker docs](https://docs.vllm.ai/en/latest/deployment/docker.html)
- Docker Hub 上 v0.31.0 的 tag 于 2026-10-04 发布：`v0.31.0`、`v0.31.0-cu129`、`v0.31.0-ubuntu2404`、`v0.31.0-cu129-ubuntu2404`、`v0.31.0-x86_64` 等；上一版 v0.30.0 发布于 2026-09-22；另有 `nightly`、`cu129-nightly`、`cu134-nightly` — [Docker Hub API](https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?page_size=20&name=v0.3)
- 基准命令（文档原文）：先下载 ShareGPT，`wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json`；再运行 `vllm bench serve --backend vllm --model <model> --endpoint /v1/completions --dataset-name sharegpt --dataset-path <path>/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 10`。可调参数有 `--request-rate`、`--max-concurrency`、`--burstiness`。支持的数据集包括 sharegpt、random、burstgpt、hf、random-mm、prefix_repetition、spec_bench、speed_bench、custom 等。报告的指标有 TTFT、TPOT、ITL 和吞吐 — [vLLM benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/)

**SGLang**
- PyPI 最新为 0.5.21，依赖 `torch==2.13.0`、`flashinfer_python[cu13]==0.6.18`、`sglang-kernel==0.4.7`、`flash-attn-4>=4.0.0b18`、`cuda-python>=13.0` — [PyPI JSON](https://pypi.org/pypi/sglang/json)
- 文档（域名已从 docs.sglang.ai 301 到 docs.sglang.io）：安装命令为 `uv pip install --prerelease=allow sglang`，要求 Python ≥3.11；“CUDA 13” 为必需；“CUDA 12 (cu129) wheels and images were retired”，最后一个 CUDA 12 版本是 `lmsysorg/sglang:v0.5.19-cu129`；Docker 部署需要 CUDA 13-compatible driver — [SGLang install](https://docs.sglang.io/get_started/install.html)
- Docker：`lmsysorg/sglang:latest`，另有 `latest-runtime`。官方 run 示例为 `docker run -it --gpus all --shm-size 32g -p 30000:30000 -v "$HF_CACHE_DIR:/root/.cache/huggingface" --ipc=host lmsysorg/sglang:latest bash`。v0.5.21 系列 tag 于 2026-10-01 发布：`v0.5.21`、`v0.5.21-cu130`、`v0.5.21-runtime`、`v0.5.21-cu130-runtime` 等 — 同上，[Docker Hub API](https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags?page_size=15&name=v0.5.2)
- **冲突**：SGLang 文档称 cu129 是“随 PyTorch 2.14 升级”退役的，但 PyPI 上的 0.5.21 仍然固定 `torch==2.13.0`。可能文档跟随的是 main 分支，比发行版超前。「证据不足」，以 `pip show` 的实际结果为准。

**PyTorch 与 CUDA 驱动**
- PyPI torch 最新为 2.14.1，Linux 默认 wheel 依赖 `cuda-toolkit==13.0.3`、`nvidia-nccl-cu13==2.30.7`、`nvidia-cudnn-cu13==9.24.0.43` — [PyPI torch JSON](https://pypi.org/pypi/torch/json)。注意：vLLM/SGLang 固定的是 2.13.0，不是最新的 2.14.1。
- pytorch.org/get-started 抓取失败（ECONNRESET），torch 2.13.0 有哪些 CUDA 变体「证据不足」。
- NVIDIA CUDA Release Notes（最新为 CUDA 13.4 Update 2）：
  - minor version compatibility：13.x 要求驱动 ≥580，12.x 要求 ≥525
  - 13.0 对应 R580，13.1 对应 R590，13.2 对应 R595，13.4 对应 R615
  - 12.9 GA 要求 ≥575.51.03，12.9 U1 要求 ≥575.57.08，12.8 GA 要求 ≥570.26
  - 从 CUDA 13.4 起，Linux 安装包不再捆绑驱动
  — [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

**Hugging Face 下载**
- CLI 已改名为 `hf`，不再是 `huggingface-cli`（文档示例版本为 huggingface_hub v2.2.0）。安装方式：`curl -LsSf https://hf.co/cli/install.sh | bash`、`pip install -U "huggingface_hub"` 或 `uvx hf`。登录：`hf auth login`，或 `hf auth login --token $HF_TOKEN`。
- 下载示例：`hf download <repo>`、`hf download gpt2 config.json`、`--include "*.safetensors" --exclude ...`、`--repo-type dataset`、`--revision v1.1`、`--local-dir <dir>`、`--cache-dir <dir>`。
- 默认缓存在 `~/.cache/huggingface/hub`，由 `HF_HOME` 控制。token 存放在 `~/.cache/huggingface/token`。相关环境变量有 `HF_TOKEN`、`HF_HUB_DOWNLOAD_TIMEOUT`（默认 10s）。
- 来源：[HF CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

**NCCL / 拓扑**
- nccl-tests 编译：`make`，或 `make MPI=1 MPI_HOME=... CUDA_HOME=... NCCL_HOME=...`。单机示例：`./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8`。参数含义：`-b` 是起始大小，`-e` 是结束大小，`-f` 是倍增因子，`-g` 是每线程 GPU 数 — [nccl-tests README](https://github.com/NVIDIA/nccl-tests)
- algbw = S/t；allreduce 的 busbw = algbw × 2(n−1)/n，“reflects how optimally the hardware is used” — [PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
- H800 SXM 的 NVLink 带宽从 H100 的 900 GB/s 降到 400 GB/s，算力和 80 GB 显存不变，目的是满足 2022-10 出口管制中的互联带宽限制。单卡 LLM 推理性能与 H100 SXM 接近 — [Epoch AI](https://epoch.ai/gradient-updates/us-export-controls-china-ai)（二手分析；NVIDIA 没有公开 H800 数据表，「未核实（一手）」）。显存带宽各 SKU 说法不一，有 3.0 TB/s 和 2.0 TB/s（HBM2e/PCIe）两种 — [getdeploying](https://getdeploying.com/gpus/nvidia-h100-vs-nvidia-h800)，「证据不足」

### Inferences
- **H800 环境检查清单**（以下命令基于上述来源改写，未在用户机器上运行）：
  1. `nvidia-smi`：确认驱动 ≥580（SGLang 0.5.21 的硬性要求；vLLM 默认 cu129 wheel 用 ≥575.51.03 即可）。
  2. `nvidia-smi topo -m`：查看 4 张卡之间是 NV#（NVLink）还是 PIX/SYS。这条命令是 nvidia-smi 的标准子命令，但本次没有抓取官方文档页，「未核实来源」。
  3. `nvcc --version`：只有编译 nccl-tests 或扩展时才需要。用 Docker 时可以忽略宿主机 toolkit。
  4. nccl-tests：4 卡改用 `-g 4`，并把 `-e` 扩大到 1G–8G 以观察饱和 busbw，例如 `./build/all_reduce_perf -b 8 -e 4G -f 2 -g 4`。在 400 GB/s NVLink 下，busbw 预期明显低于 H100 的公开数据，具体数值必须实测。
  5. 每个引擎各建一个 uv venv（vLLM 与 SGLang 的 flashinfer/kernel 依赖可能冲突），或直接用钉死 tag 的镜像：`vllm/vllm-openai:v0.31.0` 或 `-cu129`，`lmsysorg/sglang:v0.5.21`。
  6. `export HF_HOME=/data/hf`（放在大容量 NVMe 上），然后把该目录挂载进容器的 `/root/.cache/huggingface`。
- vLLM 同时发布了 `v0.31.0` 和 `v0.31.0-cu129` 两个 tag，暗示默认 tag 可能已经是 CUDA 13 构建，而 wheel 文档仍写 12.9 是默认值。这一点「未核实」，拉镜像后用 `python -c "import torch;print(torch.version.cuda)"` 确认。
- CUDA 13 是否支持 Hopper（sm90）：PyPI torch 2.14.1 默认使用 cu13，SGLang 文档也把 H100 列在支持范围内，可以推定支持；但本次没有抓取 CUDA 13 的架构支持表，「未核实」。

### Gaps
- vLLM 0.31.0 与 SGLang 0.5.21 的 GitHub release 页没有直接打开，发布日期按 Docker Hub tag 时间推定。
- Docker 镜像体积「未核实」。
- 若机器在国内，需要 HF 镜像或 ModelScope 下载方式；本次没有核实相关一手文档，且任务约束不允许改动网络设置。

---

## Q5 4×H800 上可用的模型与数据集（不训练）

### Takeaway
从 0.6B 到 32B dense，再到 30B-A3B MoE、MLA 架构的 DeepSeek-V2-Lite，以及 VL、ASR、Omni 多模态模型，都能在 4×80 GB 上用 bf16 推理。除 Llama-3.1 需要申请 gated 权限外，其余模型无门槛，大多为 Apache-2.0。全部下载约需 320 GB（推算），建议预留 ≥0.6–1 TB 本地 NVMe。

### Cited Findings
| 模型 | 参数量（HF API） | bf16 体积（推算） | 许可证 / gated | 实验用途 | 来源 |
|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 751,632,384 | ≈1.5 GB | Apache-2.0 / 否 | nano-vllm/mini-sglang 跑通、单卡 profiling | [API](https://huggingface.co/api/models/Qwen/Qwen3-0.6B) |
| Qwen/Qwen3-8B | 8.19B | ≈16.4 GB | Apache-2.0 / 否 | 单卡 vLLM vs SGLang 基准、prefix cache | [API](https://huggingface.co/api/models/Qwen/Qwen3-8B) |
| Qwen/Qwen3-32B | 32.76B | ≈65.5 GB | Apache-2.0 / 否 | TP=2/4 扩展性、通信开销 | [API](https://huggingface.co/api/models/Qwen/Qwen3-32B) |
| Qwen/Qwen3-30B-A3B | 30.5B 总 / 3.3B 激活；48 层；Q/KV 头 32/4；128 专家激活 8；上下文 32,768（YaRN 可到 131,072）；需要 transformers≥4.51.0、vllm≥0.8.5、sglang≥0.4.6.post1 | ≈61 GB | Apache-2.0 / 否 | MoE、EP vs TP | [API](https://huggingface.co/api/models/Qwen/Qwen3-30B-A3B)，[README](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| meta-llama/Llama-3.1-8B-Instruct | 8,030,261,248 | ≈16.1 GB（API 显示仓库存储 32.1 GB，「未核实」原因） | llama3.1 / **gated（需人工审核）** | vLLM 文档示例对齐、EAGLE 草稿模型生态 | [API](https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct) |
| deepseek-ai/DeepSeek-V2-Lite(-Chat) | 15.7B（卡片写 16B 总 / 2.4B 激活）；MLA；上下文 32k | ≈31 GB | 代码 MIT；模型适用 DeepSeek Model Agreement，“supports commercial use” / 否 | MLA 的 KV 压缩、MoE | [API](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V2-Lite-Chat)，[README](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) |
| Qwen/Qwen2.5-VL-7B-Instruct | 8,292,166,656 | ≈16.6 GB | Apache-2.0 / 否 | 视觉编码器 + LLM，接近 VLA 的 prefill 形态 | [API](https://huggingface.co/api/models/Qwen/Qwen2.5-VL-7B-Instruct) |
| Qwen/Qwen3-VL-8B-Instruct | 8,767,123,696 | ≈17.5 GB | Apache-2.0 / 否 | 同上（新一代） | [API](https://huggingface.co/api/models/Qwen/Qwen3-VL-8B-Instruct) |
| openai/whisper-large-v3 | 1,543,490,560（F16） | ≈3.1 GB | Apache-2.0 / 否 | encoder-decoder、音频前处理瓶颈 | [API](https://huggingface.co/api/models/openai/whisper-large-v3) |
| Qwen/Qwen2.5-Omni-7B | 10,732,225,440（0.449B F32 + 10.28B BF16） | ≈22.4 GB | Apache-2.0 / 否 | any-to-any 多阶段流水 | [API](https://huggingface.co/api/models/Qwen/Qwen2.5-Omni-7B) |
| Qwen/Qwen3-Omni-30B-A3B-Instruct | 35,259,818,545 | ≈70.5 GB | Apache-2.0 / 否 | 大型多模态 MoE（可选） | [API](https://huggingface.co/api/models/Qwen/Qwen3-Omni-30B-A3B-Instruct) |

| 数据集 | 说明 | 许可证 | 来源 |
|---|---|---|---|
| anon8231489123/ShareGPT_Vicuna_unfiltered → `ShareGPT_V3_unfiltered_cleaned_split.json` | vLLM 官方基准文档使用的下载 URL（见 Q4）；最后修改 2023-04-12 | 卡片写 apache-2.0（内容来源于 ShareGPT 用户对话，合规性「未核实」） | [HF API](https://huggingface.co/api/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) |
| vLLM 内置 random / prefix_repetition / spec_bench / random-mm 等 | 不需要下载，可控输入输出长度，适合做消融 | — | [vLLM benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/) |

### Inferences
- 显存粗算：Qwen3-32B 的权重约 65.5 GB，单张 80 GB 卡只剩约 10 GB 给 KV cache，所以应以 TP=2/4 为主要实验。Qwen3-30B-A3B 的权重约 61 GB，同理建议 TP/EP=2 或 4。Qwen3-Omni-30B-A3B 约 70.5 GB，必须多卡。
- 每 token 的 KV 大小 = 2 × 层数 × KV 头数 × head_dim × 2 字节。以 Qwen3-30B-A3B 为例，48 层、4 个 KV 头已核实，head_dim 若为 128（「未核实」，需查 config.json），则 KV ≈ 96 KiB/token。写作时应让读者用 config.json 自行计算，或使用 Handbook 的 KV Cache Memory Calculator。
- 7 天内推荐的最小模型集：Qwen3-0.6B（读源码与调试）、Qwen3-8B（单卡基准）、Qwen3-32B（TP）、Qwen3-30B-A3B（MoE）、Qwen2.5-VL-7B 或 Qwen3-VL-8B（多模态 prefill，衔接 OxyGen/VLA 经验），合计约 160 GB。Llama-3.1-8B 只在需要 gated 生态时下载，Omni 与 Whisper 为可选。
- 磁盘：上表全部合计约 320 GB（推算），加上 Docker 镜像和 venv，建议预留 ≥0.6 TB。

### Gaps
- Qwen3 系列（8B/32B）各自的层数、KV 头数和 head_dim 没有逐一核对 config.json。
- Llama-3.1-8B 仓库 32.1 GB 的构成（可能含 `original/` 目录）「未核实」。
- DeepSeek-V2-Lite 在 vLLM 0.31/SGLang 0.5.21 上的支持情况没有专门核实。
- Qwen3-Omni 在 vLLM 主线中的支持状态「未核实」。

---

## Q6 能力阶梯：七天能达到什么，生产级推理 infra 还需要什么

### Takeaway
七天高强度学习可以达到“概念掌握 + 可运行的对照实验”：能读懂 nano-vllm/mini-sglang，能在 4×H800 上用 vLLM/SGLang 复现吞吐-时延曲线，能解释 chunked prefill、prefix cache、TP、spec decoding 对 TTFT/ITL 的影响。生产级能力的核心是另一组东西：SLO 与容量规划、指标与告警、以队列为信号的自动扩缩容、多租户流控、K8s 上的 KV 感知路由与 PD 分离（llm-d/Dynamo/KServe）、冷启动、故障恢复和回归基准。这些需要真实流量与集群，七天内只能通过阅读加单机模拟来“接触”。

### Cited Findings
- **指标**：vLLM 在 `/metrics` 暴露以下 Prometheus 指标，生产可观测性就建立在这些指标上：
  - Gauge：`vllm:num_requests_running`、`vllm:num_requests_waiting`、`vllm:kv_cache_usage_perc`
  - Counter：`vllm:prompt_tokens_total`、`vllm:generation_tokens_total`、`vllm:prefix_cache_queries`、`vllm:prefix_cache_hits`
  - Histogram：`vllm:time_to_first_token_seconds`、`vllm:inter_token_latency_seconds`、`vllm:e2e_request_latency_seconds`、`vllm:request_prefill_time_seconds`、`vllm:request_decode_time_seconds`
  — [vLLM metrics](https://docs.vllm.ai/en/latest/design/metrics.html)
- **SLO 与 goodput**：Handbook 把 goodput 视为比原始吞吐更有意义的指标，用来衡量满足 SLO 的产出 — [handbook.modular.com](https://handbook.modular.com/)；DistServe 以 goodput 为优化目标 — [arXiv 2401.09670](https://arxiv.org/abs/2401.09670)
- **自动扩缩容**：GKE 最佳实践推荐两种扩缩信号：
  - queue size：“Queue size directly correlates to request latency”，建议阈值从 3–5 起调。
  - batch size：在更严格的时延目标下使用，原因是 “Larger batch sizes increase throughput but also raise latency due to the prefill phase of some requests interrupting the decode phase”。
  - GPU 内存或 CPU 指标单独使用不够。
  — [GKE autoscaling best practices](https://docs.cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling)
- **llm-d**（Apache-2.0，2026-10-15 仍在 push，最新 v0.7 于 2026-05 发布）的 “well-lit paths”：
  - Intelligent Routing：基于 prefix-cache 与负载感知，按预测时延调度
  - Advanced KV-Cache Management：分层 offload 到 CPU/磁盘，配合全局索引
  - Serving Large Models：PD 分离与 wide-EP
  - Operational Excellence：多租户 flow control 与 SLO-aware autoscaling
  - Batch Processing
  组件包括 vLLM/SGLang、inference scheduler、Gateway API Inference Extension — [llm-d README](https://github.com/llm-d/llm-d)，[API](https://api.github.com/repos/llm-d/llm-d)
- **NVIDIA Dynamo** 的主要能力：
  - PD 分离
  - KV-aware router（README 称 TTFT 快 2 倍）
  - SLA-driven Planner（README 举例 “80% fewer SLA breaches” at “5% lower TCO”）
  - KVBM，可把 KV 从 GPU 逐级 offload 到 CPU、SSD、远端存储
  - NIXL/ModelExpress，README 称冷启动快 7 倍
  - K8s Operator
  - 后端支持 vLLM、SGLang、TensorRT-LLM
  README 写的是 Apache 2.0，GitHub API 的 SPDX 字段则显示 NOASSERTION — [Dynamo README](https://github.com/ai-dynamo/dynamo)，[API](https://api.github.com/repos/ai-dynamo/dynamo)。以上性能数字均为厂商自述。
- **KServe**：“Standardized Distributed Generative and Predictive AI Inference Platform … on Kubernetes”，Apache-2.0，2026-10-15 仍在 push — [API](https://api.github.com/repos/kserve/kserve)
- **vLLM production-stack**：“vLLM's reference system for K8S-native cluster-wide deployment”，Apache-2.0 — [API](https://api.github.com/repos/vllm-project/production-stack)
- **多实例调度与迁移**：Llumnix 通过请求实时迁移改善尾延迟并降低成本，是多实例调度的学术代表 — [arXiv 2406.03243](https://arxiv.org/abs/2406.03243)

### Inferences
**能力阶梯（供写作者改写为表格）**

| 等级 | 标志性能力 | 可验证产出 | 7 天可达？ |
|---|---|---|---|
| L0 概念掌握 | 能讲清 prefill/decode 的计算访存特征、continuous batching、PagedAttention、prefix/radix cache、chunked prefill、PD 分离、spec decoding、TP/EP；会算 KV 显存 | 一页纸“自绘架构图 + KV 显存计算” | 是（第 1–2 天） |
| L1 源码级理解 | 能在 nano-vllm/mini-sglang 中定位 scheduler、block 分配、CUDA graph 捕获、TP 通信，并做小改动（如打印每步 batch 组成、修改 block_size） | 带注释的调用链笔记，一个能跑的小 patch | 是（第 2–4 天） |
| L2 可运行实验 | 用 `vllm bench serve` 和 SGLang 的对应工具做受控对照：request-rate 扫描、TP=1/2/4、开关 prefix cache/chunked prefill/spec decode；跑 nccl-tests 和 `topo -m`，并解释 busbw；用 `/metrics` 观察 KV 使用率和排队 | 吞吐-时延（P50/P99 TTFT、ITL）曲线，加上复现脚本和环境锁定 | 是（第 3–7 天），这是七天的核心交付 |
| L3 工程级调优 | 用 Nsight/torch profiler 定位瓶颈；按 SLO 选并行策略与 batch 上限；在 4 卡上做 1P+1D 或 2P+2D 的 PD 分离实验；做量化（FP8）的精度与速度权衡 | 带 profiler 证据的调优报告 | 部分可达（只能做 1–2 个点） |
| L4 生产级 | SLO 与容量规划（按 goodput 推导卡数）；Prometheus/Grafana 告警；以队列或 batch 为信号的 HPA；多租户流控与优先级；K8s 上部署 llm-d/Dynamo/KServe，配合 KV 感知路由；冷启动与权重分发；故障转移与请求迁移；每个版本的回归基准；成本（$/Mtoken）核算 | 有真实流量的线上服务、runbook、回归基准 CI | 否。需要集群、真实流量和数周到数月的工程实践；7 天内只能读文档，加上单机 docker-compose 级别的模拟 |

- 判断“生产级”的关键，不在于会调多少参数，而在于能否给出**可复现、可回归、受 SLO 约束**的结论。七天的指南应把“每个实验都有固定版本、固定数据集、固定负载模型和原始数据”作为验收标准。
- 4×H800 的 NVLink 为 400 GB/s，这让 TP=4 的通信占比高于 H100。对比 TP=2×2 副本与 TP=4×1 副本，是一个有 H800 特色的好实验。数值需实测，这里不预设结论。

### Gaps
- 没有找到一手资料把“七天学习能达到什么”量化为能力等级。上表是基于官方文档能力项的推断，不是引用。
- 本次没有找到可核实的公开回归基准方法论来源，例如 vLLM 的 performance dashboard 或 CI 基准页。「证据不足」
- KServe 的 LLMInferenceService 等具体 LLM 功能没有打开文档核实。
- llm-d/Dynamo 能否在单机 4 卡上做最小演示（如 kind/minikube），「未核实」。

---

## 附：线索文件（learning_path_deadlines_rankings.md）复核结论
- 复核一致的条目：nano-vllm、mini-sglang（sgl-project 组织下）、CS336（当前 Spring 2026）、CMU 11-868（线索文件只列到 2025 Spring，实际已有 2026 Spring 和 2026 Fall）、GPU MODE 讲座仓库、Awesome-ML-SYS-Tutorial — 来源见 Q1/Q2。
- 有更新的条目：MIT 6.5940 在线索文件中写“最近可确认为 Fall 2024”，本次课程页显示已到 Fall 2026，讲次内容「证据不足」 — [hanlab.mit.edu/course](https://hanlab.mit.edu/course)。BentoML Handbook 的 URL 已迁移到 handbook.modular.com。
