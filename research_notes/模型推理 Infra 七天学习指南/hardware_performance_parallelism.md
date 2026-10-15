# H800（Hopper）上的 LLM 推理：硬件、Roofline、显存、Kernel、量化、并行与 Profiling

> 研究时间：2026-10-14（所有链接访问日期均为 2026-10-14）。版本基线：vLLM v0.31.0（GitHub latest release，2026-10-05 发布）、SGLang v0.5.21（2026-10-02 发布）、Nsight Systems 2026.5.1（Release Notes 页面最新条目）。
> 来源分级：T1 = 厂商官方/论文/官方仓库源码；T2 = 云厂商/OEM 文档；T3 = 经销商/集成商页面。「未核实」= 本轮未能取得一手证据；「证据不足」= 抓取失败。
> 「推断」中的数字是笔者用已引用参数推算出来的，不是测量结果。

## 1. H800 硬件参数（SXM5 / PCIe），以及与 H100 SXM 的对比

### Takeaway
NVIDIA 已下线 H800 官方产品页：nvidia.com/en-us/data-center/h800/ 和 nvidia.cn/data-center/h800/ 在 2026-10-14 访问均返回 404。下面的 H800 数字来自 T1 级的 DeepSeek 论文、T2 级的 Lenovo/腾讯云文档，以及 T3 级的经销商页面。可靠结论是：H800 SXM 的 Tensor Core 算力和 HBM 与 H100 SXM 相同，被削减的是 NVLink（900→400 GB/s）和 FP64。

### Cited Findings
- **官方页面状态**：H800 的美国站和中国站产品页在 2026-10-14 均返回 HTTP 404。网络搜索也表明 NVIDIA 没有公开的 H800 datasheet，参数只能从 OEM 或经销商处获得。来源：[nvidia.com/en-us/data-center/h800](https://www.nvidia.com/en-us/data-center/h800/)（404）、[nvidia.cn/data-center/h800](https://www.nvidia.cn/data-center/h800/)（404）。web.archive.org 存档无法抓取（工具限制）。
- **H100 SXM 官方参数（T1）**：FP64 34 TF；FP64 Tensor Core 67 TF；FP32 67 TF；TF32 TC 989 TF*；BF16 TC 1,979 TF*；FP16 TC 1,979 TF*；FP8 TC 3,958 TF*；INT8 TC 3,958 TOPS*。显存 80GB，带宽 3.35 TB/s，TDP 最高 700W（可配置），NVLink 900 GB/s，PCIe Gen5 128 GB/s。页面脚注“* With sparsity”，即带 * 的数字均含 2:4 结构化稀疏，dense 值为其一半。同一页面上 H100 NVL 为 94GB、3.9 TB/s、NVLink 600 GB/s。来源：[NVIDIA H100 产品页](https://www.nvidia.com/en-us/data-center/h100/)
- **Hopper 架构（T1，NVIDIA 技术博客，2022-03-22）**：完整 GH100 有 144 个 SM；H100 SXM5 为 132 个 SM，H100 PCIe 为 114 个 SM。每个 SM 有 128 个 FP32 CUDA core、4 个第四代 Tensor Core、256 KB 的 shared memory 与 L1 合并存储。全卡 L2 为 50 MB。第四代 NVLink 每卡 18 条 link，合计 900 GB/s。PCIe Gen5 合计 128 GB/s，即每方向 64 GB/s。新特性包括 TMA、Thread Block Cluster / Distributed Shared Memory、异步事务 barrier 和 FP8 Transformer Engine。博客称 H100 FP8 Tensor Core 吞吐是 A100 FP16 的 6 倍。来源：[NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- **H800 SXM 的 SM 数（T1）**：DeepSeek-V3 报告写到“20 out of the 132 SMs available in the H800 GPU”被用于通信。来源：[DeepSeek-V3 Technical Report, arXiv 2412.19437](https://arxiv.org/html/2412.19437)
- **H800 SXM 的 NVLink（T1）**：DeepSeek 的 ISCA'25 论文原文是“the NVLink bandwidth in H800 SXM nodes is reduced from 900 GB/s to 400 GB/s”，以及“NVLink provides 200GB/s bandwidth (of which about 160GB/s can actually be achieved)”。论文还说 H800 为合规降低了 FP64 算力和 NVLink 带宽；每个节点配 8 张 400G IB CX7 网卡，每张约 50 GB/s，论文计算时按 40 GB/s 有效带宽取值。来源：[Insights into DeepSeek-V3, arXiv 2505.09343 (ISCA'25)](https://arxiv.org/html/2505.09343)。DeepSeek-V3 报告的说法是“NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s)”。来源：[arXiv 2412.19437](https://arxiv.org/html/2412.19437)
- **H800 SXM 的算力（T2，腾讯云）**：HCCPNV5 实例配置为“8 × NVIDIA H800 NVLink 80GB (FP32 64 TFLOPS, TF32 494 TFLOPS, BF16 989 TFLOPS, 400GB/s NVLink)”，并配 3.2 Tbps RDMA。这组数字是 dense 值：989 正好是 H100 SXM 稀疏 1,979 的一半。该页面没有给出 FP8 和显存带宽。页面更新时间为 2026-02-04。来源：[腾讯云实例规格](https://www.tencentcloud.com/document/product/1236/62812)
- **H800 SXM 的显存带宽和 TDP（T3）**：经销商和集成商页面列出 80GB HBM3、3.35 TB/s、700W、FP64 1 TF，以及带稀疏的 BF16 1,979 TF 和 INT8 3,958 TOPS。但同一页面的宣传文字又写“3 TB/s”，存在自相矛盾。来源：[强川科技 H800 SXM 页面](https://m.qiangchuan.com/item/7228.html)、[LTHPC G8228G3](https://lthpc.com/cms/shenduxuexi/79.html)。旁证：FlashMLA 在 H800 SXM5 上实测达到“3000 GB/s in memory-bound configuration”（见第 4 节），说明有效带宽至少 ≥3 TB/s。
- **H800 PCIe 80GB（T2，Lenovo，产品已撤市）**：FP64 0.8 TF；FP32 51 TF；TF32 756 TF*；FP16/BF16 1,513 TF*；FP8 3,026 TF*；INT8 3,026 TOPS*，其中 * 表示“With structural sparsity enabled”。显存 80GB HBM2e，带宽 2 TB/s；NVLink 400 GB/s（通过 NVLink bridge 连接）；PCIe Gen5 x16（128 GB/s）；功耗 350W；MIG 最多 7 个 10GB 实例。另有 H800 NVL 94GB 型号：HBM3、3.9 TB/s、400W。来源：[Lenovo Press LP1814](https://lenovopress.lenovo.com/lp1814-thinksystem-nvidia-h800-pcie-gen5-gpu)。注意：Lenovo 页面写的“18,432 CUDA cores / 528 Tensor cores”是完整 GH100 芯片的数字，很可能有误；其他来源给出的是 114 SM / 14,592 core。

**汇总表**（dense = 不含稀疏；括号内为来源级别）

| 指标 | H100 SXM（T1） | H800 SXM5 | H800 PCIe 80GB（T2 Lenovo） |
|---|---|---|---|
| BF16/FP16 TC dense | 989 TF（1,979 稀疏÷2） | 989 TF（T2 腾讯云） | ≈756 TF（1,513 稀疏÷2） |
| FP8 TC dense | 1,979 TF（3,958 稀疏÷2） | ≈1,979 TF（推断：TC 与 H100 相同；T3 给出 INT8 3,958 稀疏） | ≈1,513 TF |
| TF32 TC dense | 494.5 TF | 494 TF（T2） | ≈378 TF |
| FP64 / FP64 TC | 34 / 67 TF | ≈1 TF（T3；T1 只说“reduced”） | 0.8 / 0.8 TF |
| 显存 / 带宽 | 80GB HBM3 / 3.35 TB/s | 80GB HBM3 / 3.35 TB/s（T3） | 80GB HBM2e / 2 TB/s |
| NVLink | 900 GB/s，18 link | 400 GB/s（T1 DeepSeek）；单向 200，实际可达约 160 | 400 GB/s（bridge） |
| PCIe | Gen5 128 GB/s | Gen5（推断） | Gen5 x16 128 GB/s |
| SM / L2 | 132 / 50 MB | 132（T1 DeepSeek）/ 50 MB（推断，未核实） | 未核实（他源称 114 SM） |
| TDP | ≤700W | 700W（T3） | 350W |

### Inferences
- 推理算 roofline 时，H800 SXM 可以直接沿用 H100 SXM 的 dense 算力和 3.35 TB/s 带宽。对推理有实质影响的差别只有 NVLink：多卡 TP 的 all-reduce 带宽约为 H100 的 4/9。FP64 被削减与 LLM 推理无关。
- 用户手上的“4×H800”是 SXM5（HGX 板，经 NVSwitch 全互联）还是 PCIe 卡（只有两两之间用 bridge 相连），必须先用 `nvidia-smi topo -m` 确认。若是 PCIe 卡，TP=4 时跨对的流量会走 PCIe（每方向 64 GB/s），并行策略的结论会完全改变。

### Gaps
- 没有找到 NVIDIA 现行的 H800 SXM5 官方 datasheet。H800 SXM 的 FP8 峰值、3.35 TB/s 带宽和 L2 容量只有 T3 来源或推断，标为「未核实」。
- H800 SXM 在 `nvidia-smi topo -m` 中显示的 NVLink 条数（社区常见说法是“NV8”）未找到一手证据，标为「未核实」。

## 2. Roofline：ridge point、prefill 与 decode 的算术强度，以及 8B 模型的 decode 上限

### Takeaway
H800 SXM 的 BF16 ridge point 约为 295 FLOP/B，FP8 约为 590 FLOP/B。decode 阶段 GEMM 的算术强度约等于 batch 中的 token 数，所以 batch 小于约 300 时受带宽限制。decode attention 的强度约等于 GQA 组大小（Hq/Hkv），与 batch 无关，因此一直是带宽受限。

### Cited Findings
- 计算所用参数：BF16 dense 989 TF（[腾讯云](https://www.tencentcloud.com/document/product/1236/62812)、[H100 页](https://www.nvidia.com/en-us/data-center/h100/)）；带宽 3.35 TB/s（[H100 页](https://www.nvidia.com/en-us/data-center/h100/)；H800 的该值为 T3）。
- Qwen3-8B 配置：36 层，hidden 4096，intermediate 12288，32 个 Q 头，8 个 KV 头，head_dim 128，vocab 151,936，`tie_word_embeddings=false`；共 8,190,735,360 个 BF16 参数。来源：[Qwen3-8B config.json](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json)、[HF API](https://huggingface.co/api/models/Qwen/Qwen3-8B)
- Pope 等人的论文用解析模型选择推理切分方式，并指出 MQA 降低 KV 显存后可以支持长 32 倍的上下文。来源：[Efficiently Scaling Transformer Inference, arXiv 2211.05102](https://arxiv.org/abs/2211.05102)
- DeepSeek 的 KV/token 对比数据：DeepSeek-V3（MLA）为 70.272 KB，Qwen-2.5 72B（GQA）为 327.680 KB，LLaMA-3.1 405B 为 516.096 KB。来源：[arXiv 2505.09343](https://arxiv.org/html/2505.09343)
- FlashMLA 的 dense MLA decode 在 H800 SXM5 上，memory-bound 配置下达到 3000 GB/s，compute-bound 配置下达到 660 TFLOPS（CUDA 12.8）。来源：[FlashMLA @ba89a34 README](https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7)

### Inferences（公式与推算）
- **Ridge point**：I* = 峰值 FLOP/s ÷ 带宽。
  - BF16：989e12 / 3.35e12 ≈ **295 FLOP/B**。
  - FP8：1979e12 / 3.35e12 ≈ **591 FLOP/B**。
  - H800 PCIe 的 BF16：756e12 / 2.0e12 ≈ 378 FLOP/B。
- **线性层**（输入 [B,K] × 权重 [K,N]，BF16）：FLOPs = 2BKN；访存 ≈ 2KN + 2B(K+N) 字节。当 B ≪ K, N 时，强度 ≈ **B FLOP/B**。
  - prefill 时 B = 批内全部 prompt token 数，通常有数千，远超 295，属于 compute-bound。
  - decode 时 B = 并发序列数，需要 B 约 300 才能碰到 ridge。
  - 换成 FP8 权重和 FP8 计算：强度 ≈ 2B，ridge 为 591，所以交点 B 仍约 295；但带宽受限区间内的速度翻倍。
  - W4A16（Marlin）：强度 ≈ 4B，交点 B ≈ 74。这与 Marlin 论文“batch 16–32 内接近 4× 加速、64–128 时加速逐渐下降”的结论在量级上一致。
- **decode attention**（每条序列读自己的 KV，上下文长度 L）：FLOPs ≈ 4·L·Hq·d，KV 访存 ≈ 2·2·L·Hkv·d 字节（BF16），强度 ≈ **Hq/Hkv**。Qwen3-8B 为 4，Llama 类 GQA 一般为 4–8。增大 batch 只是让需要读的 KV 总量线性增加，强度不变，所以永远是带宽受限。
  - 例外是 MLA：DeepSeek 的 128 个 Q 头共享一个 576 维 latent KV，单个 query token 的强度 ≈ 128×(2×576+2×512)/(576×2) ≈ 242 FLOP/B，接近 ridge。这就是 FlashMLA 能出现 compute-bound 配置的原因。
- **算例：Qwen3-8B，BF16，单卡 H800，只计带宽下限**
  - 每步需读的权重：16.38 GB 减去 embedding 表 1.245 GB（decode 只读当步 token 对应的行），约 15.1 GB。
  - 每 token 的 KV = 2 × 8 × 128 × 2 B × 36 层 = **147,456 B（144 KiB）**。
  - **batch=1**（上下文约 1k，KV 约 0.15 GB 可忽略）：15.1e9 / 3.35e12 ≈ 4.5 ms/步，上限约 **220 tok/s**。实际带宽利用率 80–90% 时约为 180–200 tok/s。
  - **batch=64，平均上下文 2048**：KV 读量 = 64 × 2048 × 147,456 ≈ 19.3 GB，每步共约 34.4 GB，约 10.3 ms，聚合上限约 **6,200 tok/s**（每序列约 97 tok/s）。若平均上下文为 512：约 20.0 GB/步，约 6.0 ms，约 10,700 tok/s。
  - 同一步的 GEMM 计算量为 2 × 8.19e9 × 64 ≈ 1.05 TFLOP，在 989 TF 下约 1.06 ms，远小于访存时间，所以依然是带宽受限。
  - 结论：加大 batch 能摊薄权重的读取，但摊不薄 KV 的读取。长上下文、大 batch 的 decode 是 KV 带宽在主导。

### Gaps
- H800 的实测可达 HBM 带宽（如 STREAM 类 copy 测试）没有公开的一手数据，需要用实验 2 自己测。

## 3. 显存预算与 vLLM 如何确定 KV 容量

### Takeaway
显存由五部分组成：权重、KV cache、峰值激活与 workspace、CUDA graph 池、非 torch 内存（NCCL、cuBLAS、CUDA context）。vLLM 的计算公式是：KV 可用 = total × `gpu_memory_utilization`（v0.31.0 默认值 **0.92**）− profile run 测得的 non-KV 内存 − CUDA graph 估计值。从 v0.21.0 起，CUDA graph 估计默认计入。

### Cited Findings
- v0.31.0 的 `CacheConfig.gpu_memory_utilization: float = Field(default=0.92, gt=0, le=1)`。它是“per-instance limit”：同卡运行两个实例时，可以各自设为 0.5。来源：[vllm/config/cache.py @v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/cache.py)。注：早期版本的默认值是 0.9，本轮未核实具体是哪个版本改的。
- `request_memory()`：`requested_memory = ceil(total_memory * gpu_memory_utilization)`。如果启动时空闲显存小于这个值，直接报错。来源：[vllm/v1/worker/utils.py @v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/worker/utils.py)
- `determine_available_memory()` 的流程分三步：
  1. 用 dummy 输入执行 `profile_run`，在 `memory_profiling(weights_memory=...)` 中测量。
  2. 若 cudagraph_mode ≠ NONE，调用 `profile_cudagraph_memory()`。
  3. 计算 `available_kv_cache_memory_bytes = requested_memory - non_kv_cache_memory - cudagraph_memory_estimate_applied`。
  
  日志会写明 CUDA graph 显存估计“default since v0.21.0”，可以用 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` 关闭。如果设置了 `kv_cache_memory_bytes`，则跳过 profiling，同时不再遵守 `gpu_memory_utilization`。来源：[vllm/v1/worker/gpu_worker.py @v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/worker/gpu_worker.py)
- profiling 期间，代码用 `max_split_size_mb=20` 限制 allocator 切分，防止 MoE workspace 留下的碎片被算进已消耗内存（源码注释）。同一文件还有一条断言：如果同一 GPU 上其他进程在 profiling 期间释放了显存，会报 “Error in memory profiling”。来源：同上。
- 启动日志会给出容量结论，例如“GPU KV cache size: 643,232 tokens”和“Maximum concurrency for 40,960 tokens per request: 15.70x”。来源：[parallelism_scaling.md @v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/serving/parallelism_scaling.md)
- 文档给出的省显存手段：TP、量化、调小 `max_model_len` / `max_num_seqs`、用 `cudagraph_capture_sizes` 减少捕获的尺寸、`enforce_eager=True`（完全不用 CUDA graph）。来源：[conserving_memory.md @v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/configuration/conserving_memory.md)
- PagedAttention 以分页方式管理 KV，做到“near-zero waste”，相比 FasterTransformer/Orca 吞吐提升 2–4×。来源：[arXiv 2309.06180](https://arxiv.org/abs/2309.06180)

### Inferences（算例：Qwen3-8B，单卡 H800 80GB）
- 假设 torch 报告的总显存约 79.6 GiB（需实测），乘以 0.92 得到 requested ≈ 73.2 GiB。
- 依次扣除：权重 15.26 GiB、峰值激活（视 `max_num_batched_tokens` 而定，假设约 2 GiB）、非 torch 内存约 1 GiB、CUDA graph 约 1 GiB。剩余 KV ≈ 54 GiB。
- 54 GiB ÷ 144 KiB/token ≈ **39.3 万 token**。按每请求 32k token 算，最大并发约 12；按 8k 算约 48。
- 通用公式：
  - 最大并发 ≈ KV_bytes ÷ (每 token 的 KV 字节 × 每请求 token 数)
  - 每 token 的 KV 字节 = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes
  - 用 TP=t 时，每卡只存 n_kv_heads/t 个头（若 kv_heads < t，则会复制）。
- 换成 FP8 KV cache，每 token KV 减半，容量约翻倍。
- 可以和 DeepSeek 的数据交叉验证：Qwen2.5-72B 按上式为 2 × 80 × 8 × 128 × 2 = 327,680 B/token，与 DeepSeek 论文给出的 327.680 KB 完全一致。

### Gaps
- NCCL buffer 的具体占用（与 `NCCL_BUFFSIZE` 和通道数有关）没有核实一手文档。
- “0.9→0.92”默认值的变更版本未核实。

## 4. 关键 Kernel：FlashAttention 1–4、Flash-Decoding、FlashInfer、PagedAttention、DeepSeek 三件套、CUDA Graph / torch.compile

### Takeaway
在 H800 上，FA3 仍是 Hopper 专用的主力 attention kernel。FA4 的论文以 Blackwell 为主，但 README 称其同时为 Hopper 和 Blackwell 优化。需要特别注意：**FlashMLA 在 2026-09-30 的 main 分支已移除 Hopper 支持**，H800 用户须固定到 commit `ba89a34`。DeepGEMM 仍支持 SM90。DeepEP 已升级到 V2.5（NCCL Gin 后端，要求 CUDA 13.1+ 和 PyTorch 2.10+）。

### Cited Findings
- **FlashAttention-2**：在 A100 上达到理论 FLOPs 的 50–73%。FA1 只有 25–40%。来源：[arXiv 2307.08691](https://arxiv.org/abs/2307.08691)
- **FlashAttention-3**：FA2 在 H100 上利用率只有 35%。FA3 用了三项技术：
  1. 借助 Tensor Core 与 TMA 的异步性，用 warp specialization 让计算与数据搬运重叠；
  2. 在块内交错执行 matmul 与 softmax；
  3. FP8 块量化加 incoherent processing。
  
  效果：FP16 提速 1.5–2.0×，达到 740 TFLOPs/s（75% 利用率）；FP8 接近 1.2 PFLOPs/s，数值误差比基线 FP8 低 2.6×。来源：[arXiv 2407.08608](https://arxiv.org/abs/2407.08608)。README 中 FA3 仍标为 beta，要求“H100 / H800 GPU, CUDA >= 12.3”，推荐 CUDA 12.8；已发布 FP16/BF16 的前向和反向、FP8 前向；安装方式为 `cd hopper && python setup.py install`。来源：[flash-attention README](https://github.com/Dao-AILab/flash-attention)
- **FlashAttention-4**：论文 2026-03-05 发布，用 CuTe-DSL 实现。针对 Blackwell 的不对称扩展做了三件事：完全异步的 MMA、软件模拟 exp、tensor memory 加 2-CTA MMA。在 B200 上 BF16 达 1613 TFLOPs/s（71%），比 cuDNN 9.13 快 1.3×，比 Triton 快 2.7×；编译速度比 C++ 模板快 20–30×。来源：[arXiv 2603.05451](https://arxiv.org/abs/2603.05451)。README 写“optimized for Hopper and Blackwell GPUs (e.g. H100, B200)”，安装 `pip install flash-attn-4`，调用 `from flash_attn.cute import flash_attn_func`。来源：[flash-attention README](https://github.com/Dao-AILab/flash-attention)
- **Flash-Decoding**（Dao、Haziza、Massa、Sizov，2023-10）：FlashAttention 只在 batch 和 query 块两个维度上并行，decode 时 batch=1 只能用到不足 1% 的 GPU。Flash-Decoding 先把 KV 切块并行计算，再用 log-sum-exp 归约。超长序列生成最多快 8×，attention 本身最多快 50×（CodeLlama-34B，A100，序列长 512–64k）。已集成于 FlashAttention ≥2.2 和 xFormers ≥0.0.22。来源：[CRFM blog](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- **FlashInfer**：用 block-sparse 格式统一 KV 存储，提供 JIT 定制模板和兼容 CUDAGraph 的负载均衡调度，已被 SGLang、vLLM、MLC 采用。相比编译器后端，ITL 降低 29–69%；长上下文延迟降低 28–30%；并行生成提速 13–17%。来源：[arXiv 2501.01005](https://arxiv.org/abs/2501.01005)
- **FlashMLA**：
  - 2026-09-30 的 README 写明：“removed support for the Hopper architecture and for earlier models (including DeepSeek V3 / V3.2 / V4.0)”，并改变了 FP8/FP4 KV 格式。需要 Hopper 支持的用户请切换到 commit `ba89a3466e9470ad08ab39738d4e7bb66989e1e7`。当前要求 SM100/SM103 和 CUDA 13.1+。来源：[FlashMLA README (main)](https://github.com/deepseek-ai/FlashMLA)
  - 在 `ba89a34` 版本中，H800 SXM5（CUDA 12.8）的成绩：dense MLA decode 达 3000 GB/s（memory-bound）和 660 TFLOPS（compute-bound）；稀疏 decode（FP8 KV、BF16 矩阵乘）410 TFLOPS；稀疏 prefill 640 TFLOPS。2025-04-22 的更新称 compute-bound 场景提升 5–15%，达到 660 TFLOPS。来源：[FlashMLA @ba89a34](https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7)
- **DeepGEMM**：
  - 2025-04-18 起在 H800 上 FP8 “up to 1550 TFLOPS”，接近 1,979 dense 峰值的 78%（推算）。
  - 同时支持 SM90 和 SM100；要求 CUDA 12.9+、PyTorch 2.3+、CUTLASS 4.0+；kernel 运行时 JIT 编译。
  - SM90 只支持 NT layout，scale 用 FP32。
  - grouped GEMM 有两种：contiguous（用于 prefill）和 masked（用于 CUDA graph 下的 decode，可接 DeepEP 低延迟输出）。
  - 2026 年新增 Mega MoE 等功能。
  
  来源：[DeepGEMM README](https://github.com/deepseek-ai/DeepGEMM)。按 WebFetch 摘要，FP8 的 scale 粒度为“激活 1×128、权重 128×128”，与 DeepSeek-V3 报告一致；原始 README 中未逐字复核。
- **DeepEP v1.2.1 的 H800 测试**（NVLink 约 160 GB/s，CX7 IB 约 50 GB/s）：
  - normal kernel（4096 token，hidden 7168，top-8，FP8 dispatch / BF16 combine）：
    - 节点内 EP8：dispatch 153 GB/s，combine 158 GB/s（NVLink）；
    - 跨节点 EP16/32/64：43/58/51 GB/s（RDMA）。
  - low-latency kernel（128 token）：EP8 dispatch 77 µs、combine 114 µs；EP256 分别为 194 µs、360 µs。
  
  来源：[DeepEP v1.2.1 README](https://github.com/deepseek-ai/DeepEP/tree/v1.2.1)
- **DeepEP 当前版本**（V2/V2.5）：引入 NCCL Gin 后端，完全移除 V1 和 NVSHMEM 依赖。要求 Hopper 或更新 GPU、CUDA 13.1+、PyTorch 2.10+、NCCL 2.32.3+。说明中写“EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported”，与 V1 宣传的“hook 式重叠不占 SM”不同。来源：[DeepEP README (main)](https://github.com/deepseek-ai/DeepEP)
- **vLLM 的 CUDA graph 与 torch.compile**（v0.31.0）：
  - 默认 `FULL_AND_PIECEWISE`：uniform decode 用 full graph，其他情况用 piecewise graph。这是通常最快的模式，但最占显存、捕获最慢。
  - `FULL_DECODE_ONLY` 适合 P/D 分离部署中的 decode 实例。
  - piecewise 模式会让 attention 等不兼容 graph 的算子保持 eager 执行。
  - 编译模式为 `CompilationMode.VLLM_COMPILE`。
  
  来源：[cuda_graphs.md @v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/design/cuda_graphs.md)、[conserving_memory.md](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/configuration/conserving_memory.md)
- vLLM 的 nsys 样例中，H800/H100 上占比最高的 kernel 是 `sm90_xmma_gemm_bf16...`（cuBLAS 的 Hopper GEMM，约 83%）、`FlashAttnFwdSm90`（4.1%）和 `fused_add_rms_norm`。来源：[profiling.md @v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/contributing/profiling.md)

### Inferences
- 对 8B 这类小模型做 decode 时，单步 GPU 时间只有几毫秒，每层约 10 个 kernel，几十层下来有数百次 launch。不用 CUDA graph 时，CPU 发射开销（每次约数微秒）和 Python 调度开销会以 GPU 空泡的形式出现在 timeline 上。这就是 vLLM 默认开启 full decode graph 的原因。可以用 `enforce_eager=True` 做对照实验来观察。
- 如果要在 H800 上复现 DeepSeek 的 MLA kernel，必须使用 FlashMLA 的 `ba89a34` 及更早版本。不要直接克隆 main 分支。

### Gaps
- FlashInfer、Triton、CUTLASS 的当前版本号本轮未查询，标为「未核实」。CUTLASS 4.x 仅从 DeepGEMM 的依赖要求得知。
- CUDA graph 对 launch 开销的量化数据缺少本轮可引用的一手材料。

## 5. Hopper 上的量化

### Takeaway
Hopper 的 Tensor Core 原生支持 FP8（E4M3/E5M2）和 INT8。FP4（NVFP4/MXFP4）以及 MXFP8 的块缩放属于 Blackwell 的硬件特性。在 H800 上，性价比最高的路线是 FP8 W8A8 加 FP8 KV cache；若 batch 小、追求单流速度，可以用 INT4 weight-only（AWQ/GPTQ + Marlin）。

### Cited Findings
- FP8 有两种编码：E4M3（4 位指数 3 位尾数，没有 inf，只有一个 NaN 位型，以此扩展范围）和 E5M2（遵循 IEEE 特殊值约定）。175B 模型训练可以达到 16-bit 的效果；对 int8 难以量化的模型，FP8 PTQ 也可行。来源：[FP8 Formats for Deep Learning, arXiv 2209.05433](https://arxiv.org/abs/2209.05433)
- Transformer Engine 文档给出：E4M3 最大 ±448，E5M2 最大 ±57344。前向用 E4M3，反向用 E5M2，两者组合即 HYBRID。delayed scaling 的公式是 scale = (FP8_MAX/amax)/2^margin，需要 SM89 以上。MXFP8（E8M0 幂次 scale）属于 Blackwell；API 中另有 Float8BlockScaling、NVFP4BlockScaling 等。来源：[TE 2.19 FP8 Delayed Scaling](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.19/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html)、[TE FP8 primer (2.6)](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.6/user-guide/examples/fp8_primer.html)。注意：这些内容来自搜索结果摘要，页面本身未单独抓取；旧的 fp8_primer URL 已经 404。
- vLLM v0.31.0 的硬件兼容表：
  - llm-compressor FP8 W8A8：支持 Ada、Hopper，**不支持** Ampere。
  - Marlin（GPTQ/AWQ/FP8/FP4）：支持 Turing* 到 Hopper。
  - AWQ、GPTQ、INT8 W8A8：均支持 Hopper。
  
  来源：[quantization/README.md @v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/features/quantization/README.md)
- vLLM 的 FP8 KV cache 支持两种粒度：per-tensor 和 per-attention-head（后者仅限 Flash Attention 后端，需要 llm-compressor 校准）。不校准时 scale 默认为 1.0，官方推荐用数据集校准。可选 `kv_cache_dtype` 为 `fp8`、`fp8_e4m3`、`fp8_e5m2`。使用 FA3 后端加 FP8 KV 时，Q 也会被量化到 FP8。可以用 `--kv-cache-dtype-skip-layers` 跳过对量化敏感的层（如 sliding-window 层）。来源：[quantized_kvcache.md @v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/features/quantization/quantized_kvcache.md)
- DeepSeek-V3 的细粒度量化方案：激活按 1×128（per-token-per-128-channel），权重按 128×128 块。H800 Tensor Core 做 FP8 累加时精度受限：DeepSeek-V3 报告说“only uses the highest 14 bits of each mantissa product”，ISCA 论文则写“highest 13 fraction bits… accumulated to FP22 registers”，两处表述不一致，予以保留。DeepSeek 的应对是每隔一段把部分和提升到 CUDA core 上用 FP32 累加。来源：[arXiv 2412.19437](https://arxiv.org/html/2412.19437)、[arXiv 2505.09343](https://arxiv.org/html/2505.09343)
- **SmoothQuant**：通过数学等价变换，把激活中的离群值“迁移”到权重上，从而实现 INT8 W8A8；最多 1.56× 加速，显存减半。来源：[arXiv 2211.10438](https://arxiv.org/abs/2211.10438)
- **AWQ**：只需保护约 1% 的显著通道。按激活统计确定缩放，不依赖反向传播。来源：[arXiv 2306.00978](https://arxiv.org/abs/2306.00978)
- **GPTQ**：基于二阶信息的一次性量化，175B 模型约 4 GPU 小时即可量化到 3–4 bit。来源：[arXiv 2210.17323](https://arxiv.org/abs/2210.17323)
- **Marlin**：4-bit 权重在 batch 16–32 时接近 4× 加速，64–128 时加速递减；集成到 vLLM 后端到端最多快 2.8×。来源：[arXiv 2408.11743](https://arxiv.org/abs/2408.11743)

### Inferences
- 按第 2 节的 roofline 分析：
  - decode 小 batch 时，W4A16 的读量是 BF16 的 1/4，速度最快，但到 batch 约 74 就进入 compute 区。而且 Hopper 上 INT4 没有对应的 Tensor Core 路径，需要先反量化再走 BF16 MMA。
  - FP8 W8A8 在 decode 时读量减半，在 prefill 时算力翻倍，是兼顾两头的方案。
- 精度风险：未校准的 FP8 KV（scale=1）、per-tensor 激活 scale 遇到离群值、FP8 累加精度问题，都可能损伤长链推理的精度。上线前需要做任务级评测，不能只看 perplexity。这是基于以上来源的推断。

### Gaps
- 本轮没有抓取 NVIDIA 官方 NVFP4 博客，所以“FP4 Tensor Core 仅 Blackwell 支持”的依据目前只有 TE 文档摘要和 H100 规格表里没有 FP4 一行，属于间接证据。
- 本轮没有收集 FP8 和 INT4 在具体模型上的精度对比数据。

## 6. 推理并行：DP / TP / PP / EP / CP，NCCL，以及 4×H800 推荐

### Takeaway
TP 每层在前向中需要 2 次 all-reduce，每次每 token 的数据量为 2h 字节（BF16）。H800 的 NVLink 只有 H100 的 4/9，所以 TP 在 prefill 时通信占比明显更高，decode 时则主要受 all-reduce 延迟限制。在 4×H800 上，凡是单卡放得下的模型，优先用 DP（多副本）。只有显存放不下或需要压低单流延迟时，才上 TP。MoE 可以考虑 DP-attention 加 EP。

### Cited Findings
- **Megatron TP**：MLP 的第一个 GEMM 按列切分，GeLU 在各卡本地计算，第二个 GEMM 按行切分；attention 按头做列切分，输出投影按行切分。原文：“only two all-reduces in the forward path and two in the backward path”。f 和 g 互为共轭：f 在前向是恒等、反向是 all-reduce，g 正好相反。来源：[Megatron-LM, arXiv 1909.08053](https://arxiv.org/abs/1909.08053)（第 3 节、图 3）
- **nccl-tests 的带宽口径**：algbw = S/t。all-reduce 的 busbw = algbw × 2(n−1)/n，理论最优时间 t = (S/B) × 2(n−1)/n。reduce-scatter 和 all-gather 的系数为 (n−1)/n。busbw 可以直接和硬件峰值比较。来源：[nccl-tests PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
- **nccl-tests 用法**：用 `make` 编译（可指定 `CUDA_HOME`、`NCCL_HOME`；多机需加 `MPI=1`）。单机示例 `./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8`。参数含义：`-b/-e` 为起止字节数，`-f` 为倍增因子，`-g` 为每线程 GPU 数，`-n` 为迭代次数（默认 20），`-w` 为 warmup 次数，`-d` 为数据类型，`-G` 表示用 CUDA graph 捕获。来源：[nccl-tests README](https://github.com/NVIDIA/nccl-tests)
- **`nvidia-smi topo -m` 图例**：`NV#` 表示“traversing a bonded set of # NVLinks”；`PIX` 表示经过单个 PCIe 桥/交换机；`PXB` 表示经过多个 PCIe 桥；`PHB` 表示经过 PCIe Host Bridge；`NODE` 表示经过 NUMA 节点内的互连；`SYS` 表示经过跨 socket 的 SMP 互连。来源：[nvidia-smi 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
- **vLLM 官方建议**：单卡放得下就不做分布式；单机放不下单卡时用 TP（4 卡机设 `tensor_parallel_size=4`）；跨机用 TP+PP。若 GPU 数不能整除，或没有 NVLink（如 L40S），用 PP。vLLM 的 TP 实现的就是 Megatron 算法。来源：[parallelism_scaling.md @v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/serving/parallelism_scaling.md)
- **vLLM 的 DP 与 EP**：
  - `--data-parallel-size=N` 启动 N 个独立 engine，`--max-num-seqs` 按每个 DP rank 计算。
  - 对 MoE，attention 用 DP，expert 层默认组成大小为 DP×TP 的 TP 组；加 `--enable-expert-parallel` 后改为 EP，且 EP_SIZE = TP_SIZE × DP_SIZE。
  - MoE 的各 DP rank 必须同步前向，空闲 rank 也要执行 dummy forward。
  - all2all 后端：`allgather_reducescatter`（默认）、`deepep_high_throughput`（prefill）、`deepep_low_latency`（decode，支持 CUDA graph）。
  
  来源：[data_parallel_deployment.md](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/serving/data_parallel_deployment.md)、[expert_parallel_deployment.md](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/serving/expert_parallel_deployment.md)（均为 v0.31.0）
- **Context Parallelism**（Meta）：提出 pass-KV 和 pass-Q 两种 ring attention。在 128 张 H100 上，Llama3 405B 的 1M 上下文 prefill 用时 77 s（并行效率 93%），128K 用时 3.8 s；在 RDMA 和 TCP 互连下都能扩展。来源：[arXiv 2411.01783](https://arxiv.org/abs/2411.01783)
- **几款 Qwen 模型参数**（来自 config.json 和 HF API）：
  - Qwen3-32B：64 层，h=5120，64 个 Q 头 / 8 个 KV 头，32.76B 参数。
  - Qwen2.5-72B：80 层，h=8192，64/8 头，72.71B 参数。
  - Qwen3-30B-A3B：48 层，h=2048，32/4 头，128 个专家、每 token 激活 top-8，30.53B 参数。
  
  来源：[Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json)、[Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/raw/main/config.json)、[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json)

### Inferences（通信量估算与 4×H800 推荐）
- **TP 每步通信**：每层 2 次 all-reduce，每次消息大小 S = tokens × h × 2 B。TP=4 时，每卡每次的传输量 ≈ 1.5S。
  - 例 1：Qwen3-8B、TP4、8192 token 的 prefill。S = 64 MiB，每次约 67e6 × 1.5 / busbw。假设 busbw 约 150 GB/s（未核实，用实验 1 实测），每次约 0.67 ms，72 次合计约 48 ms。同一步每卡的计算约 34 ms（按峰值），或约 57 ms（按 60% MFU）。可见在 H800 上，TP4 prefill 的通信与计算同量级，必须做重叠或换策略。H100 NVLink 带宽是其 2.25 倍，同样场景约 21 ms。
  - 例 2：decode batch 64 时 S 只有 512 KiB，带宽项仅约 5 µs，单次耗时由 NCCL 的固定延迟（通常十几微秒，未核实）决定。72 次 all-reduce 合计约 1 ms 量级，相对 8B 模型约 4–10 ms 的单步时间不可忽略。
- **EP 通信量**（DeepSeek-V3 量级）：dispatch 每 token ≈ top-k × h × 1 B（FP8）= 8 × 7168 ≈ 57 KB；combine ≈ 8 × 7168 × 2 B ≈ 115 KB。实际会因节点限制路由和本地专家去重而更少。
- **4×H800 SXM 推荐**（前提是 `topo -m` 显示 4 卡两两之间都是 NV#；若是 PCIe 卡加 bridge，则避免跨对做 TP4）：
  - **8B**（BF16 16.4 GB）：**DP=4**，起 4 个副本，吞吐最高。追求单流延迟时可以试 TP2×DP2，但每层 all-reduce 的延迟会抵消一部分收益。
  - **32B**（BF16 65.5 GB，单卡只剩约 7 GB 给 KV）：吞吐优先用 **FP8 W8A8 + DP=4**（每卡约 33 GB 权重）；保持 BF16 时用 **TP2×DP2**；对延迟极敏感时用 TP4。KV 每 token 256 KiB（BF16）。
  - **70B 级**（BF16 约 145 GB）：**BF16 用 TP4**，每卡约 36 GB 权重、约 30 GB KV；或 **FP8 + TP2×DP2**，每卡约 36 GB。两者在 prefill 通信上的取舍需要实测。不推荐 PP，因为单机 NVLink 的 TP 延迟更低，而 PP 在推理时有气泡，会降低单请求速度。
  - **30B-A3B MoE**（BF16 61 GB，每 token 激活约 3B 参数）：单卡放得下，**DP=4** 是基线。若想腾出显存给 KV，可试 `--data-parallel-size 4 --enable-expert-parallel`（DP-attention + EP4，单机用默认的 allgather_reducescatter 后端）。还可以和 TP4 对比；注意它只有 4 个 KV 头，TP4 时每卡 1 个头。

### Gaps
- H800 上 4 卡 NCCL all-reduce 的实测 busbw，以及是否启用 NVLS（NVLink SHARP），没有一手数据（「未核实」），需要实验测量。
- SGLang 的 `--enable-dp-attention` 等参数：在 v0.5.21 的 `server_args.py` 中 grep 没有命中（可能文件路径或写法不同），参数名标为「未核实」。只核实了版本号：[SGLang releases](https://github.com/sgl-project/sglang/releases)。

## 7. DeepSeek-V3/R1 在 H800 上的推理部署

### Takeaway
DeepSeek 的推理部署可以概括为“大 EP + DP-attention + 冗余专家 + 双 micro-batch 计算通信重叠”。每台 H800 节点平均可处理约 73.7k tok/s 的 prefill 输入（含 cache 命中），或约 14.8k tok/s 的 decode 输出。

### Cited Findings
- **V3 技术报告中的部署**：
  - prefill 最小单元为 4 节点 32 卡。attention 用 TP4+SP 加 DP8，MoE 用 EP32。设 32 个冗余专家，每卡托管 8 个原始专家加 1 个冗余专家。两个 micro-batch 交替执行，以重叠计算与通信。
  - decode 最小单元为 40 节点 320 卡。attention 用 TP4+SP 加 DP80，MoE 用 EP320，每卡只放 1 个专家，另有 64 卡托管冗余专家和共享专家。dispatch/combine 通过 IB 点对点直传，使用 IBGDA。
  
  来源：[arXiv 2412.19437](https://arxiv.org/html/2412.19437)
- **Day 6 推理系统概览**（2025-02-27/28 统计）：
  - prefill：Routed Expert EP32、MLA/Shared Expert DP32，部署单元 4 节点，32 个冗余专家，每卡 9 个 routed 专家加 1 个 shared 专家。
  - decode：EP144、DP144，部署单元 18 节点，每卡 2 个 routed 专家加 1 个 shared 专家。
  - prefill 用 dual-batch overlap；decode 用把 attention 拆成两段的 5 级流水；设有 prefill、decode、EP 三种负载均衡器。
  - 吞吐：每节点平均约 73.7k tok/s（prefill 输入，含 cache 命中），或约 14.8k tok/s（decode 输出）。
  - 24 小时统计：输入 608B token，其中 342B（56.3%）命中磁盘 KV cache；输出 168B token；平均输出速度 20–22 tok/s；平均 KV 长度 4,989；峰值 278 个节点，平均 226.75 个节点。
  - 成本收益：按 H800 租金 $2/GPU·h 计算，日成本 $87,072；按 R1 定价的理论日收入 $562,027，“cost profit margin 545%”。DeepSeek 自己注明实际收入远低于此，原因包括 V3 定价更低、免费服务、夜间折扣。
  
  来源：[open-infra-index Day 6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)
- **ISCA'25 论文的理论上限**：基于 400Gbps IB 的 EP，decode TPOT 理论上限约 14.76 ms（约 67 tok/s）；换成 GB200 NVL72（900 GB/s）约为 0.82 ms（约 1200 tok/s）。来源：[arXiv 2505.09343](https://arxiv.org/html/2505.09343)

### Inferences
- 两份文档的配置不同：V3 报告是 decode 40 节点、EP320；Day 6 是 18 节点、EP144。说明线上配置在报告发布后有调整。引用时应注明出处和时间。
- 4×H800 单机复现不了这种规模，但同样的思路可以缩小验证：DP-attention + EP 能让每卡只存一部分专家、把显存留给 KV；双 micro-batch 能让通信藏在计算后面。可以在 Qwen3-30B-A3B 上用 vLLM 的 DP+EP 做缩小版实验。

### Gaps
- Day 6 的吞吐数字依赖 cache 命中率和输出长度分布，不能直接换算成单机的期望值。

## 8. Profiling 方法

### Takeaway
推荐的流程是：先用 nsys 看整体 timeline，判断 GPU 有没有空泡、各类 kernel 的占比、NCCL 是否与计算重叠；再用 torch.profiler / vLLM `--profiler-config` 定位到 Python 调用栈；最后只在单个热点 kernel 上用 Nsight Compute 做深入分析。vLLM 从 v0.13.0 起用 `--profiler-config` 替代了旧的环境变量。

### Cited Findings
- vLLM v0.31.0 的 profiling 文档（[profiling.md](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/contributing/profiling.md)）：
  - **torch profiler**：`vllm serve <model> --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile"}'`。可选项包括 `torch_profiler_record_shapes`、`torch_profiler_with_memory`、`torch_profiler_with_stack`（默认开）、`torch_profiler_with_flops`、`torch_profiler_activities` 等。用 `POST /start_profile` 和 `/stop_profile` 控制采集区间（支持 `profile_prefix`、`delay_iterations`、`max_iterations`），或给 `vllm bench serve` 加 `--profile`。trace 用 Perfetto（ui.perfetto.dev）查看。文档说明 `--profiler-config` 需要 vLLM ≥ v0.13.0。还警告最终用户不要在生产中开启 profiling；flush 70B 模型约 100 个请求的 trace 在 H100 上要约 10 分钟。
  - **Nsight Systems**：推荐参数 `--trace-fork-before-exec=true --cuda-graph-trace=node`，并建议设置 `VLLM_WORKER_MULTIPROC_METHOD=spawn`。对 server 做 profiling 时加 `--capture-range=cudaProfilerApi --capture-range-end repeat`，同时给 vLLM 加 `--profiler-config.profiler cuda`。分析时用 `nsys stats report1.nsys-rep`，可以得到 `cuda_gpu_kern_sum` 等汇总表。
  - **Triton Proton**：`"profiler": "proton"`，支持 CUDA graph attribution。
- Nsight Systems 用户指南（2026.5.1）记载了以下参数：`--capture-range=cudaProfilerApi|nvtx|hotkey`、`--capture-range-end`、`--cuda-graph-trace=<granularity>[:<launch origin>][:<nvtx mode>]`、`--gpu-metrics-devices=[none|all|cuda-visible|<index>]`（采样 SM 和 Tensor Core 活跃度等 GPU 指标）、`--cuda-memory-usage`、`--trace-fork-before-exec`。来源：[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)、[Release Notes](https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html)

### Inferences（看 timeline 时重点看什么）
1. **GPU 空泡**：两个 kernel 之间的空白。decode 时空白多，通常说明 CPU 调度、Python 或 launch 开销太大，可以对比开关 CUDA graph 的效果。
2. **kernel 占比**：prefill 时 GEMM（`sm90_xmma_gemm`、cutlass、DeepGEMM）应占主导。decode 时 attention 和小算子（RMSNorm、激活函数）的占比会升高。
3. **通信**：TP 时看 `ncclDevKernel_AllReduce` 的占比，以及它是否和计算串行。EP 时看 dispatch/combine 的耗时。
4. **gpu-metrics**：把 SM active / Tensor active 与 roofline 推算对照。decode 时 Tensor 利用率低是正常现象。
5. **CPU 侧**：调度器（scheduler）每步的耗时，以及 detokenize 是否阻塞了下一步。

### Gaps
- `torch.profiler` 的 API 细节（`schedule`、`tensorboard_trace_handler`、TensorBoard 插件是否已弃用）本轮未抓取 PyTorch 官方页面，标为「未核实」，请以 [PyTorch profiler 文档](https://docs.pytorch.org/docs/stable/profiler.html) 为准。
- Nsight Compute 的 CLI（`ncu --set full -k <regex> -c <n>` 等）本轮未核实语法，标为「未核实」。

## 9. 4×H800 动手实验（每项都写明目标、命令、预期和验收）

### Takeaway
下面四个实验按“拓扑与通信 → 单卡 roofline → 显存预算 → 并行策略与 profiling”的顺序安排，总计约 1–1.5 天 GPU 时间。命令中只使用了上文已核实的参数，未核实的部分已标注。

### Cited Findings
命令语法的来源：[nccl-tests README](https://github.com/NVIDIA/nccl-tests)、[nvidia-smi 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)、[vLLM profiling.md / parallelism_scaling.md / data_parallel_deployment.md / expert_parallel_deployment.md @v0.31.0](https://github.com/vllm-project/vllm/tree/v0.31.0/docs)。

### Inferences（实验设计）
- **实验 1：拓扑与 NCCL 带宽**（约 30 分钟，4 卡）
  - 目标：确认机器是 SXM（NVSwitch）还是 PCIe，并测出 all-reduce 的 busbw。
  - 步骤：
    1. 运行 `nvidia-smi topo -m`。
    2. 在 nccl-tests 目录执行 `make`（需要 CUDA 和 NCCL 头文件）。
    3. 运行 `./build/all_reduce_perf -b 8 -e 8G -f 2 -g 4`，再用 `-g 2` 重复一次；可选 `-d bfloat16`（数据类型名称未核实）。
  - 预期：SXM 机型上 4 卡两两之间都是 `NV#`；大消息的 busbw 稳定在 H800 NVLink 单向 200 GB/s 以内（DeepSeek 实测约 160 GB/s）；小消息的耗时是十几到几十微秒的固定延迟。
  - 验收：记录 busbw 曲线，以及 8 KiB、512 KiB、64 MiB 三个点的耗时，并代回第 6 节的 TP 通信估算。
- **实验 2：decode roofline 验证**（约 1 小时，单卡）
  - 目标：验证“batch 摊薄权重读取，但摊不薄 KV 读取”。
  - 步骤：运行 `vllm bench latency --model Qwen/Qwen3-8B --num-iters-warmup 5 --num-iters 10 --batch-size {1,16,64,256} --input-len {512,2048} --output-len 128`。
  - 预期：batch=1 时 tok/s 接近但低于约 220 的上限；batch=64、上下文 2k 时聚合吞吐低于约 6,200 tok/s；batch 从 64 增加到 256 时，单 token 延迟明显上升。
  - 验收：画出实测与理论上限的对比表，带宽利用率在 60–90% 之间即可。
- **实验 3：显存预算**（约 1 小时，单卡）
  - 目标：复核第 3 节的 KV 容量公式。
  - 步骤：分别用 `--gpu-memory-utilization 0.80` 和 `0.92` 运行 `vllm serve Qwen/Qwen3-8B`；再加 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` 对比；可选在 Python API 中设 `enforce_eager=True`，以及 `kv_cache_dtype="fp8"`。记录日志中的 “Available KV cache memory”、“GPU KV cache size”、“Maximum concurrency”。
  - 预期：每 token 的 KV 为 144 KiB；FP8 KV 下 token 容量约翻倍；利用率每增加 0.12，KV 增加约 9.5 GiB。
  - 验收：日志中的 token 数与 KV_bytes ÷ 147,456 的误差在 5% 以内。
- **实验 4：并行策略对比与 nsys 分析**（约半天，4 卡）
  - 目标：为 32B 和 MoE 模型选择并行方案。
  - 步骤：
    1. Qwen3-32B 分别用 `--tensor-parallel-size 4` 和 `--data-parallel-size 2 --tensor-parallel-size 2` 两种配置。
    2. Qwen3-30B-A3B 分别用 `--data-parallel-size 4`、`--data-parallel-size 4 --enable-expert-parallel`、`--tensor-parallel-size 4` 三种配置。
    3. 用 `vllm bench serve --backend vllm --model ... --dataset-name sharegpt --dataset-path sharegpt.json --num-prompts N` 测 TTFT、TPOT 和吞吐（random 数据集的参数未核实）。
    4. 选最优配置，加 `nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end repeat vllm serve ... --profiler-config.profiler cuda` 和 `VLLM_WORKER_MULTIPROC_METHOD=spawn`，用 `--profile` 少量采样，然后运行 `nsys stats`。
  - 预期：高并发下 DP 配置的吞吐最高，低并发下 TP4 的 TPOT 最低；TP 配置的 timeline 上 AllReduce kernel 的占比在 prefill 时明显。
  - 验收：给出每种配置在两档并发下的表格，并写出 NCCL kernel 占 GPU 时间的百分比。

### Gaps
- `vllm bench latency` 是否接受所有 engine 参数（如 `--tensor-parallel-size`）未逐项核实；nccl-tests 的 `-d` 数据类型命名未核实。
- 实验中的预期数字都是推算值，需要以实测结果为准。
