# 推理引擎与 Serving/调度系统（vLLM / SGLang 为主）研究笔记

> 访问日期：所有链接均于 2026-10-14 访问（部分源码于 2026-10-15 复查）。版本基准：**vLLM v0.31.0**（tag `v0.31.0`）、**SGLang v0.5.21**（tag `v0.5.21`）。源码与文档优先读取对应 git tag 下的文件（raw.githubusercontent.com），因此下文的默认值都绑定到上述版本；升级后必须重新核对。
> 标记约定：「未核实」= 本次没有抓到一手来源；「证据不足」= 抓取失败或来源互相矛盾。

---

## 1. 核心概念与奠基论文（batching、PagedAttention、chunked prefill、抢占、prefix caching、PD 分离、speculative decoding）

### Takeaway
现代引擎的调度核心是 **iteration-level（continuous）batching + 分页 KV cache + token budget 驱动的 chunked prefill**，在此基础上叠加 prefix caching（vLLM 的 block hash / SGLang 的 radix tree）、PD 分离和引擎内 speculative decoding。vLLM V1 的调度器源码明确写着“不区分 prefill 阶段和 decode 阶段”，每个请求只是让 `num_computed_tokens` 追上 `num_tokens_with_spec`。这样一个统一抽象就把 chunked prefill、prefix caching 和 spec decode 都包住了。

### Cited Findings
**Batching 演进**
- Static batching：凑齐一批后一起跑完，短请求要等长请求结束；dynamic batching：按时间窗或批大小凑批，粒度仍是“请求”。Orca 提出 **iteration-level scheduling**：每生成一个 token（一次 iteration）就重新决定 batch 成员，完成的请求立即退出，新请求立即加入，并用 selective batching 处理不同长度的 attention。这就是后来所说的 continuous batching / in-flight batching。— [Orca, OSDI'22（USENIX 页面）](https://www.usenix.org/conference/osdi22/presentation/yu)（「未核实」：本次没有抓取该页面，会议和作者信息来自任务说明与常识，Orca 不在 arXiv 上）
- vLLM V1 调度器注释（v0.31.0）："There's no 'decoding phase' nor 'prefill phase' in the scheduler. Each request just has the num_computed_tokens and num_tokens_with_spec … general enough to cover chunked prefills, prefix caching, speculative decoding" — [vllm/v1/core/sched/scheduler.py@v0.31.0 L562-573](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/core/sched/scheduler.py)

**PagedAttention / block table**
- 论文："Efficient Memory Management for Large Language Model Serving with PagedAttention"，Kwon 等，arXiv 2309.06180，arXiv 注释为 "SOSP 2023"。— [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- 思想：KV cache 按固定大小的 block（页）分配，每个请求用一张 block table 把逻辑块映射到物理块，从而消除外部碎片，并支持块级共享（copy-on-write、prefix 共享）。vLLM v0.31.0 的 `CacheConfig.DEFAULT_BLOCK_SIZE = 16`（`block_size=None` 表示由平台或 attention backend 决定，FlashAttention backend 支持 `MultipleOf(16)`）。— [vllm/config/cache.py@v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/cache.py)、[flash_attn.py@v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/attention/backends/flash_attn.py)

**Chunked prefill / stall-free batching**
- SARATHI（arXiv 2308.16369, 2023-08）提出用 chunked prefill 让 decode “搭便车”；Sarathi-Serve 的论文为 "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve"（arXiv 2403.02310，Agrawal 等）。— [arXiv:2308.16369](https://arxiv.org/abs/2308.16369)、[arXiv:2403.02310](https://arxiv.org/abs/2403.02310)（会议 OSDI'24 来自任务说明，arXiv 元数据中没有会议注释，「未核实」）
- vLLM 实现：每步有一个 `token_budget`（= `max_num_batched_tokens` 派生的 `max_num_scheduled_tokens`）。调度器**先排 RUNNING 请求（decode 优先），再排 WAITING 请求**，长 prompt 在预算不足时会被切块。`long_prefill_token_threshold`（默认 0 = 不限制）用于限制单个长 prefill 占用的预算；v0.31.0 起该阈值会按等待中的 prefill 数量自适应，batch 里只有一个请求时不切。— [scheduler.py@v0.31.0 L582-628, L872](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/core/sched/scheduler.py)、[vllm/config/scheduler.py@v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/scheduler.py)、[v0.31.0 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)

**Preemption（recompute vs swap）与准入控制**
- vLLM V1（v0.31.0）中 `allocate_slots` 失败时会抢占：FCFS 策略取 `self.running[-1]`（最后加入的请求），priority 策略取 `(priority, arrival_time)` 最大者。`_preempt_request` 释放该请求的全部 KV blocks，设 `num_computed_tokens = 0`，状态置为 `PREEMPTED`，然后 `waiting.prepend_request` 放回等待队列队首，即 **recompute 式抢占**。scheduler.py 中没有 swap 路径（我对该文件 grep "swap" 无结果）。— [scheduler.py@v0.31.0 L744-800, L1538-1581](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/core/sched/scheduler.py)
- 准入相关参数：`max_num_seqs`（运行中请求上限，也决定 runner buffer 和 CUDA graph 容量）；v0.31.0 新增 `max_num_active_seqs`，只限制进入 RUNNING 的请求数，不缩小 runner 和 graph 容量；waiting 队列被改为“已持有 KV blocks 的请求优先调度”。— [vllm/config/scheduler.py@v0.31.0 L63-76](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/scheduler.py)、[release notes v0.31.0](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)
- SGLang 把这个过程称为 retract（KV 不够时撤回请求）。`schedule_conservativeness` 默认 1.0，帮助文字说明“如果请求频繁被 retract，就调大该值”。— [arg_groups/fields/schedule.py@v0.5.21](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/arg_groups/fields/schedule.py)

**Prefix caching**
- vLLM APC：对每个**满** block 计算 `hash(parent_hash, block_tokens, extra_hashes)`，extra 中包括 LoRA ID、多模态输入 hash 和 `cache_salt`。只缓存满块。v0.11 起默认 hash 算法为 `sha256`，可通过 `--prefix-caching-hash-algo` 选 `sha256 / sha256_cbor / xxhash / xxhash_cbor`。所有 `KVCacheBlock` 预先分配在 BlockPool 中，空闲块放在 free queue 里，按 **LRU** 从队首驱逐；请求结束时块以**逆序**加入队尾，所以请求尾部的块先被驱逐。命中的块会被 “touch”（ref_cnt+1 并移出 free queue）。— [docs/design/prefix_caching.md@v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/design/prefix_caching.md)
- SGLang RadixAttention：论文 "SGLang: Efficient Execution of Structured Language Model Programs"，Zheng 等，arXiv 2312.07104。— [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)（NeurIPS'24 来自任务说明，arXiv 元数据中没有注明，「未核实」）
- 在 v0.5.21 中，`RadixCache`（`match_prefix / insert / evict / cache_unfinished_req / inc_lock_ref / dec_lock_ref`）位于 `python/sglang/srt/mem_cache/radix_cache.py`。`radix_eviction_policy` 默认 `"lru"`，可选 lfu/slru/priority/tlru。`disable_radix_cache` 默认 False，即默认开启。v0.5.21 release notes 称 "The prefix cache now runs on a Rust core by default"，对应仓库中的 `rust/sglang-radix-tree/`。— [radix_cache.py@v0.5.21](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/mem_cache/radix_cache.py)、[memory.py@v0.5.21](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/arg_groups/fields/memory.py)、[SGLang v0.5.21 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.21)

**Prefill–Decode 分离与 KV 传输**
- DistServe：arXiv 2401.09670，注释 "OSDI 2024"，提出以 goodput（满足 SLO 的吞吐）为目标拆分 prefill/decode。— [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)
- Splitwise：arXiv 2311.18677（Patel 等，"phase splitting"）。— [arXiv:2311.18677](https://arxiv.org/abs/2311.18677)（ISCA'24 「未核实」）
- Mooncake：arXiv 2407.00079，"A KVCache-centric Disaggregated Architecture for LLM Serving"（Qin 等）。— [arXiv:2407.00079](https://arxiv.org/abs/2407.00079)（FAST'25 「未核实」）
- KV 传输库在 PyPI 上的最新版本（2026-10-14 查询）：`nixl` 1.5.0（2026-09-30）、`lmcache` 0.5.5（2026-09-12，另有 0.5.6rc）、`mooncake-transfer-engine` 0.3.13.post1（2026-08-31）。— [PyPI nixl](https://pypi.org/project/nixl/)、[PyPI lmcache](https://pypi.org/project/lmcache/)、[PyPI mooncake-transfer-engine](https://pypi.org/project/mooncake-transfer-engine/)
- vLLM v0.31.0 的 KV connector 都在 `vllm/distributed/kv_transfer/kv_connector/v1/` 下，包括 `nixl/`（含 pull/push scheduler 和 worker）、`lmcache_connector.py`、`mooncake/`（含 `store/`）、`offloading_connector.py`、`multi_connector.py`、`hf3fs/`、`moriio/`。对应文档：`docs/features/disagg_prefill.md`、`nixl_connector_usage.md`、`mooncake_connector_usage.md`、`kv_offloading_usage.md`。— [vLLM 源码树@v0.31.0](https://github.com/vllm-project/vllm/tree/v0.31.0/vllm/distributed/kv_transfer/kv_connector/v1)
- SGLang PD 分离文档写明 "Currently, we support Mooncake and NIXL as the transfer engine"，`disaggregation_transfer_backend` 默认 `"mooncake"`，bootstrap 端口默认 8998。v0.5.21 新增 PD 实例无需重启即可切换 prefill/decode 角色。— [pd_disaggregation.mdx@v0.5.21](https://github.com/sgl-project/sglang/blob/v0.5.21/docs/docs/advanced_features/pd_disaggregation.mdx)、[disagg.py@v0.5.21](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/arg_groups/fields/disagg.py)、[release v0.5.21](https://github.com/sgl-project/sglang/releases/tag/v0.5.21)

**Speculative decoding**
- Leviathan 等，"Fast Inference from Transformers via Speculative Decoding"，arXiv 2211.17192（ICML 2023 Oral）；Chen 等，"Accelerating LLM Decoding with Speculative Sampling"，arXiv 2302.01318。两篇都用 rejection sampling 保证输出分布与目标模型一致。— [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)、[arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- Medusa（多解码头）arXiv 2401.10774；EAGLE（特征层草稿）arXiv 2401.15077；EAGLE-2（动态草稿树）arXiv 2406.16858；EAGLE-3（training-time test，多层特征融合）arXiv 2503.01840；MTP 见 DeepSeek-V3 技术报告 arXiv 2412.19437。— [2401.10774](https://arxiv.org/abs/2401.10774)、[2401.15077](https://arxiv.org/abs/2401.15077)、[2406.16858](https://arxiv.org/abs/2406.16858)、[2503.01840](https://arxiv.org/abs/2503.01840)、[2412.19437](https://arxiv.org/abs/2412.19437)
- vLLM v0.31.0 的 spec decode 源码 `vllm/v1/spec_decode/` 包括 `eagle.py`、`medusa.py`、`ngram_proposer.py` / `ngram_proposer_gpu.py`、`suffix_decoding.py`、`draft_model.py`、`dflash.py`；拒绝采样器在 `vllm/v1/sample/rejection_sampler.py`；配置在 `vllm/config/speculative.py`。— [源码树@v0.31.0](https://github.com/vllm-project/vllm/tree/v0.31.0/vllm/v1/spec_decode)
- SGLang v0.5.21 的 `speculative_algorithm` 内置 "EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH, DSPARK, UNO"，相关参数有 `speculative_num_steps / speculative_eagle_topk / speculative_num_draft_tokens / speculative_draft_model_path`（默认都是 None，自动推导）。— [spec.py@v0.5.21](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/arg_groups/fields/spec.py)

### Inferences
- 对熟悉 OxyGen 的学习者来说，可以这样类比：OxyGen 的“一次观测 prefill 被多个 expert 共享的统一 KV cache”对应 vLLM 的 block 级 prefix 共享（ref_cnt）和 SGLang 的 radix 节点共享；它的“跨帧 continuous batching 语言解码”就是 Orca 的 iteration-level scheduling 在 VLA 场景中的特化。
- vLLM V1 只做 recompute 抢占，所以 KV 紧张时会表现为 `vllm:num_preemptions` 上升、TTFT 和 E2E 尾延迟变差，而不是出现 PCIe swap 流量。

### Gaps
- Orca、Sarathi-Serve、SGLang、Splitwise、Mooncake 的会议归属（OSDI'22/OSDI'24/NeurIPS'24/ISCA'24/FAST'25）本次没有逐一抓取会议页面核对，标为「未核实」。arXiv 元数据只对 PagedAttention（SOSP 2023）、DistServe（OSDI 2024）和 Leviathan（ICML 2023）给出了会议注释。

---

## 2. vLLM 现状：版本、V1 进程架构、默认开启的特性、关键参数

### Takeaway
最新稳定版为 **vLLM 0.31.0**（GitHub 发布于 2026-10-05，PyPI 上传于 2026-10-05），约两周一个小版本。V1 是多进程架构：API server ↔(ZMQ)↔ EngineCore busy loop（含 Scheduler 和 KVCacheManager）→ Executor → 每 GPU 一个 Worker / ModelRunner。在 CUDA 上，**Model Runner V2（MRV2）在满足条件时已是默认**。prefix caching、chunked prefill（模型支持时）和 async scheduling 默认开启；默认 `gpu_memory_utilization=0.92`。

### Cited Findings
**版本**
- PyPI `vllm` 最新版 0.31.0，上传于 2026-10-05T07:41Z。之前的版本：0.30.0（09-22）、0.29.0（09-09）、0.28.0（08-26）、0.27.1（08-11）。`requires_python <3.15,>=3.10`。0.31.0 依赖固定 `torch==2.13.0`、`flashinfer-python==0.7.0.post1`、`transformers>=5.10.4,<5.18.0`、`xgrammar==0.2.7`。— [PyPI vllm](https://pypi.org/project/vllm/0.31.0/)
- GitHub release v0.31.0 发布于 2026-10-05T06:44Z，包含 717 commits / 307 contributors。要点：DeepSeek-V4.1-Flash 优化；新增 `vllm preload` 权重常驻守护进程（快速重启）；MRV2 支持 draft-model 投机；MoonEP all2all（`--all2all-backend moonep`）；新增 `--max-num-active-seqs`；`--long-prefill-token-threshold` 自适应；`--enforce-eager` 现在也会关闭 JIT kernel warmup；`tokenizer_mode="slow"` 被移除。— [vLLM v0.31.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)

**进程架构（docs/design/arch_overview.md@v0.31.0）**
- API Server 进程负责 HTTP、tokenize 和多模态加载，用 ZMQ 与 engine core 通信。默认 1 个，DP 时自动扩展为 DP 个（可用 `--api-server-count` 设置），并与所有 engine core 多对多连接。
- Engine Core 进程运行 scheduler、管理 KV cache，busy loop 不断调度并下发给 GPU worker。每个 DP rank 一个（`vllm/v1/engine/core.py`）。
- GPU Worker 进程每 GPU 一个，数量为 TP×PP（每个 engine core），见 `vllm/v1/executor/multiproc_executor.py` 和 `vllm/v1/worker/gpu_worker.py`。DP>1 时还会多一个 DP Coordinator（`vllm/v1/engine/coordinator.py`）。
- 进程数举例：`vllm serve -tp=4` 共 1 API + 1 core + 4 worker = 6 个进程；`-tp=2 -dp=4` 共 4+4+8+1 = 17 个。— [arch_overview.md@v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/design/arch_overview.md)

**MRV2 / persistent batch / async scheduling**
- MRV2 设计文档说明：V1 的 persistent batch 把持久状态张量直接当作输入，导致增删请求时需要复杂的重排，并且要维护 `CachedRequestState`。MRV2 把持久状态（预分配 `max_num_reqs` 行，多数平台默认 1024）与每步的输入张量解耦，在 GPU 上 gather；抢占被视为完成，恢复时作为新状态重新加入；整体 “Async-First”，即 step N 在 GPU 上执行时 CPU 已在准备 step N+1。— [docs/design/model_runner_v2.md@v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/design/model_runner_v2.md)
- 选择逻辑 `VllmConfig.use_v2_model_runner`：环境变量 `VLLM_USE_V2_MODEL_RUNNER` 有设置时以它为准；否则在 ROCm 某些架构上默认用 V1；没有 Triton 时用 V1；存在 MRV2 不支持的特性时回退 V1 并打印 warning；其余情况 `return True`。— [vllm/config/vllm.py@v0.31.0 L701-761](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/vllm.py)
- async scheduling：`SchedulerConfig.async_scheduling` 默认为 None，由 VllmConfig 解析，一般情况下置为 True。以下情况会关闭：pooling 模型、部分 spec decode 方法、executor 不支持、某些 PP>1 场景、ROCm DeepEP HT+DBO。— [vllm/config/vllm.py@v0.31.0 L1585-1635](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/vllm.py)

**编译 / CUDA graph**
- `optimization_level` 默认 **O2**。O>0 时 compilation mode 为 `VLLM_COMPILE`（torch.compile）。O2/O3 的 `cudagraph_mode = FULL_AND_PIECEWISE`。`enforce_eager=True` 会把 mode 置为 NONE、cudagraph_mode 置为 NONE。动态投机等场景会降级为 PIECEWISE。另有 `performance_mode`，默认 `"balanced"`，可选 `"interactivity"` 和 `"throughput"`；throughput 模式会把默认的 `max_num_batched_tokens` 与 `max_num_seqs` 翻倍。— [vllm/config/vllm.py@v0.31.0 L300-351, L442-453, L1694-1979](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/vllm.py)、[arg_utils.py@v0.31.0 L3064-3067](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/engine/arg_utils.py)

**关键参数与默认值（v0.31.0）**

| 参数 | 含义 | 默认（v0.31.0） | 来源 |
|---|---|---|---|
| `gpu_memory_utilization` | 本实例可用的显存比例（权重+激活+KV） | **0.92** | [cache.py](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/cache.py) |
| `kv_cache_memory_bytes` | 直接指定每 GPU 的 KV 字节数 | None（按利用率推导） | 同上 |
| `enable_prefix_caching` | APC | True（`model_config.is_prefix_caching_supported`） | cache.py / [arg_utils.py L2926](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/engine/arg_utils.py) |
| `prefix_caching_hash_algo` | block hash 算法 | `sha256` | cache.py |
| `block_size` | 每 block 的 token 数 | None→16（平台/后端可改） | cache.py |
| `enable_chunked_prefill` | chunked prefill | 跟随模型是否支持（通常 True） | arg_utils.py L2925 |
| `max_num_batched_tokens` | 每步 token 预算 | ≥70 GiB 且非 A100（H100/H800 类）：**serve 8192 / LLM 类 16384**；≥160 GiB：16384/16384 | [arg_utils.py L2858-2887](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/engine/arg_utils.py) |
| `max_num_seqs` | 每步最多请求数 | 上述 GPU：1024（其他 GPU 256） | 同上 |
| `max_num_active_seqs` | RUNNING 准入上限 | None | [scheduler.py config](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/scheduler.py) |
| `long_prefill_token_threshold` | 长 prefill 单步上限 | 0（不限） | 同上 |
| `policy` | 调度策略 | `fcfs`（另有 `priority`） | 同上 |
| `async_scheduling` | CPU 调度与 GPU 执行重叠 | 解析后通常为 True | vllm.py |
| `max_model_len` | 序列最大长度（prompt+生成） | 来自模型 config | scheduler.py config |
| `enforce_eager` | 关闭 compile 和 CUDA graph | False | vllm.py L1694 |
| `optimization_level` (`-O`) | 启动时间与性能的折中 | O2 | vllm.py L442 |
| TP / PP / DP | `-tp` / PP / `-dp`（`--data-parallel-size`） | 1 | arch_overview.md |

- 其他参数：`kv_cache_dtype`（auto/fp8 等）、`quantization`、`enable_expert_parallel` 和 `--all2all-backend` 都存在于 v0.31.0 配置体系中。v0.31.0 有一个破坏性变更：在线 FP8 量化不再用 `quantization="fp8"`，改用 `fp8_per_tensor` 简写。— [release v0.31.0](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)（各参数的完整取值列表本次未逐一核对）

**源码阅读地图（全部路径已在 v0.31.0 git tree 中确认存在）**

| 层 | 文件 |
|---|---|
| CLI / 入口 | `vllm/entrypoints/cli/main.py`、`cli/serve.py`、`cli/benchmark/{serve,throughput,latency,sweep}.py` |
| OpenAI server | `vllm/entrypoints/openai/api_server.py`、`openai/chat_completion/serving.py`、`openai/completion/serving.py`；launcher 在 `vllm/entrypoints/launchers/api_server/` |
| 前端引擎 | `vllm/v1/engine/async_llm.py`（AsyncLLM）、`llm_engine.py`、`core_client.py`（ZMQ client）、`input_processor.py`、`output_processor.py`、`detokenizer.py` |
| EngineCore | `vllm/v1/engine/core.py`（busy loop）、`coordinator.py`（DP） |
| 调度 | `vllm/v1/core/sched/scheduler.py`、`async_scheduler.py`、`request_queue.py`、`output.py`（SchedulerOutput） |
| KV 管理 | `vllm/v1/core/kv_cache_manager.py`、`block_pool.py`、`kv_cache_coordinator.py`、`single_type_kv_cache_manager.py`、`kv_cache_utils.py`（block hash）；设计文档 `docs/design/hybrid_kv_cache_manager.md` |
| Executor / Worker | `vllm/v1/executor/{multiproc_executor,uniproc_executor,ray_executor}.py`、`vllm/v1/worker/gpu_worker.py` |
| Model runner | V1：`vllm/v1/worker/gpu_model_runner.py` + `gpu_input_batch.py`（persistent batch）+ `block_table.py`；**MRV2：`vllm/v1/worker/gpu/model_runner.py`**、`gpu/input_batch.py`、`gpu/cudagraph_utils.py` |
| 采样 | `vllm/v1/sample/sampler.py`、`rejection_sampler.py` |
| Attention | `vllm/v1/attention/backends/{flash_attn,flashinfer,triton_attn,flex_attention,...}.py`、`registry.py` |
| 编译 | `vllm/compilation/{backends,piecewise_backend,cuda_graph,decorators}.py`、`vllm/v1/cudagraph_dispatcher.py`；文档 `docs/design/{cuda_graphs,torch_compile,optimization_levels}.md` |
| 指标 | `vllm/v1/metrics/{loggers,stats,prometheus}.py` |
| 配置 | `vllm/config/{scheduler,cache,parallel,compilation,speculative,vllm}.py`、`vllm/engine/arg_utils.py` |

— [vLLM git tree@v0.31.0](https://github.com/vllm-project/vllm/tree/v0.31.0)

### Inferences
- arch_overview.md 仍然说 “OpenAI server 使用 `AsyncLLMEngine`（vllm/engine/async_llm_engine.py）”，但 V1 的实际实现是 `vllm/v1/engine/async_llm.py`（AsyncLLM）。阅读时以 v1 目录为准，`vllm/engine/` 下的文件主要是兼容层（推断，未逐行确认）。
- 学习路线建议：先读 V1 的 `gpu_model_runner.py`，理解 persistent batch 的经典实现；再读 MRV2（`v1/worker/gpu/`），对比 MRV2 设计文档中列出的 V1 问题。这样比只读一个版本更容易体会“为什么要重写”。

### Gaps
- 每个 attention backend 在 H800（SM90）上的默认选择逻辑本次未核对（`vllm/platforms/cuda.py`），「未核实」。
- MRV2 的 `_get_v2_model_runner_unsupported_features()` 具体列表未展开，因此无法断言“某个模型在 H800 上一定走 MRV2”。可在启动日志中看是否出现 "Model Runner V2 does not yet support … using the V1 model runner" 这条 warning 来确认。

---

## 3. vLLM 安装、serve、bench 命令与 Prometheus 指标

### Takeaway
v0.31.0 的 release notes 写明 **PyPI 默认 wheel 为 CUDA 13.0**（`pip install vllm` 或 `uv pip install vllm --torch-backend=auto`），CUDA 12.9 的 wheel 在 release assets 中，Docker 有 `vllm/vllm-openai:v0.31.0` 和 `-cu129` 两个标签。但同一 tag 下的安装文档仍写着“二进制以 CUDA 12.9 编译”，**两处来源互相矛盾**，以 release notes 为准，并在 H800 机器上先确认驱动版本。`vllm bench serve` 默认不忽略 EOS、不做 warmup、默认 dataset 为 `random`、默认 `--request-rate inf`。

### Cited Findings
- release notes v0.31.0 的发布产物：PyPI (CUDA 13.0) `pip install vllm`；`uv pip install vllm --torch-backend=auto`；Docker `docker pull vllm/vllm-openai:v0.31.0`（CUDA 13.0，默认）和 `vllm/vllm-openai:v0.31.0-cu129`；assets 中有 CUDA 12.9 / 13.0 的 x86_64 与 arm64 wheel。— [release v0.31.0](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)
- 与之矛盾：`docs/getting_started/installation/gpu.cuda.inc.md@v0.31.0` 写的是 "vLLM contains pre-compiled C++ and CUDA (12.9) binaries" 和 "binaries are compiled with CUDA 12.9 … by default"，并给出指定 CUDA 版本 wheel 的安装方式：`uv pip install "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_28_${CPU_ARCH}.whl" --extra-index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"`。同一文档还要求 GPU compute capability ≥7.5、仅支持 Linux、Python 3.10–3.13（而 PyPI 元数据写的是 <3.15）。— [gpu.cuda.inc.md@v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/getting_started/installation/gpu.cuda.inc.md)、[gpu.md@v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/getting_started/installation/gpu.md)。**判定为「证据不足」：默认 CUDA 版本以 release notes（CUDA 13.0）为较新、较具体的来源。**
- Python-only 开发安装：`VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto`，会下载 base commit 的预编译库和 `vllm-rs` 二进制。— 同上文档
- `vllm bench serve` 参数（`vllm/benchmarks/serve.py@v0.31.0`）：`--backend`（默认 `openai`）、`--endpoint`（默认 `/v1/completions`）、`--request-rate`（默认 inf；非 inf 时按 Poisson 过程，或按 `--burstiness` 用 gamma 分布）、`--max-concurrency`（默认 None）、`--num-warmups`（默认 0）、`--ignore-eos`、`--percentile-metrics`（可选 ttft,tpot,itl,e2el 等，默认 "ttft,tpot,itl"）、`--metric-percentiles`（默认 "99"）、`--goodput KEY:VALUE`（ttft/tpot/e2el，单位 ms，引用 DistServe 的定义）、`--ramp-up-strategy linear|exponential`、`--save-result`、`--result-dir`、`--plot-timeline`。数据集参数在 `vllm/benchmarks/datasets/datasets.py`：`--dataset-name` 默认 `random`，可选 sharegpt/burstgpt/sonnet/random/random-mm/hf/custom/prefix_repetition/spec_bench/speed_bench/timed_trace 等；另有 `--num-prompts`、`--random-input-len`、`--random-output-len`、`--random-range-ratio`、`--random-prefix-len`、`--prefix-repetition-{prefix-len,suffix-len,num-prefixes,output-len}`、`--dataset-path`。— [serve.py@v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/benchmarks/serve.py)、[datasets.py@v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/benchmarks/datasets/datasets.py)
- `vllm bench latency` 参数：`--input-len --output-len --batch-size --n --num-iters-warmup --num-iters --output-json --profile`。`vllm bench throughput` 参数：`--dataset-name --input-len --output-len --num-prompts --num-warmups --async-engine --prefix-len --output-json` 等。另外还有 `vllm bench sweep / startup / mm_processor` 子命令。— [latency.py](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/benchmarks/latency.py)、[throughput.py](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/benchmarks/throughput.py)、[cli/benchmark/](https://github.com/vllm-project/vllm/tree/v0.31.0/vllm/entrypoints/cli/benchmark)
- Prometheus 指标（`vllm/v1/metrics/loggers.py@v0.31.0` 中出现的名称）：
  - 延迟类：`vllm:time_to_first_token_seconds`（TTFT）、`vllm:inter_token_latency_seconds`（ITL）、`vllm:request_time_per_output_token_seconds`（每请求 TPOT）、`vllm:e2e_request_latency_seconds`、`vllm:request_queue_time_seconds`、`vllm:request_prefill_time_seconds`、`vllm:request_decode_time_seconds`、`vllm:request_inference_time_seconds`
  - 负载与容量：`vllm:num_requests_running`、`vllm:num_requests_waiting`、`vllm:num_requests_waiting_by_reason`、`vllm:kv_cache_usage_perc`、`vllm:num_preemptions`、`vllm:iteration_tokens_total`
  - 缓存：`vllm:prefix_cache_queries`、`vllm:prefix_cache_hits`、`vllm:external_prefix_cache_{queries,hits}`、`vllm:prompt_tokens_cached`、`vllm:kv_block_lifetime_seconds`、`vllm:kv_block_idle_before_evict_seconds`、`vllm:kv_block_reuse_gap_seconds`
  - 吞吐计数：`vllm:prompt_tokens`、`vllm:generation_tokens`（以 counter 导出时 Prometheus 客户端通常会加 `_total` 后缀，属推断）
  - 投机：`vllm:spec_decode_num_drafts`、`vllm:spec_decode_num_draft_tokens`、`vllm:spec_decode_num_accepted_tokens`、`vllm:spec_decode_num_accepted_tokens_per_pos`（`vllm/v1/spec_decode/metrics.py`）
  - — [loggers.py@v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/metrics/loggers.py)、[spec_decode/metrics.py@v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/spec_decode/metrics.py)；指标说明见 [docs/design/metrics.md](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/design/metrics.md)、[docs/usage/metrics.md](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/usage/metrics.md)

### Inferences
- H800 服务器需先运行 `nvidia-smi` 查看驱动支持的 CUDA 版本。如果驱动只支持 12.x，就用 cu129 wheel 或 `vllm/vllm-openai:v0.31.0-cu129` 镜像。SGLang v0.5.21 **只提供 CUDA 13**（见第 4 节），所以想在同一台机器上对比两个引擎，最好统一升级到支持 CUDA 13 的驱动。CUDA 13.0 对驱动的最低要求（常见说法为 R580）本次「未核实」，需查 NVIDIA CUDA Toolkit Release Notes。
- 比较 vLLM 与 SGLang 的 benchmark 数字时要注意两边默认值不同：vLLM bench 默认**不** ignore EOS、warmup 为 0、dataset 为 random；SGLang bench 默认 ignore EOS（用 `--disable-ignore-eos` 关闭）、warmup 为 1、dataset 为 sharegpt。必须显式对齐这些参数。

- CLI 写法（2026-10-15 补充核对）：`arg_utils.py@v0.31.0` 显式注册了 `--gpu-memory-utilization`（L1326）、`--enable-prefix-caching`（L1339）、`--max-num-batched-tokens`（L1668）、`--max-num-seqs`（L1682）、`--enable-chunked-prefill`（L1713）、`--speculative-config` / `-sc`（L1793）。布尔字段统一使用 `argparse.BooleanOptionalAction`（L379），因此可以用 `--no-enable-prefix-caching` 关闭 APC（这是 argparse 的标准行为，属推断，但依据充分）。`vllm/config/speculative.py` 中 `method` 的取值包括 `"ngram"、"medusa"、"mlp_speculator"、"draft_model"、"suffix"、"custom_class"`，以及 EAGLE、ngram-GPU、DSpark 类型；字段有 `num_speculative_tokens`、`model`、`prompt_lookup_max`、`prompt_lookup_min`。示例：`--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4}'`（JSON 写法依据字段名组合，属推断）。— [arg_utils.py@v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/engine/arg_utils.py)、[speculative.py@v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/speculative.py)

### Gaps
- EAGLE 系列 method 的具体字符串（`EagleModelTypes` 展开后的值）未展开核对。

---

## 4. SGLang 现状：版本、架构、RadixAttention、overlap 调度、router/gateway、PD、参数、bench

### Takeaway
最新版 **SGLang 0.5.21**（PyPI 上传于 2026-10-01，GitHub release 发布于 2026-10-02 UTC），**仅支持 CUDA 13**。进程结构为 HTTP server + TokenizerManager（主进程）→ Scheduler 子进程（每 TP rank 一个，内含 TpModelWorker → ModelRunner）→ DetokenizerManager 子进程；DP 时还有 DataParallelController。默认开启 overlap scheduler 和 radix cache，默认调度策略 fcfs。路由层是 Rust 实现的 **SGLang Model Gateway**（Python 包仍叫 `sglang-router`，入口为 `python -m sglang_router.launch_router`）。

### Cited Findings
- PyPI `sglang` 0.5.21，上传于 2026-10-01T07:45Z。最近版本：0.5.20（09-18）、0.5.19（09-04）、0.5.18（08-21）。依赖：`torch==2.13.0`、`sglang-kernel==0.4.7`（替代旧的 sgl-kernel，后者在 PyPI 上停在 2026-01 的 0.3.21）、`flashinfer_python[cu13]==0.6.18`、`cuda-python>=13.0`。— [PyPI sglang](https://pypi.org/project/sglang/0.5.21/)、[PyPI sgl-kernel](https://pypi.org/project/sgl-kernel/)
- release v0.5.21（779 PRs / 227 contributors）：PD 实例可以在线切换 prefill/decode 角色；prefix cache 默认运行在 Rust core 上；新增 `/v1/decisions` 和 `/v1/score` API。升级命令 `uv pip install --prerelease=allow sglang==0.5.21`；Docker `lmsysorg/sglang:v0.5.21`（NVIDIA CUDA 13）。文档域名已变为 **docs.sglang.io**。— [SGLang v0.5.21 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.21)
- 安装文档：`pip install uv; uv pip install --prerelease=allow sglang`（uv<0.12.0 不加 `--prerelease=allow` 会静默装成 0.5.9）。文档原文："SGLang requires CUDA 13. The CUDA 12 (cu129) wheels and images are retired… SGLang 0.5.19 is the last release with a CUDA 12 lane." Docker 示例：`docker run --gpus all --shm-size 32g -p 30000:30000 -v ~/.cache/huggingface:/root/.cache/huggingface --env "HF_TOKEN=<secret>" --ipc=host lmsysorg/sglang:latest python3 -m sglang.launch_server --model-path … --host 0.0.0.0 --port 30000`。文档建议固定不可变 tag，因为 `latest` 是可变的。— [install.mdx@v0.5.21](https://github.com/sgl-project/sglang/blob/v0.5.21/docs/docs/get-started/install.mdx)
- 进程与类（v0.5.21 源码）：`entrypoints/engine.py` 用 `mp.Process` 启动 `run_scheduler_process_func`、`run_data_parallel_controller_process`、`run_detokenizer_process_func`；`managers/tokenizer_manager.py: TokenizerManager`；`managers/scheduler.py: Scheduler`（`event_loop_normal`、`event_loop_overlap`、`get_next_batch_to_run`、`get_new_batch_prefill`、`update_running_batch`、`run_batch`、`process_batch_result`）；`managers/tp_worker.py: TpModelWorker.forward_batch_generation`；`model_executor/model_runner.py: ModelRunner`；`managers/schedule_policy.py: SchedulePolicy / PrefillAdder / CacheAwarePolicy`；`managers/schedule_batch.py: Req / ScheduleBatch`；`model_executor/forward_batch_info.py`（ForwardBatch）；`managers/detokenizer_manager.py: DetokenizerManager`；`mem_cache/{radix_cache,memory_pool,chunk_cache,hicache_storage}.py`；`disaggregation/{prefill,decode}.py`；`speculative/eagle_worker_v2.py` 等。— [sglang git tree@v0.5.21](https://github.com/sgl-project/sglang/tree/v0.5.21/python/sglang/srt)
- Overlap（“zero-overhead”）调度的实现：`event_loop_overlap` 维护一个 `result_queue`。每轮先 `get_next_batch_to_run`，`run_batch(batch)` 发射当前 batch 后，**再**处理上一 batch 的结果（`pop_and_process`），从而让 CPU 的调度和后处理与 GPU 前向重叠；当 `is_disable_overlap_for_batch` 为真时，先处理上一 batch 再发射。`disable_overlap_schedule` 默认 False。— [scheduler.py@v0.5.21 L1906-2010](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/scheduler.py)、[schedule.py@v0.5.21](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/arg_groups/fields/schedule.py)（该设计最初见 LMSYS 的 SGLang v0.4 博客，本次未抓取，「未核实」）
- 关键 server args（v0.5.21 默认值，定义在 `python/sglang/srt/arg_groups/fields/*.py`）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--tp-size`(`--tensor-parallel-size`) / `--dp-size` / `--pp-size` / `--ep-size` | 1 | 并行度 |
| `--mem-fraction-static` | None（自动） | 权重+KV 池的静态显存比例 |
| `--max-running-requests` | None | 运行中请求上限 |
| `--max-total-tokens` | None | KV 池 token 数（调试用） |
| `--chunked-prefill-size` | None（自动；-1 表示关闭） | chunk 大小 |
| `--max-prefill-tokens` | 16384 | prefill batch token 上限 |
| `--schedule-policy` | `fcfs`（可选 lpm/random/dfs-weight/lof/priority/routing-key/hrrn/shortest-prefill-first） | |
| `--schedule-conservativeness` | 1.0 | |
| `--page-size` | None | |
| `--disable-radix-cache` | False | |
| `--radix-eviction-policy` | `lru` | |
| `--disable-overlap-schedule` | False | |
| `--enable-mixed-chunk` | False | prefill 与 decode 混批 |
| `--enable-dp-attention` | False | attention 用 DP，FFN 用 TP |
| `--load-balance-method` | `auto` | DP 负载均衡 |
| `--attention-backend` | None（自动） | |
| `--kv-cache-dtype` | `auto`（fp8_e5m2/fp8_e4m3/bf16/nvfp4 等） | |
| `--enable-torch-compile` | False | |
| `--enable-hierarchical-cache` | False | HiCache（host 内存层） |
| `--enable-metrics` | **False** | Prometheus 指标需显式开启 |
| `--disaggregation-mode` | `null`（prefill/decode） | |
| `--disaggregation-transfer-backend` | `mooncake` | |

  — [schedule.py](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/arg_groups/fields/schedule.py)、[memory.py](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/arg_groups/fields/memory.py)、[parallel.py](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/arg_groups/fields/parallel.py)、[exec_.py](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/arg_groups/fields/exec_.py)、[model.py](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/arg_groups/fields/model.py)、[observability.py](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/arg_groups/fields/observability.py)、[disagg.py](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/arg_groups/fields/disagg.py)
- Model Gateway：文档称其为 "high-performance model-routing gateway"，可路由到任意 OpenAI 兼容后端。安装方式为 `pip install sglang-router` 或 Docker `lmsysorg/sgl-model-gateway:latest`。启动示例：`python -m sglang_router.launch_router --worker-urls http://worker1:8000 http://worker2:8000 --policy cache_aware`（也可用 Rust 二进制 `sgl-model-gateway`）。和 worker 一起启动：`python -m sglang_router.launch_server --model … --dp-size 4`。源码在仓库 `sgl-model-gateway/`。注意 PyPI 上 `sglang-router` 的最新版是 0.3.2（2026-01-15），比主仓库旧，安装前请确认版本。— [sgl_model_gateway.mdx@v0.5.21](https://github.com/sgl-project/sglang/blob/v0.5.21/docs/docs/advanced_features/sgl_model_gateway.mdx)、[PyPI sglang-router](https://pypi.org/project/sglang-router/)
- PD 分离的最小示例（文档原文）：
  ```bash
  python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disaggregation-mode prefill --port 30000 --disaggregation-ib-device mlx5_roce0
  python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disaggregation-mode decode --port 30001 --base-gpu-id 1 --disaggregation-ib-device mlx5_roce0
  python -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 8000
  ```
  需要先 `uv pip install mooncake-transfer-engine`（NIXL 后端则用 `pip install nixl`）。— [pd_disaggregation.mdx@v0.5.21](https://github.com/sgl-project/sglang/blob/v0.5.21/docs/docs/advanced_features/pd_disaggregation.mdx)
- Benchmark：`python -m sglang.bench_serving` 在 v0.5.21 中已是**弃用路径**（会发出 FutureWarning），实现迁移到 `python -m sglang.benchmark.serving`。参数：`--backend`（默认 sglang，也可用 vllm 等）、`--dataset-name`（默认 **sharegpt**，可选 random/random-ids/generated-shared-prefix/mooncake/image/mmmu/longbench_v2/speed-bench/agentic-trace/custom/openai）、`--num-prompts`、`--random-input-len`、`--random-output-len`、`--random-range-ratio`、`--request-rate`（默认 inf）、`--max-concurrency`、`--warmup-requests`（默认 1）、`--disable-ignore-eos`（即**默认 ignore EOS**）、`--flush-cache`、`--gsp-*`（共享前缀数据集）、`--output-file`、`--pd-separated`。文档示例：`python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 --num-prompts 1000 --model meta-llama/Llama-3.1-8B-Instruct`。— [benchmark/serving.py@v0.5.21](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/benchmark/serving.py)、[bench_serving.py@v0.5.21](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/bench_serving.py)、[bench_serving.mdx@v0.5.21](https://github.com/sgl-project/sglang/blob/v0.5.21/docs/docs/developer_guide/bench_serving.mdx)
- SGLang 指标（需 `--enable-metrics`）：`sglang:time_to_first_token_seconds`、`sglang:time_per_output_token_seconds`、`sglang:e2e_request_latency_seconds`、`sglang:num_running_reqs`、`sglang:num_queue_reqs`、`sglang:token_usage`、`sglang:cache_hit_rate`、`sglang:gen_throughput`、`sglang:prompt_tokens_total`、`sglang:generation_tokens_total`、`sglang:spec_num_draft_tokens` 等。— [production_metrics.mdx@v0.5.21](https://github.com/sgl-project/sglang/blob/v0.5.21/docs/docs/references/production_metrics.mdx)

### Inferences
- 两个引擎的结构可以这样对照：SGLang Scheduler 进程 ≈ vLLM EngineCore + Worker 中 rank0 的组合（SGLang 的 scheduler 与 TpModelWorker 在同一进程，每个 TP rank 一个 scheduler 进程）。vLLM 则把调度（EngineCore）与执行（Worker）放在不同进程，通过 executor 广播。这个结论来自上面的进程启动代码和 arch_overview，属于推断，建议用 `ps -ef`/`py-spy dump` 实际观察确认。
- SGLang 的 `chunked_prefill_size` 和 `mem_fraction_static` 默认是 None，按 GPU 显存和模型自动推导。实验时应把推导结果从启动日志中记录下来。

### Gaps
- `chunked_prefill_size`、`mem_fraction_static`、`cuda_graph_max_bs` 在 H800 80GB 上的自动推导值未核实（需看启动日志）。
- LMSYS 博客（RadixAttention 2024-01、v0.4 zero-overhead scheduler 2024-12）本次未抓取。

---

## 5. 其他引擎与编排层（截至 2026-10-14 的状态）

### Takeaway
下表只给出能核实的版本信息；各项目的定位描述大多来自既有知识，本次未逐一抓取 README 核对，标为「未核实」，交给撰写者时请保留这一标记。

### Cited Findings

| 项目 | PyPI 最新（上传日期） | 定位（「未核实」部分已标注） |
|---|---|---|
| TensorRT-LLM (`tensorrt-llm`) | 稳定版 1.2.1（2026-04-20）；预发布 1.3.0rc29（09-29）、1.4.0rc0（10-13） | NVIDIA 官方引擎，in-flight batching，PyTorch backend，`trtllm-serve`（「未核实」：默认 backend 与 CLI 未抓取） |
| NVIDIA Dynamo (`ai-dynamo`) | 1.5.1（2026-10-07），1.6.0.dev 每日构建 | 数据中心级分布式推理编排（PD 分离、KV 感知路由，后端可选 vLLM/SGLang/TRT-LLM）（「未核实」） |
| NVIDIA AIPerf (`aiperf`) | 0.13.0（2026-09-24） | 压测工具；`genai-perf` 停在 0.0.16（2025-08-26），推断已被 AIPerf 取代 |
| GuideLLM (`guidellm`) | 0.8.1（2026-10-08） | vllm-project 下的 SLO 导向压测工具（rate sweep）（「未核实」：CLI 未抓取） |
| LMDeploy (`lmdeploy`) | 0.18.0（2026-09-28） | 上海 AI Lab，TurboMind/PyTorch 双引擎（「未核实」） |
| Ray (`ray`，含 Ray Serve LLM) | 2.59.0（2026-10-02） | 基于 Ray 的多副本、多模型编排，底层用 vLLM（「未核实」） |
| HF TGI | 客户端包 `text-generation` 停在 0.7.0（2024-03） | 服务端维护状态「未核实」：本次没有抓取 README，不能断言是否已进入 maintenance mode |
| llama.cpp | （本次未查询） | 面向 CPU/端侧与 GGUF 量化的单机推理，不是数据中心多卡 serving 的首选（「未核实」） |
| llm-d | （本次未查询） | 基于 K8s 和 vLLM 的分布式推理栈（Gateway API inference extension、PD）（「未核实」） |

— [PyPI tensorrt-llm](https://pypi.org/project/tensorrt-llm/)、[PyPI ai-dynamo](https://pypi.org/project/ai-dynamo/)、[PyPI aiperf](https://pypi.org/project/aiperf/)、[PyPI genai-perf](https://pypi.org/project/genai-perf/)、[PyPI guidellm](https://pypi.org/project/guidellm/)、[PyPI lmdeploy](https://pypi.org/project/lmdeploy/)、[PyPI ray](https://pypi.org/project/ray/)、[PyPI text-generation](https://pypi.org/project/text-generation/)

### Inferences
- 7 天的学习应以 vLLM 与 SGLang 源码为主线。Dynamo、llm-d 和 Ray Serve 属于“多实例编排层”，解决的是路由、PD、扩缩容和 KV 感知调度；单机 4×H800 上用 SGLang router 或 vLLM DP 就能体验同类概念。

### Gaps
- TGI 的维护状态、TRT-LLM 1.x 默认 backend、llm-d 与 llama.cpp 的最新版本都未核实（受时间预算限制没有抓取）。

---

## 6. 指标定义与 benchmark 方法学

### Takeaway
需要同时报告延迟分布（TTFT、TPOT/ITL、E2E 的 P50/P90/P99）和吞吐（req/s、output tok/s）。在 SLO 约束下比较 **goodput**。负载用 **open-loop request-rate 扫描**（模拟真实到达）或 **closed-loop concurrency 扫描**（固定并发），两者不能混着比较。

### Cited Findings
- 定义（与 vLLM bench 的实现一致）：TTFT = 请求发出到第一个 token；ITL = 相邻 token 的间隔（流式）；TPOT = 每请求 (E2E − TTFT)/(输出 token 数 − 1)；E2EL = 端到端延迟。`--percentile-metrics` 可选 ttft/tpot/itl/e2el；`--goodput ttft:… tpot:… e2el:…` 的定义引用 DistServe（arXiv 2401.09670）。— [vllm/benchmarks/serve.py@v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/benchmarks/serve.py)、[DistServe](https://arxiv.org/abs/2401.09670)（TPOT 的公式属于常规定义，与源码的逐行对应未展开核对）
- Open vs closed loop：`--request-rate` 为有限值时按 Poisson 到达（`--burstiness` 改成 gamma 分布），为 inf 时所有请求一次性发出；`--max-concurrency` 限制同时在途的请求数，两者叠加时实际速率可能低于设定值。— [serve.py@v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/benchmarks/serve.py)
- Warmup 与 EOS：vLLM `--num-warmups` 默认 0、`--ignore-eos` 默认关闭；SGLang `--warmup-requests` 默认 1、默认 ignore EOS。— 同上，[sglang benchmark/serving.py@v0.5.21](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/benchmark/serving.py)
- 数据集：vLLM 有 random（默认）、sharegpt（需 `--dataset-path` 指向 ShareGPT json）、prefix_repetition、burstgpt、spec_bench、timed_trace；SGLang 有 sharegpt（默认）、random、generated-shared-prefix、mooncake trace 等。— 同上
- 工具：vllm bench、sglang.benchmark.serving、GuideLLM 0.8.1、AIPerf 0.13.0（版本见第 5 节）。

### Inferences
- 方法学检查清单：①固定模型、dtype、TP、max_model_len，并记录 git tag 与 GPU 时钟；②预热（vLLM 侧加 `--num-warmups`）；③输出长度可控（双方都 ignore EOS，或都不 ignore）；④每个负载点至少跑 1–2 分钟，或请求数 ≥ 10×并发；⑤同时抓服务端 `/metrics`，用 queue time 与 kv usage 解释客户端看到的尾延迟；⑥测 prefix cache 时，两次运行之间要清缓存或换随机种子。

### Gaps
- GuideLLM 与 AIPerf 的 CLI 语法本次未核实，实验中不使用。

---

## 7. 4×H800 动手实验（只用上文核实过的语法；未核实的部分已标出）

### Takeaway
以下 4 个实验总计约 1–1.5 个 GPU·天以内，都用一个 7–8B 级别的稠密模型（如 `Qwen/Qwen3-8B`，选型本身不是版本敏感事实）。先在 vLLM 0.31.0 / SGLang 0.5.21 上确认环境（驱动需支持 CUDA 13；vLLM 也可使用 `-cu129` 镜像）。

### Cited Findings
（命令语法来源见第 3、4 节对应文件；下文不再重复链接）

**实验 A：continuous batching 与 token budget（vLLM，1 GPU，约 1.5 h）**
- 目标：观察并发增大时吞吐上升、TPOT/ITL 变差的曲线，以及 `max_num_batched_tokens` 对 TTFT 和 ITL 的取舍（chunked prefill）。
- 步骤：
  ```bash
  CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B --port 8000      # 默认 serve 预算 8192 tokens（H800 属于 ≥70GiB 分支）
  for c in 1 8 32 128 512; do
    vllm bench serve --model Qwen/Qwen3-8B --base-url http://127.0.0.1:8000 \
      --dataset-name random --random-input-len 1024 --random-output-len 256 \
      --num-prompts $((c*10)) --max-concurrency $c --ignore-eos --num-warmups 10 \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
      --save-result --result-dir ./expA
  done
  curl -s http://127.0.0.1:8000/metrics | grep -E "vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions|iteration_tokens_total)"
  ```
  然后把 `--max-num-batched-tokens` 分别设为 2048 和 16384 重启（flag 已在 arg_utils.py L1668 核实），用 `--random-input-len 8192` 的长 prompt 混合负载复测。
- 预期：并发 1→128 时 output tok/s 大幅上升，TPOT P50 缓慢增加；预算小时长 prompt 的 TTFT 变大但 ITL P99 更稳，预算大时相反。
- 验收：画出“并发–吞吐–TPOT”曲线，并用 `vllm:num_requests_waiting` 和 `kv_cache_usage_perc` 解释拐点。

**实验 B：prefix caching 的收益（vLLM APC vs SGLang RadixAttention，2 GPU 并行，约 2 h）**
- 步骤（vLLM，GPU0）：
  ```bash
  vllm bench serve --model Qwen/Qwen3-8B --base-url http://127.0.0.1:8000 \
    --dataset-name prefix_repetition --prefix-repetition-prefix-len 2048 \
    --prefix-repetition-suffix-len 128 --prefix-repetition-num-prefixes 8 \
    --prefix-repetition-output-len 128 --num-prompts 800 --max-concurrency 64 --ignore-eos
  ```
  对照组用 `vllm serve Qwen/Qwen3-8B --no-enable-prefix-caching` 重启（依据 BooleanOptionalAction 推断，启动日志中应能看到 prefix caching 已关闭），记录 `vllm:prefix_cache_hits / vllm:prefix_cache_queries`。
- 步骤（SGLang，GPU1）：
  ```bash
  CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000 --enable-metrics
  python -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 --port 30000 \
    --dataset-name generated-shared-prefix --gsp-num-groups 8 --gsp-prompts-per-group 100 \
    --gsp-system-prompt-len 2048 --gsp-question-len 128 --gsp-output-len 128 --max-concurrency 64
  ```
  对照组加 `--disable-radix-cache` 重启，记录 `sglang:cache_hit_rate`。
- 预期：开启缓存后 TTFT P50 显著下降（共享 2048 token 前缀，prefill 量可减少一个数量级），命中率接近 (前缀长)/(前缀长+后缀长)。
- 验收：两引擎各自给出开/关缓存的 TTFT、吞吐和命中率表。再阅读 `block_pool.py`（LRU free queue）与 `radix_cache.py`（`match_prefix/evict`），写出各自数据结构的差异。

**实验 C：引擎对比与并行策略（4 GPU，约 3 h）**
- 步骤：vLLM `vllm serve Qwen/Qwen3-8B -tp=4` 对比 `vllm serve Qwen/Qwen3-8B -dp=4`（DP 时会出现 4 API server + 4 engine core + 1 coordinator，用 `ps -ef | grep -i vllm` 数进程，验证 arch_overview 的进程数公式）；SGLang `--tp-size 4` 对比 `python -m sglang_router.launch_server --model Qwen/Qwen3-8B --dp-size 4 --host 0.0.0.0 --port 30000`。负载用 open-loop：`vllm bench serve … --request-rate {2,4,8,16,32} --goodput ttft:500 tpot:50 --ignore-eos`（两边都用 vllm bench 打，`--backend openai`，确保客户端一致）。
- 预期：8B 模型单卡就放得下，因此高负载下 DP=4 的吞吐和 goodput 通常优于 TP=4；TP=4 的单请求 TPOT 更低。H800 的 NVLink 带宽低于 H100（「未核实」：H800 NVLink 规格没有找到 NVIDIA 一手页面），会放大 TP 的通信开销。
- 验收：一张 rate–goodput 曲线图（4 条线）和进程数截图。

**实验 D（可选，取决于网卡）：SGLang PD 分离（2 GPU，约 2–3 h）**
- 步骤：按第 4 节的文档命令，prefill 在 GPU0（`--port 30000`）、decode 在 GPU1（`--base-gpu-id 1 --port 30001`），router 在 8000 端口；`--disaggregation-ib-device` 需填写本机 RDMA 设备名（用 `ibv_devices` 查看）。没有 RDMA 网卡时 Mooncake/NIXL 能否走纯 NVLink/TCP「未核实」，可能无法完成。
- 对比：同为 2 GPU 时，与 `--dp-size 2` 的统一引擎在“长 prompt（8k）+ 短输出”和“短 prompt + 长输出”两种负载下比较 TTFT P99 与 TPOT P99。
- 预期：PD 分离消除了 prefill 对 decode 的打断，ITL/TPOT 尾延迟更稳定，但 TTFT 会多出 KV 传输时间；负载不均衡时 PD 的资源利用率反而更低。
- 验收：写出何时选 PD、何时选 chunked prefill 的结论，并引用 DistServe 的 goodput 论证。

### Inferences
- 每个实验都要配一个“读代码”任务：A 读 `scheduler.py::schedule()`，B 读 `block_pool.py` / `radix_cache.py`，C 读 `core.py`、`coordinator.py`、`engine.py::_launch_subprocesses`，D 读 `disaggregation/{prefill,decode}.py` 与 vLLM `kv_connector/v1/nixl/`。

### Gaps
- H800 的硬件规格（HBM 容量与带宽、NVLink 带宽）没有找到 NVIDIA 一手规格页，「未核实」。
- 实验 D 在无 RDMA 环境下的可行性「证据不足」。spec decode 实验因 EAGLE 草稿模型名称与 method 字符串未核实，没有列为正式实验；如需加做，可用第 3 节中已核实的 ngram method 字段。
