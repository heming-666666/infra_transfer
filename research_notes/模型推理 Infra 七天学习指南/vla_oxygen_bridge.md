# OxyGen 与 openpi π0 / π0.5 推理路径：从 VLA 复现到通用推理 Infra 的桥梁

> 检索与访问日期：2026-10-14。代码事实绑定到具体 commit：OxyGen `1e8890681090c7f423f85866b5db877916ba65e6`（2026-10-10，"Add optional model extensions for Xiaomi, StarVLA and Qwen3.5"）；openpi `215abfb217dbac7d5f1273282331b9b1866c0479`（2026-08-24，访问时 main 的 HEAD）。论文以 arXiv 2603.14371 **v3（2026-10-13）** 为准。
> 标注约定：「未核实」= 本次没有找到一手来源；「推断」= 根据已引用事实做的计算或推理，不是原文结论。
> 本地线索文件 `D:\Code\research_notes\AI Infra 论文分布与 VLA 推理选题\vla_inference_landscape.md` 只用来找方向。凡是本次没有重新打开一手来源的条目，都会明确标注。

---

## Q1. OxyGen：问题、方法、实现、结果、运行方式、局限与接收状态

### Takeaway
OxyGen 解决的问题是：MoT（Mixture-of-Transformers）VLA 在同一帧观测上同时跑"动作 + 语言"两个任务时会重复 prefill，两个任务还会争抢 GPU。它的做法是把 KV cache 当作跨任务、跨帧共享的一等资源，提供两个优化：
- **cross-task KV sharing**：一次 prefill，同时供动作专家和语言专家使用。
- **cross-frame continuous batching**：把各帧尚未完成的语言请求放进同一个 batch，每帧只推进 k 个 token。

在 π0.5 上，OxyGen 相对 isolated execution 最高提速 3.7×；在 RTX 4090 上可以同时做到 >200 tokens/s 的语言吞吐和 70 Hz 的动作频率。v3 又加入了 Xiaomi-Robotics-0、StarVLA 和 Qwen3.5 0.8B–9B 的扩展实验。

论文 v3 和 README 都写明已被 NeurIPS 2026 接收，README 还写明是 Main Track poster。但这两处都是作者自述，我没能在 neurips.cc 或 OpenReview 上独立查到。

### Cited Findings

**身份与版本**
- arXiv 2603.14371 有三个版本：v1 2026-03-15，v2 2026-05-18，v3 2026-10-13。v3 的 Comments 字段为 "NeurIPS 2026"，分类为 cs.RO 与 cs.AI。作者为 Xiangyu Li、Huaizhi Tang、Xin Ding、Weijun Wang、Ting Cao（通讯作者）、Yunxin Liu，单位是清华 AIR、清华电子系和中国科大 — [arXiv abs](https://arxiv.org/abs/2603.14371)；[arXiv HTML v3](https://arxiv.org/html/2603.14371v3)
- v3 HTML 页眉原文："Accepted at the 40th Conference on Neural Information Processing Systems (NeurIPS 2026)" — [arXiv HTML v3](https://arxiv.org/html/2603.14371v3)
- README 的 News 写道："2026-09: Paper accepted to NeurIPS 2026 Main Track as poster. See you in Sydney!" 但 README 里的 BibTeX 仍是 `arXiv preprint` — [OxyGen README @1e88906](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/README.md)
- 独立核实失败：
  - OpenReview API 全文搜索 "OxyGen Unified KV Cache" 没有返回该论文。
  - Web 搜索没有找到 neurips.cc/virtual/2026 上的 OxyGen 页面。同一次搜索能找到其他 NeurIPS 2026 Sydney poster 页面，说明这些页面已经开始上线。
  - 结论：接收状态 =「作者自述（arXiv v3 + README），官方列表未核实」 — [OpenReview API 查询](https://api2.openreview.net/notes/search?term=OxyGen%20Unified%20KV%20Cache&limit=5)

**问题设定**
- 具身智能体需要从同一份观测出发，并行执行操控、对话、记忆构建等多个任务，而且各任务的时间约束不同。MoT VLA 用不同专家输出不同模态，但现有系统采用 isolated execution：每个任务各跑一次同一模型的前向。论文把根因归结为"孤立的 KV cache 管理"，后果有两个：冗余 prefill，以及资源争用 — [arXiv abs](https://arxiv.org/abs/2603.14371)；[v3 HTML](https://arxiv.org/html/2603.14371v3)
- v3 把这种设定称为 "Multi-task parallelism with asymmetric deadlines"：
  - 动作生成必须满足**逐帧截止期**，才能维持目标动作频率 f_min；
  - 语言生成可以跨越多帧，受 **soft deadline** 约束；
  - 优化目标是"在满足动作截止期的前提下最大化语言吞吐"。
  - 原文用的是 per-frame deadline 与 soft deadline，没有直接写 "hard deadline"。本地线索中"硬截止期"的说法来自 v1，本次没有复核 v1。 — [v3 HTML](https://arxiv.org/html/2603.14371v3)
- 论文明确区分了 OxyGen 与 LLM serving 的已有技术：vLLM 和 SGLang 的 prefix caching、continuous batching 已经能避免共享前缀的重复计算，但"Simply enabling prefix caching and continuous batching in an LLM server does not provide this coordination"。论文强调这里的特殊之处：同一份 prefill 结果要同时服务两个专家——一个是在 denoising 中反复读取前缀的 flow-matching 专家，一个是维护自身增长 cache 的自回归专家；同时语言解码还必须放进周期性动作推理的时间预算内 — [v3 HTML](https://arxiv.org/html/2603.14371v3)

**方法**
- 统一 KV 管理器 ℳ 维护每个请求的可恢复状态 σ_t。其中 y_t 是已生成的 token；δ_t∈{0,1} 表示是否已终止（输出 EOS 或达到最大长度 N）。ℳ 提供 Store / Retrieve / Update / Remove 四个 CRUD 操作，并维护活跃请求集合 ℛ — [v3 HTML](https://arxiv.org/html/2603.14371v3)
- 每帧分两个阶段：
  1. 对新观测 o_t 做一次 prefill，得到 𝒦_t。这份 cache 同时分发给动作专家（做 S 步 denoising 得到动作块 A_t）和一个新建的语言状态 σ_t（存放在请求 ID r_t 下）。
  2. r_t 加入 ℛ，ℳ 把全部 m 个活跃状态取出并拼成一个 batch，执行 k 步批量自回归解码。已完成的请求被逐出，其余请求连同新增 token 保存到下一帧。
  — [v3 HTML](https://arxiv.org/html/2603.14371v3)
- 解码预算 k 决定每个语言请求每帧前进多少。论文原话："For simplicity, our current implementation uses a fixed k"；部署时可以通过 profiling 目标硬件来选 k。论文还指出，可恢复状态接口不要求一次前向只出一个 token，也就是说它可以与投机解码之类的方法组合 — [v3 HTML](https://arxiv.org/html/2603.14371v3)
- 指标定义：
  - action frequency = H / T，即动作块长度除以平均端到端帧时延。
  - π0.5 默认 H=10、S=10。
  - 语言吞吐 = 所有活跃请求每秒生成的 token 总数。
  - 注意：这里的 action frequency 是吞吐型指标，不是反应时延。
  — [v3 HTML](https://arxiv.org/html/2603.14371v3)
- 由于 openpi 没有发布 π0.5 论文中的语言生成代码，OxyGen 的语言解码基线按社区复现 `openpi_subtask_generation` 实现 — [v3 HTML](https://arxiv.org/html/2603.14371v3)；[README](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/README.md)

**实现（代码位置，均在 commit 1e88906）**
- 基于 openpi。仓库的第一个提交是 "From openpi — Squashed commits"（2026-03-14），许可证为 Apache-2.0。README 写明的演进如下：
  - 2026-03：JAX 后端 + 统一 KV 管理；
  - 2026-05：PyTorch 推理 + `torch.compile`（在 A100 与 Jetson AGX Thor 上验证）；
  - 2026-08：suffix-only LoRA 与 LIBERO 文本记忆示例；
  - 2026-10：Xiaomi-Robotics-0 / StarVLA 扩展。
  — [README](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/README.md)；[GitHub commits API](https://api.github.com/repos/air-embodied-brain/OxyGen/commits?per_page=8)
- `src/openpi/models/kv_cache_manager.py`：
  - `ContinuousBatchManager` 用 `active_states: dict[request_id → IncrementalTextState]` 管理状态，提供 `store_state / get_state / remove_state / get_status / get_all_active_requests / cancel_request / clear_all`。
  - 文件里还保留了已废弃的 `KVCacheState`，以及按最大 cache 长度做 `jnp.pad` 再 `jnp.stack` 的 legacy `prepare_batch`。
  — [kv_cache_manager.py](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/src/openpi/models/kv_cache_manager.py)
- `src/openpi/models/pi05.py`：
  - `IncrementalTextState` 被注册为 JAX pytree。
    - data 字段：`rng, last_logits, output_tokens, kv_cache, current_step[B], is_finished[B], prefill_len[B]`。
    - meta（静态）字段：`prefill_size, suffix_offset, max_decoding_steps, cache_size(=prefill_size+max_decoding_steps)`。
  - `prefill(..., max_decoding_steps)` 在 attention mask 末尾预留解码槽位。KV 布局是不连续的：有效前缀位于 `[0, prefill_len)`，解码 token 从 `prefill_size` 开始顺序写入，中间是 padding。因此必须用 `_gather_last_valid_token` 取种子 embedding。
  - 其他关键函数：`sample_actions_with_kv`（用共享 prefill 做 denoising）、`init_incremental_state`、`generate_n_tokens`（每个元素有独立的 current_step，所以能批处理处于不同进度的请求）、`sample_text_actions_shared_kv`、`sample_text_actions_dependent`（π0.5 论文式的"先文本后动作"串行版本，断言 batch=1），以及 `profile_sample_text / profile_sample_actions`。
  — [pi05.py](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/src/openpi/models/pi05.py)
- `src/openpi/policies/policy.py::infer_text_actions_continuous_batch` 是 OxyGen 的主路径（PyTorch 对应 `_infer_text_actions_continuous_batch_pytorch`），流程如下：
  1. 区分新请求和恢复请求；
  2. 新请求做**一次** `_prefill(max_decoding_steps=...)`，复用于 `_init_incremental_state` 和 `_sample_actions_with_kv`；
  3. 恢复请求默认不再生成动作；
  4. `_stack_incremental_states` 把所有状态拼成一个 batch，调用一次 `_generate_n_tokens(tokens_to_generate=steps_per_frame)`；
  5. `block_until_ready` 后拆分状态，未完成的回写 manager，已完成的移除。
  - 计时字段有 `pre_proc_ms / prefill_actions_ms / text_gen_ms / post_proc_ms / total_ms / batch_size / new_requests / resumed_requests`。
  — [policy.py](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/src/openpi/policies/policy.py)
- 调度逻辑不是独立模块，而是逐帧循环。`experiments/continuous_batching/runner.py` 按到达模式（默认 `uniform_arrivals(rate=1)`）生成新请求，与活跃请求一起调用上面的方法，并记录 `frame_ms`。另外三种设置的对照 runner 位于 `experiments/{baseline,shared_kv,parallel_mps}/runner.py` — [runner.py](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/experiments/continuous_batching/runner.py)；[Experiments.md](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/experiments/Experiments.md)
- 四种对照设置：
  - `baseline`：顺序执行 `infer_actions()` + `infer_text()`；
  - `shared_kv`：一次 prefill 供两头使用；
  - `continuous_batching`：OxyGen 完整路径；
  - `parallel_mps`：两个 worker 在 NVIDIA MPS 下并行。
  - PyTorch 版 continuous batching 使用**固定大小的文本 KV cache**，用一次 batched decode 推进所有活跃请求。
  — [Experiments.md](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/experiments/Experiments.md)
- OxyGen 自定义的模型配置（`src/openpi/training/config.py`）：

  | 配置 | action_horizon | max_token_len | 其他 |
  |---|---|---|---|
  | `pi05_o2_arx` | 20 | 100 | — |
  | `pi05_o2_droid` | 10 | 100 | — |
  | `pi05_o2_aloha` | 10 | 100 | — |
  | `pi05_o2_libero` | 10 | 100 | `discrete_state_input=False` |

  — [config.py @1e88906](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/src/openpi/training/config.py)
- 模型扩展（`model_extensions/`，每个框架各用一个 Python 3.12 环境，要求 CUDA 12.8 驱动，建议 24 GB 显存）：
  - Xiaomi-Robotics-0：一个语言专家 + 一个 flow-matching 动作专家，实现在 `common/cache.py` 和 `xiaomi/benchmark.py`；
  - StarVLA Qwen3VL-PI_v3：可在同一 Qwen3-VL 骨干上加挂 GR00T 与 OFT 动作专家，最多 4 个专家，实现在 `starvla/experts.py` 和 `starvla/runtime.py`；
  - Qwen3.5 0.8B–9B：Qwen3.5 是 full attention + linear attention 的混合结构，runtime 同时批处理 KV 和 conv/recurrent state。
  - 默认使用合成输入和随机参数。
  — [model_extensions/README.md](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/model_extensions/README.md)

**运行命令（README / Experiments.md @1e88906）**
- 安装：`GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .`。环境要求 Ubuntu 22.04+、Python 3.11、显存 >8 GB（完整实验建议 >16 GB）。JAX 后端的测试环境是 RTX 4090 + 驱动 580.126.09 + CUDA 13.0 — [README](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/README.md)
- 快速测试：`create_policy("pi05_o2_arx")` → `create_synthetic_observation(...)` → `policy.infer_text(obs, max_decoding_steps=20)`。首次运行会把 `pi05_base` 下载到 `~/.cache/openpi` — [README](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/README.md)
- 完整 JAX 实验：
  ```
  uv run python -m experiments.run_experiments --settings baseline shared_kv continuous_batching parallel_mps \
      --policies pi05_o2_aloha pi05_o2_droid pi05_o2_libero \
      --checkpoint-dir /tmp/oxygen_random_init_checkpoint \
      --results-dir experiments/results_4090_jax --gpu 0
  ```
  另有 `experiments.run_overhead`、`experiments.run_workload_sweep`，以及 `experiments.analysis.{aggregate_metrics,compute_speedup_cb,compute_speedup_ablation,compute_workload_sweep,plot_all}`。
  - 传入一个不存在的 checkpoint 路径即可使用随机初始化做时延测试（输出无意义，但张量形状和调度路径是真实的）。
  - 结果写入 `{results_dir}/{setting}/{policy}/denoise{N}_decode{M}.json`。
  — [Experiments.md](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/experiments/Experiments.md)
- PyTorch：
  - `OPENPI_TORCH_COMPILE=1 ... --pytorch-device cuda:0`。checkpoint 目录中有 `model.safetensors`，或者传了 `--pytorch-device`，都会选择 PyTorch。
  - 子开关 `OPENPI_TORCH_COMPILE_{DENOISE,PREFILL,TEXT_DECODE}` 默认都是 1。`OPENPI_TORCH_COMPILE_DENOISE_CUDAGRAPHS` 默认 0，打开后 denoise 使用 `reduce-overhead` 模式。
  - 在 Thor 上编译后的单 token 解码曾产生非法采样概率，所以稳定模式会设置 `OPENPI_TORCH_COMPILE_TEXT_DECODE=0`。
  - JAX→PyTorch 转换：`scripts/convert_jax_model_to_pytorch.py`。
  — [Experiments.md](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/experiments/Experiments.md)
- 主要 CLI 参数：
  - `--num-denoise-steps`（S）、`--max-decoding-steps`（N）、`--steps-per-frame`（k）、`--total-frames`（默认 50）、`--arrival-pattern`、`--num-measured-runs`（默认 3）、`--warmup-runs`（默认 1）。
  - `N % k ≠ 0` 的组合会被跳过。
  - 分析脚本给出的指标：`frame_latency_ms`、`action_frequency_hz = ACTION_HORIZON / frame_latency`、`language_throughput_tps`、`avg_batch_size`、`avg_request_wall_ms`。
  — [Experiments.md](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/experiments/Experiments.md)

**结果（v3）**
- 硬件：RTX 4090 和 Jetson AGX Thor。最高 3.7× 加速，RTX 4090 上同时做到 >200 tokens/s 和 70 Hz — [arXiv abs](https://arxiv.org/abs/2603.14371)
- 消融（RTX 4090，LIBERO，S=10）：
  - 仅开 KV sharing 时，N=5 的帧时延从 200.3 ms 降到 145.1 ms（1.4×）。
  - 在此基础上开 batching，N=30、k=5 时帧时延从 469.3 ms 降到 175.8 ms（再快 2.7×），动作频率从 21.3 Hz 升到 56.9 Hz。
  - 随着语言输出变长，开启 batching 后动作频率基本保持不变：RTX 4090 约 60 Hz，Thor 约 27 Hz。
  — [v3 HTML](https://arxiv.org/html/2603.14371v3)
- 加速比主要受 k 影响（N 越大、k 越小，加速越多）；配置种类（LIBERO/ALOHA/DROID）和 S 的影响较小。MPS 并行在 4090 上收益有限，在 Thor 上反而变慢，原因是两个进程仍会重复计算前缀并争抢 GPU — [v3 HTML](https://arxiv.org/html/2603.14371v3)
- 时延分解（Table 6，π0.5-LIBERO，S=10）：一次 prefill 约 44.0 ms，是动作 denoising（20.4 ms）的两倍多。"Ours w/o Batching" 在 N=1 时把帧时延降低 36.4%，N=5 时降低 25.9%；绝对节省约 47–48 ms，N=30 时相对贡献降到 9.4%。（原句没有写硬件，同一节其他句子写的是 RTX 4090，推断为 4090。） — [v3 HTML](https://arxiv.org/html/2603.14371v3)
- 内存与能耗（Table 3，RTX 4090）：

  | 方法 | 峰值显存 (GiB) | 平均功率 (W) | 每请求能耗 (J) | 能耗变化 |
  |---|---|---|---|---|
  | Baseline | 6.43 | 293.5 | 117.4 | — |
  | Parallel | 12.49 | 324.4 | 120.9 | +3.0% |
  | Ours w/o Batching | 6.43 | 287.9 | 97.7 | −16.7% |
  | Ours (batch 2) | 7.35 | 173.3 | 39.1 | −66.7% |
  | Ours (batch 4) | 7.41 | 154.4 | 25.8 | −78.0% |

  列名是我依据正文 "Memory (GiB), Power (W), and Energy/Request (J)" 对应的。正文结论：显存开销 +15%，功率最多 −47%，能耗最多 −78%；原因是 batched decoding 把 VLM 权重读取摊薄到多个语言请求上。 — [v3 HTML](https://arxiv.org/html/2603.14371v3)
- 动作质量：Table 2 用 10 个种子评估 LIBERO 成功率。其中 LIBERO-10 一行为 "92.40 92.86 [92.40, 93.32]"，我推测是"openpi 官方值 / OxyGen 均值 [置信区间]"，列含义未核实 — [v3 HTML](https://arxiv.org/html/2603.14371v3)
- 跨模型（RTX 4090，N=30，k=5）：
  - Xiaomi-Robotics-0（S=5）加速 3.23×，Qwen3VL-PI_v3（S=4）加速 3.60×。
  - StarVLA 增加到 3 / 4 个专家时加速 3.53× / 3.69×。在 N=k=5（没有跨帧请求）时，2/3/4 个专家的加速分别为 1.23× / 1.40× / 1.59×，每增加一个专家，prefix reuse 每帧约省 25 ms。
  - Qwen3.5 骨干从 0.8B 到 9B，加速 3.58×–3.79×，基线帧时延从 429.0 ms 增加到 731.9 ms。
  - TP=1 时加速 3.50×，TP=2 时 3.71×；但开 TP=2 后帧时延反而从 493.4 ms 升到 535.3 ms，即两块 GPU 没有降低时延。（这一句对应的具体模型未提取。）
  — [v3 HTML](https://arxiv.org/html/2603.14371v3)
- 人形机器人实测（板载 Jetson AGX Thor，3 路 224×224 RGB，H=10，S=10，N=30，k=5）：
  - isolated 1030 ms = 动作路径 207.5 ms + 语言 822.3 ms；
  - OxyGen 393 ms = 198.0 ms + 195.4 ms。
  — [v3 HTML](https://arxiv.org/html/2603.14371v3)
- 负载扫描：在 uniform / Poisson 到达和长短请求混合下，到达率越高，batch 越大，语言吞吐越高。附录 D.3 的稳定性实验记录到 GPU 温度 87–90 °C，可能影响持续时延 — [v3 HTML](https://arxiv.org/html/2603.14371v3)

**自述局限（v3 原文）**
- 评估只覆盖"并发的动作 + 语言生成"。以下方面仍需验证：更多输出模态（如视频）的统一模型；与异步控制和模型级优化的结合；覆盖数据采集、多任务训练与闭环部署的端到端真实应用。另外 k 是固定值，没有自适应 — [v3 HTML](https://arxiv.org/html/2603.14371v3)

### Inferences
- 在 OxyGen 主路径里，"恢复请求"默认不重新生成动作。每帧只有一个新请求用当前观测做 prefill 并产出动作；旧语言请求继续基于它所属那一帧的旧观测解码。所以语言输出天然带有"陈旧观测"语义，适合记忆、旁白这类任务，不适合用来做动作条件。
- 这与 π0.5 论文的分层推理（先预测子任务文本，再以文本为条件出动作）不同。后者的语言解码处在动作关键路径上，对应的实现是 `sample_text_actions_dependent`，只支持 batch=1。
- OxyGen 的 KV 管理属于"静态预留 + 每帧 stack/split"：每个请求预留 `prefill_size+N` 个槽位，不分页。它的 batching 是帧粒度的 iteration-level scheduling，每帧 k 步，新请求只在帧边界加入。跨帧状态拼接本身有拷贝开销（论文 Table 7 量化了这一开销，具体数值本次未提取）。
- 前缀 KV 的计算（推断）：π0/π0.5 的 Gemma-2B 是 MQA（num_kv_heads=1，见 Q2），每个 token 的 KV 只有 2×18×1×256×2 B = 18,432 B。所以 KV 显存不是 OxyGen 的瓶颈；Table 3 中显存仅 +15% 与这一点一致。瓶颈在 prefill 计算和解码时的权重带宽。

### Gaps
- NeurIPS 2026 接收没有在官方列表或 OpenReview 上独立核实。v3 页眉只写 "Accepted"，没有写 poster；"poster" 只出现在 README 中。
- Table 7（KV 管理开销占帧时延的比例，JAX 与 PyTorch 对比）的具体数值、Table 2 的完整成功率、TP 实验对应的模型，本次均未提取。
- OxyGen 没有在 H800/Hopper 上报告结果（只有 RTX 4090、Thor 和 A100 PyTorch 验证）。H800 上的行为未核实。

---

## Q2. openpi π0 / π0.5 一次推理的完整路径（以及 π0-FAST 对照）

### Takeaway
openpi 一次推理的流程是：
1. 输入变换：图像 resize-with-pad 到 224×224 并归一化到 [-1,1]；状态做归一化（π0.5 默认还会离散成 256 个 bin 并写进文本 prompt）；prompt 用 PaliGemma SentencePiece 分词，π0.5 的 max_token_len=200，π0 为 48。
2. **前缀 prefill**：SigLIP So400m/14 把每张图编码为 256 个 token，与文本 token 拼接后，在 Gemma-2B 中以双向注意力做一次前向，生成 KV cache。
3. **flow matching**：默认 10 步 Euler 积分。每一步只把 H 个动作 token 送进约 300M 参数的 action expert，读取缓存的前缀 KV，不追加 KV。
4. 输出变换：反归一化。

upstream openpi 的 π0.5 只支持 flow-matching 头，不做语言解码。π0-FAST 则是纯自回归解码 FAST token，在 4090 上每个 chunk 约 750 ms，而 π0 约 100 ms。

### Cited Findings

**模型结构**
- PaliGemma = SigLIP-So400m 视觉编码器 + Gemma-2B 语言模型，总计 3B — [PaliGemma arXiv 2407.07726](https://arxiv.org/abs/2407.07726)
- openpi 代码中的 SigLIP 配置为 `variant="So400m/14", pool_type="none"`（保留全部 patch token） — [pi0.py @215abfb](https://raw.githubusercontent.com/Physical-Intelligence/openpi/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py)
- Gemma 两个变体的配置：

  | 变体 | width | depth | mlp_dim | num_heads | num_kv_heads | head_dim |
  |---|---|---|---|---|---|---|
  | `gemma_2b` | 2048 | 18 | 16384 | 8 | 1 | 256 |
  | `gemma_300m`（action expert） | 1024 | 18 | 4096 | 8 | 1 | 256 |

  两者的层数和头配置相同，所以可以逐层做联合注意力：动作 token 的 query 直接读取前缀的 K/V — [gemma.py @215abfb](https://raw.githubusercontent.com/Physical-Intelligence/openpi/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/gemma.py)
- `Pi0Config` 默认值：`dtype="bfloat16"`，`paligemma_variant="gemma_2b"`，`action_expert_variant="gemma_300m"`，`action_dim=32`，`action_horizon=50`，`max_token_len = 200 if pi05 else 48`，`discrete_state_input = pi05`，`pytorch_compile_mode="max-autotune"` — [pi0_config.py @215abfb](https://raw.githubusercontent.com/Physical-Intelligence/openpi/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py)
- 各发布配置的 action_horizon：

  | 配置 | action_horizon | 其他 |
  |---|---|---|
  | `pi05_droid` | 15 | — |
  | `pi05_libero` | 10 | `discrete_state_input=False` |
  | `pi05_aloha` | 默认 50 | — |
  | `pi0_droid` | 10 | — |
  | `pi0_fast_droid` | 10 | action_dim=8 |
  | `pi0_fast_libero` | 10 | action_dim=7，max_token_len=180 |

  — [config.py @215abfb](https://raw.githubusercontent.com/Physical-Intelligence/openpi/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py)

**预处理**
- 图像：
  - `IMAGE_KEYS = (base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb)`，`IMAGE_RESOLUTION = (224, 224)`。
  - uint8 图像先做 `/255*2-1` 转到 [-1,1]，再 `resize_with_pad`（仅训练时做增强，腕部相机不做部分增强）。
  — [model.py @215abfb](https://raw.githubusercontent.com/Physical-Intelligence/openpi/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/model.py)
- LIBERO 只有两路相机，`right_wrist_0_rgb` 用全零图填充。π0 / π0.5 中这一路的 `image_mask=False`，π0-FAST 中为 True — [libero_policy.py](https://raw.githubusercontent.com/Physical-Intelligence/openpi/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/libero_policy.py)
- 状态和动作的归一化：只要 `model_type != PI0`（即 π0.5 和 FAST），就用分位数归一化（`use_quantile_norm`）；π0 用均值方差。推理输出经 `Unnormalize` 等 output transforms 还原 — [config.py @215abfb](https://raw.githubusercontent.com/Physical-Intelligence/openpi/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py)
- 分词（`PaligemmaTokenizer`）：
  - π0.5 格式：先把状态离散化，`np.digitize(state, np.linspace(-1,1,257)[:-1]) - 1`，得到 256 个 bin；再拼成 `"Task: {prompt}, State: {s1 s2 ...};\nAction: "`，加 BOS 后编码。
  - π0 格式：`prompt` 编码后单独追加 `"\n"`。
  - 不足 `max_len` 时右侧 pad 并给出 mask；超出时打 warning 并截断。
  — [tokenizer.py @215abfb](https://raw.githubusercontent.com/Physical-Intelligence/openpi/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/tokenizer.py)
- π0.5 论文："The robot proprioceptive state is discretized and input to the model as text tokens" — [π0.5 arXiv HTML](https://arxiv.org/html/2504.16054)

**注意力掩码与前缀 / 后缀**
- `make_attn_mask` 对 `ar_mask` 做 cumsum：一个 token 只能看 cumsum 小于等于自己的 token — [pi0.py](https://raw.githubusercontent.com/Physical-Intelligence/openpi/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py)
- 前缀：图像 token 与文本 token 的 `ar_mask` 全为 False，即前缀内部是完全双向注意力。
- 后缀：
  - π0：先放一个 state token（`state_proj`，ar=True，开启新的块）；再放动作 token，`[True] + [False]*(H-1)`，动作块内部双向。
  - π0.5：没有 state token。时间步 t 经过 `time_mlp_in/out` 变成 adaRMS 条件（`adarms_cond`）；π0 则是把动作和时间拼接后过 MLP。
  — [pi0.py](https://raw.githubusercontent.com/Physical-Intelligence/openpi/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py)
- π0 论文把它描述为三块 blockwise causal mask：`[images, language]`、`[state q]`、`[actions]`；块内双向，块不能看未来的块 — [π0 arXiv HTML](https://arxiv.org/html/2410.24164)

**prefill 一次 + 多步去噪复用 KV**
- `sample_actions(num_steps=10)` 的流程：
  1. `embed_prefix` → `make_attn_mask` → `positions = cumsum(prefix_mask)-1`；
  2. 运行 `PaliGemma.llm([prefix_tokens, None], ...)` 得到 `kv_cache`；
  3. 在 `jax.lax.while_loop` 中每步执行 `embed_suffix(x_t, time)`，后缀位置为 `sum(prefix_mask) + cumsum(suffix_mask) - 1`；以 `[None, suffix_tokens]` 加 `kv_cache` 前向；`v_t = action_out_proj(...)`；更新 `x_t + dt*v_t`。
  - 其中 `dt = -1/num_steps`，t=1 为噪声、t=0 为目标。代码注释承认这与 π0 论文的约定相反。
  — [pi0.py](https://raw.githubusercontent.com/Physical-Intelligence/openpi/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py)
- π0 论文："We use 10 integration steps (δ = 0.1)"；"inference can be implemented efficiently by caching the attention keys and values for the prefix o_t and only recomputing the suffix corresponding to the action tokens for each integration step"；H=50；对灵巧任务最高可达 50 Hz 控制 — [π0 arXiv HTML](https://arxiv.org/html/2410.24164)

**策略封装与计时**
- `Policy.infer`：`_input_transform` → `_sample_actions`（JAX 下用 `nnx_utils.module_jit` 包装）→ `_output_transform`。返回的 `policy_timing.infer_ms` 只统计模型调用这一段，用 `time.monotonic` 计时 — [policy.py @215abfb](https://raw.githubusercontent.com/Physical-Intelligence/openpi/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/policy.py)
- PyTorch 版：`self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)`，默认 `"max-autotune"` — [pi0_pytorch.py](https://raw.githubusercontent.com/Physical-Intelligence/openpi/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/pi0_pytorch.py)
- README 写明：PyTorch 支持 π0 / π0.5，不支持 π0-FAST；已在 LIBERO 上验证；需要把 `transformers_replace` 拷入 transformers 包。推理显存需 >8 GB（RTX 4090 即可）。2025-09 发布了 π0.5 和 PyTorch 支持 — [openpi README @215abfb](https://raw.githubusercontent.com/Physical-Intelligence/openpi/215abfb217dbac7d5f1273282331b9b1866c0479/README.md)

**π0.5 是否做语言解码**
- 论文中的推理流程是：先自回归解码高层子任务文本，再以文本为条件做 10 步 denoising 出动作。原文："At inference time we then use standard autoregressive decoding for text tokens ℓ̂ followed by 10 denoising steps, conditioned on text tokens"。训练分两阶段：预训练用 FAST 离散 token，后训练加入 action expert（knowledge insulation）。论文同时指出，离散动作表示因为需要昂贵的自回归解码，不适合实时推理 — [π0.5 arXiv HTML](https://arxiv.org/html/2504.16054)
- openpi README："in this repository, we currently only support the flow matching head for both π0.5 training and inference"。也就是说，开源版没有子任务解码，OxyGen 的语言路径来自社区复现 — [openpi README](https://raw.githubusercontent.com/Physical-Intelligence/openpi/215abfb217dbac7d5f1273282331b9b1866c0479/README.md)；[v3 HTML](https://arxiv.org/html/2603.14371v3)

**π0-FAST 对照**
- `Pi0FASTConfig.max_token_len=250`，`sample_actions(max_decoding_steps=256)`，`PALIGEMMA_EOS_TOKEN=1`。prefill 时同样把 attention mask pad 到 `prefill_size + max_decoding_steps`，预留 KV 槽位。OxyGen 的 `prefill(max_decoding_steps)` 沿用了同一模式 — [pi0_fast.py](https://raw.githubusercontent.com/Physical-Intelligence/openpi/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_fast.py)
- FAST 用 DCT + BPE 压缩动作，每个 chunk 每条手臂约 30 个 token（双臂约 60 个）。π0 在 4090 上预测 1 秒动作块"within 100ms"，π0-FAST 约需 750 ms。原因有两点：要解码 30–60 个 token（π0 只需 10 步去噪），而且要用完整的 2B 骨干解码（π0 用的是 300M action expert） — [FAST arXiv HTML](https://arxiv.org/html/2501.09747)

**时延数字**
- π0 论文 Table I（RTX 4090，3 路相机）：

  | 部分 | 时延 |
  |---|---|
  | image encoders | 14 ms |
  | observation forward pass | 32 ms |
  | x10 action forward pass (flow) | 27 ms |
  | network latency（off-board 时） | 13 ms |
  | total on-board | 73 ms |
  | total off-board | 86 ms |

  — [π0 arXiv HTML](https://arxiv.org/html/2410.24164)
- "Running VLAs at Real-time Speed"（π0，单张 RTX 4090）：openpi/JAX 在 1/2/3 路视角下为 43.8 / 53.7 / 67.6 ms，优化后为 20.0 / 27.3 / 36.8 ms。手段包括用 CUDA graph 消除 CPU 开销、简化计算图、重写 kernel — [arXiv HTML 2510.26742](https://arxiv.org/html/2510.26742)

### Inferences
- 每帧前缀 token 数（推断）：π0.5 默认为 3×256 + 200 = 968；OxyGen 的 `pi05_o2_*` 为 3×256 + 100 = 868。LIBERO 中被 mask 掉的那张零图仍会被编码，只是在注意力里不可见，因此 SigLIP 的计算并没有省下来。
- 每样本 KV（推断，假设 bf16 存储）：
  - π0.5 默认：968 × 18,432 B ≈ 17.0 MiB；
  - OxyGen 预留 N=30 时：(868+30) × 18,432 B ≈ 15.8 MiB。
  - 对比 Llama-2-7B（32 层 × 32 个 KV 头 × head_dim 128）每 token 约 512 KiB——其配置本次未核实。可见 VLA 的 KV 很小，瓶颈在算力和权重读取。
- 前缀内部是双向注意力，π0.5 又把状态写进了文本，所以哪怕图像没变，第 1 层以后所有前缀 token 的 KV 也会随状态变化。因此 LLM 式的精确 prefix caching 不能跨帧直接复用，只能像 VLA-Cache 那样做近似复用。
- 4090 上的两组数字相互吻合：π0 的前缀约 46 ms，10 步去噪 27 ms；OxyGen 的 π0.5 前缀 44.0 ms，去噪 20.4 ms。也就是说，在消费级 GPU 上 prefill 是主要开销。

### Gaps
- π0.5 的 action expert 参数量：π0.5 论文只说 "significantly smaller"，"300M" 来自 FAST 论文对 π0 的描述和 `gemma_300m` 变体名，π0.5 原文的具体数字未核实。
- Physical Intelligence 没有在本次检索到的来源中公布 π0.5 的官方推理时延。
- H800 上的 openpi 时延没有一手数据。

---

## Q3. VLA 阶段 ↔ 通用 LLM serving 概念映射（以及类比在哪里失效）

### Takeaway
VLA 的前缀编码相当于 prefill；FAST token 和 OxyGen 的语言解码相当于 decode；flow-matching 去噪最接近"固定步数、非自回归、反复读取同一前缀 KV 的多次前向"；OxyGen 的跨帧语言 batching 相当于帧粒度的 continuous batching；动作截止期相当于周期性的逐迭代时延 SLO。

类比主要在五处失效：前缀是双向注意力；每帧观测都是全新的，前缀很难跨帧复用；去噪不追加 KV；batch=1 的闭环控制；"频率"指标是吞吐型而非时延型。

### Cited Findings（映射表的事实依据）

| VLA / OxyGen 阶段 | 代码位置 | 对应的 LLM serving 概念 | 类比在哪里失效 |
|---|---|---|---|
| 图像 resize/归一化 + 状态离散化 + 分词 | `transforms`、`tokenizer.py`、`model.py::preprocess_observation` | tokenization / 多模态预处理（CPU 侧开销） | 状态会写进 prompt（π0.5），所以 prompt 每帧都变 |
| SigLIP 编码 + Gemma-2B 前缀前向生成 KV | `pi0.py::sample_actions` 前半段；OxyGen 的 `pi05.py::prefill` | **prefill**（compute-bound，决定 TTFT） | 前缀**双向**注意力（prefix-LM）；每帧都是新请求，跨帧 KV 不能精确复用 |
| 10 步 flow matching，每步 H 个 token 读取前缀 KV | `while_loop` / `sample_actions_with_kv` | 像 S 次"并行解码"迭代，但不追加 KV、步数固定 | 非自回归、形状静态、时延可预测，适合 CUDA graph；不存在输出长度方差 |
| π0-FAST 的 FAST token；OxyGen 的语言 token | `pi0_fast.py::sample_actions`；`generate_n_tokens` | **decode**（memory-bound，TPOT） | 用 2B 全骨干解码，动作 token 每块约 30–60 个 |
| 一次 prefill 同时供动作和语言两个专家 | `infer_text_actions_continuous_batch` | 前缀共享 / prefix caching（vLLM/SGLang） | 由两个**不同权重**的专家读取同一 KV；同一帧内复用，而不是跨请求复用 |
| 跨帧语言 batching，每帧推进 k 个 token | `ContinuousBatchManager` + `_stack_incremental_states` | **continuous batching / iteration-level scheduling** | 迭代粒度是帧，新请求只在帧边界加入；KV 静态预留，不分页 |
| 动作每帧截止期 f_min；语言为软截止期 | 论文问题定义 | 延迟 **SLO**（per-iteration / TBT 类）与 E2E/吞吐 SLO 的混合调度 | 动作是周期性实时流；论文的 action frequency = H/T，是吞吐而不是反应时延 |
| MPS 并行 isolated execution | `parallel_mps` | 多模型 / 多进程共置（spatial sharing） | 两个进程会重复 prefill，显存接近翻倍 |

- 用到的一手事实：
  - OxyGen 论文明确把 vLLM/SGLang 的 prefix caching 和 continuous batching 作为参照，并说明简单开启这两项"does not provide this coordination" — [v3 HTML](https://arxiv.org/html/2603.14371v3)
  - π0 的前缀 KV 缓存与后缀重算 — [π0 arXiv HTML](https://arxiv.org/html/2410.24164)
  - 前缀双向注意力 — [pi0.py](https://raw.githubusercontent.com/Physical-Intelligence/openpi/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py)
  - FAST 的自回归解码代价 — [FAST HTML](https://arxiv.org/html/2501.09747)
  - action frequency 的定义 — [Experiments.md](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/experiments/Experiments.md)
  - MPS 下显存接近翻倍 — [v3 HTML](https://arxiv.org/html/2603.14371v3)
- 单机器人 batch=1 是常态。"Beyond LLM Serving"（ASPLOS'27 方向的刻画论文，arXiv 2610.05062）指出：每个控制周期就是一个推理截止期，超时就会让机器人基于陈旧观测行动；单机器人 batch-1 推理不在 LLM serving 系统的设计点上；动作张量的维度决定一个阶段是 memory-bound 还是 compute-bound — [arXiv 2610.05062](https://arxiv.org/abs/2610.05062)
- PhyAI 在 Hopper 系列 GPU 上测得：batch=1 时，π0.5 action expert 只占 8.8% 的 FLOPs，却占 57.2% 的时延；batch=32 时这一占比降到 13.5%，吞吐约 100 samples/s — [arXiv 2608.03682](https://arxiv.org/abs/2608.03682)

### Inferences
- 对初学者最有用的一句话：**prefill ≈ 图像+文本前缀编码；decode ≈ 语言/FAST token；flow matching ≈ "S 次只算 H 个 query token、复用同一份 KV 的小前向"**。去噪步的 arithmetic intensity 极低（300M 权重只处理 10–50 个 token），在 H800 这类高算力卡上会更偏 memory/launch-bound。这与 PhyAI 在 Hopper 上 57.2% 的观察一致。
- OxyGen 的固定 k 类似 chunked-prefill 系统里的"每迭代 token 预算"：都是用有界的每轮工作量保护时延敏感流（这里是动作）。不过两者保护的对象和预算单位不同。该类比的 LLM 侧文献（Orca、Sarathi-Serve 等）本次未核实，需要与其他研究员的笔记交叉确认。
- "Little 定律"式推断：uniform 到达 rate=1 帧⁻¹ 时，每个请求存活 N/k 帧，所以稳态 batch ≈ N/k。例如 N=30、k=5 时 ≈ 6。这可以作为实验 E3 的验收值。

### Gaps
- Orca / vLLM PagedAttention / Sarathi-Serve 的原始论文本次没有打开核实。映射表中这些名称只作为概念对应，需要其他笔记补充来源。

---

## Q4. 相关 VLA 推理 / serving 系统（简表）

### Takeaway
与 OxyGen 最接近的工作分三类：
- 帧间视觉 KV 近似复用：VLA-Cache；
- 跨请求的 prefill/decode 流水线：ActionFlow；
- 多机器人、多模型的 SLO serving：Robion、Kairos。

另有三类可做背景：kernel/引擎优化（Running VLAs at Real-time Speed、PhyAI）和性能建模/刻画（VLA-Perf、Beyond LLM Serving）。

### Cited Findings
- **VLA-Cache**（NeurIPS 2025）：training-free，复用相邻帧中变化很小的静态视觉 token 的 KV，对任务相关 token 重新计算，并按层自适应决定复用比例；CUDA 时延最高 1.7× 加速，控制频率 +15% — [arXiv 2502.02175](https://arxiv.org/abs/2502.02175)
- **ActionFlow**（arXiv 2025-12）：面向边缘设备，用 Cross-Request Pipelining 把 memory-bound 的 decode 与 compute-bound 的 prefill 跨时间步混合批处理，配合 Unified KV Ring Buffer；OpenVLA-7B 的 FPS 提升 2.55× — [arXiv 2512.20276](https://arxiv.org/abs/2512.20276)
- **Robion**（arXiv 2026-09）：在单个 GPU 内用两条 stream 把 VLM 阶段和 ADiT 阶段解耦，并限制 VLM stream 的 SM 数量；按剩余 SLO 时间最少优先调度；在 98% SLO 达成率下承载的机器人负载比 vLLM-Omni 高 6.7×，比 Monolithic 方案高 1.5× — [arXiv 2609.12075](https://arxiv.org/abs/2609.12075)
- **Kairos**（arXiv 2026-05）：多机器人 serving，把 generate–execute 循环当作一等公民；端到端任务时延降低 31.8–66.5% — [arXiv 2605.11381](https://arxiv.org/abs/2605.11381)
- **Running VLAs at Real-time Speed**（arXiv 2025-10）：单张消费级 GPU 上做到 π0 级多视角 30 Hz 帧率、最高 480 Hz 轨迹；数字见 Q2；代码在 github.com/Dexmal/realtime-vla — [arXiv 2510.26742](https://arxiv.org/abs/2510.26742)
- **PhyAI**（arXiv 2026-08）：统一的 VLA/WAM 推理引擎，比官方实现快 1.40–4.65×；包含上文 Hopper 上的 action expert 时延占比数据 — [arXiv 2608.03682](https://arxiv.org/abs/2608.03682)
- **VLA-Perf**（arXiv 2026-02）：针对 VLA 推理的解析性能模型，给出 15 条 takeaway — [arXiv 2602.18397](https://arxiv.org/abs/2602.18397)
- **Beyond LLM Serving**（arXiv 2026-10）：在边缘 GPU 服务器和两种板载 SoC 上刻画 4 个 VLA，包含 43,200 个闭环 episode — [arXiv 2610.05062](https://arxiv.org/abs/2610.05062)

### Inferences
- 把 OxyGen 放进谱系：它是"单设备、单模型、多任务"的 KV 协调。VLA-Cache 是单任务的跨帧近似复用，ActionFlow 是单任务的跨请求流水线，Robion/Kairos 是多机器人 serving。

### Gaps
- 本地线索中的 ROSA（2607.01088）、Armory（2608.00337）、EagleVLA（2607.12659）、RTC（2506.07339）本次没有重新打开一手来源复核，需要的话请标为「未核实」后再使用。
- ActionFlow 与 Robion 的会议归属本次未查到。

---

## Q5. 在 H800 上对 OxyGen / openpi 复现做的插桩实验

### Takeaway
下面 7 个实验从"单次推理分解"逐步走到"调度与 SLO"，每个都给出目标、插桩位置、预期观察和验收标准。

所有计时都要注意两点：
- **JAX 是异步派发的**，必须对输出调用 `block_until_ready()`；
- **PyTorch 必须 `torch.cuda.synchronize()`**，并丢弃 warmup 和编译轮次。

预期值以 4090 上的论文数据为参照；H800 上的数值未核实，需要用户自己测得。

### Cited Findings（实验设计所依赖的事实）
- 可直接复用的钩子：
  - `Pi05.profile_sample_text / profile_sample_actions / prefile_sample_text_actions_shared_kv`（函数名原文如此拼写），返回 prefill、actions、text 的分段时间；
  - `infer_text_actions_continuous_batch` 的 `policy_timing`；
  - runner 记录的 `frame_ms / n_new / n_resumed / total_tokens_this_frame / is_warmup`。
  — [pi05.py](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/src/openpi/models/pi05.py)；[policy.py](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/src/openpi/policies/policy.py)；[Experiments.md](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/experiments/Experiments.md)
- 注意 `prefill_actions_ms` 把 prefill 和动作去噪**合在一起**计时。要分开测，需要在 `_prefill` 之后加同步 — [policy.py](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/src/openpi/policies/policy.py)
- 随机初始化即可测时延，不需要下载 checkpoint — [Experiments.md](https://raw.githubusercontent.com/air-embodied-brain/OxyGen/1e8890681090c7f423f85866b5db877916ba65e6/experiments/Experiments.md)

**E1 阶段分解：prefill、单步去噪、单步解码**
- 目标：在 H800 上得到"prefill : 一步 denoise : 一步 decode"的比例。
- 插桩：
  - 在 `policy.py::infer_text_actions_continuous_batch` 中，`_prefill` 之后对结果调用 `jax.block_until_ready`，再分别计时 `_sample_actions_with_kv` 和 `_generate_n_tokens`；
  - 把 `--num-denoise-steps` 设为 1 和 10 做差分，得到单步代价；把 `steps_per_frame` 设为 1 和 5 做差分，同理；
  - 可选：用 `jax.profiler` 或 Nsight Systems 统计每步的 kernel 数。
- 预期：
  - 4090 参考值为 prefill 44.0 ms、S=10 去噪 20.4 ms；
  - Hopper 上去噪占比可能上升（PhyAI 测得 bs=1 时 action expert 占 57.2%）；
  - 单步 decode 应接近"读一遍 2B 权重"的带宽下限。
- 验收：
  - 分段之和与 `frame_ms` 相差不超过 5%；
  - 100 次运行的变异系数 < 5%；
  - 在笔记中写下实测比例，并与 4090 的参考值对照。

**E2 前缀 KV 大小 vs 图像数 / prompt 长度**
- 目标：验证"每 token 18 KiB、KV 很小"这一推断。
- 插桩：
  - 在 `prefill` 返回后打印 `kv_cache` 中各张量的 shape、dtype 和 nbytes；
  - 改变 `max_token_len`（48/100/200）与图像 key 的数量（需要修改输入变换，真正删除 key；仅改 `image_mask` 不会减少 token 数）。
  - 分辨率由 SigLIP 固定为 224，不建议修改。
- 预期：
  - 字节数 = token 数 × 18,432 × batch；
  - 一张图 = 256 个 token；
  - prefill 时间大致随 token 数线性增长（对 2B 模型，注意力的二次项可以忽略——此为推断）。
- 验收：实测字节数与公式误差在 1% 以内；画出"token 数 → prefill ms"曲线。

**E3 并发任务数 / N / k / 到达率扫描**
- 目标：复现论文"N 越大、k 越小，加速越多"的趋势，并验证稳态 batch≈N/k。
- 插桩：
  - `run_experiments --settings baseline shared_kv continuous_batching --max-decoding-steps 10 20 30 --steps-per-frame 1 5 10`；
  - `run_workload_sweep` 扫 Poisson 到达；
  - 读取 `avg_batch_size` 与 `language_throughput_tps`。
- 预期：uniform rate=1 时 `avg_batch_size≈N/k`；加速比随 N/k 增大。
- 验收：
  - batch 与 N/k 偏差 ≤ 0.5；
  - 加速比的单调趋势与论文 Fig.5/10 一致；
  - 报告 p50/p99 帧时延。

**E4 JAX vs PyTorch + torch.compile**
- 目标：量化编译和 CUDA graph 对小 kernel 密集型去噪循环的作用。
- 插桩：
  - 分别以 `OPENPI_TORCH_COMPILE=0/1` 和 `OPENPI_TORCH_COMPILE_DENOISE_CUDAGRAPHS=0/1` 运行，与 JAX 对照；
  - 记录首次编译耗时与稳态时延；
  - 用 `OPENPI_TORCH_COMPILE_PREFILL=0` 做 eager 数值对照。
- 预期：CUDA graph 主要降低去噪阶段的 launch 开销（realtime-vla 在 4090 上用 CUDA graph 等手段把 π0 从 53.7 ms 降到 27.3 ms）。
- 验收：
  - 表格列出三种后端的 prefill / denoise / decode 稳态时延与编译时间；
  - 同样噪声下 JAX 与 PyTorch 的动作差异小（阈值自定，例如相对 L2 < 1e-2，此值为建议而非来源）。

**E5 动作截止期的松弛度**
- 目标：把"action frequency"换算成真正的截止期，检查超时率。
- 插桩：
  - 设定 f_min，例如 50 Hz（π0 的灵巧任务上限），则帧预算 T_max = H/f_min，H=10 时为 200 ms；
  - 对每帧计算 slack = T_max − frame_ms；
  - 另外单独记录"新请求从到达到动作就绪"的时延（即 `prefill_actions_ms`）。这才是反应时延；H/T 只是吞吐。
- 预期：continuous batching 下的 slack 随 N 基本保持不变，baseline 下随 N 变差。
- 验收：
  - 报告 miss rate（slack<0 的帧比例）与 p99 slack；
  - 找出满足 miss rate < 1% 的最大 k。
  - 扩展练习：实现一个按上一帧 slack 自适应调整 k 的控制器，对应论文中"fixed k"这一局限。

**E6 动作去噪的 batch 扩展（多机器人视角）**
- 目标：在 H800 上找到 denoise 从 memory/launch-bound 进入 compute-bound 的拐点。
- 插桩：用 `profile_sample_actions` 对 batch 1/2/4/8/16/32 计时，算出每样本时延和吞吐。
- 预期：小 batch 时总时延基本平坦，摊薄后每样本时延大幅下降（PhyAI 在 Hopper 上 bs=32 时约 100 samples/s）。
- 验收：画出 batch→吞吐曲线并标出拐点；用这个结果说明为什么 Robion/Kairos 要跨机器人做 batching。

**E7 isolated vs MPS vs 多卡**
- 目标：理解"空间共享"为何不如"共享 KV"。
- 插桩：在单卡上跑 `parallel_mps`；可选：在 4×H800 上每卡起一个独立实例，模拟 4 台机器人。
- 预期：与论文一致，MPS 下显存接近翻倍、收益有限。
- 验收：报告峰值显存（`nvidia-smi` 或 JAX memory stats）、帧时延和功率（`nvidia-smi --query-gpu=power.draw`），并与 Table 3 的趋势对比。

### Inferences
- 推荐顺序：E1 → E2 → E4 → E3 → E5 → E6 → E7。这一顺序依次对应"推理分解 → 内存 → 编译 → 调度 → SLO → 扩展"，正好可以搭成通用推理 Infra 课程的骨架。

### Gaps
- OxyGen 的测试环境是 CUDA 13.0 + 4090、A100 PyTorch、Thor。H800 上的 JAX/CUDA 兼容性和性能未核实。
- 本次没有核实 H800 的峰值算力和带宽规格，带宽下限的估算需要用户查官方 datasheet 后代入。
