# AGENT_SETUP：在 4×H800 服务器上配置《模型推理 Infra 七天学习》环境

> 本文件写给执行配置的 AI agent。请严格按顺序执行，每步完成后核对“成功标准”，
> 不满足就停下并把“需要汇报的内容”发给人类，不要自己发挥。
> 资料核实日期：2026-10-14/15。版本号会变，以当天官方文档为准。

---

## 0. 规则（必须遵守）

**可以做：**
- 在 `$WORK`（默认 `~/infra7`）目录内创建、写入文件。
- 用 `uv` 在 `$WORK/envs/` 下创建虚拟环境并安装 Python 包。
- 克隆公开 Git 仓库，下载公开模型和数据集。

**禁止做（遇到需要时，停下来问人类）：**
- 不要用 `sudo`，不要升级或降级 NVIDIA 驱动、CUDA，不要改系统配置。
- 不要修改网络、代理、防火墙设置。下载失败时只汇报报错原文。
- 不要删除或覆盖 `$WORK` 以外的任何文件；`$WORK` 里已存在的目录也不要删。
- 不要做任何模型训练。
- 不要把 HF token、密码写进文件或日志。需要 token 时请人类输入。

**每一步都要记录：** 执行的命令、退出码、输出的最后 20 行。

---

## 1. 前提条件

| 项目 | 要求 | 怎么检查 |
|---|---|---|
| 操作系统 | Linux x86_64（Ubuntu 22.04+ 推荐） | `uname -a` |
| GPU | 4 × NVIDIA H800 80GB | `nvidia-smi` |
| 驱动 | ≥ 580（SGLang 0.5.21 需要 CUDA 13）；只用 vLLM cu129 时 ≥ 575.51.03 | `nvidia-smi` 第一行 |
| 磁盘 | `$WORK` 所在分区 ≥ 200 GB 可用（全量模型约 320 GB） | `df -h ~` |
| 工具 | `git`、`wget`、`uv`；编 nccl-tests 需要 `nvcc` 和 `make` | `command -v git wget uv nvcc make` |

`uv` 未安装时：请人类决定是否安装（官方文档：https://docs.astral.sh/uv/getting-started/installation/ ）。

---

## 2. 把学习包拷到服务器

学习包在 Windows 机器的 `D:\Code\inference_infra_guide_20261014\`。在服务器上放到
`~/infra7/guide`（与 `scripts/config.sh` 中 `LABS_DIR=$WORK/guide/labs` 对应）：

```bash
# 在 Windows（Git Bash）上执行，user@server 换成真实地址：
scp -r /d/Code/inference_infra_guide_20261014 user@server:~/infra7/guide
```

**成功标准：** 服务器上 `ls ~/infra7/guide/scripts ~/infra7/guide/labs` 能看到脚本和 `lab01_bpe_toy.py` 等 10 个文件。

如果放在别的位置，执行脚本前先设置：`export LABS_DIR=/实际路径/labs`。

之后所有命令都在 `~/infra7/guide` 目录下执行：`cd ~/infra7/guide`

---

## 3. 第 0 步：环境检查（只读）

```bash
bash scripts/00_check_env.sh 2>&1 | tee ~/infra7/logs_00_check.txt
```

**成功标准：**
- 列出 4 张 `NVIDIA H800`，每张约 80 GB（81559 MiB 左右）。
- 打印 `OK: 驱动 >= 580`。若打印 `WARN`：可以继续，但第 4 步只装 vLLM（见下文），并告诉人类。
- `nvidia-smi topo -m` 的 GPU 之间显示 `NV#`（# 为数字）。若显示 `PIX`/`SYS`，说明不是 NVLink 互联，告诉人类（影响第 5 天并行实验结论，不影响配置）。

**需要汇报的内容：** GPU 型号与数量、驱动版本、topo 矩阵、可用磁盘。

---

## 4. 第 1 步：创建 Python 环境

```bash
bash scripts/01_setup_envs.sh all      # 驱动 < 580 时改成：bash scripts/01_setup_envs.sh vllm
```

脚本做的事：
1. `$WORK/envs/vllm`：`uv pip install "vllm==0.31.0" --torch-backend=auto`，再执行 `vllm download-kernels`。
2. `$WORK/envs/sglang`：`uv pip install --prerelease=allow "sglang==0.5.21"`。

**成功标准：** 最后打印类似
`vllm 0.31.0 torch 2.13.0... cuda ...` 和 `sglang 0.5.21 torch 2.13.0...`，并显示 `hf` 版本号。

**常见问题：**
| 现象 | 处理 |
|---|---|
| `uv: command not found` | 停下，请人类安装 uv |
| 下载超时 / 连接错误 | 停下，汇报报错原文；不要改代理或换源 |
| `No matching distribution` | 汇报 Python 版本（`uv python list`）和报错；不要换版本号 |
| SGLang 安装后 import 报 CUDA 错误 | 多半是驱动 < 580，改为只用 vLLM，告诉人类 |

> 已知不确定点：vLLM 0.31.0 的 release notes 说 PyPI 默认 wheel 是 CUDA 13.0，
> 而同版本安装文档说默认 CUDA 12.9。`--torch-backend=auto` 会按驱动自动选择，
> 安装后用上面打印的 `cuda` 字段确认实际版本，写进汇报。

---

## 5. 第 2 步：克隆源码仓库

```bash
bash scripts/02_clone_repos.sh
```

会在 `$WORK/repos/` 下克隆并切换到指定版本：

| 目录 | 仓库 | 版本 |
|---|---|---|
| vllm | https://github.com/vllm-project/vllm | tag v0.31.0 |
| sglang | https://github.com/sgl-project/sglang | tag v0.5.21 |
| nano-vllm | https://github.com/GeeeekExplorer/nano-vllm | main |
| mini-sglang | https://github.com/sgl-project/mini-sglang | main |
| gpt-fast | https://github.com/meta-pytorch/gpt-fast | main |
| nccl-tests | https://github.com/NVIDIA/nccl-tests | master |
| openpi | https://github.com/Physical-Intelligence/openpi | commit 215abfb |
| OxyGen | https://github.com/air-embodied-brain/OxyGen | commit 1e88906 |

**成功标准：** 每个仓库打印一行 `OK: 名字 @ 短hash`；已存在的打印“跳过”。
如果人类已经在别处复现过 OxyGen，可以跳过这里的 OxyGen，告诉人类即可。

---

## 6. 第 3 步：下载模型和数据集

```bash
bash scripts/03_download_models.sh core       # 第 1–4 天够用，约 37 GB
bash scripts/03_download_models.sh extended   # 第 5 天前再下，额外约 125 GB
```

| 档位 | 模型（Hugging Face ID） |
|---|---|
| core | Qwen/Qwen3-0.6B，Qwen/Qwen3-8B，Qwen/Qwen2.5-VL-7B-Instruct，openai/whisper-large-v3 |
| extended | 加 Qwen/Qwen3-32B，Qwen/Qwen3-30B-A3B |
| all | 再加 Qwen/Qwen3-VL-8B-Instruct，Qwen/Qwen2.5-Omni-7B，deepseek-ai/DeepSeek-V2-Lite-Chat |

另外下载 ShareGPT：`$WORK/data/ShareGPT_V3_unfiltered_cleaned_split.json`。

**成功标准：** 每个模型 `hf download` 结束后打印本地路径；最后 `du -sh` 显示大小与上表大致相符。
以上模型都是公开的（Apache-2.0 或 DeepSeek 许可），不需要 token。
`meta-llama/Llama-3.1-8B-Instruct` 是 gated 模型，脚本没有包含；需要时请人类先在网页申请并执行 `hf auth login`。

---

## 7. 第 4 步：通信带宽测试

```bash
bash scripts/04_nccl_tests.sh
```

**成功标准：** 输出表格最后几行（大消息，约 1G–4G）有 `busbw` 数值，结果保存在
`$WORK/results/nccl_allreduce_4gpu.txt`。把最大 busbw 写进汇报（H800 的 NVLink 比 H100 窄，数值会明显低于 H100 公开数据，这是正常的）。
`make` 失败（找不到 nvcc/NCCL）时：汇报报错，跳过此步，不影响后续。

---

## 8. 第 5 步：运行 labs 自检

```bash
bash scripts/05_run_labs.sh
```

**成功标准：** 最后打印 `全部通过`。每个 lab 会打印 `N/N checks passed`。
`lab09` 在 GPU 上可能有 1 个 `SKIP`（CUPTI 不可用时），不算失败。
有 `FAIL` 时：把日志路径和失败的 lab 名字汇报给人类，不要改 lab 代码。

> 说明：lab07 用 `torchrun --nproc_per_node=4 ... --backend nccl`，这条在作者本机（Windows）
> 没法运行，第一次在 H800 上运行。若它失败而其他都通过，汇报其报错即可。

---

## 9. 第 6 步：vLLM 服务冒烟测试

终端 A：
```bash
bash scripts/06_vllm_serve.sh Qwen/Qwen3-0.6B 0
```
等到日志出现服务已启动（可访问 http://localhost:8000）。

终端 B：
```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "Hello", "max_tokens": 16}'
curl -s http://localhost:8000/metrics | grep -E '^vllm:' | head
bash scripts/07_vllm_bench.sh Qwen/Qwen3-0.6B smoke
```

**成功标准：** `/v1/models` 返回模型名；completions 返回 JSON 文本；`/metrics` 有 `vllm:` 开头的指标；
`$WORK/results/smoke/` 下有 5 个结果 JSON。完成后在终端 A 按 Ctrl+C 停止服务。

---

## 10. 第 7 步（可选）：SGLang 冒烟测试

仅当第 0 步驱动 ≥ 580 且第 1 步 SGLang 安装成功时执行。

终端 A：`bash scripts/08_sglang_serve_bench.sh serve Qwen/Qwen3-0.6B 0`
终端 B：`curl http://localhost:30000/v1/models`，然后 `bash scripts/08_sglang_serve_bench.sh bench Qwen/Qwen3-0.6B`

**成功标准：** 返回模型名；`$WORK/results/sglang_*/` 下有 5 个 `concurrency_*.txt`。

---

## 11. 最终汇报模板（发给人类）

```text
[环境] GPU: 4x ____  驱动: ____  topo: NV# / PIX / SYS
[Python 环境] vllm ____ (torch ____, cuda ____)；sglang ____ / 未安装（原因）
[仓库] 成功 __/8，失败：____
[模型] 已下载档位：core / extended / all，占用 ____ GB
[nccl-tests] 最大 busbw ____ GB/s（消息大小 ____）
[labs] 通过 __/10，失败：____（日志路径 ____）
[vLLM 冒烟] 通过 / 失败（报错原文 ____）
[SGLang 冒烟] 通过 / 失败 / 跳过
[需要人类处理的问题] ____
```

---

## 12. 链接清单（所有用到的仓库、文件、工具、文档）

### 12.1 推理引擎与官方文档
- vLLM 仓库：https://github.com/vllm-project/vllm
- vLLM 文档：https://docs.vllm.ai/
- vLLM GPU 安装：https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html
- vLLM Docker：https://docs.vllm.ai/en/latest/deployment/docker.html
- vLLM 压测 CLI：https://docs.vllm.ai/en/latest/benchmarking/cli/
- vLLM 架构总览（v0.31.0）：https://github.com/vllm-project/vllm/blob/v0.31.0/docs/design/arch_overview.md
- vLLM 前缀缓存设计：https://github.com/vllm-project/vllm/blob/v0.31.0/docs/design/prefix_caching.md
- vLLM CUDA Graph 设计：https://github.com/vllm-project/vllm/blob/v0.31.0/docs/design/cuda_graphs.md
- vLLM 指标说明：https://github.com/vllm-project/vllm/blob/v0.31.0/docs/usage/metrics.md
- vLLM profiling：https://github.com/vllm-project/vllm/blob/v0.31.0/docs/contributing/profiling.md
- vLLM 并行与扩展：https://github.com/vllm-project/vllm/blob/v0.31.0/docs/serving/parallelism_scaling.md
- SGLang 仓库：https://github.com/sgl-project/sglang
- SGLang 文档：https://docs.sglang.io/
- SGLang 安装：https://docs.sglang.io/get_started/install.html
- SGLang PD 分离：https://github.com/sgl-project/sglang/blob/v0.5.21/docs/docs/advanced_features/pd_disaggregation.mdx
- SGLang 指标：https://github.com/sgl-project/sglang/blob/v0.5.21/docs/docs/references/production_metrics.mdx

### 12.2 关键源码文件（按阅读顺序）
- vLLM 调度器：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/core/sched/scheduler.py
- vLLM KV cache 管理：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/core/kv_cache_manager.py
- vLLM 块池：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/core/block_pool.py
- vLLM EngineCore：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/engine/core.py
- vLLM AsyncLLM 前端：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/engine/async_llm.py
- vLLM Model Runner（V1）：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/worker/gpu_model_runner.py
- vLLM Model Runner V2：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/worker/gpu/model_runner.py
- vLLM 采样器：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/sample/sampler.py
- vLLM 引擎参数：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/engine/arg_utils.py
- SGLang 调度器：https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/scheduler.py
- SGLang Radix Cache：https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/mem_cache/radix_cache.py
- HF Qwen3 模型：https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3/modeling_qwen3.py
- HF Cache 实现：https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/cache_utils.py
- HF generate：https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/generation/utils.py
- HF Qwen2-VL 图像预处理（smart_resize）：https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
- OxyGen KV 管理：https://github.com/air-embodied-brain/OxyGen/blob/1e8890681090c7f423f85866b5db877916ba65e6/src/openpi/models/kv_cache_manager.py
- OxyGen π0.5 模型：https://github.com/air-embodied-brain/OxyGen/blob/1e8890681090c7f423f85866b5db877916ba65e6/src/openpi/models/pi05.py
- OxyGen 策略入口：https://github.com/air-embodied-brain/OxyGen/blob/1e8890681090c7f423f85866b5db877916ba65e6/src/openpi/policies/policy.py
- OxyGen 实验说明：https://github.com/air-embodied-brain/OxyGen/blob/1e8890681090c7f423f85866b5db877916ba65e6/experiments/Experiments.md

### 12.3 学习用小型代码仓库
- nano-vllm（约 1.2k 行，读 vLLM 的最短路径）：https://github.com/GeeeekExplorer/nano-vllm
- mini-sglang（约 5k 行，Radix Cache/分块预填充/重叠调度）：https://github.com/sgl-project/mini-sglang
- gpt-fast（单请求 decode 极致优化）：https://github.com/meta-pytorch/gpt-fast
- minbpe（BPE 分词器）：https://github.com/karpathy/minbpe
- nanoGPT：https://github.com/karpathy/nanoGPT
- 中文 ML 系统笔记：https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial

### 12.4 工具
- uv：https://docs.astral.sh/uv/
- Hugging Face CLI（`hf download` / `hf auth login`）：https://huggingface.co/docs/huggingface_hub/en/guides/cli
- nccl-tests：https://github.com/NVIDIA/nccl-tests
- nccl-tests 带宽含义：https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
- Nsight Systems：https://developer.nvidia.com/nsight-systems
- Nsight Compute：https://developer.nvidia.com/nsight-compute
- Perfetto（打开 trace JSON）：https://ui.perfetto.dev
- CUDA 驱动版本对应表：https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- Docker 镜像：https://hub.docker.com/r/vllm/vllm-openai ，https://hub.docker.com/r/lmsysorg/sglang

### 12.5 模型卡
- https://huggingface.co/Qwen/Qwen3-0.6B
- https://huggingface.co/Qwen/Qwen3-8B
- https://huggingface.co/Qwen/Qwen3-32B
- https://huggingface.co/Qwen/Qwen3-30B-A3B
- https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- https://huggingface.co/openai/whisper-large-v3
- https://huggingface.co/Qwen/Qwen2.5-Omni-7B
- https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat
- https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct （gated，需申请）

### 12.6 数据集
- ShareGPT（vLLM 压测文档使用）：https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered

### 12.7 本学习包内的文件
- 指南正文：`reports/模型推理 Infra 七天学习指南.md`（及同名 .pdf）
- labs 说明与测试记录：`labs/README.md`，`research_notes/模型推理 Infra 七天学习指南/code_labs.md`
- 脚本：`scripts/`（统一配置在 `scripts/config.sh`）
