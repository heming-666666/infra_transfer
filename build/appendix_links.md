
## 附录：代码仓库、文件与工具链接清单

本附录把正文和参考资料中用到的链接以完整 URL 列出，方便直接复制。核实日期 2026-10-15；源码文件链接固定在对应 tag 或 commit 上。配套的环境搭建操作手册见 `../AGENT_SETUP.md`，脚本见 `../scripts/`，代码实验见 `../labs/`。

### 推理引擎仓库与官方文档

- vLLM 仓库：https://github.com/vllm-project/vllm
- vLLM v0.31.0 发布说明：https://github.com/vllm-project/vllm/releases/tag/v0.31.0
- vLLM 文档首页：https://docs.vllm.ai/
- vLLM GPU 安装：https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html
- vLLM Docker 部署：https://docs.vllm.ai/en/latest/deployment/docker.html
- vLLM 压测 CLI：https://docs.vllm.ai/en/latest/benchmarking/cli/
- SGLang 仓库：https://github.com/sgl-project/sglang
- SGLang v0.5.21 发布说明：https://github.com/sgl-project/sglang/releases/tag/v0.5.21
- SGLang 文档首页：https://docs.sglang.io/
- SGLang 安装：https://docs.sglang.io/get_started/install.html
- vLLM-Omni：https://github.com/vllm-project/vllm-omni
- llm-d：https://github.com/llm-d/llm-d
- NVIDIA Dynamo：https://github.com/ai-dynamo/dynamo
- KServe：https://github.com/kserve/kserve
- vLLM production-stack：https://github.com/vllm-project/production-stack

### 关键源码与设计文档（按阅读顺序）

vLLM（tag v0.31.0）：

- 架构总览：https://github.com/vllm-project/vllm/blob/v0.31.0/docs/design/arch_overview.md
- 调度器：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/core/sched/scheduler.py
- KV cache 管理：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/core/kv_cache_manager.py
- 块池：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/core/block_pool.py
- 前缀缓存设计：https://github.com/vllm-project/vllm/blob/v0.31.0/docs/design/prefix_caching.md
- EngineCore 主循环：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/engine/core.py
- AsyncLLM 前端：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/engine/async_llm.py
- Model Runner（V1）：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/worker/gpu_model_runner.py
- Model Runner V2：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/worker/gpu/model_runner.py
- MRV2 设计文档：https://github.com/vllm-project/vllm/blob/v0.31.0/docs/design/model_runner_v2.md
- CUDA Graph 设计：https://github.com/vllm-project/vllm/blob/v0.31.0/docs/design/cuda_graphs.md
- 采样器：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/sample/sampler.py
- 引擎参数与默认值：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/engine/arg_utils.py
- 调度配置：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/scheduler.py
- 缓存配置：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/cache.py
- 显存与 worker：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/worker/gpu_worker.py
- 压测实现：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/benchmarks/serve.py
- 指标说明：https://github.com/vllm-project/vllm/blob/v0.31.0/docs/usage/metrics.md
- Profiling：https://github.com/vllm-project/vllm/blob/v0.31.0/docs/contributing/profiling.md
- 并行部署：https://github.com/vllm-project/vllm/blob/v0.31.0/docs/serving/parallelism_scaling.md
- 多模态配置：https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/multimodal.py

SGLang（tag v0.5.21）：

- 调度器：https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/scheduler.py
- Radix Cache：https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/mem_cache/radix_cache.py
- 压测实现：https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/benchmark/serving.py
- PD 分离文档：https://github.com/sgl-project/sglang/blob/v0.5.21/docs/docs/advanced_features/pd_disaggregation.mdx
- Model Gateway 文档：https://github.com/sgl-project/sglang/blob/v0.5.21/docs/docs/advanced_features/sgl_model_gateway.mdx
- 指标文档：https://github.com/sgl-project/sglang/blob/v0.5.21/docs/docs/references/production_metrics.mdx

Hugging Face transformers（tag v5.19.0）：

- Qwen3 模型：https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3/modeling_qwen3.py
- Cache 实现：https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/cache_utils.py
- generate 实现：https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/generation/utils.py
- Qwen2-VL 图像预处理（smart_resize）：https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
- Qwen3-VL 处理器：https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_vl/processing_qwen3_vl.py
- Qwen3-VL 模型：https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py

OxyGen（commit 1e88906）与 openpi（commit 215abfb）：

- OxyGen 仓库：https://github.com/air-embodied-brain/OxyGen
- OxyGen KV 管理：https://github.com/air-embodied-brain/OxyGen/blob/1e8890681090c7f423f85866b5db877916ba65e6/src/openpi/models/kv_cache_manager.py
- OxyGen π0.5 模型：https://github.com/air-embodied-brain/OxyGen/blob/1e8890681090c7f423f85866b5db877916ba65e6/src/openpi/models/pi05.py
- OxyGen 策略入口：https://github.com/air-embodied-brain/OxyGen/blob/1e8890681090c7f423f85866b5db877916ba65e6/src/openpi/policies/policy.py
- OxyGen 实验说明：https://github.com/air-embodied-brain/OxyGen/blob/1e8890681090c7f423f85866b5db877916ba65e6/experiments/Experiments.md
- openpi 仓库：https://github.com/Physical-Intelligence/openpi
- openpi π0 模型：https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py

其他源码与组件：

- Whisper 音频前处理：https://github.com/openai/whisper/blob/v20250625/whisper/audio.py
- Qwen-VL 视频/图像工具：https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
- DeepSeek 推理系统（Day 6）：https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
- FlashMLA（最后支持 Hopper 的 commit）：https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7
- DeepGEMM：https://github.com/deepseek-ai/DeepGEMM
- DeepEP：https://github.com/deepseek-ai/DeepEP
- FlashAttention：https://github.com/Dao-AILab/flash-attention
- FlashInfer：https://github.com/flashinfer-ai/flashinfer

### 学习用代码仓库

- nano-vllm（约 1.2k 行）：https://github.com/GeeeekExplorer/nano-vllm
- mini-sglang（约 5k 行）：https://github.com/sgl-project/mini-sglang
- gpt-fast：https://github.com/meta-pytorch/gpt-fast
- minbpe：https://github.com/karpathy/minbpe
- nanoGPT：https://github.com/karpathy/nanoGPT
- nanochat：https://github.com/karpathy/nanochat
- Awesome-ML-SYS-Tutorial（中文）：https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial
- 本学习包代码实验：`../labs/`（说明见 `../labs/README.md`）

### 工具

- uv：https://docs.astral.sh/uv/
- Hugging Face CLI：https://huggingface.co/docs/huggingface_hub/en/guides/cli
- nccl-tests：https://github.com/NVIDIA/nccl-tests
- nccl-tests 带宽说明：https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
- Nsight Systems：https://developer.nvidia.com/nsight-systems
- Nsight Systems 用户手册：https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- Nsight Compute：https://developer.nvidia.com/nsight-compute
- Perfetto（打开 trace）：https://ui.perfetto.dev
- nvidia-smi 文档：https://docs.nvidia.com/deploy/nvidia-smi/index.html
- CUDA 驱动版本对应：https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- vLLM Docker 镜像：https://hub.docker.com/r/vllm/vllm-openai
- SGLang Docker 镜像：https://hub.docker.com/r/lmsysorg/sglang

### 模型卡

- Qwen3-0.6B：https://huggingface.co/Qwen/Qwen3-0.6B
- Qwen3-8B：https://huggingface.co/Qwen/Qwen3-8B
- Qwen3-32B：https://huggingface.co/Qwen/Qwen3-32B
- Qwen3-30B-A3B：https://huggingface.co/Qwen/Qwen3-30B-A3B
- Qwen2.5-VL-7B-Instruct：https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- Qwen3-VL-8B-Instruct：https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- whisper-large-v3：https://huggingface.co/openai/whisper-large-v3
- Qwen2.5-Omni-7B：https://huggingface.co/Qwen/Qwen2.5-Omni-7B
- DeepSeek-V2-Lite-Chat：https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat
- Llama-3.1-8B-Instruct（gated，需申请）：https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

### 数据集

- ShareGPT（vLLM 压测文档所用）：https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
- 直接下载文件：https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

### 课程与文档

- Stanford CS336：http://cs336.stanford.edu/
- CMU 11-868 LLM Systems：https://llmsystem.github.io/
- MIT 6.5940 EfficientML：https://hanlab.mit.edu/course
- GPU MODE 讲座：https://github.com/gpu-mode/lectures
- HF 推理优化文档：https://huggingface.co/docs/transformers/main/en/llm_optims
- LLM Inference Handbook：https://handbook.modular.com/

### 论文（建议先读前 8 篇）

1. PagedAttention / vLLM：https://arxiv.org/abs/2309.06180
2. Orca：https://www.usenix.org/conference/osdi22/presentation/yu
3. FlashAttention：https://arxiv.org/abs/2205.14135
4. Sarathi-Serve：https://arxiv.org/abs/2403.02310
5. DistServe：https://arxiv.org/abs/2401.09670
6. SGLang / RadixAttention：https://arxiv.org/abs/2312.07104
7. Speculative Decoding（Leviathan 等）：https://arxiv.org/abs/2211.17192
8. Mooncake：https://arxiv.org/abs/2407.00079

扩展阅读：

- Efficiently Scaling Transformer Inference（Pope 等）：https://arxiv.org/abs/2211.05102
- FlashAttention-2：https://arxiv.org/abs/2307.08691
- FlashAttention-3：https://arxiv.org/abs/2407.08608
- FlashInfer：https://arxiv.org/abs/2501.01005
- Splitwise：https://arxiv.org/abs/2311.18677
- Medusa：https://arxiv.org/abs/2401.10774
- EAGLE：https://arxiv.org/abs/2401.15077
- EAGLE-3：https://arxiv.org/abs/2503.01840
- Llumnix：https://arxiv.org/abs/2406.03243
- Megatron-LM：https://arxiv.org/abs/1909.08053
- DeepSeek-V3 技术报告：https://arxiv.org/abs/2412.19437
- XGrammar：https://arxiv.org/abs/2411.15100
- OxyGen：https://arxiv.org/abs/2603.14371
- π0：https://arxiv.org/abs/2410.24164
- π0.5：https://arxiv.org/abs/2504.16054
- Qwen3-Omni 技术报告：https://arxiv.org/abs/2509.17765
