# 模型推理 Infra 代码实验（labs）

10 个自包含的小实验，从分词一路走到多模态输入，配合《模型推理 Infra 七天学习指南》使用。
全部使用随机权重或合成数据，不需要下载任何模型或数据集。每个实验都有 `--test`
自检模式，逐行打印 `PASS` / `FAIL`，有失败时退出码非 0。

## 索引

| 实验 | 讲什么 | 本机运行命令（在 `labs/` 目录下） | 本机测试状态 | 到 H800 上做什么 |
|---|---|---|---|---|
| lab01_bpe_toy.py | 字节级 BPE 训练/编码/解码；流式反分词扣住不完整 UTF-8 | `python lab01_bpe_toy.py --test` | CPU 8/8 PASS（两个环境） | 无需 GPU；可换成真实分词器对照 |
| lab02_tiny_llama.py | Llama/Qwen3 结构、RoPE、GQA、prefill / decode_step、KV cache 字节数 | `python lab02_tiny_llama.py --device cpu --test` | CPU 4/4 PASS；CUDA(4070 Ti S) 4/4 PASS | `--device cuda`，加大 `--bench-prompt` 看 prefill/decode 差异 |
| lab03_hf_equivalence.py | 把 HF Llama/Qwen3 权重映射进 lab02，logits 逐位对齐 | `python lab03_hf_equivalence.py --device cuda --test` | CPU 4/4、CUDA 4/4 PASS（需 transformers，仅 GPU 环境有）；`--pretrained <本地目录>` 已在随机小模型上演练 | `--pretrained <hf_id>` 加载真实小模型对比（**未在本机运行**） |
| lab04_sampling.py | 重复惩罚、温度、top-k、top-p、min-p、指数竞赛采样 | `python lab04_sampling.py --device cpu --test` | CPU 16/16；CUDA 16/16 PASS | 无需改动 |
| lab05_paged_kv.py | 块池/引用计数/块表/slot mapping/paged attention/前缀缓存+LRU | `python lab05_paged_kv.py --device cpu --test` | CPU 9/9；CUDA 9/9 PASS | 对照 vLLM 的 KV cache 管理源码阅读 |
| lab06_scheduler_sim.py | 静态批 vs 连续批 vs 分块预填充；TTFT/TPOT；抢占重算 | `python lab06_scheduler_sim.py --test` | 13/13 PASS（两个环境） | 用 vLLM/SGLang 实测，比较趋势是否一致（成本模型只是示意） |
| lab07_tensor_parallel.py | 列并行/行并行 + all_reduce；all_reduce 带宽 | `python lab07_tensor_parallel.py --spawn 2 --backend gloo --test` | 2 进程 gloo：2/2 PASS（两个环境） | `torchrun --nproc_per_node=4 lab07_tensor_parallel.py --backend nccl --test`（**未在本机运行**） |
| lab08_capacity_calculator.py | 参数显存、KV/token、最大 KV token 数、屋顶线拐点、decode 吞吐上界 | `python lab08_capacity_calculator.py --test` | 13/13 PASS（两个环境） | 传入真实 config.json 与 H800 官方规格数字 |
| lab09_profile.py | torch.profiler 算子表、Chrome/Perfetto trace、NVTX、CUDA events 计时 | `python lab09_profile.py --device cuda --test` | CPU 3/3；CUDA 3/3 PASS + 1 SKIP（本机 CUPTI 不可用，无 GPU kernel 轨迹） | `nsys profile -t cuda,nvtx,osrt -o out/lab09_nsys python lab09_profile.py --device cuda --no-torch-profiler`（**未在本机运行**） |
| lab10_multimodal_toy.py | 图像 patch 化/2×2 合并/投影/占位符替换；STFT + mel 滤波器组 | `python lab10_multimodal_toy.py --device cpu --test` | CPU 11/11；CUDA 11/11 PASS | 可加大图像尺寸、改 n_mels 观察 token 数变化 |

本机测试的命令原文和输出摘录见 `../research_notes/模型推理 Infra 七天学习指南/code_labs.md`，
完整日志在 `out/*.log`。

## 运行环境（本机实测）

- CPU 环境：Python 3.11.15，torch 2.13.0+cpu，numpy 2.3.5（gloo 可用；无 transformers）。
- GPU 环境：Python 3.10.20，torch 2.5.1+cu121，transformers 5.13.0，tokenizers 0.22.2，
  safetensors 0.8.0；NVIDIA GeForce RTX 4070 Ti SUPER 16 GB，驱动 610.88。
- 系统：Windows 11 + Git Bash。代码只用标准库与 PyTorch（lab03 另需 transformers），
  路径用 `pathlib`，可直接在 Linux（H800 服务器）上运行。
- 依赖要求：Python ≥ 3.10，PyTorch ≥ 2.x；lab03 需要 transformers 5.x（4.x 的配置字段名
  如 `rope_theta` 与 5.x 的 `rope_parameters` 不同，未测试）。

## 生成的文件（`out/`）

- `out/*.log`：本机验证日志（`cpu_*` = CPU 环境，`gpu_*` = GPU 环境 + CUDA，
  `gpuenv_*` = GPU 环境的 Python 3.10 跑纯 CPU 实验）。
- `out/lab09_trace_cpu.json`、`out/lab09_trace_cuda.json`：可在 https://ui.perfetto.dev 打开。
- `out/tiny_hf_qwen3/`：`lab03 --save-tiny` 保存的随机小 Qwen3（HF 格式，约 2.7 MB），
  用于离线演练 `--pretrained` 路径，也可作为 lab08 的 config.json 示例。

## 已知限制

1. **Windows 上 torchrun 不可用**：实测 torch 2.13.0+cpu 与 2.5.1+cu121 的
   `python -m torch.distributed.run` 即使设置 `USE_LIBUV=0`，也会在 rendezvous 阶段报
   “use_libuv was requested but PyTorch was built without libuv support”（launcher 内部
   创建 TCPStore 时没有传 `use_libuv`）。因此 lab07 提供 `--spawn N` 自带启动器
   （同样走 `env://` 初始化）。torchrun 命令只在 Linux 上使用，本机未验证。
2. **NCCL / 多卡 / H800 均未在本机运行**：lab07 的 `--backend nccl`、lab09 的 nsys、
   lab03 的 `--pretrained <hf_id>`（需要下载权重）都只做了语法检查（`py_compile`），
   以及能在本机做的替代演练（gloo、本地随机小模型目录）。
3. **本机 torch.profiler 抓不到 GPU kernel**：GPU 环境 `torch.profiler.supported_activities()`
   只返回 CPU（CUPTI 不可用），lab09 会打印警告并把 kernel 检查标为 SKIP；
   GPU 时间以 CUDA events 为准。在 Linux + 正常 CUPTI 的环境中应能看到 kernel。
4. **lab06 的耗时模型是示意性的**：不是任何硬件的实测值，只用于比较调度策略的趋势。
   默认负载下两种连续批的吞吐相同（都已跟上到达速率），差别体现在 TTFT/TPOT 上。
5. **lab08 是估算**：忽略 bias、激活显存（用 `--reserve-gb` 预留）、通信与 kernel 效率；
   MoE 只按 Qwen-MoE 式字段近似，MLA（`kv_lora_rank`）不适用。硬件数字必须自己查官方规格后传入。
6. **lab03 只覆盖 lab02 能表达的子集**：默认 RoPE、无 bias、silu、全注意力层的 Llama/Qwen3；
   遇到 rope scaling（如 llama3 类型）、attention bias、滑窗等配置会直接报错退出。
   `--pretrained` 的容差（fp32 相对误差 1e-3、bf16 5e-2）是经验值，未在真实权重上标定。
7. 计时数字（lab02/07/09）来自本机小模型，主要受 Python/kernel 启动开销支配，
   不能代表 H800 上的性能；GPU 上 decode 比 CPU 还慢正是这个原因。
8. 文件长度：lab02、lab05、lab06 各约 260 行，超过“每个约 200 行”的目标；全部 10 个文件
   共约 2,000 行，每行显示宽度 ≤ 88 列（中文按 2 列计）。
9. Windows 控制台默认 GBK 时，各实验会把 stdout/stderr 重新设为 UTF-8；
   若终端仍显示乱码，可设置 `PYTHONUTF8=1`。
