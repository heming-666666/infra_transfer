# 模型推理 Infra 七天学习包

生成日期：2026-10-15。面向：已复现 OxyGen、有 4×H800 的 infra 初学者。

## 目录里有什么

| 路径 | 给谁看 | 内容 |
|---|---|---|
| `reports/模型推理 Infra 七天学习指南.md` | 人 | 学习指南正文（七天计划、原理、实验、验收） |
| `reports/模型推理 Infra 七天学习指南.pdf` | 人 | 上面的 PDF 版 |
| `AGENT_SETUP.md` | agent | 在 H800 服务器上一步步配置环境的操作手册 |
| `scripts/` | agent / 人 | 配置与实验脚本（Linux bash），按编号顺序运行 |
| `labs/` | 人 | 10 个代码实验，每个都有 `--test` 自检 |
| `research_notes/` | 参考 | 指南依据的研究笔记（含来源链接） |
| `build/` | 维护者 | 把 Markdown 转成 PDF 的模板和过滤器 |

## 最快开始（在 H800 服务器上）

```bash
# 1. 把整个目录拷到服务器，例如放在 ~/infra7/guide
# 2. 依次运行：
bash scripts/00_check_env.sh            # 检查 GPU/驱动/拓扑/磁盘
bash scripts/01_setup_envs.sh           # 建 vLLM 和 SGLang 两个环境
bash scripts/02_clone_repos.sh          # 克隆要读的源码仓库
bash scripts/03_download_models.sh core # 下载核心模型（约 37 GB）
bash scripts/05_run_labs.sh             # 跑 labs 自检
```

详细步骤、预期输出和出错处理见 `AGENT_SETUP.md`。

## 脚本一览

| 脚本 | 作用 |
|---|---|
| `config.sh` | 唯一需要修改的配置（工作目录、版本、默认模型） |
| `00_check_env.sh` | 只读检查：GPU、驱动 ≥580、`nvidia-smi topo -m`、磁盘 |
| `01_setup_envs.sh` | uv 建环境：vLLM 0.31.0、SGLang 0.5.21 |
| `02_clone_repos.sh` | 克隆 vLLM、SGLang、nano-vllm、mini-sglang、gpt-fast、nccl-tests、openpi、OxyGen |
| `03_download_models.sh` | 下载模型（core / extended / all 三档）和 ShareGPT |
| `04_nccl_tests.sh` | 4 卡 all_reduce 带宽测试 |
| `05_run_labs.sh` | 在 GPU 上跑 10 个 labs 自检 |
| `06_vllm_serve.sh` | 启动 vLLM 服务（可选 TP/DP/关闭前缀缓存） |
| `07_vllm_bench.sh` | 对 vLLM 服务做并发扫描压测 |
| `08_sglang_serve_bench.sh` | 启动 SGLang 服务 / 压测 |

## 验证状态（诚实说明）

- `labs/`：在本机（Windows，CPU 与 RTX 4070 Ti SUPER）实际运行，全部 `--test` 通过；
  NCCL/torchrun/nsys/H800 部分未在本机运行。详见 `labs/README.md`。
- `scripts/`：只做了 `bash -n` 语法检查和辅助函数测试，**未在 H800 上运行**。
  命令写法来自 2026-10-14/15 核实的官方文档（vLLM 0.31.0、SGLang 0.5.21、nccl-tests、hf CLI）。
- 指南 PDF（40 页）：由 `bash build/build_pdf.sh` 生成，已逐页渲染检查（无越界文字、目录页码正确）。
- 指南中引用的 labs 代码摘录与 `labs/*.py` 逐行一致；Day 2 的手写 prefill/decode 循环
  已用随机小 Qwen3 在本机验证，生成结果与 `generate()` 完全一致（`build/check_day2_loop.py`）。
- 版本会变：运行前以当天 `vllm --version`、`pip show sglang` 和官方文档为准。
