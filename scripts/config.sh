#!/usr/bin/env bash
# 所有脚本共用的配置。只改这个文件即可，其他脚本不用改。
# 用法：其他脚本开头会执行 `source "$(dirname "$0")/config.sh"`

# 工作根目录：仓库、虚拟环境、日志、结果都放在这里
export WORK="${WORK:-$HOME/infra7}"

# Hugging Face 缓存目录（模型权重下载到这里，约需 160–320 GB 磁盘）
export HF_HOME="${HF_HOME:-$WORK/hf_home}"

# 版本锁定（2026-10-15 核实的最新稳定版）
export VLLM_VERSION="0.31.0"
export SGLANG_VERSION="0.5.21"
export PY_VERSION="3.12"          # vLLM 文档：3.11–3.14；SGLang 文档：>=3.11

# 本指南的 labs 目录（从 Windows 拷到服务器后的位置）
export LABS_DIR="${LABS_DIR:-$WORK/guide/labs}"

# 默认实验模型
export MODEL_SMALL="Qwen/Qwen3-0.6B"
export MODEL_MAIN="Qwen/Qwen3-8B"

mkdir -p "$WORK" "$HF_HOME" "$WORK/logs" "$WORK/results"
