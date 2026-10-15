#!/usr/bin/env bash
# 第 3 步：下载模型权重和基准数据集到 $HF_HOME。
#   core     ≈ 37 GB ：Qwen3-0.6B, Qwen3-8B, Qwen2.5-VL-7B, whisper-large-v3 + ShareGPT
#   extended ≈ 125 GB：再加 Qwen3-32B, Qwen3-30B-A3B（第 5 天并行实验用）
#   all      ≈ 320 GB：再加 Qwen3-VL-8B, Qwen2.5-Omni-7B, DeepSeek-V2-Lite-Chat
# 用法：bash scripts/03_download_models.sh [core|extended|all]   默认 core
set -euo pipefail
source "$(dirname "$0")/config.sh"
source "$WORK/envs/vllm/bin/activate"     # 使用其中安装的 hf 命令
tier="${1:-core}"

CORE=(Qwen/Qwen3-0.6B Qwen/Qwen3-8B Qwen/Qwen2.5-VL-7B-Instruct openai/whisper-large-v3)
EXTENDED=(Qwen/Qwen3-32B Qwen/Qwen3-30B-A3B)
ALL=(Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen2.5-Omni-7B deepseek-ai/DeepSeek-V2-Lite-Chat)

models=("${CORE[@]}")
[[ "$tier" == "extended" || "$tier" == "all" ]] && models+=("${EXTENDED[@]}")
[[ "$tier" == "all" ]] && models+=("${ALL[@]}")

for m in "${models[@]}"; do
  echo "== 下载 $m"
  hf download "$m"            # 下载到 $HF_HOME/hub，vLLM/SGLang 会自动找到
done

# ShareGPT 数据集（vLLM 基准文档使用的 URL）
mkdir -p "$WORK/data"
f="$WORK/data/ShareGPT_V3_unfiltered_cleaned_split.json"
[ -f "$f" ] || wget -O "$f" \
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
du -sh "$HF_HOME" "$WORK/data"
