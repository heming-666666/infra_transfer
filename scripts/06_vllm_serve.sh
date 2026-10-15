#!/usr/bin/env bash
# 第 6 步：启动 vLLM OpenAI 兼容服务（前台运行，Ctrl+C 停止）。
# 用法：bash scripts/06_vllm_serve.sh [模型] [GPU编号] [额外参数...]
# 例子：
#   bash scripts/06_vllm_serve.sh Qwen/Qwen3-8B 0                          # 单卡
#   bash scripts/06_vllm_serve.sh Qwen/Qwen3-8B 0,1,2,3 -tp=4              # 4 卡张量并行
#   bash scripts/06_vllm_serve.sh Qwen/Qwen3-8B 0,1,2,3 -dp=4              # 4 卡数据并行
#   bash scripts/06_vllm_serve.sh Qwen/Qwen3-8B 0 --no-enable-prefix-caching
set -euo pipefail
source "$(dirname "$0")/config.sh"
source "$WORK/envs/vllm/bin/activate"
model="${1:-$MODEL_MAIN}"; gpus="${2:-0}"; shift $(( $# >= 2 ? 2 : $# ))
export CUDA_VISIBLE_DEVICES="$gpus"
echo "模型=$model GPU=$gpus 额外参数=$*  服务地址 http://localhost:8000"
vllm serve "$model" --port 8000 "$@" 2>&1 | tee "$WORK/logs/vllm_serve_$(date +%H%M%S).log"
