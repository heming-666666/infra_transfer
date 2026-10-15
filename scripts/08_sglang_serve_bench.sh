#!/usr/bin/env bash
# 第 8 步：SGLang 启动 + 压测（与 vLLM 对比用）。需要驱动 >= 580（CUDA 13）。
# 用法：
#   bash scripts/08_sglang_serve_bench.sh serve [模型] [GPU编号] [额外参数...]
#   bash scripts/08_sglang_serve_bench.sh bench [模型]
# 注意：SGLang 压测默认忽略 EOS、warmup=1；vLLM 默认相反，对比前要对齐（见指南）。
set -euo pipefail
source "$(dirname "$0")/config.sh"
source "$WORK/envs/sglang/bin/activate"
mode="${1:-serve}"; model="${2:-$MODEL_MAIN}"

if [ "$mode" = "serve" ]; then
  gpus="${3:-0}"; shift $(( $# >= 3 ? 3 : $# ))
  export CUDA_VISIBLE_DEVICES="$gpus"
  python -m sglang.launch_server --model-path "$model" --port 30000 \
    --enable-metrics "$@" 2>&1 | tee "$WORK/logs/sglang_serve_$(date +%H%M%S).log"
else
  out="$WORK/results/sglang_$(date +%H%M%S)"; mkdir -p "$out"
  for c in 1 4 16 64 128; do
    python -m sglang.benchmark.serving --backend sglang \
      --dataset-name random --random-input-len 1024 --random-output-len 256 \
      --num-prompts $((c * 10)) --max-concurrency "$c" \
      | tee "$out/concurrency_$c.txt"
  done
  echo "结果在 $out"
fi
