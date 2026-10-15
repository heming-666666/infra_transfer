#!/usr/bin/env bash
# 第 7 步：对已启动的 vLLM 服务做并发扫描压测（另开一个终端运行）。
# 前提：06_vllm_serve.sh 正在运行。
# 用法：bash scripts/07_vllm_bench.sh [模型] [结果子目录名]
# 输出：$WORK/results/<名字>/ 下的 JSON（含 TTFT/TPOT/ITL/吞吐）
set -euo pipefail
source "$(dirname "$0")/config.sh"
source "$WORK/envs/vllm/bin/activate"
model="${1:-$MODEL_MAIN}"; name="${2:-bench_$(date +%H%M%S)}"
out="$WORK/results/$name"; mkdir -p "$out"

for c in 1 4 16 64 128; do
  echo "== 并发 $c"
  vllm bench serve --backend vllm --model "$model" \
    --dataset-name random --random-input-len 1024 --random-output-len 256 \
    --num-prompts $((c * 10)) --max-concurrency "$c" \
    --ignore-eos --num-warmups 10 \
    --save-result --result-dir "$out"
done
echo "结果在 $out"
