#!/usr/bin/env bash
# 第 5 步：在 H800 上运行全部 labs 自检（每个打印 PASS/FAIL）。
# 前提：已把指南的 labs/ 目录拷到 $LABS_DIR；01_setup_envs.sh 已建好 vllm 环境。
# 用法：bash scripts/05_run_labs.sh
set -uo pipefail
source "$(dirname "$0")/config.sh"
source "$WORK/envs/vllm/bin/activate"
cd "$LABS_DIR"
log="$WORK/logs/labs_$(date +%Y%m%d_%H%M).log"
fail=0

run() {
  echo "== $*" | tee -a "$log"
  "$@" >> "$log" 2>&1; rc=$?
  tail -n 2 "$log"
  [ "$rc" -eq 0 ] || { echo "FAIL (退出码 $rc)"; fail=1; }
}

run python lab01_bpe_toy.py --test
run python lab02_tiny_llama.py --device cuda --test
run python lab03_hf_equivalence.py --device cuda --test
run python lab04_sampling.py --device cuda --test
run python lab05_paged_kv.py --device cuda --test
run python lab06_scheduler_sim.py --test
run torchrun --nproc_per_node=4 lab07_tensor_parallel.py --backend nccl --test
run python lab08_capacity_calculator.py --test
run python lab09_profile.py --device cuda --test
run python lab10_multimodal_toy.py --device cuda --test

echo "日志：$log"
[ "$fail" -eq 0 ] && echo "全部通过" || { echo "有失败项，查看日志"; exit 1; }
