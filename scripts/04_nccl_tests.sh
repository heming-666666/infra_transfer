#!/usr/bin/env bash
# 第 4 步：编译 nccl-tests，测 4 卡 all_reduce 带宽（理解 TP 通信开销）。
# 前提：02_clone_repos.sh 已克隆 nccl-tests；系统有 nvcc 和 NCCL。
# 用法：bash scripts/04_nccl_tests.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$WORK/repos/nccl-tests"
[ -x build/all_reduce_perf ] || make -j
# -b 起始大小 -e 结束大小 -f 倍增因子 -g 使用的 GPU 数（nccl-tests README）
./build/all_reduce_perf -b 8 -e 4G -f 2 -g 4 | tee "$WORK/results/nccl_allreduce_4gpu.txt"
echo "看最后几行的 busbw（GB/s）。busbw = algbw × 2(n-1)/n。"
