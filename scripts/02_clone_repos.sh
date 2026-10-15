#!/usr/bin/env bash
# 第 2 步：把要读的源码仓库克隆到 $WORK/repos，并切到指南引用的版本。
# 已存在的目录会跳过，不会覆盖。
# 用法：bash scripts/02_clone_repos.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"
mkdir -p "$WORK/repos" && cd "$WORK/repos"

# 格式：目录名  仓库URL  版本（tag/commit；main 表示最新）
REPOS=(
  "vllm        https://github.com/vllm-project/vllm.git            v0.31.0"
  "sglang      https://github.com/sgl-project/sglang.git           v0.5.21"
  "nano-vllm   https://github.com/GeeeekExplorer/nano-vllm.git     main"
  "mini-sglang https://github.com/sgl-project/mini-sglang.git      main"
  "gpt-fast    https://github.com/meta-pytorch/gpt-fast.git        main"
  "nccl-tests  https://github.com/NVIDIA/nccl-tests.git            master"
  "openpi      https://github.com/Physical-Intelligence/openpi.git 215abfb217dbac7d5f1273282331b9b1866c0479"
  "OxyGen      https://github.com/air-embodied-brain/OxyGen.git    1e8890681090c7f423f85866b5db877916ba65e6"
)

for line in "${REPOS[@]}"; do
  read -r name url rev <<< "$line"
  if [ -d "$name" ]; then echo "跳过（已存在）：$name"; continue; fi
  # GIT_LFS_SKIP_SMUDGE=1：不拉取大文件（OxyGen/openpi README 的做法）
  GIT_LFS_SKIP_SMUDGE=1 git clone "$url" "$name"
  if [ "$rev" != "main" ] && [ "$rev" != "master" ]; then
    git -C "$name" checkout --quiet "$rev"
  fi
  echo "OK: $name @ $(git -C "$name" rev-parse --short HEAD)"
done
