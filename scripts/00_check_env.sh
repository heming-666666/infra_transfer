#!/usr/bin/env bash
# 第 0 步：检查 GPU、驱动、拓扑、磁盘。只读，不修改任何东西。
# 用法：bash scripts/00_check_env.sh
set -u
source "$(dirname "$0")/config.sh"

echo "== 1. GPU 列表（应看到 4 张 H800）"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

echo "== 2. 驱动版本检查"
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
major=${drv%%.*}
echo "driver=$drv"
if [ "$major" -ge 580 ]; then
  echo "OK: 驱动 >= 580，可运行 CUDA 13（SGLang 0.5.21 必需）"
else
  echo "WARN: 驱动 < 580。SGLang 0.5.21 只支持 CUDA 13，需要驱动 >= 580；"
  echo "      vLLM 可改用 cu129 版本（驱动 >= 575.51.03）。不要自行升级驱动，先问管理员。"
fi

echo "== 3. GPU 互联拓扑（NV# 表示 NVLink；PIX/SYS 表示走 PCIe）"
nvidia-smi topo -m

echo "== 4. 磁盘空间（HF_HOME=$HF_HOME）"
df -h "$HF_HOME"

echo "== 5. 工具"
for t in git uv python3 docker nvcc wget; do
  printf "%-8s " "$t"; command -v "$t" || echo "(未安装)"
done
