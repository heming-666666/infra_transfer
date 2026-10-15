#!/usr/bin/env bash
# 第 1 步：创建两个互相隔离的 Python 环境（vLLM 与 SGLang 的依赖可能冲突）。
#   $WORK/envs/vllm   : vLLM 0.31.0（也用来跑 labs）
#   $WORK/envs/sglang : SGLang 0.5.21（需要 CUDA 13 / 驱动 >= 580）
# 前提：已安装 uv（https://docs.astral.sh/uv/）。
# 用法：bash scripts/01_setup_envs.sh [vllm|sglang|all]   默认 all
set -euo pipefail
source "$(dirname "$0")/config.sh"
which_env="${1:-all}"

if [[ "$which_env" == "vllm" || "$which_env" == "all" ]]; then
  uv venv "$WORK/envs/vllm" --python "$PY_VERSION"
  source "$WORK/envs/vllm/bin/activate"
  # --torch-backend=auto：按本机驱动自动选择 CUDA 版本的 torch（vLLM 官方安装文档）
  uv pip install "vllm==$VLLM_VERSION" --torch-backend=auto
  vllm download-kernels          # 官方文档要求的安装后步骤
  hf --version                    # hf 命令行随 vLLM 的依赖 huggingface_hub 一起装好
  python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
  deactivate
fi

if [[ "$which_env" == "sglang" || "$which_env" == "all" ]]; then
  uv venv "$WORK/envs/sglang" --python "$PY_VERSION"
  source "$WORK/envs/sglang/bin/activate"
  # SGLang 官方安装文档写法（需要 --prerelease=allow）
  uv pip install --prerelease=allow "sglang==$SGLANG_VERSION"
  python -c "import sglang, torch; print('sglang', sglang.__version__, 'torch', torch.__version__)"
  deactivate
fi
echo "完成。激活方式：source $WORK/envs/vllm/bin/activate"
