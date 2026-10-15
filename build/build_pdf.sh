#!/usr/bin/env bash
# 把 reports/ 下的指南 Markdown 转成 PDF，并把每页渲染成 PNG 供版面检查。
# 依赖：pandoc ≥3.9、xelatex（ctex/xeCJK/fvextra）、pdftoppm、Microsoft YaHei/Consolas/Segoe UI 字体
# 用法：bash build/build_pdf.sh
set -euo pipefail
cd "$(dirname "$0")/.."
T="模型推理 Infra 七天学习指南"

pandoc "reports/$T.md" -f markdown+autolink_bare_uris \
  --lua-filter=build/breakcode.lua -H build/header.tex \
  --pdf-engine=xelatex -o "reports/$T.pdf" \
  -V mainfont="Segoe UI" -V geometry:margin=2cm -V papersize=a4 \
  -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
  --shift-heading-level-by=-1 --toc --toc-depth=1 -V toc-title="目录" \
  -V fontsize=11pt --syntax-highlighting=tango \
  -M title="从 OxyGen 出发，七天打通模型推理 Infra" -M date="2026-10-15" \
  2>&1 | grep -v -E '^$|elevated privileges|MiKTeX administrator' || true

pdfinfo "reports/$T.pdf" | grep -E 'Pages|File size'
rm -rf build/pages && mkdir -p build/pages
pdftoppm -r 60 -png "reports/$T.pdf" build/pages/p 2>/dev/null
ls build/pages | wc -l
