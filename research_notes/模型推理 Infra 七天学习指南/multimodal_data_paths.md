# 多模态推理数据通路：从像素/帧/波形到 Prefill token，再到语音/图像生成与引擎服务化

> 研究日期 / 访问日期：2026-10-14（所有 URL 均于该日访问）。
> 版本基线（GitHub Releases API 实测）：transformers **v5.19.0**（2026-10-06）、vLLM **v0.31.0**（2026-10-05）、SGLang **v0.5.21**（2026-10-02）、diffusers **v0.41.0**（2026-10-06）、vLLM-Omni **v0.30.0**（2026-09-25）、xDiT **0.7.0**（2026-09-25）、openai/whisper **v20250625**、qwen-vl-utils PyPI **0.0.14**。
> 来源：[transformers releases](https://github.com/huggingface/transformers/releases/tag/v5.19.0)；[vLLM v0.31.0](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)；[SGLang v0.5.21](https://github.com/sgl-project/sglang/releases/tag/v0.5.21)；[diffusers v0.41.0](https://github.com/huggingface/diffusers/releases/tag/v0.41.0)；[vLLM-Omni v0.30.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.30.0)；[xDiT 0.7.0](https://github.com/xdit-project/xDiT/releases/tag/0.7.0)；[whisper v20250625](https://github.com/openai/whisper/releases/tag/v20250625)；[qwen-vl-utils PyPI](https://pypi.org/project/qwen-vl-utils/)
> 方法说明：精确数字主要通过 curl 直接读取官方源码（raw.githubusercontent.com，固定 tag）与 Hugging Face 官方 config.json/preprocessor_config.json 获得，而非二手文章。本轮中断后未完成的部分（扩散/DiT 缓存技巧、SGLang 多模态参数、Gemma 3 报告原文、Qwen2.5-Omni 报告原文、vLLM 设计文档页面）均在各节 Gaps 中标注「未核实」。

---

## Q1. 通用 VLM 推理架构：编码器 → 投影/合并 → LLM；占位 token 与 embedding scatter；多模态位置编码；为何视觉 token 主导 prefill 与 KV cache

### Takeaway
主流 VLM 的推理链路是「CPU 预处理（解码/resize/归一化/切 patch）→ ViT 类视觉编码器（GPU）→ merger/projector 把视觉特征映射到 LLM hidden size 并压缩 token 数 → 在文本 token 序列中预留同等数量的占位 token，用 `masked_scatter` 把视觉 embedding 写入 → 与文本一起做 prefill」。视觉 token 一旦写入就和文本 token 完全一样占用 KV cache；一张 1080p 图在 Qwen2.5-VL 里约 2.7k token、一段 1 分钟 1080p 视频约 4.3 万 token，因此多模态请求的 prefill 计算量与 KV 占用常常被视觉 token 主导。

### Cited Findings
- **占位符 + scatter（Qwen3-VL，transformers v5.19.0）**：processor 把每张图展开为 `image_token * num_image_tokens`，其中 `num_image_tokens = image_grid_thw.prod() // merge_size**2`；模型侧通过 `get_placeholder_mask` 得到掩码并执行 `inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)`（视频同理）。— [processing_qwen3_vl.py](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_vl/processing_qwen3_vl.py)；[modeling_qwen3_vl.py](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
- **Qwen 系特殊 token id**（Qwen2.5-VL-7B / Qwen3-VL-8B 相同）：`vision_start_token_id=151652`、`vision_end_token_id=151653`（Qwen3-VL config）、`image_token_id=151655`、`video_token_id=151656`。— [Qwen2.5-VL-7B config.json](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json)；[Qwen3-VL-8B config.json](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json)
- **Qwen2.5-VL-7B 结构参数**：视觉塔 `depth=32, hidden_size=1280, patch_size=14, spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=3584`，**窗口注意力** `window_size=112`，仅 `fullatt_block_indexes=[7,15,23,31]` 四层做全局注意力；LLM `hidden_size=3584, num_hidden_layers=28, num_attention_heads=28, num_key_value_heads=4, max_position_embeddings=128000`；`rope_scaling: {type: mrope, mrope_section: [16,24,24]}`；`vision_config.tokens_per_second=2`。— [Qwen2.5-VL-7B config.json](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json)
- **Qwen2.5-VL 绝对时间对齐（M-RoPE 时间维）**：`get_rope_index` 中视频时间位置间隔 `time_interval = tokens_per_second * second_per_grid_ts`，注释给出 `interval = tokens_per_second * temporal_patch_size / fps`。— [modeling_qwen2_5_vl.py](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)
- **Qwen3-VL-8B 结构参数**：视觉塔 `depth=27, hidden_size=1152, num_heads=16, patch_size=16, spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=2304, out_hidden_size=4096, deepstack_visual_indexes=[8,16,24]`；LLM `hidden_size=4096, num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8, head_dim=128, max_position_embeddings=262144, rope_theta=5000000`；`rope_scaling: {mrope_interleaved: true, mrope_section: [24,20,20]}`。— [Qwen3-VL-8B config.json](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json)
- **Qwen3-VL Interleaved-MRoPE（源码已核实）**：`recomposition_frequencies` 以 T 维频率为底，把 H 维写入 `slice(1, mrope_section[1]*3, 3)`、W 维写入 `slice(2, mrope_section[2]*3, 3)`，即 T/H/W 在频率维上按 3 交错排布（而非 Qwen2/2.5-VL 的分块 [T|H|W]）。— [modeling_qwen3_vl.py](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
- **Qwen3-VL DeepStack（源码已核实）**：视觉塔在第 8/16/24 层各接一个独立 merger（`deepstack_merger_list`）产出多层视觉特征；LLM 前 `len(deepstack_visual_embeds)`（即 3）层中，`_deepstack_process` 在视觉 token 位置执行 `hidden_states[visual_pos_masks] += visual_embeds`。— [modeling_qwen3_vl.py](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
- **Qwen3-VL 文本时间戳**：视频 prompt 按帧组构造为 `<{t:.1f} seconds>` + `<|vision_start|>` + `frame_seqlen` 个视频 token + `<|vision_end|>`；时间戳取同一 temporal patch 内首末帧时间的平均；`get_rope_index` 注释说明因用时间戳分隔视频，`video_grid_thw` 需按 t 拆开。— [processing_qwen3_vl.py](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_vl/processing_qwen3_vl.py)；[modeling_qwen3_vl.py](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
- **LLaVA-1.5 投影器**：`projector_hidden_act="gelu"`，`vision_feature_layer=-2`（取 CLIP 倒数第二层），`vision_feature_select_strategy="default"`（去掉 CLS）。— [llava-1.5-7b-hf config.json](https://huggingface.co/llava-hf/llava-1.5-7b-hf/blob/main/config.json)
- **InternVL pixel shuffle**：`num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))`，`pixel_shuffle(vit_embeds, scale_factor=0.5)` 把 2×2 相邻 patch 的通道拼接，token 数降为 1/4；文本侧占位符为 `<IMG_CONTEXT>`。— [InternVL3-8B modeling_internvl_chat.py](https://huggingface.co/OpenGVLab/InternVL3-8B/blob/main/modeling_internvl_chat.py)
- **Gemma 3 池化**：`patches_per_image = image_size // patch_size`，`tokens_per_side = sqrt(mm_tokens_per_image)`，`kernel_size = patches_per_image // tokens_per_side`，用 `nn.AvgPool2d` 平均池化；`mm_tokens_per_image` 默认 256。— [modeling_gemma3.py](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/gemma3/modeling_gemma3.py)；[configuration_gemma3.py](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/gemma3/configuration_gemma3.py)

### Inferences
- **每 token KV 字节数（BF16，推导）**：KV/token = 2(K,V) × 层数 × KV头数 × head_dim × 2B。
  - Qwen2.5-VL-7B：2×28×4×128×2 = **57,344 B = 56 KiB/token**（head_dim=3584/28=128）。
  - Qwen3-VL-8B：2×36×8×128×2 = **147,456 B = 144 KiB/token**。
  - Qwen3-Omni-30B-A3B Thinker：2×48×4×128×2 = **98,304 B = 96 KiB/token**（参数见 Q5）。
- **示例**：1920×1080 图片 → Qwen2.5-VL 约 2,691 token ≈ 147 MiB KV；Qwen3-VL-8B 约 2,040 token ≈ 287 MiB KV（计算见 Q2）。对比 LLaVA-1.5 恒定 576 token。这说明「动态分辨率」模型的 KV 与 prefill FLOPs 随输入分辨率线性（按像素）增长，服务端必须用 max_pixels/token 预算来控成本。
- **编码器成本**：Qwen2.5-VL 的 ViT 在合并前处理 4 倍于 LLM 视觉 token 的 patch（1080p 例中 10,764 个 patch），全局注意力为 O(n²)；这正是 Qwen2.5-VL 只保留 4 层全局注意力、其余用 112 px 窗口注意力的工程动机（推断，依据为上述 config 字段）。
- **DeepStack 对服务的影响（推断）**：DeepStack 使一次视觉编码产出「主特征 + 3 份层级特征」，引擎的 encoder cache/embedding 传输体积相应增大，且这些特征必须在 LLM 前 3 层被注入，不能仅靠「普通 token embedding」表示。
- **M-RoPE 对 prefix caching/解码的影响（推断）**：多模态位置 id 是 3 维的，图像 token 的 position 不随 token 序号线性递增，解码阶段需要 `rope_deltas` 偏移（modeling 中可见 `self.rope_deltas`），引擎实现需为每个请求保存该偏移。

### Gaps
- Qwen3-VL 技术报告原文（对 Interleaved-MRoPE / DeepStack / 文本时间戳动机的官方表述）本轮未抓取；以上仅以 transformers 源码与 config 为证据。
- vLLM 中 Qwen3-VL DeepStack 特征如何存入 encoder cache（是否沿 hidden 维拼接）未核实。

---

## Q2. 图像预处理的精确数字（Qwen2.5-VL / Qwen3-VL / LLaVA-1.5 / LLaVA-NeXT / InternVL / Gemma 3）

### Takeaway
Qwen 系是「原生动态分辨率」：边长取整到 28（Qwen2/2.5-VL：14 px patch × 2 合并）或 32（Qwen3-VL：16 px × 2），每个 LLM 视觉 token 对应 28×28=784 或 32×32=1024 像素；LLaVA-1.5 固定 576；LLaVA-NeXT/InternVL 走「切块（tiling）+ 缩略图」；Gemma 3 每图（每 crop）固定 256 token。

### Cited Findings
- **Qwen2-VL/2.5-VL `smart_resize`（transformers v5.19.0）**：签名 `smart_resize(height, width, factor=28, min_pixels=56*56, max_pixels=14*14*4*1280)`；宽高比 >200 报错；`h_bar = round(h/factor)*factor`；若超过 max_pixels：`beta = sqrt(h*w/max_pixels)`，`h_bar = max(factor, floor(h/beta/factor)*factor)`；若低于 min_pixels：`beta = sqrt(min_pixels/(h*w))`，`ceil` 取整。类默认 `size={"shortest_edge": 56*56, "longest_edge": 28*28*1280}`、`resample=BICUBIC`、`patch_size=14, temporal_patch_size=2, merge_size=2`、`image_mean/std = OPENAI_CLIP_MEAN/STD`。单图在 temporal 维被 `expand` 复制成 2 帧，每个 patch 展平维度 `channel*temporal_patch_size*patch_size*patch_size`（=3×2×14×14=1176）。v5.19.0 中向 processor 调用传 `min_pixels/max_pixels` 已被标注弃用，将于 **v5.23** 移除，应改传 `size={'longest_edge':..., 'shortest_edge':...}`。— [image_processing_qwen2_vl.py @v5.19.0](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)
- **Qwen2.5-VL-7B-Instruct 实际预处理配置**：`min_pixels=3136`（=56²，即 4 个 token）、`max_pixels=12845056`（=16384×28²，即 16384 个 token）、`patch_size=14, temporal_patch_size=2, merge_size=2`、`image_mean=[0.48145466, 0.4578275, 0.40821073]`、`image_std=[0.26862954, 0.26130258, 0.27577711]`（OpenAI CLIP 均值方差）。注意：模型仓库的 max_pixels 比 transformers 类默认（1280 token）大得多。— [Qwen2.5-VL-7B preprocessor_config.json](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/preprocessor_config.json)
- **qwen-vl-utils（官方参考实现，QwenLM/Qwen3-VL 仓库，最后修改 commit fe12058 / 2025-11-17）**：`MAX_RATIO=200, SPATIAL_MERGE_SIZE=2, IMAGE_MIN_TOKEN_NUM=4, IMAGE_MAX_TOKEN_NUM=16384`；`fetch_image(..., image_patch_size=14)`，`patch_factor = image_patch_size*2`，默认 `min_pixels = 4*patch_factor²`、`max_pixels = 16384*patch_factor²`（Qwen3-VL 需传 `image_patch_size=16`）。— [vision_process.py](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py)
- **Qwen3-VL-8B-Instruct 预处理配置**：`size={"shortest_edge": 65536, "longest_edge": 16777216}`、`patch_size=16, temporal_patch_size=2, merge_size=2`、`image_mean=[0.5,0.5,0.5]`、`image_std=[0.5,0.5,0.5]`、`image_processor_type="Qwen2VLImageProcessorFast"`。— [Qwen3-VL-8B preprocessor_config.json](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/preprocessor_config.json)
- **LLaVA-1.5（llava-hf/llava-1.5-7b-hf）**：CLIP `image_size=336, patch_size=14`；预处理 `CLIPImageProcessor`，`size.shortest_edge=336` 后 `center_crop 336×336`、`resample=3`（BICUBIC）、`rescale_factor=1/255`、CLIP 均值方差同上；processor `num_additional_image_tokens=1`、`vision_feature_select_strategy="default"`；token 数公式 `(h//patch)*(w//patch) + num_additional_image_tokens`，default 策略再减 1 → (336/14)² = **576**。— [preprocessor_config.json](https://huggingface.co/llava-hf/llava-1.5-7b-hf/blob/main/preprocessor_config.json)；[processor_config.json](https://huggingface.co/llava-hf/llava-1.5-7b-hf/blob/main/processor_config.json)；[processing_llava.py](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/llava/processing_llava.py)
- **LLaVA-NeXT AnyRes（llava-hf/llava-v1.6-mistral-7b-hf）**：`image_grid_pinpoints=[[336,672],[672,336],[672,672],[1008,336],[336,1008]]`，tile 为 336；token 数 = 基础图 `patches_h*patches_w(+CLS 处理)` + 高分辨率网格「去 padding 后」特征数 `unpadded_features` + 每行一个换行 token `newline_features=current_height`（`_get_number_of_features` / `_get_unpadded_features`）。— [config.json](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/blob/main/config.json)；[processing_llava_next.py](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/llava_next/processing_llava_next.py)
- **InternVL3-8B 动态切块**：`force_image_size=448, patch_size=14, downsample_ratio=0.5, ps_version="v2", dynamic_image_size=true, min_dynamic_patch=1, max_dynamic_patch=12, use_thumbnail=true`；`dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail)` 枚举所有 i×j≤12 的网格，选最接近原宽高比者，resize 后切成 448×448 块，块数>1 时追加一张 448×448 缩略图；归一化用 ImageNet `MEAN=(0.485,0.456,0.406)`、`STD=(0.229,0.224,0.225)`；视频示例 `load_video(..., max_num=1, num_segments=32)`（每帧 1 块、均匀采 32 段）。— [InternVL3-8B config.json](https://huggingface.co/OpenGVLab/InternVL3-8B/blob/main/config.json)；[InternVL3-8B README（模型卡代码）](https://huggingface.co/OpenGVLab/InternVL3-8B)
- **Gemma 3（transformers v5.19.0）**：processor `image_seq_length=256`，每张图展开为 `"\n\n" + <boi> + 256×<image> + <eoi> + "\n\n"`；Pan & Scan 默认 `do_pan_and_scan=False, pan_and_scan_min_crop_size=256, pan_and_scan_max_num_crops=4, pan_and_scan_min_ratio_to_activate=1.2`；启用后 prompt 为 `"Here is the original image {seq} and here are some crops to help you see better " + num_crops × {seq}`；图像处理器 `resample=BILINEAR`、`image_mean/std = IMAGENET_STANDARD_MEAN/STD`（即 0.5/0.5）；config `boi_token_index=255999, eoi_token_index=256000`；文本侧 `sliding_window=4096`、`sliding_window_pattern=6`（每 6 层 1 层全局注意力）。— [processing_gemma3.py](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/gemma3/processing_gemma3.py)；[image_processing_gemma3.py](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/gemma3/image_processing_gemma3.py)；[configuration_gemma3.py](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/gemma3/configuration_gemma3.py)

### Inferences（按上述公式手算，可在实验 1 中验证）
| 模型 | 像素/视觉 token | 1920×1080 输入 | LLM 视觉 token | ViT patch 数 | 单图上限 |
|---|---|---|---|---|---|
| Qwen2.5-VL（max_pixels=12,845,056） | 28×28=784 | resize→1092×1932（round(38.57)=39, round(68.57)=69） | 39×69=**2,691** | 78×138=10,764 | 16,384 |
| Qwen3-VL-8B（longest_edge=16,777,216） | 32×32=1024 | resize→1088×1920（34×60） | **2,040** | 68×120=8,160 | 16,384 |
| LLaVA-1.5 | 14×14（336² 内） | 短边缩到 336 后中心裁剪（丢弃两侧） | **576** | 576(+CLS) | 576 |
| LLaVA-NeXT-Mistral | 336 tile | 选 pinpoint 网格 + 基础图 | 576 + 网格去 padding 特征 + 行换行 | — | 2×2 网格时约 576+48×48+48=2,928（未扣 unpad） |
| InternVL3 | 448 tile=256 token | 网格（如 4×2）+缩略图 | (块数+1)×256 | 每块 1024 | 13×256=**3,328** |
| Gemma 3 | 每图/每 crop 固定 256 | 不开 P&S：缩放到 896×896（宽高比失真） | **256**（开 P&S 最多 (1+4)×256=1,280） | 64×64=4,096 | 1,280 |

- 归一化差异要点：Qwen2/2.5-VL 与 LLaVA-1.5 用 OpenAI CLIP 均值方差；Qwen3-VL、Gemma 3 用 0.5/0.5（SigLIP 风格）；InternVL 用 ImageNet 均值方差。自写预处理或做「GPU 预处理」时用错常数会静默降低精度。
- **CPU 侧成本（推断）**：JPEG 解码 + BICUBIC resize + 归一化 + patch 重排都在 CPU（transformers v5 的 `TorchvisionBackend` 可在 tensor 上做），对 4K 图或长视频的解码是单请求 TTFT 的显著部分；vLLM v0.31.0 已有 `mm_device_do_normalize`（默认 True）与 processor cache（见 Q7）来缓解。精确耗时需实验测量。

- **Gemma 3 技术报告原文**：“We use a 400M variant of the SigLIP encoder”；“The Gemma vision encoder takes as input square images resized to 896 x 896”；“condensing the vision embeddings into a fixed size of 256 vectors”；Pan & Scan “segments images into non-overlapping crops of equal size, covering the whole image, and resize them to 896×896 pixels”。结合 transformers 池化公式：896/14 = 64×64 = 4096 个 patch → `AvgPool2d(4)` → 16×16 = **256** token（patch_size=14 取自 SigLIP 惯例，Gemma 3 config 因 gated 未直接读到）。— [Gemma 3 Technical Report, arXiv 2503.19786](https://arxiv.org/html/2503.19786)

### Gaps
- google/gemma-3-4b-it 在 HF 上为 gated（curl 返回 “Access … is restricted”），其 preprocessor_config/config（含 patch_size=14）未直接读取。
- LLaVA-NeXT 博客中「672×672, 336×1344, 1344×336」的表述与 HF Mistral 版 pinpoints（含 1008×336）不完全一致，原始博客本轮未抓取核对，以 HF config 为准。
- CPU 预处理的实测耗时没有可引用的官方数字。

---

## Q3. 视频：采样、时间 patch、token 预算公式、时间戳、长视频爆炸与缓解

### Takeaway
Qwen 系视频默认按 **2 fps** 采样、帧数夹在 **[4, 768]** 且为 2 的倍数，相邻 **2 帧合成一个 temporal patch**；每个 temporal patch 的 token 数被限制在 **128–768**，整段视频总像素预算默认约 **0.9×128000 个 token**。因此 1 分钟 1080p 视频约 4.3 万 token，10 分钟以上视频直接顶到 ~11.5 万 token 的预算上限，单请求 KV 可达数 GiB。

### Cited Findings
- **qwen-vl-utils 视频常量**：`VIDEO_MIN_TOKEN_NUM=128, VIDEO_MAX_TOKEN_NUM=768, FPS=2.0, FRAME_FACTOR=2, FPS_MIN_FRAMES=4, FPS_MAX_FRAMES=768, MAX_NUM_WORKERS_FETCH_VIDEO=8, MODEL_SEQ_LEN = env MODEL_SEQ_LEN 默认 128000`。`smart_nframes`：`nframes = total_frames / video_fps * fps`，夹到 `[ceil2(min_frames), floor2(min(max_frames,total_frames))]` 后 `floor_by_factor(nframes, 2)`；帧索引 `torch.linspace(start,end,nframes).round()`。— [vision_process.py](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py)
- **qwen-vl-utils 每帧像素预算**：`image_factor = image_patch_size*2`；`VIDEO_FRAME_MIN_PIXELS = 128*factor²`、`VIDEO_FRAME_MAX_PIXELS = 768*factor²`；`total_pixels = ele.get("total_pixels", MODEL_SEQ_LEN*factor²*0.9)`；`max_pixels = max(min(VIDEO_FRAME_MAX_PIXELS, total_pixels/nframes*FRAME_FACTOR), int(min_pixels*1.05))`，再用该值做 `smart_resize`。— [vision_process.py](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py)
- **transformers Qwen3-VL 视频处理器（v5.19.0）**：类默认 `fps=2, min_frames=4, max_frames=768, do_sample_frames=True, size={"shortest_edge":128*32*32, "longest_edge":32*32*768}`；帧索引 `np.linspace(0, total-1, num_frames).round()`；无 metadata 时 fps 回退为 24 并告警；视频版 `smart_resize` 对 `t_bar*h_bar*w_bar` 整体施加像素上下限；帧数不是 2 的倍数时复制最后一帧补齐。**新增 `cap_pixels_per_frame` / `max_video_tokens`(默认 768)**：开启后每帧像素 `max(min(768*factor², longest_edge//num_frames), int(shortest_edge*1.05))`，以复现 qwen-vl-utils；**未设置时保持“不封顶”旧行为并告警，默认值计划在 v5.22 改为 True**。— [video_processing_qwen3_vl.py @v5.19.0](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py)
- **Qwen3-VL-8B 视频预处理配置**：`size={"shortest_edge": 4096, "longest_edge": 25165824}`、patch 16、temporal 2、merge 2、均值方差 0.5。— [video_preprocessor_config.json](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/video_preprocessor_config.json)
- **时间戳**：Qwen2.5-VL 用 M-RoPE 时间维的绝对时间间隔（`tokens_per_second=2`）；Qwen3-VL 改为在每个 temporal patch 前插入文本时间戳 `<x.x seconds>`（见 Q1 引用）。— [modeling_qwen2_5_vl.py](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)；[processing_qwen3_vl.py](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_vl/processing_qwen3_vl.py)
- **vLLM 的视频 token 剪枝**：v0.31.0 `MultiModalConfig` 有 `video_pruning_rate`（[0,1)，>0 时启用）与 `video_pruning_method`（默认 `"evs"` = Efficient Video Sampling，另有 `"vidcom2"`）。— [vllm/config/multimodal.py @v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/multimodal.py)
- **InternVL 视频**：模型卡示例按 `num_segments=32` 均匀分段取帧、每帧 `max_num=1`（单块 256 token）→ 32×256=8,192 token。— [InternVL3-8B 模型卡](https://huggingface.co/OpenGVLab/InternVL3-8B)

### Inferences（Qwen2.5-VL，factor=28，qwen-vl-utils 默认，手算）
- **token 预算公式**：视频 token ≈ (nframes/2) × (h'/28) × (w'/28)，其中 h'w' ≤ min(768×784, total_pixels×2/nframes)，total_pixels = 128000×784×0.9 = 90,316,800。
- **例 1：60 s、30 fps、1080p** → nframes = 1800/30×2 = 120 帧 → 60 个 temporal patch；每帧上限 min(602,112, 90,316,800×2/120=1,505,280) = 602,112 px；beta=√(2,073,600/602,112)=1.856 → 560×1008 → 每组 20×36 = 720 token → 总计 **60×720 = 43,200 token**，KV ≈ 43,200×56 KiB ≈ **2.31 GiB**（单请求）。
- **例 2：10 分钟视频** → 2 fps 得 1200 帧，被 `FPS_MAX_FRAMES=768` 截断 → 384 组；每帧上限 min(602,112, 90,316,800×2/768=235,200) → 每组约 300 token → 总计约 **115,200 token**（= 0.9×128000，被总预算封顶），KV ≈ 6.3 GiB（Qwen2.5-VL-7B）/ 若换 Qwen3-VL-8B 量级再×2.6。
- **缓解手段**（工程层面）：降 fps/限 max_frames、调小 `total_pixels`/`max_pixels`、vLLM `video_pruning_rate`（EVS）、`--limit-mm-per-prompt` 中给 video 配 `num_frames/width/height`（仅影响 profiling 用的 dummy 输入，见 Q7）、encoder 侧数据并行（`mm_encoder_tp_mode=data`）。
- **注意 transformers 与 qwen-vl-utils 的差异**：在 v5.19.0 中，HF 视频处理器默认不做每帧封顶，短视频可能把整个 `longest_edge` 预算（Qwen3-VL-8B：25,165,824 px ≈ 12,288 token）花在少数帧上；用于测试时务必显式设置 `cap_pixels_per_frame`，或统一使用 qwen-vl-utils，否则 token 数对不上。

### Gaps
- vLLM 内部对 Qwen3-VL 视频是否默认采用 qwen-vl-utils 语义的每帧上限（vs. HF 处理器行为）未核实。
- EVS 的剪枝质量/速度权衡数据未获取。

---

## Q4. 音频输入：Whisper 前端、编码器-解码器解码、Qwen2-Audio / Qwen2.5-Omni / Qwen3-Omni 的 token 速率、流式 ASR

### Takeaway
Whisper：16 kHz，n_fft=400（25 ms 窗）、hop=160（10 ms 移），30 s 一块 = 480,000 采样 = 3000 帧 Mel，conv2 stride 2 → **1500 个编码器位置（20 ms/位置，50/s）**；large-v3 用 128 Mel，large-v2 及更早用 80 Mel。Qwen2-Audio/Qwen2.5-Omni 在 Whisper 式编码器后再做 stride-2 平均池化 → **25 token/s（40 ms/token）**；Qwen3-Omni 的 AuT 下采样 8 倍 → **12.5 Hz（约 80 ms/token，实现上每 1 s 块 13 token）**。

### Cited Findings
- **openai/whisper audio.py（v20250625）**：`SAMPLE_RATE=16000, N_FFT=400, HOP_LENGTH=160, CHUNK_LENGTH=30, N_SAMPLES=480000, N_FRAMES=3000, N_SAMPLES_PER_TOKEN=HOP_LENGTH*2`（注释“the initial convolutions has stride 2”）, `FRAMES_PER_SECOND=100`（10 ms/帧）, `TOKENS_PER_SECOND=50`（20 ms/音频 token）；`mel_filters` 只支持 `n_mels ∈ {80,128}`；`torch.stft(audio, 400, 160, window=hann_window(400))`。— [whisper/audio.py](https://github.com/openai/whisper/blob/v20250625/whisper/audio.py)
- **Whisper 编码器**：`conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)`、`conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)`，加正弦位置编码，且断言输入形状必须等于位置编码形状（即固定 30 s 输入）。— [whisper/model.py](https://github.com/openai/whisper/blob/v20250625/whisper/model.py)
- **HF whisper-large-v3**：`num_mel_bins=128, max_source_positions=1500, max_target_positions=448, encoder_layers=32, decoder_layers=32, d_model=1280`；预处理 `chunk_length=30, feature_size=128, hop_length=160, n_fft=400, n_samples=480000, nb_max_frames=3000, sampling_rate=16000`。**large-v3-turbo**：`encoder_layers=32, decoder_layers=4, num_mel_bins=128`。**large-v2**：`num_mel_bins=80`。— [large-v3 config](https://huggingface.co/openai/whisper-large-v3/blob/main/config.json)；[large-v3 preprocessor_config](https://huggingface.co/openai/whisper-large-v3/blob/main/preprocessor_config.json)；[large-v3-turbo config](https://huggingface.co/openai/whisper-large-v3-turbo/blob/main/config.json)；[large-v2 config](https://huggingface.co/openai/whisper-large-v2/blob/main/config.json)
- **Qwen2-Audio 输出长度（transformers v5.19.0）**：编码器后 `avg_pooler = nn.AvgPool1d(2, stride=2)`；`_get_feat_extract_output_lengths`: `input_lengths = (L-1)//2 + 1`（conv stride 2），`output_lengths = (input_lengths-2)//2 + 1`（池化）。30 s：3000 → 1500 → **750 token**。— [modeling_qwen2_audio.py](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py)
- **Qwen2.5-Omni-7B**：音频编码器 `num_mel_bins=128, encoder_layers=32, d_model=1280, n_window=100, output_dim=3584`；`position_id_per_seconds=25, seconds_per_chunk=2`；`get_rope_index` 中音频长度 `((L-1)//2+1-2)//2+1`（同 Qwen2-Audio，25 token/s），音视频交织按 `t_ntoken_per_chunk = position_id_per_seconds * seconds_per_chunk`（= 50 个时间位置/2 s 块）。— [Qwen2.5-Omni-7B config.json](https://huggingface.co/Qwen/Qwen2.5-Omni-7B/blob/main/config.json)；[modeling_qwen2_5_omni.py](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py)
- **Qwen3-Omni AuT**：报告原文：“resample to 16 kHz and convert the raw waveform into a 128 channel mel-spectrogram with a 25 ms window and a 10 ms hop”；“filter bank features … downsampled 8 times using Conv2D blocks before the attention layers, reducing the token rate to 12.5 Hz”；“each frame … corresponds to approximately an 80 ms segment”；AuT 从头训练于 2000 万小时音频，“employs block-wise window attention to enable real-time prefill caching”。— [Qwen3-Omni Technical Report, arXiv 2509.17765](https://arxiv.org/html/2509.17765)
- **Qwen3-Omni 实现**：`_get_feat_extract_output_lengths(input_lengths, n_window=50)`：`chunk_len = n_window*2 = 100` 帧（1 s），完整 1 s 块各产出 **13** token，余数部分经三次 stride-2 计算；config `n_window=50, n_window_infer=800, conv_chunksize=500, downsample_hidden_size=480, output_dim=2048, position_id_per_seconds=13`。— [modeling_qwen3_omni_moe.py](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py)；[Qwen3-Omni-30B-A3B-Instruct config.json](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/config.json)

### Inferences
| 前端 | 帧率 | 30 s 音频 → token/位置 | 备注 |
|---|---|---|---|
| Whisper 编码器 | 100 Mel 帧/s → 50 位置/s | 1500（固定，不足 30 s 也 pad 到 30 s） | 编码器-解码器，解码器最多 448 token |
| Qwen2-Audio / Qwen2.5-Omni | 25 token/s（40 ms） | 750 | 作为 LLM prefill token |
| Qwen3-Omni AuT | 12.5 Hz（实现 13/s 块） | 30×13 = 390 | token 量约为前者的一半 |
- 对 Qwen3-Omni Thinker（96 KiB/token），1 小时音频 ≈ 3600×13 = 46,800 token ≈ 4.3 GiB KV；对 Qwen2.5-Omni 则为 90,000 token。
- **Whisper 服务特点（推断）**：编码器每 30 s 窗口一次性算（固定 1500 位置，计算量与真实时长无关，短音频浪费），解码器是小型自回归 decoder，带 cross-attention KV（每请求一份 1500 长度的 cross KV，生成过程中不变）。流式 ASR 需要切块 + 重叠/局部一致性策略；Qwen3-Omni 的块级窗口注意力 + 1 s 块（`n_window`）设计本身就是为流式预填充缓存服务。
- large-v3-turbo 把解码器从 32 层减到 4 层，说明 Whisper 推理瓶颈主要在自回归解码而非编码器（推断）。

### Gaps
- Whisper 论文原文（arXiv 2212.04356）与 Qwen2-Audio 报告（arXiv 2407.10759）中“40 ms/帧”原句本轮未抓取，40 ms 结论由源码公式推得。
- 流式 ASR 在 vLLM/SGLang 中的具体支持（如 realtime API）未核实。

---

## Q5. 语音输出与 Omni 模型：Thinker–Talker、TMRoPE、Qwen3-Omni（MoE、多码本、Code2Wav、首包延迟）与多阶段服务

### Takeaway
Omni 模型是「理解模型（Thinker，大 LLM）+ 语音生成模型（Talker，小 LLM）+ 波形渲染器（Qwen2.5-Omni：DiT+BigVGAN；Qwen3-Omni：轻量因果 ConvNet Code2Wav）」的多阶段流水线。Qwen3-Omni 把 Thinker 与 Talker 都改为 MoE，Talker 每步自回归出 1 帧（12.5 Hz、16 码本，MTP 补齐残差码本），Code2Wav 逐帧合成波形，报告冷启动理论端到端首包延迟 **234 ms**。服务这类模型需要跨阶段流式传递 hidden states / codec token，并独立调度各阶段的 batching 与显存。

### Cited Findings
- **Qwen3-Omni 五点升级（原文）**：(1) Thinker 和 Talker 均升级为 MoE；(2) 用 AuT 取代 Whisper 音频编码器；(3) 语音生成采用多码本表示；(4) Talker 从单轨变为多轨 codec 建模，“autoregressively predicting multiple codebook layers via MTP modules”，波形阶段 “Code2Wav replaces block-wise DiT with a lightweight convolutional network (ConvNet)”；(5) 输入与输出音频码率降至 12.5 Hz，输出 codec 支持“single-frame, immediate speech synthesis”。— [arXiv 2509.17765](https://arxiv.org/html/2509.17765)
- **首包延迟**：“In cold-start settings (no prior context), Qwen3-Omni achieves a theoretical end-to-end first-packet latency of 234 ms”。— [arXiv 2509.17765](https://arxiv.org/html/2509.17765)
- **解耦**：“Talker no longer consumes the Thinker’s high-level text representations and conditions only on audio and visual multimodal features”，因此 Thinker 与 Talker 可以使用不同 system prompt；“Both the Thinker and Talker adopt Mixture-of-Experts (MoE) architectures to support high concurrency and fast inference”。— [arXiv 2509.17765](https://arxiv.org/html/2509.17765)
- **TM-RoPE（Qwen3-Omni）**：把 RoPE 分解为时间/高/宽三维；文本三维共享位置 id（等价 1D RoPE）；音频共享位置 id 并叠加绝对时间编码，“each temporal ID corresponds to a duration of 80 ms”；视频帧时间 id 按真实时间戳调整以保持 80 ms/ID。— [arXiv 2509.17765](https://arxiv.org/html/2509.17765)
- **Qwen3-Omni-30B-A3B-Instruct 配置**：Thinker 文本 `num_hidden_layers=48, hidden_size=2048, num_attention_heads=32, num_key_value_heads=4, head_dim=128, num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768, max_position_embeddings=65536, rope: mrope_interleaved=true, mrope_section=[24,20,20]`；Thinker 视觉 `depth=27, hidden=1152, patch_size=16, spatial_merge=2, temporal_patch=2, deepstack_visual_indexes=[8,16,24], out_hidden_size=2048`（与 Qwen3-VL 同族）；Talker `accept_hidden_layer=24, num_code_groups=16, thinker_hidden_size=2048`，Talker 文本 `num_hidden_layers=20, hidden=1024, num_key_value_heads=2, num_experts=128, num_experts_per_tok=6, vocab_size=3072`；code predictor（MTP）`num_hidden_layers=5, hidden=1024, vocab_size=2048, num_code_groups=16`；Code2Wav `codebook_size=2048, num_quantizers=16, hidden=1024, num_hidden_layers=8, sliding_window=72, upsample_rates=[8,5,4,3], upsampling_ratios=[2,2], decoder_dim=1536`。— [config.json](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/config.json)
- **Qwen2.5-Omni-7B 配置**：Thinker 文本 `hidden_size=3584, num_hidden_layers=28, num_key_value_heads=4, vocab=152064`；Talker `hidden_size=896, num_hidden_layers=24, num_key_value_heads=4, vocab_size=8448`；token2wav 含 `dit_config{depth=22, mel_dim=80, num_embeds=8193}` 与 `bigvgan_config{mel_dim=80,...}`；`position_id_per_seconds=25, seconds_per_chunk=2`。transformers 注释以 “25 tokens per second … interval = 25*2/1 = 50” 解释 TMRoPE 时间间隔。— [Qwen2.5-Omni-7B config.json](https://huggingface.co/Qwen/Qwen2.5-Omni-7B/blob/main/config.json)；[modeling_qwen2_5_omni.py](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py)
- **服务框架**：vLLM 官方组织下存在独立项目 vLLM-Omni，最新 release v0.30.0（2026-09-25）。— [vllm-project/vllm-omni v0.30.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.30.0)

### Inferences
- **Code2Wav 帧长（推导）**：每个 codec 帧上采样 8×5×4×3×2×2 = **1,920 个采样点**；若输出采样率为 24 kHz，则 1 帧 = 80 ms，正好对应 12.5 Hz（与报告一致）。也就是 Talker 每 decode 1 步（+MTP 5 层小模型补 15 个残差码本）→ Code2Wav 即可吐出 80 ms 音频，这就是“单帧即时合成”的含义。
- **多阶段服务 vs 单 LLM**：(a) 三个模型各有 KV/状态（Thinker KV 96 KiB/token；Talker 20 层×2 KV 头×128×2×2B = 20 KiB/token；Code2Wav 滑窗 72）；(b) 阶段间是流式生产者-消费者，需要 chunk 级传递 Thinker 第 24 层 hidden（`accept_hidden_layer=24`）或多模态特征；(c) 首包延迟 = Thinker prefill + 若干 Thinker/Talker decode 步 + MTP + Code2Wav 首帧，各阶段可分别放在不同 GPU、分别做 continuous batching；(d) 与 OxyGen 的类比：一次 prefill 结果被多个“专家/阶段”消费，调度的关键是阶段间的数据依赖与 batching 粒度。
- Qwen2.5-Omni 的波形阶段是 DiT（22 层）+ BigVGAN，Qwen3-Omni 换成 8 层滑窗 Transformer + 卷积上采样，从“块级扩散”换成“逐帧因果解码”，这是首包延迟下降的结构性原因（推断，与报告第 (4)(5) 点一致）。

### Gaps
- Code2Wav 输出采样率（24 kHz）、Thinker/Talker 参数规模（报告中出现“3B-A0.3B”字样，推测为 Talker 规模）以及报告中不同并发下的分阶段延迟表本轮未完整抽取（抓取第 5 节全文时超时）→ 「证据不足」。
- Qwen2.5-Omni 技术报告（arXiv 2503.20215）原文未抓取（滑窗 DiT、首包设计），仅以 config/源码为证据。
- vLLM-Omni 的架构文档、支持模型列表与命令行未核实（仅确认项目与版本存在）。

---

## Q6. 图像/视频生成推理：扩散/DiT 流水线、CFG、步数与调度器、缓存技巧、序列并行、服务框架、与自回归生成对比

### Takeaway
扩散推理 = 文本编码器（一次）→ DiT/UNet 去噪 N 步（主成本，每步一次完整前向；开启 CFG 时每步需条件/无条件两次前向，常以 batch×2 实现）→ VAE 解码（一次，视频/高分辨率时显存峰值大）。它没有自回归 LLM 意义上的“随长度增长的 KV cache”，瓶颈是“步数 × 每步 token 数²”的注意力计算；优化方向是减步（蒸馏/更好的 solver）、跨步特征缓存（TeaCache、First-Block Cache）、多卡序列并行（xDiT 的 USP）。本节大部分细节因中断未能逐项核实。

### Cited Findings
- 版本存在性（已核实）：diffusers v0.41.0（2026-10-06）；xDiT 0.7.0（2026-09-25）；vLLM-Omni v0.30.0（2026-09-25）；SGLang v0.5.21 源码树中存在 `python/sglang/multimodal_gen/` 目录（GitHub contents API 返回 13 个条目），表明 SGLang 内置了多模态生成（扩散）子系统。— [diffusers v0.41.0](https://github.com/huggingface/diffusers/releases/tag/v0.41.0)；[xDiT 0.7.0](https://github.com/xdit-project/xDiT/releases/tag/0.7.0)；[vLLM-Omni v0.30.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.30.0)；[sglang multimodal_gen @v0.5.21](https://github.com/sgl-project/sglang/tree/v0.5.21/python/sglang/multimodal_gen)

### Inferences（以下均为「未核实」的通用知识，写作者使用时需标注或另行核对）
- CFG：ε = ε_uncond + s·(ε_cond − ε_uncond)，每步两次前向 → 计算约 ×2（可 batch 化或分到两张卡做 “CFG 并行”）。
- DiT（Peebles & Xie, arXiv 2212.09748）：把 VAE latent 切 patch 成 token 序列，用 Transformer 去噪；token 数 = (H/8/p)×(W/8/p)（典型 VAE 下采样 8、patch p=2），视频再乘以潜空间帧数，因此高分辨率/长视频的注意力是 O(n²) 主导。
- TeaCache / First-Block Cache：利用相邻去噪步输出差异小，跳过部分步骤或复用残差；xDiT 的 USP = Ulysses（all-to-all 切 head）+ Ring Attention（切序列）混合序列并行，另有 PipeFusion、CFG 并行。
- 与自回归图像生成对比：AR 方案（图像 token + LLM decode）复用 LLM 推理栈（KV cache、continuous batching），但生成一张图需数千步 decode；扩散方案步数少（十几到几十步）但每步全序列计算、无增量 KV。

### Gaps
- DiT 论文、diffusers CFG 实现、TeaCache/First-Block Cache 原始仓库、xDiT USP 文档、vLLM-Omni 与 SGLang diffusion 文档本轮均未抓取 → 以上通用知识全部「未核实」，不得作为已验证数字使用。

---

## Q7. 引擎支持：vLLM 多模态管线、SGLang 多模态支持与典型瓶颈

### Takeaway
vLLM v0.31.0 的多模态配置集中在 `MultiModalConfig`：每 prompt 模态数量上限（默认每模态 999，可附带 dummy 尺寸用于 profiling）、HF processor 参数覆盖、processor 结果缓存（默认 4 GiB，按进程复制）、encoder 的 TP 模式（`weights`/`data`）、视频剪枝、仅编码器模式（用于编码器解耦），以及 `--ec-transfer-config`（encoder cache 传输配置，用于 E/P 分离）。典型瓶颈依次是：CPU 解码/resize → 视觉编码器 → 巨量视觉 token 的 LLM prefill 与 KV 占用。

### Cited Findings
- **CLI 参数存在性（vLLM v0.31.0 `engine/arg_utils.py` 实测）**：`--limit-mm-per-prompt`、`--mm-processor-kwargs`、`--mm-processor-cache-gb`、`--mm-processor-cache-type`、`--mm-encoder-tp-mode`、`--enable-prefix-caching`、`--max-num-batched-tokens`、`--ec-transfer-config`。— [vllm/engine/arg_utils.py @v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/engine/arg_utils.py)
- **`limit_per_prompt`**：“Defaults to 999 for each modality”；旧格式 `{"image": 16, "video": 2}`；可配置格式 `{"video": {"count": 1, "num_frames": 32, "width": 512, "height": 512}, "image": {"count": 5, "width": 512, "height": 512}}`；可混用。— [vllm/config/multimodal.py @v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/multimodal.py)
- **`mm_processor_cache_gb`**：默认 **4** GiB，用于避免重复处理已见过的多模态输入；该缓存在每个 API 进程和 engine core 进程各复制一份，总占用 `mm_processor_cache_gb * (api_server_count + data_parallel_size)`；单个超出预算的条目不缓存（告警）；设为 0 完全禁用（不推荐）。`mm_processor_cache_type` 默认 `"lru"`；`mm_shm_cache_max_object_size_mb` 默认 128（共享内存缓存）。— 同上
- **`mm_hasher_algorithm`** 默认 `"blake3"`（多模态内容哈希，用于缓存键/前缀缓存识别同一图像）。— 同上
- **`mm_encoder_tp_mode`**：默认 `"weights"`（同一引擎内把每层权重切到各 TP rank）；`"data"`：每个 TP rank 持有完整编码器权重，把批内输入数据切分并行处理（“batch-level DP”，区别于 `--data-parallel-size`），仅部分模型支持，不支持时回退。— 同上
- **其他字段**：`mm_encoder_only`（跳过语言部分，“usually only valid in disaggregated Encoder process”）、`language_model_only`、`enable_mm_embeds`（允许直接传入预计算 embedding）、`media_io_kwargs`、`mm_device_do_normalize`（默认 True）、`mm_encoder_attn_backend`、`mm_encoder_attn_dtype`（可选 `"fp8"`）、`skip_mm_profiling`、`interleave_mm_strings`、`mm_tensor_ipc`（默认 `"direct_rpc"`）、`mm_ipc_gpu_memory_gb`（默认 0）、`video_pruning_rate`/`video_pruning_method`。— 同上

### Inferences
- **请求路径（据字段推断）**：API 进程内 HF processor（受 `mm_processor_kwargs` 控制，结果入 processor cache，键为 blake3 内容哈希）→ engine core 调度时为含图请求安排 encoder 计算（受 encoder 预算约束）→ 编码器输出存 encoder cache → 按 chunked prefill 切片把对应 embedding scatter 进 LLM 输入 → 同一图片的 KV 可经前缀缓存复用（哈希进入 block hash）。
- **`--mm-processor-kwargs` 的 transformers v5 注意事项**：对 Qwen2/2.5-VL，向处理器传 `min_pixels/max_pixels` 在 transformers v5.19.0 已弃用（v5.23 移除），新写法是 `{"size": {"longest_edge": N, "shortest_edge": M}}`；Qwen3-VL HF 配置本身已经使用 `size` 字段。
- **瓶颈判断（推断）**：小图高并发 → CPU 预处理/IPC 是瓶颈（看 API 进程 CPU 占用、processor cache 命中）；大图/视频 → encoder + prefill 主导 TTFT；长视频 → KV 容量限制并发数（例：Qwen2.5-VL-7B 单个 11.5 万 token 视频请求 ≈ 6.3 GiB KV）。
- `mm_encoder_tp_mode=data` 的价值：ViT（如 Qwen2.5-VL 32 层×1280 hidden）参数小，按权重 TP 切分时通信占比高；整份复制、按图像切分往往更快（需实验验证）。

### Gaps
- vLLM 文档站的设计页（multimodal processing、encoder cache、EPD 解耦）未抓取，`--ec-transfer-config` 的具体用法（connector 名称、角色）未核实；encoder compute budget 的具体参数名未核实。
- **SGLang 多模态参数未核实**：v0.5.21 的 `python/sglang/srt/server_args.py` 仅 698 行，未找到 `--enable-multimodal` 等字符串（参数定义可能已迁移到其他模块）；SGLang 的 EPD/encoder 分离支持同样未核实。

---

## Q8. VLA 的定位（简述）

### Takeaway
VLA（如 π0.5）可视为「多模态输入（多视角图像 + 语言 + 本体状态）+ 动作输出」：前半段与 VLM 的 prefill 数据通路一致（视觉编码 → 视觉 token → 共享 KV），后半段把 LLM 解码替换/并联为动作专家（flow matching/扩散式动作头）。OxyGen 的“一次观测 prefill 被动作专家与语言专家共享”正是本笔记 Q1 的视觉 token 成本问题在机器人场景的复用式解法。详细内容由另一研究者负责。

### Cited Findings
- OxyGen 项目地址（用户已复现）：— [air-embodied-brain/OxyGen](https://github.com/air-embodied-brain/OxyGen)

### Inferences
- 对 VLA，每帧观测的图像 token 数直接决定控制频率上限；本笔记的 token 公式（Q2）可直接用于估算每帧 prefill 成本。

### Gaps
- π0.5 的视觉编码器/每帧 token 数本轮未核实（由其他研究者覆盖）。

---

## Q9. 4×H800 动手实验（不训练；命令中 vLLM 参数名已对照 v0.31.0 源码核实，其余 Python API 为 transformers 常规用法）

### Takeaway
四个实验按“先算 token、再测 TTFT、再测缓存、再测编码器并行/视频剪枝”的顺序，把 Q1–Q3、Q7 的公式和配置落到可测数据上。

### Cited Findings
- 参数名与取值依据：[vllm/engine/arg_utils.py @v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/engine/arg_utils.py)；[vllm/config/multimodal.py @v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/multimodal.py)；预处理公式依据见 Q2/Q3 引用。

### Inferences（实验设计）

**实验 1：离线 token 计数器（CPU，0 GPU，约 1 小时）**
- 目标：验证 Q2/Q3 手算表。
- 步骤：在已有 transformers 环境（不新装包）中，对 256²、1280×720、1920×1080、3840×2160 合成图，分别用 `AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")` 与 `"Qwen/Qwen3-VL-8B-Instruct"` 处理，读取输出的 `image_grid_thw`，计算 `prod(thw)//4`；再用 `"llava-hf/llava-1.5-7b-hf"` 统计 `input_ids` 中 image token 个数。视频：Qwen3-VL 处理器分别设置 `cap_pixels_per_frame=True/False`，比较 `video_grid_thw`。
- 预期：1080p → Qwen2.5-VL 2,691、Qwen3-VL 2,040、LLaVA-1.5 576；Qwen3-VL 短视频在不封顶时 token 明显偏多。
- 验收：实测与手算误差为 0（公式确定性）；并记录每张图预处理耗时（`time.perf_counter`）。

**实验 2：分辨率扫描 TTFT 与 KV 占用（1×H800，约 2 小时）**
- 命令骨架（参数名已核实）：
  `vllm serve Qwen/Qwen2.5-VL-7B-Instruct --limit-mm-per-prompt '{"image": 4, "video": 1}' --mm-processor-kwargs '{"size": {"longest_edge": 12845056, "shortest_edge": 3136}}' --max-num-batched-tokens 16384`
- 步骤：用 OpenAI 兼容 `/v1/chat/completions` 发送 `image_url` 消息，分辨率从 448² 到 4K，`max_tokens=1`，测 TTFT；改 `longest_edge` 为 `1280*28*28` 重复。
- 预期：TTFT 随视觉 token 近线性上升，4K 图被 max_pixels 截到 ≤16,384 token；降低 longest_edge 后 TTFT 显著下降。
- 验收：画出 TTFT–token 数曲线，并用 56 KiB/token 校验日志中 KV 使用变化。

**实验 3：缓存效应（1×H800，约 1 小时）**
- 步骤：同一张 1080p 图 + 不同问题连续发送 50 次；A 组默认（`--mm-processor-cache-gb` 4、prefix caching 开启），B 组 `--mm-processor-cache-gb 0`；可再加 C 组 `--no-enable-prefix-caching` 对照（v0.31.0 布尔参数使用 `argparse.BooleanOptionalAction`，故自动生成 `--no-` 形式）。
- 预期：A 组第 2 次起 TTFT 大幅下降（预处理 + 编码 + 图像部分 prefill 均被复用）；B 组仍复用 KV 但每次重做 HF 预处理。
- 验收：报告首请求/后续请求 TTFT 比值，以及 API 进程 CPU 时间差。

**实验 4：编码器 TP 模式与视频剪枝（4×H800，约 3 小时）**
- 命令骨架：`vllm serve Qwen/Qwen2.5-VL-72B-Instruct --tensor-parallel-size 4 --mm-encoder-tp-mode data`，对照 `--mm-encoder-tp-mode weights`；视频请求（60 s、1080p）下再对比 `--video-pruning-rate 0.5`（v0.31.0 arg_utils 第 1490 行已核实该参数；方法默认 `evs`）。
- 预期：多图批量下 `data` 模式编码吞吐更高；剪枝后视频 token 与 TTFT 近似按比例下降，答案质量轻微变化。
- 验收：记录图像/秒、TTFT p50/p99 与 3–5 个视频问答的人工质量对比。

### Gaps
- 实验命令未在真实 H800 上运行，属于“参数名已静态核实、行为未运行验证”；Qwen2.5-VL-72B 是否支持 `mm_encoder_tp_mode=data`（不支持时会回退）未核实。
