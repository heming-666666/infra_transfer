# 文本 LLM 推理核心链路：从原始文本到流式输出（tokenization → forward → prefill/decode → KV cache → sampling → 结构化解码 → detokenization）

> 研究日期 / 访问日期：2026-10-14（所有 URL 均在该日访问）。
> 版本基线（PyPI JSON，访问 2026-10-14）：transformers **5.19.0**（2026-10-06 发布）、vLLM **0.31.0**（2026-10-05）、SGLang **0.5.21**（2026-10-01）、xgrammar 0.2.8、llguidance 1.9.1、outlines 1.3.3、tiktoken 0.14.0、tokenizers 0.23.3、torch 2.14.1、sentencepiece 0.2.2 — 来源 https://pypi.org/pypi/{transformers,vllm,sglang,xgrammar,llguidance,outlines,tiktoken,tokenizers,torch,sentencepiece}/json
> 代码引用：transformers 以 tag `v5.19.0` 为准（部分文件读的是 main@2026-10-14，已标注）；vLLM 以 tag `v0.31.0` 或 main（tree sha `d36158f9`，2026-10-14）为准；SGLang 为 main@2026-10-14。
> 方法说明：本机无 transformers/tokenizers（按约束不安装），所以 Qwen3 的 token 统计是我用纯 Python 按官方 `tokenizer.json`（NFC + Split 正则 + ByteLevel + BPE merges）在内存中复现的，并用 `"Hello world" → [9707, 1879]` 做了自检。参数量、KV、FLOPs 数字由 Python 根据官方 config.json 计算。

---

## Q1 Tokenization、预处理、chat template、padding 与流式 detokenization

### Takeaway
推理时，tokenizer 是一条确定性的流水线：normalize → pre-tokenize（正则切分）→ BPE merges（byte-level，或 SentencePiece 式的 ▁ + byte fallback）→ 加 special tokens / chat template。字节级 BPE 会把 1 个汉字（UTF-8 3 字节）切到多个 token 里，所以流式输出必须先缓冲，等字节凑成完整字符再吐出。HF TextStreamer 和 vLLM 增量 detokenizer 都是这么做的。

### Cited Findings
**Byte-level BPE（Qwen3）**
- Qwen3-0.6B 的 `tokenizer.json` 结构如下（访问 2026-10-14）— [Qwen3-0.6B tokenizer.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer.json)
  - normalizer 为 `{"type":"NFC"}`。
  - pre_tokenizer 是 Sequence[Split(regex, Isolated), ByteLevel(add_prefix_space=false, use_regex=false)]。
  - 切分正则为 `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`。
  - decoder 为 ByteLevel，model 为 BPE，`byte_fallback: false`。
  - 词表 151,643 项、merges 151,387 条、added_tokens 26 个。
- Qwen3 正则对数字单字切分（`\p{N}`），Llama 3 是 `\p{N}{1,3}`（最多 3 位一组）。其余部分与 Qwen3 相同 — [llama-models llama3/tokenizer.py](https://github.com/meta-llama/llama-models/blob/main/models/llama3/tokenizer.py)
- Llama 3 tokenizer 基于 tiktoken，`num_reserved_special_tokens = 256`。special tokens 包括 `<|begin_of_text|>`（bos_id）、`<|end_of_text|>`（eos_id）、`<|eot_id|>`（end of turn）— [llama-models llama3/tokenizer.py](https://github.com/meta-llama/llama-models/blob/main/models/llama3/tokenizer.py)
- tiktoken README 称其 "reversible and lossless"、会压缩文本，并且比可比的开源 tokenizer 快 3–6 倍 — [tiktoken README](https://github.com/openai/tiktoken/blob/main/README.md)

**SentencePiece 式（Llama 2 / Gemma）**
- SentencePiece 把输入当作原始 Unicode 字符序列，空格转义成元符号 `▁`（U+2581），因此 detokenize 就是简单拼接后把 `▁` 换回空格，"lossless"。它支持 BPE 与 unigram，README 以 Gemma 3（256k vocab）为 BPE 模型示例 — [SentencePiece README](https://github.com/google/sentencepiece/blob/master/README.md)
- Llama 2 tokenizer 的 HF 转换版（`hf-internal-testing/llama-tokenizer`，测试用副本，**非 Meta 官方仓库**）— [hf-internal-testing/llama-tokenizer](https://huggingface.co/hf-internal-testing/llama-tokenizer/blob/main/tokenizer.json)
  - normalizer：Prepend "▁"，再 Replace " "→"▁"；无 pre_tokenizer。
  - BPE 设 `byte_fallback: true`，vocab 32,000，前三个 token 为 `<unk>`、`<s>`、`</s>`。
  - decoder：Replace ▁→空格 → ByteFallback → Fuse → Strip（去掉开头 1 个空格）。
  - byte fallback 的含义：词表外字符拆成 `<0xE9>` 这类字节 token，decoder 再把它们拼回 UTF-8。

**Special tokens、chat template、生成参数（Qwen3）**
- Qwen3-0.6B 的 `tokenizer_config.json` — [tokenizer_config.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json)
  - `add_bos_token: false`、`bos_token: null`、`eos_token: "<|im_end|>"`、`pad_token: "<|endoftext|>"`、`model_max_length: 131072`。
  - added tokens 包括 151643 `<|endoftext|>`、151644 `<|im_start|>`、151645 `<|im_end|>`、151657 `<tool_call>`、151667 `<think>`、151668 `</think>`。
- chat_template（Jinja，4,168 字符）结尾为 `{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}`。也就是说关闭思考时，模板会预填一个空的 think 块；历史 assistant 消息中的 reasoning 内容会被切分重写 — [同上](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json)
- Qwen3-0.6B 的 `generation_config.json`：`do_sample: true, temperature: 0.6, top_k: 20, top_p: 0.95, eos_token_id: [151645, 151643], pad_token_id: 151643` — [generation_config.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/generation_config.json)
- config.json 中 `vocab_size: 151936`，大于 tokenizer 实际的 151,643+26=151,669，说明 embedding/LM head 有填充行 — [Qwen3-0.6B config.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)
- HF chat templating 文档：`add_generation_prompt=True` 会在末尾加 `<|im_start|>assistant`，提示模型开始写 assistant 回复；Llama 这类 assistant 前没有 special token 的模型，该参数无效；不能与 `continue_final_message` 同时使用 — [chat_templating.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/chat_templating.md)

**attention_mask / position_ids / left padding**
- HF LLM 教程要求把 `padding_side` 设为 `"left"`，理由是 "a LLM is not trained to continue generation from padding tokens"。tokenizer 返回 input_ids 和 attention_mask — [llm_tutorial.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/llm_tutorial.md)
- transformers v5.19.0 的 `Qwen3Model.forward` 在未传 position_ids 时用 `position_ids = torch.arange(T) + past_key_values.get_seq_length()`。该文件中已没有 `cache_position` 参数（出现次数 0）— [modeling_qwen3.py@v5.19.0](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3/modeling_qwen3.py)

**流式 detokenization**
- HF `TextStreamer.put` 的行为 — [streamers.py (main)](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py)
  - 每次把新 token 追加到 `token_cache`，然后整体 `decode`。
  - 文本以 `\n` 结尾时，全部输出并清空缓存。
  - 最后一个字符属于 CJK 统一表意文字区时，立即输出。
  - 其他情况只输出到最后一个空格（避免半个单词）。
  - 仅支持 batch size 1。
- vLLM 慢路径 `detokenize_incrementally` 维护 `prefix_offset/read_offset`。当 `new_text.endswith("�")` 时返回空串（注释："utf-8 char at the end means it's a potential unfinished byte sequence from byte fallback tokenization"）— [vllm/tokenizers/detokenizer_utils.py (main)](https://github.com/vllm-project/vllm/blob/main/vllm/tokenizers/detokenizer_utils.py)
- vLLM 快路径 `FastIncrementalDetokenizer` 使用 HF tokenizers 的 `tokenizers.decoders.DecodeStream(ids=prompt_token_ids, skip_special_tokens=...)`，逐 token 调用 `stream.step(tokenizer, id)`。遇到 "non-monotonic, invalid UTF-8" 异常时会重建 stream。stop 字符串在 detokenizer 层用 `stop_buffer_length` 做尾部缓冲 — [vllm/v1/engine/detokenizer.py (main)](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/detokenizer.py)

**每个汉字对应多少 token**
- Qwen 官方文档经验值："As a rule of thumb, 1 token is 3~4 characters for English texts and 1.5~1.8 characters for Chinese texts." — [Qwen Docs: Key Concepts](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)
- 本人实测（Qwen3 tokenizer.json，纯 Python 复现 BPE，可复现但不是官方数字）：
  - 一段 181 字符（168 个汉字）的现代中文技术/日常文本 → 110 tokens，即 **1.645 字符/token（0.608 token/字符）**，落在官方 1.5–1.8 区间内。
  - 等义英文 460 字符、68 词 → 81 tokens（5.68 字符/token）。
  - 中文前若干 token 为 `人工智能 | 正在 | 深刻 | 改变 | 世界 | 。| 大 | 语言 | 模型 | …`。
  - 生僻字 `龘`（UTF-8 `e9 be 98`）被切成 token 82912（`e9be`）+ 246（`98`），`靐`、`爩` 同样是 2 个 token，且每个 token 单独都不是合法 UTF-8；`齉` 是 1 个 token（122125）。

### Inferences
- 一个请求的预处理顺序：messages → chat template（Jinja 渲染出字符串，含 `<|im_start|>` 等 special token 文本）→ tokenizer encode（special token 整体匹配，普通文本走 NFC→正则→BPE）→ `input_ids [B,T]` + `attention_mask [B,T]`。Qwen 不加 BOS，Llama 3 加 `<|begin_of_text|>`。
- 为什么 decoder-only 批量生成要 left padding：
  - 每步都取 `logits[:, -1]` 作为下一 token 的分布，并把新 token 拼到序列末尾。
  - right padding 时，短序列的最后位置是 pad，新 token 会接在 pad 之后。
  - left padding 让所有序列的"最新位置"对齐在最右列；pad 位置由 attention_mask 屏蔽，position_ids 需按 mask 计算，或由框架处理。
- vLLM 这类按 token 拼接（packed / varlen）的引擎没有 padding 问题；left padding 是 HF 静态 batch `[B,T]` 张量的产物。
- 对 TextStreamer 的推断：中文无空格时，如果末字符是 `�`（不完整字节），它不属于 CJK 区，也不以 `\n` 结尾。于是走 `text[print_len : rfind(" ")+1]` 分支，`rfind` 返回 -1，切片为空，残缺字节被自然地缓冲住。这是从代码推出的结论，建议用练习 1 验证。

### Gaps
- Llama-2 / Gemma 官方 tokenizer 仓库是 gated 的，未直接读取，用的是 HF 测试副本。Gemma 的 normalizer / byte_fallback 细节「未核实」。
- 中文 token 率只有 Qwen 官方经验值和我对单段样本的实测，没有找到官方的大规模语料统计。
- Qwen3 model card 中 `apply_chat_template(..., enable_thinking=False)` 这一 kwarg 用法本次未抓取原文。从模板变量名可推断该用法，但 kwarg 透传机制「未核实」。

---

## Q2 Decoder-only Transformer 前向（Llama-3.x / Qwen3 dense 为范本）与官方配置

### Takeaway
Qwen3 / Llama 3 的每层结构都是 pre-norm：`x + Attn(RMSNorm(x))`，再 `+ SwiGLU-MLP(RMSNorm(·))`。Attention 用 GQA + RoPE；Qwen3 额外在每个 head 的 Q、K 上做 RMSNorm（q_norm/k_norm）。prefill 和 decode 跑的是同一套算子，区别只在序列维：prefill 是 T，decode 是 1，而 K/V 的长度 S 等于已缓存长度加当前长度。

### Cited Findings
**官方 config.json（访问 2026-10-14）**

| 模型 | hidden | layers | heads(Q) | kv_heads | head_dim | FFN | vocab | rope_theta | 其它 |
|---|---|---|---|---|---|---|---|---|---|
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json) | 1024 | 28 | 16 | 8 | 128 | 3072 | 151936 | 1e6 | tie_word_embeddings=true，max_pos 40960，rms_eps 1e-6 |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) | 4096 | 36 | 32 | 8 | 128 | 12288 | 151936 | 1e6 | untied |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json) | 5120 | 64 | 64 | 8 | 128 | 25600 | 151936 | 1e6 | 注意 64×128=8192≠5120 |
| [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json) | 2048 | 48 | 32 | 4 | 128 | moe_intermediate 768 | 151936 | 1e6 | num_experts=128，num_experts_per_tok=8，norm_topk_prob=true，decoder_sparse_step=1，mlp_only_layers=[] |
| Llama-3.1-8B | 4096 | 32 | 32 | 8 | 128 | 14336 | 128256 | 5e5 | rope_scaling llama3（factor 8，orig 8192），max_pos 131072，rms_eps 1e-5 |
| [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) | 7168 | 61 | 128 | 128（名义） | — | dense 18432 / moe 2048 | 129280 | 1e4 + YaRN(factor 40) | 见下 |

- Llama-3.1-8B 官方 HF 仓库是 gated 的（未登录访问返回 "Access … is restricted"），表中数值取自镜像 [unsloth/Meta-Llama-3.1-8B config.json](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json)。该镜像带 `unsloth_fixed: true`，`pad_token_id: 128004` 可能是镜像修改的。
- Meta 官方 [llama-models sku_list.py](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py) 给出的 3.1-8B arch_args 与镜像一致：`dim 4096, n_layers 32, n_heads 32, n_kv_heads 8, ffn_dim_multiplier 1.3, multiple_of 1024, norm_eps 1e-5, rope_theta 500000, use_scaled_rope True`。按 FFN 维度公式可推出 14336。
- DeepSeek-V3 的 MLA/MoE 字段 — [DeepSeek-V3 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)
  - MLA：`q_lora_rank 1536, kv_lora_rank 512, qk_nope_head_dim 128, qk_rope_head_dim 64, v_head_dim 128`。
  - MoE：`n_routed_experts 256, n_shared_experts 1, num_experts_per_tok 8, first_k_dense_replace 3, n_group 8, topk_group 4, scoring_func sigmoid, topk_method noaux_tc, routed_scaling_factor 2.5`。
  - 其它：`num_nextn_predict_layers 1`（MTP），权重量化为 fp8 e4m3、block 128×128。

**逐算子形状（Qwen3-8B；prefill 输入 `[B,T]`，decode 输入 `[B,1]`；S = 已缓存长度 + 当前长度）**
算子定义来自 [modeling_qwen3.py@v5.19.0](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3/modeling_qwen3.py)：

| # | 算子（代码位置） | 形状 / 公式 |
|---|---|---|
| 1 | `embed_tokens` | `[B,T]` → `[B,T,4096]` |
| 2 | `input_layernorm`（Qwen3RMSNorm） | 先转 fp32，`x·rsqrt(mean(x²)+eps)`，再乘 weight；形状不变 |
| 3 | `q_proj` / `k_proj` / `v_proj` | `[B,T,4096]` → Q `[B,T,32·128]`，K/V `[B,T,8·128]`；view 成 `(…, -1, 128)` 后 transpose → Q `[B,32,T,128]`，K/V `[B,8,T,128]` |
| 4 | `q_norm` / `k_norm` | 对最后一维 128（per-head）做 RMSNorm，在 RoPE 之前。代码注释："unlike olmo, only on the head dim" |
| 5 | RoPE | `q·cos + rotate_half(q)·sin`；`rotate_half` 把后半维取负后与前半维交换；cos/sin 由 `rotary_emb(hidden, position_ids)` 给出，形状 `[B,T,128]` |
| 6 | `past_key_values.update(k, v, layer_idx)` | 返回完整 K/V：`[B,8,S,128]` |
| 7 | `repeat_kv` | 用 expand + reshape 把 `[B,8,S,128]` 变成 `[B,32,S,128]`；`num_key_value_groups = 32/8 = 4` |
| 8 | 打分 | `scores = Q·Kᵀ·head_dim^-0.5`：`[B,32,T,S]`；加 causal/padding mask（加性 −inf）；在 fp32 中做 softmax |
| 9 | 加权求和 | `attn·V` → `[B,32,T,128]` → transpose/reshape → `[B,T,4096]` |
| 10 | `o_proj` + 残差 | `[B,T,4096]` → `[B,T,4096]`，然后 `+residual` |
| 11 | `post_attention_layernorm` → MLP | `down(silu(gate(x)) * up(x))`：gate/up 输出 `[B,T,12288]`，down 输出 `[B,T,4096]`；然后 `+residual` |
| 12 | 末端 | 36 层之后接 `norm` → `lm_head` |

- `lm_head` 的切片：`logits = lm_head(hidden[:, -logits_to_keep:])`，得到 `[B, T 或 1, 151936]`。`generate` 只取 `outputs.logits[:, -1]` 并转 fp32 — [generation/utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py)
- 两个 config 层面的细节 — [modeling_qwen3.py@v5.19.0](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3/modeling_qwen3.py)
  - 0.6B 的 `tie_word_embeddings=true`：`_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`，LM head 与 embedding 共享权重。
  - attention 实现通过 `ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, eager_attention_forward)` 可替换为 sdpa、flash 等。
- 该文件由 `modular_qwen3.py` 自动生成，改代码应读 modular 文件 — [modeling_qwen3.py 文件头](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3/modeling_qwen3.py)

**论文出处（arXiv API 元数据）**
- Transformer：Vaswani et al. 2017 — [1706.03762](https://arxiv.org/abs/1706.03762)
- MQA：Shazeer 2019 — [1911.02150](https://arxiv.org/abs/1911.02150)
- GQA：Ainslie et al. 2023。GQA 是 MQA 的推广，KV head 数介于 1 和 query head 数之间；摘要称 uptrained GQA "quality close to multi-head attention with comparable speed to MQA" — [2305.13245](https://arxiv.org/abs/2305.13245)
- RoPE：Su et al. RoFormer 2021 — [2104.09864](https://arxiv.org/abs/2104.09864)
- RMSNorm：Zhang & Sennrich 2019 — [1910.07467](https://arxiv.org/abs/1910.07467)
- SwiGLU：Shazeer 2020 GLU Variants — [2002.05202](https://arxiv.org/abs/2002.05202)
- MLA：DeepSeek-V2 2024 — [2405.04434](https://arxiv.org/abs/2405.04434)
- DeepSeek-V3 技术报告 — [2412.19437](https://arxiv.org/abs/2412.19437)
- Qwen3 技术报告 — [2505.09388](https://arxiv.org/abs/2505.09388)
- Llama 3 Herd — [2407.21783](https://arxiv.org/abs/2407.21783)

### Inferences
- RoPE 公式（HF 的 rotate_half 约定）：
  - 维度对 (i, i+d/2) 共用频率 θ_i = base^(−2i/d)，其中 base = rope_theta。
  - 位置 m 处：`q' = q·cos(mθ) + rotate_half(q)·sin(mθ)`。
  - 由于 q_m·k_n 只依赖 m−n，RoPE 只作用于 Q/K，**缓存的是已经旋转过的 K**。因此 decode 时只需对新 token 做一次 RoPE。
- Qwen3-32B 的 q_proj 是 5120→8192 的"扩张"投影，o_proj 是 8192→5120。读形状时不要默认 `heads×head_dim = hidden`。
- MoE 层（Qwen3-30B-A3B）中，MLP 换成：router logits `[B·T,128]` → top-8 → 被选专家的 SwiGLU（每个专家的中间维 768）→ 加权求和（`norm_topk_prob=true` 表示对 top-k 权重重新归一化）。attention 部分与 dense 模型相同。

### Gaps
- HF `generate` 是否在 prefill 阶段自动设置 `logits_to_keep=1`（避免算出完整的 `[B,T,V]` logits）本次未核实。`_sample` 中只看到取 `[:, -1]`。
- DeepSeek-V3 的 MLA 推理时"吸收"（absorb）W_UK/W_UV 的实现细节未展开，需读 vLLM 或 DeepSeek 的推理代码。

---

## Q3 计算与访存算术：参数量、FLOPs、KV cache、为何 prefill 是 compute-bound、decode 是 memory-bound

### Takeaway
每生成一个 token，前向约需 2N FLOPs（N 为参数量），外加随上下文线性增长的 attention 项。但 decode 每一步都要把**全部权重**（以及该序列的 KV）从 HBM 读一遍，batch=1 时算术强度约为 1 FLOP/byte，远低于 GPU 的 ridge point（约 300 FLOP/byte 量级），所以受带宽限制。prefill 一次处理 T 个 token，算术强度约为 T，因此受算力限制。

### Cited Findings
- Pope et al. 2022 原文：
  - "An N-parameter decoder-only model requires 2N matmul FLOPs in the forward pass per token seen…"；
  - attention matmul 对大模型而言 "typically add a much smaller number of FLOPs per token"；
  - 权重和 KV cache "need to be transferred from HBM to the compute cores … once per forward pass (prefill or decode step)"，这段时间称为 "memory time"；
  - "At small batch sizes and sequence lengths, the time to load weights dominates. At larger batch sizes and sequence lengths (e.g. 2048+ tokens with batch size 512+), the time to load the KV cache dominates."
  - 来源：[Efficiently Scaling Transformer Inference, arXiv 2211.05102（ar5iv HTML）](https://ar5iv.labs.arxiv.org/html/2211.05102)
- Pope 摘要：MQA 的低内存需求可以支撑最多 32× 更长的上下文；在 500B+ 模型上低 batch 的 decode 延迟为 29 ms/token — [2211.05102](https://arxiv.org/abs/2211.05102)
- Shazeer 2019 摘要：增量推理慢 "due to the memory-bandwidth cost of repeatedly loading the large 'keys' and 'values' tensors"；MQA 让 K/V 在所有 head 之间共享 — [1911.02150](https://arxiv.org/abs/1911.02150)
- DeepSeek-V2 Table 1 给出每 token 每层的 KV 元素数（l 为层数）— [DeepSeek-V2 HTML](https://arxiv.org/html/2405.04434v5)
  - MHA：2·n_h·d_h·l
  - GQA：2·n_g·d_h·l
  - MQA：2·d_h·l
  - MLA：(d_c + d_h^R)·l ≈ (9/2)·d_h·l
  - V2 中 d_c = 4d_h、d_h^R = d_h/2，"equal to GQA with only 2.25 groups"。摘要称相对 DeepSeek 67B，KV cache 减少 93.3%，最大生成吞吐提高到 5.76×。
- NVIDIA H100 产品页规格 — [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/)
  - 显存带宽：SXM 3.35 TB/s（80GB），NVL 3.9 TB/s（94GB）。
  - FP8 Tensor Core 3,958 TFLOPS（带 * "With sparsity"）。
  - NVLink 600GB/s（该行对应 NVL/PCIe 规格栏）。
- H800 的官方规格页（nvidia.com / nvidia.cn 的 `/data-center/h800/`）返回 404，**H800 规格「未核实」**。

**计算结果（bf16 = 2 bytes；由官方 config 计算，公式见 Inferences）**

| 模型 | 总参数 | 非 embedding | bf16 权重 | KV/token（bf16） | 若为 MHA | 32K 上下文 KV/序列 | 2N GFLOPs/token | attn GFLOPs/token @4K ctx |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.596B | 0.440B | 1.19 GB | 112 KiB | 224 KiB | 3.50 GiB | 1.19 | 0.94 |
| Qwen3-8B | 8.191B | 6.946B | 16.38 GB | 144 KiB | 576 KiB | 4.50 GiB | 15.14 | 2.42 |
| Qwen3-32B | 32.762B | 31.206B | 65.52 GB | 256 KiB | 2048 KiB | 8.00 GiB | 63.97 | 8.59 |
| Llama-3.1-8B | 8.030B | 6.980B | 16.06 GB | 128 KiB | 512 KiB | 4.00 GiB | 15.01 | 2.15 |
| Qwen3-30B-A3B | 30.53B | — | 61 GB | 96 KiB | — | 3.0 GiB | 激活约 3.04B → 约 6.1 | — |
| DeepSeek-V3（MLA） | — | — | — | 68.6 KiB（576 元素 × 61 层） | 解压后的 K(192)+V(128)×128 heads 为 4.77 MiB | — | — | — |

- 计算口径：
  - "2N" = 2 ×（非 embedding 参数 + LM head）。
  - Qwen3-30B-A3B 的"激活 3.04B"按 attention + 8 个专家 + router + LM head 计。官方宣称 "A3B / 3.3B 激活" 的具体口径「未核实」。
  - DeepSeek-V3 的 671B 总参 / 37B 激活取自其技术报告摘要的常见表述，本次未抓取摘要正文，「未核实」。

### Inferences
- **KV cache 公式**：`bytes/token = 2(K,V) × n_layers × n_kv_heads × head_dim × bytes_per_elem`。
  - 例：Qwen3-8B 为 2×36×8×128×2 = 147,456 B = 144 KiB。
  - 32 条 32K 上下文的并发请求需要 32 × 4.5 GiB = 144 GiB，单卡 80GB 装不下。这正是 PagedAttention、KV 量化、MLA 存在的理由。
  - GQA 把 KV 缩小 n_heads/n_kv_heads 倍（8B 为 4×，32B 为 8×）；MQA 缩到 1/n_heads；MLA 改为缓存低秩潜向量 c_KV（512）和共享的 RoPE key（64）。
- **FLOPs 中的 attention 项**：每个 decode token 在上下文长度 L 时需要 QKᵀ 和 AV 两次 matmul，各 2·n_h·d_h·L FLOPs，合计约 4·n_layers·n_h·d_h·L。
  - Qwen3-8B 在 4K 上下文时约 2.4 GFLOPs，约为 2N 的 16%；32K 时约 19 GFLOPs，超过 2N。长上下文下 attention 不能再忽略。
  - 这与 Pope "attention FLOPs 通常较小" 的说法在短上下文下一致。此项是我的推导。
- **Roofline 估算**（带宽用 H100 SXM 的 3.35 TB/s；BF16 dense 峰值约 989 TFLOPS 这一数字未在官方页面抓到，「未核实」，仅作量级参考）：
  - Qwen3-8B 在 batch=1 decode 时，每步读权重 16.4 GB ÷ 3.35 TB/s ≈ **4.9 ms/token 下界**（≤ 约 204 tok/s）。计算只需 15 GFLOPs ≈ 0.015 ms，强度约 1 FLOP/B。
  - ridge point ≈ 989e12 / 3.35e12 ≈ 295 FLOP/B，所以 decode 要把 batch 拉到约几百个 token 才接近 compute-bound；而 KV 读取量随 batch 线性增长，会进一步拖低强度。
  - prefill T=2048 时强度约 2048 FLOP/B，已经 compute-bound。
  - 这就是 continuous batching（把多请求的 decode 合批）和 chunked prefill（把 prefill 与 decode 混合）的物理动机。
- 用户熟悉的 OxyGen：它共享一次 observation prefill 并跨帧合批语言 decode，本质也是在摊薄"每步读全部权重"的固定成本。

### Gaps
- H800 的 HBM 带宽、BF16 峰值和 NVLink 带宽没有官方规格页可查（404），「未核实」。练习中建议用实测带宽（例如 `nvidia-smi` 或拷贝基准）替代。
- Kaplan et al. 2020 的 FLOPs 表格本次未抓取。attention 项公式是我的推导，没有直接引用原文。

---

## Q4 Hugging Face transformers（v5.19）中的 KV cache 与 generate 的 prefill→decode

### Takeaway
v5 的 Cache 是"按层的 Layer 对象列表"：DynamicLayer 用 `torch.cat` 增长，StaticLayer 预分配固定长度、用 `index_copy_` 原地写入。`generate` 先调用 `_prefill` 跑完整 prompt，再在 while 循环里每步喂 1 个 token。静态形状让 `torch.compile` 和 CUDA graphs 不必因长度变化而反复重编译或重捕获。在 v5.19 里，`cache_position` 已从模型 forward 中移除，只在 StaticLayer 内部和 remote-code 兼容路径里出现。

### Cited Findings
- v5.19.0 `cache_utils.py` 的类层次 — [cache_utils.py@v5.19.0](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/cache_utils.py)
  - Layer 类：`CacheLayerMixin`、`DynamicLayer`、`DynamicSlidingWindowLayer`、`DynamicIndexedLayer`、`StaticLayer`、`StaticSlidingWindowLayer`、`StaticIndexedLayer`、`QuantizedLayer`（Quanto / HQQ）、各类 `LinearAttention*Layer`。
  - Cache 类：`Cache`、`DynamicCache`、`StaticCache`、`QuantizedCache`、`EncoderDecoderCache`、`MtpCache`、`DFlashCache`。
  - 原 `SlidingWindowCache`、`HybridCache` 等已不是独立类，改为通过 Layer 组合实现。这是从类列表得出的推断。
- `DynamicLayer.update`：`self.keys = torch.cat([self.keys, key_states], dim=-2)`。`StaticLayer.update`：`cache_position = torch.arange(kv_length) + self.cumulative_length`，然后 `self.keys.index_copy_(2, cache_position, key_states)`（MPS 上退回索引赋值）— [同上](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/cache_utils.py)
- `generation/utils.py@v5.19.0` 中 `cache_position` 只出现在 remote-code 兼容段（"This arg has been … Please open a PR on the remote code hub repo to remove any usage of `cache_position`"）。Qwen3 的 forward 中完全没有它 — [generation/utils.py@v5.19.0](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/generation/utils.py)
- `_sample` 流程（main，与 v5.19 的结构一致，`_prefill` 存在）— [generation/utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py)
  1. `outputs = self._prefill(...)`。支持 `generation_config.prefill_chunk_size` 分块 prefill，此时把 `torch._dynamo` 的 `cache_size_limit` 设为 64。
  2. 循环：`prepare_inputs_for_generation` → `model_forward`（满足条件时用 `get_compiled_call(compile_config)` 编译后的版本）→ `_update_model_kwargs_for_generation`。
  3. 取 `next_token_logits = outputs.logits[:, -1].to(fp32)` → `logits_processor(input_ids, logits)` → `torch.multinomial` 或 `argmax`。
  4. 已结束的序列用 pad 填充；`torch.cat` 拼接到 input_ids；用 `stopping_criteria` 更新 `unfinished_sequences`。
- 模型内部 `Qwen3Model.forward`：`use_cache` 且未传 cache 时，新建 `DynamicCache(config=self.config)` — [modeling_qwen3.py@v5.19.0](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3/modeling_qwen3.py)
- kv_cache 文档 — [kv_cache.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/kv_cache.md)
  - DynamicCache 是所有模型的默认 cache，"prevents you from taking advantage of most just-in-time (JIT) optimizations because the cache size isn't fixed"。
  - `cache_implementation="static"` 会启用 StaticCache，并对 greedy/sample 的 decode 阶段自动编译。
  - offloading 通过 `"offloaded"` / `"offloaded_static"` 启用；量化通过 `"quantized"` 启用。
  - 对比表：

    | Cache | Sliding | Offload | torch.compile | 内存 |
    |---|---|---|---|---|
    | Dynamic | Yes | Yes | No | Medium |
    | Static | Yes | Yes | Yes | High |
    | Quantized | No | No | No | Low |

  - 文档示例：`StaticCache(config=model.config, max_cache_len=1024)` 预填充 prompt 后复用，从同一个 prefill 生成多个续写（与 OxyGen 共享 prefix 的思路同构）。
- cache_explanation 文档："StaticLayer and StaticSlidingWindowLayer have a fixed sequence length … compatible with torch.compile" — [cache_explanation.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/cache_explanation.md)

### Inferences
- 为什么静态形状重要：
  - torch.compile 按输入形状特化（guard），CUDA graph 捕获固定的指针和形状。
  - DynamicCache 每步 `cat` 都会产生新长度和新分配，导致重编译，或无法 replay graph。
  - StaticCache 预分配 `[B, n_kv, max_len, d]`，写入位置变成张量数据（`cumulative_length + arange`）而不是形状，attention 再用 mask 屏蔽未写入的部分。
  - vLLM/SGLang 的 paged KV + 按 batch-size 分桶的 CUDA graph 是同一思想的工业化版本。
- decode 步的 `position_ids = past_len + arange(1)`，每层 `update` 把新 K/V 写到第 past_len 位。prefill 一次性写入前 T 个位置。

### Gaps
- `_valid_auto_compile_criteria` 的具体条件，以及默认的 compile_config，本次未逐行核实。
- transformers v5 移除 `cache_position` 的 release note 原文未抓取，结论来自 v5.19.0 源码。

---

## Q5 Sampling：定义、logits 处理顺序、logprobs、随机性与非确定性、停止条件

### Takeaway
HF 和 vLLM 的处理顺序不同：
- HF：penalty 类 processor → 用户自定义 processor → temperature → top-k → top-p → min-p → …
- vLLM：先做 argmax 会改变的处理（min_tokens、logit_bias、penalties），再 temperature → min-p → top-k/top-p。

非确定性的主因不是"GPU 原子操作的随机竞争"，而是 kernel 缺乏 batch invariance：同一个请求在不同 batch 大小下，归约顺序不同。vLLM 提供 `VLLM_BATCH_INVARIANT=1`（beta）。

### Cited Findings
**HF transformers 的处理顺序**（`_get_logits_processor`，main L1406–1640）— [generation/utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py)
1. UnbatchedClassifierFreeGuidance
2. SequenceBias
3. EncoderRepetitionPenalty
4. RepetitionPenalty
5. NoRepeatNGram
6. EncoderNoRepeatNGram
7. NoBadWords
8. MinLength
9. MinNewTokensLength
10. PrefixConstrained
11. ForcedBOS / ForcedEOS
12. InfNanRemove
13. ExponentialDecayLengthPenalty
14. SuppressTokens / SuppressTokensAtBegin
15. 合并用户的 `logits_processor`
16. 仅当 `do_sample` 时：Temperature → TopH → TopK → TopP → MinP（注释说明 min-p 在 temperature 之后，参见 llama.cpp#3841）→ Typical → Epsilon → Eta
17. Watermark
18. `LogitNormalization` 永远在最后（`renormalize_logits=True` 时）

**vLLM 0.31.0 Sampler 的 docstring 顺序** — [vllm/v1/sample/sampler.py@v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/sample/sampler.py)
1. 若请求 logprobs，先在**原始 logits** 上计算（默认 `logprobs_mode="raw_logprobs"`；注释说明这与 V0 不同，V0 用的是处理后的 logits）。
2. 转为 fp32。
3. allowed_token_ids 白名单。
4. bad words。
5. 非 argmax-invariant 的 processor：min_tokens、logit_bias。
6. penalties：repetition → frequency → presence。
7. 采样：
   - 非全随机时先算 greedy；
   - temperature（< 1e-5 视为 greedy）；
   - argmax-invariant 的 processor（默认 min_p）；
   - top-k / top-p；
   - 采样。
8. 收集 top-`logprobs` 加上被采样 token 的 logprob（最多 `logprobs+1` 个）。

**vLLM 的具体公式**
- penalties（main）：repetition 走自定义算子 `apply_repetition_penalties(logits, prompt_mask, output_mask, p)`，作用于出现在 prompt **或** output 中的 token。之后 `logits -= frequency_penalty × output_bin_counts`、`logits -= presence_penalty × (output_counts>0)`。注释："We follow the definition in OpenAI API" — [vllm/model_executor/layers/utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/utils.py)
- min-p：`probs = softmax(logits)`；凡 `probs < min_p × max(probs)` 的位置置为 −inf — [builtin.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/logits_processor/builtin.py)
- logit_bias：`logits[idx] += bias`，属于非 argmax-invariant — [builtin.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/logits_processor/builtin.py)
- 随机采样不用 `torch.multinomial`（"causes CPU-GPU synchronization"）。做法是 `q.exponential_()` 后取 `argmax(probs / q)`。带 seed 的请求用各自的 `torch.Generator` 逐行生成 q。top-k/top-p 在有 Triton 时走 Triton kernel，否则用 PyTorch 的排序实现 — [topk_topp_sampler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py)

**vLLM SamplingParams 默认值** — [vllm/sampling_params.py](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py)
- `temperature=1.0`（0 表示 greedy）、`top_p=1.0`、`top_k=0`（0 或 -1 表示不限）、`min_p=0.0`。
- `presence/frequency_penalty=0.0`、`repetition_penalty=1.0`、`seed=None`。
- `stop`：输出中不含 stop 字符串，除非 `include_stop_str_in_output`。
- `stop_token_ids`：输出包含这些 token，除非它们是 special token。
- `ignore_eos=False`、`max_tokens=16`、`min_tokens=0`、`logprobs=None`、`logit_bias`、`allowed_token_ids`、`skip_special_tokens=True`。

**seed 与可复现性** — [docs/usage/reproducibility.md](https://github.com/vllm-project/vllm/blob/main/docs/usage/reproducibility.md)
- 默认不保证可复现。
- 离线模式可设 `VLLM_ENABLE_V1_MULTIPROCESSING=0`（使调度确定），或开启 batch invariance；在线模式只能开 batch invariance。
- 即便如此，也只在相同硬件、相同 vLLM 版本下可复现。
- V1 的全局 `seed` 默认为 0。

**Batch invariance** — [docs/features/batch_invariance.md](https://github.com/vllm-project/vllm/blob/main/docs/features/batch_invariance.md)
- 目前为 beta。
- 用 `VLLM_BATCH_INVARIANT=1` 开启，例如 `VLLM_BATCH_INVARIANT=1 vllm serve meta-llama/Llama-3.1-8B-Instruct`。
- 硬件要求为 NVIDIA compute capability ≥ 8.0（另支持部分 AMD、Intel XPU）。
- 实现位于 `vllm/model_executor/determinism/batch_invariant.py`。

**Thinking Machines 实验**（Horace He 等，2025-09-10）— [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- 设置：Qwen3-235B-A22B-Instruct-2507，temperature 0，1000 次 completion，每次 1000 tokens。
- 结果：得到 **80 种不同输出**，最常见的出现 78 次；前 102 个 token 全部相同。
- 结论：常见的 "concurrency + floating point" 假说不是主因；真正原因是 forward 缺乏 "batch invariance"，请求的输出依赖于同批的其他请求。
- 解决方案是 batch-invariant kernel，见 thinking-machines-lab/batch-invariant-ops 和 vLLM 的 deterministic 模式。

**论文定义**
- Nucleus（top-p）：从 "dynamic nucleus of the probability distribution" 中采样 — [Holtzman et al. 1904.09751](https://arxiv.org/abs/1904.09751)
- min-p："uses the top token's probability as a scaling factor" 的动态截断 — [Nguyen et al. 2407.01082](https://arxiv.org/abs/2407.01082)

### Inferences
- **数值例子（4 个 token 的 logits = [2.0, 1.0, 0.5, −1.0]）**
  - T=1 时 softmax ≈ [0.610, 0.224, 0.136, 0.030]。
  - T=0.5 时 logits 变为 [4, 2, 1, −2]，softmax ≈ [0.842, 0.114, 0.042, 0.002]。
  - 在 T=0.5 上做 top-p=0.9：累积概率 0.842 → 0.956 ≥ 0.9，保留前 2 个，重新归一化为 [0.881, 0.119]。
  - min-p=0.1：阈值 0.0842，同样保留前 2 个。
  - top-k=1 等价于 greedy。
  - repetition_penalty=1.2 作用于已出现的 token 0：正 logit 除以 1.2，2.0→1.667；负 logit 乘以 1.2，例如 −1.0→−1.2。此为 HF/CTRL 惯例，vLLM 的算子语义与之一致，但本次只看到算子调用，未读 CUDA 实现，「未核实」细节。
  - 顺序不同会改变结果：min-p 在 temperature 前后的阈值不同。HF 与 vLLM 都把 min-p 放在 temperature 之后，但 vLLM 把 min-p 放在 top-k/top-p 之前，HF 放在之后，极端设置下两者结果可能不同。
- `argmax(p/q)`（q ~ Exp(1)）是 Gumbel-max 技巧的等价形式（指数竞赛）：选中 i 的概率正好是 p_i。它避免了 multinomial 的同步开销。
- **beam search**：保持 k 条前缀，按累计 log-prob 扩展并截断。适合翻译等确定性任务，对开放式对话会放大 Holtzman 所说的退化问题。vLLM 的 beam search 接口细节本次未核实。
- **停止条件的层次**：
  - token 级：EOS（Qwen3 generation_config 中为 [151645, 151643]）、`stop_token_ids`、`max_tokens`，在 engine/scheduler 中检查；
  - 字符串级：`stop`，在 detokenizer 中检查，需要缓冲尾部字符，以免把 stop 串的前缀先发给用户；
  - `min_tokens` 通过 MinTokens processor 在达到长度前屏蔽 stop token。

### Gaps
- vLLM 的 `apply_repetition_penalties` CUDA kernel 的精确公式未读源码。
- vLLM beam search 和 `n>1` 的实现路径未覆盖。
- HF 的 `TopHLogitsWarper` 是新加入的 warper，其定义未查。

---

## Q6 结构化 / 约束解码：XGrammar、llguidance、Outlines 的 mask 机制、开销与默认后端

### Takeaway
三者都是每一步算一个 vocab 大小的 bitmask，把非法 token 的 logits 置为 −inf：
- Outlines 把 regex/JSON 编译成 FSM，并预建"状态 → 合法 token"的索引；
- XGrammar 处理 CFG（下推自动机）：把 token 分为上下文无关（预先检查好）和上下文相关（运行时用 persistent stack 检查）两类，并让 mask 计算与 GPU 前向重叠；
- llguidance 用 Earley 解析 + 词法分析，CPU 上约 50μs/token。

默认后端：vLLM 为 `auto`（先试 xgrammar，不支持时回退 guidance，必要时再回退 outlines）；SGLang 默认 xgrammar。

### Cited Findings
- Outlines（Willard & Louf 2023）把生成重写为有限状态机的状态转移，在词表上构建索引，用于 regex 和 CFG 引导。摘要称 "adds little overhead" — [2307.09702](https://arxiv.org/abs/2307.09702)
- XGrammar（Dong et al. 2024）的四个关键点："dividing the vocabulary into context-independent tokens that can be prechecked and context-dependent tokens"、扩展语法上下文以减少上下文相关 token、"persistent stack"、"overlap grammar computation with GPU executions"。摘要称比现有方案快至多 100×，端到端 "near-zero overhead" — [2411.15100](https://arxiv.org/abs/2411.15100)
- llguidance：给定 CFG、tokenizer 和已有前缀，算出合法 token mask。"Mask computation takes approximately 50μs of single-core CPU time for a tokenizer with 128k tokens"（依赖语法，对 JSON schema 衍生语法成立），"negligible startup costs"。2025-06-23 进入 v1.0.0 — [llguidance README](https://github.com/guidance-ai/llguidance/blob/main/README.md)
- vLLM 结构化输出配置 — [config/structured_outputs.py@main](https://github.com/vllm-project/vllm/blob/main/vllm/config/structured_outputs.py)
  - `StructuredOutputsConfig.backend` 可选 `"auto" | "xgrammar" | "guidance" | "outlines" | "lm-format-enforcer"`，默认 `"auto"`。
  - 官方注释："behavior is subject to change in each release"。
- `auto` 逻辑（v0.31.0）— [sampling_params.py@v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/sampling_params.py)
  - 先 `validate_xgrammar_grammar`，成功则用 xgrammar。
  - 失败时回退到 guidance。
  - 若是非 tekken 的 Mistral tokenizer，或 schema 含 guidance 不支持的特性，则回退 outlines。
  - 特例：Tekken Mistral tokenizer 配 Lark 语法时直接走 guidance。
- 文档说明可通过 `--structured-outputs-config.backend` 指定后端。旧字段 `guided_json/guided_regex/...` 已于 v0.12.0 移除，改为 `{"structured_outputs": {"json": ...}}` 或 `StructuredOutputsParams(json=...)` — [docs/features/structured_outputs.md@v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/features/structured_outputs.md)
- SGLang 默认后端：
  - `ServerArgs.grammar_backend` 字段默认为 `None` — [arg_groups/fields/exec_.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/arg_groups/fields/exec_.py)
  - `handle_grammar_backend` 在其为 None 时解析为 `"xgrammar"` — [arg_groups/serving_hook.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/arg_groups/serving_hook.py)
  - 文档原文："XGrammar(default)… If no backend is specified, XGrammar will be used as the default"，也可用 `--grammar-backend llguidance` 或 `none` — [structured_outputs.mdx](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/structured_outputs.mdx)

### Inferences
- 每步流程：
  1. 用上一步的 token 推进 grammar 状态（matcher.accept）；
  2. 在 CPU 上为下一步填 `[B, ceil(V/32)]` 的 int32 bitmask；
  3. GPU 把 bitmask 应用到 logits（非法位置置 −inf）后再采样。
- 开销来源：
  - 每步 CPU 侧 mask 计算（llguidance 约 50μs，XGrammar 依靠预检查和缓存）；
  - grammar 编译（首次请求时编译，可缓存）；
  - CPU 与 GPU 之间的同步。
- XGrammar 的"与 GPU 重叠"做法：在 GPU 跑第 t 步前向的同时，CPU 计算第 t 步的 mask。
- 对 Qwen3 这类 thinking 模型，需要 reasoning parser 配合，让约束只作用于 `</think>` 之后的内容。SGLang 的 `enable_strict_thinking`、`reasoner_grammar_backend.py` 和 vLLM 的 `reasoning_parser` 都与此相关，细节「未核实」。

### Gaps
- 没有拿到 XGrammar / llguidance / Outlines 在同一硬件上的 per-token 微基准数字（论文中的 100× 是相对旧方案的）。建议用练习 4 自测。
- vLLM `auto` 模式下 bitmask 应用的 kernel 位置（`vllm/v1/worker/gpu/structured_outputs.py`）未细读。

---

## Q7 动手练习（适配 4×H800；只用本次核实过的 API；版本按上述基线）

### Takeaway
四个练习都在单卡或 CPU 上，每个 1–3 小时：
1. tokenizer、chat template 与流式缓冲；
2. 手写 prefill/decode 循环，对照 roofline 理解访存瓶颈；
3. vLLM 的采样顺序、logprobs 与 batch invariance；
4. 结构化输出后端的开销对比。

### Cited Findings
- 可用 API 及其出处：
  - `tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)`、`DynamicCache(config=model.config)`、`StaticCache(config=model.config, max_cache_len=1024)`、`cache_implementation="static"` — [kv_cache.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/kv_cache.md)
  - `AutoTokenizer.from_pretrained(..., padding_side="left")` — [llm_tutorial.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/llm_tutorial.md)
  - `TextStreamer(tok)` 与 `model.generate(**inputs, streamer=streamer, max_new_tokens=20)` — [streamers.py](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py)
  - `LLM(model=..., tensor_parallel_size=1)`、`SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100, seed=42)`、`llm.generate(prompts, sampling_params)`、`VLLM_BATCH_INVARIANT=1` — [batch_invariance.md](https://github.com/vllm-project/vllm/blob/main/docs/features/batch_invariance.md)
  - `--structured-outputs-config.backend`、`StructuredOutputsParams(json=...)` — [structured_outputs.md](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/features/structured_outputs.md)

### Inferences
**练习 1：tokenizer、chat template 与流式 UTF-8（CPU，约 1.5 小时）**
- 目标：亲眼看到 prompt 字符串、token id、半个汉字的 token。
- 步骤：
  1. 加载 `Qwen/Qwen3-0.6B` 的 tokenizer，用 `apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` 打印字符串。
  2. 对比 `enable_thinking=False` 时末尾是否多出 `<think>\n\n</think>\n\n`（kwarg 透传「未核实」，以实际输出为准）。
  3. 对 "Hello world" 和 "龘" 调用 `encode`，逐个 token 打印 `tok.convert_ids_to_tokens`。
  4. 写一个 20 行的增量解码器：每来一个 token 就 decode 全部缓存；输出以 "�" 结尾时暂不输出。再与 TextStreamer 的行为对照。
- 预期：`Hello world` → [9707, 1879]；`龘` → [82912, 246]；单独 decode 82912 得到 "�"。
- 验收：自写 streamer 对含生僻字的中文从不输出 "�"，最终文本与一次性 decode 逐字节一致。

**练习 2：手写 prefill / decode 循环 + 对照 roofline（1×H800，约 3 小时）**
- 目标：量化 prefill（compute-bound）与 decode（memory-bound）的差异，以及 KV cache 的作用。
- 步骤：
  1. Qwen3-8B 以 bf16 加载。
  2. 手写循环：先 `out = model(**inputs, past_key_values=DynamicCache(config=model.config), use_cache=True)`，之后每步只喂 `next_token[:, None]`。
  3. 用 `torch.cuda.synchronize()` 加计时，记录 T∈{128, 512, 2048} 的 prefill 时间，以及 B∈{1, 8, 32, 128} 的每步 decode 时间。
  4. 对照组 A：不用 cache，每步喂完整序列。
  5. 对照组 B：`generate(..., cache_implementation="static")`。
  6. 每次检查 `past_key_values.get_seq_length()` 与 `torch.cuda.memory_allocated()` 的增量。
- 预期：
  - B=1 时 decode 每 token ≥ 16.4 GB ÷ 实测 HBM 带宽（以 H100 SXM 3.35 TB/s 计约 4.9 ms，H800 规格未核实）；
  - B 从 1 增到 32，每步时间几乎不变（memory-bound）；
  - KV 显存增量约为 144 KiB × token 数 × B；
  - 不用 cache 时每步耗时随长度线性增长。
- 验收：画出"tokens/s vs B"曲线，并用 roofline 公式解释拐点；KV 实测值与公式误差 < 10%。

**练习 3：vLLM 的采样顺序、logprobs 与非确定性（1×H800，约 2 小时）**
- 目标：复现 batch 依赖的非确定性，并验证 batch invariance。
- 步骤：
  1. `LLM(model="Qwen/Qwen3-8B", tensor_parallel_size=1)`。
  2. 用 `SamplingParams(temperature=0, max_tokens=256, logprobs=5)` 对同一 prompt 分别跑：单独 1 条，以及混在 64 条不同 prompt 里。
  3. 比较两次的 token 序列和首个分歧位置的 logprob 差。
  4. 设 `VLLM_BATCH_INVARIANT=1` 后重复。
  5. 再用 temperature=0.7、seed=42 测试 min_p / top_p 的组合，对照 Q5 的数值例子手算一个 4-token 玩具分布。
- 预期：默认模式下长输出可能在中途分歧，概率较小但可观测；开启 batch invariance 后应完全一致，吞吐有所下降。
- 验收：报告分歧率和吞吐比；能指出 vLLM 默认返回的 logprobs 来自原始 logits（raw_logprobs），而不是 temperature 处理之后的分布。

**练习 4：结构化输出后端开销（1×H800，约 1.5 小时）**
- 目标：测量 grammar mask 的成本。
- 步骤：
  1. 分别以 `--structured-outputs-config.backend xgrammar` 和 `guidance` 启动 `vllm serve Qwen/Qwen3-8B`。
  2. 用 OpenAI 客户端在 `extra_body={"structured_outputs": {"json": schema}}` 中传一个嵌套 JSON schema。
  3. 并发 1 / 32 / 128，对比无约束时的 TPOT 与吞吐，并统计首请求编译延迟。
- 预期：稳态 TPOT 增加很小（XGrammar 宣称接近零开销，llguidance 约 50μs/token CPU）；首次 schema 编译有可见延迟；输出 100% 能被 `json.loads` 解析并通过 schema 校验。
- 验收：给出两个后端在三个并发度下的 TPOT 差异表，以及 schema 合法率。

### Gaps
- 以上命令未在 H800 上实跑，预期值来自公式和文档。vLLM / transformers 需由用户按官方方式自行安装（本次按约束不安装）。
- `enable_thinking` 的 kwarg 透传以实际运行结果为准。
