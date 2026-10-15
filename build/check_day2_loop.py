# 验证指南 Day 2 手写 prefill/decode 循环（用随机小 Qwen3 代替 Qwen3-8B，不需要下载）
import time, torch
from transformers import AutoModelForCausalLM, DynamicCache
name = "labs/out/tiny_hf_qwen3"
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda()
torch.manual_seed(0)
inputs = {"input_ids": torch.randint(0, model.config.vocab_size, (1, 32), device="cuda")}
inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
cache = DynamicCache(config=model.config)
torch.cuda.synchronize(); t0 = time.perf_counter()
out = model(**inputs, past_key_values=cache, use_cache=True)      # prefill
torch.cuda.synchronize(); print("prefill s", time.perf_counter() - t0)
nxt = out.logits[:, -1].argmax(-1, keepdim=True)
toks = [nxt]
for _ in range(64):                                                # decode
    out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
    nxt = out.logits[:, -1].argmax(-1, keepdim=True); toks.append(nxt)
print(cache.get_seq_length(), torch.cuda.memory_allocated() / 2**30)
ref = model.generate(**inputs, max_new_tokens=65, do_sample=False)[:, 32:]
mine = torch.cat(toks, 1)
print("seq_len == 32+64:", cache.get_seq_length() == 96, "| matches generate():", torch.equal(ref, mine))
