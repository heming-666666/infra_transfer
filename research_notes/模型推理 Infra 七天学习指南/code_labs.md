# 代码实验（labs）验证记录

记录时间：2026-10-15。代码目录：`D:\Code\inference_infra_guide_20261014\labs\`。
本文只记录**实际执行过**的命令和输出摘录。没有执行的部分标为「未在本机运行」，并写明原因。
完整日志见 `labs/out/*.log`（每次验证都覆盖重写）。

## 1. 环境（实测）

| 项 | CPU 环境 | GPU 环境 |
|---|---|---|
| 解释器 | `D:\anaconda\envs\holosoma_retargeting\python.exe` | `D:\anaconda\envs\env_isaaclab\python.exe` |
| Python | 3.11.15 | 3.10.20 |
| torch | 2.13.0+cpu（gloo 可用） | 2.5.1+cu121（CUDA 可用） |
| 其他 | numpy 2.3.5；**没有 transformers** | transformers 5.13.0、tokenizers 0.22.2、safetensors 0.8.0、numpy 2.2.6 |
| 硬件 | — | NVIDIA GeForce RTX 4070 Ti SUPER，16376 MiB，驱动 610.88（`nvidia-smi`） |

Shell 为 Windows 11 上的 Git Bash。下文命令都在 `labs/` 目录下执行，并约定：

```bash
C=/d/anaconda/envs/holosoma_retargeting/python.exe   # CPU 环境
G=/d/anaconda/envs/env_isaaclab/python.exe           # GPU 环境
```

没有安装、升级或卸载任何包，也没有下载任何模型权重或数据集。

## 2. 汇总

| 实验 | CPU 环境 `--test` | GPU 环境 `--test` | 备注 |
|---|---|---|---|
| lab01 | 8/8 PASS | 8/8 PASS（py3.10） | 纯 Python |
| lab02 | 4/4 PASS | `--device cuda` 4/4 PASS | |
| lab03 | 无法运行（CPU 环境没有 transformers） | `--device cpu` 4/4 PASS；`--device cuda` 4/4 PASS | `--pretrained <hf_id>` 未在本机运行 |
| lab04 | 16/16 PASS | `--device cuda` 16/16 PASS | |
| lab05 | 9/9 PASS | `--device cuda` 9/9 PASS | |
| lab06 | 13/13 PASS | 13/13 PASS（py3.10） | 纯 Python |
| lab07 | `--spawn 2` gloo 2/2 PASS | `--spawn 2` gloo 2/2 PASS | torchrun 在 Windows 上失败（见 §5）；NCCL 未在本机运行 |
| lab08 | 13/13 PASS | 13/13 PASS（py3.10） | 纯 Python |
| lab09 | 3/3 PASS | `--device cuda` 3/3 PASS + 1 SKIP | CUPTI 不可用，GPU kernel 轨迹检查 SKIP |
| lab10 | 11/11 PASS | `--device cuda` 11/11 PASS | |

两个环境都跑了 `python -m py_compile`，10 个文件全部通过：

```bash
for P in "$C" "$G"; do for f in lab*.py; do "$P" -m py_compile "$f" || echo "COMPILE FAIL $P $f"; done; done
# 输出：没有任何 COMPILE FAIL（另外 echo 了 "py_compile done"）
```

最终一轮完整验证中，所有命令的退出码都是 0（来自最终一轮的逐条汇总）：

```
cpu_lab01_bpe_toy exit=0 PASS=8 FAIL=0 SKIP=0
cpu_lab02_tiny_llama exit=0 PASS=4 FAIL=0 SKIP=0
cpu_lab04_sampling exit=0 PASS=16 FAIL=0 SKIP=0
cpu_lab05_paged_kv exit=0 PASS=9 FAIL=0 SKIP=0
cpu_lab06_scheduler_sim exit=0 PASS=13 FAIL=0 SKIP=0
cpu_lab08_capacity_calculator exit=0 PASS=13 FAIL=0 SKIP=0
cpu_lab09_profile exit=0 PASS=3 FAIL=0 SKIP=0
cpu_lab10_multimodal_toy exit=0 PASS=11 FAIL=0 SKIP=0
cpu_lab07_tensor_parallel exit=0 PASS=2 FAIL=0 SKIP=0
gpuenv_cpu_lab03_hf_equivalence exit=0 PASS=4 FAIL=0 SKIP=0
gpu_lab02_tiny_llama exit=0 PASS=4 FAIL=0 SKIP=0
gpu_lab03_hf_equivalence exit=0 PASS=4 FAIL=0 SKIP=0
gpu_lab04_sampling exit=0 PASS=16 FAIL=0 SKIP=0
gpu_lab05_paged_kv exit=0 PASS=9 FAIL=0 SKIP=0
gpu_lab09_profile exit=0 PASS=3 FAIL=0 SKIP=1
gpu_lab10_multimodal_toy exit=0 PASS=11 FAIL=0 SKIP=0
gpuenv_lab01_bpe_toy exit=0 PASS=8 FAIL=0 SKIP=0
gpuenv_lab06_scheduler_sim exit=0 PASS=13 FAIL=0 SKIP=0
gpuenv_lab08_capacity_calculator exit=0 PASS=13 FAIL=0 SKIP=0
gpuenv_lab07_tensor_parallel exit=0 PASS=2 FAIL=0 SKIP=0
gpu_lab03_pretrained_localdir exit=0 PASS=2 FAIL=0 SKIP=0
gpu_lab03_pretrained_localdir_bf16 exit=0 PASS=2 FAIL=0 SKIP=0
cpu_lab08_on_tiny_config exit=0 PASS=0 FAIL=0 SKIP=0   # 计算器正常运行模式，无断言
```

## 3. 逐个实验的命令与输出摘录

### lab01_bpe_toy.py（字节级 BPE + 流式反分词）

```bash
"$C" lab01_bpe_toy.py --device cpu --test > out/cpu_lab01_bpe_toy.log
```

```
learned 289 merges, vocab size 545
longest tokens: ['词元延', '的请求', '键值缓存', '推理服务', '词元延迟']
sample: '推理引擎用 KV cache 加速解码，鲲鹏🚀' -> 50 bytes, 26 tokens
naive per-token : 推理|�|�|�|�|�|用| KV| cache| |加|�|�|�|解码|，|�|�|...
streaming pieces: 推理||引|||擎|用| KV| cache| |加|||速|解码|，|||鲲|||鹏||||🚀|
PASS merges learned >= 200 | 289 merges
PASS round-trip decode(encode(s)) == s | 302 texts
PASS streaming output == full decode
PASS streaming emits no U+FFFD
PASS byte-by-byte streaming ok
PASS matches codecs incremental decoder
PASS naive per-token decode DOES produce U+FFFD
PASS unseen CJK char -> multiple tokens | encode('鲲')=[233, 178, 178]
8/8 checks passed
```

说明：设定的上限是 300 次合并，但语料里再找不到出现 ≥2 次的相邻对，训练提前停在 289 次。
「鲲」没在语料中出现过，被编码成 3 个字节 token（UTF-8 为 e9 b2 b2）。

### lab02_tiny_llama.py（最小解码器 + KV cache）

```bash
"$C" lab02_tiny_llama.py --device cpu --test
"$G" lab02_tiny_llama.py --device cuda --test
```

```
device=cpu torch=2.13.0+cpu params=3.03M
PASS prefill+decode logits == full forward | max|diff|=1.431e-06
PASS greedy with cache == without cache | tokens=[180, 454, 377, 208, 79, 426, 287, 169]...
PASS KV bytes formula (torch.float32) | measured=163840 = 2*4*2*32*4B*40tok*2batch
PASS KV bytes formula (torch.bfloat16) | measured=81920 = 2*4*2*32*2B*40tok*2batch
prefill 256 tok: 8.58 ms (33.5 us/tok) | decode: 2.07 ms/tok | KV=576 KiB

device=cuda torch=2.5.1+cu121 params=3.03M
PASS prefill+decode logits == full forward | max|diff|=6.258e-07
PASS greedy with cache == without cache | tokens=[368, 110, 475, 354, 308, 489, 362, 217]...
PASS KV bytes formula (torch.float32) | measured=163840 = 2*4*2*32*4B*40tok*2batch
PASS KV bytes formula (torch.bfloat16) | measured=81920 = 2*4*2*32*2B*40tok*2batch
prefill 256 tok: 6.69 ms (26.1 us/tok) | decode: 4.81 ms/tok | KV=576 KiB
```

说明：模型只有 3M 参数，GPU 上一次 decode 要发射几十个小 kernel，耗时主要是 Python 和
kernel 启动开销，所以 GPU 的 decode 反而比 CPU 慢。这些计时不能代表 H800 上的性能。
两个环境里贪心生成的 token 不同，是因为 `torch.randint` 在 CUDA 上用的是另一套随机数发生器；
两边各自都满足“有 cache = 无 cache”。

### lab03_hf_equivalence.py（HF 权重 → lab02，logits 对齐）

```bash
"$G" lab03_hf_equivalence.py --device cpu --test
"$G" lab03_hf_equivalence.py --device cuda --test
```

```
device=cpu torch=2.5.1+cu121 transformers=5.13.0
PASS llama: lab02 logits == HF logits (fp32) | max|diff|=0.000e+00
PASS llama: lab02 greedy(KV cache) == HF generate
PASS qwen3: lab02 logits == HF logits (fp32) | max|diff|=0.000e+00
PASS qwen3: lab02 greedy(KV cache) == HF generate

device=NVIDIA GeForce RTX 4070 Ti SUPER torch=2.5.1+cu121 transformers=5.13.0
PASS llama: lab02 logits == HF logits (fp32) | max|diff|=2.384e-06
PASS llama: lab02 greedy(KV cache) == HF generate
PASS qwen3: lab02 logits == HF logits (fp32) | max|diff|=1.907e-06
PASS qwen3: lab02 greedy(KV cache) == HF generate
```

测试配置：vocab 320、hidden 128、3 层、8 个 Q 头、2 个 KV 头、head_dim 32（故意让
head_dim ≠ hidden/heads），`attn_implementation="eager"`。Qwen3 用绑定词嵌入并带 q_norm/k_norm，
Llama 不绑定。所有权重都重新随机化（norm 权重设为 1+0.1·randn），这样一旦映射出错就能发现。
本地测试路径在导入 transformers 之前就设置了 `HF_HUB_OFFLINE=1`。

**失败路径（验证测试确实能发现错误）**：

- 故意把 `mlp.gate_proj` 和 `mlp.up_proj` 的映射对调，然后调用 `run_tiny(cpu)`：
  `FAIL llama ... max|diff|=4.163e+00`，`FAIL qwen3 ... max|diff|=3.908e+00`，两个 greedy 检查也都 FAIL。
- 把 rope 改成 `{"rope_type": "linear", ...}` 再调用 `load_into_lab02`：脚本以“不支持的配置”退出。

**`--pretrained` 路径**：没有用真实的 hub 模型 ID 跑过——那需要下载权重，「未在本机运行」。
作为替代演练，先把随机小 Qwen3 存成 HF 格式，再离线从这个本地目录加载：

```bash
HF_HUB_OFFLINE=1 "$G" lab03_hf_equivalence.py --save-tiny out/tiny_hf_qwen3
HF_HUB_OFFLINE=1 "$G" lab03_hf_equivalence.py --device cuda --pretrained out/tiny_hf_qwen3
HF_HUB_OFFLINE=1 "$G" lab03_hf_equivalence.py --device cuda --dtype bfloat16 --pretrained out/tiny_hf_qwen3
```

```
[info] 无可用分词器（AssertionError），使用随机 token
PASS pretrained logits match (rel max diff, torch.float32) | 4.769e-07
PASS pretrained greedy == HF generate
...
PASS pretrained logits match (rel max diff, torch.bfloat16) | 0.000e+00
PASS pretrained greedy == HF generate
```

### lab04_sampling.py（logits 处理器与采样）

```bash
"$C" lab04_sampling.py --device cpu --test
"$G" lab04_sampling.py --device cuda --test
```

```
PASS kept set top_k=3 | [1, 3, 5]
PASS kept set top_k=1 keeps ties | [1, 2]
PASS kept set top_p=0.3 | [1]
PASS kept set top_p=0.6 | [1, 3]
PASS kept set top_p=0.75 | [1, 3, 5]
PASS kept set top_p=0.92 | [0, 1, 3, 4, 5]
PASS kept set min_p=0.3 (thr 0.12) | [1, 3, 5]
PASS kept set min_p=0.2 (thr 0.08) | [0, 1, 3, 5]
PASS kept set min_p=0.06 (thr 0.024) | [0, 1, 3, 4, 5, 6]
PASS kept set top_k=4 then top_p=0.8 | [1, 3, 5]
PASS temperature 0.5 == p^2 / sum(p^2)
PASS repetition penalty | [[1.0, -2.0, 0.5, 3.0]]
PASS temperature 0 -> greedy
PASS empirical dist ...top_k=4, top_p=0.8... | N=200000 max|freq-exp|=0.00106 samples_outside_kept=0   (CPU)
PASS empirical dist ...temperature=0.5, min_p=0.2... | N=200000 max|freq-exp|=0.00001 samples_outside_kept=0 (CPU)
PASS same seed -> same samples
16/16 checks passed
（CUDA 上两个经验分布的误差分别为 0.00160、0.00021，同样 16/16）
```

固定 logits 取 log([0.10, 0.40, 0.02, 0.25, 0.05, 0.15, 0.03])，保留集合可以手算。
经验分布的期望值来自闭式解：{1: 0.5, 3: 0.3125, 5: 0.1875}，以及 {1: 0.16/0.2225, 3: 0.0625/0.2225}。
对第二种情形，在 CPU 上用种子 0–3 各采样 20 万次，概率为 0.719 的那个 token 的频率分别是
0.71911、0.71884、0.71997、0.71906。可见 0.00001 只是种子 0 恰好接近，不是实现上的巧合。
测试阈值 0.005（N=20 万时约为 4.5σ）。

### lab05_paged_kv.py（分页 KV + 前缀缓存）

```bash
"$C" lab05_paged_kv.py --device cpu --test
"$G" lab05_paged_kv.py --device cuda --test
```

```
PASS paged attention == contiguous attention | seq1 block_table=[0, 1, 6, 7, 12, 13]...
PASS no leak after interleaved alloc/free
PASS requests sharing system prompt hit 2 blocks | hit tokens=[0, 8, 8] pool.hits=4
PASS shared blocks ref-counted
PASS shared-block attention == contiguous + no leak
PASS cache survives after requests finish | hit=8
PASS LRU eviction under pressure | evictions=7 hit_after=0
PASS block accounting never leaks | free=12/12
PASS LRU evicts least-recently-freed block first | X blocks alive=[True, False] Y blocks alive=[True, True]
9/9 checks passed   （CPU、CUDA 相同）
```

“精确 LRU”检查：先后缓存前缀 X 和 Y，然后新请求占用 8 个块。只有 X 的尾块被驱逐，
因为请求结束时块是逆序释放的，尾块最先回到队首；X 的首块和 Y 的两个块都还在。

### lab06_scheduler_sim.py（调度模拟；耗时模型为示意值）

```bash
"$C" lab06_scheduler_sim.py --test
```

```
300 reqs, Poisson 6.0/s, KV=4096 blocks x 16 tok  (illustrative cost model, ms)
policy           tok/s   TTFT50   TTFT99  TPOT50  TPOT99 preempt peakKV
static             416    20025    41055    30.0    31.3       0   1543
prefill_first      738       50      127    15.1    27.9       0   1247
chunked            738       69      183    14.0    25.2       0   1174
PASS <各策略>: every request gets exactly its tokens          （5 个配置）
PASS <各策略>: KV capacity never exceeded                     （5 个配置）
PASS prefill_first@tight: ... peak=300/300 preemptions=58
PASS chunked@tight: ... peak=300/300 preemptions=839
PASS chunked: per-iteration tokens <= budget | max=512
PASS tight KV forces preemption (recompute path exercised) | 897
PASS continuous batching beats static in throughput | static=416, prefill_first=738, chunked=738
13/13 checks passed
```

解读（只看趋势）：静态批跟不上 6 req/s 的到达速率，排队把 TTFT 推到了几十秒。两种连续批都跟上了
到达速率，所以吞吐相同；分块预填充把单步 token 数限制在 512，TPOT p99 更低（25.2 vs 27.9 ms），
代价是 TTFT 略高。KV 容量收紧到 300 块时，分块预填充的抢占次数（839）远多于预填充优先（58）：
它允许更多请求同时处于运行状态，容量紧张时容易反复抢占、重算。所有耗时都来自示意性的成本模型，
不是实测值。

### lab07_tensor_parallel.py（TP + all_reduce）

```bash
"$C" lab07_tensor_parallel.py --spawn 2 --backend gloo --test
"$G" lab07_tensor_parallel.py --spawn 2 --backend gloo --test --max-mb 4
```

```
backend=gloo world=2 device=cpu torch=2.13.0+cpu
weights/rank: 491520 of 983040 params (0.500)
comm per block forward: 2 all_reduce x 64x256 fp32 = 131072 bytes
PASS TP block output == single-process reference | max|diff|=1.431e-06
PASS all_reduce sum correct | 3.0
       bytes   time(us)  algbw GB/s  busbw GB/s
        4096      398.1       0.010       0.010
       16384      413.4       0.040       0.040
       65536      683.8       0.096       0.096
      262144     1271.2       0.206       0.206
     1048576    25317.7       0.041       0.041
     4194304    43286.2       0.097       0.097
    16777216   174101.1       0.096       0.096
（torch 2.5.1 环境：同样 2/2 PASS，max|diff|=1.431e-06）
```

本机 gloo 带宽只有约 0.1–0.3 GB/s（同一台机器上两个进程走 TCP 回环，`OMP_NUM_THREADS=1`），
只能用来理解“消息越大，固定延迟占比越小”，不能与 NVLink 上的 NCCL 相比。
运行时会出现 `[c10d] The client socket has failed to connect to [主机名]:29512` 警告，
这是建立连接前的重试，之后连接成功，测试也通过。

失败路径：`"$C" lab07_tensor_parallel.py --backend nccl`，在只有 CPU 的环境下打印
“nccl 需要 CUDA；在 CPU 上请用 --backend gloo”，退出码 1。

**torchrun（未能在本机运行）**：以下 4 种尝试都失败，报同一个错：

```bash
"$C" -m torch.distributed.run --nproc_per_node=2 lab07_tensor_parallel.py --backend gloo --test
USE_LIBUV=0 "$C" -m torch.distributed.run --nproc_per_node=2 lab07_tensor_parallel.py --backend gloo --test
USE_LIBUV=0 "$C" -m torch.distributed.run --nproc_per_node=2 --rdzv-backend=static \
    --master-addr=127.0.0.1 --master-port=29511 lab07_tensor_parallel.py --backend gloo --test
USE_LIBUV=0 "$G" -m torch.distributed.run --nproc_per_node=2 lab07_tensor_parallel.py --backend gloo --test
# → DistStoreError / RuntimeError: use_libuv was requested but PyTorch was built without libuv support
```

查源码发现：torch 2.13 的 `elastic/rendezvous/static_tcp_rendezvous.py` 和
`c10d_rendezvous_backend.py` 在创建 `TCPStore(...)` 时都没有传 `use_libuv`，而 C++ 端默认是 True。
直接调用 `TCPStore(..., use_libuv=False)` 可以成功；只设 `USE_LIBUV=0` 而不传参数，会报同样的错。
torch 2.5.1 同样在 `static_tcp_rendezvous.py` 处失败。所以 lab07 增加了 `--spawn N`：
用 `torch.multiprocessing` 启动进程，设置 torchrun 本来会设置的 RANK/WORLD_SIZE/MASTER_* 环境变量，
再走同样的 `init_process_group(env://)`。在 Windows 上，`env://` 默认不使用 libuv。
`torchrun --nproc_per_node=4 ... --backend nccl` 在 H800 上的运行「未在本机运行」
（本机只有 1 张卡，而且 Windows 不支持 NCCL）。

### lab08_capacity_calculator.py（容量 / 屋顶线计算器）

```bash
"$C" lab08_capacity_calculator.py --test
```

```
PASS params | got=1352 want=1352
PASS matmul params | got=1232 want=1232
PASS KV bytes/token = 2*2*1*4*2 | got=32 want=32
PASS max KV tokens = floor((0.5e9-2704)/32) | got=15624915 want=15624915
PASS ridge = 100/2 | got=50 want=50
PASS decode FLOPs | got=6208 want=6208
PASS decode bytes | got=3344 want=3344
PASS decode time = 3344/2e12 | got=1.672e-09 want=1.672e-09
PASS prefill FLOPs | got=566400 want=566400
PASS prefill bytes | got=5904 want=5904
PASS decode B=2 is memory-bound
PASS prefill 100 tok is compute-bound (AI 95.9 > 50)
PASS tp=2 with 1 KV head replicates KV (per-GPU KV not halved)
```

测试用的是内联的小配置（hidden 8、2 层、2 个 Q 头、1 个 KV 头、head_dim 4、FFN 16、vocab 10），
硬件数字也是编的（1 GB、2 TB/s、100 TFLOPS），每一项都能手算核对。脚本里没有写入任何真实 GPU
或真实模型的规格。

读真实 config.json 的路径：用 lab03 保存的随机小 Qwen3 的 config.json，配上编的硬件数字：

```bash
"$C" lab08_capacity_calculator.py out/tiny_hf_qwen3/config.json --hbm-gb 10 --hbm-tbps 1 \
    --tflops 100 --tp 2 --ctx 256 --prompt 128
```

```
params=0.001B  weights=0.00 GB (per GPU 0.00 GB, tp=2)
KV/token=0.8 KiB (per GPU 0.4 KiB)  max KV tokens=23,435,718  -> ~91545 seqs @ 256 tok
ridge point = 100.0 FLOP/byte
prefill 128 tok: AI=136.4 (compute-bound) >= 0.001 ms
```

交叉核对：lab08 算出的参数量与 HF 模型 `sum(p.numel() for p in model.parameters())` 完全一致——
绑定词嵌入的 Qwen3 都是 684096，不绑定的 Llama 都是 724864。

### lab09_profile.py（torch.profiler / trace / NVTX / CUDA events）

```bash
"$C" lab09_profile.py --device cpu --test
"$G" lab09_profile.py --device cuda --test
```

```
device=cpu torch=2.13.0+cpu dtype=float32 params=19.4M
prefill 256 tok: 28.50 ms (111.3 us/tok) | decode: 4.26 ms/tok  [perf_counter]
  aten::mm  26.35% self CPU, 493 calls ... （算子表前几行）
trace -> ...\labs\out\lab09_trace_cpu.json (9.3 MB)
PASS trace has prefill/decode_step regions
PASS top-ops table lists matmul ops | ['aten::mm', 'aten::bmm', 'aten::matmul', 'aten::linear']
PASS prefill per-token cost < decode per-token cost | 0.111 vs 4.260 ms

device=NVIDIA GeForce RTX 4070 Ti SUPER torch=2.5.1+cu121 dtype=float32 params=19.4M
prefill 256 tok: 4.27 ms (16.7 us/tok) | decode: 5.16 ms/tok  [CUDA events]
[warn] 本环境 torch.profiler 不支持 CUDA 活动（CUPTI 不可用），只记录 CPU 侧算子；...
trace -> ...\labs\out\lab09_trace_cuda.json (7.3 MB)
PASS trace has prefill/decode_step regions
PASS top-ops table lists matmul ops | ['aten::mm', 'aten::bmm', 'aten::matmul']
SKIP trace has CUDA kernel events | CUPTI 不可用，无法在本机验证
PASS prefill per-token cost < decode per-token cost | 0.017 vs 5.164 ms
```

关于 CUPTI：GPU 环境里 `torch.profiler.supported_activities()` 只返回 `{ProfilerActivity.CPU}`
（`kineto_available()` 为 True，`torch/lib` 下也有 `cupti64_2023.1.1.dll`，但没有启用 CUDA 活动）。用最小例子 profile 一次 1024×1024 matmul，
同样抓不到 GPU kernel 事件。所以本机无法验证 GPU kernel 轨迹，这一项如实标为 SKIP，没有记为 PASS。
NVTX 的 `range_push/pop` 在 CUDA 上执行时没有报错，但本机没有安装 nsys，
NVTX 区间能否在 nsys 里看到「未在本机运行」。

### lab10_multimodal_toy.py（多模态数据通路）

```bash
"$C" lab10_multimodal_toy.py --device cpu --test
"$G" lab10_multimodal_toy.py --device cuda --test
```

```
image (3, 300, 451) -> (3, 224, 320) -> patches (280, 768) -> merged (70, 768) -> visual tokens (70, 256)
text+image sequence: 82 tokens (70 image placeholders)
audio 30.0s @16kHz = 480000 samples -> log-mel (80, 3001) (center=False: 2998 frames) -> audio tokens (1501, 256); empty mel filters=0
PASS patch count = (H/p)(W/p) | 280 patches
PASS conv patch-embed == linear on patchified pixels | max|diff|=1.25e-06   (CUDA)
PASS 2x2 merge -> N/4 tokens | M=70
PASS merged token k = concat of its 2x2 spatial neighbours
PASS placeholders == visual tokens, seq len = text + M
PASS visual embeddings scattered in order
PASS text embeddings untouched
PASS STFT frames (center=True) = 1 + N//hop | 3001
PASS STFT frames (center=False) = 1 + (N-n_fft)//hop | 2998
PASS audio tokens = floor((F+2-3)/s)+1 | 1501
PASS 440 Hz sine peaks at bin round(440/(sr/n_fft)) = 11 | bin=11
11/11 checks passed
```

图像尺寸、patch、维度、归一化参数等默认值都是示意性的，不对应任何真实模型。
附带观察：`--n-mels 128` 时，HTK 公式配 n_fft=400 会出现 4 个全零（空）的 mel 滤波器
（`empty mel filters=4`）。原因是低频端滤波器比 40 Hz 的频点间隔还窄。

## 4. 调试过程中发现并修复的问题（如实记录）

1. lab01：第一次用 sed 往源码里写 `\ufffd` 转义时，GNU sed 把 `\u` 当成“下一个字母转大写”，
   结果写进去的是 `"Fffd"`。这时“streaming emits no U+FFFD”这一项虽然显示 PASS，但其实是空检查。
   改为 `REPL = chr(0xFFFD)` 后重新验证：反向检查“naive per-token decode DOES produce U+FFFD”通过，
   说明检测是有效的。
2. lab05：测试代码里有三处问题——多写了一行无用语句、交错写入时 K/V 切片长度不对、`ok` 被覆盖而不是累加，
   已修复。另外补了一个能真正检验 LRU 驱逐顺序的检查。
3. lab09：GPU 上第一次运行时，“trace has CUDA kernel events”为 FAIL（0 个 kernel）。
   排查后确认原因是本机 CUPTI 不可用，不是代码问题。改为：只有在支持 CUDA 活动时才请求它，否则打印警告并标为 SKIP。
4. lab10：GPU 上卷积与线性层的差异是 3.53e-4，导致 FAIL。原因是 `torch.backends.cudnn.allow_tf32`
   默认为 True，而 `matmul.allow_tf32` 默认为 False。关闭 cuDNN 的 TF32 后差异降到 1.25e-6，并在代码注释里说明了原因。
5. lab03：用 `--pretrained <本地目录>` 时，transformers 5.13 在目录里没有词表文件的情况下返回一个“空分词器”，
   编码结果为 `[]`，导致输入为 0 个 token 而崩溃。现在把空结果也当作“没有分词器”，改用随机 token。
6. 环境：任务中途 shell 环境被重置，`PYTHONUTF8=1` 消失，stdout/stderr 变回 GBK，中文出现乱码。
   现在每个实验在启动时都会把 stdout/stderr 重设为 UTF-8，上面的最终验证就是在没有 `PYTHONUTF8` 的情况下跑的。
7. 排版：所有行按显示宽度（中文计 2 列）检查，均 ≤ 88 列（用脚本检查，结果为 0 行超宽）。

## 5. 未在本机运行的部分及原因

| 内容 | 原因 |
|---|---|
| lab03 `--pretrained <hf_id>`（真实 hub 模型） | 需要下载权重，本任务禁止；只用本地随机小模型目录演练了这条代码路径 |
| lab07 `torchrun --nproc_per_node=4 --backend nccl` | 本机只有 1 张 GPU，且 Windows 不支持 NCCL；需要在 H800 上运行 |
| lab07 用 torchrun 启动（含 gloo） | 本机 torch 2.13 / 2.5.1 的 torchrun 在 Windows 上因 libuv 问题无法建立 rendezvous（见 lab07 一节）；Linux 上应可用，但未验证 |
| lab09 `nsys profile ...` | 本机没有安装 Nsight Systems（`which nsys` 找不到）；按要求未安装 |
| lab09 的 GPU kernel 轨迹 | 本机 CUPTI 不可用（`supported_activities()` 只有 CPU） |
| 任何 H800 上的性能数字 | 本机只做了 CPU 和 RTX 4070 Ti SUPER 上的正确性验证，计时只来自玩具模型 |

## 6. 文件清单（行数）

```
  210 lab01_bpe_toy.py
  257 lab02_tiny_llama.py
  166 lab03_hf_equivalence.py
  171 lab04_sampling.py
  262 lab05_paged_kv.py
  269 lab06_scheduler_sim.py
  187 lab07_tensor_parallel.py
  170 lab08_capacity_calculator.py
  150 lab09_profile.py
  170 lab10_multimodal_toy.py
 2012 total
```

lab02、lab05、lab06 超过了“每个约 200 行”的目标（包含自检代码）；总量约 2,000 行。
