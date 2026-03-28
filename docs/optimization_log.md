# 优化记录

## 环境

- 本地开发 (旧): NVIDIA A10 24GB (sm_86), Python 3.10.9, PyTorch 2.10.0+cu128, transformers 5.2.0
- 本地开发 (当前): NVIDIA A800 80GB PCIe (sm_80), Python 3.12.3, PyTorch 2.8.0+cu128, transformers 5.4.0, triton 3.4.0
- 评测环境: NVIDIA A800 80GB (sm_80), Python 3.12.3, PyTorch 2.8.0+cu128, flash_attn 2.8.3 预装

## Baseline

| 指标 | 值 |
|------|-----|
| TTFT | 133.13 ms |
| Throughput | 37.22 tok/s |
| 样本数 | 30 |
| 配置 | FP16, device_map="cuda:0", 无 attn_implementation, use_cache=True |

## Phase 1: 配置级优化

### 1a: flash_attention_2 + CUDA config + device_map

| 指标 | 值 | vs Baseline |
|------|-----|-------------|
| TTFT | 128.20 ms | -3.7% |
| Throughput | 29.79 tok/s | **-20.0%** |
| 样本数 | 10 | |

改动:
- `attn_implementation="flash_attention_2"`
- TF32 enabled, flash SDP enabled, mem_efficient SDP enabled
- `torch.set_float32_matmul_precision('high')`

发现: flash_attn_varlen 在 A10 上 throughput 严重退化。vision encoder 用 `flash_attn_varlen_func` 处理变长序列，A10 上效率不如 sdpa。评测环境 A800 可能不同，代码保留条件检测。

### 1b: flash_attention_2 + CUDA config + .to(device)

| 指标 | 值 | vs Baseline |
|------|-----|-------------|
| TTFT | 125.99 ms | -5.4% |
| Throughput | 29.21 tok/s | **-21.5%** |
| 样本数 | 10 | |

改动: 在 1a 基础上用 `.to(device)` 替代 `device_map=device`，意图避免 accelerate dispatch hooks。

发现: throughput 无改善，说明 device_map 的 accelerate hooks 开销在单卡场景下很小。`.to(device)` 与 `device_map` 性能几乎无差异。

### 1c: sdpa + CUDA config + device_map (当前最优)

| 指标 | 值 | vs Baseline |
|------|-----|-------------|
| TTFT | 121.35 ms | **-8.8%** |
| Throughput | 37.57 tok/s | +0.9% |
| 样本数 | 10 | |

改动: 在 1a 基础上将 `attn_implementation` 改为 `"sdpa"`。

发现: sdpa 在 A10 上是最优 attention backend。TTFT 改善主要来自 CUDA config（TF32 等）。

## P0: 强制 SDPA (修复 A800 吞吐量退化)

A800 官方评测发现 flash_attention_2 导致 throughput 仅 33.07 tok/s（基线 ~55），退化 ~40%。
根因: flash_attn 在 decode 阶段 CPU dispatch 开销 8.2ms/step >> GPU 计算 0.56ms/step。

### 修复

强制使用 `attn_impl = "sdpa"`，移除 flash_attn 条件判断。SDPA 让 PyTorch 自动选择最优后端。

| 指标 | 值 | vs Step 2 (A800 flash_attn) |
|------|-----|-------------|
| TTFT | 62.42 ms | -8.6% |
| Throughput | 41.93 tok/s | **+26.8%** |
| Accuracy | 0.6933 | -1.6% |
| Score | 0.4470 | |
| 样本数 | 5000 | A800 |

## Phase 4: OptimizedDecoderLayer (融合线性投影)

自定义 `OptimizedDecoderLayer` 替换 HF 的 `Qwen3VLTextDecoderLayer`，融合线性投影减少 kernel launch。

### 优化内容

1. **Fused QKV**: q_proj/k_proj/v_proj 权重拼接为 [4096, 2048]，3 次 GEMM → 1 次
2. **Fused gate_up**: gate_proj/up_proj 权重拼接为 [12288, 2048]，2 次 GEMM → 1 次
3. RMSNorm/SiLU 保持 native PyTorch（Triton 在 decode 小 tensor 下更慢）

每 decode step 减少 84 次 kernel launch (28 层 × 3 次)。

### A10 结果

| 指标 | 值 | vs P0 baseline (sdpa) |
|------|-----|-------------|
| TTFT | 126.35 ms | +1.9% |
| Throughput | 39.58 tok/s | **+13.0%** |
| 样本数 | 10 | A10 |

## Triton Fused RMSNorm + RoPE

在 OptimizedDecoderLayer 基础上，用 Triton 自定义 kernel 替换 PyTorch native 实现。

### 优化内容

1. **Triton RMSNorm**: 8 PyTorch ops → 1 Triton kernel per norm call
2. **Triton RoPE**: 5 kernels (cat/neg/mul/mul/add) → 1 Triton kernel per call

### A10 结果

| Step | 优化 | TTFT | Throughput | vs Step 8 |
|------|------|------|------------|-----------|
| 10 | + Triton RMSNorm | 115.87ms | 48.11 tok/s | +21.5% |
| 11 | + Triton RoPE | 117.89ms | 50.83 tok/s | +28.4% |

## Step 12: Decode 阶段 Triton → Native Ops 分支

### 问题发现

通过 viztracer (max_stack_depth=40) 深度分析 decode 阶段，发现 **Triton Python dispatch 是最大瓶颈**：

| 指标 | 数值 |
|------|------|
| 每步 wall time | 25.7ms (= 39 tok/s) |
| 每步 GPU kernel 时间 | 4.3ms (= 231 tok/s 理论上限) |
| GPU 利用率 | 仅 17% |
| Triton Python dispatch / step | 11.2ms (占 44%) |

Triton kernel 在 decode 阶段 tensor 极小 (batch=1, seq=1) 时：
- `_rms_norm`: Python dispatch 69us/call vs GPU kernel 2.3us/call (**30x**)
- `_rope_fwd`: Python dispatch 62us/call vs GPU kernel 1.7us/call (**36x**)

每步 168 次 Triton launch (28层 × 6次)，Python 侧开销 11.2ms >> GPU 计算。

### 优化方案

在 `_rms_norm` 和 `_apply_rotary_emb` 中加入 tensor 大小分支 (`M <= 32`)：
- decode (小 tensor): 使用 `F.rms_norm` / PyTorch native RoPE (dispatch ~5-10us)
- prefill (大 tensor): 继续使用 Triton kernel (amortize dispatch over large data)

### A10 结果

| 指标 | Step 11 (all Triton) | Step 12 (native decode) | 变化 |
|------|---------------------|------------------------|------|
| TTFT | 117.89ms | 111.81ms | **-5.2%** |
| Throughput | 50.83 tok/s | 54.90 tok/s | **+8.0%** |
| 样本数 | 10 | 10 | A10 |

### 启示

Decode 阶段 CPU 开销远超 GPU 计算，native ops 分支只解决了 Triton dispatch 部分。
剩余瓶颈 (HF generate Python loop、item() sync、cudaLaunchKernel) 需要 CUDA Graph + 自定义 decode loop 根本性解决。

## Phase 4: Custom Decode + StaticCache + CUDA Graph

### 实现范围

本阶段的真实 benchmark 入口是 `model.model.generate(...)`，因此改动全部落在 `evaluation_wrapper.py` 的 `ModelWrapper.generate()`。

当前实现分两层：

1. **Prefill 保持 eager**
   - 完整输入继续走普通 `self._model(...)`
   - 首轮 forward 传入 `StaticCache`
   - 提取首 token logits、`rope_deltas`、prefill 后的 cache 状态

2. **Decode 改为自定义 greedy loop**
   - 默认先走 eager custom decode
   - 在 CUDA 条件满足时，对单 token decode step 做 `torch.cuda.CUDAGraph` capture/replay
   - trace 模式 (`ENABLE_VIZTRACE=1`) 自动关闭 cudagraph，只保留 eager custom decode

### 关键实现点

- 新增 `GreedyDecodeEngine`
  - 只在 `batch_size == 1`
  - `do_sample == False`
  - `use_cache == True`
  - `max_new_tokens <= 128`
  - 且无额外复杂 generation kwargs 时启用
- 新增 `DecodeCudaGraphRunner`
  - 固定 placeholder:
    - `input_ids_buf: (1, 1)`
    - `cache_position_buf: (1,)`
    - `position_ids_buf: (3, 1, 1)`
    - `attention_mask_buf: (1, max_cache_len)`
  - 只 capture 单步 decode
  - capture 前做 warmup
  - 通过 snapshot/restore `StaticCache`，避免 warmup/capture 破坏 prefill 后的真实起始状态
- decode 热路径不再调用 `cache.get_seq_length()`
- `position_ids` 显式按 `current_position + rope_deltas` 构造

### A10 冒烟与对照结果

测试环境：
- GPU: NVIDIA A10 24GB
- PyTorch: 2.10.0+cu128
- transformers: 5.2.0
- 命令均带 `TQDM_DISABLE=1`

#### 1 样本 benchmark

| 路径 | TTFT | Throughput | 结果文件 |
|------|------|------------|----------|
| HF generate | 107.00 ms | 56.82 tok/s | `results/result_hf_generate_smoke.json` |
| Custom eager + StaticCache | 109.37 ms | 51.83 tok/s | `results/result_custom_decode_smoke.json` |
| Custom + StaticCache + CUDAGraph | 108.54 ms | 83.96 tok/s | `results/result_cudagraph_smoke.json` |

结论：
- graph 路径成功进入并稳定运行
- graph 吞吐明显高于 eager custom decode 与 HF generate
- 三条路径生成文本完全一致

#### 3 样本 benchmark

| 路径 | TTFT | Throughput | 结果文件 |
|------|------|------------|----------|
| HF generate | 125.15 ms | 45.54 tok/s | `results/result_hf_generate_3.json` |
| Custom eager + StaticCache | 129.32 ms | 50.81 tok/s | `results/result_custom_decode_3.json` |
| Custom + StaticCache + CUDAGraph | 127.83 ms | 80.76 tok/s | `results/result_cudagraph_3.json` |

对照结论：
- `HF == eager == graph`，3/3 样本文本完全一致
- eager custom decode 已经略高于 HF throughput (`50.81 vs 45.54 tok/s`)
- 加入 cudagraph 后 throughput 进一步提升到 `80.76 tok/s`
- 相对 HF 3 样本基线，throughput 提升约 `77.3%`
- TTFT 小幅波动（约 +2.7ms），当前主要收益集中在 decode throughput

#### 10 样本 benchmark

| 路径 | TTFT | Throughput | 结果文件 |
|------|------|------------|----------|
| HF generate | 112.80 ms | 54.19 tok/s | `results/result_hf_generate_10.json` |
| Custom eager + StaticCache | 114.51 ms | 52.78 tok/s | `results/result_custom_decode_10.json` |
| Custom + StaticCache + CUDAGraph | 113.65 ms | 82.00 tok/s | `results/result_cudagraph_10.json` |

对照结论：
- `HF == eager == graph`，10/10 样本文本完全一致
- custom eager 在 10 样本下未显著优于 HF，说明“只替换 loop + StaticCache”本身收益有限，capture 成本和 host 逻辑仍有影响
- cudagraph 路径保持稳定收益，throughput 从 `54.19` 提升到 `82.00 tok/s`
- 相对 HF 10 样本基线，throughput 提升约 `51.3%`
- TTFT 与 HF 基本持平（`113.65ms vs 112.80ms`），说明收益主要集中在 decode 阶段，符合设计预期

### 当前结论

- `StaticCache + custom decode + single-token CUDAGraph` 已在真实 benchmark 路径上跑通
- 现阶段已经满足“可以继续扩样本和做 trace/profile 验证”的条件
- 下一步重点应放在：
  - 更大样本数（10/更高）验证收益是否稳定
  - graph 前后 trace 对比，确认 `cat/concat` 与 Python loop 开销下降
  - 评估是否需要做 graph bucket 复用，降低每请求 capture 成本

## Step 16: 修复 transformers 5.4.0 兼容性 + A800 CUDAGraph 验证

### 环境变化

本地开发环境升级为 A800 80GB PCIe + Python 3.12.3 + PyTorch 2.8.0+cu128 + transformers 5.4.0。

### 发现并修复的 3 个兼容性问题

1. **`mm_token_type_ids` 缺失**: 5.4.0 `apply_chat_template` 新增 `mm_token_type_ids` 返回值，不在 `SUPPORTED_KWARGS` 中，导致 CUDAGraph 静默 fallback。
2. **EOS token 解析失败**: `Qwen3VLConfig` 不再有 `eos_token_id` 属性，拿到 None，EOS 永不检测。
3. **StaticCache `cumulative_length` 未恢复**: 5.4.0 `StaticLayer` 用内部 `cumulative_length` 替代外部 `cache_position`，warmup 推进越界。

### Step 9 分数暴跌根因

Step 9 (score 0.0820) 的根因是 EOS 检测失败——custom decode 不停止，throughput 测试结果异常。

### A800 结果

| 指标 | Step 7 A800 | Step 16 A800 | 变化 |
|------|-------------|--------------|------|
| TTFT | 62.42ms | 62.27ms | -0.2% |
| Throughput | 41.93 tok/s | **75.53 tok/s** | **+80.1%** |

## 踩坑记录

### torch.compile + flash_attn_varlen 不兼容

尝试 `torch.compile(visual, mode="default", dynamic=True)` 编译 vision encoder，warmup 时报错:
```
TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
flash_attn._flash_attn_varlen_forward() Expected a value of type 'int'
for argument 'max_seqlen_q' but instead found type 'FakeTensor'
```
原因: flash_attn_varlen_func 的 max_seqlen_q/k 参数是 tensor，但 C++ 侧声明为 SymInt，Dynamo FakeTensor 无法 cast 为 C++ int。这是 flash_attn 与 torch.compile 的已知不兼容。

结论: **不能对使用 flash_attention_2 的 vision encoder 做 torch.compile**。

### torch.compile mode="reduce-overhead" CUDA Graph 冲突

尝试 `mode="reduce-overhead"` 编译 vision encoder 和 language model，报错:
```
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been
overwritten by a subsequent run.
```
原因: reduce-overhead 模式内部使用 CUDA Graphs。DeepStack 机制从 vision encoder 抽取中间层特征传给 language model 的前几层，两个独立编译的模块共享 tensor，CUDA Graph replay 时覆盖了这些 tensor。

结论: **DeepStack 跨模块 tensor 共享与 CUDA Graphs 不兼容**。如需 torch.compile 须用 mode="default"，且不能 compile vision encoder（flash_attn 限制）。

### transformers 5.x tqdm 在非 TTY 环境阻塞

通过 Bash tool 执行 benchmark 时，进程卡在模型加载（`convert_and_load_state_dict_in_model`）。CPU idle 99%，磁盘 IO 为 0。

原因: transformers 5.x 的权重加载用 tqdm 进度条，每个参数都调 `pbar.refresh()`。在非 TTY 环境（stdin 被重定向）下，tqdm 的 terminal 检测导致阻塞。

解决: `TQDM_DISABLE=1` 环境变量。
