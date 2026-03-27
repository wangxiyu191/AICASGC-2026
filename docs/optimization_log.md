# 优化记录

## 环境

- 本地开发: NVIDIA A10 24GB (sm_86), Python 3.10.9, PyTorch 2.10.0+cu128, transformers 5.2.0
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
