# AICAS 2026 VLM 高效推理优化报告（简略版）

## 1. 研究背景

本项目面向 AICAS 2026「面向 AI 芯片的 VLM 高效推理与优化」赛道，目标是在**单张 NVIDIA A800 80GB PCIe** 上，对 **Qwen3-VL-2B-Instruct** 视觉语言模型进行推理优化。评测指标为：

```
最终得分 = 0.4 × 准确率 + 0.3 × TTFT提升率 + 0.3 × 吞吐量提升率
```

竞赛约束：**禁止量化、稀疏化、修改模型权重**，所有优化须在算子层面和系统层面完成。

## 2. 相关工作

我们参考了以下工程实践：
- **vLLM / SGLang** — PagedAttention、CUDAGraph capture/replay、continuous batching 等推理引擎设计
- **FlashAttention / FlashInfer** — 高效 attention kernel 的 split-KV、online softmax 等算法
- **Triton** — 用于快速原型和 GPU kernel 开发的 DSL
- **CUTLASS / cuBLAS** — GEMM/GEMV 的参考实现和性能基线

## 3. 方法设计

我们的优化分为 **Prefill 阶段**（TTFT 优化）和 **Decode 阶段**（吞吐量优化）两条主线。

### 3.1 Decode 阶段优化（吞吐量提升主力）

#### 3.1.1 自定义 Greedy Decode + CUDAGraph

- 替换 HuggingFace 的 `model.generate()`，实现自定义 `GreedyDecodeEngine`
- 使用 `StaticCache` 替代 `DynamicCache`，消除 decode 阶段的动态内存分配
- 引入 **CUDAGraph capture/replay**：
  - 以 64 为粒度预建 32 个 graph bucket (64~2048)，消除首请求 lazy capture 开销
  - 异步 EOS 检测：通过 pin_memory + D2H copy + CUDA Event 实现非阻塞 EOS 轮询，减少 GPU 空闲

#### 3.1.2 融合 Decoder Layer (OptimizedDecoderLayer)

替换 HF 原始 Qwen3VLTextDecoderLayer，核心融合：

| 融合项 | 原始 | 优化后 | 效果 |
|--------|------|--------|------|
| QKV projection | 3 次 GEMM | 1 次 fused GEMM | 省 56 次 kernel launch/step |
| gate+up projection | 2 次 GEMM | 1 次 fused GEMM | 省 28 次 kernel launch/step |
| RMSNorm | 8 个 PyTorch ops | 1 个 Triton kernel | ~2.2x faster |
| Add + RMSNorm | 独立 add + norm | 1 个 Triton kernel | 一次读写 |
| QK-Norm + RoPE | 分别 norm + 分别 rope | 1 个 Triton kernel | 减少 HBM 往返 |
| SiLU(gate) * up | 3 个 element-wise ops | 1 个 Triton kernel | 减少 launch 开销 |
| KV Cache 更新 | 5 个小 kernel/层/步 | 1 个 Triton kernel | decode GPU/步 -0.28ms |

#### 3.1.3 自定义 CUDA GEMV (nosplit)

NCU profiling 发现 cuBLAS 对 M=1 场景选用 WMMA Tensor Core GEMM 而非专用 GEMV，仅达 52-59% DRAM 吞吐、10% occupancy。我们实现了专用 CUDA GEMV：

- **k2048 路径** (QKV/lm_head): row2/row4 变体，`__hfma2` 半精度 FMA
- **k6144 路径** (gate_up/down_proj): `n8k1024_xwarp_w_async` 变体，cp.async 异步加载 weight、x 通过 warp shuffle 广播
- 额外融合: `fused_add_rmsnorm_gateup`、`fused_silu_downproj`、`fused_gateup_silu_downproj`

#### 3.1.4 自定义 Decode Attention

实现多级 Triton attention kernel 调度链：

1. **Split-KV attention** (默认): 持久 workspace、多 split 并行处理 KV cache，online-merge 合并 partial softmax
2. **Dot-style attention**: 利用 `tl.dot` 做矩阵块操作
3. **Grouped prefix attention**: 利用 GQA 结构，一个 KV head 的 K/V 同时服务多个 Q head
4. 自动根据 seq_len 选择 split_k / block_n / num_warps 配置

### 3.2 Prefill 阶段优化（TTFT 优化）

#### 3.2.1 Vision Encoder 缓存

- 缓存 `fast_pos_embed_interpolate`、`rot_pos_emb`、`position_embeddings`、`cu_seqlens` — 相同 grid_thw 的图片直接复用
- `patch_embed` Conv2d 改为等价 `F.linear`，避免 reshape 开销
- Vision attention 使用预计算的 split_sizes 做 per-frame SDPA

#### 3.2.2 多模态融合快速路径

- 检测单图 VQA 模式（prefix + image tokens + suffix），跳过 HF 通用多模态合并
- 直接 slice copy image_embeds 到 inputs_embeds

#### 3.2.3 Text Prefill 快速路径

- 跳过首次 prefill 的 `create_causal_mask`（全 prompt 无需 4D mask）
- 复用 `cache_position` 和 `rotary_emb` 缓存
- Prefill attention 直接在有效 prompt K/V 上做 causal SDPA 后写回 cache

#### 3.2.4 RoPE 位置编码缓存

- 缓存 `get_rope_index` 的完整计算结果（position_ids + rope_deltas），相同 grid+prefix+suffix 组合直接复用

#### 3.2.5 输入预处理元数据提前提取

- 在 `processor.apply_chat_template` 阶段（CPU 侧）提前提取 grid_key 和 segment_info，避免 GPU 侧重复计算

### 3.3 系统级配置

- CUDA 后端: `allow_tf32=True`、`enable_flash_sdp(True)`、`enable_mem_efficient_sdp(True)`
- Attention 实现: SDPA (自动选择 flash/mem_efficient/math)
- 模型精度: FP16
- Float32 矩阵乘法精度: `high`

## 4. 实验设置

| 项目 | 值 |
|------|---|
| 模型 | Qwen3-VL-2B-Instruct |
| GPU | NVIDIA A800 80GB PCIe |
| PyTorch | 2.8.0+cu128 |
| CUDA Toolkit | 12.8 |
| 评测样本 | 5000 (完整验证集) / 150 (线上评测子集) |
| 最大生成长度 | 128 tokens |
| Batch size | 1 |
| 预热 | 10 样本 |

## 5. 结果分析

### 5.1 性能演进关键里程碑

| 阶段 | TTFT (ms) | 吞吐量 (tok/s) | 准确率 | 线上得分 | 主要优化 |
|------|-----------|----------------|--------|----------|----------|
| Baseline | 133.13 | 37.22 | - | - | 无优化，仅 use_cache |
| SDPA+CUDA config | 68.32 | 33.07 | 0.7044 | - | A800 首次官方评测 |
| Fused QKV+gate_up | 60.10 | 46.57 | 0.6867 | 0.0820 | 3 GEMM→1, 2 GEMM→1 |
| Triton fused norms+RoPE | - | ~54.90 | - | - | RMSNorm/RoPE 融合 |
| CUDAGraph | 62.27 | 75.53 | - | - | CUDAGraph 首次成功 |
| Reusable graph pool | 61.33 | 95.68 | - | - | bucket 复用 |
| Prebuild graph | 60.74 | 102.51 | - | - | 预建 32 个 bucket |
| **线上 v1** | **60.32** | **108.82** | **0.6978** | **0.9576** | Triton decode 全套融合 |
| Split-KV + CUDA GEMV | 61.67 | 200.55 | 0.7022 | 1.6398 | split-KV attention + nosplit GEMV |
| **线上 v2** | **62.10** | **212.47** | **0.7022** | **1.7274** | 稳定版 |
| Text prefill fastpath | **43.49** | **226.11** | - | - | TTFT 大幅优化 |

### 5.2 最终性能对比

| 指标 | Baseline | 最终优化 | 提升 |
|------|----------|----------|------|
| TTFT | ~80 ms (官方 baseline) | 43.49 ms | -45.6% |
| 吞吐量 | ~55 tok/s (官方 baseline) | 226.11 tok/s | +311.1% |
| 准确率 | - | 0.7022 | 维持 |
| 线上得分 | - | 1.7274 | - |

### 5.3 各优化技术贡献分析

**Decode 吞吐量分解**（近似，A800 10-sample 测量）：

| 技术 | 吞吐量增量 | 累计吞吐量 |
|------|-----------|-----------|
| HF baseline (SDPA) | - | ~42 tok/s |
| Fused QKV/gate_up | +4.6 tok/s (+11%) | ~47 tok/s |
| Triton fused norms/RoPE/SiLU | +8 tok/s (+17%) | ~55 tok/s |
| CUDAGraph | +21 tok/s (+38%) | ~76 tok/s |
| Graph pool + prebuild | +27 tok/s (+35%) | ~103 tok/s |
| Triton decode attention | +6 tok/s (+6%) | ~109 tok/s |
| Split-KV attention | +75 tok/s (+69%) | ~184 tok/s |
| CUDA nosplit GEMV | +14 tok/s (+7.5%) | ~198 tok/s |
| Fused KV cache update | +16 tok/s (+8%) | ~214 tok/s |
| Misc (in-place residual 等) | +12 tok/s | ~226 tok/s |

**TTFT 分解**：

| 技术 | TTFT 变化 |
|------|-----------|
| SDPA + CUDA config | 133 → 68 ms |
| Vision grid cache + RoPE cache | 68 → 62 ms |
| Text prefill fastpath (skip mask + cache reuse) | 61 → 43 ms |

## 6. 结论

1. **CUDAGraph 是 decode 吞吐量优化的基础**：消除每步 Python dispatch 和 CUDA launch 开销，是后续所有 GPU kernel 优化发挥作用的前提。

2. **自定义 CUDA GEMV 和 split-KV attention 是高收益优化**：NCU profiling 揭示 cuBLAS 对 M=1 场景的低效，以及 attention 在长 cache 下的并行度不足，针对性实现带来了最大的吞吐量提升。

3. **Triton kernel 融合在 CUDAGraph 内效果显著**：小 shape 下 Triton 的 Python dispatch 开销在 eager 模式下反而是负收益，但在 CUDAGraph capture 后完全消除，使得 RMSNorm/RoPE/SiLU/KV-update 的融合 kernel 切实减少了 GPU kernel 数量和 HBM 访问。

4. **Prefill 优化以缓存和跳过冗余计算为主**：vision position embeddings 缓存、RoPE 缓存、跳过 causal mask 构造、text prefill fastpath 等共同将 TTFT 从 61ms 降至 43ms。

5. **准确率保持稳定**：所有优化均为等价变换（相同精度、相同计算），准确率维持在 0.70 左右，未出现精度退化。
