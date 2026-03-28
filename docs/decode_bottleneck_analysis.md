# CUDAGraph Decode 瓶颈分析

> 分析日期: 2026-03-28
> 基于: Step 24 (线上 Score 0.9576, Throughput 108.82 tok/s)
> 工具: viztracer (`log_torch=True`, CUPTI) + `analyze_traces.py`

---

## 1. 当前架构

```
Prefill (eager)                     Decode (CUDAGraph replay)
─────────────────                   ─────────────────────────
pixel_values → VisionEncoder        input_ids_buf (1,1)
            ↓                                ↓
text_tokens → LM prefill            graph.replay() ← 捕获的单步 forward
            ↓                                ↓
StaticCache 初始化                   argmax → next_token_buf
rope_deltas 提取                             ↓
            ↓                       async EOS poll (host ring buffer)
首 token argmax                              ↓
            ↓                       循环 max_new_tokens 次
CUDAGraph capture (如 bucket 未预建)
```

关键组件:
- `OptimizedDecoderLayer`: fused QKV + fused gate_up + Triton RMSNorm/RoPE (prefill) / native (decode)
- `DecodeCudaGraphRunner`: 按 `(device, bucket_max_cache_len)` 池化, 预建 32 个 bucket (64..2048)
- `GreedyDecodeEngine`: custom greedy loop, async EOS polling, device cache_position

## 2. Decode 每步耗时分解

### GPU Kernel (10.15ms/步 = ~99 tok/s 理论上限)

| 类别 | 时间 | 占比 | kernel 数 | avg/kernel | 说明 |
|------|------|------|-----------|------------|------|
| **Elementwise** | **4.29ms** | **42.2%** | **~500+** | **2-4us** | Copy/Mul/SiLU/Cast/Fill |
| GEMM | 2.28ms | 22.5% | ~60 | 38us | cublas GEMM (batch=1 仍选 tensor core tile) |
| Attention | 2.01ms | 19.8% | ~29 | 67us | SDPA→mem_efficient (fmha_cutlassF) |
| GEMV | 1.01ms | 10.0% | ~57 | 18us | cublas GEMV (QKV fused, gate_up fused) |
| Reduction | 0.36ms | 3.5% | ~114 | 3us | RMSNorm variance |
| Cat | 0.14ms | 1.4% | ~58 | 2us | |

### Host (CPU + CUDA runtime)

| 操作 | 时间/步 | 说明 |
|------|---------|------|
| CPU ops | 2.14ms | 已不是瓶颈 |
| cudaGraphLaunch | 3.23ms | host 分发 graph replay |
| cudaEventSynchronize | ~6ms | async EOS 阻塞等待 |

### 关键结论

**瓶颈已从 CPU dispatch 转移到 GPU kernel 效率。** CUDAGraph 消除了 kernel launch 开销，但 graph 内部数百个微小 elementwise kernel 各自独立访问 HBM，带宽利用率仅 ~20%。

## 3. Elementwise 瓶颈拆解

每个 decode step 内 elementwise 的前几大来源:

| kernel 类型 | 每步次数 | 每次耗时 | 来源 |
|-------------|---------|---------|------|
| Copy (aten::copy_) | ~354 | 4.4us | 残差连接、KV cache 写入、dtype 转换 |
| Mul (aten::mul) | ~484 | 3.1us | RoPE cos\*x, SiLU\*up, RMSNorm scale |
| Generic elementwise | ~516 | 1.8us | add, neg, silu, fill, cast |
| SiLU | ~29 | 3.2us | MLP activation |

**每步 ~1400 个 elementwise kernel**, 每个只做一次简单 HBM 读写。kernel 间不共享寄存器/SRAM，反复从 HBM load-compute-store 造成带宽浪费。

## 4. 带宽利用率估算

- A800 HBM 带宽: ~2039 GB/s
- FP16 权重总量: ~4 GB (2B 参数)
- 理论 decode 上限 (纯 weight load): ~500 tok/s
- 当前 GPU kernel 时间: 10.15ms/步 → 99 tok/s
- 带宽利用率: 99 / 500 ≈ **20%**
- 竞赛前三名: ~330 tok/s → ~66% 带宽利用率

差距的 3.3x 中:
- ~2x 来自 elementwise HBM roundtrip 浪费 (42% GPU 时间)
- ~1x 来自 attention/GEMM 算子效率差距

## 5. 优化方向与预期收益

### 方向 A: torch.compile + CUDAGraph (预期 throughput 1.5-2x)

将 `_run_step()` 内的模型 forward 用 `torch.compile` 编译后再 capture 进 CUDAGraph。编译器会自动融合 elementwise → 减少 kernel 数和 HBM roundtrip。

- 优点: 最省力, 编译器自动优化
- 风险: StaticCache 与 torch.compile 兼容性, CUDAGraph capture 编译后代码的兼容性
- 验证: 对比编译前后 graph 内 kernel 数

### 方向 B: Triton 手写 fused kernel (预期 throughput 1.3-1.5x)

针对 decode 阶段的高频 elementwise 模式做手写 fusion:

1. **Fused residual + RMSNorm**: `hidden = residual + attn_out; rms_norm(hidden)` → 1 kernel
2. **Fused SiLU * up**: `F.silu(gate) * up` → 1 kernel
3. **Fused add + RMSNorm + linear**: 更深度的 fusion

- 优点: 精准控制, 无编译器黑盒
- 风险: 开发量大, 需要针对 decode (M=1) 和 prefill (M>32) 分别优化

### 方向 C: FlashInfer decode attention (预期 attention 部分 1.5-2x)

替换 SDPA mem_efficient → FlashInfer 的 `BatchDecodeWithPagedKVCacheWrapper` 或 single decode kernel。

- 预期: attention 部分 2.01ms → ~1ms, 整体提升 ~10%
- 风险: 需要集成 FlashInfer, 处理 KV cache 格式差异

### 方向 D: Vision encoder 优化 (TTFT 60ms → 40ms)

当前 TTFT 60ms, Top 1 为 42ms。prefill GPU kernel 50.6ms 中 vision encoder + GEMM + attention 占主要部分。

- `torch.compile` vision encoder (需绕过 flash_attn_varlen 不兼容)
- ViT CUDAGraph 捕获

### 优先级建议

```
[1] 方向 A: torch.compile + CUDAGraph ← 收益最大/成本最低
[2] 方向 B: Triton fused kernels      ← 如 A 不可行的备选
[3] 方向 C: FlashInfer decode attn     ← 独立增量优化
[4] 方向 D: Vision encoder 优化        ← TTFT 专项
```
