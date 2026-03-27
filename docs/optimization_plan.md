# Qwen3-VL-2B-Instruct 推理优化计划

> 目标: 在不修改模型参数的前提下，优化 TTFT 和 Throughput
> 评分: 0.4 * Accuracy + 0.3 * TTFT_Improvement + 0.3 * Throughput_Improvement
> 硬件: NVIDIA A800 80GB PCIe (sm_80), CUDA 12.8, PyTorch 2.8.0+cu128
> 约束: 禁止量化/稀疏化/修改模型参数，仅修改 evaluation_wrapper.py + 自定义模块

---

## 当前状态

**最新 A800 成绩 (P0 sdpa):** Score 0.4470, Accuracy 0.6933, TTFT 62.42ms, Throughput 41.93 tok/s

**已完成优化:**
- [x] P0: 强制 SDPA (修复 flash_attn_2 吞吐量退化 ~40%)
- [x] Phase 4: OptimizedDecoderLayer - fused QKV + fused gate_up (A10 throughput +13%)
- [x] CUDA config: TF32, flash/mem_efficient SDP

**待 A800 验证:** fused QKV + gate_up (commit `[historical commit]`)

---

## 环境信息

| 项目 | 开发环境 | 评测环境 |
|------|---------|---------|
| Python | 3.10.9 | 3.12.3 |
| PyTorch | 2.10.0+cu128 | 2.8.0+cu128 |
| CUDA | 12.8 | 12.8 |
| GPU | A10 (sm_86) | A800 (sm_80) |
| Triton | 3.6.0 | TBD |
| transformers | 5.2.0 | TBD |

---

## 基线数据

| 指标 | 官方基线 | 本地实测 (A10) |
|------|---------|---------------|
| TTFT | ~80ms | TBD |
| Throughput | ~55 tok/s | TBD |
| Accuracy | TBD | TBD |

---

## 优化路线图

### Phase 1: 配置级优化 (低风险, 高收益)

#### 1.1 torch.compile 模型编译 — ❌ 失败
- 结果: 动态 KV Cache 导致 recompile，throughput -35%
- 详见 optimization_attempts.md Phase 2 章节

#### 1.2 Flash SDPA 确认与优化 — ✅ 已完成
- 结果: SDPA 是最优 attention backend，flash_attention_2 在 decode 阶段反而有害
- 强制 `attn_implementation="sdpa"`

#### 1.3 StaticCache KV 缓存预分配 — ❌ 失败
- 结果: 与动态 VLM 输入不兼容，cache_position 处理复杂
- 详见 optimization_attempts.md StaticCache 章节

#### 1.4 Attention mask 优化 — 待尝试
- 方案: 使用 `is_causal=True` 替代显式 attention mask
- 原理: 减少 mask 张量的内存分配和传输

### Phase 2: Vision Encoder 专项优化 (针对 TTFT)

#### 2.1 安装 flash-attn 库
- 方案: `pip install flash-attn` (需 sm_80+ 支持)
- 原理: 专用 Flash Attention 实现比 PyTorch SDPA 更快
- 特别是 `flash_attn_varlen_func` 对 ViT 可变长度序列更优

#### 2.2 Vision Encoder 单独编译
- 方案: `model.visual = torch.compile(model.visual, mode="max-autotune")`
- 原理: ViT 输入 shape 相对稳定, 编译效果好, 避免 LLM 动态 shape 干扰

#### 2.3 Vision Attention monkey-patch
- 方案: 替换 ViT attention 为 flash_attn 实现
- 参考: SGLang `VisionFlash3Attention`, vLLM `vit_flash_attn_wrapper`
- 目标组件: `model.visual.blocks[*].attn`

#### 2.4 ViT CUDA Graph 捕获
- 方案: 按 seqlen 分桶捕获 Vision Encoder 执行图
- 参考: SGLang `ViTCudaGraphRunner`
- 原理: 消除 27 层 ViT 的 kernel launch 开销

### Phase 3: Triton 自定义 Kernel (算子级优化)

#### 3.1 Triton Fused RMSNorm — ❌ 无效
- 结果: decode 小 tensor (batch=1, seq_len=1) 下 Triton launch 开销 159us > native 59us
- Triton kernel launch 固定 ~100us，112 次/step = +11.2ms 额外开销

#### 3.2 Triton Fused Rotary Embedding (RoPE) — 待尝试
- 方案: 融合 RoPE 计算到 attention kernel
- 原理: 减少单独的 RoPE kernel 调用

#### 3.3 Triton Fused SwiGLU FFN — ❌ 无效
- 结果: swiglu 14.1us vs native silu*up 12.0us，收益可忽略

#### 3.4 Fused QKV Projection — ✅ 已完成
- 实现: 合并 Q/K/V 权重为 [4096, 2048]，使用 F.linear 1 次 GEMM
- 结果: 配合 fused gate_up，A10 throughput +13%

#### 3.5 Fused gate_up Projection — ✅ 已完成
- 实现: 合并 gate_proj/up_proj 权重为 [12288, 2048]，使用 F.linear 1 次 GEMM

### Phase 4: 系统级优化 (精调)

#### 4.1 内存预分配与池化
- 方案: 预分配所有中间张量缓冲区
- 原理: 避免运行时 cudaMalloc/cudaFree

#### 4.2 CUDA Stream 流水线
- 方案: Vision Encoder 和 embedding 处理使用独立 stream
- 原理: 计算与数据传输重叠

#### 4.3 生成配置调优
- 方案: 调整 generate() 参数 (use_cache, pad_token_id 等)
- 原理: 避免不必要的计算和内存操作

---

## 实施优先级矩阵（更新后）

```
已完成:
  ✅ Flash SDPA (P0)
  ✅ Fused QKV + gate_up (OptimizedDecoderLayer)
  ✅ CUDA config (TF32, SDP)

已验证无效:
  ❌ torch.compile (动态 shape recompile)
  ❌ StaticCache (VLM 动态输入不兼容)
  ❌ Triton RMSNorm/SwiGLU (decode 小 tensor launch 开销)
  ❌ flash_attention_2 (decode CPU dispatch 开销)
  ❌ 自定义 decode loop (模型内部 sync 瓶颈)

下一步优先级:
  [高] CUDA Graph 捕获 decode 阶段
  [高] 消除 item()/is_nonzero 同步点 (TTFT)
  [中] Vision Encoder 优化 (TTFT)
  [低] dtype 转换优化
```

---

## 风险与注意事项

1. **A10 vs A800**: 开发在 A10 (sm_86), 评测在 A800 (sm_80), 预编译 .so 需针对 sm_80
2. **torch.compile recompile**: VLM 动态 shape 可能导致频繁重编译, 需设 `dynamic=True` 或 padding
3. **transformers 版本**: 本地 5.2.0, 需确认评测端版本兼容性
4. **accuracy 至上**: 权重 0.4, 任何优化后必须立即验证 accuracy 不下降
5. **提交限制**: 评测不自动编译 .cu, 需提交预编译 .so 或纯 Triton/Python
