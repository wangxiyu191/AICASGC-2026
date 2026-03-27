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
- [x] Triton fused RMSNorm + RoPE (A10 throughput +28.4% vs step 8)
- [x] Step 12: decode 阶段 Triton→native ops 分支 (A10 throughput +8.0% vs step 11)

**待 A800 验证:** Step 12 (Triton RMSNorm/RoPE + native decode 分支)

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

#### 3.1 Triton Fused RMSNorm — ✅ 有效 (prefill), 需 native 分支 (decode)
- 结果: prefill 大 tensor 有效 (+21.5%), decode 小 tensor Triton dispatch 65us >> GPU 2us
- 解决: M<=32 时 fallback 到 F.rms_norm

#### 3.2 Triton Fused Rotary Embedding (RoPE) — ✅ 有效 (prefill), 需 native 分支 (decode)
- 结果: prefill 有效 (+5.7%), decode 同样有 dispatch 开销问题
- 解决: seq_len<=32 时 fallback 到 PyTorch native RoPE

#### 3.3 Triton Fused SwiGLU FFN — ❌ 无效
- 结果: swiglu 14.1us vs native silu*up 12.0us，收益可忽略

#### 3.4 Fused QKV Projection — ✅ 已完成
- 实现: 合并 Q/K/V 权重为 [4096, 2048]，使用 F.linear 1 次 GEMM
- 结果: 配合 fused gate_up，A10 throughput +13%

#### 3.5 Fused gate_up Projection — ✅ 已完成
- 实现: 合并 gate_proj/up_proj 权重为 [12288, 2048]，使用 F.linear 1 次 GEMM

### Phase 4: CUDA Graph + 自定义 Decode Loop (系统级优化)

> **这是当前最大的优化杠杆。** Trace 分析显示 decode 阶段 GPU 利用率仅 17%（GPU 4.3ms/step vs wall 25.7ms/step），
> CPU dispatch + Python overhead + sync 是绝对瓶颈。CUDA Graph 可一次性消除所有 kernel launch 和 Python dispatch 开销。

#### 4.1 自定义 Greedy Decode Loop — 高优先级
- **目标**: 跳过 HF `generate()` 的 Python 开销（500ms+ Other CPU ops）和同步点
- **方案**: 自定义 greedy decode，直接调用 model forward + argmax
- **消除**: `aten::item` (63ms sync) + `aten::is_nonzero` (31ms sync) + HF logits processing
- **关键**: 不用 stopping criteria（固定生成 max_new_tokens），后处理时截断 EOS
- **风险**: 需要正确处理 KV cache、attention mask、cache_position 的更新

#### 4.2 StaticCache 预分配 — 4.1 的前置依赖
- **目标**: 固定 KV cache tensor 地址，满足 CUDA Graph 的 static memory 要求
- **方案**: 预分配 `max_seq_len` 大小的 cache，用 index_copy 替代 cat
- **注意**: 之前 StaticCache 尝试失败是因为 HF generate 兼容性问题，自定义 loop 可以绕过
- **消除**: Cat/Concat 操作 (12.8ms GPU, 4056 calls)

#### 4.3 CUDA Graph 捕获 Decode Step — 核心目标
- **方案**: 捕获完整的单步 decode (28层 forward + argmax) 为 CUDA Graph，replay N 次
- **前提**: 4.1 (自定义 loop) + 4.2 (StaticCache) 都完成后才能 capture
- **预期收益**: 从 ~25.7ms/step → ~4.3ms/step (纯 GPU kernel 时间)
  - 理论 throughput: 从 ~55 tok/s → ~200+ tok/s
  - 消除: 25K cudaLaunchKernel (125ms) + 所有 Python dispatch
- **注意事项**:
  - Prefill 阶段形状动态，不能 capture，只 capture decode
  - Triton kernel 与 graph capture 兼容性需验证（decode 阶段已用 native ops，风险降低）
  - attention_mask 和 cache_position 每步需 in-place 更新（用 placeholder tensor）
- **参考**: SGLang `CudaGraphRunner`, vLLM `CUDAGraphRunner`

#### 4.4 Fused SiLU*up Kernel — 附加优化
- **方案**: 合并 `F.silu(gate) * up` 为单个 Triton kernel（仅 prefill 走 Triton）
- **预期**: 减少 1876 次 elementwise kernel（SiLU + Mul → 1 fused）

#### 4.5 Fused Residual Add + RMSNorm — 附加优化
- **方案**: 合并 `residual + hidden_states` 和 `_rms_norm()` 为单个 kernel
- **预期**: 减少 1 次 memory read/write per layer per step

---

## 实施优先级矩阵（更新后）

```
已完成:
  ✅ Flash SDPA (P0)
  ✅ Fused QKV + gate_up (OptimizedDecoderLayer)
  ✅ CUDA config (TF32, SDP)
  ✅ Triton RMSNorm + RoPE (prefill), native 分支 (decode)

已验证无效:
  ❌ torch.compile (动态 shape recompile)
  ❌ StaticCache via HF generate (VLM 动态输入不兼容, 但自定义 loop 可绕过)
  ❌ flash_attention_2 (decode CPU dispatch 开销)

下一步 (CUDA Graph 路径):
  [1] 自定义 greedy decode loop (消除 HF generate overhead + sync)
  [2] StaticCache 预分配 (固定 tensor 地址, 消除 cat)
  [3] CUDA Graph capture decode step (消除所有 kernel launch overhead)
  
  预期: throughput 从 ~55 tok/s → 200+ tok/s (A10 理论上限 231 tok/s)
```

---

## 风险与注意事项

1. **A10 vs A800**: 开发在 A10 (sm_86), 评测在 A800 (sm_80), 预编译 .so 需针对 sm_80
2. **torch.compile recompile**: VLM 动态 shape 可能导致频繁重编译, 需设 `dynamic=True` 或 padding
3. **transformers 版本**: 本地 5.2.0, 需确认评测端版本兼容性
4. **accuracy 至上**: 权重 0.4, 任何优化后必须立即验证 accuracy 不下降
5. **提交限制**: 评测不自动编译 .cu, 需提交预编译 .so 或纯 Triton/Python
