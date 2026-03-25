# Qwen3-VL-2B-Instruct 推理优化计划

> 目标: 在不修改模型参数的前提下，优化 TTFT 和 Throughput
> 评分: 0.4 * Accuracy + 0.3 * TTFT_Improvement + 0.3 * Throughput_Improvement
> 硬件: NVIDIA A800 80GB PCIe (sm_80), CUDA 12.8, PyTorch 2.8.0+cu128
> 约束: 禁止量化/稀疏化/修改模型参数，仅修改 evaluation_wrapper.py + 自定义模块

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

#### 1.1 torch.compile 模型编译
- 方案: `torch.compile(model, mode="reduce-overhead")` 或分部件编译
- 原理: 算子自动融合, 减少 Python 开销和 kernel launch 次数
- 参考: vLLM `@support_torch_compile()` 对 Vision Transformer 的应用
- 风险: VLM 动态 shape (不同图像分辨率) 可能导致频繁 recompile
- 备选: 仅编译 Vision Encoder (`model.visual`) 规避动态 shape

#### 1.2 Flash SDPA 确认与优化
- 方案: 确认 `torch.backends.cuda.enable_flash_sdp(True)` 已启用
- 原理: PyTorch 内置 Flash Attention 后端, 对 prefill 大序列有效
- 参考: SGLang 6 种 Vision Attention 后端自动选择

#### 1.3 StaticCache KV 缓存预分配
- 方案: 使用 `cache_implementation="static"` 或手动预分配 KV 缓存
- 原理: 避免 decode 阶段每步动态分配内存, 减少 cudaMalloc 开销
- 参考: SGLang 静态内存分配策略

#### 1.4 Attention mask 优化
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

#### 3.1 Triton Fused RMSNorm
- 方案: Triton kernel 替换 torch RMSNorm
- 原理: Qwen3 每层 2 个 RMSNorm + QK-Norm, 28 层共 ~112 次调用
- 参考: SGLang/vLLM 的融合 RMSNorm 实现

#### 3.2 Triton Fused Rotary Embedding (RoPE)
- 方案: 融合 RoPE 计算到 attention kernel
- 原理: 减少单独的 RoPE kernel 调用

#### 3.3 Triton Fused SwiGLU FFN
- 方案: 融合 gate_proj + up_proj + SiLU + element-wise multiply
- 原理: 减少中间张量内存分配, 减少全局内存访问

#### 3.4 Fused QKV Projection
- 方案: 合并 Q/K/V 三个 Linear 为一次 GEMM
- 参考: vLLM `QKVParallelLinear`
- 原理: 减少 3 次 GEMM 调用为 1 次

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

## 实施优先级矩阵

```
                     收益高                 收益低
实现容易   [P1] torch.compile        [P1] Flash SDPA 确认
           [P1] StaticCache          [P1] Attention mask
            
实现中等   [P2] flash_attn 安装      [P2] Conv3D 优化
           [P2] Vision compile       [P4] CUDA Stream
            
实现困难   [P2] CUDA Graph (ViT)     [P3] 完整 Triton Attention
           [P3] Triton RMSNorm       [P3] Fused QKV/FFN
```

---

## 风险与注意事项

1. **A10 vs A800**: 开发在 A10 (sm_86), 评测在 A800 (sm_80), 预编译 .so 需针对 sm_80
2. **torch.compile recompile**: VLM 动态 shape 可能导致频繁重编译, 需设 `dynamic=True` 或 padding
3. **transformers 版本**: 本地 5.2.0, 需确认评测端版本兼容性
4. **accuracy 至上**: 权重 0.4, 任何优化后必须立即验证 accuracy 不下降
5. **提交限制**: 评测不自动编译 .cu, 需提交预编译 .so 或纯 Triton/Python
