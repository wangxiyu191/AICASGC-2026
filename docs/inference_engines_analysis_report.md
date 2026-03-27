# Qwen3-VL-2B-Instruct 推理引擎深度分析报告

> 目标环境：NVIDIA A800 80GB PCIe, Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8
> 分析对象：SGLang, vLLM, llama.cpp
> 分析日期：2026-03-26

---

## 目录

1. [Qwen3-VL 模型架构概览](#1-qwen3-vl-模型架构概览)
2. [SGLang 引擎分析](#2-sglang-引擎分析)
3. [vLLM 引擎分析](#3-vllm-引擎分析)
4. [llama.cpp 引擎分析](#4-llamacpp-引擎分析)
5. [三引擎运行流程对比](#5-三引擎运行流程对比)
6. [关键优化技术对比](#6-关键优化技术对比)
7. [各引擎独特优化亮点](#7-各引擎独特优化亮点)
8. [对竞赛优化的启示与建议](#8-对竞赛优化的启示与建议)
9. [关键源码索引](#9-关键源码索引)

---

## 1. Qwen3-VL 模型架构概览

Qwen3-VL 是通义千问系列的视觉语言模型，推理流程分为 4 个核心阶段：

```
图像输入 --> [Vision Encoder] --> [跨模态投影] --> [与文本Token融合] --> [LLM Decoder 生成]
```

### 1.1 核心组件

| 组件 | 规格 |
|------|------|
| Vision Encoder | 27层 ViT, 1152维隐藏层, 16头注意力 |
| Patch Embedding | 3D卷积 (kernel=[2,16,16]), 支持视频时间维度 |
| 位置编码 | 3D M-RoPE (Multi-axis Rotary Position Embedding) |
| DeepStack | 从第 8/16/24 层提取中间特征，多尺度融合 |
| Patch Merger | spatial_merge_size=2, 将 4 个 patch 合并为 1 个 token |
| 跨模态投影 | 两层 MLP: FC1 -> GELU -> FC2, 输出维度=3584 |
| 语言模型 | Qwen3 架构, GQA, QK-Norm (RMSNorm), SwiGLU FFN |
| 词表大小 | 151936 |
| 特殊 Token | image_token_id=151655, video_token_id=151656 |

### 1.2 推理两阶段

- **Prefill 阶段**：处理完整输入（图像编码 + 文本编码 + 跨模态融合 + 首次前向传播），产出第一个 token。对应 TTFT 指标。
- **Decode 阶段**：逐 token 自回归生成，利用 KV Cache 避免重复计算。对应 Throughput 指标。

### 1.3 性能基线

- TTFT: ~80ms
- Throughput: ~55 tokens/sec
- 测试配置：max_new_tokens=128, do_sample=False

---

## 2. SGLang 引擎分析

### 2.1 模型注册与加载

**关键文件：**
- 配置：`sglang/python/sglang/srt/configs/qwen3_vl.py` (27.9 KB)
- 模型：`sglang/python/sglang/srt/models/qwen3_vl.py` (53.2 KB)
- MoE变体：`sglang/python/sglang/srt/models/qwen3_vl_moe.py` (15.2 KB)
- 处理器：`sglang/python/sglang/srt/multimodal/processors/qwen_vl.py` (586行)

**配置类层级：**

```
Qwen3VLConfig
├── Qwen3VLTextConfig
│   ├── vocab_size: 151936
│   ├── hidden_size: 4096 (2B版本为1536)
│   ├── num_hidden_layers: 32 (2B版本为28)
│   ├── num_attention_heads: 32 (2B版本为12)
│   └── max_position_embeddings: 128000
└── Qwen3VLVisionConfig
    ├── depth: 27
    ├── hidden_size: 1152
    ├── num_heads: 16
    ├── patch_size: 16
    ├── spatial_merge_size: 2
    ├── temporal_patch_size: 2
    ├── out_hidden_size: 3584 (2B版本为1536)
    ├── num_position_embeddings: 2304
    └── deepstack_visual_indexes: [8, 16, 24]
```

**模型注册映射：**
- `Qwen3VLForConditionalGeneration` -> 基础版本
- `Qwen3VLMoeForConditionalGeneration` -> MoE 版本

### 2.2 Vision Encoder 实现

SGLang 的 Vision Encoder 实现是三个引擎中**优化程度最高**的，提供了 6 种 Attention 后端：

**Patch Embedding：**
```python
class Qwen3VLVisionPatchEmbed:
    # Conv3dLayer: kernel=[2,16,16], stride=[2,16,16]
    # 输入: (B, 3, T, H, W) -> 输出: (-, 1152)
```

**位置编码：**
- VocabParallelEmbedding (2304 位置, 1152 隐层)
- `fast_pos_embed_interpolate_from_list()` 快速双线性插值
- 支持可变分辨率自适应

**Vision Attention 后端（6种）：**

| 后端 | 类名 | 说明 |
|------|------|------|
| Flash Attention 3 | `VisionFlash3Attention` | 最新 FA 实现 |
| Flash Attention 4 | `VisionFlash4Attention` | 下一代 FA |
| FlashInfer cuDNN | `VisionFlashInferAttention` | cuDNN 加速 prefill |
| Triton | `VisionTritonAttention` | Triton kernel 实现 |
| SDPA | `VisionSdpaAttention` | PyTorch 原生 |
| ROCm AITER | - | AMD GPU 专用 |

文件位置：`sglang/python/sglang/srt/layers/attention/vision.py` (36.3 KB)

**FlashInfer cuDNN 配置：**
```python
# 工作空间：128 MB
# Seqlen buckets: [4K, 8K, 16K, 32K, 64K, 128K]
# Batch buckets: [8, 16, 32, 64]
cudnn_batch_prefill_with_kv_cache()
```

**Vision Transformer Block：**
```
输入 hidden_states
    |
    v
LayerNorm (eps=1e-6)
    |
    v
VisionAttention (选择最优后端)
    |
    v (残差连接)
LayerNorm (eps=1e-6)
    |
    v
Qwen3_VisionMLP (gate_proj + up_proj -> GELU -> down_proj)
    |
    v (残差连接)
输出 hidden_states

# 在第 8/16/24 层额外执行 DeepStack 特征提取
```

### 2.3 语言模型优化

**Attention 实现（Qwen3Attention）：**
- QK Norm: RMSNorm 在 Q/K 头维度
- Grouped Query Attention (GQA)
- RadixAttention: KV 缓存管理核心
- RoPE 支持多种类型: default, linear, dynamic, yarn, longrope, llama3

**Decoder Layer 结构：**
```
输入 hidden_states
    |
    v
RMSNorm -> Self-Attention (Qwen3Attention, 含 RadixAttention)
    |            |-- QKV 投影 (Tensor Parallel)
    |            |-- QK-Norm (RMSNorm)
    |            |-- RoPE
    |            |-- Paged Attention
    |            |-- Output 投影
    v (残差连接)
RMSNorm -> MLP (Qwen3MLP)
    |            |-- gate_proj + up_proj -> SiLU -> down_proj
    v (残差连接)
输出 hidden_states
```

**KV Cache 管理 - Radix Tree：**
- 树结构管理 KV Cache，支持前缀共享
- 多请求之间自动复用公共前缀的 KV Cache
- 文件：`sglang/python/sglang/srt/layers/quantization/kv_cache.py`

### 2.4 跨模态连接器

**核心函数：** `general_mm_embed_routine()` (mm_utils.py:1147)

```
视觉特征 (来自 Vision Encoder, DeepStack)
    |
    v
Qwen3VLMoeVisionPatchMerger
    |-- ColumnParallelLinear (1152*4 -> out_hidden_size)
    |-- GELU 激活
    |-- RowParallelLinear (-> out_hidden_size)
    |-- 空间合并 (spatial_merge_size=2, 4x压缩)
    v
MultimodalInputs 处理
    |-- 特征缓存管理 (GPU/CPU 平衡)
    |-- 特征哈希验证
    v
embed_mm_inputs() / embed_mm_inputs_chunked()
    |-- 视觉 token 与文本 token 交错
    |-- 位置编码调整
    v
融合后的输入 -> 语言模型
```

### 2.5 关键优化技术

**2.5.1 CUDA Graph（ViT 专用）**

文件：`sglang/python/sglang/srt/multimodal/vit_cuda_graph_runner.py`

```python
class ViTCudaGraphRunner:
    # 按 (batch_size, seq_len) 分桶缓存执行图
    block_graphs: Dict[Hashable, torch.cuda.CUDAGraph]
    
    # 旋转位置缓存（地址稳定性）
    sin_cos_ws: Optional[Tuple[torch.Tensor, torch.Tensor]]
    
    # DeepStack 兼容
    _deepstack_visual_indexes: list
    _deepstack_merger_list: nn.ModuleList
```

- 捕获 Vision Transformer 块的执行图
- 按 batch size 和 seqlen 分桶参数化
- 消除 kernel launch 开销

**2.5.2 Tensor/Pipeline/Expert 并行**

```python
# Tensor Parallel
layers.*.self_attn.q_proj: "colwise"
layers.*.self_attn.k_proj: "colwise"
layers.*.self_attn.v_proj: "colwise"
layers.*.self_attn.o_proj: "rowwise"
layers.*.mlp.*_proj: colwise/rowwise

# Pipeline Parallel
pp_group.is_first_rank  # 包含 embedding 层
pp_group.is_last_rank   # 包含 LM Head

# Expert Parallel (MoE版本)
# DeepEP 支持
```

**2.5.3 Continuous Batching**

- 调度器：`sglang/python/sglang/srt/managers/schedule_batch.py`
- 多模态数据项管理
- GPU 特征缓冲区预分配 (`SGLANG_MM_BUFFER_SIZE_MB`)

**2.5.4 量化支持**

| 方法 | 文件 |
|------|------|
| INT8 (W8A8) | `w8a8_int8.py` |
| FP8 (W8A8) | `w8a8_fp8.py` |
| FP4 | `fp4_utils.py`, `w4afp8.py` |
| AWQ | `awq_triton.py` |
| Quark | `quark/` 子目录 |

注意：Vision Encoder 通常不应用量化，仅语言模型部分量化。

**2.5.5 DP-Attention (Data Parallel Attention)**

- 激活方式：`--mm-enable-dp-encoder`
- Vision Encoder 独立数据并行处理
- 文件：`sglang/python/sglang/srt/layers/dp_attention.py`

### 2.6 可调配置参数

```bash
# 多模态并发控制
--mm-max-concurrent-calls <value>
--mm-per-request-timeout <seconds>

# 内存优化
--keep-mm-feature-on-device       # 避免 D2H 拷贝
SGLANG_USE_CUDA_IPC_TRANSPORT=1   # 进程间内存共享
SGLANG_VLM_CACHE_SIZE_MB=0        # 禁用特征缓存

# Prefill 优化
--chunked-prefill-size 8192

# 静态内存分配
--mem-fraction-static 0.65

# Vision Attention 后端选择
--mm-attention-backend flashinfer_cudnn  # 或 fa3/fa4

# ViT CUDA Graph
SGLANG_VIT_ENABLE_CUDA_GRAPH=1
```

---

## 3. vLLM 引擎分析

### 3.1 模型注册与加载

**关键文件：**
- 模型：`vllm/vllm/model_executor/models/qwen3_vl.py` (62.8 KB)
- 基类：`vllm/vllm/model_executor/models/qwen2_5_vl.py` (56.8 KB)
- 注册：`vllm/vllm/model_executor/models/registry.py`

**注册映射：**
```python
# registry.py 中 _MULTIMODAL_MODELS 字典
"Qwen3VLForConditionalGeneration": ("qwen3_vl", "Qwen3VLForConditionalGeneration")
"Qwen3VLMoeForConditionalGeneration": ("qwen3_vl_moe", "Qwen3VLMoeForConditionalGeneration")
```

**模型类层级：**
```
Qwen3VLForConditionalGeneration (nn.Module)
├── visual: Qwen3_VisionTransformer          # 视觉编码器
├── language_model: Qwen3LLMForCausalLM      # 语言模型（继承 Qwen3ForCausalLM）
├── SupportsLoRA                              # LoRA 适配器支持
├── SupportsMRoPE                             # M-RoPE 支持
└── SupportsPP                                # Pipeline Parallel 支持
```

### 3.2 Vision Encoder 实现

**核心组件（qwen3_vl.py 第 126-609 行）：**

```
Qwen3_VisionPatchEmbed
    |-- Conv3dLayer (kernel=[temporal_patch_size, patch_size, patch_size])
    |-- stride = kernel_size（无重叠）
    |-- 输入: (B, 3, T, H, W) -> (-, hidden_size)

Qwen3_VisionBlock (x27层)
    |-- LayerNorm
    |-- Qwen2_5_VisionAttention（复用 Qwen2.5-VL 实现）
    |-- MLP
    |-- 残差连接

Qwen3_VisionPatchMerger
    |-- LayerNorm
    |-- FC1 (hidden_size * spatial_merge_size^2 -> intermediate)
    |-- GELU
    |-- FC2 (intermediate -> out_hidden_size)
```

**Vision Attention 后端（3种）：**

| 后端 | 枚举值 | 说明 |
|------|--------|------|
| Flash Attention | `FLASH_ATTN` | `flash_attn_varlen_func`，支持可变长度序列 |
| PyTorch SDPA | `TORCH_SDPA` | `vit_torch_sdpa_wrapper` |
| XFormers | `XFORMERS` | `vit_xformers_attn_wrapper` |

后端选择函数（`attention/ops/vit_attn_wrappers.py`）：
```python
def get_vit_attn_backend(head_size, dtype, attn_backend_override=None):
    # 优先级：override > 环境变量 > 平台默认
    # Flash Attention 条件：FP16/BF16, CUDA, 算力 >= 80
```

**Attention 前向传播细节（qwen2_5_vl.py 第 362-430 行）：**
```python
def forward(self, x, cu_seqlens, rotary_pos_emb, max_seqlen, seqlens):
    # 1. QKV 投影 (QKVParallelLinear)
    x, _ = self.qkv(x)
    
    # 2. 重排张量
    qkv = einops.rearrange(x, "s b (three head head_dim) -> b s three head head_dim")
    
    # 3. 应用 RoPE (3D 旋转位置编码)
    qk_rotated = apply_rotary_pos_emb_vision(qk_reshaped, rotary_pos_emb)
    
    # 4. 选择 Attention 后端执行
    if is_flash_attn:
        context_layer = vit_flash_attn_wrapper(q, k, v, cu_seqlens, max_seqlen)
    elif torch_sdpa:
        context_layer = vit_torch_sdpa_wrapper(q, k, v, cu_seqlens)
    elif xformers:
        context_layer = vit_xformers_attn_wrapper(q, k, v, seqlens)
    
    # 5. 输出投影
    output, _ = self.proj(context_layer)
    return output
```

**RoPE 实现（qwen2_5_vl.py 第 591-624 行）：**
```python
class Qwen2_5_VisionRotaryEmbedding:
    # 3D RoPE：时间、高度、宽度维度
    # 频率缓存机制 (_freqs_cached)
    # 动态长度适配
```

**torch.compile 支持：**
```python
@support_torch_compile()
class Qwen3_VisionTransformer:
    ...

@support_torch_compile()
class Qwen2_5_VisionBlock:
    ...
```

### 3.3 语言模型优化

**Qwen3LLMForCausalLM（qwen3_vl.py 第 1143-1173 行）：**
- 继承 `Qwen3ForCausalLM`
- 基于 `text_config` 配置
- 支持可选词嵌入绑定

**Attention 实现：**
- 使用 vLLM 通用注意力层 (`vllm.attention`)
- 后端自动选择 (`get_attn_backend`)
- PagedAttention 是核心 KV Cache 管理策略

**PagedAttention 细节：**
```python
# KV Cache 规范
class FullAttentionSpec(KVCacheSpec):
    # 每个 block 的大小 = 2 * block_size * num_kv_heads * head_size * dtype_size
    # 动态块分配避免内存碎片
    # 页表管理

class SlidingWindowSpec(KVCacheSpec):
    # 滑动窗口注意力的 KV Cache 变体
```

**融合线性层：**
- `ColumnParallelLinear`: 列并行（QKV 投影）
- `RowParallelLinear`: 行并行（输出投影）
- `ParallelLMHead`: 分布式词表投影
- 支持 `quant_config` 参数进行量化

### 3.4 跨模态连接器

**Patch Merger（qwen3_vl.py 第 251-300 行）：**
```python
class Qwen3_VisionPatchMerger(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)                    # LayerNorm
        x_parallel, _ = self.linear_fc1(x)  # 投影到中间维度
        x_parallel = self.act_fn(x_parallel) # GELU
        out, _ = self.linear_fc2(x_parallel) # 投影到输出维度
        return out
```

**多模态嵌入融合（qwen3_vl.py 第 1482-1501 行）：**
```python
def embed_multimodal():
    # 1. 解析多模态输入（图像/视频）
    # 2. 通过视觉编码器处理
    # 3. 通过 Patch Merger 映射到语言模型空间
    # 4. 与文本 embeddings 合并（占位符替换）
```

**MRoPE 位置编码（qwen3_vl.py 第 1448-1477 行）：**
```python
def get_mrope_input_positions():
    # 3D 位置编码 (时间, 高度, 宽度)
    # 支持多个视觉对象的位置信息融合
```

**DeepStack 实现：**
```python
# 预分配多层 embedding 缓冲区
self.deepstack_input_embeds = [
    torch.zeros(max_num_batched_tokens, hidden_size)
    for _ in range(deepstack_num_level)
]
# 从视觉编码器的中间层提取特征，逐层注入语言模型
```

### 3.5 关键优化技术

**3.5.1 PagedAttention（vLLM 首创）**

KV Cache 按固定大小的 block 分配，通过页表映射管理：
- 消除 KV Cache 预分配浪费
- 支持 KV Cache 共享（Beam Search、前缀共享）
- 文件：`vllm/vllm/v1/kv_cache_interface.py` (14.0 KB)

**3.5.2 Continuous Batching**

- 请求级别的动态 batch 组装
- 每次迭代可加入/移除请求
- Prefill 和 Decode 可交错执行

**3.5.3 torch.compile 集成**

```python
@support_torch_compile()
class Qwen3_VisionTransformer:
    # 动态维度标记确保编译图正确性
    # 算子融合，减少 kernel launch 开销
```

**3.5.4 DP 分片 Vision Model**

文件：`vllm/vllm/model_executor/models/vision.py` (19.0 KB)

```python
def run_dp_sharded_mrope_vision_model():
    # 1. 按补丁数量负载均衡分配到多 GPU
    # 2. 本地处理
    # 3. All-gather 汇总结果
    # 4. 恢复原始顺序
```

**3.5.5 量化支持**

- 通过 `quant_config` 参数统一配置
- 支持 INT8/FP8 量化（BitsAndBytes, GPTQ, AWQ 等）
- 视觉编码器的量化线性层：`QKVParallelLinear(quant_config=quant_config)`

**3.5.6 LoRA 支持**

```python
class Qwen3VLForConditionalGeneration(SupportsLoRA, ...):
    # 原生 LoRA 适配器支持
    # 可在推理时动态加载不同 LoRA 权重
```

### 3.6 完整推理流程

```
Prefill 阶段:
1. 图像预处理 (smart_resize, 归一化)
   |-- 因子: 28 (patch对齐)
   |-- 最小像素: 4*28*28 = 3136
   |-- 宽高比限制: 200:1
   
2. Vision Encoder
   |-- Patch Embedding (Conv3D)
   |-- 27层 ViT Block (选择最优 Attention 后端)
   |-- DeepStack 中间层特征提取
   |-- Patch Merger (4x 空间压缩)
   
3. 跨模态融合
   |-- 构建 MRoPE 位置编码 (3D)
   |-- 视觉 token 替换文本占位符
   |-- 生成融合后的 input embeddings
   
4. LLM Forward Pass
   |-- 28层 Decoder (Qwen3)
   |-- PagedAttention (KV Cache 分配)
   |-- 输出第一个 token

Decode 阶段:
1. 单 token 输入
2. PagedAttention (复用 KV Cache)
3. 逐层 Decoder 处理
4. Logits -> 采样 -> 输出 token
```

---

## 4. llama.cpp 引擎分析

### 4.1 模型转换与加载

llama.cpp 使用自定义的 GGUF 格式，需要从 HuggingFace 模型转换。

**关键文件：**
- 转换脚本：`llama.cpp/convert_hf_to_gguf.py` (行 4772-4980)
- 视觉编码器图：`llama.cpp/tools/mtmd/models/qwen3vl.cpp`
- 文本模型图：`llama.cpp/src/models/qwen3vl.cpp`
- CLIP 核心：`llama.cpp/tools/mtmd/clip.cpp`
- 模型结构定义：`llama.cpp/tools/mtmd/clip-model.h`

**转换类：**
```python
class Qwen3VLVisionModel(MmprojModel)  # 视觉编码器转换 (行 4772-4888)
class Qwen3VLTextModel(Qwen3Model)      # 文本模型转换 (行 4917-4933)
class Qwen3VLMoeTextModel(Qwen3MoeModel) # MoE 文本模型转换 (行 4937-4980)
```

**转换流程：**
```
HuggingFace PyTorch 权重
    |
    v
convert_hf_to_gguf.py
    |
    |-- Vision Model 转换:
    |   |-- Conv3D -> 2个 Conv2D 分解（沿时间维度拆分）
    |   |-- DeepStack 层处理 (norm/FC1/FC2)
    |   |-- 多模态投影层 (mm_0_w, mm_1_w)
    |   |-- 位置嵌入: image_size = sqrt(num_position_embeddings) * patch_size
    |   
    |-- Text Model 转换:
    |   |-- MoE 专家张量排列 (permute 0,2,1)
    |   |-- 跳过视觉相关张量
    |
    |-- GGUF 元数据写入:
    |   |-- projector_type = QWEN3VL
    |   |-- vision_use_gelu = True
    |   |-- spatial_merge_size
    |   |-- num_deepstack_layers
    |   |-- attention_layernorm_eps
    |
    v
vision.gguf + text.gguf (GGUF 格式文件)
```

### 4.2 Vision Encoder 实现（C/C++）

**核心数据结构（clip-model.h）：**
```cpp
struct clip_layer {
    // Attention 组件
    ggml_tensor * k_w, * q_w, * v_w;     // K/Q/V 权重
    ggml_tensor * k_norm, * q_norm;       // K/Q 归一化 (Qwen3-VL 特性)
    ggml_tensor * o_w, * o_b;             // 输出投影
    
    // FFN 组件
    ggml_tensor * ff_up_w, * ff_gate_w, * ff_down_w;
    
    // LayerNorm
    ggml_tensor * ln_1_w, * ln_2_w;
    
    // Qwen3VL DeepStack
    ggml_tensor * deepstack_norm_w;
    ggml_tensor * deepstack_fc1_w;
    ggml_tensor * deepstack_fc2_w;
};
```

**前向传播（qwen3vl.cpp 中 `clip_graph_qwen3vl::build()`）：**
```
1. Patch Embedding
   |-- Conv2D_0 (patch_embeddings_0) + Conv2D_1 (patch_embeddings_1)
   |-- 两个卷积输出相加（Conv3D 分解实现）
   |-- Permute: [w, h, c, b] -> [c, w, h, b]

2. 位置编码
   |-- 可学习位置嵌入 (resize_position_embeddings)
   |-- 支持分辨率自适应插值

3. Pre LayerNorm (可选)

4. Transformer Layers (27层)
   对每层:
   |-- LayerNorm
   |-- QKV 投影 (单个融合矩阵乘法)
   |-- Q/K 分离并应用 M-RoPE
   |   |-- ggml_rope_multi(): 多轴旋转位置编码
   |   |-- mrope_sections: 4个部分, GGML_ROPE_TYPE_VISION
   |   |-- base=10000, d_head/2 维度
   |-- Attention 计算
   |   |-- Flash Attention: ggml_flash_attn_ext() (可选)
   |   |-- 标准 Attention: ggml_mul_mat (Q*K^T) -> softmax -> *V
   |-- 输出投影
   |-- 残差连接
   |-- LayerNorm
   |-- FFN (up + gate -> SiLU -> down)
   |-- 残差连接
   |
   |-- [DeepStack 处理] (在指定层):
       |-- reshape: [n_embd, n_pos] -> [n_embd * merge_factor, n_pos / merge_factor]
       |-- LayerNorm (deepstack_norm)
       |-- FFN with GELU (deepstack_fc1 -> GELU -> deepstack_fc2)
       |-- 累积拼接到 deepstack_features

5. Post LayerNorm (可选)

6. 多模态投影
   |-- reshape: [n_embd, n_pos] -> [n_embd*4, n_pos/4] (4x空间合并)
   |-- FFN: mm_0_w -> GELU -> mm_1_w
   |-- 拼接 DeepStack 特征 (ggml_concat)

7. 输出: 投影后的视觉嵌入
```

### 4.3 图像预处理（clip.cpp 第 3050-3250 行）

```cpp
// Qwen3VL 特定预处理
case PROJECTOR_TYPE_QWEN3VL:
{
    // 1. 计算目标尺寸（保留纵横比）
    const clip_image_size new_size = img_tool::calc_size_preserved_ratio(
        original_size,
        params.patch_size * 2,        // 对齐因子
        params.image_min_pixels,      // 最小像素数
        params.image_max_pixels);     // 最大像素数
    
    // 2. 双线性插值调整大小
    img_tool::resize(*img, resized, new_size, RESIZE_ALGO_BILINEAR, false);
    
    // 3. 归一化: (pixel/255.0 - mean) / std
    normalize_image_u8_to_f32(resized, *img_f32, params.image_mean, params.image_std);
}
```

### 4.4 语言模型实现（src/models/qwen3vl.cpp）

```cpp
llm_build_qwen3vl::llm_build_qwen3vl(const llama_model & model, 
                                      const llm_graph_params & params) {
    // 1. Token 嵌入
    inpL = build_inp_embd(model.tok_embd);
    
    // 2. 位置信息和 KV Cache
    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();
    
    // 3. 28层 Decoder
    for (int il = 0; il < n_layer; ++il) {
        // RMSNorm
        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        
        // QKV 投影 (支持 LoRA)
        Qcur = build_lora_mm(layer.wq, cur);
        Kcur = build_lora_mm(layer.wk, cur);
        Vcur = build_lora_mm(layer.wv, cur);
        
        // QK-Norm (Qwen3-VL 特性)
        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        
        // M-RoPE (多轴旋转位置编码)
        Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, ...);
        Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, ...);
        
        // Self-Attention (含 KV Cache)
        cur = build_attn(inp_attn, layer.wo, layer.bo, Qcur, Kcur, Vcur, ...);
        
        // 残差连接
        cur = ggml_add(ctx0, cur, inpL);
        
        // DeepStack 残差注入
        if (il < n_deepstack_layers) {
            ggml_tensor * ds = ggml_view_2d(ctx0, res->t_inp_embd, 
                                            n_embd, n_tokens, ..., 
                                            (il + 1) * n_embd * sizeof(float));
            cur = ggml_add(ctx0, cur, ds);
        }
        
        // RMSNorm + FFN (SwiGLU)
        cur = build_ffn(cur, layer.ffn_up, NULL, NULL,
                        layer.ffn_gate, NULL, NULL,
                        layer.ffn_down, NULL, NULL,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
    }
    
    // 4. 最终 RMSNorm + LM Head
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cur = build_lora_mm(model.output, cur);
}
```

### 4.5 量化策略

**支持的量化格式：**

| 格式 | 比特数 | 块大小 | 精度等级 | 适用场景 |
|------|--------|--------|---------|---------|
| Q2_K | 2-3 | 256 | 最低 | 极端内存受限 |
| Q3_K_S/M/L | 3 | 256 | 低 | 低内存场景 |
| Q4_0 | 4 | 32 | 中低 | 快速推理 |
| Q4_K_M | 4 | 256 | 中 | 平衡选择 |
| Q5_0 | 5 | 32 | 中高 | 中等质量 |
| Q5_K_M | 5 | 256 | 高 | 高质量 |
| Q6_K | 6 | 256 | 很高 | 接近原精度 |
| Q8_0 | 8 | 32 | 极高 | 近无损 |
| BF16 | 16 | - | 浮点 | 无损 |
| F32 | 32 | - | 浮点 | 原始精度 |

**量化特殊处理：**
- 位置嵌入始终保持 F32 精度
- Vision Encoder 通常使用更高精度量化
- 文件：`ggml/src/ggml-quants.c`

### 4.6 C/C++ 性能优化

**4.6.1 Flash Attention 实现（ggml-cpu/ops.cpp 第 8130-8860 行）**

```cpp
// 标准 Flash Attention
static void ggml_compute_forward_flash_attn_ext_f16_one_chunk(...) {
    // 分块处理 Q, K, V
    // 在线 softmax 优化（避免全局 softmax）
    // IO 复杂度从 O(N^2) 降至 O(N)
}

// 平铺 Flash Attention（更高效的块分解）
static void ggml_compute_forward_flash_attn_ext_tiled(...) {
    // Q 块大小 (bq): 64-128
    // K/V 块大小 (bkv): 16-32
    // 减少寄存器溢出
}
```

**4.6.2 SIMD 指令集优化**

| 架构 | 指令集 | 文件 |
|------|--------|------|
| x86_64 | AVX-512, AVX2, AVX, SSE | `ggml/src/ggml-cpu/arch/x86/quants.c` (3820行) |
| ARM | NEON, INT8 | `ggml/src/ggml-cpu/arch/arm/quants.c` (4136行) |

- 矩阵乘法：分块计算，优化缓存命中
- 量化/反量化：SIMD 并行处理
- 向量点积：批量 SIMD 操作

**4.6.3 CUDA 后端优化**

| 功能 | 文件 |
|------|------|
| Flash Attention | `ggml/src/ggml-cuda/fattn.cu` |
| Conv2D | `ggml/src/ggml-cuda/conv2d.cu` |
| 反量化 | `ggml/src/ggml-cuda/dequantize.cuh` |
| 归一化 | `ggml/src/ggml-cuda/norm.cuh` |

**4.6.4 计算图优化**

```
图优化流程:
1. ggml_cgraph 构建
2. 节点融合（合并兼容操作）
3. 内存分配（最小化峰值内存）
4. ggml_backend_sched 调度（自动选择最优后端）

后端调度:
ggml_backend_sched_new(backends, n_backends)
-> ggml_backend_sched_graph_compute(sched, cgraph)
-> 自动为每个算子选择最高效的后端
```

**4.6.5 多后端支持**

| 后端 | 说明 |
|------|------|
| CPU | C 实现 + SIMD (AVX-512/AVX2/NEON) |
| CUDA | GPU kernel |
| Vulkan | 计算 shader（含 Vision RoPE: `rope_vision.comp`） |
| Metal | macOS/iOS GPU |

### 4.7 完整推理调用链

```
main()
  |
  v
llama_model_load_from_file()
  |-- 读取 text.gguf (语言模型)
  |-- clip_init() (加载 vision.gguf)
  
llama_eval_image()
  |-- clip_image_preprocess()
  |   |-- calc_size_preserved_ratio() (计算目标尺寸)
  |   |-- resize() (双线性插值)
  |   |-- normalize_image_u8_to_f32() (归一化)
  |
  |-- clip_image_batch_encode()
  |   |-- clip_graph_qwen3vl::build() (构建计算图)
  |   |   |-- Patch Embedding (2x Conv2D)
  |   |   |-- 27层 ViT (M-RoPE, Attention, FFN)
  |   |   |-- DeepStack 特征提取
  |   |   |-- 多模态投影 (FFN with GELU)
  |   |
  |   |-- ggml_backend_sched_graph_compute() (执行)
  
llama_eval_tokens()
  |-- 连接视觉嵌入 + 文本 token
  |-- llm_build_qwen3vl() (构建语言模型图)
  |   |-- Token Embedding
  |   |-- 28层 Decoder (M-RoPE, Attention, FFN, DeepStack注入)
  |   |-- LM Head
  |
  |-- ggml_backend_sched_graph_compute() (执行)
  |-- 采样 -> 输出 token
```

---

## 5. 三引擎运行流程对比

### 5.1 统一流程视角

```
                SGLang                    vLLM                      llama.cpp
                ------                    ----                      ---------
图像输入     HF Processor             HF Processor              clip_image_preprocess
             smart_resize             smart_resize              calc_size_preserved_ratio
             factor=28                factor=28                 bilinear resize
             
Patch Emb    Conv3dLayer              Conv3dLayer               2x Conv2D (分解)
             原生3D卷积                原生3D卷积                 兼容性分解
             
位置编码     VocabParallelEmbed       Qwen2_5_VisionRotaryEmb   ggml_rope_multi
             + 快速双线性插值          + _freqs_cached           + GGML_ROPE_TYPE_VISION
             
Vision Attn  6种后端自动选择           3种后端自动选择             ggml_flash_attn_ext
             FA3/FA4/cuDNN/Triton     FA/SDPA/XFormers          CPU/CUDA 实现
             
DeepStack    逐层注入语言模型          预分配缓冲区               拼接到投影输出
             separate_deepstack       deepstack_input_embeds    ggml_concat
             
Patch Merge  ColumnParallel+RowParallel  FC1->GELU->FC2         build_ffn (GELU)
             Tensor Parallel           支持量化                   GGML 算子
             
LLM Attn     RadixAttention           PagedAttention            KV Cache + FA
             Radix Tree 管理          分页块管理                  连续内存预分配
             
Batching     Continuous Batching      Continuous Batching       单请求串行
             Radix Tree 前缀共享      PagedAttention             无 batch 支持
             
生成策略     Chunk Prefill            Chunk Prefill             逐 token 生成
             CUDA Graph               torch.compile              计算图复用
```

### 5.2 Prefill 阶段对比

| 步骤 | SGLang | vLLM | llama.cpp |
|------|--------|------|-----------|
| 图像预处理 | Python (HF Processor) | Python (HF Processor) | C++ (零拷贝) |
| Vision 编码 | GPU (CUDA Graph 可选) | GPU (torch.compile 可选) | 多后端调度 |
| 跨模态融合 | GPU (embed_mm_inputs) | GPU (embed_multimodal) | CPU+GPU 混合 |
| LLM Prefill | GPU (RadixAttention) | GPU (PagedAttention) | GPU/CPU |

### 5.3 Decode 阶段对比

| 步骤 | SGLang | vLLM | llama.cpp |
|------|--------|------|-----------|
| KV Cache | Radix Tree（前缀共享） | Paged Blocks | 连续内存 |
| Attention | RadixAttention | PagedAttention | ggml_flash_attn |
| 调度 | Continuous Batching | Continuous Batching | 串行 |
| 内存效率 | 极高（前缀共享） | 高（分页管理） | 中（预分配） |

---

## 6. 关键优化技术对比

### 6.1 Attention 优化

| 特性 | SGLang | vLLM | llama.cpp |
|------|--------|------|-----------|
| Flash Attention 版本 | FA3/FA4 | FA2+ | 自实现 (分块+在线softmax) |
| Vision 专用 FA | 6种后端 | 3种后端 | ggml_flash_attn_ext |
| PagedAttention | Radix Tree | Block Table | 无 |
| cuDNN Attention | FlashInfer cuDNN | 无 | 无 |
| Triton Attention | VisionTritonAttention | 无 | 无 |

### 6.2 内存管理

| 特性 | SGLang | vLLM | llama.cpp |
|------|--------|------|-----------|
| KV Cache 架构 | Radix Tree | Paged Block | 连续预分配 |
| 前缀共享 | 原生支持 | 支持 | 不支持 |
| 内存碎片 | 极低 | 低 | 无碎片 |
| 动态分配 | 树结构增长 | 按需分配 block | 固定大小 |
| 内存利用率 | 最高 | 高 | 中 |

### 6.3 量化支持

| 格式 | SGLang | vLLM | llama.cpp |
|------|--------|------|-----------|
| INT8 (W8A8) | w8a8_int8 | BitsAndBytes/GPTQ | Q8_0 |
| FP8 | w8a8_fp8 | 支持 | 不支持 |
| INT4 | FP4, AWQ | BitsAndBytes, GPTQ, AWQ | Q4_0, Q4_K_M |
| 2-3 bit | 无 | 无 | Q2_K, Q3_K |
| 5-6 bit | 无 | 无 | Q5_K, Q6_K |
| 混合精度 | Vision FP16 + LLM INT8 | Vision FP16 + LLM 量化 | 逐张量精度控制 |

### 6.4 并行化

| 特性 | SGLang | vLLM | llama.cpp |
|------|--------|------|-----------|
| Tensor Parallel | 全面支持 | 全面支持 | 不支持 |
| Pipeline Parallel | 支持 | 支持 | 不支持 |
| Expert Parallel | 支持 (MoE) | 支持 | 不支持 |
| Data Parallel | DP-Attention | DP 分片 Vision | 不支持 |
| 多线程 CPU | 无 | 无 | n_threads 线程池 |

### 6.5 图优化与编译

| 特性 | SGLang | vLLM | llama.cpp |
|------|--------|------|-----------|
| CUDA Graph | ViT 专用 CudaGraphRunner | 支持 | 不适用 |
| torch.compile | 可集成 | `@support_torch_compile` | 不适用 (C++) |
| 计算图优化 | 无 (即时执行) | 无 (即时执行) | ggml 图优化+调度 |
| 算子融合 | Flash Attention 内含 | Flash Attention 内含 | 图级节点融合 |

### 6.6 硬件后端

| 后端 | SGLang | vLLM | llama.cpp |
|------|--------|------|-----------|
| NVIDIA CUDA | 主要平台 | 主要平台 | 支持 |
| AMD ROCm | 支持 | 支持 | 支持 |
| CPU (x86/ARM) | 不支持 | 不支持 | 主要平台 (SIMD) |
| Vulkan | 不支持 | 不支持 | 支持 |
| Metal (Apple) | 不支持 | 不支持 | 支持 |
| Ascend NPU | 支持 | 不支持 | 不支持 |
| Intel XPU | 支持 | 支持 | 不支持 |

---

## 7. 各引擎独特优化亮点

### 7.1 SGLang 独特优势

1. **Radix Tree KV Cache**
   - 核心创新：使用 Radix Tree（基数树）数据结构管理 KV Cache
   - 优势：天然支持前缀共享，多请求间自动复用公共前缀的 KV Cache
   - 场景：同一图像多个问题时，视觉 KV Cache 可完全复用

2. **6种 Vision Attention 后端**
   - FA3, FA4, FlashInfer cuDNN, Triton, SDPA, ROCm AITER
   - 根据硬件和序列长度自动选择最优实现
   - FlashInfer cuDNN 对 prefill 阶段特别有效

3. **ViT CUDA Graph Runner**
   - 专门为 Vision Transformer 捕获 CUDA 执行图
   - 按 (batch_size, seqlen) 分桶缓存
   - 消除 Vision 编码阶段的 kernel launch 开销
   - 兼容 DeepStack 中间层提取

4. **DP-Attention for Vision**
   - Vision Encoder 可独立进行数据并行
   - 不依赖 Tensor Parallel 的分片策略
   - 适合视觉输入负载不均的场景

### 7.2 vLLM 独特优势

1. **PagedAttention（行业标准）**
   - vLLM 首创，已成为 LLM 推理的标准 KV Cache 管理策略
   - 将 KV Cache 按固定大小 block 分配
   - 通过页表映射管理，消除内存预分配浪费
   - 支持 Beam Search、前缀共享等复杂场景

2. **torch.compile 深度集成**
   - Vision Transformer 组件标注 `@support_torch_compile()`
   - 动态维度标记确保编译图正确性
   - 算子自动融合（LayerNorm + Attention、FFN 内部融合等）
   - 减少 Python 解释器开销和 kernel launch 次数

3. **DP 分片 MRoPE Vision Model**
   - `run_dp_sharded_mrope_vision_model()` 专门为多模态设计
   - 按补丁数量进行负载均衡分配
   - All-gather 汇总并恢复原始顺序
   - 适合大分辨率图像的多 GPU 处理

4. **LoRA 原生支持**
   - `SupportsLoRA` 接口允许推理时动态加载适配器
   - 可在不同请求使用不同 LoRA 权重
   - 无需重新加载完整模型

### 7.3 llama.cpp 独特优势

1. **极致量化覆盖**
   - 从 Q2_K (2-3 bit) 到 F32 的完整量化频谱
   - K 系列量化（Q4_K_M 等）使用 256 块大小，精度优于传统 4-bit
   - 逐张量精度控制（如位置嵌入保持 F32）
   - 内存效率最高（Q4_K_M 下 2B 模型仅需约 1.5GB）

2. **Conv3D 分解技术**
   - 将 3D 卷积分解为 2 个 2D 卷积相加
   - 兼容不支持 3D 卷积的后端（CPU, Vulkan 等）
   - 数学等价，无精度损失
   - 降低实现复杂度

3. **CPU SIMD 深度优化**
   - AVX-512: 512位向量操作，单次处理 16 个 float32
   - AVX2: 256位向量操作
   - ARM NEON: 移动端高效计算
   - 量化/反量化操作全部 SIMD 化
   - 矩阵乘法分块优化缓存命中率

4. **计算图调度系统**
   - `ggml_backend_sched`: 自动为每个算子选择最优后端
   - 同一推理可混合使用 CPU + GPU
   - 图级节点融合减少中间结果内存
   - 零拷贝张量传递

5. **纯 C/C++ 零依赖**
   - 无 Python 解释器开销
   - 最小化系统依赖
   - 二进制可直接部署
   - 适合嵌入式/边缘场景

---

## 8. 对竞赛优化的启示与建议

### 8.1 竞赛约束回顾

- **环境**：单卡 A800 80GB, Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8
- **限制**：只能修改 `evaluation_wrapper.py`，通过 monkey patch 优化模型
- **评分**：`0.4 * Accuracy + 0.3 * TTFT_Improvement + 0.3 * Throughput_Improvement`
- **基线**：TTFT ~80ms, Throughput ~55 tokens/sec

### 8.2 可借鉴的优化方向（按优先级排序）

#### 高优先级（易实现，收益显著）

| 优化 | 参考引擎 | 实现方式 | 预期效果 |
|------|---------|---------|---------|
| **Flash Attention 启用** | SGLang/vLLM | `torch.backends.cuda.enable_flash_sdp(True)` 或使用 `flash_attn` 库 | TTFT 降低 20-40% |
| **FP16 + INT8 混合量化** | 三者 | BitsAndBytes `load_in_8bit` 或自定义 W8A8 | 吞吐量提升 30-50% |
| **torch.compile** | vLLM | `torch.compile(model, mode="reduce-overhead")` | 算子融合，减少 kernel launch |
| **KV Cache 配置优化** | SGLang | `use_cache=True` + StaticCache 预分配 | 减少内存碎片，稳定延迟 |

#### 中优先级（中等难度，收益可观）

| 优化 | 参考引擎 | 实现方式 | 预期效果 |
|------|---------|---------|---------|
| **Vision Encoder CUDA Graph** | SGLang | 捕获 ViT 前向传播图，按 seqlen 分桶 | Prefill 加速 15-30% |
| **算子替换（LayerNorm/RMSNorm）** | SGLang/llama.cpp | Triton kernel 替换 torch RMSNorm | 减少 kernel 开销 |
| **Attention 融合（QKV 投影）** | vLLM | 将 Q/K/V 三个投影合并为一个矩阵乘法 | 减少 GEMM 调用次数 |
| **FP8 量化** | SGLang/vLLM | `torch.float8_e4m3fn` 推理 | 在 A800 上带宽效率翻倍 |

#### 低优先级（高难度，边际收益）

| 优化 | 参考引擎 | 实现方式 | 预期效果 |
|------|---------|---------|---------|
| **自定义 Triton kernel** | SGLang | 编写 Fused Attention/FFN kernel | 极致优化 |
| **Conv3D 分解优化** | llama.cpp | 2x Conv2D 替代 Conv3D | 兼容性和效率 |
| **DeepStack 优化** | 三者 | 中间特征缓存和融合优化 | 跨模态加速 |
| **KV Cache 量化** | SGLang | INT8 KV Cache | 内存节省 50% |

### 8.3 推荐优化实施路径

```
Phase 1: 基础优化（最大 ROI）
├── 1.1 启用 Flash Attention (SDPA 或 flash_attn 库)
├── 1.2 torch.compile 编译模型
├── 1.3 静态 KV Cache 预分配
└── 1.4 确认 FP16 精度正确

Phase 2: 量化优化
├── 2.1 INT8 量化 (BitsAndBytes load_in_8bit)
├── 2.2 验证准确率不降
├── 2.3 若 A800 支持, 尝试 FP8
└── 2.4 混合精度策略 (Vision FP16 + LLM INT8)

Phase 3: 算子级优化
├── 3.1 Triton RMSNorm kernel
├── 3.2 融合 QKV 投影
├── 3.3 Vision Encoder CUDA Graph
└── 3.4 Patch Embedding 优化

Phase 4: 系统级优化
├── 4.1 内存预分配策略
├── 4.2 CUDA stream 流水线
├── 4.3 CPU-GPU 重叠计算
└── 4.4 缓存预热策略
```

### 8.4 关键注意事项

1. **准确率优先**：评分公式中准确率权重为 0.4，量化时必须验证 VQA 准确率不显著下降
2. **Vision Encoder 谨慎量化**：三个引擎都倾向保持 Vision Encoder 较高精度（FP16），仅量化语言模型
3. **位置编码不量化**：llama.cpp 明确将位置嵌入保持 F32，这对 M-RoPE 精度至关重要
4. **A800 特性利用**：
   - 算力 8.0 (sm_80)，支持 FP16 Tensor Core
   - 80GB HBM，2TB/s 带宽
   - 原生支持 Flash Attention 2/3
   - 支持 FP8 Tensor Core 操作
5. **预编译 .so 提交**：若使用自定义 CUDA kernel，需在 CUDA 12.8 环境预编译

---

## 9. 关键源码索引

### 9.1 SGLang

| 组件 | 文件路径 | 关键类/函数 |
|------|---------|-----------|
| 模型配置 | `sglang/python/sglang/srt/configs/qwen3_vl.py` | `Qwen3VLConfig`, `Qwen3VLVisionConfig` |
| 模型实现 | `sglang/python/sglang/srt/models/qwen3_vl.py` | `Qwen3VLForConditionalGeneration` |
| MoE 变体 | `sglang/python/sglang/srt/models/qwen3_vl_moe.py` | `Qwen3VLMoeForConditionalGeneration` |
| 语言模型 | `sglang/python/sglang/srt/models/qwen3.py` | `Qwen3Attention`, `Qwen3MLP` |
| Vision Attention | `sglang/python/sglang/srt/layers/attention/vision.py` | `VisionFlash3Attention`, `VisionSdpaAttention` |
| 图像处理器 | `sglang/python/sglang/srt/multimodal/processors/qwen_vl.py` | `QwenVLImageProcessor` |
| CUDA Graph | `sglang/python/sglang/srt/multimodal/vit_cuda_graph_runner.py` | `ViTCudaGraphRunner` |
| 多模态融合 | `sglang/python/sglang/srt/models/mm_utils.py` | `general_mm_embed_routine()` |
| KV Cache 量化 | `sglang/python/sglang/srt/layers/quantization/kv_cache.py` | - |
| 调度器 | `sglang/python/sglang/srt/managers/schedule_batch.py` | - |

### 9.2 vLLM

| 组件 | 文件路径 | 关键类/函数 |
|------|---------|-----------|
| 模型主体 | `vllm/vllm/model_executor/models/qwen3_vl.py` | `Qwen3VLForConditionalGeneration` |
| 基类 (2.5) | `vllm/vllm/model_executor/models/qwen2_5_vl.py` | `Qwen2_5_VisionAttention`, `Qwen2_5_VisionTransformer` |
| 模型注册 | `vllm/vllm/model_executor/models/registry.py` | `_MULTIMODAL_MODELS` |
| 视觉工具 | `vllm/vllm/model_executor/models/vision.py` | `get_vit_attn_backend`, `run_dp_sharded_mrope_vision_model` |
| Attention 后端 | `vllm/vllm/attention/backends/registry.py` | `AttentionBackendEnum` |
| Attention 层 | `vllm/vllm/attention/layer.py` | `check_upstream_fa_availability` |
| Attention Wrapper | `vllm/vllm/attention/ops/vit_attn_wrappers.py` | `vit_flash_attn_wrapper`, `vit_torch_sdpa_wrapper` |
| KV Cache | `vllm/vllm/v1/kv_cache_interface.py` | `KVCacheSpec`, `FullAttentionSpec` |

### 9.3 llama.cpp

| 组件 | 文件路径 | 关键类/函数 |
|------|---------|-----------|
| 模型转换 | `llama.cpp/convert_hf_to_gguf.py` | `Qwen3VLVisionModel`, `Qwen3VLTextModel` |
| Vision 编码器 | `llama.cpp/tools/mtmd/models/qwen3vl.cpp` | `clip_graph_qwen3vl::build()` |
| 文本模型 | `llama.cpp/src/models/qwen3vl.cpp` | `llm_build_qwen3vl` |
| CLIP 核心 | `llama.cpp/tools/mtmd/clip.cpp` | `clip_image_encode()`, `clip_image_preprocess()` |
| 模型结构 | `llama.cpp/tools/mtmd/clip-model.h` | `clip_model`, `clip_layer` |
| 图基类 | `llama.cpp/tools/mtmd/clip-graph.h` | `clip_graph` |
| Flash Attention | `llama.cpp/ggml/src/ggml-cpu/ops.cpp` | `ggml_compute_forward_flash_attn_ext_f16` |
| 量化 | `llama.cpp/ggml/src/ggml-quants.c` | `quantize_row_q4_0_ref` 等 |
| CUDA FA | `llama.cpp/ggml/src/ggml-cuda/fattn.cu` | CUDA Flash Attention kernel |
| x86 SIMD | `llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` | AVX-512/AVX2 量化 |
| ARM SIMD | `llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c` | NEON 量化 |

---

## 附录：三引擎技术栈总览

```
               SGLang                  vLLM                   llama.cpp
              --------                ------                 -----------
语言        Python + CUDA           Python + CUDA           C/C++ + CUDA
模型格式    PyTorch (HF)            PyTorch (HF)            GGUF
核心优化    RadixAttention          PagedAttention          GGML 量化推理
              + CUDA Graph            + torch.compile        + SIMD 优化
              + FA3/FA4               + FA2+                 + 计算图调度
              + Continuous Batch      + Continuous Batch     + 多后端混合

适用场景    服务端高并发              服务端高并发              边缘/嵌入式
            多用户请求               多用户请求               单请求低延迟
            多 GPU 集群              多 GPU 集群              资源受限设备

成熟度      生产就绪                 生产就绪（行业标准）      生产就绪
社区活跃    高                       最高                     高
```

---

*报告结束。如需深入某个具体优化方向的实现细节，可进一步查阅上述源码索引中的文件。*
