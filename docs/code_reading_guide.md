# 代码阅读指南 — 提交包中评测时实际运行的代码

本文档帮助你理解 `create_submission.sh` 打包的代码在评测环境中是如何被调用和执行的。

---

## 1. 评测入口：benchmark.py 如何调用我们的代码

评测环境使用**官方 `benchmark.py`**（ZIP 中的 benchmark.py 会被忽略），其核心流程：

```
benchmark.py
  └─ from evaluation_wrapper import VLMModel
  └─ vlm = VLMModel(model_path, device)
  └─ 访问 vlm.processor / vlm.model / vlm.device
  └─ 对每个样本：
       inputs = vlm.processor.apply_chat_template(...)  # 预处理
       inputs = inputs.to(device)
       output_ids = vlm.model.generate(**inputs, ...)    # 推理
       # → 测量 TTFT（首 token 时间）和 Throughput（吞吐量）
```

关键接口：
- `vlm.processor` → 返回 `AutoProcessor`
- `vlm.model` → 返回 `ModelWrapper`（透明包装，非原始 HF 模型）
- `vlm.model.generate(...)` → 实际的推理入口

---

## 2. 提交包文件清单与作用

`create_submission.sh` 打包的文件：

```
submission.zip
├── evaluation_wrapper.py          # [核心] VLMModel 类，所有优化入口
├── optimized_layer.py             # [核心] 融合的 decoder layer + Triton kernels
├── cuda_decode_gemv.py            # CUDA GEMV 的兼容 facade
├── cuda_decode_gemv_ops/          # 按算子族拆分后的 GEMV Python 包
├── cuda_decode_attention.py       # CUDA decode attention 的 Python 加载器
├── requirements.txt               # 依赖声明
├── cuda_extensions/               # CUDA kernel 源码 + 预编译 .so
│   ├── cuda_decode_gemv_ext*.so   # 预编译的 GEMV CUDA extension
│   ├── cuda_decode_attention_ext*.so  # 预编译的 attention CUDA extension
│   ├── decode_gemv_kernel.cu      # GEMV kernel 薄入口
│   ├── decode_gemv_kernel_*.cuh   # GEMV kernel 分片（common/matvec/fused/xwarp/launchers）
│   ├── decode_gemv_ext.cpp        # GEMV pybind11 绑定
│   ├── decode_attention_kernel.cu # attention kernel 源码
│   ├── decode_attention_ext.cpp   # attention pybind11 绑定
│   ├── setup.py                   # attention extension 构建脚本
│   └── setup_decode_gemv.py       # GEMV extension 构建脚本
└── (my_kernel/ 如存在)             # 其他自定义算子
```

---

## 3. evaluation_wrapper.py 阅读路线

这是最核心的文件，建议按以下顺序阅读：

### 3.1 VLMModel.__init__（第 1053-1095 行）— 模型加载与优化安装

```
VLMModel.__init__(model_path, device)
  ├── _configure_cuda()              # TF32/FlashSDP/MemEfficientSDP 配置
  ├── AutoProcessor.from_pretrained  # 加载 tokenizer + image processor
  ├── AutoModelForImageTextToText    # 加载 Qwen3-VL-2B, FP16, SDPA backend
  ├── _optimize_kv_cache()           # use_cache=True, pad_token_id 修正
  ├── _replace_decoder_layers()      # → 调用 optimized_layer.replace_decoder_layers()
  ├── _optimize_prefill_path()       # 安装 prefill 阶段各种 fast path
  └── ModelWrapper(model)            # 包装模型，注入自定义 decode 引擎
```

### 3.2 ModelWrapper.generate（第 1002-1047 行）— 推理调度

```
ModelWrapper.generate(*args, **kwargs)
  ├── 判断是否启用自定义 decode (ENABLE_CUSTOM_DECODE=1)
  ├── 判断请求是否满足 CUDAGraph 条件
  ├── 路径 A: GreedyDecodeEngine.generate()  ← 绝大多数情况
  └── 路径 B: model.generate() (HF 原生 fallback)
```

### 3.3 GreedyDecodeEngine.generate（第 130-295 行）— 核心推理循环

这是吞吐量优化的核心，阅读重点：

```
GreedyDecodeEngine.generate()
  ├── [Prefill 阶段]
  │   ├── 准备 attention_mask, StaticCache
  │   ├── 获取或创建 CUDAGraph runner (按 bucket 复用)
  │   └── model(input_ids, pixel_values, ..., past_key_values=StaticCache)
  │       → 首次前向，输出 rope_deltas + 第一个 token
  │
  ├── [CUDAGraph 捕获]
  │   └── graph_runner.capture_if_needed()
  │       → 在 side stream 上 warmup → 在 default stream 上 capture
  │
  └── [Decode 循环] (两种模式)
      ├── 异步 EOS 检测模式 (_generate_with_async_eos_poll)
      │   → graph_runner.replay() + stage_host_token() + poll_staged_tokens()
      │   → EOS token 通过 pin_memory + D2H copy + event 异步检测
      └── 同步模式
          → 逐 token: graph_runner.replay() → argmax → EOS 检查
```

### 3.4 DecodeCudaGraphRunner（第 727-969 行）— CUDAGraph 生命周期

```
DecodeCudaGraphRunner
  ├── __init__: 分配所有静态 buffer (input_ids, position_ids, logits, cache, mask...)
  ├── prebuild_graph(): 模型初始化时预建（32 个 bucket，64-2048 步长）
  ├── prepare_for_request(): 每个请求复用 runner，重置 cache/mask
  ├── capture_if_needed(): warmup + torch.cuda.graph() capture
  ├── replay(): 写入 inputs → graph.replay() → 读取 next_token
  └── stage_host_token() / poll_staged_tokens(): 异步 EOS 检测
```

---

## 4. optimized_layer.py 阅读路线

### 4.1 OptimizedDecoderLayer（第 21-335 行）— 融合的 decoder layer

每个 Qwen3VL decoder layer 被替换为此类，forward 流程：

```
OptimizedDecoderLayer.forward(hidden_states, position_embeddings, ...)
  1. Input RMSNorm          → _rms_norm() (Triton/F.rms_norm 自适应)
  2. Fused QKV projection   → 1 GEMM (原先 3 个) → nosplit GEMV 或 F.linear
  3. QK-Norm + RoPE         → _fused_qk_rms_norm_rope() (Triton 融合)
  4. KV Cache update        → _static_cache_single_token_update() (Triton 单 kernel)
  5. Attention              → _small_decode_attention() (split-KV Triton / dot-style)
  6. O_proj                 → nosplit GEMV 或 Triton GEMV 或 F.linear
  7. Residual + Post-attn RMSNorm  → _fused_add_rms_norm() (Triton 融合)
  8. Fused gate+up + SiLU   → 1 GEMM (原先 2 个) + nosplit GEMV
  9. Down_proj              → nosplit GEMV
  10. Residual add
```

### 4.2 关键 kernel 函数索引

| 函数 | 说明 | 何时生效 |
|------|------|----------|
| `_rms_norm()` | RMSNorm: 大 M 用 Triton, decode 用 F.rms_norm 或 Triton small | 每层 2 次 |
| `_fused_rms_norm_rope()` | RMSNorm + RoPE 融合 | Q/K 各 1 次 |
| `_fused_qk_rms_norm_rope()` | Q+K 的 norm+rope 合并为 1 次 launch | decode 时 |
| `_fused_add_rms_norm()` | residual add + RMSNorm 融合 | 每层 1 次 |
| `_fused_silu_mul()` | SiLU(gate)*up 融合 | MLP 激活 |
| `_small_decode_attention()` | 自定义 decode attention 调度 | decode 每步每层 |
| `_static_cache_single_token_update()` | 单 token KV 写入融合 | decode 每步每层 |
| `_cuda_nosplit_gemv()` | CUDA GEMV (替代 cuBLAS M=1) | QKV/o_proj/gate_up/down_proj |
| `_triton_decode_gemv()` | Triton GEMV (fallback) | nosplit 不可用时 |

### 4.3 Decode Attention 调度链

```
_small_decode_attention()
  ├── 优先: split-KV (持久 workspace, 多 split 并行)
  │         _small_decode_attention_grouped_prefix_cachepos_splitkv()
  ├── 次选: dot-style (tl.dot 矩阵块操作)
  │         _small_decode_attention_grouped_prefix_cachepos_dot()
  ├── 再选: prefix+cachepos (基于 cache_position 的 prefix 长度)
  │         _small_decode_attention_grouped_prefix_cachepos()
  ├── 再选: grouped (GQA 复用 K/V)
  │         _small_decode_attention_grouped()
  └── 最后: per-head
          _small_decode_attention_per_head()
```

---

## 5. CUDA Extensions 阅读路线

### 5.1 cuda_decode_gemv.py / cuda_decode_gemv_ops/ — GEMV 加载器

- `cuda_decode_gemv.py` 只保留兼容入口，真正实现拆到 `cuda_decode_gemv_ops/{runtime,matvec,fused,profile}.py`
- `runtime.py` 负责自动查找预编译 `.so`，如果过时则重新编译
- 核心函数: `nosplit_matvec_half()` — 评测时默认使用的 GEMV
  - 对 down_proj (2048x6144) 自动路由到 `k6144_n8k1024_xwarp_w_async` 变体
- 融合函数: `fused_add_rmsnorm_gateup_half()`, `fused_silu_downproj_half()`, `fused_gateup_silu_downproj_half()` 等

### 5.2 cuda_decode_attention.py — Attention 加载器

- `grouped_prefix_cachepos_splitkv()` — split-KV attention 的 CUDA 实现

### 5.3 CUDA kernel 源码

- `decode_gemv_kernel.cu`: 仅保留编译入口，按顺序 `#include` 各个 GEMV 子模块
- `decode_gemv_kernel_*.cuh`: nosplit GEMV 实现分片，覆盖 k2048/k6144、row2/row4、async、xwarp、launchers 等
- `decode_attention_kernel.cu`: split-KV decode attention kernel

---

## 6. Prefill 优化路径 (_optimize_prefill_path)

在 `evaluation_wrapper.py` 第 1126-1983 行，通过 monkey-patch 安装多个 prefill fast path：

```
_optimize_prefill_path()
  ├── prefill:input_meta_cache
  │   → processor.apply_chat_template 拦截，提前提取 grid/segment 元数据
  │
  ├── prefill:vision_grid_cache
  │   → 缓存 vision encoder 的 position_embeddings, cu_seqlens, split_sizes
  │   → 缓存 fast_pos_embed_interpolate, rot_pos_emb
  │   → patch_embed 改为 F.linear (避免 Conv2d reshape)
  │   → vision attention 使用预计算的 split_sizes 做 per-frame SDPA
  │
  ├── prefill:mm_slice_fastpath
  │   → 跳过 HF 的通用多模态合并，直接 slice copy image_embeds
  │   → 跳过 create_causal_mask (全 prompt 无需 mask)
  │
  ├── prefill:text_fastpath
  │   → 复用 cache_position / rotary_emb 缓存
  │   → prefill attention 直接在有效 prompt 上做 causal SDPA
  │
  └── prefill:rope_fastpath
      → 缓存 RoPE position_ids 和 rope_deltas 计算结果
```

---

## 7. 环境变量开关速查

评测时生效的关键环境变量（默认值即评测配置）：

| 环境变量 | 默认 | 作用 |
|----------|------|------|
| `ENABLE_CUSTOM_DECODE` | 1 | 启用自定义 greedy decode 引擎 |
| `ENABLE_CUDAGRAPH` | 1 | 启用 CUDAGraph capture/replay |
| `ENABLE_CUDA_NOSPLIT_GEMV` | 1 | 启用 CUDA nosplit GEMV |
| `ENABLE_TRITON_SMALL_RMSNORM` | 1 | decode 用 Triton small RMSNorm |
| `ENABLE_TRITON_SMALL_ADD_RMSNORM` | 1 | decode 用 Triton fused add+RMSNorm |
| `ENABLE_TRITON_SMALL_QK_NORM_ROPE` | 1 | decode 用 Triton fused QK-norm+RoPE |
| `ENABLE_TRITON_SMALL_DECODE_ATTENTION` | 1 | 启用自定义 decode attention |
| `ENABLE_TRITON_SMALL_DECODE_ATTENTION_DOT` | 1 | 启用 dot-style attention |
| `ENABLE_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV` | 1 | 启用 split-KV attention |
| `ENABLE_TRITON_SMALL_DECODE_STATIC_KV` | 1 | Triton 单 token KV cache 写入 |
| `ENABLE_TRITON_SMALL_DECODE_GEMV` | 1 | Triton GEMV (nosplit 的 fallback) |
| `ENABLE_PREFILL_VISION_GRID_CACHE` | 1 | prefill vision grid 缓存 |
| `ENABLE_PREFILL_ROPE_FASTPATH` | 1 | prefill RoPE 缓存 |
| `ENABLE_PREFILL_PATCH_EMBED_LINEAR` | 1 | patch_embed Conv2d → F.linear |
| `ENABLE_PREFILL_MM_SLICE_FASTPATH` | 1 | 多模态 slice 快速路径 |
| `ENABLE_PREFILL_TEXT_FASTPATH` | 1 | text prefill 快速路径 |
| `CUDAGRAPH_PREBUILD_MAX_CACHE_LEN` | 2048 | 预建 graph 的最大 cache 长度 |
| `CUDAGRAPH_BUCKET_SIZE` | 64 | graph runner 按 64 粒度复用 |

---

## 8. 推荐阅读顺序

1. **先读 `create_submission.sh`** — 了解哪些文件进入提交包
2. **再读 `VLMModel.__init__`** — 了解优化安装入口
3. **再读 `ModelWrapper.generate` → `GreedyDecodeEngine.generate`** — 了解推理主循环
4. **再读 `OptimizedDecoderLayer.forward`** — 了解单层融合逻辑
5. **按需深入** Triton kernel / CUDA extension / prefill fast path
