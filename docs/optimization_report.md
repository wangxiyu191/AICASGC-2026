# AICAS 2026 — 面向 AI 芯片的 VLM 高效推理与优化

## 技术报告

---

## 摘要

本文总结了我们对 **Qwen3-VL-2B-Instruct** 在 **单张 NVIDIA A800 80GB PCIe** 上的推理优化工作。优化严格遵守赛题约束：**不做量化、稀疏化、蒸馏，也不修改任何模型参数或结构**。最终官方 5000 样本评测成绩为 **TTFT 36.66 ms、吞吐量 284.48 tok/s、准确率 0.6956、综合得分 2.3765**，相对官方 Baseline 的 TTFT ~80 ms、吞吐量 ~55 tok/s，分别提升 **54.2%** 和 **417.2%**。

从工程视角看，最终成绩主要来自三类关键优化：

- **Decode 执行流重构**：自定义 greedy decode + `StaticCache` + `CUDAGraph`，把单 token decode 的 Python / launch 开销压到最低。
- **Decoder layer 小算子与 GEMV/attention 内核替换**：包括 fused QKV/gate_up 权重拼接、CUDA nosplit GEMV、CUDA v4 q-head split-KV attention、fused RMSNorm+QKV、fused Q-RoPE+KV cache、persistent post-attn MLP。
- **Prefill 快速路径**：单图输入的跨模态 slice fastpath、vision geometry cache、vision RoPE 融合、text prefill no-mask fastpath、layer fastcall、prefill Triton 融合 kernel，以及 RoPE/position 相关缓存。

---

## 1. 研究背景与问题定义

### 1.1 赛题概述

AICAS 2026「面向 AI 芯片的 VLM 高效推理与优化」赛道要求选手对 **Qwen3-VL-2B-Instruct** 做端到端推理加速。评测模式为 **batch_size=1**，每个样本包含一张图片和一个自然语言问题，模型生成不超过 128 个 token 的回答。评测公式为：

$$
\text{Score}
= 0.4 \times \text{Accuracy}
+ 0.3 \times \frac{\text{Baseline\_TTFT} - \text{TTFT}}{\text{Baseline\_TTFT}}
+ 0.3 \times \frac{\text{Throughput} - \text{Baseline\_Throughput}}{\text{Baseline\_Throughput}}
$$

其中：

- **TTFT** 覆盖从输入准备、图像编码、跨模态融合、prefill，到输出首 token 的全部延迟。
- **Throughput** 是端到端 token 生成速率（tok/s）。
- **Accuracy** 是 5000 个样本上的 VQA 准确率。

### 1.2 约束条件

- **禁止量化、稀疏化、蒸馏或修改模型参数/架构**
- **禁止外部 API / 网络调用**
- **禁止预计算答案、硬编码输出或任何针对 benchmark 顺序的违规复用**
- 评测环境无外网，CUDA 扩展需随提交包一并提供
- 评测入口固定为官方 `benchmark.py`，优化只能通过 `evaluation_wrapper.py` 及其导入模块生效

### 1.3 硬件与软件环境

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA A800 80GB PCIe (SM80, HBM2e 2039 GB/s) |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA Toolkit | 12.8 (nvcc 12.8.61) |
| Flash Attention | 2.8.3（本地已预装） |
| 模型 | Qwen3-VL-2B-Instruct |
| 精度 | FP16 |
| Batch size | 1 |
| 最大生成长度 | 128 tokens |

---

## 2. 相关工作

我们的工程方案主要借鉴了以下实践：

- **vLLM / SGLang**：为 `StaticCache`、decode 引擎重构、`CUDAGraph` 预建与复用提供了系统设计启发。
- **FlashAttention / FlashInfer**：其 split-KV、online softmax merge 等思想直接影响了我们 decode attention 的 Triton/CUDA kernel 设计。
- **Triton**：用于 prefill 阶段大量小算子和 kernel-count 压缩的快速原型开发。
- **CUTLASS / cuBLAS**：作为 GEMM/GEMV 的基线对照，帮助我们定位 `M=1` decode 场景下 cuBLAS 的低效点。

---

## 3. 方法设计

整体上，我们把优化划分为三类：

- **Decode 侧优化**：决定 steady-state throughput，是最终收益的主体。
- **Prefill 侧优化**：决定 TTFT，是近阶段 TTFT 改善的主要来源。
- **系统级与执行流优化**：确保上面两类优化能在 batch=1 的真实 benchmark 路径中稳定生效。

提交版本的实际执行结构如下：

```text
benchmark.py (官方, 不可修改)
  └── from evaluation_wrapper import VLMModel
      └── VLMModel.__init__
          ├── _configure_cuda()
          ├── AutoProcessor / AutoModel
          ├── _optimize_kv_cache()
          ├── _replace_decoder_layers()
          ├── _optimize_prefill_path()
          ├── _optimize_lm_head()
          └── ModelWrapper(model)

      vlm.model.generate(...)
        └── ModelWrapper.generate(...)
            ├── 路径 A: GreedyDecodeEngine.generate()   # 优先路径
            └── 路径 B: HF model.generate()             # 不满足条件时 fallback
```

### 3.1 Decode 侧优化

#### 3.1.1 自定义 Greedy Decode + StaticCache + CUDAGraph

官方 `model.generate()` 在 batch=1、单 token decode 场景下的主要瓶颈在于：

- Python dispatch 多
- 每步 kernel launch 多
- `DynamicCache` 带来额外的 shape / storage 变化
- 首 token 之后的 decode 仍走 HuggingFace 通用路径

因此我们实现了自定义 `GreedyDecodeEngine`，核心设计是：

- **`StaticCache` 替代 `DynamicCache`**：decode 阶段 KV cache 形状固定，只更新内容，不改变布局。
- **按 cache 长度 bucket 化复用 graph runner**：bucket 粒度为 64，默认预建到 2048。
- **初始化阶段直接预建 32 个 CUDAGraph bucket**：避免首请求 lazy capture。
- **decode step 整体 graph capture**：把 decoder layers、attention、MLP、lm_head、argmax 纳入一条静态 graph。
- **异步 EOS 检测**：通过 pinned host ring + D2H async copy + event 轮询，把 EOS 检测从主计算流里拆出去。

这一方向的收益可以从阶段性 benchmark 与官方评测中直接观察到：

- A800 上首次让 CUDAGraph 正常运行后，吞吐量达到 **75.53 tok/s**
- graph runner bucket 复用后提升到 **95.68 tok/s**
- 初始化时预建全部 bucket 后，进一步提升到 **102.51 tok/s**

也就是说，仅 decode 执行流重构本身，就把吞吐量从早期的 ~45-76 tok/s 区间拉到了 **100 tok/s 级别**。

#### 3.1.2 Decoder Layer 的实际执行链

提交版本采用的 decode 单层执行链如下：

```text
input hidden
  → fused_rmsnorm_qkv                 # CUDA，默认开
  → q / k / v split
  → fused_q_rope_kv_cache             # CUDA，默认开
    or qk_norm_rope + static_kv       # Triton fallback
  → CUDA v4 q-head split-KV attention # 优先路径
    or Triton split-KV / dot fallback
  → o_proj via nosplit GEMV
  → fused post_attn_mlp persistent    # CUDA persistent，默认开
    or gate_up+silu+down_proj fused fallback
  → residual output
```

这条链路里，最终性能取决于**一组稳定、可 graph-capture、对 batch=1 形状高度特化的小链路融合**的协同效果。

#### 3.1.3 最终采用的 Decode 优化

| 优化项 | 采用情况 | 原理 | 证据口径 |
|--------|----------|------|----------|
| 自定义 greedy decode + `StaticCache` | 最终采用 | 去掉 HF 通用 decode 控制流和动态 cache 管理 | 自定义 decode 与 CUDAGraph 建设阶段的多轮 benchmark |
| `CUDAGraph` bucket 复用 + 预建 | 最终采用 | 让 decode step 走静态 graph，减少 Python 与 launch 开销 | A800 上 CUDAGraph 成功到 graph pool 预建的阶段性 benchmark |
| Fused QKV / gate_up 权重拼接 | 最终采用 | 3 个投影变 1 个 QKV，2 个投影变 1 个 gate_up | 早期 decoder layer 融合阶段 benchmark |
| CUDA nosplit GEMV | 最终采用 | 用专用 matvec 替代 cuBLAS 在 `M=1` 场景的低效 GEMM 路由 | A800 上 ON/OFF 对照 benchmark |
| CUDA fused RMSNorm + QKV | 最终采用 | 把 input RMSNorm 和 QKV matvec 融成一段 decode 小算子 | 提交版本默认链路 |
| CUDA fused Q-RoPE + KV cache | 最终采用 | 单 token decode 中把 Q 的 RoPE、K/V 规范化与 cache 写入压缩到更少 launch | v3 与 v4 官方评测对比 |
| CUDA v4 q-head split-KV attention | 最终采用 | 利用 16Q:8KV 的 GQA 结构，在 KV split 之外再切分 Q-head 并行度 | A800 本地 10-sample benchmark 与最终官方评测 |
| Persistent post-attn MLP 融合 | 最终采用 | 将 `post_attn add+rmsnorm → gate_up → silu → down_proj → residual_add` 合并为 persistent kernel | A800 本地 benchmark 对照 |
| Fast lm_head GEMV | 最终采用 | `lm_head` 的单 token路径走 nosplit GEMV；argmax 仍用 `torch.argmax`，但直接在 graph 内对 `outputs.logits` 取值 | 代码默认开，单 token 路径为 fast GEMV + graph 内 argmax |

提交版本中，decode attention 优先采用 **CUDA v4 q-head split-KV**，Triton split-KV / dot-style 作为 fallback 保留。`_optimize_lm_head()` 对应的是单 token `lm_head` 的 GEMV 加速，最终取值仍由 `torch.argmax` 完成。

#### 3.1.4 Decode 侧收益总结

从最终结果看，decode 侧收益主要来自三类因素：首先是 `StaticCache`、`CUDAGraph` bucket 复用与预建所带来的执行流重构；其次是针对 `M=1` 形态的 CUDA nosplit GEMV 替换；最后是 fused Q-RoPE + KV cache、q-head split-KV attention、persistent post-attn MLP 等小算子链路压缩与融合。对应的 A/B 数据与阶段性收益拆解统一放在第 5.4 节。

#### 3.1.5 最终版本中未启用的 Decode 优化

这部分代码仍保留在仓库中，但**不属于最终提交配置**：

| 路径 | 状态 | 说明 |
|------|----------|------|
| prompt-prefix reuse | 已删除 / 禁用 | 该历史方案已被主办方明确禁止，不属于最终提交实现 |
| full-layer softsync | 未启用，且早期异常记录不作为最终成绩说明依据 | 历史版本曾出现 `down_proj` row coverage bug，因此相关高分观测值未用于最终结果说明 |
| o_proj+post_attn MLP softsync/persistent | 实验保留，未启用 | 代码保留，但不属于最终提交配置 |
| partition-aware GEMV | 未启用 | 代码已实现，但 `ENABLE_PARTITION_GEMV=0`，不属于最终提交实现 |
| persistent attention / o_proj prefetch | 未启用 | 仅用于实验验证，未纳入最终提交配置 |

---

### 3.2 Prefill 侧优化

和 decode 相比，prefill 的关键不只是算子本身，而是：

- 视觉侧和文本侧是否重复做了不必要的准备工作
- 是否引入了 host/device 同步
- 是否产生了大量 DirectCopy / chunk copy / 小 elementwise kernel

因此我们把 prefill 优化拆成三部分：**视觉侧缓存与 fastpath、跨模态与 text prefill fastpath、prefill 小算子融合**。

#### 3.2.1 Vision Encoder：缓存几何信息，减少 per-image 重复工作

提交版本实际采用的视觉侧优化包括：

- **grid/geometry cache**：缓存 `fast_pos_embed_interpolate`、`rot_pos_emb`、`position_embeddings`、`cu_seqlens`、attention/pooler `split_sizes`。
- **patch_embed 改写为等价 `F.linear`**：在输入形状满足条件时，绕开 `Conv3d` 路径中的额外 reshape/调度成本。
- **vision attention 按 frame split 做 SDPA**：减少通用路径里的额外处理。
- **fused vision RoPE**：将 `apply_rotary_pos_emb_vision` 的一长串 PyTorch kernel 融成单个 Triton kernel。

这部分的关键收益在于**大幅削减了视觉侧 prefill 的 kernel 数和重复几何准备**。

#### 3.2.2 单图输入的跨模态优化

针对单图、单轮图文问答这一输入结构，我们实现了对应的跨模态数据通路优化：

- 识别 `prefix + image tokens + suffix` 的单图结构
- 提前计算 `position_ids`，避免 HF 原始路径里在 vision 之后才去做 `tolist()`/CPU 侧逻辑
- 直接把 `image_embeds` slice-copy 到 `inputs_embeds` 对应区间
- 避免通用多模态 merge 逻辑里的逐 token 分类和不必要拷贝

这里有一个很实际的收益点：在 `evaluation_wrapper.py` 里我们把 `compute_3d_position_ids()` 提前到 vision encoder 之前，是为了**把一次隐式 GPU→CPU 同步挪到 700+ 个 vision kernels 之前**。代码注释中记录，这样可以消除 vision 与 text decoder 之间约 **3.5 ms** 的空洞。

#### 3.2.3 Text Prefill No-Mask Fastpath

对于首次 prefill，我们建立了和 decode 分离的文本快路径：

- **跳过首次 prefill 的显式 causal mask 构造**
- **缓存 `cache_position` 和 rotary embedding**
- **prefill attention 直接在有效 prompt K/V 上做 causal SDPA，再写回 `StaticCache`**
- **`PrefillFastpathContext` + layer fastcall**：每层直接走优化过的 `prefill_forward_no_mask`，减少 HF 层调用框架的参数拆装

这条路径对应的正式 10-sample benchmark 把 TTFT 从 **61.11 ms** 降到 **43.49 ms**；后续又通过显式 layer fastcall 与共享上下文继续压缩了 per-layer Python 调度。

#### 3.2.4 Prefill Triton Kernel 套件

后续针对 prefill 又做了四类 kernel-count 压缩：

- **fused QK-norm + RoPE**：同时修复了 RoPE 精度和非连续 tensor 读取问题
- **packed SiLU*mul**：直接从 packed gate_up 张量读两半，去掉 `chunk()` 相关 copy kernel
- **fused add + RMSNorm**
- **vision RoPE 融合 + 消除 text decoder DirectCopy**

这几项叠加后，TTFT 从 40ms+ 进一步压到 38ms 级别，并在最终官方线上成绩里进入 36ms 区间。

#### 3.2.5 Prefill 侧收益总结

Prefill 侧的关键收益集中在三部分：text prefill no-mask fastpath、显式 layer fastcall 与共享上下文带来的 Python 调度压缩，以及 fused QK-norm+RoPE、packed SiLU*mul、fused vision RoPE 与 DirectCopy 消除所带来的 kernel-count 压缩。最终 TTFT 的改善主要来自通用多模态路径中无效准备、隐式同步与大量小 kernel 的削减。对应的 A/B 数据与阶段性收益拆解统一放在第 5.4 节。

#### 3.2.6 最终版本中未启用的 Prefill 优化

| 路径 | 状态 | 说明 |
|------|----------|------|
| cuDNN SDPA prefill | 未启用 | 仅保留开关，未纳入最终提交实现 |
| contiguous StaticCache prefill writer | 未启用 | 作为特定实验保留 |
| 其他 benchmark 专用缓存复用 | 已清理 | 不再出现在最终提交实现中 |

---

### 3.3 系统级配置

| 配置项 | 设置 | 说明 |
|--------|----------|------|
| CUDA 模块加载 | `CUDA_MODULE_LOADING=EAGER` | 避免运行时 lazy module loading |
| TF32 | `allow_tf32=True` | 打开 Tensor Core 对应路径 |
| SDPA backend | FlashSDP + MemEfficientSDP | 允许 PyTorch 选择最优 attention backend |
| 模型 dtype | FP16 | 沿用官方提供的 FP16 配置，提交版本未对模型 dtype 做额外修改 |
| `use_cache` | `True` | decode / prefill 都依赖 `StaticCache` |

---

## 4. 实现细节

### 4.1 主要执行路径与核心模块

#### 4.1.1 主要执行路径

提交包在评测环境中的实际调用入口是官方 `benchmark.py`，它不会执行提交包内自带的 benchmark 脚本，而是直接导入 `evaluation_wrapper.py` 中的 `VLMModel`。主执行链可以概括为：

```text
official benchmark.py
  └── from evaluation_wrapper import VLMModel
      └── vlm = VLMModel(model_path, device)
          ├── vlm.processor  -> AutoProcessor
          ├── vlm.device     -> CUDA device
          └── vlm.model      -> ModelWrapper
               └── generate(...)
                    ├── 优先: GreedyDecodeEngine.generate()
                    └── 回退: HuggingFace model.generate()
```

因此，最终提交版本的主要优化集中在两条热路径上：

- **初始化与路径安装**：`evaluation_wrapper.py` 中的 `VLMModel.__init__()` 负责模型加载、decoder layer 替换、prefill fastpath 安装、`lm_head` 优化与执行环境配置。
- **单次请求的推理热路径**：`ModelWrapper.generate()` 优先进入 `GreedyDecodeEngine.generate()`，再由 `optimized_layer.py` 中的 `OptimizedDecoderLayer.forward()` 调度各类 decode / prefill kernel。

#### 4.1.2 核心模块与职责对应

为了把前文中的优化项与具体实现对应起来，提交版本中最关键的代码模块及其职责如下：

| 模块 | 代码位置 | 在最终执行中的作用 |
|------|----------|--------------------|
| 初始化与优化安装 | `evaluation_wrapper.py::VLMModel.__init__()` | 完成模型加载、环境配置、decoder layer 替换、prefill fastpath 安装与 `lm_head` 优化 |
| 推理入口分流 | `evaluation_wrapper.py::ModelWrapper.generate()` | 在自定义 decode 路径与 HuggingFace fallback 之间完成请求分流 |
| Decode 主循环 | `evaluation_wrapper.py::GreedyDecodeEngine.generate()` | 负责 prefill、graph runner 复用、decode 循环与异步 EOS 检测 |
| Graph 生命周期管理 | `evaluation_wrapper.py::DecodeCudaGraphRunner` | 负责 `CUDAGraph` 的静态 buffer、capture、replay 与 bucket 预建 |
| 单层执行调度 | `optimized_layer.py::OptimizedDecoderLayer.forward()` | 负责单层 decode/prefill 的实际执行链，以及各融合路径的启用条件 |
| 算子路由与 fallback | `optimized_layer.py` 中的 attention / GEMV / Triton 路由函数 | 负责 CUDA v4 q-head split-KV、nosplit GEMV、persistent post-attn MLP 等关键路径的实际调度 |
| Python loader 层 | `cuda_decode_attention.py`、`cuda_decode_small_ops.py`、`cuda_decode_gemv_ops/*.py` | 负责 extension 加载、形状分派与 Python 侧兼容封装 |
| CUDA 实现层 | `cuda_extensions/` | 负责最终 CUDA kernel、pybind11 绑定与扩展构建 |

#### 4.1.3 代码结构与规模

提交版本中与性能最相关的代码规模如下：

| 文件/目录 | 行数 | 主要职责 |
|-----------|------|----------|
| `evaluation_wrapper.py` | 2263 | 评测入口、`VLMModel`、`GreedyDecodeEngine`、`DecodeCudaGraphRunner`、prefill fastpath |
| `optimized_layer.py` | 3921 | `OptimizedDecoderLayer`、decode/prefill Triton 路由与 fallback 逻辑 |
| `cuda_extensions/` 中 `.cu/.cuh/.cpp` | 17718 | CUDA GEMV、persistent/grid-sync MLP、decode attention、small ops |
| `cuda_decode_gemv_ops/*.py` + loader | 4311 | GEMV runtime/route/profile/fused 包装层 |

### 4.2 最终版本启用的 Kernel 实现概述

本节仅列出最终版本在 benchmark 热路径中实际启用的**自定义 kernel**。`CUDAGraph`、cache bucket、slice fastpath、`PrefillFastpathContext` 等执行流与缓存优化不属于单独 GPU kernel，因此不在本节逐项展开。

#### 4.2.1 Decode 侧 Kernel

| Kernel / 算子 | 实现概述 | Python 入口 | 底层实现与代码位置 |
|---------------|----------|-------------|--------------------|
| `fused_rmsnorm_qkv_half` | 将单 token decode 的 input RMSNorm 与 QKV projection 合并，避免先做 RMSNorm 再单独启动 QKV GEMV。 | `optimized_layer.py::_fused_rmsnorm_qkv()` | Python 包装在 `cuda_decode_gemv_ops/fused.py`，PyTorch extension 入口在 `cuda_extensions/decode_gemv_ext.cpp`，核心 kernel 为 `cuda_extensions/decode_gemv_kernel_fused.cuh` 中的 `fused_rmsnorm_qkv_k2048_hfma2_fwd_kernel`。 |
| `fused_q_rope_kv_cache_update` | 在一条 CUDA kernel 中完成 Q/K 的 norm+RoPE，并把 K/V 直接写入 `StaticCache`，减少 decode 小算子链路中的 launch 与中间张量往返。 | `optimized_layer.py::_fused_q_rope_kv_cache_update()` | 扩展绑定在 `cuda_extensions/decode_small_ops_ext.cpp`，核心实现位于 `cuda_extensions/decode_small_ops_kernel.cu` 的 `fused_q_rope_kv_cache_update_fwd_kernel`。 |
| `decode_attn_v4_cta_qhead_splitkv_fused` | 针对 GQA 结构把 decode attention 组织为 `q-head + split-KV` 的 CTA 级并行，并在 kernel 内完成 partial merge，是最终版本的 decode attention 优先路径。 | `optimized_layer.py` 中 decode attention 路由调用 `cuda_decode_attention.decode_attn_v4_cta_qhead_splitkv_fused()` | Python 包装在 `cuda_decode_attention.py`，扩展绑定在 `cuda_extensions/decode_attention_ext.cpp`，核心 kernel 为 `cuda_extensions/decode_attention_v4_kernel.cu` 中的 `decode_attn_v4_cta_qhead_splitkv_fused_kernel`。 |
| `nosplit_matvec_half` kernel 家族 | 覆盖 `o_proj`、`down_proj`、`lm_head` 等单 token matvec；运行时按 shape 分派到 `k=2048` / `k=6144` 的专用 CUDA kernel，以避免 cuBLAS 在 `M=1` 下走低效 GEMM 路由。 | `optimized_layer.py::_cuda_nosplit_gemv()`；`evaluation_wrapper.py::_optimize_lm_head()` | Python 包装在 `cuda_decode_gemv_ops/matvec.py` 与 `cuda_decode_gemv_ops/runtime.py`，扩展入口在 `cuda_extensions/decode_gemv_ext.cpp`，launcher 在 `cuda_extensions/decode_gemv_kernel_launchers.cuh`，kernel 实现在 `cuda_extensions/decode_gemv_kernel_matvec.cuh` 与 `cuda_extensions/decode_gemv_kernel_xwarp.cuh`。 |
| `fused_post_attn_mlp_persistent_half` | cooperative persistent kernel，把 `post_attn add+rmsnorm → gate_up → SiLU*up → down_proj → residual_add` 合并到一条 decode kernel 中，减少中间结果写回与重复 launch。 | `optimized_layer.py::_fused_post_attn_mlp_persistent()` | Python 包装在 `cuda_decode_gemv_ops/fused.py`，扩展入口在 `cuda_extensions/decode_gemv_ext.cpp`，核心 kernel 为 `cuda_extensions/decode_gemv_gridsync.cu` 中的 `fused_post_attn_mlp_persistent_k2048_k6144_fwd_kernel`。 |

#### 4.2.2 Prefill 侧 Kernel

| Kernel / 算子 | 实现概述 | Python 入口 | 底层实现与代码位置 |
|---------------|----------|-------------|--------------------|
| `_fused_qk_rms_norm_rope_fwd_kernel` | 在 full-prompt prefill 中同时对 Q、K 做 RMSNorm 与 RoPE，减少两路独立 norm/rope 的读写与 launch。 | `optimized_layer.py::_fused_qk_rms_norm_rope()` | Triton kernel 直接定义在 `optimized_layer.py`，对应实现名为 `_fused_qk_rms_norm_rope_fwd_kernel`。 |
| `_fused_rms_norm_rope_fwd_kernel` | 单张量版本的 RMSNorm+RoPE kernel，用于与双路 QK 融合路径配套的兼容分支。 | `optimized_layer.py::_fused_rms_norm_rope()` | Triton kernel 直接定义在 `optimized_layer.py`，实现名为 `_fused_rms_norm_rope_fwd_kernel`。 |
| `_fused_add_rms_norm_fwd_kernel` | 将 residual add 与 RMSNorm 合成一条 Triton kernel，在 prefill MLP 入口减少一次中间张量写回。 | `optimized_layer.py::_fused_add_rms_norm()` | Triton kernel 直接定义在 `optimized_layer.py`，实现名为 `_fused_add_rms_norm_fwd_kernel`。 |
| `_fused_silu_mul_packed_fwd_kernel` | 直接消费 packed `gate_up [M, 2N]` 输出，完成 `SiLU(gate) * up`，避免 `chunk()` 带来的额外 copy kernel。 | `optimized_layer.py::_fused_silu_mul_packed()` | Triton kernel 直接定义在 `optimized_layer.py`，实现名为 `_fused_silu_mul_packed_fwd_kernel`。 |
| `_fused_vision_rope_fwd_kernel` | 将 vision encoder 中 `apply_rotary_pos_emb_vision` 的多步 PyTorch 运算融合成一个 Triton kernel，显著减少 vision prefill 阶段的小 kernel 数量。 | `optimized_layer.py::fused_vision_rope()`，调用发生在 `evaluation_wrapper.py` 的 vision prefill 路径 | Triton kernel 直接定义在 `optimized_layer.py`，实现名为 `_fused_vision_rope_fwd_kernel`。 |

需要说明的是，prefill 路径中的 `qkv / o_proj / down_proj` 在最终版本仍主要使用 PyTorch/底层 GEMM 路径；最终版本在 prefill 阶段重点启用的是上述 Triton 小算子融合与 fastpath 调度，而不是把 full-prompt GEMM 全部替换成自定义 CUDA kernel。

### 4.3 最终采用的开关与实验性开关

#### 4.3.1 最终采用的默认配置

| 环境变量 | 默认值 | 含义 |
|----------|--------|------|
| `ENABLE_CUSTOM_DECODE` | 1 | 自定义 greedy decode 引擎 |
| `ENABLE_CUDAGRAPH` | 1 | decode 走 `CUDAGraph` |
| `ENABLE_CUDA_NOSPLIT_GEMV` | 1 | 单 token matvec 使用自定义 CUDA GEMV |
| `ENABLE_CUDA_FUSED_RMSNORM_QKV` | 1 | decode 的 input RMSNorm + QKV 融合 |
| `ENABLE_CUDA_FUSED_Q_ROPE_KV_CACHE` | 1 | decode 的 Q-RoPE + KV cache 写入融合 |
| `ENABLE_CUDA_FUSED_POST_ATTN_MLP_PERSISTENT` | 1 | persistent post-attn MLP 融合 |
| `ENABLE_CUDA_FUSED_GATEUP_SILU_DOWNPROJ` | 1 | gateup+silu+downproj 融合 fallback |
| `CUDA_FUSED_GATEUP_SILU_DOWNPROJ_VARIANT` | `persistent` | 默认使用 persistent 变体 |
| `ENABLE_CUDA_SMALL_DECODE_ATTENTION_V4` | 1 | 开启 CUDA v4 decode attention |
| `ENABLE_CUDA_SMALL_DECODE_ATTENTION_V4_QHEAD_SPLITKV` | 1 | 默认优先 q-head split-KV |
| `ENABLE_PREFILL_VISION_GRID_CACHE` | 1 | vision 几何与位置缓存 |
| `ENABLE_PREFILL_MM_SLICE_FASTPATH` | 1 | 单图输入的跨模态 slice fastpath |
| `ENABLE_PREFILL_TEXT_FASTPATH` | 1 | text prefill fastpath |
| `ENABLE_PREFILL_LAYER_FASTCALL` | 1 | prefill 层级快速调用 |
| `ENABLE_TRITON_PREFILL_FUSED_ADD_RMSNORM` | 1 | prefill fused add+rmsnorm |
| `ENABLE_TRITON_PREFILL_FUSED_SILU_MUL` | 1 | prefill packed SiLU*mul |
| `ENABLE_TRITON_PREFILL_FUSED_QK_NORM_ROPE` | 1 | prefill fused QK-norm+RoPE |
| `ENABLE_TRITON_PREFILL_FUSED_RMS_NORM_ROPE` | 1 | prefill fused RMSNorm+RoPE |
| `ENABLE_FUSED_LM_HEAD_ARGMAX` | 1 | 历史命名保留；实际作用是单 token `lm_head` 走 fast GEMV，而不是独立 fused argmax kernel |

#### 4.3.2 未纳入最终提交的配置

| 环境变量 | 默认值 | 状态 |
|----------|--------|------|
| `ENABLE_CUDA_FUSED_FULL_LAYER_SOFTSYNC` | 0 | 实验路径，且历史高收益已被判定无效 |
| `ENABLE_CUDA_FUSED_O_PROJ_POST_ATTN_MLP_PERSISTENT` | 0 | 实验路径 |
| `ENABLE_CUDA_FUSED_O_PROJ_POST_ATTN_MLP_SOFTSYNC` | 0 | 实验路径 |
| `ENABLE_PARTITION_GEMV` | 0 | 代码保留，但不属于最终提交默认配置 |
| `ENABLE_PREFILL_TEXT_CUDNN_SDPA` | 0 | 可选实验开关 |
| `ENABLE_CUDA_SMALL_DECODE_ATTENTION_O_PROJ_PREFETCH` | 0 | 可选实验开关 |
| `ENABLE_CUDA_PERSISTENT_ATTENTION` | 0 | 可选实验开关 |

### 4.4 `CUDAGraph` 兼容性与静态化策略

为了让 decode 真正稳定跑在 graph 里，我们做了几项关键静态化处理：

- graph runner 预分配输入、mask、`position_ids`、`cache_position`、`next_token`、`StaticCache`
- decode attention 的 split-KV workspace 挂在 layer 生命周期上，避免 capture 后再分配
- `lm_head` 的输出 buffer 固定地址
- EOS 检测使用 host ring buffer 异步轮询，不阻断 graph replay
- `_run_step()` 内直接对 `outputs.logits[:, -1, :]` 做 `argmax` 并拷贝到 `_next_token_buf`，不再额外保留一份独立 logits staging buffer

这也是为什么最终采用的优化大多都要求：

- `batch_size=1`
- `dtype=torch.float16`
- `shape` 固定
- `stride` 满足特定条件

因为只有这样，kernel 替换和 graph capture 才能稳定地进入真正的 benchmark 路径。

---

## 5. 实验结果

### 5.1 评测设置

| 项目 | 值 |
|------|---|
| 评测样本 | 150（线上排行榜）/ 5000（完整验证集） |
| batch size | 1 |
| 最大生成长度 | 128 tokens |
| 预热样本 | 10 |
| 线上 GPU | NVIDIA A800 80GB PCIe |
| Baseline TTFT | ~80 ms |
| Baseline Throughput | ~55 tok/s |

### 5.2 性能演进关键里程碑

| 阶段 | TTFT (ms) | 吞吐量 (tok/s) | 准确率 | 线上得分 | 说明 |
|------|-----------|----------------|--------|----------|------|
| Baseline (无优化) | ~80 | ~55 | - | - | 官方 Baseline |
| A800 首次评测 | 68.32 | 33.07 | 0.7044 | - | 仅有基础配置 |
| Fused QKV/gate_up | 60.10 | 46.57 | 0.6867 | 0.0820 | 进入 fused decoder layer 阶段 |
| CUDAGraph 首次成功 | 62.27 | 75.53 | - | - | decode 执行流重构开始见效 |
| Graph pool + 预建 | 60.74 | 102.51 | - | - | graph runner 复用成为最终实现的一部分 |
| **线上评测 v1** | **60.32** | **108.82** | **0.6978** | **0.9576** | decode Triton 路径基本成型 |
| Split-KV + CUDA GEMV | 61.67 | 200.55 | 0.7022 | 1.6398 | decode throughput 进入 200 tok/s 级别 |
| **线上评测 v2** | **62.10** | **212.47** | **0.7022** | **1.7274** | 稳定版提交 |
| Text prefill fastpath | 43.49 | 226.11 | - | - | TTFT 首次大幅下降 |
| Persistent post-attn MLP | 42.56 | 280.62 | - | - | decode MLP persistent 启用 |
| Prefill fused QK-norm+RoPE | 46.18 | 239.75 | - | - | prefill kernel 套件开始接管 |
| Prefill fused vision RoPE | 38.78 | 247.95 | - | - | kernel-count 压缩带来明显 TTFT 收益 |
| **线上评测 v3** | **36.18** | **255.65** | **0.6956** | **2.1623** | prefill 侧优化成熟 |
| **线上评测 v4** | **36.60** | **271.25** | **0.6956** | **2.2775** | fused Q-RoPE + KV cache 启用后的提交版本 |
| Prefill fastcall + shared context（本地 10-sample） | 35.13 | 262.16 | - | - | 显式 prefill layer fastcall 启用，并把共享上下文提到 loop 外 |
| qhead_splitkv4 + `max_seq_len=4096` 启用（本地 10-sample） | 35.24 | 269.23 | - | - | A800 上 decode attention 优先路径切到 qhead split-KV |
| **线上评测 v5（最新）** | **36.66** | **284.48** | **0.6956** | **2.3765** | `commit [historical commit]` 官方评测；对应上面两条本地过渡记录收敛后的提交结果 |

> 口径说明：
>
> - prompt-prefix reuse 已被主办方判定违规，不计入最终实现。
> - full-layer softsync 的历史高收益观测值建立在已修复的问题版本上，因此不用于解释最终成绩。
> - 表中 `Prefill fastcall + shared context` 与 `qhead_splitkv4 + max_seq_len=4096` 都属于 A800 本地 10-sample benchmark，用来解释 `v4 -> v5` 的关键收敛过程。
> - 这些本地记录不应机械地和官方 5000-sample 结果逐项一一映射；它们的作用是补足 v4 到 v5 之间真正进入最终实现的关键变化。

### 5.3 最终性能对比

| 指标 | 官方 Baseline | 最终优化（线上评测 v5） | 变化 |
|------|---------------|--------------------------|------|
| TTFT | ~80 ms | 36.66 ms | **-54.2%** |
| 吞吐量 | ~55 tok/s | 284.48 tok/s | **+417.2%** |
| 准确率 | - | 0.6956 | 保持稳定 |
| 综合得分 | - | 2.3765 | - |

### 5.4 关键增益拆解

#### Decode 侧

| 优化 | 变化 | 说明 |
|------|------|------|
| graph runner 复用 + 预建 | `75.53 → 102.51 tok/s` | decode 执行流重构的核心收益 |
| dot-style attention / 去掉冗余 contiguous / Triton 单 token KV 更新 | `183.05 → 194.24 tok/s` | decode 小算子链路压缩的阶段性收益 |
| CUDA nosplit GEMV | `+6.7% throughput, -4.4% TTFT` | 直接验证 cuBLAS 在 `M=1` 下并非最优 |
| gateup+silu+downproj persistent | `273.30 → 276.43 tok/s` | 单独看收益不大，但稳定且 bit-exact |
| post-attn MLP persistent | `276.17 → 280.62 tok/s` | 把 MLP 融合再往前扩一层 |
| fused Q-RoPE + KV cache（v3 → v4 官方） | `255.65 → 271.25 tok/s` | 后期 decode 小算子链路的关键收益 |
| qhead_splitkv4 + `max_seq_len=4096` 启用 | `35.42 → 35.24 ms`，`262.21 → 269.23 tok/s` | v4 之后进入 v5 的关键 decode attention 收益 |

#### Prefill 侧

| 优化 | 变化 | 说明 |
|------|------|------|
| text prefill fastpath | `61.11 → 43.49 ms` | 首次把 TTFT 明显拉下 40ms 级别 |
| prefill layer fastcall 启用 | `37.54 → 36.03 ms` | 显式 prefill 层级调用路径进入最终实现 |
| `PrefillFastpathContext` 共享上下文 | `36.03 → 35.13 ms` | 再降 `0.90 ms` TTFT，继续压缩 per-layer Python 调度 |
| fused prefill QK-norm+RoPE | `49.49 → 46.18 ms` | 修复精度/stride 问题后正式成为最终实现的一部分 |
| packed SiLU*mul | `46.18 → 46.11 ms` | 小幅但稳定的 kernel-count 压缩 |
| 去 DirectCopy + fused vision RoPE | `46.11 → 38.78 ms` | 最关键的 prefill kernel 数减少收益 |

---

## 6. 正确性验证

我们把正确性放在与性能同等重要的位置。一个优化要进入最终提交配置，至少需要满足以下要求：

1. **逐步 A/B 验证**：新优化默认先关闭，通过 10 样本或 100 样本 A/B 对照后再决定是否提升为默认。
2. **答案一致性检查**：CUDA nosplit GEMV、persistent decode MLP、prefill fused QK-norm+RoPE、packed SiLU*mul、fused vision RoPE 等优化都做过 `10/10` 或 `100/100` 一致性验证。
3. **关键算子数值精度验证**：除了最终答案级一致性之外，我们还持续对关键融合算子做与原始分步实现的数值对比，覆盖 decode attention、nosplit GEMV、post-attn MLP 融合、prefill fused QK-norm+RoPE、packed SiLU*mul、fused vision RoPE 等路径，并检查不同输入长度、实际 benchmark shape 与非连续张量场景下的数值一致性。
4. **数值级排障**：例如在 fused QK-norm+RoPE 路径中，我们明确修复了两个问题：
   - RoPE 计算精度从 FP16 改回与原路径一致的 FP32 口径
   - PyTorch 2.8 下非连续 view 的 stride 问题导致读错 K/V，随后改为显式支持对应 stride
5. **线上准确率跟踪**：最终几个官方评测版本准确率稳定在 `0.6956` 左右，没有因为最终采用的优化出现新的退化。

---

## 7. 未纳入最终提交配置的较大方向探索

### 7.1 Partition-aware GEMV

我们曾尝试利用 A800 上 HBM/L2 访存局部性做 partition-aware GEMV，希望把 page-level 延迟差异转化为单 token matvec 的带宽收益。围绕这一方向，先后做了顺序 `256B` stride latency trace、基址 offset 扫描、逐 SM 的“地址-延迟”曲线对比，以及双副本权重、`8KB` skew、`smid` 路由等实验。

实验结果表明，这类局部性现象更接近一个与地址相位相关的 `256KB` 级周期，而不是可以稳定利用的简单 odd/even `8KB` 规则。进一步的 GEMV 端到端对照中，`[64, 2048]` 与 `[256, 2048]` 等较小形状仅出现 `+0.5%` 到 `+5.1%` 的轻微波动；`[1024, 2048]`、`[4096, 2048]` 与 `[8192, 2048]` 等较大形状则转为回退。综合收益稳定性、双副本显存开销与运行时路由复杂度，这一方向未纳入最终提交配置。

### 7.2 Full-layer cooperative fusion

另一个投入较大的方向，是把 decode attention、`o_proj` 和 post-attention MLP 组织进同一个 cooperative kernel，通过阶段同步减少 kernel launch 与中间结果往返。这一方向早期一度获得较高局部收益，但在后续核查中发现历史版本存在 `down_proj` row coverage 问题，因此相关高分记录未用于最终结果说明。

在修复正确性问题后，我们继续推进这一方向，并把其中的 attention 阶段升级为 CTA 级 `qhead splitkv4`。这一步使 full-layer 路径的性能明显恢复：10-sample benchmark 中，旧版 full-layer 为 `250.07 tok/s`，升级后提升到 `261.44 tok/s`。不过，在相同口径下，分离式实现的参考配置仍可达到 `269.32 tok/s`。因此，full-layer cooperative fusion 证明了进一步压缩 decode 执行链的可行性，但截至提交版本，尚未超过最终采用的分离式高性能实现。

### 7.3 基于 full-layer 的跨算子 prefetch

在 full-layer cooperative fusion 之上，我们还尝试了跨算子参数预取，包括 attention 阶段对 `o_proj` 权重的 sidecar prefetch、owner-row prefetch，以及在阶段同步前的自旋等待窗口中提前预取下一阶段参数。相关 profile 显示，这些方法在 kernel 局部层面确实可以改善后续阶段的访存，例如 `qsplit4` full-layer 上，较优的 all-idle `o_proj` prefetch 配置可把单层 profile 总时间压到 `90.006 us`，对应 10-sample benchmark 达到 `261.88 tok/s`。

但从端到端结果看，这些收益仍不足以改变 full-layer 路径与最终分离式实现之间的相对排序；同时，更激进的方案，如整核提升到 `32 warp/block`，或在最终自旋阶段加入 owner-style next-stage prefetch，都会带来明显回退。基于这一点，跨算子 prefetch 被保留为后续可继续推进的融合方向，但未纳入最终提交配置。

---

## 8. 结论

在不修改模型参数、不使用量化/稀疏化的前提下，我们最终把 Qwen3-VL-2B-Instruct 在单张 A800 上的官方评测成绩推进到：

- **TTFT 36.66 ms**
- **Throughput 284.48 tok/s**
- **Accuracy 0.6956**
- **Score 2.3765**

从最终实现的视角看，几个关键结论如下：

1. **Decode 执行流必须先重构**：`StaticCache` + `CUDAGraph` + 自定义 greedy decode 是所有后续 decode 小算子优化成立的基础。
2. **真正高收益的是一串稳定的 batch=1 特化替换**：CUDA nosplit GEMV、CUDA v4 q-head split-KV attention、fused Q-RoPE + KV cache、persistent post-attn MLP，这些一起定义了提交版本的 decode 性能。
3. **TTFT 的决定性收益来自 prefill fastpath 与 kernel-count 压缩**：text no-mask fastpath、single-image slice fastpath、vision geometry cache、fused vision RoPE、消除 DirectCopy，共同把 TTFT 从 60ms 级别压到了 36ms 级别。

---

## 参考文献

1. Bai, S., Cai, Y., Chen, R., et al. *Qwen3-VL Technical Report*. arXiv:2511.21631, 2025. https://arxiv.org/abs/2511.21631
2. Dao, T., Fu, D. Y., Ermon, S., Rudra, A., Re, C. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. Advances in Neural Information Processing Systems 35, 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html
3. Kwon, W., Li, Z., Zhuang, S., et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. arXiv:2309.06180, 2023. https://arxiv.org/abs/2309.06180
4. Zheng, L., Yin, L., Xie, Z., et al. *SGLang: Efficient Execution of Structured Language Model Programs*. arXiv:2312.07104, 2023. https://arxiv.org/abs/2312.07104
5. Tillet, P., Kung, H.-T., Cox, D. *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*. Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, 2019. DOI: 10.1145/3315508.3329973
6. NVIDIA. *CUDA C++ Programming Guide*, Release 12.8. NVIDIA Documentation, 2024. https://docs.nvidia.com/cuda/archive/12.8.0/pdf/CUDA_C_Programming_Guide.pdf
7. NVIDIA. *cuBLAS Library User Guide*, Release 12.8. NVIDIA Documentation, 2024. https://docs.nvidia.com/cuda/archive/12.8.0/pdf/CUBLAS_Library.pdf
