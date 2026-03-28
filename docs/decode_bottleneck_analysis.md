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

### 3.1 这些 `Copy / Mul / Reduction` 在当前代码里分别来自哪里

结合最新默认配置（`small decode add+rmsnorm` 已开启）和当前实现，主要来源可以直接映射到下面几段：

- `Reduction`
  - 主要是剩余 **3 个 RMSNorm**：
    - input RMSNorm: [`optimized_layer.py:102`](../optimized_layer.py#L102)
    - q RMSNorm: [`optimized_layer.py:110`](../optimized_layer.py#L110)
    - k RMSNorm: [`optimized_layer.py:111`](../optimized_layer.py#L111)
  - post-attn RMSNorm 原本也会贡献一组 reduction，但现在已经由 [`_fused_add_rms_norm()`](../optimized_layer.py#L179) 合掉了，所以 trace 里 `Reduction` 已明显下降。

- `Mul`
  - 第一大头是 **decode RoPE** 的逐元素乘加：
    - [`optimized_layer.py:327`](../optimized_layer.py#L327) 到 [`optimized_layer.py:338`](../optimized_layer.py#L338)
    - 这里会展开成 `x1*cos - x2*sin` 和 `x2*cos + x1*sin`
  - 第二部分是 RMSNorm 的 scale：
    - decode 小 shape 下 `_rms_norm()` 走 `F.rms_norm`，内部会有逐元素乘
    - 位置在 [`optimized_layer.py:161`](../optimized_layer.py#L161) 到 [`optimized_layer.py:176`](../optimized_layer.py#L176)
  - 第三部分是 MLP 里的 `SiLU(gate) * up`：
    - [`optimized_layer.py:142`](../optimized_layer.py#L142) 到 [`optimized_layer.py:148`](../optimized_layer.py#L148)
    - 但从单独 trace 看，这一块已经不是主要瓶颈，收益明显小于 add+rmsnorm

- `Copy`
  - 第一大头是 **KV cache 写入**：
    - layer 内调用点在 [`optimized_layer.py:117`](../optimized_layer.py#L117) 到 [`optimized_layer.py:122`](../optimized_layer.py#L122)
    - 实际实现是 `StaticLayer.update()` 里的 `index_copy_`：
      - [`venv/lib/python3.12/site-packages/transformers/cache_utils.py:333`](../venv/lib/python3.12/site-packages/transformers/cache_utils.py#L333)
      - [`venv/lib/python3.12/site-packages/transformers/cache_utils.py:334`](../venv/lib/python3.12/site-packages/transformers/cache_utils.py#L334)
  - 第二部分是 attention 输出整理：
    - [`optimized_layer.py:129`](../optimized_layer.py#L129) 的 `.reshape(...).contiguous()`
  - 第三部分是 decode 小张量在 view/split/cat/materialize 边界上的中间结果落地：
    - `qkv.split(...)` 见 [`optimized_layer.py:106`](../optimized_layer.py#L106)
    - RoPE 的 `torch.cat(...)` 见 [`optimized_layer.py:337`](../optimized_layer.py#L337)
    - 最后一层残差 `residual + hidden_states` 见 [`optimized_layer.py:152`](../optimized_layer.py#L152)

### 3.2 对应该怎么优化

从 trace 和代码结构看，下一步更值得做的是继续压掉 “剩余 RMSNorm + RoPE + cache write” 这几类小 kernel，而不是继续深挖 `SiLU`。

建议优先级：

1. **做 decode `q_norm + k_norm + rope` 融合**
   - 直接吃掉一部分 `Reduction + Mul + Cat`
   - 这段现在是最典型的 “多个超小 kernel 串起来” 模式
   - 对 batch=1, seq=1 decode 特别友好

2. **做 decode `input_rmsnorm + qkv` 融合**
   - 把 input RMSNorm 的 reduction 和后面的 QKV 读写边界压掉
   - 这是层内剩下最肥的一段未融合前处理
   - 难度会比 add+rmsnorm 高，因为要把 norm 与 linear 的访存布局一起处理

3. **专门处理 KV cache 写入**
   - 当前 `index_copy_` 是通用路径，灵活但不够轻
   - decode 单 token 下，本质上是对固定位置写一小片连续块，更适合定制化写回
   - 这块主要目标是压 `Copy` 和少量 runtime 开销

4. **`SiLU(gate) * up` 先保持默认开启，但不再作为主攻方向**
   - 精度已确认没问题
   - trace 也说明它只能吃掉很小一部分 `Mul`
   - 后续除非顺手并入更大 MLP fusion，否则单独继续挖意义不大

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

## 6. 历史 trace 演进对比：host 端还能不能继续抠？

用同一套 `analyze_traces.py` 对历史 trace 做横向对比后，可以更明确地回答这个问题。

| 路径 | decode GPU/步 | decode CPU op/步 | CPU/GPU | 主要 host 热点 | 结论 |
|------|---------------|------------------|---------|----------------|------|
| `traces_cudagraph/` | 9.46ms | 30.33ms | 3.2x | `item/is_nonzero` 17.01ms/步, `cudaStreamSynchronize` 8.51ms/步 | 仍明显 host-bound |
| `traces_cudagraph_async2_cachepos_device/` | 9.47ms | 4.68ms | 0.5x | `cudaEventSynchronize` 总计 447.3ms / 72 次, `cudaStreamSynchronize` 约 0.01ms/步 | host 主瓶颈大部分已解掉 |
| `traces/` (最新) | 10.15ms | 2.14ms | 0.2x | `cudaGraphLaunch` 3.23ms/步, `cudaEventSynchronize` ~6.0ms/步 | 进一步 host 微调收益有限，主矛盾在 graph 内 GPU kernel |

补充判断:

- 从 `traces_cudagraph/` 到 `traces_cudagraph_async2_cachepos_device/`，decode CPU op/步从 `30.33ms` 降到 `4.68ms`，`item()` 与 `_write_inputs()` 的同步点基本都被清掉了。
- 最新 trace 中 `CPU op` 只剩 `2.14ms/步`，但 graph 内 GPU kernel 仍要 `10.15ms/步`。这说明继续优化 Python loop、普通 `aten::copy_` 或 EOS 判断，已经不在收益最大的象限。
- 最新 trace 的 `cudaGraphLaunch` / `cudaEventSynchronize` 绝对时间仍不小，但 `viztracer + CUPTI` 会显著放大 host runtime 时间，因此这些数值更适合看相对变化，不适合直接拿来和 benchmark 的真实 tok/s 做一一对应。

## 7. Score 杠杆分析：TTFT 还是 Throughput 更值得优先押注？

比赛分数公式:

```
Score = 0.4 * Accuracy + 0.3 * TTFT提升率 + 0.3 * Throughput提升率
```

按 README 的 baseline (`TTFT≈80ms`, `Throughput≈55 tok/s`) 粗略展开：

- `TTFT` 每下降 `1ms`，分数约增加 `0.3 / 80 = 0.00375`
- `Throughput` 每增加 `1 tok/s`，分数约增加 `0.3 / 55 = 0.00545`

即：

- `+1 tok/s` 的边际收益，大致相当于 `TTFT -1.45ms`
- 若把 `TTFT 60ms → 42ms`，理论只增加约 `0.0675` 分
- 若把 `Throughput 108.8 → 140 tok/s`，理论增加约 `0.170` 分

因此在当前分数段，**throughput 仍是第一杠杆，TTFT 是第二杠杆**。除非 TTFT 能一次性拿到 15ms 以上的大改善，否则优先级不应高于 decode throughput。

## 8. 方向优先级修正

结合当前代码状态、trace 证据和离线环境限制，建议把后续方向收敛为：

### P1. `torch.compile` 单步 decode forward，再 capture 进 CUDAGraph

这是当前最值得先做的实验，但要收缩作用域，避免重复踩旧坑。

- 不要再尝试 compile 整个 vision encoder
- 不要用 `reduce-overhead`，避免再次和内部 CUDAGraph 冲突
- 只编译 **decode 单步静态 shape** 的 forward 路径，优先考虑：
  - `DecodeCudaGraphRunner._run_step()` 的最小包装模块
  - 或只编译 `model.model.language_model` / `OptimizedDecoderLayer.forward`
- 目标不是先看 benchmark，而是先看 trace：
  - graph 内 elementwise kernel 数是否明显下降
  - `Elementwise 4.29ms/步` 是否能先压到 `<=3ms/步`

原因:

- 当前最大热点正是编译器最擅长处理的 elementwise fusion
- decode 路径 shape 已静态化，比早期 phase2 的动态 KV cache 条件好很多
- 即使 benchmark 只提升 `10-20%`，也是当前分数段最有价值的增益来源

### P2. 手写 Triton decode fusion，优先做两个 kernel

如果 `torch.compile` 在 `StaticCache` / SDPA / graph capture 上仍不稳定，就转手写 fusion，不要继续在 compile 兼容性上反复拉扯。

优先级建议：

1. `residual add + RMSNorm`
2. `SiLU(gate) * up`

原因:

- 它们直接对应 trace 中最高频的 `Copy/Mul/SiLU/Reduction`
- 与现有 `OptimizedDecoderLayer` 结构天然贴合，改动边界比较清晰
- 比“重写整个 attention / MLP”风险小得多

### P3. TTFT 专项优化放到 throughput 路线之后

TTFT 目前的主要证据是 prefill GPU kernel `50.6ms/样本`，其中 `GEMM + FlashAttn + Elementwise` 占绝大头。它不是没有空间，但当前收益杠杆弱于 throughput。

更合理的触发条件：

- `P1/P2` 都做完，throughput 已接近平台天花板
- 或线上排名证明 TTFT 比吞吐更缺口明显

具体再考虑：

- vision/text prefill 分拆 trace
- vision encoder 单独 benchmark
- 是否有不依赖外部安装的 prefill graph/compile 方案

### P4. FlashInfer / 外部 decode kernel 暂时降级

理论上是有效方向，但当前环境是离线评测机，安装受限，仓库里也没有现成集成。除非本地已经有可直接复用的实现，否则它不是下一步的最优起点。

## 9. 最新实验回填：Triton decode fusion 的第一轮结论

2026-03-28 已做过第一轮小范围实证，结论比“直接全面上 Triton”更细一点：

- `small decode add+rmsnorm + silu*up` 只在 **CUDAGraph warmup/capture** 时启用，10 样本可把吞吐从 `102.52` 提到 `108.39~109.11 tok/s`
- 但 TTFT 同时从 `60.90ms` 升到约 `63.2ms`
- 与 `step22` 的 10 样本预测文本只做到 `9/10 exact match`
- 只保留 `small silu*up` 时，吞吐回落到 `102.78 tok/s`，说明主要收益来自 `small add+rmsnorm`

因此当前判断是：

- **方向是对的**：decode graph 内的 Triton fusion 确实能换到真实吞吐收益
- **实现还不够稳**：带 reduction 的 `add+rmsnorm` 目前有数值漂移风险，不能直接默认开启
- 下一步如果继续走 Triton，重点不是“再多写几个 kernel”，而是先把 `add+rmsnorm` 的数值一致性和 TTFT 回归问题压住

## 10. 第二轮验证：只开 `small add+rmsnorm`

进一步把实验缩到更干净的形态：

- `ENABLE_TRITON_SMALL_ADD_RMSNORM=1`
- `ENABLE_TRITON_SMALL_SILU_MUL=0`
- 仍然只在 **CUDAGraph warmup/capture** 时临时开启

### Trace 结论

同一 `sample_idx=0`、同一 `run_viztrace.py --mode cudagraph` 条件下：

| 指标 | baseline | Triton add+rmsnorm | 变化 |
|------|----------|--------------------|------|
| decode GPU/步 | 9.26ms | 8.71ms | **-5.9%** |
| decode CPU op/步 | 1.93ms | 1.82ms | -5.7% |
| Elementwise | 4.12ms | 3.58ms | **-13.1%** |
| Reduction | 0.37ms | 0.28ms | **-24.3%** |
| FlashAttn | 1.42ms | 1.42ms | 持平 |
| GEMV | 1.02ms | 1.02ms | 持平 |
| cudaGraphLaunch | 2.12ms/步 | 1.84ms/步 | -13.2% |

这说明收益确实来自 **graph 内部 GPU kernel 数和 HBM roundtrip 的减少**，不是 benchmark 噪声。

### 精度/稳定性结论

本地数据集没有标准答案字段，因此无法直接复算官方 accuracy；但可以做输出稳定性回归。

- `100` 样本基线：`64.50ms / 104.15 tok/s`
- `100` 样本 Triton add+rmsnorm：`64.13ms / 110.44 tok/s`
- 与基线对比：**预测文本 exact match 100/100**

因此至少在当前 100 样本回归上：

- 没看到精度/输出漂移问题
- 吞吐提升约 `+6.0%`
- TTFT 没有再出现 10 样本实验时那种回退

这也说明第一轮 `9/10 exact match` 的问题，更可能来自“fusion 组合过多”或小样本波动，而不是 `small add+rmsnorm` 这个方向本身不可用。

## 11. 为什么 `silu-only` 收益很小？

这个问题已经用 trace 验证过了：**不是精度问题，而是它本来就不在 decode 主热点上。**

### 精度先排除

- 独立 kernel 测试：`_fused_silu_mul` 在 `(1,1,4096/6144/8192)` 的 decode 小 shape 上，与 `F.silu(gate) * up` 做 50 轮随机对比，结果是 **bit-identical**
- `100` 样本回归：`ENABLE_TRITON_SMALL_ADD_RMSNORM=0`、`ENABLE_TRITON_SMALL_SILU_MUL=1` 与基线 **exact match 100/100**

所以 `silu-only` 不是“有精度风险才没上”，而是“安全但不值钱”。

### Trace 结论

基线 vs `silu-only`：

| 指标 | baseline | silu-only | 变化 |
|------|----------|-----------|------|
| decode GPU/步 | 9.26ms | 9.22ms | `-0.4%` |
| Elementwise | 4.12ms | 4.00ms | `-2.9%` |
| Reduction | 0.37ms | 0.37ms | 几乎不变 |
| `SiLU` hotspot | ~0.087ms/步 | ~0.019ms/步 | 有下降，但绝对值小 |

结论：

- `SiLU` kernel 自己确实被优化掉了一部分
- 但它在 decode 总时延里本来只占很小一块
- 现在真正大的还是 `Copy / Mul / generic elementwise / Reduction / Attention / GEMV`

也就是说，`silu-only` 没什么收益，**不是因为 fusion 没生效，而是因为它只打到了 MLP 激活这一小段，而这段本来就不是主矛盾**。

### 对下一步优化的意义

如果继续往下做 Triton / CUDA Graph 图内 fusion，优先级应是：

1. `residual + rmsnorm` 相关的 `Copy + Reduction + Mul`
2. `q_norm + k_norm + rope` 一类仍然制造大量 `Mul/Copy` 的路径
3. `lm_head` / 线性投影的单 token GEMV 路径

`silu*up` 可以保留为低风险小优化，但它不应该再作为主线投入时间。

### 优先级建议

```
[1] P1: 单步 decode `torch.compile` + CUDAGraph
[2] P2: Triton decode fusion (`residual+RMSNorm`, `SiLU*up`)
[3] P3: Vision / prefill TTFT 专项优化
[4] P4: FlashInfer / 外部 decode kernel（仅在本地可直接落地时考虑）
```

## 12. 新工具补充：steady-state decode 分析（`generate_2 - generate_1`）

2026-03-28 又抓了一版新 trace：

- 目录：[`traces/nvtx_decode_breakdown_rf`](../traces/nvtx_decode_breakdown_rf)
- 说明：尝试给 decode forward 加 NVTX/`record_function` 分段，但在 `viztracer + CUDAGraph replay` 这条链路下，图内用户 range 没有直接保留到 trace 里
- 因此改为在 [`analyze_traces.py`](../analyze_traces.py) 中新增 **`generate_2 - generate_1` 的 steady-state decode 估算**

这样可以把 `generate_2` 里的 prompt prefill 和首 token 影响尽量扣掉，更接近“稳定 decode 单步”。

### steady-state decode 每步 GPU 时间

新工具输出：

| 类别 | steady-state GPU/步 | 占比 |
|------|---------------------|------|
| Elementwise | **3.47ms** | **42.2%** |
| GEMM | 1.90ms | 23.1% |
| FlashAttn | 1.33ms | 16.2% |
| GEMV | 1.04ms | 12.7% |
| Reduction | 0.28ms | 3.4% |
| Cat/Concat | 0.12ms | 1.4% |
| Norm | 0.07ms | 0.9% |

总计约 **`8.22ms / steady decode step`**。

### 按结构估算到 decode 路径

工具里的“Decode structure estimate”给出了一版更接近代码结构的拆解：

| 路径 | steady-state GPU/步 | 单层均摊 |
|------|---------------------|----------|
| `qkv_proj` | `0.413ms` | `14.8us/layer` |
| `gate_up_proj` | `1.125ms` | `40.2us/layer` |
| `attention` | `1.335ms` | `47.7us/layer` |
| `o_proj + down_proj` | `1.039ms` | `37.1us/layer` |
| `post_attn_add_rmsnorm` | `0.075ms` | `2.7us/layer` |
| `input/qk norm + rope` | `0.398ms` | `14.2us/layer` |
| `SiLU activation` | `0.069ms` | `2.5us/layer` |
| `lm_head` | `0.357ms` | 非 per-layer |
| **fragmented `Copy / Mul / misc`** | **`3.410ms`** | **`121.8us/layer`** |

这里最关键的是最后一行：

- 现在 **真正还没被吃掉的大头**，不是 attention，也不是单个线性层
- 而是图内剩下的 **碎片化 `Copy / Mul / Cast / Fill / index / generic elementwise`**
- 它单独就还有 **`3.41ms/步`**，比 attention (`1.33ms`) 还大很多

### 这对下一步优化的含义

因此接下来的优先级可以再收紧一点：

1. **继续压 `Copy / Mul / generic elementwise`**
   - 主战场仍然是 `q_norm + k_norm + rope`
   - 以及 layer 内各种小张量 materialization / residual / cache write 边界

2. **`gate_up` / `down_proj` 不算小，但已经是比较“整块”的 GEMM/GEMV**
   - 它们当然有空间
   - 但从当前 trace 看，优先级仍低于碎片 elementwise 大盘

3. **`post_attn_add_rmsnorm` 已经不再是主矛盾**
   - 现在只剩 `0.075ms/步`
   - 说明默认开启的 `small add+rmsnorm` 已经把这一块压得比较干净

4. **`SiLU` 仍然不是主线**
   - steady-state 里也只有 `0.069ms/步`
   - 继续单独围着它打，杠杆还是不够

## 13. 新实验：`q_norm + k_norm + rope` 的 decode 小 shape Triton fusion

在上面的 steady-state 分析里，最值得先打的一段就是：

- `q_norm`
- `k_norm`
- `rope`

因为它同时制造：

- `Reduction`
- `Mul`
- `Cat`
- 一部分 generic elementwise / copy 边界

### 实现方式

在 [`optimized_layer.py`](../optimized_layer.py) 中新增了 `_fused_rms_norm_rope()`：

- 把 `RMSNorm + RoPE` 合到一个 Triton kernel
- 只在 **decode 小 shape** 且 `prefer_small_decode_triton=True` 时启用
- 仍然保留环境变量开关 `ENABLE_TRITON_SMALL_QK_NORM_ROPE`
- 数值上与原来的 decode 路径
  - `F.rms_norm`
  - `_apply_rotary_emb`
  保持对齐

独立微基准（`batch in {1,2}`, `heads in {8,16}`, `D=128`）：

- 原路径：约 `121~128us`
- 新 kernel：约 `49~51us`
- 单段加速约 **`2.4x~2.6x`**

### 10 样本 benchmark

同机对照：

| 配置 | TTFT | Throughput |
|------|------|------------|
| 当前默认（qk+rope fusion 关） | `160.49ms` | `63.68 tok/s` |
| 打开 `ENABLE_TRITON_SMALL_QK_NORM_ROPE=1` | `126.09ms` | `78.25 tok/s` |

- 吞吐提升约 **`+22.9%`**
- 与 baseline **`10/10 exact match`**

### 100 样本 benchmark

实验版：

- `66.44ms / 135.47 tok/s`

与 [`results/result_step26_baseline_100.json`](../results/result_step26_baseline_100.json) 对比：

- 吞吐 `104.15 -> 135.47 tok/s`，约 **`+30.1%`**
- TTFT `64.50 -> 66.44ms`，约 `+3.0%`
- 预测文本 **`100/100 exact match`**

### Trace 结论

实验 trace 在：

- [`traces/compare/triton_qk_norm_rope`](../traces/compare/triton_qk_norm_rope)

用 `generate_2 - generate_1` steady-state 分析后，decode GPU/步从上一版 fresh baseline 的约 `8.22ms` 降到 `6.34ms`。

关键变化：

| 指标 | baseline steady-state | qk+rope fusion | 变化 |
|------|------------------------|----------------|------|
| decode GPU/步 | `8.22ms` | `6.34ms` | **`-22.9%`** |
| Elementwise | `3.47ms` | `1.77ms` | **`-49.0%`** |
| Reduction | `0.28ms` | `0.10ms` | **`-64.3%`** |
| Cat/Concat | `0.12ms` | `~0ms` | 基本吃掉 |
| FlashAttn | `1.33ms` | `1.33ms` | 持平 |
| GEMV | `1.04ms` | `1.03ms` | 持平 |

新的 steady-state 结构估算里：

- `input/qk norm + rope` 从 `0.398ms/步` 降到 `0.104ms/步`
- `fragmented Copy/Mul/misc` 从 `3.41ms/步` 降到 `1.84ms/步`

这说明这次收益不是 benchmark 偶然波动，而是确实把此前最大的 decode 小 kernel 热点吃掉了一大块。

### 当前判断

这条 fusion 已经满足三个条件：

1. **trace 收益明确**
2. **100 样本输出稳定**
3. **端到端吞吐提升足够大**

因此它和 `small add+rmsnorm` 一样，已经足够进入默认配置。

## 14. 继续补：decode input RMSNorm 的小 shape Triton

在 `q_norm + k_norm + rope` 合掉之后，剩下最明确的 RMSNorm 热点其实就只剩每层开头那一个 input RMSNorm。

### 实现

在 [`optimized_layer.py`](../optimized_layer.py) 里补了：

- `ENABLE_TRITON_SMALL_RMSNORM`
- 只对 **decode 小 shape** 的 `_rms_norm(..., prefer_small_triton=True)` 生效
- 当前实际只覆盖 layer 开头的 input RMSNorm

独立微基准（`(1,1,2048)`）：

- `F.rms_norm`: 约 `52.9us`
- Triton small RMSNorm: 约 `32.1us`

### 10 样本验证

同轮对照：

| 配置 | TTFT | Throughput |
|------|------|------------|
| 当前主线 | `162.73ms` | `75.10 tok/s` |
| + `ENABLE_TRITON_SMALL_RMSNORM=1` | `110.04ms` | `79.01 tok/s` |

- 吞吐约 `+5.2%`
- 输出 **`10/10 exact match`**

这里 TTFT 波动偏大，因此不把这组 `10` 样本的 TTFT 数值本身当强结论，更主要看 throughput 和 trace。

### Trace 结论

trace 在：

- [`traces/compare/triton_small_rmsnorm`](../traces/compare/triton_small_rmsnorm)

相对上一阶段的 `triton_qk_norm_rope`：

| 指标 | qk+rope 后 | 再开 small RMSNorm | 变化 |
|------|------------|--------------------|------|
| steady-state decode GPU/步 | `6.34ms` | `5.84ms` | **`-7.9%`** |
| Elementwise | `1.77ms` | `1.28ms` | **`-27.7%`** |
| Reduction | `0.10ms` | `0.01ms` | **`-90%`** |
| FlashAttn | `1.33ms` | `1.34ms` | 持平 |
| GEMV | `1.03ms` | `1.02ms` | 持平 |

steady-state 结构估算里：

- `fragmented Copy/Mul/misc` 进一步从 `1.84ms/步` 压到 `1.42ms/步`
- `input/qk norm + rope` 只剩 `0.017ms/步`

这说明 input RMSNorm 这条虽然不是像 `qk+rope` 那样的大块收益，但它确实把剩余的 `Reduction + Elementwise` 又往下压了一截。

### 当前判断

这条优化的收益比 `qk+rope` 小，但满足：

- 数值对齐
- trace 收益明确
- `10` 样本输出稳定

因此也可以先作为默认配置保留，后面如果更大样本出现回退，再单独回滚。
