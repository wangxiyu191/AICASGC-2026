# Decode GEMV 性能分析与优化尝试

> 更新日期: 2026-03-29
> 目标: 总结 decode 阶段 `o_proj / down_proj / lm_head` 的性能现状、已验证尝试与后续优化方向

---

## 1. 背景

最新 steady-state decode trace 显示，当前 decode 阶段的主时间块已经大致变成：

- `o_proj + down_proj` GEMV: `~1.04 ms / step`
- Attention: `~0.67 ms / step`
- `lm_head`: `~0.36 ms / step`

对应 trace 分解见：

- [traces/compare/triton_decode_step39_static_kv_write](../traces/compare/triton_decode_step39_static_kv_write)

也就是说，Attention 已经不是唯一主热点，decode 线性层的 GEMV 已经足够大，值得单独梳理。

---

## 2. 目标算子与形状

Qwen3-VL-2B 当前 decode 单 token 下，相关 GEMV 的核心形状是：

| 算子 | 输入 | 权重 | 备注 |
|------|------|------|------|
| `o_proj` | `[1, 1, 2048]` | `[2048, 2048]` | 每层 1 次 |
| `down_proj` | `[1, 1, 6144]` | `[2048, 6144]` | 每层 1 次 |
| `lm_head` | `[1, 1, 2048]` | `[151936, 2048]` | 每 token 1 次 |

按每个 token 统计：

- `o_proj + down_proj` 共 `28 * 2 = 56` 次
- `lm_head` 共 `1` 次

其中 `o_proj / down_proj` 属于"小而频繁"的 per-layer GEMV，`lm_head` 则是"大而固定"的 single-call GEMV。

---

## 3. 真实路径 microbenchmark 工具

当前使用：

- [bench_decode_linears.py](../bench_decode_linears.py)

它不是脱离真实路径的人造 matvec，而是：

1. 跑真实 sample 的 benchmark 风格 prefill + custom decode
2. 在指定 `layer_idx / decode_step` 抓取 decode 时的真实输入张量
3. 直接对不同线性实现做 benchmark

这点很重要，因为 decode 的真实 bucket、cache 长度、张量 layout 都会影响结论。

---

## 4. 早期已测实现（eager 模式）

目前对每个目标 GEMV 都测了这些实现：

- `F.linear` 基线
- `F.linear(weight.contiguous())`
- `x @ weight.t()`
- 自写 Triton 普通 `matvec`
- 自写 Triton `split-K` matvec (`split_k=4/8`)

测试点位：

- sample0 / layer0 / decode_step0
  - [results/linear_microbench_step40_smoke.json](../results/linear_microbench_step40_smoke.json)
- sample0 / layer14 / decode_step16
  - [results/linear_microbench_step40_layer14_step16.json](../results/linear_microbench_step40_layer14_step16.json)

另外对 `lm_head` 还做了一轮 split-K 参数扫：

- [results/lm_head_splitk_sweep_step40.json](../results/lm_head_splitk_sweep_step40.json)

### 4.1 `o_proj`

两组点位结果都一致：

- `F.linear`: `~0.015 ms`
- `weight.contiguous()`：几乎持平，最多 `~1%~3%`
- `matmul(weight.t())`：更慢，约 `0.77x`
- Triton 普通 matvec：明显更慢，约 `0.33x`
- Triton split-K：更慢，约 `0.18x~0.20x`

### 4.2 `down_proj`

两组点位也一致：

- `F.linear`: `~0.028 ms`
- `weight.contiguous()`：几乎持平，约 `+0%~1%`
- `matmul(weight.t())`：几乎持平
- Triton 普通 matvec：只有 `~0.51x~0.59x`
- Triton split-K：只有 `~0.30x~0.34x`

### 4.3 `lm_head`

`lm_head` 是唯一一个"自写 kernel 有一点接近基线"的点，但仍没赢：

- `F.linear`: `~0.358~0.359 ms`
- Triton 普通 matvec：约 `0.91x`
- Triton split-K4：约 `0.96x`

### 4.4 早期结论

**在 eager 模式下，所有 Triton 替换都比 cuBLAS 慢。** 主要原因是 Triton Python dispatch 开销 (~29us/call) 吃掉了 GPU 级的收益。这个结论在后续 NCU 分析中被证实。

---

## 5. NCU 深度分析（2026-03-29）

### 5.1 工具链

新增了专门的 NCU profiling 工具：

- [profile_gemv_ncu.py](../profile_gemv_ncu.py) — 使用 `cudaProfilerStart/Stop` API 精确控制 NCU 采集窗口
- [ncu_gemv.sh](../ncu_gemv.sh) — `--profile-from-start off` + 单 pass metrics 采集
- [sweep_gemv.py](../sweep_gemv.py) — CUDAGraph 模式下的配置扫描

NCU 产物统一在 `results/ncu/`：

| 文件 | 内容 |
|------|------|
| `gemv_o_proj_f_linear_metrics.ncu-rep` | cuBLAS o_proj |
| `gemv_down_proj_f_linear_metrics.ncu-rep` | cuBLAS down_proj |
| `gemv_lm_head_f_linear_metrics.ncu-rep` | cuBLAS lm_head |
| `gemv_o_proj_triton_metrics.ncu-rep` | 旧 Triton matvec (BLOCK_M=64) |
| `gemv_o_proj_triton_v2_metrics.ncu-rep` | 新 Triton v2 (BLOCK_M=4) |

### 5.2 关键发现：cuBLAS 选错了 kernel

NCU 揭示了一个核心问题：**cuBLAS 对 M=1 小 N 的 GEMV 选用了 WMMA Tensor Core GEMM，而不是专用 GEMV kernel。**

| 目标 | cuBLAS kernel 类型 | Block | Grid | GPU 时间 | DRAM% | Occupancy |
|------|-------------------|-------|------|----------|-------|-----------|
| o_proj (2048x2048) | **WMMA GEMM** | 32 | 8 | 30.4 us | 52.5% | **10.3%** |
| down_proj (2048x6144) | **WMMA GEMM** | 32 | 8 | 76.2 us | 58.9% | **10.4%** |
| lm_head (151936x2048) | **gemv2T** | 128 | 18992 | 1086 us | **95.9%** | 65.7% |

关键对比：

- **o_proj/down_proj**: cuBLAS 启动了一个仅 8 个 block、每 block 32 线程的 WMMA GEMM kernel，用了 120 个寄存器/线程。grid=8 在 72 个 SM（A10）上只占了 11% 的 SM，DRAM 带宽利用率仅 52-59%，occupancy 仅 10%。这是明显的 kernel 选择错误——M=1 不需要 Tensor Core GEMM。
- **lm_head**: cuBLAS 正确选择了 gemv2T 专用 kernel（grid=18992），DRAM 利用率高达 95.87%，已接近理论最优。

### 5.3 旧 Triton matvec 的问题

NCU 对比显示旧 Triton matvec (BLOCK_M=64, grid=32) 比 cuBLAS 还差：

| 指标 | cuBLAS WMMA | 旧 Triton matvec | 新 Triton v2 |
|------|------------|-----------------|-------------|
| Grid | 8 | 32 | 512 |
| DRAM throughput | 52.5% | 19.1% | **63.3%** |
| Occupancy | 10.3% | 8.3% | **57.7%** |
| GPU cycles | 31,236 | 53,710 | **22,648** |
| Registers/thread | 120 | 112 | **30** |

旧 Triton matvec BLOCK_M=64 导致 grid 太小(32)，同时 112 个寄存器/线程严重限制了 occupancy。

### 5.4 新 Triton v2 kernel 设计思路

根据 NCU 分析，设计了高并行度的 GEMV kernel：

```
核心变化：
- BLOCK_M: 64 → 1~4  (更高 grid 并行度)
- Grid:    32 → 512~2048  (充分占满 SMs)
- Regs:    112 → 30  (高 occupancy)
- 策略:    每个 block 只处理 1~4 行输出，沿 K 维 512 步长迭代
```

kernel 代码（已集成到 `optimized_layer.py`）：

```python
@triton.jit
def _decode_gemv_fwd_kernel(
    X, W, OUT, K: tl.constexpr, N, stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start_k in range(0, K, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        x = tl.load(X + offs_k).to(tl.float32)
        w_ptrs = W + offs_m[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w = tl.load(w_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(OUT + offs_m, acc.to(OUT.dtype.element_ty), mask=mask_m)
```

最优配置（CUDAGraph 下 sweep 确认）：`BLOCK_M=1, BLOCK_K=512, num_warps=4, num_stages=2`

### 5.5 Python dispatch 是关键盲点

在 eager 模式下，所有 Triton 配置都给出 ~45us（0.53x cuBLAS），看起来没有改善。但 NCU 显示 GPU 级别时间：

```
Triton v2 GPU: 22,648 cycles ≈ 13.4 us
cuBLAS GPU:    31,236 cycles ≈ 18.4 us
```

差值 ~29us 正好是 Triton Python dispatch 的开销。**CUDAGraph 消除了这个开销后，Triton v2 才显示出真实优势。**

---

## 6. CUDAGraph 模式验证（2026-03-29）

### 6.1 微基准结果

在 CUDAGraph 下测量 Triton v2 vs cuBLAS（A10）：

| 目标 | cuBLAS (F.linear) | Triton v2 | 加速比 |
|------|-------------------|-----------|--------|
| o_proj (2048x2048) | 22.60 us | **17.26 us** | **1.31x** |
| down_proj (2048x6144) | 59.62 us | **47.67 us** | **1.25x** |

有效带宽（从 CUDAGraph wall-clock 计算）：

| 目标 | 权重大小 | Triton 有效带宽 | A10 利用率 |
|------|---------|---------------|-----------|
| o_proj | 8 MB | 486 GB/s | **81.0%** |
| down_proj | 24 MB | 528 GB/s | **88.0%** |

### 6.2 端到端 benchmark（A10, 10 样本）

| 配置 | TTFT | Throughput | 结果文件 |
|------|------|------------|---------|
| GEMV OFF (cuBLAS) | 118.24 ms | 94.59 tok/s | [results/result_step55_gemv_off_10.json](../results/result_step55_gemv_off_10.json) |
| **GEMV ON (Triton v2)** | 118.62 ms | **99.40 tok/s** | [results/result_step55_gemv_on_10.json](../results/result_step55_gemv_on_10.json) |

- 吞吐提升：**+5.1%**
- 输出一致性：**10/10 exact match**
- TTFT 基本不变

### 6.3 lm_head 不做替换

NCU 确认 cuBLAS 对 lm_head 已正确选择 gemv2T kernel，DRAM 利用率 95.87%。没有优化空间。

---

## 7. 集成方案

代码集成到 `optimized_layer.py`，通过环境变量 `ENABLE_TRITON_SMALL_DECODE_GEMV` 控制（默认 `1` 开启）：

```python
def _triton_decode_gemv(x, weight, bias, prefer_small_triton=False):
    # 条件: prefer_small_triton=True, CUDA tensor, x.shape[-2]==1 (单 token decode)
    # 不满足条件返回 None，caller fallback 到 F.linear
    # ...
    BLOCK_M, BLOCK_K = 1, 512
    _decode_gemv_fwd_kernel[(triton.cdiv(N, BLOCK_M),)](
        x_vec, weight, out_vec, K, N,
        weight.stride(0), weight.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
```

替换位置：

- `OptimizedDecoderLayer.forward()` 的 o_proj 调用（约第 187 行）
- `OptimizedDecoderLayer.forward()` 的 down_proj 调用（约第 213 行）
- `__init__()` 新增 `self.o_proj_weight/bias` 和 `self.down_proj_weight/bias` 引用

A800 验证计划：[docs/a800_gemv_validation.md](a800_gemv_validation.md)

---

## 8. Roofline 分析：进一步优化空间

### 8.1 当前效率

从 CUDAGraph wall-clock 反推 Triton v2 的有效带宽已达 A10 峰值的 81-88%：

```
o_proj:    8 MB / 17.26 us = 486 GB/s → 81% of A10 peak (600 GB/s)
down_proj: 24 MB / 47.67 us = 528 GB/s → 88% of A10 peak (600 GB/s)
```

NCU 报告的 63% DRAM throughput 是 BLOCK_M=4 版本的指标；实际部署的 BLOCK_M=1 有更高的有效带宽。

### 8.2 A800 理论空间

| 场景 | o_proj/layer | down_proj/layer | 28层合计/token | vs Triton@~80% 节省 |
|------|-------------|----------------|---------------|-------------------|
| 理想 100% | 4.11 us | 12.34 us | 461 us | +137 us |
| Triton@~80% | 5.14 us | 15.43 us | 576 us | baseline |
| **CUDA@85%** | 4.84 us | 14.52 us | 542 us | +56 us |
| **CUDA@90%** | 4.57 us | 13.71 us | 512 us | +86 us |

端到端影响（decode step ~4.05 ms）：

| 优化 | GEMV 耗时/token | decode step | 吞吐提升 |
|------|----------------|------------|---------|
| 当前 Triton@80% | 0.576 ms | 4.05 ms | baseline |
| CUDA@85% | 0.542 ms | 4.02 ms | **+1.4%** |
| CUDA@90% | 0.512 ms | 3.99 ms | **+2.2%** |

**结论：纯 GEMV kernel 优化的端到端空间约 1-3%，性价比有限。**

### 8.3 A800 tail occupancy 问题

BLOCK_M=1 在 A800 上有明显的 wave tail waste：

```
A800: 108 SMs, max concurrent blocks = 16/SM × 108 = 1728
o_proj BLOCK_M=1: grid = 2048
  Wave 1: 1728 blocks (100% 利用率)
  Wave 2: 320 blocks (18.5% 利用率)
  平均利用率: 59.3%
```

CUDA persistent kernel 可以消除这个问题，但对总体性能影响有限（~1-2%）。

---

## 9. 额外尝试：`lm_head` 后处理小优化

还试过一个很便宜的小改动：

- graph decode 中不再先把 `outputs.logits` copy 到 `_logits_buf`
- 直接对 `outputs.logits[:, -1, :]` 做 `argmax`

对应结果：

- [results/result_step40_direct_argmax_10.json](../results/result_step40_direct_argmax_10.json)

结果是：

- step39 基线: `194.24 tok/s`
- direct argmax: `191.71 tok/s`

结论：

- 这条线没有收益，已经回退

---

## 10. 已证伪方向汇总

### 10.1 方法论层面

1. **简单 API 替换** — `F.linear` → `matmul(weight.t())` / `contiguous weight`：无收益
2. **Eager 模式 Triton 替换 cuBLAS** — Python dispatch 开销 (~29us/call) 完全吃掉 GPU 级收益
3. **简单 split-K Triton matvec** — 对所有目标都不如基线
4. **`torch.mv` / `torch.addmv`** — 底层调用同一个 cuBLAS kernel，无收益

### 10.2 已确认有效

1. **高并行度 Triton GEMV + CUDAGraph** — o_proj 1.31x, down_proj 1.25x, 端到端 +5.1%
2. **根因诊断**：cuBLAS 对小 N 的 M=1 GEMV 选择 WMMA GEMM 是性能瓶颈源头
3. **lm_head cuBLAS gemv2T 已最优** — 95.87% DRAM 利用率，无需替换

---

## 11. 后续方向与优先级

### 11.1 CUDA GEMV kernel（收益 1-3%，工程量中）

如果要写 CUDA 替代当前 Triton，主要优化点：

1. **float4 向量化加载**: 用 `__ldg` + `uint4` 保证 128 字节合并访问
2. **Shared memory 缓存 x 向量**: 每 block 加载 x (4KB) 到 shared memory 一次，避免重复 L2 访问
3. **Warp shuffle reduction**: `__shfl_xor_sync` 替代 Triton 的 `tl.sum`
4. **Persistent kernel**: 固定 `num_SMs` 个 block，消除 tail occupancy 问题

预估收益：从当前 ~80% 带宽提到 ~85-90%，端到端 +1-3%。**性价比有限。**

### 11.2 GEMV + epilogue 融合（收益 2-5%，工程量中高）

把 bias + residual add 融合进 GEMV 内部：

```
当前流程: GEMV → store → load → add bias → store → load → residual add → store
融合后:   GEMV → bias → residual add → store (一次写)
```

对 batch=1 来说 activation 只有 4-12 KB，HBM 节省不大。但 28 层 × 2 算子 = 56 次 launch 的节省有意义。

### 11.3 更高价值方向（非纯 GEMV）

参考 roofline 各算子的绝对 gap 排序：

| 方向 | 绝对 gap/token | 占总 gap 比 | 推荐 |
|------|---------------|-----------|------|
| attention 并行度 | 0.63 ms | 27% | 最高优先级 |
| elementwise/misc 压缩 | 0.44 ms | 19% | 值得做 |
| down_proj GEMV | 0.45 ms | 19% | 已做（Triton v2） |
| gate_up GEMM | 0.29 ms | 13% | cuBLAS 已接近最优 |
| qkv GEMM | 0.19 ms | 8% | cuBLAS 已接近最优 |
| o_proj GEMV | 0.14 ms | 6% | 已做（Triton v2） |

**o_proj/down_proj 的 Triton v2 GEMV 已经把 GEMV 从"高优先级"变成了"中低优先级"。** 继续投入 CUDA 写 GEMV kernel 的边际收益有限，更应该转向 attention 并行度和 elementwise 压缩。

### 11.4 综合判断

| 方向 | 预估收益 | 工程量 | 推荐 |
|------|---------|--------|------|
| 纯 CUDA GEMV kernel | +1-3% | 中 | 性价比低 |
| GEMV + epilogue 融合 | +2-5% | 中高 | 可选 |
| Persistent GEMV | +1-3% | 高 | 不推荐 |
| **attention 并行度** | **+5-10%** | **高** | **最高优先级** |
| **elementwise/misc** | **+3-5%** | **中** | **值得做** |

---

## 12. 结论

### 12.1 已完成

- NCU 确认了 cuBLAS 对小 N GEMV 的 kernel 选择问题（WMMA GEMM 而非 gemv2T）
- 新 Triton v2 GEMV kernel 在 CUDAGraph 下实现了 1.25-1.31x 单 kernel 加速，端到端 +5.1%
- 有效带宽已达 A10 峰值的 81-88%，接近 memory-bound roofline

### 12.2 当前状态

- `ENABLE_TRITON_SMALL_DECODE_GEMV=1` 默认开启
- o_proj 和 down_proj 在 decode 单 token + CUDAGraph 路径使用 Triton v2
- lm_head 保持 cuBLAS gemv2T（已最优）
- A800 验证待做（见 [docs/a800_gemv_validation.md](a800_gemv_validation.md)）

### 12.3 一句话结论

**GEMV 优化已从"高优先级热点"变为"接近 roofline 的已优化项"。** 当前 Triton v2 达到 81-88% 有效带宽利用率，继续投入 CUDA 只能再挤 1-3%。后续精力应转向 attention 并行度（gap 最大）和 elementwise 压缩。
