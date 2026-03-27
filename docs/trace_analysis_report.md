# Trace Profiling 分析报告

## 测试环境

- GPU: NVIDIA A800 80GB PCIe
- 模型: Qwen3-VL-2B-Instruct (28 decoder layers)
- 测试条件: 去除 warmup 和 verify，仅保留性能评测部分
- Trace 文件: 12 个 generate_*.json, 共 1.57 GB
- 分析工具: `analyze_traces.py`

## 样本概况

| 文件 | 类型 | Decode Steps | GPU(ms) | CPU(ms) | CPU/GPU |
|------|------|-------------|---------|---------|---------|
| generate_1 | TTFT | 1 | 106 | 232 | 2.2x |
| generate_3 | TTFT | 1 | 106 | 232 | 2.2x |
| generate_5 | TTFT | 1 | 165 | 395 | 2.4x |
| generate_7 | TTFT | 1 | 83 | 174 | 2.1x |
| generate_9 | TTFT | 1 | 84 | 173 | 2.1x |
| generate_11 | TTFT | 1 | 107 | 237 | 2.2x |
| generate_2 | Throughput | 72 | 893 | 5137 | 5.8x |
| generate_4 | Throughput | 3 | 128 | 364 | 2.8x |
| generate_6 | Throughput | 128 | 1599 | 9215 | 5.8x |
| generate_8 | Throughput | 128 | 1493 | 8795 | 5.9x |
| generate_10 | Throughput | 53 | 658 | 3705 | 5.6x |
| generate_12 | Throughput | 42 | 567 | 3068 | 5.4x |

关键观察: 所有样本均有 24 次 FlashAttnVarlenFunc (视觉编码器), decode 阶段每步 28 次 FlashAttnFunc (28 decoder layers).

---

## 一、TTFT (Prefill 阶段) 分析

**平均每样本: GPU 108.5 ms, CPU 240.2 ms, CPU/GPU = 2.2x**

### GPU Kernel 时间分布

| 类别 | 时间(ms) | 占比 |
|------|---------|------|
| GEMM (prefill large) | 55.0 | 50.7% |
| Elementwise | 30.5 | 28.1% |
| FlashAttn | 13.0 | 12.0% |
| Cat/Concat | 4.3 | 4.0% |
| Reduction | 1.9 | 1.8% |
| Norm | 1.2 | 1.1% |
| GEMV | 1.1 | 1.0% |
| Conv (Vision) | 1.0 | 0.9% |

### CPU 开销时间分布

| 类别 | 时间(ms) | 占比 | 说明 |
|------|---------|------|------|
| **item() sync** | **77.8** | **32.4%** | GPU->CPU 同步, 每次强制等待 GPU 完成 |
| FlashAttn dispatch | 41.6 | 17.3% | FA 的 CPU 端调度远超 GPU 端计算 |
| Linear/MatMul dispatch | 31.7 | 13.2% | Python/PyTorch 调度开销 |
| Type conv/copy | 28.7 | 12.0% | fp16<->fp32 类型转换 |
| Other | 28.9 | 12.0% | 其他小型 ops |
| Math/RMSNorm ops | 17.7 | 7.4% | pow+mean+rsqrt 分开的 RMSNorm |
| is_nonzero sync | 9.0 | 3.7% | 另一种 GPU sync |
| Cat/Arange/Sum | 4.8 | 2.0% | |

**TTFT 关键瓶颈:** `item()` 同步 (77.8ms) + `is_nonzero` sync (9.0ms) 占 36%, 这些是 GPU->CPU 数据同步, 完全阻塞 GPU pipeline.

---

## 二、Throughput (Decode 阶段) 分析

**每步平均: GPU 12.53 ms, CPU 71.09 ms, CPU/GPU = 5.7x**

| 指标 | 数值 |
|------|------|
| 理论最大吞吐 (GPU only) | ~80 tok/s |
| 实际估算吞吐 (CPU bound) | ~14 tok/s |
| 每步 kernel launch | 1791 次, 8.26 ms |
| 每步 stream sync | 32 次, 1.22 ms |

### GPU Kernel 每步时间分布

| 类别 | 时间(ms) | 占比 | 说明 |
|------|---------|------|------|
| GEMM (decode small) | 5.30 | 42.3% | cutlass WMMA, avg 32-48us/call, decode 阶段 linear 层 |
| Elementwise | 2.88 | 23.0% | Mul/Add/Copy/SiLU 等 |
| GEMV | 1.65 | 13.1% | gemv2T, decode 阶段的 vector-matrix 乘法 |
| GEMM (prefill large) | 0.77 | 6.2% | prefill 相关(可能是 cross-attention) |
| Reduction | 0.76 | 6.1% | reduce kernel |
| Cat/Concat | 0.58 | 4.6% | KV cache concat |
| FlashAttn | 0.56 | 4.4% | splitkv_kernel + combine |
| Norm | 0.02 | 0.1% | LayerNorm/RMSNorm GPU kernel |

### CPU 开销每步时间分布

| 类别 | 时间(ms) | 占比 | 说明 |
|------|---------|------|------|
| **Linear/MatMul dispatch** | **20.68** | **29.1%** | aten::linear -> matmul -> mm 调度链 |
| **Math/RMSNorm ops** | **13.60** | **19.1%** | pow+mean+rsqrt 各自独立 dispatch |
| **Type conv/copy** | **12.07** | **17.0%** | aten::to, _to_copy, copy_ |
| FlashAttn dispatch | 8.23 | 11.6% | FA CPU 侧调度 |
| Other | 8.66 | 12.2% | |
| Cat/Arange/Sum | 3.32 | 4.7% | |
| item() sync | 3.27 | 4.6% | |
| is_nonzero sync | 1.27 | 1.8% | |

**Decode 关键瓶颈:** CPU 调度开销 (71ms) 是 GPU 计算时间 (12.5ms) 的 5.7 倍. 每步 1791 次 kernel launch 是极其严重的碎片化问题.

---

## 三、全局汇总

### 按类别的总时间占比 (所有 trace 合并)

| Category | Count | Total(ms) | 占比 |
|----------|-------|-----------|------|
| FEE (Python函数) | 178,188 | 336,794 | 83.7% |
| cpu_op | 2,780,580 | 31,726 | 7.9% |
| Trace | 12 | 22,933 | 5.7% |
| kernel (GPU) | 838,656 | 5,990 | 1.5% |
| cuda_runtime | 924,052 | 4,605 | 1.1% |

### Top GPU Kernels

| Rank | Time(ms) | 占比 | Calls | Avg(us) | 类别 |
|------|---------|------|-------|---------|------|
| 1 | 1132 | 18.9% | 35,280 | 32.1 | GEMM decode (cutlass WMMA 128x2) |
| 2 | 1127 | 18.8% | 23,520 | 47.9 | GEMM decode (cutlass WMMA 128x1) |
| 3 | 476 | 7.9% | 161,052 | 3.0 | Elementwise Mul |
| 4 | 465 | 7.8% | 432 | 1077.3 | GEMV (gemv2T) |
| 5 | 409 | 6.8% | 198,386 | 2.1 | Elementwise (vectorized 4-wide) |

### Top CPU Operations

| Rank | Time(ms) | 占比 | Calls | Avg(us) | 名称 |
|------|---------|------|-------|---------|------|
| 1 | 3522 | 11.1% | 86,352 | 40.8 | aten::linear |
| 2 | 3019 | 9.5% | 85,536 | 35.3 | aten::matmul |
| 3 | 2413 | 7.6% | 85,104 | 28.4 | aten::mm |
| 4 | 2104 | 6.6% | 165,444 | 12.7 | aten::to |
| 5 | 1984 | 6.3% | 114,096 | 17.4 | aten::_to_copy |
| 6 | 1883 | 5.9% | 161,124 | 11.7 | aten::mul |
| 7 | 1879 | 5.9% | 12,096 | 155.4 | FlashAttnFunc |
| 8 | 1451 | 4.6% | 12,096 | 120.0 | flash_attn::_flash_attn_forward |
| 9 | 1362 | 4.3% | 112,044 | 12.2 | aten::add |
| 10 | 1228 | 3.9% | 116,460 | 10.5 | aten::copy_ |

---

## 四、核心结论

### 系统状态: 严重 CPU-bound

GPU 利用率极低. GPU kernel 总计 ~6s, 但 CPU 调度+同步总计 ~32s. 整个推理流程大部分时间 GPU 在空闲等待 CPU.

### 性能瓶颈优先级排序

#### P0: CPU Dispatch 开销 (decode 每步 ~60ms, 占 CPU 时间 85%)

- `aten::linear` 调度链 (linear -> matmul -> mm) 每步 20.7ms
- `aten::to/_to_copy/copy_` 类型转换每步 12.1ms
- `Math/RMSNorm` (pow+mean+rsqrt 各自独立 dispatch) 每步 13.6ms
- FlashAttn CPU 调度每步 8.2ms

**根因**: PyTorch eager mode 下每个 op 都需要经过 Python -> C++ dispatcher -> CUDA launch 的完整路径.
每步 1791 次 kernel launch, 每次 avg 4.6us launch overhead.

#### P1: GPU-CPU 同步点 (TTFT 中占 36%)

- `aten::item()`: TTFT 中 77.8ms (32.4%), 87 次/样本
- `aten::is_nonzero`: TTFT 中 9.0ms (3.7%), 34 次/样本
- 每次 item() 调用强制 CPU 等待 GPU 完成所有 pending work

#### P2: Kernel 碎片化

- 每个 decode step 有 1791 次独立 kernel launch
- 大量小 kernel (elementwise avg 2-3us, 但 launch overhead 4.6us)
- 相当于 "launch 一个 kernel 比 kernel 本身执行还贵"

---

## 五、优化建议

### 1. torch.compile (预期收益: decode 吞吐 3-5x)

一次性解决 CPU dispatch, kernel fusion, type conversion 问题.
将大量碎片化小 kernel fuse 成少量大 kernel, 大幅减少 launch overhead.

### 2. 消除 item() / is_nonzero 同步 (预期收益: TTFT -30%)

定位代码中的 `.item()` 和 `is_nonzero` 调用, 用 tensor 操作替代.
这些通常出现在 generation loop 的 stopping criteria 和 logits processing 中.

### 3. CUDA Graph for Decode (预期收益: decode 吞吐 2-3x)

对 decode 阶段的固定计算图使用 CUDA Graph 一次捕获、多次 replay.
消除每步 1791 次 kernel launch 的开销.

### 4. Fused RMSNorm (预期收益: decode 每步 -10ms CPU)

当前 RMSNorm 拆分成 pow + mean + rsqrt 三个独立 op.
用 fused Triton/CUDA kernel 替换, 减少 3x dispatch + 3x kernel launch.

### 5. 减少 aten::to/_to_copy (预期收益: decode 每步 -8ms CPU)

检查是否存在不必要的 fp16<->fp32 转换.
确保模型全程在 fp16/bf16 下运行, 避免中间类型提升.

### 6. KV Cache 预分配

当前存在 Cat/Concat 操作 (每步 0.58ms GPU, 3.3ms CPU), 可能是动态拼接 KV cache.
预分配固定大小 KV cache, 用 index_copy_ 替代 cat.
