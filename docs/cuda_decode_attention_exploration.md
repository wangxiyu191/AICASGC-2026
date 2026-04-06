# CUDA Decode Attention Kernel 探索全记录

本文详细记录从 Triton split-KV decode attention 迁移到手写 CUDA kernel 的完整探索过程，包括每个版本的设计动机、架构选择、性能结果，以及过程中积累的经验教训。

## 1. 背景与目标

### 1.1 问题定义

在 AICASGC 2026 比赛中，需要优化 Qwen3-VL-2B-Instruct 在单张 A800 80GB PCIe (sm_80) 上的推理性能。Decode 阶段的 attention 是关键算子之一。

模型配置：

| 参数 | 值 |
|------|-----|
| num_q_heads | 16 |
| num_kv_heads | 8 |
| num_key_value_groups | 2 |
| head_dim | 128 |
| batch_size | 1 |
| q_len (decode) | 1 |
| dtype | float16 |

评测场景：单 batch decode，典型 valid_len 约 600-900 tokens，max_cache_len = 832。

### 1.2 为什么要写 CUDA kernel

在 Triton 侧已经做了大量优化（step 31→36: 去 repeat_kv、grouped-GQA、cache_position prefix derivation、launch config tuning；step 45→53: split-KV workspace 持久化、packed stats、online merge），Triton 版 split-KV attention 已经相当成熟。

写 CUDA 的目标是进一步逼近或超越 Triton 编译器的指令调度能力，在不依赖 Triton 的环境下也能部署。

### 1.3 基准线

CUDAGraph capture + replay，real data，sample0/layer14/step36，valid_len=691，max_cache_len=832：

- **Triton split-KV**：0.0197ms（A800 上的稳定基准）

## 2. Triton 侧的演进简述

作为 CUDA 探索的前导背景，简要列出 Triton attention 的关键迭代：

| Step | 优化点 | 关键结果 |
|------|--------|---------|
| 31 | 首版 Triton single-token decode attention，去 repeat_kv | steady-state decode GPU/步 6.04→5.40ms |
| 33 | Grouped-GQA：1 program 处理 1 kv_head + 2 q_head | microbench 0.0971→0.0693ms |
| 35 | cache_position prefix derivation，不读 bool mask 尾部 | microbench 再快 8-17% |
| 36 | Launch config tuning (BLOCK_N/warps/stages bucket) | attention 1.47→1.01ms/步 |
| 45-53 | split-KV：workspace 持久化、packed stats、online merge | A10 上 throughput +4.1% |

到此 Triton 侧基本挖空，进入 CUDA 手写阶段。

## 3. CUDA v1：朴素 warp-sum baseline

**对应实验**：result_experiments.csv step 62

**文件**：`cuda_extensions/decode_attention_kernel.cu` — `grouped_prefix_cachepos_splitkv_fwd_kernel`

### 3.1 设计

最直接的翻译：

- **Grid**：`(num_kv_heads, split_k)` — 每个 block 处理 1 个 kv_head 的 1 个 split chunk
- **Block**：64 threads = 2 warps，每个 warp 处理同组 2 个 q_head 中的 1 个
- **数据布局**：lane 覆盖 4 dims（lane × 4），`head_dim=128` 恰好 32 lanes × 4 dims
- **QK dot product**：`warp_sum()` 做 5 轮 `__shfl_down_sync` 归约，然后 `__shfl_sync(0, 0)` 广播
- **Online softmax**：标准 FlashAttention 流式 (m, l, acc) 更新，逐 token 处理
- **V 累加**：与 QK 在同一 token 循环内完成，每次都要做 rescale

### 3.2 迭代过程

1. **首版 block-wide reduction**：0.0791ms，比 Triton 慢 ~5.5x
2. **改 warp-specialized**（去掉跨 warp 归约）：0.0421ms
3. **扫 split_k**：split_k=4/8/16/24/32/48/64 → 0.0775/0.0419/0.0255/0.0210/0.0194/0.0191/0.0203ms，最佳 split_k≈48 → **0.0191ms**

### 3.3 结论

v1 best = 0.0191ms，但这是在高 split_k(48) 下取得的，仍比 Triton 0.0145ms 慢约 **38%**（注：当时 A800 Triton 基准与后续略有差异，因为不同 sample/step 下的 valid_len 不同）。

**瓶颈分析**：Grid 只有 `num_kv_heads(8) × split_k`，split_k 过高后 merge 开销上升，且 warp_sum 每 token 要做 5 轮 shfl reduction。

## 4. CUDA v1-qhead：Q-head 粒度 grid

**对应实验**：result_experiments.csv step 63

**文件**：`cuda_extensions/decode_attention_kernel.cu` — `grouped_prefix_cachepos_splitkv_qhead_fwd_kernel`

### 4.1 设计改动

核心变化只有一个：把 grid 从 `num_kv_heads × split_k` 改成 `num_q_heads × split_k`。

- **Grid**：`(num_q_heads=16, split_k)` — 每个 block 只处理 1 个 q_head
- **Block**：32 threads = 1 warp，不再需要 warp 内区分两个 q_head
- 其余逻辑不变

### 4.2 结果

- 同 case 对比：v1-qhead ≈ 0.0198ms vs v1 ≈ 0.0421ms，快约 2.1x
- split_k sweep：split_k=8/16/24/32/48/64/96 → 0.0431/0.0267/0.0217/0.0197/0.0201/0.0211/0.0256ms
- 最佳 split_k≈32 → **0.0197ms**

### 4.3 结论

Grid 做大确实是对的方向，但仍比 Triton 慢 **35%**。warp_sum 的 5 轮 reduction 和逐 token 的 QKV 联合处理仍然是性能瓶颈。

## 5. CUDA v3：thread-group=4 + shared memory

**说明**：v2 是中间过渡版本（实验性质，未留完整记录），v3 是 thread-group 路线的第一个成型版本。

### 5.1 设计思路

从 v1 的"1 warp = 1 q_head, lane 覆盖 4 dims"跳到一个完全不同的线程组织：

- **Thread-group=4**：每 4 个线程组成 1 个 group，处理 1 个 token 的 dot product
  - 每个 thread 负责 `head_dim / 4 = 32` dims
  - dot product 归约只需 2 轮 shfl（`xor 1`, `xor 2`），比 warp_sum 的 5 轮少很多
- **8 groups/warp**：1 个 warp 内 8 个 group 同时处理 8 个 token → 更好的 ILP
- **2 warps/CTA**：对应 2 个 q_head（grouped-GQA, num_key_value_groups=2）
- **Grid**：`(num_kv_heads=8, split_k)`

### 5.2 scores 存储：shared memory

v3 的 scores 存在 shared memory 中：

- 每个 warp 在 smem 里维护 scores 数组
- 需要 `__syncthreads()` 做 warp 间同步
- Cross-group max 通过 smem reduction 完成

### 5.3 结果

- CUDAGraph formal benchmark：**v3 = 0.0225ms**
- 比 v1-qhead(0.0197ms) 慢，但 v3 的 grid 更小（8×split_k vs 16×split_k）
- **比 Triton 0.0197ms 慢 14.2%**

### 5.4 分析

v3 走对了 thread-group=4 这个方向（QK reduction 轮数从 5 降到 2），但 shared memory + syncthreads 的同步开销和 smem bank conflict 限制了进一步提升。

## 6. CUDA v4-shfl：寄存器 + cross-group shfl（pre-tiled 版本）

### 6.1 核心改进

相比 v3，v4 的第一版（pre-tiled）做了两个关键变化：

1. **去掉 shared memory**：scores 完全存在寄存器中
2. **Cross-group 归约用 shfl**：3 轮 `shfl_xor(4, 8, 16)` 完成 8 个 group 的 max 归约

保持 thread-group=4 的基本布局不变。

### 6.2 问题：rescale 频率过高

pre-tiled 版本的处理流程是每处理 8 个 token（1 个 warp step）就做一次 rescale：

```
for each 8-token step:
    compute 8 scores (each group handles 1 token)
    cross-group max via shfl
    rescale acc *= exp(old_m - new_m)  ← 每 8 token 做一次
    accumulate V
```

这意味着对于 64 个 token，要做 8 次 rescale。每次 rescale 要对 32 个 float 寄存器做 `*= alpha`，开销显著。

### 6.3 结果

- **v4 shfl (pre-tiled) = 0.0212ms**
- 比 v3 的 0.0225ms 快一些，但改善幅度有限

## 7. CUDA v4-tiled：最终最优版本

**对应实验**：result_experiments.csv step 134

**文件**：`cuda_extensions/decode_attention_v4_kernel.cu`

### 7.1 核心设计：tiled two-pass

这是整个探索过程中最关键的设计突破。将每个 64-token tile 的处理分成两个 pass：

**Pass 1 (K pass)**：
```
for step in 0..7:                          // 8 steps × 8 groups = 64 tokens
    load K[token], compute QK dot product
    2-round shfl reduce within group
    store score in local_scores[step]       // register array, 8 slots
    update tile_max = max(tile_max, score)

// 3-round shfl_xor(4,8,16) → cross-group max
tile_max = global_max_across_all_groups
```

**Rescale（每 tile 仅 1 次）**：
```
alpha = exp(old_m - new_m)
acc[0..31] *= alpha                         // 32 floats
l *= alpha
m = new_m
```

**Pass 2 (V pass)**：
```
for step in 0..7:
    p = exp(local_scores[step] - m)
    load V[token]
    acc[i] += p * V[i]                      // weighted accumulate
    l += p
```

### 7.2 为什么 tiled two-pass 更快

关键在于 **rescale 频率**：

| 版本 | rescale 频率 | 64 tokens 的 rescale 次数 |
|------|-------------|--------------------------|
| v1 (warp_sum) | 每 1 token | 64 |
| v4-shfl (pre-tiled) | 每 8 token | 8 |
| **v4-tiled** | **每 64 token** | **1** |
| Triton split-KV | 每 BLOCK_N token | 1 (BLOCK_N=64~128) |
| v3 | 每 tile | 1 |

v4-tiled 的 rescale 频率与 Triton / v3 对齐，而且做到了：

- **零 shared memory**：REG=123, SHARED=0, STACK=0
- **零 syncthreads**：不需要 warp 间同步
- Cross-group 通信完全通过 shfl 完成

### 7.3 内核参数

```
Constants:
  kHeadDim = 128
  kThreadGroup = 4
  kGroupsPerWarp = 8      (32/4)
  kDimsPerThread = 32     (128/4)
  kTileSteps = 8
  kTileN = 64             (8 groups × 8 steps)

Forward kernel:
  Grid: (num_kv_heads=8, split_k)
  Block: 64 threads (2 warps)
  __launch_bounds__(64)
  REG=123, SHARED=0

Merge kernel:
  Grid: (num_kv_heads=8,)
  Block: 128 threads
  __launch_bounds__(128)
```

### 7.4 寄存器压力分析

v4 forward kernel 每个线程的主要寄存器占用：

| 变量 | 寄存器数 |
|------|---------|
| q_reg[32] | 32 |
| acc[32] | 32 |
| local_scores[8] | 8 |
| m_val, l_val | 2 |
| K/V load 临时 (uint4 + half) | ~10 |
| dot, score, alpha 等标量 | ~5 |
| 地址计算 | ~10 |
| **合计** | **~99** (nvcc 报告 123) |

123 regs 在 sm_80 上理论 occupancy 约 25%，与 Triton 的 168 regs / 18.75% occupancy 同级，说明这个 kernel 确实是 ILP-bound 而非 occupancy-bound。

### 7.5 性能结果

CUDAGraph formal benchmark (real data, valid_len=691, max_cache_len=832)：

| 变体 | 延迟 (ms) | 相对 Triton |
|------|----------|------------|
| Triton split-KV | 0.0197 | baseline |
| **v4 tiled** | **0.0207** | **+5.1%** |
| v3 smem | 0.0225 | +14.2% |
| v1-qhead | 0.0197~0.0421 | +0%~+114% (取决于 split_k) |

split_k sweep 显示 v4 在所有 split_k 和 seq_len 组合下一致优于 v3，快 8-18%。

### 7.6 剩余 5.1% gap 归因

1. **nvcc vs Triton 编译器的指令调度差异**：Triton 通过 MLIR→LLVM IR→PTX 的编译链，能做更激进的指令交错和 latency hiding
2. **Cross-group merge 开销**：3 轮 shfl × 32 dims 的 merge 在每个 tile 结尾引入约 96 条额外指令
3. **Thread-group=4 布局的 load 指令数**：每 token 的 K/V 需要 4 个 thread 各发 4 次 128-bit load（uint4），共 16 次 load；而 Triton 的 coalesced block-level load 可能更高效

## 8. CUDA v5：dim-parallel（匹配 Triton 线程布局）

**对应实验**：result_experiments.csv step 135

### 8.1 设计动机

既然 v4 最后 5% gap 可能源自线程布局差异，v5 尝试直接匹配 Triton 的 dim-parallel 布局：

- **row_id = lane / 16**：2 行，每行处理 1 个 token
- **col_id = lane % 16**：16 列，每列处理 8 dims
- 1 个 warp step 处理 2 tokens（vs v4 的 8 tokens）
- 32 tokens per tile
- 每 thread 在寄存器中存 16 个 scores

### 8.2 结果

- 数值验证：通过（误差在容差内）
- **性能：比 v4 慢 60%+**
- REG=64, SHARED=0

### 8.3 失败原因分析

| 维度 | v4 (thread-group=4) | v5 (dim-parallel) |
|------|--------------------|--------------------|
| QK reduction 轮数 | 2 (xor 1, xor 2) | 4 (xor 1, 2, 4, 8) |
| Tokens/warp step | 8 | 2 |
| Dims/thread | 32 | 8 |
| ILP (tokens in flight) | 8 | 2 |

根本原因：

1. **Reduction 开销**：thread-group=16 需要 4 轮 shfl reduction，是 thread-group=4 的 2 倍
2. **ILP 不足**：每 step 只处理 2 token（v4 处理 8 token），内存请求的交错（interleaving）远不够掩盖延迟
3. **nvcc 的局限**：Triton 编译器能在 MLIR 层面做更激进的 software pipelining 和指令重排；nvcc 对手写 CUDA 的优化空间受限于程序员写出的循环结构

### 8.4 关键教训

**nvcc 编译下，thread-group=4 布局在指令效率上显著优于 dim-parallel 布局，即使后者有更好的全局内存合并（coalescing）特性。** 这说明在 decode attention 这种 memory-latency-bound 的 kernel 中，ILP（通过更多 tokens in flight）比 memory coalescing 更重要。

## 9. 版本演进总览

```
v1 (warp_sum)
  ├─ 1 warp = 1 q_head, lane covers 4 dims
  ├─ 5-round shfl reduction per token
  ├─ 0.0191ms (split_k=48), 38% slower than Triton
  │
  └─ v1-qhead (grid: num_q_heads × split_k)
       ├─ 0.0197ms (split_k=32), 35% slower
       └─ 瓶颈: warp_sum + 逐 token rescale

v3 (thread-group=4 + smem)
  ├─ 4 threads/group, 8 groups/warp, 8 tokens in flight
  ├─ 2-round shfl reduction per token
  ├─ scores in shared memory + __syncthreads
  ├─ 0.0225ms, 14.2% slower
  └─ 瓶颈: smem bank conflict + sync overhead

v4-shfl (thread-group=4 + register scores)
  ├─ 去 smem, 去 syncthreads
  ├─ cross-group max via 3-round shfl_xor
  ├─ 0.0212ms
  └─ 瓶颈: 每 8 token rescale 一次，过频

v4-tiled (thread-group=4 + 64-token tile two-pass) ★ BEST
  ├─ K pass: compute all 64 scores → register array
  ├─ ONE rescale per tile
  ├─ V pass: weighted accumulate
  ├─ REG=123, SHARED=0
  ├─ 0.0207ms, 5.1% gap to Triton
  └─ 当前最优 CUDA kernel

v5 (dim-parallel, matching Triton layout)
  ├─ row_id=lane/16, col_id=lane%16
  ├─ 2 tokens/step, 4-round shfl reduction
  ├─ 0.033ms+, 60%+ slower than v4
  └─ 方向已放弃
```

## 10. 性能对比汇总

以 CUDAGraph formal benchmark, real data, valid_len=691, max_cache_len=832 为基准：

| Kernel | 延迟 (ms) | 相对 Triton | REG | SMEM | Split-K | 关键特征 |
|--------|----------|------------|-----|------|---------|---------|
| Triton split-KV | 0.0197 | — | 168 | — | 16 | Triton 编译器优化 |
| CUDA v4 tiled | 0.0207 | +5.1% | 123 | 0 | 16 | thread-group=4, 64-tile two-pass |
| CUDA v3 smem | 0.0225 | +14.2% | ~100 | yes | 16 | thread-group=4, smem scores |
| CUDA v4 shfl | 0.0212 | +7.6% | ~110 | 0 | 16 | thread-group=4, per-8 rescale |
| CUDA v1-qhead | 0.0197 | +0%~35% | ~60 | 0 | 32-48 | warp_sum, q-head grid |
| CUDA v5 dim-par | 0.033+ | +60%+ | 64 | 0 | 16 | dim-parallel, 4-round reduction |

## 11. 经验与收获

### 11.1 Thread 布局选择

**结论：thread-group=4 > dim-parallel (for nvcc)**

- thread-group=4：每 group 4 线程 × 32 dims/thread，QK 只需 2 轮 shfl，8 tokens in flight
- dim-parallel：每 token 16 线程 × 8 dims/thread，QK 需 4 轮 shfl，仅 2 tokens in flight
- Triton 的 dim-parallel 能赢，是因为 Triton 编译器在 MLIR 层面做了更深的 software pipelining；但 nvcc 面对手写 CUDA 时做不到同等程度的指令交错

**推论**：当你用 nvcc 写 hand-written CUDA kernel 时，优先选择能带来更多 ILP（更多 tokens in flight）的布局，而不是追求更好的内存合并。

### 11.2 Rescale 频率至关重要

从 v4-shfl 到 v4-tiled 的提升几乎完全来自一个改变：**把 rescale 从每 8 token 降到每 64 token**。

rescale 操作本身：
```c
float alpha = expf(m_val - new_m);
for (int i = 0; i < 32; ++i) acc[i] *= alpha;
l_val *= alpha;
```

看起来只是 33 次乘法 + 1 次 exp，但它在 register file 上形成了一个 long dependency chain，打断了 load-compute 的 pipeline。把 rescale 推迟到 64 token 边界，让 K pass 和 V pass 各自保持完整的 pipeline，效果显著。

### 11.3 ILP > Occupancy

| 变体 | REG | Theoretical Occupancy | 延迟 |
|------|-----|----------------------|------|
| Triton | 168 | 18.75% | 0.0197ms |
| v4 tiled | 123 | ~25% | 0.0207ms |
| v5 | 64 | ~50% | 0.033ms+ |

寄存器最少、occupancy 最高的 v5 反而最慢。这个 kernel 是典型的 **ILP-bound**：

- 工作量小（batch=1, q_len=1, valid_len~700）
- GPU 总 SM 有 108 个，但 grid 最多 8×16=128 blocks
- 每个 thread 需要大量寄存器来保持足够多的 token 同时 in flight
- 占用率低不是问题，单 warp 内的指令级并行才是关键

### 11.4 Shared Memory 不一定帮忙

v3 用 smem 存 scores 并做 cross-group reduction，结果比不用 smem 的 v4 慢。原因：

- smem 访问本身有延迟（~20 cycles on sm_80）
- bank conflict 在 strided access pattern 下难以完全避免
- `__syncthreads()` 引入 barrier stall
- 对于只有 64 线程的小 CTA，寄存器 + shfl 是更好的通信方式

### 11.5 nvcc vs Triton 编译器

5% 的最终 gap 很可能主要来自编译器差异，而非算法差异：

- Triton 通过 MLIR → LLVM IR → PTX 的多层优化管线，能做全局的指令调度和 software pipelining
- nvcc 对手写 CUDA 的优化虽然也不弱，但受限于程序员写出的循环结构和 pragma 提示
- 在 ILP-bound 场景下，**指令交错（instruction interleaving）的质量**直接决定性能，这恰恰是编译器优化的强项

### 11.6 Grid 粒度的选择

v1 的 `(num_kv_heads, split_k)` vs v1-qhead 的 `(num_q_heads, split_k)` 实验说明：
- 在 grouped-GQA 下，如果硬件并行度不足，把 grid 做到 q_head 粒度确实能带来收益
- 但 v3/v4 回到 `(num_kv_heads, split_k)` 后反而更好，因为 thread-group 布局让 CTA 内已经有足够的并行度
- Grid 粒度的最优选择取决于 CTA 内部的线程组织方式

## 12. 最终结论

1. **v4 tiled two-pass 是当前最优 CUDA decode attention kernel**，距 Triton 仅 5.1% 差距
2. 这 5% 的差距主要来自 nvcc vs Triton 编译器的指令调度能力差异，在手写 CUDA 侧已经接近理论极限
3. **在 Triton 可用的环境下，v4 不构成主线替换理由**；但在 Triton 不可用（如纯 CUDA 部署）的场景下，v4 是可靠的选择
4. 整个探索过程中最大的三个教训：
   - **thread-group 大小决定 ILP**：group=4 显著优于 group=16
   - **rescale 频率决定 pipeline 质量**：越少越好
   - **低 occupancy + 高 ILP > 高 occupancy + 低 ILP**

## 13. 保留文件

清理后仓库中保留的 CUDA attention 相关文件：

| 文件 | 说明 |
|------|------|
| `cuda_extensions/decode_attention_kernel.cu` | v1 baseline + v1-qhead kernel |
| `cuda_extensions/decode_attention_v4_kernel.cu` | v4 tiled two-pass kernel（最优） |
| `cuda_extensions/decode_attention_ext.cpp` | Pybind11 绑定 |
| `cuda_extensions/setup.py` | 编译配置 |
| `cuda_decode_attention.py` | Python wrapper + launch config |
| `bench_decode_attention.py` | Microbenchmark |

已删除的实验文件（v2/v3/v5 kernel 源码、各种 bench/test/profile 脚本共 18 个文件）的实验结果记录在 `result_experiments.csv` 中。
