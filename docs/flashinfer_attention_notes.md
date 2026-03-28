# FlashInfer Attention / Intra-kernel Trace 学习笔记

> 调研目标: 从 `./reference/flashinfer` 提炼对当前仓库后续 Attention 优化最有价值的实现技巧，以及可借鉴的“算子内 trace”方案。

## 1. 先说结论

FlashInfer 里最值得我们借鉴的，不是某一个单独 kernel 技巧，而是下面这套组合：

1. **decode attention 按 GQA group 专门做 kernel 特化**
2. **把 decode 也尽量路由到 tensor-core / prefill 风格的更强 kernel**
3. **把 metadata 规划和真正 run 分离，graph 场景再做一层 fast-plan**
4. **如果要看 kernel 内部阶段，不依赖 NVTX，而是内核自己写 timestamp 到 profiler buffer**

对我们当前仓库最直接的启发是：

- 现在已经做了“按 `kv_head + 2 q_head` 复用 K/V”的第一步，但仍然是 **scalar streaming softmax kernel**
- 下一步如果 Attention 还要继续提速，优先级最高的方向不是再抠零碎 copy，而是**想办法把 decode attention 推向更接近 FlashInfer `use_tensor_cores=True` 这类路线**
- 如果后面需要精确回答“Attention kernel 里面到底是 load / qk / softmax / pv 哪一段慢”，就该上 **FlashInfer 这种 profiler buffer + perfetto**，而不是继续依赖 graph 外部 trace

---

## 2. Attention 实现上学到的东西

### 2.1 GQA 不是逻辑开关，而是 kernel 维度

在 [`flashinfer/xqa.py`](./reference/flashinfer/flashinfer/xqa.py#L273) 里，`head_group_ratio = num_q_heads // num_kv_heads` 会直接进入 JIT module key；也就是说：

- `group_size` 不是 run 时普通参数
- 它是**参与 kernel 生成 / 特化**的核心维度

这和我们现在的判断一致：Qwen3-VL-2B 是 `16 q / 8 kv`，group size 固定就是 `2`，完全值得单独做专门实现。

对我们的启发：

- 当前仓库不需要追求“任意 GQA 比例的通用 decode kernel”
- 先把 `group_size=2, head_dim=128, batch=1, q_len=1` 这条主路径做深，比做通用实现更值

### 2.2 FlashInfer 的 decode 不只是一条“纯 decode kernel”

在 [`flashinfer/decode.py`](./reference/flashinfer/flashinfer/decode.py#L1043) 附近，`use_tensor_cores=True` 时，decode wrapper 实际会切到 `get_batch_prefill_module(...)` 这条路，而不是继续走普通 CUDA-core decode kernel。

这说明它们的思路是：

- **decode/prefill 在接口上分开**
- 但在 kernel 选型上，如果 GQA 比较大、shape 合适，decode 也可以借用更强的 prefill-style kernel

这对我们非常重要，因为我们当前 attention kernel 的主要限制正是：

- 单 token decode
- `group_size=2`
- `head_dim=128`
- 仍然是逐块 streaming 标量/向量累加

后续更值得研究的方向：

1. 是否能把当前 decode attention 改成更接近 `mma` / tensor-core 的实现
2. 是否能把“一组 query 头 + 一块 K/V”的计算重排成更适合 tensor core 的 tile
3. 是否可以容忍少量 workspace，换取更高吞吐

### 2.3 他们非常重视 `plan/run` 分离

在 [`flashinfer/decode.py`](./reference/flashinfer/flashinfer/decode.py#L825) 的 `plan()` 里，FlashInfer 会提前完成：

- `indptr / indices / last_page_len` 等 metadata 处理
- `kv_len` 计算
- backend 选择
- module 选择 / JIT
- 部分 host->device copy

而真正的 [`run()`](./reference/flashinfer/flashinfer/decode.py#L1210) 只做：

- q / kv cache shape 与 dtype 检查
- 输出 buffer 准备
- 按 cached plan info 发 kernel

这背后的收益不只是“代码干净”，而是：

- 把 host 侧同步和 metadata 构造从 steady-state decode 中拿走
- 让同一份 plan 能被多层复用
- 给 CUDA Graph 留出更稳定的输入形状和缓冲区

对我们的启发：

- 以后做 Attention microbench / 替换 kernel 时，最好也分清楚：
  - 哪些是 **per-bucket / per-sample 一次性准备**
  - 哪些是 **每步 decode 真正 unavoidable 的工作**
- 否则很容易把 host metadata 成本和 kernel 成本混在一起

### 2.4 graph 场景还要再做一层 `fast_decode_plan`

FlashInfer 甚至单独写了 [`fast_decode_plan()`](./reference/flashinfer/flashinfer/decode.py#L2742)，目标很明确：

- 去掉 graph buffer 上不必要的 device-to-device copy
- 去掉 metadata 上不必要的 host-to-device copy

它说明一个经验：

- 即便已经有 plan/run 分离
- **CUDA Graph 兼容路径仍然值得再做专门瘦身**

这点和我们当前路径高度相关。我们已经知道 graph 外的 host 开销大部分被清掉了，但 graph-compatible 的 metadata 准备仍可能有余量。

可以借鉴到我们仓库的地方：

1. decode attention microbench / benchmark 里，把“构造 valid_len / mask / cache_position 的辅助工作”继续压到 graph 外
2. 尽量保证 steady-state 只改固定 GPU buffer，而不是反复生成新张量

### 2.5 persistent kernel + reduction 分拆，是他们处理长序列的重要方式

在 [`include/flashinfer/attention/persistent_template.cuh`](./reference/flashinfer/include/flashinfer/attention/persistent_template.cuh#L53) 和 [`include/flashinfer/attention/persistent.cuh`](./reference/flashinfer/include/flashinfer/attention/persistent.cuh#L265) 里，可以看到它们把 attention 工作拆成：

- `Runner1`
- `Runner2`
- `Reduction`

大意是：

- 前面 persistent runners 负责不同 tile / chunk 的主体 attention 计算
- 后面 reduction runner 合并 partial states / partial lse

这类设计在我们当前单 batch decode 上不一定要原样照搬，但它说明：

- 真正高性能的 attention 往往不是“单一 kernel 一把梭”
- 而是**调度 + partial state + merge** 的整体方案

对我们当前任务的含义是：

- 如果后面要继续冲极限，很可能需要接受一定的 workspace / merge 开销
- 不能把思路永远局限在“一个最简 Triton kernel 直接扫完整段 KV”

---

## 3. 算子内 trace 手法

### 3.1 它不是 NVTX，而是 kernel 自己写时间戳

FlashInfer 的核心做法在 [`include/flashinfer/profiler.cuh`](./reference/flashinfer/include/flashinfer/profiler.cuh#L76)：

- 编译时加 `-DFLASHINFER_ENABLE_PROFILER`
- kernel 额外接一个 `uint64 profiler_buffer`
- 用 `mov.u32 %globaltimer_lo` 取 GPU 全局时间戳
- 把 `(sm_id, block_id, event_idx, begin/end/instant, timestamp)` 编码后写到 buffer

它记录的不是“kernel 外部持续时间”，而是：

- 同一个 CTA / group 内部
- 某几个命名阶段的开始、结束、瞬时打点

这正是我们现在缺的能力。

### 3.2 它为什么能看到 kernel 内部阶段

因为每个阶段都会显式插：

- `PROFILER_EVENT_START(...)`
- `PROFILER_EVENT_END(...)`
- `PROFILER_EVENT_INSTANT(...)`

而这些宏最终只是：

1. 生成 tag
2. 读 `globaltimer`
3. 写入 buffer
4. 用 `__threadfence_block()` 压住重排

对应代码见：

- [`profiler.cuh:89`](./reference/flashinfer/include/flashinfer/profiler.cuh#L89)
- [`profiler.cuh:104`](./reference/flashinfer/include/flashinfer/profiler.cuh#L104)
- [`profiler.cuh:114`](./reference/flashinfer/include/flashinfer/profiler.cuh#L114)
- [`profiler.cuh:124`](./reference/flashinfer/include/flashinfer/profiler.cuh#L124)

### 3.3 导出链路也很简单

Python 侧在 [`flashinfer/profiler/__init__.py`](./reference/flashinfer/flashinfer/profiler/__init__.py#L52) 做的事情是：

- 把 `profiler_buffer` 拷回 CPU
- 解码 tag
- 按 `(block, group, event)` 建 perfetto track
- 输出 `.perfetto-trace`

这个链路有两个优点：

1. **不依赖 graph 内 NVTX 是否可见**
2. **可以看到单 kernel 内多个阶段的重叠和顺序**

### 3.4 代价也很明确

FlashInfer 在 [`profiler/README.md`](./reference/flashinfer/profiler/README.md) 里明确写了：

- instrumentation 是 intrusive 的
- `__threadfence_block()` 会拖慢 kernel

所以这个 profiler 更适合：

- 小规模定点诊断
- attention microbench
- 单 kernel 阶段分析

不适合：

- 直接拿它的绝对时间当 benchmark 结论

### 3.5 这套方法怎么迁到我们仓库

短期内最现实的落地方式，不是去改 Triton，而是：

1. **给我们后续可能写的 CUDA attention kernel 预留 profiler 版本**
2. 在 `bench_decode_attention.py` 旁边补一个专门的 profiler runner
3. 只在 `ENABLE_*_PROFILER=1` 时编译/运行 profiler 版本
4. 输出单独的 perfetto trace 到 `traces/` 子目录

原因很简单：

- Triton 里要复刻 `globaltimer + 自定义事件 buffer + perfetto` 这整套能力，复杂度并不低
- 但如果后面 Attention 真的进入“手写 CUDA kernel / tensor-core kernel”阶段，这套 profiler 就能原样借鉴

所以我对这块的判断是：

- **值得学**
- **值得以后落地**
- **但最好等我们有更重型的 Attention kernel 时再接入**

---

## 4. 对当前仓库的具体启发

### 4.1 继续优化 Attention 时，优先级最高的方向

1. **tensor-core 化的 decode attention**
   - FlashInfer 明确把 `use_tensor_cores=True` 当作 decode 主路线之一
   - 我们当前 grouped-GQA Triton attention 已经解决了“重复读 K/V”的第一层问题
   - 下一步更该研究“如何把 group=2, head_dim=128 的 decode attention 组织成更适合 tensor core 的 tile”

2. **继续强化 plan/run 分离**
   - 尽量不要把任何可缓存的 metadata 留在 steady-state decode 步里
   - 尤其是 graph 路径，值得像 `fast_decode_plan()` 一样单独瘦身

3. **把 Attention 的验证工具继续做强**
   - FlashInfer 的常见工作流是：单独 wrapper / 单独 microbench / 单独 profiler
   - 我们已经有 `bench_decode_attention.py`，下一步可以继续补：
     - 多 layer
     - 多 decode step
     - 多 valid_len / bucket
     - 单独的 profiler 模式

### 4.2 哪些东西暂时不用急着抄

1. **paged KV cache 体系本身**
   - FlashInfer 很多设计是围绕 paged KV 展开的
   - 我们当前仓库主路径还是 `StaticCache` 连续布局，不能机械照搬

2. **SM90+/SM100 专属能力**
   - 例如 XQA、PDL、某些 fp8 / NVFP4 路线
   - 我们当前硬件是 A800 (SM80)，大部分只能学思想，不能直接拿代码

3. **persistent + reduction 的完整大工程**
   - 这是冲极限时的方向
   - 对我们当前阶段，更现实的是先验证 tensor-core 化 decode attention 是否值得

---

## 5. 建议的下一步

结合当前仓库状态，我建议后面按这个顺序推进：

1. 先继续用 [`bench_decode_attention.py`](../bench_decode_attention.py) 扩几组真实 case
   - `layer_idx ∈ {0, 14, 27}`
   - `decode_step ∈ {0, 16, 64}`
   - 看 grouped-GQA 的收益是不是稳定

2. 如果收益稳定，再尝试一版 **更偏 tensor-core / tile 化** 的 decode attention
   - 不一定一步到位写完整 CUDA kernel
   - 但方向要从“继续省几个 copy”转向“提升 attention kernel 本体效率”

3. 当 Attention kernel 复杂到需要看内部阶段时，再接 FlashInfer 风格的 profiler
   - 单独环境变量控制
   - 单独 microbench 场景使用
   - trace 文件统一放 `traces/` 子目录

---

## 6. 这次调研里最值得反复记住的一句话

**真正高收益的 Attention 优化，往往不是再省一个外围小 copy，而是把 kernel 组织方式改到更适合 GQA、tensor core 和长期可观测。**
