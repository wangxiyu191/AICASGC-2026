# Decode Attention 优化纪要

本文记录 2026-03-29 这一轮 decode attention，尤其是 split-KV 路线的优化过程、`ncu` 结论、保留实现与已回退实验。

## 1. 目标与约束

- 目标：优化单 batch decode attention，优先看 TTFT、吞吐与正确性。
- 约束：不改模型参数；开发 profiling 机为 A10，目标评测机为 A800 80GB PCIe。
- 方法：
  - 单算子对照：`bench_decode_attention.py`
  - 端到端确认：`benchmark.py`
  - kernel 级分析：`profile_attention_ncu.py` + `/usr/local/cuda-12.6/bin/ncu`

## 2. 当前默认保留版本

截至 2026-03-29，默认保留的是 packed-stats + online-merge 的 split-KV decode attention 主线：

- 默认仍开启 `ENABLE_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV=1`
- split-KV workspace 挂在 layer 生命周期里，并在 graph capture/replay 中复用
- partial 结果布局为：
  - `partial_acc[num_kv_heads, split_k, num_key_value_groups, head_dim]`
  - `partial_stats[num_kv_heads, split_k, num_key_value_groups, 2]`
- merge kernel 使用 online single-pass accumulation，不再做两遍扫描

当前代码位置：

- split-KV 调度与 workspace 复用：[optimized_layer.py](../optimized_layer.py#L930)
- split-KV partial kernel：[optimized_layer.py](../optimized_layer.py#L1442)
- split-KV merge kernel：[optimized_layer.py](../optimized_layer.py#L1522)

当前默认确认结果：

- 10 样本 benchmark：`results/result_splitkv_default_confirm_10.json`
- A10 实测：TTFT `115.93 ms`，Throughput `96.30 tok/s`

这版相对最近几个可比版本的取舍：

- 比 step47 默认开启 split-KV 的老版本吞吐更高：`95.77 -> 96.30 tok/s`
- 比 step50 首轮 packed-stats + online-merge 复测结果也略高：`96.13 -> 96.30 tok/s`
- TTFT 不是历史最小值，但处于同一量级，没有为了极小 TTFT 牺牲吞吐

因此当前默认实现以这版为准。

## 3. 优化过程摘要

### 3.1 从 prefix/dot attention 到 split-KV

最初 `ncu` 看到的问题不是算得慢，而是 decode attention grid 太小：

- prefix/dot attention 常见 grid 只有 8 blocks
- GPU 没有被充分铺开

所以引入 split-KV，把单个 decode attention 拆成：

- `partial`：每个 split 独立算局部 attention 累加
- `merge`：合并不同 split 的局部结果

但第一版 split-KV 在 graph 外不值，原因很明确：

- 多了一次 merge kernel launch
- 还多了 `partial_acc/partial_l/partial_m/out` 的临时分配

只有把 workspace 挂进 graph 内部长期复用之后，这条路线才转成正收益。

### 3.2 当前主线为什么保留

后续把实现收敛到两点：

- `partial_l + partial_m` 打包成 `partial_stats[..., 2]`
- merge 从 two-pass 改成 online single-pass

这两项修改没有制造额外端到端回归，且让代码路径更简单，适合作为默认实现保留。

## 4. `ncu` 看到的核心瓶颈

### 4.1 partial kernel 不是单纯 HBM 打满

对 split-KV partial kernel 的几轮 `ncu` 观察，结论比较稳定：

- 主瓶颈更接近 `L1TEX`/memory-latency stall
- 同时伴随明显寄存器压力
- occupancy 偏低

典型现象：

- `No Eligible` 很高
- `Issued Warp / Scheduler` 偏低
- theoretical occupancy 和 achieved occupancy 都不高
- long scoreboard / L1TEX scoreboard 占主导

这意味着问题不是“再把 DRAM 带宽榨高一点”就能解决。

### 4.2 merge kernel 的问题是 underfilled

online merge 虽然减少了工作量，但 `ncu` 仍显示：

- grid 只有 `(8, 1, 1)`
- occupancy 仍然很低
- `No Eligible` 依旧在 90% 左右

所以 merge 的主要问题不是公式没写对，而是并行度天然不足。

## 5. 已验证有效的改动

### 5.1 graph 内持久 workspace

这是 split-KV 从“不值”转为“值得默认开启”的关键。

- graph 外：launch + allocation 开销把 kernel 内收益吃掉
- graph 内：launch 开销被摊薄，workspace 复用后 split-KV 才有稳定正收益

### 5.2 packed stats + online merge

这是当前主线实现。

收益特点：

- 代码更简洁
- 数据布局更规整
- 没有引入端到端明显回归
- 是目前吞吐最好的默认版本基础

## 6. 已证伪或暂不保留的路线

### 6.1 继续扫 launch 参数

做过多轮 `split_k / block_n / num_warps / num_stages` 扫描后，结论很明确：

- microbench 偶尔能赢一点
- 但很难外推成整条 graph decode 的正收益
- 当前默认启发式已经接近 A10 上的局部最优

因此这条路不再继续当成主方向。

### 6.2 per-q split-KV

单算子看起来不差，但端到端不成立。

根因：

- 破坏 grouped GQA 下的 K/V reuse
- 整体上不是主线方向

已移除。

### 6.3 head-dim chunking

试过把 partial forward 的 `HEAD_DIM=128` 拆成更小 chunk，目的是缩短 live range、减寄存器压力。

结果：

- `ncu` 指标方向是对的
- occupancy、eligible warps 有改善
- 但端到端 benchmark 没赢，反而回退

说明这类改动虽然改善了某些 kernel 内部指标，但新增的指令/访存成本抵消了收益。

### 6.4 `partial_out + lse`

试过在 partial kernel 内直接存：

- `partial_out = acc / l`
- `lse = log(l) + m`

然后复用现有 online merge，通过 `exp(lse)` 重新加权。

结果：

- microbench 基本和默认持平
- 数值误差更大
- 没拿到足够强的 kernel-level `ncu` 证据支持继续保留

所以这条分支已回退，不留在默认代码里。

## 7. 当前结论

对 decode attention 来说，本轮已经得到比较清晰的边界：

- split-KV 值得保留，但前提是 graph 内持久 workspace
- Triton 级别的小修小补已经接近挖空
- 当前真正的剩余空间不在 launch 参数微调
- partial kernel 的进一步突破大概率要靠更强的 memory staging / pipelining

## 8. 下一步建议

下一条更值得投入的路线是原生 CUDA，而不是继续在 Triton 上做小改：

- 给 split-KV partial kernel 做 shared-memory staging
- 尝试 pipelined K/V load-compute overlap
- 需要时单独重写 combine/merge kernel

对于 “split-KV 和 merge 能不能直接 fuse” 这个问题，当前判断是：

- 在现有多 CTA split-KV 结构下，很难在 Triton 里做成真正有价值的 fuse
- 因为最终还是会卡在跨 CTA 的归并与同步
- 真要做深，基本要进 CUDA 自定义 kernel

## 9. 对应结果记录

详细 step 记录见：

- [result.csv](../result.csv#L62)

本轮最关键的结果文件：

- `results/result_splitkv_default_confirm_10.json`
- `results/attention_microbench_splitkv_online_merge_layer14_step16.json`
- `results/result_splitkv_online_merge_10.json`
- `results/result_splitkv_online_merge_10_rerun.json`
- `results/attention_microbench_splitkv_chunkedpartial_layer14_step16.json`
- `results/result_splitkv_chunkedpartial_10.json`
- `results/attention_microbench_splitkv_chunkedpartial64_layer14_step16.json`
- `results/result_splitkv_chunkedpartial64_10.json`
- `results/attention_microbench_splitkv_partialout_lse_layer14_step16.json`

`ncu` 报告统一在 `results/ncu/` 下，二进制报告较大，仓库默认不跟踪。
