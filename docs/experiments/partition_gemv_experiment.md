# HBM Partition GEMV 实验报告

> 日期: 2026-04-05
> 状态: **结论明确 — 放弃此方向**

## 1. 实验目标

验证 A800 上 HBM 访存延迟的 ~14% 差异（同分区 vs 跨分区）是否能转化为：
- 可观的访存带宽提升
- 端到端的 GEMV 性能提升

## 2. 实验结果

### 2.1 数据汇总

测试脚本: `bench_same_vs_cross_partition.py`
结果文件: `results/partition_same_vs_cross.json`

| 测试 | 形状 | 标准(us) | Partition(us) | 加速比 | 标准带宽 | Partition带宽 |
|------|------|---------|--------------|--------|---------|--------------|
| tiny | [64,2048] | 7.7 | 7.6 | 1.005x | 34.2 GB/s | 34.4 GB/s |
| small | [256,2048] | 8.4 | 8.0 | 1.051x | 124.8 GB/s | 131.2 GB/s |
| medium | [1024,2048] | 10.5 | 10.8 | 0.968x | 399.6 GB/s | 387.0 GB/s |
| large | [4096,2048] | 21.4 | 22.0 | 0.974x | 784.7 GB/s | 764.2 GB/s |
| xlarge | [8192,2048] | 35.2 | 35.7 | 0.985x | 953.5 GB/s | 939.4 GB/s |

### 2.2 结论

**Partition GEMV 没有带来一致的性能提升。**

- 小模型（N≤256）：基本持平或微弱优势（+0.5% ~ +5.1%），可能是统计波动
- 中等模型（N≥1024）：反而更慢（-1.5% ~ -3.2%）
- **没有任何场景下超过 +5% 的提升**

## 3. 原因分析

### 3.1 延迟 ≠ 带宽

- **延迟**：单个请求从发出到数据返回的时间（~451 vs ~593 cycles，差异 14%）
- **带宽**：单位时间内能传输的数据量

在 GEMV 中：
1. 每个 SM 有大量独立的访存请求（N/108 个 row-pairs）
2. GPU 通过 **warp 调度隐藏延迟**：当一些 warp 等待数据时，其他 warp 继续执行
3. **最终瓶颈是 HBM 总带宽**，不是单个请求的延迟

### 3.2 GPU 访存系统的特性

- HBM 控制器统一调度所有请求，不区分"快路径"和"慢路径"
- NoC（Network-on-Chip）批量处理 SM 到 HBM 的请求
- 即使每个 SM 访问"自己的快页面"，**总的 HBM 访存量相同**

### 3.3 为什么 small 测试有 +5.1%？

可能是统计波动或测量误差。在 cold-L2 场景下，200 次迭代的平均值仍可能有 ±3-5% 的波动。

即使 +5.1% 是真实的，这个收益太小，不足以证明 partition 优化的复杂性：
- 需要双副本权重（2x 显存）
- 需要运行时探测 SM/chunk 类别
- 需要专门的 kernel 实现

## 4. 决策

### 4.1 放弃 HBM Partition GEMV 方向

**理由**：
1. 没有一致的性能提升
2. 最大收益仅 +5%（small 测试），且可能是波动
3. 大模型下反而更慢
4. 实现复杂度高（双副本、探测、专用 kernel）

### 4.2 可能的替代方向

如果还想利用 HBM 分区特性，需要考虑：
1. **降低数据量**：不使用双副本，尝试单副本 + 分区感知布局
   - 但需要控制物理页面分配，CUDA 不支持
2. **其他访存优化**：weight packing、shared memory tiling、register cache 等
3. **计算优化**：kernel fusion、operator scheduling 等

## 5. 实验代码索引

- `bench_same_vs_cross_partition.py`: 主要实验脚本
- `bench_partition_bandwidth.py`: 辅助带宽测试
- `results/partition_same_vs_cross.json`: 实验结果
- `docs/experiments/partition_gemv_experiment.md`: 本实验报告
