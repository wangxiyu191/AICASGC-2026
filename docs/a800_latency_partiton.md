# A800 HBM 延迟 Checkerboard 完整规律 (实验验证版)

> 更新日期: 2026-04-05  
> 硬件: NVIDIA A800 80GB PCIe (108 SM, 40MB L2, HBM2e)  
> 验证状态: **128/128 chunks, 106/106 regime B, 页公式 100% 精确**  
> 真实权重验证: **193/193 chunks, 108/108 SM, page_class 公式 100% 精确**

## 1. 核心现象

A800 上，HBM cold-miss 访问延迟存在一个 **三层 XOR checkerboard 交互**：

```
latency_class(smid, addr) = sm_class(smid) XOR page_class(addr) XOR chunk_class(addr)
```

- `latency_class == 0` → **快路径**: ~451 cycles (均值 522 - 振幅 71)
- `latency_class == 1` → **慢路径**: ~593 cycles (均值 522 + 振幅 71)

**振幅 71 cycles ≈ 14% 的均值延迟，差异显著。**

## 2. 三层公式详解

### 2.1 Page Class (第一层 - 8KB 粒度)

**精度: 100% 精确，所有实验条件下均成立。**

```c
page_class(addr) = ((addr >> 13) & 1) ^ ((addr >> 17) & 1)
```

等价表达: 取 8KB 页在 2MiB chunk 内的索引 `p = (addr >> 13) & 0xFF`，则 `page_class = (p & 1) ^ ((p >> 4) & 1)`。

| 特性 | 值 |
|------|------|
| 粒度 | 8KB (2^13) |
| 相邻页 | 交替 (01010101...) |
| 反相周期 | 128KB (16 pages) |
| 完整重复周期 | 256KB (32 pages) |
| 每 2MiB chunk | 256 pages, 8 个完整周期 |
| 前32页 class | `01010101010101011010101010101010` |

### 2.2 SM Class (第二层 - 108 SM 二分类)

**精度: 首次冷探测时 R²=0.975；同一分配地址下完全可复现。**

108 个 SM 固定分为两类：

**分析方法**: 基于 `results/sequential_stride_latency_256b_1mib_108sm_a800_20260404_cycles_table.csv`（4096 行，108 SM × 256B stride 遍历 1MiB chunk），对比每个 SM 在 page_class=0 和 page_class=1 地址上的平均延迟。分析脚本: `tools/analyze_sm_class.py`。

**分类结果** (|delta| min=136.2, max=148.0, 分离极其干净，无歧义):

> **注意**: 以下 SM 分类列表是在**特定 A800 卡**上测得的示例值。SM 分类是**硬件实例相关**的，不同 GPU 卡的分类表不同。实际使用时必须通过 `hbm_latency_partition.probe_sm_classes()` 运行时探测。

- **sm_class = 0 (Group A, 56 SMs)**: page_class=0 时快路径 (~451 cycles), page_class=1 时慢路径 (~593 cycles)
  ```
  [1, 3, 4, 7, 9, 10, 12, 14, 16, 17, 20, 21, 22, 23, 28, 30,
   33, 34, 36, 39, 41, 43, 45, 46, 52, 53, 54, 55, 56, 57, 58, 59,
   60, 63, 66, 67, 68, 69, 70, 71, 72, 73, 76, 77, 78, 79, 81, 83,
   85, 87, 88, 91, 92, 93, 102, 103]
  ```

- **sm_class = 1 (Group B, 52 SMs)**: page_class=1 时快路径 (~451 cycles), page_class=0 时慢路径 (~594 cycles)
  ```
  [0, 2, 5, 6, 8, 11, 13, 15, 18, 19, 24, 25, 26, 27, 29, 31,
   32, 35, 37, 38, 40, 42, 44, 47, 48, 49, 50, 51, 61, 62, 64, 65,
   74, 75, 80, 82, 84, 86, 89, 90, 94, 95, 96, 97, 98, 99, 100, 101,
   104, 105, 106, 107]
  ```

**Bitfield 表示** (sm_class=1 的 SM 对应 bit=1, 此为示例值):
```python
# 示例值，实际值因卡而异，需运行时探测
SM_CLASS_BITFIELD = 0x0F3FC6550C03600F9569AF0CA965
```

**快速查询** (bitfield 作为 kernel 参数传入，运行时从探测结果填充):
```c
// sm_class(smid): 返回 0 或 1
// sm_class_lo / sm_class_hi 由运行时探测得到，作为 kernel 参数传入
int sm_class(int smid, uint64_t sm_class_lo, uint64_t sm_class_hi) {
    if (smid < 64) return (sm_class_lo >> smid) & 1;
    return (sm_class_hi >> (smid - 64)) & 1;
}
```

**SM 编号与 class 的交替规律**: 观察数据可见，相邻 SM 编号通常交替属于不同 class（但并非严格交替），这可能与 GPC/TPC 拓扑有关。

> 数据来源: `export_sequential_stride_table.py` 从 `bench_sequential_stride_latency.py` 产出的 JSON 导出。

### 2.2.1 真实 GEMV 权重验证 (2026-04-05)

**验证脚本**: `bench_sm_partition_on_weights.py`  
**结果**: `results/sm_partition_on_weights_2mb_align.json`

在 Qwen3-VL-2B-Instruct 的真实 GEMV 权重 tensor（layer 14）上验证 SM 分区和 page_class 公式。

**关键发现: 必须从 2MB 对齐地址开始探测**  
权重 tensor 的 base_addr 可能不在 2MB 边界上（CUDA 分配粒度小于 2MB）。
未对齐时，延迟差消失（所有 page 平均 ~510 cycles，无快慢分离）。
从 tensor 内第一个 2MB 对齐地址开始切 chunk 后，所有 chunk 都显示清晰分离。

| 权重 | shape | size | base对齐? | skip | chunks | page_class OK | chunk_class分布 |
|------|-------|------|-----------|------|--------|---------------|-----------------|
| qkv | [4096,2048] | 16MB | 否(+512KB) | 512KB | 15 | **15/15** | cc0=5, cc1=10 |
| gate_up | [12288,2048] | 48MB | 否(+512KB) | 512KB | 47 | **47/47** | cc0=26, cc1=21 |
| down_proj | [2048,6144] | 24MB | 否(+512KB) | 512KB | 23 | **23/23** | cc0=6, cc1=17 |
| lm_head | [151936,2048] | 593.5MB | **是** | 0 | 108 | **108/108** | cc0=108, cc1=0 |

**汇总 (全量扫描 2026-04-05)**: 678/678 chunks page_class 100%, 108/108 SM 100%。

### 2.2.2 Chunk Class 粒度确认

**结论: chunk_class 最小粒度 = 2MB (huge page)**

全量扫描 678 chunks，将相邻 1MB chunks 两两配对为 2MB pairs，检查同一 2MB 内两个 1MB 是否共享 chunk_class：

| 权重 | 2MB pairs | SAME | DIFF |
|------|-----------|------|------|
| qkv | 7 | 7 | 0 |
| gate_up | 23 | 23 | 0 |
| down_proj | 11 | 11 | 0 |
| lm_head | 296 | 296 | 0 |
| **合计** | **337** | **337** | **0** |

337/337 pairs 全部一致 → chunk_class 不会在 2MB 内部变化。

同时 run-length 分析表明存在 run_length=1 (2MB) 的单独 2MB 块（gate_up/lm_head 中出现），因此 chunk_class 的粒度**恰好是 2MB**，不是更大。

这与 CUDA 使用 2MB huge pages 管理 GPU 显存一致：同一物理 huge page 内的两个 1MB chunk 映射到同一 HBM channel/bank，因此 chunk_class 相同。
