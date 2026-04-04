#!/usr/bin/env python3
"""
分析 sequential_stride_latency CSV，确定 108 个 SM 的二分类 (sm_class)。

原理：
  latency_class(smid, addr) = sm_class(smid) XOR page_class(addr) XOR chunk_class(addr)
  page_class(addr) = ((addr >> 13) & 1) ^ ((addr >> 17) & 1)

  CSV 中所有数据来自同一个 1MiB chunk，因此 chunk_class 是常量。
  对于 page_class=0 的地址行，latency_class = sm_class XOR C
  对于 page_class=1 的地址行，latency_class = sm_class XOR C XOR 1

  所以，对比每个 SM 在 page_class=0 和 page_class=1 上的平均延迟，
  可以判断 sm_class：
    - 如果 page_class=0 上更快 → sm_class XOR C = 0（快路径）
    - 如果 page_class=1 上更快 → sm_class XOR C = 1
"""

import csv
import sys
import numpy as np

CSV_PATH = "results/sequential_stride_latency_256b_1mib_108sm_a800_20260404_cycles_table.csv"

def page_class(addr):
    """page_class = ((addr >> 13) & 1) ^ ((addr >> 17) & 1)"""
    return ((addr >> 13) & 1) ^ ((addr >> 17) & 1)

def main():
    # 读取 CSV
    rows = []
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"总行数: {len(rows)}")

    num_sms = 108
    # 收集每个 SM 在 page_class=0 和 page_class=1 下的延迟
    lat_pc0 = [[] for _ in range(num_sms)]  # page_class == 0
    lat_pc1 = [[] for _ in range(num_sms)]  # page_class == 1

    # 跳过第一行 (access_idx=0) 因为全是 26，可能是特殊值
    for row in rows:
        access_idx = int(row["access_idx"])
        if access_idx == 0:
            continue
        addr_offset = int(row["addr_offset_bytes"])
        pc = page_class(addr_offset)

        for sm in range(num_sms):
            lat = int(row[f"smid_{sm}"])
            if pc == 0:
                lat_pc0[sm].append(lat)
            else:
                lat_pc1[sm].append(lat)

    # 计算每个 SM 在两种 page_class 下的平均延迟
    mean_pc0 = np.array([np.mean(lat_pc0[sm]) for sm in range(num_sms)])
    mean_pc1 = np.array([np.mean(lat_pc1[sm]) for sm in range(num_sms)])

    print(f"\npage_class=0 样本数: {len(lat_pc0[0])}, page_class=1 样本数: {len(lat_pc1[0])}")
    print(f"全局平均 page_class=0: {np.mean(mean_pc0):.1f}, page_class=1: {np.mean(mean_pc1):.1f}")

    # 对每个 SM，计算 delta = mean_pc0 - mean_pc1
    # delta > 0 → SM 在 page_class=0 时更慢 → sm_class XOR C = 1
    # delta < 0 → SM 在 page_class=0 时更快 → sm_class XOR C = 0
    delta = mean_pc0 - mean_pc1

    # 用 sign 分类（不关心 C 的具体值，先分出两组）
    # 定义 group_A = delta < 0 的 SM (page_class=0时更快)
    # 定义 group_B = delta > 0 的 SM (page_class=1时更快)
    group_A = sorted([sm for sm in range(num_sms) if delta[sm] < 0])
    group_B = sorted([sm for sm in range(num_sms) if delta[sm] >= 0])

    print(f"\nGroup A ({len(group_A)} SMs): page_class=0 时更快 (低延迟)")
    print(f"  SMs: {group_A}")
    print(f"  平均延迟 pc0={np.mean(mean_pc0[group_A]):.1f}, pc1={np.mean(mean_pc1[group_A]):.1f}")

    print(f"\nGroup B ({len(group_B)} SMs): page_class=1 时更快 (低延迟)")
    print(f"  SMs: {group_B}")
    print(f"  平均延迟 pc0={np.mean(mean_pc0[group_B]):.1f}, pc1={np.mean(mean_pc1[group_B]):.1f}")

    # 计算分离度
    delta_abs = np.abs(delta)
    print(f"\n|delta| 统计: min={np.min(delta_abs):.1f}, max={np.max(delta_abs):.1f}, "
          f"mean={np.mean(delta_abs):.1f}, median={np.median(delta_abs):.1f}")

    # 输出为可直接使用的 Python 常量
    print("\n# ========== Python 常量 ==========")
    print(f"SM_CLASS_A = {group_A}  # {len(group_A)} SMs")
    print(f"SM_CLASS_B = {group_B}  # {len(group_B)} SMs")

    # 也输出 bitfield 形式
    sm_class_bits = 0
    for sm in group_B:
        sm_class_bits |= (1 << sm)
    print(f"\n# Bitfield: group_B 的 SM 对应 bit=1")
    print(f"SM_CLASS_BITFIELD = 0x{sm_class_bits:028X}")

    # 输出每个SM的详细数据
    print("\n# ========== 每个 SM 详细数据 ==========")
    print(f"{'SM':>4} {'mean_pc0':>10} {'mean_pc1':>10} {'delta':>10} {'class':>6}")
    for sm in range(num_sms):
        cls = "A" if delta[sm] < 0 else "B"
        print(f"{sm:>4} {mean_pc0[sm]:>10.1f} {mean_pc1[sm]:>10.1f} {delta[sm]:>10.1f} {cls:>6}")


if __name__ == "__main__":
    main()
