# A800 Decode GEMV 验证计划

> 日期: 2026-03-29
> 背景: NCU 分析发现 cuBLAS 对 M=1 小 N 的 o_proj(2048x2048)/down_proj(2048x6144) 选用了 WMMA GEMM 而非专用 GEMV kernel，DRAM throughput 仅 52-59%、occupancy 仅 10%。新增 Triton `_decode_gemv_fwd_kernel` (BLOCK_M=1, BLOCK_K=512, warps=4, stages=2) 在 CUDAGraph 内达到 63%+ DRAM throughput 和 58% occupancy。A10 开发机 10 样本对照已确认 throughput +5.1% 且输出 exact match 10/10。

## 1. 当前状态

`ENABLE_TRITON_SMALL_DECODE_GEMV` 默认值为 `1`（默认开启）。

已知本地 A10 结果：

- baseline (GEMV OFF): [results/result_step55_gemv_off_10.json](../results/result_step55_gemv_off_10.json)
  - `TTFT 118.24 ms`
  - `Throughput 94.59 tok/s`
- GEMV ON: [results/result_step55_gemv_on_10.json](../results/result_step55_gemv_on_10.json)
  - `TTFT 118.62 ms`
  - `Throughput 99.40 tok/s`
  - `10/10 exact match`

辅助证据：

- NCU profiling 脚本: [profile_gemv_ncu.py](../profile_gemv_ncu.py)
- NCU shell 脚本: [ncu_gemv.sh](../ncu_gemv.sh)
- GEMV sweep 脚本: [sweep_gemv.py](../sweep_gemv.py)

## 2. 验证目标

需要确认三件事：

1. A800 上默认开启 GEMV 后，完整 benchmark throughput 是否稳定提升。
2. 默认开启 GEMV 后，答案是否与默认关闭版本保持一致。
3. A800 上 NCU 是否继续支持"Triton GEMV 比 cuBLAS WMMA GEMM 更优"的结论，或出现与 A10 不同的 kernel 选择行为。

## 3. 必做验证

### 3.1 默认开启 vs 默认关闭 benchmark

命令：

```bash
cd .

TQDM_DISABLE=1 ENABLE_TRITON_SMALL_DECODE_GEMV=0 ./venv/bin/python benchmark.py \
  --model-path ./Qwen3-VL-2B-Instruct \
  --dataset-path ./data \
  --output results/result_a800_gemv_off_10.json \
  --num-samples 10

TQDM_DISABLE=1 ENABLE_TRITON_SMALL_DECODE_GEMV=1 ./venv/bin/python benchmark.py \
  --model-path ./Qwen3-VL-2B-Instruct \
  --dataset-path ./data \
  --output results/result_a800_gemv_on_10.json \
  --num-samples 10
```

需要记录：

- `avg_ttft_ms`
- `avg_throughput_tokens_per_sec`
- 两次运行的输出 exact match

期望结果：

- throughput 至少不退化
- 理想目标：`+3%` 以上 throughput 提升
- TTFT 不明显变差
- 输出 `10/10 exact match`

### 3.2 100 样本稳定性验证

命令：

```bash
cd .

TQDM_DISABLE=1 ENABLE_TRITON_SMALL_DECODE_GEMV=0 ./venv/bin/python benchmark.py \
  --model-path ./Qwen3-VL-2B-Instruct \
  --dataset-path ./data \
  --output results/result_a800_gemv_off_100.json \
  --num-samples 100

TQDM_DISABLE=1 ENABLE_TRITON_SMALL_DECODE_GEMV=1 ./venv/bin/python benchmark.py \
  --model-path ./Qwen3-VL-2B-Instruct \
  --dataset-path ./data \
  --output results/result_a800_gemv_on_100.json \
  --num-samples 100
```

期望结果：

- throughput 提升方向和 10 样本一致
- 输出 `100/100 exact match`
- 无 graph capture/replay 异常

### 3.3 NCU 确认 cuBLAS kernel 选择

A800 (sm_80) 的 cuBLAS 可能选择与 A10 (sm_86) 不同的 kernel。需确认：

命令：

```bash
cd .

# 先确认 NCU 可用
/usr/local/cuda-12.6/bin/ncu --version

# 对 cuBLAS baseline 做 NCU
bash ncu_gemv.sh o_proj f_linear metrics
bash ncu_gemv.sh down_proj f_linear metrics

# 对 Triton GEMV 做 NCU
bash ncu_gemv.sh o_proj triton_gemv_v2 metrics
bash ncu_gemv.sh down_proj triton_gemv_v2 metrics
```

关注项：

- cuBLAS 是否仍选择 WMMA GEMM 而非 gemv2T？
- 如果 A800 cuBLAS 选了专用 GEMV kernel，Triton 替代可能不再有收益
- Triton kernel 的 DRAM throughput 和 occupancy 在 A800 上是否依然优于 cuBLAS

## 4. 默认保留条件

满足以下条件时，建议保留 GEMV 默认开启：

- A800 上 10 样本和 100 样本 benchmark 均不退化
- 至少一组 benchmark 显示 `+3%` 以上 throughput 提升
- 输出 exact match 全通过
- NCU 确认 cuBLAS 仍选择次优 kernel（WMMA GEMM 而非 gemv2T）

若不满足（尤其是 A800 cuBLAS 选择了专用 GEMV kernel），则回退默认值到 `0`，仅保留实验开关。

## 5. 技术细节

### 5.1 已替换的 GEMV 目标

| 目标 | 形状 (N x K) | cuBLAS kernel | 问题 |
|------|-------------|---------------|------|
| o_proj | 2048 x 2048 | WMMA GEMM | DRAM 52.53%, Occ 10.31% |
| down_proj | 2048 x 6144 | WMMA GEMM | DRAM 58.85%, Occ 10.36% |

### 5.2 未替换的 GEMV 目标

| 目标 | 形状 (N x K) | cuBLAS kernel | 原因 |
|------|-------------|---------------|------|
| lm_head | 151936 x 2048 | gemv2T | DRAM 95.87%, 已接近最优 |

### 5.3 Triton kernel 配置

- `BLOCK_M=1, BLOCK_K=512, num_warps=4, num_stages=2`
- 每个 thread block 处理 1 行输出，沿 K 维度以 512 步长迭代
- FP32 累加，最后转回输入 dtype 存储
- bias 在 kernel 外单独加（避免 kernel 复杂度）
