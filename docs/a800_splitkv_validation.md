# A800 Split-KV Attention 验证计划

> 日期: 2026-03-29
> 背景: decode attention 的 split-KV 路径在 A10 上，配合 graph 内持久 workspace 复用后，已经从负收益转为正收益。下一步需要在目标 A800 80GB PCIe 环境上确认它是否值得保留为默认路径。

## 1. 当前状态

当前默认配置已经将 `ENABLE_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV` 设为默认开启。

已知本地 A10 结果：

- baseline: [results/result_step45_baseline_graph_10.json](../results/result_step45_baseline_graph_10.json)
  - `TTFT 117.70 ms`
  - `Throughput 92.05 tok/s`
- split-KV + persistent workspace: [results/result_step45_splitkv_workspace_graph_10.json](../results/result_step45_splitkv_workspace_graph_10.json)
  - `TTFT 115.81 ms`
  - `Throughput 95.81 tok/s`
  - `10/10 exact match`

辅助证据：

- split-KV microbench: [results/attention_microbench_splitkv_layer14_step16.json](../results/attention_microbench_splitkv_layer14_step16.json)
- prefix NCU: [results/ncu/ncu_attention_prefix_layer14_step16_sections_2024.ncu-rep](../results/ncu/ncu_attention_prefix_layer14_step16_sections_2024.ncu-rep)
- split-KV NCU: [results/ncu/ncu_attention_splitkv_layer14_step16_sections_2024.ncu-rep](../results/ncu/ncu_attention_splitkv_layer14_step16_sections_2024.ncu-rep)

## 2. 验证目标

需要确认三件事：

1. A800 上默认开启 split-KV 后，完整 benchmark throughput 是否稳定提升。
2. 默认开启 split-KV 后，答案是否与默认关闭版本保持一致。
3. A800 上 `ncu` 是否继续支持“split-KV partial kernel 本体更优，但整体收益受 graph 外成本影响”的判断，或出现与 A10 不同的瓶颈。

## 3. 必做验证

### 3.1 默认开启 vs 默认关闭 benchmark

命令：

```bash
cd .

TQDM_DISABLE=1 ENABLE_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV=0 ./venv/bin/python benchmark.py \
  --model-path ./Qwen3-VL-2B-Instruct \
  --dataset-path ./data \
  --output results/result_a800_splitkv_off_10.json \
  --num-samples 10

TQDM_DISABLE=1 ENABLE_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV=1 ./venv/bin/python benchmark.py \
  --model-path ./Qwen3-VL-2B-Instruct \
  --dataset-path ./data \
  --output results/result_a800_splitkv_on_10.json \
  --num-samples 10
```

需要记录：

- `avg_ttft_ms`
- `avg_throughput_tokens_per_sec`
- 两次运行的输出 exact match

期望结果：

- throughput 至少不退化
- 理想目标：`+2%` 以上 throughput 提升
- TTFT 不明显变差
- 输出 `10/10 exact match`

### 3.2 100 样本稳定性验证

命令：

```bash
cd .

TQDM_DISABLE=1 ENABLE_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV=0 ./venv/bin/python benchmark.py \
  --model-path ./Qwen3-VL-2B-Instruct \
  --dataset-path ./data \
  --output results/result_a800_splitkv_off_100.json \
  --num-samples 100

TQDM_DISABLE=1 ENABLE_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV=1 ./venv/bin/python benchmark.py \
  --model-path ./Qwen3-VL-2B-Instruct \
  --dataset-path ./data \
  --output results/result_a800_splitkv_on_100.json \
  --num-samples 100
```

期望结果：

- throughput 提升方向和 10 样本一致
- 输出 `100/100 exact match`
- 无 graph capture/replay 异常

### 3.3 Attention microbench 对照

命令：

```bash
cd .

TQDM_DISABLE=1 ENABLE_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV=0 ./venv/bin/python bench_decode_attention.py \
  --sample-idx 0 \
  --layer-idx 14 \
  --decode-step 16 \
  --output results/attention_microbench_a800_splitkv_off_layer14_step16.json

TQDM_DISABLE=1 ENABLE_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV=1 ./venv/bin/python bench_decode_attention.py \
  --sample-idx 0 \
  --layer-idx 14 \
  --decode-step 16 \
  --output results/attention_microbench_a800_splitkv_on_layer14_step16.json
```

关注项：

- `triton_grouped_cachepos_prefix`
- `triton_grouped_cachepos_prefix_dot`
- `triton_grouped_cachepos_splitkv`
- `triton_current`

期望结果：

- split-KV 不一定在单独 microbench 上最优
- 但如果 graph benchmark 明显提升，而 microbench 只是持平或略慢，可以接受
- 数值误差需保持在当前量级，不出现明显漂移

## 4. NCU 复核

建议继续使用：

```bash
/usr/local/cuda-12.6/bin/ncu
```

### 4.1 baseline prefix kernel

目标 kernel：

- `_small_decode_attention_grouped_prefix_cachepos_fwd_kernel`

重点看：

- `Grid Size`
- `Registers Per Thread`
- `Theoretical / Achieved Occupancy`
- `Memory Throughput`
- `No Eligible`

### 4.2 split-KV partial + merge kernel

目标 kernel：

- `_small_decode_attention_grouped_prefix_cachepos_splitkv_fwd_kernel`
- `_small_decode_attention_grouped_prefix_cachepos_splitkv_merge_kernel`

重点看：

- partial kernel 是否仍然比 baseline 更高 grid、更低 regs/thread
- merge kernel 占比是否可接受
- A800 上 partial kernel 的 grid 是否更接近填满 SM

期望结果：

- partial kernel 的并行度收益在 A800 上至少不弱于 A10
- merge kernel 不应成为新的显著热点

## 5. 默认保留条件

满足以下条件时，建议保留 split-KV 默认开启：

- A800 上 10 样本和 100 样本 benchmark 均不退化
- 至少一组 benchmark 显示 `+2%` 以上 throughput 提升
- 输出 exact match 全通过
- `ncu` 没有暴露出新的明显大热点

若不满足，则回退默认值到 `0`，仅保留实验开关。
