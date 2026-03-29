# CUDA Extensions Notes

这份文档记录当前仓库里实验级 CUDA kernel 的位置、编译方式、microbench 命令和 trace 命令，方便直接查看和复现。

## 文件说明

- `decode_attention_ext.cpp`
  - attention CUDA extension 的 pybind 入口。
- `decode_attention_kernel.cu`
  - attention CUDA kernel 实现。
- `setup.py`
  - attention CUDA extension 的本地构建脚本。
- `decode_gemv_ext.cpp`
  - decode GEMV CUDA extension 的 pybind 入口。
- `decode_gemv_kernel.cu`
  - decode GEMV CUDA kernel 实现。
- `setup_decode_gemv.py`
  - decode GEMV CUDA extension 的本地构建脚本。
- `cuda_decode_attention_ext*.so`
  - attention extension 编译产物，运行时自动生成。
- `cuda_decode_gemv_ext*.so`
  - GEMV extension 编译产物，运行时自动生成。

仓库根目录下还有两个 Python loader：

- [`cuda_decode_attention.py`](../cuda_decode_attention.py)
- [`cuda_decode_gemv.py`](../cuda_decode_gemv.py)

它们会在首次调用时自动检查源码时间戳并触发本地编译，不需要手动先 `python setup.py build_ext`。

## 编译方式

当前实现使用 PyTorch `BuildExtension`，并显式关闭 `ninja`：

- attention: `cuda_extensions/setup.py`
- GEMV: `cuda_extensions/setup_decode_gemv.py`

依赖条件：

- `./venv`
- `/usr/local/cuda/bin/nvcc`
- 当前机器可用 CUDA 编译环境

如果你想手动编译：

```bash
cd ./cuda_extensions
PATH=/usr/local/cuda/bin:$PATH ../venv/bin/python setup.py build_ext --inplace
PATH=/usr/local/cuda/bin:$PATH ../venv/bin/python setup_decode_gemv.py build_ext --inplace
```

## Attention Kernel 测试

### 1. 运行 attention microbench

这条命令会：

- 加载真实模型和数据
- 捕获 `sample_idx=0, layer_idx=14, decode_step=36` 的真实 attention 张量
- 对比 Triton / CUDA / SDPA 多个 attention 变体
- 使用 `cudaEvent` 做 GPU-only 计时

```bash
cd .
TQDM_DISABLE=1 ./venv/bin/python bench_decode_attention.py \
  --sample-idx 0 \
  --layer-idx 14 \
  --decode-step 36 \
  --warmup 5 \
  --iters 50 \
  --graph \
  --graph-capture-repeats 64 \
  --output results/attention_microbench_cudaevent_layer14_step36.json
```

结果文件：

- [`results/attention_microbench_cudaevent_layer14_step36.json`](../results/attention_microbench_cudaevent_layer14_step36.json)

当前重点看这几个字段：

- `triton_grouped_cachepos_splitkv`
- `cuda_grouped_cachepos_splitkv`
- `cuda_qhead_grouped_cachepos_splitkv`

### 2. 只看更早的 CUDA attention 实验结果

- [`results/attention_microbench_cuda_splitkv_smoke_layer14_step36.json`](../results/attention_microbench_cuda_splitkv_smoke_layer14_step36.json)
- [`results/attention_microbench_cuda_splitkv_smoke_layer14_step36_v2.json`](../results/attention_microbench_cuda_splitkv_smoke_layer14_step36_v2.json)
- [`results/attention_microbench_cuda_qhead_splitkv_smoke_layer14_step36.json`](../results/attention_microbench_cuda_qhead_splitkv_smoke_layer14_step36.json)
- [`results/attention_microbench_cuda_splitkv_sweep_layer14_step36.json`](../results/attention_microbench_cuda_splitkv_sweep_layer14_step36.json)

## GEMV Kernel 测试

### 1. 运行 linear microbench

这条命令会：

- 捕获真实 decode 时的 `qkv / gate_up / o_proj / down_proj / lm_head` 输入
- 对比 `F.linear`、Triton matvec、CUDA GEMV 变体
- 使用 `cudaEvent` 做 GPU-only 计时

```bash
cd .
TQDM_DISABLE=1 ./venv/bin/python bench_decode_linears.py \
  --sample-idx 0 \
  --layer-idx 14 \
  --decode-step 36 \
  --warmup 5 \
  --iters 50 \
  --graph \
  --graph-capture-repeats 64 \
  --output results/linear_microbench_fp16_cudaevent_sample0_layer14_step36.json
```

结果文件：

- [`results/linear_microbench_fp16_cudaevent_sample0_layer14_step36.json`](../results/linear_microbench_fp16_cudaevent_sample0_layer14_step36.json)

当前更值得关注的是：

- `o_proj_benchmarks`
- `down_proj_benchmarks`

### 2. CUDA GEMV 实验结果

- [`results/linear_microbench_cuda_gemv_smoke_sample0_layer14_step36.json`](../results/linear_microbench_cuda_gemv_smoke_sample0_layer14_step36.json)
- [`results/linear_microbench_cuda_gemv_row4_sample0_layer14_step36.json`](../results/linear_microbench_cuda_gemv_row4_sample0_layer14_step36.json)

主要看这些变体：

- `cuda_splitk8_matvec`
- `cuda_splitk16_matvec`
- `cuda_row4_splitk8_matvec`
- `cuda_row4_splitk16_matvec`

## 抓 Trace 看当前默认路径

### 1. 用专用 trace runner 抓单样本 trace

```bash
cd .
TQDM_DISABLE=1 ./venv/bin/python run_viztrace.py \
  --warmup 5 \
  --sample-idx 0 \
  --mode cudagraph \
  --output-dir traces/compare/my_trace
```

生成：

- `traces/compare/my_trace/generate_1.json`
- `traces/compare/my_trace/generate_2.json`

### 2. 分析 trace

```bash
cd .
./venv/bin/python analyze_traces.py traces/compare/my_trace \
  --top 12 \
  --step-samples 3 \
  --step-op-limit 35 \
  --layer-samples 3 \
  --layer-neighborhood 8
```

## 用 benchmark + 环境变量抓 trace

如果你想看“直接走 benchmark.py 时”生成的 trace，可以这样跑：

```bash
cd .
TQDM_DISABLE=1 \
ENABLE_VIZTRACE=1 \
VIZTRACE_OUTPUT_DIR=./traces/benchmark_env/my_trace \
VIZTRACE_WARMUP_CALLS=1 \
ENABLE_CUSTOM_DECODE=1 \
ENABLE_CUDAGRAPH=1 \
ENABLE_CUDAGRAPH_WITH_VIZTRACE=1 \
./venv/bin/python benchmark.py \
  --model-path ./Qwen3-VL-2B-Instruct \
  --dataset-path ./data \
  --output results/result_trace_benchmark_env_1.json \
  --num-samples 1
```

现成样例：

- [`traces/benchmark_env/current_default_fp16_20260329`](../traces/benchmark_env/current_default_fp16_20260329)

注意：

- trace 会极大拖慢运行速度
- trace 数值只用来看算子顺序和热点，不直接拿来和正常 benchmark 的 tok/s 比

## 当前结论

截至当前实验结果：

- attention CUDA 版已经能编译、能跑、能做 microbench
- 但在 GPU-only 口径下，当前 CUDA attention 仍慢于 Triton attention
- GEMV CUDA 版也已经能编译、能跑、能做 microbench
- 但目前仍慢于现有 Triton `splitk8` GEMV

更完整的实验结论见：

- [`result.csv`](../result.csv)
