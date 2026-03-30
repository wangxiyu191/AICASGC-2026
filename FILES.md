# 项目文件索引

AICAS 2026 竞赛项目 — Qwen3-VL-2B-Instruct 推理优化。

## 核心文件

| 文件 | 说明 |
|------|------|
| [evaluation_wrapper.py](./evaluation_wrapper.py) | **参赛者核心修改文件**，包含 VLMModel 类和所有优化入口 |
| [optimized_layer.py](./optimized_layer.py) | 优化的 decoder layer（Fused QKV/gate_up、Triton 小 shape 融合、decode attention、GEMV 等） |
| [benchmark.py](./benchmark.py) | 官方基准测试工具（不可修改），测量 TTFT/吞吐量/准确率 |

## CUDA Extensions

| 文件 | 说明 |
|------|------|
| [cuda_decode_gemv.py](./cuda_decode_gemv.py) | CUDA nosplit GEMV 的 Python 接口（decode 阶段 o_proj/down_proj 等） |
| [cuda_decode_attention.py](./cuda_decode_attention.py) | 实验级 CUDA decode attention 的 Python 接口 |
| [cuda_extensions/](./cuda_extensions/) | CUDA kernel 源码目录 |
| `cuda_extensions/decode_gemv_ext.cpp` / `decode_gemv_kernel.cu` | nosplit GEMV CUDA kernel（含 k2048/k6144 专用路径、row2/row4 变体、internal profiling） |
| `cuda_extensions/decode_attention_ext.cpp` / `decode_attention_kernel.cu` | 实验级 CUDA decode attention kernel |
| `cuda_extensions/setup.py` / `setup_decode_gemv.py` | Extension 构建脚本 |

## Benchmark 与 Profiling 工具

| 文件 | 说明 |
|------|------|
| [bench_decode_attention.py](./bench_decode_attention.py) | 真实路径 decode attention microbenchmark（GPU-only cudaEvent 计时，支持 graph capture） |
| [bench_decode_linears.py](./bench_decode_linears.py) | 真实路径 decode linear microbenchmark（qkv/gate_up/o_proj/down_proj/lm_head） |
| [bench_decode_layer_chain.py](./bench_decode_layer_chain.py) | 单层 decode chain CUDAGraph microbenchmark（含 viztrace 集成） |
| [bench_rmsnorm.py](./bench_rmsnorm.py) | RMSNorm microbenchmark |
| [run_viztrace.py](./run_viztrace.py) | VizTracer profiling 脚本，分别采集 TTFT 和 throughput trace |
| [analyze_traces.py](./analyze_traces.py) | Trace 分析脚本，解析 viztracer JSON 输出并生成瓶颈报告（含 decode step 抽样时间线） |
| [profile_attention_ncu.py](./profile_attention_ncu.py) | NCU attention profiling 辅助脚本 |
| [profile_gemv_ncu.py](./profile_gemv_ncu.py) | NCU GEMV profiling 辅助脚本 |
| [profile_nosplit_internal.py](./profile_nosplit_internal.py) | nosplit GEMV 内部 block-timing profiler（基于 %globaltimer） |
| [ncu.sh](./ncu.sh) / [ncu_gemv.sh](./ncu_gemv.sh) / [ncu_nosplit.sh](./ncu_nosplit.sh) | Nsight Compute 批量采集脚本（metrics/deep/kernel_deep 多档） |
| [sweep_gemv.py](./sweep_gemv.py) | GEMV 参数扫描脚本 |

## 验证与对比工具

| 文件 | 说明 |
|------|------|
| [verify_numerical.py](./verify_numerical.py) | 数值验证脚本，对比优化后模型输出与原始模型的一致性 |
| [verify_real_data.py](./verify_real_data.py) | 真实数据验证脚本，在实际数据集上验证优化后的正确性 |
| [compare_results.py](./compare_results.py) | benchmark 结果对比脚本 |
| [diagnose_throughput.py](./diagnose_throughput.py) | 吞吐量诊断脚本 |

## 数据与结果

| 文件/目录 | 说明 |
|-----------|------|
| [result.csv](./result.csv) | **里程碑记录**：基线、官方评测、合入主线的优化、当前默认配置 benchmark（33 条） |
| [result_experiments.csv](./result_experiments.csv) | **实验记录**：未合入或已回退的探索、消融、A/B 对照、验证（40 条） |
| [result_tooling.csv](./result_tooling.csv) | **工具/分析记录**：NCU profiling、trace 工具链、microbench、研究调研（28 条） |
| [results/](./results/) | benchmark JSON 测试结果存放目录 |
| [traces/](./traces/) | viztrace/trace 文件存放目录 |

## 部署

| 文件 | 说明 |
|------|------|
| [create_submission.sh](./create_submission.sh) | 提交包创建脚本，打包竞赛提交文件 |

## 文档

| 文件 | 说明 |
|------|------|
| [docs/README.md](./docs/README.md) | 优化文档索引（计划、日志、踩坑记录、Trace 分析、引擎对比） |
| [FILES.md](./FILES.md) | 本文件 — 项目文件索引 |
| [AGENTS.md](./AGENTS.md) | AI 助手工作指引与约定 |
| [README_CN.md](./README_CN.md) | 竞赛介绍与评测说明 |
| [UPLOAD_README.md](./UPLOAD_README.md) | 第二赛季提交说明 |
