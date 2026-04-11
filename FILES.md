# 项目文件索引

AICAS 2026 竞赛项目 — Qwen3-VL-2B-Instruct 推理优化。

## 核心文件

| 文件 | 说明 |
|------|------|
| [evaluation_wrapper.py](./evaluation_wrapper.py) | **参赛者核心修改文件**，包含 VLMModel 类和所有优化入口 |
| [optimized_layer.py](./optimized_layer.py) | 优化的 decoder layer（Fused QKV/gate_up、Triton 小 shape 融合、decode attention、GEMV）及 prefill Triton kernel（fused QK-norm+RoPE、packed SiLU*mul、fused vision RoPE） |
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
| [bench_gemv_bandwidth.py](./bench_gemv_bandwidth.py) | decode GEMV 带宽拆分 microbenchmark（flat 顺序读 / row4-xwarp 纯读 pattern / full GEMV） |
| [bench_l2_partition_locality.py](./bench_l2_partition_locality.py) | dual-copy / `smid` 路由 / `8KB` skew 的 L2 locality 专项 probe，含 synthetic weight-stream 与真实 `down_proj` GEMV 验证 |
| [bench_8kb_interleave_latency.py](./bench_8kb_interleave_latency.py) | 单次 kernel 内按 `smid` 逐块记录 cold-L2 pointer-chase 延迟，直接对比 even/odd `8KB` chunk 的原始 HBM-load 延迟 |
| [bench_interleave_granularity_latency.py](./bench_interleave_granularity_latency.py) | 固定每个 SM 总窗口大小后，扫描 `128B~8KB` 粒度的 odd/even pointer-chase 延迟，辅助区分 interleave 粒度与 TLB/page 覆盖影响 |
| [bench_sequential_stride_latency.py](./bench_sequential_stride_latency.py) | 单 block / 单线程串行 pointer-chase 顺序扫固定地址步长区域，记录每一次 load 的周期级 latency，用于观察 `256B` 顺序访问下的 page-level 周期结构 |
| [export_sequential_stride_table.py](./export_sequential_stride_table.py) | 将顺序 stride latency JSON 导出成“行=访问地址、列=smid”的矩阵 CSV，便于直接做热力图或表格分析 |
| [export_sequential_page_mean_table.py](./export_sequential_page_mean_table.py) | 直接运行顺序 stride probe，并把每个 `SM x 8KB page` 的均值导出成更小的 `page x smid` CSV，适合在大地址区间上找二维规律 |
| [analyze_sequential_stride_checkerboard.py](./analyze_sequential_stride_checkerboard.py) | 对顺序 stride latency 的 `8KB page x smid` 矩阵做二维 checkerboard 拟合，恢复 `SM 类 x 地址页类` 规律、周期和位公式 |
| [summarize_sequential_stride_mapping_findings.py](./summarize_sequential_stride_mapping_findings.py) | 从现有 sequential-stride 结果 JSON 自动生成 Markdown 结论文档，汇总 `1MiB` / `128MiB` / chunk-regime 三层发现 |
| [bench_decode_layer_chain.py](./bench_decode_layer_chain.py) | 单层 decode chain CUDAGraph microbenchmark（含 viztrace 集成） |
| [bench_prefill.py](./bench_prefill.py) | 真实路径 prefill forward benchmark（host/GPU 双计时，支持 bucketed cache 与 viztrace） |
| [bench_rmsnorm.py](./bench_rmsnorm.py) | RMSNorm microbenchmark |
| [run_nsys_prefill.py](./run_nsys_prefill.py) | nsys prefill profiler（CUPTI+NVTX，低开销替代 VizTracer） |
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
| [result.csv](./result.csv) | **里程碑记录**：基线、官方评测、合入主线的优化、当前默认配置 benchmark |
| [result_experiments.csv](./result_experiments.csv) | **实验记录**：未合入或已回退的探索、消融、A/B 对照、验证 |
| [result_tooling.csv](./result_tooling.csv) | **工具/分析记录**：NCU profiling、trace 工具链、microbench、研究调研 |
| [results/](./results/) | benchmark JSON 测试结果存放目录 |
| [traces/](./traces/) | viztrace/trace 文件存放目录 |

## 部署

| 文件 | 说明 |
|------|------|
| [create_submission.sh](./create_submission.sh) | 提交包创建脚本，打包竞赛提交文件 |

## 文档

| 文件 | 说明 |
|------|------|
| [docs/README.md](./docs/README.md) | 优化文档索引（活跃文档与归档文档） |
| [CLAUDE.md](./CLAUDE.md) | Claude Code AI 助手项目指引 |
| [FILES.md](./FILES.md) | 本文件 — 项目文件索引 |
| [AGENTS.md](./AGENTS.md) | AI 助手工作指引与约定 |
| [README_CN.md](./README_CN.md) | 竞赛介绍与评测说明 |
| [UPLOAD_README.md](./UPLOAD_README.md) | 第二赛季提交说明 |
