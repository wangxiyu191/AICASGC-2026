# 优化文档索引

Qwen3-VL-2B-Instruct 推理优化过程中的分析报告与记录文档。

> 当前状态 (2026-04-11, step 89): TTFT 38.78ms, Throughput 247.95 tok/s (100 samples, A800)

## 活跃文档

| 文档 | 说明 |
|------|------|
| [decode_fusion_plan.md](./decode_fusion_plan.md) | decode 深融合计划：优先级、实施顺序、改动文件、验证口径与止损条件 |
| [decode_attention_optimization.md](./decode_attention_optimization.md) | decode attention / split-KV 路线的优化过程、`ncu` 结论、保留实现与已证伪实验汇总 |
| [decode_gemv_analysis.md](./decode_gemv_analysis.md) | decode 阶段 `o_proj / down_proj / lm_head` 的 GEMV 性能分析、失败尝试与后续 CUDA/fusion 判断 |
| [cuda_decode_attention_exploration.md](./cuda_decode_attention_exploration.md) | CUDA decode attention kernel 完整探索记录，Triton split-KV 仍为默认的决策依据 |
| [optimization_attempts.md](./optimization_attempts.md) | 优化尝试与踩坑总结，含失败经验、已验证无效方向与未尝试方向 |
| [optimization_report.md](./optimization_report.md) | 竞赛提交用最终技术报告，包含主要执行路径、核心模块、最终启用 kernel 与实验结果说明 |
| [nsys_profiling_guide.md](./nsys_profiling_guide.md) | nsys 采集与分析流程: 命令、SQLite 分析模板、已有 trace 索引 |
| [a800_splitkv_validation.md](./a800_splitkv_validation.md) | A800 split-KV decode attention 验证计划与结论 |
| [a800_gemv_validation.md](./a800_gemv_validation.md) | A800 GEMV 验证计划与结论 |
| [a800_latency_partiton.md](./a800_latency_partiton.md) | A800 HBM 延迟 checkerboard 完整规律（实验验证版） |
| [flashinfer_attention_notes.md](./flashinfer_attention_notes.md) | FlashInfer Attention / 算子内 trace 调研笔记，kernel 设计与 profiler 思路参考 |
| [experiments/partition_gemv_experiment.md](./experiments/partition_gemv_experiment.md) | HBM partition GEMV 实验记录（结论：放弃此方向，保留作为避坑参考） |

## 归档文档 (stale/)

以下文档已过时，移至 `stale/` 目录保留备查：

| 文档 | 说明 |
|------|------|
| [stale/optimization_plan.md](./stale/optimization_plan.md) | 历史优化计划，停留在 phase4c 前 |
| [stale/cudagraph_implementation_plan.md](./stale/cudagraph_implementation_plan.md) | CUDAGraph 实施计划，已全部落地 |
| [stale/trace_analysis_report.md](./stale/trace_analysis_report.md) | CUDAGraph 前的旧 trace 分析，瓶颈已从 CPU-bound 转为 GPU-bound |
| [stale/inference_engines_analysis_report.md](./stale/inference_engines_analysis_report.md) | 推理引擎调研，架构参考可用，落地建议部分过时 |
| [stale/optimization_log.md](./stale/optimization_log.md) | steps 0-24 逐步日志，已被 result.csv 完全覆盖 |
| [stale/decode_bottleneck_analysis.md](./stale/decode_bottleneck_analysis.md) | step 24 (108 tok/s) 时的 decode kernel 分析，当前 pipeline 完全不同 |
| [stale/decode_compute_memory_budget.md](./stale/decode_compute_memory_budget.md) | 基于 step 24 的算力/访存预算估算，数字已过时 |
