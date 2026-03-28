# 优化文档索引

Qwen3-VL-2B-Instruct 推理优化过程中的分析报告与记录文档。

> 当前状态 (2026-03-28): 线上 Score 0.9576, Accuracy 0.6978, TTFT 60.32ms, Throughput 108.82 tok/s

## 活跃文档

| 文档 | 说明 |
|------|------|
| [decode_bottleneck_analysis.md](./decode_bottleneck_analysis.md) | **当前重点** CUDAGraph decode 阶段的 GPU kernel 级瓶颈分析与优化方向，基于最新 viztracer trace |
| [optimization_log.md](./optimization_log.md) | 逐步优化记录 (step 0 → 24)，包含每次修改的配置与性能数据 |
| [optimization_attempts.md](./optimization_attempts.md) | 优化尝试与踩坑总结，含失败经验、已验证无效方向与未尝试方向 |

## 归档文档 (stale/)

以下文档已过时，移至 `stale/` 目录保留备查：

| 文档 | 说明 |
|------|------|
| [stale/optimization_plan.md](./stale/optimization_plan.md) | 历史优化计划，停留在 phase4c 前 |
| [stale/cudagraph_implementation_plan.md](./stale/cudagraph_implementation_plan.md) | CUDAGraph 实施计划，已全部落地 |
| [stale/trace_analysis_report.md](./stale/trace_analysis_report.md) | CUDAGraph 前的旧 trace 分析，瓶颈已从 CPU-bound 转为 GPU-bound |
| [stale/inference_engines_analysis_report.md](./stale/inference_engines_analysis_report.md) | 推理引擎调研，架构参考可用，落地建议部分过时 |
