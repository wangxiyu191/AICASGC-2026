# 优化文档索引

Qwen3-VL-2B-Instruct 推理优化过程中的分析报告与记录文档。

## 文档列表

| 文档 | 说明 |
|------|------|
| [optimization_plan.md](./optimization_plan.md) | `[历史规划/部分过时]` 整体优化计划；内容停留在 `phase4c` 前后的计划视角，最新结论以 `optimization_log.md` 和 `result.csv` 为准 |
| [cudagraph_implementation_plan.md](./cudagraph_implementation_plan.md) | `[历史实施文档/部分过时]` CUDA Graph 路线实施文档；`phase4c` 已落地，本文件主要保留实现拆解、风险点和验证思路供回顾 |
| [optimization_log.md](./optimization_log.md) | 逐步优化记录，包含每次修改的配置与性能数据 |
| [optimization_attempts.md](./optimization_attempts.md) | 优化尝试与踩坑总结，含失败经验与后续建议 |
| [trace_analysis_report.md](./trace_analysis_report.md) | `[旧 Trace/部分过时]` 基于 `phase4c` 前 trace 的瓶颈分析；前半段现象分析仍可参考，但“优化建议”部分已有多项被后续实验更新 |
| [inference_engines_analysis_report.md](./inference_engines_analysis_report.md) | `[外部调研/部分过时]` 推理引擎深度对比分析；架构和源码索引可参考，但第 8 节中的部分优化建议已过时，且包含当前规则不允许的量化方向 |
