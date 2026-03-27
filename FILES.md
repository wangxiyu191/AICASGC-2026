# 项目文件索引

AICAS 2026 竞赛项目 — Qwen3-VL-2B-Instruct 推理优化。

## 核心文件

| 文件 | 说明 |
|------|------|
| [evaluation_wrapper.py](./evaluation_wrapper.py) | **参赛者核心修改文件**，包含 VLMModel 类和所有优化入口 |
| [optimized_layer.py](./optimized_layer.py) | 优化的 decoder layer（Fused QKV/gate_up 线性投影） |
| [benchmark.py](./benchmark.py) | 官方基准测试工具（不可修改），测量 TTFT/吞吐量/准确率 |

## 工具脚本

| 文件 | 说明 |
|------|------|
| [run_viztrace.py](./run_viztrace.py) | VizTracer profiling 脚本，分别采集 TTFT 和 throughput trace |
| [analyze_traces.py](./analyze_traces.py) | Trace 分析脚本，解析 viztracer JSON 输出并生成瓶颈报告 |
| [diagnose_throughput.py](./diagnose_throughput.py) | 吞吐量诊断脚本，对比不同场景下的性能差异 |
| [verify_numerical.py](./verify_numerical.py) | 数值验证脚本，对比优化后模型输出与原始模型的一致性 |
| [verify_real_data.py](./verify_real_data.py) | 真实数据验证脚本，在实际数据集上验证优化后的正确性 |
| [create_submission.sh](./create_submission.sh) | 提交包创建脚本，打包竞赛提交文件 |

## 数据与结果

| 文件/目录 | 说明 |
|-----------|------|
| [result.csv](./result.csv) | 逐步优化记录（每步的 TTFT、吞吐量、GPU、备注） |
| [results/](./results/) | benchmark JSON 测试结果存放目录 |

## 文档

| 文件 | 说明 |
|------|------|
| [docs/README.md](./docs/README.md) | 优化文档索引（计划、日志、踩坑记录、Trace 分析、引擎对比） |
| [AGENTS.md](./AGENTS.md) | AI 助手工作指引与约定 |
| [README_CN.md](./README_CN.md) | 竞赛介绍与评测说明 |
| [UPLOAD_README.md](./UPLOAD_README.md) | 第二赛季提交说明 |
