# 代码阅读说明

本文档仅作为提交包中的**辅助索引**。关于最终提交版本的执行路径、实际启用的优化、关键 kernel 与代码位置，请以 [optimization_report.md](./optimization_report.md) 为主。

---

## 1. 建议优先阅读技术报告

主办方若希望快速理解“评测时实际执行了什么代码、主要优化在哪一层生效、应该如何阅读源码”，建议直接阅读技术报告中的以下部分：

- **第 3 节 方法设计**：说明官方 `benchmark.py` 进入提交包后的实际执行结构，以及 decode / prefill / 系统级优化的整体设计。
- **第 4.1 节 主要执行路径与核心模块**：说明从 `evaluation_wrapper.py` 到 `optimized_layer.py` 的主执行链，以及各核心模块在最终执行中的职责。
- **第 4.2 节 最终版本启用的 Kernel 实现概述**：列出最终热路径中实际启用的自定义 kernel、Python 入口和底层实现位置。
- **第 4.3 节 最终采用的开关与实验性开关**：给出最终提交配置与未采用实验路径的区分。
- **第 4.4 节 `CUDAGraph` 兼容性与静态化策略**：说明 decode 图捕获成立所依赖的静态化前提。

---

## 2. 提交包中最值得阅读的文件

| 文件/目录 | 主要职责 |
|-----------|----------|
| `evaluation_wrapper.py` | 官方评测入口、`VLMModel`、`GreedyDecodeEngine`、`DecodeCudaGraphRunner`、prefill fastpath |
| `optimized_layer.py` | `OptimizedDecoderLayer`、decode/prefill Triton 路由、fallback 逻辑 |
| `cuda_decode_attention.py` | decode attention CUDA extension 的 Python 加载与调用入口 |
| `cuda_decode_small_ops.py` | decode small ops CUDA extension 的 Python 加载与调用入口 |
| `cuda_decode_gemv_ops/` | nosplit GEMV、融合算子、runtime loader、profile 工具 |
| `cuda_extensions/` | CUDA kernel、pybind11 绑定与扩展构建脚本 |

---

## 3. 最短阅读路线

如果只关心最终成绩对应的热路径，建议按如下顺序阅读：

1. `evaluation_wrapper.py::VLMModel.__init__()`
2. `evaluation_wrapper.py::ModelWrapper.generate()`
3. `evaluation_wrapper.py::GreedyDecodeEngine.generate()`
4. `optimized_layer.py::OptimizedDecoderLayer.forward()`
5. `optimization_report.md` 第 4.2 节中对应 kernel 的底层实现位置

---

## 4. 说明

此前版本的 `code_reading_guide.md` 曾包含较多基于具体行号的展开说明。由于 `evaluation_wrapper.py`、`optimized_layer.py` 与 CUDA 扩展仍在持续演进，这类逐行号导读容易过时。因此，提交包中保留本文件作为简洁索引，而把正式、统一的实现说明收敛到技术报告中。
