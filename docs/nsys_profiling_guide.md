# nsys Profiling Guide

本项目使用 nsys (Nsight Systems) 进行低开销 GPU profiling, 替代 VizTracer 的 `log_torch=True` 模式。

## 为什么用 nsys 而不是 VizTracer

| 工具 | CPU 开销 | 原理 |
|------|---------|------|
| VizTracer `log_torch=True` | 600-900% | 内部用 `torch.profiler.profile(activities={CPU,CUDA})`, 通过 RecordFunction hook 拦截每个算子 |
| VizTracer 不开 torch | ~68% | 只记 Python 函数调用, 无 GPU kernel 信息 |
| **nsys** | **~3%** | 走 CUPTI activity API, 内核级采集, 无 Python hook |

nsys 只记录 CUDA Runtime API 调用和 GPU kernel 执行, 不插入 Python-level hook, 所以 CPU timeline 几乎不失真。

## 环境

本项目 nsys 二进制路径 (随 NCU 安装包附带):

```bash
NSYS=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
$NSYS --version   # 验证可用
```

> NCU (ncu) 本身因 `RmProfilingAdminOnly=1` + 容器缺少 `CAP_SYS_ADMIN`/`CAP_PERFMON` 而无法采集硬件性能计数器, 只有 `--set none` 可用。但 nsys 不需要特权, 正常工作。

## 采集流程

### 1. 使用 run_nsys_prefill.py 脚本

脚本通过 `cudaProfilerStart()/Stop()` 控制采集范围, 只抓取实际推理部分, 排除模型加载和 warmup。

```bash
# 完整命令: 采集 prefill trace
TQDM_DISABLE=1 $NSYS profile \
    --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    -o traces/nsys_<label>/report \
    ./venv/bin/python run_nsys_prefill.py \
        --warmup 3 --sample-idx 0 --repeats 1
```

参数说明:

| 参数 | 作用 |
|------|------|
| `--trace=cuda,nvtx` | 采集 CUDA Runtime/Driver API + GPU kernel + NVTX 标注 |
| `--capture-range=cudaProfilerApi` | 只在代码调用 `cudaProfilerStart()` 后开始采集 |
| `--capture-range-end=stop` | 遇到 `cudaProfilerStop()` 时结束采集 |
| `-o <path>` | 输出 `.nsys-rep` 文件路径 (不含扩展名) |
| `--force-overwrite=true` | 覆盖已有报告 |

脚本参数:

| 参数 | 默认 | 作用 |
|------|------|------|
| `--warmup` | 3 | warmup 轮数 (不采集) |
| `--sample-idx` | 0 | 使用 dataset 中第几个 sample |
| `--repeats` | 3 | 采集区间内重复几次 TTFT |
| `--no-nsys` | - | 不启动 nsys, 只用 cudaEvent 计时 (调试/基线用) |

### 2. 不使用脚本 (直接 profile benchmark.py)

```bash
# 会采集全量 (含模型加载/warmup), 数据量大
TQDM_DISABLE=1 $NSYS profile \
    --trace=cuda,nvtx \
    --force-overwrite=true \
    -o traces/nsys_<label>/report \
    ./venv/bin/python benchmark.py \
        --model-path ./Qwen3-VL-2B-Instruct \
        --dataset-path ./data \
        --output results/result_nsys_<label>.json \
        --num-samples 1
```

不推荐: 没有 capture range 控制, 会采集几十秒的无关数据。

## 导出与分析

### 导出 SQLite (推荐用于编程分析)

```bash
$NSYS export --type sqlite \
    --force-overwrite=true \
    -o traces/nsys_<label>/report.sqlite \
    traces/nsys_<label>/report.nsys-rep
```

### 导出 JSON (可用 chrome://tracing 查看)

```bash
$NSYS export --type json \
    traces/nsys_<label>/report.nsys-rep \
    > traces/nsys_<label>/report.json
```

### 快速统计

```bash
$NSYS stats traces/nsys_<label>/report.nsys-rep
```

## SQLite 分析模板

导出 SQLite 后可用 Python 分析。核心表:

| 表名 | 内容 |
|------|------|
| `CUPTI_ACTIVITY_KIND_KERNEL` | GPU kernel 执行记录 (start, end, name, grid, block) |
| `CUPTI_ACTIVITY_KIND_RUNTIME` | CUDA Runtime API 调用 (cudaLaunchKernel, cudaStreamSynchronize 等) |
| `NVTX_EVENTS` | NVTX range push/pop 标注 |
| `StringIds` | 字符串 id→value 映射 (kernel name 等通过 id 引用) |

### 常用分析 SQL

#### 1. 查找 Sync 调用 (定位 CPU 阻塞点)

```python
conn.execute('''
    SELECT r.start, r.end, (r.end - r.start) as dur_ns, s.value as name
    FROM CUPTI_ACTIVITY_KIND_RUNTIME r
    JOIN StringIds s ON r.nameId = s.id
    WHERE s.value LIKE '%ynchronize%'
    ORDER BY dur_ns DESC
    LIMIT 20
''')
```

#### 2. 提取 GPU kernel 时间线

```python
conn.execute('''
    SELECT k.start, k.end, (k.end - k.start) as dur_ns,
           COALESCE(dk.value, sk.value, 'unknown') as name
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    LEFT JOIN StringIds dk ON k.demangledName = dk.id
    LEFT JOIN StringIds sk ON k.shortName = sk.id
    ORDER BY k.start
''')
```

#### 3. 找 Top-N inter-kernel gaps (GPU 气泡)

```python
kernels = [...]  # 按 start 排序的 kernel 列表
gaps = []
for i in range(1, len(kernels)):
    gap = kernels[i].start - kernels[i-1].end
    if gap > 0:
        gaps.append((gap, kernels[i-1].name, kernels[i].name))
gaps.sort(reverse=True)
```

#### 4. GPU 利用率统计

```python
total_wall = kernels[-1].end - kernels[0].start
total_busy = sum(k.end - k.start for k in kernels)
utilization = total_busy / total_wall
```

#### 5. 识别 Vision→Text 过渡 (找第一个 _rms_norm_fwd_kernel)

```python
for i, k in enumerate(kernels):
    if '_rms_norm_fwd' in k.name and i > 10:
        text_start = k.start
        vision_end = kernels[i-1].end
        gap = text_start - vision_end
        print(f"Vision->Text gap: {gap/1e3:.1f}us")
        break
```

## 已有 Trace 索引

| 路径 | 说明 | 日期 |
|------|------|------|
| `traces/nsys_prefill_0411/` | Prefill baseline (position_ids 在 vision 之后, HF generate 路径) | 2026-04-11 |
| `traces/nsys_posids_reorder/` | position_ids 前移到 vision 之前 (HF generate 路径) | 2026-04-11 |
| `traces/nsys_greedydecode/` | **GreedyDecodeEngine 路径** (与 benchmark.py 一致) | 2026-04-11 |

### 04-11 HF generate 路径对比 (position_ids reorder)

| 指标 | Before | After | 变化 |
|------|--------|-------|------|
| Wall time (1 run) | 55.02ms | 52.29ms | -2.73ms |
| GPU busy | 33.00ms | 35.17ms | +2.17ms |
| GPU idle gaps | 22.02ms | 17.12ms | -4.90ms |
| GPU 利用率 | 60.0% | 67.3% | +7.3pp |
| Text decoder 起始 | 30.42ms | 24.71ms | -5.71ms |
| 最大 kernel gap | 1550us | 260us | -1290us |

### 04-11 关键发现: HF generate vs GreedyDecodeEngine 路径差异

`run_nsys_prefill.py` 之前使用 `vlm._model.generate()` (HF generate), 与
benchmark.py 实际路径 (ModelWrapper -> GreedyDecodeEngine -> `self._model()` 直接调用)
不同。两条路径的 text_fastpath 行为差异巨大:

| 指标 | HF generate | GreedyDecodeEngine |
|------|-------------|-------------------|
| Wall time | 52,291us | **42,661us** (-9.6ms) |
| GPU busy | 35,171us | 35,038us |
| GPU 利用率 | 67.3% | **82.1%** (+14.9pp) |
| cudaStreamSync > 100us | 3,517us | **196us** (仅末尾) |
| 最大 kernel gap | 1,550us | **352us** (仅开头) |
| text_fastpath | 失败 (HF 传 cache_position) | **通过** |

原因: HF `generate()` 内部 `prepare_inputs_for_generation` 计算 `cache_position` 并传递,
导致 `_language_model_forward_prefill_fastpath` 的 `cache_position is None` 检查失败,
fallback 到 HF 标准 forward (创建 causal mask 触发 3.5ms sync)。
GreedyDecodeEngine 直接调用模型, 不经过 `prepare_inputs_for_generation`, `cache_position`
保持 None, text_fastpath 正常激活。

已修复 `run_nsys_prefill.py` 使用 `vlm.model` (ModelWrapper) 匹配真实 benchmark 路径。

## 注意事项

1. **nsys 报告路径**: `-o` 指定的是不含扩展名的路径, nsys 自动添加 `.nsys-rep`
2. **SQLite 导出是额外步骤**: 采集完 `.nsys-rep` 后需要手动 `nsys export --type sqlite` 才能得到 `.sqlite`
3. **Trace 文件存放**: 统一放 `traces/` 目录下, 对比实验用子目录区分
4. **NCU 不可用**: 当前环境 NCU 只能 `--set none` (kernel 名 + grid/block dims), 无法采集硬件性能计数器
5. **采集时 repeats**: 建议 `--repeats 1` 减小 trace 大小; 需要多次对比时用 `--no-nsys` 模式取 cudaEvent 均值
