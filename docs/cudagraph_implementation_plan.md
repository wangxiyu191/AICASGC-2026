# CUDA Graph 详细实施计划

> 目标: 在不修改模型参数的前提下，为 Qwen3-VL-2B-Instruct 的 decode 阶段引入 `StaticCache + custom greedy decode + torch.cuda.CUDAGraph`
> 范围: 仅优化 decode；prefill/vision encoder 保持现有路径
> 约束:
> - 不允许量化、稀疏化或任何模型参数改动
> - 评测为单 batch，核心指标是 TTFT、吞吐、准确度
> - 评测机为单卡 NVIDIA A800 80GB PCIe
> - 开发和测试必须在 `./venv` 中完成
> - 所有 benchmark 结果写入 `results/`，每次优化动作和测试结果记录到 `result.csv`
> - 任何涉及 `from_pretrained` 的脚本必须带 `TQDM_DISABLE=1`

---

## 1. 背景与目标

当前代码已经完成以下几类优化:

- `sdpa` 固定为 attention backend，避免 `flash_attention_2` 在 decode 阶段的 CPU dispatch 退化
- `OptimizedDecoderLayer` 已融合 QKV / gate_up projection
- `RMSNorm` / `RoPE` 已实现 prefill 走 Triton、decode 走 native ops 的分支

这些优化把 decode 阶段的纯 GPU 计算时间压到了较低水平，但仍未解决系统级瓶颈。

### 当前关键结论

- decode 每步 wall time 约 `25.7ms`
- decode 每步 GPU kernel 时间约 `4.3ms`
- GPU 利用率仅约 `17%`
- 剩余瓶颈主要来自:
  - HF `generate()` Python loop
  - `aten::item` / `aten::is_nonzero` 同步
  - 大量 `cudaLaunchKernel`
  - Dynamic KV cache 的 `cat/concat`

### 本阶段目标

本轮实现不追求一次完成所有外围优化，而是先建立一个稳定、可验证、可逐步扩展的 decode 执行框架:

1. 以自定义 greedy decode 替代 HF `generate()` 的 decode loop
2. 用 `StaticCache` 固定 KV cache 地址
3. 仅对单 token decode step 做 `CUDA Graph capture/replay`
4. 保持输出与现有 greedy 生成一致
5. 为后续进一步做 fused residual/norm、额外 graph 分桶预留结构

---

## 2. 当前代码与接口现状

### 2.1 真实 benchmark 调用入口

benchmark 并不是调用 `VLMModel.generate()`，而是直接调用:

```python
model_obj = model.model
output_ids = model_obj.generate(...)
```

因此所有 decode 路径改造必须落在:

- `evaluation_wrapper.py`
- `ModelWrapper.generate()`

只改 `VLMModel.generate()` 对 benchmark 无效。

### 2.2 当前 wrapper 状态

当前 `ModelWrapper.generate()` 只是透传:

- 非 trace 模式: `self._model.generate(*args, **kwargs)`
- trace 模式: 用 `viztracer` 包一层后再调用 `self._model.generate`

这意味着:

- benchmark 的 TTFT/throughput 仍然走 HF generation stack
- 现有 fused layer / native decode branch 只优化了 `forward` 内部算子
- Python loop、sync 和 launch overhead 还没有被消除

### 2.3 当前模型行为中对 CUDA Graph 有利的点

本地 `transformers 5.2.0` 里已经具备以下条件:

- 首次 prefill 后，只要 `past_key_values` 非空，后续 decode 不会再执行 vision encoder
- `Qwen3-VL` 的 decode 位置编码可以沿着 `cache_position + rope_deltas` 推进
- `StaticCache` 底层使用 `index_copy_` 写入 KV，不再 `cat`

因此 decode 具备做 static-address capture 的基本条件。

---

## 3. 总体设计

### 3.1 执行策略

采用“两阶段执行”:

1. **Prefill 阶段**
   - 继续用普通 eager forward
   - 输入为完整的 `input_ids + pixel_values + image_grid_thw + attention_mask`
   - 输出:
     - 首 token logits
     - 已填充的 `StaticCache`
     - `rope_deltas`
     - decode 起始位置

2. **Decode 阶段**
   - 输入固定为单 token (`shape=(1, 1)`)
   - 使用固定长度的 `StaticCache`
   - 通过 placeholder tensor 原地更新:
     - `input_ids`
     - `cache_position`
     - `position_ids`
     - `attention_mask`
   - 用 `torch.cuda.CUDAGraph` 捕获一次 decode step，再多次 replay

### 3.2 第一版范围控制

第一版先严格限制在 benchmark 主路径:

- `batch_size == 1`
- `do_sample == False`
- `use_cache == True`
- 文本 greedy decode
- 默认只支持性能测试用的 `max_new_tokens <= 128`

超出范围的调用统一 fallback 到原始 `self._model.generate(...)`，确保:

- 不影响 accuracy 评估的长输出场景
- 不影响调试脚本
- 不因边界条件导致提交风险

### 3.3 推荐分层结构

建议在 `evaluation_wrapper.py` 内引入以下内部组件:

- `ModelWrapper`
  - benchmark 对外入口
  - 参数检查与 fallback
- `GreedyDecodeEngine`
  - 管理 prefill、decode loop、输出拼接
- `DecodeCudaGraphRunner`
  - 管理 placeholder、warmup、capture、replay
- `DecodeState`
  - 封装本次请求的 cache、position、attention_mask、输出 buffer

第一版如果不想拆太多文件，也至少应在同文件内把职责按类拆开，避免 `ModelWrapper.generate()` 变成超长函数。

---

## 4. 关键实现点拆解

## 4.1 自定义 greedy decode loop

### 目标

完全绕过 HF generation loop 的以下逻辑:

- stopping criteria
- logits processor / logits warper
- 逐步 Python bookkeeping
- 由此带来的 `item()` / `is_nonzero()` 同步

### 输入

来自 benchmark 的标准输入:

- `input_ids`
- `attention_mask`
- `pixel_values`
- `image_grid_thw`
- `max_new_tokens`
- `do_sample=False`
- `temperature=0.0`
- `use_cache=True`

### 输出

与 HF `generate()` 保持相同的输出格式:

- 返回完整 `output_ids`
- 包含 prompt + generated tokens

### decode 规则

- 固定 greedy: `argmax(logits[:, -1, :])`
- 循环次数由 `max_new_tokens` 决定
- 不在循环内部用 Python `break` 做 EOS stopping 作为主路径
- 可在循环结束后做一次后处理截断 EOS 后续 token

### 为什么第一版不做动态提前停止

因为 benchmark 的 throughput 测试关心端到端 token 生成速率，固定长度更适合 graph replay:

- 形状稳定
- 控制流稳定
- 避免每步 host 侧判断

如果需要兼顾 accuracy 阶段长输出，可对 `max_new_tokens > 128` 直接 fallback。

---

## 4.2 StaticCache 接入

### 目标

把 decode 的 KV cache 从动态增长改为静态预分配，满足 graph 的地址稳定要求。

### 初始化方式

使用:

```python
from transformers.cache_utils import StaticCache

past_key_values = StaticCache(
    config=self._model.config,
    max_cache_len=input_len + max_new_tokens,
)
```

注意:

- 传 `self._model.config` 即可，`StaticCache` 内部会 `get_text_config(decoder=True)`
- 不再使用旧版本尝试中的 `batch_size/device/dtype` 参数

### 关键注意事项

- prefill 时就把 `StaticCache` 传入模型
- capture 前必须确保 cache 已经完成 lazy initialization
- decode 期间只允许原地写入，不允许重新创建 cache 对象

### 风险点

- `get_seq_length()` 内部会扫描缓存占用，不适合放在 decode 热路径反复调用
- decode 逻辑必须自己维护当前位置计数器，不能每步依赖 `cache.get_seq_length()`

---

## 4.3 position_ids / cache_position / attention_mask

这是第一版最容易出错的部分，也是之前 StaticCache 尝试失败的核心。

### `cache_position`

decode 第 `t` 步的 `cache_position` 应由 host 侧显式维护:

- prefill 后起始值 = `input_len`
- 每生成一个 token，自增 `1`

建议使用固定 shape placeholder:

```python
self.cache_position_buf = torch.zeros((1,), dtype=torch.long, device=device)
```

每步只做:

```python
self.cache_position_buf[0] = current_position
```

### `position_ids`

对 Qwen3-VL 来说，decode 后续位置编码需要沿用 prefill 产出的 `rope_deltas`。

建议策略:

- prefill 后读取 `outputs.rope_deltas`
- decode 时显式构造 `position_ids`
- 避免在 graph 内再次走复杂推导路径

推荐 placeholder 形状:

```python
position_ids.shape == (3, 1, 1)
```

数值更新逻辑应等价于:

```python
text_pos = current_position + rope_delta
position_ids[...] = text_pos
```

第一版只支持单图片、单 batch、单 token decode，因此这里完全可以直接做定长原地写入。

### `attention_mask`

第一版推荐保留显式 2D `attention_mask`，不要一开始就尝试去掉它。

原因:

- 行为更接近现有 HF 路径
- 更容易做数值对齐
- debug 时更容易排查

实现上:

- 预分配长度 `max_cache_len`
- prefill 区间填 1
- decode 每步只把当前位置置为 1

推荐 placeholder:

```python
attention_mask_buf.shape == (1, max_cache_len)
```

传给模型时可切片到固定长度也可直接整张传入。第一版优先选择“整张传入”并验证开销，再视结果决定是否进一步裁切。

---

## 4.4 DecodeCudaGraphRunner 设计

### 核心职责

- 管理 decode 输入 placeholder
- 执行 warmup
- 捕获 graph
- replay graph
- 暴露“给定 next input token，返回 next logits / next token”的接口

### 需要固定的张量

至少包括:

- `input_ids_buf`: `(1, 1)`，`torch.long`
- `cache_position_buf`: `(1,)`，`torch.long`
- `position_ids_buf`: `(3, 1, 1)`，`torch.long`
- `attention_mask_buf`: `(1, max_cache_len)`，mask dtype 以现有模型路径兼容为准
- `logits_buf` 或 `next_token_buf`
- `StaticCache`

### 推荐捕获内容

捕获完整的单步 decode:

```python
outputs = model(
    input_ids=input_ids_buf,
    attention_mask=attention_mask_buf,
    position_ids=position_ids_buf,
    past_key_values=static_cache,
    use_cache=True,
    cache_position=cache_position_buf,
)
logits = outputs.logits[:, -1, :]
next_token = torch.argmax(logits, dim=-1)
```

这样 replay 后 host 侧只需要:

1. 把上一步 token 写入 `input_ids_buf`
2. 更新 `cache_position_buf`
3. 更新 `position_ids_buf`
4. 更新 `attention_mask_buf`
5. `graph.replay()`
6. 从 `next_token_buf` 读结果并写入输出 buffer

### warmup 要求

capture 前至少做 2~3 次 eager warmup，确保:

- cudnn/cublas 内部 workspace 稳定
- `StaticCache` 已 lazy init
- SDPA 后端已选定
- graph capture 不包含首次初始化开销

### graph 缓存策略

第一版只做一个 bucket:

- key: `(max_cache_len,)` 或 `(input_len_bucket, max_new_tokens)`

更保守一点，可以先按“本次请求独享”创建 graph runner，不做跨请求复用；等闭环稳定后再做 bucket cache。

---

## 4.5 与现有 trace 能力兼容

当前 `ModelWrapper` 已支持 `ENABLE_VIZTRACE=1`。

建议:

- trace 模式默认关闭 `CUDA Graph` 主路径，继续走 eager 或 custom decode eager
- 原因是 `viztracer` 对 graph replay 的可观测性有限，且会扰乱时序

实现上可以增加环境变量:

- `ENABLE_CUSTOM_DECODE=1`
- `ENABLE_CUDAGRAPH=1`
- `ENABLE_VIZTRACE=1`

优先级建议:

- `ENABLE_VIZTRACE=1` 时，禁用 cudagraph replay，只保留 custom decode eager

---

## 5. 分阶段实施步骤

## Step A: 建立 custom decode eager 基线

### 目标

先不引入 graph，只替换 generation loop，验证:

- 输出是否正确
- `position_ids/cache_position/attention_mask` 是否完全打通
- `StaticCache` 是否可用

### 代码改动

- 在 `evaluation_wrapper.py` 新增 custom decode engine
- `ModelWrapper.generate()` 根据条件切换到 custom decode
- 暂不引入 `torch.cuda.CUDAGraph`

### 验证标准

- 小样本逐 token 与 HF `generate()` 对齐
- benchmark 至少不崩溃
- TTFT 允许暂时持平或略差
- throughput 不应明显退化到旧 WIP 水平

### 退出条件

只有 Step A 数值稳定，才能进入 Step B。

## Step B: 替换 DynamicCache 为 StaticCache

### 目标

确认 decode 不再出现 `cat/concat` 类 cache 增长行为。

### 验证手段

- 用小样本跑 `torch.profiler` / `viztracer`
- 检查 trace 中 `cat` 时间是否明显下降
- 确认 decode 热路径中没有 `get_seq_length()` 的重复扫描

## Step C: 引入单步 decode graph capture

### 目标

只捕获 `seq_len=1` 的 decode step。

### 实现要求

- 先 eager warmup，再 capture
- capture 期间不得创建新 tensor 或新 cache
- graph 外只做 placeholder 写入与输出拷贝

### 验证标准

- 数值与 Step B 一致
- throughput 明显提升
- 运行过程中无 graph overwrite / illegal memory access

## Step D: 做 fallback 与稳定性收尾

### 目标

保证提交安全，不因为 graph 入口过于激进而影响 accuracy 或异常输入。

### fallback 条件建议

以下任一条件不满足即 fallback:

- `batch_size != 1`
- `do_sample == True`
- `use_cache != True`
- `pixel_values_videos is not None`
- `max_new_tokens > 128`
- 捕获初始化失败
- 任意一次 replay 抛异常

fallback 路径统一回到:

```python
self._model.generate(*args, **kwargs)
```

---

## 6. 建议的代码改动清单

优先修改:

- `evaluation_wrapper.py`

可能需要小改:

- `optimized_layer.py`

### `evaluation_wrapper.py` 预期新增内容

- `DecodeCudaGraphRunner`
- `GreedyDecodeEngine`
- 环境变量开关
- fallback 分流逻辑
- 输出拼接逻辑

### `optimized_layer.py` 可能需要检查的点

- decode 分支不能在 graph 热路径里有多余的 shape-dependent Python 分支
- 尽量避免隐式新 tensor 分配
- 确认 `_apply_rotary_emb` / `_rms_norm` 的 decode 分支对 `seq_len=1` 稳定

如果 graph capture 报错，再回头看是否需要把 decode 分支里某些 `torch.cat` 或临时 tensor 改成更静态的写法。

---

## 7. 验证计划

## 7.1 数值正确性

建议新增或复用脚本，比较以下几条路径:

- 原始 HF `generate()`
- custom decode eager
- custom decode + StaticCache
- custom decode + StaticCache + CUDAGraph

比较项目:

- 生成 token 序列是否完全一致
- 至少对 3~10 个真实样本做比对
- 覆盖:
  - 短问题
  - 长 prompt
  - 不同图片尺寸

## 7.2 性能验证

统一使用:

```bash
TQDM_DISABLE=1 source ./venv/bin/activate
```

建议顺序:

1. `--num-samples 3` 快速冒烟
2. `--num-samples 10` 对比 TTFT/throughput
3. 必要时再扩到更大样本

结果文件统一放:

- `results/result_cudagraph_smoke.json`
- `results/result_cudagraph_10.json`
- 后续稳定版再按阶段命名

## 7.3 trace / profile 验证

目标不是每步都 trace，而是定点确认:

- decode 热路径中 `cat` 是否消失
- Python loop 是否显著缩短
- `cudaLaunchKernel` 数量是否下降

建议:

- graph 前先跑一次 custom decode eager trace
- graph 后只做少量 profiler 验证，避免 trace 影响结论

---

## 8. result.csv 记录规范

每完成一个稳定阶段，记录一行结果，建议命名:

- `13,Phase4a: custom greedy decode eager,...`
- `14,Phase4b: custom decode + StaticCache,...`
- `15,Phase4c: custom decode + StaticCache + CUDAGraph,...`

备注中必须写清:

- 是否启用 graph
- 是否仍保留 fallback
- 样本数
- 是否只在 `max_new_tokens<=128` 启用

---

## 9. commit 切分建议

建议至少分 3 个 commit，便于回滚:

1. `phase4a: add custom greedy decode eager path`
2. `phase4b: switch decode cache to StaticCache`
3. `phase4c: add CUDA Graph runner for single-token decode`

如果某一步结果不理想，也应保留对应 commit，方便后续比较。

---

## 10. 主要风险与应对

### 风险 1: `position_ids` 计算错误导致输出漂移

应对:

- 先做 eager 版逐 token 对齐
- 把 `rope_deltas` 和 `cache_position` 打印出来核对前几步

### 风险 2: `StaticCache` 虽可运行，但热路径里仍有隐藏动态逻辑

应对:

- 避免在 loop 中调用 `get_seq_length()`
- profiler 检查 `cat`/分配行为是否真的消失

### 风险 3: graph capture 与某些 decode 分支不兼容

应对:

- 保持第一版仅支持 `seq_len=1` decode
- capture 失败立即 fallback，不阻塞 benchmark

### 风险 4: accuracy 场景受影响

应对:

- 对长输出或复杂场景默认 fallback
- 先保证 benchmark 性能路径稳定，再逐步扩大适用范围

### 风险 5: A10 开发通过，但 A800 表现不同

应对:

- 避免依赖仅在 sm_86 上偶然有效的技巧
- 尽量走 PyTorch/transformers 原生 cache 与 graph 机制
- 保持实现纯 Python + PyTorch，不增加额外编译依赖

---

## 11. 第一版验收标准

满足以下条件即可认为 `cudagraph` 第一版可提交测试:

1. `benchmark.py --num-samples 3` 稳定通过
2. custom decode 路径生成结果与原始 greedy 基本一致
3. decode trace 中 `cat/concat` 显著减少或消失
4. throughput 相对当前 Step 12 有明确提升
5. TTFT 没有出现不可接受的恶化
6. `result.csv` 与 `results/` 已同步更新

---

## 12. 建议的直接行动顺序

按这个顺序执行，风险最低:

1. 在 `evaluation_wrapper.py` 里重建 custom decode eager，不启 graph
2. 用 `StaticCache` 跑通 prefill + decode，并做逐 token 对齐
3. 加入 placeholder buffer，去掉 decode 热路径里的动态 tensor 创建
4. 实现 `DecodeCudaGraphRunner`，先 capture 单步 decode
5. 跑小样本 benchmark，记录 `result.csv`
6. 如性能成立，再补 fallback、环境变量和文档

这个顺序的重点是: 先把“正确执行”跑通，再把“graph replay”压上去，不要同时调三件事。
