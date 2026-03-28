# Decode GEMV 性能分析与优化尝试

> 更新日期: 2026-03-28
> 目标: 总结 decode 阶段 `o_proj / down_proj / lm_head` 的性能现状、已验证尝试与后续优化方向

---

## 1. 背景

最新 steady-state decode trace 显示，当前 decode 阶段的主时间块已经大致变成：

- `o_proj + down_proj` GEMV: `~1.04 ms / step`
- Attention: `~0.67 ms / step`
- `lm_head`: `~0.36 ms / step`

对应 trace 分解见：

- [traces/compare/triton_decode_step39_static_kv_write](../traces/compare/triton_decode_step39_static_kv_write)

也就是说，Attention 已经不是唯一主热点，decode 线性层的 GEMV 已经足够大，值得单独梳理。

---

## 2. 目标算子与形状

Qwen3-VL-2B 当前 decode 单 token 下，相关 GEMV 的核心形状是：

| 算子 | 输入 | 权重 | 备注 |
|------|------|------|------|
| `o_proj` | `[1, 1, 2048]` | `[2048, 2048]` | 每层 1 次 |
| `down_proj` | `[1, 1, 6144]` | `[2048, 6144]` | 每层 1 次 |
| `lm_head` | `[1, 1, 2048]` | `[151936, 2048]` | 每 token 1 次 |

按每个 token 统计：

- `o_proj + down_proj` 共 `28 * 2 = 56` 次
- `lm_head` 共 `1` 次

其中 `o_proj / down_proj` 属于“小而频繁”的 per-layer GEMV，`lm_head` 则是“大而固定”的 single-call GEMV。

---

## 3. 真实路径 microbenchmark 工具

当前使用：

- [bench_decode_linears.py](../bench_decode_linears.py)

它不是脱离真实路径的人造 matvec，而是：

1. 跑真实 sample 的 benchmark 风格 prefill + custom decode
2. 在指定 `layer_idx / decode_step` 抓取 decode 时的真实输入张量
3. 直接对不同线性实现做 benchmark

这点很重要，因为 decode 的真实 bucket、cache 长度、张量 layout 都会影响结论。

---

## 4. 已测实现

目前对每个目标 GEMV 都测了这些实现：

- `F.linear` 基线
- `F.linear(weight.contiguous())`
- `x @ weight.t()`
- 自写 Triton 普通 `matvec`
- 自写 Triton `split-K` matvec (`split_k=4/8`)

测试点位：

- sample0 / layer0 / decode_step0
  - [results/linear_microbench_step40_smoke.json](../results/linear_microbench_step40_smoke.json)
- sample0 / layer14 / decode_step16
  - [results/linear_microbench_step40_layer14_step16.json](../results/linear_microbench_step40_layer14_step16.json)

另外对 `lm_head` 还做了一轮 split-K 参数扫：

- [results/lm_head_splitk_sweep_step40.json](../results/lm_head_splitk_sweep_step40.json)

---

## 5. 核心结果

### 5.1 `o_proj`

两组点位结果都一致：

- `F.linear`: `~0.015 ms`
- `weight.contiguous()`：几乎持平，最多 `~1%~3%`
- `matmul(weight.t())`：更慢，约 `0.77x`
- Triton 普通 matvec：明显更慢，约 `0.33x`
- Triton split-K：更慢，约 `0.18x~0.20x`

结论：

- `o_proj` 这类 `2048 -> 2048` 的 decode GEMV，当前 `F.linear` 已经很难通过“直接改调用方式”打赢

### 5.2 `down_proj`

两组点位也一致：

- `F.linear`: `~0.028 ms`
- `weight.contiguous()`：几乎持平，约 `+0%~1%`
- `matmul(weight.t())`：几乎持平
- Triton 普通 matvec：只有 `~0.51x~0.59x`
- Triton split-K：只有 `~0.30x~0.34x`

结论：

- `down_proj` 这类 `6144 -> 2048` 的 decode GEMV，同样没有简单替换收益

### 5.3 `lm_head`

`lm_head` 是唯一一个“自写 kernel 有一点接近基线”的点，但仍没赢：

- `F.linear`: `~0.358~0.359 ms`
- `weight.contiguous()`：几乎持平，最好约 `+0.3%`
- `matmul(weight.t())`：几乎持平
- Triton 普通 matvec：约 `0.91x`
- Triton split-K4：约 `0.96x`
- Triton split-K8：约 `0.94x`

`lm_head` split-K sweep 最优也只是：

- baseline: `0.3574 ms`
- best trial: `0.3633 ms`

也就是仍然比 baseline 慢。

---

## 6. 额外尝试：`lm_head` 后处理小优化

还试过一个很便宜的小改动：

- graph decode 中不再先把 `outputs.logits` copy 到 `_logits_buf`
- 直接对 `outputs.logits[:, -1, :]` 做 `argmax`

对应结果：

- [results/result_step40_direct_argmax_10.json](../results/result_step40_direct_argmax_10.json)

结果是：

- step39 基线: `194.24 tok/s`
- direct argmax: `191.71 tok/s`

结论：

- 这条线没有收益，已经回退

---

## 7. 当前判断

截至目前，可以比较明确地下结论：

### 7.1 已基本证伪的方向

1. **简单 API 替换**
   - `F.linear` -> `matmul(weight.t())`
   - `F.linear` -> `contiguous weight`
   - 这类不会带来实质收益

2. **普通 Triton matvec 直接替换 cuBLAS**
   - 对 `o_proj / down_proj` 明显更慢
   - 对 `lm_head` 也没有超过 baseline

3. **简单 split-K Triton matvec**
   - 对 `o_proj / down_proj` 反而显著更差
   - 对 `lm_head` 虽然更接近 baseline，但仍未打平

### 7.2 目前最合理的判断

- `o_proj / down_proj` 这两块，**单独替换 GEMV 本体** 基本没有 easy win
- `lm_head` 可能还有一点空间，但也不像是“随手写个 Triton kernel 就能明显超 cuBLAS”

所以如果后面专门搞 GEMV，方向不应该是“继续换 GEMV API”，而应该是：

1. **更深的融合**
   - 把 GEMV 前后的 elementwise / residual / norm / activation 一起吃掉
   - 让优势来自“减少中间落地与 kernel 边界”，而不是单纯比 GEMV 本体快

2. **只针对 `lm_head` 做更激进的定制**
   - 它 shape 特殊，`N` 很大、`K=2048` 固定、输出只需要 top-1
   - 更像一个“`gemv + argmax`”问题，而不是普通线性层

3. **把 `o_proj / down_proj` 当作融合入口，而不是替换目标**
   - `o_proj` 可以考虑和 attention output 的布局整理一起看
   - `down_proj` 可以考虑和前面的 `SiLU * up` / residual 链一起看

---

## 8. Triton 还是 CUDA

当前阶段我的判断是：

- **如果只是继续做“单 GEMV kernel 替换”**，Triton 意义已经不大了
- **如果后面要认真搞 GEMV，尤其是 `lm_head` 或更深融合，基本会更偏向 CUDA**

原因：

1. `o_proj / down_proj` 已经说明：想靠普通 Triton matvec 打赢 cuBLAS 不现实
2. `lm_head` 真正有潜在价值的方向更像：
   - persistent kernel
   - warp-specialized reduction
   - `gemv + argmax` 融合
   - 更细的寄存器 / shared memory / launch 控制
3. 这类东西 Triton 不是完全不能做，但实现空间和可控性都会比 CUDA 更受限

所以如果后面单开 GEMV 专项，我更倾向这样分工：

- **`o_proj / down_proj`**
  - 先优先想“融合”，不是想“换 GEMV 内核”

- **`lm_head`**
  - 如果要继续深挖，直接按 **CUDA 专项优化** 来规划更合理

---

## 9. 建议的后续路线

按优先级排序，建议是：

1. **先把 `lm_head` 当成独立专项**
   - 目标不是 `Linear -> custom GEMV`
   - 而是 `lm_head + argmax` 一体化
   - 如果要继续，优先考虑 CUDA

2. **`o_proj / down_proj` 不要再花时间做单算子替换**
   - 当前证据已经足够说明这条路不值

3. **GEMV 后续应主要看融合**
   - `attention output -> o_proj`
   - `SiLU*up -> down_proj`
   - `lm_head -> argmax`

4. **继续保留 microbench 工具**
   - 后面如果上 CUDA 原型，第一验证入口就还是这套真实路径 microbench

---

## 10. 一句话结论

当前 decode GEMV 的结论不是“完全没得做”，而是：

- **`o_proj / down_proj`：单算子替换基本没戏，后续应看融合**
- **`lm_head`：可能还有空间，但更像 CUDA 专项问题，而不是 Triton 小修小补**
