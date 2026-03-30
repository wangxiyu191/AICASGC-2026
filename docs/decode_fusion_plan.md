# Decode 融合优化计划

更新时间：2026-03-31

## 1. 背景

当前 A800 主线已经完成：

- fused QKV / fused gate_up
- decode 小 shape 的 RMSNorm / QK-Norm+RoPE / attention / StaticCache 写回优化
- decode `o_proj/down_proj` 的 CUDA nosplit GEMV
- CUDAGraph decode 主路径

现阶段继续深挖单个 kernel 仍可能有收益，但已经进入明显的边际收益递减区间：

- `attention` 还有少量空间，但单点优化上限有限
- `qkv/gate_up/down_proj/o_proj` 的单算子 microbench 已经较接近当前口径下的局部最优
- 从 trace 和单层 replay bench 看，下一步更值得投入的是 **跨算子融合**，目标是减少 launch、减少中间张量落地、减少 HBM roundtrip

因此，后续 decode 优化的主方向切换为：

1. 优先做 decode 单 token 路径的深融合
2. 先做风险低、收益实的局部 fuse
3. 每一步都先用单层链式 bench 验证，再决定是否进整机 benchmark

## 2. 当前 decode 层主链

当前 decode 单层主链位于 [optimized_layer.py](../optimized_layer.py)：

```text
input_rmsnorm
-> qkv
-> q_norm_rope + k_norm_rope
-> kv_cache_update
-> attention
-> o_proj
-> post_attn_add_rmsnorm
-> gate_up
-> silu_mul
-> down_proj
-> residual_add
```

其中当前最适合继续融合的段主要有两组：

1. `post_attn_add_rmsnorm -> gate_up -> silu_mul -> down_proj`
2. `attention -> o_proj`

## 3. 优先级排序

先说明一个原则：

- 后续融合优先级不再只看“理论上能少几个 kernel”
- 而是看 **实现风险 / 对当前已站住 kernel 的破坏程度 / 单层 replay 预估收益**

按这个标准，优先级调整为：

1. `attention` 输出布局直连 `o_proj`
2. `silu_mul + down_proj`
3. `post_attn_add_rmsnorm + gate_up`
4. `gate_up` kernel 本体继续优化（作为备选支线）

### 3.1 第一优先级：`attention -> o_proj` 邻域融合

不建议第一步就做“真正的 attention 数学 + o_proj 完全同 kernel 融合”。

更现实的切入顺序：

1. 先做 attention 输出布局直连 `o_proj`
2. 再评估是否需要继续做更深层的融合

可选方向：

- 让 `_small_decode_attention` 直接写到 `o_proj` 需要的线性 layout
- 尽量消掉 `reshape/contiguous/handoff` 这一段
- 复用 layer 生命周期内的 workspace

预期收益：

- 小到中等

工程复杂度：

- 中等

风险：

- attention 路径分支多（split-KV / dot / prefix）
- 但如果只改输出布局、不改 attention 数学，风险明显低于真正的 attention 融合

优先原因：

- 改动边界比 `norm + gate_up` 小得多
- 不需要碰当前已经较强的 `o_proj` nosplit kernel
- 更容易先做出一个干净的正收益

### 3.2 第二优先级：`silu_mul + down_proj` 融合

目标：

- 去掉独立的 `silu_mul` launch
- 避免 materialize 6144 维中间向量后再喂给 `down_proj`

当前形态：

```text
gate_up(hidden_states)
-> gate, up = chunk(2)
-> silu(gate) * up
-> down_proj
```

目标形态：

```text
gate_up(hidden_states)
-> fused_silu_downproj(gate_up, down_proj_weight, bias)
```

核心思路：

- `gate_up` 仍保留为一次 fused linear
- 在 `down_proj` kernel 内部直接从 `gate_up` 的前后两半读取
- kernel 内边算 `silu(gate) * up`，边做 `down_proj` reduction
- 不再生成独立的 `hidden_states = silu(gate) * up`

需要先确认的一点：

- 当前 `silu_mul` 本身的纯算术代价不大
- 真正可能节省的是独立 launch 和中间向量 handoff
- 因此开始写 kernel 之前，必须先用单层 replay trace 量化 `silu_mul -> down_proj` 之间的真实 gap
- 如果这一段 gap 很小，则该方向的整机上限可能也很有限

预期收益：

- 小到中等
- 主要来自减少一次 launch 和一次中间向量读写，而不是节省大量算术

工程复杂度：

- 中等

风险：

- kernel 内寄存器压力上升
- `down_proj` 原本是当前比较强的 nosplit kernel，融合后可能因为额外 epilogue 逻辑损失主循环效率
- 如果融合后 occupancy 掉得太多，可能把原本的收益全部吃掉

优先原因：

- 仍然是可做的方向
- 但不再把它视为“默认第一优先级”
- 只有在单层 trace 证明 handoff 成本确实存在时才推进

### 3.3 第三优先级：`post_attn_add_rmsnorm + gate_up` 融合

目标：

- 不再先把 2048 维归一化结果写回，再由 `gate_up` 重新读取

当前形态：

```text
residual + attn_output
-> rms_norm
-> gate_up
```

目标形态：

```text
fused_add_rmsnorm_gateup(residual, attn_output, gate_up_weight)
```

核心思路：

- 不能简单假设“把 add+rmsnorm reduction 直接塞进 gate_up GEMV 主循环”就能成立
- 更现实的思路应是：
  1. 保留 `add + rmsnorm` 作为一个独立逻辑块
  2. 优先消除其输出写回与 `gate_up` 输入读取之间的 handoff
  3. 只有在确认需要更深融合时，再考虑 persistent kernel / cooperative group 方案

预期收益：

- 中等偏高

工程复杂度：

- 中高

风险：

- reduction + matvec 混合后更容易出现 occupancy / register spill 问题
- 若需要 CTA 间同步，工程复杂度和 graph 兼容性风险都会明显上升
- 数值正确性验证要更仔细

优先原因：

- 这段本身仍然值得做
- 但它是当前三项里实现难度最高的一项
- 因此应放到 `attention -> o_proj` 和 `silu_mul + down_proj` 之后

### 3.4 暂缓项：`input_rmsnorm + qkv`、`q/k_norm_rope + qkv` 更深融合

这些方向理论上也可做，但暂时不是第一优先：

- `qkv` 当前已经接近较优实现
- 把 norm / rope / qkv 进一步缠在一起会显著增加工程复杂度
- 短期内更可能“做很久但整机收益有限”

因此建议暂缓，除非前两项都已经做完且收益不够。

### 3.5 备选支线：`gate_up` kernel 本体继续优化

虽然当前主方向已经切到融合，但 `gate_up` 仍是 decode 层里的大头之一。

如果前三个融合方向验证后收益都不明显，可回到这一条支线：

- 看 `gate_up` 主循环是否还能做更好的 pipeline / double-buffering
- 看 weight layout 是否还有更适合向量化加载的组织方式

注意：

- 这条线不作为当前主线优先级
- 但它是一个合理的止损备选方向

## 4. 粗收益估计（待用单层 viztrace 复核）

下表只作为当前 planning 用的粗估，不作为最终收益承诺：

| 方向 | 当前热点位置 | 估计节省（单层） | 备注 |
|------|-------------|------------------|------|
| attention 输出布局直连 `o_proj` | attention 输出 handoff | ~2-3 us | 需先用单层 viztrace 复核 contiguous/handoff 是否真实存在 |
| `silu_mul + down_proj` | MLP 后半段 handoff | ~2-5 us | 上限可能不高，开始前必须先量化 gap |
| `post_attn_add_rmsnorm + gate_up` | MLP 前半段 handoff | ~5-8 us | 实现难度最高，收益估计也最不确定 |
| `gate_up` kernel 本体优化 | `gate_up` GEMV | ~2-5 us | 作为备选支线，不是当前主线 |

结论：

- 当前最值得先做的是 **低风险、可快速验证正收益** 的方向
- 因此优先顺序仍应是：先 `attention` 输出布局直连，再看 `silu_mul + down_proj`

## 5. 建议实施顺序

### Phase A：attention 输出布局直连 `o_proj`

改动范围：

- [optimized_layer.py](../optimized_layer.py)
- [cuda_extensions/decode_attention_kernel.cu](../cuda_extensions/decode_attention_kernel.cu)
- [cuda_extensions/decode_attention_ext.cpp](../cuda_extensions/decode_attention_ext.cpp)
- [cuda_decode_attention.py](../cuda_decode_attention.py)

建议做法：

- 先不改 attention 数学
- 只改 attention 输出 layout / handoff 方式
- 尽量让输出直接满足 `o_proj` 使用要求
- 如果收益不明显，及时止损，不继续做更深 attention 融合

通过标准：

- 单层 replay / viztrace 里 attention 到 `o_proj` 的 handoff 明显缩短
- 输出与旧路径对齐
- 10 sample benchmark 没有正确性回退

### Phase B：`fused_silu_downproj`

改动范围：

- [cuda_extensions/decode_gemv_kernel.cu](../cuda_extensions/decode_gemv_kernel.cu)
- [cuda_extensions/decode_gemv_ext.cpp](../cuda_extensions/decode_gemv_ext.cpp)
- [cuda_decode_gemv.py](../cuda_decode_gemv.py)
- [optimized_layer.py](../optimized_layer.py)

建议做法：

- 新增 `fused_silu_downproj_half(...)` CUDA 接口
- 只先支持 decode 单 token、fp16、当前主用 shape
- 默认关闭，挂 env 开关
- 如果新 kernel 条件不满足，自动回退到旧路径

通过标准：

- 单层链式 bench 总 replay 时间显著下降
- 真实 payload 输出与旧路径对齐
- 10 sample benchmark 不回退准确率

### Phase C：`fused_add_rmsnorm_gateup`

改动范围：

- [optimized_layer.py](../optimized_layer.py)
- 新增 Triton 或 CUDA kernel（优先看实现复杂度）

建议做法：

- 先做 decode 专用
- 只覆盖 `M=1`、单 batch、当前主 shape
- 保留旧路径作 fallback

通过标准：

- 单层链式 bench 的 MLP 前半段明显缩短
- 整层 replay 时间继续下降
- benchmark 不出现 TTFT 明显回退

### Phase D：`gate_up` kernel 本体继续优化（备选）

改动范围：

- [cuda_extensions/decode_gemv_kernel.cu](../cuda_extensions/decode_gemv_kernel.cu)
- [cuda_extensions/decode_gemv_ext.cpp](../cuda_extensions/decode_gemv_ext.cpp)
- [cuda_decode_gemv.py](../cuda_decode_gemv.py)

建议做法：

- 只在前三项都没有足够收益时启用
- 优先尝试对当前 `gate_up` nosplit kernel 做 pipeline / layout 微调
- 继续遵守 default-off + fallback 的约束

## 6. 验证方法

每一步都使用同一套验证顺序，避免直接冲整机 benchmark：

1. 算子级/局部正确性
2. 真实 payload microbench
3. 单层链式 replay bench
4. 单层 viztrace
5. 10 sample benchmark

推荐工具：

- [bench_decode_linears.py](../bench_decode_linears.py)
- [bench_decode_attention.py](../bench_decode_attention.py)
- [bench_decode_layer_chain.py](../bench_decode_layer_chain.py)

重点看两类结果：

- `bench_decode_layer_chain.py` 的单层总 replay 时间是否稳定下降
- viztrace 里 layer 内主链是否真的少了 launch / 少了中间 handoff

另外新增两项必须检查：

1. CUDAGraph replay 稳定性
2. graph 路径是否引入新的 runtime allocation

建议做法：

- 对每个新融合 kernel，固定 payload 连续 replay 1000 次，检查输出是否稳定
- 在 capture 前后、重复 replay 前后记录 `torch.cuda.memory_allocated()`，确认没有异常增量
- 若新 fuse 依赖 workspace，必须在 layer 生命周期内预分配，不允许在 replay 内 `torch.empty`

## 7. 实施约束

为了避免“功能对了但整机反而更差”，后续融合实现必须遵守：

1. 默认关闭，新 fuse 先用 env 开关保护
2. graph replay 路径不能引入新的 runtime allocation
3. workspace 必须复用，优先挂到 layer 生命周期
4. fallback 路径必须完整保留
5. 每一步都在 A800 上复核，不用 A10 结论直接拍板

## 8. 止损条件

出现以下情况时应停止继续加深该方向：

1. 单层 replay 只改善极小，但整机 benchmark 无明显变化
2. kernel 复杂度显著上升，调参成本远高于预期收益
3. graph capture/replay 稳定性下降
4. 数值正确性验证变得脆弱，需要大量特殊分支修补

## 9. 当前建议结论

下一步建议直接开始做：

1. attention 输出布局直连 `o_proj`
2. `fused_silu_downproj`
3. `fused_add_rmsnorm_gateup`

而不是优先去碰：

1. attention 数学本体继续深挖
2. qkv 单 kernel 继续打磨
3. 更深的 `qkv + norm + rope` 全链融合

原因很简单：

- 第一项实现风险最低，更适合先建立正收益
- 第二项仍可做，但必须先量化 handoff gap，避免高估收益
- 第三项价值依然存在，但实现复杂度最高，应放到最后
