# Decode 算力与访存预算

> 分析日期: 2026-03-28
> 模型: Qwen3-VL-2B-Instruct
> 目标: 估算 decode 阶段每 token 的 FLOPs、访存需求、理论上限，并与当前 trace/benchmark 对照

---

## 1. 结构参数

从模型配置读取到的 text decoder 关键参数：

| 参数 | 值 |
|------|----|
| hidden_size | 2048 |
| intermediate_size | 6144 |
| num_hidden_layers | 28 |
| num_attention_heads | 16 |
| num_key_value_heads | 8 |
| head_dim | 128 |
| vocab_size | 151936 |

分析假设：

- decode 为 `batch=1, seq=1`
- 权重 dtype 为 FP16（2 bytes）
- 线性层 FLOPs 统一按 `2 * in_dim * out_dim`（mul+add）估算
- attention cache 长度记为 `L`

---

## 2. 每层 decode 的主算子预算

### 2.1 线性层

| 算子 | 形状 | 参数量 | 权重读 | FLOPs / token |
|------|------|--------|--------|---------------|
| fused QKV | `2048 -> 4096` | 8,388,608 | 16 MB | 16.78 MF |
| o_proj | `2048 -> 2048` | 4,194,304 | 8 MB | 8.39 MF |
| fused gate_up | `2048 -> 12288` | 25,165,824 | 48 MB | 50.33 MF |
| down_proj | `6144 -> 2048` | 12,582,912 | 24 MB | 25.17 MF |

每层合计：

- 权重读：`96 MB`
- FLOPs：`100.66 MF`

28 层合计：

- 权重读：`2.625 GiB`
- FLOPs：`2.819 GF`

### 2.2 Attention

单 token decode 的 attention 计算主要是：

1. `q @ k^T`
2. softmax
3. `attn @ v`

忽略 softmax 常数项后，每层主 FLOPs 约为：

```text
4 * num_heads * head_dim * L
= 4 * 16 * 128 * L
= 8192 * L
```

举例：

| cache 长度 L | 每层 attention FLOPs | 28 层 attention FLOPs |
|--------------|----------------------|------------------------|
| 512 | 4.19 MF | 117.44 MF |
| 768 | 6.29 MF | 176.16 MF |
| 1024 | 8.39 MF | 234.88 MF |

attention 访存方面，最关键的是 KV cache 读取。每层最小读取量（K+V）为：

```text
2 * num_kv_heads * head_dim * L * 2 bytes * 2
= 4096 * L bytes
```

举例：

| cache 长度 L | 每层 KV 读 | 28 层 KV 读 |
|--------------|------------|-------------|
| 512 | 2 MB | 109 MB |
| 768 | 3 MB | 164 MB |
| 1024 | 4 MB | 219 MB |

可以看到：**attention FLOPs 和 KV bytes 都随 `L` 线性增长，但在当前长度区间里，仍然明显小于全层权重读取。**

### 2.3 lm_head

decode 每 token 还要走一次 full vocab `lm_head`：

| 算子 | 形状 | 参数量 | 权重读 | FLOPs / token |
|------|------|--------|--------|---------------|
| lm_head | `2048 -> 151936` | 311,164,928 | 0.580 GiB | 622.33 MF |

这一步本身就是一个巨大的单 token GEMV。

---

## 3. 每 token 总预算

### 3.1 只看线性层权重

- 28 层线性层：`2.625 GiB`
- `lm_head`：`0.580 GiB`

总计：

- **权重读约 `3.205 GiB / token`**
- **FLOPs 约 `3.441 GF / token`**  
  这里包含 28 层线性 + lm_head，不含 attention 与小算子

### 3.2 加上 attention 的典型长度

以 `L = 768` 为例：

- 权重读：`3.205 GiB`
- KV cache 读：`0.164 GiB`
- 合计：`3.369 GiB / token`

FLOPs：

- 线性 + lm_head：`3.441 GF`
- attention：`0.176 GF`
- 合计：约 **`3.617 GF / token`**

这说明：

- 从 **算力** 看，decode 每 token 只有几 GF，远低于 A800 的 FP16 Tensor Core 峰值，完全不是算力上限问题
- 从 **访存** 看，每 token 要读 3GB 量级的数据，明显是 HBM 带宽主导

---

## 4. 理论上限估算

按 A800 HBM 带宽约 `2039 GB/s`，仅从权重+KV 读取上限估算：

| cache 长度 L | 每 token bytes | 带宽上限 |
|--------------|----------------|----------|
| 512 | 3.314 GiB | 573 tok/s |
| 768 | 3.369 GiB | 564 tok/s |
| 1024 | 3.423 GiB | 555 tok/s |

即使按 `L=768`，纯带宽理论上限也还有 **~560 tok/s**。

如果换成时间尺度看，这个 token 级下限大致是：

```text
1 / 564 tok/s ≈ 1.77 ms / token
```

这和“按权重读 + KV 读”直接估出来的 roofline 下限是一致的。

### 4.1 按主算子分解的 A800 理想平均耗时

下面把最新 steady-state decode trace 的主要 stage，和 A800 上按

```text
ideal_time = max(FLOPs / 312 TFLOP/s, bytes / 2039 GB/s)
```

算出的理想平均耗时并排放在一起。

说明：

- 线性层按权重读取主导估算
- attention 按 `L=768`、只取 K/V 读取这个主项估算
- 小算子（RMSNorm / RoPE / SiLU）按 activation bytes 粗估，它们的“倍率”很容易很夸张，但绝对值通常很小
- 实测列来自最新 trace [traces/compare/triton_decode_attention_dot](../traces/compare/triton_decode_attention_dot) 的 steady-state decode 分解

| 算子 | 理想带宽时间 | 理想算力时间 | A800 理想下限 | 最新实测 | 实测/理想 |
|------|--------------|--------------|---------------|----------|-----------|
| qkv_proj / layer | 8.23 us | 0.05 us | **8.23 us** | 14.7 us | 1.8x |
| gate_up_proj / layer | 24.68 us | 0.16 us | **24.68 us** | 34.9 us | 1.4x |
| o_proj / layer | 4.11 us | 0.03 us | **4.11 us** | 9.4 us | 2.3x |
| down_proj / layer | 12.34 us | 0.08 us | **12.34 us** | 28.3 us | 2.3x |
| attention(KV) / layer, `L=768` | 1.54 us | 0.02 us | **1.54 us** | 23.9 us | 15.5x |
| input+qk_norm+rope / layer | 0.034 us | ~0 | **0.034 us** | 0.62 us | 18.2x |
| post_attn_add_rmsnorm / layer | 0.010 us | ~0 | **0.010 us** | 2.5 us | 253.6x |
| SiLU / layer | 0.012 us | ~0 | **0.012 us** | 2.5 us | 207.4x |
| lm_head / token | 305.21 us | 2.00 us | **305.21 us** | 356.7 us | 1.2x |

把这些主项加总后：

- **A800 理想下限约 `1.74 ms / token`**
- **最新 steady-state trace 实测约 `4.02 ms / token`**
- 也就是相对这些主项的 roofline 下限，**还有约 `2.3x` 空间**

这个 `1.74 ms` 和上面的 token 级总下限 `1.77 ms` 基本一致，说明这张分解表和整体预算是对齐的。

### 4.2 为什么 Attention 看起来还差很多

上表里的 Attention `1.54 us / layer` 是 **full-chip roofline**，也就是默认整张 A800 的 HBM 带宽都能被这一层 decode attention 吃满。

但这对当前 decode shape 明显过于乐观，因为当前主路径的 Attention kernel 并行度非常低：

- 模型是 `16 q heads / 8 kv heads`
- 当前 grouped-GQA decode attention 是 **1 个 `kv_head` 对应 1 个 CTA**
- 所以一层 Attention kernel 只有 **8 个 CTA**
- 这张 A800 80GB PCIe 实测有 **108 个 SM**

也就是说，在不做 split-KV / multi-CTA merge 的前提下，这个 kernel 同时最多只能占住：

```text
8 / 108 ≈ 7.4% 的 SM
```

如果粗略假设 HBM 可用带宽也按这个并行度同比缩放，那么当前 shape 下的 **并行度受限 roofline** 大致是：

| 指标 | 数值 |
|------|------|
| K/V bytes / layer (`L=768`) | `3,145,728 B` |
| full-chip HBM roofline | `1.54 us / layer` |
| shape-limited HBM ceiling | `2039 * 8 / 108 ≈ 151 GB/s` |
| shape-limited roofline | **`20.8 us / layer`** |
| step37 实测 | **`23.9 us / layer`** |

换句话说：

- 如果和 **full-chip roofline** 比，当前 Attention 看起来还是 `15.5x`
- 但如果和 **当前 shape 的并行度受限 roofline** 比，实际上只差：

```text
23.9 / 20.8 ≈ 1.15x
```

也就是每层只差大约 `3.1 us`，28 层合计约 `0.087 ms / token`。

这更符合直觉：我们已经把 Attention 从早期的 `~52 us / layer` 压到了现在的 `~24 us / layer`，再拿 full-chip roofline 去看，会夸大“还差很多”的感觉。

因此更准确的结论是：

- **当前 Attention 离 full-chip 理论值仍远**
- 但这主要是 **decode 小 shape 的并行度天花板**，不完全是 kernel 本身低效
- 在当前 `8 CTA / layer` 这个结构下，step37 的 Attention 其实已经比较接近可达上限

### 4.3 这个差距主要卡在哪

从上表看，最重要的不是“倍率最大”的那些小算子，而是**绝对时间差**最大的几项：

| 算子 | 实测 - 理想 |
|------|-------------|
| attention / 28 layers | **~0.63 ms / token** |
| down_proj / 28 layers | **~0.45 ms / token** |
| gate_up_proj / 28 layers | **~0.29 ms / token** |
| qkv_proj / 28 layers | **~0.19 ms / token** |
| o_proj / 28 layers | **~0.14 ms / token** |
| lm_head | **~0.06 ms / token** |

可以直接得到两个判断：

1. **attention 仍然是 full-chip roofline 下差距最大的主力项**
   - 但这里要区分两层含义：
   - 相对 full-chip roofline，当前 attention 的确还高约 `15.5x`
   - 相对当前 decode shape 的并行度受限 roofline，当前 attention 其实只高约 `1.15x`
   - 所以它后续不是“完全没优化好”，而是已经开始明显受制于并行度天花板

2. **大线性层已经相对接近带宽下限**
   - `qkv / gate_up / o_proj / down_proj / lm_head` 大多只比理想下限高 `1.2x ~ 2.4x`
   - 它们当然还有空间，但已经不像 attention 那样是一个数量级的差距

所以如果只看 full-chip 理论值，Attention 还是最该盯的一段；但如果把当前 decode 并行度也考虑进去，那么 **下一步 Attention 真正值得做的不是继续抠单 CTA 内部细节，而是想办法提高可并行度**。

当前已经验证的最新本地实验（decode attention dot-style fast path 开启）：

- `10` 样本 benchmark：`63.56 ms / 183.05 tok/s`
- latest steady-state trace：`4.35 ms / step`，约对应 `204 tok/s` 的 GPU-only 上限

因此当前有效带宽利用率大致为：

```text
183 / 564 ≈ 32.5%
```

如果只看 GPU steady-state 上限，则大致为：

```text
204 / 564 ≈ 36.2%
```

也就是说，当前和理论上限之间仍然还有约 `2.8x ~ 3.1x` 空间。

---

## 5. 为什么现在还是跑不快？

理论上限不是问题，现实问题在于 **HBM roundtrip 浪费**。

latest trace（`triton_decode_attention_dot`）显示：

| 类别 | decode GPU/步 | 占比 |
|------|---------------|------|
| GEMM | 1.75ms | 40.2% |
| GEMV | 1.06ms | 24.3% |
| Attention | 0.67ms | 15.4% |
| Elementwise | 0.57ms | 13.2% |
| Norm | 0.28ms | 6.4% |

主热点不是“大矩阵乘本身算不动”，而是：

- attention
- 单 token GEMV / 小 GEMM
- 仍然残留的 `Copy/Mul/generic elementwise`

这些小 kernel 把本来应该接近“读权重一次就做完”的过程，拆成了大量：

```text
HBM 读 -> 小计算 -> HBM 写 -> 再读 -> 再算 -> 再写
```

因此虽然理论上是带宽受限，但按 full-chip 口径看，当前 steady-state 也只吃到了约 `36%` 的有效带宽。

---

## 6. 对后续优化的启示

### 已经证明有效

- `small add+rmsnorm`
  它能同时减少 `Copy + Mul + Reduction`，所以有约 `+6%` 的真实吞吐收益

### 收益小但安全

- `small silu*up`
  独立 kernel 数值上是 bit-identical，100 样本输出也 100/100 一致，但因为 `SiLU` 自身占比小，收益只有 `~+0.8%`

### 最值得继续做

1. **如果继续做 Attention，优先级应转向“提高并行度”**
   比如更低开销的 split-KV / multi-CTA merge，或者更接近 FlashInfer 的 decode/prefill-style 组织，而不是只在单 CTA 内再抠一点点标量优化

2. **同时应该开始重视 decode GEMV**
   最新 trace 下 `o_proj + down_proj` 已经有 `~1.06 ms / step`，绝对时间比 Attention 还大；其中 `down_proj` 的理论差距也已经接近 Attention

3. **继续打 elementwise / reduction**
   重点仍是 `Copy/Mul/Reduction`，不是继续围着 `SiLU` 打转
   这条路径还会制造大量 `Mul/Copy`

4. **检查 `lm_head` 的单 token GEMV 上限**
   它单独就有 `0.58 GiB` 权重读，是很大的固定成本

5. **attention decode kernel 专项优化**
   但目标应从“再抠单 CTA 内部细节”切到“提高并行度”；否则很容易已经接近当前 shape 的局部上限

---

## 7. 当前判断

从理论预算和 trace 同时看，当前 decode 已经不是“CPU dispatch 问题”，而是：

- **架构上属于 HBM 带宽受限**
- **实现上同时受到小 shape Attention 并行度不足、decode GEMV 成本、以及图内 elementwise/reduction 过碎的限制**

因此下一阶段最合理的方向不是再做通用编译器赌博，而是继续做 **decode 图内的 targeted optimization**：

- Attention 侧优先想办法提高并行度
- GEMV 侧优先看 `o_proj/down_proj`
- elementwise 侧继续压 `Copy/Mul/Reduction`
