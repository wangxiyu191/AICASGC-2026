# 优化尝试记录与踩坑总结

> 记录于 2026-03-27，避免后续重复踩坑

## 环境信息

- 开发环境 (旧): NVIDIA A10 24GB (sm_86), Python 3.10.9, PyTorch 2.10.0+cu128, transformers 5.2.0
- 开发环境 (当前): NVIDIA A800 80GB PCIe (sm_80), Python 3.12.3, PyTorch 2.8.0+cu128, transformers 5.4.0, triton 3.4.0
- 评测环境: NVIDIA A800 80GB (sm_80), Python 3.12.3, PyTorch 2.8.0+cu128
- 模型: Qwen3-VL-2B-Instruct

---

## Phase 1: 配置级优化

### 1a: flash_attention_2 + CUDA config

```python
attn_implementation="flash_attention_2"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
```

**结果:** TTFT 128ms, Throughput 29.8 tok/s (baseline 133ms/37.2)

**问题:** Throughput 下降 20%，A10 上 flash_attn_varlen 效率不如 sdpa

### 1b: device_map vs .to(device)

**结论:** 单卡场景下 device_map 和 .to(device) 性能几乎无差异

### 1c: sdpa + CUDA config (当前最优)

```python
attn_implementation="sdpa"
```

**结果:** TTFT 121ms (-9%), Throughput 37.6 tok/s (持平)

---

## Phase 2: torch.compile

### 尝试

```python
self._model.model.language_model = torch.compile(
    self._model.model.language_model,
    mode="default",
    dynamic=True,
    fullgraph=False,
)
```

### 问题

1. **不能编译 vision encoder** - flash_attn_varlen_func 与 torch.compile 不兼容
   ```
   TorchRuntimeError: flash_attn._flash_attn_varlen_forward() 
   Expected a value of type 'int' for argument 'max_seqlen_q' 
   but instead found type 'FakeTensor'
   ```

2. **KV Cache 动态初始化导致频繁重编译**
   ```
   torch._dynamo hit config.recompile_limit (8)
   reason: past_key_values.layers[7].is_initialized == False
   ```

### 结果

TTFT 122ms (+0.5%), Throughput 24.5 tok/s (-34.8%)

**结论:** 动态 KV Cache 导致 recompile，性能反而下降

---

## Phase 3: 自定义 Decode Loop

### 尝试思路

根据 trace 分析，CPU dispatch 开销是 GPU 计算时间的 5.7 倍，主要瓶颈:
- `item()` 同步: 34.4% CPU 时间
- FlashAttn dispatch: 17.4%
- Linear/MatMul dispatch: 12.4%

尝试实现自定义 generate loop:
1. 避免使用 `.item()` 同步
2. 使用 tensor 操作检测 EOS
3. 预分配输出 tensor

### 实现代码

```python
def _generate_optimized(self, *args, **kwargs):
    # ... 省略输入处理 ...
    
    with torch.no_grad():
        outputs = self._model(**model_inputs)
    
    next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    
    # 使用 tensor 操作检测 EOS，避免 .item() 同步
    done_mask = (next_tokens == eos_token_id).squeeze(-1)
    
    for step in range(1, max_new_tokens):
        if done_mask.all():
            break
        # ...
```

### 问题

1. **模型内部 forward 仍有同步点** - `item()` 和 `is_nonzero` 在模型内部被调用
2. **缺少必要的参数传递** - VLM 需要 pixel_values, image_grid_thw 等参数
3. **性能反而下降** - TTFT 151ms vs 127ms

### 结论

自定义 decode loop 无法解决模型内部的同步问题，需要直接优化模型 forward

---

## StaticCache 尝试

### 问题 [redacted]`python
past_key_values = StaticCache(
    config=text_config,
    max_cache_len=max_cache_len,
    batch_size=batch_size,
    device=device,
    dtype=self._model.dtype,
)
```

1. **需要正确的 config** - 使用 `config.get_text_config(decoder=True)`，而非 `config`
2. **cache_position 计算复杂** - 需要正确传递给模型
3. **与动态输入不兼容** - VLM 输入长度不固定

---

## Trace 分析关键发现

### TTFT (Prefill) 阶段

| 类别 | 时间(ms) | 占比 |
|------|---------|------|
| GEMM | 59.7 | 50.4% |
| Elementwise | 32.8 | 27.7% |
| FlashAttn | 15.2 | 12.8% |
| **CPU item() sync** | 91.4 | 34.4% |
| **CPU FlashAttn dispatch** | 46.3 | 17.4% |

### Decode 阶段 (每步)

| 类别 | 时间(ms) | 占比 |
|------|---------|------|
| GEMM decode | 5.3 | 42.3% |
| Elementwise | 2.9 | 23.0% |
| **CPU Linear dispatch** | 20.7 | 29.1% |
| **CPU RMSNorm (pow+mean+rsqrt)** | 13.6 | 19.1% |
| **CPU Type conv/copy** | 12.1 | 17.0% |

### CPU/GPU 时间比

- TTFT: 266ms / 118ms = **2.2x**
- Decode: 71ms / 12.5ms = **5.7x**

**严重 CPU-bound，GPU 利用率极低**

---

## P0: 强制 SDPA (修复吞吐量退化)

### 问题

A800 官方评测显示 throughput 仅 33.07 tok/s（基线 ~55），存在 ~40% 退化。

### 根因

`flash_attention_2` 在 decode 阶段有极高的 CPU dispatch 开销：
- FlashAttn CPU dispatch: 8.2ms/step vs GPU 计算仅 0.56ms/step
- SDPA 自动选择最优后端 (prefill 用 flash, decode 用 math/mem_efficient)

### 修复

```python
attn_impl = "sdpa"  # 不再条件判断 flash_attn 可用性
```

**A10 结果:** TTFT 124ms, Throughput 35 tok/s (vs flash_attn_2 的 29.8 tok/s)
**A800 结果:** TTFT 62.42ms, Throughput 41.93 tok/s, Accuracy 0.6933, Score 0.4470

---

## Phase 4: OptimizedDecoderLayer (融合线性投影)

### 思路

根据 Trace 分析，decode 阶段 CPU-bound 严重 (CPU/GPU = 5.7x)，主要瓶颈是 kernel launch 次数过多。
参考 SGLang/vLLM 的融合实现，编写自定义 `OptimizedDecoderLayer` 替换 HF 的 `Qwen3VLTextDecoderLayer`。

### 实现

两项融合（均使用 `F.linear`，无 Triton 开销）：

1. **Fused QKV projection**: 合并 q_proj/k_proj/v_proj 权重为 [4096, 2048]，1 次 GEMM 替代 3 次
2. **Fused gate_up projection**: 合并 gate_proj/up_proj 权重为 [12288, 2048]，1 次 GEMM 替代 2 次

每个 decode step 减少 84 次 kernel launch (28 层 × 3 次 QKV + 28 层 × gate/up 各减 1 次 = 84)。

### 踩坑：Triton 小 tensor 开销

最初实现包含 `flash_attn.ops.triton.layer_norm.rms_norm_fn` 和 `flash_attn.ops.activations.swiglu`，
但 **Triton 在 decode 小 tensor (batch=1, seq_len=1) 场景下反而更慢**：

| 算子 | Triton | Native PyTorch | 倍数 |
|------|--------|---------------|------|
| rms_norm_fn | 159.4 us | 58.6 us | 2.7x 慢 |
| fused add+rmsnorm | 193.3 us | — | 更慢 |
| swiglu | 14.1 us | silu*up 12.0 us | ~1.2x 慢 |

原因：Triton kernel launch 固定开销（~100us），在小 tensor 上占比过大。
112 次 RMSNorm/step × 100us 额外开销 = +11.2ms/step。

**结论：** RMSNorm/SwiGLU 保持 native PyTorch 实现，仅使用 fused linear projections。

### 踩坑：SDPA GQA 属性名

替换 decoder layer 后报错：
```
RuntimeError: The size of tensor a (16) must match the size of tensor b (8)
```

原因：SDPA 的 `sdpa_attention_forward` 检查 `module.num_key_value_groups` 属性决定是否做 repeat_kv。
最初命名为 `self.num_kv_groups` 导致属性缺失，SDPA 对 16 头 Q 和 8 头 KV 做了错误广播。

修复：`self.num_key_value_groups = attn.num_key_value_groups`

### 结果

**A10:** TTFT 126.35ms, Throughput 39.58 tok/s (+13% vs sdpa baseline 的 35 tok/s)

---

## 未尝试的方向

### 可能有效的优化

1. **CUDA Graph 捕获 decode 阶段** - 固定 input shape，消除全部 kernel launch 开销
2. **torch.compile + StaticCache** - 需解决动态 shape recompile 问题
3. **消除模型内部 item()/is_nonzero 同步点** - 改善 TTFT
4. **减少 dtype 转换开销** - RMSNorm float32 转换在 trace 中占比 17%

### 已验证无效/收益有限的方向

1. **仅优化 generate 循环** - 瓶颈在模型内部 forward
2. **torch.compile 无脑用** - 动态 KV Cache shape 导致 recompile
3. **StaticCache 简单套用** - 需要正确处理 cache_position
4. **Triton RMSNorm/SwiGLU** - decode 小 tensor 下 launch 开销反超计算收益
5. **flash_attention_2** - decode 阶段 CPU dispatch 开销导致吞吐量退化 ~40%

---

## 代码结构注意事项

### Qwen3-VL 模型结构

```
Qwen3VLForConditionalGeneration
├── model: Qwen3VLModel
│   ├── visual: Qwen3VLVisionModel  # 不编译 (flash_attn_varlen)
│   └── language_model: Qwen3VLTextModel  # 可尝试编译
└── lm_head: Linear
```

### 关键配置获取

```python
# EOS token ID 在 text_config 中
text_config = model.config.get_text_config(decoder=True)
eos_token_id = text_config.eos_token_id  # 151645

# 不是在 model.config.eos_token_id
```

---

## 性能数据汇总

| Step | 优化内容 | TTFT (ms) | Throughput (tok/s) | 环境 | 备注 |
|------|---------|-----------|-------------------|------|------|
| 0 | Baseline FP16 | 133.13 | 37.22 | A10 | |
| 1a | flash_attn_2 | 128.20 | 29.79 | A10 | throughput -20% |
| 1c | sdpa | 121.35 | 37.57 | A10 | TTFT -9% |
| 2 | 官方 A800 评测 (flash_attn_2) | 68.32 | 33.07 | A800 | Score 未知 |
| 3 | torch.compile | 121.99 | 24.51 | A10 | recompile 导致退化 |
| 5 | 自定义 decode loop | 151.11 | 25.29 | A10 | 模型内部 sync 瓶颈 |
| 6 | P0: force sdpa | 123.99 | 35.03 | A10 | 修复 flash_attn 退化 |
| 7 | A800 sdpa 验证 | 62.42 | 41.93 | A800 | Score 0.4470, Acc 0.6933 |
| 8 | **fused QKV+gate_up** | 126.35 | 39.58 | A10 | **throughput +13%** |

**当前最优:** P0 (sdpa) + OptimizedDecoderLayer (fused QKV + gate_up)
