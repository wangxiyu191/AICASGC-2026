# 优化尝试记录与踩坑总结

> 记录于 2026-03-27，避免后续重复踩坑

## 环境信息

- 开发环境: NVIDIA A10 24GB (sm_86), Python 3.10.9, PyTorch 2.10.0+cu128
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

## 未尝试的方向

### 可能有效的优化

1. **融合 RMSNorm kernel** - 将 pow+mean+rsqrt 合并为单个 Triton kernel
2. **优化 Attention mask 处理** - 减少不必要的 tensor 操作
3. **CUDA Graph 捕获 decode 阶段** - 但需要固定输入 shape
4. **替换模型内部算子** - Monkey patch 模型的 attention/RMSNorm

### 可能无效的方向

1. **仅优化 generate 循环** - 瓶颈在模型内部
2. **torch.compile 无脑用** - 动态 shape 导致 recompile
3. **StaticCache 简单套用** - 需要正确处理 cache_position

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

| Step | 优化内容 | TTFT (ms) | Throughput (tok/s) | 环境 |
|------|---------|-----------|-------------------|------|
| 0 | Baseline FP16 | 133.13 | 37.22 | A10 |
| 1a | flash_attn_2 | 128.20 | 29.79 | A10 |
| 1c | sdpa (最优) | 121.35 | 37.57 | A10 |
| 2 | 官方 A800 评测 | 68.32 | 33.07 | A800 |
| 2 | torch.compile | 121.99 | 24.51 | A10 |
| 3 | 自定义 decode loop | 151.11 | 25.29 | A10 |

**结论:** 当前最优配置仍是 Phase 1c (sdpa + CUDA config)

---

## 后续建议

1. **直接优化模型 forward 内部算子** - 而非 generate 循环
2. **实现 fused RMSNorm Triton kernel** - 减少 dispatch 次数
3. **研究消除模型内部的 sync 点** - 需要深入 transformers 源码
4. **在 A800 环境测试** - A10 和 A800 性能特性可能不同
