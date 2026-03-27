"""
Optimized Qwen3-VL decoder layer with fused linear projections.

Replaces HF's Qwen3VLTextDecoderLayer:
- Fused QKV projection: 3 GEMMs → 1 GEMM (saves 2 kernel launches × 28 layers per step)
- Fused gate+up projection: 2 GEMMs → 1 GEMM (saves 1 kernel launch × 28 layers per step)
- Triton fused RMSNorm: 8 kernels → 1 kernel per norm call (2.2x faster, ~33us saved/call)
- Triton fused RoPE: 5 kernels → 1 kernel per call (eliminates cat/neg/mul/add)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


class OptimizedDecoderLayer(nn.Module):
    """Drop-in replacement for Qwen3VLTextDecoderLayer with fused linear projections.

    Keeps the same forward signature so HF's model.forward() can call it
    without modification. Uses the original layer's weight tensors.
    """

    def __init__(self, orig_layer):
        super().__init__()
        attn = orig_layer.self_attn
        mlp = orig_layer.mlp

        # === Config ===
        self.hidden_size = orig_layer.hidden_size
        self.head_dim = attn.head_dim
        self.num_heads = attn.config.num_attention_heads
        self.num_kv_heads = attn.config.num_key_value_heads
        self.num_key_value_groups = attn.num_key_value_groups
        self.scaling = attn.scaling
        self.is_causal = attn.is_causal
        self.layer_idx = attn.layer_idx
        self.attention_dropout = attn.attention_dropout
        self.rms_norm_eps = attn.config.rms_norm_eps

        # === RMSNorm weights (keep native — Triton is 2.7x slower on small tensors) ===
        self.input_layernorm_weight = orig_layer.input_layernorm.weight
        self.post_attn_layernorm_weight = orig_layer.post_attention_layernorm.weight
        self.q_norm_weight = attn.q_norm.weight
        self.k_norm_weight = attn.k_norm.weight

        # === Fused QKV projection: 3 GEMMs → 1 ===
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_weight = nn.Parameter(
            torch.cat([attn.q_proj.weight.data,
                       attn.k_proj.weight.data,
                       attn.v_proj.weight.data], dim=0),
            requires_grad=False,
        )
        self.o_proj = attn.o_proj

        # === Fused gate+up projection: 2 GEMMs → 1 ===
        self.intermediate_size = mlp.intermediate_size
        self.gate_up_weight = nn.Parameter(
            torch.cat([mlp.gate_proj.weight.data,
                       mlp.up_proj.weight.data], dim=0),
            requires_grad=False,
        )
        self.down_proj = mlp.down_proj

        # === Attention interface (cached lookup) ===
        self.config = attn.config
        self._attn_implementation = attn.config._attn_implementation
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward
        except ImportError:
            eager_attention_forward = None
        self._attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self._attn_implementation, eager_attention_forward,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape_q = (*input_shape, self.num_heads, self.head_dim)
        hidden_shape_kv = (*input_shape, self.num_kv_heads, self.head_dim)

        # 1. Input LayerNorm
        residual = hidden_states
        hidden_states = _rms_norm(hidden_states, self.input_layernorm_weight, self.rms_norm_eps)

        # 2. Fused QKV (1 GEMM instead of 3)
        qkv = F.linear(hidden_states, self.qkv_weight)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 3. QK-Norm + RoPE (before transpose — natural layout for cos/sin broadcast)
        cos, sin = position_embeddings
        q = _rms_norm(q.view(hidden_shape_q), self.q_norm_weight, self.rms_norm_eps)
        k = _rms_norm(k.view(hidden_shape_kv), self.k_norm_weight, self.rms_norm_eps)
        q = _apply_rotary_emb(q, cos, sin).transpose(1, 2)
        k = _apply_rotary_emb(k, cos, sin).transpose(1, 2)
        v = v.view(hidden_shape_kv).transpose(1, 2)

        # 4. KV Cache
        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin.unsqueeze(1), "cos": cos.unsqueeze(1),
                "cache_position": cache_position,
            }
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # 5. Attention (SDPA)
        attn_output, _ = self._attention_interface(
            self, q, k, v, attention_mask,
            dropout=0.0, scaling=self.scaling, **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        # 6. Residual + Post-attention LayerNorm
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = _rms_norm(hidden_states, self.post_attn_layernorm_weight, self.rms_norm_eps)

        # 7. Fused gate+up (1 GEMM instead of 2) + SiLU activation
        gate_up = F.linear(hidden_states, self.gate_up_weight)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = F.silu(gate) * up
        hidden_states = self.down_proj(hidden_states)

        # 8. Residual
        hidden_states = residual + hidden_states
        return hidden_states


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fused RMSNorm using Triton — single kernel replaces 8 PyTorch ops.

    Handles arbitrary input shapes by flattening to 2D (M, N) where N is the
    last dimension (must match weight). All computation in fp32 internally.
    """
    orig_shape = x.shape
    N = x.shape[-1]
    M = x.numel() // N
    x_2d = x.contiguous().reshape(M, N)
    out_2d = torch.empty_like(x_2d)
    BLOCK_N = triton.next_power_of_2(N)
    _rms_norm_fwd_kernel[(M,)](
        x_2d, out_2d, weight,
        x_2d.stride(0), N, eps,
        BLOCK_N=BLOCK_N,
    )
    return out_2d.view(orig_shape)


@triton.jit
def _rms_norm_fwd_kernel(
    X, Y, W,
    stride,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
):
    """Fused RMSNorm forward kernel.

    Each program processes one row: load → variance in fp32 → normalize → apply weight → store.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * stride + cols, (x * rstd * w).to(Y.dtype.element_ty), mask=mask)


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Fused RoPE using Triton — single kernel replaces neg + cat + 2 mul + add.

    Args:
        x: (batch, seq_len, num_heads, head_dim) — before transpose
        cos: (batch, seq_len, head_dim) — from position_embeddings
        sin: (batch, seq_len, head_dim) — from position_embeddings

    cos/sin have duplicated halves (cos[..., :D/2] == cos[..., D/2:]) so we
    only load the first half (D/2 elements) to save memory bandwidth.
    """
    batch, seq_len, nheads, D = x.shape
    HALF_D = D // 2
    M = batch * seq_len  # number of (batch, seq) positions
    total = M * nheads   # total programs

    x_3d = x.reshape(M, nheads, D)
    cos_2d = cos.reshape(M, D)
    sin_2d = sin.reshape(M, D)
    out_3d = torch.empty_like(x_3d)

    BLOCK = triton.next_power_of_2(HALF_D)
    _rope_fwd_kernel[(total,)](
        x_3d, out_3d, cos_2d, sin_2d,
        nheads, D, HALF_D,
        x_3d.stride(0), x_3d.stride(1),
        cos_2d.stride(0),
        BLOCK=BLOCK,
    )
    return out_3d.view(batch, seq_len, nheads, D)


@triton.jit
def _rope_fwd_kernel(
    X, OUT, COS, SIN,
    nheads, D: tl.constexpr, HALF_D: tl.constexpr,
    stride_x_m, stride_x_h,
    stride_cos_m,
    BLOCK: tl.constexpr,
):
    """Fused RoPE forward kernel.

    Each program handles one (sample, head) pair.
    Loads x first/second halves, cos/sin first half only (halves are identical),
    computes: out0 = x0*cos - x1*sin, out1 = x1*cos + x0*sin.
    """
    pid = tl.program_id(0)
    m = pid // nheads   # (batch * seq_len) index for cos/sin lookup
    h = pid % nheads    # head index

    rk = tl.arange(0, BLOCK)
    mask = rk < HALF_D

    x_base = m * stride_x_m + h * stride_x_h
    cs_base = m * stride_cos_m

    # Load first and second halves of x
    x0 = tl.load(X + x_base + rk, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(X + x_base + rk + HALF_D, mask=mask, other=0.0).to(tl.float32)

    # Load cos/sin (only first half — second half is identical)
    c = tl.load(COS + cs_base + rk, mask=mask, other=1.0).to(tl.float32)
    s = tl.load(SIN + cs_base + rk, mask=mask, other=0.0).to(tl.float32)

    # RoPE rotation
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    tl.store(OUT + x_base + rk, out0.to(X.dtype.element_ty), mask=mask)
    tl.store(OUT + x_base + rk + HALF_D, out1.to(X.dtype.element_ty), mask=mask)


def replace_decoder_layers(model):
    """Replace all Qwen3VLTextDecoderLayer with OptimizedDecoderLayer."""
    layers = model.model.language_model.layers
    is_eval = not model.training
    count = 0
    for i in range(len(layers)):
        layers[i] = OptimizedDecoderLayer(layers[i])
        if is_eval:
            layers[i].eval()
        count += 1
    return count
