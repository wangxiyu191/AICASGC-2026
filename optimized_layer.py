"""
Optimized Qwen3-VL decoder layer with fused linear projections.

Replaces HF's Qwen3VLTextDecoderLayer:
- Fused QKV projection: 3 GEMMs → 1 GEMM (saves 2 kernel launches × 28 layers per step)
- Fused gate+up projection: 2 GEMMs → 1 GEMM (saves 1 kernel launch × 28 layers per step)
- Triton fused RMSNorm: 8 kernels → 1 kernel per norm call (2.2x faster, ~33us saved/call)
- Triton fused RoPE: 5 kernels → 1 kernel per call (eliminates cat/neg/mul/add)
- CUDA Graph-only small-shape Triton fusion: residual+RMSNorm and SiLU*up
"""
from contextlib import contextmanager
import os
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
        self._prefer_small_decode_triton = False

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
        prefer_small_decode_triton = self._prefer_small_decode_triton
        enable_decode_nvtx = _should_enable_decode_nvtx(hidden_states, past_key_values)

        # 1. Input LayerNorm
        residual = hidden_states
        with _nvtx_range("decode.input_rmsnorm", enable_decode_nvtx):
            hidden_states = _rms_norm(hidden_states, self.input_layernorm_weight, self.rms_norm_eps)

        # 2. Fused QKV (1 GEMM instead of 3)
        with _nvtx_range("decode.qkv_proj", enable_decode_nvtx):
            qkv = F.linear(hidden_states, self.qkv_weight)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 3. QK-Norm + RoPE (before transpose — natural layout for cos/sin broadcast)
        cos, sin = position_embeddings
        with _nvtx_range("decode.qk_norm_rope", enable_decode_nvtx):
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
            with _nvtx_range("decode.kv_cache_update", enable_decode_nvtx):
                k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # 5. Attention (SDPA)
        with _nvtx_range("decode.attention", enable_decode_nvtx):
            attn_output, _ = self._attention_interface(
                self, q, k, v, attention_mask,
                dropout=0.0, scaling=self.scaling, **kwargs,
            )
        with _nvtx_range("decode.o_proj", enable_decode_nvtx):
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)

        # 6. Residual + Post-attention LayerNorm
        with _nvtx_range("decode.post_attn_add_rmsnorm", enable_decode_nvtx):
            residual, hidden_states = _fused_add_rms_norm(
                residual,
                attn_output,
                self.post_attn_layernorm_weight,
                self.rms_norm_eps,
                prefer_small_triton=prefer_small_decode_triton,
            )

        # 7. Fused gate+up (1 GEMM instead of 2) + SiLU activation
        with _nvtx_range("decode.gate_up_silu", enable_decode_nvtx):
            gate_up = F.linear(hidden_states, self.gate_up_weight)
            gate, up = gate_up.chunk(2, dim=-1)
            hidden_states = _fused_silu_mul(
                gate,
                up,
                prefer_small_triton=prefer_small_decode_triton,
            )
        with _nvtx_range("decode.down_proj", enable_decode_nvtx):
            hidden_states = self.down_proj(hidden_states)

        # 8. Residual
        with _nvtx_range("decode.residual_add", enable_decode_nvtx):
            hidden_states = residual + hidden_states
        return hidden_states


_TRITON_ROWS_THRESHOLD = 32
_ENABLE_TRITON_SMALL_ADD_RMSNORM = os.environ.get("ENABLE_TRITON_SMALL_ADD_RMSNORM", "1") == "1"
_ENABLE_TRITON_SMALL_SILU_MUL = os.environ.get("ENABLE_TRITON_SMALL_SILU_MUL", "0") == "1"
_ENABLE_DECODE_NVTX = os.environ.get("ENABLE_DECODE_NVTX", "0") == "1"


@contextmanager
def _nvtx_range(name: str, enabled: bool):
    if enabled:
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
        return
    yield


def _should_enable_decode_nvtx(hidden_states: torch.Tensor, past_key_values) -> bool:
    return (
        _ENABLE_DECODE_NVTX
        and hidden_states.is_cuda
        and hidden_states.dim() >= 2
        and hidden_states.shape[-2] == 1
        and past_key_values is not None
    )


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm: Triton for prefill (large M), F.rms_norm for decode (small M)."""
    N = x.shape[-1]
    M = x.numel() // N
    if M <= _TRITON_ROWS_THRESHOLD:
        return F.rms_norm(x, (N,), weight, eps)
    orig_shape = x.shape
    x_2d = x.contiguous().reshape(M, N)
    out_2d = torch.empty_like(x_2d)
    BLOCK_N = triton.next_power_of_2(N)
    _rms_norm_fwd_kernel[(M,)](
        x_2d, out_2d, weight,
        x_2d.stride(0), N, eps,
        BLOCK_N=BLOCK_N,
    )
    return out_2d.view(orig_shape)


def _fused_add_rms_norm(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    prefer_small_triton: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (residual_plus_x, rms_norm(residual_plus_x)) with one Triton kernel when useful."""
    N = residual.shape[-1]
    if (
        (not prefer_small_triton)
        or (not _ENABLE_TRITON_SMALL_ADD_RMSNORM)
        or residual.device.type != "cuda"
        or x.device.type != "cuda"
        or weight.device.type != "cuda"
        or residual.stride(-1) != 1
        or x.stride(-1) != 1
        or weight.stride(-1) != 1
    ):
        summed = residual + x
        normed = F.rms_norm(summed, (N,), weight, eps)
        return summed, normed

    M = residual.numel() // N
    residual_2d = residual.view(M, N)
    x_2d = x.view(M, N)
    summed_2d = torch.empty_like(residual_2d)
    normed_2d = torch.empty_like(residual_2d)
    BLOCK_N = triton.next_power_of_2(N)
    _fused_add_rms_norm_fwd_kernel[(M,)](
        residual_2d,
        x_2d,
        summed_2d,
        normed_2d,
        weight,
        residual_2d.stride(0),
        x_2d.stride(0),
        summed_2d.stride(0),
        normed_2d.stride(0),
        N,
        eps,
        BLOCK_N=BLOCK_N,
    )
    return summed_2d.view_as(residual), normed_2d.view_as(residual)


def _fused_silu_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
    prefer_small_triton: bool = False,
) -> torch.Tensor:
    """Fuse SiLU(gate) * up into one kernel when useful."""
    if (
        (not prefer_small_triton)
        or (not _ENABLE_TRITON_SMALL_SILU_MUL)
        or gate.device.type != "cuda"
        or up.device.type != "cuda"
        or gate.stride(-1) != 1
        or up.stride(-1) != 1
    ):
        return F.silu(gate) * up

    gate_flat = gate.reshape(-1)
    up_flat = up.reshape(-1)
    out_flat = torch.empty_like(gate_flat)
    total = gate_flat.numel()
    BLOCK = 1024
    _fused_silu_mul_fwd_kernel[(triton.cdiv(total, BLOCK),)](
        gate_flat,
        up_flat,
        out_flat,
        total,
        BLOCK=BLOCK,
    )
    return out_flat.view_as(gate)


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


@triton.jit
def _fused_add_rms_norm_fwd_kernel(
    RESIDUAL,
    X,
    OUT_SUM,
    OUT_NORM,
    W,
    residual_stride,
    x_stride,
    out_sum_stride,
    out_norm_stride,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
):
    """Compute residual+x and its RMSNorm in one pass."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    residual = tl.load(RESIDUAL + row * residual_stride + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
    summed = residual + x
    var = tl.sum(summed * summed, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    tl.store(OUT_SUM + row * out_sum_stride + cols, summed.to(OUT_SUM.dtype.element_ty), mask=mask)
    tl.store(OUT_NORM + row * out_norm_stride + cols, (summed * rstd * w).to(OUT_NORM.dtype.element_ty), mask=mask)


@triton.jit
def _fused_silu_mul_fwd_kernel(
    GATE,
    UP,
    OUT,
    total_elements,
    BLOCK: tl.constexpr,
):
    """Compute SiLU(gate) * up."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements
    gate = tl.load(GATE + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(UP + offsets, mask=mask, other=0.0).to(tl.float32)
    silu = gate / (1.0 + tl.exp(-gate))
    tl.store(OUT + offsets, (silu * up).to(OUT.dtype.element_ty), mask=mask)


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE: Triton for prefill (large seq), native PyTorch for decode (seq=1)."""
    batch, seq_len, nheads, D = x.shape
    HALF_D = D // 2

    if seq_len <= _TRITON_ROWS_THRESHOLD:
        x1 = x[..., :HALF_D]
        x2 = x[..., HALF_D:]
        cos_half = cos[..., :HALF_D].unsqueeze(2)
        sin_half = sin[..., :HALF_D].unsqueeze(2)
        return torch.cat([x1 * cos_half - x2 * sin_half,
                          x2 * cos_half + x1 * sin_half], dim=-1)

    M = batch * seq_len
    total = M * nheads

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


def set_small_decode_triton_enabled(model, enabled: bool) -> bool:
    """Toggle decode-only small-shape Triton fusion for optimized layers."""
    layers = model.model.language_model.layers
    previous = False
    for layer in layers:
        if isinstance(layer, OptimizedDecoderLayer):
            previous = layer._prefer_small_decode_triton
            layer._prefer_small_decode_triton = enabled
    return previous
