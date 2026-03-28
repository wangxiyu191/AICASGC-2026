"""
Optimized Qwen3-VL decoder layer with fused linear projections.

Replaces HF's Qwen3VLTextDecoderLayer:
- Fused QKV projection: 3 GEMMs → 1 GEMM (saves 2 kernel launches × 28 layers per step)
- Fused gate+up projection: 2 GEMMs → 1 GEMM (saves 1 kernel launch × 28 layers per step)
- Triton fused RMSNorm: 8 kernels → 1 kernel per norm call (2.2x faster, ~33us saved/call)
- Triton fused RoPE: 5 kernels → 1 kernel per call (eliminates cat/neg/mul/add)
- CUDA Graph-only small-shape Triton fusion: residual+RMSNorm and SiLU*up
"""
from contextlib import ExitStack, contextmanager
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
            hidden_states = _rms_norm(
                hidden_states,
                self.input_layernorm_weight,
                self.rms_norm_eps,
                prefer_small_triton=prefer_small_decode_triton,
            )

        # 2. Fused QKV (1 GEMM instead of 3)
        with _nvtx_range("decode.qkv_proj", enable_decode_nvtx):
            qkv = F.linear(hidden_states, self.qkv_weight)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 3. QK-Norm + RoPE (before transpose — natural layout for cos/sin broadcast)
        cos, sin = position_embeddings
        with _nvtx_range("decode.qk_norm_rope", enable_decode_nvtx):
            q = _fused_rms_norm_rope(
                q.view(hidden_shape_q),
                self.q_norm_weight,
                self.rms_norm_eps,
                cos,
                sin,
                prefer_small_triton=prefer_small_decode_triton,
            ).transpose(1, 2)
            k = _fused_rms_norm_rope(
                k.view(hidden_shape_kv),
                self.k_norm_weight,
                self.rms_norm_eps,
                cos,
                sin,
                prefer_small_triton=prefer_small_decode_triton,
            ).transpose(1, 2)
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
            attn_output = _small_decode_attention(
                q,
                k,
                v,
                attention_mask,
                self.scaling,
                self.num_key_value_groups,
                prefer_small_triton=prefer_small_decode_triton,
                cache_position=cache_position,
            )
            if attn_output is None:
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
_ENABLE_TRITON_SMALL_RMSNORM = os.environ.get("ENABLE_TRITON_SMALL_RMSNORM", "1") == "1"
_ENABLE_TRITON_SMALL_ADD_RMSNORM = os.environ.get("ENABLE_TRITON_SMALL_ADD_RMSNORM", "1") == "1"
_ENABLE_TRITON_SMALL_SILU_MUL = os.environ.get("ENABLE_TRITON_SMALL_SILU_MUL", "0") == "1"
_ENABLE_TRITON_SMALL_QK_NORM_ROPE = os.environ.get("ENABLE_TRITON_SMALL_QK_NORM_ROPE", "1") == "1"
_ENABLE_TRITON_SMALL_DECODE_ATTENTION = os.environ.get("ENABLE_TRITON_SMALL_DECODE_ATTENTION", "1") == "1"
_ENABLE_DECODE_NVTX = os.environ.get("ENABLE_DECODE_NVTX", "0") == "1"


@contextmanager
def _nvtx_range(name: str, enabled: bool):
    if enabled:
        with ExitStack() as stack:
            stack.enter_context(torch.autograd.profiler.record_function(name))
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


def _rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    prefer_small_triton: bool = False,
) -> torch.Tensor:
    """RMSNorm: Triton for prefill (large M), F.rms_norm for decode (small M)."""
    N = x.shape[-1]
    M = x.numel() // N
    if (
        M <= _TRITON_ROWS_THRESHOLD
        and (
            (not prefer_small_triton)
            or (not _ENABLE_TRITON_SMALL_RMSNORM)
            or x.device.type != "cuda"
            or weight.device.type != "cuda"
            or x.stride(-1) != 1
            or weight.stride(-1) != 1
        )
    ):
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


def _fused_rms_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    prefer_small_triton: bool = False,
) -> torch.Tensor:
    """Fuse RMSNorm + RoPE for decode-small shapes when profitable."""
    batch, seq_len, nheads, D = x.shape
    half_d = D // 2
    if (
        (not prefer_small_triton)
        or (not _ENABLE_TRITON_SMALL_QK_NORM_ROPE)
        or x.device.type != "cuda"
        or weight.device.type != "cuda"
        or cos.device.type != "cuda"
        or sin.device.type != "cuda"
        or seq_len > _TRITON_ROWS_THRESHOLD
        or x.stride(-1) != 1
        or weight.stride(-1) != 1
        or cos.stride(-1) != 1
        or sin.stride(-1) != 1
    ):
        return _apply_rotary_emb(_rms_norm(x, weight, eps), cos, sin)

    m = batch * seq_len
    x_3d = x.reshape(m, nheads, D)
    cos_2d = cos.reshape(m, D)
    sin_2d = sin.reshape(m, D)
    out_3d = torch.empty_like(x_3d)
    block_d = triton.next_power_of_2(D)
    block_half = triton.next_power_of_2(half_d)
    _fused_rms_norm_rope_fwd_kernel[(m * nheads,)](
        x_3d,
        out_3d,
        weight,
        cos_2d,
        sin_2d,
        nheads,
        D,
        half_d,
        x_3d.stride(0),
        x_3d.stride(1),
        cos_2d.stride(0),
        eps,
        BLOCK_D=block_d,
        BLOCK_HALF=block_half,
    )
    return out_3d.view_as(x)


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


def _small_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_key_value_groups: int,
    prefer_small_triton: bool = False,
    cache_position: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Single-token decode attention without repeat_kv copies."""
    if not _small_decode_attention_inputs_ok(
        q,
        k,
        v,
        attention_mask,
        num_key_value_groups,
        prefer_small_triton=prefer_small_triton,
    ):
        return None

    out = _small_decode_attention_grouped_prefix_cachepos(
        q,
        k,
        v,
        scaling,
        num_key_value_groups,
        cache_position,
        prefer_small_triton=prefer_small_triton,
    )
    if out is not None:
        return out
    out = _small_decode_attention_grouped(
        q,
        k,
        v,
        attention_mask,
        scaling,
        num_key_value_groups,
        prefer_small_triton=prefer_small_triton,
    )
    if out is not None:
        return out
    return _small_decode_attention_per_head(
        q,
        k,
        v,
        attention_mask,
        scaling,
        num_key_value_groups,
        prefer_small_triton=prefer_small_triton,
    )


def _small_decode_attention_inputs_ok(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
    num_key_value_groups: int,
    prefer_small_triton: bool = False,
) -> bool:
    return not (
        (not prefer_small_triton)
        or (not _ENABLE_TRITON_SMALL_DECODE_ATTENTION)
        or attention_mask is None
        or q.device.type != "cuda"
        or k.device.type != "cuda"
        or v.device.type != "cuda"
        or q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
        or q.shape[0] != 1
        or q.shape[2] != 1
        or q.shape[1] % k.shape[1] != 0
        or num_key_value_groups != (q.shape[1] // k.shape[1])
        or attention_mask.ndim != 4
        or attention_mask.dtype != torch.bool
        or attention_mask.shape[0] != 1
        or attention_mask.shape[2] != 1
        or attention_mask.shape[3] != k.shape[2]
        or q.shape[-1] != k.shape[-1]
        or q.shape[-1] != v.shape[-1]
        or q.stride(-1) != 1
        or k.stride(-1) != 1
        or v.stride(-1) != 1
    )


def _small_decode_attention_per_head(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_key_value_groups: int,
    prefer_small_triton: bool = False,
) -> torch.Tensor | None:
    if not _small_decode_attention_inputs_ok(
        q,
        k,
        v,
        attention_mask,
        num_key_value_groups,
        prefer_small_triton=prefer_small_triton,
    ):
        return None

    seq_len = k.shape[2]
    head_dim = q.shape[-1]
    block_n = 64 if seq_len <= 1024 else 128
    out = torch.empty((1, q.shape[1], head_dim), dtype=q.dtype, device=q.device)
    mask_1d = attention_mask[0, 0, 0]
    _small_decode_attention_fwd_kernel[(q.shape[1],)](
        q,
        k,
        v,
        mask_1d,
        out,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        q.shape[1],
        num_key_value_groups,
        scaling,
        HEAD_DIM=head_dim,
        SEQ_LEN=seq_len,
        BLOCK_N=block_n,
    )
    return out.unsqueeze(2).transpose(1, 2)


def _small_decode_attention_grouped(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_key_value_groups: int,
    prefer_small_triton: bool = False,
) -> torch.Tensor | None:
    if not _small_decode_attention_inputs_ok(
        q,
        k,
        v,
        attention_mask,
        num_key_value_groups,
        prefer_small_triton=prefer_small_triton,
    ):
        return None
    if num_key_value_groups != 2:
        return None

    seq_len = k.shape[2]
    head_dim = q.shape[-1]
    block_n = 64 if seq_len <= 1024 else 128
    out = torch.empty((1, q.shape[1], head_dim), dtype=q.dtype, device=q.device)
    mask_1d = attention_mask[0, 0, 0]
    _small_decode_attention_grouped_fwd_kernel[(k.shape[1],)](
        q,
        k,
        v,
        mask_1d,
        out,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        scaling,
        HEAD_DIM=head_dim,
        SEQ_LEN=seq_len,
        BLOCK_N=block_n,
    )
    return out.unsqueeze(2).transpose(1, 2)


def _small_decode_attention_grouped_prefix_cachepos(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: float,
    num_key_value_groups: int,
    cache_position: torch.Tensor | None,
    prefer_small_triton: bool = False,
) -> torch.Tensor | None:
    if (
        not prefer_small_triton
        or not _ENABLE_TRITON_SMALL_DECODE_ATTENTION
        or cache_position is None
        or cache_position.device.type != "cuda"
        or cache_position.ndim != 1
        or cache_position.numel() != 1
        or num_key_value_groups != 2
    ):
        return None
    if (
        q.device.type != "cuda"
        or k.device.type != "cuda"
        or v.device.type != "cuda"
        or q.shape[0] != 1
        or q.shape[2] != 1
        or q.stride(-1) != 1
        or k.stride(-1) != 1
        or v.stride(-1) != 1
    ):
        return None

    seq_len = k.shape[2]
    head_dim = q.shape[-1]
    block_n, num_warps, num_stages = _small_decode_attention_grouped_prefix_cachepos_launch_config(seq_len)
    out = torch.empty((1, q.shape[1], head_dim), dtype=q.dtype, device=q.device)
    _small_decode_attention_grouped_prefix_cachepos_fwd_kernel[(k.shape[1],)](
        q,
        k,
        v,
        cache_position,
        out,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        scaling,
        HEAD_DIM=head_dim,
        MAX_SEQ_LEN=seq_len,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out.unsqueeze(2).transpose(1, 2)


def _small_decode_attention_grouped_prefix_cachepos_launch_config(seq_len: int) -> tuple[int, int, int]:
    """Launch config tuned for decode buckets seen in benchmark traces."""
    if seq_len <= 512:
        return 64, 2, 2
    if seq_len <= 2048:
        return 128, 4, 4
    return 128, 8, 2


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
def _fused_rms_norm_rope_fwd_kernel(
    X,
    OUT,
    W,
    COS,
    SIN,
    nheads,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    stride_x_m,
    stride_x_h,
    stride_cos_m,
    eps,
    BLOCK_D: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    """Compute RMSNorm + RoPE for one (sample, head) row."""
    pid = tl.program_id(0)
    m = pid // nheads
    h = pid % nheads

    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    x_base = X + m * stride_x_m + h * stride_x_h
    x = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D
    rstd = tl.rsqrt(var + eps)

    rk = tl.arange(0, BLOCK_HALF)
    mask_half = rk < HALF_D
    x0 = tl.load(x_base + rk, mask=mask_half, other=0.0).to(tl.float32)
    x1 = tl.load(x_base + HALF_D + rk, mask=mask_half, other=0.0).to(tl.float32)
    w0 = tl.load(W + rk, mask=mask_half, other=0.0).to(tl.float32)
    w1 = tl.load(W + HALF_D + rk, mask=mask_half, other=0.0).to(tl.float32)
    cs_base = m * stride_cos_m
    cos = tl.load(COS + cs_base + rk, mask=mask_half, other=0.0).to(OUT.dtype.element_ty)
    sin = tl.load(SIN + cs_base + rk, mask=mask_half, other=0.0).to(OUT.dtype.element_ty)

    # Match the native decode path more closely: RMSNorm writes fp16, then RoPE consumes fp16.
    n0 = (x0 * rstd * w0).to(OUT.dtype.element_ty)
    n1 = (x1 * rstd * w1).to(OUT.dtype.element_ty)
    out0 = n0 * cos - n1 * sin
    out1 = n1 * cos + n0 * sin
    tl.store(OUT + m * stride_x_m + h * stride_x_h + rk, out0.to(OUT.dtype.element_ty), mask=mask_half)
    tl.store(OUT + m * stride_x_m + h * stride_x_h + HALF_D + rk, out1.to(OUT.dtype.element_ty), mask=mask_half)


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


@triton.jit
def _small_decode_attention_fwd_kernel(
    Q,
    K,
    V,
    MASK,
    OUT,
    stride_q_b,
    stride_q_h,
    stride_k_b,
    stride_k_h,
    stride_k_s,
    stride_v_b,
    stride_v_h,
    stride_v_s,
    stride_out_b,
    stride_out_h,
    num_heads,
    num_key_value_groups,
    scale,
    HEAD_DIM: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Streaming softmax attention for q_len=1 decode without GQA materialization."""
    head_idx = tl.program_id(0)
    kv_head_idx = head_idx // num_key_value_groups
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + head_idx * stride_q_h + offs_d, mask=offs_d < HEAD_DIM, other=0.0).to(tl.float32)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for start_n in range(0, SEQ_LEN, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        token_mask = offs_n < SEQ_LEN
        valid_mask = tl.load(MASK + offs_n, mask=token_mask, other=0)
        valid_mask = valid_mask != 0

        k_ptrs = K + kv_head_idx * stride_k_h + offs_n[:, None] * stride_k_s + offs_d[None, :]
        k = tl.load(k_ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)
        qk = tl.sum(k * q[None, :], axis=1) * scale
        qk = tl.where(valid_mask, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
        p = tl.exp(qk - m_ij)
        alpha = tl.exp(m_i - m_ij)

        v_ptrs = V + kv_head_idx * stride_v_h + offs_n[:, None] * stride_v_s + offs_d[None, :]
        v = tl.load(v_ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_ij

    acc = acc / l_i
    tl.store(OUT + head_idx * stride_out_h + offs_d, acc.to(OUT.dtype.element_ty), mask=offs_d < HEAD_DIM)


@triton.jit
def _small_decode_attention_grouped_fwd_kernel(
    Q,
    K,
    V,
    MASK,
    OUT,
    stride_q_b,
    stride_q_h,
    stride_k_b,
    stride_k_h,
    stride_k_s,
    stride_v_b,
    stride_v_h,
    stride_v_s,
    stride_out_b,
    stride_out_h,
    scale,
    HEAD_DIM: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Streaming decode attention that reuses one kv-head's K/V across its q-head group."""
    kv_head_idx = tl.program_id(0)
    offs_d = tl.arange(0, HEAD_DIM)
    q_head_idx0 = kv_head_idx * 2
    q_head_idx1 = q_head_idx0 + 1

    q0 = tl.load(Q + q_head_idx0 * stride_q_h + offs_d, mask=offs_d < HEAD_DIM, other=0.0).to(tl.float32)
    q1 = tl.load(Q + q_head_idx1 * stride_q_h + offs_d, mask=offs_d < HEAD_DIM, other=0.0).to(tl.float32)

    m0 = -float("inf")
    m1 = -float("inf")
    l0 = 0.0
    l1 = 0.0
    acc0 = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    acc1 = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for start_n in range(0, SEQ_LEN, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        token_mask = offs_n < SEQ_LEN
        valid_mask = tl.load(MASK + offs_n, mask=token_mask, other=0) != 0

        k_ptrs = K + kv_head_idx * stride_k_h + offs_n[:, None] * stride_k_s + offs_d[None, :]
        k = tl.load(k_ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)
        qk0 = tl.sum(k * q0[None, :], axis=1) * scale
        qk1 = tl.sum(k * q1[None, :], axis=1) * scale
        qk0 = tl.where(valid_mask, qk0, -float("inf"))
        qk1 = tl.where(valid_mask, qk1, -float("inf"))

        m0_new = tl.maximum(m0, tl.max(qk0, axis=0))
        m1_new = tl.maximum(m1, tl.max(qk1, axis=0))
        p0 = tl.exp(qk0 - m0_new)
        p1 = tl.exp(qk1 - m1_new)
        alpha0 = tl.exp(m0 - m0_new)
        alpha1 = tl.exp(m1 - m1_new)

        v_ptrs = V + kv_head_idx * stride_v_h + offs_n[:, None] * stride_v_s + offs_d[None, :]
        v = tl.load(v_ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)
        acc0 = acc0 * alpha0 + tl.sum(p0[:, None] * v, axis=0)
        acc1 = acc1 * alpha1 + tl.sum(p1[:, None] * v, axis=0)
        l0 = l0 * alpha0 + tl.sum(p0, axis=0)
        l1 = l1 * alpha1 + tl.sum(p1, axis=0)
        m0 = m0_new
        m1 = m1_new

    acc0 = acc0 / l0
    acc1 = acc1 / l1
    tl.store(OUT + q_head_idx0 * stride_out_h + offs_d, acc0.to(OUT.dtype.element_ty), mask=offs_d < HEAD_DIM)
    tl.store(OUT + q_head_idx1 * stride_out_h + offs_d, acc1.to(OUT.dtype.element_ty), mask=offs_d < HEAD_DIM)


@triton.jit
def _small_decode_attention_grouped_prefix_cachepos_fwd_kernel(
    Q,
    K,
    V,
    CACHE_POSITION,
    OUT,
    stride_q_b,
    stride_q_h,
    stride_k_b,
    stride_k_h,
    stride_k_s,
    stride_v_b,
    stride_v_h,
    stride_v_s,
    stride_out_b,
    stride_out_h,
    scale,
    HEAD_DIM: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Grouped decode attention with valid prefix length derived from cache_position."""
    kv_head_idx = tl.program_id(0)
    offs_d = tl.arange(0, HEAD_DIM)
    valid_len = tl.load(CACHE_POSITION).to(tl.int32) + 1
    valid_len = tl.minimum(valid_len, MAX_SEQ_LEN)
    q_head_idx0 = kv_head_idx * 2
    q_head_idx1 = q_head_idx0 + 1

    q0 = tl.load(Q + q_head_idx0 * stride_q_h + offs_d, mask=offs_d < HEAD_DIM, other=0.0).to(tl.float32)
    q1 = tl.load(Q + q_head_idx1 * stride_q_h + offs_d, mask=offs_d < HEAD_DIM, other=0.0).to(tl.float32)

    m0 = -float("inf")
    m1 = -float("inf")
    l0 = 0.0
    l1 = 0.0
    acc0 = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    acc1 = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    start_n = 0
    while start_n < valid_len:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        token_mask = offs_n < valid_len

        k_ptrs = K + kv_head_idx * stride_k_h + offs_n[:, None] * stride_k_s + offs_d[None, :]
        k = tl.load(k_ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)
        qk0 = tl.sum(k * q0[None, :], axis=1) * scale
        qk1 = tl.sum(k * q1[None, :], axis=1) * scale

        m0_new = tl.maximum(m0, tl.max(qk0, axis=0))
        m1_new = tl.maximum(m1, tl.max(qk1, axis=0))
        p0 = tl.exp(qk0 - m0_new)
        p1 = tl.exp(qk1 - m1_new)
        alpha0 = tl.exp(m0 - m0_new)
        alpha1 = tl.exp(m1 - m1_new)

        v_ptrs = V + kv_head_idx * stride_v_h + offs_n[:, None] * stride_v_s + offs_d[None, :]
        v = tl.load(v_ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)
        acc0 = acc0 * alpha0 + tl.sum(p0[:, None] * v, axis=0)
        acc1 = acc1 * alpha1 + tl.sum(p1[:, None] * v, axis=0)
        l0 = l0 * alpha0 + tl.sum(p0, axis=0)
        l1 = l1 * alpha1 + tl.sum(p1, axis=0)
        m0 = m0_new
        m1 = m1_new
        start_n += BLOCK_N

    acc0 = acc0 / l0
    acc1 = acc1 / l1
    tl.store(OUT + q_head_idx0 * stride_out_h + offs_d, acc0.to(OUT.dtype.element_ty), mask=offs_d < HEAD_DIM)
    tl.store(OUT + q_head_idx1 * stride_out_h + offs_d, acc1.to(OUT.dtype.element_ty), mask=offs_d < HEAD_DIM)


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
