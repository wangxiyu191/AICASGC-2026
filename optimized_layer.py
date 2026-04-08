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
from transformers.cache_utils import StaticCache, StaticLayer


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
        self.o_proj_weight = attn.o_proj.weight
        self.o_proj_bias = attn.o_proj.bias

        # === Fused gate+up projection: 2 GEMMs → 1 ===
        self.intermediate_size = mlp.intermediate_size
        self.gate_up_weight = nn.Parameter(
            torch.cat([mlp.gate_proj.weight.data,
                       mlp.up_proj.weight.data], dim=0),
            requires_grad=False,
        )
        self.down_proj = mlp.down_proj
        self.down_proj_weight = mlp.down_proj.weight
        self.down_proj_bias = mlp.down_proj.bias

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
        self._splitkv_workspace = None
        if _ENABLE_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV:
            self._ensure_splitkv_workspace()

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
        prefill_static_cache_no_mask = kwargs.pop("prefill_static_cache_no_mask", False)
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
            qkv = _cuda_partition_gemv(hidden_states, f"layer{self.layer_idx}.qkv")
            if qkv is None:
                qkv = _cuda_nosplit_gemv(hidden_states, self.qkv_weight, None)
            if qkv is None:
                qkv = F.linear(hidden_states, self.qkv_weight)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 3. QK-Norm + RoPE (before transpose — natural layout for cos/sin broadcast)
        cos, sin = position_embeddings
        with _nvtx_range("decode.qk_norm_rope", enable_decode_nvtx):
            q_view = q.view(hidden_shape_q)
            k_view = k.view(hidden_shape_kv)
            fused_qk = _fused_qk_rms_norm_rope(
                q_view,
                k_view,
                self.q_norm_weight,
                self.k_norm_weight,
                self.rms_norm_eps,
                cos,
                sin,
                prefer_small_triton=prefer_small_decode_triton,
            )
            if fused_qk is None:
                q_view = _fused_rms_norm_rope(
                    q_view,
                    self.q_norm_weight,
                    self.rms_norm_eps,
                    cos,
                    sin,
                    prefer_small_triton=prefer_small_decode_triton,
                )
                k_view = _fused_rms_norm_rope(
                    k_view,
                    self.k_norm_weight,
                    self.rms_norm_eps,
                    cos,
                    sin,
                    prefer_small_triton=prefer_small_decode_triton,
                )
            else:
                q_view, k_view = fused_qk
            q = q_view.transpose(1, 2)
            k = k_view.transpose(1, 2)
            v = v.view(hidden_shape_kv).transpose(1, 2)

        # 4. KV Cache
        attn_key_states = k
        attn_value_states = v
        cache_kwargs = None
        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin.unsqueeze(1), "cos": cos.unsqueeze(1),
                "cache_position": cache_position,
            }
            if not prefill_static_cache_no_mask:
                with _nvtx_range("decode.kv_cache_update", enable_decode_nvtx):
                    updated_kv = _static_cache_single_token_update(
                        past_key_values,
                        self.layer_idx,
                        k,
                        v,
                        cache_position,
                    )
                    if updated_kv is None:
                        attn_key_states, attn_value_states = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
                    else:
                        attn_key_states, attn_value_states = updated_kv

        # 5. Attention (SDPA)
        with _nvtx_range("decode.attention", enable_decode_nvtx):
            attn_output = _small_decode_attention(
                q,
                attn_key_states,
                attn_value_states,
                attention_mask,
                self.scaling,
                self.num_key_value_groups,
                prefer_small_triton=prefer_small_decode_triton,
                cache_position=cache_position,
                splitkv_workspace=self._splitkv_workspace,
            )
            if attn_output is None:
                attn_output, _ = self._attention_interface(
                    self, q, attn_key_states, attn_value_states, attention_mask,
                    dropout=0.0, scaling=self.scaling, **kwargs,
                )
        if past_key_values is not None and prefill_static_cache_no_mask:
            past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_output = attn_output.reshape(*input_shape, -1)
        if not attn_output.is_contiguous():
            attn_output = attn_output.contiguous()

        with _nvtx_range("decode.o_proj_post_attn_mlp_persistent_full", enable_decode_nvtx):
            fused_o_proj_post_attn_mlp_out = _fused_o_proj_post_attn_mlp_persistent(
                residual,
                attn_output,
                self.o_proj_weight,
                self.o_proj_bias,
                self.post_attn_layernorm_weight,
                self.gate_up_weight,
                self.down_proj_weight,
                self.down_proj_bias,
                self.rms_norm_eps,
                prefer_small_triton=prefer_small_decode_triton,
            )
            if fused_o_proj_post_attn_mlp_out is not None:
                return fused_o_proj_post_attn_mlp_out

        with _nvtx_range("decode.o_proj", enable_decode_nvtx):
            gemv_out = _cuda_partition_gemv(attn_output, f"layer{self.layer_idx}.o_proj")
            if gemv_out is None:
                gemv_out = _cuda_nosplit_gemv(attn_output, self.o_proj_weight, self.o_proj_bias)
            if gemv_out is None:
                gemv_out = _triton_decode_gemv(
                    attn_output, self.o_proj_weight, self.o_proj_bias,
                    prefer_small_triton=prefer_small_decode_triton,
                )
            attn_output = gemv_out if gemv_out is not None else self.o_proj(attn_output)

        with _nvtx_range("decode.post_attn_mlp_persistent_full", enable_decode_nvtx):
            fused_post_attn_mlp_out = _fused_post_attn_mlp_persistent(
                residual,
                attn_output,
                self.post_attn_layernorm_weight,
                self.gate_up_weight,
                self.down_proj_weight,
                self.down_proj_bias,
                self.rms_norm_eps,
                prefer_small_triton=prefer_small_decode_triton,
            )
            if fused_post_attn_mlp_out is not None:
                return fused_post_attn_mlp_out

        # 6. Residual + Post-attention LayerNorm
        gate_up = None
        with _nvtx_range("decode.post_attn_add_rmsnorm", enable_decode_nvtx):
            fused_gate_up = _fused_add_rmsnorm_gateup(
                residual,
                attn_output,
                self.post_attn_layernorm_weight,
                self.gate_up_weight,
                self.rms_norm_eps,
                prefer_small_triton=prefer_small_decode_triton,
            )
            if fused_gate_up is None:
                residual, hidden_states = _fused_add_rms_norm(
                    residual,
                    attn_output,
                    self.post_attn_layernorm_weight,
                    self.rms_norm_eps,
                    prefer_small_triton=prefer_small_decode_triton,
                )
            else:
                residual, gate_up = fused_gate_up
                hidden_states = None

        # 7. Fused gate+up (1 GEMM instead of 2) + SiLU activation
        with _nvtx_range("decode.gate_up_silu", enable_decode_nvtx):
            fused_mlp_out = None
            if gate_up is None:
                fused_mlp_out = _fused_gateup_silu_downproj(
                    hidden_states,
                    self.gate_up_weight,
                    self.down_proj_weight,
                    self.down_proj_bias,
                    prefer_small_triton=prefer_small_decode_triton,
                )
            if fused_mlp_out is None and gate_up is None:
                gate_up = _cuda_partition_gemv(hidden_states, f"layer{self.layer_idx}.gate_up")
                if gate_up is None:
                    gate_up = _cuda_nosplit_gemv(hidden_states, self.gate_up_weight, None)
                if gate_up is None:
                    gate_up = F.linear(hidden_states, self.gate_up_weight)

            if fused_mlp_out is None:
                fused_down_proj = _fused_silu_downproj(
                    gate_up,
                    self.down_proj_weight,
                    self.down_proj_bias,
                    prefer_small_triton=prefer_small_decode_triton,
                )
                if fused_down_proj is None:
                    gate, up = gate_up.chunk(2, dim=-1)
                    hidden_states = _fused_silu_mul(
                        gate,
                        up,
                        prefer_small_triton=prefer_small_decode_triton,
                    )
                else:
                    hidden_states = fused_down_proj
            else:
                fused_down_proj = fused_mlp_out
                hidden_states = fused_mlp_out
        with _nvtx_range("decode.down_proj", enable_decode_nvtx):
            residual_epilogue_bias = None
            if fused_down_proj is None:
                if _ENABLE_DOWNPROJ_RESIDUAL_EPILOGUE and self.down_proj_bias is None:
                    residual_epilogue_bias = residual
                down_proj_bias = self.down_proj_bias if residual_epilogue_bias is None else residual_epilogue_bias
                gemv_out = _cuda_nosplit_gemv(hidden_states, self.down_proj_weight, down_proj_bias)
                if gemv_out is None:
                    gemv_out = _triton_decode_gemv(
                        hidden_states, self.down_proj_weight, down_proj_bias,
                        prefer_small_triton=prefer_small_decode_triton,
                    )
                if gemv_out is not None:
                    hidden_states = gemv_out
                else:
                    hidden_states = self.down_proj(hidden_states)
                    if residual_epilogue_bias is not None:
                        hidden_states = hidden_states + residual_epilogue_bias

        # 8. Residual
        with _nvtx_range("decode.residual_add", enable_decode_nvtx):
            if residual_epilogue_bias is None:
                hidden_states = _add(hidden_states, residual, prefer_small_triton=prefer_small_decode_triton)
        return hidden_states

    def _ensure_splitkv_workspace(self, split_k: int | None = None):
        if split_k is None:
            split_k = _default_splitkv_workspace_split_k()
        if self.qkv_weight.device.type != "cuda":
            self._splitkv_workspace = None
            return
        layout = _splitkv_workspace_layout(
            num_kv_heads=self.num_kv_heads,
            num_key_value_groups=self.num_key_value_groups,
            head_dim=self.head_dim,
            split_k=split_k,
        )
        if (
            self._splitkv_workspace is not None
            and self._splitkv_workspace["split_k"] == split_k
            and self._splitkv_workspace["partial_acc"].device == self.qkv_weight.device
        ):
            return
        partial_acc = torch.empty(layout["partial_acc_shape"], dtype=torch.float32, device=self.qkv_weight.device)
        partial_stats = torch.empty(layout["partial_stats_shape"], dtype=torch.float32, device=self.qkv_weight.device)
        out = torch.empty(
            (1, self.num_heads, self.head_dim),
            dtype=self.qkv_weight.dtype,
            device=self.qkv_weight.device,
        )
        self._splitkv_workspace = {
            "split_k": split_k,
            "partial_acc": partial_acc,
            "partial_stats": partial_stats,
            "out": out,
        }


_TRITON_ROWS_THRESHOLD = 32
_ENABLE_TRITON_SMALL_RMSNORM = os.environ.get("ENABLE_TRITON_SMALL_RMSNORM", "1") == "1"
_ENABLE_CUDA_RMSNORM = os.environ.get("ENABLE_CUDA_RMSNORM", "0") == "1"
_ENABLE_TRITON_SMALL_ADD_RMSNORM = os.environ.get("ENABLE_TRITON_SMALL_ADD_RMSNORM", "1") == "1"
_ENABLE_TRITON_PREFILL_FUSED_ADD_RMSNORM = os.environ.get("ENABLE_TRITON_PREFILL_FUSED_ADD_RMSNORM", "1") == "1"
_ENABLE_CUDA_FUSED_ADD_RMSNORM = os.environ.get("ENABLE_CUDA_FUSED_ADD_RMSNORM", "0") == "1"
_ENABLE_CUDA_FUSED_ADD_RMSNORM_GATEUP = os.environ.get("ENABLE_CUDA_FUSED_ADD_RMSNORM_GATEUP", "0") == "1"
_ENABLE_CUDA_FUSED_POST_ATTN_MLP_PERSISTENT = os.environ.get("ENABLE_CUDA_FUSED_POST_ATTN_MLP_PERSISTENT", "0") == "1"
_ENABLE_CUDA_FUSED_O_PROJ_POST_ATTN_MLP_PERSISTENT = os.environ.get(
    "ENABLE_CUDA_FUSED_O_PROJ_POST_ATTN_MLP_PERSISTENT", "0"
) == "1"
_ENABLE_CUDA_FUSED_GATEUP_SILU_DOWNPROJ = os.environ.get("ENABLE_CUDA_FUSED_GATEUP_SILU_DOWNPROJ", "0") == "1"
_ENABLE_CUDA_FUSED_GATEUP_SILU_DOWNPROJ_MEGA = os.environ.get("ENABLE_CUDA_FUSED_GATEUP_SILU_DOWNPROJ_MEGA", "0") == "1"
_CUDA_FUSED_GATEUP_SILU_DOWNPROJ_VARIANT = os.environ.get(
    "CUDA_FUSED_GATEUP_SILU_DOWNPROJ_VARIANT",
    "persistent",
).strip().lower()
_ENABLE_TRITON_SMALL_SILU_MUL = os.environ.get("ENABLE_TRITON_SMALL_SILU_MUL", "0") == "1"
_ENABLE_TRITON_PREFILL_FUSED_SILU_MUL = os.environ.get("ENABLE_TRITON_PREFILL_FUSED_SILU_MUL", "1") == "1"
_ENABLE_CUDA_FUSED_SILU_MUL = os.environ.get("ENABLE_CUDA_FUSED_SILU_MUL", "0") == "1"
_ENABLE_CUDA_FUSED_SILU_DOWNPROJ = os.environ.get("ENABLE_CUDA_FUSED_SILU_DOWNPROJ", "0") == "1"
_ENABLE_CUDA_ADD = os.environ.get("ENABLE_CUDA_ADD", "0") == "1"
_ENABLE_DOWNPROJ_RESIDUAL_EPILOGUE = os.environ.get("ENABLE_DOWNPROJ_RESIDUAL_EPILOGUE", "0") == "1"
_ENABLE_TRITON_SMALL_QK_NORM_ROPE = os.environ.get("ENABLE_TRITON_SMALL_QK_NORM_ROPE", "1") == "1"
_ENABLE_TRITON_PREFILL_FUSED_RMS_NORM_ROPE = os.environ.get("ENABLE_TRITON_PREFILL_FUSED_RMS_NORM_ROPE", "0") == "1"
_ENABLE_TRITON_SMALL_FUSED_QK_NORM_ROPE = os.environ.get("ENABLE_TRITON_SMALL_FUSED_QK_NORM_ROPE", "0") == "1"
_ENABLE_TRITON_PREFILL_FUSED_QK_NORM_ROPE = os.environ.get("ENABLE_TRITON_PREFILL_FUSED_QK_NORM_ROPE", "0") == "1"
_ENABLE_CUDA_FUSED_QK_NORM_ROPE = os.environ.get("ENABLE_CUDA_FUSED_QK_NORM_ROPE", "0") == "1"
_ENABLE_TRITON_SMALL_DECODE_ATTENTION = os.environ.get("ENABLE_TRITON_SMALL_DECODE_ATTENTION", "1") == "1"
_ENABLE_TRITON_SMALL_DECODE_ATTENTION_DOT = os.environ.get("ENABLE_TRITON_SMALL_DECODE_ATTENTION_DOT", "1") == "1"
_ENABLE_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV = os.environ.get("ENABLE_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV", "1") == "1"
_ENABLE_CUDA_PERSISTENT_ATTENTION = os.environ.get("ENABLE_CUDA_PERSISTENT_ATTENTION", "0") == "1"
_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV_SPLIT_K = int(os.environ.get("TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV_SPLIT_K", "0"))
_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV_BLOCK_N = int(os.environ.get("TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV_BLOCK_N", "0"))
_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV_NUM_WARPS = int(os.environ.get("TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV_NUM_WARPS", "0"))
_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV_NUM_STAGES = int(os.environ.get("TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV_NUM_STAGES", "0"))
_ENABLE_TRITON_SMALL_DECODE_STATIC_KV = os.environ.get("ENABLE_TRITON_SMALL_DECODE_STATIC_KV", "1") == "1"
_ENABLE_TRITON_SMALL_DECODE_GEMV = os.environ.get("ENABLE_TRITON_SMALL_DECODE_GEMV", "1") == "1"
_ENABLE_CUDA_NOSPLIT_GEMV = os.environ.get("ENABLE_CUDA_NOSPLIT_GEMV", "0") == "1"
_ENABLE_PARTITION_GEMV = os.environ.get("ENABLE_PARTITION_GEMV", "0") == "1"
_PARTITION_DATA = None  # Set by evaluation_wrapper after probing
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
        and prefer_small_triton
        and _ENABLE_CUDA_RMSNORM
        and x.device.type == "cuda"
        and weight.device.type == "cuda"
        and x.stride(-1) == 1
        and weight.stride(-1) == 1
    ):
        import cuda_decode_small_ops
        return cuda_decode_small_ops.rms_norm(x, weight, eps)
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
    allow_triton = (
        (
            prefer_small_triton
            and _ENABLE_TRITON_SMALL_QK_NORM_ROPE
            and seq_len <= _TRITON_ROWS_THRESHOLD
        )
        or (
            (not prefer_small_triton)
            and _ENABLE_TRITON_PREFILL_FUSED_RMS_NORM_ROPE
        )
    )
    if (
        (not allow_triton)
        or x.device.type != "cuda"
        or weight.device.type != "cuda"
        or cos.device.type != "cuda"
        or sin.device.type != "cuda"
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


def _fused_qk_rms_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    prefer_small_triton: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fuse decode q_norm_rope + k_norm_rope into one launch when the head grouping matches."""
    if (
        prefer_small_triton
        and _ENABLE_CUDA_FUSED_QK_NORM_ROPE
        and q.device.type == "cuda"
        and k.device.type == "cuda"
        and q_weight.device.type == "cuda"
        and k_weight.device.type == "cuda"
        and cos.device.type == "cuda"
        and sin.device.type == "cuda"
        and q.ndim == 4
        and k.ndim == 4
        and q.shape[:2] == k.shape[:2]
        and q.shape[-1] == 128
        and k.shape[-1] == 128
        and q.shape[2] == k.shape[2] * 2
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and q_weight.stride(-1) == 1
        and k_weight.stride(-1) == 1
        and cos.stride(-1) == 1
        and sin.stride(-1) == 1
    ):
        import cuda_decode_small_ops
        return cuda_decode_small_ops.fused_qk_rms_norm_rope(
            q,
            k,
            q_weight,
            k_weight,
            eps,
            cos,
            sin,
        )

    allow_triton = (
        (
            prefer_small_triton
            and _ENABLE_TRITON_SMALL_FUSED_QK_NORM_ROPE
            and q.shape[1] <= _TRITON_ROWS_THRESHOLD
        )
        or (
            (not prefer_small_triton)
            and _ENABLE_TRITON_PREFILL_FUSED_QK_NORM_ROPE
        )
    )
    if (
        (not allow_triton)
        or q.device.type != "cuda"
        or k.device.type != "cuda"
        or q_weight.device.type != "cuda"
        or k_weight.device.type != "cuda"
        or cos.device.type != "cuda"
        or sin.device.type != "cuda"
        or q.ndim != 4
        or k.ndim != 4
        or q.shape[:2] != k.shape[:2]
        or q.shape[-1] != k.shape[-1]
        or q.stride(-1) != 1
        or k.stride(-1) != 1
        or q_weight.stride(-1) != 1
        or k_weight.stride(-1) != 1
        or cos.stride(-1) != 1
        or sin.stride(-1) != 1
    ):
        return None

    batch, seq_len, q_heads, d = q.shape
    kv_heads = k.shape[2]
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        return None
    groups = q_heads // kv_heads
    if groups <= 0 or groups > 4:
        return None
    half_d = d // 2
    if d != 128 or half_d * 2 != d:
        return None

    m = batch * seq_len
    q_3d = q.reshape(m, q_heads, d)
    k_3d = k.reshape(m, kv_heads, d)
    cos_2d = cos.reshape(m, d)
    sin_2d = sin.reshape(m, d)
    q_out = torch.empty_like(q_3d)
    k_out = torch.empty_like(k_3d)
    _fused_qk_rms_norm_rope_fwd_kernel[(m * kv_heads,)](
        q_3d,
        k_3d,
        q_out,
        k_out,
        q_weight,
        k_weight,
        cos_2d,
        sin_2d,
        kv_heads,
        q_3d.stride(0),
        q_3d.stride(1),
        k_3d.stride(0),
        k_3d.stride(1),
        cos_2d.stride(0),
        eps,
        GROUPS=groups,
        D=d,
        HALF_D=half_d,
        BLOCK_D=triton.next_power_of_2(d),
        BLOCK_HALF=triton.next_power_of_2(half_d),
    )
    return q_out.view_as(q), k_out.view_as(k)


def _fused_add_rms_norm(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    prefer_small_triton: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (residual_plus_x, rms_norm(residual_plus_x)) with one Triton kernel when useful."""
    N = residual.shape[-1]
    allow_triton = (
        (
            prefer_small_triton
            and _ENABLE_TRITON_SMALL_ADD_RMSNORM
        )
        or (
            (not prefer_small_triton)
            and _ENABLE_TRITON_PREFILL_FUSED_ADD_RMSNORM
        )
    )
    if (
        prefer_small_triton
        and _ENABLE_CUDA_FUSED_ADD_RMSNORM
        and residual.device.type == "cuda"
        and x.device.type == "cuda"
        and weight.device.type == "cuda"
        and residual.stride(-1) == 1
        and x.stride(-1) == 1
        and weight.stride(-1) == 1
    ):
        import cuda_decode_small_ops
        return cuda_decode_small_ops.fused_add_rms_norm(residual, x, weight, eps)

    if (
        (not allow_triton)
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


def _fused_add_rmsnorm_gateup(
    residual: torch.Tensor,
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    eps: float,
    prefer_small_triton: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        not prefer_small_triton
        or not _ENABLE_CUDA_FUSED_ADD_RMSNORM_GATEUP
        or residual.device.type != "cuda"
        or x.device.type != "cuda"
        or norm_weight.device.type != "cuda"
        or gate_up_weight.device.type != "cuda"
        or residual.dtype != torch.float16
        or x.dtype != torch.float16
        or norm_weight.dtype != torch.float16
        or gate_up_weight.dtype != torch.float16
        or residual.ndim < 2
        or residual.shape[-2] != 1
        or residual.shape != x.shape
        or residual.shape[-1] != 2048
        or gate_up_weight.shape != (12288, 2048)
        or residual.stride(-1) != 1
        or x.stride(-1) != 1
        or norm_weight.stride(-1) != 1
        or gate_up_weight.stride(-1) != 1
    ):
        return None

    import cuda_decode_gemv
    return cuda_decode_gemv.fused_add_rmsnorm_gateup_half(
        residual,
        x,
        norm_weight,
        gate_up_weight,
        None,
        eps,
    )


def _fused_post_attn_mlp_persistent(
    residual: torch.Tensor,
    attn_output: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_proj_bias: torch.Tensor | None,
    eps: float,
    prefer_small_triton: bool = False,
) -> torch.Tensor | None:
    if (
        not prefer_small_triton
        or not _ENABLE_CUDA_FUSED_POST_ATTN_MLP_PERSISTENT
        or residual.device.type != "cuda"
        or attn_output.device.type != "cuda"
        or norm_weight.device.type != "cuda"
        or gate_up_weight.device.type != "cuda"
        or down_proj_weight.device.type != "cuda"
        or residual.dtype != torch.float16
        or attn_output.dtype != torch.float16
        or norm_weight.dtype != torch.float16
        or gate_up_weight.dtype != torch.float16
        or down_proj_weight.dtype != torch.float16
        or residual.ndim < 2
        or residual.shape[-2] != 1
        or residual.shape != attn_output.shape
        or residual.shape[-1] != 2048
        or norm_weight.numel() != 2048
        or gate_up_weight.shape != (12288, 2048)
        or down_proj_weight.shape != (2048, 6144)
        or residual.stride(-1) != 1
        or attn_output.stride(-1) != 1
        or norm_weight.stride(-1) != 1
        or gate_up_weight.stride(-1) != 1
        or down_proj_weight.stride(-1) != 1
        or (down_proj_bias is not None and down_proj_bias.dtype != torch.float16)
    ):
        return None

    import cuda_decode_gemv
    _hidden_states, out = cuda_decode_gemv.fused_post_attn_mlp_persistent_half(
        residual,
        attn_output,
        norm_weight,
        gate_up_weight,
        down_proj_weight,
        down_proj_bias,
        eps,
    )
    return out


def _fused_o_proj_post_attn_mlp_persistent(
    residual: torch.Tensor,
    attn_input: torch.Tensor,
    o_proj_weight: torch.Tensor,
    o_proj_bias: torch.Tensor | None,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_proj_bias: torch.Tensor | None,
    eps: float,
    prefer_small_triton: bool = False,
) -> torch.Tensor | None:
    if (
        not prefer_small_triton
        or not _ENABLE_CUDA_FUSED_O_PROJ_POST_ATTN_MLP_PERSISTENT
        or residual.device.type != "cuda"
        or attn_input.device.type != "cuda"
        or o_proj_weight.device.type != "cuda"
        or norm_weight.device.type != "cuda"
        or gate_up_weight.device.type != "cuda"
        or down_proj_weight.device.type != "cuda"
        or residual.dtype != torch.float16
        or attn_input.dtype != torch.float16
        or o_proj_weight.dtype != torch.float16
        or norm_weight.dtype != torch.float16
        or gate_up_weight.dtype != torch.float16
        or down_proj_weight.dtype != torch.float16
        or residual.ndim < 2
        or residual.shape[-2] != 1
        or residual.shape != attn_input.shape
        or residual.shape[-1] != 2048
        or o_proj_weight.shape != (2048, 2048)
        or norm_weight.numel() != 2048
        or gate_up_weight.shape != (12288, 2048)
        or down_proj_weight.shape != (2048, 6144)
        or residual.stride(-1) != 1
        or attn_input.stride(-1) != 1
        or o_proj_weight.stride(-1) != 1
        or norm_weight.stride(-1) != 1
        or gate_up_weight.stride(-1) != 1
        or down_proj_weight.stride(-1) != 1
        or (o_proj_bias is not None and o_proj_bias.dtype != torch.float16)
        or (down_proj_bias is not None and down_proj_bias.dtype != torch.float16)
    ):
        return None

    import cuda_decode_gemv
    _hidden_states, out = cuda_decode_gemv.fused_o_proj_post_attn_mlp_persistent_half(
        residual,
        attn_input,
        o_proj_weight,
        o_proj_bias,
        norm_weight,
        gate_up_weight,
        down_proj_weight,
        down_proj_bias,
        eps,
    )
    return out


def _fused_silu_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
    prefer_small_triton: bool = False,
) -> torch.Tensor:
    """Fuse SiLU(gate) * up into one kernel when useful."""
    allow_triton = (
        (
            prefer_small_triton
            and _ENABLE_TRITON_SMALL_SILU_MUL
        )
        or (
            (not prefer_small_triton)
            and _ENABLE_TRITON_PREFILL_FUSED_SILU_MUL
        )
    )
    if (
        prefer_small_triton
        and _ENABLE_CUDA_FUSED_SILU_MUL
        and gate.device.type == "cuda"
        and up.device.type == "cuda"
        and gate.stride(-1) == 1
        and up.stride(-1) == 1
    ):
        import cuda_decode_small_ops
        return cuda_decode_small_ops.fused_silu_mul(gate, up)

    if (
        (not allow_triton)
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


def _fused_silu_downproj(
    gate_up: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_proj_bias: torch.Tensor | None,
    prefer_small_triton: bool = False,
) -> torch.Tensor | None:
    if (
        not prefer_small_triton
        or not _ENABLE_CUDA_FUSED_SILU_DOWNPROJ
        or _ENABLE_DOWNPROJ_RESIDUAL_EPILOGUE
        or gate_up.device.type != "cuda"
        or down_proj_weight.device.type != "cuda"
        or gate_up.dtype != torch.float16
        or down_proj_weight.dtype != torch.float16
        or gate_up.ndim < 2
        or gate_up.shape[-2] != 1
        or gate_up.shape[-1] != 12288
        or down_proj_weight.shape != (2048, 6144)
        or gate_up.stride(-1) != 1
        or down_proj_weight.stride(-1) != 1
        or (down_proj_bias is not None and down_proj_bias.dtype != torch.float16)
    ):
        return None

    import cuda_decode_gemv
    return cuda_decode_gemv.fused_silu_downproj_half(
        gate_up,
        down_proj_weight,
        down_proj_bias,
    )


def _fused_gateup_silu_downproj(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_proj_bias: torch.Tensor | None,
    prefer_small_triton: bool = False,
) -> torch.Tensor | None:
    if (
        not prefer_small_triton
        or not _ENABLE_CUDA_FUSED_GATEUP_SILU_DOWNPROJ
        or _ENABLE_DOWNPROJ_RESIDUAL_EPILOGUE
        or x.device.type != "cuda"
        or gate_up_weight.device.type != "cuda"
        or down_proj_weight.device.type != "cuda"
        or x.dtype != torch.float16
        or gate_up_weight.dtype != torch.float16
        or down_proj_weight.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
        or x.shape[-1] != 2048
        or gate_up_weight.shape != (12288, 2048)
        or down_proj_weight.shape != (2048, 6144)
        or x.stride(-1) != 1
        or gate_up_weight.stride(-1) != 1
        or down_proj_weight.stride(-1) != 1
        or (down_proj_bias is not None and down_proj_bias.dtype != torch.float16)
    ):
        return None

    import cuda_decode_gemv
    if down_proj_bias is None and _ENABLE_CUDA_FUSED_GATEUP_SILU_DOWNPROJ_MEGA:
        return cuda_decode_gemv.fused_gateup_silu_downproj_mega_half(
            x,
            gate_up_weight,
            down_proj_weight,
        )
    if down_proj_bias is None:
        if _CUDA_FUSED_GATEUP_SILU_DOWNPROJ_VARIANT == "persistent":
            _hidden_states, out = cuda_decode_gemv.fused_gateup_silu_downproj_persistent_half(
                x,
                gate_up_weight,
                down_proj_weight,
            )
            return out
        if _CUDA_FUSED_GATEUP_SILU_DOWNPROJ_VARIANT == "gridsync":
            _hidden_states, out = cuda_decode_gemv.fused_gateup_silu_downproj_gridsync_half(
                x,
                gate_up_weight,
                down_proj_weight,
            )
            return out
        if _CUDA_FUSED_GATEUP_SILU_DOWNPROJ_VARIANT == "softsync":
            _hidden_states, out = cuda_decode_gemv.fused_gateup_silu_downproj_softsync_half(
                x,
                gate_up_weight,
                down_proj_weight,
            )
            return out
    _hidden_states, out = cuda_decode_gemv.fused_gateup_silu_downproj_half(
        x,
        gate_up_weight,
        down_proj_weight,
        down_proj_bias,
    )
    return out


def _add(
    a: torch.Tensor,
    b: torch.Tensor,
    prefer_small_triton: bool = False,
) -> torch.Tensor:
    if (
        prefer_small_triton
        and _ENABLE_CUDA_ADD
        and a.device.type == "cuda"
        and b.device.type == "cuda"
        and a.stride(-1) == 1
        and b.stride(-1) == 1
    ):
        import cuda_decode_small_ops
        return cuda_decode_small_ops.add(a, b)
    return a + b


# ---------------------------------------------------------------------------
# Triton GEMV kernel for decode-phase o_proj / down_proj
#
# cuBLAS uses WMMA Tensor Core GEMM for M=1 small-N GEMV, achieving only
# 52-59% DRAM throughput and 10% occupancy (NCU validated).
# This dedicated GEMV kernel achieves 63%+ DRAM throughput and 58% occupancy
# by using proper 1D reduction instead of 2D WMMA tiles.
# Under CUDAGraph: 1.25-1.31x faster than cuBLAS for o_proj/down_proj.
# ---------------------------------------------------------------------------

@triton.jit
def _decode_gemv_fwd_kernel(
    X,           # [K] input vector (flattened)
    W,           # [N, K] weight matrix (row-major)
    OUT,         # [N] output vector
    K: tl.constexpr,
    N,
    stride_wn,   # W.stride(0) = K
    stride_wk,   # W.stride(1) = 1
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for start_k in range(0, K, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        x = tl.load(X + offs_k).to(tl.float32)
        w_ptrs = W + offs_m[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w = tl.load(w_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    tl.store(OUT + offs_m, acc.to(OUT.dtype.element_ty), mask=mask_m)


def _triton_decode_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    prefer_small_triton: bool = False,
) -> torch.Tensor | None:
    """Replace F.linear for decode-phase GEMV (M=1) when inside CUDAGraph.

    Returns None if conditions not met (caller falls back to F.linear).
    """
    if (
        not prefer_small_triton
        or not _ENABLE_TRITON_SMALL_DECODE_GEMV
        or x.device.type != "cuda"
        or weight.device.type != "cuda"
        or x.ndim < 2
        or x.shape[-2] != 1  # only for single-token decode
    ):
        return None

    x_vec = x.reshape(-1)
    N, K = weight.shape
    out_vec = torch.empty(N, dtype=x.dtype, device=x.device)
    BLOCK_M = 1
    BLOCK_K = 512
    _decode_gemv_fwd_kernel[(triton.cdiv(N, BLOCK_M),)](
        x_vec,
        weight,
        out_vec,
        K,
        N,
        weight.stride(0),
        weight.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    out = out_vec.view(*x.shape[:-1], N)
    if bias is not None:
        out = out + bias
    return out


def _cuda_nosplit_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    """Replace F.linear for decode-phase GEMV (M=1) using CUDA nosplit kernel.

    Returns None if conditions not met (caller falls back).
    """
    if (
        not _ENABLE_CUDA_NOSPLIT_GEMV
        or x.device.type != "cuda"
        or weight.device.type != "cuda"
        or x.dtype != torch.float16
        or weight.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
    ):
        return None

    import cuda_decode_gemv
    return cuda_decode_gemv.nosplit_matvec_half(x, weight, bias)


def _cuda_partition_gemv(
    x: torch.Tensor,
    layer_key: str,
) -> torch.Tensor | None:
    """Partition-aware GEMV for k=2048 with interleaved weight layout.

    Returns None if partition data not available (caller falls back).
    """
    if (
        not _ENABLE_PARTITION_GEMV
        or _PARTITION_DATA is None
        or layer_key not in _PARTITION_DATA["weights"]
        or x.device.type != "cuda"
        or x.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
    ):
        return None

    wd = _PARTITION_DATA["weights"][layer_key]
    import cuda_decode_gemv
    return cuda_decode_gemv.nosplit_matvec_half_partition(
        x, wd["buf"], None, wd["chunk_classes"],
        _PARTITION_DATA["sm_class_lo"], _PARTITION_DATA["sm_class_hi"],
        wd["n_rows"],
    )


def _static_cache_single_token_update(
    past_key_values,
    layer_idx: int,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_position: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Single-token StaticCache update specialized for CUDA decode."""
    if (
        not _ENABLE_TRITON_SMALL_DECODE_STATIC_KV
        or not isinstance(past_key_values, StaticCache)
        or cache_position is None
        or cache_position.device.type != "cuda"
        or cache_position.ndim != 1
        or cache_position.numel() != 1
        or key_states.device.type != "cuda"
        or value_states.device.type != "cuda"
        or key_states.ndim != 4
        or value_states.ndim != 4
        or key_states.shape[0] != 1
        or value_states.shape[0] != 1
        or key_states.shape[2] != 1
        or value_states.shape[2] != 1
        or key_states.shape[:3] != value_states.shape[:3]
        or key_states.shape[-1] != value_states.shape[-1]
    ):
        return None

    layer = past_key_values.layers[layer_idx]
    if not isinstance(layer, StaticLayer):
        return None
    cumulative_length = getattr(layer, "cumulative_length", None)
    if not isinstance(cumulative_length, torch.Tensor) or cumulative_length.device.type != "cuda":
        return None

    if not layer.is_initialized:
        layer.lazy_initialization(key_states, value_states)

    head_dim = key_states.shape[-1]
    block_d = triton.next_power_of_2(head_dim)
    if block_d > 256:
        return None

    _static_cache_single_token_update_fwd_kernel[(key_states.shape[1],)](
        key_states,
        value_states,
        layer.keys,
        layer.values,
        cache_position,
        cumulative_length,
        key_states.stride(1),
        key_states.stride(2),
        key_states.stride(3),
        value_states.stride(1),
        value_states.stride(2),
        value_states.stride(3),
        layer.keys.stride(1),
        layer.keys.stride(2),
        layer.keys.stride(3),
        layer.values.stride(1),
        layer.values.stride(2),
        layer.values.stride(3),
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        num_warps=1,
        num_stages=1,
    )
    return layer.keys, layer.values


def _small_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_key_value_groups: int,
    prefer_small_triton: bool = False,
    cache_position: torch.Tensor | None = None,
    splitkv_workspace: dict | None = None,
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

    if (
        _ENABLE_CUDA_PERSISTENT_ATTENTION
        and prefer_small_triton
        and cache_position is not None
        and cache_position.device.type == "cuda"
        and cache_position.ndim == 1
        and cache_position.numel() == 1
        and q.dtype == torch.float16
        and k.dtype == torch.float16
        and v.dtype == torch.float16
        and q.shape[-1] == 128
        and num_key_value_groups == 2
    ):
        import cuda_decode_gemv
        out = cuda_decode_gemv.decode_attn_persistent_half(
            q,
            k,
            v,
            cache_position,
            scale=scaling,
            workspace=splitkv_workspace,
        )
        return out

    out = _small_decode_attention_grouped_prefix_cachepos_splitkv(
        q,
        k,
        v,
        scaling,
        num_key_value_groups,
        cache_position,
        prefer_small_triton=prefer_small_triton,
        workspace=splitkv_workspace,
    )
    if out is not None:
        return out

    if _ENABLE_TRITON_SMALL_DECODE_ATTENTION_DOT:
        out = _small_decode_attention_grouped_prefix_cachepos_dot(
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


@triton.jit
def _static_cache_single_token_update_fwd_kernel(
    key_ptr,
    value_ptr,
    keys_cache_ptr,
    values_cache_ptr,
    cache_position_ptr,
    cumulative_length_ptr,
    key_stride_h,
    key_stride_s,
    key_stride_d,
    value_stride_h,
    value_stride_s,
    value_stride_d,
    keys_cache_stride_h,
    keys_cache_stride_s,
    keys_cache_stride_d,
    values_cache_stride_h,
    values_cache_stride_s,
    values_cache_stride_d,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    kv_head = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    pos = tl.load(cache_position_ptr).to(tl.int64)

    key_ptrs = key_ptr + kv_head * key_stride_h + offs_d * key_stride_d
    value_ptrs = value_ptr + kv_head * value_stride_h + offs_d * value_stride_d
    keys_cache_ptrs = keys_cache_ptr + kv_head * keys_cache_stride_h + pos * keys_cache_stride_s + offs_d * keys_cache_stride_d
    values_cache_ptrs = values_cache_ptr + kv_head * values_cache_stride_h + pos * values_cache_stride_s + offs_d * values_cache_stride_d

    tl.store(keys_cache_ptrs, tl.load(key_ptrs, mask=mask_d), mask=mask_d)
    tl.store(values_cache_ptrs, tl.load(value_ptrs, mask=mask_d), mask=mask_d)

    if kv_head == 0:
        tl.store(cumulative_length_ptr, pos + 1)


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


def _small_decode_attention_grouped_prefix_cachepos_dot(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: float,
    num_key_value_groups: int,
    cache_position: torch.Tensor | None,
    prefer_small_triton: bool = False,
) -> torch.Tensor | None:
    """Experimental grouped decode attention using tl.dot for qk / pv blocks."""
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
    _small_decode_attention_grouped_prefix_cachepos_dot_fwd_kernel[(k.shape[1],)](
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


def _small_decode_attention_grouped_prefix_cachepos_splitkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: float,
    num_key_value_groups: int,
    cache_position: torch.Tensor | None,
    prefer_small_triton: bool = False,
    workspace: dict | None = None,
) -> torch.Tensor | None:
    if (
        not prefer_small_triton
        or not _ENABLE_TRITON_SMALL_DECODE_ATTENTION
        or not _ENABLE_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV
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
    split_k, block_n, num_warps, num_stages = _small_decode_attention_grouped_prefix_cachepos_splitkv_launch_config(seq_len)
    if split_k <= 1:
        return None
    layout = _splitkv_workspace_layout(
        num_kv_heads=k.shape[1],
        num_key_value_groups=num_key_value_groups,
        head_dim=head_dim,
        split_k=split_k,
    )

    if (
        workspace is not None
        and workspace.get("split_k") == split_k
        and workspace["partial_acc"].device == q.device
        and workspace["out"].dtype == q.dtype
    ):
        partial_acc = workspace["partial_acc"]
        partial_stats = workspace["partial_stats"]
        out = workspace["out"]
    else:
        partial_acc = torch.empty(layout["partial_acc_shape"], dtype=torch.float32, device=q.device)
        partial_stats = torch.empty(layout["partial_stats_shape"], dtype=torch.float32, device=q.device)
        out = torch.empty((1, q.shape[1], head_dim), dtype=q.dtype, device=q.device)

    _small_decode_attention_grouped_prefix_cachepos_splitkv_fwd_kernel[(k.shape[1], split_k)](
        q,
        k,
        v,
        cache_position,
        partial_acc,
        partial_stats,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_stats.stride(0),
        partial_stats.stride(1),
        partial_stats.stride(2),
        partial_stats.stride(3),
        scaling,
        HEAD_DIM=head_dim,
        MAX_SEQ_LEN=seq_len,
        BLOCK_N=block_n,
        SPLIT_K=split_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _small_decode_attention_grouped_prefix_cachepos_splitkv_merge_kernel[(k.shape[1],)](
        partial_acc,
        partial_stats,
        out,
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_stats.stride(0),
        partial_stats.stride(1),
        partial_stats.stride(2),
        partial_stats.stride(3),
        out.stride(0),
        out.stride(1),
        HEAD_DIM=head_dim,
        SPLIT_K=split_k,
    )
    return out.unsqueeze(2).transpose(1, 2)


def _small_decode_attention_grouped_prefix_cachepos_launch_config(seq_len: int) -> tuple[int, int, int]:
    """Launch config tuned for decode buckets seen in benchmark traces."""
    if seq_len <= 512:
        return 64, 2, 2
    if seq_len <= 2048:
        return 128, 4, 4
    return 128, 8, 2


def _small_decode_attention_grouped_prefix_cachepos_splitkv_launch_config(seq_len: int) -> tuple[int, int, int, int]:
    """Split-KV config for increasing decode attention parallelism on long-ish caches."""
    if seq_len < 512:
        split_k, block_n, num_warps, num_stages = 1, 64, 2, 2
    elif seq_len < 1024:
        split_k, block_n, num_warps, num_stages = 8, 64, 2, 2
    else:
        split_k, block_n, num_warps, num_stages = 4, 128, 2, 2
    if _TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV_SPLIT_K > 0:
        split_k = _TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV_SPLIT_K
    if _TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV_BLOCK_N > 0:
        block_n = _TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV_BLOCK_N
    if _TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV_NUM_WARPS > 0:
        num_warps = _TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV_NUM_WARPS
    if _TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV_NUM_STAGES > 0:
        num_stages = _TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV_NUM_STAGES
    return split_k, block_n, num_warps, num_stages


def _default_splitkv_workspace_split_k() -> int:
    if _TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV_SPLIT_K > 0:
        return _TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV_SPLIT_K
    return 4


def _splitkv_workspace_layout(
    *,
    num_kv_heads: int,
    num_key_value_groups: int,
    head_dim: int,
    split_k: int,
) -> dict:
    return {
        "partial_acc_shape": (num_kv_heads, split_k, num_key_value_groups, head_dim),
        "partial_stats_shape": (num_kv_heads, split_k, num_key_value_groups, 2),
    }


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
def _fused_qk_rms_norm_rope_fwd_kernel(
    Q,
    K,
    Q_OUT,
    K_OUT,
    Q_W,
    K_W,
    COS,
    SIN,
    kv_heads,
    stride_q_m,
    stride_q_h,
    stride_k_m,
    stride_k_h,
    stride_cos_m,
    eps,
    GROUPS: tl.constexpr,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    """Compute q_norm_rope and k_norm_rope for one KV head-group in one program."""
    pid = tl.program_id(0)
    m = pid // kv_heads
    kv_h = pid % kv_heads
    q_head_base = kv_h * GROUPS

    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    rk = tl.arange(0, BLOCK_HALF)
    mask_half = rk < HALF_D
    cs_base = m * stride_cos_m
    cos = tl.load(COS + cs_base + rk, mask=mask_half, other=0.0).to(Q_OUT.dtype.element_ty)
    sin = tl.load(SIN + cs_base + rk, mask=mask_half, other=0.0).to(Q_OUT.dtype.element_ty)

    k_base = K + m * stride_k_m + kv_h * stride_k_h
    k = tl.load(k_base + cols, mask=mask, other=0.0).to(tl.float32)
    k_var = tl.sum(k * k, axis=0) / D
    k_rstd = tl.rsqrt(k_var + eps)
    k0 = tl.load(k_base + rk, mask=mask_half, other=0.0).to(tl.float32)
    k1 = tl.load(k_base + HALF_D + rk, mask=mask_half, other=0.0).to(tl.float32)
    kw0 = tl.load(K_W + rk, mask=mask_half, other=0.0).to(tl.float32)
    kw1 = tl.load(K_W + HALF_D + rk, mask=mask_half, other=0.0).to(tl.float32)
    kn0 = (k0 * k_rstd * kw0).to(K_OUT.dtype.element_ty)
    kn1 = (k1 * k_rstd * kw1).to(K_OUT.dtype.element_ty)
    tl.store(K_OUT + m * stride_k_m + kv_h * stride_k_h + rk, kn0 * cos - kn1 * sin, mask=mask_half)
    tl.store(K_OUT + m * stride_k_m + kv_h * stride_k_h + HALF_D + rk, kn1 * cos + kn0 * sin, mask=mask_half)

    for group_idx in range(GROUPS):
        q_h = q_head_base + group_idx
        q_base = Q + m * stride_q_m + q_h * stride_q_h
        q = tl.load(q_base + cols, mask=mask, other=0.0).to(tl.float32)
        q_var = tl.sum(q * q, axis=0) / D
        q_rstd = tl.rsqrt(q_var + eps)
        q0 = tl.load(q_base + rk, mask=mask_half, other=0.0).to(tl.float32)
        q1 = tl.load(q_base + HALF_D + rk, mask=mask_half, other=0.0).to(tl.float32)
        qw0 = tl.load(Q_W + rk, mask=mask_half, other=0.0).to(tl.float32)
        qw1 = tl.load(Q_W + HALF_D + rk, mask=mask_half, other=0.0).to(tl.float32)
        qn0 = (q0 * q_rstd * qw0).to(Q_OUT.dtype.element_ty)
        qn1 = (q1 * q_rstd * qw1).to(Q_OUT.dtype.element_ty)
        tl.store(Q_OUT + m * stride_q_m + q_h * stride_q_h + rk, qn0 * cos - qn1 * sin, mask=mask_half)
        tl.store(Q_OUT + m * stride_q_m + q_h * stride_q_h + HALF_D + rk, qn1 * cos + qn0 * sin, mask=mask_half)


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


@triton.jit
def _small_decode_attention_grouped_prefix_cachepos_dot_fwd_kernel(
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
    """Experimental grouped decode attention using matrix-style dot blocks."""
    kv_head_idx = tl.program_id(0)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_h = tl.arange(0, 16)
    mask_h = offs_h < 2
    valid_len = tl.load(CACHE_POSITION).to(tl.int32) + 1
    valid_len = tl.minimum(valid_len, MAX_SEQ_LEN)
    q_head_idx = kv_head_idx * 2 + tl.minimum(offs_h, 1)

    q_ptrs = Q + q_head_idx[:, None] * stride_q_h + offs_d[None, :]
    q = tl.load(q_ptrs, mask=mask_h[:, None] & (offs_d[None, :] < HEAD_DIM), other=0.0)

    e_max = tl.zeros((16,), dtype=tl.float32)
    e_sum = tl.zeros((16,), dtype=tl.float32)
    acc = tl.zeros((16, HEAD_DIM), dtype=tl.float32)

    start_n = 0
    while start_n < valid_len:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        token_mask = offs_n < valid_len

        k_ptrs = K + kv_head_idx * stride_k_h + offs_d[:, None] + offs_n[None, :] * stride_k_s
        k = tl.load(k_ptrs, mask=(offs_d[:, None] < HEAD_DIM) & token_mask[None, :], other=0.0)
        qk = tl.dot(q, k, out_dtype=tl.float32) * scale
        qk = tl.where(mask_h[:, None] & token_mask[None, :], qk, -float("inf"))

        prev_e_max = tl.where(mask_h, e_max, 0.0)
        n_e_max = tl.where(mask_h, tl.maximum(tl.max(qk, axis=1), prev_e_max), 0.0)
        re_scale = tl.where(mask_h, tl.exp(prev_e_max - n_e_max), 0.0)
        p = tl.where(mask_h[:, None], tl.exp(qk - n_e_max[:, None]), 0.0)

        v_ptrs = V + kv_head_idx * stride_v_h + offs_n[:, None] * stride_v_s + offs_d[None, :]
        v = tl.load(v_ptrs, mask=token_mask[:, None] & (offs_d[None, :] < HEAD_DIM), other=0.0)
        acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        e_sum = e_sum * re_scale + tl.sum(p, axis=1)
        e_max = n_e_max
        start_n += BLOCK_N

    out = acc / e_sum[:, None]
    out_ptrs = OUT + q_head_idx[:, None] * stride_out_h + offs_d[None, :]
    tl.store(out_ptrs, out.to(OUT.dtype.element_ty), mask=mask_h[:, None] & (offs_d[None, :] < HEAD_DIM))


@triton.jit
def _small_decode_attention_grouped_prefix_cachepos_splitkv_fwd_kernel(
    Q,
    K,
    V,
    CACHE_POSITION,
    PARTIAL_ACC,
    PARTIAL_STATS,
    stride_q_b,
    stride_q_h,
    stride_k_b,
    stride_k_h,
    stride_k_s,
    stride_v_b,
    stride_v_h,
    stride_v_s,
    stride_pa_kh,
    stride_pa_split,
    stride_pa_head,
    stride_ps_kh,
    stride_ps_split,
    stride_ps_head,
    stride_ps_stat,
    scale,
    HEAD_DIM: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    kv_head_idx = tl.program_id(0)
    split_idx = tl.program_id(1)
    offs_d = tl.arange(0, HEAD_DIM)
    valid_len = tl.load(CACHE_POSITION).to(tl.int32) + 1
    valid_len = tl.minimum(valid_len, MAX_SEQ_LEN)
    chunk_size = tl.cdiv(valid_len, SPLIT_K)
    chunk_start = split_idx * chunk_size
    chunk_end = tl.minimum(chunk_start + chunk_size, valid_len)

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

    start_n = chunk_start
    while start_n < chunk_end:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        token_mask = offs_n < chunk_end
        k_ptrs = K + kv_head_idx * stride_k_h + offs_n[:, None] * stride_k_s + offs_d[None, :]
        k = tl.load(k_ptrs, mask=token_mask[:, None] & (offs_d[None, :] < HEAD_DIM), other=0.0).to(tl.float32)
        qk0 = tl.sum(k * q0[None, :], axis=1) * scale
        qk1 = tl.sum(k * q1[None, :], axis=1) * scale

        next_m0 = tl.maximum(m0, tl.max(qk0, axis=0))
        next_m1 = tl.maximum(m1, tl.max(qk1, axis=0))
        alpha0 = tl.exp(m0 - next_m0)
        alpha1 = tl.exp(m1 - next_m1)
        p0 = tl.exp(qk0 - next_m0)
        p1 = tl.exp(qk1 - next_m1)
        v_ptrs = V + kv_head_idx * stride_v_h + offs_n[:, None] * stride_v_s + offs_d[None, :]
        v = tl.load(v_ptrs, mask=token_mask[:, None] & (offs_d[None, :] < HEAD_DIM), other=0.0).to(tl.float32)
        acc0 = acc0 * alpha0 + tl.sum(p0[:, None] * v, axis=0)
        acc1 = acc1 * alpha1 + tl.sum(p1[:, None] * v, axis=0)
        l0 = l0 * alpha0 + tl.sum(p0, axis=0)
        l1 = l1 * alpha1 + tl.sum(p1, axis=0)
        m0 = next_m0
        m1 = next_m1
        start_n += BLOCK_N

    partial_acc_base = PARTIAL_ACC + kv_head_idx * stride_pa_kh + split_idx * stride_pa_split
    tl.store(partial_acc_base + 0 * stride_pa_head + offs_d, acc0, mask=offs_d < HEAD_DIM)
    tl.store(partial_acc_base + 1 * stride_pa_head + offs_d, acc1, mask=offs_d < HEAD_DIM)

    partial_stats_base = PARTIAL_STATS + kv_head_idx * stride_ps_kh + split_idx * stride_ps_split
    tl.store(partial_stats_base + 0 * stride_ps_head + 0 * stride_ps_stat, l0)
    tl.store(partial_stats_base + 1 * stride_ps_head + 0 * stride_ps_stat, l1)
    tl.store(partial_stats_base + 0 * stride_ps_head + 1 * stride_ps_stat, m0)
    tl.store(partial_stats_base + 1 * stride_ps_head + 1 * stride_ps_stat, m1)
@triton.jit
def _small_decode_attention_grouped_prefix_cachepos_splitkv_merge_kernel(
    PARTIAL_ACC,
    PARTIAL_STATS,
    OUT,
    stride_pa_kh,
    stride_pa_split,
    stride_pa_head,
    stride_ps_kh,
    stride_ps_split,
    stride_ps_head,
    stride_ps_stat,
    stride_out_b,
    stride_out_h,
    HEAD_DIM: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    kv_head_idx = tl.program_id(0)
    offs_d = tl.arange(0, HEAD_DIM)
    max0 = -float("inf")
    max1 = -float("inf")
    sum0 = 0.0
    sum1 = 0.0
    acc0 = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    acc1 = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for split_idx in range(SPLIT_K):
        stats_base = PARTIAL_STATS + kv_head_idx * stride_ps_kh + split_idx * stride_ps_split
        acc_base = PARTIAL_ACC + kv_head_idx * stride_pa_kh + split_idx * stride_pa_split

        m0 = tl.load(stats_base + 0 * stride_ps_head + 1 * stride_ps_stat)
        m1 = tl.load(stats_base + 1 * stride_ps_head + 1 * stride_ps_stat)
        l0 = tl.load(stats_base + 0 * stride_ps_head + 0 * stride_ps_stat)
        l1 = tl.load(stats_base + 1 * stride_ps_head + 0 * stride_ps_stat)
        part_acc0 = tl.load(acc_base + 0 * stride_pa_head + offs_d, mask=offs_d < HEAD_DIM, other=0.0)
        part_acc1 = tl.load(acc_base + 1 * stride_pa_head + offs_d, mask=offs_d < HEAD_DIM, other=0.0)

        next_max0 = tl.maximum(max0, m0)
        next_max1 = tl.maximum(max1, m1)
        prev_scale0 = tl.exp(max0 - next_max0)
        prev_scale1 = tl.exp(max1 - next_max1)
        part_scale0 = tl.exp(m0 - next_max0)
        part_scale1 = tl.exp(m1 - next_max1)

        acc0 = acc0 * prev_scale0 + part_acc0 * part_scale0
        acc1 = acc1 * prev_scale1 + part_acc1 * part_scale1
        sum0 = sum0 * prev_scale0 + l0 * part_scale0
        sum1 = sum1 * prev_scale1 + l1 * part_scale1
        max0 = next_max0
        max1 = next_max1

    out0 = acc0 / sum0
    out1 = acc1 / sum1
    q_head_idx0 = kv_head_idx * 2
    q_head_idx1 = q_head_idx0 + 1
    tl.store(OUT + q_head_idx0 * stride_out_h + offs_d, out0.to(OUT.dtype.element_ty), mask=offs_d < HEAD_DIM)
    tl.store(OUT + q_head_idx1 * stride_out_h + offs_d, out1.to(OUT.dtype.element_ty), mask=offs_d < HEAD_DIM)


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
            if enabled and _ENABLE_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV:
                layer._ensure_splitkv_workspace()
    return previous


def set_cuda_mlp_fusion_flags(
    *,
    add_rmsnorm_gateup: bool | None = None,
    post_attn_mlp_persistent: bool | None = None,
    o_proj_post_attn_mlp_persistent: bool | None = None,
    silu_downproj: bool | None = None,
    gateup_silu_downproj: bool | None = None,
    gateup_silu_downproj_mega: bool | None = None,
) -> tuple[bool, bool, bool, bool, bool, bool]:
    global _ENABLE_CUDA_FUSED_ADD_RMSNORM_GATEUP
    global _ENABLE_CUDA_FUSED_POST_ATTN_MLP_PERSISTENT
    global _ENABLE_CUDA_FUSED_O_PROJ_POST_ATTN_MLP_PERSISTENT
    global _ENABLE_CUDA_FUSED_SILU_DOWNPROJ
    global _ENABLE_CUDA_FUSED_GATEUP_SILU_DOWNPROJ
    global _ENABLE_CUDA_FUSED_GATEUP_SILU_DOWNPROJ_MEGA
    previous = (
        _ENABLE_CUDA_FUSED_ADD_RMSNORM_GATEUP,
        _ENABLE_CUDA_FUSED_POST_ATTN_MLP_PERSISTENT,
        _ENABLE_CUDA_FUSED_O_PROJ_POST_ATTN_MLP_PERSISTENT,
        _ENABLE_CUDA_FUSED_SILU_DOWNPROJ,
        _ENABLE_CUDA_FUSED_GATEUP_SILU_DOWNPROJ,
        _ENABLE_CUDA_FUSED_GATEUP_SILU_DOWNPROJ_MEGA,
    )
    if add_rmsnorm_gateup is not None:
        _ENABLE_CUDA_FUSED_ADD_RMSNORM_GATEUP = add_rmsnorm_gateup
    if post_attn_mlp_persistent is not None:
        _ENABLE_CUDA_FUSED_POST_ATTN_MLP_PERSISTENT = post_attn_mlp_persistent
    if o_proj_post_attn_mlp_persistent is not None:
        _ENABLE_CUDA_FUSED_O_PROJ_POST_ATTN_MLP_PERSISTENT = o_proj_post_attn_mlp_persistent
    if silu_downproj is not None:
        _ENABLE_CUDA_FUSED_SILU_DOWNPROJ = silu_downproj
    if gateup_silu_downproj is not None:
        _ENABLE_CUDA_FUSED_GATEUP_SILU_DOWNPROJ = gateup_silu_downproj
    if gateup_silu_downproj_mega is not None:
        _ENABLE_CUDA_FUSED_GATEUP_SILU_DOWNPROJ_MEGA = gateup_silu_downproj_mega
    return previous
