#!/usr/bin/env python3
"""Decode GEMV fused-op and attention wrappers."""

from __future__ import annotations

import os

import torch

from .runtime import _load_ext

def fused_add_rmsnorm_gateup_half(
    residual: torch.Tensor,
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    workspace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        residual.device.type != "cuda"
        or x.device.type != "cuda"
        or norm_weight.device.type != "cuda"
        or weight.device.type != "cuda"
        or residual.dtype != torch.float16
        or x.dtype != torch.float16
        or norm_weight.dtype != torch.float16
        or weight.dtype != torch.float16
        or residual.ndim < 2
        or residual.shape[-2] != 1
        or residual.shape != x.shape
    ):
        raise RuntimeError("fused_add_rmsnorm_gateup_half expects fp16 CUDA single-token decode tensors")

    residual_vec = residual.reshape(-1).contiguous()
    x_vec = x.reshape(-1).contiguous()
    norm_vec = norm_weight.reshape(-1).contiguous()
    n, k = weight.shape
    if (
        workspace is not None
        and workspace.get("type") == "fused_add_rmsnorm_gateup"
        and tuple(workspace["residual_out"].shape) == (k,)
        and tuple(workspace["gate_up_out"].shape) == (n,)
        and workspace["residual_out"].device == residual.device
        and workspace["gate_up_out"].device == residual.device
        and workspace["residual_out"].dtype == residual.dtype
        and workspace["gate_up_out"].dtype == residual.dtype
    ):
        residual_out = workspace["residual_out"]
        gate_up_out = workspace["gate_up_out"]
    else:
        residual_out = torch.empty((k,), dtype=residual.dtype, device=residual.device)
        gate_up_out = torch.empty((n,), dtype=residual.dtype, device=residual.device)

    ext = _load_ext()
    ext.fused_add_rmsnorm_gateup_half(
        residual_vec,
        x_vec,
        norm_vec,
        weight,
        bias if bias is not None else torch.empty(0, device=residual.device, dtype=residual.dtype),
        residual_out,
        gate_up_out,
        eps,
        bias is not None,
    )
    return residual_out.view_as(residual), gate_up_out.view(*residual.shape[:-1], n)


def fused_rmsnorm_qkv_half(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    if (
        x.device.type != "cuda"
        or norm_weight.device.type != "cuda"
        or weight.device.type != "cuda"
        or x.dtype != torch.float16
        or norm_weight.dtype != torch.float16
        or weight.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
    ):
        raise RuntimeError("fused_rmsnorm_qkv_half expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    norm_vec = norm_weight.reshape(-1).contiguous()
    n, _k = weight.shape
    out_vec = torch.empty((n,), dtype=x.dtype, device=x.device)

    ext = _load_ext()
    ext.fused_rmsnorm_qkv_half(
        x_vec,
        norm_vec,
        weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        out_vec,
        eps,
        bias is not None,
    )
    return out_vec.view(*x.shape[:-1], n)


def fused_silu_downproj_half(
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: dict | None = None,
) -> torch.Tensor:
    if (
        gate_up.device.type != "cuda"
        or weight.device.type != "cuda"
        or gate_up.dtype != torch.float16
        or weight.dtype != torch.float16
        or gate_up.ndim < 2
        or gate_up.shape[-2] != 1
    ):
        raise RuntimeError("fused_silu_downproj_half expects fp16 CUDA single-token decode tensors")

    gate_up_vec = gate_up.reshape(-1).contiguous()
    n, _k = weight.shape
    if (
        workspace is not None
        and workspace.get("type") == "fused_silu_downproj"
        and tuple(workspace["out"].shape) == (n,)
        and workspace["out"].device == gate_up.device
        and workspace["out"].dtype == gate_up.dtype
    ):
        out_vec = workspace["out"]
    else:
        out_vec = torch.empty((n,), dtype=gate_up.dtype, device=gate_up.device)

    ext = _load_ext()
    ext.fused_silu_downproj_half(
        gate_up_vec,
        weight,
        bias if bias is not None else torch.empty(0, device=gate_up.device, dtype=gate_up.dtype),
        out_vec,
        bias is not None,
    )
    return out_vec.view(*gate_up.shape[:-1], n)


def fused_gateup_silu_downproj_half(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        x.device.type != "cuda"
        or gate_up_weight.device.type != "cuda"
        or down_proj_weight.device.type != "cuda"
        or x.dtype != torch.float16
        or gate_up_weight.dtype != torch.float16
        or down_proj_weight.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
    ):
        raise RuntimeError("fused_gateup_silu_downproj_half expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    n, _k = down_proj_weight.shape
    if (
        workspace is not None
        and workspace.get("type") == "fused_gateup_silu_downproj"
        and tuple(workspace["hidden_out"].shape) == (6144,)
        and tuple(workspace["out"].shape) == (n,)
        and workspace["hidden_out"].device == x.device
        and workspace["out"].device == x.device
        and workspace["hidden_out"].dtype == x.dtype
        and workspace["out"].dtype == x.dtype
    ):
        hidden_out = workspace["hidden_out"]
        out_vec = workspace["out"]
    else:
        hidden_out = torch.empty((6144,), dtype=x.dtype, device=x.device)
        out_vec = torch.empty((n,), dtype=x.dtype, device=x.device)

    ext = _load_ext()
    ext.fused_gateup_silu_downproj_half(
        x_vec,
        gate_up_weight,
        down_proj_weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        hidden_out,
        out_vec,
        bias is not None,
    )
    return hidden_out.view(*x.shape[:-1], 6144), out_vec.view(*x.shape[:-1], n)


def fused_post_attn_mlp_persistent_half(
    residual: torch.Tensor,
    attn_output: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    workspace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        residual.device.type != "cuda"
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
    ):
        raise RuntimeError("fused_post_attn_mlp_persistent_half expects fp16 CUDA single-token decode tensors")

    residual_vec = residual.reshape(-1).contiguous()
    attn_vec = attn_output.reshape(-1).contiguous()
    norm_vec = norm_weight.reshape(-1).contiguous()
    n, _k = down_proj_weight.shape
    if (
        workspace is not None
        and workspace.get("type") == "fused_post_attn_mlp_persistent"
        and tuple(workspace["hidden_out"].shape) == (6144,)
        and tuple(workspace["out"].shape) == (n,)
        and workspace["hidden_out"].device == residual.device
        and workspace["out"].device == residual.device
        and workspace["hidden_out"].dtype == residual.dtype
        and workspace["out"].dtype == residual.dtype
    ):
        hidden_out = workspace["hidden_out"]
        out_vec = workspace["out"]
    else:
        hidden_out = torch.empty((6144,), dtype=residual.dtype, device=residual.device)
        out_vec = torch.empty((n,), dtype=residual.dtype, device=residual.device)

    ext = _load_ext()
    ext.fused_post_attn_mlp_persistent_half(
        residual_vec,
        attn_vec,
        norm_vec,
        gate_up_weight,
        down_proj_weight,
        bias if bias is not None else torch.empty(0, device=residual.device, dtype=residual.dtype),
        hidden_out,
        out_vec,
        eps,
        bias is not None,
    )
    return hidden_out.view(*residual.shape[:-1], 6144), out_vec.view(*residual.shape[:-1], n)


def fused_post_attn_mlp_softsync_half(
    residual: torch.Tensor,
    attn_output: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    workspace: dict | None = None,
    threads_per_block: int = 1024,
    launch_blocks: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        residual.device.type != "cuda"
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
    ):
        raise RuntimeError("fused_post_attn_mlp_softsync_half expects fp16 CUDA single-token decode tensors")

    residual_vec = residual.reshape(-1).contiguous()
    attn_vec = attn_output.reshape(-1).contiguous()
    norm_vec = norm_weight.reshape(-1).contiguous()
    n, _k = down_proj_weight.shape
    if (
        workspace is not None
        and workspace.get("type") == "fused_post_attn_mlp_softsync"
        and tuple(workspace["barrier_count"].shape) == (1,)
        and tuple(workspace["barrier_generation"].shape) == (1,)
        and tuple(workspace["hidden_out"].shape) == (6144,)
        and tuple(workspace["out"].shape) == (n,)
        and workspace["barrier_count"].device == residual.device
        and workspace["barrier_generation"].device == residual.device
        and workspace["hidden_out"].device == residual.device
        and workspace["out"].device == residual.device
        and workspace["barrier_count"].dtype == torch.int32
        and workspace["barrier_generation"].dtype == torch.int32
        and workspace["hidden_out"].dtype == residual.dtype
        and workspace["out"].dtype == residual.dtype
    ):
        barrier_count = workspace["barrier_count"]
        barrier_generation = workspace["barrier_generation"]
        hidden_out = workspace["hidden_out"]
        out_vec = workspace["out"]
    else:
        barrier_count = torch.zeros((1,), dtype=torch.int32, device=residual.device)
        barrier_generation = torch.zeros((1,), dtype=torch.int32, device=residual.device)
        hidden_out = torch.empty((6144,), dtype=residual.dtype, device=residual.device)
        out_vec = torch.empty((n,), dtype=residual.dtype, device=residual.device)

    ext = _load_ext()
    ext.fused_post_attn_mlp_softsync_half(
        residual_vec,
        attn_vec,
        norm_vec,
        gate_up_weight,
        down_proj_weight,
        bias if bias is not None else torch.empty(0, device=residual.device, dtype=residual.dtype),
        barrier_count,
        barrier_generation,
        hidden_out,
        out_vec,
        eps,
        bias is not None,
        threads_per_block,
        0 if launch_blocks is None else int(launch_blocks),
    )
    return hidden_out.view(*residual.shape[:-1], 6144), out_vec.view(*residual.shape[:-1], n)


def fused_o_proj_post_attn_mlp_persistent_half(
    residual: torch.Tensor,
    attn_input: torch.Tensor,
    o_proj_weight: torch.Tensor,
    o_proj_bias: torch.Tensor | None,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_proj_bias: torch.Tensor | None,
    eps: float,
    workspace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        residual.device.type != "cuda"
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
    ):
        raise RuntimeError(
            "fused_o_proj_post_attn_mlp_persistent_half expects fp16 CUDA single-token decode tensors"
        )

    residual_vec = residual.reshape(-1).contiguous()
    attn_vec = attn_input.reshape(-1).contiguous()
    norm_vec = norm_weight.reshape(-1).contiguous()
    n, _k = down_proj_weight.shape
    if (
        workspace is not None
        and workspace.get("type") == "fused_o_proj_post_attn_mlp_persistent"
        and tuple(workspace["hidden_out"].shape) == (6144,)
        and tuple(workspace["out"].shape) == (n,)
        and workspace["hidden_out"].device == residual.device
        and workspace["out"].device == residual.device
        and workspace["hidden_out"].dtype == residual.dtype
        and workspace["out"].dtype == residual.dtype
    ):
        hidden_out = workspace["hidden_out"]
        out_vec = workspace["out"]
    else:
        hidden_out = torch.empty((6144,), dtype=residual.dtype, device=residual.device)
        out_vec = torch.empty((n,), dtype=residual.dtype, device=residual.device)

    ext = _load_ext()
    ext.fused_o_proj_post_attn_mlp_persistent_half(
        residual_vec,
        attn_vec,
        o_proj_weight,
        o_proj_bias if o_proj_bias is not None else torch.empty(0, device=residual.device, dtype=residual.dtype),
        norm_vec,
        gate_up_weight,
        down_proj_weight,
        down_proj_bias if down_proj_bias is not None else torch.empty(0, device=residual.device, dtype=residual.dtype),
        hidden_out,
        out_vec,
        eps,
        o_proj_bias is not None,
        down_proj_bias is not None,
    )
    return hidden_out.view(*residual.shape[:-1], 6144), out_vec.view(*residual.shape[:-1], n)


def fused_o_proj_post_attn_mlp_softsync_half(
    residual: torch.Tensor,
    attn_input: torch.Tensor,
    o_proj_weight: torch.Tensor,
    o_proj_bias: torch.Tensor | None,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_proj_bias: torch.Tensor | None,
    eps: float,
    workspace: dict | None = None,
    gate_up_active_sms: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        residual.device.type != "cuda"
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
    ):
        raise RuntimeError(
            "fused_o_proj_post_attn_mlp_softsync_half expects fp16 CUDA single-token decode tensors"
        )

    residual_vec = residual.reshape(-1).contiguous()
    attn_vec = attn_input.reshape(-1).contiguous()
    norm_vec = norm_weight.reshape(-1).contiguous()
    n, _k = down_proj_weight.shape
    if (
        workspace is not None
        and workspace.get("type") == "fused_o_proj_post_attn_mlp_softsync"
        and tuple(workspace["barrier_count"].shape) == (1,)
        and tuple(workspace["barrier_generation"].shape) == (1,)
        and tuple(workspace["hidden_out"].shape) == (6144,)
        and tuple(workspace["out"].shape) == (n,)
        and workspace["barrier_count"].device == residual.device
        and workspace["barrier_generation"].device == residual.device
        and workspace["hidden_out"].device == residual.device
        and workspace["out"].device == residual.device
        and workspace["barrier_count"].dtype == torch.int32
        and workspace["barrier_generation"].dtype == torch.int32
        and workspace["hidden_out"].dtype == residual.dtype
        and workspace["out"].dtype == residual.dtype
    ):
        barrier_count = workspace["barrier_count"]
        barrier_generation = workspace["barrier_generation"]
        hidden_out = workspace["hidden_out"]
        out_vec = workspace["out"]
    else:
        barrier_count = torch.zeros((1,), dtype=torch.int32, device=residual.device)
        barrier_generation = torch.zeros((1,), dtype=torch.int32, device=residual.device)
        hidden_out = torch.empty((6144,), dtype=residual.dtype, device=residual.device)
        out_vec = torch.empty((n,), dtype=residual.dtype, device=residual.device)

    ext = _load_ext()
    ext.fused_o_proj_post_attn_mlp_softsync_half(
        residual_vec,
        attn_vec,
        o_proj_weight,
        o_proj_bias if o_proj_bias is not None else torch.empty(0, device=residual.device, dtype=residual.dtype),
        norm_vec,
        gate_up_weight,
        down_proj_weight,
        down_proj_bias if down_proj_bias is not None else torch.empty(0, device=residual.device, dtype=residual.dtype),
        barrier_count,
        barrier_generation,
        hidden_out,
        out_vec,
        eps,
        o_proj_bias is not None,
        down_proj_bias is not None,
        gate_up_active_sms,
    )
    return hidden_out.view(*residual.shape[:-1], 6144), out_vec.view(*residual.shape[:-1], n)


def fused_decode_full_layer_softsync_half(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_position: torch.Tensor,
    o_proj_weight: torch.Tensor,
    o_proj_bias: torch.Tensor | None,
    post_attn_norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_proj_bias: torch.Tensor | None,
    eps: float,
    attn_scale: float,
    workspace: dict | None = None,
    launch_blocks: int | None = None,
) -> torch.Tensor:
    if (
        hidden_states.device.type != "cuda"
        or qkv_weight.device.type != "cuda"
        or input_layernorm_weight.device.type != "cuda"
        or q_norm_weight.device.type != "cuda"
        or k_norm_weight.device.type != "cuda"
        or cos.device.type != "cuda"
        or sin.device.type != "cuda"
        or k_cache.device.type != "cuda"
        or v_cache.device.type != "cuda"
        or cache_position.device.type != "cuda"
        or o_proj_weight.device.type != "cuda"
        or post_attn_norm_weight.device.type != "cuda"
        or gate_up_weight.device.type != "cuda"
        or down_proj_weight.device.type != "cuda"
        or hidden_states.dtype != torch.float16
        or qkv_weight.dtype != torch.float16
        or input_layernorm_weight.dtype != torch.float16
        or q_norm_weight.dtype != torch.float16
        or k_norm_weight.dtype != torch.float16
        or cos.dtype != torch.float16
        or sin.dtype != torch.float16
        or k_cache.dtype != torch.float16
        or v_cache.dtype != torch.float16
        or o_proj_weight.dtype != torch.float16
        or post_attn_norm_weight.dtype != torch.float16
        or gate_up_weight.dtype != torch.float16
        or down_proj_weight.dtype != torch.float16
        or hidden_states.ndim < 2
        or hidden_states.shape[-2] != 1
    ):
        raise RuntimeError(
            "fused_decode_full_layer_softsync_half expects fp16 CUDA single-token decode tensors"
        )

    hidden_vec = hidden_states.reshape(-1).contiguous()
    cos_vec = cos.reshape(-1).contiguous()
    sin_vec = sin.reshape(-1).contiguous()
    n = hidden_states.shape[-1]
    attn_split_k = max(
        2,
        int(os.environ.get("CUDA_FUSED_FULL_LAYER_SOFTSYNC_ATTN_QHEAD_SPLITKV_SPLIT_K", "4")),
    )
    num_q_heads = k_cache.shape[1] * 2
    if (
        workspace is not None
        and workspace.get("type") == "fused_decode_full_layer_softsync"
        and tuple(workspace["barrier_count"].shape) == (1,)
        and tuple(workspace["barrier_generation"].shape) == (1,)
        and tuple(workspace["kv_ready_count"].shape) == (1,)
        and tuple(workspace["attn_done_count"].shape) == (1,)
        and tuple(workspace["attn_partial_acc"].shape) == (num_q_heads, attn_split_k, 128)
        and tuple(workspace["attn_partial_stats"].shape) == (num_q_heads, attn_split_k, 2)
        and tuple(workspace["attn_split_done_count"].shape) == (num_q_heads,)
        and tuple(workspace["qkv_out"].shape) == (4096,)
        and tuple(workspace["hidden_out"].shape) == (6144,)
        and tuple(workspace["out"].shape) == (n,)
        and workspace["barrier_count"].device == hidden_states.device
        and workspace["barrier_generation"].device == hidden_states.device
        and workspace["kv_ready_count"].device == hidden_states.device
        and workspace["attn_done_count"].device == hidden_states.device
        and workspace["attn_partial_acc"].device == hidden_states.device
        and workspace["attn_partial_stats"].device == hidden_states.device
        and workspace["attn_split_done_count"].device == hidden_states.device
        and workspace["qkv_out"].device == hidden_states.device
        and workspace["hidden_out"].device == hidden_states.device
        and workspace["out"].device == hidden_states.device
        and workspace["barrier_count"].dtype == torch.int32
        and workspace["barrier_generation"].dtype == torch.int32
        and workspace["kv_ready_count"].dtype == torch.int32
        and workspace["attn_done_count"].dtype == torch.int32
        and workspace["attn_partial_acc"].dtype == torch.float32
        and workspace["attn_partial_stats"].dtype == torch.float32
        and workspace["attn_split_done_count"].dtype == torch.int32
        and workspace["qkv_out"].dtype == hidden_states.dtype
        and workspace["hidden_out"].dtype == hidden_states.dtype
        and workspace["out"].dtype == hidden_states.dtype
    ):
        barrier_count = workspace["barrier_count"]
        barrier_generation = workspace["barrier_generation"]
        kv_ready_count = workspace["kv_ready_count"]
        attn_done_count = workspace["attn_done_count"]
        attn_partial_acc = workspace["attn_partial_acc"]
        attn_partial_stats = workspace["attn_partial_stats"]
        attn_split_done_count = workspace["attn_split_done_count"]
        qkv_out = workspace["qkv_out"]
        hidden_out = workspace["hidden_out"]
        out_vec = workspace["out"]
    else:
        barrier_count = torch.zeros((1,), dtype=torch.int32, device=hidden_states.device)
        barrier_generation = torch.zeros((1,), dtype=torch.int32, device=hidden_states.device)
        kv_ready_count = torch.zeros((1,), dtype=torch.int32, device=hidden_states.device)
        attn_done_count = torch.zeros((1,), dtype=torch.int32, device=hidden_states.device)
        attn_partial_acc = torch.empty(
            (num_q_heads, attn_split_k, 128),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        attn_partial_stats = torch.empty(
            (num_q_heads, attn_split_k, 2),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        attn_split_done_count = torch.zeros((num_q_heads,), dtype=torch.int32, device=hidden_states.device)
        qkv_out = torch.empty((4096,), dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_out = torch.empty((6144,), dtype=hidden_states.dtype, device=hidden_states.device)
        out_vec = torch.empty((n,), dtype=hidden_states.dtype, device=hidden_states.device)

    if launch_blocks is None:
        launch_blocks = int(os.environ.get("CUDA_FUSED_FULL_LAYER_SOFTSYNC_BLOCKS", "0"))

    ext = _load_ext()
    ext.fused_decode_full_layer_softsync_half(
        hidden_vec,
        qkv_weight,
        input_layernorm_weight.reshape(-1).contiguous(),
        q_norm_weight.reshape(-1).contiguous(),
        k_norm_weight.reshape(-1).contiguous(),
        cos_vec,
        sin_vec,
        k_cache,
        v_cache,
        cache_position,
        o_proj_weight,
        o_proj_bias if o_proj_bias is not None else torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype),
        post_attn_norm_weight.reshape(-1).contiguous(),
        gate_up_weight,
        down_proj_weight,
        down_proj_bias if down_proj_bias is not None else torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype),
        barrier_count,
        barrier_generation,
        kv_ready_count,
        attn_done_count,
        attn_partial_acc,
        attn_partial_stats,
        attn_split_done_count,
        qkv_out,
        hidden_out,
        out_vec,
        eps,
        attn_scale,
        o_proj_bias is not None,
        down_proj_bias is not None,
        int(launch_blocks),
    )
    return out_vec.view(*hidden_states.shape[:-1], n)


def fused_gateup_silu_downproj_mega_half(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    workspace: dict | None = None,
) -> torch.Tensor:
    if (
        x.device.type != "cuda"
        or gate_up_weight.device.type != "cuda"
        or down_proj_weight.device.type != "cuda"
        or x.dtype != torch.float16
        or gate_up_weight.dtype != torch.float16
        or down_proj_weight.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
    ):
        raise RuntimeError("fused_gateup_silu_downproj_mega_half expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    n, _k = down_proj_weight.shape
    if (
        workspace is not None
        and workspace.get("type") == "fused_gateup_silu_downproj_mega"
        and tuple(workspace["out"].shape) == (n,)
        and workspace["out"].device == x.device
        and workspace["out"].dtype == x.dtype
    ):
        out_vec = workspace["out"]
    else:
        out_vec = torch.empty((n,), dtype=x.dtype, device=x.device)

    out_vec.zero_()
    ext = _load_ext()
    ext.fused_gateup_silu_downproj_mega_half(
        x_vec,
        gate_up_weight,
        down_proj_weight,
        out_vec,
    )
    return out_vec.view(*x.shape[:-1], n)


def fused_gateup_silu_downproj_mega_partial_half(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    workspace: dict | None = None,
) -> torch.Tensor:
    if (
        x.device.type != "cuda"
        or gate_up_weight.device.type != "cuda"
        or down_proj_weight.device.type != "cuda"
        or x.dtype != torch.float16
        or gate_up_weight.dtype != torch.float16
        or down_proj_weight.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
    ):
        raise RuntimeError("fused_gateup_silu_downproj_mega_partial_half expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    grid_blocks = (6144 + 256 - 1) // 256
    if (
        workspace is not None
        and workspace.get("type") == "fused_gateup_silu_downproj_mega_partial"
        and tuple(workspace["partial_out"].shape) == (grid_blocks, 2048)
        and workspace["partial_out"].device == x.device
        and workspace["partial_out"].dtype == x.dtype
    ):
        partial_out = workspace["partial_out"]
    else:
        partial_out = torch.empty((grid_blocks, 2048), dtype=x.dtype, device=x.device)

    ext = _load_ext()
    ext.fused_gateup_silu_downproj_mega_partial_half(
        x_vec,
        gate_up_weight,
        down_proj_weight,
        partial_out,
    )
    return partial_out


def fused_gateup_silu_downproj_softsync_half(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    workspace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        x.device.type != "cuda"
        or gate_up_weight.device.type != "cuda"
        or down_proj_weight.device.type != "cuda"
        or x.dtype != torch.float16
        or gate_up_weight.dtype != torch.float16
        or down_proj_weight.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
    ):
        raise RuntimeError("fused_gateup_silu_downproj_softsync_half expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    if (
        workspace is not None
        and workspace.get("type") == "fused_gateup_silu_downproj_softsync"
        and tuple(workspace["barrier_count"].shape) == (1,)
        and tuple(workspace["barrier_phase"].shape) == (1,)
        and tuple(workspace["hidden_out"].shape) == (6144,)
        and tuple(workspace["out"].shape) == (2048,)
        and workspace["barrier_count"].device == x.device
        and workspace["barrier_phase"].device == x.device
        and workspace["hidden_out"].device == x.device
        and workspace["out"].device == x.device
    ):
        barrier_count = workspace["barrier_count"]
        barrier_phase = workspace["barrier_phase"]
        hidden_out = workspace["hidden_out"]
        out_vec = workspace["out"]
    else:
        barrier_count = torch.zeros((1,), dtype=torch.int32, device=x.device)
        barrier_phase = torch.zeros((1,), dtype=torch.int32, device=x.device)
        hidden_out = torch.empty((6144,), dtype=x.dtype, device=x.device)
        out_vec = torch.empty((2048,), dtype=x.dtype, device=x.device)

    barrier_count.zero_()
    barrier_phase.zero_()
    ext = _load_ext()
    ext.fused_gateup_silu_downproj_softsync_half(
        x_vec,
        gate_up_weight,
        down_proj_weight,
        barrier_count,
        barrier_phase,
        hidden_out,
        out_vec,
    )
    return hidden_out.view(*x.shape[:-1], 6144), out_vec.view(*x.shape[:-1], 2048)


def fused_gateup_silu_downproj_gridsync_half(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    workspace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        x.device.type != "cuda"
        or gate_up_weight.device.type != "cuda"
        or down_proj_weight.device.type != "cuda"
        or x.dtype != torch.float16
        or gate_up_weight.dtype != torch.float16
        or down_proj_weight.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
    ):
        raise RuntimeError("fused_gateup_silu_downproj_gridsync_half expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    if (
        workspace is not None
        and workspace.get("type") == "fused_gateup_silu_downproj_gridsync"
        and tuple(workspace["hidden_out"].shape) == (6144,)
        and tuple(workspace["out"].shape) == (2048,)
        and workspace["hidden_out"].device == x.device
        and workspace["out"].device == x.device
    ):
        hidden_out = workspace["hidden_out"]
        out_vec = workspace["out"]
    else:
        hidden_out = torch.empty((6144,), dtype=x.dtype, device=x.device)
        out_vec = torch.empty((2048,), dtype=x.dtype, device=x.device)

    ext = _load_ext()
    ext.fused_gateup_silu_downproj_gridsync_half(
        x_vec,
        gate_up_weight,
        down_proj_weight,
        hidden_out,
        out_vec,
    )
    return hidden_out.view(*x.shape[:-1], 6144), out_vec.view(*x.shape[:-1], 2048)


def fused_gateup_silu_downproj_persistent_half(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    workspace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        x.device.type != "cuda"
        or gate_up_weight.device.type != "cuda"
        or down_proj_weight.device.type != "cuda"
        or x.dtype != torch.float16
        or gate_up_weight.dtype != torch.float16
        or down_proj_weight.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
    ):
        raise RuntimeError("fused_gateup_silu_downproj_persistent_half expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    if (
        workspace is not None
        and workspace.get("type") == "fused_gateup_silu_downproj_persistent"
        and tuple(workspace["hidden_out"].shape) == (6144,)
        and tuple(workspace["out"].shape) == (2048,)
        and workspace["hidden_out"].device == x.device
        and workspace["out"].device == x.device
    ):
        hidden_out = workspace["hidden_out"]
        out_vec = workspace["out"]
    else:
        hidden_out = torch.empty((6144,), dtype=x.dtype, device=x.device)
        out_vec = torch.empty((2048,), dtype=x.dtype, device=x.device)

    ext = _load_ext()
    ext.fused_gateup_silu_downproj_persistent_half(
        x_vec,
        gate_up_weight,
        down_proj_weight,
        hidden_out,
        out_vec,
    )
    return hidden_out.view(*x.shape[:-1], 6144), out_vec.view(*x.shape[:-1], 2048)


def decode_attn_persistent_half(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_position: torch.Tensor,
    scale: float,
    workspace: dict | None = None,
) -> torch.Tensor:
    """Persistent cooperative-groups decode attention (1024 threads, 1 block/SM).

    All 32 warps participate in attention via block-internal 16-way split-KV
    with shared memory reduction.  No external partial buffers needed.

    Args:
        q: [1, num_q_heads, 1, head_dim] or [1, num_q_heads, head_dim] query tensor (fp16)
        k_cache: [1, num_kv_heads, max_seq_len, head_dim] key cache (fp16)
        v_cache: [1, num_kv_heads, max_seq_len, head_dim] value cache (fp16)
        cache_position: [1] int64 tensor with current position (seq_len = pos + 1)
        scale: attention scale (1/sqrt(head_dim))
        workspace: optional pre-allocated output buffer
    Returns:
        out: [1, num_q_heads, 1, head_dim] attention output (fp16)
    """
    if (
        q.device.type != "cuda"
        or k_cache.device.type != "cuda"
        or v_cache.device.type != "cuda"
        or q.dtype != torch.float16
        or k_cache.dtype != torch.float16
        or v_cache.dtype != torch.float16
    ):
        raise RuntimeError("decode_attn_persistent_half expects fp16 CUDA tensors")

    # Normalize q shape to [num_q_heads, head_dim]
    if q.ndim == 4:
        num_q_heads, head_dim = q.shape[1], q.shape[3]
        q_2d = q.view(num_q_heads, head_dim).contiguous()
    elif q.ndim == 3:
        num_q_heads, head_dim = q.shape[1], q.shape[2]
        q_2d = q.view(num_q_heads, head_dim).contiguous()
    else:
        raise RuntimeError("q must be 3D or 4D")

    if (
        workspace is not None
        and workspace.get("type") == "decode_attn_persistent"
        and workspace["out"].device == q.device
    ):
        out_2d = workspace["out"]
    else:
        out_2d = torch.empty(
            (num_q_heads, head_dim),
            dtype=torch.float16, device=q.device,
        )
        if workspace is not None:
            workspace["type"] = "decode_attn_persistent"
            workspace["out"] = out_2d

    ext = _load_ext()
    ext.decode_attn_persistent_half(
        q_2d,
        k_cache,
        v_cache,
        cache_position,
        out_2d,
        scale,
    )
    return out_2d.view(1, num_q_heads, 1, head_dim)

__all__ = [
    "fused_add_rmsnorm_gateup_half",
    "fused_rmsnorm_qkv_half",
    "fused_silu_downproj_half",
    "fused_gateup_silu_downproj_half",
    "fused_post_attn_mlp_persistent_half",
    "fused_post_attn_mlp_softsync_half",
    "fused_o_proj_post_attn_mlp_persistent_half",
    "fused_o_proj_post_attn_mlp_softsync_half",
    "fused_decode_full_layer_softsync_half",
    "fused_gateup_silu_downproj_mega_half",
    "fused_gateup_silu_downproj_mega_partial_half",
    "fused_gateup_silu_downproj_softsync_half",
    "fused_gateup_silu_downproj_gridsync_half",
    "fused_gateup_silu_downproj_persistent_half",
    "decode_attn_persistent_half",
]
