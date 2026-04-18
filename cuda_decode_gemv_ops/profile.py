#!/usr/bin/env python3
"""Decode GEMV profiling and trace wrappers."""

from __future__ import annotations

import os

import torch

from .runtime import (
    FULL_LAYER_PROFILE_SLOT_NAMES,
    GRIDSYNC_PROFILE_SLOT_NAMES,
    NOSPLIT_PROFILE_SLOT_NAMES,
    NOSPLIT_VARIANTS,
    NOSPLIT_XWARP_DETAIL_SLOT_NAMES,
    OPROJ_MLP_PROFILE_SLOT_NAMES,
    PERSISTENT_PROFILE_SLOT_NAMES,
    SOFTSYNC_PROFILE_SLOT_NAMES,
    _load_ext,
    _nosplit_grid_blocks,
    _nosplit_profile_kernel_name,
    _nosplit_profile_variant_kernel_name,
    _nosplit_variant_grid_blocks,
)

def fused_decode_full_layer_softsync_profile_half(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
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
            "fused_decode_full_layer_softsync_profile_half expects fp16 CUDA single-token decode tensors"
        )

    hidden_vec = hidden_states.reshape(-1).contiguous()
    cos_vec = cos.reshape(-1).contiguous()
    sin_vec = sin.reshape(-1).contiguous()
    n = hidden_states.shape[-1]

    if launch_blocks is None:
        launch_blocks = int(os.environ.get("CUDA_FUSED_FULL_LAYER_SOFTSYNC_BLOCKS", "0"))

    props = torch.cuda.get_device_properties(hidden_states.device)
    num_sms = props.multi_processor_count
    requested_blocks = int(launch_blocks)
    grid_blocks = max(1, min(num_sms, requested_blocks if requested_blocks > 0 else num_sms))
    num_slots = len(FULL_LAYER_PROFILE_SLOT_NAMES)
    attn_split_k = max(
        2,
        int(os.environ.get("CUDA_FUSED_FULL_LAYER_SOFTSYNC_ATTN_QHEAD_SPLITKV_SPLIT_K", "4")),
    )
    num_q_heads = int(k_cache.shape[1]) * 2

    if (
        workspace is not None
        and workspace.get("type") == "fused_decode_full_layer_softsync_profile"
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
        and workspace.get("profile") is not None
        and workspace["profile"].shape == (grid_blocks, num_slots)
        and workspace["profile"].device == hidden_states.device
        and workspace["profile"].dtype == torch.int64
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
        profile = workspace["profile"]
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
        profile = torch.empty((grid_blocks, num_slots), dtype=torch.int64, device=hidden_states.device)

    ext = _load_ext()
    ext.fused_decode_full_layer_softsync_profile_half(
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
        profile,
        eps,
        attn_scale,
        o_proj_bias is not None,
        down_proj_bias is not None,
        int(launch_blocks),
    )
    full_layer_gqa_ratio = 2
    num_q_heads = int(k_cache.shape[1]) * full_layer_gqa_ratio
    enable_attn_qsplit = os.environ.get(
        "ENABLE_CUDA_FUSED_FULL_LAYER_SOFTSYNC_ATTN_QHEAD_SPLITKV", "1"
    ) == "1"
    attn_qsplit_max_seq = int(
        os.environ.get("CUDA_FUSED_FULL_LAYER_SOFTSYNC_ATTN_QHEAD_SPLITKV_MAX_SEQ_LEN", "4096")
    )
    use_attn_qsplit = (
        enable_attn_qsplit
        and grid_blocks >= num_q_heads * attn_split_k
        and (attn_qsplit_max_seq <= 0 or int(k_cache.shape[2]) <= attn_qsplit_max_seq)
    )
    attn_active_blocks = min(
        grid_blocks,
        num_q_heads * attn_split_k if use_attn_qsplit else num_q_heads,
    )
    o_proj_idle_only = os.environ.get(
        "ENABLE_CUDA_FUSED_FULL_LAYER_SOFTSYNC_O_PROJ_IDLE_ONLY", "0"
    ) == "1"
    partial_attn_flags = os.environ.get(
        "ENABLE_CUDA_FUSED_FULL_LAYER_SOFTSYNC_PARTIAL_ATTN_FLAGS", "0"
    ) == "1"
    idle_blocks = max(0, grid_blocks - attn_active_blocks)
    o_proj_active_blocks = idle_blocks if o_proj_idle_only else grid_blocks

    meta = {
        "kernel_name": "fused_decode_full_layer_softsync_profile",
        "grid_blocks": grid_blocks,
        "num_q_heads": num_q_heads,
        "attn_qhead_splitkv": use_attn_qsplit,
        "attn_split_k": attn_split_k if use_attn_qsplit else 1,
        "attn_active_blocks": attn_active_blocks,
        "idle_blocks": idle_blocks,
        "partial_attn_flags": partial_attn_flags,
        "o_proj_idle_only": o_proj_idle_only,
        "o_proj_active_blocks": o_proj_active_blocks,
        "rows_per_wave": grid_blocks * 16,
        "downproj_row_passes": (2048 + grid_blocks * 16 - 1) // (grid_blocks * 16),
        "slot_names": list(FULL_LAYER_PROFILE_SLOT_NAMES),
    }
    return (
        hidden_out.view(*hidden_states.shape[:-1], 6144),
        out_vec.view(*hidden_states.shape[:-1], n),
        profile,
        meta,
    )


def fused_o_proj_post_attn_mlp_persistent_profile_half(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
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
            "fused_o_proj_post_attn_mlp_persistent_profile_half expects fp16 CUDA single-token decode tensors"
        )

    residual_vec = residual.reshape(-1).contiguous()
    attn_vec = attn_input.reshape(-1).contiguous()
    norm_vec = norm_weight.reshape(-1).contiguous()
    n, _k = down_proj_weight.shape
    props = torch.cuda.get_device_properties(residual.device)
    grid_blocks = props.multi_processor_count
    num_slots = len(OPROJ_MLP_PROFILE_SLOT_NAMES)

    if (
        workspace is not None
        and workspace.get("type") == "oproj_mlp_persistent_profile"
        and tuple(workspace["hidden_out"].shape) == (6144,)
        and tuple(workspace["out"].shape) == (n,)
        and workspace["hidden_out"].device == residual.device
        and workspace["out"].device == residual.device
        and workspace.get("profile") is not None
        and workspace["profile"].shape == (grid_blocks, num_slots)
        and workspace["profile"].device == residual.device
        and workspace["profile"].dtype == torch.int64
    ):
        hidden_out = workspace["hidden_out"]
        out_vec = workspace["out"]
        profile = workspace["profile"]
    else:
        hidden_out = torch.empty((6144,), dtype=residual.dtype, device=residual.device)
        out_vec = torch.empty((n,), dtype=residual.dtype, device=residual.device)
        profile = torch.empty(
            (grid_blocks, num_slots),
            dtype=torch.int64,
            device=residual.device,
        )

    ext = _load_ext()
    ext.fused_o_proj_post_attn_mlp_persistent_profile_half(
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
        profile,
        eps,
        o_proj_bias is not None,
        down_proj_bias is not None,
        gate_up_active_sms,
    )
    meta = {
        "kernel_name": "fused_o_proj_post_attn_mlp_persistent_profile",
        "grid_blocks": grid_blocks,
        "slot_names": list(OPROJ_MLP_PROFILE_SLOT_NAMES),
    }
    return hidden_out.view(*residual.shape[:-1], 6144), out_vec.view(*residual.shape[:-1], n), profile, meta


def fused_o_proj_post_attn_mlp_softsync_profile_half(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
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
            "fused_o_proj_post_attn_mlp_softsync_profile_half expects fp16 CUDA single-token decode tensors"
        )

    residual_vec = residual.reshape(-1).contiguous()
    attn_vec = attn_input.reshape(-1).contiguous()
    norm_vec = norm_weight.reshape(-1).contiguous()
    n, _k = down_proj_weight.shape
    props = torch.cuda.get_device_properties(residual.device)
    grid_blocks = props.multi_processor_count
    num_slots = len(OPROJ_MLP_PROFILE_SLOT_NAMES)

    if (
        workspace is not None
        and workspace.get("type") == "oproj_mlp_softsync_profile"
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
        and workspace.get("profile") is not None
        and workspace["profile"].shape == (grid_blocks, num_slots)
        and workspace["profile"].device == residual.device
        and workspace["profile"].dtype == torch.int64
    ):
        barrier_count = workspace["barrier_count"]
        barrier_generation = workspace["barrier_generation"]
        hidden_out = workspace["hidden_out"]
        out_vec = workspace["out"]
        profile = workspace["profile"]
    else:
        barrier_count = torch.zeros((1,), dtype=torch.int32, device=residual.device)
        barrier_generation = torch.zeros((1,), dtype=torch.int32, device=residual.device)
        hidden_out = torch.empty((6144,), dtype=residual.dtype, device=residual.device)
        out_vec = torch.empty((n,), dtype=residual.dtype, device=residual.device)
        profile = torch.empty(
            (grid_blocks, num_slots),
            dtype=torch.int64,
            device=residual.device,
        )

    ext = _load_ext()
    ext.fused_o_proj_post_attn_mlp_softsync_profile_half(
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
        profile,
        eps,
        o_proj_bias is not None,
        down_proj_bias is not None,
        gate_up_active_sms,
    )
    meta = {
        "kernel_name": "fused_o_proj_post_attn_mlp_softsync_profile",
        "grid_blocks": grid_blocks,
        "slot_names": list(OPROJ_MLP_PROFILE_SLOT_NAMES),
    }
    return hidden_out.view(*residual.shape[:-1], 6144), out_vec.view(*residual.shape[:-1], n), profile, meta


def fused_gateup_silu_downproj_softsync_profile_half(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    workspace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Run softsync fused gateup+silu+downproj with per-block phase profiling.

    Returns (hidden_out, out, profile, meta).
    profile shape: [grid_blocks, len(SOFTSYNC_PROFILE_SLOT_NAMES)] (int64 timestamps in ns).
    """
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
        raise RuntimeError("softsync_profile_half expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    grid_blocks = 2048 // 8  # 256

    if (
        workspace is not None
        and workspace.get("type") == "softsync_profile"
        and tuple(workspace["barrier_count"].shape) == (1,)
        and tuple(workspace["barrier_phase"].shape) == (1,)
        and tuple(workspace["hidden_out"].shape) == (6144,)
        and tuple(workspace["out"].shape) == (2048,)
        and workspace["barrier_count"].device == x.device
        and workspace["barrier_phase"].device == x.device
        and workspace["hidden_out"].device == x.device
        and workspace["out"].device == x.device
        and workspace.get("profile") is not None
        and workspace["profile"].shape == (grid_blocks, len(SOFTSYNC_PROFILE_SLOT_NAMES))
        and workspace["profile"].device == x.device
        and workspace["profile"].dtype == torch.int64
    ):
        barrier_count = workspace["barrier_count"]
        barrier_phase = workspace["barrier_phase"]
        hidden_out = workspace["hidden_out"]
        out_vec = workspace["out"]
        profile = workspace["profile"]
    else:
        barrier_count = torch.zeros((1,), dtype=torch.int32, device=x.device)
        barrier_phase = torch.zeros((1,), dtype=torch.int32, device=x.device)
        hidden_out = torch.empty((6144,), dtype=x.dtype, device=x.device)
        out_vec = torch.empty((2048,), dtype=x.dtype, device=x.device)
        profile = torch.empty(
            (grid_blocks, len(SOFTSYNC_PROFILE_SLOT_NAMES)),
            dtype=torch.int64,
            device=x.device,
        )

    barrier_count.zero_()
    barrier_phase.zero_()
    ext = _load_ext()
    ext.fused_gateup_silu_downproj_softsync_profile_half(
        x_vec,
        gate_up_weight,
        down_proj_weight,
        barrier_count,
        barrier_phase,
        hidden_out,
        out_vec,
        profile,
    )
    meta = {
        "kernel_name": "fused_gateup_silu_downproj_softsync_profile",
        "grid_blocks": grid_blocks,
        "slot_names": list(SOFTSYNC_PROFILE_SLOT_NAMES),
    }
    return hidden_out.view(*x.shape[:-1], 6144), out_vec.view(*x.shape[:-1], 2048), profile, meta


def fused_gateup_silu_downproj_gridsync_profile_half(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    workspace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Run gridsync fused gateup+silu+downproj with per-block phase profiling.

    Returns (hidden_out, out, profile, meta).
    profile shape: [grid_blocks, len(GRIDSYNC_PROFILE_SLOT_NAMES)] (int64 timestamps in ns).
    """
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
        raise RuntimeError("gridsync_profile_half expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    grid_blocks = 2048 // 8  # 256

    if (
        workspace is not None
        and workspace.get("type") == "gridsync_profile"
        and tuple(workspace["hidden_out"].shape) == (6144,)
        and tuple(workspace["out"].shape) == (2048,)
        and workspace["hidden_out"].device == x.device
        and workspace["out"].device == x.device
        and workspace.get("profile") is not None
        and workspace["profile"].shape == (grid_blocks, len(GRIDSYNC_PROFILE_SLOT_NAMES))
        and workspace["profile"].device == x.device
        and workspace["profile"].dtype == torch.int64
    ):
        hidden_out = workspace["hidden_out"]
        out_vec = workspace["out"]
        profile = workspace["profile"]
    else:
        hidden_out = torch.empty((6144,), dtype=x.dtype, device=x.device)
        out_vec = torch.empty((2048,), dtype=x.dtype, device=x.device)
        profile = torch.empty(
            (grid_blocks, len(GRIDSYNC_PROFILE_SLOT_NAMES)),
            dtype=torch.int64,
            device=x.device,
        )

    ext = _load_ext()
    ext.fused_gateup_silu_downproj_gridsync_profile_half(
        x_vec,
        gate_up_weight,
        down_proj_weight,
        hidden_out,
        out_vec,
        profile,
    )
    meta = {
        "kernel_name": "fused_gateup_silu_downproj_gridsync_profile",
        "grid_blocks": grid_blocks,
        "slot_names": list(GRIDSYNC_PROFILE_SLOT_NAMES),
    }
    return hidden_out.view(*x.shape[:-1], 6144), out_vec.view(*x.shape[:-1], 2048), profile, meta


def fused_gateup_silu_downproj_persistent_profile_half(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    workspace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
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
        raise RuntimeError("persistent_profile_half expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    props = torch.cuda.get_device_properties(x.device)
    grid_blocks = props.multi_processor_count

    if (
        workspace is not None
        and workspace.get("type") == "persistent_profile"
        and tuple(workspace["hidden_out"].shape) == (6144,)
        and tuple(workspace["out"].shape) == (2048,)
        and workspace["hidden_out"].device == x.device
        and workspace["out"].device == x.device
        and workspace.get("profile") is not None
        and workspace["profile"].shape == (grid_blocks, len(PERSISTENT_PROFILE_SLOT_NAMES))
        and workspace["profile"].device == x.device
        and workspace["profile"].dtype == torch.int64
    ):
        hidden_out = workspace["hidden_out"]
        out_vec = workspace["out"]
        profile = workspace["profile"]
    else:
        hidden_out = torch.empty((6144,), dtype=x.dtype, device=x.device)
        out_vec = torch.empty((2048,), dtype=x.dtype, device=x.device)
        profile = torch.empty(
            (grid_blocks, len(PERSISTENT_PROFILE_SLOT_NAMES)),
            dtype=torch.int64,
            device=x.device,
        )

    ext = _load_ext()
    ext.fused_gateup_silu_downproj_persistent_profile_half(
        x_vec,
        gate_up_weight,
        down_proj_weight,
        hidden_out,
        out_vec,
        profile,
    )
    meta = {
        "kernel_name": "fused_gateup_silu_downproj_persistent_profile",
        "grid_blocks": grid_blocks,
        "slot_names": list(PERSISTENT_PROFILE_SLOT_NAMES),
    }
    return hidden_out.view(*x.shape[:-1], 6144), out_vec.view(*x.shape[:-1], 2048), profile, meta


def nosplit_matvec_half_profile(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if (
        x.device.type != "cuda"
        or weight.device.type != "cuda"
        or x.dtype != torch.float16
        or weight.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
    ):
        raise RuntimeError("nosplit_matvec_half_profile expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    n, k = weight.shape
    grid_blocks = _nosplit_grid_blocks(n, k)

    if (
        workspace is not None
        and workspace.get("type") == "nosplit_profile"
        and tuple(workspace["out"].shape) == (n,)
        and tuple(workspace["profile"].shape) == (grid_blocks, len(NOSPLIT_PROFILE_SLOT_NAMES))
        and workspace["out"].device == x.device
        and workspace["profile"].device == x.device
        and workspace["out"].dtype == x.dtype
        and workspace["profile"].dtype == torch.int64
    ):
        out_vec = workspace["out"]
        profile = workspace["profile"]
    else:
        out_vec = torch.empty((n,), dtype=x.dtype, device=x.device)
        profile = torch.empty(
            (grid_blocks, len(NOSPLIT_PROFILE_SLOT_NAMES)),
            dtype=torch.int64,
            device=x.device,
        )

    ext = _load_ext()
    ext.nosplit_matvec_half_profile(
        x_vec,
        weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        out_vec,
        profile,
        bias is not None,
    )
    meta = {
        "kernel_name": _nosplit_profile_kernel_name(n, k),
        "grid_blocks": grid_blocks,
        "slot_names": list(NOSPLIT_PROFILE_SLOT_NAMES),
    }
    return out_vec.view(*x.shape[:-1], n), profile, meta


def nosplit_matvec_half_profile_variant(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    variant: str,
    workspace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if variant not in NOSPLIT_VARIANTS:
        raise ValueError(f"unknown nosplit variant: {variant}")
    if (
        x.device.type != "cuda"
        or weight.device.type != "cuda"
        or x.dtype != torch.float16
        or weight.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
    ):
        raise RuntimeError("nosplit_matvec_half_profile_variant expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    n, k = weight.shape
    grid_blocks = _nosplit_variant_grid_blocks(n, k, variant)

    if (
        workspace is not None
        and workspace.get("type") == f"nosplit_profile_variant:{variant}"
        and tuple(workspace["out"].shape) == (n,)
        and tuple(workspace["profile"].shape) == (grid_blocks, len(NOSPLIT_PROFILE_SLOT_NAMES))
        and workspace["out"].device == x.device
        and workspace["profile"].device == x.device
        and workspace["out"].dtype == x.dtype
        and workspace["profile"].dtype == torch.int64
    ):
        out_vec = workspace["out"]
        profile = workspace["profile"]
    else:
        out_vec = torch.empty((n,), dtype=x.dtype, device=x.device)
        profile = torch.empty(
            (grid_blocks, len(NOSPLIT_PROFILE_SLOT_NAMES)),
            dtype=torch.int64,
            device=x.device,
        )

    ext = _load_ext()
    ext.nosplit_matvec_half_profile_variant(
        x_vec,
        weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        out_vec,
        profile,
        bias is not None,
        NOSPLIT_VARIANTS[variant],
    )
    meta = {
        "kernel_name": _nosplit_profile_variant_kernel_name(variant),
        "grid_blocks": grid_blocks,
        "slot_names": list(NOSPLIT_PROFILE_SLOT_NAMES),
        "variant": variant,
    }
    return out_vec.view(*x.shape[:-1], n), profile, meta


def nosplit_matvec_half_profile_xwarp_detail(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if (
        x.device.type != "cuda"
        or weight.device.type != "cuda"
        or x.dtype != torch.float16
        or weight.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
    ):
        raise RuntimeError("nosplit_matvec_half_profile_xwarp_detail expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    n, k = weight.shape
    if k != 6144:
        raise RuntimeError("nosplit_matvec_half_profile_xwarp_detail currently supports only k=6144")
    grid_blocks = (n + 7) // 8
    warp_groups = 8

    if (
        workspace is not None
        and workspace.get("type") == "nosplit_profile_xwarp_detail"
        and tuple(workspace["out"].shape) == (n,)
        and tuple(workspace["trace"].shape) == (grid_blocks, warp_groups, len(NOSPLIT_XWARP_DETAIL_SLOT_NAMES))
        and workspace["out"].device == x.device
        and workspace["trace"].device == x.device
        and workspace["out"].dtype == x.dtype
        and workspace["trace"].dtype == torch.int64
    ):
        out_vec = workspace["out"]
        trace = workspace["trace"]
    else:
        out_vec = torch.empty((n,), dtype=x.dtype, device=x.device)
        trace = torch.empty(
            (grid_blocks, warp_groups, len(NOSPLIT_XWARP_DETAIL_SLOT_NAMES)),
            dtype=torch.int64,
            device=x.device,
        )

    ext = _load_ext()
    ext.nosplit_matvec_half_profile_xwarp_detail(
        x_vec,
        weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        out_vec,
        trace,
        bias is not None,
    )
    meta = {
        "kernel_name": "nosplit_matvec_k6144_n8k1024_xwarp_w_async_trace_fwd_kernel",
        "grid_blocks": grid_blocks,
        "warp_groups": warp_groups,
        "slot_names": list(NOSPLIT_XWARP_DETAIL_SLOT_NAMES),
        "variant": "k6144_n8k1024_xwarp_w_async",
    }
    return out_vec.view(*x.shape[:-1], n), trace, meta


def nosplit_xwarp_wait_sync_calibration(
    grid_blocks: int,
    workspace: dict | None = None,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, dict]:
    if grid_blocks <= 0:
        raise RuntimeError("nosplit_xwarp_wait_sync_calibration expects grid_blocks > 0")
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("nosplit_xwarp_wait_sync_calibration expects a CUDA device")
    warp_groups = 8

    if (
        workspace is not None
        and workspace.get("type") == "nosplit_xwarp_wait_sync_calibration"
        and tuple(workspace["trace"].shape) == (grid_blocks, warp_groups, len(NOSPLIT_XWARP_DETAIL_SLOT_NAMES))
        and workspace["trace"].device == device
        and workspace["trace"].dtype == torch.int64
    ):
        trace = workspace["trace"]
    else:
        trace = torch.empty(
            (grid_blocks, warp_groups, len(NOSPLIT_XWARP_DETAIL_SLOT_NAMES)),
            dtype=torch.int64,
            device=device,
        )

    ext = _load_ext()
    ext.nosplit_xwarp_wait_sync_calibration(trace)
    meta = {
        "kernel_name": "nosplit_xwarp_wait_sync_calibration_kernel",
        "grid_blocks": grid_blocks,
        "warp_groups": warp_groups,
        "slot_names": list(NOSPLIT_XWARP_DETAIL_SLOT_NAMES),
    }
    return trace, meta


def nosplit_matvec_half_profile_xwarp_detail_with_smid(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    if (
        x.device.type != "cuda"
        or weight.device.type != "cuda"
        or x.dtype != torch.float16
        or weight.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
    ):
        raise RuntimeError("nosplit_matvec_half_profile_xwarp_detail_with_smid expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    n, k = weight.shape
    if k != 6144:
        raise RuntimeError("nosplit_matvec_half_profile_xwarp_detail_with_smid currently supports only k=6144")
    grid_blocks = (n + 7) // 8
    warp_groups = 8

    if (
        workspace is not None
        and workspace.get("type") == "nosplit_profile_xwarp_detail_with_smid"
        and tuple(workspace["out"].shape) == (n,)
        and tuple(workspace["trace"].shape) == (grid_blocks, warp_groups, len(NOSPLIT_XWARP_DETAIL_SLOT_NAMES))
        and tuple(workspace["smid"].shape) == (grid_blocks, warp_groups)
        and workspace["out"].device == x.device
        and workspace["trace"].device == x.device
        and workspace["smid"].device == x.device
        and workspace["out"].dtype == x.dtype
        and workspace["trace"].dtype == torch.int64
        and workspace["smid"].dtype == torch.int32
    ):
        out_vec = workspace["out"]
        trace = workspace["trace"]
        smid = workspace["smid"]
    else:
        out_vec = torch.empty((n,), dtype=x.dtype, device=x.device)
        trace = torch.empty(
            (grid_blocks, warp_groups, len(NOSPLIT_XWARP_DETAIL_SLOT_NAMES)),
            dtype=torch.int64,
            device=x.device,
        )
        smid = torch.empty(
            (grid_blocks, warp_groups),
            dtype=torch.int32,
            device=x.device,
        )

    ext = _load_ext()
    ext.nosplit_matvec_half_profile_xwarp_detail_with_smid(
        x_vec,
        weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        out_vec,
        trace,
        smid,
        bias is not None,
    )
    meta = {
        "kernel_name": "nosplit_matvec_k6144_n8k1024_xwarp_w_async_trace_fwd_kernel",
        "grid_blocks": grid_blocks,
        "warp_groups": warp_groups,
        "slot_names": list(NOSPLIT_XWARP_DETAIL_SLOT_NAMES),
        "variant": "k6144_n8k1024_xwarp_w_async",
    }
    return out_vec.view(*x.shape[:-1], n), trace, smid, meta


def nosplit_matvec_half_profile_xwarp_sms2_cta_trace(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    if (
        x.device.type != "cuda"
        or weight.device.type != "cuda"
        or x.dtype != torch.float16
        or weight.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
    ):
        raise RuntimeError("nosplit_matvec_half_profile_xwarp_sms2_cta_trace expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    n, k = weight.shape
    if k != 6144:
        raise RuntimeError("nosplit_matvec_half_profile_xwarp_sms2_cta_trace currently supports only k=6144")

    props = torch.cuda.get_device_properties(x.device)
    full_grid_blocks = (n + 7) // 8
    launch_grid_blocks = min(full_grid_blocks, max(1, props.multi_processor_count * 2))
    max_iters = (full_grid_blocks + launch_grid_blocks - 1) // launch_grid_blocks

    if (
        workspace is not None
        and workspace.get("type") == "nosplit_profile_xwarp_sms2_cta_trace"
        and tuple(workspace["out"].shape) == (n,)
        and tuple(workspace["trace"].shape) == (launch_grid_blocks, max_iters, 2)
        and tuple(workspace["smid"].shape) == (launch_grid_blocks,)
        and tuple(workspace["row_base"].shape) == (launch_grid_blocks, max_iters)
        and workspace["out"].device == x.device
        and workspace["trace"].device == x.device
        and workspace["smid"].device == x.device
        and workspace["row_base"].device == x.device
        and workspace["out"].dtype == x.dtype
        and workspace["trace"].dtype == torch.int64
        and workspace["smid"].dtype == torch.int32
        and workspace["row_base"].dtype == torch.int32
    ):
        out_vec = workspace["out"]
        trace = workspace["trace"]
        smid = workspace["smid"]
        row_base = workspace["row_base"]
    else:
        out_vec = torch.empty((n,), dtype=x.dtype, device=x.device)
        trace = torch.empty((launch_grid_blocks, max_iters, 2), dtype=torch.int64, device=x.device)
        smid = torch.empty((launch_grid_blocks,), dtype=torch.int32, device=x.device)
        row_base = torch.empty((launch_grid_blocks, max_iters), dtype=torch.int32, device=x.device)

    ext = _load_ext()
    ext.nosplit_matvec_half_profile_xwarp_sms2_cta_trace(
        x_vec,
        weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        out_vec,
        trace,
        smid,
        row_base,
        bias is not None,
    )
    meta = {
        "kernel_name": "nosplit_matvec_k6144_n8k1024_xwarp_w_async_sms2_cta_trace_fwd_kernel",
        "launch_grid_blocks": launch_grid_blocks,
        "full_grid_blocks": full_grid_blocks,
        "max_iters": max_iters,
        "sms": int(props.multi_processor_count),
        "slot_names": ["iter_start", "iter_end"],
        "variant": "k6144_n8k1024_xwarp_w_async_sms2",
    }
    return out_vec.view(*x.shape[:-1], n), trace, smid, row_base, meta


def nosplit_matvec_half_profile_xwarp_sms2_balanced_cta_trace(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    if (
        x.device.type != "cuda"
        or weight.device.type != "cuda"
        or x.dtype != torch.float16
        or weight.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
    ):
        raise RuntimeError("nosplit_matvec_half_profile_xwarp_sms2_balanced_cta_trace expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    n, k = weight.shape
    if k != 6144:
        raise RuntimeError("nosplit_matvec_half_profile_xwarp_sms2_balanced_cta_trace currently supports only k=6144")

    props = torch.cuda.get_device_properties(x.device)
    full_grid_blocks = (n + 7) // 8
    launch_grid_blocks = min(full_grid_blocks, max(1, props.multi_processor_count * 2))
    max_rows_per_block = (n + launch_grid_blocks - 1) // launch_grid_blocks
    max_iters = (max_rows_per_block + 7) // 8

    if (
        workspace is not None
        and workspace.get("type") == "nosplit_profile_xwarp_sms2_balanced_cta_trace"
        and tuple(workspace["out"].shape) == (n,)
        and tuple(workspace["trace"].shape) == (launch_grid_blocks, max_iters, 2)
        and tuple(workspace["smid"].shape) == (launch_grid_blocks,)
        and tuple(workspace["row_base"].shape) == (launch_grid_blocks, max_iters)
        and workspace["out"].device == x.device
        and workspace["trace"].device == x.device
        and workspace["smid"].device == x.device
        and workspace["row_base"].device == x.device
        and workspace["out"].dtype == x.dtype
        and workspace["trace"].dtype == torch.int64
        and workspace["smid"].dtype == torch.int32
        and workspace["row_base"].dtype == torch.int32
    ):
        out_vec = workspace["out"]
        trace = workspace["trace"]
        smid = workspace["smid"]
        row_base = workspace["row_base"]
    else:
        out_vec = torch.empty((n,), dtype=x.dtype, device=x.device)
        trace = torch.empty((launch_grid_blocks, max_iters, 2), dtype=torch.int64, device=x.device)
        smid = torch.empty((launch_grid_blocks,), dtype=torch.int32, device=x.device)
        row_base = torch.empty((launch_grid_blocks, max_iters), dtype=torch.int32, device=x.device)

    ext = _load_ext()
    ext.nosplit_matvec_half_profile_xwarp_sms2_balanced_cta_trace(
        x_vec,
        weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        out_vec,
        trace,
        smid,
        row_base,
        bias is not None,
    )
    meta = {
        "kernel_name": "nosplit_matvec_k6144_n8k1024_xwarp_w_async_sms2_balanced_cta_trace_fwd_kernel",
        "launch_grid_blocks": launch_grid_blocks,
        "full_grid_blocks": full_grid_blocks,
        "max_iters": max_iters,
        "sms": int(props.multi_processor_count),
        "slot_names": ["iter_start", "iter_end"],
        "variant": "k6144_n8k1024_xwarp_w_async_sms2_balanced",
    }
    return out_vec.view(*x.shape[:-1], n), trace, smid, row_base, meta

__all__ = [
    "FULL_LAYER_PROFILE_SLOT_NAMES",
    "GRIDSYNC_PROFILE_SLOT_NAMES",
    "NOSPLIT_PROFILE_SLOT_NAMES",
    "NOSPLIT_XWARP_DETAIL_SLOT_NAMES",
    "OPROJ_MLP_PROFILE_SLOT_NAMES",
    "PERSISTENT_PROFILE_SLOT_NAMES",
    "SOFTSYNC_PROFILE_SLOT_NAMES",
    "fused_decode_full_layer_softsync_profile_half",
    "fused_o_proj_post_attn_mlp_persistent_profile_half",
    "fused_o_proj_post_attn_mlp_softsync_profile_half",
    "fused_gateup_silu_downproj_softsync_profile_half",
    "fused_gateup_silu_downproj_gridsync_profile_half",
    "fused_gateup_silu_downproj_persistent_profile_half",
    "nosplit_matvec_half_profile",
    "nosplit_matvec_half_profile_variant",
    "nosplit_matvec_half_profile_xwarp_detail",
    "nosplit_xwarp_wait_sync_calibration",
    "nosplit_matvec_half_profile_xwarp_detail_with_smid",
    "nosplit_matvec_half_profile_xwarp_sms2_cta_trace",
    "nosplit_matvec_half_profile_xwarp_sms2_balanced_cta_trace",
]
