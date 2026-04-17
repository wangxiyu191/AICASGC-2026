#!/usr/bin/env python3
"""Decode GEMV matvec, routing, and probe wrappers."""

from __future__ import annotations

import torch

from .runtime import DUALCOPY_ROUTE_MODES, NOSPLIT_VARIANTS, _load_ext

def splitk_matvec(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    split_k: int,
    workspace: dict | None = None,
) -> torch.Tensor:
    if (
        x.device.type != "cuda"
        or weight.device.type != "cuda"
        or x.ndim < 2
        or x.shape[-2] != 1
        or split_k <= 0
    ):
        raise RuntimeError("splitk_matvec expects CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    n, _k = weight.shape
    if (
        workspace is not None
        and workspace.get("split_k") == split_k
        and tuple(workspace["scratch"].shape) == (n,)
        and workspace["scratch"].device == x.device
        and workspace["out"].dtype == x.dtype
    ):
        scratch = workspace["scratch"]
        out_vec = workspace["out"]
    else:
        scratch = torch.empty((n,), dtype=torch.float32, device=x.device)
        out_vec = torch.empty((n,), dtype=x.dtype, device=x.device)

    ext = _load_ext()
    ext.splitk_matvec(
        x_vec,
        weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        split_k,
        scratch,
        out_vec,
        bias is not None,
    )
    return out_vec.view(*x.shape[:-1], n)


def splitk_matvec_row4(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    split_k: int,
    workspace: dict | None = None,
) -> torch.Tensor:
    if (
        x.device.type != "cuda"
        or weight.device.type != "cuda"
        or x.ndim < 2
        or x.shape[-2] != 1
        or split_k <= 0
    ):
        raise RuntimeError("splitk_matvec_row4 expects CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    n, _k = weight.shape
    if (
        workspace is not None
        and workspace.get("split_k") == split_k
        and tuple(workspace["scratch"].shape) == (n,)
        and workspace["scratch"].device == x.device
        and workspace["out"].dtype == x.dtype
    ):
        scratch = workspace["scratch"]
        out_vec = workspace["out"]
    else:
        scratch = torch.empty((n,), dtype=torch.float32, device=x.device)
        out_vec = torch.empty((n,), dtype=x.dtype, device=x.device)

    ext = _load_ext()
    ext.splitk_matvec_row4(
        x_vec,
        weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        split_k,
        scratch,
        out_vec,
        bias is not None,
    )
    return out_vec.view(*x.shape[:-1], n)


def splitk_matvec_row8_shared(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    split_k: int,
    workspace: dict | None = None,
) -> torch.Tensor:
    if (
        x.device.type != "cuda"
        or weight.device.type != "cuda"
        or x.ndim < 2
        or x.shape[-2] != 1
        or split_k <= 0
    ):
        raise RuntimeError("splitk_matvec_row8_shared expects CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    n, _k = weight.shape
    if (
        workspace is not None
        and workspace.get("split_k") == split_k
        and tuple(workspace["scratch"].shape) == (n,)
        and workspace["scratch"].device == x.device
        and workspace["out"].dtype == x.dtype
    ):
        scratch = workspace["scratch"]
        out_vec = workspace["out"]
    else:
        scratch = torch.empty((n,), dtype=torch.float32, device=x.device)
        out_vec = torch.empty((n,), dtype=x.dtype, device=x.device)

    ext = _load_ext()
    ext.splitk_matvec_row8_shared(
        x_vec,
        weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        split_k,
        scratch,
        out_vec,
        bias is not None,
    )
    return out_vec.view(*x.shape[:-1], n)


def splitk_matvec_row4_shared_vec_half(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    split_k: int,
    workspace: dict | None = None,
) -> torch.Tensor:
    if (
        x.device.type != "cuda"
        or weight.device.type != "cuda"
        or x.dtype != torch.float16
        or weight.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
        or split_k <= 0
    ):
        raise RuntimeError("splitk_matvec_row4_shared_vec_half expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    n, _k = weight.shape
    if (
        workspace is not None
        and workspace.get("split_k") == split_k
        and tuple(workspace["scratch"].shape) == (n,)
        and workspace["scratch"].device == x.device
        and workspace["out"].dtype == x.dtype
    ):
        scratch = workspace["scratch"]
        out_vec = workspace["out"]
    else:
        scratch = torch.empty((n,), dtype=torch.float32, device=x.device)
        out_vec = torch.empty((n,), dtype=x.dtype, device=x.device)

    ext = _load_ext()
    ext.splitk_matvec_row4_shared_vec_half(
        x_vec,
        weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        split_k,
        scratch,
        out_vec,
        bias is not None,
    )
    return out_vec.view(*x.shape[:-1], n)


def splitk_matvec_row4_shared_vec_half_v2(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    split_k: int,
    workspace: dict | None = None,
) -> torch.Tensor:
    if (
        x.device.type != "cuda"
        or weight.device.type != "cuda"
        or x.dtype != torch.float16
        or weight.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
        or split_k <= 0
    ):
        raise RuntimeError("splitk_matvec_row4_shared_vec_half_v2 expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    n, _k = weight.shape
    if (
        workspace is not None
        and workspace.get("split_k") == split_k
        and tuple(workspace["scratch"].shape) == (n,)
        and workspace["scratch"].device == x.device
        and workspace["out"].dtype == x.dtype
    ):
        scratch = workspace["scratch"]
        out_vec = workspace["out"]
    else:
        scratch = torch.empty((n,), dtype=torch.float32, device=x.device)
        out_vec = torch.empty((n,), dtype=x.dtype, device=x.device)

    ext = _load_ext()
    ext.splitk_matvec_row4_shared_vec_half_v2(
        x_vec,
        weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        split_k,
        scratch,
        out_vec,
        bias is not None,
    )
    return out_vec.view(*x.shape[:-1], n)


def persistent_matvec_half(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: dict | None = None,
) -> torch.Tensor:
    if (
        x.device.type != "cuda"
        or weight.device.type != "cuda"
        or x.dtype != torch.float16
        or weight.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
    ):
        raise RuntimeError("persistent_matvec_half expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    n, _k = weight.shape
    if (
        workspace is not None
        and workspace.get("type") == "persistent"
        and tuple(workspace["out"].shape) == (n,)
        and workspace["out"].device == x.device
        and workspace["out"].dtype == x.dtype
    ):
        out_vec = workspace["out"]
    else:
        out_vec = torch.empty((n,), dtype=x.dtype, device=x.device)

    ext = _load_ext()
    ext.persistent_matvec_half(
        x_vec,
        weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        out_vec,
        bias is not None,
    )
    return out_vec.view(*x.shape[:-1], n)


def nosplit_matvec_half(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: dict | None = None,
) -> torch.Tensor:
    if (
        x.device.type != "cuda"
        or weight.device.type != "cuda"
        or x.dtype != torch.float16
        or weight.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
    ):
        raise RuntimeError("nosplit_matvec_half expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    n, _k = weight.shape
    if (
        workspace is not None
        and workspace.get("type") == "nosplit"
        and tuple(workspace["out"].shape) == (n,)
        and workspace["out"].device == x.device
        and workspace["out"].dtype == x.dtype
    ):
        out_vec = workspace["out"]
    else:
        out_vec = torch.empty((n,), dtype=x.dtype, device=x.device)

    # Cold-L2 microbench shows the real down_proj shape (2048x6144) is
    # consistently better on the xwarp async path than the row4 auto path.
    if n == 2048 and _k == 6144:
        return nosplit_matvec_half_variant(
            x,
            weight,
            bias,
            "k6144_n8k1024_xwarp_w_async",
        )

    ext = _load_ext()
    ext.nosplit_matvec_half(
        x_vec,
        weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        out_vec,
        bias is not None,
    )
    return out_vec.view(*x.shape[:-1], n)


def nosplit_hfma2_matvec_half(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: dict | None = None,
) -> torch.Tensor:
    if (
        x.device.type != "cuda"
        or weight.device.type != "cuda"
        or x.dtype != torch.float16
        or weight.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
    ):
        raise RuntimeError("nosplit_hfma2 expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    n, _k = weight.shape
    if (
        workspace is not None
        and workspace.get("type") == "nosplit_hfma2"
        and tuple(workspace["out"].shape) == (n,)
        and workspace["out"].device == x.device
        and workspace["out"].dtype == x.dtype
    ):
        out_vec = workspace["out"]
    else:
        out_vec = torch.empty((n,), dtype=x.dtype, device=x.device)

    ext = _load_ext()
    ext.nosplit_hfma2_matvec_half(
        x_vec,
        weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        out_vec,
        bias is not None,
    )
    return out_vec.view(*x.shape[:-1], n)


def nosplit_matvec_half_variant(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    variant: str,
    workspace: dict | None = None,
) -> torch.Tensor:
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
        raise RuntimeError("nosplit_matvec_half_variant expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    n, _k = weight.shape
    if (
        workspace is not None
        and workspace.get("type") == f"nosplit_variant:{variant}"
        and tuple(workspace["out"].shape) == (n,)
        and workspace["out"].device == x.device
        and workspace["out"].dtype == x.dtype
    ):
        out_vec = workspace["out"]
    else:
        out_vec = torch.empty((n,), dtype=x.dtype, device=x.device)

    ext = _load_ext()
    ext.nosplit_matvec_half_variant(
        x_vec,
        weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        out_vec,
        bias is not None,
        NOSPLIT_VARIANTS[variant],
    )
    return out_vec.view(*x.shape[:-1], n)


def weight_stream_read_half(
    weight: torch.Tensor,
    workspace: dict | None = None,
) -> torch.Tensor:
    if (
        weight.device.type != "cuda"
        or weight.dtype != torch.float16
        or weight.ndim != 2
        or not weight.is_contiguous()
    ):
        raise RuntimeError("weight_stream_read_half expects contiguous fp16 CUDA weight tensor [N, K]")

    num_sms = torch.cuda.get_device_properties(weight.device).multi_processor_count
    grid_blocks = min(max(1, (weight.numel() // 8 + 255) // 256), max(1, num_sms * 8))
    out_elems = grid_blocks * 8

    if (
        workspace is not None
        and workspace.get("type") == "weight_stream_read_half"
        and tuple(workspace["out"].shape) == (out_elems,)
        and workspace["out"].device == weight.device
        and workspace["out"].dtype == torch.int32
    ):
        out = workspace["out"]
    else:
        out = torch.empty((out_elems,), dtype=torch.int32, device=weight.device)

    ext = _load_ext()
    ext.weight_stream_read_half(weight, out)
    return out


def weight_stream_read_dual_half(
    weight_a: torch.Tensor,
    weight_b: torch.Tensor,
    vecs_per_block: int,
    route_mode: str = "force_a",
    route_table: torch.Tensor | None = None,
    smid_threshold: int | None = None,
    workspace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if route_mode not in DUALCOPY_ROUTE_MODES:
        raise ValueError(f"unknown dualcopy route_mode: {route_mode}")
    if (
        weight_a.device.type != "cuda"
        or weight_b.device.type != "cuda"
        or weight_a.dtype != torch.float16
        or weight_b.dtype != torch.float16
        or weight_a.ndim != 2
        or weight_b.ndim != 2
        or not weight_a.is_contiguous()
        or not weight_b.is_contiguous()
        or tuple(weight_a.shape) != tuple(weight_b.shape)
    ):
        raise RuntimeError("weight_stream_read_dual_half expects matching contiguous fp16 CUDA weight tensors [N, K]")
    if vecs_per_block <= 0:
        raise ValueError("vecs_per_block must be > 0")

    num_sms = torch.cuda.get_device_properties(weight_a.device).multi_processor_count
    if smid_threshold is None:
        smid_threshold = num_sms // 2

    total_vecs = weight_a.numel() // 8
    if total_vecs * 8 != weight_a.numel():
        raise RuntimeError("weight_stream_read_dual_half requires element count divisible by 8")
    if total_vecs % vecs_per_block != 0:
        raise RuntimeError("weight_stream_read_dual_half requires total_vecs divisible by vecs_per_block")

    grid_blocks = total_vecs // vecs_per_block
    warps_per_block = 256 // 32
    route_key = f"weight_stream_read_dual_half:{route_mode}:{vecs_per_block}:{grid_blocks}"
    if (
        workspace is not None
        and workspace.get("type") == route_key
        and tuple(workspace["out"].shape) == (grid_blocks * warps_per_block,)
        and tuple(workspace["trace"].shape) == (grid_blocks, 2)
        and tuple(workspace["smid"].shape) == (grid_blocks,)
        and tuple(workspace["route"].shape) == (grid_blocks,)
        and workspace["out"].device == weight_a.device
    ):
        out = workspace["out"]
        trace = workspace["trace"]
        smid = workspace["smid"]
        route = workspace["route"]
    else:
        out = torch.empty((grid_blocks * warps_per_block,), dtype=torch.int32, device=weight_a.device)
        trace = torch.empty((grid_blocks, 2), dtype=torch.int64, device=weight_a.device)
        smid = torch.empty((grid_blocks,), dtype=torch.int32, device=weight_a.device)
        route = torch.empty((grid_blocks,), dtype=torch.int32, device=weight_a.device)
        if workspace is not None:
            workspace["type"] = route_key
            workspace["out"] = out
            workspace["trace"] = trace
            workspace["smid"] = smid
            workspace["route"] = route

    if route_table is None:
        route_table = torch.empty((0,), dtype=torch.int32, device=weight_a.device)
    elif (
        route_table.device != weight_a.device
        or route_table.dtype != torch.int32
        or route_table.ndim != 1
        or not route_table.is_contiguous()
    ):
        raise RuntimeError("route_table must be a contiguous int32 CUDA tensor")

    ext = _load_ext()
    ext.weight_stream_read_dual_half(
        weight_a,
        weight_b,
        out,
        trace,
        smid,
        route,
        route_table,
        vecs_per_block,
        DUALCOPY_ROUTE_MODES[route_mode],
        smid_threshold,
    )
    return out, trace, smid, route


def l2_8kb_pointer_chase_pair(
    chain: torch.Tensor,
    start_even: torch.Tensor,
    start_odd: torch.Tensor,
    flush_buf: torch.Tensor,
    chain_len: int,
    swap_order: bool = False,
    workspace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        chain.device.type != "cuda"
        or start_even.device != chain.device
        or start_odd.device != chain.device
        or flush_buf.device != chain.device
        or chain.dtype != torch.int64
        or start_even.dtype != torch.int64
        or start_odd.dtype != torch.int64
        or flush_buf.dtype != torch.int32
        or chain.ndim != 1
        or start_even.ndim != 1
        or start_odd.ndim != 1
        or flush_buf.ndim != 1
        or not chain.is_contiguous()
        or not start_even.is_contiguous()
        or not start_odd.is_contiguous()
        or not flush_buf.is_contiguous()
    ):
        raise RuntimeError("l2_8kb_pointer_chase_pair expects contiguous CUDA tensors: chain[int64], start_even[int64], start_odd[int64], flush_buf[int32]")

    num_blocks = start_even.numel()
    if start_odd.numel() != num_blocks:
        raise RuntimeError("start_even/start_odd size mismatch")
    if flush_buf.numel() % max(1, num_blocks) != 0:
        raise RuntimeError("flush_buf size must divide evenly by block count")
    if chain_len <= 0:
        raise RuntimeError("chain_len must be > 0")

    if (
        workspace is not None
        and workspace.get("type") == "l2_8kb_pointer_chase_pair"
        and tuple(workspace["trace"].shape) == (num_blocks, 4)
        and tuple(workspace["smid"].shape) == (num_blocks,)
        and tuple(workspace["final_idx"].shape) == (num_blocks, 2)
        and tuple(workspace["flush_sink"].shape) == (num_blocks,)
        and workspace["trace"].device == chain.device
    ):
        trace = workspace["trace"]
        smid = workspace["smid"]
        final_idx = workspace["final_idx"]
        flush_sink = workspace["flush_sink"]
    else:
        trace = torch.empty((num_blocks, 4), dtype=torch.int64, device=chain.device)
        smid = torch.empty((num_blocks,), dtype=torch.int32, device=chain.device)
        final_idx = torch.empty((num_blocks, 2), dtype=torch.int64, device=chain.device)
        flush_sink = torch.empty((num_blocks,), dtype=torch.int32, device=chain.device)
        if workspace is not None:
            workspace["type"] = "l2_8kb_pointer_chase_pair"
            workspace["trace"] = trace
            workspace["smid"] = smid
            workspace["final_idx"] = final_idx
            workspace["flush_sink"] = flush_sink

    ext = _load_ext()
    ext.l2_8kb_pointer_chase_pair(
        chain,
        start_even,
        start_odd,
        flush_buf,
        trace,
        smid,
        final_idx,
        flush_sink,
        chain_len,
        int(bool(swap_order)),
    )
    return trace, smid, final_idx, flush_sink


def l2_sequential_pointer_chase_trace(
    chain: torch.Tensor,
    start: torch.Tensor,
    flush_buf: torch.Tensor,
    chain_len: int,
    workspace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        chain.device.type != "cuda"
        or start.device != chain.device
        or flush_buf.device != chain.device
        or chain.dtype != torch.int64
        or start.dtype != torch.int64
        or flush_buf.dtype != torch.int32
        or chain.ndim != 1
        or start.ndim != 1
        or flush_buf.ndim != 1
        or not chain.is_contiguous()
        or not start.is_contiguous()
        or not flush_buf.is_contiguous()
    ):
        raise RuntimeError("l2_sequential_pointer_chase_trace expects contiguous CUDA tensors: chain[int64], start[int64], flush_buf[int32]")

    num_blocks = start.numel()
    if num_blocks <= 0:
        raise RuntimeError("start must be non-empty")
    if flush_buf.numel() % num_blocks != 0:
        raise RuntimeError("flush_buf size must divide evenly by block count")
    if chain_len <= 0:
        raise RuntimeError("chain_len must be > 0")

    if (
        workspace is not None
        and workspace.get("type") == "l2_sequential_pointer_chase_trace"
        and tuple(workspace["latency"].shape) == (num_blocks, chain_len)
        and tuple(workspace["smid"].shape) == (num_blocks,)
        and tuple(workspace["final_idx"].shape) == (num_blocks,)
        and tuple(workspace["flush_sink"].shape) == (num_blocks,)
        and workspace["latency"].device == chain.device
    ):
        latency = workspace["latency"]
        smid = workspace["smid"]
        final_idx = workspace["final_idx"]
        flush_sink = workspace["flush_sink"]
    else:
        latency = torch.empty((num_blocks, chain_len), dtype=torch.int64, device=chain.device)
        smid = torch.empty((num_blocks,), dtype=torch.int32, device=chain.device)
        final_idx = torch.empty((num_blocks,), dtype=torch.int64, device=chain.device)
        flush_sink = torch.empty((num_blocks,), dtype=torch.int32, device=chain.device)
        if workspace is not None:
            workspace["type"] = "l2_sequential_pointer_chase_trace"
            workspace["latency"] = latency
            workspace["smid"] = smid
            workspace["final_idx"] = final_idx
            workspace["flush_sink"] = flush_sink

    ext = _load_ext()
    ext.l2_sequential_pointer_chase_trace(
        chain,
        start,
        flush_buf,
        latency,
        smid,
        final_idx,
        flush_sink,
        chain_len,
    )
    return latency, smid, final_idx, flush_sink


def nosplit_weight_only_half_variant(
    weight: torch.Tensor,
    variant: str,
    workspace: dict | None = None,
) -> torch.Tensor:
    if variant not in NOSPLIT_VARIANTS:
        raise ValueError(f"unknown nosplit variant: {variant}")
    if (
        weight.device.type != "cuda"
        or weight.dtype != torch.float16
        or weight.ndim != 2
    ):
        raise RuntimeError("nosplit_weight_only_half_variant expects fp16 CUDA weight tensor [N, K]")

    n, _k = weight.shape
    if (
        workspace is not None
        and workspace.get("type") == f"nosplit_weight_only:{variant}"
        and tuple(workspace["out"].shape) == (n,)
        and workspace["out"].device == weight.device
        and workspace["out"].dtype == torch.int32
    ):
        out = workspace["out"]
    else:
        out = torch.empty((n,), dtype=torch.int32, device=weight.device)

    ext = _load_ext()
    ext.nosplit_weight_only_half_variant(weight, out, NOSPLIT_VARIANTS[variant])
    return out


def nosplit_matvec_half_dualcopy_routed(
    x: torch.Tensor,
    weight_a: torch.Tensor,
    weight_b: torch.Tensor,
    bias: torch.Tensor | None,
    route_mode: str = "table",
    route_table: torch.Tensor | None = None,
    smid_threshold: int | None = None,
    workspace: dict | None = None,
) -> torch.Tensor:
    if route_mode not in DUALCOPY_ROUTE_MODES:
        raise ValueError(f"unknown dualcopy route_mode: {route_mode}")
    if (
        x.device.type != "cuda"
        or weight_a.device.type != "cuda"
        or weight_b.device.type != "cuda"
        or x.dtype != torch.float16
        or weight_a.dtype != torch.float16
        or weight_b.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
        or weight_a.ndim != 2
        or weight_b.ndim != 2
        or tuple(weight_a.shape) != tuple(weight_b.shape)
        or weight_a.shape[1] != x.shape[-1]
    ):
        raise RuntimeError("nosplit_matvec_half_dualcopy_routed expects matching fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    n, _k = weight_a.shape
    if smid_threshold is None:
        smid_threshold = torch.cuda.get_device_properties(weight_a.device).multi_processor_count // 2
    if (
        workspace is not None
        and workspace.get("type") == f"nosplit_dualcopy:{route_mode}"
        and tuple(workspace["out"].shape) == (n,)
        and workspace["out"].device == x.device
        and workspace["out"].dtype == x.dtype
    ):
        out_vec = workspace["out"]
    else:
        out_vec = torch.empty((n,), dtype=x.dtype, device=x.device)
        if workspace is not None:
            workspace["type"] = f"nosplit_dualcopy:{route_mode}"
            workspace["out"] = out_vec

    if route_table is None:
        route_table = torch.empty((0,), dtype=torch.int32, device=x.device)
    elif (
        route_table.device != x.device
        or route_table.dtype != torch.int32
        or route_table.ndim != 1
        or not route_table.is_contiguous()
    ):
        raise RuntimeError("route_table must be a contiguous int32 CUDA tensor")

    ext = _load_ext()
    ext.nosplit_matvec_half_dualcopy_routed(
        x_vec,
        weight_a,
        weight_b,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        out_vec,
        route_table,
        bias is not None,
        DUALCOPY_ROUTE_MODES[route_mode],
        smid_threshold,
    )
    return out_vec.view(*x.shape[:-1], n)


def nosplit_matvec_half_partition(
    x: torch.Tensor,
    weight_buf: torch.Tensor,
    bias: torch.Tensor | None,
    chunk_classes: torch.Tensor,
    sm_class_lo: int,
    sm_class_hi: int,
    n_rows: int,
    workspace: dict | None = None,
) -> torch.Tensor:
    """Partition-aware GEMV for k=2048 with interleaved weight layout."""
    if (
        x.device.type != "cuda"
        or weight_buf.device.type != "cuda"
        or x.dtype != torch.float16
        or weight_buf.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
    ):
        raise RuntimeError("nosplit_matvec_half_partition expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    if (
        workspace is not None
        and workspace.get("type") == "nosplit_partition"
        and tuple(workspace["out"].shape) == (n_rows,)
        and workspace["out"].device == x.device
        and workspace["out"].dtype == x.dtype
    ):
        out_vec = workspace["out"]
    else:
        out_vec = torch.empty((n_rows,), dtype=x.dtype, device=x.device)
        if workspace is not None:
            workspace["type"] = "nosplit_partition"
            workspace["out"] = out_vec

    ext = _load_ext()
    # Convert to signed int64 for pybind11 (C++ int64_t)
    _lo = sm_class_lo if sm_class_lo < (1 << 63) else sm_class_lo - (1 << 64)
    _hi = sm_class_hi if sm_class_hi < (1 << 63) else sm_class_hi - (1 << 64)
    ext.nosplit_matvec_half_partition(
        x_vec,
        weight_buf,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        out_vec,
        chunk_classes,
        _lo,
        _hi,
        n_rows,
        bias is not None,
    )
    return out_vec.view(*x.shape[:-1], n_rows)


def nosplit_matvec_half_partition_v2(
    x: torch.Tensor,
    weight_buf: torch.Tensor,
    bias: torch.Tensor | None,
    chunk_class_bits: int,
    sm_class_lo: int,
    sm_class_hi: int,
    n_rows: int,
    workspace: dict | None = None,
) -> torch.Tensor:
    """Partition-aware GEMV v2 for k=2048 — zero extra memory reads."""
    if (
        x.device.type != "cuda"
        or weight_buf.device.type != "cuda"
        or x.dtype != torch.float16
        or weight_buf.dtype != torch.float16
        or x.ndim < 2
        or x.shape[-2] != 1
    ):
        raise RuntimeError("nosplit_matvec_half_partition_v2 expects fp16 CUDA single-token decode tensors")

    x_vec = x.reshape(-1).contiguous()
    if (
        workspace is not None
        and workspace.get("type") == "nosplit_partition_v2"
        and tuple(workspace["out"].shape) == (n_rows,)
        and workspace["out"].device == x.device
        and workspace["out"].dtype == x.dtype
    ):
        out_vec = workspace["out"]
    else:
        out_vec = torch.empty((n_rows,), dtype=x.dtype, device=x.device)
        if workspace is not None:
            workspace["type"] = "nosplit_partition_v2"
            workspace["out"] = out_vec

    ext = _load_ext()
    # Convert to signed int64 for pybind11 (C++ int64_t)
    _lo = sm_class_lo if sm_class_lo < (1 << 63) else sm_class_lo - (1 << 64)
    _hi = sm_class_hi if sm_class_hi < (1 << 63) else sm_class_hi - (1 << 64)
    _cc = chunk_class_bits if chunk_class_bits < (1 << 63) else chunk_class_bits - (1 << 64)
    ext.nosplit_matvec_half_partition_v2(
        x_vec,
        weight_buf,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        out_vec,
        _cc,
        _lo,
        _hi,
        n_rows,
        bias is not None,
    )
    return out_vec.view(*x.shape[:-1], n_rows)

__all__ = [
    "splitk_matvec",
    "splitk_matvec_row4",
    "splitk_matvec_row8_shared",
    "splitk_matvec_row4_shared_vec_half",
    "splitk_matvec_row4_shared_vec_half_v2",
    "persistent_matvec_half",
    "nosplit_matvec_half",
    "nosplit_hfma2_matvec_half",
    "nosplit_matvec_half_variant",
    "weight_stream_read_half",
    "weight_stream_read_dual_half",
    "l2_8kb_pointer_chase_pair",
    "l2_sequential_pointer_chase_trace",
    "nosplit_weight_only_half_variant",
    "nosplit_matvec_half_dualcopy_routed",
    "nosplit_matvec_half_partition",
    "nosplit_matvec_half_partition_v2",
]
