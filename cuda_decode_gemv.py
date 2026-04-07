#!/usr/bin/env python3
"""Runtime loader for experimental CUDA decode GEMV kernels."""

from __future__ import annotations

import importlib.util
import importlib.machinery
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import torch


_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _ROOT / "cuda_extensions"
_MODULE_NAME = "cuda_decode_gemv_ext"
NOSPLIT_PROFILE_SLOT_NAMES = (
    "kernel_start",
    "tile0_load_done",
    "tile0_compute_done",
    "tile1_load_done",
    "tile1_compute_done",
    "tile2_load_done",
    "tile2_compute_done",
    "tile3_load_done",
    "tile3_compute_done",
    "tile4_load_done",
    "tile4_compute_done",
    "tile5_load_done",
    "tile5_compute_done",
    "kernel_done",
)
NOSPLIT_XWARP_DETAIL_SLOT_NAMES = (
    "prologue_x_issue_start",
    "prologue_x_issue_end",
    "prologue_w_issue_start",
    "prologue_w_issue_end",
    "prologue_wait_start",
    "prologue_wait_end",
    "prologue_sync_start",
    "prologue_sync_end",
    "tile0_x_issue_start",
    "tile0_x_issue_end",
    "tile0_w_issue_start",
    "tile0_w_issue_end",
    "tile0_compute_start",
    "tile0_compute_end",
    "tile0_wait_start",
    "tile0_wait_end",
    "tile0_sync_start",
    "tile0_sync_end",
    "tile1_x_issue_start",
    "tile1_x_issue_end",
    "tile1_w_issue_start",
    "tile1_w_issue_end",
    "tile1_compute_start",
    "tile1_compute_end",
    "tile1_wait_start",
    "tile1_wait_end",
    "tile1_sync_start",
    "tile1_sync_end",
    "tile2_x_issue_start",
    "tile2_x_issue_end",
    "tile2_w_issue_start",
    "tile2_w_issue_end",
    "tile2_compute_start",
    "tile2_compute_end",
    "tile2_wait_start",
    "tile2_wait_end",
    "tile2_sync_start",
    "tile2_sync_end",
    "tile3_x_issue_start",
    "tile3_x_issue_end",
    "tile3_w_issue_start",
    "tile3_w_issue_end",
    "tile3_compute_start",
    "tile3_compute_end",
    "tile3_wait_start",
    "tile3_wait_end",
    "tile3_sync_start",
    "tile3_sync_end",
    "tile4_x_issue_start",
    "tile4_x_issue_end",
    "tile4_w_issue_start",
    "tile4_w_issue_end",
    "tile4_compute_start",
    "tile4_compute_end",
    "tile4_wait_start",
    "tile4_wait_end",
    "tile4_sync_start",
    "tile4_sync_end",
    "tile5_x_issue_start",
    "tile5_x_issue_end",
    "tile5_w_issue_start",
    "tile5_w_issue_end",
    "tile5_compute_start",
    "tile5_compute_end",
    "tile5_wait_start",
    "tile5_wait_end",
    "tile5_sync_start",
    "tile5_sync_end",
    "epilogue_start",
    "epilogue_end",
)
NOSPLIT_VARIANTS = {
    "auto": 0,
    "k2048_row2": 1,
    "k2048_row4": 2,
    "k6144_row4": 3,
    "persistent": 4,
    "persistent_hfma2": 5,
    "k6144_weight_async": 6,
    "k6144_n8k1024_async": 7,
    "k6144_n8k1024_xw_async": 8,
    "k6144_n8k512_s3_xw_async": 9,
    "k6144_persistent_xfull_w2_async": 10,
    "k6144_persistent_xfull_row1_warp8_async": 11,
    "k6144_persistent_xfull_n8k1024_w_async": 12,
    "k6144_n8k1024_xwarp_w_async": 13,
    "k6144_n8k1024_xwarp_w_async_sms2": 14,
    "k6144_n8k1024_xwarp_w_async_sms2_balanced": 15,
    "k2048_row8_persistent": 16,
    "k2048_row4_persistent": 17,
}
DUALCOPY_ROUTE_MODES = {
    "force_a": 0,
    "force_b": 1,
    "smid_half": 2,
    "smid_parity": 3,
    "table": 4,
}


def _find_prebuilt_ext() -> Path | None:
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = _SRC_DIR / f"{_MODULE_NAME}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _ext_source_files() -> tuple[Path, ...]:
    files = {_SRC_DIR / "setup_decode_gemv.py"}
    for pattern in ("*.cpp", "*.cu", "*.cuh", "*.h"):
        files.update(_SRC_DIR.glob(pattern))
    return tuple(sorted(path for path in files if path.exists()))


def _should_rebuild_ext(prebuilt_ext: Path | None) -> bool:
    if prebuilt_ext is None:
        return True
    ext_mtime_ns = prebuilt_ext.stat().st_mtime_ns
    return any(path.stat().st_mtime_ns > ext_mtime_ns for path in _ext_source_files())


@lru_cache(maxsize=1)
def _load_ext():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    prebuilt_ext = _find_prebuilt_ext()
    if _should_rebuild_ext(prebuilt_ext):
        env = dict(os.environ)
        env.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
        env["PATH"] = f"/usr/local/cuda/bin:{env.get('PATH', '')}"
        subprocess.run(
            [sys.executable, "setup_decode_gemv.py", "build_ext", "--inplace"],
            cwd=_SRC_DIR,
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        prebuilt_ext = _find_prebuilt_ext()
    if prebuilt_ext is None:
        raise RuntimeError("failed to build cuda_decode_gemv_ext")

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, prebuilt_ext)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load extension spec from {prebuilt_ext}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nosplit_grid_blocks(n: int, k: int) -> int:
    if k == 2048 and n <= 2048:
        return (n + 1) // 2
    return (n + 3) // 4


def _nosplit_variant_grid_blocks(n: int, k: int, variant: str) -> int:
    if variant in (
        "k6144_n8k1024_async",
        "k6144_n8k1024_xw_async",
        "k6144_n8k1024_xwarp_w_async",
        "k6144_n8k1024_xwarp_w_async_sms2",
        "k6144_n8k1024_xwarp_w_async_sms2_balanced",
    ):
        return (n + 7) // 8
    return _nosplit_grid_blocks(n, k)


def _nosplit_profile_kernel_name(n: int, k: int) -> str:
    if k == 2048 and n <= 2048:
        return "nosplit_matvec_k2048_row2_hfma2_fwd_kernel"
    if k == 2048:
        return "nosplit_matvec_k2048_hfma2_fwd_kernel"
    if k == 6144:
        return "nosplit_matvec_k6144_hfma2_fwd_kernel"
    raise RuntimeError("nosplit internal profiling currently supports only k=2048/6144 specialized kernels")


def _nosplit_profile_variant_kernel_name(variant: str) -> str:
    mapping = {
        "k6144_row4": "nosplit_matvec_k6144_hfma2_fwd_kernel",
        "k6144_n8k1024_async": "nosplit_matvec_k6144_n8k1024_async_fwd_kernel",
        "k6144_n8k1024_xw_async": "nosplit_matvec_k6144_n8k1024_xw_async_fwd_kernel",
        "k6144_n8k1024_xwarp_w_async": "nosplit_matvec_k6144_n8k1024_xwarp_w_async_fwd_kernel",
        "k6144_n8k1024_xwarp_w_async_sms2": "nosplit_matvec_k6144_n8k1024_xwarp_w_async_sms2_fwd_kernel",
        "k6144_n8k1024_xwarp_w_async_sms2_balanced": "nosplit_matvec_k6144_n8k1024_xwarp_w_async_sms2_balanced_fwd_kernel",
    }
    try:
        return mapping[variant]
    except KeyError as exc:
        available = ", ".join(sorted(mapping))
        raise RuntimeError(f"nosplit profile variant currently supports only: {available}") from exc


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


SOFTSYNC_PROFILE_SLOT_NAMES = (
    "kernel_start",            # 0
    "x_load_done",             # 1
    "gate_up_compute_done",    # 2
    "barrier_start",           # 3
    "barrier_done",            # 4
    "hidden_load_done",        # 5
    "tile0_load_done",         # 6
    "tile0_compute_done",      # 7
    "tile1_load_done",         # 8
    "tile1_compute_done",      # 9
    "tile2_compute_done",      # 10
    "kernel_done",             # 11
)

PERSISTENT_PROFILE_SLOT_NAMES = (
    "kernel_start",         # 0
    "x_load_done",          # 1
    "gateup_iter0_done",    # 2
    "gateup_iter1_done",    # 3  (0 if block does only 1 iteration)
    "gridsync_start",       # 4
    "gridsync_done",        # 5
    "down_hidden_loaded",   # 6
    "down_compute_done",    # 7
    "kernel_done",          # 8
)


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


# ---------------------------------------------------------------------------
# __grid_sync() versions: use cooperative_groups::this_grid().sync()
# No barrier_count/barrier_phase tensors needed.
# ---------------------------------------------------------------------------

GRIDSYNC_PROFILE_SLOT_NAMES = SOFTSYNC_PROFILE_SLOT_NAMES


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
