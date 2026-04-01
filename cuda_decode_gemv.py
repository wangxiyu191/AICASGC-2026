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
}


def _find_prebuilt_ext() -> Path | None:
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = _SRC_DIR / f"{_MODULE_NAME}{suffix}"
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def _load_ext():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    prebuilt_ext = _find_prebuilt_ext()
    if prebuilt_ext is None:
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
