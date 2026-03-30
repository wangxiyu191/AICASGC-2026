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
    "kernel_done",
)
NOSPLIT_VARIANTS = {
    "auto": 0,
    "k2048_row2": 1,
    "k2048_row4": 2,
    "k6144_row4": 3,
    "persistent": 4,
    "persistent_hfma2": 5,
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


def _nosplit_profile_kernel_name(n: int, k: int) -> str:
    if k == 2048 and n <= 2048:
        return "nosplit_matvec_k2048_row2_hfma2_fwd_kernel"
    if k == 2048:
        return "nosplit_matvec_k2048_hfma2_fwd_kernel"
    if k == 6144:
        return "nosplit_matvec_k6144_hfma2_fwd_kernel"
    raise RuntimeError("nosplit internal profiling currently supports only k=2048/6144 specialized kernels")


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
