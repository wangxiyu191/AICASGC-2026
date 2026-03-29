#!/usr/bin/env python3
"""Runtime loader for experimental CUDA decode GEMV kernels."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import torch


_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _ROOT / "cuda_extensions"


@lru_cache(maxsize=1)
def _load_ext():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    ext_suffix = next(iter(__import__("importlib.machinery").machinery.EXTENSION_SUFFIXES))
    so_candidates = sorted(_SRC_DIR.glob(f"cuda_decode_gemv_ext*{ext_suffix}"))
    latest_src_mtime = max(
        (_SRC_DIR / "decode_gemv_ext.cpp").stat().st_mtime,
        (_SRC_DIR / "decode_gemv_kernel.cu").stat().st_mtime,
        (_SRC_DIR / "setup_decode_gemv.py").stat().st_mtime,
    )
    needs_build = (not so_candidates) or so_candidates[0].stat().st_mtime < latest_src_mtime
    if needs_build:
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
        so_candidates = sorted(_SRC_DIR.glob(f"cuda_decode_gemv_ext*{ext_suffix}"))
    if not so_candidates:
        raise RuntimeError("failed to build cuda_decode_gemv_ext")

    spec = importlib.util.spec_from_file_location("cuda_decode_gemv_ext", so_candidates[0])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load extension spec from {so_candidates[0]}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
