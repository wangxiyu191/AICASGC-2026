#!/usr/bin/env python3
"""Runtime loader for an experimental CUDA decode-attention kernel."""

from __future__ import annotations

import os
import subprocess
import sys
import importlib.util
from functools import lru_cache
from pathlib import Path

import torch


_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _ROOT / "cuda_extensions"


@lru_cache(maxsize=1)
def _load_ext():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    ext_suffix = next(iter(__import__("importlib.machinery").machinery.EXTENSION_SUFFIXES))
    so_candidates = sorted(_SRC_DIR.glob(f"cuda_decode_attention_ext*{ext_suffix}"))
    latest_src_mtime = max(
        (_SRC_DIR / "decode_attention_ext.cpp").stat().st_mtime,
        (_SRC_DIR / "decode_attention_kernel.cu").stat().st_mtime,
        (_SRC_DIR / "setup.py").stat().st_mtime,
    )
    needs_build = (not so_candidates) or so_candidates[0].stat().st_mtime < latest_src_mtime
    if needs_build:
        env = dict(os.environ)
        env.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
        env["PATH"] = f"/usr/local/cuda/bin:{env.get('PATH', '')}"
        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=_SRC_DIR,
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        so_candidates = sorted(_SRC_DIR.glob(f"cuda_decode_attention_ext*{ext_suffix}"))
    if not so_candidates:
        raise RuntimeError("failed to build cuda_decode_attention_ext")

    spec = importlib.util.spec_from_file_location(
        "cuda_decode_attention_ext",
        so_candidates[0],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load extension spec from {so_candidates[0]}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workspace_layout(*, num_kv_heads: int, num_key_value_groups: int, head_dim: int, split_k: int) -> dict:
    return {
        "partial_acc_shape": (num_kv_heads, split_k, num_key_value_groups, head_dim),
        "partial_stats_shape": (num_kv_heads, split_k, num_key_value_groups, 2),
        "out_shape": (1, num_kv_heads * num_key_value_groups, head_dim),
    }


def _qhead_workspace_layout(*, num_heads: int, head_dim: int, split_k: int) -> dict:
    return {
        "partial_acc_shape": (num_heads, split_k, head_dim),
        "partial_stats_shape": (num_heads, split_k, 2),
        "out_shape": (1, num_heads, head_dim),
    }


def grouped_prefix_cachepos_splitkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: float,
    num_key_value_groups: int,
    cache_position: torch.Tensor,
    split_k: int,
    workspace: dict | None = None,
) -> torch.Tensor | None:
    if (
        num_key_value_groups != 2
        or split_k <= 1
        or q.device.type != "cuda"
        or k.device.type != "cuda"
        or v.device.type != "cuda"
        or cache_position.device.type != "cuda"
        or q.shape[0] != 1
        or q.shape[2] != 1
        or q.shape[-1] > 128
    ):
        return None

    head_dim = q.shape[-1]
    layout = _workspace_layout(
        num_kv_heads=k.shape[1],
        num_key_value_groups=num_key_value_groups,
        head_dim=head_dim,
        split_k=split_k,
    )
    if (
        workspace is not None
        and workspace.get("split_k") == split_k
        and tuple(workspace["partial_acc"].shape) == layout["partial_acc_shape"]
        and tuple(workspace["partial_stats"].shape) == layout["partial_stats_shape"]
        and tuple(workspace["out"].shape) == layout["out_shape"]
        and workspace["out"].dtype == q.dtype
        and workspace["partial_acc"].device == q.device
    ):
        partial_acc = workspace["partial_acc"]
        partial_stats = workspace["partial_stats"]
        out = workspace["out"]
    else:
        partial_acc = torch.empty(layout["partial_acc_shape"], dtype=torch.float32, device=q.device)
        partial_stats = torch.empty(layout["partial_stats_shape"], dtype=torch.float32, device=q.device)
        out = torch.empty(layout["out_shape"], dtype=q.dtype, device=q.device)

    ext = _load_ext()
    ext.grouped_prefix_cachepos_splitkv(
        q,
        k,
        v,
        cache_position,
        float(scaling),
        partial_acc,
        partial_stats,
        out,
    )
    return out.unsqueeze(2).transpose(1, 2)


def grouped_prefix_cachepos_splitkv_qhead(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: float,
    num_key_value_groups: int,
    cache_position: torch.Tensor,
    split_k: int,
    workspace: dict | None = None,
) -> torch.Tensor | None:
    if (
        num_key_value_groups != 2
        or split_k <= 1
        or q.device.type != "cuda"
        or k.device.type != "cuda"
        or v.device.type != "cuda"
        or cache_position.device.type != "cuda"
        or q.shape[0] != 1
        or q.shape[2] != 1
        or q.shape[-1] > 128
    ):
        return None

    layout = _qhead_workspace_layout(
        num_heads=q.shape[1],
        head_dim=q.shape[-1],
        split_k=split_k,
    )
    if (
        workspace is not None
        and workspace.get("split_k") == split_k
        and tuple(workspace["partial_acc"].shape) == layout["partial_acc_shape"]
        and tuple(workspace["partial_stats"].shape) == layout["partial_stats_shape"]
        and tuple(workspace["out"].shape) == layout["out_shape"]
        and workspace["out"].dtype == q.dtype
        and workspace["partial_acc"].device == q.device
    ):
        partial_acc = workspace["partial_acc"]
        partial_stats = workspace["partial_stats"]
        out = workspace["out"]
    else:
        partial_acc = torch.empty(layout["partial_acc_shape"], dtype=torch.float32, device=q.device)
        partial_stats = torch.empty(layout["partial_stats_shape"], dtype=torch.float32, device=q.device)
        out = torch.empty(layout["out_shape"], dtype=q.dtype, device=q.device)

    ext = _load_ext()
    ext.grouped_prefix_cachepos_splitkv_qhead(
        q,
        k,
        v,
        cache_position,
        float(scaling),
        partial_acc,
        partial_stats,
        out,
    )
    return out.unsqueeze(2).transpose(1, 2)
