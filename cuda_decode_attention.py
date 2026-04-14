#!/usr/bin/env python3
"""Runtime loader for an experimental CUDA decode-attention kernel."""

from __future__ import annotations

import os
import subprocess
import sys
import importlib.util
import importlib.machinery
from functools import lru_cache
from pathlib import Path

import torch


_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _ROOT / "cuda_extensions"
_MODULE_NAME = "cuda_decode_attention_ext"
_BUILD_SOURCES = (
    _SRC_DIR / "setup.py",
    _SRC_DIR / "decode_attention_ext.cpp",
    _SRC_DIR / "decode_attention_kernel.cu",
    _SRC_DIR / "decode_attention_v4_kernel.cu",
)


def _find_prebuilt_ext() -> Path | None:
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = _SRC_DIR / f"{_MODULE_NAME}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _needs_rebuild(prebuilt_ext: Path | None) -> bool:
    if prebuilt_ext is None:
        return True
    ext_mtime = prebuilt_ext.stat().st_mtime
    return any(src.exists() and src.stat().st_mtime > ext_mtime for src in _BUILD_SOURCES)


@lru_cache(maxsize=1)
def _load_ext():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    prebuilt_ext = _find_prebuilt_ext()
    if _needs_rebuild(prebuilt_ext):
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
        prebuilt_ext = _find_prebuilt_ext()
    if prebuilt_ext is None:
        raise RuntimeError("failed to build cuda_decode_attention_ext")

    spec = importlib.util.spec_from_file_location(
        _MODULE_NAME,
        prebuilt_ext,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load extension spec from {prebuilt_ext}")
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
        and workspace.get("split_k", 0) >= split_k
        and workspace["partial_acc"].shape[0] == layout["partial_acc_shape"][0]
        and workspace["partial_acc"].shape[2:] == layout["partial_acc_shape"][2:]
        and workspace["partial_stats"].shape[0] == layout["partial_stats_shape"][0]
        and workspace["partial_stats"].shape[2:] == layout["partial_stats_shape"][2:]
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
        int(split_k),
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
        and workspace.get("split_k", 0) >= split_k
        and workspace["partial_acc"].shape[0] == layout["partial_acc_shape"][0]
        and workspace["partial_acc"].shape[2:] == layout["partial_acc_shape"][2:]
        and workspace["partial_stats"].shape[0] == layout["partial_stats_shape"][0]
        and workspace["partial_stats"].shape[2:] == layout["partial_stats_shape"][2:]
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
        int(split_k),
    )
    return out.unsqueeze(2).transpose(1, 2)


def _v4_launch_config(seq_len: int) -> int:
    """Return split_k for v4 tiled kernel."""
    if seq_len < 512:
        return 16
    elif seq_len < 1024:
        return 16
    else:
        return 32


def decode_attn_v4_splitkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: float,
    num_key_value_groups: int,
    cache_position: torch.Tensor,
    split_k: int | None = None,
    workspace: dict | None = None,
    prefetch_weight: torch.Tensor | None = None,
    prefetch_blocks: int = 0,
) -> torch.Tensor | None:
    if (
        num_key_value_groups != 2
        or q.device.type != "cuda"
        or k.device.type != "cuda"
        or v.device.type != "cuda"
        or cache_position.device.type != "cuda"
        or q.shape[0] != 1
        or q.shape[2] != 1
        or q.shape[-1] != 128
        or q.dtype != torch.float16
    ):
        return None

    seq_len = k.shape[2]
    if split_k is None:
        split_k = _v4_launch_config(seq_len)

    head_dim = q.shape[-1]
    layout = _workspace_layout(
        num_kv_heads=k.shape[1],
        num_key_value_groups=num_key_value_groups,
        head_dim=head_dim,
        split_k=split_k,
    )
    if (
        workspace is not None
        and workspace.get("split_k", 0) >= split_k
        and workspace["partial_acc"].shape[0] == layout["partial_acc_shape"][0]
        and workspace["partial_acc"].shape[2:] == layout["partial_acc_shape"][2:]
        and workspace["partial_stats"].shape[0] == layout["partial_stats_shape"][0]
        and workspace["partial_stats"].shape[2:] == layout["partial_stats_shape"][2:]
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

    valid_prefetch = (
        prefetch_blocks > 0
        and prefetch_weight is not None
        and prefetch_weight.device == q.device
        and prefetch_weight.dtype == torch.float16
        and prefetch_weight.dim() == 2
        and tuple(prefetch_weight.shape) == (2048, 2048)
        and prefetch_weight.is_contiguous()
    )
    prefetch_tensor = prefetch_weight if valid_prefetch else q.reshape(-1)[:0]

    ext = _load_ext()
    ext.decode_attn_v4_splitkv(
        q, k, v, cache_position,
        float(scaling),
        partial_acc, partial_stats, out,
        int(num_key_value_groups),
        int(split_k),
        prefetch_tensor,
        int(prefetch_blocks if valid_prefetch else 0),
    )
    return out.unsqueeze(2).transpose(1, 2)
