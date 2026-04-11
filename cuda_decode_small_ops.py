#!/usr/bin/env python3
"""Runtime loader for experimental CUDA small decode ops."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import torch


_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _ROOT / "cuda_extensions"
_MODULE_NAME = "cuda_decode_small_ops_ext"
_BUILD_SOURCES = (
    _SRC_DIR / "setup_decode_small_ops.py",
    _SRC_DIR / "decode_small_ops_ext.cpp",
    _SRC_DIR / "decode_small_ops_kernel.cu",
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
            [sys.executable, "setup_decode_small_ops.py", "build_ext", "--inplace"],
            cwd=_SRC_DIR,
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        prebuilt_ext = _find_prebuilt_ext()
    if prebuilt_ext is None:
        raise RuntimeError("failed to build cuda_decode_small_ops_ext")

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, prebuilt_ext)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load extension spec from {prebuilt_ext}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fused_qk_rms_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        q.device.type != "cuda"
        or k.device.type != "cuda"
        or q_weight.device.type != "cuda"
        or k_weight.device.type != "cuda"
        or cos.device.type != "cuda"
        or sin.device.type != "cuda"
        or q.dtype != torch.float16
        or k.dtype != torch.float16
        or q_weight.dtype != torch.float16
        or k_weight.dtype != torch.float16
        or cos.dtype != torch.float16
        or sin.dtype != torch.float16
    ):
        raise RuntimeError("fused_qk_rms_norm_rope expects fp16 CUDA tensors")

    if q.ndim != 4 or k.ndim != 4:
        raise RuntimeError("fused_qk_rms_norm_rope expects q/k as [B, S, H, D]")
    if q.shape[:2] != k.shape[:2]:
        raise RuntimeError("q/k batch-seq shapes must match")
    if q.shape[-1] != 128 or k.shape[-1] != 128:
        raise RuntimeError("fused_qk_rms_norm_rope currently supports only head_dim=128")
    if q.shape[2] != k.shape[2] * 2:
        raise RuntimeError("fused_qk_rms_norm_rope currently supports q_heads == 2 * kv_heads")
    if q_weight.numel() != 128 or k_weight.numel() != 128:
        raise RuntimeError("fused_qk_rms_norm_rope expects q/k norm weights of shape [128]")

    q_in = q.contiguous()
    k_in = k.contiguous()
    cos_in = cos.contiguous()
    sin_in = sin.contiguous()
    if cos_in.shape != q_in.shape[:2] + (128,) or sin_in.shape != q_in.shape[:2] + (128,):
        raise RuntimeError("fused_qk_rms_norm_rope expects cos/sin shaped [B, S, 128]")

    m = q_in.shape[0] * q_in.shape[1]
    q_3d = q_in.reshape(m, q_in.shape[2], 128)
    k_3d = k_in.reshape(m, k_in.shape[2], 128)
    cos_2d = cos_in.reshape(m, 128)
    sin_2d = sin_in.reshape(m, 128)
    q_out = torch.empty_like(q_3d)
    k_out = torch.empty_like(k_3d)

    ext = _load_ext()
    ext.fused_qk_rms_norm_rope(
        q_3d,
        k_3d,
        q_weight.contiguous(),
        k_weight.contiguous(),
        cos_2d,
        sin_2d,
        q_out,
        k_out,
        eps,
    )
    return q_out.view_as(q_in), k_out.view_as(k_in)


def fused_q_rope_kv_cache_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_position: torch.Tensor,
) -> torch.Tensor:
    if (
        q.device.type != "cuda"
        or k.device.type != "cuda"
        or v.device.type != "cuda"
        or q_weight.device.type != "cuda"
        or k_weight.device.type != "cuda"
        or cos.device.type != "cuda"
        or sin.device.type != "cuda"
        or k_cache.device.type != "cuda"
        or v_cache.device.type != "cuda"
        or cache_position.device.type != "cuda"
        or q.dtype != torch.float16
        or k.dtype != torch.float16
        or v.dtype != torch.float16
        or q_weight.dtype != torch.float16
        or k_weight.dtype != torch.float16
        or cos.dtype != torch.float16
        or sin.dtype != torch.float16
        or k_cache.dtype != torch.float16
        or v_cache.dtype != torch.float16
        or cache_position.dtype != torch.long
    ):
        raise RuntimeError("fused_q_rope_kv_cache_update expects fp16 CUDA tensors and int64 cache_position")

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise RuntimeError("fused_q_rope_kv_cache_update expects q/k/v as [B, S, H, D]")
    if q.shape[:2] != k.shape[:2] or k.shape != v.shape:
        raise RuntimeError("q/k/v shapes must match decode layout")
    if q.shape[0] != 1 or q.shape[1] != 1 or k.shape[0] != 1 or k.shape[1] != 1:
        raise RuntimeError("fused_q_rope_kv_cache_update currently supports single-token decode only")
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        raise RuntimeError("fused_q_rope_kv_cache_update currently supports only head_dim=128")
    if q.shape[2] != k.shape[2] * 2:
        raise RuntimeError("fused_q_rope_kv_cache_update currently supports q_heads == 2 * kv_heads")
    if q_weight.numel() != 128 or k_weight.numel() != 128:
        raise RuntimeError("fused_q_rope_kv_cache_update expects q/k norm weights of shape [128]")
    if cos.shape != q.shape[:2] + (128,) or sin.shape != q.shape[:2] + (128,):
        raise RuntimeError("fused_q_rope_kv_cache_update expects cos/sin shaped [B, S, 128]")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise RuntimeError("fused_q_rope_kv_cache_update expects k_cache/v_cache as [B, KVH, T, D]")
    if k_cache.shape[0] != 1 or v_cache.shape[0] != 1:
        raise RuntimeError("fused_q_rope_kv_cache_update currently supports batch=1 cache")
    if k_cache.shape[1] != k.shape[2] or v_cache.shape[1] != v.shape[2]:
        raise RuntimeError("cache kv-head count mismatch")
    if k_cache.shape[-1] != 128 or v_cache.shape[-1] != 128:
        raise RuntimeError("cache head_dim mismatch")
    if cache_position.ndim != 1 or cache_position.numel() != 1:
        raise RuntimeError("cache_position must be shape [1]")

    q_in = q.contiguous()
    k_in = k.contiguous()
    v_in = v.contiguous()
    cos_in = cos.contiguous()
    sin_in = sin.contiguous()
    q_3d = q_in.reshape(1, q_in.shape[2], 128)
    k_3d = k_in.reshape(1, k_in.shape[2], 128)
    v_3d = v_in.reshape(1, v_in.shape[2], 128)
    cos_2d = cos_in.reshape(1, 128)
    sin_2d = sin_in.reshape(1, 128)
    q_out = torch.empty_like(q_3d)

    ext = _load_ext()
    ext.fused_q_rope_kv_cache_update(
        q_3d,
        k_3d,
        v_3d,
        q_weight.contiguous(),
        k_weight.contiguous(),
        cos_2d,
        sin_2d,
        q_out,
        k_cache,
        v_cache,
        cache_position.contiguous(),
        eps,
    )
    return q_out.view_as(q_in)


def fused_silu_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    if (
        gate.device.type != "cuda"
        or up.device.type != "cuda"
        or gate.dtype != torch.float16
        or up.dtype != torch.float16
    ):
        raise RuntimeError("fused_silu_mul expects fp16 CUDA tensors")
    if gate.shape != up.shape:
        raise RuntimeError("fused_silu_mul expects gate/up with identical shapes")

    gate_in = gate.contiguous()
    up_in = up.contiguous()
    out = torch.empty_like(gate_in)
    ext = _load_ext()
    ext.fused_silu_mul(gate_in.reshape(-1), up_in.reshape(-1), out.reshape(-1))
    return out.view_as(gate_in)


def fused_add_rms_norm(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        residual.device.type != "cuda"
        or x.device.type != "cuda"
        or weight.device.type != "cuda"
        or residual.dtype != torch.float16
        or x.dtype != torch.float16
        or weight.dtype != torch.float16
    ):
        raise RuntimeError("fused_add_rms_norm expects fp16 CUDA tensors")
    if residual.shape != x.shape:
        raise RuntimeError("fused_add_rms_norm expects residual/x with identical shapes")
    if residual.shape[-1] != weight.numel():
        raise RuntimeError("fused_add_rms_norm weight size mismatch")

    residual_in = residual.contiguous()
    x_in = x.contiguous()
    rows = residual_in.numel() // residual_in.shape[-1]
    cols = residual_in.shape[-1]
    residual_2d = residual_in.reshape(rows, cols)
    x_2d = x_in.reshape(rows, cols)
    out_sum = torch.empty_like(residual_2d)
    out_norm = torch.empty_like(residual_2d)
    ext = _load_ext()
    ext.fused_add_rms_norm(
        residual_2d,
        x_2d,
        weight.contiguous(),
        out_sum,
        out_norm,
        eps,
    )
    return out_sum.view_as(residual_in), out_norm.view_as(residual_in)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if (
        x.device.type != "cuda"
        or weight.device.type != "cuda"
        or x.dtype != torch.float16
        or weight.dtype != torch.float16
    ):
        raise RuntimeError("rms_norm expects fp16 CUDA tensors")
    if x.shape[-1] != weight.numel():
        raise RuntimeError("rms_norm weight size mismatch")

    x_in = x.contiguous()
    rows = x_in.numel() // x_in.shape[-1]
    cols = x_in.shape[-1]
    x_2d = x_in.reshape(rows, cols)
    out = torch.empty_like(x_2d)
    ext = _load_ext()
    ext.rms_norm(
        x_2d,
        weight.contiguous(),
        out,
        eps,
    )
    return out.view_as(x_in)


def add(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    if (
        a.device.type != "cuda"
        or b.device.type != "cuda"
        or a.dtype != torch.float16
        or b.dtype != torch.float16
    ):
        raise RuntimeError("add expects fp16 CUDA tensors")
    if a.shape != b.shape:
        raise RuntimeError("add expects identical shapes")

    a_in = a.contiguous()
    b_in = b.contiguous()
    out = torch.empty_like(a_in)
    ext = _load_ext()
    ext.add(a_in.reshape(-1), b_in.reshape(-1), out.reshape(-1))
    return out.view_as(a_in)
