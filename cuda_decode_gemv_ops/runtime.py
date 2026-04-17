#!/usr/bin/env python3
"""Shared runtime and public constants for experimental CUDA decode GEMV kernels."""

from __future__ import annotations
import importlib.util
import importlib.machinery
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
import torch

_ROOT = Path(__file__).resolve().parent.parent


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


FULL_LAYER_PROFILE_SLOT_NAMES = (
    "kernel_start",
    "input_rmsnorm_done",
    "qkv_done",
    "sync1_start",
    "sync1_done",
    "qkvcache_done",
    "sync2_start",
    "sync2_done",
    "attention_done",
    "sync3_start",
    "sync3_done",
    "oproj_done",
    "sync4_start",
    "sync4_done",
    "post_rmsnorm_done",
    "gateup_silu_done",
    "sync5_start",
    "sync5_done",
    "downproj_done",
    "kernel_done",
    "gt_start",
    "gt_end",
    "sm_id",
    "attn_prefetch_lines",
)


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


OPROJ_MLP_PROFILE_SLOT_NAMES = (
    "kernel_start",         # 0  (clock64)
    "x_load_done",          # 1  (clock64)
    "oproj_done",           # 2  (clock64)
    "gridsync1_start",      # 3  (clock64)
    "gridsync1_done",       # 4  (clock64)
    "resnorm_done",         # 5  (clock64)
    "gateup_iter0_done",    # 6  (clock64)
    "gateup_iter1_done",    # 7  (clock64, 0 if block does only 1 iteration)
    "gridsync2_start",      # 8  (clock64)
    "gridsync2_done",       # 9  (clock64)
    "down_hidden_loaded",   # 10 (clock64)
    "down_compute_done",    # 11 (clock64)
    "kernel_done",          # 12 (clock64)
    "gt_start",             # 13 (globaltimer, calibration)
    "gt_end",               # 14 (globaltimer, calibration)
    "sm_id",                # 15 (SM ID from %smid)
)


GRIDSYNC_PROFILE_SLOT_NAMES = SOFTSYNC_PROFILE_SLOT_NAMES

__all__ = [
    "NOSPLIT_PROFILE_SLOT_NAMES",
    "NOSPLIT_XWARP_DETAIL_SLOT_NAMES",
    "NOSPLIT_VARIANTS",
    "DUALCOPY_ROUTE_MODES",
    "FULL_LAYER_PROFILE_SLOT_NAMES",
    "SOFTSYNC_PROFILE_SLOT_NAMES",
    "PERSISTENT_PROFILE_SLOT_NAMES",
    "OPROJ_MLP_PROFILE_SLOT_NAMES",
    "GRIDSYNC_PROFILE_SLOT_NAMES",
]
