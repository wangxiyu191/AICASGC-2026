#!/usr/bin/env python3
"""Shared CUDA benchmark helpers."""

from __future__ import annotations

import torch


DEFAULT_BENCHMARK_L2_FLUSH_MB = 64


def make_l2_flush_runner(device, flush_mb: int = DEFAULT_BENCHMARK_L2_FLUSH_MB):
    if flush_mb <= 0:
        return None
    device = torch.device(device)
    flush_bytes = flush_mb * 1024 * 1024
    flush_elems = (flush_bytes + 3) // 4
    flush_buf = torch.zeros(flush_elems, device=device, dtype=torch.int32)

    def _runner():
        # Touch a working set larger than L2 to evict hot lines before replay.
        flush_buf.add_(1)

    return _runner


def benchmark_style_cuda_reset(pre_replay=None):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if pre_replay is not None:
        pre_replay()
        torch.cuda.synchronize()
