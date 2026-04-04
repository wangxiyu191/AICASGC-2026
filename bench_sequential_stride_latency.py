#!/usr/bin/env python3
"""Trace per-access cold-L2 latency for a sequential pointer chase."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmark import get_system_info
import cuda_decode_gemv


def _build_sequential_chain(
    num_blocks: int,
    span_bytes: int,
    stride_bytes: int,
    base_offset_bytes: int,
    device: torch.device,
):
    elem_size = torch.tensor([], dtype=torch.int64).element_size()
    if stride_bytes % elem_size != 0:
        raise ValueError("stride_bytes must be divisible by element size")
    if span_bytes % stride_bytes != 0:
        raise ValueError("span_bytes must be divisible by stride_bytes")
    if base_offset_bytes % elem_size != 0:
        raise ValueError("base_offset_bytes must be divisible by element size")

    step_elems = stride_bytes // elem_size
    chain_len = span_bytes // stride_bytes
    base_offset_elems = base_offset_bytes // elem_size
    block_span_elems = chain_len * step_elems
    block_alloc_elems = block_span_elems + base_offset_elems
    total_elems = num_blocks * block_alloc_elems

    chain = torch.empty((total_elems,), dtype=torch.int64, device=device)
    chain_cpu = torch.zeros((total_elems,), dtype=torch.int64)
    start = torch.empty((num_blocks,), dtype=torch.int64)

    for block in range(num_blocks):
        region_base = block * block_alloc_elems + base_offset_elems
        nodes = region_base + torch.arange(chain_len, dtype=torch.int64) * step_elems
        chain_cpu[nodes[:-1]] = nodes[1:]
        chain_cpu[nodes[-1]] = nodes[-1]
        start[block] = nodes[0]

    chain.copy_(chain_cpu, non_blocking=False)
    start = start.to(device=device, non_blocking=False)

    meta = {
        "span_bytes": span_bytes,
        "span_kb": span_bytes / 1024.0,
        "stride_bytes": stride_bytes,
        "base_offset_bytes": base_offset_bytes,
        "base_offset_kb": base_offset_bytes / 1024.0,
        "chain_len": chain_len,
        "step_elems": step_elems,
        "block_span_elems": block_span_elems,
        "block_alloc_elems": block_alloc_elems,
        "accesses_per_8kb_page": 8192 // stride_bytes,
        "pages_8kb_per_block": span_bytes // 8192,
    }
    return chain, start, meta


def _build_flush_buffer(num_blocks: int, flush_mb_per_block: int, device: torch.device):
    elems_per_block = (flush_mb_per_block * 1024 * 1024) // 4
    if elems_per_block <= 0:
        raise ValueError("flush_mb_per_block too small")
    return torch.randint(
        0,
        2**31 - 1,
        (num_blocks * elems_per_block,),
        device=device,
        dtype=torch.int32,
    )


def _query_sm_clock_mhz() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.current.sm", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    for line in out.stdout.splitlines():
        line = line.strip()
        if line:
            return float(line)
    return None


def _cycles_to_ns(latency_cycles: torch.Tensor, sm_clock_mhz: float | None) -> torch.Tensor:
    latency_f64 = latency_cycles.to(dtype=torch.float64)
    if sm_clock_mhz is None or sm_clock_mhz <= 0:
        return latency_f64
    return latency_f64 * (1000.0 / sm_clock_mhz)


def _summarize_trace(latency_cycles: torch.Tensor, stride_bytes: int, page_bytes: int, sm_clock_mhz: float | None) -> dict:
    latency_f64 = _cycles_to_ns(latency_cycles, sm_clock_mhz)
    accesses_per_page = page_bytes // stride_bytes
    page_count = latency_f64.numel() // accesses_per_page
    page_view = latency_f64.view(page_count, accesses_per_page)
    page_means = page_view.mean(dim=1)
    mod_means = page_view.mean(dim=0)

    even_pages = page_means[0::2]
    odd_pages = page_means[1::2]

    return {
        "mean_ns": float(latency_f64.mean().item()),
        "median_ns": float(latency_f64.median().item()),
        "min_ns": float(latency_f64.min().item()),
        "max_ns": float(latency_f64.max().item()),
        "even_page_mean_ns": float(even_pages.mean().item()) if even_pages.numel() else 0.0,
        "odd_page_mean_ns": float(odd_pages.mean().item()) if odd_pages.numel() else 0.0,
        "odd_minus_even_page_mean_ns": (
            float(odd_pages.mean().item() - even_pages.mean().item())
            if even_pages.numel() and odd_pages.numel()
            else 0.0
        ),
        "page_mean_ns": [float(x) for x in page_means.tolist()],
        "mod_page_mean_ns": [float(x) for x in mod_means.tolist()],
    }


def main():
    parser = argparse.ArgumentParser(description="Trace sequential 256B-stride cold-L2 latency")
    parser.add_argument("--span-kb", type=int, default=1024)
    parser.add_argument("--stride-bytes", type=int, default=256)
    parser.add_argument("--page-bytes", type=int, default=8192)
    parser.add_argument("--base-offset-kb", type=int, default=0)
    parser.add_argument("--flush-mb-per-block", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--output",
        type=str,
        default="results/sequential_stride_latency_256b_1mib_a800_20260404.json",
    )
    args = parser.parse_args()

    if args.page_bytes % args.stride_bytes != 0:
        raise ValueError("page_bytes must be divisible by stride_bytes")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    sm_clock_mhz = _query_sm_clock_mhz()
    span_bytes = args.span_kb * 1024
    base_offset_bytes = args.base_offset_kb * 1024

    chain, start, chain_meta = _build_sequential_chain(
        num_blocks=args.num_blocks,
        span_bytes=span_bytes,
        stride_bytes=args.stride_bytes,
        base_offset_bytes=base_offset_bytes,
        device=device,
    )
    flush_buf = _build_flush_buffer(args.num_blocks, args.flush_mb_per_block, device)

    output = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "system": get_system_info(),
            "device_name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "sm_clock_mhz_approx": sm_clock_mhz,
            "num_sms": props.multi_processor_count,
            "num_blocks": args.num_blocks,
            "repeats": args.repeats,
            "page_bytes": args.page_bytes,
            "flush_mb_per_block": args.flush_mb_per_block,
            "chain_meta": chain_meta,
        },
        "repeats": [],
    }

    workspace = {}
    for repeat_idx in range(args.repeats):
        latency, smid, final_idx, flush_sink = cuda_decode_gemv.l2_sequential_pointer_chase_trace(
            chain,
            start,
            flush_buf,
            chain_len=chain_meta["chain_len"],
            workspace=workspace,
        )
        torch.cuda.synchronize()

        latency_cpu = latency.cpu()
        smid_cpu = smid.cpu()
        final_idx_cpu = final_idx.cpu()
        flush_sink_cpu = flush_sink.cpu()

        repeat_entry = {
            "repeat": repeat_idx,
            "per_block": [],
            "flush_sink": [int(x) for x in flush_sink_cpu.tolist()],
        }
        for block_idx in range(args.num_blocks):
            lat_block_cycles = latency_cpu[block_idx]
            lat_block_ns = _cycles_to_ns(lat_block_cycles, sm_clock_mhz)
            repeat_entry["per_block"].append(
                {
                    "block_idx": block_idx,
                    "smid": int(smid_cpu[block_idx].item()),
                    "final_idx": int(final_idx_cpu[block_idx].item()),
                    "latency_cycles": [int(x) for x in lat_block_cycles.tolist()],
                    "latency_ns_approx": [float(x) for x in lat_block_ns.tolist()],
                    "summary": _summarize_trace(
                        lat_block_cycles,
                        args.stride_bytes,
                        args.page_bytes,
                        sm_clock_mhz,
                    ),
                }
            )
        output["repeats"].append(repeat_entry)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    for repeat_entry in output["repeats"]:
        for block in repeat_entry["per_block"]:
            summary = block["summary"]
            print(
                f"[repeat {repeat_entry['repeat']}] "
                f"block={block['block_idx']} smid={block['smid']} "
                f"mean={summary['mean_ns']:.3f} median={summary['median_ns']:.3f} "
                f"even8k={summary['even_page_mean_ns']:.3f} "
                f"odd8k={summary['odd_page_mean_ns']:.3f} "
                f"odd-even={summary['odd_minus_even_page_mean_ns']:.3f}"
            )
    print(f"[output] {output_path}")


if __name__ == "__main__":
    main()
