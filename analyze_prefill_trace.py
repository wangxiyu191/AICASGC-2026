#!/usr/bin/env python3
"""Analyze one prefill VizTracer JSON and split it into model phases.

This tool is purpose-built for the prefill-only traces produced by
`bench_prefill.py --viztrace`. It focuses on one request and reports:

1. High-level prefill phases:
   - vision_only
   - bridge_before_text
   - text_prefill_total
   - lm_head_tail
2. Per-phase GPU/CPU accumulated time and top operators.
3. Heuristic deeper split for the vision phase:
   - patch_embed_and_conv_bootstrap
   - prep_before_first_vision_attn
   - vision_block_attn_loop
   - tail_after_last_vision_attn

The script works from a single trace file and does not load the model.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

from analyze_traces import categorize_cpu_op, categorize_gpu_kernel, simplify_kernel_name


def _load_trace(trace_path: Path):
    with trace_path.open() as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    fee_events = [ev for ev in events if ev.get("cat") == "FEE" and ev.get("ph") == "X"]
    cpu_events = [ev for ev in events if ev.get("cat") == "cpu_op" and ev.get("ph") == "X"]
    gpu_events = [
        ev
        for ev in events
        if ev.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and ev.get("ph") == "X"
    ]
    return {
        "events": events,
        "fee_events": fee_events,
        "cpu_events": cpu_events,
        "gpu_events": gpu_events,
    }


def _find_longest_fee(fee_events, needle: str):
    matches = [ev for ev in fee_events if needle in ev.get("name", "")]
    if not matches:
        return None
    return max(matches, key=lambda ev: ev.get("dur", 0))


def _find_longest_fee_any(fee_events, needles: list[str]):
    matches = [
        ev
        for ev in fee_events
        if any(needle in ev.get("name", "") for needle in needles)
    ]
    if not matches:
        return None
    return max(matches, key=lambda ev: ev.get("dur", 0))


def _event_end(ev) -> float:
    return ev.get("ts", 0.0) + ev.get("dur", 0.0)


def _overlap_us(ev, start_ts: float, end_ts: float) -> float:
    ev_start = ev.get("ts", 0.0)
    ev_end = _event_end(ev)
    return max(0.0, min(ev_end, end_ts) - max(ev_start, start_ts))


def _sorted_items(d, top_k: int):
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


def _count_matching(events, start_ts: float, end_ts: float, predicate: Callable[[dict], bool]) -> int:
    count = 0
    for ev in events:
        if _overlap_us(ev, start_ts, end_ts) <= 0:
            continue
        if predicate(ev):
            count += 1
    return count


def _summarize_window(
    name: str,
    start_ts: float,
    end_ts: float,
    anchor_ts: float,
    cpu_events,
    gpu_events,
    top_k: int,
    note: str | None = None,
):
    span_us = max(0.0, end_ts - start_ts)
    gpu_groups = defaultdict(float)
    gpu_ops = defaultdict(float)
    cpu_groups = defaultdict(float)
    cpu_ops = defaultdict(float)
    gpu_event_count = 0
    cpu_event_count = 0

    for ev in gpu_events:
        overlap_us = _overlap_us(ev, start_ts, end_ts)
        if overlap_us <= 0:
            continue
        gpu_event_count += 1
        if ev.get("cat") == "kernel":
            group = categorize_gpu_kernel(ev.get("name", ""))
            short_name = simplify_kernel_name(ev.get("name", ""))
        else:
            group = ev.get("cat", "unknown")
            short_name = f"[{ev.get('cat', 'gpu')}] {ev.get('name', '')}"
        gpu_groups[group] += overlap_us
        gpu_ops[short_name] += overlap_us

    for ev in cpu_events:
        overlap_us = _overlap_us(ev, start_ts, end_ts)
        if overlap_us <= 0:
            continue
        cpu_event_count += 1
        group = categorize_cpu_op(ev.get("name", ""))
        cpu_groups[group] += overlap_us
        cpu_ops[ev.get("name", "")] += overlap_us

    gpu_accum_us = sum(gpu_groups.values())
    cpu_accum_us = sum(cpu_groups.values())

    result = {
        "name": name,
        "start_offset_ms": (start_ts - anchor_ts) / 1000.0,
        "end_offset_ms": (end_ts - anchor_ts) / 1000.0,
        "span_ms": span_us / 1000.0,
        "gpu_accumulated_ms": gpu_accum_us / 1000.0,
        "cpu_op_accumulated_ms": cpu_accum_us / 1000.0,
        "gpu_event_count": gpu_event_count,
        "cpu_event_count": cpu_event_count,
        "gpu_accum_vs_span_pct": (gpu_accum_us / span_us * 100.0) if span_us > 0 else 0.0,
        "top_gpu_groups": [
            {"name": group, "total_ms": total_us / 1000.0}
            for group, total_us in _sorted_items(gpu_groups, top_k)
        ],
        "top_gpu_ops": [
            {"name": op, "total_ms": total_us / 1000.0}
            for op, total_us in _sorted_items(gpu_ops, top_k)
        ],
        "top_cpu_groups": [
            {"name": group, "total_ms": total_us / 1000.0}
            for group, total_us in _sorted_items(cpu_groups, top_k)
        ],
        "top_cpu_ops": [
            {"name": op, "total_ms": total_us / 1000.0}
            for op, total_us in _sorted_items(cpu_ops, top_k)
        ],
        "attention_kernel_count": _count_matching(
            gpu_events,
            start_ts,
            end_ts,
            lambda ev: ev.get("cat") == "kernel"
            and ("flash" in ev.get("name", "").lower() or "fmha" in ev.get("name", "").lower()),
        ),
        "conv_kernel_count": _count_matching(
            gpu_events,
            start_ts,
            end_ts,
            lambda ev: ev.get("cat") == "kernel" and categorize_gpu_kernel(ev.get("name", "")) == "Conv (Vision)",
        ),
    }
    if note:
        result["note"] = note
    return result


def _extract_major_windows(fee_events):
    windows = {
        "run_prefill_once": _find_longest_fee(fee_events, "_run_prefill_once"),
        "full_forward": _find_longest_fee(fee_events, "Qwen3VLForConditionalGeneration.forward"),
        "model_forward": _find_longest_fee_any(
            fee_events,
            [
                "Qwen3VLModel.forward",
                "_model_forward_slice_fastpath",
            ],
        ),
        "vision_only": _find_longest_fee_any(
            fee_events,
            [
                "Qwen3VLModel.get_image_features",
                "_get_image_features_cached",
            ],
        ),
        "placeholder_mask": _find_longest_fee(fee_events, "Qwen3VLModel.get_placeholder_mask"),
        "compute_3d_position_ids": _find_longest_fee(fee_events, "Qwen3VLModel.compute_3d_position_ids"),
        "text_prefill_total": _find_longest_fee_any(
            fee_events,
            [
                "Qwen3VLTextModel.forward",
                "_language_model_forward_prefill_fastpath",
            ],
        ),
    }
    return windows


def _build_phase_windows(major_windows):
    vision = major_windows["vision_only"]
    text = major_windows["text_prefill_total"]
    model = major_windows["model_forward"]
    full = major_windows["full_forward"]

    if vision is None or text is None or model is None or full is None:
        missing = [k for k, v in major_windows.items() if k in ("vision_only", "text_prefill_total", "model_forward", "full_forward") and v is None]
        raise ValueError(f"Missing required FEE windows in trace: {missing}")

    image_start = vision["ts"]
    image_end = _event_end(vision)
    text_start = text["ts"]
    text_end = _event_end(text)
    model_end = _event_end(model)
    full_end = _event_end(full)

    if image_end > text_start:
        raise ValueError("Trace windows overlap unexpectedly: vision_only ends after text_prefill_total starts")
    if model_end > full_end:
        raise ValueError("Trace windows overlap unexpectedly: model_forward ends after full_forward ends")

    return {
        "vision_only": (image_start, image_end),
        "bridge_before_text": (image_end, text_start),
        "text_prefill_total": (text_start, text_end),
        "lm_head_tail": (model_end, full_end),
    }


def _build_vision_subphases(vision_window, gpu_events):
    start_ts, end_ts = vision_window
    window_gpu = [
        ev
        for ev in gpu_events
        if ev.get("cat") == "kernel" and _overlap_us(ev, start_ts, end_ts) > 0
    ]
    window_gpu.sort(key=lambda ev: ev.get("ts", 0.0))

    conv_events = [
        ev for ev in window_gpu if categorize_gpu_kernel(ev.get("name", "")) == "Conv (Vision)"
    ]
    attn_events = [
        ev for ev in window_gpu if "flash" in ev.get("name", "").lower() or "fmha" in ev.get("name", "").lower()
    ]

    subphases = []
    current_start = start_ts

    if conv_events:
        conv_end = max(_event_end(ev) for ev in conv_events)
        subphases.append(
            (
                "patch_embed_and_conv_bootstrap",
                start_ts,
                min(conv_end, end_ts),
                "From vision window start through the last Conv (Vision) kernel.",
            )
        )
        current_start = min(conv_end, end_ts)

    if attn_events:
        first_attn_start = min(ev["ts"] for ev in attn_events)
        last_attn_end = max(_event_end(ev) for ev in attn_events)
        if first_attn_start > current_start:
            subphases.append(
                (
                    "prep_before_first_vision_attn",
                    current_start,
                    first_attn_start,
                    "Heuristic pre-attention setup: positional interpolation / rotary prep / early block prologue before the first vision attention kernel.",
                )
            )
        subphases.append(
            (
                "vision_block_attn_loop",
                first_attn_start,
                last_attn_end,
                "Heuristic block-loop window spanning from the first to the last vision attention kernel.",
            )
        )
        if last_attn_end < end_ts:
            subphases.append(
                (
                    "tail_after_last_vision_attn",
                    last_attn_end,
                    end_ts,
                    "Heuristic tail after the last vision attention kernel; typically includes the end of the last block plus merger/deepstack epilogue.",
                )
            )
    elif current_start < end_ts:
        subphases.append(
            (
                "vision_residual_window",
                current_start,
                end_ts,
                "Fallback: no vision attention kernel found, so the residual vision window is kept as one segment.",
            )
        )

    return {
        "vision_attention_kernel_count": len(attn_events),
        "vision_conv_kernel_count": len(conv_events),
        "subphases": subphases,
    }


def _count_deepstack_calls(fee_events, start_ts: float, end_ts: float):
    matched = []
    for ev in fee_events:
        if "_deepstack_process" not in ev.get("name", ""):
            continue
        if _overlap_us(ev, start_ts, end_ts) <= 0:
            continue
        matched.append(ev)
    return {
        "count": len(matched),
        "total_ms": sum(ev.get("dur", 0.0) for ev in matched) / 1000.0,
        "max_ms": max((ev.get("dur", 0.0) for ev in matched), default=0.0) / 1000.0,
    }


def _print_phase_summary(phase):
    print(f"\n[{phase['name']}]")
    print(
        "  "
        f"offset={phase['start_offset_ms']:.3f}..{phase['end_offset_ms']:.3f} ms  "
        f"span={phase['span_ms']:.3f} ms  "
        f"gpu_accum={phase['gpu_accumulated_ms']:.3f} ms  "
        f"cpu_accum={phase['cpu_op_accumulated_ms']:.3f} ms"
    )
    if "note" in phase:
        print(f"  note: {phase['note']}")
    print(
        "  "
        f"gpu_events={phase['gpu_event_count']}  cpu_events={phase['cpu_event_count']}  "
        f"attention_kernels={phase['attention_kernel_count']}  conv_kernels={phase['conv_kernel_count']}"
    )
    if phase["top_gpu_groups"]:
        print("  top gpu groups:")
        for item in phase["top_gpu_groups"]:
            print(f"    {item['name']}: {item['total_ms']:.3f} ms")
    if phase["top_gpu_ops"]:
        print("  top gpu ops:")
        for item in phase["top_gpu_ops"]:
            print(f"    {item['name']}: {item['total_ms']:.3f} ms")
    if phase["top_cpu_groups"]:
        print("  top cpu groups:")
        for item in phase["top_cpu_groups"]:
            print(f"    {item['name']}: {item['total_ms']:.3f} ms")


def main():
    parser = argparse.ArgumentParser(description="Analyze one prefill VizTracer JSON")
    parser.add_argument("trace", type=str, help="Path to one VizTracer JSON file from bench_prefill.py --viztrace")
    parser.add_argument("--top-k", type=int, default=8, help="Top groups/ops to print per phase")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    trace_path = Path(args.trace)
    loaded = _load_trace(trace_path)
    fee_events = loaded["fee_events"]
    cpu_events = loaded["cpu_events"]
    gpu_events = loaded["gpu_events"]

    major_windows = _extract_major_windows(fee_events)
    phase_windows = _build_phase_windows(major_windows)
    run_prefill = major_windows["run_prefill_once"]
    anchor_ts = run_prefill["ts"] if run_prefill is not None else phase_windows["vision_only"][0]

    phases = []
    for phase_name in ("vision_only", "bridge_before_text", "text_prefill_total", "lm_head_tail"):
        start_ts, end_ts = phase_windows[phase_name]
        phases.append(
            _summarize_window(
                phase_name,
                start_ts,
                end_ts,
                anchor_ts,
                cpu_events=cpu_events,
                gpu_events=gpu_events,
                top_k=args.top_k,
            )
        )

    vision_subphase_meta = _build_vision_subphases(phase_windows["vision_only"], gpu_events)
    vision_subphases = []
    for sub_name, start_ts, end_ts, note in vision_subphase_meta["subphases"]:
        vision_subphases.append(
            _summarize_window(
                sub_name,
                start_ts,
                end_ts,
                anchor_ts,
                cpu_events=cpu_events,
                gpu_events=gpu_events,
                top_k=args.top_k,
                note=note,
            )
        )

    text_start, text_end = phase_windows["text_prefill_total"]
    deepstack_summary = _count_deepstack_calls(fee_events, text_start, text_end)

    output = {
        "trace": str(trace_path),
        "major_windows": {
            key: (
                None
                if ev is None
                else {
                    "name": ev.get("name", ""),
                    "start_offset_ms": (ev.get("ts", 0.0) - anchor_ts) / 1000.0,
                    "dur_ms": ev.get("dur", 0.0) / 1000.0,
                }
            )
            for key, ev in major_windows.items()
        },
        "phases": phases,
        "vision_detail": {
            "heuristic": (
                "vision subphases are inferred directly from the trace: "
                "conv tail, first/last vision attention kernel, and residual window."
            ),
            "vision_attention_kernel_count": vision_subphase_meta["vision_attention_kernel_count"],
            "vision_conv_kernel_count": vision_subphase_meta["vision_conv_kernel_count"],
            "subphases": vision_subphases,
        },
        "text_detail": {
            "deepstack_process": deepstack_summary,
        },
    }

    print(f"Trace: {trace_path}")
    if run_prefill is not None:
        print(f"Anchor: _run_prefill_once span={run_prefill.get('dur', 0.0)/1000.0:.3f} ms")
    print("\n== Prefill Phases ==")
    for phase in phases:
        _print_phase_summary(phase)

    print("\n== Vision Detail ==")
    print(
        "  "
        f"vision_attention_kernels={vision_subphase_meta['vision_attention_kernel_count']}  "
        f"vision_conv_kernels={vision_subphase_meta['vision_conv_kernel_count']}"
    )
    for phase in vision_subphases:
        _print_phase_summary(phase)

    print("\n== Text Detail ==")
    print(
        "  "
        f"deepstack_process count={deepstack_summary['count']}  "
        f"total={deepstack_summary['total_ms']:.3f} ms  "
        f"max={deepstack_summary['max_ms']:.3f} ms"
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"\nSaved JSON to: {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
