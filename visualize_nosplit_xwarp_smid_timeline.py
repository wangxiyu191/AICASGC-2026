#!/usr/bin/env python3
"""Render an SM-level CTA timeline for the k=6144 xwarp async decode GEMV trace."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("ENABLE_CUSTOM_DECODE", "1")
os.environ.setdefault("ENABLE_CUDAGRAPH", "0")
os.environ.setdefault("CUDAGRAPH_PREBUILD_MAX_CACHE_LEN", "0")

import torch
import torch.nn.functional as F
from datasets import load_from_disk

import cuda_decode_gemv
from bench_decode_attention import _prepare_inputs
from bench_decode_linears import _run_capture
from evaluation_wrapper import VLMModel


PHASE_COLORS = {
    "x_issue": "#4e79a7",
    "w_issue": "#59a14f",
    "compute": "#f28e2b",
    "wait": "#e15759",
    "sync": "#af7aa1",
    "epilogue": "#76b7b2",
}

WAVE_COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#76b7b2",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
]


def _derive_phase(segment_label: str) -> str:
    if segment_label.startswith("prologue_"):
        tail = segment_label[len("prologue_") :]
    elif segment_label.startswith("tile"):
        tail = segment_label.split("_", 1)[1]
    else:
        tail = segment_label
    return tail


def _segment_defs(slot_names: list[str]) -> list[dict[str, object]]:
    if len(slot_names) % 2 != 0:
        raise RuntimeError("slot_names must come in start/end pairs")

    defs: list[dict[str, object]] = []
    for idx in range(0, len(slot_names), 2):
        start_name = slot_names[idx]
        end_name = slot_names[idx + 1]
        if not start_name.endswith("_start") or not end_name.endswith("_end"):
            raise RuntimeError(f"unexpected slot pair: {start_name}, {end_name}")
        label = start_name[: -len("_start")]
        if label != end_name[: -len("_end")]:
            raise RuntimeError(f"slot mismatch: {start_name}, {end_name}")
        defs.append(
            {
                "start_idx": idx,
                "end_idx": idx + 1,
                "label": label,
                "phase": _derive_phase(label),
            }
        )
    return defs


def _round_us(value: float) -> float:
    return round(float(value), 3)


def _collect_trace_payload(args: argparse.Namespace) -> tuple[dict, torch.Tensor, torch.Tensor]:
    print(
        f"[Timeline] sample={args.sample_idx} layer={args.layer_idx} "
        f"step={args.decode_step} target={args.target} repeats={args.repeats}"
    )
    vlm = VLMModel(args.model_path)
    dataset = load_from_disk(args.dataset_path)
    sample = dataset[args.sample_idx]
    inputs = _prepare_inputs(vlm, sample["image"], sample["question"])
    payloads, capture_meta = _run_capture(
        vlm,
        inputs,
        decode_step=args.decode_step,
        max_new_tokens=args.max_new_tokens,
        bucketed=args.bucketed,
        layer_idx=args.layer_idx,
    )

    payload = payloads[args.target]
    x = payload["x"]
    weight = payload["weight"]
    bias = payload["bias"]
    if weight.shape[1] != 6144:
        raise RuntimeError(f"{args.target} weight shape {tuple(weight.shape)} is not k=6144")

    ref = F.linear(x, weight, bias)
    trace_reps = []
    smid_reps = []
    max_abs_diff = 0.0
    mean_abs_diff = 0.0
    meta = None

    with torch.no_grad():
        cuda_decode_gemv.nosplit_matvec_half(x, weight, bias)
        torch.cuda.synchronize()
        for rep in range(args.repeats):
            out, trace, smid, meta = cuda_decode_gemv.nosplit_matvec_half_profile_xwarp_detail_with_smid(x, weight, bias)
            torch.cuda.synchronize()
            if rep == 0:
                diff = (out - ref).abs()
                max_abs_diff = float(diff.max().item())
                mean_abs_diff = float(diff.float().mean().item())
            trace_reps.append(trace.cpu())
            smid_reps.append(smid.cpu())

    trace_all = torch.stack(trace_reps, dim=0)
    smid_all = torch.stack(smid_reps, dim=0)
    metadata = {
        "question_id": int(sample["question_id"]),
        "sample_idx": int(args.sample_idx),
        "layer_idx": int(args.layer_idx),
        "decode_step": int(args.decode_step),
        "bucketed": bool(args.bucketed),
        "repeats": int(args.repeats),
        "prompt_length": int(capture_meta["prompt_length"]),
        "max_cache_len": int(capture_meta["max_cache_len"]),
        "target": args.target,
        "x_shape": list(x.shape),
        "weight_shape": list(weight.shape),
        "has_bias": bias is not None,
        "kernel_name": meta["kernel_name"],
        "grid_blocks": int(meta["grid_blocks"]),
        "warp_groups": int(meta["warp_groups"]),
        "slot_names": list(meta["slot_names"]),
        "max_abs_diff_vs_f_linear": max_abs_diff,
        "mean_abs_diff_vs_f_linear": mean_abs_diff,
    }
    return metadata, trace_all, smid_all


def _assign_lanes(blocks: list[dict[str, object]]) -> int:
    lane_ends: list[float] = []
    for block in sorted(blocks, key=lambda item: (float(item["start_us"]), int(item["block"]))):
        start_us = float(block["start_us"])
        lane = 0
        while lane < len(lane_ends) and start_us < lane_ends[lane]:
            lane += 1
        if lane == len(lane_ends):
            lane_ends.append(float(block["end_us"]))
        else:
            lane_ends[lane] = float(block["end_us"])
        block["lane"] = lane
    return max(1, len(lane_ends))


def _build_processed_timeline(metadata: dict, trace_all: torch.Tensor, smid_all: torch.Tensor) -> dict:
    slot_names = list(metadata["slot_names"])
    segment_defs = _segment_defs(slot_names)
    repeats = int(trace_all.shape[0])
    grid_blocks = int(trace_all.shape[1])
    all_durations = []
    repeat_entries = []
    global_smid_set: set[int] = set()

    for rep in range(repeats):
        trace_rep = trace_all[rep].to(torch.int64)
        smid_rep = smid_all[rep].to(torch.int32)
        valid_ctas = trace_rep[:, :, -1] > 0
        positive_starts = trace_rep[:, :, 0][trace_rep[:, :, 0] > 0]
        if positive_starts.numel() == 0:
            continue
        kernel_base = int(positive_starts.min().item())
        block_entries = []

        for block in range(grid_blocks):
            valid_warps = valid_ctas[block]
            if not bool(valid_warps.any()):
                continue
            start_ticks = trace_rep[block, :, 0][valid_warps]
            end_ticks = trace_rep[block, :, -1][valid_warps]
            start_tick = int(start_ticks.min().item())
            end_tick = int(end_ticks.max().item())
            if end_tick <= start_tick:
                continue

            valid_smid_values = smid_rep[block][valid_warps]
            smid_values = [int(v) for v in valid_smid_values.tolist() if int(v) >= 0]
            smid = smid_values[0] if smid_values else -1
            if smid >= 0:
                global_smid_set.add(smid)

            warp_idx = int(torch.nonzero(valid_warps, as_tuple=False)[0].item())
            rep_trace = trace_rep[block, warp_idx]
            segments = []
            for seg in segment_defs:
                seg_start = int(rep_trace[int(seg["start_idx"])].item())
                seg_end = int(rep_trace[int(seg["end_idx"])].item())
                if seg_start <= 0 or seg_end <= seg_start:
                    continue
                segments.append(
                    {
                        "label": str(seg["label"]),
                        "phase": str(seg["phase"]),
                        "start_us": _round_us((seg_start - kernel_base) / 1000.0),
                        "end_us": _round_us((seg_end - kernel_base) / 1000.0),
                        "dur_us": _round_us((seg_end - seg_start) / 1000.0),
                    }
                )

            block_entry = {
                "block": int(block),
                "smid": int(smid),
                "start_us": _round_us((start_tick - kernel_base) / 1000.0),
                "end_us": _round_us((end_tick - kernel_base) / 1000.0),
                "dur_us": _round_us((end_tick - start_tick) / 1000.0),
                "segments": segments,
            }
            block_entries.append(block_entry)
            all_durations.append(float(block_entry["dur_us"]))

        block_entries.sort(key=lambda item: (int(item["smid"]), float(item["start_us"]), int(item["block"])))
        unique_starts = sorted(set(float(item["start_us"]) for item in block_entries))
        wave_by_start = {start_us: wave_id for wave_id, start_us in enumerate(unique_starts)}
        wave_counts: dict[int, int] = {}
        for block_entry in block_entries:
            wave_id = int(wave_by_start[float(block_entry["start_us"])])
            block_entry["wave_id"] = wave_id
            wave_counts[wave_id] = wave_counts.get(wave_id, 0) + 1

        sm_groups: dict[int, list[dict[str, object]]] = {}
        for block_entry in block_entries:
            sm_groups.setdefault(int(block_entry["smid"]), []).append(block_entry)
        sm_lane_counts: dict[str, int] = {}
        for smid, blocks in sm_groups.items():
            sm_lane_counts[str(smid)] = _assign_lanes(blocks)

        durations = torch.tensor([float(item["dur_us"]) for item in block_entries], dtype=torch.float64)
        block_summary = {
            "count": int(len(block_entries)),
            "mean_us": _round_us(float(durations.mean().item())) if durations.numel() else 0.0,
            "median_us": _round_us(float(durations.median().item())) if durations.numel() else 0.0,
            "p95_us": _round_us(float(torch.quantile(durations, 0.95).item())) if durations.numel() else 0.0,
            "max_us": _round_us(float(durations.max().item())) if durations.numel() else 0.0,
        }
        repeat_entries.append(
            {
                "rep": int(rep),
                "kernel_end_us": _round_us(max(float(item["end_us"]) for item in block_entries)) if block_entries else 0.0,
                "active_smid_count": int(len(sm_groups)),
                "max_lanes_per_smid": int(max((int(item["lane"]) + 1) for item in block_entries)) if block_entries else 0,
                "block_duration_summary": block_summary,
                "block_rows": block_entries,
                "smid_order": sorted(int(smid) for smid in sm_groups.keys()),
                "sm_lane_counts": {str(k): int(v) for k, v in sm_lane_counts.items()},
                "wave_count": int(len(unique_starts)),
                "wave_starts_us": [_round_us(v) for v in unique_starts],
                "wave_counts": {str(k): int(v) for k, v in sorted(wave_counts.items())},
            }
        )

    duration_tensor = torch.tensor(all_durations, dtype=torch.float64)
    global_thresholds = {
        "p95_us": _round_us(float(torch.quantile(duration_tensor, 0.95).item())) if duration_tensor.numel() else 0.0,
        "p99_us": _round_us(float(torch.quantile(duration_tensor, 0.99).item())) if duration_tensor.numel() else 0.0,
        "max_us": _round_us(float(duration_tensor.max().item())) if duration_tensor.numel() else 0.0,
        "mean_us": _round_us(float(duration_tensor.mean().item())) if duration_tensor.numel() else 0.0,
    }

    worst_rep = 0
    if repeat_entries:
        worst_rep = max(range(len(repeat_entries)), key=lambda idx: repeat_entries[idx]["block_duration_summary"]["max_us"])

    return {
        "metadata": metadata,
        "phase_colors": PHASE_COLORS,
        "wave_colors": {str(i): color for i, color in enumerate(WAVE_COLORS)},
        "global_thresholds": global_thresholds,
        "active_smids": sorted(global_smid_set),
        "default_repeat_index": int(worst_rep),
        "repeats": repeat_entries,
    }


def _build_html(processed: dict) -> str:
    title = "nosplit xwarp smid timeline"
    escaped_title = html.escape(title)
    data_json = json.dumps(processed, ensure_ascii=False, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escaped_title}</title>
  <style>
    body {{
      margin: 0;
      font-family: "DejaVu Sans Mono", "SFMono-Regular", Consolas, monospace;
      background: #f5f1e8;
      color: #1f1f1f;
    }}
    .page {{
      padding: 18px 22px 28px 22px;
    }}
    .title {{
      font-size: 20px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .meta {{
      font-size: 12px;
      margin-bottom: 14px;
      line-height: 1.5;
    }}
    .controls {{
      display: flex;
      gap: 14px;
      align-items: center;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }}
    .controls label {{
      font-size: 13px;
      font-weight: 600;
    }}
    .summary {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    .card {{
      border: 1px solid #b9b2a3;
      background: #fffaf0;
      padding: 10px 12px;
      min-width: 180px;
      box-shadow: 0 1px 0 rgba(0, 0, 0, 0.04);
    }}
    .card .k {{
      font-size: 11px;
      color: #6b665b;
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .card .v {{
      font-size: 18px;
      font-weight: 700;
    }}
    .legend {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 8px;
      font-size: 12px;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border: 1px solid rgba(0, 0, 0, 0.25);
      display: inline-block;
    }}
    .notes {{
      font-size: 12px;
      color: #534f47;
      margin-bottom: 10px;
      line-height: 1.45;
    }}
    .section-title {{
      font-size: 13px;
      font-weight: 700;
      margin: 14px 0 8px 0;
    }}
    .svg-wrap {{
      overflow-x: auto;
      border: 1px solid #c7c0b1;
      background: #fffdf8;
      margin-bottom: 14px;
    }}
    svg {{
      display: block;
    }}
    .axis {{
      stroke: #7c776c;
      stroke-width: 1;
    }}
    .tick {{
      stroke: #d8d1c2;
      stroke-width: 1;
    }}
    .sm-label {{
      font-size: 11px;
      fill: #3f3a32;
      dominant-baseline: middle;
      text-anchor: end;
    }}
    .axis-label {{
      font-size: 12px;
      fill: #3f3a32;
    }}
    .tick-label {{
      font-size: 10px;
      fill: #6f695f;
      text-anchor: middle;
    }}
    .row-bg {{
      fill: #fcf7eb;
    }}
    .row-sep {{
      stroke: #eee8d7;
      stroke-width: 1;
    }}
    .bar-outline {{
      fill: none;
      stroke-width: 0.9;
    }}
    .footer {{
      margin-top: 8px;
      font-size: 11px;
      color: #6f695f;
      line-height: 1.45;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="title">SM-level CTA timeline for nosplit k6144 xwarp async GEMV</div>
    <div class="meta" id="meta"></div>
    <div class="controls">
      <label for="rep-select">Repeat</label>
      <select id="rep-select"></select>
      <button id="prev-btn" type="button">Prev</button>
      <button id="next-btn" type="button">Next</button>
      <span id="repeat-headline"></span>
    </div>
    <div class="summary" id="summary"></div>
    <div class="legend" id="legend"></div>
    <div class="notes">
      First panel is the recommended view: each CTA is colored only by wave/start cohort, so you can read launch waves and SM tail directly.
      Second panel keeps the old per-phase colors, but those colors come from one representative warp and can look sparse; do not use that panel to infer that a CTA skipped `x` load.
    </div>
    <div class="section-title">Wave View</div>
    <div class="svg-wrap">
      <svg id="wave-timeline"></svg>
    </div>
    <div class="section-title">Phase View</div>
    <div class="svg-wrap">
      <svg id="phase-timeline"></svg>
    </div>
    <div class="footer" id="footer"></div>
  </div>
  <script>
    const payload = {data_json};
    const selectEl = document.getElementById("rep-select");
    const summaryEl = document.getElementById("summary");
    const legendEl = document.getElementById("legend");
    const metaEl = document.getElementById("meta");
    const footerEl = document.getElementById("footer");
    const headlineEl = document.getElementById("repeat-headline");
    const waveSvg = document.getElementById("wave-timeline");
    const phaseSvg = document.getElementById("phase-timeline");

    function fmt(value) {{
      return Number(value).toFixed(3).replace(/\\.000$/, "");
    }}

    function niceStep(maxValue) {{
      const rough = maxValue / 10.0;
      const power = Math.pow(10, Math.floor(Math.log10(Math.max(rough, 1e-6))));
      const scaled = rough / power;
      let step = 1;
      if (scaled > 5) step = 10;
      else if (scaled > 2) step = 5;
      else if (scaled > 1) step = 2;
      return step * power;
    }}

    function buildLegend() {{
      const items = Object.entries(payload.wave_colors).slice(0, 10).map(([wave, color]) =>
        `<span><i class="swatch" style="background:${{color}}"></i>wave${{wave}}</span>`
      );
      items.push(`<span>|</span>`);
      items.push(...Object.entries(payload.phase_colors).map(([phase, color]) =>
        `<span><i class="swatch" style="background:${{color}}"></i>${{phase}}</span>`
      ));
      items.push(`<span><i class="swatch" style="background:transparent;border:1.6px solid #222"></i>&ge; global p95 duration</span>`);
      items.push(`<span><i class="swatch" style="background:transparent;border:1.8px solid #d81b60"></i>&ge; global p99 duration</span>`);
      legendEl.innerHTML = items.join("");
    }}

    function buildMeta() {{
      const m = payload.metadata;
      metaEl.textContent =
        `target=${{m.target}} kernel=${{m.kernel_name}}  sample=${{m.sample_idx}} layer=${{m.layer_idx}} decode_step=${{m.decode_step}}  ` +
        `x=${{m.x_shape.join("x")}} weight=${{m.weight_shape.join("x")}}  repeats=${{m.repeats}}  ` +
        `max_abs_diff=${{m.max_abs_diff_vs_f_linear.toExponential(3)}} mean_abs_diff=${{m.mean_abs_diff_vs_f_linear.toExponential(3)}}`;
      footerEl.textContent =
        `Rendered from raw CTA trace + smid. Active SM count across captured repeats: ${{payload.active_smids.length}}. ` +
        `Global CTA duration mean=${{fmt(payload.global_thresholds.mean_us)}}us, p95=${{fmt(payload.global_thresholds.p95_us)}}us, ` +
        `p99=${{fmt(payload.global_thresholds.p99_us)}}us, max=${{fmt(payload.global_thresholds.max_us)}}us.`;
    }}

    function buildSummary(rep) {{
      const slowBlocks = rep.block_rows
        .slice()
        .sort((a, b) => b.dur_us - a.dur_us)
        .slice(0, 5)
        .map((row) => `b${{row.block}}@sm${{row.smid}}=${{fmt(row.dur_us)}}us`)
        .join("  ");
      const waveSummary = rep.wave_starts_us
        .map((start, idx) => `w${{idx}}@${{fmt(start)}}us:${{rep.wave_counts[String(idx)] || 0}}`)
        .join("  ");

      summaryEl.innerHTML = [
        ["kernel span", `${{fmt(rep.kernel_end_us)}} us`],
        ["active sms", `${{rep.active_smid_count}}`],
        ["max lanes / sm", `${{rep.max_lanes_per_smid}}`],
        ["wave count", `${{rep.wave_count}}`],
        ["cta median", `${{fmt(rep.block_duration_summary.median_us)}} us`],
        ["cta p95", `${{fmt(rep.block_duration_summary.p95_us)}} us`],
        ["cta max", `${{fmt(rep.block_duration_summary.max_us)}} us`],
        ["wave starts", waveSummary || "none"],
        ["top slow blocks", slowBlocks || "none"],
      ].map(([k, v]) => `<div class="card"><div class="k">${{k}}</div><div class="v">${{v}}</div></div>`).join("");
    }}

    function renderCommon(rep) {{
      const left = 78;
      const right = 22;
      const top = 34;
      const bottom = 34;
      const smGap = 6;
      const laneHeight = 9;
      const laneGap = 2;
      const width = 1800;
      const innerWidth = width - left - right;
      const sms = rep.smid_order.slice().sort((a, b) => a - b);
      const rowLayouts = [];
      let cursorY = top;
      for (const sm of sms) {{
        const lanes = rep.sm_lane_counts[String(sm)] || 1;
        const bodyH = lanes * laneHeight + (lanes - 1) * laneGap;
        rowLayouts.push({{ sm, lanes, y: cursorY, height: bodyH }});
        cursorY += bodyH + smGap;
      }}
      const height = cursorY + bottom;
      const scale = innerWidth / Math.max(rep.kernel_end_us, 1e-9);
      const tickStep = niceStep(rep.kernel_end_us);
      const tickCount = Math.ceil(rep.kernel_end_us / tickStep);
      const rowMap = Object.fromEntries(rowLayouts.map((row) => [String(row.sm), row]));
      return {{ left, right, top, bottom, laneHeight, laneGap, width, innerWidth, height, scale, tickCount, tickStep, rowLayouts, rowMap }};
    }}

    function renderWave(rep, layout) {{
      const parts = [];
      parts.push(`<rect x="0" y="0" width="${{layout.width}}" height="${{layout.height}}" fill="#fffdf8"></rect>`);
      parts.push(`<line class="axis" x1="${{layout.left}}" y1="${{layout.top - 10}}" x2="${{layout.left}}" y2="${{layout.height - layout.bottom + 4}}"></line>`);
      parts.push(`<line class="axis" x1="${{layout.left}}" y1="${{layout.height - layout.bottom + 4}}" x2="${{layout.width - layout.right}}" y2="${{layout.height - layout.bottom + 4}}"></line>`);
      for (let t = 0; t <= layout.tickCount; t++) {{
        const value = Math.min(rep.kernel_end_us, t * layout.tickStep);
        const x = layout.left + value * layout.scale;
        parts.push(`<line class="tick" x1="${{x}}" y1="${{layout.top - 10}}" x2="${{x}}" y2="${{layout.height - layout.bottom + 4}}"></line>`);
        parts.push(`<text class="tick-label" x="${{x}}" y="${{layout.height - 10}}">${{fmt(value)}}</text>`);
      }}
      parts.push(`<text class="axis-label" x="${{layout.width / 2}}" y="${{layout.height - 2}}">time (us)</text>`);
      parts.push(`<text class="axis-label" x="14" y="18">sm id</text>`);

      for (const row of layout.rowLayouts) {{
        parts.push(`<rect class="row-bg" x="${{layout.left}}" y="${{row.y - 1}}" width="${{layout.innerWidth}}" height="${{row.height + 2}}" rx="2"></rect>`);
        parts.push(`<text class="sm-label" x="${{layout.left - 8}}" y="${{row.y + row.height / 2}}">sm${{row.sm}}</text>`);
        parts.push(`<line class="row-sep" x1="${{layout.left}}" y1="${{row.y + row.height + 3}}" x2="${{layout.width - layout.right}}" y2="${{row.y + row.height + 3}}"></line>`);
      }}

      for (const block of rep.block_rows) {{
        const row = layout.rowMap[String(block.smid)];
        if (!row) continue;
        const lane = block.lane || 0;
        const y = row.y + lane * (layout.laneHeight + layout.laneGap);
        const x0 = layout.left + block.start_us * layout.scale;
        const totalW = Math.max((block.end_us - block.start_us) * layout.scale, 0.8);
        const fill = payload.wave_colors[String(block.wave_id % 10)] || "#999";
        const outlineColor =
          block.dur_us >= payload.global_thresholds.p99_us ? "#d81b60" :
          block.dur_us >= payload.global_thresholds.p95_us ? "#222222" : "#6d675d";
        const outlineWidth =
          block.dur_us >= payload.global_thresholds.p99_us ? 1.8 :
          block.dur_us >= payload.global_thresholds.p95_us ? 1.2 : 0.8;
        const tip = [
          `block=${{block.block}}`,
          `smid=${{block.smid}}`,
          `wave=${{block.wave_id}}`,
          `lane=${{lane}}`,
          `start=${{fmt(block.start_us)}}us`,
          `end=${{fmt(block.end_us)}}us`,
          `dur=${{fmt(block.dur_us)}}us`,
        ].join(" | ");
        parts.push(`<g><title>${{tip}}</title><rect x="${{x0}}" y="${{y}}" width="${{totalW}}" height="${{layout.laneHeight}}" fill="${{fill}}" opacity="0.95"></rect><rect class="bar-outline" x="${{x0}}" y="${{y}}" width="${{totalW}}" height="${{layout.laneHeight}}" stroke="${{outlineColor}}" stroke-width="${{outlineWidth}}"></rect></g>`);
      }}
      waveSvg.setAttribute("width", layout.width);
      waveSvg.setAttribute("height", layout.height);
      waveSvg.innerHTML = parts.join("");
    }}

    function renderPhase(rep, layout) {{
      const parts = [];
      parts.push(`<rect x="0" y="0" width="${{layout.width}}" height="${{layout.height}}" fill="#fffdf8"></rect>`);
      parts.push(`<line class="axis" x1="${{layout.left}}" y1="${{layout.top - 10}}" x2="${{layout.left}}" y2="${{layout.height - layout.bottom + 4}}"></line>`);
      parts.push(`<line class="axis" x1="${{layout.left}}" y1="${{layout.height - layout.bottom + 4}}" x2="${{layout.width - layout.right}}" y2="${{layout.height - layout.bottom + 4}}"></line>`);
      for (let t = 0; t <= layout.tickCount; t++) {{
        const value = Math.min(rep.kernel_end_us, t * layout.tickStep);
        const x = layout.left + value * layout.scale;
        parts.push(`<line class="tick" x1="${{x}}" y1="${{layout.top - 10}}" x2="${{x}}" y2="${{layout.height - layout.bottom + 4}}"></line>`);
        parts.push(`<text class="tick-label" x="${{x}}" y="${{layout.height - 10}}">${{fmt(value)}}</text>`);
      }}
      parts.push(`<text class="axis-label" x="${{layout.width / 2}}" y="${{layout.height - 2}}">time (us)</text>`);
      parts.push(`<text class="axis-label" x="14" y="18">sm id</text>`);

      for (const row of layout.rowLayouts) {{
        parts.push(`<rect class="row-bg" x="${{layout.left}}" y="${{row.y - 1}}" width="${{layout.innerWidth}}" height="${{row.height + 2}}" rx="2"></rect>`);
        parts.push(`<text class="sm-label" x="${{layout.left - 8}}" y="${{row.y + row.height / 2}}">sm${{row.sm}}</text>`);
        parts.push(`<line class="row-sep" x1="${{layout.left}}" y1="${{row.y + row.height + 3}}" x2="${{layout.width - layout.right}}" y2="${{row.y + row.height + 3}}"></line>`);
      }}

      for (const block of rep.block_rows) {{
        const row = layout.rowMap[String(block.smid)];
        if (!row) continue;
        const lane = block.lane || 0;
        const y = row.y + lane * (layout.laneHeight + layout.laneGap);
        const x0 = layout.left + block.start_us * layout.scale;
        const totalW = Math.max((block.end_us - block.start_us) * layout.scale, 0.6);
        const outlineColor =
          block.dur_us >= payload.global_thresholds.p99_us ? "#d81b60" :
          block.dur_us >= payload.global_thresholds.p95_us ? "#222222" : "#6d675d";
        const outlineWidth =
          block.dur_us >= payload.global_thresholds.p99_us ? 1.8 :
          block.dur_us >= payload.global_thresholds.p95_us ? 1.2 : 0.8;
        const segParts = [];
        for (const seg of block.segments) {{
          const segX = layout.left + seg.start_us * layout.scale;
          const segW = Math.max((seg.end_us - seg.start_us) * layout.scale, 0.45);
          const fill = payload.phase_colors[seg.phase] || "#999999";
          segParts.push(`<rect x="${{segX}}" y="${{y}}" width="${{segW}}" height="${{layout.laneHeight}}" fill="${{fill}}" opacity="0.96"></rect>`);
        }}
        const tip = [
          `block=${{block.block}}`,
          `smid=${{block.smid}}`,
          `lane=${{lane}}`,
          `start=${{fmt(block.start_us)}}us`,
          `end=${{fmt(block.end_us)}}us`,
          `dur=${{fmt(block.dur_us)}}us`,
        ].join(" | ");
        parts.push(`<g><title>${{tip}}</title>${{segParts.join("")}}<rect class="bar-outline" x="${{x0}}" y="${{y}}" width="${{totalW}}" height="${{layout.laneHeight}}" stroke="${{outlineColor}}" stroke-width="${{outlineWidth}}"></rect></g>`);
      }}
      phaseSvg.setAttribute("width", layout.width);
      phaseSvg.setAttribute("height", layout.height);
      phaseSvg.innerHTML = parts.join("");
    }}

    function render(repIndex) {{
      const rep = payload.repeats[repIndex];
      headlineEl.textContent = `default=${{payload.default_repeat_index}}, showing repeat #${{rep.rep}}`;
      buildSummary(rep);
      const layout = renderCommon(rep);
      renderWave(rep, layout);
      renderPhase(rep, layout);
    }}

    for (const rep of payload.repeats) {{
      const opt = document.createElement("option");
      opt.value = rep.rep;
      opt.textContent = `repeat #${{rep.rep}}`;
      selectEl.appendChild(opt);
    }}
    selectEl.value = String(payload.repeats[payload.default_repeat_index].rep);
    buildLegend();
    buildMeta();
    render(payload.default_repeat_index);

    function selectByOffset(delta) {{
      const current = payload.repeats.findIndex((rep) => String(rep.rep) === selectEl.value);
      const next = Math.min(payload.repeats.length - 1, Math.max(0, current + delta));
      selectEl.value = String(payload.repeats[next].rep);
      render(next);
    }}

    selectEl.addEventListener("change", () => {{
      const idx = payload.repeats.findIndex((rep) => String(rep.rep) === selectEl.value);
      if (idx >= 0) render(idx);
    }});
    document.getElementById("prev-btn").addEventListener("click", () => selectByOffset(-1));
    document.getElementById("next-btn").addEventListener("click", () => selectByOffset(1));
  </script>
</body>
</html>
"""


def _default_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    stem = f"nosplit_xwarp_smid_timeline_{args.target}_k6144_layer{args.layer_idx}_step{args.decode_step}_r{args.repeats}"
    return (
        Path(f"results/{stem}.json"),
        Path(f"traces/nosplit_xwarp/{stem}.html"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize CTA-level SM timeline for k6144 xwarp async GEMV")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--layer-idx", type=int, default=14)
    parser.add_argument("--decode-step", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--bucketed", action="store_true", default=True)
    parser.add_argument("--no-bucketed", dest="bucketed", action="store_false")
    parser.add_argument("--target", type=str, default="down_proj", choices=["down_proj"])
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--input-json", type=str, default=None, help="Render HTML from an existing processed timeline JSON.")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-html", type=str, default=None)
    args = parser.parse_args()

    if args.input_json is not None:
        processed = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    else:
        metadata, trace_all, smid_all = _collect_trace_payload(args)
        processed = _build_processed_timeline(metadata, trace_all, smid_all)

    default_json, default_html = _default_output_paths(args)
    output_json = Path(args.output_json) if args.output_json is not None else default_json
    output_html = Path(args.output_html) if args.output_html is not None else default_html

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(processed, indent=2, ensure_ascii=False), encoding="utf-8")
    output_html.write_text(_build_html(processed), encoding="utf-8")

    repeats = processed["repeats"]
    default_rep = processed["default_repeat_index"]
    summary = repeats[default_rep]["block_duration_summary"] if repeats else {}
    print(f"[Timeline] processed JSON saved to {output_json}")
    print(f"[Timeline] HTML saved to {output_html}")
    if repeats:
        print(
            "[Timeline] default repeat "
            f"#{repeats[default_rep]['rep']}: median={summary['median_us']:.3f}us "
            f"p95={summary['p95_us']:.3f}us max={summary['max_us']:.3f}us "
            f"active_sm={repeats[default_rep]['active_smid_count']} "
            f"max_lanes_per_sm={repeats[default_rep]['max_lanes_per_smid']}"
        )


if __name__ == "__main__":
    main()
